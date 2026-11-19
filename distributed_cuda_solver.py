import os
import sys
import json
import math
import time
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Non-periodic 2x2 Cartesian process grid
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart_comm.Get_coords(rank)  # (row_rank, col_rank)
    nbr_up, nbr_down = cart_comm.Shift(direction=0, disp=1)
    nbr_left, nbr_right = cart_comm.Shift(direction=1, disp=1)

    neighbors = {
        "up": nbr_up,
        "down": nbr_down,
        "left": nbr_left,
        "right": nbr_right,
    }
    return cart_comm, rank, coords, neighbors, gpu_uuid, instance_seed


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ==============================================================================
def initialize_subdomains(coords, neighbors, instance_seed: int):
    row_rank, col_rank = coords
    ny, nx = GLOBAL_NY // 2, GLOBAL_NX // 2  # 256 x 256 owned interior
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    # Local padded arrays of shape (258, 258): indices 1..256 interior, 0 and 257 ghost
    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    kappa = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    source = np.zeros((ny + 2, nx + 2), dtype=np.float64)

    # Seeded parameters for Gaussian pulses and heterogeneous diffusivity
    rng = np.random.default_rng(instance_seed)
    p1_x, p1_y = rng.uniform(0.25, 0.45), rng.uniform(0.25, 0.45)
    p2_x, p2_y = rng.uniform(0.55, 0.75), rng.uniform(0.55, 0.75)
    amp1, amp2 = rng.uniform(0.8, 1.2), rng.uniform(0.6, 1.0)
    freq_x = rng.uniform(2.0 * math.pi, 4.0 * math.pi)
    freq_y = rng.uniform(2.0 * math.pi, 4.0 * math.pi)

    # Global index offsets for owned cells (1..256 map to global 1..512)
    gi_offset = row_rank * ny
    gj_offset = col_rank * nx

    li = np.arange(ny + 2)
    lj = np.arange(nx + 2)
    gi = gi_offset + li
    gj = gj_offset + lj
    Y, X = np.meshgrid(gi * dy, gj * dx, indexing="ij")

    # Spatially varying diffusivity kappa in [0.08, 0.24]
    kappa[:, :] = 0.16 + 0.08 * np.sin(freq_x * X) * np.cos(freq_y * Y)

    # Initial thermal field: two seeded Gaussian pulses
    u[:, :] = (
        amp1 * np.exp(-((X - p1_x) ** 2 + (Y - p1_y) ** 2) / 0.015)
        + amp2 * np.exp(-((X - p2_x) ** 2 + (Y - p2_y) ** 2) / 0.020)
    )

    # Localized steady source term S(x, y)
    source[:, :] = 0.25 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)

    # Enforce homogeneous Dirichlet boundary conditions (u = 0.0) on global outer edges
    if neighbors["up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
        u[1, :] = 0.0
    if neighbors["down"] == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
        u[ny, :] = 0.0
    if neighbors["left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
        u[:, 1] = 0.0
    if neighbors["right"] == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0
        u[:, nx] = 0.0

    return u, kappa, source, dx, dy, ny, nx


# ==============================================================================
# TASK 3: GPU Halo Pack / Unpack Kernels & Pinned Asynchronous MPI Exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        # Interior top row (1) and bottom row (ny)
        send_up[idx] = u[1, idx + 1]
        send_down[idx] = u[ny, idx + 1]
    if idx < ny:
        # Non-contiguous interior left col (1) and right col (nx)
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(
    u, recv_up, recv_down, recv_left, recv_right,
    has_up, has_down, has_left, has_right, ny, nx
):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = recv_up[idx] if has_up else 0.0
        u[ny + 1, idx + 1] = recv_down[idx] if has_down else 0.0
    if idx < ny:
        u[idx + 1, 0] = recv_left[idx] if has_left else 0.0
        u[idx + 1, nx + 1] = recv_right[idx] if has_right else 0.0


class HaloBuffers:
    """Pre-allocated device and page-locked host staging buffers for zero-allocation steps."""
    def __init__(self, ny: int, nx: int):
        self.d_send_up = cuda.device_array(nx, dtype=np.float64)
        self.d_send_down = cuda.device_array(nx, dtype=np.float64)
        self.d_send_left = cuda.device_array(ny, dtype=np.float64)
        self.d_send_right = cuda.device_array(ny, dtype=np.float64)

        self.d_recv_up = cuda.device_array(nx, dtype=np.float64)
        self.d_recv_down = cuda.device_array(nx, dtype=np.float64)
        self.d_recv_left = cuda.device_array(ny, dtype=np.float64)
        self.d_recv_right = cuda.device_array(ny, dtype=np.float64)

        self.h_send_up = cuda.pinned_array(nx, dtype=np.float64)
        self.h_send_down = cuda.pinned_array(nx, dtype=np.float64)
        self.h_send_left = cuda.pinned_array(ny, dtype=np.float64)
        self.h_send_right = cuda.pinned_array(ny, dtype=np.float64)

        self.h_recv_up = cuda.pinned_array(nx, dtype=np.float64)
        self.h_recv_down = cuda.pinned_array(nx, dtype=np.float64)
        self.h_recv_left = cuda.pinned_array(ny, dtype=np.float64)
        self.h_recv_right = cuda.pinned_array(ny, dtype=np.float64)


def post_halo_pack_and_mpi(cart_comm, d_u, bufs: HaloBuffers, neighbors: dict, stream, ny: int, nx: int):
    """Stage A & initial Stage B: Pack halos on GPU, async D2H to pinned memory, post Isend/Irecv."""
    threads = 128
    blocks = (max(ny, nx) + threads - 1) // threads
    pack_halos_kernel[blocks, threads, stream](
        d_u, bufs.d_send_up, bufs.d_send_down, bufs.d_send_left, bufs.d_send_right, ny, nx
    )

    # Async D2H DMA transfers into page-locked host staging buffers
    bufs.d_send_up.copy_to_host(bufs.h_send_up, stream=stream)
    bufs.d_send_down.copy_to_host(bufs.h_send_down, stream=stream)
    bufs.d_send_left.copy_to_host(bufs.h_send_left, stream=stream)
    bufs.d_send_right.copy_to_host(bufs.h_send_right, stream=stream)
    stream.synchronize()

    # Post non-blocking tag-matched MPI_Irecv and MPI_Isend calls
    reqs = []
    # Tag convention: 10=from_up/to_down, 11=from_down/to_up, 20=from_left/to_right, 21=from_right/to_left
    reqs.append(cart_comm.Irecv(bufs.h_recv_up, source=neighbors["up"], tag=10))
    reqs.append(cart_comm.Irecv(bufs.h_recv_down, source=neighbors["down"], tag=11))
    reqs.append(cart_comm.Irecv(bufs.h_recv_left, source=neighbors["left"], tag=20))
    reqs.append(cart_comm.Irecv(bufs.h_recv_right, source=neighbors["right"], tag=21))

    reqs.append(cart_comm.Isend(bufs.h_send_down, dest=neighbors["down"], tag=10))
    reqs.append(cart_comm.Isend(bufs.h_send_up, dest=neighbors["up"], tag=11))
    reqs.append(cart_comm.Isend(bufs.h_send_right, dest=neighbors["right"], tag=20))
    reqs.append(cart_comm.Isend(bufs.h_send_left, dest=neighbors["left"], tag=21))
    return reqs


def complete_halo_mpi_and_unpack(reqs, d_u, bufs: HaloBuffers, neighbors: dict, stream, ny: int, nx: int):
    """Stage C: Waitall on MPI requests, async H2D DMA, unpack halos into ghost apron."""
    MPI.Request.Waitall(reqs)

    has_up = 1 if neighbors["up"] != MPI.PROC_NULL else 0
    has_down = 1 if neighbors["down"] != MPI.PROC_NULL else 0
    has_left = 1 if neighbors["left"] != MPI.PROC_NULL else 0
    has_right = 1 if neighbors["right"] != MPI.PROC_NULL else 0

    if has_up:
        bufs.d_recv_up.copy_to_device(bufs.h_recv_up, stream=stream)
    if has_down:
        bufs.d_recv_down.copy_to_device(bufs.h_recv_down, stream=stream)
    if has_left:
        bufs.d_recv_left.copy_to_device(bufs.h_recv_left, stream=stream)
    if has_right:
        bufs.d_recv_right.copy_to_device(bufs.h_recv_right, stream=stream)

    threads = 128
    blocks = (max(ny, nx) + threads - 1) // threads
    unpack_halos_kernel[blocks, threads, stream](
        d_u, bufs.d_recv_up, bufs.d_recv_down, bufs.d_recv_left, bufs.d_recv_right,
        has_up, has_down, has_left, has_right, ny, nx
    )


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(
    u, u_next, kappa, source,
    dt, inv_dx2, inv_dy2, lam,
    r_start, r_end, c_start, c_end,
    bc_up, bc_down, bc_left, bc_right,
    ny, nx
):
    # 18 x 18 shared memory tiles for u and kappa
    s_u = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    s_k = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Global cell coordinates within the bounding box [r_start..r_end], [c_start..c_end]
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx

    sy = ty + 1
    sx = tx + 1

    # Load center cell into shared memory (clamped safely to array bounds 0..ny+1, 0..nx+1)
    ci = min(max(i, 0), ny + 1)
    cj = min(max(j, 0), nx + 1)
    s_u[sy, sx] = u[ci, cj]
    s_k[sy, sx] = kappa[ci, cj]

    # Boundary threads load the 1-cell halo apron into s_u and s_k
    if ty == 0:
        ui = max(ci - 1, 0)
        s_u[0, sx] = u[ui, cj]
        s_k[0, sx] = kappa[ui, cj]
    if ty == TILE_DIM - 1 or i == r_end:
        di = min(ci + 1, ny + 1)
        s_u[sy + 1, sx] = u[di, cj]
        s_k[sy + 1, sx] = kappa[di, cj]
    if tx == 0:
        lj = max(cj - 1, 0)
        s_u[sy, 0] = u[ci, lj]
        s_k[sy, 0] = kappa[ci, lj]
    if tx == TILE_DIM - 1 or j == c_end:
        rj = min(cj + 1, nx + 1)
        s_u[sy, sx + 1] = u[ci, rj]
        s_k[sy, sx + 1] = kappa[ci, rj]

    # Synchronize all 8 warps in the 16x16 threadblock before stencil reads
    cuda.syncthreads()

    if r_start <= i <= r_end and c_start <= j <= c_end:
        # Enforce homogeneous Dirichlet u = 0.0 on outer physical boundaries
        if (bc_up and i == 1) or (bc_down and i == ny) or (bc_left and j == 1) or (bc_right and j == nx):
            u_next[i, j] = 0.0
            return

        u_c = s_u[sy, sx]
        k_c = s_k[sy, sx]

        # Arithmetic average interface conductivities
        k_down = 0.5 * (s_k[sy + 1, sx] + k_c)
        k_up = 0.5 * (s_k[sy - 1, sx] + k_c)
        k_right = 0.5 * (s_k[sy, sx + 1] + k_c)
        k_left = 0.5 * (s_k[sy, sx - 1] + k_c)

        flux_y = (k_down * (s_u[sy + 1, sx] - u_c) - k_up * (u_c - s_u[sy - 1, sx])) * inv_dy2
        flux_x = (k_right * (s_u[sy, sx + 1] - u_c) - k_left * (u_c - s_u[sy, sx - 1])) * inv_dx2

        u_next[i, j] = u_c + dt * (flux_y + flux_x - lam * (u_c * u_c * u_c) + source[i, j])


def launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, r0, r1, c0, c1, stream):
    rows = r1 - r0 + 1
    cols = c1 - c0 + 1
    if rows <= 0 or cols <= 0:
        return
    blocks = ((cols + TILE_DIM - 1) // TILE_DIM, (rows + TILE_DIM - 1) // TILE_DIM)
    threads = (TILE_DIM, TILE_DIM)
    tiled_stencil_kernel[blocks, threads, stream](
        d_u, d_u_next, d_k, d_s,
        dt, inv_dx2, inv_dy2, LAMBDA_DAMP,
        r0, r1, c0, c1,
        bcs[0], bcs[1], bcs[2], bcs[3],
        ny, nx
    )


def step_serialized(cart_comm, d_u, d_u_next, d_k, d_s, bufs, neighbors, bcs, dt, inv_dx2, inv_dy2, ny, nx, stream):
    reqs = post_halo_pack_and_mpi(cart_comm, d_u, bufs, neighbors, stream, ny, nx)
    complete_halo_mpi_and_unpack(reqs, d_u, bufs, neighbors, stream, ny, nx)
    launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, 1, ny, 1, nx, stream)
    stream.synchronize()


def step_overlapped(
    cart_comm, d_u, d_u_next, d_k, d_s, bufs, neighbors, bcs,
    dt, inv_dx2, inv_dy2, ny, nx, stream_boundary, stream_interior
):
    # Stage A + start Stage B: Pack border strips & post non-blocking MPI Isend/Irecv
    reqs = post_halo_pack_and_mpi(cart_comm, d_u, bufs, neighbors, stream_boundary, ny, nx)

    # Stage B (Concurrent): Launch deep interior [2..ny-1, 2..nx-1] on stream_interior
    launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, 2, ny - 1, 2, nx - 1, stream_interior)

    # Stage C: Waitall MPI, async H2D, unpack halos, and update 4 outer border rings on stream_boundary
    complete_halo_mpi_and_unpack(reqs, d_u, bufs, neighbors, stream_boundary, ny, nx)

    # Top ring (row 1), Bottom ring (row ny), Left ring (rows 2..ny-1, col 1), Right ring (rows 2..ny-1, col nx)
    launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, 1, 1, 1, nx, stream_boundary)
    launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, ny, ny, 1, nx, stream_boundary)
    launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, 2, ny - 1, 1, 1, stream_boundary)
    launch_box(d_u, d_u_next, d_k, d_s, dt, inv_dx2, inv_dy2, bcs, ny, nx, 2, ny - 1, nx, nx, stream_boundary)

    stream_interior.synchronize()
    stream_boundary.synchronize()


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def compute_global_metrics(cart_comm, d_u, d_u_prev, dx, dy, ny, nx):
    h_u = d_u.copy_to_host()[1:ny + 1, 1:nx + 1]
    local_energy = float(np.sum(h_u) * dx * dy)
    global_energy = cart_comm.allreduce(local_energy, op=MPI.SUM)

    if d_u_prev is not None:
        h_prev = d_u_prev.copy_to_host()[1:ny + 1, 1:nx + 1]
        local_sq_diff = float(np.sum((h_u - h_prev) ** 2))
        global_l2 = math.sqrt(cart_comm.allreduce(local_sq_diff, op=MPI.SUM))
    else:
        global_l2 = 0.0
    return global_energy, global_l2


def inspect_ptx_audit():
    ptx_dict = tiled_stencil_kernel.inspect_asm()
    ptx_code = next(iter(ptx_dict.values()))
    st_shared = ptx_code.count("st.shared.f64")
    ld_shared = ptx_code.count("ld.shared.f64")
    bar_sync = ptx_code.count("bar.sync")
    passed = (st_shared > 0) and (ld_shared > 0) and (bar_sync > 0)
    return {
        "st_shared_f64": st_shared,
        "ld_shared_f64": ld_shared,
        "bar_sync": bar_sync,
        "ptx_audit_passed": passed,
    }


def run_solver(github_user: str):
    cart_comm, rank, coords, neighbors, gpu_uuid, instance_seed = init_distributed_context(github_user)
    u_init, kappa, source, dx, dy, ny, nx = initialize_subdomains(coords, neighbors, instance_seed)

    # Global maximum diffusivity and explicit 2D CFL time step
    local_kmax = float(np.max(kappa))
    global_kmax = cart_comm.allreduce(local_kmax, op=MPI.MAX)
    dt_max = (dx * dx) / (4.0 * global_kmax)
    dt = 0.8 * dt_max
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    bcs = (
        1 if neighbors["up"] == MPI.PROC_NULL else 0,
        1 if neighbors["down"] == MPI.PROC_NULL else 0,
        1 if neighbors["left"] == MPI.PROC_NULL else 0,
        1 if neighbors["right"] == MPI.PROC_NULL else 0,
    )

    d_k = cuda.to_device(kappa)
    d_s = cuda.to_device(source)
    bufs = HaloBuffers(ny, nx)
    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    # Warm-up JIT compilation before timing
    d_u_warm = cuda.to_device(u_init)
    d_un_warm = cuda.to_device(u_init)
    step_overlapped(
        cart_comm, d_u_warm, d_un_warm, d_k, d_s, bufs, neighbors, bcs,
        dt, inv_dx2, inv_dy2, ny, nx, stream_boundary, stream_interior
    )
    cuda.synchronize()
    ptx_report = inspect_ptx_audit()

    # --------------------------------------------------------------------------
    # Mode 1: Serialized Baseline (400 Steps)
    # --------------------------------------------------------------------------
    d_u_ser = cuda.to_device(u_init)
    d_un_ser = cuda.to_device(u_init)
    cart_comm.Barrier()
    ev_s0, ev_s1 = cuda.event(timing=True), cuda.event(timing=True)
    t0_ser = time.perf_counter()
    ev_s0.record(stream_boundary)

    for step in range(1, STEPS + 1):
        step_serialized(
            cart_comm, d_u_ser, d_un_ser, d_k, d_s, bufs, neighbors, bcs,
            dt, inv_dx2, inv_dy2, ny, nx, stream_boundary
        )
        d_u_ser, d_un_ser = d_un_ser, d_u_ser

    ev_s1.record(stream_boundary)
    ev_s1.synchronize()
    cart_comm.Barrier()
    serialized_wall_ms = (time.perf_counter() - t0_ser) * 1000.0
    serialized_cuda_ms = cuda.event_elapsed_time(ev_s0, ev_s1)

    # --------------------------------------------------------------------------
    # Mode 2: Dual-Stream Overlapped Execution (400 Steps) + Energy Trace
    # --------------------------------------------------------------------------
    d_u_ovl = cuda.to_device(u_init)
    d_un_ovl = cuda.to_device(u_init)
    energy_trace = []

    e0, l2_0 = compute_global_metrics(cart_comm, d_u_ovl, None, dx, dy, ny, nx)
    energy_trace.append({"step": 0, "global_energy": e0, "l2_residual": l2_0})

    cart_comm.Barrier()
    ev_o0, ev_o1 = cuda.event(timing=True), cuda.event(timing=True)
    t0_ovl = time.perf_counter()
    ev_o0.record(stream_boundary)

    for step in range(1, STEPS + 1):
        step_overlapped(
            cart_comm, d_u_ovl, d_un_ovl, d_k, d_s, bufs, neighbors, bcs,
            dt, inv_dx2, inv_dy2, ny, nx, stream_boundary, stream_interior
        )
        if step % 100 == 0:
            e_val, l2_val = compute_global_metrics(cart_comm, d_un_ovl, d_u_ovl, dx, dy, ny, nx)
            energy_trace.append({"step": step, "global_energy": e_val, "l2_residual": l2_val})
        d_u_ovl, d_un_ovl = d_un_ovl, d_u_ovl

    ev_o1.record(stream_boundary)
    ev_o1.synchronize()
    cart_comm.Barrier()
    overlapped_wall_ms = (time.perf_counter() - t0_ovl) * 1000.0
    overlapped_cuda_ms = cuda.event_elapsed_time(ev_o0, ev_o1)

    # Verify numerical parity between serialized and overlapped states
    final_ser = d_u_ser.copy_to_host()[1:ny + 1, 1:nx + 1]
    final_ovl = d_u_ovl.copy_to_host()[1:ny + 1, 1:nx + 1]
    local_max_diff = float(np.max(np.abs(final_ser - final_ovl)))
    parity_max_diff = cart_comm.allreduce(local_max_diff, op=MPI.MAX)

    max_ser_ms = cart_comm.allreduce(serialized_wall_ms, op=MPI.MAX)
    max_ovl_ms = cart_comm.allreduce(overlapped_wall_ms, op=MPI.MAX)

    if rank == 0:
        print("=" * 72)
        print("DISTRIBUTED MPI + CUDA 2D ANISOTROPIC REACTION-DIFFUSION SOLVER")
        print("=" * 72)
        print(f"GitHub User     : {github_user}")
        print(f"GPU UUID        : {gpu_uuid}")
        print(f"Instance Seed   : {instance_seed} (0x{instance_seed:016X})")
        print(f"MPI Topology    : 4 ranks in 2x2 Cartesian grid (256x256 interior/rank)")
        print(f"Grid / CFL      : dx={dx:.6e}, k_max={global_kmax:.6f}, dt_max={dt_max:.6e}, dt={dt:.6e}")
        print("-" * 72)
        print("PTX ASSEMBLY AUDIT (tiled_stencil_kernel):")
        print(f"  st.shared.f64 : {ptx_report['st_shared_f64']}")
        print(f"  ld.shared.f64 : {ptx_report['ld_shared_f64']}")
        print(f"  bar.sync      : {ptx_report['bar_sync']}")
        print(f"  Audit Passed  : {ptx_report['ptx_audit_passed']}")
        print("-" * 72)
        print("ENERGY & CONVERGENCE TRACE (Allreduce MPI.SUM):")
        for item in energy_trace:
            print(f"  Step {item['step']:3d} | Global Energy E = {item['global_energy']:.10f} | L2 Residual = {item['l2_residual']:.6e}")
        print("-" * 72)
        print("BENCHMARK & PARITY VERIFICATION:")
        print(f"  Serialized Wall Time : {max_ser_ms:.3f} ms (Rank 0 CUDA Event: {serialized_cuda_ms:.3f} ms)")
        print(f"  Overlapped Wall Time : {max_ovl_ms:.3f} ms (Rank 0 CUDA Event: {overlapped_cuda_ms:.3f} ms)")
        print(f"  Parity Max Diff      : {parity_max_diff:.6e} (Threshold < 1e-12)")
        print("=" * 72)

        assert ptx_report["ptx_audit_passed"], "PTX audit failed!"
        assert parity_max_diff < 1e-12, f"Parity check failed: {parity_max_diff}"

        manifest = {
            "github_username": github_user,
            "gpu_uuid": gpu_uuid,
            "instance_seed_hex": f"0x{instance_seed:016X}",
            "mpi_ranks": 4,
            "cartesian_dims": [2, 2],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_interior_grid": [ny, nx],
            "steps": STEPS,
            "dx": dx,
            "k_max": global_kmax,
            "dt_max": dt_max,
            "dt": dt,
            "ptx_audit": ptx_report,
            "energy_trace": energy_trace,
            "serialized_ms": round(max_ser_ms, 4),
            "overlapped_ms": round(max_ovl_ms, 4),
            "rank0_serialized_cuda_event_ms": round(serialized_cuda_ms, 4),
            "rank0_overlapped_cuda_event_ms": round(overlapped_cuda_ms, 4),
            "parity_max_diff": parity_max_diff,
        }
        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Saved submission_manifest.json successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 distributed_cuda_solver.py <GITHUB_USERNAME>")
    run_solver(sys.argv[1])
