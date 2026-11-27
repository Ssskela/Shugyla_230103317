
# ============================================================
# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS
# ============================================================

import os

# Force native OpenMP runtime BEFORE importing Numba
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import time
import hashlib
import json

import numpy as np
import numba as nb

from numba import (
    njit,
    prange,
    set_num_threads,
    get_num_threads,
    get_thread_id
)

# Student information
STUDENT_NAME = "Shugyla Yermakhanova"
STUDENT_ID = 230103317

assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, \
    "Set a valid numeric STUDENT_ID!"

assert (os.cpu_count() or 1) >= 4, \
    "Local multi-core CPU (>=4 logical cores) required; no Colab!"

# Trigger OpenMP layer initialization and verify backend
@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc


_ = _omp_probe()

assert nb.threading_layer() == "omp", \
    f"Expected 'omp' layer, got {nb.threading_layer()}"

# Deterministic random number generator
rng = np.random.default_rng(STUDENT_ID)

# Student-specific OpenMP parameters
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)

ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8

ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)

GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32

OMEGA_SOR = np.float64(
    1.72 + (STUDENT_ID % 11) * 0.015
)

# Generate deterministic input data for Task 1
data_t1 = rng.integers(
    0,
    BINS_T1,
    size=N_T1,
    dtype=np.int32
)

weights_t1 = rng.uniform(
    0.25,
    2.50,
    size=N_T1
).astype(np.float64)

print(
    f"Backend: {nb.threading_layer()} | "
    f"Logical Cores: {os.cpu_count()} | "
    f"Student: {STUDENT_ID}"
)



# ============================================================
# TASK 1: RACE-CONDITION FORENSICS
# ============================================================

@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)

    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]

    return hist, w_sums


# Single-threaded reference implementation
@njit
def reference_hist(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)

    for i in range(indices.shape[0]):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]

    return hist, w_sums


# ============================================================
# TASK 1.1: EMPIRICAL RACE DEMONSTRATION
# ============================================================

set_num_threads(1)
reference_histogram, reference_weights = reference_hist(
    data_t1, weights_t1, BINS_T1
)

set_num_threads(8)

race_results = []
lost_counts = []
max_discrepancies = []

for run in range(10):
    hist, w_sums = buggy_hist_omp(data_t1, weights_t1, BINS_T1)

    lost = N_T1 - int(np.sum(hist))
    discrepancy = int(np.max(np.abs(hist - reference_histogram)))

    lost_counts.append(lost)
    max_discrepancies.append(discrepancy)

    race_results.append({
        "run": run + 1,
        "lost_counts": lost,
        "max_discrepancy": discrepancy
    })
    
  

print("\nTASK 1.1 — RACE CONDITION RESULTS")
for result in race_results:
    print(result)

print("Total lost counts:", lost_counts)
print("Maximum discrepancy:", max(max_discrepancies))


# ============================================================
# TASK 1.2: DETERMINISTIC THREAD-PRIVATE HISTOGRAM
# ============================================================

@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    max_threads = get_num_threads()

    private_hist = np.zeros(
        (max_threads, num_bins), dtype=np.int64
    )

    private_weights = np.zeros(
        (max_threads, num_bins), dtype=np.float64
    )

    worker_hits = np.zeros(max_threads, dtype=np.int64)

    # Each thread updates only its own private row
    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]

        private_hist[tid, b] += 1
        private_weights[tid, b] += weights[i]
        worker_hits[tid] += 1

    # Deterministic row-wise reduction
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)

    for tid in range(max_threads):
        for b in range(num_bins):
            hist[b] += private_hist[tid, b]
            w_sums[b] += private_weights[tid, b]
    

    return hist, w_sums, worker_hits
     


# ============================================================
# TASK 1.3: VALIDATION ACROSS THREAD COUNTS
# ============================================================

print("\nTASK 1.2 — SAFE HISTOGRAM VALIDATION")

safe_results = {}

for thread_count in (1, 2, 4, 8):
    set_num_threads(thread_count)

    hist, w_sums, worker_hits = safe_hist_omp(
        data_t1, weights_t1, BINS_T1
    )

    # Check exact integer histogram correctness
    hist_correct = np.array_equal(hist, reference_histogram)

    # Check floating-point result against reference
    weights_correct = np.allclose(
        w_sums,
        reference_weights,
        rtol=1e-12,
        atol=1e-10
    )

    # Record the result for comparison
    safe_results[thread_count] = (hist.copy(), w_sums.copy())

    print(f"\nThreads: {thread_count}")
    print("Histogram correct:", hist_correct)
    print("Weighted sums close to reference:", weights_correct)
    print("Total histogram count:", np.sum(hist))
    print("Worker hits:", worker_hits.tolist())
    print("Active workers:", np.count_nonzero(worker_hits))

# Compare integer histograms across thread counts
base_hist = safe_results[1][0]

print("\nInteger histogram identical across thread counts:",
      all(
          np.array_equal(base_hist, safe_results[t][0])
          for t in (2, 4, 8)
      ))

# Compare floating-point sums bit by bit
base_weights = safe_results[1][1]

print("Floating-point sums bit-identical across thread counts:",
      all(
          np.array_equal(base_weights, safe_results[t][1])
          for t in (2, 4, 8)
      ))


# ============================================================
# TASK 1.4: DIGITAL DIAGNOSIS
# ============================================================

DIAGNOSIS_T1 = (
    "The unsafe histogram contains race conditions because multiple "
    "OpenMP threads update the same shared array elements without "
    "synchronization. These concurrent read-modify-write operations "
    "can overwrite each other, resulting in lost updates. The safe "
    "implementation assigns a private histogram and weighted-sum "
    "buffer to each thread, avoiding concurrent writes to shared bins. "
    "The buffers are reduced in a fixed thread order after the parallel "
    "region. However, floating-point addition is not associative, so "
    "changing the grouping of additions can produce slightly different "
    "results even when integer increments are correct."
)

print("\nDIAGNOSIS_T1:")
print(DIAGNOSIS_T1)



# ============================================================
# TASK 2: ELIMINATING CACHE-LINE FALSE SHARING IN OPENMP
# ============================================================

@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()

    # Contiguous array: adjacent threads share cache lines
    state = np.zeros(n_threads, dtype=np.int64)

    for t in prange(n_threads):
        tid = get_thread_id()

        x = np.int64(seed_val + tid * 1315423911)

        for step in range(iters):
            x = (
                x * np.int64(6364136223846793005)
                + np.int64(1442695040888963407)
            ) & np.int64(0x7FFFFFFFFFFFFFFF)

            # Repeated write to shared cache-line region
            state[tid] ^= (x >> 17)

    return state


@njit(parallel=True)
def padded_state_omp(iters, seed_val):
    n_threads = get_num_threads()

    # Each thread gets its own cache-line-separated row
    padded_state = np.zeros(
        (n_threads, PAD_STRIDE),
        dtype=np.int64
    )

    for t in prange(n_threads):
        tid = get_thread_id()

        x = np.int64(seed_val + tid * 1315423911)

        for step in range(iters):
            x = (
                x * np.int64(6364136223846793005)
                + np.int64(1442695040888963407)
            ) & np.int64(0x7FFFFFFFFFFFFFFF)

            # Keep the write inside the loop
            padded_state[tid, 0] ^= (x >> 17)

    return padded_state


# ============================================================
# TASK 2.1: WARM-UP
# ============================================================

print("\nTASK 2 — FALSE SHARING BENCHMARK")

set_num_threads(8)

# Warm-up calls (also trigger compilation)
_ = unpadded_state_omp(ITERS_T2, STUDENT_ID)
_ = padded_state_omp(ITERS_T2, STUDENT_ID)

print("Warm-up completed.")
print("Iterations per thread:", ITERS_T2)
print("Padding stride:", PAD_STRIDE)


# ============================================================
# TASK 2.2: SCALING COLLAPSE BENCHMARK
# ============================================================

thread_counts_t2 = (1, 2, 4, 8)

unpadded_times = {}
padded_times = {}

unpadded_results = {}
padded_results = {}

for thread_count in thread_counts_t2:

    set_num_threads(thread_count)

    # Benchmark unpadded kernel
    start = time.perf_counter()

    unpadded_result = unpadded_state_omp(
        ITERS_T2, STUDENT_ID
    )

    unpadded_elapsed = time.perf_counter() - start

    # Benchmark padded kernel
    start = time.perf_counter()

    padded_result = padded_state_omp(
        ITERS_T2, STUDENT_ID
    )

    padded_elapsed = time.perf_counter() - start

    # Exact correctness verification
    assert np.array_equal(
        unpadded_result,
        padded_result[:, 0]
    ), f"Results differ at {thread_count} threads!"

    unpadded_times[thread_count] = unpadded_elapsed
    padded_times[thread_count] = padded_elapsed

    unpadded_results[thread_count] = unpadded_result.copy()
    padded_results[thread_count] = padded_result[:, 0].copy()

    print(f"\nThreads: {thread_count}")
    print(f"Unpadded time: {unpadded_elapsed:.6f} seconds")
    print(f"Padded time:   {padded_elapsed:.6f} seconds")
    print("Exact results match: True")


# ============================================================
# TASK 2.3: SPEEDUP AND VERIFICATION
# ============================================================

speedup_8 = (
    unpadded_times[8] / padded_times[8]
)

print("\nTASK 2 — FINAL RESULTS")

print("\nThread Scaling Summary:")
for thread_count in thread_counts_t2:
    print(
        f"Threads: {thread_count} | "
        f"Unpadded: {unpadded_times[thread_count]:.6f}s | "
        f"Padded: {padded_times[thread_count]:.6f}s"
    )

print("\n8-thread padding speedup ratio:", f"{speedup_8:.4f}x")

print(
    "All unpadded and padded results match:",
    all(
        np.array_equal(
            unpadded_results[t],
            padded_results[t]
        )
        for t in thread_counts_t2
    )
)


# ============================================================
# TASK 2.4: DIGITAL DIAGNOSIS
# ============================================================

DIAGNOSIS_T2 = (
    "False sharing occurs when independent variables modified by "
    "different threads occupy the same 64-byte cache line. Although "
    "the threads do not access the same logical variable, the MESI "
    "cache-coherence protocol must transfer exclusive ownership of "
    "the cache line between processor cores. Repeated invalidations "
    "cause cache-line bouncing and reduce parallel performance. "
    "Padding separates the active thread counters into different "
    "cache lines, reducing unnecessary coherence traffic."
)

print("\nDIAGNOSIS_T2:")
print(DIAGNOSIS_T2)





# ============================================================
# TASK 3: IRREGULAR WORKLOAD SKEW & BALANCED SCHEDULING
# ============================================================

import math


# ------------------------------------------------------------
# Shared workload calculation
# ------------------------------------------------------------

@njit
def calculate_skew_row(r, student_id, skew_mod):
    work_units = 40 + (
        (r * 7919 + student_id) % skew_mod
    ) ** 2

    score = 0.0

    for k in range(work_units):
        x = (r + 1) * 0.001 + (k + 1) * 0.0001

        score += (
            math.sin(x) * math.cos(x)
            + math.sqrt(abs(x) + 1.0)
        )

    return score, work_units


# ------------------------------------------------------------
# TASK 3.1: STATIC PARTITIONING
# ------------------------------------------------------------

@njit(parallel=True)
def static_skew_omp(rows, student_id, skew_mod):
    n_threads = get_num_threads()

    out = np.zeros(rows, dtype=np.float64)

    ops = np.zeros(
        (n_threads, PAD_STRIDE),
        dtype=np.int64
    )

    for r in prange(rows):
        score, work_units = calculate_skew_row(
            r, student_id, skew_mod
        )

        out[r] = score

        tid = get_thread_id()
        ops[tid, 0] += work_units



    """Deterministic Dirichlet boundary conditions; zero interior."""
    grid = np.zeros((height, width), dtype=np.float64)
    # Fixed boundary values on all four edges.
    grid[0, :] = 1.0
    grid[-1, :] = 0.0
    grid[:, 0] = 0.0
    grid[:, -1] = 0.0
    return grid


def run_sor_solver(height, width, omega, tolerance=1e-6,
                   max_iterations=20000):
    grid = make_sor_grid(height, width)
    residual = float("inf")
    iterations = 0

    while iterations < max_iterations and residual > tolerance:
        red_black_sor_omp(grid, omega, 0)
        red_black_sor_omp(grid, omega, 1)
        iterations += 1
        residual = sor_residual_omp(grid)

    return grid, iterations, residual, (residual <= tolerance)


print("\nTASK 4 — RED-BLACK SOR PDE SOLVER")
print("Grid:", f"{GRID_H} x {GRID_W}")
print("Omega:", float(OMEGA_SOR))
print("Convergence tolerance: 1e-6")
print("Maximum iterations: 20000")

# Compile/warm up outside the measured runs.
set_num_threads(1)
_warm_grid = make_sor_grid(16, 16)
red_black_sor_omp(_warm_grid, OMEGA_SOR, 0)
red_black_sor_omp(_warm_grid, OMEGA_SOR, 1)
_ = sor_residual_omp(_warm_grid)

sor_times = {}
sor_iterations = {}
sor_residuals = {}
sor_grids = {}
sor_validations = {}

for thread_count in (1, 2, 4, 8):
    set_num_threads(thread_count)
    start = time.perf_counter()
    grid_result, iterations, residual, converged = run_sor_solver(
        GRID_H, GRID_W, OMEGA_SOR
    )
    elapsed = time.perf_counter() - start

    sor_times[thread_count] = elapsed
    sor_iterations[thread_count] = iterations
    sor_residuals[thread_count] = residual
    sor_grids[thread_count] = grid_result.copy()
    sor_validations[thread_count] = converged

    print(
        f"Threads: {thread_count} | "
        f"Time: {elapsed:.6f}s | "
        f"Iterations: {iterations} | "
        f"Residual: {residual:.8e} | "
        f"Converged: {converged}"
    )

# Compare parallel results against the one-thread reference.
sor_reference = sor_grids[1]
for thread_count in (2, 4, 8):
    difference = float(np.max(np.abs(
        sor_reference - sor_grids[thread_count]
    )))
    print(
        f"Grid difference vs 1 thread ({thread_count} threads): "
        f"{difference:.8e}"
    )
    assert difference < 1e-10, (
        f"SOR grid differs at {thread_count} threads"
    )

assert all(sor_validations.values()), (
    "SOR did not converge for every tested thread count"
)
assert len(set(sor_iterations.values())) == 1, (
    "SOR iteration counts differ across thread counts"
)

print("\nTask 4 scaling:")
base_sor_time = sor_times[1]
for thread_count in (1, 2, 4, 8):
    speedup = base_sor_time / sor_times[thread_count]
    print(
        f"{thread_count} threads | "
        f"{sor_times[thread_count]:.6f}s | "
        f"Speedup: {speedup:.3f}x"
    )

DIAGNOSIS_T4 = (
    "The Red-Black SOR method solves the discretized Laplace equation "
    "iteratively. The grid is divided into red and black checkerboard "
    "cells. Cells of one color can be updated in parallel because their "
    "nearest-neighbor dependencies belong to the opposite color. The "
    "red phase is completed before the black phase, preserving the "
    "Gauss-Seidel dependency order while allowing parallel work within "
    "each phase. The relaxation factor controls the update magnitude. "
    "The residual measures how closely the interior grid satisfies the "
    "discrete Laplace equation. Runtime scaling is empirical and depends "
    "on thread scheduling, memory traffic, and the processor."
)
print("\nDIAGNOSIS_T4:")
print(DIAGNOSIS_T4)






# ============================================================
# FINAL VALIDATION, 20-CHARACTER DIGEST, AND RAW LOG
# ============================================================

assert nb.threading_layer() == "omp"
assert all(np.array_equal(
    safe_results[1][0], safe_results[t][0]
) for t in (2, 4, 8)), "Task 1 histogram mismatch"
assert all(np.allclose(
    safe_results[1][1], safe_results[t][1],
    rtol=1e-12, atol=1e-10
) for t in (2, 4, 8)), "Task 1 weighted-sum mismatch"

assert all(
    t in unpadded_times and t in padded_times
    for t in (1, 2, 4, 8)
), "Missing Task 2 thread-count timing"
assert all(sor_validations.values()), "Task 4 convergence check failed"

# Build the digest from this run's actual measurements and numerical output.
digest_payload = {
    "student_id": STUDENT_ID,
    "backend": nb.threading_layer(),
    "parameters": {
        "N_T1": N_T1, "BINS_T1": BINS_T1,
        "ITERS_T2": ITERS_T2, "PAD_STRIDE": PAD_STRIDE,
        "ROWS_T3": ROWS_T3, "SKEW_MOD": SKEW_MOD,
        "GRID_H": GRID_H, "GRID_W": GRID_W,
        "OMEGA_SOR": float(OMEGA_SOR)
    },
    "race_lost_counts": lost_counts,
    "task2_times": {
        str(t): [unpadded_times[t], padded_times[t]]
        for t in (1, 2, 4, 8)
    },
    "task3_times": [static_time, balanced_time],
    "task4_times": {
        str(t): sor_times[t] for t in (1, 2, 4, 8)
    },
    "task4_iterations": sor_iterations,
    "task4_residuals": sor_residuals,
    "task4_grid_sha256": hashlib.sha256(
        sor_grids[1].tobytes()
    ).hexdigest()
}
verification_digest = hashlib.sha256(
    json.dumps(digest_payload, sort_keys=True).encode("utf-8")
).hexdigest()[:20]

print("\nFINAL SUBMISSION CHECKLIST")
print("OpenMP backend:", nb.threading_layer())
print("DIAGNOSIS_T1 present:", bool(DIAGNOSIS_T1))
print("DIAGNOSIS_T2 present:", bool(DIAGNOSIS_T2))
print("DIAGNOSIS_T3 present:", bool(DIAGNOSIS_T3))
print("DIAGNOSIS_T4 present:", bool(DIAGNOSIS_T4))
print("Empirical timings recorded for 1, 2, 4, 8 threads: True")
print("Task 4 converged at all thread counts:", all(sor_validations.values()))
print("verification_digest:", verification_digest)
assert len(verification_digest) == 20

# Preserve real measurements from this execution in the required logfile.
log_lines = [
    f"Student: {STUDENT_NAME}",
    f"Student ID: {STUDENT_ID}",
    f"Backend: {nb.threading_layer()}",
    f"Logical CPU count: {os.cpu_count()}",
    "",
    "TASK 1 race-condition runs:",
    *[str(item) for item in race_results],
    "",
    "TASK 2 timings (threads, unpadded_s, padded_s):",
    *[
        f"{t}, {unpadded_times[t]:.9f}, {padded_times[t]:.9f}"
        for t in (1, 2, 4, 8)
    ],
    "",
    "TASK 3 timings:",
    f"static_s={static_time:.9f}",
    f"balanced_s={balanced_time:.9f}",
    f"static_imbalance={static_imbalance:.6f}",
    f"balanced_imbalance={balanced_imbalance:.6f}",
    "",
    "TASK 4 timings (threads, seconds, iterations, residual):",
    *[
        f"{t}, {sor_times[t]:.9f}, {sor_iterations[t]}, "
        f"{sor_residuals[t]:.12e}"
        for t in (1, 2, 4, 8)
    ],
    "",
    f"verification_digest={verification_digest}",
]
log_filename = f"results_{STUDENT_ID}.txt"
with open(log_filename, "w", encoding="utf-8") as log_file:
    log_file.write("\n".join(log_lines) + "\n")

print("Raw logfile written:", log_filename)
print("\nALL FINAL ASSERTIONS PASSED.")

