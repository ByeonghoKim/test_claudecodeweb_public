"""
stream_add.py — Stream 64 MB vectors and estimate runtime of just the add step.

Layout
------
1. stream_vectors()   – generator that yields numpy float32 arrays, each exactly 64 MB
2. timed_add()        – adds two vectors and returns (result, elapsed_seconds)
3. main()             – drives the stream, accumulates timing stats, prints a report
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MB = 1024 * 1024
VECTOR_BYTES = 64 * MB                      # 64 MB per vector
DTYPE = np.float32                           # 4 bytes / element
ELEMENTS_PER_VECTOR = VECTOR_BYTES // np.dtype(DTYPE).itemsize   # 16 777 216
NUM_PAIRS = 8                                # how many (a, b) pairs to stream


# ---------------------------------------------------------------------------
# 1. Streaming source
# ---------------------------------------------------------------------------
def stream_vectors(num_pairs: int = NUM_PAIRS):
    """
    Yield (index, a, b) tuples where a and b are 64 MB float32 vectors.

    The generator materialises only one pair at a time so peak memory is
    bounded to ~3 × 64 MB = 192 MB (a, b, and the result buffer).
    """
    rng = np.random.default_rng(seed=42)
    for i in range(num_pairs):
        a = rng.random(ELEMENTS_PER_VECTOR, dtype=DTYPE)
        b = rng.random(ELEMENTS_PER_VECTOR, dtype=DTYPE)
        yield i, a, b


# ---------------------------------------------------------------------------
# 2. Timed add — isolates *just* the addition
# ---------------------------------------------------------------------------
def timed_add(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Add two vectors and return (result, elapsed_seconds).

    The timer wraps only the np.add call so I/O, allocation, and everything
    else is excluded from the measurement.
    """
    t0 = time.perf_counter()
    result = np.add(a, b)          # ← the only line being timed
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# 3. Runtime estimation helpers
# ---------------------------------------------------------------------------
def estimate_remaining(elapsed_so_far: list[float], pairs_left: int) -> float:
    """
    Return a running-average estimate of remaining add time (seconds).

    Uses the mean of all observed add durations so the estimate stabilises
    as more samples arrive.
    """
    if not elapsed_so_far:
        return float("nan")
    mean_per_pair = sum(elapsed_so_far) / len(elapsed_so_far)
    return mean_per_pair * pairs_left


def throughput_gbs(elapsed_sec: float) -> float:
    """Return add throughput in GB/s (two input vectors read + one output written)."""
    bytes_touched = 3 * VECTOR_BYTES        # read a, read b, write result
    return (bytes_touched / elapsed_sec) / (1024 ** 3)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print(f"  Stream-add benchmark")
    print(f"  Vector size : {VECTOR_BYTES / MB:.0f} MB  ({ELEMENTS_PER_VECTOR:,} float32 elements)")
    print(f"  Pairs       : {NUM_PAIRS}")
    print("=" * 62)

    add_times: list[float] = []

    for idx, a, b in stream_vectors(NUM_PAIRS):
        # ── just the add ────────────────────────────────────────────────
        result, elapsed = timed_add(a, b)
        # ────────────────────────────────────────────────────────────────

        add_times.append(elapsed)
        pairs_done = idx + 1
        pairs_left = NUM_PAIRS - pairs_done
        est_remaining = estimate_remaining(add_times, pairs_left)

        print(
            f"  pair {pairs_done:>2}/{NUM_PAIRS} │ "
            f"add = {elapsed * 1000:7.3f} ms │ "
            f"{throughput_gbs(elapsed):5.2f} GB/s │ "
            f"est. remaining add time = "
            + (f"{est_remaining * 1000:.1f} ms" if pairs_left else "done")
        )

        # Discard result immediately — we are benchmarking, not accumulating
        del result

    # ── Summary ─────────────────────────────────────────────────────────
    total   = sum(add_times)
    mean    = total / len(add_times)
    minimum = min(add_times)
    maximum = max(add_times)

    print("-" * 62)
    print(f"  Total add time : {total  * 1000:8.3f} ms")
    print(f"  Mean  add time : {mean   * 1000:8.3f} ms  ({throughput_gbs(mean):.2f} GB/s avg)")
    print(f"  Min   add time : {minimum* 1000:8.3f} ms")
    print(f"  Max   add time : {maximum* 1000:8.3f} ms")
    print("=" * 62)


if __name__ == "__main__":
    main()
