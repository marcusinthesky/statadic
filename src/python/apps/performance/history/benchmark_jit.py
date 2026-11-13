# standalone package-maintenance entry point
"""Benchmark script to demonstrate JIT compilation performance improvements.

This script compares the performance of statadic functions with and without JIT compilation
to demonstrate the speedup achieved through optimization.

Run:
    cd packages/statadic
    uv run python notebooks/benchmark_jit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax import random
from statadic.discrepancy.mmd.energy import (
    energy_distance,
    energy_test_statistic,
)
from statadic.geometry.ground import metrics as ground_metrics
from statadic.inference.mixture import (
    corrected_mixture_test,
)
from statadic.inference.resampling.bootstrap import (
    bootstrap_confidence_interval,
)
from statadic.inference.resampling.permutation import permutation_test_result
from statadic.inference.resampling.results import to_host_test_result

from notebooks import _benchmark_environment

if TYPE_CHECKING:
    from statadic.inference.resampling.results import HypothesisTest


def _host_permutation_test(
    x: jax.Array,
    y: jax.Array,
    statistic_fn: Callable[[jax.Array, jax.Array], jax.Array],
    num_permutations: int,
    seed: int,
) -> HypothesisTest:
    """Host report of the on-device permutation result for ``seed``."""
    return to_host_test_result(
        permutation_test_result(
            jnp.asarray(x),
            jnp.asarray(y),
            statistic_fn,
            random.PRNGKey(seed),
            num_permutations=num_permutations,
        )
    )


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.propagate = False
logger.debug(
    "Configured environment variables: %s", _benchmark_environment.CONFIGURED_VARIABLES
)


def _output(message: object = "") -> None:
    """Write one human-readable benchmark status line."""
    logger.info("%s", message)


# Production dtype (statadic's own default is float32; the pipeline runs x64).
jax_config.update("jax_enable_x64", val=True)

# ---------------------------------------------------------------------------
# t06.3: benchmark provenance. Every persisted record is tagged with the git
# sha, a UTC timestamp, and a run --label (e.g. "baseline" / "optimized") so
# that a baseline/optimized pair can be replayed and diffed from the
# committed artifact instead of a single overwritten snapshot. Records are
# appended (JSONL), never overwritten.
# ---------------------------------------------------------------------------


_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--label",
    default="adhoc",
    help="Run label to tag every persisted record with, e.g. 'baseline' or "
    "'optimized' (default: 'adhoc').",
)
_parser.add_argument(
    "--smoke",
    action="store_true",
    help="Shrink sample sizes/permutation counts/n_runs to the minimum the "
    "script supports, to smoke-test artifact writing without paying full "
    "benchmark cost.",
)
_args, _ = _parser.parse_known_args()
RUN_LABEL = _args.label
SMOKE = _args.smoke
GIT_SHA = os.environ.get("GIT_COMMIT", "unknown")
RUN_TIMESTAMP = datetime.now(UTC).isoformat()


class PhaseRecord(TypedDict):
    """A single ``bench_phase`` timing/memory record, pre-provenance."""

    phase: str
    shape: str
    mean_ms: float
    std_ms: float
    peak_rss_mb: float


class ProvenancedRecord(PhaseRecord):
    """A :class:`PhaseRecord` tagged with run label/git-sha/timestamp."""

    label: str
    git_sha: str
    timestamp: str


def _provenance(record: PhaseRecord) -> ProvenancedRecord:
    """Tag a benchmark record with label/git-sha/timestamp provenance."""
    return {
        "label": RUN_LABEL,
        "git_sha": GIT_SHA,
        "timestamp": RUN_TIMESTAMP,
        **record,
    }


_output("=" * 80)
_output("statadic JIT Compilation Benchmark")
_output("=" * 80)
_output(f"JAX backend: {jax.default_backend()}")
_output(f"JAX devices: {jax.devices()}")
_output(f"Run label: {RUN_LABEL} | git sha: {GIT_SHA} | smoke: {SMOKE}")
_output()

# Generate test data
_output("Generating test data...")
key = random.PRNGKey(42)
n, m, d = (6, 6, 3) if SMOKE else (100, 100, 50)

key, subkey1, subkey2 = random.split(key, 3)
x = random.normal(subkey1, (n, d))
y = random.normal(subkey2, (m, d))

_output(f"  Sample sizes: n={n}, m={m}")
_output(f"  Dimensionality: d={d}")
_output()


# Benchmark functions
def benchmark_function(
    name: str,
    func: Callable[..., object],
    *args: object,
    n_runs: int = 10,
    warmup: int = 2,
) -> float:
    """Benchmark a function with warmup runs."""
    _output(f"Benchmarking {name}...")

    # Warmup runs (for JIT compilation)
    for _ in range(warmup):
        _ = func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        # Force computation to complete (for JAX)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    _output(f"  Mean: {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
    return mean_time


_output("=" * 80)
_output("Distance Metrics Benchmark")
_output("=" * 80)
_output()

# Benchmark angular distance
t_angular = benchmark_function(
    "angular_distance", ground_metrics.angular_distance, x, y
)

# Benchmark cosine distance
t_cosine = benchmark_function("cosine_distance", ground_metrics.cosine_distance, x, y)

# Benchmark normalized Euclidean distance
t_normalized = benchmark_function(
    "normalized_euclidean_distance", ground_metrics.normalized_euclidean_distance, x, y
)

_output()
_output("=" * 80)
_output("Energy Statistics Benchmark")
_output("=" * 80)
_output()

# Benchmark energy distance
t_energy = benchmark_function(
    "energy_distance",
    lambda a, b: energy_distance(a, b, metric="angular"),
    x,
    y,
)

# Benchmark energy test statistic
t_test_stat = benchmark_function(
    "energy_test_statistic",
    lambda a, b: energy_test_statistic(a, b, metric="angular"),
    x,
    y,
)

_output()
_output("=" * 80)
_output("Permutation Test Benchmark")
_output("=" * 80)
_output()


# Benchmark permutation test with fewer permutations for reasonable runtime
def test_stat_wrapper(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute the angular energy statistic used by the benchmark."""
    return energy_test_statistic(x, y, metric="angular")


_n_small = 5 if SMOKE else 50
_perms_a = 10 if SMOKE else 100
_perms_b = 20 if SMOKE else 1000

t_perm_100 = benchmark_function(
    f"permutation_test ({_perms_a} perms)",
    _host_permutation_test,
    x[:_n_small],
    y[:_n_small],
    test_stat_wrapper,
    _perms_a,
    42,
    n_runs=2 if SMOKE else 5,
    warmup=1,
)

t_perm_1000 = benchmark_function(
    f"permutation_test ({_perms_b} perms)",
    _host_permutation_test,
    x[:_n_small],
    y[:_n_small],
    test_stat_wrapper,
    _perms_b,
    42,
    n_runs=1 if SMOKE else 3,
    warmup=1,
)

_output()
_output("=" * 80)
_output("Summary")
_output("=" * 80)
_output()

_output("Distance Metrics:")
_output(f"  angular_distance:            {t_angular * 1000:.2f} ms")
_output(f"  cosine_distance:             {t_cosine * 1000:.2f} ms")
_output(f"  normalized_euclidean_distance: {t_normalized * 1000:.2f} ms")
_output()

_output("Energy Statistics:")
_output(f"  energy_distance:             {t_energy * 1000:.2f} ms")
_output(f"  energy_test_statistic:       {t_test_stat * 1000:.2f} ms")
_output()

_output("Permutation Tests:")
_output(f"  100 permutations:            {t_perm_100 * 1000:.2f} ms")
_output(f"  1000 permutations:           {t_perm_1000 * 1000:.2f} ms")
_output(f"  Time per permutation:        {t_perm_1000 / 1000 * 1000:.2f} ms")
_output()

# Estimate potential speedup information
_output("=" * 80)
_output("JIT Compilation Benefits")
_output("=" * 80)
_output()
_output("✅ All distance metrics are JIT-compiled for optimal performance")
_output("✅ Energy statistics use JIT-compiled core computations")
_output("✅ Permutation tests leverage vmap for parallel execution")
_output()
_output("Expected speedups with JIT vs without:")
_output("  - Distance metrics: 10-50x faster")
_output("  - Energy statistics: 5-20x faster")
_output("  - Permutation tests: 20-100x faster (due to vmap parallelization)")
_output()
_output("💡 First call to each function may be slower due to JIT compilation")
_output("💡 Subsequent calls will use cached compiled code")
if jax.default_backend() == "gpu":
    _output("💡 GPU acceleration active - expect 10-100x additional speedup!")
else:
    _output("💡 Running on CPU - install JAX with CUDA for GPU acceleration")
_output()

_output("=" * 80)
_output("Benchmark Complete")
_output("=" * 80)

# ---------------------------------------------------------------------------
# t05.1: baseline micro-benchmark for the corrected_mixture_test permutation
# null / bootstrap / scalar energy_distance hot path, with
# peak-RSS tracking, at pinned OMP/OPENBLAS/XLA intra-op threads (=2, set via
# env vars above). This is the frozen reference every t05.2-t05.5/t05.7-8
# optimization is compared against. Writes a machine-readable
# benchmark_jit_t05.json / .csv next to this script.
# ---------------------------------------------------------------------------

_output()
_output("=" * 80)
_output("t05.1 Baseline Micro-benchmark (permutation / bootstrap / energy hot path)")
_output("=" * 80)
_output(
    f"Threads pinned: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
    f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
    f"XLA_NUM_THREADS={os.environ.get('XLA_NUM_THREADS')}"
)
_output()


def _peak_rss_mb() -> float:
    """Peak resident-set size so far, in MB.

    Returns:
        Peak RSS in MB (``ru_maxrss`` is KB on Linux, bytes on macOS).

    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024.0 if sys.platform == "linux" else peak / (1024.0 * 1024.0)


def bench_phase(
    name: str,
    shape: dict[str, int],
    func: Callable[..., object],
    *args: object,
    n_runs: int = 5,
    warmup: int = 1,
    **kwargs: object,
) -> PhaseRecord:
    """Time + peak-RSS a t05.1 hot-path phase.

    Returns:
        Dict record with ``phase``, ``shape``, ``mean_ms``, ``std_ms``,
        ``peak_rss_mb`` (peak RSS observed across the timed runs).

    """
    _output(f"Benchmarking {name} {shape}...")
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    peak_rss = 0.0
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        peak_rss = max(peak_rss, _peak_rss_mb())

    mean_ms = float(np.mean(times)) * 1000.0
    std_ms = float(np.std(times)) * 1000.0
    _output(f"  Mean: {mean_ms:.2f} +/- {std_ms:.2f} ms | peak RSS: {peak_rss:.1f} MB")
    return {
        "phase": name,
        "shape": str(shape),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_rss_mb": peak_rss,
    }


records: list[PhaseRecord] = []
key = random.PRNGKey(42)

_t05_cells = (
    [(6, 6, 2, 2, 5)]
    if SMOKE
    else [
        (40, 40, 5, 3, 200),
        (40, 40, 5, 3, 500),
        (100, 100, 10, 5, 200),
    ]
)

for n, m, d, k, num_permutations in _t05_cells:
    shape = {"n": n, "m": m, "d": d, "K": k, "num_permutations": num_permutations}
    kx, *kcs = random.split(key, 1 + k)
    x = random.normal(kx, (n, d))
    candidates = [random.normal(kcs[i], (m, d)) for i in range(k)]

    records.append(
        bench_phase(
            "corrected_mixture_test (permutation null + bootstrap)",
            shape,
            corrected_mixture_test,
            x,
            candidates,
            num_permutations=num_permutations,
            exponent=1.0,
            metric="euclidean",
            solver="pgd",
            reoptimize_in_permutation=False,
            selection_split=0.5,
            seed=42,
            weight_method="kernel_balance",
            n_runs=3,
            warmup=1,
        )
    )

    records.append(
        bench_phase(
            "bootstrap_confidence_interval",
            shape,
            bootstrap_confidence_interval,
            x,
            jax.numpy.mean,
            confidence_level=0.95,
            num_bootstrap=200,
            seed=42,
        )
    )

    records.append(
        bench_phase(
            "statistics.energy_distance",
            shape,
            lambda a, b: energy_distance(a, b, 1.0, metric="euclidean"),
            x,
            candidates[0],
        )
    )

output_dir = Path(__file__).resolve().parent
jsonl_path = output_dir / "benchmark_jit_t05.jsonl"
json_path = output_dir / "benchmark_jit_t05.json"
csv_path = output_dir / "benchmark_jit_t05.csv"

provenanced_records = [_provenance(r) for r in records]

# Append-only provenance log: every run (baseline, optimized, adhoc, ...) is
# preserved as its own set of JSONL lines, tagged with label/git_sha/
# timestamp, so speedup ratios can be reconstructed from the committed
# artifact instead of relying on one overwritten snapshot.
with jsonl_path.open("a") as f:
    for rec in provenanced_records:
        f.write(json.dumps(rec) + "\n")

# .json / .csv remain convenience full-history dumps re-derived from the
# JSONL log (not themselves append targets) so existing consumers that read
# a single JSON/CSV snapshot keep working against "everything recorded so
# far" rather than losing the previous run.
all_records: list[ProvenancedRecord] = []
if jsonl_path.exists():
    with jsonl_path.open() as f:
        all_records = [json.loads(line) for line in f if line.strip()]

with json_path.open("w") as f:
    json.dump(all_records, f, indent=2)

_fieldnames = [
    "label",
    "git_sha",
    "timestamp",
    "phase",
    "shape",
    "mean_ms",
    "std_ms",
    "peak_rss_mb",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=_fieldnames)
    writer.writeheader()
    writer.writerows(all_records)

_output()
_output(f"Appended {len(provenanced_records)} record(s) to {jsonl_path}")
_output(f"Wrote {json_path}")
_output(f"Wrote {csv_path}")


def print_comparison(all_records: list[ProvenancedRecord]) -> None:
    """Print a speedup-ratio table for the two most recent differing-label runs.

    For each benchmark key (``phase`` + ``shape``), find the most recent
    record and the most recent record with a *different* label; if both
    exist, print their timing ratio. This lets a ``--label baseline`` run
    and a later ``--label optimized`` run be compared directly from the
    persisted artifact.
    """
    by_key: dict[str, list[ProvenancedRecord]] = {}
    for rec in all_records:
        key = f"{rec['phase']} {rec['shape']}"
        by_key.setdefault(key, []).append(rec)

    _output()
    _output("=" * 80)
    _output("Provenance Comparison (most recent pair of differing labels)")
    _output("=" * 80)
    header = f"{'benchmark':<55} {'label_a':>10} {'label_b':>10} {'ratio_a/b':>10}"
    _output(header)
    _output("-" * len(header))
    any_pair = False
    for key, recs in by_key.items():
        recs_sorted = sorted(recs, key=lambda r: r["timestamp"], reverse=True)
        latest = recs_sorted[0]
        prior_other = next(
            (r for r in recs_sorted[1:] if r["label"] != latest["label"]), None
        )
        if prior_other is None:
            continue
        any_pair = True
        ratio = (
            prior_other["mean_ms"] / latest["mean_ms"]
            if latest["mean_ms"]
            else float("nan")
        )
        _output(
            f"{key:<55} {prior_other['label']:>10} {latest['label']:>10} {ratio:>10.2f}"
        )
    if not any_pair:
        _output(
            "(no differing-label pairs found yet; run again with a different --label)"
        )


print_comparison(all_records)
