"""Run each case/backend pair in a fresh pinned worker. Run with Python 3.13 on Linux."""

import argparse
import contextlib
import importlib
import itertools
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def measure(module, case, side, trial):
    import numpy as np

    suite = importlib.import_module(module)
    run = suite.prepare(case, side)
    if side == "statadic" and "jax" in sys.modules:
        sys.modules["jax"].effects_barrier()
    if side == "reference" and "jax" in sys.modules:
        raise RuntimeError("Reference imported JAX")
    baseline = next(
        int(line.split()[1])
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    ms = []
    for _ in range(6):
        then = time.perf_counter_ns()
        result = np.asarray(run(), dtype=np.float64)
        ms.append((time.perf_counter_ns() - then) / 1e6)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite output")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    key = f"{module}.{case}.{side}.{trial}"
    np.save(HERE / "results" / (key + ".npy"), result)
    record = dict(
        module=module,
        case=case,
        side=side,
        trial=trial,
        ms=ms,
        rss_kib=rss,
        baseline_kib=baseline,
        cpu=list(os.sched_getaffinity(0)),
    )
    (HERE / "results" / (key + ".json")).write_text(json.dumps(record))


def schedule(cpu, trials, requests):
    for trial, item in itertools.product(trials, requests):
        module, case = item.split("/")
        for side in (
            ("statadic", "reference")
            if (trial + requests.index(item)) % 2
            else ("reference", "statadic")
        ):
            key = f"{module}.{case}.{side}.{trial}"
            if (HERE / "results" / (key + ".json")).exists():
                continue
            env = dict(
                os.environ,
                JAX_ENABLE_X64="1",
                JAX_PLATFORMS="cpu",
                JAX_ENABLE_COMPILATION_CACHE="0",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                BLIS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1",
                PYTHONHASHSEED="0",
                MPLBACKEND="Agg",
            )
            python = (
                os.getenv("STATADIC_REFERENCE_PYTHON", sys.executable)
                if side == "reference"
                else sys.executable
            )
            cmd = [
                "taskset",
                "-c",
                str(cpu),
                python,
                str(HERE / "measure.py"),
                "worker",
                module,
                case,
                side,
                str(trial),
            ]
            with (HERE / "logs" / (key + ".log")).open("w") as f:
                try:
                    r = subprocess.run(cmd, env=env, stdout=f, stderr=f, timeout=180)
                    print(key, r.returncode, flush=True)
                except subprocess.TimeoutExpired:
                    print(key, "timeout", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        measure(*sys.argv[2:5], int(sys.argv[5]))
    else:
        schedule(
            int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
        )
