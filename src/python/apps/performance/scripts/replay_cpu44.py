"""Replay the pinned 44-case supplement on a committed STATADIC source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "history" / "cpu44"
MODULES = (
    "measure.py",
    "collect.py",
    "spatial_cases.py",
    "survival_cases.py",
    "scale_fit.py",
    "econometrics.py",
    "econ_meta.py",
    "more_refs.py",
    "other_refs.py",
    "pot_control.py",
    "lock-constraints.txt",
)
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--peer-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cpus", required=True, help="Comma-separated available Linux CPU IDs"
    )
    args = parser.parse_args()
    cpus = [int(c) for c in args.cpus.split(",")]
    if len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        parser.error("CPU IDs must be distinct and available to the process.")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("source", "logs", "results"):
        (output / name).mkdir()
    git = ["git", "-C", str(args.repo)]
    sha = subprocess.check_output(
        [
            *git,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{args.revision}^{{commit}}",
        ],
        text=True,
    ).strip()
    with (output / "source.tar").open("wb") as file:
        subprocess.run(
            [*git, "archive", sha, "src/python/packages/statadic"],
            stdout=file,
            check=True,
        )
    with tarfile.open(output / "source.tar") as tar:
        tar.extractall(output / "source", filter="data")
    (output / "source.tar").unlink()
    for name in MODULES:
        shutil.copy2(FIXTURES / name, output / name)
    # Read the original collector's own workload inventory in its reference environment.
    inventory = (
        "import json; from collect import GROUPS; "
        "cases = [m + '/' + k for m, keys in GROUPS for k in keys.split()]; "
        "print(json.dumps(cases))"
    )
    cases = json.loads(
        subprocess.check_output(
            [str(args.python.absolute()), "-c", inventory], cwd=output
        )
    )
    if len(cases) != 44 or len(set(cases)) != 44:
        raise ValueError("Unexpected original workload inventory")
    py, peer = str(args.python.absolute()), str(args.peer_python.absolute())
    for interpreter, name in ((py, "main"), (peer, "peer")):
        with (output / f"{name}-environment.txt").open("w") as file:
            subprocess.run(
                ["uv", "pip", "freeze", "--python", interpreter],
                stdout=file,
                check=True,
            )
    # Reuse deterministic case-to-core assignments at every code revision.
    lanes = {cpu: [] for cpu in cpus}
    for case, cpu in zip(cases, cycle(cpus), strict=False):
        lanes[cpu].append(case)
    manifest = {
        "revision": sha,
        "fixture_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in MODULES
        },
        "lanes": lanes,
        "platform": platform.platform(),
        "cpu_model": next(
            (
                v.partition(":")[2].strip()
                for v in Path("/proc/cpuinfo").read_text().splitlines()
                if v.startswith("model name")
            ),
            "unknown",
        ),
        "python": subprocess.check_output([py, "--version"], text=True).strip(),
        "trials": 3,
        "warm_calls": 5,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JAX_", "XLA_", "PYTHONPATH"))
    }
    env.update(dict.fromkeys(THREADS, "1"))
    env.update(
        JAX_ENABLE_X64="1",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_COMPILATION_CACHE="0",
        PYTHONHASHSEED="0",
        MPLBACKEND="Agg",
    )
    package = output / "source/src/python/packages/statadic/src"

    def measure(lane: tuple[int, list[str]]) -> list[str]:
        cpu, assigned = lane
        failures = []
        for trial in range(3):
            for index, case in enumerate(assigned):
                module, key = case.split("/")
                sides = (
                    ("statadic", "reference")
                    if (trial + index) % 2
                    else ("reference", "statadic")
                )
                for side in sides:
                    label = f"{module}.{key}.{side}.{trial}"
                    child = env.copy()
                    interpreter = py
                    if side == "statadic":
                        child["PYTHONPATH"] = str(package)
                    elif case in {"more_refs/skbio-mantel", "more_refs/pypfopt-sample"}:
                        interpreter = peer
                    command = [
                        "taskset",
                        "-c",
                        str(cpu),
                        interpreter,
                        "measure.py",
                        "worker",
                        module,
                        key,
                        side,
                        str(trial),
                    ]
                    with (output / "logs" / f"{label}.log").open("w") as log:
                        try:
                            subprocess.run(
                                command,
                                cwd=output,
                                env=child,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=240,
                                check=True,
                            )
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                        ) as exc:
                            print(f"{label}: {exc}", file=log)
                            failures.append(label)
        return failures

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        failures = [f for group in pool.map(measure, lanes.items()) for f in group]
    if failures:
        raise SystemExit(f"Worker failures (see logs): {failures}")
    subprocess.run([py, "collect.py", "--strict"], cwd=output, env=env, check=True)
    shutil.rmtree(
        output / "source"
    )  # Exact inputs remain available at manifest.revision.


if __name__ == "__main__":
    run()
