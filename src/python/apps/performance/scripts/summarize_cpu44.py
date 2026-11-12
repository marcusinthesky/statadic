"""Verify two CPU44 replays and export their trial-median comparison and raw timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
from replay_cpu44 import FIXTURES, MODULES

SIDES = ("statadic", "reference")
ROLES = ("parent", "candidate")
METRICS = ("first_ms", "warm_ms", "peak_mib")
ORIGINAL_POT_CONTROL = (
    "f8e27da4ba158a46f70673cebadafd8e9c9cdb5be0b6dfd69c107eab7abcfd31"
)
ORIGINAL_ZIP = "3fe52498a31116ccee03e5d4aa170c2dc346ea19101c639282d7271b057d30ad"


def inventory() -> dict:
    # Load labels and tolerances from the repository fixture code, not a result
    # directory whose Python files have not yet been verified.
    query = (
        "import json; from collect import GROUPS, metadata; "
        "print(json.dumps({m+'/'+k:metadata(m,k) "
        "for m,ks in GROUPS for k in ks.split()}))"
    )
    cases = json.loads(
        subprocess.check_output([sys.executable, "-c", query], cwd=FIXTURES)
    )
    if len(cases) != 44:
        raise ValueError("Expected exactly 44 distinct pinned workloads")
    return cases


def read_manifest(directory: Path, cases: dict) -> tuple[dict, dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if not re.fullmatch("[0-9a-f]{40}", manifest["revision"]):
        raise ValueError(f"Invalid source revision in {directory}")
    if (manifest["trials"], manifest["warm_calls"]) != (3, 5):
        raise ValueError(f"Wrong trial or call count in {directory}")
    hashes = {}
    for name in MODULES:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        tracked = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        original_pot = name == "pot_control.py" and digest == ORIGINAL_POT_CONTROL
        if digest != tracked and not original_pot:
            raise ValueError(f"Changed fixture or lock in {directory}: {name}")
        hashes[name] = digest
    if "fixture_sha256" in manifest:
        if manifest["fixture_sha256"] != hashes:
            raise ValueError(f"Invalid fixture digest manifest in {directory}")
    elif manifest.get("supplement_sha256") != ORIGINAL_ZIP:
        raise ValueError(f"Unrecognized original supplement in {directory}")

    pins = set((FIXTURES / "lock-constraints.txt").read_text().splitlines())
    environments = {}
    for name in ("main", "peer"):
        raw = (directory / f"{name}-environment.txt").read_bytes()
        lines = raw.decode().splitlines()
        # The editable/local STATADIC build is supplied by the committed source
        # recorded in manifest.revision. Third-party packages follow the lock.
        installed = {s for s in lines if s and not s.startswith("statadic @ file:")}
        required = (
            {
                "jax",
                "jaxlib",
                "spreg",
                "lifelines",
                "linearmodels",
                "glum",
                "arch",
                "scikit-learn",
                "numpy",
                "scipy",
                "statsmodels",
                "dcor",
                "pot",
            }
            if name == "main"
            else {"scikit-bio", "pyportfolioopt", "numpy"}
        )
        if installed - pins or not required <= {
            s.partition("==")[0] for s in installed
        }:
            raise ValueError(
                f"Missing or unlocked packages in {directory}/{name}-environment.txt"
            )
        environments[name] = hashlib.sha256(raw).hexdigest()

    assignments = [
        (case, int(cpu)) for cpu, ids in manifest["lanes"].items() for case in ids
    ]
    cpus = dict(assignments)
    if (
        len(assignments) != len(cases)
        or set(cpus) != set(cases)
        or min(cpus.values()) < 0
    ):
        raise ValueError(f"Invalid case-to-CPU assignment in {directory}")
    expected = {
        f"{case.replace('/', '.')}.{side}.{trial}.{ext}"
        for case in cases
        for side in SIDES
        for trial in range(3)
        for ext in ("json", "npy")
    }
    if {p.name for p in (directory / "results").iterdir()} != expected:
        raise ValueError(f"Missing or unexpected worker results in {directory}")
    return manifest, cpus, environments


def read_worker(
    directory: Path, case_id: str, side: str, trial: int, cpu: int
) -> tuple[dict, np.ndarray]:
    module, case = case_id.split("/")
    base = directory / "results" / f"{module}.{case}.{side}.{trial}"
    row = json.loads(Path(f"{base}.json").read_text())
    if (row["module"], row["case"], row["side"], row["trial"], row["cpu"]) != (
        module,
        case,
        side,
        trial,
        [cpu],
    ):
        raise ValueError(f"Worker identity or affinity mismatch in {base}")
    times = row["ms"]
    if len(times) != 6 or any(
        type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times
    ):
        raise ValueError(f"Expected six finite, positive timings in {base}")
    if any(type(row[k]) is not int or row[k] <= 0 for k in ("baseline_kib", "rss_kib")):
        raise ValueError(f"Invalid RSS record in {base}")
    output = np.load(f"{base}.npy", allow_pickle=False)
    if (
        output.dtype != np.dtype("float64")
        or not output.size
        or not np.isfinite(output).all()
    ):
        raise ValueError(f"Invalid numerical output in {base}")
    return row, output


def error(actual: np.ndarray, target: np.ndarray, meta: dict, label: str) -> float:
    if actual.shape != target.shape or not np.allclose(
        actual, target, rtol=meta["rtol"], atol=meta["atol"]
    ):
        raise ValueError(f"Numerical comparison failed for {label}")
    return float(np.max(np.abs(actual - target)))


def order(ratios: list[float]) -> str:
    if all(v < 1 for v in ratios):
        return "faster"
    if all(v > 1 for v in ratios):
        return "slower"
    return "equal" if all(v == 1 for v in ratios) else "mixed"


def compare_case(
    case: str, meta: dict, dirs: tuple[Path, Path], revisions: list[str], cpu: int
) -> tuple[dict, list]:
    summary = {
        "case_id": case,
        **{
            k: meta[k]
            for k in (
                "category",
                "method",
                "reference",
                "fixture",
                "work",
                "rtol",
                "atol",
                "source",
            )
        },
    }
    rows = []
    warm = {role: {side: [] for side in SIDES} for role in ROLES}
    for trial in range(3):
        records = {
            (role, side): read_worker(folder, case, side, trial, cpu)
            for role, folder in zip(ROLES, dirs, strict=True)
            for side in SIDES
        }
        for side in SIDES:
            change = error(
                records["candidate", side][1],
                records["parent", side][1],
                meta,
                f"{case}, trial {trial}, {side} candidate vs parent",
            )
            for role, revision in zip(ROLES, revisions, strict=True):
                data, actual = records[role, side]
                paired = error(
                    records[role, "statadic"][1],
                    records[role, "reference"][1],
                    meta,
                    f"{case}, trial {trial}, {role} STATADIC vs reference",
                )
                ms = data["ms"]
                warm[role][side].append(median(ms[1:]))
                rows.append(
                    {
                        "case_id": case,
                        "run": role,
                        "revision": revision,
                        "side": side,
                        "trial": trial,
                        "cpu": cpu,
                        "first_ms": ms[0],
                        **{f"warm_{i}_ms": v for i, v in enumerate(ms[1:], 1)},
                        "baseline_kib": data["baseline_kib"],
                        "peak_kib": data["rss_kib"],
                        "output_size": actual.size,
                        "statadic_peer_max_abs_error": paired,
                        "candidate_parent_max_abs_error": change,
                    }
                )

    for role in ROLES:
        for side in SIDES:
            group = [r for r in rows if (r["run"], r["side"]) == (role, side)]
            values = (
                median(r["first_ms"] for r in group),
                median(warm[role][side]),
                median(r["peak_kib"] for r in group) / 1024,
            )
            summary.update(
                {f"{role}_{side}_{m}": v for m, v in zip(METRICS, values, strict=True)}
            )
    for target, role, side in (
        ("parent", "parent", "statadic"),
        ("reference", "candidate", "reference"),
    ):
        for metric, key in zip(METRICS, ("first", "warm", "peak"), strict=True):
            summary[f"candidate_over_{target}_{key}"] = (
                summary[f"candidate_statadic_{metric}"]
                / summary[f"{role}_{side}_{metric}"]
            )
        ratios = [
            a / b
            for a, b in zip(
                warm["candidate"]["statadic"], warm[role][side], strict=True
            )
        ]
        summary[f"warm_vs_{target}"] = order(ratios)
    for key in ("statadic_peer_max_abs_error", "candidate_parent_max_abs_error"):
        summary[key] = max(r[key] for r in rows)
    return summary, rows


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Supply a fresh output directory")
    cases = inventory()
    dirs = (args.parent, args.candidate)
    (parent, cpus, env), (candidate, other_cpus, other_env) = (
        read_manifest(p, cases) for p in dirs
    )
    if (
        env != other_env
        or cpus != other_cpus
        or any(parent[k] != candidate[k] for k in ("platform", "cpu_model", "python"))
    ):
        raise ValueError(
            "Parent and candidate environments or assigned CPU lanes differ"
        )
    revisions = [parent["revision"], candidate["revision"]]
    summaries, trials = [], []
    for case, meta in cases.items():
        record, raw = compare_case(case, meta, dirs, revisions, cpus[case])
        summaries.append(record)
        trials.extend(raw)

    counts = {
        "revisions": dict(zip(ROLES, revisions, strict=True)),
        "cases": len(cases),
        "workers": len(trials),
    }
    for target in ("parent", "reference"):
        counts[f"warm_vs_{target}"] = dict(
            Counter(r[f"warm_vs_{target}"] for r in summaries)
        )
        for key in ("first", "warm", "peak"):
            values = [r[f"candidate_over_{target}_{key}"] for r in summaries]
            counts[f"candidate_vs_{target}_{key}"] = {
                "lower": sum(v < 1 for v in values),
                "equal": sum(v == 1 for v in values),
                "higher": sum(v > 1 for v in values),
            }
    args.output.mkdir(parents=True, exist_ok=False)
    write_table(args.output / "summary.csv", summaries)
    write_table(args.output / "trials.csv", trials)
    (args.output / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
