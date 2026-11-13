"""Retain isolated measurements and compare only compatible campaigns."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from statadic_evidence import (
    fingerprint,
    inventory,
    sha256,
    verify_inventory,
    write_archive,
    write_json,
)

from statadic_performance.identities import WORKLOADS, variant_names

MATCH_FIELDS = (
    "workload_fingerprint",
    "harness_fingerprint",
    "dtype",
    "backend",
    "hardware",
    "devices",
    "versions",
    "python",
    "cache_policy",
    "threads",
    "compiler_settings",
    "memory_only",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def source_inventory(root: Path) -> dict[str, str]:
    """Re-enumerate measured sources so additions cannot escape the final guard."""
    sources = [
        *sorted((root / "src/python/packages/statadic/src").rglob("*.py")),
        *sorted((root / "src/python/apps/performance/src").rglob("*.py")),
        root / "src/python/apps/performance/src/statadic_performance/identities.json",
    ]
    return {str(path.relative_to(root)): sha256(path) for path in sources}


def verify(path: Path) -> dict:
    manifest = load(path / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Performance campaign is incomplete")
    verify_inventory(path, manifest["files"])
    expected = {
        f"{name}-{dtype}-{session}.json"
        for name in manifest["workloads"]
        for dtype in manifest["dtypes"]
        for session in range(manifest["sessions"])
    }
    if set(manifest["records"]) != expected or len(manifest["records"]) != len(
        expected
    ):
        raise ValueError("Missing or unexpected measurement records")
    expected_memory = {
        f"{name}-{dtype}-memory.json"
        for name in manifest["workloads"]
        for dtype in manifest["dtypes"]
    }
    if set(manifest["memory_records"]) != expected_memory or len(
        manifest["memory_records"]
    ) != len(expected_memory):
        raise ValueError("Missing or unexpected isolated memory records")
    for record in manifest["records"]:
        data = load(path / record)
        for variant in data["variants"]:
            if variant["numerical_witness"]["status"] != "passed":
                raise ValueError("Numerical witness did not pass")
            if len(variant["warm_seconds"]) != manifest["samples"]:
                raise ValueError("Incomplete retained timing distribution")
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in variant["warm_seconds"]
            ):
                raise ValueError("Invalid timing distribution")
    for record in manifest["memory_records"]:
        data = load(path / record)
        if not data["memory_only"] or data["process_peak_rss_bytes"] <= 0:
            raise ValueError("Invalid isolated memory evidence")
    expected_cold = {
        f"{name}-{dtype}-cold-{variant}-{session}.json"
        for name in manifest["workloads"]
        for dtype in manifest["dtypes"]
        for variant in manifest["variant_names"][name]
        for session in range(manifest["sessions"])
    }
    if set(manifest["cold_records"]) != expected_cold or len(
        manifest["cold_records"]
    ) != len(expected_cold):
        raise ValueError("Missing or unexpected cold-call records")
    for record in manifest["cold_records"]:
        data = load(path / record)
        if data["measurement_mode"] != "cold-jit-call" or len(data["variants"]) != 1:
            raise ValueError("Cold-call latency requires one variant in a fresh worker")
        variant = data["variants"][0]
        if (
            variant["cold_jit_call_seconds"] <= 0
            or variant["numerical_witness"]["status"] != "passed"
        ):
            raise ValueError("Invalid cold-call numerical or timing evidence")
    return manifest


def run(output: Path, *, smoke: bool, profile: bool = False) -> dict:
    if output.exists():
        raise ValueError(
            "Choose a new output directory; retained history is never overwritten"
        )
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    lock = root / "src/python/uv.lock"
    output.mkdir(parents=True)
    dtypes = ["float32", "float64"]
    sessions = 1 if smoke else 3
    source_files = source_inventory(root)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started": datetime.now(UTC).isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "source_files": source_files,
        "source_fingerprint": fingerprint(source_files),
        "lock_sha256": sha256(lock),
        "workloads": list(WORKLOADS),
        "dtypes": dtypes,
        "sessions": sessions,
        "samples": 2 if smoke else 30,
        "warmups": 1 if smoke else 5,
        "smoke": smoke,
        "records": [],
        "memory_records": [],
        "cold_records": [],
        "variant_names": variant_names(smoke),
    }
    write_json(output / "manifest.json", manifest)
    environment = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(key, None)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_COMPILATION_CACHE="false",
        JAX_DISABLE_MOST_OPTIMIZATIONS="0",
    )
    try:
        for name in WORKLOADS:
            for dtype in dtypes:
                for session in range(sessions + 1):
                    memory = session == sessions
                    filename = f"{name}-{dtype}-{'memory' if memory else session}.json"
                    command = [
                        sys.executable,
                        "-m",
                        "statadic_performance.worker",
                        name,
                        "--dtype",
                        dtype,
                        "--output",
                        str(output.resolve() / filename),
                    ]
                    if smoke:
                        command.append("--smoke")
                    if memory:
                        command.append("--memory-only")
                    if profile and session == 0:
                        command += [
                            "--profile",
                            str(output.resolve() / "traces" / f"{name}-{dtype}"),
                        ]
                    with tempfile.TemporaryDirectory(
                        prefix="statadic-performance-cache-"
                    ) as cache:
                        with (output / filename.replace(".json", ".log")).open(
                            "w"
                        ) as log:
                            subprocess.run(
                                command,
                                env=environment
                                | {
                                    "JAX_ENABLE_X64": "1"
                                    if dtype == "float64"
                                    else "0",
                                    "JAX_COMPILATION_CACHE_DIR": cache,
                                },
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                    imported_source = load(output / filename)["statadic_source"]
                    expected_source = (
                        root / "src/python/packages/statadic/src/statadic/__init__.py"
                    )
                    if Path(imported_source).resolve() != expected_source.resolve():
                        raise ValueError(
                            "Measured STATADIC import does not match "
                            "recorded source tree"
                        )
                    manifest["memory_records" if memory else "records"].append(filename)
                    write_json(output / "manifest.json", manifest)
                    if not memory:
                        for variant in manifest["variant_names"][name]:
                            cold_filename = (
                                f"{name}-{dtype}-cold-{variant}-{session}.json"
                            )
                            cold_command = [
                                sys.executable,
                                "-m",
                                "statadic_performance.worker",
                                name,
                                "--dtype",
                                dtype,
                                "--output",
                                str(output.resolve() / cold_filename),
                                "--cold-variant",
                                variant,
                            ]
                            if smoke:
                                cold_command.append("--smoke")
                            with tempfile.TemporaryDirectory(
                                prefix="statadic-cold-cache-"
                            ) as cache:
                                with (
                                    output / cold_filename.replace(".json", ".log")
                                ).open("w") as log:
                                    subprocess.run(
                                        cold_command,
                                        env=environment
                                        | {
                                            "JAX_ENABLE_X64": "1"
                                            if dtype == "float64"
                                            else "0",
                                            "JAX_COMPILATION_CACHE_DIR": cache,
                                        },
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                        check=True,
                                    )
                            cold_source = load(output / cold_filename)[
                                "statadic_source"
                            ]
                            if Path(cold_source).resolve() != expected_source.resolve():
                                raise ValueError(
                                    "Cold worker imported a different STATADIC source"
                                )
                            manifest["cold_records"].append(cold_filename)
                            write_json(output / "manifest.json", manifest)
        if (
            source_files != source_inventory(root)
            or sha256(lock) != manifest["lock_sha256"]
        ):
            raise ValueError("Performance inputs changed during measurement")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished"] = datetime.now(UTC).isoformat()
        manifest["files"] = inventory(output)
        write_json(output / "manifest.json", manifest)
    verify(output)
    write_archive(output, output.with_name(output.name + ".tar"))
    return manifest


def compare(baseline: Path, candidate: Path) -> dict:
    left, right = verify(baseline), verify(candidate)
    for key in (
        "lock_sha256",
        "workloads",
        "dtypes",
        "sessions",
        "samples",
        "warmups",
        "smoke",
        "variant_names",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched campaign field: {key}")
    if set(left["records"]) != set(right["records"]):
        raise ValueError("Unmatched session inventory")
    rows = []
    for name in left["records"]:
        before, after = load(baseline / name), load(candidate / name)
        for key in MATCH_FIELDS:
            if key not in before or key not in after or before[key] != after[key]:
                raise ValueError(f"Unmatched measurement field: {key}: {name}")
        if [v["variant"] for v in before["variants"]] != [
            v["variant"] for v in after["variants"]
        ]:
            raise ValueError("Unmatched variants")
        for first, second in zip(before["variants"], after["variants"], strict=True):
            old, new = (
                statistics.median(first["warm_seconds"]),
                statistics.median(second["warm_seconds"]),
            )
            rows.append(
                {
                    "record": name,
                    "variant": first["variant"],
                    "baseline_median_seconds": old,
                    "candidate_median_seconds": new,
                    "candidate_over_baseline": new / old if old else None,
                    "phases": {
                        phase: {"baseline": first[phase], "candidate": second[phase]}
                        for phase in (
                            "transfer_seconds",
                            "lowering_seconds",
                            "compilation_seconds",
                            "first_call_seconds",
                            "output_host_materialization_seconds",
                        )
                    },
                    "compiler_memory": {
                        "baseline": first["compiler_memory"],
                        "candidate": second["compiler_memory"],
                    },
                }
            )
    memory_rows = []
    for name in left["memory_records"]:
        before, after = load(baseline / name), load(candidate / name)
        for key in MATCH_FIELDS:
            if key not in before or key not in after or before[key] != after[key]:
                raise ValueError(f"Unmatched memory field: {key}: {name}")
        memory_rows.append(
            {
                "record": name,
                "baseline_process_peak_rss_bytes": before["process_peak_rss_bytes"],
                "candidate_process_peak_rss_bytes": after["process_peak_rss_bytes"],
            }
        )
    cold_rows = []
    if set(left["cold_records"]) != set(right["cold_records"]):
        raise ValueError("Unmatched cold session inventory")
    for name in left["cold_records"]:
        before, after = load(baseline / name), load(candidate / name)
        for key in MATCH_FIELDS:
            if key not in before or key not in after or before[key] != after[key]:
                raise ValueError(f"Unmatched cold measurement field: {key}: {name}")
        cold_rows.append(
            {
                "record": name,
                "baseline_cold_jit_call_seconds": before["variants"][0][
                    "cold_jit_call_seconds"
                ],
                "candidate_cold_jit_call_seconds": after["variants"][0][
                    "cold_jit_call_seconds"
                ],
            }
        )
    return {
        "schema_version": 1,
        "baseline_revision": left["revision"],
        "candidate_revision": right["revision"],
        "interpretation": (
            "Matched per-session timing ratios; descriptive evidence, "
            "not an automatic optimization claim"
        ),
        "comparisons": rows,
        "memory_comparisons": memory_rows,
        "cold_call_comparisons": cold_rows,
    }


def report(path: Path) -> str:
    manifest = verify(path)
    lines = [
        "# CPU performance evidence",
        "",
        f"Revision: `{manifest['revision']}`.",
        "",
        "Warm measurements are elapsed distributions; "
        "memory is whole-worker RSS and compiler estimates.",
        "",
        "| Session | Variant | Median warm seconds |",
        "| --- | --- | ---: |",
    ]
    for name in manifest["records"]:
        record = load(path / name)
        for variant in record["variants"]:
            lines.append(
                f"| {name} | {variant['variant']} | "
                f"{statistics.median(variant['warm_seconds']):.6g} |"
            )
    lines += [
        "",
        "| Isolated worker | Whole-process peak RSS bytes |",
        "| --- | ---: |",
    ]
    for name in manifest["memory_records"]:
        record = load(path / name)
        lines.append(f"| {name} | {record['process_peak_rss_bytes']} |")
    lines += ["", "| Fresh worker | Combined cold JIT-call seconds |", "| --- | ---: |"]
    for name in manifest["cold_records"]:
        variant = load(path / name)["variants"][0]
        lines.append(f"| {name} | {variant['cold_jit_call_seconds']:.6g} |")
    lines += [
        "",
        "Compiler memory availability and allocation estimates "
        "are retained in each raw variant record.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    for name in ("smoke", "bench", "profile"):
        command = commands.add_parser(name)
        command.add_argument(
            "--output",
            type=Path,
            default=Path("build/performance")
            / f"{name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}",
        )
    command = commands.add_parser("compare")
    command.add_argument("baseline", type=Path)
    command.add_argument("candidate", type=Path)
    command.add_argument("--output", type=Path)
    command = commands.add_parser("report")
    command.add_argument("campaign", type=Path)
    command.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(Path(__file__).resolve().parents[2] / "tests"),
            ],
            check=True,
        )
    elif args.command in {"smoke", "bench", "profile"}:
        run(
            args.output,
            smoke=args.command == "smoke",
            profile=args.command == "profile",
        )
    elif args.command == "compare":
        if args.output and any(
            args.output.resolve().is_relative_to(source.resolve())
            for source in (args.baseline, args.candidate)
        ):
            raise ValueError("Write comparisons outside immutable campaign evidence")
        result = compare(args.baseline, args.candidate)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2))
    else:
        text = report(args.campaign)
        if args.output:
            if args.output.resolve().is_relative_to(args.campaign.resolve()):
                raise ValueError("Write reports outside immutable campaign evidence")
            # Replace the destination inode instead of truncating an external
            # hard link to retained evidence.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=args.output.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(text)
                os.replace(temporary, args.output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            print(text, end="")


if __name__ == "__main__":
    main()
