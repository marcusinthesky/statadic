"""Retain matched wheel/reference measurements under an explicit resource policy.

The implemented pilot cases are a subset of the alpha workload roster.
They cannot qualify the full scope. Missing cases and absent wheel bindings fail;
strict superiority rejects adverse sessions, while report policy retains losses.
No summary status can replace the raw records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tomllib
import zipfile
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

from statadic_performance.measurement import harness_fingerprint
from statadic_performance.reference_workloads import CASES, fixture

# Exact subcases, rather than a broad family label, own matched outputs. Extend
# this table with reviewed adapters before claiming another workload is covered.
PILOT_CASES = {case: specification["workload"] for case, specification in CASES.items()}
# Each workload is bounded to the enumerated input/output contracts. These
# entries enable measurement, not approval or family-wide scientific coverage.
# Full 500x20 joint SAR-AFT fit plus identified information has no matched
# reference adapter yet, so the complete alpha scope still fails closed.
QUALIFICATION_CASES: dict[str, tuple[str, ...]] = {
    "linear-fit-and-inference": ("ols-hc3", "wls-hc3"),
    "canonical-glm": ("binomial-logit", "poisson-offset"),
    "dispersion-likelihood": ("gamma-estimated", "nb2-estimated"),
    "covariance-shrinkage": ("constant-correlation-shrinkage",),
    "dyadic-node-refits": ("dyadic-node-refits", "dyadic-effects-refits"),
    "affine-gmm": ("affine-gmm-actual-weight", "affine-gmm-iterated-hac"),
    "weighted-mmd-energy": (
        "weighted-rbf-value-gradient",
        "weighted-energy-value-gradient",
    ),
    "empirical-transport": ("quantile-transport-unequal",),
    "measure-projections": ("simplex-energy-projection", "kernel-target-projection"),
    "qp-gmv": ("capped-qp",),
    "supplied-row-refits": ("row-pairs-refits",),
    "spatial-operators": ("slx-classical", "sar-gaussian-ml"),
    "spatial-panel": ("spatial-panel-sar-within", "observed-spatial-panel-sar"),
    "ordinary-aft": (
        "weibull-aft",
        "lognormal-aft",
        "weibull-aft-entry",
        "lognormal-aft-entry",
    ),
}
SIDES = ("statadic", "reference")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _positive(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"Invalid positive measurement: {label}")
    return float(value)


def _scope(scope: dict) -> None:
    request = scope.get("request", {})
    if not isinstance(request, dict):
        raise ValueError("Expected the frozen canonical resolved release scope")
    workloads = request.get("performance_workloads")
    if (
        not isinstance(scope.get("fingerprint"), str)
        or len(scope["fingerprint"]) != 64
        or not isinstance(workloads, list)
        or not workloads
        or any(not isinstance(item, str) or not item for item in workloads)
        or len(workloads) != len(set(workloads))
    ):
        raise ValueError("Expected the frozen canonical resolved release scope")
    _performance_policy(scope)


def _performance_policy(scope: dict) -> str:
    policy = scope["request"].get("performance_policy", "superiority")
    if policy not in ("superiority", "report"):
        raise ValueError("Unknown explicit performance policy")
    return policy


def _locked_distributions(lock: Path, record: dict) -> None:
    locked: dict[str, set[str]] = {}
    for package in tomllib.loads(lock.read_text()).get("package", []):
        name = package["name"].lower().replace("_", "-")
        locked.setdefault(name, set()).add(package["version"])
    installed = record.get("installed_distributions")
    if not isinstance(installed, dict) or "numpy" not in installed:
        raise ValueError("Missing actual worker dependency inventory")
    for name, version in installed.items():
        # Installer tooling is not imported by a measured operation. All actual
        # installed application/numerical dependencies must resolve in uv.lock.
        if name in {"pip", "uv"}:
            continue
        if version not in locked.get(name, set()):
            raise ValueError(
                f"Worker dependency differs from uv.lock: {name}=={version}"
            )


def verify_reference_campaign(
    path: Path,
    *,
    wheel: Path,
    scope: dict,
    lock: Path,
    require_qualification: bool = True,
) -> dict:
    """Recompute each matched session's ratios from retained executed records.

    Release evidence is keyed by every required workload. The default policy
    requires strict resource wins; an explicit report policy retains completed
    bounded measurements and adverse ratios without certifying superiority.
    The caller supplies its actual wheel, resolved scope and dependency lock;
    matching strings inside an untrusted status record are insufficient.
    """
    _scope(scope)
    report_only = _performance_policy(scope) == "report"
    manifest = load(path / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Reference performance campaign is incomplete")
    verify_inventory(path, manifest["files"])
    if manifest.get("wheel_sha256") != sha256(wheel):
        raise ValueError("Reference performance wheel changed")
    if sha256(path / "inputs/candidate.whl") != sha256(wheel):
        raise ValueError("Retained performance wheel differs from release artifact")
    if manifest.get("lock_sha256") != sha256(lock):
        raise ValueError("Reference performance dependency lock changed")
    if sha256(path / "inputs/uv.lock") != sha256(lock):
        raise ValueError("Retained performance lock changed")
    if load(path / "inputs/scope.json") != scope:
        raise ValueError("Reference performance resolved scope changed")
    if manifest.get("scope_fingerprint") != scope["fingerprint"]:
        raise ValueError("Reference performance has a different scope fingerprint")
    if manifest.get("harness_fingerprint") != harness_fingerprint():
        raise ValueError("Reference performance harness changed")
    with zipfile.ZipFile(wheel) as archive:
        wheel_payload = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("statadic/") and not name.endswith("/")
        }
    if not wheel_payload:
        raise ValueError("Candidate wheel has no STATADIC payload")
    cases = manifest.get("cases", [])
    if (
        not isinstance(cases, list)
        or not cases
        or len(cases) != len(set(cases))
        or any(case not in PILOT_CASES for case in cases)
    ):
        raise ValueError("Missing, duplicate or unknown reference cases")
    required = scope["request"]["performance_workloads"]
    if any(PILOT_CASES[case] not in required for case in cases):
        raise ValueError("Reference case lies outside the explicit release scope")
    sessions = manifest.get("sessions")
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise ValueError("Invalid reference session count")
    for field, minimum in (("samples", 2), ("warmups", 1)):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid reference sampling protocol: {field}")
    if manifest.get("purpose") not in {"pilot", "qualification"}:
        raise ValueError("Unknown reference measurement purpose")
    if require_qualification:
        if report_only:
            missing = set(required) - {PILOT_CASES[case] for case in cases}
            if missing:
                raise ValueError(
                    f"Missing required performance observations: {sorted(missing)}"
                )
        elif manifest.get("purpose") != "qualification" or sessions != 3:
            raise ValueError("Pilot evidence cannot qualify three independent sessions")
    if require_qualification and not report_only:
        missing = set(required) - QUALIFICATION_CASES.keys()
        expected_cases = {
            case
            for workload in required
            for case in QUALIFICATION_CASES.get(workload, ())
        }
        if missing or set(cases) != expected_cases:
            raise ValueError(
                f"Incomplete required performance workloads/cases: {sorted(missing)}"
            )
        if manifest.get("samples") != 30 or manifest.get("warmups") != 5:
            raise ValueError("Qualification requires five warmups and thirty samples")
    expected = {
        f"{case}-{side}-{session}.json"
        for case in cases
        for side in SIDES
        for session in range(sessions)
    }
    records = manifest.get("records", [])
    if set(records) != expected or len(records) != len(expected):
        raise ValueError("Missing, duplicate or unexpected performance workers")
    rows = []
    identities = set()
    session_hardware = None
    for case in cases:
        for session in range(sessions):
            pair = {}
            for side in SIDES:
                record = load(path / f"{case}-{side}-{session}.json")
                if (
                    record.get("schema_version") != 1
                    or record.get("case") != case
                    or record.get("side") != side
                    or record.get("session") != session
                    or record.get("dtype") != "float64"
                    or record.get("backend") != "cpu"
                    or record.get("warmups") != manifest["warmups"]
                    or record.get("cache_policy") != "fresh-process;persistent-disabled"
                ):
                    raise ValueError(
                        "Worker identity or protocol differs from its schedule"
                    )
                identity = record.get("process_identity")
                if (
                    not isinstance(identity, str)
                    or not identity
                    or identity in identities
                ):
                    raise ValueError("Repeated or absent independent process identity")
                identities.add(identity)
                _locked_distributions(lock, record)
                installed = record["installed_distributions"]
                needed = (
                    {"numpy", "jax", "jaxlib", "statadic"}
                    if side == "statadic"
                    else {"numpy", CASES[case]["distribution"]}
                )
                if not needed <= installed.keys():
                    raise ValueError("Measured implementation dependencies are absent")
                binding = record.get("installed_wheel")
                if side == "statadic":
                    if not isinstance(binding, dict) or binding.get(
                        "wheel_sha256"
                    ) != sha256(wheel):
                        raise ValueError(
                            "Candidate worker lacks the actual wheel binding"
                        )
                    if (
                        binding.get("payload_files") != wheel_payload
                        or record.get("jax_imported") is not True
                    ):
                        raise ValueError("Candidate package payload was not verified")
                    if binding.get("version") != installed["statadic"]:
                        raise ValueError(
                            "Installed STATADIC version differs from wheel binding"
                        )
                    if record.get("compiler_settings") != {
                        "jax_disable_jit": False,
                        "jax_disable_most_optimizations": False,
                        "jax_default_matmul_precision": "highest",
                    }:
                        raise ValueError("Candidate compiler configuration changed")
                elif binding is not None or record.get("jax_imported") is not False:
                    raise ValueError(
                        "Reference worker imported the measured JAX implementation"
                    )
                expected_threads = {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1",
                    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
                }
                if record.get("threads") != expected_threads:
                    raise ValueError(
                        "Worker thread or compiler policy differs from the schedule"
                    )
                warm = record.get("warm_seconds", [])
                if not isinstance(warm, list) or len(warm) != manifest["samples"]:
                    raise ValueError("Incomplete warm timing distribution")
                for value in warm:
                    _positive(value, "warm_seconds")
                _positive(record.get("first_call_seconds"), "first_call_seconds")
                _positive(record.get("process_peak_rss_bytes"), "whole-process RSS")
                if session_hardware is None:
                    session_hardware = record.get("hardware")
                if not session_hardware or record.get("hardware") != session_hardware:
                    raise ValueError(
                        "Reference workers have different hardware or affinity"
                    )
                pair[side] = record
            first, second = pair["statadic"], pair["reference"]
            for field in ("fixture", "python", "threads", "timing_scope", "rss_scope"):
                if field not in first or first[field] != second.get(field):
                    raise ValueError(f"Unmatched reference measurement field: {field}")
            fixture_record = first["fixture"]
            expected_fixture, _ = fixture(case, pilot=manifest["purpose"] == "pilot")
            if (
                fixture_record != expected_fixture
                or fingerprint(fixture_record) != manifest["fixtures"][case]
            ):
                raise ValueError("Reference fixture does not match the retained design")
            left, right = first.get("values"), second.get("values")
            if (
                not isinstance(left, list)
                or not isinstance(right, list)
                or not left
                or len(left) != len(right)
                or first.get("shape") != [len(left)]
                or second.get("shape") != [len(right)]
            ):
                raise ValueError("Reference output shapes differ")
            rtol, atol = CASES[case]["rtol"], CASES[case]["atol"]
            for actual, reference in zip(left, right, strict=True):
                if (
                    isinstance(actual, bool)
                    or isinstance(reference, bool)
                    or not isinstance(actual, (int, float))
                    or not isinstance(reference, (int, float))
                    or not math.isfinite(actual)
                    or not math.isfinite(reference)
                    or abs(actual - reference) > atol + rtol * abs(reference)
                ):
                    raise ValueError(f"Independent reference output mismatch: {case}")
            warm_ratio = _positive(
                statistics.median(first["warm_seconds"])
                / statistics.median(second["warm_seconds"]),
                "warm_time_ratio",
            )
            rss_ratio = _positive(
                first["process_peak_rss_bytes"] / second["process_peak_rss_bytes"],
                "total_peak_rss_ratio",
            )
            rows.append(
                {
                    "case": case,
                    "workload": PILOT_CASES[case],
                    "session": session,
                    "warm_time_ratio": warm_ratio,
                    "total_peak_rss_ratio": rss_ratio,
                    "passed": warm_ratio <= 0.90 and rss_ratio <= 0.90,
                    "candidate_process": first["process_identity"],
                    "reference_process": second["process_identity"],
                }
            )
    if (
        require_qualification
        and not report_only
        and not all(row["passed"] for row in rows)
    ):
        raise ValueError(
            "Required warm-time and total-RSS win failed in at least one session"
        )
    evidence = {}
    if require_qualification:
        for workload in required:
            evidence[workload] = {
                "status": "reported" if report_only else "passed",
                "lane": "matched-reference-performance",
                "scope_fingerprint": scope["fingerprint"],
                "wheel_sha256": sha256(wheel),
                "campaign_manifest_sha256": sha256(path / "manifest.json"),
                "cases": [case for case in cases if PILOT_CASES[case] == workload]
                if report_only
                else list(QUALIFICATION_CASES[workload]),
                "sessions": [
                    {
                        "id": str(session),
                        "warm_time_ratio": max(
                            row["warm_time_ratio"]
                            for row in rows
                            if row["workload"] == workload and row["session"] == session
                        ),
                        "total_peak_rss_ratio": max(
                            row["total_peak_rss_ratio"]
                            for row in rows
                            if row["workload"] == workload and row["session"] == session
                        ),
                    }
                    for session in range(sessions)
                ],
            }
            if report_only:
                evidence[workload].update(
                    performance_policy="report", measurement_purpose=manifest["purpose"]
                )
    return {
        "purpose": manifest["purpose"],
        "observations": rows,
        "contract_evidence": evidence,
    }


def run(
    output: Path,
    *,
    wheel: Path,
    scope: dict,
    lock: Path,
    python: Path,
    reference_python: Path,
    cases: list[str],
    pilot: bool = True,
    sessions: int = 1,
) -> dict:
    """Run a serialized, alternating schedule in the supplied environments."""
    _scope(scope)
    report_only = _performance_policy(scope) == "report"
    if output.exists():
        raise ValueError("Choose a new immutable reference campaign directory")
    if not pilot and not report_only:
        missing = (
            set(scope["request"]["performance_workloads"]) - QUALIFICATION_CASES.keys()
        )
        if missing:
            raise ValueError(
                f"Required workload adapters are not implemented: {sorted(missing)}"
            )
        sessions = 3
    if (
        not cases
        or len(cases) != len(set(cases))
        or any(case not in PILOT_CASES for case in cases)
    ):
        raise ValueError("Select distinct implemented reference cases")
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise ValueError("Expected a positive independent session count")
    wheel, lock = wheel.resolve(), lock.resolve()
    expected_wheel, expected_lock = sha256(wheel), sha256(lock)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    shutil.copyfile(wheel, output / "inputs/candidate.whl")
    shutil.copyfile(lock, output / "inputs/uv.lock")
    write_json(output / "inputs/scope.json", scope)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "purpose": "pilot" if pilot else "qualification",
        "started": datetime.now(UTC).isoformat(),
        "wheel_sha256": expected_wheel,
        "lock_sha256": expected_lock,
        "scope_fingerprint": scope["fingerprint"],
        "harness_fingerprint": harness_fingerprint(),
        "cases": cases,
        "sessions": sessions,
        "samples": 3 if pilot else 30,
        "warmups": 1 if pilot else 5,
        "fixtures": {},
        "records": [],
        "python": str(python.absolute()),
        "reference_python": str(reference_python.absolute()),
    }
    write_json(output / "manifest.json", manifest)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        and not key.startswith("JAX_")
    } | {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        "PYTHONHASHSEED": "0",
    }
    worker = Path(__file__).with_name("reference_worker.py").resolve()
    try:
        for session in range(sessions):
            for index, case in enumerate(cases):
                sides = SIDES if (session + index) % 2 else tuple(reversed(SIDES))
                for side in sides:
                    name = f"{case}-{side}-{session}.json"
                    command = [
                        str(
                            (
                                python if side == "statadic" else reference_python
                            ).absolute()
                        ),
                        "-I",
                        str(worker),
                        case,
                        side,
                        "--wheel",
                        str((output / "inputs/candidate.whl").resolve()),
                        "--output",
                        str((output / name).resolve()),
                        "--session",
                        str(session),
                        "--samples",
                        str(manifest["samples"]),
                        "--warmups",
                        str(manifest["warmups"]),
                    ]
                    if pilot:
                        command.append("--pilot")
                    with (output / name.replace(".json", ".log")).open("w") as log:
                        subprocess.run(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=600,
                        )
                    record = load(output / name)
                    manifest["fixtures"].setdefault(
                        case, fingerprint(record["fixture"])
                    )
                    manifest["records"].append(name)
                    write_json(output / "manifest.json", manifest)
        if (
            sha256(wheel) != expected_wheel
            or sha256(lock) != expected_lock
            or harness_fingerprint() != manifest["harness_fingerprint"]
        ):
            raise ValueError(
                "Wheel, dependency lock or harness changed during measurement"
            )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished"] = datetime.now(UTC).isoformat()
        manifest["files"] = inventory(output)
        write_json(output / "manifest.json", manifest)
        write_archive(output, output.with_name(output.name + ".tar"))
    return verify_reference_campaign(
        output,
        wheel=wheel,
        scope=scope,
        lock=lock,
        require_qualification=not pilot,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "qualify", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--wheel", type=Path, required=True)
        command.add_argument(
            "--scope",
            type=Path,
            required=True,
            help="Frozen canonical resolved scope JSON",
        )
        command.add_argument("--lock", type=Path, required=True)
        command.add_argument(
            "--output" if name != "verify" else "--campaign", type=Path, required=True
        )
        if name != "verify":
            command.add_argument("--python", type=Path, required=True)
            command.add_argument("--reference-python", type=Path, required=True)
            command.add_argument(
                "--cases",
                nargs="+",
                choices=tuple(PILOT_CASES),
                default=list(PILOT_CASES),
            )
            if name == "pilot":
                command.add_argument("--sessions", type=int, default=1)
        else:
            command.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()
    scope = load(args.scope)
    if args.command == "verify":
        result = verify_reference_campaign(
            args.campaign,
            wheel=args.wheel,
            scope=scope,
            lock=args.lock,
            require_qualification=not args.allow_pilot,
        )
    else:
        result = run(
            args.output,
            wheel=args.wheel,
            scope=scope,
            lock=args.lock,
            python=args.python,
            reference_python=args.reference_python,
            cases=args.cases,
            pilot=args.command == "pilot",
            sessions=getattr(args, "sessions", 3),
        )
    json.dump(result, sys.stdout, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
