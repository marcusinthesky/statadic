"""Qualify retained distributions against locked, isolated scientific references."""

import argparse
import hashlib
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from statadic_evidence import (
    extract_archive,
    inventory,
    verify_inventory,
    write_archive,
    write_json,
)

ROOT = Path(__file__).resolve().parents[5]
PROJECT = ROOT / "src/python"
PACKAGE = PROJECT / "packages/statadic"
OUTPUT = ROOT / "build/release"
BUILD_SOURCE_DATE_EPOCH = "315532800"


def run_logged(command, *, cwd, env, log_path):
    """Stream a lane to the console and retain its exact combined output bytes."""
    with (
        log_path.open("wb") as log,
        subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process,
    ):
        assert process.stdout is not None
        while chunk := process.stdout.read1(65536):
            log.write(chunk)
            log.flush()
            console = getattr(sys.stdout, "buffer", None)
            if console is None:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            else:
                console.write(chunk)
                console.flush()
        return subprocess.CompletedProcess(command, process.wait())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_execution_policy() -> None:
    """Select this release driver's CPU/float64 scope before campaign inspection."""
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    import jax

    # A Python caller may already have imported JAX. This consumer owns its
    # explicit precision policy; runtime library modules never change it.
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise ValueError("Release inspection requires the supported CPU backend")


def source_snapshot() -> dict[str, str]:
    listing = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return {
        name: digest(ROOT / name)
        for name in sorted(set(listing.decode().split("\0")) - {""})
        if (ROOT / name).is_file()
    }


def verify_check_evidence(bundle: Path, manifest: dict) -> None:
    """Recheck the checkout gate, including imported complete shard evidence."""
    checks = [lane for lane in manifest["lanes"] if lane["name"] == "check"]
    if (
        len(checks) != 1
        or checks[0].get("status") != "passed"
        or checks[0].get("returncode") != 0
    ):
        raise ValueError("Release requires exactly one successful checkout check")
    check = checks[0]
    if "check_receipt" not in manifest:
        if (
            check.get("command") != ["just", "check"]
            or not (bundle / "check.log").is_file()
        ):
            raise ValueError("Release lacks its fresh checkout check log")
        return
    if manifest.get("mode") != "preflight":
        raise ValueError("Only preflight can reuse a CI checkout check")
    provenance = manifest["check_receipt"]
    if (
        provenance.get("path") != "check-receipt"
        or check.get("command") != ["verify-ci-check-receipt", "check-receipt"]
        or not provenance.get("run_id")
        or not provenance.get("run_attempt")
    ):
        raise ValueError("Invalid CI checkout check provenance")
    verifier = runpy.run_path(str(ROOT / "infra/nix/ci_shards.py"))["verify_receipt"]
    receipt_dir = bundle / "check-receipt"
    verifier(
        receipt_dir,
        source_files=manifest["source_files"],
        revision=manifest["revision"],
        architecture=manifest["platform"]["machine"],
        run_id=provenance["run_id"],
        run_attempt=provenance["run_attempt"],
    )
    if digest(receipt_dir / "receipt.json") != provenance.get("sha256"):
        raise ValueError("CI checkout check receipt changed")


def retain_preflight_check(directory: Path, output: Path, manifest: dict) -> None:
    """Reuse completed checks only from this exact hosted workflow execution."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if os.environ.get("GITHUB_ACTIONS") != "true" or not run_id or not attempt:
        raise ValueError("CI check reuse requires the current GitHub run and attempt")
    verifier = runpy.run_path(str(ROOT / "infra/nix/ci_shards.py"))["verify_receipt"]
    verifier(
        directory,
        source_files=manifest["source_files"],
        revision=manifest["revision"],
        architecture=platform.machine(),
        run_id=run_id,
        run_attempt=attempt,
    )
    retained = output / "check-receipt"
    shutil.copytree(directory, retained)
    manifest["check_receipt"] = {
        "path": "check-receipt",
        "sha256": digest(retained / "receipt.json"),
        "run_id": run_id,
        "run_attempt": attempt,
    }
    manifest["lanes"].append(
        {
            "name": "check",
            "command": ["verify-ci-check-receipt", "check-receipt"],
            "backend": "cpu",
            "precision": "float64",
            "status": "passed",
            "returncode": 0,
            "provenance": "Complete checkout checks from this workflow run",
        }
    )
    verify_check_evidence(output, manifest)
    print("Reused verified checkout checks from this workflow run", flush=True)


def junit_summary(path: Path) -> dict:
    """Preserve per-cell dispositions independently of test execution success."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    result = {
        "selected": len(cases),
        "failed": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
        "studies": [],
    }
    for case in cases:
        for prop in case.findall("properties/property"):
            if prop.get("name") == "qualification_report":
                result["studies"].append(
                    {"test": case.get("name"), "report": json.loads(prop.get("value"))}
                )
    if not cases or any(result[k] for k in ("failed", "errors", "skipped")):
        raise ValueError(f"Incomplete required test lane: {path}: {result}")
    return result


def verified_contracts(bundle: Path, resolved: dict, wheel: Path) -> dict:
    """Reconstruct contract evidence from its executed, retained producers."""
    from statadic_docs_labs.consumer_acceptance import verify_acceptance
    from statadic_docs_labs.numerical_contracts import verify_results
    from statadic_performance.reference import verify_reference_campaign

    evidence = {}
    if (bundle / "numerical-contracts.json").is_file():
        evidence["numerical_contracts"] = verify_results(
            bundle, scope=resolved, wheel=wheel, package=PACKAGE
        )
    if (bundle / "consumer-acceptance").is_dir():
        result = verify_acceptance(bundle / "consumer-acceptance", wheel=wheel)
        evidence["consumer_contracts"] = {
            identity: record
            | {
                "scope_fingerprint": resolved["fingerprint"],
                "wheel_sha256": digest(wheel),
                "lane": "consumer-acceptance",
            }
            for identity, record in result["consumer_contracts"].items()
        }
    if (bundle / "reference-performance").is_dir():
        result = verify_reference_campaign(
            bundle / "reference-performance",
            wheel=wheel,
            scope=resolved,
            lock=bundle / "uv.lock",
        )
        evidence["performance_workloads"] = result["contract_evidence"]
    return evidence


def documentation_command(exports: list[Path]) -> list[str]:
    """Build release pages from an explicit, already verified export selection."""
    command = [
        sys.executable,
        str(PROJECT / "apps/docs/scripts/build.py"),
        "--release-evidence",
    ]
    for path in exports:
        command.extend(("--study-export", str(path.resolve())))
    return command


def documentation_evidence(bundle: Path, records: list[dict]) -> dict:
    """Bind the rendered site to retained exports without relying on local paths."""
    metadata = json.loads((bundle / "site/build-provenance.json").read_text())
    exports = {
        record["study"]: digest(
            bundle / "method-evaluations" / record["study"] / "evidence-manifest.json"
        )
        for record in records
    }
    rendered = metadata.get("study_exports")
    if metadata.get("release_evidence") is not True or not isinstance(rendered, list):
        raise ValueError("Release documentation lacks an explicit evidence selection")
    hashes = [
        row.get("manifest_sha256") if isinstance(row, dict) else None
        for row in rendered
    ]
    if any(not isinstance(value, str) for value in hashes) or sorted(hashes) != sorted(
        exports.values()
    ):
        raise ValueError("Release documentation renders different study exports")
    return {"release_evidence": True, "exports": exports}


def verify_reference_indexes(bundle: Path, manifest: dict) -> None:
    """Re-derive installed-wheel reference receipts from retained raw reports."""
    if manifest.get("reference_index_version") != 1:
        raise ValueError("Release evidence requires reference index version 1")
    reconstruct = runpy.run_path(str(PACKAGE / "scripts/check_references.py"))[
        "reference_receipt"
    ]
    for name in ("external", "precision-float32", "precision-float64"):
        lane = "external-float64" if name == "external" else name
        retained = json.loads(
            (bundle / "reference-index" / lane / "reference-index.json").read_text()
        )
        actual = reconstruct(
            bundle / "reference-catalogue.json", bundle / f"{name}.xml", lane
        )
        records = [
            record
            for record in manifest["lanes"]
            if record["name"] == f"{name}-reference-index"
        ]
        if len(records) != 1 or records[0].get("status") != "passed":
            raise ValueError(f"Missing completed reference index: {lane}")
        if actual != retained or actual != records[0].get("reference_index"):
            raise ValueError(f"Reference index differs from retained execution: {lane}")


def verify(
    bundle: Path,
    *,
    require_qualification: bool = True,
    expected_scope: dict | None = None,
) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported release evidence schema")
    retained_scope = manifest.get("qualification_scope")
    if retained_scope is not None:
        if manifest.get("schema_version") != 2:
            raise ValueError("Scoped evidence requires release schema 2")
        if require_qualification and expected_scope is None:
            raise ValueError("Scoped qualification requires an explicit expected scope")
        if expected_scope is not None and retained_scope["request"] != expected_scope:
            raise ValueError("Release qualification scope changed")
    elif expected_scope is not None:
        raise ValueError("Release bundle lacks the expected qualification scope")
    if manifest["status"] != "passed":
        raise ValueError("Release validation did not pass")
    if (
        require_qualification
        and manifest.get("mode", "qualification") != "qualification"
    ):
        raise ValueError(
            "Preflight evidence does not establish scientific qualification"
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if manifest["revision"] != revision:
        raise ValueError("Source revision changed after release validation")
    if manifest["source_files"] != source_snapshot():
        raise ValueError("Source changed after release validation")
    try:
        verify_inventory(bundle, manifest["files"])
    except ValueError as error:
        raise ValueError(f"Release evidence changed: {error}") from error
    verify_check_evidence(bundle, manifest)
    verify_reference_indexes(bundle, manifest)
    rendered = documentation_evidence(bundle, manifest.get("method_evaluations", []))
    if rendered != manifest.get("documentation_evidence"):
        raise ValueError("Release documentation evidence changed")
    if require_qualification:
        from statadic_docs_labs.campaign_gate import inspect_campaigns, match_wheel

        if retained_scope is None and not manifest.get("scientific_scope", {}).get(
            "all_registered_methods_evaluated"
        ):
            raise ValueError("Release lacks all-method scientific evaluation evidence")
        records = inspect_campaigns(
            ROOT,
            [
                bundle / "method-evaluations" / record["study"]
                for record in manifest.get("method_evaluations", [])
            ],
            scope=expected_scope,
        )
        if retained_scope is not None:
            from statadic_docs_labs.qualification_scope import (
                require_contract_evidence,
                resolve_scope,
            )

            actual = resolve_scope(ROOT, expected_scope)
            if actual != retained_scope:
                raise ValueError("Resolved release scope changed")
        (wheel,) = (bundle / "artifacts").glob("*.whl")
        match_wheel(records, wheel)
        if retained_scope is not None:
            from statadic_docs_labs.qualification_lanes import verify_results

            execution = verify_results(
                bundle, scope=actual, wheel=wheel, package=PACKAGE
            )
            if execution != manifest.get("qualification_execution"):
                raise ValueError("Scoped qualification execution changed")
            evidence = verified_contracts(bundle, actual, wheel)
            if evidence != manifest.get("contract_evidence", {}):
                raise ValueError("Contract evidence differs from retained execution")
            require_contract_evidence(actual, evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="Verify a retained evidence bundle")
    parser.add_argument(
        "--verify-archive", type=Path, help="Extract and verify an archive"
    )
    parser.add_argument(
        "--allow-preflight",
        action="store_true",
        help="Verify preflight integrity without claiming qualification",
    )
    parser.add_argument(
        "--mode", choices=("preflight", "qualification"), default="qualification"
    )
    parser.add_argument(
        "--preflight-check-receipt",
        type=Path,
        help="Reuse a complete same-run CI checkout receipt in hosted preflight only",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--scope",
        type=Path,
        help="Explicit bounded qualification selection; default remains all-method",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        help="Complete matched reference campaign for the same wheel/scope/lock",
    )
    parser.add_argument(
        "--consumer-repo",
        type=Path,
        help="Read-only consumer application repository for adapted wheel fixtures",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        action="append",
        default=[],
        help="Retained publication export from a complete installed-wheel evaluation",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Retain all evidence, including hidden files, in a deterministic tar",
    )
    args = parser.parse_args()
    if args.preflight_check_receipt is not None and (
        args.mode != "preflight" or args.verify or args.verify_archive
    ):
        parser.error("CI checkout receipts apply only to executing preflight")
    configure_execution_policy()
    scope = None
    if (
        args.performance is not None or args.consumer_repo is not None
    ) and args.scope is None:
        parser.error(
            "Consumer and matched performance evidence require an explicit scope"
        )
    if args.scope is not None:
        from statadic_docs_labs.qualification_scope import read_scope

        scope = read_scope(args.scope)
        if args.mode == "preflight":
            parser.error("Preflight cannot qualify a declared scope")
    if args.verify_archive:
        with tempfile.TemporaryDirectory(prefix="statadic-evidence-") as temporary:
            extracted = Path(temporary) / "bundle"
            extract_archive(args.verify_archive, extracted)
            verify(
                extracted,
                require_qualification=not args.allow_preflight,
                expected_scope=scope,
            )
        print("Archived evidence matches its files and current sources")
        return 0
    if args.verify:
        verify(
            args.verify,
            require_qualification=not args.allow_preflight,
            expected_scope=scope,
        )
        print("Release evidence matches its files and current sources")
        return 0
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Use an empty evidence directory; retain or move {output} first"
        )
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "UV_WORKING_DIRECTORY",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
    ):
        environment.pop(name, None)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
        PYTHONNOUSERSITE="1",
        STATADIC_TEST_MODE="fresh",
        SOURCE_DATE_EPOCH=BUILD_SOURCE_DATE_EPOCH,
    )
    manifest = {
        "schema_version": 2 if scope is not None else 1,
        "reference_index_version": 1,
        "status": "running",
        "mode": args.mode,
        "build_policy": {
            "source_date_epoch": BUILD_SOURCE_DATE_EPOCH,
            "statadic_route": "uv build --sdist --wheel",
        },
        "started": datetime.now(UTC).isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        "source_files": source_snapshot(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "backend": "cpu",
        },
        "lanes": [],
        "formal_scope": {
            "claim_review": {
                claim["id"]: claim.get("review", "not recorded")
                for claim in json.loads(
                    (ROOT / "src/lean/registry/claims.json").read_text()
                )["claims"]
            },
            "source_review": {
                claim["id"]: (
                    claim["review"]
                    if claim["kind"] == "algorithm-refinement"
                    else claim["source"]["review"]
                )
                for claim in json.loads(
                    (ROOT / "src/lean/registry/claims.json").read_text()
                )["claims"]
            },
            "implementation_verification": "Not established by Lean certification",
        },
    }

    def execute(name: str, command: list[str], cwd=ROOT, x64="1") -> None:
        print(f"Release lane: {name}", flush=True)
        start = time.monotonic()
        lane = {
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "backend": "cpu",
            "precision": "float64" if x64 == "1" else "float32",
            "status": "running",
        }
        manifest["lanes"].append(lane)
        write_json(output / "manifest.json", manifest)
        completed = run_logged(
            command,
            cwd=cwd,
            env=environment | {"JAX_ENABLE_X64": x64},
            log_path=output / f"{name}.log",
        )
        lane.update(
            returncode=completed.returncode,
            seconds=time.monotonic() - start,
            status="passed" if completed.returncode == 0 else "failed",
        )
        write_json(output / "manifest.json", manifest)
        completed.check_returncode()

    # Retain the input manifest before running any study, including on interruption.
    write_json(output / "manifest.json", manifest)
    try:
        if scope is not None:
            from statadic_docs_labs.qualification_scope import resolve_scope

            manifest["qualification_scope"] = resolve_scope(ROOT, scope)
            write_json(output / "manifest.json", manifest)
        if args.mode == "qualification":
            from statadic_docs_labs.campaign_gate import inspect_campaigns

            manifest["method_evaluations"] = inspect_campaigns(
                ROOT, args.campaign, scope=scope
            )
        if args.preflight_check_receipt is None:
            execute("check", ["just", "check"])
        else:
            retain_preflight_check(args.preflight_check_receipt, output, manifest)
            write_json(output / "manifest.json", manifest)
        if args.mode == "preflight":
            execute("release-documentation", documentation_command([]))
        # Preserve completed proof/docs evidence even if a later scientific lane fails.
        shutil.copytree(PROJECT / "apps/docs/build/html", output / "site")
        shutil.copy2(
            ROOT / "src/lean/build/certification.json", output / "certification.json"
        )
        for name, path in {
            "uv.lock": PROJECT / "uv.lock",
            "devenv.lock": ROOT / "devenv.lock",
            "lake-manifest.json": ROOT / "src/lean/lake-manifest.json",
        }.items():
            shutil.copy2(path, output / name)
        artifacts = output / "artifacts"
        execute(
            "build",
            [
                "uv",
                "build",
                "--project",
                str(PROJECT),
                "--offline",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--package",
                "statadic",
                "--sdist",
                "--wheel",
                "--out-dir",
                str(artifacts),
            ],
        )
        manifest["built_artifacts"] = {
            str(path.relative_to(artifacts)): digest(path)
            for path in artifacts.iterdir()
            if path.is_file()
        }
        if args.mode == "qualification":
            from statadic_docs_labs.campaign_gate import match_wheel

            (evaluated_wheel,) = artifacts.glob("*.whl")
            match_wheel(manifest["method_evaluations"], evaluated_wheel)
            for record in manifest["method_evaluations"]:
                shutil.copytree(
                    record["path"], output / "method-evaluations" / record["study"]
                )
            execute(
                "release-documentation",
                documentation_command(
                    [
                        output / "method-evaluations" / record["study"]
                        for record in manifest["method_evaluations"]
                    ]
                ),
            )
            shutil.rmtree(output / "site")
            shutil.copytree(PROJECT / "apps/docs/build/html", output / "site")
            manifest["scientific_scope"] = {
                "all_registered_methods_evaluated": scope is None,
                "universal_statistical_validity": False,
                "dispositions": "See retained per-scenario summaries",
            }
        manifest["documentation_evidence"] = documentation_evidence(
            output, manifest.get("method_evaluations", [])
        )
        execute(
            "distribution",
            [
                sys.executable,
                str(PACKAGE / "scripts/check_distribution.py"),
                "--artifacts-dir",
                str(artifacts),
                "--evidence-dir",
                str(output / "distribution"),
            ],
        )
        with tempfile.TemporaryDirectory(prefix="statadic-release-") as temporary:
            harness = Path(temporary)
            shutil.copytree(PACKAGE / "tests", harness / "tests")
            shutil.copy2(PACKAGE / "pyproject.toml", harness / "pyproject.toml")
            shutil.copy2(
                PACKAGE / "scripts/check_references.py", harness / "check_references.py"
            )
            retained_catalogue = output / "reference-catalogue.json"
            shutil.copy2(PROJECT / "apps/studies/catalogue.json", retained_catalogue)
            python = harness / "venv/bin/python"
            execute(
                "environment",
                ["uv", "venv", "--python", sys.executable, str(python.parent.parent)],
                cwd=harness,
            )
            (wheel,) = artifacts.glob("*.whl")
            execute(
                "dependencies",
                [
                    sys.executable,
                    str(ROOT / "infra/release/check_offline_install.py"),
                    "--profile",
                    "release",
                    "--output",
                    str(output / "installed-environment"),
                    "--python",
                    str(python),
                    "--wheels",
                    str(wheel),
                ],
                cwd=harness,
            )
            execute(
                "origin",
                [
                    str(python),
                    "-I",
                    "-c",
                    "import statadic, sys; from pathlib import Path; "
                    "from importlib.metadata import version; "
                    "origin = Path(statadic.__file__).resolve(); "
                    "assert origin.is_relative_to(Path(sys.prefix).resolve()); "
                    "import jax; print(origin, version('statadic'), "
                    "version('jax'), jax.devices())",
                ],
                cwd=harness,
            )
            base = [
                str(python),
                "-I",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:monitor",
                "-p",
                "no:tach",
                "-p",
                "no:layout-enforcer",
                "-n",
                "0",
                "--no-testmon",
                "-o",
                "junit_family=legacy",
            ]
            if environment.get("STATADIC_CI_SEED"):
                seed = environment["STATADIC_CI_SEED"]
                base.append(f"--hypothesis-seed={seed}")
                print(f"Installed-package Hypothesis seed: {seed}", flush=True)
            if scope is not None:
                from statadic_docs_labs.numerical_contracts import (
                    retain_results,
                    selected_files,
                )

                report = output / "numerical-contracts.xml"
                execute(
                    "numerical-contracts",
                    [
                        *base,
                        *selected_files(scope["numerical_contracts"]),
                        "-m",
                        "not slow",
                        f"--junitxml={report}",
                    ],
                    cwd=harness,
                )
                retain_results(
                    output,
                    scope=manifest["qualification_scope"],
                    wheel=wheel,
                    package=PACKAGE,
                )
                if args.consumer_repo is not None:
                    from statadic_docs_labs.consumer_acceptance import run_acceptance

                    consumer_lane = {
                        "name": "consumer-acceptance",
                        "status": "running",
                        "backend": "cpu",
                        "precision": "float64",
                        "wheel_sha256": digest(wheel),
                        "source": str(args.consumer_repo.resolve()),
                    }
                    manifest["lanes"].append(consumer_lane)
                    write_json(output / "manifest.json", manifest)
                    started = time.monotonic()
                    run_acceptance(
                        args.consumer_repo,
                        output / "consumer-acceptance",
                        python=python,
                        wheel=wheel,
                    )
                    consumer_lane.update(
                        status="passed", seconds=time.monotonic() - started
                    )
                    write_json(output / "manifest.json", manifest)
            for name, x64, selection in (
                (
                    "external",
                    "1",
                    [
                        "-m",
                        "external_reference",
                        "--require-external-references",
                        "--hypothesis-profile=statadic-reference",
                    ],
                ),
                (
                    "precision-float32",
                    "0",
                    [
                        "-m",
                        "external_reference",
                        "--require-external-references",
                        "tests/testing/test_precision_references.py",
                        "tests/testing/test_survival_references.py",
                    ],
                ),
                (
                    "precision-float64",
                    "1",
                    [
                        "-m",
                        "external_reference",
                        "--require-external-references",
                        "tests/testing/test_precision_references.py",
                        "tests/testing/test_survival_references.py",
                    ],
                ),
                ("qualification", "1", ["-m", "qualification"]),
            ):
                if name == "qualification" and args.mode == "preflight":
                    manifest["scientific_qualification"] = "not run in hosted preflight"
                    continue
                scoped_qualification = name == "qualification" and scope is not None
                if scoped_qualification:
                    from statadic_docs_labs.qualification_lanes import (
                        retain_results as retain_qualification,
                    )
                    from statadic_docs_labs.qualification_lanes import (
                        selected_nodes,
                    )

                    nodes = selected_nodes(manifest["qualification_scope"])
                    selection = ["-m", "qualification", *nodes]
                    if not nodes:
                        manifest["qualification_execution"] = retain_qualification(
                            output,
                            scope=manifest["qualification_scope"],
                            wheel=wheel,
                            package=PACKAGE,
                        )
                        write_json(output / "manifest.json", manifest)
                        continue
                report_name = (
                    "qualification-scoped.xml"
                    if scoped_qualification
                    else f"{name}.xml"
                )
                report = output / report_name
                execute(
                    name,
                    [*base, *selection, f"--junitxml={report}"],
                    cwd=harness,
                    x64=x64,
                )
                manifest["lanes"][-1]["results"] = junit_summary(report)
                if name != "qualification":
                    lane = "external-float64" if name == "external" else name
                    receipt = output / "reference-index" / lane
                    execute(
                        f"{name}-reference-index",
                        [
                            str(python),
                            "-I",
                            str(harness / "check_references.py"),
                            "--reconcile-report",
                            str(report),
                            "--lane",
                            lane,
                            "--catalogue",
                            str(retained_catalogue),
                            "--output",
                            str(receipt),
                        ],
                        cwd=harness,
                        x64=x64,
                    )
                    manifest["lanes"][-1]["reference_index"] = json.loads(
                        (receipt / "reference-index.json").read_text()
                    )
                if scoped_qualification:
                    manifest["qualification_execution"] = retain_qualification(
                        output,
                        scope=manifest["qualification_scope"],
                        wheel=wheel,
                        package=PACKAGE,
                    )
                    write_json(output / "manifest.json", manifest)
        if manifest["source_files"] != source_snapshot():
            raise ValueError("Sources changed during qualification")
        if manifest["built_artifacts"] != {
            str(path.relative_to(artifacts)): digest(path)
            for path in artifacts.iterdir()
            if path.is_file()
        }:
            raise ValueError("Built artifacts changed during qualification")
        if scope is not None:
            from statadic_docs_labs.qualification_scope import require_contract_evidence

            if args.performance is not None:
                shutil.copytree(args.performance, output / "reference-performance")
            manifest["contract_evidence"] = verified_contracts(
                output, manifest["qualification_scope"], wheel
            )
            if args.performance is not None:
                manifest["lanes"].append(
                    {
                        "name": "matched-reference-performance",
                        "status": "passed",
                        "verification": "Raw worker outputs and ratios re-derived",
                        "wheel_sha256": digest(wheel),
                    }
                )
            require_contract_evidence(
                manifest["qualification_scope"], manifest["contract_evidence"]
            )
        manifest["status"] = "passed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["error"] = "Interrupted; incomplete evidence cannot qualify"
    except (OSError, ValueError, ET.ParseError, subprocess.CalledProcessError) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        print(error, file=sys.stderr)
    finally:
        manifest["finished"] = datetime.now(UTC).isoformat()
        manifest["files"] = inventory(output)
        write_json(output / "manifest.json", manifest)
        if args.archive:
            write_archive(output, args.archive)
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
