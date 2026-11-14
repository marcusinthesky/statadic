# ruff: noqa: INP001  # standalone package-maintenance entry point
"""Run mandatory live comparisons and retain precision-specific evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import NotRequired, TypedDict
from xml.etree import ElementTree as ET

_REQUIRED = ("spreg", "statsmodels", "linearmodels", "lifelines")
_LANES = ("external-float64", "precision-float32", "precision-float64")


def input_identity(package: Path, catalogue: Path | None = None) -> dict[str, object]:
    """Bind reference evidence to the imported implementation and locked tests."""
    import statadic  # noqa: PLC0415 -- report reconciliation needs only the stdlib

    imported = Path(statadic.__file__).resolve().parent
    files = set(imported.rglob("*.py"))
    for directory in ("tests", "scripts"):
        files.update((package / directory).rglob("*.py"))
    files.add(package / "pyproject.toml")
    if (package / "justfile").is_file():
        files.add(package / "justfile")
    if catalogue is not None:
        files.add(catalogue.resolve())
    for parent in (package, *package.parents):
        if (parent / "uv.lock").is_file():
            files.add(parent / "uv.lock")
            break
    return {
        "imported_package": str(imported),
        "distribution_origin": json.loads(
            distribution("statadic").read_text("direct_url.json") or "{}"
        ),
        "files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        },
    }


class ReferenceResults(TypedDict):
    """Completed comparison inventory and outcomes from one pytest process."""

    tests: int
    failures: int
    errors: int
    skipped: int
    comparators: dict[str, int]
    nodes: list[str]
    node_comparators: dict[str, list[str]]


class ReferenceMapping(TypedDict):
    """One authored, bounded comparison; execution is retained separately."""

    python_symbols: list[str]
    construction_paths: list[str]
    nodeids: list[str]
    oracle_kind: str
    distribution: str
    call: str
    quantities: list[str]
    conventions: str
    limits: str
    lanes: list[str]
    configuration: NotRequired[str]


class ReferenceClaim(TypedDict):
    """Existing catalogue identity and its exact comparison declaration."""

    owner_kind: str
    owner_id: str
    mapping: ReferenceMapping


def read_results(path: Path, required: tuple[str, ...] = ()) -> ReferenceResults:
    """Reject empty, incomplete or unsuccessful reference evidence."""
    cases = list(ET.parse(path).getroot().iter("testcase"))  # noqa: S314  # local pytest output
    counts: dict[str, int] = {}
    nodes: dict[str, list[str]] = {}
    for case in cases:
        identities = []
        dependencies: list[str] = []
        for prop in case.findall("properties/property"):
            if prop.get("name") == "reference_dependencies":
                dependencies = json.loads(prop.get("value", "[]"))
                for name in dependencies:
                    counts[name] = counts.get(name, 0) + 1
            if prop.get("name") == "reference_nodeid":
                identities.append(prop.get("value", ""))
        if len(identities) != 1 or not identities[0].startswith("tests/"):
            message = "Reference comparison lacks an exact reference_nodeid"
            raise ValueError(message)
        if identities[0] in nodes:
            message = f"Duplicate reference node: {identities[0]}"
            raise ValueError(message)
        nodes[identities[0]] = dependencies
    result: ReferenceResults = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "comparators": counts,
        "nodes": list(nodes),
        "node_comparators": nodes,
    }
    if not cases or result["failures"] or result["errors"] or result["skipped"]:
        message = f"Reference comparisons did not all complete successfully: {result}"
        raise ValueError(message)
    missing = sorted(set(required) - counts.keys())
    if missing:
        message = f"No completed reference comparisons for: {', '.join(missing)}"
        raise ValueError(message)
    return result


def load_reference_mappings(path: Path) -> list[ReferenceClaim]:
    """Read explicit catalogue data without importing the internal studies app."""
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    records = [("method", record) for record in catalogue.get("methods", [])] + [
        ("scientific-variant", variant)
        for procedure in catalogue.get("scientific_procedures", [])
        for variant in procedure["variants"]
    ]
    result: list[ReferenceClaim] = []
    for kind, record in records:
        for index, mapping in enumerate(record.get("reference_mappings", [])):
            lanes = mapping.get("lanes", [])
            nodeids = mapping.get("nodeids", [])
            invalid_nodes = (
                not nodeids
                or len(nodeids) != len(set(nodeids))
                or any(
                    not node.startswith("tests/") or "::" not in node
                    for node in nodeids
                )
            )
            invalid_lanes = (
                not set(lanes) <= set(_LANES)
                or len(lanes) != len(set(lanes))
                or (mapping.get("oracle_kind") == "external" and not lanes)
            )
            if (
                invalid_nodes
                or invalid_lanes
                or mapping.get("oracle_kind")
                not in {"external", "independent-formula", "trusted-fixture"}
                or not mapping.get("distribution")
            ):
                message = f"Invalid reference mapping: {record['id']}[{index}]"
                raise ValueError(message)
            result.append(
                {"owner_kind": kind, "owner_id": record["id"], "mapping": mapping}
            )
    return result


def reconcile_reference_mappings(
    mappings: list[ReferenceClaim], name: str, results: ReferenceResults
) -> dict[str, object]:
    """Require exact declared comparisons; expose executions without a mapping."""
    covered: set[str] = set()
    claims = []
    for entry in mappings:
        mapping = entry["mapping"]
        if name not in mapping["lanes"]:
            continue
        for node in mapping["nodeids"]:
            if node not in results["node_comparators"]:
                message = f"Declared reference node did not execute in {name}: {node}"
                raise ValueError(message)
            if mapping["distribution"] not in results["node_comparators"][node]:
                message = (
                    f"Reference comparator mismatch for {node}: "
                    f"{mapping['distribution']}"
                )
                raise ValueError(message)
            covered.add(node)
        claims.append(entry)
    return {
        "executed_mappings": claims,
        "mapped_nodes": sorted(covered),
        "unmapped_executed_nodes": sorted(set(results["nodes"]) - covered),
    }


def _run_lane(
    name: str,
    recipe: str,
    x64: str,
    package: Path,
    output: Path,
    required: tuple[str, ...],
    mappings: list[ReferenceClaim] | None = None,
) -> dict[str, object]:
    report = output / f"{name}.xml"
    command = ["just", recipe, f"--junitxml={report}"]
    seed = os.environ.get("STATADIC_CI_SEED")
    if seed:
        command.append(f"--hypothesis-seed={seed}")
    environment = dict(os.environ, JAX_ENABLE_X64=x64)
    start = time.monotonic()
    sys.stdout.write(
        f"Reference lane {name}: Python {platform.python_version()}, "
        f"JAX_ENABLE_X64={x64}, STATADIC_CI_SEED={seed or '<unset>'}\n"
    )
    sys.stdout.write(f"Reference command: {shlex.join(command)}\n")
    sys.stdout.flush()
    with (output / f"{name}.log").open("wb") as log:
        process = subprocess.Popen(  # fixed repository recipes  # noqa: S603
            command,
            cwd=package,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            message = "Reference subprocess output pipe was not created"
            raise RuntimeError(message)
        with process.stdout:
            while chunk := os.read(process.stdout.fileno(), 65536):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        returncode = process.wait()
    lane: dict[str, object] = {
        "name": name,
        "command": command,
        "jax_enable_x64": x64 == "true",
        "returncode": returncode,
        "seconds": time.monotonic() - start,
        "status": "failed",
    }
    if returncode == 0:
        try:
            results = read_results(report, required)
            lane["results"] = results
            if mappings is not None:
                lane["reference_index"] = reconcile_reference_mappings(
                    mappings, name, results
                )
            lane["status"] = "passed"
        except (OSError, ValueError, ET.ParseError) as error:
            lane["error"] = str(error)
    return lane


def reference_receipt(catalogue: Path, report: Path, lane: str) -> dict[str, object]:
    """Reconstruct exact-node reference claims from retained catalogue and JUnit."""
    mappings = load_reference_mappings(catalogue)
    results = read_results(report)
    return {
        "lane": lane,
        "catalogue_sha256": hashlib.sha256(catalogue.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "results": results,
        **reconcile_reference_mappings(mappings, lane, results),
    }


def main() -> None:
    """Run full references and both precision subsets in separate processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/reference-validation")
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        help="Reconcile exact reference mappings from an explicit catalogue file.",
    )
    parser.add_argument(
        "--reconcile-report",
        type=Path,
        help="Reconcile one already executed installed-wheel JUnit report.",
    )
    parser.add_argument("--lane", choices=_LANES)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    catalogue = arguments.catalogue.resolve() if arguments.catalogue else None
    mappings = load_reference_mappings(catalogue) if catalogue else []
    if arguments.reconcile_report is not None or arguments.lane is not None:
        if (
            arguments.reconcile_report is None
            or arguments.lane is None
            or catalogue is None
        ):
            parser.error(
                "Report reconciliation requires --reconcile-report, "
                "--lane and --catalogue"
            )
        receipt = reference_receipt(
            catalogue, arguments.reconcile_report, arguments.lane
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "reference-index.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        return
    if output.exists() and any(output.iterdir()):
        message = f"Retain the existing evidence and use an empty directory: {output}"
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    lanes: list[dict[str, object]] = []
    manifest = {
        "status": "running",
        "input_identity": input_identity(package, catalogue),
        "reference_catalogue": str(catalogue) if catalogue else None,
        "reference_mappings": mappings,
        "unexecuted_reference_mappings": [
            entry for entry in mappings if not entry["mapping"]["lanes"]
        ],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            str(dist.metadata["Name"]): dist.version for dist in distributions()
        },
        "layout_enforcer": json.loads(
            distribution("pytest-layout-enforcer").read_text("direct_url.json") or "{}"
        ),
        "lanes": lanes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, recipe, x64, required in (
        ("external-float64", "test-external", "true", _REQUIRED),
        (
            "precision-float32",
            "test-precision-references",
            "false",
            ("statsmodels", "lifelines"),
        ),
        (
            "precision-float64",
            "test-precision-references",
            "true",
            ("statsmodels", "lifelines"),
        ),
    ):
        lane = _run_lane(
            name,
            recipe,
            x64,
            package,
            output,
            required,
            mappings if catalogue is not None else None,
        )
        lanes.append(lane)
        if input_identity(package, catalogue) != manifest["input_identity"]:
            lane["status"] = "failed"
            lane["error"] = "Reference implementation, lock or definitions changed"
        manifest["status"] = "failed" if lane["status"] != "passed" else "running"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if lane["status"] != "passed":
            log_path = output / f"{name}.log"
            message = f"Reference lane {name} failed; inspect {log_path}"
            raise RuntimeError(message)
    manifest["status"] = "passed"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
