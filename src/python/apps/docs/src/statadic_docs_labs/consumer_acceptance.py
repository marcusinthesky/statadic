"""Prepare frozen consumer application fixtures for an installed candidate wheel.

This is pre-migration numerical acceptance, not a production DVC replay or a
statistical/performance qualification. The application repository is read-only.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

from statadic_evidence import inventory, verify_inventory, write_json

CONSUMER_REVISION = "2aa5a191bb193edded80a569e7ae754ff78240b9"
_QUALIFICATION = (
    "src/python/apps/pipeline/src/pipeline/stages/shared/statadic_qualification.py"
)
_HOTELLING = "src/python/apps/hotelling/src"
_WORKFLOW_SOURCES = {
    "consumer-dyadic": "src/python/apps/pipeline/src/pipeline/stages/papers/paper1/"
    "dyadic_confound.py",
    "consumer-frontier": "src/python/apps/pipeline/src/pipeline/stages/papers/paper3/"
    "representation_ablation.py",
    "consumer-spatial-covariance": (
        "src/python/apps/pipeline/src/pipeline/stages/papers/paper5/"
        "spatial_covariance.py"
    ),
    "consumer-spatial-estimator": (
        "src/python/apps/pipeline/src/pipeline/stages/papers/paper5/estimate.py"
    ),
}
_FUNCTIONS = {
    "_fixture",
    "_comparison",
    "_require_available",
    "numerical_qualification",
    "_projected_mmd_objective",
    "_projected_mmd_reference",
    "_weighted_transport_objective",
    "_composition_objective",
    "_hotelling_fixture",
    "_hotelling_operator_reference",
    "_objective_witnesses",
    "_spatial_panel_witness",
    "_gmm_witness",
    "_glm_witness",
    "_glm_fit_witness",
}


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _EstimateMigration(ast.NodeTransformer):
    """Adapt only the selected public fitting verb and covariance composition."""

    def __init__(self, *, covariance: bool = False) -> None:
        self.lines: list[int] = []
        self.covariance = covariance

    def visit_Call(self, node: ast.Call) -> ast.Call:  # noqa: N802 -- AST protocol
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            self.lines.append(node.lineno)
            node.func.attr = "estimate"
        if (
            self.covariance
            and isinstance(node.func, ast.Name)
            and node.func.id == "LedoitWolf"
            and not node.args
            and not node.keywords
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="Estimand", ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id="Covariance", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    ],
                    keywords=[ast.keyword(arg="estimator", value=node)],
                ),
                node,
            )
        return node


def adapt_numerical_source(source: str) -> tuple[str, list[int]]:
    """Extract the actual deterministic witnesses, retaining their independent oracles.

    Exclude pipeline serialization/identity imports and benchmark functions.
    No numerical tolerance, fixture, expected value or assertion is rewritten.
    """
    module = ast.parse(source)
    selected = []
    found = set()
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            if node.name not in _FUNCTIONS:
                continue
            found.add(node.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "pipeline"
        ):
            continue
        elif isinstance(node, ast.If):
            if not isinstance(node.test, ast.Name) or node.test.id != "TYPE_CHECKING":
                continue
        elif not isinstance(
            node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
        ):
            continue
        selected.append(node)
    if found != _FUNCTIONS:
        raise ValueError(
            f"Frozen numerical fixture functions changed: {_FUNCTIONS - found}"
        )
    module.body = selected
    migration = _EstimateMigration()
    migration.visit(module)
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n", sorted(migration.lines)


def adapt_spatial_covariance_source(source: str) -> tuple[str, list[int]]:
    """Retain the actual innovation/total-risk calculation without IO imports."""
    names = {
        "SpatialCovarianceEstimate",
        "spatial_ledoit_wolf_covariance",
        "_symmetrize",
        "_spatialize_covariance",
    }
    selected = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise ValueError("Frozen spatial covariance functions changed")
    module = ast.parse(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "from typing import cast\nimport jax\nimport numpy as np\n"
        "from statadic.api.estimand import Covariance, Estimand\n"
        "from statadic.estimation.covariance import LedoitWolf\n_MATRIX_DIMENSIONS = 2\n"
    )
    module.body.extend(selected)
    migration = _EstimateMigration(covariance=True)
    migration.visit(module)
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n", sorted(migration.lines)


def run_acceptance(
    consumer_repo: Path,
    output: Path,
    *,
    python: Path,
    wheel: Path,
    revision: str = CONSUMER_REVISION,
) -> dict:
    """Run bounded adapted fixtures with an already installed, isolated wheel.

    The caller owns creation of the locked environment and wheel installation.
    ``python`` must be its interpreter. Only frozen Hotelling application source
    enters its explicit import path; STATADIC is verified against wheel bytes.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("consumer application requires an exact Git revision")
    if output.resolve().is_relative_to(consumer_repo.resolve()):
        raise ValueError("Acceptance output must be outside consumer application")
    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(consumer_repo),
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        ],
        text=True,
    ).strip()
    if commit != revision:
        raise ValueError("consumer application revision did not resolve exactly")
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(
        ["git", "-C", str(consumer_repo), "show", f"{revision}:{_QUALIFICATION}"]
    )
    adapted, lines = adapt_numerical_source(source.decode())
    (output / "qualification-original.py").write_bytes(source)
    (output / "qualification-adapted.py").write_text(adapted)
    for filename in ("uv.lock", "pyproject.toml"):
        payload = subprocess.check_output(
            [
                "git",
                "-C",
                str(consumer_repo),
                "show",
                f"{revision}:src/python/{filename}",
            ]
        )
        (output / f"consumer-{filename}").write_bytes(payload)
    workflow_sources = {}
    for identity, path in _WORKFLOW_SOURCES.items():
        payload = subprocess.check_output(
            ["git", "-C", str(consumer_repo), "show", f"{revision}:{path}"]
        )
        (output / f"{identity}-original.py").write_bytes(payload)
        workflow_sources[identity] = {"path": path, "sha256": _hash(payload)}
        if identity == "consumer-spatial-covariance":
            spatial_code, spatial_lines = adapt_spatial_covariance_source(
                payload.decode()
            )
            (output / "spatial-covariance-adapted.py").write_text(spatial_code)
            workflow_sources[identity]["fit_to_estimate_lines"] = spatial_lines
    archive = subprocess.check_output(
        ["git", "-C", str(consumer_repo), "archive", revision, "--", _HOTELLING]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as retained:
        if any(
            not (member.isfile() or member.isdir()) for member in retained.getmembers()
        ):
            raise ValueError("Application snapshots require regular source files")
        retained.extractall(output / "consumer-source", filter="data")
    worker = Path(__file__).with_name("consumer_fixtures.py")
    verifier = (
        Path(__file__).resolve().parents[3] / "studies/src/statadic_studies/installation.py"
    )
    shutil.copyfile(worker, output / "consumer_fixtures.py")
    shutil.copyfile(verifier, output / "wheel_installation.py")
    inputs = {
        path.relative_to(output).as_posix(): _hash(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    request = {
        "schema_version": 2,
        "consumer_revision": revision,
        "consumer_source": _QUALIFICATION,
        "source_sha256": _hash(source),
        "adaptation": {"fit_to_estimate_lines": lines, "numerics_changed": False},
        "hotelling_root": f"consumer-source/{_HOTELLING}",
        "files": inputs,
        "wheel_sha256": _hash(wheel.read_bytes()),
        "sample_size": 64,
        "scope": "adapted deterministic fixtures; full DVC replay and "
        "production migration pending",
        "workflow_sources": workflow_sources,
    }
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PYTHON"):
        environment.pop(name, None)
    environment.update(
        JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1"
    )
    with (output / "execution.log").open("w") as log:
        subprocess.run(
            [
                str(python.absolute()),
                "-I",
                str((output / "consumer_fixtures.py").absolute()),
                str(wheel.absolute()),
            ],
            cwd=output,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = json.loads((output / "results.json").read_text())
    for relative, expected in inputs.items():
        if _hash((output / relative).read_bytes()) != expected:
            raise ValueError(f"Acceptance input changed during execution: {relative}")
    write_json(
        output / "manifest.json",
        {"schema_version": 2, "status": "passed", "files": inventory(output)},
    )
    verify_acceptance(output, wheel=wheel)
    return result


def verify_acceptance(output: Path, *, wheel: Path) -> dict:
    """Check retained fixture/adaptation/wheel bindings before release admission."""
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or manifest.get("status") != "passed":
        raise ValueError("Incomplete consumer application acceptance")
    verify_inventory(output, manifest["files"])
    request = json.loads((output / "request.json").read_text())
    result = json.loads((output / "results.json").read_text())
    if (
        request.get("schema_version") != 2
        or result.get("schema_version") != 2
        or request.get("consumer_revision") != CONSUMER_REVISION
        or request.get("consumer_source") != _QUALIFICATION
        or request.get("sample_size") != 64
    ):
        raise ValueError("Consumer fixture revision or design changed")
    if request.get("wheel_sha256") != _hash(wheel.read_bytes()):
        raise ValueError("Consumer candidate wheel changed")
    for relative, expected in request["files"].items():
        if _hash((output / relative).read_bytes()) != expected:
            raise ValueError(f"Consumer input changed: {relative}")
    sources = request.get("workflow_sources")
    if not isinstance(sources, dict) or set(sources) != set(_WORKFLOW_SOURCES):
        raise ValueError("Consumer workflow source selection changed")
    for identity, path in _WORKFLOW_SOURCES.items():
        if (
            not isinstance(sources[identity], dict)
            or sources[identity].get("path") != path
            or sources[identity].get("sha256")
            != _hash((output / f"{identity}-original.py").read_bytes())
        ):
            raise ValueError(f"Frozen consumer workflow source changed: {identity}")
    original = (output / "qualification-original.py").read_text()
    if request.get("source_sha256") != _hash(original.encode()):
        raise ValueError("Frozen consumer source changed")
    adapted, lines = adapt_numerical_source(original)
    if (output / "qualification-adapted.py").read_text() != adapted or request[
        "adaptation"
    ] != {"fit_to_estimate_lines": lines, "numerics_changed": False}:
        raise ValueError("Consumer numerical adaptation changed")
    spatial_code, spatial_lines = adapt_spatial_covariance_source(
        (output / "consumer-spatial-covariance-original.py").read_text()
    )
    if (
        output / "spatial-covariance-adapted.py"
    ).read_text() != spatial_code or request["workflow_sources"][
        "consumer-spatial-covariance"
    ]["fit_to_estimate_lines"] != spatial_lines:
        raise ValueError("Consumer Spatial covariance adaptation changed")
    for name, source in {
        "consumer_fixtures.py": Path(__file__).with_name("consumer_fixtures.py"),
        "wheel_installation.py": Path(__file__).resolve().parents[3]
        / "studies/src/statadic_studies/installation.py",
    }.items():
        if (output / name).read_bytes() != source.read_bytes():
            raise ValueError(f"Consumer harness changed: {name}")
    expected = {
        "consumer-dyadic",
        "consumer-frontier",
        "consumer-spatial-covariance",
        "consumer-operating-game",
        "hotelling",
    }
    contracts = result.get("consumer_contracts", {})
    if (
        result.get("status") != "passed"
        or result.get("request") != request
        or set(contracts) != expected
        or any(record.get("status") != "passed" for record in contracts.values())
    ):
        raise ValueError("Required consumer fixtures did not all pass")
    if result.get("installed_wheel") != {
        "wheel_sha256": request["wheel_sha256"],
        "payload_verified": True,
    }:
        raise ValueError("Consumer fixtures lack the executed installed-wheel binding")
    return result
