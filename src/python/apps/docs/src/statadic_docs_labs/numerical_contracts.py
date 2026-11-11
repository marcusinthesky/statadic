"""Bind bounded numerical contract lanes to installed-wheel test evidence.

The large censored-panel qualification has no adapter here. Small Gaussian
controls cannot satisfy its separate 500 by 20 accuracy/refit obligation.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from statadic_evidence import fingerprint, sha256

CONTRACT_FILES = {
    "empirical-probabilities": (
        "sample/test_packed.py",
        "sample/test_packed_measures.py",
        "sample/test_evidence.py",
        "discrepancy/mmd/test_weighted.py",
    ),
    "energy-construction": (
        "discrepancy/test_provenance.py",
        "discrepancy/mmd/test_energy_independent_reference.py",
    ),
    "target-projections": (
        "geometry/barycentre/mmd/test_weights.py",
        "geometry/barycentre/mmd/energy/test_weights.py",
    ),
    "source-barycentres": (
        "geometry/barycentre/wasserstein/test_measure.py",
        "geometry/barycentre/wasserstein/test_measure_reference.py",
    ),
    "psd-spectral": ("optimize/test_psd.py",),
    "qp-kkt": (
        "optimize/test_simplex.py",
        "optimize/test_simplex_qp.py",
        "optimize/test_simplex_diagnostics.py",
    ),
    "jax-objective-derivatives": (
        "integration/geometry/test_simplex_objectives.py",
        "discrepancy/transport/test_univariate.py",
        "optimize/test_psd.py",
    ),
    "row-refit-schedules": (
        "inference/resampling/test_schedule_stationary_variance_reference.py",
        "inference/resampling/test_bootstrap.py",
    ),
    "spatial-operators": (
        "operators/test_spatial.py",
        "estimation/spatial/test_graph_likelihood_reference.py",
    ),
    "observed-panel-alignment": (
        "operators/test_panel.py",
        "api/test_survival_composition.py",
        "api/test_spatial_graph_regression.py",
    ),
    "ordinary-aft-prediction-functionals": (
        "model/survival/test_aft.py",
        "api/test_survival_host.py",
    ),
    "spatial-lognormal-prediction-functionals": ("api/test_survival_composition.py",),
}


def selected_files(contracts: list[str]) -> list[str]:
    """Select available bounded lanes, retaining unsupported obligations as gaps."""
    return sorted(
        {
            f"tests/{name}"
            for identity in contracts
            for name in CONTRACT_FILES.get(identity, ())
        }
    )


def read_results(report: Path, files: list[str]) -> dict:
    """Require execution of each named file, with no failures or skips."""
    cases = list(ET.parse(report).getroot().iter("testcase"))
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ):
        raise ValueError("Incomplete numerical contract lane")
    by_file = {}
    for filename in files:
        suffix = filename.removesuffix(".py").replace("/", ".")
        nodes = [
            f"{case.get('classname')}::{case.get('name')}"
            for case in cases
            if (
                case.get("classname", "") == suffix
                or case.get("classname", "").startswith(suffix + ".")
                or ("." + suffix + ".") in ("." + case.get("classname", "") + ".")
            )
        ]
        if not nodes:
            raise ValueError(f"Numerical contract file was not executed: {filename}")
        by_file[filename] = nodes
    return by_file


def retain_results(bundle: Path, *, scope: dict, wheel: Path, package: Path) -> dict:
    """Create evidence only for named contracts whose actual wheel lane passed."""
    files = selected_files(scope["request"]["numerical_contracts"])
    report = bundle / "numerical-contracts.xml"
    by_file = read_results(report, files)
    source_files = {name: sha256(package / name) for name in files}
    definition = fingerprint(CONTRACT_FILES)
    result = {
        identity: {
            "status": "passed",
            "scope_fingerprint": scope["fingerprint"],
            "wheel_sha256": sha256(wheel),
            "lane": "numerical-contracts",
            "report_sha256": sha256(report),
            "definition_fingerprint": definition,
            "files": {name: source_files[name] for name in selected_files([identity])},
            "nodes": {name: by_file[name] for name in selected_files([identity])},
        }
        for identity in scope["request"]["numerical_contracts"]
        if identity in CONTRACT_FILES
    }
    (bundle / "numerical-contracts.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def verify_results(bundle: Path, *, scope: dict, wheel: Path, package: Path) -> dict:
    """Re-derive retained numerical evidence without trusting passing statuses."""
    retained = json.loads((bundle / "numerical-contracts.json").read_text())
    files = selected_files(scope["request"]["numerical_contracts"])
    by_file = read_results(bundle / "numerical-contracts.xml", files)
    for identity, record in retained.items():
        if (
            identity not in scope["request"]["numerical_contracts"]
            or identity not in CONTRACT_FILES
        ):
            raise ValueError("Unexpected retained numerical contract")
        expected_files = {
            name: sha256(package / name) for name in selected_files([identity])
        }
        if record != {
            "status": "passed",
            "scope_fingerprint": scope["fingerprint"],
            "wheel_sha256": sha256(wheel),
            "lane": "numerical-contracts",
            "report_sha256": sha256(bundle / "numerical-contracts.xml"),
            "definition_fingerprint": fingerprint(CONTRACT_FILES),
            "files": expected_files,
            "nodes": {name: by_file[name] for name in expected_files},
        }:
            raise ValueError(f"Numerical evidence changed: {identity}")
    return retained
