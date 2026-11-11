"""Scope supplemental statistical tests without weakening canonical campaigns.

Every selected variant still needs its exact current campaign cells. The nodes
here are additional prespecified procedures, never replacements for those cells
or for a numerical contract such as the large joint-Gaussian accuracy gate.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from statadic_evidence import fingerprint, sha256


def _nodes(module: str, function: str, *parameters: str) -> tuple[str, ...]:
    base = f"tests/qualification/{module}.py::{function}"
    return (
        tuple(f"{base}[{parameter}]" for parameter in parameters)
        if parameters
        else (base,)
    )


_OLS = _nodes("test_linear", "test_gaussian_linear_native_inference", "ols")
_WLS = _nodes("test_linear", "test_gaussian_linear_native_inference", "wls")
_DIAGNOSTICS = _nodes(
    "test_linear_diagnostics", "test_gaussian_null_size_of_residual_diagnostics"
)
_SAR = _nodes(
    "test_panels",
    "test_gaussian_sar_native_information_inference",
    "rho0-n48",
    "rho0.4-n96",
)
_COVARIANCE = _nodes(
    "test_covariance",
    "test_shrinkage_population_loss_distribution",
    "misspecified-target",
    "matched-target",
)

SUPPLEMENTAL_NODES = {
    "linear.ols-wls.ols": _OLS,
    "linear.ols-wls.analytic-precision-wls": _WLS,
    "linear.covariance.classical": _OLS + _WLS,
    "parameter.inference.student-t-coefficient-interval": _OLS + _WLS,
    "parameter.inference.f-linear-restriction": _OLS + _WLS,
    "parameter.inference.normal-coefficient-interval": _SAR,
    "parameter.inference.chi-square-linear-restriction": _SAR,
    "linear.residual-diagnostics.breusch-godfrey-1": _DIAGNOSTICS,
    "linear.residual-diagnostics.breusch-godfrey-4": _DIAGNOSTICS,
    "linear.residual-diagnostics.breusch-pagan-original": _DIAGNOSTICS,
    "linear.residual-diagnostics.breusch-pagan-studentized": _DIAGNOSTICS,
    "linear.residual-diagnostics.jarque-bera": _DIAGNOSTICS,
    "linear.residual-diagnostics.reset-chi2": _DIAGNOSTICS,
    "linear.residual-diagnostics.reset-f": _DIAGNOSTICS,
    "linear.residual-diagnostics.white": _DIAGNOSTICS,
    "gmm.tests.efficient-retained-fit-hansen-j": _nodes(
        "test_moments", "test_fitted_hansen_j_strongly_identified_iid_mean_model"
    ),
    "panel.estimation.within": _nodes(
        "test_panels", "test_gaussian_panel_native_t_inference", "within"
    ),
    "spatial.covariance-impacts.sar-expected-information-covariance": _SAR,
    (
        "dyadic.estimation-bootstrap.multinomial-node-bootstrap-coefficient-inference"
    ): _nodes(
        "test_panels", "test_dyadic_complete_node_bootstrap_coverage", "n16", "n32"
    ),
    "covariance.sample-shrinkage.sample-ddof0": _COVARIANCE,
    "covariance.sample-shrinkage.ledoit-wolf-constant-correlation": _COVARIANCE,
    "mantel.inference.pearson-node-permutation": _nodes(
        "test_resampling", "test_mantel_independent_node_null", "n8", "n16"
    ),
    "sampler.alternatives.normal": _nodes(
        "test_samples", "test_sampler_known_marginal_law", "normal"
    ),
    "sampler.alternatives.mean-shift": _nodes(
        "test_samples", "test_sampler_known_marginal_law", "mean-shift"
    ),
    "sampler.alternatives.scale": _nodes(
        "test_samples", "test_sampler_known_marginal_law", "scale"
    ),
    "sampler.alternatives.student-t": _nodes(
        "test_samples", "test_sampler_known_marginal_law", "student-t12"
    )
    + _nodes("test_samples", "test_student_t_common_scale_joint_law"),
    "sampler.alternatives.mixture": _nodes(
        "test_samples", "test_deterministic_contamination_population_mean"
    ),
    "survival.independent.weibull-ordinary-observed-information-wald": _nodes(
        "test_survival",
        "test_independent_aft_wald_qualification",
        "ordinary-n128-weibull",
        "ordinary-n512-weibull",
    ),
    "survival.independent.weibull-delayed-observed-information-wald": _nodes(
        "test_survival",
        "test_independent_aft_wald_qualification",
        "delayed-n128-weibull",
        "delayed-n512-weibull",
    ),
    "survival.independent.lognormal-ordinary-observed-information-wald": _nodes(
        "test_survival",
        "test_independent_aft_wald_qualification",
        "ordinary-n128-lognormal",
        "ordinary-n512-lognormal",
    ),
    "survival.independent.lognormal-delayed-observed-information-wald": _nodes(
        "test_survival",
        "test_independent_aft_wald_qualification",
        "delayed-n128-lognormal",
        "delayed-n512-lognormal",
    ),
}

_REPORT = "qualification-scoped.xml"
_RETAINED = "qualification-scoped.json"
_COMMON_SOURCES = (
    "pyproject.toml",
    "tests/conftest.py",
    "tests/_validation/calibration.py",
)


def _routes(scope: dict) -> dict:
    required = scope["request"]["required_variants"]
    if (
        not required
        or len(required) != len(set(required))
        or set(required) != set(scope["variants"])
    ):
        raise ValueError("Qualification variants differ from the required scope")
    routes = {}
    for identity in sorted(required):
        variant = scope["variants"][identity]
        mappings = variant.get("study_mappings")
        if (
            variant.get("id") != identity
            or not isinstance(mappings, list)
            or not mappings
        ):
            raise ValueError(
                f"Required variant has no canonical campaign cells: {identity}"
            )
        cells = set()
        for mapping in mappings:
            if (
                not isinstance(mapping.get("study"), str)
                or not mapping["study"].strip()
                or not isinstance(mapping.get("scope"), str)
                or not mapping["scope"].strip()
                or any(
                    not isinstance(mapping.get(field), list)
                    or not mapping[field]
                    or any(
                        not isinstance(value, str) or not value.strip()
                        for value in mapping[field]
                    )
                    for field in ("methods", "scenarios")
                )
            ):
                raise ValueError(f"Invalid canonical qualification mapping: {identity}")
            cells.update(
                (mapping["study"], scenario, method)
                for scenario in mapping["scenarios"]
                for method in mapping["methods"]
            )
        nodes = sorted(SUPPLEMENTAL_NODES.get(identity, ()))
        routes[identity] = {
            "campaign_cells": [
                {"study": study, "scenario": scenario, "method": method}
                for study, scenario, method in sorted(cells)
            ],
            "supplemental_nodes": nodes,
            "supplemental_disposition": "required"
            if nodes
            else "campaign-backed-no-supplement",
        }
    return routes


def selected_nodes(scope: dict) -> list[str]:
    """Return exact stable supplemental nodes; every variant stays campaign-backed.

    An empty result means no supplemental execution is required. Callers must
    skip pytest in that case, rather than falling back to an unscoped selection.
    """
    return sorted(
        {
            node
            for route in _routes(scope).values()
            for node in route["supplemental_nodes"]
        }
    )


def read_results(report: Path, nodes: list[str]) -> list[str]:
    """Require every expected parameter cell exactly once, without extra cases."""
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError(
            "Supplemental qualification requires nonempty unique expected nodes"
        )
    root = ET.parse(report).getroot()
    if any(element.tag in ("failure", "error", "skipped") for element in root.iter()):
        raise ValueError("Incomplete scoped qualification lane")
    actual = [
        f"{case.get('classname', '').replace('.', '/')}.py::{case.get('name', '')}"
        for case in root.iter("testcase")
    ]
    if len(actual) != len(set(actual)) or sorted(actual) != sorted(nodes):
        raise ValueError("Scoped qualification nodes are missing, extra or duplicated")
    return sorted(actual)


def _evidence(bundle: Path, *, scope: dict, wheel: Path, package: Path) -> dict:
    routes = _routes(scope)
    nodes = selected_nodes(scope)
    report = bundle / _REPORT
    if nodes:
        executed = read_results(report, nodes)
        sources = set(_COMMON_SOURCES) | {node.split("::", 1)[0] for node in nodes}
        if any(
            "test_dyadic_complete_node_bootstrap_coverage" in node for node in nodes
        ):
            sources.add("tests/_dyadic.py")
    else:
        if report.exists():
            raise ValueError(
                "Unexpected qualification report for campaign-only selection"
            )
        executed, sources = [], set()
    return {
        "schema_version": 1,
        "status": "passed",
        "lane": "qualification-scoped",
        "supplemental_execution": "passed" if nodes else "not-required-campaign-backed",
        "campaign_qualification": "independently-required",
        "scope_fingerprint": scope["fingerprint"],
        "wheel_sha256": sha256(wheel),
        "definition_fingerprint": fingerprint(SUPPLEMENTAL_NODES),
        "owner_sha256": sha256(Path(__file__)),
        "routes": routes,
        "source_files": {name: sha256(package / name) for name in sorted(sources)},
        "expected_nodes": nodes,
        "executed_nodes": executed,
        "report_sha256": sha256(report) if nodes else None,
    }


def retain_results(bundle: Path, *, scope: dict, wheel: Path, package: Path) -> dict:
    """Bind exact executed tests, or an explicit campaign-only empty selection."""
    result = _evidence(bundle, scope=scope, wheel=wheel, package=package)
    (bundle / _RETAINED).write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_results(bundle: Path, *, scope: dict, wheel: Path, package: Path) -> dict:
    """Rederive current mapping/source/wheel/report bindings, not passing labels."""
    retained = json.loads((bundle / _RETAINED).read_text())
    if retained != _evidence(bundle, scope=scope, wheel=wheel, package=package):
        raise ValueError("Scoped qualification evidence changed")
    return retained
