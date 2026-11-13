"""Resolve an explicit release selection without hiding the complete catalogue."""

from __future__ import annotations

import json
import math
from pathlib import Path

from statadic_evidence import fingerprint


def read_scope(path: Path) -> dict:
    """Read an authored selection; qualification still requires executed evidence."""
    return validate_scope(json.loads(path.read_text()))


def validate_scope(scope: dict) -> dict:
    """Reject empty selections even when called through the Python interface."""
    if (
        scope.get("schema_version") != 1
        or not isinstance(scope.get("id"), str)
        or not scope["id"].strip()
    ):
        raise ValueError("Invalid release qualification scope")
    if scope.get("performance_policy", "superiority") not in ("superiority", "report"):
        raise ValueError("Invalid release scope performance policy")
    for field in (
        "required_variants",
        "numerical_contracts",
        "consumer_contracts",
        "performance_workloads",
    ):
        items = scope.get(field)
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item.strip() for item in items)
            or len(items) != len(set(items))
        ):
            raise ValueError(f"Empty, duplicate or invalid release scope: {field}")
    return scope


def resolve_scope(repository: Path, scope: dict) -> dict:
    """Freeze exact canonical variants and retain missing source/study gates."""
    from statadic_studies.catalogue import check_catalogue

    scope = validate_scope(scope)
    coverage = check_catalogue(repository)
    catalogue = json.loads(
        (repository / "src/python/apps/studies/catalogue.json").read_text()
    )
    variants = {
        variant["id"]: variant
        for procedure in catalogue["scientific_procedures"]
        for variant in procedure["variants"]
    }
    unknown = set(scope["required_variants"]) - variants.keys()
    if unknown:
        raise ValueError(f"Unknown required scientific variants: {sorted(unknown)}")
    selected = {identity: variants[identity] for identity in scope["required_variants"]}
    missing = [
        identity
        for identity, variant in selected.items()
        if not variant["study_mappings"]
    ]
    pending = [
        identity
        for identity, variant in selected.items()
        if variant.get("source_review") != "verified-method-source"
    ]
    return {
        "request": scope,
        "variants": selected,
        "fingerprint": fingerprint({"request": scope, "variants": selected}),
        "missing_study": missing,
        "source_review_pending": pending,
        "required_studies": sorted(
            {
                mapping["study"]
                for variant in selected.values()
                for mapping in variant["study_mappings"]
            }
        ),
        "catalogue_coverage": coverage,
    }


def require_scientific_results(resolved: dict, records: list[dict]) -> None:
    """Reject adverse required calibration cells, not unrelated study cells."""
    from statadic_studies.contracts import STUDIES

    indexed = {record["study"]: record["summary"] for record in records}
    checked = set()
    for variant in resolved["variants"].values():
        for mapping in variant["study_mappings"]:
            summary = indexed[mapping["study"]]
            for scenario in mapping["scenarios"]:
                for method in mapping["methods"]:
                    scenario_report = summary["scenarios"][scenario]
                    cell = scenario_report["methods"][method]
                    qualification = cell.get("qualification", {})
                    for key in ("size_status", "coverage_status"):
                        if key in qualification and qualification[key] != "qualified":
                            raise ValueError(
                                "Required scientific cell is not qualified: "
                                f"{variant['id']}/{scenario}/{method}/{key}"
                            )
                    if (
                        "coverage" in cell["metrics"]
                        and "coverage_status" not in qualification
                    ):
                        raise ValueError(
                            "Required coverage assessment is missing: "
                            f"{variant['id']}/{scenario}/{method}"
                        )
                    if (
                        "size" in cell["metrics"]
                        and cell.get("null_hypothesis") is True
                        and "size_status" not in qualification
                    ):
                        raise ValueError(
                            "Required size assessment is missing: "
                            f"{variant['id']}/{scenario}/{method}"
                        )
                    identity = (mapping["study"], scenario, method)
                    if identity not in checked:
                        # inspect_campaigns has already compared this current
                        # definition byte-for-byte with the frozen campaign.
                        definition = STUDIES[mapping["study"]].record()
                        _require_usable_results(cell, scenario_report, definition)
                        checked.add(identity)


def _require_usable_results(cell: dict, scenario: dict, study: dict) -> None:
    """Require frozen failure-budget evidence even for descriptive point studies."""
    from scipy.stats import beta

    valid, failed = cell.get("valid"), cell.get("failed")
    attempted, requested = scenario.get("attempted"), scenario.get("requested")
    if (
        any(type(value) is not int for value in (valid, failed, attempted, requested))
        or valid < 1
        or failed < 0
        or valid + failed != attempted
        or attempted != requested
        or scenario.get("execution") != "complete"
    ):
        raise ValueError("Required scientific cell lacks complete usable trial results")
    budget = study["failure_budget"]
    # Match the studies' one-sided 95% Clopper-Pearson failure convention.
    failure_upper = float(beta.ppf(0.95, failed + 1, valid))
    if not math.isfinite(failure_upper) or failure_upper > budget:
        raise ValueError(
            "Required scientific cell does not meet its frozen failure budget"
        )
    metrics = set(cell["metrics"])
    summaries = {
        "bias": ("conditional_bias",),
        "mse": ("conditional_mse",),
        "loss": ("conditional_mean_loss", "conditional_mean_squared_loss"),
        "frobenius_loss": ("conditional_frobenius_loss",),
        "se_accuracy": ("conditional_estimated_variance", "conditional_variance_ratio"),
    }
    for metric in metrics & summaries.keys():
        for name in summaries[metric]:
            estimate = cell.get(name, {})
            if estimate.get("n") != valid or any(
                type(estimate.get(field)) not in (int, float)
                or not math.isfinite(estimate[field])
                for field in ("mean", "mcse")
            ):
                raise ValueError(f"Required scientific metric is unavailable: {name}")
    # Finite bias/loss estimates remain descriptive. This failure gate does not
    # invent an accuracy threshold or upgrade descriptive studies to calibrated
    # size/coverage qualification.


def require_contract_evidence(resolved: dict, evidence: dict) -> None:
    """Keep required numerical, consumer and resource obligations fail-closed."""
    policy = resolved["request"].get("performance_policy", "superiority")
    if policy not in ("superiority", "report"):
        raise ValueError("Invalid release scope performance policy")
    for field in ("numerical_contracts", "consumer_contracts", "performance_workloads"):
        for identity in resolved["request"][field]:
            record = evidence.get(field, {}).get(identity)
            reporting = field == "performance_workloads" and policy == "report"
            status = "reported" if reporting else "passed"
            if not isinstance(record, dict) or record.get("status") != status:
                adjective = "reported" if reporting else "passing"
                raise ValueError(f"Missing {adjective} {field} evidence: {identity}")
            if record.get("scope_fingerprint") != resolved["fingerprint"]:
                raise ValueError(f"Contract evidence has a different scope: {identity}")
            # Only release execution can mint these records after the actual wheel
            # lanes and matched-resource verifier have completed.
            if not record.get("wheel_sha256") or not record.get("lane"):
                raise ValueError(
                    f"Contract evidence lacks wheel/lane identity: {identity}"
                )
            if field == "performance_workloads":
                if reporting and (
                    record.get("performance_policy") != "report"
                    or record.get("measurement_purpose")
                    not in ("pilot", "qualification")
                    or not isinstance(record.get("cases"), list)
                    or not record["cases"]
                    or any(
                        not isinstance(case, str) or not case
                        for case in record["cases"]
                    )
                    or len(set(record["cases"])) != len(record["cases"])
                ):
                    raise ValueError(
                        f"Missing explicit bounded performance report: {identity}"
                    )
                sessions = record.get("sessions", [])
                minimum = 1 if reporting else 3
                if (
                    not isinstance(sessions, list)
                    or any(
                        not isinstance(session, dict)
                        or type(session.get("id")) not in (str, int)
                        or session.get("id") == ""
                        for session in sessions
                    )
                    or len(sessions) < minimum
                    or len({session["id"] for session in sessions}) != len(sessions)
                ):
                    raise ValueError(
                        f"{'One' if reporting else 'Three'} independent resource "
                        f"sessions required: {identity}"
                    )
                for session in sessions:
                    for metric in ("warm_time_ratio", "total_peak_rss_ratio"):
                        ratio = session.get(metric)
                        if (
                            type(ratio) not in (int, float)
                            or not math.isfinite(ratio)
                            or ratio <= 0
                            or (not reporting and ratio > 0.90)
                        ):
                            outcome = (
                                "measurement invalid" if reporting else "win failed"
                            )
                            raise ValueError(
                                f"Required resource {outcome}: {identity}/{metric}"
                            )
