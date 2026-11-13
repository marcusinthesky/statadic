"""A declared core cannot inherit unrelated or merely completed evidence."""

import json
from types import SimpleNamespace

import pytest
from statadic_docs_labs.qualification_scope import (
    read_scope,
    require_contract_evidence,
    require_scientific_results,
    resolve_scope,
)


def scope():
    return {
        "schema_version": 1,
        "id": "bounded-alpha",
        "required_variants": ["point.exact"],
        "numerical_contracts": ["weighted-state"],
        "consumer_contracts": ["downstream"],
        "performance_workloads": ["complete-fit"],
    }


@pytest.mark.parametrize(
    "field",
    [
        "required_variants",
        "numerical_contracts",
        "consumer_contracts",
        "performance_workloads",
    ],
)
@pytest.mark.parametrize("items", [[], ["duplicate", "duplicate"], [""]])
def test_invalid_selections_fail_closed(tmp_path, field, items):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope() | {field: items}))
    with pytest.raises(ValueError, match="release scope"):
        read_scope(path)


@pytest.mark.parametrize("policy", ["unknown", None, False, {}])
def test_unknown_performance_policy_is_rejected(tmp_path, policy):
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope() | {"performance_policy": policy}))
    with pytest.raises(ValueError, match="performance policy"):
        read_scope(path)


def test_scope_freezes_exact_variant_and_keeps_unrelated_gaps(tmp_path, monkeypatch):
    from statadic_studies import catalogue

    monkeypatch.setattr(
        catalogue, "check_catalogue", lambda _: {"missing_study": ["unrelated"]}
    )
    path = tmp_path / "src/python/apps/studies/catalogue.json"
    path.parent.mkdir(parents=True)
    variant = {
        "id": "point.exact",
        "selectors": {"method": "fixed"},
        "study_mappings": [],
        "source_review": "source-unresolved",
    }
    path.write_text(json.dumps({"scientific_procedures": [{"variants": [variant]}]}))
    resolved = resolve_scope(tmp_path, scope())
    assert resolved["missing_study"] == ["point.exact"]
    assert resolved["source_review_pending"] == ["point.exact"]
    assert resolved["catalogue_coverage"]["missing_study"] == ["unrelated"]
    with pytest.raises(ValueError, match="Unknown required"):
        resolve_scope(tmp_path, scope() | {"required_variants": ["missing"]})


@pytest.mark.parametrize(
    "status", ["inconclusive", "excess-error", "descriptive", "smoke-only"]
)
def test_completed_required_cell_must_actually_qualify(status):
    resolved = {
        "variants": {
            "v": {
                "id": "v",
                "study_mappings": [
                    {"study": "s", "scenarios": ["null"], "methods": ["m"]}
                ],
            }
        }
    }
    records = [
        {
            "study": "s",
            "summary": {
                "scenarios": {
                    "null": {
                        "methods": {
                            "m": {
                                "metrics": ["coverage"],
                                "qualification": {"coverage_status": status},
                            }
                        }
                    }
                }
            },
        }
    ]
    with pytest.raises(ValueError, match="not qualified"):
        require_scientific_results(resolved, records)


def test_numerical_and_resource_obligations_cannot_be_omitted_or_averaged():
    resolved = {"request": scope(), "fingerprint": "frozen"}
    with pytest.raises(ValueError, match="Missing passing numerical"):
        require_contract_evidence(resolved, {})
    passed = {
        "status": "passed",
        "scope_fingerprint": "frozen",
        "wheel_sha256": "wheel",
        "lane": "actual-execution",
    }
    evidence = {
        field: {identity: dict(passed) for identity in scope()[field]}
        for field in (
            "numerical_contracts",
            "consumer_contracts",
            "performance_workloads",
        )
    }
    resource = evidence["performance_workloads"]["complete-fit"]
    resource["sessions"] = [
        {"id": i, "warm_time_ratio": 0.8, "total_peak_rss_ratio": 0.8} for i in range(3)
    ]
    require_contract_evidence(resolved, evidence)
    resource["sessions"][2]["total_peak_rss_ratio"] = 0.91
    with pytest.raises(ValueError, match="resource win failed"):
        require_contract_evidence(resolved, evidence)


def report_evidence():
    request = scope() | {"performance_policy": "report"}
    resolved = {"request": request, "fingerprint": "frozen"}
    passed = {
        "status": "passed",
        "scope_fingerprint": "frozen",
        "wheel_sha256": "wheel",
        "lane": "actual-execution",
    }
    evidence = {
        field: {identity: dict(passed) for identity in request[field]}
        for field in (
            "numerical_contracts",
            "consumer_contracts",
            "performance_workloads",
        )
    }
    resource = evidence["performance_workloads"]["complete-fit"]
    resource.update(
        status="reported",
        performance_policy="report",
        measurement_purpose="pilot",
        cases=["bounded-case"],
        sessions=[{"id": "0", "warm_time_ratio": 2.5, "total_peak_rss_ratio": 7.0}],
    )
    return resolved, evidence, resource


def test_explicit_report_policy_accepts_one_adverse_bounded_measurement():
    resolved, evidence, resource = report_evidence()
    require_contract_evidence(resolved, evidence)
    assert resource["sessions"][0]["total_peak_rss_ratio"] == 7.0
    assert resource["status"] == "reported"
    evidence["numerical_contracts"]["weighted-state"]["status"] = "reported"
    with pytest.raises(ValueError, match="Missing passing numerical"):
        require_contract_evidence(resolved, evidence)


@pytest.mark.parametrize("ratio", [0, -1, float("inf"), float("nan"), True, None])
def test_report_policy_rejects_unusable_resource_ratios(ratio):
    resolved, evidence, resource = report_evidence()
    resource["sessions"][0]["total_peak_rss_ratio"] = ratio
    with pytest.raises(ValueError, match="measurement invalid"):
        require_contract_evidence(resolved, evidence)


@pytest.mark.parametrize(
    "field", ["cases", "sessions", "performance_policy", "measurement_purpose"]
)
def test_report_policy_requires_the_retained_bounded_observation_identity(field):
    resolved, evidence, resource = report_evidence()
    resource.pop(field)
    with pytest.raises(ValueError, match="performance report|resource sessions"):
        require_contract_evidence(resolved, evidence)


def test_report_record_does_not_satisfy_default_legacy_superiority_scope():
    resolved, evidence, _ = report_evidence()
    resolved["request"].pop("performance_policy")
    with pytest.raises(ValueError, match="Missing passing performance"):
        require_contract_evidence(resolved, evidence)


def point_records(monkeypatch, *, valid=10_000, failed=0, budget=0.005):
    from statadic_studies.contracts import STUDIES

    monkeypatch.setitem(
        STUDIES,
        "point-study",
        SimpleNamespace(
            record=lambda: {
                "failure_budget": budget,
            }
        ),
    )
    resolved = {
        "variants": {
            "point": {
                "id": "point",
                "study_mappings": [
                    {
                        "study": "point-study",
                        "scenarios": ["known-truth"],
                        "methods": ["point"],
                    }
                ],
            }
        }
    }
    cell = {
        "metrics": ["bias", "mse"],
        "valid": valid,
        "failed": failed,
        "conditional_bias": {"mean": 0.01, "mcse": 0.002, "n": valid},
        "conditional_mse": {"mean": 0.02, "mcse": 0.003, "n": valid},
    }
    records = [
        {
            "study": "point-study",
            "summary": {
                "scenarios": {
                    "known-truth": {
                        "attempted": valid + failed,
                        "requested": valid + failed,
                        "execution": "complete",
                        "methods": {"point": cell},
                    },
                }
            },
        }
    ]
    return resolved, records, cell


def test_all_failed_point_study_cannot_qualify_from_completion(monkeypatch):
    resolved, records, _ = point_records(monkeypatch, valid=0, failed=10_000)
    with pytest.raises(ValueError, match="usable trial results"):
        require_scientific_results(resolved, records)


@pytest.mark.parametrize("valid,failed,budget", [(9900, 100, 0.005), (10_000, 0, 1e-8)])
def test_point_failure_upper_bound_must_meet_the_frozen_budget(
    monkeypatch,
    valid,
    failed,
    budget,
):
    resolved, records, _ = point_records(
        monkeypatch, valid=valid, failed=failed, budget=budget
    )
    with pytest.raises(ValueError, match="frozen failure budget"):
        require_scientific_results(resolved, records)


def test_descriptive_point_metrics_need_finite_usable_estimates(monkeypatch):
    resolved, records, cell = point_records(monkeypatch)
    require_scientific_results(resolved, records)
    assert "qualification" not in cell
    cell["conditional_bias"]["mean"] = None
    with pytest.raises(ValueError, match="metric is unavailable"):
        require_scientific_results(resolved, records)
