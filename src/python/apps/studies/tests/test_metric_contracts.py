"""Independent campaign witnesses for mixed procedures and honest operating rates."""

from __future__ import annotations

import pytest
from statadic_studies import campaign
from statadic_studies.analysis import export, summarize, verify_export
from statadic_studies.contracts import STUDIES


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(campaign, "environment", lambda: {"fixture": "locked"})


def point():
    return {
        "valid": True,
        "estimate": 0.4,
        "truth": 0.4,
        "covered": True,
        "width": 0.2,
        "pvalue": 1.0,
        "rejected": False,
    }


def test_mixed_estimators_and_tests_use_separate_contracts(
    tmp_path, monkeypatch, isolated
):
    def fake(study, scenario, replicate, seed, *, draws):
        if replicate:
            return {
                method: {"valid": False, "failure": "procedure-unavailable"}
                for method in STUDIES[study].methods
            }
        return {
            "identity-GMM": point(),
            "iterated-GMM": point(),
            "Hansen-J": {"valid": True, "pvalue": 0.01, "rejected": True},
        }

    monkeypatch.setattr(campaign, "run_trial", fake)
    root = tmp_path / "gmm"
    campaign.create(root, "gmm.affine-mean", trials=2)
    campaign.execute(root)
    result = summarize(root)
    null = result["scenarios"]["correct"]["methods"]["Hansen-J"]
    alternative = result["scenarios"]["misspecified"]["methods"]["Hansen-J"]
    assert "conditional_bias" not in null
    assert null["size_failure_inclusive"]["mean"] == 1.0
    assert alternative["power"]["mean"] == 0.5
    assert alternative["conditional_rejection"]["mean"] == 1.0
    assert "qualification" not in alternative
    point_method = result["scenarios"]["misspecified"]["methods"]["identity-GMM"]
    assert point_method["coverage_all_attempts"]["mean"] == 0.5
    assert "size_status" not in point_method["qualification"]
    assert "power" not in point_method
    destination = tmp_path / "publication"
    export(root, destination)
    assert verify_export(destination) == result
    assert "Hansen-J" in (destination / "results.md").read_text()


def test_coverage_only_needs_no_fake_decisions(tmp_path, monkeypatch, isolated):
    def fake(study, scenario, replicate, seed, *, draws):
        return {
            method: {
                "valid": True,
                "estimate": 0.1,
                "truth": 0.0,
                "covered": True,
                "width": 0.3,
                "se": 0.1,
                "true_variance": 0.02,
                "inner_draws": draws,
            }
            for method in STUDIES[study].methods
        }

    monkeypatch.setattr(campaign, "run_trial", fake)
    root = tmp_path / "coverage"
    campaign.create(root, "bootstrap.mean-coverage", trials=2, profile="smoke")
    campaign.execute(root, stop=1)
    report = summarize(root)["scenarios"]["rho0-n64"]["methods"]["iid-percentile"]
    assert report["qualification"]["coverage_status"] == "incomplete"
    assert "size_status" not in report["qualification"]
    assert "conditional_rejection" not in report
    assert report["conditional_variance_ratio"]["mean"] == pytest.approx(0.5)


def test_decision_only_preserves_failure_denominator(tmp_path, monkeypatch, isolated):
    def fake(study, scenario, replicate, seed, *, draws):
        return {
            method: {
                "valid": True,
                "rejected": False,
                "null_hypothesis": True,
                "statistic": 0.0,
                "inner_draws": draws,
            }
            if not replicate
            else {"valid": False, "failure": "no-decision"}
            for method in STUDIES[study].methods
        }

    monkeypatch.setattr(campaign, "run_trial", fake)
    root = tmp_path / "decision"
    campaign.create(root, "multiplicity.equivalence-family", trials=2)
    campaign.execute(root)
    report = summarize(root)["scenarios"]["global-null"]["methods"]["bonferroni"]
    assert report["valid"] == report["failed"] == 1
    assert report["size_failure_inclusive"]["mean"] == 0.5
    assert "conditional_bias" not in report
    assert "conditional_mse" not in report


def test_explicit_frozen_draw_budget_and_null_classification(
    tmp_path, monkeypatch, isolated
):
    with pytest.raises(ValueError, match="999"):
        campaign.create(
            tmp_path / "bad", "permutation.bernoulli-size-power", trials=2, draws=32
        )
    observed = []

    def fake(study, scenario, replicate, seed, *, draws):
        observed.append(draws)
        return {
            method: {
                "valid": True,
                "estimate": 0.0,
                "truth": 0.0,
                "pvalue": 1.0,
                "rejected": False,
                "null_hypothesis": scenario.startswith("null"),
                "inner_draws": draws,
            }
            for method in STUDIES[study].methods
        }

    monkeypatch.setattr(campaign, "run_trial", fake)
    root = tmp_path / "effort"
    campaign.create(root, "permutation.bernoulli-size-power", trials=1, profile="smoke")
    campaign.execute(root)
    assert observed == [32] * 4
    alternative = summarize(root)["scenarios"]["alternative-equal"]["methods"][
        "linear-mmd-U"
    ]
    assert "size_status" not in alternative.get("qualification", {})
    assert alternative["power"]["mean"] == 0


def test_definition_imports_do_not_require_registry_cycle():
    from statadic_studies.definitions.dependence.core import DEPENDENCE_STUDIES
    from statadic_studies.definitions.estimators import ESTIMATOR_STUDIES
    from statadic_studies.definitions.inference import INFERENCE_STUDIES

    assert set(DEPENDENCE_STUDIES) | set(ESTIMATOR_STUDIES) | set(
        INFERENCE_STUDIES
    ) <= set(STUDIES)


def test_descriptive_design_never_inherits_qualification_from_good_outcomes():
    from dataclasses import replace

    from statadic_studies.analysis import _operating_summary

    definition = STUDIES["linear.ols-covariance"]
    outcomes = [
        {"valid": True, "rejected": False, "covered": True, "width": 0.1}
    ] * 10000
    for policy, expected in (
        ("descriptive", "descriptive"),
        ("size-and-coverage", "qualified"),
    ):
        study = replace(definition, qualification=policy).record()
        report = _operating_summary(
            outcomes,
            {"size", "coverage"},
            study,
            definition.scenarios[0],
            True,
            "evaluation",
        )["qualification"]
        assert report["size_status"] == report["coverage_status"] == expected
        assert report["size_upper"] < 0.05
        assert report["coverage_lower"] > 0.95


def test_kkp_lambda_never_claims_fabricated_interval_contract():
    from statadic_studies.study_types import metrics_for

    study = STUDIES["spatial.kkp-components"].record()
    assert "coverage" not in metrics_for(study, "identity-moments-lambda")
    assert "coverage" in metrics_for(study, "identity-moments-beta")


def test_export_context_uses_frozen_design_after_current_source_changes(
    tmp_path, monkeypatch, isolated
):
    from dataclasses import replace

    root = tmp_path / "historical"
    study = STUDIES["mmd.finite-bias"]
    campaign.create(root, study.id, trials=1, profile="smoke")
    monkeypatch.setitem(
        STUDIES, study.id, replace(study, aim="New aim must not enter old evidence")
    )
    destination = tmp_path / "publication"
    export(root, destination)
    text = (destination / "results.md").read_text()
    assert study.aim in text
    assert "New aim must not enter old evidence" not in text
    for fragment in (
        study.id,
        study.dgp,
        study.estimand,
        "smoke",
        "1 requested",
        "imported-source-only",
        "alpha=0.05",
        "failure budget=0.005",
        "[Complete summary](summary.json)",
    ):
        assert fragment in text


def test_all_registered_designs_have_valid_explicit_contracts():
    from statadic_studies.study_types import metrics_for, validate_definition

    assert len(STUDIES) >= 44
    for study in STUDIES.values():
        validate_definition(study.record())
    study = STUDIES["gmm.variability-and-J"].record()
    assert "power" not in metrics_for(study, "fixed-plain-Wald")
    assert "power" in metrics_for(study, "profiled-plain-J")


def test_nonnegative_loss_summary_is_risk_not_signed_bias():
    from statadic_studies.analysis import _method_summary
    from statadic_studies.campaign import _validate_outcomes

    design = STUDIES["geometry.population-loss"].record()
    methods = design["methods"]
    outputs = [{"valid": True, "loss_value": value} for value in (1.0, 3.0)]
    for output in outputs:
        _validate_outcomes(dict.fromkeys(methods, output), design)
    outputs.append({"valid": False, "failure": "procedure-unavailable"})
    report = _method_summary(
        outputs, {"loss", "failures"}, design, "n64", True, "smoke"
    )
    assert report["valid"] == 2 and report["failed"] == 1
    assert report["failure_rate"]["mean"] == pytest.approx(1 / 3)
    assert report["conditional_mean_loss"]["mean"] == 2.0
    assert report["conditional_mean_loss"]["mcse"] == 1.0
    assert report["conditional_mean_squared_loss"]["mean"] == 5.0
    assert report["conditional_mean_squared_loss"]["mcse"] == 4.0
    assert "conditional_bias" not in report
    with pytest.raises(ValueError, match="nonnegative loss"):
        _validate_outcomes(
            dict.fromkeys(methods, {"valid": True, "loss_value": -1}), design
        )


def test_one_sided_permutation_composite_nulls_are_method_specific(
    tmp_path, monkeypatch, isolated
):
    root = tmp_path / "tails"
    identity = "permutation.signed-mean-tails"
    campaign.create(root, identity, trials=1, profile="smoke")

    def trial(study, scenario, replicate, seed, *, draws):
        return {
            method: {"valid": True, "pvalue": 0.5, "rejected": False}
            for method in STUDIES[study].methods
        }

    monkeypatch.setattr(campaign, "run_trial", trial)
    campaign.execute(root)
    report = summarize(root)["scenarios"]
    positive = report["positive-shift"]["methods"]
    negative = report["negative-shift"]["methods"]
    assert positive["less"]["null_hypothesis"]
    assert not positive["greater"]["null_hypothesis"]
    assert negative["greater"]["null_hypothesis"]
    assert not negative["less"]["null_hypothesis"]
    assert "power" not in positive["less"]
    assert "size_failure_inclusive" in positive["less"]
    assert "power" in positive["greater"]
