"""Registered additional families execute with honest failure denominators."""

import math
from dataclasses import replace

import jax.numpy as jnp
import pytest
from statadic_studies import analysis
from statadic_studies.adapters import extended, run_trial
from statadic_studies.definitions.extended import EXTENDED_STUDIES
from statadic_studies.study_types import validate_definition


def test_extended_designs_freeze_explicit_targets_and_comparison_groups():
    for study in EXTENDED_STUDIES.values():
        validate_definition(study.record())
        expected = (
            "coverage-only" if study.id.startswith("likelihood.") else "descriptive"
        )
        assert study.qualification == expected


def test_extended_adapter_rejects_unregistered_scenarios():
    """Do not infer a binary or IID cell for a misspelled scenario."""
    with pytest.raises(ValueError, match="explicitly registered"):
        run_trial("likelihood.probit-inference", "unknown", 0, 179)
    with pytest.raises(ValueError, match="explicitly registered"):
        run_trial("iv.workflow-inference", "unknown", 0, 179)


@pytest.mark.parametrize(
    "identity,scenario",
    [
        ("likelihood.gamma-inference", "ordinary"),
        ("likelihood.nb2-inference", "ordinary"),
        ("likelihood.probit-inference", "binary"),
        ("likelihood.probit-inference", "grouped"),
        ("survival.aft-mean-prediction", "weibull"),
        ("survival.aft-mean-prediction", "lognormal"),
        ("survival.aft-mean-prediction", "loglogistic"),
        ("survival.aft-mean-prediction", "loglogistic-delayed"),
        ("iv.workflow-inference", "iid"),
        ("iv.workflow-inference", "heteroskedastic"),
        ("iv.workflow-inference", "hac"),
        ("iv.workflow-inference", "clustered"),
        ("iv.workflow-inference", "weak"),
    ],
)
def test_extended_adapter_retains_each_attempt(identity, scenario):
    result = run_trial(identity, scenario, 0, 179)
    assert set(result) == set(EXTENDED_STUDIES[identity].methods)
    for outcome in result.values():
        if outcome["valid"]:
            assert math.isfinite(outcome["estimate"])
            assert math.isfinite(outcome["truth"])
            if "coverage" in EXTENDED_STUDIES[identity].metrics:
                assert outcome["width"] > 0
                assert outcome["se"] > 0
        else:
            assert outcome["failure"]
    if scenario in {
        "ordinary",
        "binary",
        "grouped",
        "weibull",
        "lognormal",
        "loglogistic",
        "loglogistic-delayed",
        "iid",
    }:
        assert any(outcome["valid"] for outcome in result.values()), (
            identity,
            scenario,
            result,
        )


def test_first_draw_failure_remains_in_coverage_denominator(monkeypatch):
    """An unavailable first draw stays attempted and counts as uncovered."""
    identity = "likelihood.gamma-inference"
    monkeypatch.setattr(
        extended,
        "_likelihood_kernel",
        lambda *_: lambda *_: (math.nan, math.nan, False),
    )
    failed = run_trial(identity, "ordinary", 0, 179)["fixed-model"]
    monkeypatch.setattr(
        extended, "_likelihood_kernel", lambda *_: lambda *_: (0.4, 0.25, True)
    )
    successful = run_trial(identity, "ordinary", 1, 179)["fixed-model"]
    summary = analysis._method_summary(
        [failed, successful],
        {"bias", "mse", "coverage", "width", "failures"},
        EXTENDED_STUDIES[identity].record(),
        "ordinary",
        True,
        "smoke",
    )
    assert summary["valid"] == 1 and summary["failed"] == 1
    assert summary["failure_rate"]["mean"] == 0.5
    assert summary["coverage_all_attempts"]["mean"] == 0.5
    assert summary["conditional_coverage"]["mean"] == 1.0


def test_iterated_study_excludes_unconverged_weight_sequence(monkeypatch):
    """A valid finite criterion solve cannot masquerade as converged iteration."""
    from statadic.estimation import iv

    class Unconverged:
        def __init__(self, *_args):
            pass

        def estimate(self, problem):
            fit = iv.TwoStageLeastSquares().estimate(problem)
            assert bool(fit.valid)
            return replace(fit, converged=jnp.asarray(False))

    monkeypatch.setattr(iv, "IteratedIVGMM", Unconverged)
    result = run_trial("iv.workflow-inference", "iid", 0, 179)
    assert result["2SLS"]["valid"]
    assert not result["iterated"]["valid"]
    assert result["iterated"]["failure"]
