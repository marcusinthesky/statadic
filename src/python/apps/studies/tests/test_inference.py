"""Truth witnesses and complete procedure smoke runs, distinct size/power semantics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from statadic_studies.adapters import inference
from statadic_studies.definitions.inference import (
    INFERENCE_NULL_SCENARIOS,
    INFERENCE_STUDIES,
)


def test_population_discrepancies_enumerate_finite_laws():
    p, q = 0.2, 0.6
    support = (0.0, 1.0)
    first = (1 - p, p)
    second = (1 - q, q)
    for method, kernel in (
        ("linear-mmd-U", lambda a, b: a * b),
        ("rbf-mmd-V", lambda a, b: np.exp(-((a - b) ** 2) / 2)),
    ):
        expected = sum(
            first[i] * first[j] * kernel(a, b)
            + second[i] * second[j] * kernel(a, b)
            - 2 * first[i] * second[j] * kernel(a, b)
            for i, a in enumerate(support)
            for j, b in enumerate(support)
        )
        assert inference.bernoulli_discrepancy_truth(method, p, q) == pytest.approx(
            expected
        )
    assert inference.bernoulli_discrepancy_truth(
        "energy-euclidean-V", p, q
    ) == pytest.approx(0.32)


def test_stationary_mean_variance_matches_covariance_sum():
    for n in (4, 16):
        for rho in (0.0, 0.3, 0.7):
            covariance = rho ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            assert inference.ar1_mean_variance(n, rho) == pytest.approx(
                covariance.sum() / n**2
            )


def test_null_and_alternative_classification_is_prespecified():
    assert INFERENCE_NULL_SCENARIOS["permutation.bernoulli-size-power"] == (
        "null-equal",
        "null-unequal",
    )
    assert INFERENCE_NULL_SCENARIOS["equivalence.bernoulli-margin"] == (
        "boundary-null",
        "separated-null",
    )
    assert "equivalent" not in INFERENCE_NULL_SCENARIOS["equivalence.bernoulli-margin"]
    for study, definition in INFERENCE_STUDIES.items():
        assert set(INFERENCE_NULL_SCENARIOS.get(study, ())) <= set(definition.scenarios)


@pytest.mark.parametrize("scenario", ["null-unequal", "alternative-equal"])
def test_complete_permutation_procedures(scenario):
    first = inference.run_inference(
        "permutation.bernoulli-size-power", scenario, 3, 20261104, draws=32
    )
    repeated = inference.run_inference(
        "permutation.bernoulli-size-power", scenario, 3, 20261104, draws=32
    )
    assert first == repeated
    assert len(first) == 5
    for record in first.values():
        assert record["valid"]
        assert record["inner_draws"] == 32
        assert record["null_hypothesis"] == scenario.startswith("null")
        assert record["pvalue"] >= 1 / 33
        assert record["rejected"] == (record["pvalue"] <= 0.05)


def test_complete_bootstrap_and_block_selection():
    outcomes = inference.run_inference(
        "bootstrap.mean-coverage", "rho0.3-n64", 2, 20261104, draws=32
    )
    assert len(outcomes) == 7
    for record in outcomes.values():
        assert record["valid"]
        assert record["truth"] == 0
        assert record["width"] >= 0
        assert record["true_variance"] > 1 / 64
        assert record["block_length"] >= 1
        assert "pvalue" not in record
    assert outcomes["stationary-length4"]["block_length"] == 4


def test_equivalence_margin_mapping_and_failure_accounting(monkeypatch):
    result = SimpleNamespace(
        equivalence_pvalue=0.01,
        energy_point_estimate=0.03,
        energy_ci=(0.01, 0.07),
        bootstrap_distribution=np.array([0.02, 0.03]),
        solver_converged=True,
    )
    monkeypatch.setattr(inference, "_equivalence_result", lambda *args: result)
    alternative = inference.run_inference(
        "equivalence.bernoulli-margin", "equivalent", 0, 3, draws=32
    )
    boundary = inference.run_inference(
        "equivalence.bernoulli-margin", "boundary-null", 0, 3, draws=32
    )
    assert all(
        not item["null_hypothesis"] and item["rejected"]
        for item in alternative.values()
    )
    assert all(
        item["null_hypothesis"] and item["rejected"] for item in boundary.values()
    )
    assert boundary["one-candidate"]["truth"] == pytest.approx(0.18)

    def unavailable(*args):
        raise ValueError("numerical procedure unavailable")

    monkeypatch.setattr(inference, "_equivalence_result", unavailable)
    failed = inference.run_inference(
        "equivalence.bernoulli-margin", "boundary-null", 0, 3, draws=32
    )
    assert all(
        not item["valid"] and item["null_hypothesis"] for item in failed.values()
    )


def test_multiplicity_uses_actual_rules_without_fabricating_family_pvalue(monkeypatch):
    pvalues = iter([0.01, 0.024, 0.9])

    def result():
        return SimpleNamespace(
            equivalence_pvalue=next(pvalues),
            solver_converged=True,
            energy_point_estimate=0.18,
            energy_ci=(0.01, 0.4),
            bootstrap_distribution=np.array([0.1, 0.2]),
        )

    monkeypatch.setattr(inference, "_equivalence_result", lambda *args: result())
    outcomes = inference.run_inference(
        "multiplicity.equivalence-family", "global-null", 0, 9, draws=32
    )
    assert outcomes["bonferroni"]["decisions"] == [True, False, False]
    assert outcomes["holm"]["decisions"] == [True, True, False]
    assert all(
        "pvalue" not in item and "truth" not in item for item in outcomes.values()
    )


def test_complete_equivalence_public_procedure():
    outcomes = inference.run_inference(
        "equivalence.bernoulli-margin", "boundary-null", 0, 20261104, draws=32
    )
    assert set(outcomes) == {"one-candidate", "two-candidates"}
    assert all(record["valid"] for record in outcomes.values())
    for record in outcomes.values():
        assert record["inner_draws"] == 32
        assert record["null_hypothesis"]
        if record["valid"]:
            assert record["truth"] == pytest.approx(0.18)
            assert 1 / 33 <= record["pvalue"] <= 1
        else:
            assert record["failure"]


def test_evaluation_budget_uses_999_inner_draws(monkeypatch):
    calls = []
    expected = object()

    def permutation(scenario, replicate, seed, draws):
        calls.append((scenario, replicate, seed, draws))
        return expected

    monkeypatch.setattr(inference, "_permutation", permutation)
    outcomes = inference.run_inference(
        "permutation.bernoulli-size-power", "null-equal", 0, 20261104
    )
    assert calls == [("null-equal", 0, 20261104, 999)]
    assert outcomes is expected


def test_complete_multiplicity_public_procedure():
    outcomes = inference.run_inference(
        "multiplicity.equivalence-family", "global-null", 0, 20261104, draws=32
    )
    assert all(item["valid"] for item in outcomes.values())
    for item in outcomes.values():
        assert item["null_hypothesis"]
        assert len(item["pairwise_pvalues"]) == 3
        assert item["statistic"] == sum(item["decisions"])
        assert item["rejected"] == any(item["decisions"])
        assert "pvalue" not in item


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="Unknown inference scenario"):
        inference.run_inference("permutation.bernoulli-size-power", "typo", 0, 20261104)
