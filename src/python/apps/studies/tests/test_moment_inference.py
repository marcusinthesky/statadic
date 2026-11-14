"""Analytic affine fitting, fixed moment quadratics and provenance witnesses."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import gel_test
from statadic.inference.gmm import hansen_j_from_fit
from statadic_studies.adapters.moments import affine_moments, run_moment_inference
from statadic_studies.definitions.moments import (
    MOMENT_INFERENCE_STUDIES,
    MOMENT_METHOD_METRICS,
    MOMENT_NULL_SCENARIOS,
)


@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_affine_gel_rejects_exact_identification_and_recovers_known_root(criterion):
    observations = 0.4 + np.random.default_rng(82).normal(size=(128, 1))
    with pytest.raises(ValueError, match="requires over-identification"):
        AffineGEL(criterion=criterion).estimate(
            AffineMomentProblem(-observations, np.ones((1, 1)))
        )
    # Uniform weights already satisfy every empirical moment exactly at .4.
    # Every carrier therefore attains its zero optimum at the known root.
    noise = np.random.default_rng(82).normal(size=(128, 3))
    noise -= noise.mean(axis=0)
    derivative = np.array([[1.0], [0.5], [-0.25]])
    fit = AffineGEL(criterion=criterion).estimate(
        AffineMomentProblem(noise - 0.4 * derivative[:, 0], derivative)
    )
    assert fit.converged and fit.rank_stable
    np.testing.assert_allclose(fit.theta, [0.4], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(fit.objective, 0.0, atol=3e-7)


@pytest.mark.parametrize("scheme", [PlainLongRun, MeanCenteredLongRun])
def test_exactly_identified_gmm_matches_sample_mean(scheme):
    observations = 0.4 + np.random.default_rng(85).normal(size=(128, 1))
    fit = FixedWeightGMM(scheme).estimate(
        AffineMomentProblem(-observations, np.ones((1, 1)), np.eye(1))
    )
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, observations.mean(axis=0), rtol=3e-5)
    tested = hansen_j_from_fit(fit, assume_correct_specification=True)
    assert not tested.overidentified and not tested.valid


@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_fixed_gel_lm_wald_match_numpy_uncentered_quadratic(criterion):
    moments = np.random.default_rng(83).normal(size=(128, 2)) + 0.1
    fit = FixedGEL(criterion=criterion).estimate(FixedMomentProblem(moments))
    expected = (
        len(moments)
        * moments.mean(axis=0)
        @ np.linalg.inv(moments.T @ moments / len(moments))
        @ moments.mean(axis=0)
    )
    for kind in ("lm", "wald"):
        tested = gel_test(fit, kind=kind, assume_iid=True)
        assert tested.converged and tested.reference_valid
        np.testing.assert_allclose(tested.stat, expected, rtol=3e-5)
        np.testing.assert_allclose(tested.pvalue, chi2.sf(expected, 2), rtol=3e-5)


@pytest.mark.parametrize("scheme", [PlainLongRun, MeanCenteredLongRun])
def test_fixed_gmm_weight_does_not_receive_efficient_J_provenance(scheme):
    moments, derivative = affine_moments(np.random.default_rng(84), "correct")
    fit = FixedWeightGMM(scheme).estimate(
        AffineMomentProblem(jnp.asarray(moments), jnp.asarray(derivative), jnp.eye(3))
    )
    expected = -(derivative.T @ moments.mean(axis=0)) / (derivative.T @ derivative)
    assert fit.valid
    np.testing.assert_allclose(fit.parameters[0], expected[0, 0], rtol=3e-5)
    tested = hansen_j_from_fit(fit, assume_correct_specification=True)
    assert not tested.valid and not tested.efficient_weight


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in MOMENT_INFERENCE_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_complete_moment_procedure_contracts(study, scenario):
    records = run_moment_inference(study, scenario, 0, 42)
    from statadic_studies.campaign import _validate_outcomes
    from statadic_studies.contracts import STUDIES

    _validate_outcomes(records, STUDIES[study].record(), scenario, None)
    assert tuple(records) == MOMENT_INFERENCE_STUDIES[study].methods
    assert set(MOMENT_METHOD_METRICS[study]) == set(records)
    assert set(MOMENT_NULL_SCENARIOS[study]) <= set(
        MOMENT_INFERENCE_STUDIES[study].scenarios
    )
    for record in records.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert all(
                np.isfinite(value)
                for value in record.values()
                if isinstance(value, float)
            )


@pytest.mark.parametrize(
    "study,producer",
    [
        ("gel.affine-profile", "AffineGEL"),
        ("gel.fixed-test-forms", "FixedGEL"),
        ("gmm.variability-and-J", "hansen_j"),
    ],
)
def test_infrastructure_runtime_error_remains_pending(monkeypatch, study, producer):
    import statadic_studies.adapters.moments as adapters

    def infrastructure_failure(*args, **kwargs):
        raise RuntimeError("simulated device initialization failed")

    monkeypatch.setattr(adapters, producer, infrastructure_failure)
    scenario = "normal-null" if study == "gel.fixed-test-forms" else "correct"
    with pytest.raises(RuntimeError, match="simulated device initialization failed"):
        adapters.run_moment_inference(study, scenario, 0, 42)
