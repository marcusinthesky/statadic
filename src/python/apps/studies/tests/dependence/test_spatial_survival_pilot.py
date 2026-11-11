"""Joint survival pilot: independent DGP and retained numerical evidence."""

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.survival import (
    SpatialSurvivalProblem,
    prepare_spatial_survival_period,
    spatial_survival_log_likelihood,
)
from statadic.operators.spatial import ObservedSpatialPeriod, prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign
from statadic_studies.adapters.dependence.spatial_survival import (
    run_spatial_survival_pilot,
    spatial_design,
)
from scipy.stats import multivariate_normal


def test_uncensored_joint_loglikelihood_matches_independent_normal_density():
    w, x = spatial_design()
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(w), tuple(range(8)), tuple(range(8))
    )
    times = np.exp(np.linspace(-0.5, 0.5, 8))
    period = prepare_spatial_survival_period(
        graph,
        x,
        times,
        np.ones(8),
        integration=IntegrationDesign.from_key(
            jax.random.key(8), replicates=2, draws=16
        ),
    )
    theta = np.array([0.1, 0.3, 0.25, np.log(0.6)])
    inverse = np.linalg.inv(np.eye(len(w)) - 0.25 * w)
    mean = inverse @ (x @ theta[:2])
    covariance = 0.6**2 * inverse @ inverse.T
    logs = np.log(np.asarray(period.duration))
    expected = multivariate_normal.logpdf(logs, mean, covariance) - logs.sum()
    actual = spatial_survival_log_likelihood(
        jnp.asarray(theta), SpatialSurvivalProblem((period,))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_native_pilot_retains_every_attempt_and_failure():
    study, scenario = "survival.spatial-joint-pilot", "rho0.25"
    result = run_spatial_survival_pilot(study, scenario, 0, 42, draws=16)
    assert set(result) == {"fixed-draw-spatial-ML", "refined-spatial-ML"}
    fixed, refined = result["fixed-draw-spatial-ML"], result["refined-spatial-ML"]
    assert fixed["attempts"] == refined["attempts"]
    assert len(fixed["attempts"]) in (1, 2)
    assert fixed["attempts"][0]["draws"] == 16
    assert all(0 <= count <= 3 for count in fixed["censored_per_period"])
    if len(fixed["attempts"]) == 2:
        assert fixed["attempts"][1]["draws"] == 32
    for record in (fixed, refined):
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert record["truth"] == 0.3
    assert refined["valid"] == (
        refined["refinement_stable"]
        and refined["attempts"][-1]["fit_valid"]
        and refined["attempts"][-1]["integration_valid"]
    )
