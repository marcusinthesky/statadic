# ruff: noqa: D103, FBT003, PLR2004
"""Observed Hessian uncertainty retains all free survival parameters."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.base import FitContext
from statadic.api.data import Observations
from statadic.api.inference import HC, ModelBased
from statadic.api.survival.inference import survival_inference
from statadic.core.results import InferenceStatus
from statadic.estimation.survival.fit import SurvivalML, SurvivalProblem
from statadic.estimation.survival.spatial import (
    LaggedLogTime,
    SpatialSurvivalML,
    SpatialSurvivalProblem,
    prepare_spatial_survival_period,
)
from statadic.inference.parameters import coefficient_intervals
from statadic.inference.survival import observed_information
from statadic.model.survival import LogNormalAFT
from statadic.operators.spatial import ObservedSpatialPeriod, prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign

if TYPE_CHECKING:
    from statadic.estimation.survival.fit import SurvivalFit
    from statadic.inference.survival.information import SurvivalUncertainty


@pytest.fixture
def fitted() -> SurvivalFit:
    logs = jnp.asarray([-1.2, -0.6, -0.1, 0.3, 0.9, 1.4])
    problem = SurvivalProblem(jnp.ones((6, 1)), jnp.exp(logs), jnp.ones(6))
    return SurvivalML(family=LogNormalAFT()).estimate(problem)


@pytest.mark.integration
@pytest.mark.numerical
def test_complete_lognormal_intercept_information(fitted: SurvivalFit) -> None:
    uncertainty = observed_information(fitted)
    assert bool(uncertainty.valid)
    variance = np.var(np.log(np.asarray(fitted.problem.duration)))
    expected = np.diag([variance / 6, 1 / 12])
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected, rtol=2e-4, atol=2e-5
    )
    assert uncertainty.fit is fitted
    assert int(uncertainty.covariance.n_observations) == 6
    assert bool(coefficient_intervals(uncertainty, reference="normal").valid)


@pytest.mark.component
@pytest.mark.contract
def test_invalid_information_and_fit_are_separate_under_jit_vmap(
    fitted: SurvivalFit,
) -> None:
    matrices = jnp.stack(
        [jnp.eye(2), jnp.diag(jnp.asarray([1.0, -1.0])), jnp.zeros((2, 2))]
    )

    def evaluate_observed_information(info: jax.Array) -> SurvivalUncertainty:
        return observed_information(replace(fitted, observed_information=info))

    results = jax.jit(jax.vmap(evaluate_observed_information))(matrices)
    np.testing.assert_array_equal(results.valid, [True, False, False])
    assert np.isnan(np.asarray(results.covariance.values[1:])).all()
    invalid = observed_information(replace(fitted, valid=jnp.asarray(False)))
    assert int(invalid.status) == InferenceStatus.INVALID_FIT


@pytest.mark.component
@pytest.mark.contract
def test_host_rejects_unsupported_covariance(fitted: SurvivalFit) -> None:
    context = FitContext(Observations({"t": jnp.ones(1)}, row_ids=(0,)), "test", "t")
    for specification in (HC(), ModelBased(finite_sample_correction=2.0)):
        with pytest.raises(ValueError, match="uncorrected ModelBased"):
            survival_inference(fitted, context, specification)


@pytest.mark.component
@pytest.mark.reference
def test_spatial_fixed_rho_uses_actual_free_parameters(fitted: SurvivalFit) -> None:
    w = np.roll(np.eye(6), 1, axis=1)
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(w), tuple(range(6)), tuple(range(6))
    )
    period = prepare_spatial_survival_period(
        graph,
        jnp.ones((6, 1)),
        fitted.problem.duration,
        jnp.ones(6),
        integration=IntegrationDesign.from_key(
            jax.random.key(4), replicates=2, draws=2
        ),
    )
    problem = SpatialSurvivalProblem((period,))
    fit = SpatialSurvivalML(rho_fixed=0.0).estimate(
        replace(problem, process=LaggedLogTime())
    )
    uncertainty = observed_information(fit)
    assert fit.parameter_names == ("beta_0", "log_scale")
    assert bool(uncertainty.valid)
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        observed_information(fitted).covariance.values,
        rtol=2e-4,
        atol=2e-5,
    )
    unavailable = observed_information(
        replace(fit, information_available=jnp.asarray(False))
    )
    assert not bool(unavailable.valid)
