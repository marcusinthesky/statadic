"""Joint observed and independent-row likelihood covariance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.likelihood import (
    EstimatedDispersion,
    GammaMLE,
    LikelihoodProblem,
)
from statadic.inference.likelihood import likelihood_covariance


@pytest.mark.component
@pytest.mark.reference
def test_joint_covariance_and_pytree_jit() -> None:
    """Use the full physical-parameter Hessian and every joint score column."""
    y = jnp.asarray([0.5, 1.0, 2.0, 3.0, 1.2, 0.8, 0.7, 2.2, 3.5, 4.0])
    fit = GammaMLE(EstimatedDispersion(), tolerance=1e-7).estimate(
        LikelihoodProblem(jnp.ones((10, 1)), y, parameter_names=("intercept",))
    )
    assert bool(fit.valid)
    bread = np.linalg.inv(np.asarray(fit.hessian))
    scores = np.asarray(fit.scores)
    model = likelihood_covariance(fit)
    robust = likelihood_covariance(fit, method="HC0")
    corrected = likelihood_covariance(fit, method="HC1")
    np.testing.assert_allclose(model.covariance.values, bread, rtol=1e-10)
    np.testing.assert_allclose(
        robust.covariance.values, bread @ scores.T @ scores @ bread, rtol=1e-10
    )
    np.testing.assert_allclose(
        corrected.covariance.values, robust.covariance.values * 10 / 8, rtol=1e-10
    )
    compiled = jax.jit(likelihood_covariance)(fit)
    np.testing.assert_allclose(compiled.covariance.values, model.covariance.values)
    assert compiled.method == "model"
