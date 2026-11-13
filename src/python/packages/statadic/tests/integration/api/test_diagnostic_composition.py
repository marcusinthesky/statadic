"""Diagnostic recipes equal explicitly fitted auxiliary regression constructions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import RESET, BreuschGodfrey, BreuschPagan
from statadic.api.hypotheses import Hypothesis, LinearRestriction, TestOutcome
from statadic.api.inference import ModelBased
from statadic.api.regression import Regression

pytestmark = pytest.mark.integration


def _columns() -> dict[str, jax.Array]:
    x = jnp.linspace(-1.5, 2.0, 60)
    return {
        "x": x,
        "y": 0.5
        + 1.2 * x
        + 0.2 * x**2
        + jnp.sin(jnp.arange(60.0) * 1.7) * (1 + 0.3 * x**2),
    }


@pytest.mark.reference
def test_studentized_bp_matches_explicit_auxiliary_regression() -> None:
    """The variance-score nR² equals the fitted squared-residual auxiliary R²."""
    data = _columns()
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    squared = fit.core.residuals**2
    auxiliary = Regression.ols("squared", Intercept() + Linear("x")).estimate(
        {"squared": squared, "x": data["x"]}
    )
    expected = squared.size * (
        1 - auxiliary.core.sse / jnp.sum((squared - squared.mean()) ** 2)
    )
    outcome = fit.diagnose(BreuschPagan())["breusch_pagan"]
    assert isinstance(outcome, TestOutcome)
    assert outcome.available
    assert outcome.statistic is not None
    np.testing.assert_allclose(outcome.statistic, expected, rtol=2e-4)


@pytest.mark.reference
def test_bg_matches_explicit_zero_padded_auxiliary_regression() -> None:
    """All original observations contribute after adding two zero-filled lags."""
    data = _columns()
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    e = fit.core.residuals
    auxiliary = Regression.ols(
        "e", Intercept() + Linear("x") + Linear("lag1") + Linear("lag2")
    ).estimate(
        {
            "e": e,
            "x": data["x"],
            "lag1": jnp.concatenate((jnp.zeros(1, dtype=e.dtype), e[:-1])),
            "lag2": jnp.concatenate((jnp.zeros(2, dtype=e.dtype), e[:-2])),
        }
    )
    expected = e.size * (1 - auxiliary.core.sse / jnp.sum((e - e.mean()) ** 2))
    outcome = fit.diagnose(BreuschGodfrey(2, order_by=None))["breusch_godfrey"]
    assert isinstance(outcome, TestOutcome)
    assert outcome.available
    assert outcome.statistic is not None
    np.testing.assert_allclose(outcome.statistic, expected, rtol=2e-4)
    assert len(auxiliary.context.row_ids) == e.size


@pytest.mark.reference
@pytest.mark.parametrize("reference", ["chi2", "f"])
def test_reset_matches_full_rank_augmented_wald(reference: str) -> None:
    """Classical augmented-fit restrictions reproduce RESET only on a full-rank span."""
    data = _columns()
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    fitted = fit.core.fitted_values
    augmented = Regression.ols(
        "y", Intercept() + Linear("x") + Linear("power2") + Linear("power3")
    ).estimate(
        {
            **data,
            "power2": fitted**2,
            "power3": fitted**3,
        }
    )
    native_reference = "F" if reference == "f" else "chi2"
    outcome = fit.diagnose(RESET(reference=native_reference))["reset"]
    assert isinstance(outcome, TestOutcome)
    assert outcome.available
    hypothesis = Hypothesis(LinearRestriction.zero("power2", "power3"))
    inference = augmented.infer(covariance=ModelBased())
    expected = (
        inference.test(hypothesis, reference="f", degrees_of_freedom=augmented.df_resid)
        if reference == "f"
        else inference.test(hypothesis, reference="chi2")
    )
    assert expected.available
    assert outcome.statistic is not None
    assert expected.statistic is not None
    assert outcome.pvalue is not None
    assert expected.pvalue is not None
    tolerance = 4e-3 if not jax.config.x64_enabled else 2e-8
    np.testing.assert_allclose(outcome.statistic, expected.statistic, rtol=tolerance)
    np.testing.assert_allclose(
        outcome.pvalue, expected.pvalue, rtol=tolerance, atol=tolerance
    )
