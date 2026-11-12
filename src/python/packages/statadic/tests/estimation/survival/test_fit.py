"""Regression and failure evidence for independent AFT fits."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.optimize import minimize
from scipy.stats import fisk, lognorm, weibull_min
from statadic.estimation.survival import (
    ExistenceStatus,
    SurvivalML,
    SurvivalProblem,
)
from statadic.model.survival import (
    LogLogisticAFT,
    LogNormalAFT,
    WeibullAFT,
    survival_mean,
)

from ..._budgets import examples

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_continuous_frozen
    from statadic.estimation.survival import SurvivalFit


@pytest.mark.component
@pytest.mark.reference
def test_loglogistic_fit_with_delayed_entry_and_fixed_effect_design() -> None:
    """Fit censored spells with entry and an explicit group-effect column."""
    rng = np.random.default_rng(521)
    x = np.column_stack([np.ones(140), np.arange(140) % 2])
    duration = np.exp(x @ np.array([0.2, 0.3]) + rng.logistic(size=140) / 0.7)
    cutoff = 5.0
    observed = np.minimum(duration, cutoff)
    event = duration <= cutoff
    entry = observed * 0.1
    problem = SurvivalProblem(x, observed, event, entry, ("intercept", "group"))
    result = SurvivalML(LogLogisticAFT()).estimate(problem)

    def objective(p: np.ndarray) -> float:
        distribution = cast(
            "rv_continuous_frozen", fisk(c=np.exp(p[-1]), scale=np.exp(x @ p[:-1]))
        )
        return -float(
            np.sum(
                np.where(
                    event, distribution.logpdf(observed), distribution.logsf(observed)
                )
                - distribution.logsf(entry)
            )
        )

    reference = minimize(objective, np.zeros(3), method="BFGS")
    assert bool(result.valid)
    assert result.parameter_names is not None
    assert result.parameter_names[-1] == "log_shape"
    np.testing.assert_allclose(result.parameters, reference.x, atol=3e-4)
    assert float(result.parameters[-1]) <= 0
    assert np.all(np.isnan(survival_mean(result.parameters, x, result.family)))


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("family", [WeibullAFT(), LogNormalAFT()])
@examples(6)
@example(seed=912)
@given(seed=st.integers(0, 2**32 - 1))
def test_fit_matches_independent_scipy_likelihood(
    seed: int,
    family: WeibullAFT | LogNormalAFT,
) -> None:
    """Match an independent SciPy objective and fitted solution."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(80), rng.normal(size=80)])
    location = x @ np.array([0.2, -0.3])
    errors = (
        rng.weibull(1.5, 80)
        if isinstance(family, WeibullAFT)
        else np.exp(rng.normal(0, 0.6, 80))
    )
    duration = np.exp(location) * errors
    cutoff = np.quantile(duration, 0.7)
    event = (duration <= cutoff).astype(int)
    observed = np.minimum(duration, cutoff)
    entry = 0.2 * observed
    problem = SurvivalProblem(x, observed, event, entry, ("intercept", "x"))
    result = SurvivalML(family=family).estimate(problem)

    def objective(p: np.ndarray) -> float:
        mu = x @ p[:-1]
        dist = cast(
            "rv_continuous_frozen",
            weibull_min(c=np.exp(p[-1]), scale=np.exp(mu))
            if isinstance(family, WeibullAFT)
            else lognorm(s=np.exp(p[-1]), scale=np.exp(mu)),
        )
        return float(
            -np.sum(
                np.where(event, dist.logpdf(observed), dist.logsf(observed))
                - dist.logsf(entry)
            )
        )

    expected = minimize(objective, np.zeros(3), method="BFGS", tol=1e-8)
    assert bool(result.valid), (result.solver, result.existence_status)
    np.testing.assert_allclose(result.parameters, expected.x, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(result.log_likelihood, -expected.fun, rtol=3e-5)
    assert np.linalg.eigvalsh(np.asarray(result.observed_information)).min() > 0
    assert result.parameter_names is not None
    assert result.parameter_names[-1] == (
        "log_shape" if isinstance(family, WeibullAFT) else "log_scale"
    )


@pytest.mark.component
@pytest.mark.contract
def test_all_censored_exact_fit_and_rank_failure_are_unavailable() -> None:
    """Keep invalid or nonidentified samples from yielding usable fits."""
    x = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    t = jnp.exp(x @ jnp.array([0.1, 0.2]))
    estimator = SurvivalML(family=LogNormalAFT(), max_steps=16)
    all_censored = estimator.estimate(SurvivalProblem(x, t, jnp.zeros(3)))
    assert not bool(all_censored.valid)
    assert int(all_censored.existence_status) == ExistenceStatus.NO_EVENTS
    exact = estimator.estimate(SurvivalProblem(x, t, jnp.ones(3)))
    assert not bool(exact.valid)
    assert int(exact.existence_status) == ExistenceStatus.EXACT_FIT_COLLAPSE
    rank_fail = estimator.estimate(
        SurvivalProblem(jnp.ones((3, 2)), jnp.array([1.0, 2.0, 4.0]), jnp.ones(3))
    )
    assert not bool(rank_fail.identified)
    invalid = estimator.estimate(
        SurvivalProblem(x, jnp.array([1.0, -1.0, 2.0]), jnp.ones(3))
    )
    assert not bool(invalid.input_valid)
    assert not bool(invalid.valid)


@pytest.mark.component
@pytest.mark.contract
def test_loglogistic_inherits_existence_diagnostics() -> None:
    """A new family retains no-event and exact-collapse diagnostics."""
    x = jnp.ones((3, 1))
    time = jnp.ones(3)
    estimator = SurvivalML(LogLogisticAFT(), max_steps=16)
    censored = estimator.estimate(SurvivalProblem(x, time, jnp.zeros(3)))
    exact = estimator.estimate(SurvivalProblem(x, time, jnp.ones(3)))
    assert not bool(censored.valid)
    assert int(censored.existence_status) == ExistenceStatus.NO_EVENTS
    assert not bool(exact.valid)
    assert int(exact.existence_status) == ExistenceStatus.EXACT_FIT_COLLAPSE


@pytest.mark.component
@pytest.mark.contract
def test_jit_fit_supports_float32_and_retains_invalid_lane() -> None:
    """Preserve caller precision and per-lane failure in compiled batching."""
    x = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=jnp.float32)
    times = jnp.array([[1.0, 2.0, 3.0, 5.0], [1.0, -2.0, 3.0, 5.0]], dtype=jnp.float32)
    estimator = SurvivalML(family=LogNormalAFT(), max_steps=100)

    def one(t: jax.Array) -> SurvivalFit:
        return estimator.estimate(SurvivalProblem(x, t, jnp.ones(4)))

    result = jax.jit(jax.vmap(one))(times)
    assert result.parameters.dtype == jnp.float32
    assert not bool(result.input_valid[1])
    assert not bool(result.valid[1])
