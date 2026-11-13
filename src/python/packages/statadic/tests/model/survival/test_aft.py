"""Independent distribution witnesses for right-censored AFT equations."""

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import gammainc, gammaln, log_ndtr
from statadic.model.survival import (
    LogLogisticAFT,
    LogNormalAFT,
    WeibullAFT,
    cumulative_hazard,
    log_density,
    log_likelihood_rows,
    log_survival,
    restricted_mean,
    survival_mean,
    survival_probability,
    survival_quantile,
)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("shape", [0.25, 0.6, 1.0, 1.5, 4.0, 10.0])
def test_loglogistic_against_scipy_and_rmst(shape: float) -> None:
    """Compare density, conditional likelihood, quantiles, means and quadrature."""
    family = LogLogisticAFT()
    x = jnp.ones((3, 1))
    theta = jnp.array([0.3, np.log(shape)])
    t = jnp.array([0.4, 1.5, 4.0])
    entry = jnp.array([0.0, 0.3, 1.0])
    event = jnp.array([1, 0, 1])
    reference = cast("rv_continuous_frozen", stats.fisk(c=shape, scale=np.exp(0.3)))
    np.testing.assert_allclose(log_density(theta, x, t, family), reference.logpdf(t))
    np.testing.assert_allclose(log_survival(theta, x, t, family), reference.logsf(t))
    expected = np.where(
        event, reference.logpdf(t), reference.logsf(t)
    ) - reference.logsf(entry)
    np.testing.assert_allclose(
        log_likelihood_rows(theta, x, t, event, entry, family), expected
    )
    np.testing.assert_allclose(survival_quantile(theta, x, 0.5, family), np.exp(0.3))
    mean = survival_mean(theta, x, family)
    if shape <= 1:
        assert np.all(np.isnan(mean))
    else:
        np.testing.assert_allclose(mean, reference.mean())
    for horizon in (0.1, 1.0, 5.0):
        expected_rmst = quad(reference.sf, 0, horizon)[0]
        actual = jax.jit(restricted_mean)(theta, x, horizon, family)
        np.testing.assert_allclose(actual, expected_rmst, rtol=2e-6)


@pytest.mark.unit
@pytest.mark.numerical
def test_survival_means_closed_form() -> None:
    """Weibull and lognormal means use their distribution formulas."""
    x = jnp.array([[1.0], [2.0]])
    theta = jnp.array([0.4, np.log(1.5)])
    np.testing.assert_allclose(
        survival_mean(theta, x, WeibullAFT()),
        np.exp(np.array([0.4, 0.8]) + gammaln(1 + 1 / 1.5)),
    )
    np.testing.assert_allclose(
        survival_mean(theta, x, LogNormalAFT()),
        np.exp(np.array([0.4, 0.8]) + 1.5**2 / 2),
    )


@pytest.mark.component
@pytest.mark.contract
def test_loglogistic_rmst_transformations_and_unavailable_tails() -> None:
    """The bounded quadrature supports transforms and masks unresolved tails."""
    family = LogLogisticAFT()
    x = jnp.array([1.0])
    theta = jnp.array([0.1, np.log(0.6)])
    horizons = jnp.array([0.0, 0.4, 2.0])

    def at_horizon(h: jax.Array) -> jax.Array:
        return restricted_mean(theta, x, h, family)

    def at_parameters(p: jax.Array) -> jax.Array:
        return restricted_mean(p, x, 2.0, family)

    values = jax.jit(jax.vmap(at_horizon))(horizons)
    assert values[0] == 0
    assert np.all(np.isfinite(values))
    gradient = jax.grad(at_parameters)(theta)
    assert np.all(np.isfinite(gradient))
    assert np.isnan(survival_mean(jnp.array([1000.0, np.log(2.0)]), x, family))
    assert np.isnan(restricted_mean(theta, x, np.inf, family))


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.reference
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_loglogistic_rmst_tiny_shape_cannot_false_converge_at_zero(dtype: str) -> None:
    """For tiny positive shapes S(t) is near one half on this unit interval."""
    with jax.enable_x64(new_val=True):
        theta = jnp.asarray([0.0, -20.0], dtype=dtype)
        x = jnp.ones(1, dtype=dtype)
        horizon = jnp.asarray(1.0, dtype=dtype)
        family = LogLogisticAFT()
        expected = quad(stats.fisk(c=np.exp(-20.0)).sf, 0, 1)[0]
        eager = restricted_mean(theta, x, horizon, family)
        compiled = jax.jit(restricted_mean)(theta, x, horizon, family)

        def evaluate(h: jax.Array) -> jax.Array:
            return restricted_mean(theta, x, h, family)

        mapped = jax.jit(jax.vmap(evaluate))(horizon[None])
        for actual in (eager, compiled, mapped[0]):
            assert actual.dtype == jnp.dtype(dtype)
            assert float(actual) >= 0.5 - 8 * np.finfo(dtype).eps
            np.testing.assert_allclose(
                actual, expected, rtol=8 * np.sqrt(np.finfo(dtype).eps)
            )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("shape", [0.6, 10.0])
def test_loglogistic_float32_quadrature(shape: float) -> None:
    """The quadrature follows caller precision in low and high shape cases."""
    theta = jnp.array([0.3, np.log(shape)], dtype=jnp.float32)
    x = jnp.array([1.0], dtype=jnp.float32)
    actual = restricted_mean(theta, x, jnp.float32(5), LogLogisticAFT())
    reference = quad(stats.fisk(c=shape, scale=np.exp(0.3)).sf, 0, 5)[0]
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, reference, rtol=5e-4)


@pytest.mark.unit
@pytest.mark.reference
def test_loglogistic_tail_log_density_remains_finite() -> None:
    """Evaluate very long durations without exponentiating their powered ratio."""
    theta = jnp.array([0.0, np.log(2.0)])
    time = jnp.array(1e200)
    density = log_density(theta, jnp.array([1.0]), time, LogLogisticAFT())
    reference = cast("rv_continuous_frozen", stats.fisk(c=2.0))
    np.testing.assert_allclose(density, reference.logpdf(1e200))


@pytest.mark.unit
@pytest.mark.numerical
def test_loglogistic_mean_near_existence_boundary() -> None:
    """Avoid cancellation when shape is just greater than one."""
    log_shape = 1e-10
    result = survival_mean(
        jnp.array([0.0, log_shape]), jnp.array([1.0]), LogLogisticAFT()
    )
    inverse = np.exp(-log_shape)
    expected = inverse / (-np.expm1(-log_shape) * np.sinc(-np.expm1(-log_shape)))
    np.testing.assert_allclose(result, expected, rtol=1e-8)


if TYPE_CHECKING:
    from jax import Array
    from scipy.stats._distn_infrastructure import rv_continuous_frozen


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("family", [WeibullAFT(), LogNormalAFT()])
def test_density_survival_entry_and_quantiles_match_scipy(
    family: WeibullAFT | LogNormalAFT,
) -> None:
    """Match separate SciPy laws with mixed events and left truncation."""
    x = np.array([[1, -0.4], [1, 0.5], [1, 1.1]])
    theta = np.array([0.3, -0.2, np.log(0.8)])
    t = np.array([0.7, 2.2, 4.5])
    a = np.array([0, 0.2, 1.3])
    d = np.array([1, 0, 1])
    eta = x @ theta[:-1]
    distribution = cast(
        "rv_continuous_frozen",
        stats.weibull_min(c=np.exp(theta[-1]), scale=np.exp(eta))
        if isinstance(family, WeibullAFT)
        else stats.lognorm(s=np.exp(theta[-1]), scale=np.exp(eta)),
    )
    np.testing.assert_allclose(
        log_density(theta, x, t, family), distribution.logpdf(t), rtol=3e-6
    )
    np.testing.assert_allclose(
        log_survival(theta, x, t, family), distribution.logsf(t), rtol=3e-6
    )
    expected = np.where(
        d, distribution.logpdf(t), distribution.logsf(t)
    ) - distribution.logsf(a)
    np.testing.assert_allclose(
        log_likelihood_rows(theta, x, t, d, a, family), expected, rtol=3e-6
    )

    def scipy_total(parameters: np.ndarray) -> float:
        location = x @ parameters[:-1]
        reference = cast(
            "rv_continuous_frozen",
            stats.weibull_min(c=np.exp(parameters[-1]), scale=np.exp(location))
            if isinstance(family, WeibullAFT)
            else stats.lognorm(s=np.exp(parameters[-1]), scale=np.exp(location)),
        )
        return float(
            np.sum(
                np.where(d, reference.logpdf(t), reference.logsf(t))
                - reference.logsf(a)
            )
        )

    step = 1e-5
    basis = np.eye(theta.size) * step
    numerical_score = np.array(
        [(scipy_total(theta + v) - scipy_total(theta - v)) / (2 * step) for v in basis]
    )

    def evaluate_total(parameters: jax.Array) -> "Array":
        return jnp.sum(log_likelihood_rows(parameters, x, t, d, a, family))

    score = jax.grad(evaluate_total)(jnp.asarray(theta))
    np.testing.assert_allclose(score, numerical_score, rtol=3e-4, atol=2e-5)
    np.testing.assert_allclose(
        survival_probability(theta, x, t, family, entry=a),
        distribution.sf(t) / distribution.sf(a),
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        cumulative_hazard(theta, x, t, family, entry=a),
        -np.log(distribution.sf(t) / distribution.sf(a)),
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        survival_quantile(theta, x, 0.3, family), distribution.ppf(0.3), rtol=3e-6
    )
    assert np.isnan(np.asarray(survival_quantile(theta, x, 1, family))).all()
    assert np.isnan(
        np.asarray(survival_probability(theta, x, 0.1, family, entry=0.2))
    ).all()


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("family", [WeibullAFT(), LogNormalAFT()])
def test_jit_gradient_zero_entry_and_restricted_mean(
    family: WeibullAFT | LogNormalAFT,
) -> None:
    """Preserve derivatives and survival behavior through JAX transformations."""
    theta = jnp.array([0.3, -0.2, -0.3])
    x = jnp.array([[1.0, 0.3], [1.0, -0.1]])
    durations = jnp.array([1.1, 2.1])
    event = jnp.array([1, 0])
    entry = jnp.array([0.0, 0.2])

    def f(p: jax.Array) -> jax.Array:
        return jnp.sum(log_likelihood_rows(p, x, durations, event, entry, family))

    assert jnp.all(jnp.isfinite(jax.jit(jax.grad(f))(theta)))

    def evaluate_survival_probability(t: jax.Array) -> "Array":
        return survival_probability(theta, x, t, family)

    curves = jax.jit(jax.vmap(evaluate_survival_probability))(
        jnp.array([0.0, 1.0, 2.0])
    )
    assert bool(jnp.all(jnp.diff(curves, axis=0) <= 0))
    actual = restricted_mean(theta, x, 2.0, family)
    for i in range(2):
        eta = float(x[i] @ theta[:-1])
        distribution = cast(
            "rv_continuous_frozen",
            stats.weibull_min(c=np.exp(float(theta[-1])), scale=np.exp(eta))
            if isinstance(family, WeibullAFT)
            else stats.lognorm(s=np.exp(float(theta[-1])), scale=np.exp(eta)),
        )
        expected = quad(distribution.sf, 0, 2)[0]
        np.testing.assert_allclose(actual[i], expected, rtol=2e-4, atol=2e-5)

    def evaluate_total(p: jax.Array) -> "Array":
        return jnp.sum(restricted_mean(p, x, 2.0, family))

    assert jnp.all(jnp.isfinite(jax.grad(evaluate_total)(theta)))


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("family", [WeibullAFT(), LogNormalAFT()])
@pytest.mark.parametrize(
    ("log_ancillary", "horizon"), [(-3.0, 1e-6), (-3.0, 1e6), (0.0, 1.0), (2.0, 1e6)]
)
def test_restricted_mean_tails_and_gradient(
    family: WeibullAFT | LogNormalAFT, log_ancillary: float, horizon: float
) -> None:
    """Compare closed forms and derivatives across small shapes and wide horizons."""
    x = jnp.asarray([[1.0]])
    theta = jnp.asarray([0.0, log_ancillary])
    value = restricted_mean(theta, x, horizon, family)[0]
    if isinstance(family, WeibullAFT):
        inverse_shape = np.exp(-log_ancillary)
        expected = np.exp(
            gammaln(1 + inverse_shape)
            + np.log(gammainc(inverse_shape, horizon ** np.exp(log_ancillary)))
        )
    else:
        sigma = np.exp(log_ancillary)
        z = np.log(horizon) / sigma
        expected = np.exp(
            np.logaddexp(
                sigma * sigma / 2 + log_ndtr(z - sigma),
                np.log(horizon) + log_ndtr(-z),
            )
        )
    np.testing.assert_allclose(value, expected, rtol=3e-4, atol=1e-9)

    def evaluate_total(p: jax.Array) -> "Array":
        return restricted_mean(p, x, horizon, family).sum()

    derivative = jax.grad(evaluate_total)(theta)
    assert bool(jnp.all(jnp.isfinite(derivative)))
