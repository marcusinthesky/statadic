# jaxtyping shape strings are runtime contracts.
"""Sharpe inference kernels, declared trial deflation and point-ratio parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.stats import norm
from statadic.estimation.sharpe import (
    sample_skew_kurtosis,
    sharpe_ratio,
    sharpe_ratio_axis,
)
from statadic.inference.resampling.block_length import stationary_bootstrap_ci
from statadic.inference.sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    multiple_sharpe_test_kernel,
    probabilistic_sharpe_ratio,
    sharpe_difference_test,
    sharpe_difference_test_kernel,
    sharpe_ratio_ci,
)
from statadic.inference.sharpe.difference import (
    _sharpe_difference_kernel,
)
from statadic.inference.sharpe.multiple import (
    _multiple_sharpe_calibration,
    _multiple_sharpe_kernel,
    _paired_sharpe_diffs,
)

from .._budgets import examples

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from statadic.inference.sharpe.difference import (
        SharpeDifferenceKernelResult,
    )
    from statadic.inference.sharpe.multiple import (
        MultipleSharpeKernelResult,
    )

_N = 500
_RAMP = 5.0


def spread_returns(size: int, *, noise: float = 1.0) -> SearchStrategy[np.ndarray]:
    """Draw return series with positive dispersion, by construction.

    A ramp wider than the generated noise keeps the sample standard deviation away
    from zero without filtering; a constant series has a declared zero Sharpe
    ratio and is covered by its own tests.

    Args:
        size: Series length, fixed per test.
        noise: Half-width of the generated noise.

    Returns:
        A strategy for float64 series.

    """
    ramp = np.linspace(-_RAMP, _RAMP, size)
    return hnp.arrays(np.float64, size, elements=st.floats(-noise, noise)).map(
        lambda values: values + ramp
    )


@pytest.mark.component
@pytest.mark.property
def test_keyed_sharpe_kernels_are_deterministic_and_transformable() -> None:
    """Explicit-key kernels return fixed-shape array pytrees under ``vmap``."""
    time = jnp.linspace(0.0, 6.0, 64)
    benchmark = 0.01 * jnp.sin(time) + 0.002 * jnp.cos(2.0 * time)
    strategies = jnp.stack(
        (
            benchmark + 0.001 + 0.003 * jnp.sin(1.7 * time),
            benchmark - 0.001 + 0.002 * jnp.cos(1.3 * time),
        ),
        axis=1,
    )
    keys = jax.random.split(jax.random.PRNGKey(19), 2)

    def evaluate_sharpe_difference_test_kernel(
        key: jax.Array,
    ) -> SharpeDifferenceKernelResult:
        return sharpe_difference_test_kernel(
            key, strategies[:, 0], benchmark, n_boot=32
        )

    difference = jax.vmap(evaluate_sharpe_difference_test_kernel)(keys)

    def evaluate_multiple_sharpe_test_kernel(
        key: jax.Array,
    ) -> MultipleSharpeKernelResult:
        return multiple_sharpe_test_kernel(key, strategies, benchmark, n_boot=32)

    multiple = jax.vmap(evaluate_multiple_sharpe_test_kernel)(keys)

    assert difference.statistic.shape == (2,)
    assert difference.pvalue.shape == (2,)
    assert difference.block_length.shape == (2,)
    assert multiple.sharpe_diff.shape == (2, 2)
    assert multiple.rw_adjusted_pvalues.shape == (2, 2)
    assert multiple.reality_check_pvalue.shape == (2,)
    repeated = sharpe_difference_test_kernel(
        keys[0], strategies[:, 0], benchmark, n_boot=32
    )
    np.testing.assert_allclose(
        difference.statistic[0], repeated.statistic, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_array_equal(difference.pvalue[0], repeated.pvalue)


def _returns(mean: float, seed: int = 0) -> np.ndarray:
    """Draw a normal return series with a chosen per-observation mean.

    Args:
        mean: Per-observation mean; the unit-variance draw makes it the Sharpe.
        seed: Seed for the NumPy generator.

    Returns:
        A ``(500,)`` return series.

    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=1.0, size=_N)


@pytest.mark.component
@pytest.mark.numerical
@examples(40)
@example(returns=_returns(mean=0.1))
@given(returns=spread_returns(_N))
def test_sharpe_ratio_is_per_observation_and_unannualised(
    returns: np.ndarray,
) -> None:
    """``mean / std`` of the series, with no sqrt(252) anywhere in it.

    Args:
        returns: Generated return series with positive dispersion.

    """
    expected = float(np.mean(returns) / np.std(returns, ddof=1))
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-9)


@pytest.mark.component
@pytest.mark.property
@examples(100)
@example(low_estimate=0.02, gap=0.18)
@given(low_estimate=st.floats(-0.3, 0.2), gap=st.floats(0.01, 0.1))
def test_probabilistic_sharpe_rises_with_the_estimate(
    low_estimate: float, gap: float
) -> None:
    """PSR is a probability, and it is monotone in ``SR_hat`` at fixed ``n``.

    Args:
        low_estimate: Generated lower estimate.
        gap: Positive distance to the higher estimate.

    """
    common = {"n_obs": _N, "skew": 0.0, "kurtosis": 3.0}
    low = probabilistic_sharpe_ratio(sr_hat=low_estimate, **common)
    high = probabilistic_sharpe_ratio(sr_hat=low_estimate + gap, **common)
    assert 0.0 <= low < high <= 1.0


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(n_obs=1, sr_hat=0.1)
@given(n_obs=st.integers(max_value=1), sr_hat=st.floats(-2, 2))
def test_probabilistic_sharpe_needs_two_observations(n_obs: int, sr_hat: float) -> None:
    """``sqrt(n - 1)`` is undefined below two: it raises rather than give NaN.

    Args:
        n_obs: Any observation count below two.
        sr_hat: Generated estimate.

    """
    with pytest.raises(ValueError, match="n_obs must be >= 2"):
        probabilistic_sharpe_ratio(sr_hat=sr_hat, n_obs=n_obs, skew=0.0, kurtosis=3.0)


@pytest.mark.component
@pytest.mark.numerical
@examples(60)
@example(n_trials=50, variance=0.04)
@given(n_trials=st.integers(1, 10_000), variance=st.floats(0.0, 4.0))
def test_expected_maximum_sharpe_is_zero_without_a_search(
    n_trials: int, variance: float
) -> None:
    """One trial, or no cross-trial dispersion, means nothing to deflate against.

    Args:
        n_trials: Generated trial count, used with zero variance.
        variance: Generated cross-trial variance, used with one trial.

    """
    assert expected_maximum_sharpe_ratio(1, variance) == pytest.approx(0.0)
    assert expected_maximum_sharpe_ratio(n_trials, 0.0) == pytest.approx(0.0)


@pytest.mark.component
@pytest.mark.property
@examples(60)
@example(variance=0.04)
@given(variance=st.floats(1e-4, 4.0))
def test_expected_maximum_sharpe_grows_with_the_number_of_trials(
    variance: float,
) -> None:
    """The best of many worthless strategies looks better the more you try.

    Args:
        variance: Generated positive cross-trial variance.

    """
    benchmarks = [expected_maximum_sharpe_ratio(n, variance) for n in (2, 10, 100)]
    assert benchmarks == sorted(benchmarks)
    assert benchmarks[0] > 0.0


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(n_trials=0, variance=0.04)
@given(n_trials=st.integers(max_value=0), variance=st.floats(0.0, 4.0))
def test_expected_maximum_sharpe_rejects_a_non_positive_trial_count(
    n_trials: int, variance: float
) -> None:
    """Zero trials is not "no deflation", it is a caller error.

    Args:
        n_trials: Any non-positive trial count.
        variance: Generated cross-trial variance.

    """
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        expected_maximum_sharpe_ratio(n_trials, variance)


@pytest.mark.component
@pytest.mark.property
def test_deflated_sharpe_falls_as_the_search_widens() -> None:
    """The deflated verdict must be strictly below the undeflated one.

    This is the whole point of the statistic: a strategy that clears PSR against
    zero may not clear it against what the best of ``n_trials`` zero-skill
    strategies would post by luck.
    """
    returns = _returns(mean=0.08)
    narrow = deflated_sharpe_ratio(returns, n_trials=2, sr_variance_across_trials=0.01)
    wide = deflated_sharpe_ratio(returns, n_trials=200, sr_variance_across_trials=0.01)

    assert isinstance(narrow, DeflatedSharpeResult)
    assert narrow.n_obs == _N
    assert wide.sr0 > narrow.sr0 > 0.0
    assert wide.deflated_sharpe < narrow.deflated_sharpe
    assert narrow.deflated_sharpe < narrow.psr_zero


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(returns=_returns(mean=0.08))
@given(returns=spread_returns(_N))
def test_a_single_trial_deflates_to_the_undeflated_verdict(
    returns: np.ndarray,
) -> None:
    """``n_trials == 1`` means no search, so ``DSR == PSR(benchmark)``.

    Args:
        returns: Generated return series with positive dispersion.

    """
    result = deflated_sharpe_ratio(returns, n_trials=1)
    assert result.sr0 == pytest.approx(0.0)
    assert result.deflated_sharpe == pytest.approx(result.psr_zero)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(returns=_returns(mean=0.08), n_trials=25)
@given(returns=spread_returns(_N), n_trials=st.integers(2, 10_000))
def test_deflating_without_trial_dispersion_is_refused(
    returns: np.ndarray, n_trials: int
) -> None:
    """There is no way to deflate ``n_trials > 1`` without the cross-trial variance.

    Args:
        returns: Generated return series with positive dispersion.
        n_trials: Any trial count above one.

    """
    with pytest.raises(ValueError, match="sr_variance_across_trials"):
        deflated_sharpe_ratio(returns, n_trials=n_trials)


def _numpy_sharpe(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Independently apply the declared zero-dispersion and sample-ddof rule."""
    deviation = np.std(values, axis=axis, ddof=1)
    return np.divide(
        np.mean(values, axis=axis),
        deviation,
        out=np.zeros_like(deviation),
        where=deviation > 0.0,
    )


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(
    returns=np.asarray([-3.0, -1.0, 0.0, 0.5, 4.0, 7.0, 2.0, -0.5]),
    trial_count=20,
    variance=0.04,
)
@given(
    returns=spread_returns(8, noise=3.0),
    trial_count=st.integers(2, 200),
    variance=st.floats(0.001, 1.0),
)
def test_sample_moments_and_deflated_sharpe_match_independent_formulas(
    returns: np.ndarray, trial_count: int, variance: float
) -> None:
    """Non-excess kurtosis, trial variance and expected-max benchmark stay aligned.

    Args:
        returns: Generated return series with positive dispersion.
        trial_count: Generated number of trials.
        variance: Generated cross-trial Sharpe variance.

    """
    centered = returns - returns.mean()
    second = np.mean(centered**2)
    skew = float(np.mean(centered**3) / second**1.5)
    kurtosis = float(np.mean(centered**4) / second**2)
    ratio = float(_numpy_sharpe(returns))
    gamma = 0.5772156649015329
    benchmark = np.sqrt(variance) * (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + gamma * norm.ppf(1.0 - 1.0 / (trial_count * np.e))
    )
    denominator = np.sqrt(1.0 - skew * ratio + (kurtosis - 1.0) * ratio**2 / 4.0)
    expected = norm.cdf((ratio - benchmark) * np.sqrt(len(returns) - 1) / denominator)

    np.testing.assert_allclose(
        sample_skew_kurtosis(returns),
        [skew, kurtosis],
        rtol=1e-12,
        # Skewness of a symmetric draw is exactly 0 in theory; a relative bound on
        # a zero target is meaningless, so allow float round-off of that size.
        atol=1e-12,
    )
    result = deflated_sharpe_ratio(returns, trial_count, variance)
    assert result.sr_hat == pytest.approx(ratio, rel=1e-12)
    assert result.sr0 == pytest.approx(benchmark, rel=1e-12)
    assert result.deflated_sharpe == pytest.approx(expected, rel=1e-12)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@examples(100)
@example(z_score=-12.0)
@example(z_score=-8.0)
@example(z_score=0.0)
@example(z_score=8.0)
@example(z_score=12.0)
@given(z_score=st.floats(-12.0, 12.0))
def test_probabilistic_sharpe_extreme_tails_match_independent_normal_cdf(
    z_score: float,
) -> None:
    """Separate tail arithmetic from finite-sample operating characteristics.

    Args:
        z_score: Generated standardised distance, including both extreme tails.

    """
    actual = probabilistic_sharpe_ratio(
        0.0, 101, skew=0.0, kurtosis=3.0, benchmark_sr=-z_score / 10.0
    )
    assert actual == pytest.approx(float(norm.cdf(z_score)), rel=1e-12, abs=0.0)


@pytest.mark.component
@pytest.mark.property
@examples(12)
@given(scale=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False))
def test_positive_common_return_scale_preserves_ratio_and_sample_moments(
    scale: float,
) -> None:
    """Returns and their benchmark must use consistent units for scale invariance."""
    returns = np.asarray([-2.0, -0.5, 0.2, 1.5, 4.0])
    assert sharpe_ratio(scale * returns) == pytest.approx(
        sharpe_ratio(returns), rel=1e-12
    )
    np.testing.assert_allclose(
        sample_skew_kurtosis(scale * returns), sample_skew_kurtosis(returns), rtol=1e-12
    )


@pytest.mark.component
@pytest.mark.reference
def test_paired_sharpe_and_stepdown_full_outputs_match_fixed_draws() -> None:
    """The same rows resample every return series; suffix maxima enforce stepdown."""
    benchmark = np.asarray([-1.0, 0.25, 1.5, -0.5, 2.0, 0.5])
    strategies = np.column_stack(
        (benchmark + 0.5, benchmark[::-1] + 0.25, 2 * benchmark - 0.25)
    )
    indices = np.asarray(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 1, 3, 4, 2, 0],
            [5, 4, 3, 2, 1, 0],
            [2, 3, 2, 0, 5, 1],
            [4, 4, 1, 5, 0, 3],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    observed = _numpy_sharpe(strategies) - _numpy_sharpe(benchmark)
    draws = np.stack(
        [
            _numpy_sharpe(strategies[rows]) - _numpy_sharpe(benchmark[rows])
            for rows in indices
        ]
    )
    standard_error = draws.std(axis=0, ddof=1)
    statistic = observed / standard_error
    centered_null = (draws - observed) / standard_error
    order = np.argsort(-statistic)
    adjusted = np.zeros(len(observed))
    running = 0.0
    for rank, index in enumerate(order):
        probability = np.mean(
            np.max(centered_null[:, order[rank:]], axis=1) >= statistic[index]
        )
        running = max(running, float(probability))
        adjusted[index] = running
    actual = _multiple_sharpe_kernel(
        jnp.asarray(strategies), jnp.asarray(benchmark), jnp.asarray(indices)
    )
    np.testing.assert_allclose(actual[0], observed, rtol=1e-12)
    np.testing.assert_allclose(actual[1], statistic, rtol=1e-12)
    np.testing.assert_allclose(actual[2], adjusted, rtol=1e-12)
    assert float(actual[3]) == pytest.approx(
        float(np.mean(np.max(centered_null, axis=1) >= statistic.max()))
    )
    np.testing.assert_allclose(
        jax.vmap(_paired_sharpe_diffs)(
            jnp.asarray(strategies[indices]), jnp.asarray(benchmark[indices])
        ),
        draws,
        rtol=1e-12,
        atol=1e-14,
    )
    paired = _sharpe_difference_kernel(
        jnp.asarray(strategies[:, 0]), jnp.asarray(benchmark), jnp.asarray(indices)
    )
    expected_probability = np.mean(
        np.abs(draws[:, 0] - draws[:, 0].mean()) >= abs(observed[0])
    )
    np.testing.assert_allclose(paired, [statistic[0], expected_probability], rtol=1e-12)
    assert np.all(np.diff(np.asarray(actual[2], dtype=float)[order]) >= 0.0)


@pytest.mark.component
@pytest.mark.regression
def test_direct_sharpe_ci_preserves_the_prehoist_callable_path() -> None:
    """The directly invoked report retains percentile endpoints and exact seed bytes."""
    returns = np.asarray([-1.0, 0.2, 1.5, -0.3, 2.0, 0.5, 0.7, -0.1])

    def old_statistic(values: jax.Array) -> jax.Array:
        column = values[:, 0] if values.ndim > 1 else values
        deviation = jnp.std(column, ddof=1)
        return jnp.where(deviation > 0, jnp.mean(column) / deviation, 0.0)

    expected = stationary_bootstrap_ci(
        returns, old_statistic, num_bootstrap=39, seed=72
    )
    actual = sharpe_ratio_ci(returns, num_bootstrap=39, seed=72)
    assert tuple(actual) == (actual.lower, actual.upper)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        sharpe_ratio_ci(np.ones(8), num_bootstrap=39), (0.0, 0.0)
    )


@pytest.mark.component
@pytest.mark.performance
def test_stationary_bootstrap_reuses_a_stable_callable_without_retracing() -> None:
    """Numeric dataset changes do not create a new static callback cache key."""
    traces: list[None] = []

    def statistic(values: jax.Array) -> jax.Array:
        traces.append(None)
        return jnp.mean(values)

    first = stationary_bootstrap_ci(np.arange(8.0), statistic, num_bootstrap=19, seed=1)
    initial_traces = len(traces)
    second = stationary_bootstrap_ci(
        np.arange(8.0) + 0.5, statistic, num_bootstrap=19, seed=2
    )
    assert initial_traces > 0
    assert len(traces) == initial_traces
    assert np.isfinite([first, second]).all()


@pytest.mark.component
@pytest.mark.regression
def test_stationary_bootstrap_preserves_unhashable_eager_callable_instances() -> None:
    """The performance hoist cannot narrow the existing eager callable surface."""

    @dataclass
    class MeanStatistic:
        """Mutable dataclasses deliberately have no hash implementation."""

        shift: float = 0.25

        def __call__(self, values: jax.Array) -> jax.Array:
            return jnp.mean(values) + self.shift

    statistic = MeanStatistic()
    returns = np.arange(8.0)

    def legacy_wrapper(sample: jax.Array) -> jax.Array:
        return jnp.asarray(statistic(sample))

    expected = stationary_bootstrap_ci(
        returns, legacy_wrapper, num_bootstrap=19, seed=73
    )
    actual = stationary_bootstrap_ci(returns, statistic, num_bootstrap=19, seed=73)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("constant", [0.1, 0.2, 0.7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_non_dyadic_constant_returns_have_exactly_zero_sharpe_and_ci(
    constant: float,
    dtype: type[np.float32 | np.float64],
) -> None:
    """Mean-rounding residue cannot invent dispersion in identical observations."""
    with jax.enable_x64(new_val=dtype is np.float64):
        values = np.full(7, constant, dtype=dtype)
        assert sharpe_ratio(values) == pytest.approx(0.0, abs=0.0)
        np.testing.assert_array_equal(
            sharpe_ratio_axis(jnp.asarray(values), axis=0), 0.0
        )
        np.testing.assert_array_equal(
            sharpe_ratio_ci(values, num_bootstrap=19), (0.0, 0.0)
        )
        assert np.isnan(sample_skew_kurtosis(values)).all()


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resolved_tiny_return_spread_is_preserved_without_an_epsilon_cutoff(
    dtype: type[np.float32 | np.float64],
) -> None:
    """A positive represented range remains estimable even far below unit scale."""
    spread = 1e-5 if dtype is np.float32 else 1e-10
    values = np.asarray(
        [1.0, 1.0 + spread, 1.0 - spread, 1.0 + 2 * spread], dtype=dtype
    )
    with jax.enable_x64(new_val=dtype is np.float64):
        ratio = sharpe_ratio(values)
        axis_ratio = sharpe_ratio_axis(jnp.asarray(values), axis=0)
        assert np.isfinite(ratio)
        assert ratio > 1.0
        assert float(axis_ratio) > 1.0
        assert np.isfinite(sample_skew_kurtosis(values)).all()


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "invalid", ["nan_stat", "inf_stat", "nan_draw", "inf_draw", "zero_se"]
)
def test_multiple_sharpe_invalid_family_is_unavailable(invalid: str) -> None:
    """Any invalid member invalidates both joint and stepdown family tails."""
    observed = jnp.array([[0.5, 0.25], [0.5, 0.25]])
    draws = jnp.array([[[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5]]] * 2)
    if invalid.endswith("stat"):
        observed = observed.at[0, 0].set(jnp.nan if invalid == "nan_stat" else jnp.inf)
    elif invalid == "zero_se":
        draws = draws.at[0, :, 0].set(0.0)
    else:
        draws = draws.at[0, 0, 0].set(jnp.nan if invalid == "nan_draw" else -jnp.inf)
    baseline = _multiple_sharpe_calibration(observed[1], draws[1])
    for evaluate in (
        jax.vmap(_multiple_sharpe_calibration),
        jax.jit(jax.vmap(_multiple_sharpe_calibration)),
    ):
        result = evaluate(observed, draws)
        assert jnp.all(jnp.isnan(result[1][0]))
        assert jnp.isnan(result[2][0])
        np.testing.assert_array_equal(result[1][1], baseline[1])
        np.testing.assert_array_equal(result[2][1], baseline[2])
    for evaluate in (
        _multiple_sharpe_calibration,
        jax.jit(_multiple_sharpe_calibration),
    ):
        result = evaluate(observed[0], draws[0])
        assert jnp.all(jnp.isnan(result[1]))
        assert jnp.isnan(result[2])


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_multiple_sharpe_nonfinite_raw_input_is_unavailable(bad: float) -> None:
    """The Sharpe ratio's degenerate-data fallback cannot hide invalid inputs."""
    strategies = jnp.array([[1.0, 0.3], [0.2, -0.4], [-0.1, 1.1], [0.4, 0.8]])
    benchmark = jnp.array([0.4, -0.1, 0.2, 0.5])
    indices = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 2, 3, 1]])
    result = _multiple_sharpe_kernel(strategies.at[0, 0].set(bad), benchmark, indices)
    assert jnp.all(jnp.isnan(result[2]))
    assert jnp.isnan(result[3])


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_paired_sharpe_public_paths_reject_nonfinite_input(bad: float) -> None:
    """Invalid input cannot become evidence through Sharpe's fallback."""
    a = jnp.array([0.1, 0.2, -0.1, 0.4])
    b = jnp.array([0.4, -0.1, 0.2, 0.5])
    invalid = a.at[0].set(bad)
    assert np.isnan(sharpe_difference_test(invalid, b, n_boot=8, block_length=2)[1])
    key = jax.random.PRNGKey(3)

    def evaluate(x: jax.Array) -> jax.Array:
        return sharpe_difference_test_kernel(key, x, b, n_boot=8).pvalue

    baseline = evaluate(a)
    assert jnp.isnan(evaluate(invalid))
    assert jnp.isnan(jax.jit(evaluate)(invalid))
    for run in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate))):
        result = run(jnp.stack([invalid, a]))
        assert jnp.isnan(result[0])
        np.testing.assert_array_equal(result[1], baseline)
