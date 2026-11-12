# jaxtyping shape strings are runtime contracts.
"""T48.4 parity: JAX Sharpe kernels against independent NumPy/SciPy oracles.

The retired implementation is reproduced here rather than imported. This keeps
SciPy independent of the production path and makes the precision comparison
meaningful after ``decision/_sharpe`` drops its last NumPy/SciPy imports.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from statadic.estimation.sharpe import (
    sample_skew_kurtosis,
    sharpe_ratio,
    sharpe_ratio_axis,
)
from statadic.inference.sharpe.deflated import (
    _probabilistic_sharpe_ratio_kernel,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from statadic.inference.sharpe.difference import _sharpe_difference_kernel
from statadic.inference.sharpe.multiple import _multiple_sharpe_kernel

_TIGHT_RTOL = 5e-13
_TIGHT_ATOL = 5e-15
_FAR_TAIL_THRESHOLD = 1e-6


def _numpy_sharpe_axis(values: np.ndarray, axis: int) -> np.ndarray:
    """Retired NumPy ``mean/std(ddof=1)`` reduction.

    Args:
        values: Numeric array to reduce.
        axis: Reduction axis.

    Returns:
        Sharpe ratios over ``axis``.

    """
    mean = values.mean(axis=axis)
    standard_deviation = values.std(axis=axis, ddof=1)
    return np.where(standard_deviation > 0.0, mean / standard_deviation, 0.0)


def _numpy_multiple_oracle(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Retired Romano-Wolf/Reality-Check arithmetic on fixed indices.

    Args:
        strategy_returns: Strategy return panel, shape ``(T, k)``.
        benchmark_returns: Benchmark returns, shape ``(T,)``.
        indices: Fixed bootstrap indices, shape ``(B, T)``.

    Returns:
        Sharpe differences, studentized statistics, adjusted p-values, and
        the Reality-Check p-value.

    """
    difference = _numpy_sharpe_axis(strategy_returns, axis=0) - float(
        _numpy_sharpe_axis(benchmark_returns, axis=0)
    )
    bootstrap_difference = (
        _numpy_sharpe_axis(strategy_returns[indices], axis=1)
        - _numpy_sharpe_axis(benchmark_returns[indices], axis=1)[:, None]
    )
    standard_error = bootstrap_difference.std(axis=0, ddof=1)
    standard_error = np.where(standard_error > 0.0, standard_error, np.inf)
    studentized = difference / standard_error
    null_bootstrap = (bootstrap_difference - difference[None, :]) / standard_error[
        None, :
    ]
    reality_check = float(
        np.mean(null_bootstrap.max(axis=1) >= float(np.max(studentized)))
    )

    order = np.argsort(-studentized)
    adjusted = np.ones(strategy_returns.shape[1], dtype=np.float64)
    previous = 0.0
    for position, strategy in enumerate(order):
        active = order[position:]
        step_maximum = null_bootstrap[:, active].max(axis=1)
        pvalue = max(float(np.mean(step_maximum >= studentized[strategy])), previous)
        adjusted[strategy] = pvalue
        previous = pvalue
    return difference, studentized, adjusted, reality_check


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("seed", range(8))
def test_sharpe_and_moments_match_numpy_scipy(seed: int) -> None:
    """Scalar, axis, skew, and non-excess kurtosis retain float64 semantics."""
    rng = np.random.default_rng(seed)
    values = rng.standard_t(df=7, size=(64, 5)) + 0.03
    expected_axis = _numpy_sharpe_axis(values, axis=0)
    with jax.enable_x64(new_val=True):
        got_axis = np.asarray(sharpe_ratio_axis(values, axis=0))
    np.testing.assert_allclose(
        got_axis,
        expected_axis,
        rtol=_TIGHT_RTOL,
        atol=_TIGHT_ATOL,
    )

    series = values[:, 0]
    expected_scalar = float(_numpy_sharpe_axis(series, axis=0))
    assert sharpe_ratio(series) == pytest.approx(
        expected_scalar,
        rel=_TIGHT_RTOL,
        abs=_TIGHT_ATOL,
    )
    skew, kurtosis = sample_skew_kurtosis(series)
    assert skew == pytest.approx(
        float(stats.skew(series, bias=True)),
        rel=_TIGHT_RTOL,
        abs=_TIGHT_ATOL,
    )
    assert kurtosis == pytest.approx(
        float(stats.kurtosis(series, fisher=False, bias=True)),
        rel=_TIGHT_RTOL,
        abs=_TIGHT_ATOL,
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("z", [-5.0, -8.0, -15.0])
def test_probabilistic_sharpe_preserves_far_lower_tail(z: float) -> None:
    """PSR remains positive and agrees with SciPy well below ``p = 1e-6``."""
    expected = float(stats.norm.cdf(z))
    got = probabilistic_sharpe_ratio(
        sr_hat=0.0,
        n_obs=2,
        skew=0.0,
        kurtosis=3.0,
        benchmark_sr=-z,
    )
    assert expected < _FAR_TAIL_THRESHOLD
    assert got > 0.0
    assert got == pytest.approx(expected, rel=2e-12, abs=0.0)


@pytest.mark.component
@pytest.mark.reference
def test_public_psr_precision_follows_the_caller() -> None:
    """The public PSR adapter no longer protects an x64-disabled caller.

    It used to open a door of its own. It does not any more, so the far-tail
    normal CDF underflows identically at the public surface and at the kernel
    — the honest statement of "precision is the caller's". A caller that owns
    the flag still gets the SciPy reference to 2e-12.
    """
    expected = float(stats.norm.cdf(-15.0))
    previous_x64 = bool(jax.config.read("jax_enable_x64"))
    try:
        with jax.enable_x64(new_val=False):
            low_precision = float(
                _probabilistic_sharpe_ratio_kernel(
                    jnp.asarray(0.0),
                    jnp.asarray(2.0),
                    jnp.asarray(0.0),
                    jnp.asarray(3.0),
                    jnp.asarray(15.0),
                )
            )
            unowned = probabilistic_sharpe_ratio(0.0, 2, 0.0, 3.0, 15.0)
        with jax.enable_x64(new_val=True):
            owned = probabilistic_sharpe_ratio(0.0, 2, 0.0, 3.0, 15.0)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    assert low_precision == 0.0  # float32 underflow is exact  # noqa: RUF069
    assert unowned == low_precision
    assert owned == pytest.approx(expected, rel=2e-12, abs=0.0)


#: Exact float64 value of ``(1 + 1e-10) - 1``, measured on jax 0.10.2 (t56.6.1).
#: In float32 the same subtraction is exactly ``0.0`` — ``1 + 1e-10`` rounds to
#: ``1.0`` well inside the float32 spacing at 1 (~1.19e-7) — which is what makes
#: this probe discriminate a genuine float64 input from a canonicalized one.
_X64_PROBE_GAP = 1.000000082740371e-10


def _x64_door(x: np.ndarray | jax.Array) -> jax.Array:
    """Door-scoped x64 context around a float64 body, as ``_sharpe`` writes it.

    Args:
        x: Two-element float64 probe.

    Returns:
        The difference of the two entries.

    """
    with jax.enable_x64(new_val=True):
        r = jnp.asarray(x, dtype=jnp.float64)
        return r[0] - r[1]


def _x64_door_sum_of_squares(x: np.ndarray | jax.Array) -> jax.Array:
    """Differentiable scalar with the same door, for the cotangent check.

    Args:
        x: Float64 probe.

    Returns:
        ``sum(x**2)`` computed inside the door's x64 context.

    """
    with jax.enable_x64(new_val=True):
        r = jnp.asarray(x, dtype=jnp.float64)
        return jnp.sum(r * r)


@pytest.mark.integration
@pytest.mark.contract
def test_door_scoped_x64_does_not_survive_jit() -> None:
    """Pin the precision boundary: a door owns eager values, the caller owns x64.

    The contract is ``docs/numerical-host-boundaries.md`` §"Precision boundary:
    who owns x64" (t56.6.1). Asserted in the *un*-preserved direction on purpose:
    if a future JAX release reordered input canonicalization so that a door's
    context reached its own arguments, ``traced_value`` would become
    ``1.0000000827e-10`` and this test would fail — instead of a manuscript
    number moving silently. Any migration that needs float64 under ``jit``, or
    under any transform whose argument is already a committed ``jax.Array``,
    must make the caller own the flag, not add a door.

    **Order-coupled to** ``tests/conftest.py:43``, which turns the process-global
    flag *on* for the whole session: that is what makes the ``new_val=False``
    block below an explicit x64-disabled caller rather than a no-op. Under a
    runner without this package's conftest the flip would restate the JAX
    default instead of deliberately dropping x64. The measured values are the
    same either way (checked at both entry states, jax 0.10.2, CPU); the
    coupling is what the block's *meaning* rests on, not its numbers.
    """
    probe = np.array([1.0 + 1e-10, 1.0], dtype=np.float64)
    batch = np.stack([probe, probe])
    previous_x64 = bool(jax.config.read("jax_enable_x64"))
    try:
        with jax.enable_x64(new_val=False):
            eager = float(_x64_door(probe))
            traced_value = jax.jit(_x64_door)(probe)
            cotangent = jax.grad(_x64_door_sum_of_squares)(probe)
            mapped_host = jax.vmap(_x64_door)(batch)
            committed = (
                jax.vmap(_x64_door)(jnp.asarray(batch)),
                jax.jit(jax.vmap(_x64_door))(batch),
                jax.vmap(jax.jit(_x64_door))(batch),
            )
        with jax.enable_x64(new_val=True):
            caller_owned = float(jax.jit(_x64_door)(probe))
    finally:
        jax.config.update("jax_enable_x64", previous_x64)

    # Eager entry: the door's context does govern the values it computes.
    assert eager == pytest.approx(_X64_PROBE_GAP, rel=1e-12, abs=0.0)
    # Under `jit` it does not: the input was canonicalized to float32 first, so
    # the door upcasts data that has already lost the gap.
    assert float(traced_value) == 0.0  # float32 rounding is exact  # noqa: RUF069
    # ...and the *dtype* is float64 even then, which is why only the value can
    # discriminate this and a dtype assertion would pass while being wrong.
    assert traced_value.dtype == jnp.float64
    # A cotangent follows the primal input dtype, so the door does not govern
    # the reverse pass either — the same fact, not a second defect.
    assert cotangent.dtype == jnp.float32
    # Bare `vmap` over a *raw NumPy* argument is the measured exception: nothing
    # commits the host buffer to a `jax.Array` before the door runs, so the
    # door's context still reaches it. The trigger is array construction, not
    # `vmap`.
    np.testing.assert_allclose(
        np.asarray(mapped_host),
        _X64_PROBE_GAP,
        rtol=1e-12,
        atol=0.0,
    )
    # It stops holding as soon as anything commits the argument first — an
    # explicit `jnp.asarray`, or a `jit` boundary on either side of the `vmap`.
    assert [np.asarray(value).tolist() for value in committed] == [[0.0, 0.0]] * 3
    # Caller-owned: correct, and the only supported way to get float64 here.
    assert caller_owned == pytest.approx(_X64_PROBE_GAP, rel=1e-12, abs=0.0)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("n_trials", [2, 10, 100, 1_000_000])
def test_expected_maximum_matches_scipy_ppf(n_trials: int) -> None:
    """The JAX inverse-normal deflation benchmark matches the SciPy oracle."""
    variance = 0.04
    gamma = 0.5772156649015329
    n = float(n_trials)
    expected = np.sqrt(variance) * (
        (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n)
        + gamma * stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    )
    assert expected_maximum_sharpe_ratio(n_trials, variance) == pytest.approx(
        float(expected),
        rel=2e-12,
        abs=_TIGHT_ATOL,
    )


@pytest.mark.component
@pytest.mark.reference
def test_difference_kernel_matches_numpy_on_fixed_indices() -> None:
    """Bootstrap stream ownership is orthogonal to paired-test arithmetic."""
    rng = np.random.default_rng(41)
    returns_a = rng.normal(0.04, 0.2, size=96)
    returns_b = rng.normal(0.02, 0.2, size=96)
    indices = rng.integers(0, returns_a.size, size=(127, returns_a.size))

    difference = float(_numpy_sharpe_axis(returns_a, axis=0)) - float(
        _numpy_sharpe_axis(returns_b, axis=0)
    )
    bootstrap = _numpy_sharpe_axis(returns_a[indices], axis=1) - _numpy_sharpe_axis(
        returns_b[indices], axis=1
    )
    standard_error = float(bootstrap.std(ddof=1))
    expected_statistic = difference / standard_error
    centered = bootstrap - bootstrap.mean()
    expected_pvalue = float(np.mean(np.abs(centered) >= np.abs(difference)))

    with jax.enable_x64(new_val=True):
        statistic, pvalue = _sharpe_difference_kernel(
            jnp.asarray(returns_a),
            jnp.asarray(returns_b),
            jnp.asarray(indices),
        )
    assert float(statistic) == pytest.approx(
        expected_statistic,
        rel=_TIGHT_RTOL,
        abs=_TIGHT_ATOL,
    )
    assert float(pvalue) == pytest.approx(expected_pvalue, abs=_TIGHT_ATOL)


@pytest.mark.component
@pytest.mark.reference
def test_multiple_kernel_matches_numpy_on_fixed_indices() -> None:
    """Vectorisation preserves the retired step-down procedure exactly in law."""
    rng = np.random.default_rng(42)
    strategies = rng.normal(size=(80, 4)) + np.array([0.1, 0.0, -0.05, 0.02])
    benchmark = rng.normal(size=80)
    indices = rng.integers(0, strategies.shape[0], size=(127, strategies.shape[0]))
    expected = _numpy_multiple_oracle(strategies, benchmark, indices)

    with jax.enable_x64(new_val=True):
        got = _multiple_sharpe_kernel(
            jnp.asarray(strategies),
            jnp.asarray(benchmark),
            jnp.asarray(indices),
        )
    for actual, oracle in zip(got[:3], expected[:3], strict=True):
        np.testing.assert_allclose(
            np.asarray(actual),
            oracle,
            rtol=_TIGHT_RTOL,
            atol=_TIGHT_ATOL,
        )
    assert float(got[3]) == pytest.approx(expected[3], abs=_TIGHT_ATOL)
    assert all(value.dtype == jnp.float64 for value in got)
