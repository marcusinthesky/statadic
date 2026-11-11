# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Bootstrap stepdown and joint max-statistic comparisons of strategy Sharpes.

Circular blocks preserve paired rows. Each strategy uses a common original
bootstrap standard deviation in every resample. The stepdown architecture is
related to Romano-Wolf's fixed-studentizer alternative; the standardized joint
maximum differs from White's original raw-maximum stationary-bootstrap setup.
Familywise control requires justified bootstrap and selection assumptions.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike  # noqa: TC002
from jaxtyping import Float, Int

from statadic.core.typing import Array, ArrayLike, PRNGKey  # noqa: TC001
from statadic.estimation.sharpe import sharpe_ratio_axis
from statadic.inference.resampling.block_length import (
    optimal_block_length,
    optimal_block_length_result,
)
from statadic.inference.sharpe.difference import (  # noqa: TC001 -- runtime annotations
    FloatScalar,
    ReturnVector,
)
from statadic.operators.longrun import (
    circular_block_bootstrap_indices,
    circular_block_bootstrap_indices_kernel,
)

__all__ = [
    "MultipleSharpeKernelResult",
    "MultipleSharpeResult",
    "StrategyPanel",
    "StrategyVector",
    "multiple_sharpe_test",
    "multiple_sharpe_test_kernel",
]

type StrategyPanel = Float[Array, "observations strategies"]
type StrategyVector = Float[Array, " strategies"]  # jaxtyping
type _BootstrapMatrix = Int[Array, "bootstrap observations"]


class MultipleSharpeResult(NamedTuple):
    """Result of a familywise multiple Sharpe-superiority test.

    Attributes:
        strategy_labels: Labels aligned with the statistics arrays.
        sharpe_diff: Observed Sharpe difference ``SR(strategy) - SR(benchmark)``
            per strategy.
        studentized: Studentized statistic ``sharpe_diff / bootstrap_se`` per
            strategy.
        rw_adjusted_pvalues: One-sided stepdown adjusted exceedance proportions
            for the fixed-studentizer bootstrap construction.
        reality_check_pvalue: Joint exceedance proportion for the maximum
            standardized Sharpe difference; a Reality-Check-inspired variant.
        block_length: Circular-block-bootstrap block length used.
        n_boot: Number of bootstrap resamples.

    """

    strategy_labels: tuple[str, ...]
    sharpe_diff: StrategyVector
    studentized: StrategyVector
    rw_adjusted_pvalues: StrategyVector
    reality_check_pvalue: float
    block_length: int
    n_boot: int


class MultipleSharpeKernelResult(NamedTuple):
    """Array-only carrier for a keyed multiple-Sharpe test."""

    sharpe_diff: StrategyVector
    studentized: StrategyVector
    rw_adjusted_pvalues: StrategyVector
    reality_check_pvalue: FloatScalar
    block_length: Int[Array, ""]


@jax.jit
def _paired_sharpe_diffs(
    block: StrategyPanel,
    bench: ReturnVector,
) -> StrategyVector:
    """Sharpe difference of every column of ``block`` vs ``bench`` (all (T,)).

    Args:
        block: Strategy return panel, shape ``(T, k)``.
        bench: Benchmark return series, shape ``(T,)``.

    Returns:
        Per-strategy Sharpe difference, shape ``(k,)``.

    """
    return sharpe_ratio_axis(block, axis=0) - sharpe_ratio_axis(bench, axis=0)


@jax.jit
def _multiple_sharpe_kernel(
    strategy_returns: StrategyPanel,
    benchmark_returns: ReturnVector,
    indices: _BootstrapMatrix,
) -> tuple[
    StrategyVector,
    StrategyVector,
    StrategyVector,
    FloatScalar,
]:
    """Evaluate bootstrap Sharpe statistics on a fixed resample index matrix.

    Args:
        strategy_returns: Strategy panel, shape ``(T, k)``.
        benchmark_returns: Benchmark series, shape ``(T,)``.
        indices: Fixed bootstrap indices, shape ``(B, T)``.

    Returns:
        Observed differences, studentized statistics, Romano-Wolf adjusted
        p-values, and the scalar Reality-Check p-value.

    """
    diff_hat = _paired_sharpe_diffs(strategy_returns, benchmark_returns)
    strategy_boot = strategy_returns[indices]
    benchmark_boot = benchmark_returns[indices]
    diff_boot = (
        sharpe_ratio_axis(strategy_boot, axis=1)
        - sharpe_ratio_axis(benchmark_boot, axis=1)[:, None]
    )

    studentized, adjusted, reality_check = _multiple_sharpe_calibration(
        diff_hat, diff_boot, dtype=strategy_returns.dtype
    )
    valid_inputs = jnp.all(jnp.isfinite(strategy_returns)) & jnp.all(
        jnp.isfinite(benchmark_returns)
    )
    return (
        diff_hat,
        studentized,
        jnp.where(valid_inputs, adjusted, jnp.nan),
        jnp.where(valid_inputs, reality_check, jnp.nan),
    )


def _multiple_sharpe_calibration(
    diff_hat: StrategyVector,
    diff_boot: Float[Array, "bootstrap strategies"],
    *,
    dtype: DTypeLike | None = None,
) -> tuple[StrategyVector, StrategyVector, FloatScalar]:
    """Calibrate the complete family only when every required value is finite."""
    standard_error = jnp.std(diff_boot, axis=0, ddof=1)
    valid = (
        jnp.all(jnp.isfinite(diff_hat))
        & jnp.all(jnp.isfinite(diff_boot))
        & jnp.all(jnp.isfinite(standard_error) & (standard_error > 0.0))
    )
    standard_error = jnp.where(standard_error > 0.0, standard_error, jnp.inf)
    studentized = diff_hat / standard_error
    null_boot = (diff_boot - diff_hat[None, :]) / standard_error[None, :]
    valid = (
        valid & jnp.all(jnp.isfinite(studentized)) & jnp.all(jnp.isfinite(null_boot))
    )

    reality_check = jnp.mean(
        jnp.max(null_boot, axis=1) >= jnp.max(studentized),
        dtype=dtype or diff_hat.dtype,
    )

    order = jnp.argsort(-studentized)
    ordered_null = null_boot[:, order]
    ordered_statistics = studentized[order]
    step_maxima = lax.associative_scan(
        jnp.maximum,
        ordered_null,
        axis=1,
        reverse=True,
    )
    raw_pvalues = jnp.mean(
        step_maxima >= ordered_statistics[None, :],
        axis=0,
        dtype=dtype or diff_hat.dtype,
    )
    monotone_pvalues = lax.associative_scan(jnp.maximum, raw_pvalues)
    adjusted = jnp.empty_like(monotone_pvalues).at[order].set(monotone_pvalues)
    return (
        studentized,
        jnp.where(valid, adjusted, jnp.nan),
        jnp.where(valid, reality_check, jnp.nan),
    )


@partial(jax.jit, static_argnames=("n_boot",))
def multiple_sharpe_test_kernel(
    key: PRNGKey,
    strategy_returns: StrategyPanel,
    benchmark_returns: ReturnVector,
    *,
    n_boot: int,
) -> MultipleSharpeKernelResult:
    """Run the multiple-Sharpe test from an explicit bootstrap key."""
    observed = _paired_sharpe_diffs(strategy_returns, benchmark_returns)
    best = jnp.argmax(observed)
    block_length = jnp.rint(
        optimal_block_length_result(
            strategy_returns[:, best] - benchmark_returns,
            scheme="circular",
        )
    ).astype(jnp.int32)
    block_length = jnp.clip(block_length, 1, strategy_returns.shape[0])
    indices = circular_block_bootstrap_indices_kernel(
        key,
        strategy_returns.shape[0],
        block_length,
        n_boot,
    )
    diff_hat, studentized, adjusted, reality_check = _multiple_sharpe_kernel(
        strategy_returns,
        benchmark_returns,
        indices,
    )
    return MultipleSharpeKernelResult(
        diff_hat,
        studentized,
        adjusted,
        reality_check,
        block_length,
    )


def multiple_sharpe_test(
    strategy_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    strategy_labels: tuple[str, ...] | None = None,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> MultipleSharpeResult:
    """Compute fixed-studentizer stepdown and joint Sharpe comparisons.

    Tests, for each strategy ``k``, ``H0_k: SR_k <= SR_benchmark`` against the
    one-sided alternative that strategy ``k`` beats the benchmark. The base
    statistic is ``d_k = SR_k - SR_bench`` divided by its circular-bootstrap
    standard error. Bootstrap differences are recentered, using the same original
    studentizer in every resample. Stepdown maxima follow the fixed-studentizer
    architecture discussed by Romano-Wolf (2005, footnote 22).

    The joint output uses the maximum standardized difference with circular
    blocks. Its correspondence to White's original stationary-bootstrap raw
    maximum remains separately reviewed. Finite-bootstrap exceedance proportions
    have no add-one correction. Familywise validity requires appropriate
    bootstrap, studentizer and block-selection assumptions; it is not guaranteed
    for arbitrary serial dependence or strategy selection.

    Args:
        strategy_returns: Strategy return panel, shape ``(T, k)`` (columns are
            strategies). A 1-D array is treated as a single strategy.
        benchmark_returns: Benchmark return series, shape ``(T,)`` (e.g. 1/N).
        strategy_labels: Optional labels for the ``k`` strategies.
        n_boot: Number of circular-block-bootstrap resamples.
        block_length: Block length ``ell``; ``None`` uses the Politis-White
            automatic length
            (:func:`statadic.inference.resampling.block_length.optimal_block_length`,
            circular scheme) on the best strategy's differential series.
        seed: RNG seed for the bootstrap.

    Returns:
        A :class:`MultipleSharpeResult`.

    Raises:
        ValueError: If the benchmark length or the label count does not match
            the strategy panel.

    """
    strat = jnp.asarray(strategy_returns)
    if strat.ndim == 1:
        strat = strat.reshape(-1, 1)
    bench = jnp.asarray(benchmark_returns).ravel()
    n_obs, n_strat = strat.shape
    if bench.shape[0] != n_obs:
        message = "benchmark and strategies must share the same length"
        raise ValueError(message)
    labels = strategy_labels or tuple(f"strategy_{j}" for j in range(n_strat))
    if len(labels) != n_strat:
        message = "strategy_labels length must match the number of strategies"
        raise ValueError(message)

    observed = _paired_sharpe_diffs(strat, bench)
    if block_length is None:
        best = int(jnp.argmax(observed))
        ell = round(
            optimal_block_length(
                strat[:, best] - bench,
                scheme="circular",
            )
        )
    else:
        ell = int(block_length)
    ell = max(1, min(ell, n_obs))

    indices = jnp.asarray(
        circular_block_bootstrap_indices(n_obs, ell, n_boot, seed=seed),
        dtype=jnp.int64,
    )
    diff_hat, studentized, adjusted, reality_check = _multiple_sharpe_kernel(
        strat,
        bench,
        indices,
    )

    return MultipleSharpeResult(
        strategy_labels=labels,
        sharpe_diff=diff_hat,
        studentized=studentized,
        rw_adjusted_pvalues=adjusted,
        reality_check_pvalue=float(reality_check),
        block_length=ell,
        n_boot=n_boot,
    )
