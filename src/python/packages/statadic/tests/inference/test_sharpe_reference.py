"""Independent fixed-draw and exact Gaussian controls for direct Sharpe CIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.random import cell_key
from statadic.estimation.sharpe import sharpe_ratio
from statadic.inference.resampling.block_length import optimal_block_length_result
from statadic.inference.resampling.schedule import stationary_row_schedule
from statadic.inference.sharpe import sharpe_ratio_ci

from .._validation.calibration import trial_rng
from ..qualification.test_returns import _return_sample

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_LONG_CASE = 11


@jax.jit
def _fixed_indices(key: jax.Array, length: jax.Array) -> jax.Array:
    """Expose the actual 999-draw schedule independently of statistic evaluation."""
    return stationary_row_schedule(key, 128, 999, length).indices


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(
    ("law", "scenario", "long_trial"),
    [("gaussian", 1000, 171), ("skewed", 1001, 180), ("dependent", 1002, 158)],
)
@pytest.mark.parametrize("case", range(12))
def test_direct_sharpe_endpoints_match_numpy_fixed_schedule(
    law: str, scenario: int, long_trial: int, case: int
) -> None:
    """Catch ratio normalization or percentile changes, including very long blocks.

    Eleven ordinary datasets and one fixed long-block witness per original law
    use the public call's exact key and selected schedule. NumPy separately
    evaluates every sample ratio and both percentile endpoints. The long-block
    cases are numerical witnesses, not a claim that a different cap is required.
    """
    trial = case if case < _LONG_CASE else long_trial
    returns = _return_sample(law, scenario, trial)
    seed = int(trial_rng(scenario, trial, stream=1).integers(0, 2**31 - 1))
    length = optimal_block_length_result(jnp.asarray(returns))
    indices = np.asarray(
        _fixed_indices(cell_key(seed, "inference", "stationary_bootstrap_ci"), length)
    )
    samples = returns[indices]
    ratios = samples.mean(axis=1) / samples.std(axis=1, ddof=1)
    expected = np.percentile(ratios, [2.5, 97.5])

    actual = sharpe_ratio_ci(returns, num_bootstrap=999, seed=seed)

    # O(128) reductions and linear percentile interpolation in float64.
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    if case == _LONG_CASE:
        assert float(length) > (128 if law != "skewed" else 20)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("count", [8, 128])
@pytest.mark.parametrize("population_sharpe", [-0.2, 0.2])
@pytest.mark.parametrize(
    ("probability", "boundary"), [(0.025, 1), (0.5, None), (0.975, 0)]
)
def test_gaussian_sharpe_exact_noncentral_t_control(
    optional_reference: Callable[[str], ModuleType],
    count: int,
    population_sharpe: float,
    probability: float,
    boundary: int | None,
) -> None:
    """Protect sqrt(T), ddof=1 and the exact Gaussian confidence-set inversion.

    Normal sample mean and sample variance are independent, giving
    sqrt(T)*SR_hat ~ nct(T-1, sqrt(T)*SR). Deterministic sufficient-statistic
    witnesses avoid a simulation tolerance or a random coverage assertion.
    This exact parametric control is not the public percentile interval.
    """
    stats = optional_reference("scipy.stats")
    optimize = optional_reference("scipy.optimize")
    root_count = np.sqrt(count)
    observed_t = stats.nct.ppf(
        probability, df=count - 1, nc=root_count * population_sharpe
    )
    centered = np.arange(count, dtype=np.float64)
    centered -= centered.mean()
    centered /= centered.std(ddof=1)
    returns = centered + observed_t / root_count

    actual_t = root_count * sharpe_ratio(returns)

    assert stats.nct.cdf(
        actual_t, df=count - 1, nc=root_count * population_sharpe
    ) == pytest.approx(probability, rel=2e-12, abs=2e-14)

    def cdf_at_sharpe(sr: float) -> float:
        """Reflect negative t to avoid opposite-tail cancellation in SciPy CDF."""
        if actual_t < 0:
            return float(stats.nct.sf(-actual_t, df=count - 1, nc=-root_count * sr))
        return float(stats.nct.cdf(actual_t, df=count - 1, nc=root_count * sr))

    limits = []
    for tail in (0.975, 0.025):

        def evaluate_centered_cdf(sr: float, probability: float = tail) -> float:
            return cdf_at_sharpe(sr) - probability

        limits.append(optimize.brentq(evaluate_centered_cdf, -10.0, 10.0, xtol=1e-13))
    if boundary is None:
        assert limits[0] < population_sharpe < limits[1]
    else:
        assert limits[boundary] == pytest.approx(population_sharpe, abs=2e-13)
