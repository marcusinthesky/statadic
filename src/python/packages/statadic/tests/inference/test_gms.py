# standalone package-maintenance entry point
"""Tests for :mod:`statadic.inference.gms`.

The indicator GMS moment-inequality diagnostic: size, power, and the
finite-sample p-value floor.
"""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.random import cell_key
from statadic.inference.gms import (
    _finite_mc_calibration,
    indicator_gms,
)
from statadic.inference.wolak import (
    wolak_test,
)

from .._budgets import examples
from .._typing import ill_typed
from ._moments import (
    MINIMUM_GMS_POWER,
    NEAR_ZERO_STATISTIC,
    SIGNIFICANCE_LEVEL,
    full_rank_panels,
)

# ---------------------------------------------------------------------------
# indicator_gms (moment-inequality diagnostic)
# ---------------------------------------------------------------------------


@pytest.mark.component
@examples(8)
@example(g=1.0 + 0.3 * np.random.default_rng(20).standard_normal((300, 3)))
@given(g=full_rank_panels(300, 3).map(lambda panel: 1.0 + 0.3 * panel))
def test_gms_slack_satisfied_not_rejected(g: np.ndarray) -> None:
    """GMS does not reject inequalities with comfortably positive slack.

    Args:
        g: Generated panel whose moments are all far above zero.

    """
    res = indicator_gms(g, alpha=0.05, n_mc=2000, seed=0)
    assert res.stat < NEAR_ZERO_STATISTIC
    assert not res.reject
    assert res.pvalue > SIGNIFICANCE_LEVEL
    assert res.n_selected == 0  # all moments selected as slack
    assert res.calibration == "indicator_gms_uncorrected"
    assert not res.rms_table_i_calibrated


@pytest.mark.component
@examples(8)
@example(g=1.0 + 0.3 * np.random.default_rng(21).standard_normal((300, 3)))
@given(g=full_rank_panels(300, 3).map(lambda panel: 1.0 + 0.3 * panel))
def test_gms_violation_rejected(g: np.ndarray) -> None:
    """GMS rejects a clear moment-inequality violation.

    Args:
        g: Generated panel with comfortably positive moments before the shift.

    """
    g = g.copy()
    g[:, 0] -= 2.5  # one moment mean ~ -1.5 → strongly violated
    res = indicator_gms(g, alpha=0.05, n_mc=2000, seed=0)
    assert res.stat > 0.0
    assert res.reject
    assert res.pvalue < SIGNIFICANCE_LEVEL


@pytest.mark.component
@pytest.mark.numerical
@examples(20)
@example(g=np.random.default_rng(22).standard_normal((100, 2)) - 20.0)
@given(g=full_rank_panels(100, 2).map(lambda panel: panel - 20.0))
def test_gms_monte_carlo_pvalue_has_finite_sample_floor(g: np.ndarray) -> None:
    """A zero-extreme Monte Carlo sample reports 1/(B+1), never zero.

    Args:
        g: Generated panel with every moment far below zero.

    """
    result = indicator_gms(g, n_mc=7, seed=0)
    assert result.pvalue == pytest.approx(1.0 / 8.0)
    assert not result.reject
    assert np.isinf(result.crit)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(g=np.random.default_rng(23).standard_normal((20, 2)), n_mc=0)
@given(g=full_rank_panels(20, 2), n_mc=st.integers(max_value=0))
def test_gms_requires_at_least_one_monte_carlo_draw(g: np.ndarray, n_mc: int) -> None:
    """Indicator GMS rejects an empty simulated reference distribution.

    Args:
        g: Generated full-rank moment panel.
        n_mc: Any non-positive draw count.

    """
    with pytest.raises(ValueError, match="positive integer"):
        indicator_gms(g, n_mc=n_mc)


@pytest.mark.component
@pytest.mark.contract
@examples(60)
@example(control=("alpha", 0.0))
@example(control=("alpha", 1.0))
@example(control=("kappa_n", 0.0))
@example(control=("eta", -0.1))
@given(
    control=st.one_of(
        st.tuples(
            st.just("alpha"),
            st.floats(allow_nan=False).filter(lambda a: not 0 < a < 1),
        ),
        st.tuples(st.just("kappa_n"), st.floats(max_value=0.0)),
        st.tuples(st.just("eta"), st.floats(max_value=-1e-9)),
    )
)
def test_gms_validates_calibration_controls(control: tuple[str, float]) -> None:
    """Invalid size, selection, or correction controls fail cleanly.

    Args:
        control: A keyword and an out-of-domain value for it.

    """
    keyword, value = control
    g = np.random.default_rng(25).standard_normal((30, 2))
    with pytest.raises(ValueError, match=r"alpha|kappa_n|eta"):
        indicator_gms(g, n_mc=10, **{keyword: ill_typed(value)})


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(x=np.random.default_rng(26).standard_normal(40))
@given(x=hnp.arrays(np.float64, 40, elements=st.floats(-5, 5)))
def test_gms_rejects_rank_deficient_moment_covariance(x: np.ndarray) -> None:
    """Duplicate moment directions cannot inherit a full-dimensional reference.

    Args:
        x: Generated series duplicated into two identical moment columns.

    """
    with pytest.raises(ValueError, match="full rank"):
        indicator_gms(np.column_stack([x, x]), n_mc=10)


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_gms_size_conservative() -> None:
    """GMS remains size-conservative under independent boundary moments."""
    # Inequality tests must be conservative: at the least-favourable boundary
    # (all moments = 0) empirical size <= nominal.
    n_rep = 300
    alpha = 0.05
    rej = 0
    for s in range(n_rep):
        rng = np.random.default_rng(5000 + s)
        g = rng.standard_normal((300, 3))  # boundary: all means = 0
        res = indicator_gms(g, alpha=alpha, n_mc=1500, seed=s)
        if res.reject:
            rej += 1
    rate = rej / n_rep
    assert rate <= alpha + 0.03  # conservative (<= nominal + slack)


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_gms_size_conservative_correlated() -> None:
    """GMS remains size-conservative under correlated boundary moments."""
    # Under strongly CORRELATED moments at the least-favourable boundary the GMS
    # QLR distance-simulation stays correctly sized (~ nominal), whereas the
    # least-favourable chi-bar-squared weight approximation becomes markedly
    # over-conservative. This is the property that makes GMS reject the panel
    # floor where the baseline Wolak (over-conservatively) does not.
    n_rep = 300
    alpha = 0.05
    k = 6
    rho = 0.7
    correlation = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    l_cholesky = np.linalg.cholesky(correlation)
    rej = 0
    for s in range(n_rep):
        rng = np.random.default_rng(12000 + s)
        g = (
            l_cholesky @ rng.standard_normal((k, 300))
        ).T  # mean 0, correlation correlation
        if indicator_gms(g, alpha=alpha, n_mc=1200, seed=s, kappa_n=1e12).reject:
            rej += 1
    rate = rej / n_rep
    assert rate <= alpha + 0.03  # correctly sized (not the ~0.003 Wolak floor)


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_gms_power_reasonable() -> None:
    """GMS retains useful power against the configured alternative."""
    n_rep = 150
    alpha = 0.05
    rej = 0
    for s in range(n_rep):
        rng = np.random.default_rng(6000 + s)
        g = rng.standard_normal((300, 3))
        g[:, 0] -= 0.8  # moderate violation on one moment
        res = indicator_gms(g, alpha=alpha, n_mc=1500, seed=s)
        if res.reject:
            rej += 1
    power = rej / n_rep
    assert power > MINIMUM_GMS_POWER


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_gms_more_powerful_than_wolak_with_slack() -> None:
    """Slack-moment selection improves power relative to the Wolak baseline."""
    # GMS's moment selection should not be *less* powerful than the least-
    # favourable Wolak test when many moments are strongly slack. Compare on a
    # DGP with one binding-violated moment and several strongly-positive ones.
    n_rep = 120
    alpha = 0.10
    gms_rej = 0
    wol_rej = 0
    for s in range(n_rep):
        rng = np.random.default_rng(7000 + s)
        g = 2.0 + 0.4 * rng.standard_normal((300, 5))  # 4 strongly slack
        g[:, 0] = -0.5 + 0.4 * rng.standard_normal(300)  # 1 violated
        if indicator_gms(g, alpha=alpha, n_mc=1500, seed=s).reject:
            gms_rej += 1
        if wolak_test(g, n_mc=1000, seed=s).pvalue < alpha:
            wol_rej += 1
    gms_power = gms_rej / n_rep
    wol_power = wol_rej / n_rep
    assert gms_power >= wol_power - 0.05


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("statistic", "draws"),
    [
        (float("nan"), [0.0, 1.0]),
        (float("inf"), [0.0, 1.0]),
        (0.0, [0.0, float("nan")]),
        (0.0, [0.0, -float("inf")]),
    ],
)
def test_nonfinite_gms_calibration_is_unavailable(
    statistic: float, draws: list[float]
) -> None:
    """Invalid observed or simulated statistics cannot support a rejection."""
    critical, pvalue, reject = _finite_mc_calibration(
        jnp.asarray(draws), statistic, eta=0.0, alpha=0.05
    )
    assert np.isnan(critical)
    assert np.isnan(pvalue)
    assert not reject


# ---------------------------------------------------------------------------
# Independent numpy witness of the statistics, selection and calibration
# ---------------------------------------------------------------------------


def _witness_hac(g: np.ndarray, bandwidth: int) -> np.ndarray:
    """Hall-centered Bartlett HAC: ``G0 + sum_l (1 - l/(b+1)) (Gl + Gl')``."""
    n_obs = g.shape[0]
    u = g - g.mean(axis=0)
    omega = u.T @ u / n_obs
    for lag in range(1, bandwidth + 1):
        gamma = u[lag:].T @ u[:-lag] / n_obs
        omega = omega + (1.0 - lag / (bandwidth + 1.0)) * (gamma + gamma.T)
    return omega


def _witness_qlr(target: np.ndarray, weight: np.ndarray) -> float:
    """Solve ``min_{t >= 0} (z - t)' W (z - t)`` by enumerating free sets.

    The minimiser has a free set ``S`` (``t_S > 0``) and ``t = 0`` elsewhere;
    on ``S`` stationarity gives ``t_S = z_S + W_SS^-1 W_S,S^c z_S^c``. The
    minimum over the feasible (``t >= 0``) candidates is the QP optimum.
    """
    k = target.size
    best = float(target @ weight @ target)  # t = 0
    for free in product([False, True], repeat=k):
        mask = np.array(free)
        if not mask.any():
            continue
        t = np.zeros(k)
        t[mask] = target[mask] + np.linalg.solve(
            weight[np.ix_(mask, mask)], weight[np.ix_(mask, ~mask)] @ target[~mask]
        )
        if np.all(t >= 0.0):
            diff = target - t
            best = min(best, float(diff @ weight @ diff))
    return best


def _witness_draws(
    cov: np.ndarray, slack: np.ndarray, sig: np.ndarray, *, n_mc: int, seed: int
) -> np.ndarray:
    """Return recentered ``N(0, cov)`` draws from the documented keyed stream."""
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    root = (vectors * np.sqrt(np.maximum(values, 1e-14))) @ vectors.T
    normals = np.asarray(
        jax.random.normal(
            cell_key(seed, "inference", "indicator_gms"), (cov.shape[0], n_mc)
        )
    )
    return np.where(
        slack[None, :], (root @ normals).T + 1e6 * sig.max(), (root @ normals).T
    )


def _witness_gms(
    g: np.ndarray,
    *,
    bandwidth: int,
    statistic: str,
    n_mc: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Statistic, selection, p-value and critical value in numpy.

    The Monte Carlo normals are the documented ``jax.random.normal`` stream for
    ``cell_key(seed, "inference", "indicator_gms")`` with shape ``(k, n_mc)``;
    everything computed from them is numpy.
    """
    n_obs = g.shape[0]
    root_t = np.sqrt(n_obs)
    gbar = g.mean(axis=0)
    cov = _witness_hac(g, bandwidth)
    sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    kappa = np.sqrt(0.3 * np.log(max(n_obs, 3)))
    slack = root_t * gbar / sig / kappa > 1.0
    draws = _witness_draws(cov, slack, sig, n_mc=n_mc, seed=seed)

    if statistic == "mmm":
        stat = float(np.sum(np.minimum(root_t * gbar / sig, 0.0) ** 2))
        sims = np.sum(np.minimum(draws / sig, 0.0) ** 2, axis=1)
    else:
        weight = np.linalg.pinv(cov)
        stat = _witness_qlr(root_t * gbar, weight)
        sims = np.array([_witness_qlr(row, weight) for row in draws])
    allowed = int(np.floor(alpha * (n_mc + 1) - 1.0 + 1e-12))
    crit = np.sort(sims)[n_mc - min(allowed, n_mc - 1) - 1] if allowed >= 0 else np.inf
    return {
        "stat": stat,
        "n_selected": int(np.sum(~slack)),
        "pvalue": (int(np.sum(sims >= stat)) + 1) / (n_mc + 1),
        "crit": float(crit),
    }


def _witness_panel() -> np.ndarray:
    """Return a fixed correlated panel: one slack, one binding, one violated."""
    rng = np.random.default_rng(2026)
    shocks = rng.standard_normal((200, 3))
    mix = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return shocks @ mix + np.array([1.5, 0.05, -0.25])


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("statistic", ["mmm", "qlr"])
def test_gms_matches_independent_numpy_witness(statistic: str) -> None:
    """MMM/QLR statistics, indicator selection and Monte Carlo calibration.

    Definitions followed (``indicator_gms`` docstring): ``z = sqrt(T) gbar /
    sigma`` with ``sigma`` from the Hall-centered Bartlett HAC; MMM is
    ``sum min(z_j, 0)^2``; QLR is ``min_{t>=0} (m - t)' Omega^-1 (m - t)`` with
    ``m = sqrt(T) gbar``; moment ``j`` is slack when ``sqrt(T) gbar_j / (sigma_j
    kappa_n) > 1`` with ``kappa_n = sqrt(0.3 ln T)``; the p-value is the add-one
    ``(1 + b) / (1 + n_mc)`` count of ``sims >= stat`` and the critical value is
    the order statistic consistent with it. The normals are the documented
    keyed ``jax.random.normal`` stream, so calibration is reproduced too.
    Statistics are compared at ``rtol=1e-10``.
    """
    g = _witness_panel()
    n_mc = 400
    with jax.enable_x64(new_val=True):
        result = indicator_gms(
            g, bandwidth=3, n_mc=n_mc, seed=7, statistic=statistic, alpha=0.05
        )
    expected = _witness_gms(g, bandwidth=3, statistic=statistic, n_mc=n_mc, seed=7)

    assert expected["stat"] > 0.0  # the panel genuinely violates a moment
    assert result.n_selected == expected["n_selected"]
    assert result.n_selected < g.shape[1]  # at least one moment is slack
    np.testing.assert_allclose(result.stat, expected["stat"], rtol=1e-10)
    assert result.pvalue == expected["pvalue"]
    np.testing.assert_allclose(result.crit, expected["crit"], rtol=1e-10)
