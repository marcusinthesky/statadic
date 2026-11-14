# standalone package-maintenance entry point
"""t46.8 parity gate: the surviving bodies vs the ones the migration retired.

Two distinct gates, deliberately separated -- conflating them is what made the
block-length case look like an impossible 1e-12 assertion:

**Move-parity.**  Code that moved path without changing.  Asserted with ``==``
(bit-identical), not a tolerance, because nothing about the arithmetic changed.

**Collapse-parity.**  Two bodies became one. Only one such collapse exists:
``simulation.hac.sharpe_ratio`` -> ``statadic.inference.sharpe.sharpe_ratio``.  The
retired body is pinned *verbatim* below rather than re-imported, so this file
keeps testing the pre-move code after t46.10 deletes the shims.

**Sharpe JAX-migration parity.** t48.4 moved the surviving Sharpe arithmetic to
x64 JAX. Generic inputs agree with the retired NumPy/SciPy path within 5e-13
relative / 5e-15 absolute (measured gaps are one ulp). One deliberate
degenerate correction remains: NumPy reports a residual standard deviation for
some exactly constant decimal arrays and produced a spurious ~5.13e15 Sharpe;
JAX returns the documented zero-dispersion value 0.0. The explicit
``0.3 +/- 1e-12`` near-constant stress fixture remains nonzero; division by
that near-zero deviation amplifies its measured reduction gap to ``1.54e-9``
relative, so it has a separate ``2e-9`` envelope.

**JAX-migration parity.**  t48.2 moved HAC arithmetic to x64 JAX/XLA. XLA may
fuse scalar division and order matrix-multiply reductions differently from
NumPy/BLAS, so that gate is numerical rather than bitwise: ``rtol=atol=1e-12``.
The seeded fixture's measured maximum absolute gap is below ``3e-15``.

**No Politis-White collapse was possible.**  ``simulation.hac`` never held a
second Politis-White implementation: ``heuristic_block_length`` is the
``T^{1/3}(1 + 2|rho_1|)`` first-lag rule and ``politis_white_block_length`` was
a pure delegation to it.  ``test_the_heuristic_is_not_politis_white`` records
the measured spread that makes the two non-interchangeable, so a later agent
cannot "finish the collapse" by pointing one at the other.

x64 note: ``optimal_block_length_numpy`` requests ``dtype=jnp.float64``, which
is a silent no-op unless ``jax_enable_x64`` is set.  The block-length
assertions here are relative and hold in both states; the parity assertions
that need exactness are pure NumPy float64 and are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest
from statadic.estimation.sharpe import sharpe_ratio, sharpe_ratio_axis
from statadic.inference.resampling.block_length import (
    heuristic_block_length,
    optimal_block_length_numpy,
)
from statadic.inference.sharpe import multiple_sharpe_test, sharpe_difference_test
from statadic.operators.longrun import (
    bartlett_weights,
    circular_block_bootstrap_indices,
    hall_centered_hac,
    newey_west,
    optimal_bandwidth,
)

_HEURISTIC_VS_PW_MIN_GAP = 0.03
_BOOTSTRAP_N_OBS = 50
_SHARPE_RTOL = 5e-13
_SHARPE_ATOL = 5e-15
_SHARPE_NEAR_CONSTANT_RTOL = 2e-9

# RUF069 is suppressed per-assertion below, never file-wide. Exact float equality
# is the *point* of a move-parity gate: the arithmetic did not change, so any
# difference at all -- not merely one past a tolerance -- means the migration
# altered a published number. A tolerance here would hide exactly what this file
# exists to catch.


def _ar1(n_observations: int, phi: float, seed: int) -> np.ndarray:
    """Seeded AR(1) fixture matching the pre-move ``tests/test_hac.py`` helper.

    Args:
        n_observations: Series length.
        phi: Autoregressive coefficient.
        seed: RNG seed.

    Returns:
        The simulated series, shape ``(T,)``.

    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n_observations)
    x = np.empty(n_observations)
    x[0] = e[0]
    for t in range(1, n_observations):
        x[t] = phi * x[t - 1] + e[t]
    return x


# ---------------------------------------------------------------------------
# The retired bodies, pinned verbatim from the pre-move sources.
# ---------------------------------------------------------------------------


def _retired_sharpe_ratio(x: np.ndarray) -> float:
    """``simulation/hac.py:228::sharpe_ratio`` as it stood before t46.8.

    Args:
        x: Return series.

    Returns:
        Sample Sharpe ratio.

    """
    x = np.asarray(x, dtype=np.float64).ravel()
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else 0.0


def _retired_boot_sharpe_diff(a_boot: np.ndarray, b_boot: np.ndarray) -> np.ndarray:
    """``simulation/hac.py:283-289`` inline bootstrap Sharpe difference.

    Args:
        a_boot: Resampled A panel, shape ``(B, T)``.
        b_boot: Resampled B panel, shape ``(B, T)``.

    Returns:
        Per-resample Sharpe difference, shape ``(B,)``.

    """
    mean_a = a_boot.mean(axis=1)
    mean_b = b_boot.mean(axis=1)
    sd_a = a_boot.std(axis=1, ddof=1)
    sd_b = b_boot.std(axis=1, ddof=1)
    sr_a = np.where(sd_a > 0, mean_a / sd_a, 0.0)
    sr_b = np.where(sd_b > 0, mean_b / sd_b, 0.0)
    return sr_a - sr_b


def _retired_paired_sharpe_diffs(block: np.ndarray, bench: np.ndarray) -> np.ndarray:
    """``simulation/sharpe_tests.py:313::_paired_sharpe_diffs`` before t46.8.

    Args:
        block: Strategy panel, shape ``(T, k)``.
        bench: Benchmark series, shape ``(T,)``.

    Returns:
        Per-strategy Sharpe difference, shape ``(k,)``.

    """
    mean = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    sr = np.where(sd > 0.0, mean / sd, 0.0)
    bsd = float(bench.std(ddof=1))
    bsr = float(bench.mean() / bsd) if bsd > 0.0 else 0.0
    return sr - bsr


# ---------------------------------------------------------------------------
# Collapse-parity: the one real duplication.
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("n_observations", [2, 8, 32])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 3.0])
def test_sharpe_collapse_matches_numpy_on_seeded_series(
    n_observations: int, scale: float
) -> None:
    """Match NumPy across short series and scales without compiling random shapes."""
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(n_observations) * scale + 0.3
    assert sharpe_ratio(returns) == pytest.approx(
        _retired_sharpe_ratio(returns),
        rel=_SHARPE_RTOL,
        abs=_SHARPE_ATOL,
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "series",
    [
        np.full(10, 0.3),  # zero dispersion
        np.array([0.4]),  # n = 1: retired body warned, both return 0.0
        np.array([0.1, 0.1]),  # n = 2, zero dispersion
        np.array([-1.0, 1.0]),  # zero mean
    ],
)
def test_sharpe_zero_dispersion_contract(series: np.ndarray) -> None:
    """Exactly constant or zero-mean series return the documented zero."""
    assert sharpe_ratio(series) == 0.0  # exact public contract  # noqa: RUF069


@pytest.mark.component
@pytest.mark.property
def test_sharpe_near_constant_series_remains_nonzero() -> None:
    """A genuinely tiny dispersion is not mistaken for an exact constant."""
    perturbation = np.tile(np.array([-1.0, 1.0]), 5) * 1e-12
    series = 0.3 + perturbation
    expected = _retired_sharpe_ratio(series)
    assert expected > 0.0
    assert sharpe_ratio(series) == pytest.approx(
        expected,
        rel=_SHARPE_NEAR_CONSTANT_RTOL,
        abs=_SHARPE_ATOL,
    )


@pytest.mark.component
@pytest.mark.regression
def test_vectorised_sharpe_matches_the_two_inline_copies() -> None:
    """The compiled axis reduction matches both retired NumPy copies."""
    rng = np.random.default_rng(11)
    boot = rng.standard_normal((64, 120))
    other = rng.standard_normal((64, 120))
    np.testing.assert_allclose(
        sharpe_ratio_axis(boot, axis=1) - sharpe_ratio_axis(other, axis=1),
        _retired_boot_sharpe_diff(boot, other),
        rtol=_SHARPE_RTOL,
        atol=_SHARPE_ATOL,
    )

    panel = rng.standard_normal((150, 4))
    bench = rng.standard_normal(150)
    np.testing.assert_allclose(
        sharpe_ratio_axis(panel, axis=0) - sharpe_ratio(bench),
        _retired_paired_sharpe_diffs(panel, bench),
        rtol=_SHARPE_RTOL,
        atol=_SHARPE_ATOL,
    )


# ---------------------------------------------------------------------------
# Move-parity: the HAC core and the block-length selectors.
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.regression
def test_hac_core_matches_the_retired_numpy_arithmetic() -> None:
    """The x64 JAX HAC stays within the stated XLA reduction-order margin."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((400, 3)) + 0.7

    b = optimal_bandwidth(g.shape[0])
    assert b == int(np.floor(4.0 * (400 / 100.0) ** (2.0 / 9.0)))
    np.testing.assert_allclose(
        bartlett_weights(b),
        1.0 - np.arange(b + 1) / (b + 1.0),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )

    for center in (False, True):
        u = g - g.mean(axis=0, keepdims=True) if center else g
        w = bartlett_weights(b)
        expected = (u.T @ u) / 400
        for k in range(1, b + 1):
            gk = (u[k:].T @ u[:-k]) / 400
            expected = expected + w[k] * (gk + gk.T)
        expected = 0.5 * (expected + expected.T)
        got = hall_centered_hac(g) if center else newey_west(g)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_circular_block_bootstrap_indices_are_seed_stable() -> None:
    """The resampler is unchanged: same seed, same index array, same shape."""
    idx = circular_block_bootstrap_indices(_BOOTSTRAP_N_OBS, 5, 20, seed=7)
    assert idx.shape == (20, _BOOTSTRAP_N_OBS)
    np.testing.assert_array_equal(
        idx, circular_block_bootstrap_indices(_BOOTSTRAP_N_OBS, 5, 20, seed=7)
    )
    assert idx.min() >= 0
    assert idx.max() < _BOOTSTRAP_N_OBS


@pytest.mark.component
@pytest.mark.property
def test_sharpe_difference_test_is_reproducible_through_the_collapse() -> None:
    """The paired test is deterministic and unchanged by the Sharpe collapse."""
    a = _ar1(400, phi=0.3, seed=1) * 0.01 + 0.0005
    b = _ar1(400, phi=0.3, seed=2) * 0.01 + 0.0005
    stat, pvalue = sharpe_difference_test(a, b, n_boot=200, seed=42)

    # Reconstruct the pre-move computation with the retired bodies.
    ell = max(1, min(heuristic_block_length(a - b), a.size))
    idx = circular_block_bootstrap_indices(a.size, ell, 200, seed=42)
    delta_boot = _retired_boot_sharpe_diff(a[idx], b[idx])
    delta_hat = _retired_sharpe_ratio(a) - _retired_sharpe_ratio(b)
    se = float(delta_boot.std(ddof=1))
    centered = delta_boot - delta_boot.mean()

    assert stat == pytest.approx(
        float(delta_hat / se),
        rel=_SHARPE_RTOL,
        abs=_SHARPE_ATOL,
    )
    assert pvalue == pytest.approx(
        float(np.mean(np.abs(centered) >= np.abs(delta_hat))),
        abs=_SHARPE_ATOL,
    )


@pytest.mark.component
@pytest.mark.regression
def test_multiple_sharpe_test_matches_the_retired_diff_helper() -> None:
    """The observed differences stay within the x64 reduction-order margin."""
    rng = np.random.default_rng(5)
    panel = rng.standard_normal((200, 3)) * 0.01
    bench = rng.standard_normal(200) * 0.01
    result = multiple_sharpe_test(panel, bench, n_boot=100, block_length=4, seed=1)
    np.testing.assert_allclose(
        result.sharpe_diff,
        _retired_paired_sharpe_diffs(panel, bench),
        rtol=_SHARPE_RTOL,
        atol=_SHARPE_ATOL,
    )


# ---------------------------------------------------------------------------
# The collapse that could NOT happen, and why.
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
def test_the_heuristic_is_not_politis_white() -> None:
    """The two block-length selectors are different estimators, not duplicates.

    ``simulation.hac`` never held a second Politis-White implementation. Pinning
    the measured spread stops a later agent from "finishing" a collapse that was
    never available: routing ``heuristic_block_length`` at
    ``optimal_block_length`` would silently move every published
    Sharpe-difference p-value.
    """
    fixtures = [
        _ar1(500, phi=0.8, seed=3),
        np.random.default_rng(4).standard_normal(500),
        _ar1(200, phi=0.5, seed=5),
        _ar1(400, phi=0.3, seed=1),
        _ar1(300, phi=0.7, seed=0),
    ]
    gaps = []
    for x in fixtures:
        pw = optimal_block_length_numpy(x, scheme="circular")
        gaps.append(abs(heuristic_block_length(x) - pw) / pw)
    assert min(gaps) > _HEURISTIC_VS_PW_MIN_GAP


@pytest.mark.integration
@pytest.mark.contract
def test_politis_white_adapter_delegates_rather_than_reimplements() -> None:
    """``optimal_block_length_numpy`` is an adapter, so there is one PW body."""
    import jax.numpy as jnp  # noqa: PLC0415
    from statadic.inference.resampling.block_length import (  # noqa: PLC0415
        optimal_block_length,
    )

    x = _ar1(300, phi=0.6, seed=9)
    for scheme in ("stationary", "circular"):
        assert optimal_block_length_numpy(x, scheme=scheme) == optimal_block_length(
            jnp.asarray(x), scheme=scheme
        )
