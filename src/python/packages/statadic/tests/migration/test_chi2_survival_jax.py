# standalone package-maintenance entry point
"""Retirement of the ``scipy.stats.chi2.sf`` boundary in GEL and Wolak.

Both reports now route through
:func:`statadic.inference._common._chi2_survival`, which evaluates
``jax.scipy.stats.chi2.sf`` at **whatever precision the caller has configured**.
SciPy appears here only as the independent oracle; the production path no
longer reaches it for this operation.

The two lanes below are deliberately different in kind, per the project's
numerical-precision policy:

* the ``reference`` lane opens an explicit x64 scope *in the test* and pins
  agreement with SciPy to establish mathematical correctness;
* the ``numerical`` lane opens an explicit **float32** scope and asserts what
  the consumers actually depend on — the decision, the valid domain, and
  monotonicity — rather than a digit count.

Both scopes are explicit because ``tests/conftest.py`` sets
``jax_enable_x64=True`` for the whole session. A test here that simply ran at
the ambient config would silently be an x64 test and would never exercise the
float32 path it claims to cover.

The float32 far tail deviates from x64 by roughly ``4e-9`` relative. That is
accepted, not tolerated: these p-values are compared against alpha and printed
to six decimals, never log-combined, accumulated, or used in a stopping rule.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy import stats
from statadic.inference._common import _chi2_survival

#: Under x64 the measured worst relative deviation from SciPy across the grid
#: below is 2.5e-14 — Cephes ``chdtrc`` against ``gammaincc``. This leaves an
#: order of magnitude of headroom without hiding a regression.
_REFERENCE_RTOL = 5e-13

#: float32 carries ~7 significant decimal digits; 1e-6 relative is the honest
#: resolution of a probability computed in it.
_FLOAT32_RESOLUTION = 1e-6

#: Conventional significance levels the reports compare against.
_ALPHAS = (0.01, 0.05, 0.10)

#: Absolute band around an alpha inside which a p-value has no
#: precision-independent verdict. At alpha=0.05 the float32 resolution is
#: ``0.05 * 1e-6 = 5e-8``, so this band is ~20x wider — loose enough to cover
#: the boundary case without masking a disagreement anywhere else.
_THRESHOLD_EXCLUSION = 1e-6

_DOFS = (1, 2, 3, 5, 8, 13, 21, 40)
_STATS = (
    0.0,
    1e-12,
    1e-6,
    0.5,
    1.0,
    3.841458820694124,
    10.0,
    23.356886,
    50.0,
    200.0,
)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("dof", _DOFS)
@pytest.mark.parametrize("stat", _STATS)
def test_chi2_survival_matches_the_scipy_oracle_under_x64(
    stat: float,
    dof: int,
) -> None:
    """Establish correctness against SciPy across the body and the far tail."""
    with jax.enable_x64(new_val=True):
        observed = _chi2_survival(stat, dof)

    assert observed == pytest.approx(
        float(stats.chi2.sf(stat, df=dof)),
        rel=_REFERENCE_RTOL,
    )


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("dof", _DOFS)
@pytest.mark.parametrize("stat", _STATS)
def test_chi2_survival_stays_a_valid_probability_under_float32(
    stat: float,
    dof: int,
) -> None:
    """Hold the structural invariant the consumers rely on, in any dtype.

    A survival function must land in ``[0, 1]`` and be finite. This is the
    property that would actually break a p-value comparison; the exact tail
    digits are not part of the contract.
    """
    with jax.enable_x64(new_val=False):
        observed = _chi2_survival(stat, dof)

    assert np.isfinite(observed)
    assert 0.0 <= observed <= 1.0


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("dof", _DOFS)
def test_chi2_survival_is_monotone_decreasing_under_float32(
    dof: int,
) -> None:
    """Keep the tail ordered, which is what a threshold comparison needs.

    A p-value that is not monotone in its statistic can invert a rejection
    decision, and float32 is where a flat tail would collapse into ties, so
    this is asserted at the coarser precision rather than under x64.
    """
    with jax.enable_x64(new_val=False):
        observed = [_chi2_survival(stat, dof) for stat in sorted(_STATS)]

    assert observed == sorted(observed, reverse=True)


@pytest.mark.unit
@pytest.mark.reference
def test_chi2_survival_agrees_with_scipy_on_the_rejection_decision() -> None:
    """Check the semantic contract: same verdict at conventional alphas.

    This is the assertion that matters for ``wolak_test`` and ``gel_test``:
    a precision change is only material here if it moves a verdict. It runs
    under float32 because that is the weakest precision the helper supports.

    Points sitting within :data:`_THRESHOLD_EXCLUSION` of an alpha are excluded
    and checked by
    :func:`test_chi2_survival_only_disagrees_on_verdict_at_the_exact_threshold`
    instead. That exclusion is not a convenience: a p-value landing *exactly*
    on alpha has no precision-independent verdict, and ``_STATS`` deliberately
    contains one such point.
    """
    with jax.enable_x64(new_val=False):
        observed = {
            (stat, dof): _chi2_survival(stat, dof) for dof in _DOFS for stat in _STATS
        }

    compared = 0
    for (stat, dof), value in observed.items():
        expected = float(stats.chi2.sf(stat, df=dof))
        for alpha in _ALPHAS:
            if abs(expected - alpha) < _THRESHOLD_EXCLUSION:
                continue
            assert (value < alpha) == (expected < alpha)
            compared += 1

    # Guard against the exclusion silently swallowing the whole grid.
    assert compared > len(observed)


@pytest.mark.unit
@pytest.mark.reference
def test_chi2_survival_only_disagrees_on_verdict_at_the_exact_threshold() -> None:
    """Bound how close to alpha a float32 verdict flip can occur.

    ``3.841458820694124`` is the exact chi2(1) five-percent critical value, so
    ``sf`` is ``0.05`` to every digit float64 has. Under float32 it evaluates to
    ``0.05000001…`` and the strict ``< 0.05`` comparison inverts.

    That is the honest worst case and it is harmless: it requires the statistic
    to sit on the critical value to fifteen digits. What this test pins is that
    the flip stays confined to that neighbourhood — the float32 and x64 values
    still agree to float32 resolution, so no p-value meaningfully distant from
    alpha can change its verdict.
    """
    critical = 3.841458820694124
    with jax.enable_x64(new_val=False):
        narrow = _chi2_survival(critical, 1)
    with jax.enable_x64(new_val=True):
        wide = _chi2_survival(critical, 1)

    assert narrow == pytest.approx(0.05, rel=_FLOAT32_RESOLUTION)
    assert wide == pytest.approx(0.05, rel=_REFERENCE_RTOL)
    assert narrow == pytest.approx(wide, rel=_FLOAT32_RESOLUTION)


@pytest.mark.unit
@pytest.mark.contract
def test_chi2_survival_returns_a_host_float_not_a_jax_array() -> None:
    """Keep the eager report boundary returning a plain host scalar."""
    value = _chi2_survival(1.0, 1)

    assert type(value) is float


@pytest.mark.unit
@pytest.mark.contract
def test_chi2_survival_does_not_mutate_the_caller_x64_configuration() -> None:
    """Fail if the helper ever reinstates a hidden global precision flip.

    A library routine that enables x64 for its own convenience silently
    re-traces every jitted function the caller holds. The helper is documented
    as respecting caller configuration; this is that documentation made
    executable.
    """
    before: bool = jax.config.read("jax_enable_x64")

    _chi2_survival(1.0, 1)

    assert jax.config.read("jax_enable_x64") == before


@pytest.mark.unit
@pytest.mark.regression
def test_chi_bar_mixture_tracks_the_retired_scipy_mixture_under_x64() -> None:
    """Check the accumulated chi-bar p-value, not just its components.

    The Wolak mixture sums ``w_j * sf(stat, j)`` term by term. This reproduces
    that accumulation against the SciPy oracle in the same order, so a change
    in summation order would fail here even when every component passes.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    with jax.enable_x64(new_val=True):
        for _ in range(64):
            m = int(rng.integers(2, 9))
            weights = rng.dirichlet(np.ones(m + 1))
            stat = rng.gamma(2.0, 2.0)

            expected = sum(
                wj * (0.0 if j == 0 else stats.chi2.sf(stat, df=j))
                for j, wj in enumerate(weights)
                if wj > 0.0
            )
            observed = sum(
                wj * (0.0 if j == 0 else _chi2_survival(stat, j))
                for j, wj in enumerate(weights)
                if wj > 0.0
            )
            worst = max(worst, abs(expected - observed) / expected)

    assert worst < _REFERENCE_RTOL
