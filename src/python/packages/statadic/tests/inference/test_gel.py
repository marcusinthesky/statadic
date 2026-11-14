# standalone package-maintenance entry point
"""Tests for :mod:`statadic.inference.gel` — GEL over-identification and profiling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import (
    gel_test,
)
from statadic.inference.gmm import (
    gmm_wald,
)

from .._typing import ill_typed

if TYPE_CHECKING:
    from statadic.estimation.gel.fit import FixedGelFit

from .._budgets import examples
from ._moments import (
    EXPECTED_COLLINEAR_DOF,
    EXPECTED_NONCONVERGED_DOF,
    GEL_TRUST_REGION_CAP,
    MINIMUM_POWER,
    NEAR_ZERO_STATISTIC,
    SIGNIFICANCE_LEVEL,
    THREE_MOMENT_DOF,
    full_rank_panels,
    rank_deficient_profile_panel,
)

# ---------------------------------------------------------------------------
# gel_test (GEL over-identification)
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_gel_test_h0_and_h1(criterion: Literal["cue", "el", "et"]) -> None:
    """GEL separates centered null moments from shifted alternatives."""
    rng = np.random.default_rng(10)
    g0 = rng.standard_normal((800, 3))
    res0 = gel_test(
        FixedGEL(criterion=criterion).estimate(FixedMomentProblem(g0)),
        kind="lr",
        assume_iid=True,
    )
    assert res0.dof == THREE_MOMENT_DOF
    assert res0.converged
    assert 0.0 <= res0.pvalue <= 1.0

    g1 = rng.standard_normal((800, 3)) + 0.8
    res1 = gel_test(
        FixedGEL(criterion=criterion).estimate(FixedMomentProblem(g1)),
        kind="lr",
        assume_iid=True,
    )
    assert res1.pvalue < SIGNIFICANCE_LEVEL


@pytest.mark.component
@pytest.mark.contract
def test_gel_requires_an_actual_fit() -> None:
    """Pre-formed moments cannot bypass fitting or its identification checks."""
    rng = np.random.default_rng(11)
    g = rng.standard_normal((500, 4))
    with pytest.raises(TypeError, match="FixedGelFit"):
        gel_test(cast("FixedGelFit", g))


@pytest.mark.component
@pytest.mark.property
def test_gel_wald_uses_the_same_reduced_row_space_as_its_dof() -> None:
    """Near-null directions cannot re-enter the quadratic after rank selection."""
    rng = np.random.default_rng(115)
    base = rng.standard_normal(500)
    moments = np.column_stack(
        [
            base,
            base + 1e-4 * rng.standard_normal(500) + 1e-4,
        ]
    )
    fit = FixedGEL(criterion="cue").estimate(FixedMomentProblem(moments))
    lr = gel_test(fit, kind="lr", assume_iid=True)
    wald = gel_test(fit, kind="wald", assume_iid=True)
    lm = gel_test(fit, kind="lm", assume_iid=True)
    assert lr.dof == wald.dof == lm.dof == 1
    assert np.isclose(wald.stat, lr.stat, rtol=1e-6, atol=1e-8)
    assert np.isclose(lm.stat, lr.stat, rtol=1e-6, atol=1e-8)


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_linear_gel_profiles_affine_parameter(
    criterion: Literal["cue", "el", "et"],
) -> None:
    """Affine GEL estimates theta inside each carrier before subtracting dof."""
    rng = np.random.default_rng(111)
    noise = 0.2 * rng.standard_normal((600, 4))
    noise -= noise.mean(axis=0, keepdims=True)
    derivative = np.array([[1.0], [0.6], [-0.4], [0.8]])
    theta_true = 0.7
    g0 = noise - theta_true * derivative[:, 0]

    res = gel_test(
        AffineGEL(criterion=criterion).estimate(AffineMomentProblem(g0, derivative)),
        assume_iid=True,
    )

    assert res.converged
    assert res.inner_converged
    assert res.outer_converged
    assert res.reference_valid
    assert res.dof == THREE_MOMENT_DOF
    assert np.isclose(res.params[0], theta_true, atol=1e-5)
    assert res.stat < NEAR_ZERO_STATISTIC


@pytest.mark.component
@pytest.mark.contract
def test_linear_gel_fails_closed_on_unidentified_parameter() -> None:
    """A rank-deficient mean Jacobian cannot license profiled GEL inference."""
    rng = np.random.default_rng(112)
    g0 = rng.standard_normal((300, 3))
    with pytest.raises(ValueError, match="does not identify"):
        gel_test(AffineGEL().estimate(AffineMomentProblem(g0, np.zeros((3, 1)))))


@pytest.mark.component
@pytest.mark.regression
def test_frozen_gel_coordinates_cannot_annihilate_the_fitted_parameter() -> None:
    """Full original derivative rank does not identify a frozen flat profile."""
    v = np.arange(1.0, 5.0)
    moments = np.column_stack((v, [-1.0, 1.0, -1.0, 1.0], np.zeros(4)))
    derivative = np.stack((-v, np.zeros(4), v), axis=1)[:, :, None]
    assert np.linalg.matrix_rank(derivative.mean(axis=0)) == 1
    # At theta=.5 the first and third columns coincide. The retained plane's
    # orthogonal complement contains (-1,0,1), the entire parameter direction.
    fitted = moments + derivative[:, :, 0] * 0.5
    retained_rank = 2
    assert np.linalg.matrix_rank(fitted) == retained_rank
    np.testing.assert_allclose(fitted @ np.array([-1.0, 0.0, 1.0]), 0)
    with pytest.raises(ValueError, match="frozen GEL moment coordinates"):
        AffineGEL().estimate(AffineMomentProblem(moments, derivative))


@pytest.mark.component
@pytest.mark.contract
def test_linear_gel_fails_closed_when_outer_profile_does_not_converge() -> None:
    """Outer nonconvergence withholds the statistic and p-value."""
    rng = np.random.default_rng(113)
    g0 = rng.standard_normal((300, 3)) - 0.4
    derivative = np.ones((3, 1))
    res = gel_test(
        AffineGEL(outer_max_iter=1).estimate(AffineMomentProblem(g0, derivative))
    )
    assert not res.outer_converged
    assert not res.converged
    assert np.isnan(res.stat)
    assert np.isnan(res.pvalue)


@pytest.mark.component
@pytest.mark.contract
def test_linear_gel_rejects_unimplemented_profiled_wald() -> None:
    """Profiled LM/Wald forms fail closed without their sandwich estimator."""
    rng = np.random.default_rng(114)
    with pytest.raises(ValueError, match="only kind='lr'"):
        ill_typed(gel_test)(
            AffineGEL().estimate(
                AffineMomentProblem(rng.standard_normal((300, 3)), np.ones((3, 1)))
            ),
            kind="wald",
        )


@pytest.mark.component
@pytest.mark.contract
def test_gel_withholds_iid_reference_without_explicit_acknowledgement() -> None:
    """Observation-level GEL exposes no p-value under unspecified dependence."""
    rng = np.random.default_rng(115)
    fixed = gel_test(
        FixedGEL().estimate(FixedMomentProblem(rng.standard_normal((300, 3))))
    )
    assert fixed.converged
    assert not fixed.reference_valid
    assert np.isfinite(fixed.stat)
    assert np.isnan(fixed.pvalue)

    g0 = rng.standard_normal((300, 3)) - 0.5
    profiled = gel_test(AffineGEL().estimate(AffineMomentProblem(g0, np.ones((3, 1)))))
    assert profiled.converged
    assert not profiled.reference_valid
    assert np.isfinite(profiled.stat)
    assert np.isnan(profiled.pvalue)


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(g=np.random.default_rng(12).standard_normal((600, 3)))
@given(g=full_rank_panels(200, 3))
def test_gel_cue_wald_matches_gmm_wald_iid(g: np.ndarray) -> None:
    """CUE-GEL Wald agrees with the IID GMM Wald reference.

    Args:
        g: Generated full-rank moment panel.

    """
    # For iid moments the CUE Wald form coincides with plain Newey-West GMM
    # Wald at bandwidth zero.
    gel = gel_test(
        FixedGEL(criterion="cue").estimate(FixedMomentProblem(g)), kind="wald"
    )
    wald = gmm_wald(g, bandwidth=0)
    assert np.isclose(gel.stat, wald.stat, rtol=1e-6)


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(g=np.random.default_rng(13).standard_normal((500, 4)))
@given(g=full_rank_panels(200, 4))
def test_gel_lm_equals_wald(g: np.ndarray) -> None:
    """GEL LM and Wald statistics coincide in the quadratic reference case.

    Args:
        g: Generated full-rank moment panel.

    """
    # With pre-formed moments and no profiled structural parameter the GEL LM
    # and Wald forms coincide exactly (both are n_observations*gbar' Omega^-1 gbar).
    fit = FixedGEL(criterion="cue").estimate(FixedMomentProblem(g))
    lm = gel_test(fit, kind="lm").stat
    wald = gel_test(fit, kind="wald").stat
    assert np.isclose(lm, wald, rtol=1e-12)


@pytest.mark.component
@pytest.mark.property
def test_gel_near_singular_high_dim_no_divergence() -> None:
    """GEL remains finite for a near-singular high-dimensional moment system."""
    # Regression (auditor HIGH finding): at the empirical dimension (m~15) with a
    # near-collinear moment second moment, the inner supremum must not run
    # to infinity with a spurious converged=True. CUE (quadratic) converges to a
    # finite statistic on the identified row-space; EL/ET may hit the trust-region
    # cap under gross misspecification and MUST then report converged=False (the
    # honest signal), never a garbage LR with converged=True.
    rng = np.random.default_rng(30)
    n_observations, m = 600, 15
    base = rng.standard_normal(
        (n_observations, 3)
    )  # rank-3 signal -> Omega near-singular
    mix = rng.standard_normal((3, m))
    g = (
        base @ mix + 1e-3 * rng.standard_normal((n_observations, m)) + 0.5
    )  # misspecified mean

    cue = gel_test(FixedGEL(criterion="cue").estimate(FixedMomentProblem(g)), kind="lr")
    assert cue.converged
    assert np.isfinite(cue.stat)
    assert np.linalg.norm(cue.lam) < GEL_TRUST_REGION_CAP
    assert cue.dof == EXPECTED_COLLINEAR_DOF

    for crit in ("el", "et"):
        res = gel_test(
            FixedGEL(criterion=crit).estimate(FixedMomentProblem(g)), kind="lr"
        )
        # Either converges to a finite stat, or honestly flags divergence.
        if not res.converged:
            assert not res.reference_valid
            assert np.isnan(res.stat)
            assert np.isnan(res.pvalue)
        else:
            assert np.isfinite(res.stat)


@pytest.mark.component
@pytest.mark.contract
def test_gel_nonconvergence_stat_is_nan() -> None:
    """A nonconverged GEL supremum must expose NaN statistics.

    When the inner supremum does not converge (λ pinned at the
    trust-region cap on an ill-posed moment set), `stat` and `pvalue` MUST be
    NaN — never a garbage value from the diverged sup — while `converged=False`
    stays honest.  Construct a grossly misspecified near-collinear EL problem
    that pins λ at the cap.
    """
    rng = np.random.default_rng(31)
    n_observations, m = 400, 15
    base = rng.standard_normal(
        (n_observations, 2)
    )  # rank-2 signal -> Omega near-singular
    mix = rng.standard_normal((2, m))
    # Large mean misspecification drives the EL sup toward the cap.
    g = base @ mix + 1e-4 * rng.standard_normal((n_observations, m)) + 3.0

    found_nonconverged = False
    for crit in ("el", "et"):
        res = gel_test(
            FixedGEL(criterion=crit).estimate(FixedMomentProblem(g)), kind="lr"
        )
        if not res.converged:
            found_nonconverged = True
            assert np.isnan(res.stat), f"{crit}: stat must be NaN on non-conv"
            assert np.isnan(res.pvalue), f"{crit}: pvalue must be NaN on non-conv"
            assert res.dof == EXPECTED_NONCONVERGED_DOF
            assert not res.reference_valid
    assert found_nonconverged, "fixture failed to trigger a non-converged sup"


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(g=np.random.default_rng(32).standard_normal((500, 3)))
@given(g=full_rank_panels(200, 3))
def test_gel_converged_stat_is_finite(g: np.ndarray) -> None:
    """The complement: when converged=True the statistic is finite and usable.

    Args:
        g: Generated full-rank moment panel.

    """
    res = gel_test(
        FixedGEL(criterion="cue").estimate(FixedMomentProblem(g)),
        kind="lr",
        assume_iid=True,
    )
    assert res.converged
    assert np.isfinite(res.stat)
    assert np.isfinite(res.pvalue)


@pytest.mark.component
@pytest.mark.contract
@examples(10)
@example(g=np.random.default_rng(116).standard_normal((30, 2)), max_iter=0)
@given(g=full_rank_panels(30, 2), max_iter=st.integers(max_value=0))
def test_gel_validates_solver_and_reference_controls(
    g: np.ndarray, max_iter: int
) -> None:
    """Invalid iteration or iid-acknowledgement controls fail at the boundary.

    Args:
        g: Generated full-rank moment panel.
        max_iter: Any non-positive iteration cap.

    """
    with pytest.raises(ValueError, match="positive integer"):
        gel_test(FixedGEL(max_iter=max_iter).estimate(FixedMomentProblem(g)))
    with pytest.raises(TypeError, match="bool"):
        gel_test(FixedGEL().estimate(FixedMomentProblem(g)), assume_iid=cast("bool", 1))


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.parametrize("criterion", ["cue", "el", "et"])
def test_gel_size_at_nominal(criterion: Literal["cue", "el", "et"]) -> None:
    """Retain the legacy bounded size screen without silently dropping failed fits."""
    # Size ~ nominal at alpha=0.05 within +/-3 binomial SE (iid moments).
    n_rep = 300
    alpha = 0.05
    rej = 0
    for s in range(n_rep):
        rng = np.random.default_rng(3000 + s)
        g = rng.standard_normal((400, 3))
        result = gel_test(
            FixedGEL(criterion=criterion).estimate(FixedMomentProblem(g)),
            kind="lr",
            assume_iid=True,
        )
        assert result.converged
        assert result.reference_valid
        assert np.isfinite(result.pvalue)
        if result.pvalue < alpha:
            rej += 1
    rate = rej / n_rep
    se = np.sqrt(alpha * (1 - alpha) / n_rep)
    assert abs(rate - alpha) <= 3 * se + 0.02


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_gel_selected_alternative_power_exceeds_null_with_paired_uncertainty() -> None:
    """Compare selected null/alternative rejection using paired simulation error."""
    n_rep = 150
    alpha = 0.05
    # Small samples and shifts keep the curve away from its zero/one limits.
    shifts = [0.0, 0.16]
    powers = []
    rejection_draws = []
    for shift in shifts:
        rej = 0
        decisions = []
        for s in range(n_rep):
            rng = np.random.default_rng(4000 + s)
            g = rng.standard_normal((120, 3)) + shift
            result = gel_test(
                FixedGEL(criterion="cue").estimate(FixedMomentProblem(g)),
                kind="lr",
                assume_iid=True,
            )
            assert result.converged
            assert result.reference_valid
            assert np.isfinite(result.pvalue)
            decisions.append(result.pvalue < alpha)
            if result.pvalue < alpha:
                rej += 1
        powers.append(rej / n_rep)
        rejection_draws.append(np.asarray(decisions, dtype=float))
    difference = rejection_draws[-1] - rejection_draws[0]
    paired_error = difference.std(ddof=1) / np.sqrt(n_rep)
    assert difference.mean() > 3 * paired_error
    assert powers[-1] > MINIMUM_POWER


# ---------------------------------------------------------------------------
# Frozen identified subspace, envelope gradient, closed-form CUE (t65.4)
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
def test_linear_gel_fails_closed_when_identified_rank_moves() -> None:
    """A rank that moves along the profile path licenses no chi-square test."""
    panel, derivative = rank_deficient_profile_panel()
    res = gel_test(
        AffineGEL(criterion="cue").estimate(AffineMomentProblem(panel, derivative)),
        assume_iid=True,
    )
    if not res.rank_stable:
        assert not res.converged
        assert np.isnan(res.stat)
        assert np.isnan(res.pvalue)
    else:  # pragma: no cover - the stable branch must still be self-consistent
        assert np.isfinite(res.stat)


@pytest.mark.component
@pytest.mark.numerical
def test_gel_cue_inner_sup_is_the_closed_form_quadratic() -> None:
    """CUE needs no iteration: its carrier makes the inner sup exactly quadratic.

    ``rho(v) = -v - v^2/2`` gives ``P(lambda) = -gbar'lambda - lambda'Omega
    lambda/2``, so ``lambda* = -Omega^+ gbar`` and ``2T*P(lambda*) = T gbar'
    Omega^+ gbar`` -- identically the Wald form on the same row space.
    """
    rng = np.random.default_rng(41)
    for seed in range(3):
        factors = rng.standard_normal((800, 4))
        panel = factors @ rng.standard_normal((4, 9)) + 1e-4 * rng.standard_normal(
            (800, 9)
        )
        del seed
        fit = FixedGEL(criterion="cue").estimate(FixedMomentProblem(panel))
        likelihood_ratio = gel_test(fit, kind="lr", assume_iid=True)
        wald = gel_test(fit, kind="wald", assume_iid=True)
        assert likelihood_ratio.converged
        np.testing.assert_allclose(
            likelihood_ratio.stat, wald.stat, rtol=1e-10, atol=0.0
        )
