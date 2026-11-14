# standalone package-maintenance entry point
"""Tests for :mod:`statadic.inference.gmm` — GMM Wald and Hansen J."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.equations import PlainLongRun
from statadic.estimation.gmm import IteratedGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gmm import (
    gmm_wald,
    gmm_wald_kernel,
    hansen_j,
    hansen_j_from_fit,
    hansen_j_kernel,
)

from .._budgets import examples
from ._moments import (
    MAX_EMPIRICAL_SIZE,
    SIGNIFICANCE_LEVEL,
    SOFT_MAX_EMPIRICAL_SIZE,
    THREE_MOMENT_DOF,
    TWO_MOMENT_DOF,
    ar1_panel,
    full_rank_panels,
)

if TYPE_CHECKING:
    from statadic.inference.gmm.hypothesis import (
        HansenJKernelResult,
        WaldKernelResult,
    )

# ---------------------------------------------------------------------------
# gmm_wald
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@given(g0=full_rank_panels(200, 3))
def test_gmm_wald_reports_dof_and_a_valid_pvalue(g0: np.ndarray) -> None:
    """The statistic carries m degrees of freedom and a probability p-value.

    Args:
        g0: Generated full-rank moment panel.

    """
    res0 = gmm_wald(g0)
    assert res0.dof == THREE_MOMENT_DOF
    assert 0.0 <= res0.pvalue <= 1.0


@pytest.mark.component
@pytest.mark.property
def test_gmm_wald_h0_and_h1() -> None:
    """GMM Wald distinguishes centered null moments from shifted alternatives."""
    rng = np.random.default_rng(0)
    g0 = rng.standard_normal((1000, 3))
    res0 = gmm_wald(g0)
    assert res0.dof == THREE_MOMENT_DOF
    assert 0.0 <= res0.pvalue <= 1.0

    g1 = rng.standard_normal((1000, 3)) + 1.0
    res1 = gmm_wald(g1)
    assert res1.pvalue < SIGNIFICANCE_LEVEL


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(invalid=np.array([1.0, 2.0]))
@example(invalid=np.array([[1.0], [np.nan]]))
@example(invalid=np.array([[1.0, 2.0]]))
@given(
    invalid=st.one_of(
        hnp.arrays(np.float64, st.integers(1, 6), elements=st.floats(-5, 5)),
        hnp.arrays(
            np.float64,
            (6, 2),
            elements=st.floats(-5, 5),
        ).map(lambda a: np.where(np.eye(6, 2, dtype=bool), np.nan, a)),
        hnp.arrays(np.float64, (1, 3), elements=st.floats(-5, 5)),
    )
)
def test_gmm_wald_rejects_invalid_moment_panels(invalid: np.ndarray) -> None:
    """GMM Wald validates dimensionality, finiteness, and observation count.

    The ndim alternative reads ``must have shape`` rather than the former
    ``two-dimensional``: t65.5 folded ``_validate_hansen_dimensions``'s panel
    half onto the shared moment-panel validator, which was already the wording
    the ``gms``/``kappa``/``wolak`` doors raised for the identical condition.
    One condition now has one message.

    Args:
        invalid: A vector, a panel with a NaN, or a single-observation panel.

    """
    with pytest.raises(ValueError, match=r"must have shape|requires|finite"):
        gmm_wald(invalid)


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(x=np.random.default_rng(35).standard_normal(200))
@given(x=hnp.arrays(np.float64, 200, elements=st.floats(-5, 5)))
def test_gmm_wald_rejects_rank_deficient_hac(x: np.ndarray) -> None:
    """A pseudoinverse cannot justify m degrees of freedom after rank loss.

    Args:
        x: Generated series duplicated into two identical moment columns.

    """
    duplicate_moments = np.column_stack([x, x])
    with pytest.raises(ValueError, match="full-rank moment covariance"):
        gmm_wald(duplicate_moments)


@pytest.mark.integration
@pytest.mark.contract
@examples(10)
@example(panel=np.random.default_rng(36).standard_normal((128, 3)))
@given(panel=full_rank_panels(128, 3))
def test_gmm_wald_kernel_is_traceable_and_matches_report(panel: np.ndarray) -> None:
    """The fixed-structure Wald core composes under JAX transforms.

    Args:
        panel: Generated full-rank moment panel.

    """
    moments = jnp.asarray(panel)
    eager = gmm_wald(np.asarray(moments), bandwidth=0)
    kernel = gmm_wald_kernel(moments, 0)

    def evaluate_gmm_wald_kernel(panel: jax.Array) -> WaldKernelResult:
        return gmm_wald_kernel(panel, 0)

    batched = jax.vmap(evaluate_gmm_wald_kernel)(jnp.stack((moments, moments + 0.1)))

    jax.make_jaxpr(evaluate_gmm_wald_kernel)(moments)
    assert np.isclose(kernel.stat, eager.stat)
    assert np.isclose(kernel.pvalue, eager.pvalue)
    assert batched.stat.shape == (2,)


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_gmm_wald_size() -> None:
    """GMM Wald reports the expected statistic dimension and p-value range."""
    n_rep = 200
    alpha = 0.05
    rejections = 0
    for s in range(n_rep):
        rng = np.random.default_rng(1000 + s)
        g = rng.standard_normal((500, 2))
        if gmm_wald(g).pvalue < alpha:
            rejections += 1
    rate = rejections / n_rep
    se = np.sqrt(alpha * (1 - alpha) / n_rep)
    assert 0.0 < rate < MAX_EMPIRICAL_SIZE
    # Prefer 3-SE calibration; HAC finite-sample may inflate size slightly.
    if abs(rate - alpha) > 3 * se:
        assert rate < SOFT_MAX_EMPIRICAL_SIZE


# ---------------------------------------------------------------------------
# hansen_j
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(g=np.random.default_rng(2).standard_normal((500, 3)))
@given(g=full_rank_panels(100, 3))
def test_hansen_j_centered_and_plain_valid_dof(g: np.ndarray) -> None:
    """Centered and plain Hansen J variants retain the correct degrees of freedom.

    Args:
        g: Generated full-rank moment panel.

    """
    res_c = hansen_j(g, n_params=1, centered=True, estimated_by_gmm=True)
    res_p = hansen_j(g, n_params=1, centered=False, estimated_by_gmm=True)
    assert res_c.dof == TWO_MOMENT_DOF
    assert res_p.dof == TWO_MOMENT_DOF
    assert 0.0 <= res_c.pvalue <= 1.0
    assert 0.0 <= res_p.pvalue <= 1.0
    assert res_c.centered is True
    assert res_p.centered is False


@pytest.mark.component
@pytest.mark.property
def test_hansen_j_plain_deflated_vs_centered_under_misspecification() -> None:
    """Hall centering prevents mean misspecification from deflating Hansen J."""
    # Under a persistent nonzero mean, plain Newey-West absorbs the mean into
    # its autocovariances and is inflated by O(T/b_T).
    # Since J = T·ḡᵀ S⁻¹ ḡ, this deflates the plain J.
    # Hall-centered J (which subtracts the mean before forming covariance). So the
    # centered statistic is the LARGER one — the inconsistent plain weighting
    # understates the divergence from the moment conditions.
    g = ar1_panel(500, 3, phi=0.5, mean=0.5, seed=3)
    j_plain = hansen_j(g, centered=False)
    j_centered = hansen_j(g, centered=True)
    assert j_centered.stat > j_plain.stat


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(
    one_moment=np.random.default_rng(32).standard_normal((200, 1)),
    moments=np.random.default_rng(33).standard_normal((200, 3)),
)
@given(one_moment=full_rank_panels(200, 1), moments=full_rank_panels(200, 3))
def test_hansen_j_rejects_non_overidentified_and_external_estimates(
    one_moment: np.ndarray, moments: np.ndarray
) -> None:
    """J cannot fabricate one dof or subtract an externally estimated parameter.

    Args:
        one_moment: Generated single-moment panel.
        moments: Generated three-moment panel.

    """
    with pytest.raises(ValueError, match="requires over-identification"):
        hansen_j(one_moment, n_params=1, estimated_by_gmm=True)

    with pytest.raises(ValueError, match="same GMM criterion"):
        hansen_j(moments, n_params=1)

    singular = np.column_stack([moments[:, 0], moments[:, 0]])
    with pytest.raises(ValueError, match="full-rank moment covariance"):
        hansen_j(singular)


@pytest.mark.integration
@pytest.mark.contract
@examples(10)
@example(panel=np.random.default_rng(37).standard_normal((128, 3)))
@given(panel=full_rank_panels(128, 3))
def test_hansen_j_kernel_is_traceable_and_matches_report(panel: np.ndarray) -> None:
    """Hall/plain selection is static while the numerical panel remains traced.

    Args:
        panel: Generated full-rank moment panel.

    """
    moments = jnp.asarray(panel)
    eager = hansen_j(np.asarray(moments), bandwidth=0)
    kernel = hansen_j_kernel(moments, 0, 0, centered=True)

    def evaluate_hansen_j_kernel(panel: jax.Array) -> HansenJKernelResult:
        return hansen_j_kernel(panel, 0, 0, centered=True)

    compiled = jax.jit(evaluate_hansen_j_kernel)(moments)
    assert np.isclose(kernel.stat, eager.stat)
    assert np.isclose(compiled.pvalue, eager.pvalue)


@pytest.mark.component
@pytest.mark.regression
def test_parameter_convergence_cannot_replace_efficient_final_weight() -> None:
    """An independent finite-update witness has no conventional J reference."""
    moments = jnp.asarray([[0.0, 1.0], [2.0, 2.0], [1.0, 4.0], [5.0, 1.0]])
    derivative = jnp.asarray([[1.0], [0.0]])
    fit = IteratedGMM(PlainLongRun, max_iter=1, tolerance=1.0).estimate(
        AffineMomentProblem(moments, derivative)
    )
    tested = hansen_j_from_fit(fit, assume_correct_specification=True)
    assert bool(fit.solver.converged)
    assert not bool(tested.efficient_weight)
    assert not bool(tested.valid)
    assert np.isnan(tested.test.statistic)
    assert np.isnan(tested.test.pvalue)
