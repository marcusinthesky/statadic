# standalone package-maintenance entry point
"""Tests for :mod:`statadic.inference.kappa`.

The delta method for entry variance, moment-Jacobian concentration as a weak
identification diagnostic, and boundary-robust p-values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import stats
from statadic.inference.kappa import (
    boundary_robust_pvalue,
    boundary_robust_pvalue_kernel,
    kappa_delta_kernel,
    kappa_delta_method,
    moment_jacobian_concentration,
    moment_jacobian_concentration_kernel,
)

from .._budgets import examples
from .._strategies import float_arrays
from ._moments import (
    WEAK_IDENTIFICATION_THRESHOLD,
)

if TYPE_CHECKING:
    from jax import Array
    from statadic.inference.kappa import (
        KappaDeltaKernelResult,
        MomentJacobianKernelResult,
    )

_KAPPA = st.floats(0.0, 3.0)  # kappa_hat is documented nonnegative
_SE = st.floats(0.01, 1.0)
# Squared-distance matrices are symmetric (the documented domain), by construction.
_D2 = float_arrays((2, 2), bound=10.0).map(lambda a: (a + a.T) / 2)

# ---------------------------------------------------------------------------
# kappa_delta_method
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.reference
@examples(100)
@example(d2=jnp.array([[0.0, 4.0], [4.0, 0.0]]), kappa_hat=1.0, kappa_se=0.1)
@given(d2=_D2, kappa_hat=_KAPPA, kappa_se=_SE)
def test_kappa_delta_method_entry_variance(
    d2: Array, kappa_hat: float, kappa_se: float
) -> None:
    """The kappa delta method propagates variance to each covariance entry.

    The covariance map is ``-kappa**2 * d2 / 2``, so each entry's variance is
    ``(kappa * d2)**2 * se**2``. The exact ``0.16`` value is pinned in a separate
    witness test.

    Args:
        d2: Generated squared-distance-like matrix.
        kappa_hat: Generated estimate.
        kappa_se: Generated standard error.

    """
    res = kappa_delta_method(np.asarray(d2), kappa_hat=kappa_hat, kappa_se=kappa_se)
    expected = (kappa_hat * np.asarray(d2)) ** 2 * kappa_se**2
    np.testing.assert_allclose(res.entry_var, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.component
@pytest.mark.numerical
def test_kappa_delta_method_entry_variance_pinned_witness() -> None:
    """The exact hand-computed value: ``(1 * 4)**2 * 0.1**2 = 0.16``."""
    d2 = np.array([[0.0, 4.0], [4.0, 0.0]])
    res = kappa_delta_method(d2, kappa_hat=1.0, kappa_se=0.1)
    assert np.isclose(res.entry_var[0, 1], (1.0 * 4.0) ** 2 * 0.1**2)
    assert np.isclose(res.entry_var[0, 1], 0.16)
    assert np.isclose(res.entry_var[0, 0], 0.0)
    assert np.isclose(res.entry_var[1, 1], 0.0)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_eager_kappa_reports_own_float64_precision() -> None:
    """Host report doors preserve float64 without changing global JAX policy."""
    d2 = np.array([[0.0, 1e-10], [1e-10, 0.0]])
    moments = np.arange(60.0).reshape(30, 2)
    derivatives = np.full_like(moments, -0.5)

    with jax.enable_x64(new_val=False):
        delta = kappa_delta_method(d2, kappa_hat=1.0, kappa_se=0.1)
        jacobian = moment_jacobian_concentration(
            moments,
            derivatives,
            bandwidth=0,
        )
        tail = boundary_robust_pvalue(0.3, 0.1)
        assert not jax.config.read("jax_enable_x64")

    assert delta.entry_var[0, 1] > 0.0
    assert np.isfinite(jacobian.concentration)
    assert np.isclose(tail, stats.norm.sf(3.0))


@pytest.mark.component
@pytest.mark.numerical
@examples(40)
@example(d2=jnp.asarray([[0.0, 4.0], [4.0, 0.0]]))
@given(d2=_D2)
def test_kappa_delta_kernel_uses_traced_derivative_at_boundary(d2: Array) -> None:
    """Autodiff gives the closed-form derivative, including κ=0.

    Args:
        d2: Generated squared-distance-like matrix.

    """
    interior = kappa_delta_kernel(d2, jnp.asarray(1.0), jnp.asarray(0.1))
    boundary = kappa_delta_kernel(d2, jnp.asarray(0.0), jnp.asarray(0.1))

    def evaluate_kappa_delta_kernel(k: jax.Array) -> KappaDeltaKernelResult:
        return kappa_delta_kernel(d2, k, jnp.asarray(0.1))

    batched = jax.vmap(evaluate_kappa_delta_kernel)(jnp.asarray([0.0, 1.0]))
    jax.make_jaxpr(kappa_delta_kernel)(
        d2,
        jnp.asarray(1.0),
        jnp.asarray(0.1),
    )
    assert np.allclose(interior.gradient, -np.asarray(d2))
    assert np.allclose(boundary.gradient, 0.0)
    assert batched.entry_var.shape == (2, 2, 2)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(
    vector=np.ones(2),
    g=np.random.default_rng(24).standard_normal((30, 2)),
    derivative=np.ones((30, 1)),
)
@given(
    vector=hnp.arrays(np.float64, 2, elements=st.floats(-10, 10)),
    g=hnp.arrays(np.float64, (30, 2), elements=st.floats(-10, 10)),
    derivative=hnp.arrays(np.float64, (30, 1), elements=st.floats(-10, 10)),
)
def test_kappa_helpers_reject_malformed_inputs(
    vector: np.ndarray, g: np.ndarray, derivative: np.ndarray
) -> None:
    """Delta and Jacobian diagnostics reject nonfinite or mismatched arrays.

    Args:
        vector: A vector where a square matrix is required.
        g: A moment panel.
        derivative: A derivative panel whose shape differs from ``g``.

    """
    with pytest.raises(ValueError, match="square matrix"):
        kappa_delta_method(vector, kappa_hat=1.0, kappa_se=0.1)
    with pytest.raises(ValueError, match="same shape"):
        moment_jacobian_concentration(g, derivative)


# ---------------------------------------------------------------------------
# moment_jacobian_concentration
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.property
def test_moment_jacobian_concentration_strong_vs_weak() -> None:
    """Jacobian concentration distinguishes strong from weak identification."""
    rng = np.random.default_rng(6)
    g = 0.1 * rng.standard_normal((500, 2))

    dg_strong = -3.0 + 0.05 * rng.standard_normal((500, 2))
    res_strong = moment_jacobian_concentration(g, dg_strong)
    assert res_strong.flagged_weak is False
    assert res_strong.concentration >= WEAK_IDENTIFICATION_THRESHOLD

    dg_weak = 1e-4 * rng.standard_normal((500, 2))
    res_weak = moment_jacobian_concentration(g, dg_weak)
    assert res_weak.flagged_weak is True
    assert res_weak.concentration < WEAK_IDENTIFICATION_THRESHOLD


@pytest.mark.component
@pytest.mark.contract
def test_moment_jacobian_kernel_is_traceable_and_matches_report() -> None:
    """The HAC sensitivity calculation remains a fixed-structure JAX core."""
    rng = np.random.default_rng(39)
    moments = rng.standard_normal((128, 2))
    derivative = -0.5 + 0.1 * rng.standard_normal((128, 2))
    eager = moment_jacobian_concentration(moments, derivative, bandwidth=0)
    kernel = moment_jacobian_concentration_kernel(
        jnp.asarray(moments),
        jnp.asarray(derivative),
        0,
    )

    def evaluate_moment_jacobian_concentration_kernel(
        panel: jax.Array, jacobian: jax.Array
    ) -> MomentJacobianKernelResult:
        return moment_jacobian_concentration_kernel(panel, jacobian, 0)

    jax.make_jaxpr(evaluate_moment_jacobian_concentration_kernel)(
        jnp.asarray(moments), jnp.asarray(derivative)
    )
    assert np.isclose(kernel.concentration, eager.concentration)
    assert bool(kernel.flagged_weak) is eager.flagged_weak


@pytest.mark.component
@pytest.mark.reference
def test_jacobian_concentration_matches_independent_plain_lag_products() -> None:
    """Check raw moment means, Bartlett lag products and the retained T factor."""
    time = np.arange(18.0)
    moments = np.column_stack((0.3 + np.sin(time), -0.2 + np.cos(time * 0.7)))
    derivative = np.column_stack((-0.5 + time / 40, 0.2 - time / 80))
    omega = moments.T @ moments / len(time)
    for lag in (1, 2):
        cross = moments[lag:].T @ moments[:-lag] / len(time)
        omega += (1 - lag / 3) * (cross + cross.T)
    dbar = derivative.mean(0)
    expected = len(time) * dbar @ np.linalg.solve(omega, dbar)

    result = moment_jacobian_concentration(moments, derivative, bandwidth=2)

    np.testing.assert_allclose(result.concentration, expected, rtol=1e-11)


# ---------------------------------------------------------------------------
# boundary_robust_pvalue
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@examples(100)
@given(kappa_hat=st.floats(1e-3, 1.0), kappa_se=_SE)
def test_boundary_robust_pvalue_equals_the_one_sided_normal_tail(
    kappa_hat: float, kappa_se: float
) -> None:
    """The chi-bar-square tail equals the one-sided normal tail for z > 0.

    Args:
        kappa_hat: Generated positive estimate.
        kappa_se: Generated standard error.

    """
    np.testing.assert_allclose(
        boundary_robust_pvalue(kappa_hat, kappa_se),
        stats.norm.sf(kappa_hat / kappa_se),
        rtol=1e-6,
        atol=1e-12,
    )
    assert boundary_robust_pvalue(0.0, kappa_se) == pytest.approx(1.0, abs=0.0)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_boundary_robust_pvalue() -> None:
    """The pinned normal-tail values at 1.645, 1.96 and 3, and the se guard."""
    np.testing.assert_equal(boundary_robust_pvalue(0.0, 0.1), 1.0)
    assert np.isclose(boundary_robust_pvalue(0.1645, 0.1), stats.norm.sf(1.645))
    assert np.isclose(boundary_robust_pvalue(0.196, 0.1), stats.norm.sf(1.96))
    assert np.isclose(boundary_robust_pvalue(0.3, 0.1), stats.norm.sf(3.0))
    with pytest.raises(ValueError, match="strictly positive"):
        boundary_robust_pvalue(1.0, 0.0)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@examples(40)
@example(estimates=jnp.asarray([0.0, 0.1645, 0.3]))
@given(estimates=float_arrays(3, bound=1.0).map(jnp.abs))
def test_boundary_tail_kernel_is_traceable_at_and_above_lower_bound(
    estimates: Array,
) -> None:
    """The mixture-law tail retains its inclusive boundary atom under vmap.

    Args:
        estimates: Generated non-negative estimates, including the boundary atom.

    """

    def evaluate_boundary_robust_pvalue_kernel(estimate: jax.Array) -> jax.Array:
        return boundary_robust_pvalue_kernel(
            estimate, jnp.asarray(0.1), jnp.asarray(0.0)
        )

    tails = jax.vmap(evaluate_boundary_robust_pvalue_kernel)(estimates)
    jax.make_jaxpr(boundary_robust_pvalue_kernel)(
        jnp.asarray(0.1),
        jnp.asarray(0.1),
        jnp.asarray(0.0),
    )
    expected = np.where(
        np.asarray(estimates) <= 0.0, 1.0, stats.norm.sf(np.asarray(estimates) / 0.1)
    )
    np.testing.assert_allclose(tails, expected, rtol=1e-6, atol=1e-12)


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(invalid=(float("nan"), 0.1, 0.0))
@example(invalid=(0.1, float("inf"), 0.0))
@example(invalid=(0.1, 0.1, float("nan")))
@example(invalid=(-0.1, 0.1, 0.0))
@given(
    invalid=st.one_of(
        st.tuples(st.sampled_from([float("nan"), float("inf")]), _SE, st.just(0.0)),
        st.tuples(
            st.floats(0.0, 1.0),
            st.sampled_from([float("nan"), float("inf")]),
            st.just(0.0),
        ),
        st.tuples(
            st.floats(0.0, 1.0), _SE, st.sampled_from([float("nan"), float("inf")])
        ),
        st.tuples(st.floats(-10.0, -1e-6), _SE, st.just(0.0)),
    )
)
def test_boundary_robust_pvalue_rejects_invalid_inputs(
    invalid: tuple[float, float, float],
) -> None:
    """Invalid estimates cannot be hidden behind a conservative p-value.

    Args:
        invalid: ``(kappa_hat, kappa_se, kappa_null)`` with a nonfinite entry or an
            estimate below the lower bound.

    """
    kappa_hat, kappa_se, kappa_null = invalid
    with pytest.raises(ValueError, match="must be"):
        boundary_robust_pvalue(kappa_hat, kappa_se, kappa_null)
