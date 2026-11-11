"""Tests for statadic.wasserstein."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given
from statadic.discrepancy.transport.sliced import (
    ProjectionDesign,
    sliced_wasserstein,
    sliced_wasserstein_under_design,
    w1_1d,
    w2_1d,
)
from statadic.sample.packed import PackedClouds

from ..._budgets import examples
from ..._strategies import float_arrays, prng_keys

_SEED = 42  # the value of the shared prng_key fixture
_KEY = jax.random.PRNGKey(_SEED)

# ---------------------------------------------------------------------------
# w2_1d
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(x=jax.random.normal(_KEY, (50,)))
@given(x=float_arrays((50,)))
def test_w2_1d_identical_samples(x: jax.Array) -> None:
    """Verify w2 1d identical samples."""
    assert float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
        )
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(x=jax.random.normal(_KEY, (200,)))
@given(x=float_arrays((200,)))
def test_w2_1d_shift(x: jax.Array) -> None:
    """Verify w2 1d shift."""
    shift = 3.0
    # For large n, sort(x+shift) = sort(x)+shift, so W2 = shift exactly.
    assert float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(x + shift)[..., None]),
        )
    ) == pytest.approx(shift, rel=1e-10)


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(
    x=jax.random.normal(_KEY, (40,)),
    y=jax.random.normal(jax.random.split(_KEY)[0], (40,)),
)
@given(x=float_arrays((40,)), y=float_arrays((40,)))
def test_w2_1d_symmetry(x: jax.Array, y: jax.Array) -> None:
    """Verify w2 1d symmetry."""
    assert float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
        )
    ) == pytest.approx(
        float(
            w2_1d(
                PackedClouds.from_array(jnp.asarray(y)[..., None]),
                PackedClouds.from_array(jnp.asarray(x)[..., None]),
            )
        ),
        rel=1e-10,
    )


# ---------------------------------------------------------------------------
# w1_1d
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(x=jax.random.normal(_KEY, (50,)))
@given(x=float_arrays((50,)))
def test_w1_1d_identical_samples(x: jax.Array) -> None:
    """Verify w1 1d identical samples."""
    assert float(
        w1_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
        )
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(x=jax.random.normal(_KEY, (200,)))
@given(x=float_arrays((200,)))
def test_w1_1d_shift(x: jax.Array) -> None:
    """Verify w1 1d shift."""
    shift = 2.5
    assert float(
        w1_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(x + shift)[..., None]),
        )
    ) == pytest.approx(shift, rel=1e-10)


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@example(
    x=jax.random.normal(_KEY, (40,)),
    y=jax.random.normal(jax.random.split(_KEY)[0], (40,)),
)
@given(x=float_arrays((40,)), y=float_arrays((40,)))
def test_w1_1d_symmetry(x: jax.Array, y: jax.Array) -> None:
    """Verify w1 1d symmetry."""
    assert float(
        w1_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
        )
    ) == pytest.approx(
        float(
            w1_1d(
                PackedClouds.from_array(jnp.asarray(y)[..., None]),
                PackedClouds.from_array(jnp.asarray(x)[..., None]),
            )
        ),
        rel=1e-10,
    )


# ---------------------------------------------------------------------------
# sliced_wasserstein: d=1 reduces to w2_1d
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@example(
    x=jax.random.normal(jax.random.split(_KEY, 3)[0], (64, 1)),
    y=jax.random.normal(jax.random.split(_KEY, 3)[1], (64, 1)),
    key=jax.random.split(_KEY, 3)[2],
)
@given(x=float_arrays((64, 1)), y=float_arrays((64, 1)), key=prng_keys())
def test_sliced_w2_d1_equals_w2_1d(x: jax.Array, y: jax.Array, key: jax.Array) -> None:
    """Sliced W2 with d=1 equals w2_1d exactly.

    When d=1 each projection direction θ ∈ {±1/1} has |θ|=1.
    W_2(θX, θY) = |θ| W_2(X, Y) = W_2(X, Y), so averaging over projections
    gives the same value as w2_1d regardless of n_projections.
    """
    sw = float(
        sliced_wasserstein(
            key,
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            n_projections=256,
            p=2,
        )
    )
    exact = float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(x[:, 0])[..., None]),
            PackedClouds.from_array(jnp.asarray(y[:, 0])[..., None]),
        )
    )
    projection_tolerance = 1e-6
    assert abs(sw - exact) < projection_tolerance, f"sliced={sw}, exact={exact}"


# ---------------------------------------------------------------------------
# sliced_wasserstein: shape, symmetry, zero self-distance
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@example(
    x=jax.random.normal(jax.random.split(_KEY)[0], (32, 4)),
    key=jax.random.split(_KEY)[1],
)
@given(x=float_arrays((32, 4)), key=prng_keys())
def test_sliced_w2_zero_self(x: jax.Array, key: jax.Array) -> None:
    """Verify sliced w2 zero self."""
    assert float(
        sliced_wasserstein(
            key,
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(x)),
        )
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@example(
    x=jax.random.normal(jax.random.split(_KEY, 3)[0], (32, 4)),
    y=jax.random.normal(jax.random.split(_KEY, 3)[1], (32, 4)),
    key=jax.random.split(_KEY, 3)[2],
)
@given(x=float_arrays((32, 4)), y=float_arrays((32, 4)), key=prng_keys())
def test_sliced_w2_symmetry(x: jax.Array, y: jax.Array, key: jax.Array) -> None:
    """Verify sliced w2 symmetry."""
    assert float(
        sliced_wasserstein(
            key,
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
        )
    ) == pytest.approx(
        float(
            sliced_wasserstein(
                key,
                PackedClouds.from_array(jnp.asarray(y)),
                PackedClouds.from_array(jnp.asarray(x)),
            )
        ),
        rel=1e-10,
    )


@pytest.mark.unit
@pytest.mark.numerical
def test_one_finite_projection_does_not_separate_distinct_measures() -> None:
    """Exhibit the identity counterexample behind the pseudometric law."""
    design = ProjectionDesign(jax.random.PRNGKey(41), n_projections=1, p=2)
    direction = design.directions(2, jnp.float64)[0]
    orthogonal = jnp.array([-direction[1], direction[0]])
    left = jnp.zeros((1, 2), dtype=jnp.float64)
    right = orthogonal[None, :]

    assert not bool(jnp.array_equal(left, right))
    assert float(
        sliced_wasserstein_under_design(
            design,
            PackedClouds.from_array(jnp.asarray(left)),
            PackedClouds.from_array(jnp.asarray(right)),
        )
    ) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Slow: W2 closed form (N(0,1) vs N(mu,1) large n → |mu|)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.statistical
@pytest.mark.slow
def test_w2_1d_gaussian_closed_form() -> None:
    """W2(N(0,1), N(μ,1)) = |μ| for the population; large n → close."""
    mu = 2.0
    key = jax.random.PRNGKey(999)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (5000,))
    y = jax.random.normal(k2, (5000,)) + mu
    got = float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
        )
    )
    # Variance of the estimator is O(1/n); allow 5 SE ≈ 5/sqrt(5000) ≈ 0.07
    sampling_tolerance = 0.1
    assert abs(got - mu) < sampling_tolerance, f"W2={got:.4f}, expected ≈{mu}"


# ---------------------------------------------------------------------------
# Slow: sliced W2 convergence for 2-D Gaussians vs analytic
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
def test_sliced_w2_2d_gaussian_convergence() -> None:
    """Sliced W2 converges to its analytic value for shifted 2-D Gaussians.

    For X ~ N(0, I_d) and Y ~ N(μ, I_d) with uniform random projections θ on
    S^{d-1}, the analytic sliced W2 is::

        SW_2 = ( E_θ[ W_2²(θ·X, θ·Y) ] )^{1/2}
             = ( E_θ[ (θ·μ)² ] )^{1/2}
             = ||μ||₂ / sqrt(d)

    because E_θ[(θ·μ)²] = ||μ||²/d for the uniform distribution on S^{d-1}.
    For d=2, μ=(1,0): analytic sliced W2 = 1/sqrt(2) ≈ 0.7071.

    Note: SW_2 ≤ W_2 = 1 in general; equality holds only in d=1.
    """
    d = 2
    mu = jnp.array([1.0, 0.0])
    analytic_sw2 = float(jnp.linalg.norm(mu)) / (d**0.5)  # ≈ 0.7071

    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    n = 2000
    x = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, d)) + mu[None, :]

    sw = float(
        sliced_wasserstein(
            k3,
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            n_projections=512,
            p=2,
        )
    )
    monte_carlo_tolerance = 0.05
    assert abs(sw - analytic_sw2) < monte_carlo_tolerance, (
        f"sliced_w2={sw:.4f}, analytic_sw2={analytic_sw2:.4f}"
    )
