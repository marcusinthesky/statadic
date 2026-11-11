# standalone package-maintenance entry point
"""Tests for statadic.sample.alternatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.sample.alternatives import AltSpec, draw

from .._budgets import examples
from .._strategies import prng_keys

# ---------------------------------------------------------------------------
# Shape / dtype for all kinds
# ---------------------------------------------------------------------------


_SHAPE_SPECS = (
    AltSpec(kind="normal", param=None),
    AltSpec(kind="student_t", param=4.0),
    AltSpec(kind="mean_shift", param=1.5),
    AltSpec(kind="scale", param=2.0),
    AltSpec(kind="mixture", param=0.1, shift=2.0),
)
_DTYPE_SPECS = (*_SHAPE_SPECS[:4], AltSpec(kind="mixture", param=0.1))
_SEED_KEY = jax.random.PRNGKey(42)


@pytest.mark.component
@pytest.mark.contract
@example(key=_SEED_KEY, spec=_SHAPE_SPECS[0])
@example(key=_SEED_KEY, spec=_SHAPE_SPECS[1])
@example(key=_SEED_KEY, spec=_SHAPE_SPECS[2])
@example(key=_SEED_KEY, spec=_SHAPE_SPECS[3])
@example(key=_SEED_KEY, spec=_SHAPE_SPECS[4])
@given(key=prng_keys(), spec=st.sampled_from(_SHAPE_SPECS))
@examples(40)
def test_draw_shape(key: jax.Array, spec: AltSpec) -> None:
    """Every alternative returns an ``(n, d)`` sample for any key."""
    n, d = 20, 5
    result = draw(key, spec, n=n, d=d)
    assert result.shape == (n, d)


@pytest.mark.component
@pytest.mark.contract
@example(key=_SEED_KEY, spec=_DTYPE_SPECS[0])
@example(key=_SEED_KEY, spec=_DTYPE_SPECS[4])
@given(key=prng_keys(), spec=st.sampled_from(_DTYPE_SPECS))
@examples(40)
def test_draw_dtype(key: jax.Array, spec: AltSpec) -> None:
    """Every alternative returns a float sample for any key."""
    result = draw(key, spec, n=10, d=3)
    assert result.dtype in (jnp.float32, jnp.float64)


@pytest.mark.component
@pytest.mark.contract
@example(key=_SEED_KEY)
@given(key=prng_keys())
@examples(40)
def test_invalid_kind(key: jax.Array) -> None:
    """An unknown alternative kind is rejected whatever the key."""
    with pytest.raises((ValueError, AssertionError)):
        draw(key, AltSpec(kind="bogus", param=None), n=5, d=2)


# ---------------------------------------------------------------------------
# Mixture contamination count is exact
# ---------------------------------------------------------------------------


@pytest.mark.component
def test_mixture_contamination_count_exact(prng_key: jax.Array) -> None:
    """The last int(n*frac) rows should be the contaminated block."""
    n, d = 100, 3
    frac = 0.15
    shift = 3.0
    spec = AltSpec(kind="mixture", param=frac, shift=shift)
    result = draw(prng_key, spec, n=n, d=d)

    n_contam = int(n * frac)
    n_clean = n - n_contam

    # Contaminated rows should have mean near `shift`; clean rows near 0.
    clean_mean = float(jnp.mean(result[:n_clean]))
    contam_mean = float(jnp.mean(result[n_clean:]))
    assert abs(clean_mean) < 1.0, f"Clean mean {clean_mean:.3f} unexpectedly large"
    contamination_mean_tolerance = 1.5
    assert abs(contam_mean - shift) < contamination_mean_tolerance, (
        f"Contaminated mean {contam_mean:.3f} not near shift={shift}"
    )
    assert result.shape[0] == n


# ---------------------------------------------------------------------------
# Statistical sanity (slow)
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.slow
def test_student_t_heavier_tails_than_normal() -> None:
    """Student-t draws should have higher empirical kurtosis than normal draws."""
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    n, d, df = 5_000, 1, 4.0

    normal_draws = draw(k1, AltSpec(kind="normal", param=None), n=n, d=d)
    t_draws = draw(k2, AltSpec(kind="student_t", param=df), n=n, d=d)

    def kurtosis(x: jax.Array) -> float:
        x = x - jnp.mean(x)
        mu4 = jnp.mean(x**4)
        mu2 = jnp.mean(x**2)
        return float(mu4 / mu2**2)

    kurt_normal = kurtosis(normal_draws)
    kurt_t = kurtosis(t_draws)
    assert kurt_t > kurt_normal, (
        f"Student-t kurtosis {kurt_t:.2f} did not exceed "
        f"normal kurtosis {kurt_normal:.2f}"
    )


@pytest.mark.component
@pytest.mark.slow
def test_mean_shift_mean_near_delta() -> None:
    """Empirical mean of mean_shift draws should be close to delta."""
    key = jax.random.PRNGKey(11)
    n, d = 10_000, 3
    delta = 2.5
    result = draw(key, AltSpec(kind="mean_shift", param=delta), n=n, d=d)
    emp_mean = float(jnp.mean(result))
    mean_estimation_tolerance = 0.1
    assert abs(emp_mean - delta) < mean_estimation_tolerance, (
        f"Empirical mean {emp_mean:.3f} not near delta={delta}"
    )


@pytest.mark.component
@pytest.mark.slow
def test_mixture_shift_matches_spec() -> None:
    """With large shift, contaminated block mean should be near shift value."""
    key = jax.random.PRNGKey(22)
    n, d = 10_000, 2
    frac, shift = 0.2, 5.0
    spec = AltSpec(kind="mixture", param=frac, shift=shift)
    result = draw(key, spec, n=n, d=d)
    n_contam = int(n * frac)
    n_clean = n - n_contam
    contam_mean = float(jnp.mean(result[n_clean:]))
    contamination_mean_tolerance = 0.2
    assert abs(contam_mean - shift) < contamination_mean_tolerance, (
        f"Contaminated mean {contam_mean:.3f} not near shift={shift}"
    )
