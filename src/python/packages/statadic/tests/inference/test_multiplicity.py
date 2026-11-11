"""Tests for familywise corrections and their assumptions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statadic.inference.multiplicity import equivalence_multiplicity_control as control

from .._budgets import examples


def _holm_oracle(p: list[float], alpha: float) -> list[bool]:
    result = [False] * len(p)
    for i, index in enumerate(sorted(range(len(p)), key=p.__getitem__)):
        if p[index] > alpha / (len(p) - i):
            break
        result[index] = True
    return result


@pytest.mark.component
@pytest.mark.reference
@examples(80, deadline=None)
@given(st.lists(st.integers(0, 1000), min_size=1, max_size=12))
def test_holm_matches_step_down_oracle(values: list[int]) -> None:
    """Compare to an independent step-down implementation, including ties."""
    p = [value / 1024 for value in values]
    np.testing.assert_array_equal(
        control(jnp.array(p), alpha=0.125), _holm_oracle(p, 0.125)
    )


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("method", ["bonferroni", "holm"])
def test_exhaustive_two_uniform_nulls(method: str) -> None:
    """All 40,000 equally likely outcomes respect the nominal familywise level."""
    grid = jnp.arange(1, 201) / 200
    pairs = jnp.stack(jnp.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)

    def run(p: jnp.ndarray) -> jnp.ndarray:
        return control(p, method=method)

    decisions = jax.vmap(run)(pairs)
    max_rejections = 2000
    assert int(jnp.any(decisions, axis=1).sum()) <= max_rejections


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("method", ["bonferroni", "holm"])
def test_jit_vmap_and_nan(method: str) -> None:
    """Compilation and batching preserve decisions; NaNs cannot be rejected."""
    p = jnp.array([[0.001, 0.02, 0.9], [0.001, jnp.nan, 0.01]])

    def run(x: jnp.ndarray) -> jnp.ndarray:
        return control(x, method=method)

    eager = jnp.stack([run(row) for row in p])
    np.testing.assert_array_equal(jax.jit(jax.vmap(run))(p), eager)
    np.testing.assert_array_equal(jax.vmap(run)(p), eager)
    assert not bool(eager[1, 1])


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("method", ["adaptive_bonferroni", "k_minus_1", "unknown"])
def test_unsupported_methods_raise(method: str) -> None:
    """Unsupported corrections do not silently fall through."""
    with pytest.raises(ValueError, match="Unknown method"):
        control(jnp.array([0.01]), method=method)


@pytest.mark.component
@pytest.mark.numerical
def test_default_is_holm_and_step_down_differs_from_bonferroni() -> None:
    """Holm continues after the first rejection at a less stringent threshold."""
    p = jnp.array([0.01, 0.024, 0.9])
    np.testing.assert_array_equal(control(p), [True, True, False])
    np.testing.assert_array_equal(control(p, method="bonferroni"), [True, False, False])


@pytest.mark.component
@pytest.mark.contract
def test_bh_refusal_explains_fdr_does_not_guarantee_fwer() -> None:
    """The refusal names the distinction between the two error rates."""
    with pytest.raises(ValueError, match="Benjamini-Hochberg controls FDR"):
        control(jnp.array([0.01]), method="bh")


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("method", ["bonferroni", "holm"])
def test_empty_family_in_eager_jit_and_vmap(method: str) -> None:
    """Empty families produce empty Boolean masks under transformations."""

    def run(p: jnp.ndarray) -> jnp.ndarray:
        return control(p, method=method)

    empty = jnp.array([])
    for result in (run(empty), jax.jit(run)(empty), jax.vmap(run)(jnp.zeros((2, 0)))):
        assert result.dtype == jnp.bool_
        assert result.shape[-1] == 0


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("method", ["bh", "unknown"])
def test_empty_family_still_rejects_unsupported_methods(method: str) -> None:
    """Empty input does not bypass method validation."""
    with pytest.raises(ValueError, match=r"Benjamini|Unknown method"):
        control(jnp.array([]), method=method)
