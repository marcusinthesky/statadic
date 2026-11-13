"""Tests for statadic.seeds."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from statadic.core.random import _label_to_uint32, cell_key

from .._budgets import examples

_SEEDS = st.integers(0, 2**32 - 1)
_LABELS = st.text(max_size=8)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(seed=42, first=10, second=8)
@given(seed=_SEEDS, first=st.integers(), second=st.integers())
def test_cell_key_deterministic(seed: int, first: int, second: int) -> None:
    """Verify cell key deterministic."""
    k1 = cell_key(seed, first, second)
    k2 = cell_key(seed, first, second)
    assert jnp.array_equal(k1, k2)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(seed=42, first=10, other_first=20, second=8, other_second=16)
@given(
    seed=_SEEDS,
    first=st.integers(),
    other_first=st.integers(),
    second=st.integers(),
    other_second=st.integers(),
)
def test_cell_key_different_labels_different_keys(
    seed: int, first: int, other_first: int, second: int, other_second: int
) -> None:
    """Verify cell key different labels different keys."""
    assume(first != other_first and second != other_second)
    k1 = cell_key(seed, first, second)
    k2 = cell_key(seed, other_first, second)
    k3 = cell_key(seed, first, other_second)
    # Equality would need a 32-bit MD5 prefix collision.
    assert not jnp.array_equal(k1, k2)
    assert not jnp.array_equal(k1, k3)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(seed=0, other_seed=1, label="n10")
@given(seed=_SEEDS, other_seed=_SEEDS, label=_LABELS)
def test_cell_key_different_seed_different_keys(
    seed: int, other_seed: int, label: str
) -> None:
    """Verify cell key different seed different keys."""
    assume(seed != other_seed)
    k1 = cell_key(seed, label)
    k2 = cell_key(other_seed, label)
    assert not jnp.array_equal(k1, k2)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(seed=42, first="n10", second="d8")
@given(seed=_SEEDS, first=_LABELS, second=_LABELS)
def test_cell_key_label_order_matters(seed: int, first: str, second: str) -> None:
    """Label ordering matters: (a, b) != (b, a) for distinct a, b."""
    assume(first != second)
    k_ab = cell_key(seed, first, second)
    k_ba = cell_key(seed, second, first)
    assert not jnp.array_equal(k_ab, k_ba)


@pytest.mark.unit
@pytest.mark.contract
def test_cell_key_no_labels() -> None:
    """cell_key with no labels is just PRNGKey(seed)."""
    k = cell_key(99)
    expected = jax.random.PRNGKey(99)
    assert jnp.array_equal(k, expected)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(label="hello", other="world")
@given(label=_LABELS, other=_LABELS)
def test_label_to_uint32_stable(label: str, other: str) -> None:
    """MD5-based hash is deterministic across calls."""
    assume(label != other)
    h1 = _label_to_uint32(label)
    h2 = _label_to_uint32(label)
    h3 = _label_to_uint32(other)
    assert h1 == h2
    # Inequality would need a 32-bit MD5 prefix collision.
    assert h1 != h3


_ANY_LABEL = st.one_of(
    st.text(), st.integers(), st.floats(), st.none(), st.booleans(), st.binary()
)


@pytest.mark.unit
@pytest.mark.property
@examples(100)
@example(label="")
@example(label="n10")
@example(label=42)
@example(label=3.14)
@example(label=None)
@given(label=_ANY_LABEL)
def test_label_to_uint32_range(label: object) -> None:
    """Verify label to uint32 range."""
    uint32_max = 0xFFFF_FFFF
    h = _label_to_uint32(label)
    assert 0 <= h <= uint32_max


@pytest.mark.component
@pytest.mark.contract
def test_cell_key_jax_valid() -> None:
    """Result must be a valid JAX PRNGKey usable for splitting."""
    key = cell_key(7, "n20", "d64")
    # Splitting a valid key should not raise
    k1, k2 = jax.random.split(key)
    assert k1.shape == k2.shape
