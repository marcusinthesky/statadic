"""Retained typed host identities and explicit external-array decoding."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.model.dyadic import DyadicSample
from statadic.operators.design import NodeIndex

from .._typing import ill_typed


@pytest.mark.unit
@pytest.mark.contract
def test_typed_sample_retains_labels_disjoint_from_mutable_sequences() -> None:
    """Retained sample identity survives later edits to caller-owned lists."""
    left = ["a", "a"]
    right = ["b", "c"]
    outcome = np.asarray([1.0, 2.0])
    sample = DyadicSample(outcome, left, right)
    index = NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)
    left[0] = "changed"
    right.append("extra")
    assert sample.endpoint_i == ("a", "a")
    assert sample.endpoint_j == ("b", "c")
    assert sample.outcome is outcome
    assert index.node_ids == ("a", "b", "c")
    np.testing.assert_array_equal(sample.outcome, [1.0, 2.0])


@pytest.mark.unit
@pytest.mark.contract
def test_external_label_arrays_decode_python_labels_and_jax_ids() -> None:
    """Array adapters recover validated host values without a claimed ID subtype."""
    mixed = DyadicSample.from_arrays(
        jnp.asarray([1.0, 2.0]),
        np.asarray([1, "a"], dtype=object),
        np.asarray(["a", 2], dtype=object),
    )
    numeric = DyadicSample.from_arrays(
        jnp.asarray([1.0, 2.0]), jnp.asarray([0, 0]), jnp.asarray([1, 2])
    )
    assert mixed.endpoint_i == (1, "a")
    assert mixed.endpoint_j == ("a", 2)
    assert numeric.endpoint_i == (0, 0)
    assert numeric.endpoint_j == (1, 2)
    assert all(type(node) is int for node in (*numeric.endpoint_i, *numeric.endpoint_j))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("value", ["abc", b"abc"])
def test_whole_string_endpoints_are_rejected(value: object) -> None:
    """One string is not a sequence of node labels."""
    from statadic.model.dyadic import DyadicInputError  # noqa: PLC0415

    with pytest.raises(DyadicInputError, match="sequence of labels"):
        DyadicSample(jnp.ones(3), ill_typed(value), ("d", "e", "f"))


@pytest.mark.unit
@pytest.mark.contract
def test_declared_array_shape_wins_over_sequence_length() -> None:
    """A two-column buffer keeps its declared shape."""
    from statadic.model.dyadic.validation import _array_shape  # noqa: PLC0415

    assert _array_shape(memoryview(np.ones((3, 2)))) == (3, 2)
    assert _array_shape(("a", "b", "c")) == (3,)
