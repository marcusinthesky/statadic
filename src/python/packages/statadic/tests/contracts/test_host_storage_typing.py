# standalone package-maintenance entry point
"""Static evidence for retained node identities and spatial storage types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.typing,
    pytest.mark.subprocess,
    pytest.mark.requires_program("pyrefly"),
]

HOST_STORAGE_TYPES = """
from collections.abc import Hashable, Sequence
from typing import assert_type
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO
from statadic.inference.dyadic import NodeCountSchedule
from statadic.model.dyadic import DyadicSample
from statadic.operators.design import NodeIndex
from statadic.operators.spatial import SpatialWeights, prepare_spatial_weights

def preserve_ids[T: Hashable](left: Sequence[T], right: Sequence[T]) -> NodeIndex[T]:
    sample = DyadicSample(jnp.ones(2), left, right)
    return NodeIndex.from_endpoints(sample.endpoint_i, sample.endpoint_j)

typed = DyadicSample[int](jnp.ones(2), (0, 0), (1, 2))
index = NodeIndex.from_endpoints(typed.endpoint_i, typed.endpoint_j)
assert_type(index, NodeIndex[int])
schedule = NodeCountSchedule(index, jnp.ones((1, 3)))
assert_type(schedule, NodeCountSchedule[int])
assert_type(schedule.counts, Array)

erased = DyadicSample.from_arrays(jnp.ones(2), jnp.array([0, 0]), jnp.array([1, 2]))
assert_type(erased, DyadicSample[Hashable])
claimed: DyadicSample[int] = erased  # E: DyadicSample

dense = prepare_spatial_weights(jnp.eye(3), allow_diagonal=True)
sparse = prepare_spatial_weights(BCOO.fromdense(jnp.eye(3)), allow_diagonal=True)
assert_type(dense, SpatialWeights[Array])
assert_type(sparse, SpatialWeights[BCOO])
assert_type(dense.values, Array)
assert_type(sparse.values, BCOO)
"""


@pytest.mark.integration
@pytest.mark.contract
def test_checker_preserves_identity_and_storage_contracts(tmp_path: Path) -> None:
    """Typed sequences retain their ID type; erased arrays cannot invent one."""
    check_types(HOST_STORAGE_TYPES, "host_storage_types.py", tmp_path)
