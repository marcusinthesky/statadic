# standalone package-maintenance entry point
"""Typed-key support through the real annotation hook and frozen legacy draws."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from statadic.core.random import cell_key, permutation_batch
from statadic.inference.resampling.bootstrap import bootstrap_estimate
from statadic.operators.longrun import (
    circular_block_bootstrap_indices_kernel,
)


def _key(seed: int, representation: str) -> jax.Array:
    """Construct either supported Threefry representation explicitly."""
    if representation == "typed":
        return jax.random.key(seed, impl="threefry2x32")
    return jax.random.PRNGKey(seed, impl="threefry2x32")


def _mean(values: jax.Array) -> jax.Array:
    """Return a scalar statistic without crossing the host boundary."""
    return jnp.mean(values)


def _draw_summary(key: jax.Array, operation: str) -> jax.Array:
    """Exercise representative keyed public numerical paths."""
    if operation == "permutation":
        return permutation_batch(key, jnp.arange(7), 4)
    if operation == "circular-bootstrap":
        return circular_block_bootstrap_indices_kernel(key, 7, 3, 4)
    estimate = bootstrap_estimate(jnp.arange(7.0), _mean, key, num_bootstrap=7)
    return jnp.stack((estimate.estimate, estimate.se, *estimate.ci))


@pytest.mark.component
@pytest.mark.regression
def test_legacy_cell_key_and_permutation_schedule_remain_frozen() -> None:
    """Typed-key admission must not reseed the retained legacy sampler."""
    key = cell_key(13, "migration", "permutation_batch")
    assert key.dtype == jnp.uint32
    assert key.shape == (2,)
    np.testing.assert_array_equal(key, [2585001211, 3090554020])
    np.testing.assert_array_equal(
        cell_key(17, "inference", "cell"), [3508635379, 2665694346]
    )
    np.testing.assert_array_equal(
        permutation_batch(key, jnp.arange(8, dtype=jnp.int32), 4),
        [
            [3, 0, 6, 5, 2, 1, 7, 4],
            [6, 5, 7, 0, 2, 1, 4, 3],
            [1, 7, 3, 5, 4, 0, 2, 6],
            [3, 1, 2, 4, 0, 5, 7, 6],
        ],
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.needs_import_hook
@pytest.mark.parametrize("representation", ["legacy", "typed"])
@pytest.mark.parametrize(
    "operation", ["permutation", "bootstrap", "circular-bootstrap"]
)
def test_scalar_keys_survive_eager_jit_and_per_key_vmap(
    representation: str, operation: str
) -> None:
    """Actual annotated consumers accept modern keys without converting them."""
    key = _key(13, representation)
    legacy = _key(13, "legacy")

    def run(draw_key: jax.Array) -> jax.Array:
        return _draw_summary(draw_key, operation)

    expected = run(legacy)
    np.testing.assert_allclose(run(key), expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(jax.jit(run)(key), expected, rtol=0.0, atol=1e-14)
    keys = jax.random.split(key, 3)
    expected_batch = jnp.stack([run(draw_key) for draw_key in keys])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(run))(keys), expected_batch, rtol=0.0, atol=1e-14
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.needs_import_hook
@pytest.mark.parametrize(
    "kind", ["float", "wrong-width", "scalar-uint", "typed-batch", "legacy-batch"]
)
def test_scalar_key_boundary_rejects_invalid_or_batched_arrays(kind: str) -> None:
    """A typed-key union must retain dtype/shape checks at a scalar consumer."""
    if kind == "float":
        key = jnp.ones(2)
    elif kind == "wrong-width":
        key = jnp.ones(3, dtype=jnp.uint32)
    elif kind == "scalar-uint":
        key = jnp.asarray(1, dtype=jnp.uint32)
    else:
        representation = "typed" if kind == "typed-batch" else "legacy"
        key = jax.random.split(_key(17, representation), 2)
    with pytest.raises(TypeCheckError):
        permutation_batch(key, jnp.arange(5), 3)
