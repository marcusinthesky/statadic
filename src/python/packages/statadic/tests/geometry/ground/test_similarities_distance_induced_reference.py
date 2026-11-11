"""Reference-point domain contracts for distance-induced kernels."""

from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from statadic.geometry.ground.metrics import ANGULAR
from statadic.geometry.ground.similarities import distance_induced_gram


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("metric", ["angular", "cosine", "normalized_euclidean"])
def test_reference_must_belong_to_metric_domain(
    metric: Literal["angular", "cosine", "normalized_euclidean"],
) -> None:
    """Normalized metrics require a reference instead of producing NaNs."""
    x = jnp.array([[3.0, 4.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="reference is required"):
        distance_induced_gram(x, x, metric=metric)
    result = distance_induced_gram(x, x, metric=metric, reference=jnp.array([1.0, 0.0]))
    assert bool(jnp.all(jnp.isfinite(result)))


@pytest.mark.component
@pytest.mark.contract
def test_declared_strategy_domain_is_checked_before_tracing() -> None:
    """The guard uses strategy metadata and permits compiled valid references."""
    x = jnp.array([[3.0, 4.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="reference is required"):
        distance_induced_gram(x, x, metric=ANGULAR)

    def evaluate(values: jax.Array) -> jax.Array:
        return distance_induced_gram(
            values, values, metric=ANGULAR, reference=jnp.array([1.0, 0.0])
        )

    compiled = jax.jit(evaluate)
    assert bool(jnp.all(jnp.isfinite(compiled(x))))
    assert bool(jnp.all(jnp.isfinite(distance_induced_gram(x, x, metric="euclidean"))))
