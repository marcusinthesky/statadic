# standalone package-maintenance entry point
"""Typed JAX/host ground strategies and their serialization boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from statadic.core.axioms import (
    METRIC,
    LawProperties,
)
from statadic.core.domains import (
    FiniteValues,
)
from statadic.geometry.ground.config import (
    InvalidGroundDistanceConfigError,
    is_euclidean,
    parse_ground_distance,
    parse_ground_distance_name,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    EuclideanLaw,
    cdist,
    declare_ground_distance,
    euclidean_distance,
)

from ..._typing import ill_typed

if TYPE_CHECKING:
    from jax import Array
    from statadic.core.axioms import (
        Axioms,
    )
    from statadic.geometry.ground.config import (
        BuiltinGroundDistance,
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("name", "strategy"),
    [
        ("euclidean", EUCLIDEAN),
        ("angular", ANGULAR),
        ("cosine", COSINE),
        ("normalized_euclidean", NORMALIZED_EUCLIDEAN),
    ],
)
def test_closed_names_and_configs_resolve_to_singletons(
    name: str,
    strategy: BuiltinGroundDistance,
) -> None:
    """Both serialized forms stop at the eager parser."""
    assert parse_ground_distance_name(name) is strategy
    assert parse_ground_distance({"kind": name}) is strategy
    assert resolve_ground_distance(strategy) is strategy


@pytest.mark.unit
@pytest.mark.contract
def test_unknown_name_and_bare_callable_are_rejected() -> None:
    """Dynamic strings fail in the parser; custom functions need a law."""
    with pytest.raises((TypeError, ValueError)):
        parse_ground_distance_name("taxicab")
    with pytest.raises(TypeError):
        resolve_ground_distance(ill_typed(euclidean_distance))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        {},
        {"kind": "taxicab"},
        {"kind": 1},
        {"kind": "euclidean", "unexpected": True},
        {"kind": "euclidean", 1: "mixed-key-type"},
    ],
)
def test_serialized_config_structure_is_validated(
    config: object,
) -> None:
    """Missing, unknown, wrong-type, and extra fields fail at the parser."""
    with pytest.raises(InvalidGroundDistanceConfigError):
        parse_ground_distance(config)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "config",
    [{}, {"kind": "taxicab"}, {"kind": 1}, {"kind": "euclidean", "extra": True}],
)
def test_resolver_rejects_invalid_serialized_mappings(
    config: dict[object, object],
) -> None:
    """The eager resolver delegates malformed mappings to the checked parser."""
    with pytest.raises(InvalidGroundDistanceConfigError):
        resolve_ground_distance(config)


@pytest.mark.unit
def test_builtin_union_can_be_narrowed() -> None:
    """TypeIs helper and runtime identity agree at the config boundary."""
    assert is_euclidean(parse_ground_distance_name("euclidean"))
    assert not is_euclidean(parse_ground_distance_name("angular"))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.needs_import_hook
def test_declared_strategy_is_import_hook_and_transform_safe() -> None:
    """Construction, call checking, JIT, and JAX-static registration compose."""

    def scaled_euclidean(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * euclidean_distance(x, y)

    strategy = declare_ground_distance(
        scaled_euclidean,
        law=EuclideanLaw(),
        name="scaled_euclidean",
        required_evidence=FiniteValues,
    )
    x = jnp.array([[0.0], [1.0]])
    assert strategy.law == EuclideanLaw()
    assert strategy.name == "scaled_euclidean"
    assert jax.tree_util.tree_leaves(strategy) == []
    assert jnp.allclose(cdist(x, x, metric=strategy), 2.0 * euclidean_distance(x, x))

    def evaluate_cdist(values: jax.Array) -> Array:
        return cdist(values, values, metric=strategy)

    assert jnp.allclose(
        jax.jit(evaluate_cdist)(x),
        cdist(x, x, metric=strategy),
    )


@pytest.mark.unit
@pytest.mark.contract
def test_jax_static_factory_rejects_unhashable_declarations() -> None:
    """Registered-static strategies cannot hide unhashable semantic metadata."""

    @dataclass
    class UnhashableLaw:
        axioms: Axioms = METRIC
        properties: LawProperties = field(default=LawProperties(0), init=False)

    with pytest.raises(TypeError, match="must be hashable"):
        declare_ground_distance(
            euclidean_distance,
            law=ill_typed(UnhashableLaw()),
            name="unhashable",
            required_evidence=FiniteValues,
        )
