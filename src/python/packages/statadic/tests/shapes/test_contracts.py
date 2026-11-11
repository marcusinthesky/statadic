"""Positive and negative contracts in the pinned shape-aware environment."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError
from statadic.sample.alternatives import _draw_normal

from .._typing import check_types

pytestmark = [pytest.mark.typing]

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _shape_environment() -> None:
    try:
        distribution("jax-pyrefly-stubs")
    except PackageNotFoundError:
        pytest.skip("Run just typecheck-shapes for the pinned shape environment")


_PRELUDE = """\
from __future__ import annotations
from typing import assert_type
from jax import Array
from jaxtyping import Float
from statadic.core.typing import ShapedArray, static_jaxtyping
"""


@pytest.mark.integration
@pytest.mark.contract
def test_projection_preserves_its_dimensions(tmp_path: Path) -> None:
    """The object's fields bind dimensions shared by later method calls."""
    check_types(
        _PRELUDE
        + """\
from statadic.sample.reduction import Projection
@static_jaxtyping("")
def check(mean: Float[Array, "3"], components: Float[Array, "3 2"],
          good: Float[Array, "5 3"], bad: Float[Array, "5 7"]) -> None:
    projection = Projection(mean, components, "pca")
    assert_type(projection.transform(good), ShapedArray[[5, 2]])
    projection.transform(bad)  # E: is not assignable
    wrong: Float[Array, "5 99"] = projection.transform(good)  # E: is not assignable
""",
        "projection.py",
        tmp_path,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_integer_arguments_determine_draw_shapes(tmp_path: Path) -> None:
    """Generated dimensions come from integer arguments, including through jit."""
    check_types(
        _PRELUDE
        + """\
import jax
from statadic.sample.alternatives import AltSpec, _draw_normal, draw
@static_jaxtyping("")
def check(key: Array) -> None:
    assert_type(_draw_normal(key, 2, 3), ShapedArray[[2, 3]])
    assert_type(draw(key, AltSpec("normal", None), 2, 3), ShapedArray[[2, 3]])
    compiled = jax.jit(_draw_normal, static_argnames=("n", "d"))
    assert_type(compiled(key, 2, 3), ShapedArray[[2, 3]])
    wrong: Float[Array, "99 100"] = _draw_normal(key, 2, 3)  # E: is not assignable
""",
        "draws.py",
        tmp_path,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_matrix_fields_retain_the_constructed_shape(tmp_path: Path) -> None:
    """Wrapping a square matrix does not erase its dimensions."""
    check_types(
        _PRELUDE
        + """\
from statadic.core.matrices import checked_dmat
@static_jaxtyping("")
def check(square: Float[Array, "3 3"], rectangle: Float[Array, "3 4"]) -> None:
    matrix = checked_dmat(square)
    assert_type(matrix.values, ShapedArray[[3, 3]])
    wrong: Float[Array, "99 100"] = matrix.values  # E: is not assignable
    checked_dmat(rectangle)  # E: is not assignable
""",
        "matrices.py",
        tmp_path,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_empirical_components_require_cloud_types(tmp_path: Path) -> None:
    """Empirical APIs require the carrier; factories check feature widths eagerly."""
    check_types(
        _PRELUDE
        + """\
from statadic.discrepancy.mmd.components import candidate_kernel_components
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.sample.packed import PackedClouds
@static_jaxtyping("")
def check(target: PackedClouds, candidates: PackedClouds,
          bad: Float[Array, "4 9"]) -> None:
    candidate_kernel_components(target, candidates, LINEAR_KERNEL)
    candidate_kernel_components(bad, candidates, LINEAR_KERNEL)  # E: is not assignable
""",
        "ragged.py",
        tmp_path,
    )


@pytest.mark.integration
@pytest.mark.contract
def test_batch_transforms_prepend_an_axis(tmp_path: Path) -> None:
    """Mapping a vector-to-scalar function returns a vector, rather than a scalar."""
    check_types(
        _PRELUDE
        + """\
import jax.numpy as jnp
from statadic.core.transforms import typed_jit, typed_vmap
from statadic.inference.resampling.blocks import map_replicates
@static_jaxtyping("d")
def total(row: Float[Array, "d"]) -> Float[Array, ""]:
    return jnp.sum(row)
@static_jaxtyping("")
def check(grid: Float[Array, "2 3"], row: Float[Array, "3"]) -> None:
    assert_type(typed_jit(total)(row), ShapedArray[[]])
    assert_type(typed_vmap(total)(grid), ShapedArray[[2]])
    assert_type(map_replicates(total, grid), ShapedArray[[2]])
    wrong: Float[Array, ""] = typed_vmap(total)(grid)  # E: is not assignable
""",
        "batching.py",
        tmp_path,
    )


@pytest.mark.component
@pytest.mark.contract
def test_draw_dimensions_are_also_checked_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong producer shape cannot satisfy a return annotation by rebinding names."""

    def wrong_normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        del key, shape
        return jnp.zeros((99, 100))

    monkeypatch.setattr(jax.random, "normal", wrong_normal)
    with pytest.raises(TypeCheckError):
        _draw_normal(jax.random.PRNGKey(0), 2, 3)


@pytest.mark.integration
@pytest.mark.contract
def test_draw_compiles_with_static_dimensions() -> None:
    """Array allocation dimensions must be static when compiling a sampler."""
    compiled = jax.jit(_draw_normal, static_argnames=("n", "d"))
    assert compiled(jax.random.PRNGKey(0), 2, 3).shape == (2, 3)
