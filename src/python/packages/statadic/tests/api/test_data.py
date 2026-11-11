"""Aligned observation boundaries and optional table transport."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.api.data import MeasureColumn, Observations
from statadic.sample.packed import PackedClouds

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import ill_typed

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
def test_explicit_wide_columns_require_x64(dtype: type) -> None:
    """Explicit wide source dtypes must not be silently narrowed."""
    values = np.array([1, 2], dtype=dtype)
    with jax.enable_x64(new_val=False):
        with pytest.raises(ValueError, match=r"x.*x64.*cast"):
            Observations({"x": values})
        with pytest.raises(ValueError, match=r"x.*x64.*cast"):
            Observations({"x": values}, row_ids=(0, 1))
        assert Observations({"x": np.array([1, 2], dtype=np.int32)}).n_rows == len(
            values
        )
        assert Observations({"x": [1, 2]}).n_rows == len(values)
    with jax.enable_x64(new_val=True):
        assert Observations({"x": values}).numeric("x").dtype == dtype


@pytest.mark.integration
@pytest.mark.contract
def test_precision_defaults_and_unsupported_column_dtype() -> None:
    """Default-width arrays remain valid and unsupported dtypes name the adapter."""
    with jax.enable_x64(new_val=False):
        for values in (np.array([1.0], dtype=np.float32), jnp.array([1.0]), [1.0]):
            assert Observations({"x": values}).n_rows == 1
        with pytest.raises(ValueError, match=r"x.*x64.*cast"):
            Observations({"x": [np.float64(1.0)]})
    pl = pytest.importorskip("polars")
    with pytest.raises(TypeError, match=r"x.*from_polars"):
        Observations({"x": pl.Series([1.0])})


@pytest.mark.unit
@pytest.mark.contract
def test_grouping_labels_preserve_types_and_reject_float_keys() -> None:
    """Grouping uses ordered integer/string keys without numeric coercion."""
    data = Observations(
        {"integer": jnp.array([2, 1]), "float": jnp.array([2.0, 1.0])},
        labels={"category": ("2", 1)},
    )
    assert data.grouping_labels("category") == ("2", 1)
    assert data.grouping_labels("integer") == (2, 1)
    with pytest.raises(ValueError, match="require integer keys"):
        data.grouping_labels("float")
    with pytest.raises(KeyError):
        data.grouping_labels("missing")


@pytest.mark.unit
@pytest.mark.contract
def test_raw_host_labels_decode_integral_scalars() -> None:
    """Host integer scalars retain exact identities without entering JAX."""
    data = Observations(
        {}, labels={"group": tuple(np.array([2**60, 2**60 + 1], dtype=np.int64))}
    )
    assert data.categorical("group") == (2**60, 2**60 + 1)
    assert all(type(value) is int for value in data.categorical("group"))
    with pytest.raises(TypeError, match="excluding bool"):
        Observations({}, labels={"group": (np.bool_(1),)})


@pytest.mark.component
@pytest.mark.contract
def test_outer_observations_keep_measure_atoms_separate() -> None:
    """Unequal supports still correspond to one row each."""
    # Arrange
    clouds = PackedClouds.from_clouds([jnp.ones((2, 1)), jnp.ones((5, 1))])
    # Act
    data = Observations(
        {"y": jnp.array([2.0, 5.0])},
        row_ids=("left", "right"),
        labels={"group": ("a", "b")},
        measures={"cloud": MeasureColumn(("left", "right"), clouds)},
    )
    subset = data.take((1,))
    # Assert
    assert data.n_rows == len(data.row_ids)
    assert subset.row_ids == ("right",)
    assert subset.measure("cloud").n_observations == 1
    np.testing.assert_array_equal(subset.numeric("y"), [5.0])


@pytest.mark.unit
@pytest.mark.contract
def test_observation_metadata_is_owned_and_typed() -> None:
    """Mutable input maps and string/int identities do not alias silently."""
    # Arrange
    columns = {"x": jnp.array([1.0, 2.0])}
    # Act
    data = Observations(columns, row_ids=(1, "1"))
    columns["x"] = jnp.zeros(2)
    # Assert
    assert data.row_ids == (1, "1")
    np.testing.assert_array_equal(data.numeric("x"), [1.0, 2.0])
    with pytest.raises(TypeError):
        ill_typed(data.columns)["extra"] = jnp.ones(2)  # immutable contract witness


@pytest.mark.unit
@pytest.mark.contract
def test_replacement_and_source_selection_preserve_alignment() -> None:
    """Replacement revalidates input and nested selections retain source positions."""

    class Source:
        names = ("cluster",)
        n_rows = 4

        def labels(self, name: str) -> tuple[object, ...]:
            assert name == "cluster"
            return ("a", "b", "a", "c")

    data = Observations({"x": [1.0, 2.0, 3.0, 4.0]}, source=Source())
    selected = data.take((3, 1, 2)).take((2, 0))
    assert selected.source_rows == (2, 3)
    assert selected.resolve_grouping("cluster") == ("a", "c")
    replaced = replace(selected, columns={"x": jnp.array([8.0, 9.0])})
    np.testing.assert_array_equal(replaced.numeric("x"), [8.0, 9.0])
    with pytest.raises(ValueError, match="shape"):
        replace(selected, columns={"x": jnp.ones(1)})


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("values", [1.0, jnp.ones(()), jnp.ones((2, 1))])
def test_constructor_rejects_nonvector_columns(values: object) -> None:
    """A raw scalar or matrix cannot establish an observation axis."""
    with pytest.raises(ValueError, match="vector shape"):
        Observations({"x": values})


@st.composite
def _columns_with_a_nonfinite_entry(draw: DrawFn) -> jax.Array:
    """Return a length-two column in which exactly one entry is NaN or infinite."""
    finite = draw(float_arrays(2))
    position = draw(st.integers(0, 1))
    bad = draw(st.sampled_from([jnp.nan, jnp.inf, -jnp.inf]))
    return finite.at[position].set(bad)


@pytest.mark.unit
@pytest.mark.contract
@example(values=jnp.array([1.0, jnp.nan]))
@example(values=jnp.array([1.0, jnp.inf]))
@given(values=_columns_with_a_nonfinite_entry())
@examples(40)
def test_missing_numeric_values_raise_without_row_dropping(values: jax.Array) -> None:
    """Column-specific dropping would destroy aligned identity."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="nonfinite"):
        Observations({"x": values})


@pytest.mark.unit
@pytest.mark.contract
def test_misalignment_and_duplicate_identities_raise() -> None:
    """Reject incompatible roles before a scientific procedure sees arrays."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="shape"):
        Observations({"x": jnp.ones(2), "y": jnp.ones(3)})
    with pytest.raises(ValueError, match="unique"):
        Observations({"x": jnp.ones(2)}, row_ids=("a", "a"))
    with pytest.raises(TypeError, match="bool"):
        Observations({"x": jnp.ones(2)}, row_ids=(True, 1))


@pytest.mark.component
@pytest.mark.contract
def test_measure_columns_require_exact_identity_alignment() -> None:
    """Same ids in another order, or other ids, must raise, never reorder."""
    # Arrange
    clouds = PackedClouds.from_clouds([jnp.ones((2, 1)), jnp.ones((3, 1))])
    y = {"y": jnp.array([2.0, 5.0])}
    # Act / Assert
    with pytest.raises(ValueError, match="identities must equal"):
        Observations(
            y,
            row_ids=("left", "right"),
            measures={"cloud": MeasureColumn(("right", "left"), clouds)},
        )
    with pytest.raises(ValueError, match="identities must equal"):
        Observations(
            y,
            row_ids=("left", "right"),
            measures={"cloud": MeasureColumn(("left", "other"), clouds)},
        )
    with pytest.raises(ValueError, match="number of measures"):
        MeasureColumn(("left",), clouds)
    with pytest.raises(ValueError, match="unique"):
        MeasureColumn(("left", "left"), clouds)


@pytest.mark.component
@pytest.mark.contract
def test_take_keeps_measure_identities_with_their_measures() -> None:
    """Selection reorders identities and measures together."""
    # Arrange
    clouds = PackedClouds.from_clouds([jnp.ones((2, 1)), jnp.ones((3, 1))])
    data = Observations(
        {"y": jnp.array([2.0, 5.0])},
        row_ids=("a", "b"),
        measures={"cloud": MeasureColumn(("a", "b"), clouds)},
    )
    # Act
    subset = data.take((1, 0))
    # Assert
    assert subset.measures["cloud"].row_ids == ("b", "a")
    np.testing.assert_array_equal(subset.measure("cloud").counts, [3, 2])


class _Source:
    """Minimal fit-time column source with one label column."""

    def __init__(self, firm: tuple[str, ...]) -> None:
        self._firm = firm

    @property
    def names(self) -> tuple[str, ...]:
        return ("x", "firm")

    @property
    def n_rows(self) -> int:
        return len(self._firm)

    def labels(self, name: str) -> tuple[str, ...]:
        assert name == "firm"
        return self._firm


@pytest.mark.unit
@pytest.mark.contract
def test_source_columns_resolve_by_name_and_follow_row_selection() -> None:
    """A source column is aligned through take() and listed once by name."""
    data = Observations(
        {"x": jnp.asarray([0.0, 1.0, 2.0])}, source=_Source(("a", "b", "c"))
    )
    assert data.available_columns == ("x", "firm")
    assert data.resolve_grouping("firm") == ("a", "b", "c")
    subset = data.take((2, 0))
    assert subset.resolve_grouping("firm") == ("c", "a")
    assert subset.take((1,)).resolve_grouping("firm") == ("a",)
    with pytest.raises(KeyError, match=r"'firm'.*available: x"):
        Observations({"x": jnp.asarray([0.0])}).resolve_grouping("firm")
    with pytest.raises(ValueError, match="aligned"):
        Observations({"x": jnp.asarray([0.0, 1.0])}, source=_Source(("a", "b", "c")))
