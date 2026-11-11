"""Optional table adapters preserve alignment and numeric transport semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from .._budgets import examples
from .._validation.tables import arrow_source, scribble, table_cases

if TYPE_CHECKING:
    from .._validation.tables import TableCase

pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")

from statadic.api.tables import (  # noqa: E402 -- skip before importing optional adapter
    from_arrow,
    from_polars,
    measures_from_arrow,
    measures_from_polars,
)

pytestmark = [
    pytest.mark.requires_optional("pyarrow"),
    pytest.mark.requires_optional("polars"),
]


@pytest.mark.integration
@pytest.mark.contract
def test_arrow_chunked_and_sliced_columns_preserve_order() -> None:
    """Chunk consolidation must not reorder or independently drop numeric rows."""
    # Arrange
    values = pa.chunked_array([[99.0, 1.0], [2.0, 3.0]], type=pa.float64())
    table = pa.table({"id": ["omit", "b", "a", "c"], "x": values}).slice(1)
    # Act
    data = from_arrow(table, numeric=("x",), row_id="id")
    # Assert
    assert data.row_ids == ("b", "a", "c")
    assert data.numeric("x").dtype == jnp.float64
    np.testing.assert_array_equal(data.numeric("x"), [1.0, 2.0, 3.0])


@pytest.mark.integration
def test_polars_and_arrow_use_the_same_observation_contract() -> None:
    """The dataframe path must preserve identities and categorical values."""
    # Arrange
    frame = pl.DataFrame({"id": [1, 2, 3], "x": [0.2, 0.4, 0.8], "g": ["a", "b", "a"]})
    # Act
    data = from_polars(frame, numeric=("x",), categorical=("g",), row_id="id")
    arrow = from_arrow(
        frame.to_arrow(), numeric=("x",), categorical=("g",), row_id="id"
    )
    # Assert
    assert data.row_ids == arrow.row_ids == (1, 2, 3)
    assert data.categorical("g") == ("a", "b", "a")
    np.testing.assert_array_equal(data.numeric("x"), arrow.numeric("x"))


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("values", [[1.0, None], [1.0, float("nan")]])
def test_arrow_missing_values_raise(values: list[float | None]) -> None:
    """DLPack cannot erase null semantics or accept missing scientific rows."""
    # Arrange
    table = pa.table({"x": values})
    # Act / Assert
    with pytest.raises(ValueError, match=r"null|nonfinite"):
        from_arrow(table, numeric=("x",))


@pytest.mark.integration
@pytest.mark.contract
def test_arrow_rejects_unsupported_numeric_and_missing_identity() -> None:
    """String coercion and absent identities are not numeric preparation."""
    # Arrange / Act / Assert
    with pytest.raises(TypeError, match="primitive"):
        from_arrow(pa.table({"x": ["1", "2"]}), numeric=("x",))
    with pytest.raises(TypeError, match="nonmissing"):
        from_arrow(
            pa.table({"id": [1, None], "x": [1.0, 2.0]}), numeric=("x",), row_id="id"
        )


def _long_table() -> pa.Table:
    return pa.table(
        {
            "id": ["b", "a", "b", "a", "a"],
            "v": [3.0, 1.0, 5.0, 2.0, 4.0],
            "m": [0.5, 0.2, 0.5, 0.3, 0.5],
        }
    )


@pytest.mark.integration
@pytest.mark.contract
def test_long_measures_use_sorted_ids_and_table_atom_order() -> None:
    """Atoms group by id; default identity order is sorted."""
    # Arrange / Act
    column = measures_from_arrow(_long_table(), row_id="id", values=("v",), mass="m")
    # Assert
    assert column.row_ids == ("a", "b")
    np.testing.assert_array_equal(column.measures.counts, [3, 2])
    np.testing.assert_allclose(column.measures.data[0, :3, 0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(column.measures.masses[0, :3], [0.2, 0.3, 0.5])


@pytest.mark.integration
@pytest.mark.contract
def test_long_measures_explicit_order_and_polars_agree() -> None:
    """An explicit order is honoured and must name exactly the table ids."""
    # Arrange
    frame = pl.from_arrow(_long_table())
    # Act
    column = measures_from_polars(
        frame, row_id="id", values=("v",), mass="m", order=("b", "a")
    )
    # Assert
    assert column.row_ids == ("b", "a")
    np.testing.assert_array_equal(column.measures.counts, [2, 3])
    with pytest.raises(ValueError, match="exactly"):
        measures_from_arrow(
            _long_table(), row_id="id", values=("v",), mass="m", order=("a",)
        )


@pytest.mark.integration
@pytest.mark.numerical
def test_long_measures_mass_rules_follow_empirical_measures() -> None:
    """Omitted masses are uniform; unnormalized masses raise unless requested."""
    # Arrange
    table = _long_table()
    # Act
    uniform = measures_from_arrow(table, row_id="id", values=("v",))
    normalized = measures_from_arrow(
        table.set_column(2, "m", pa.array([1.0, 1.0, 1.0, 1.0, 2.0])),
        row_id="id",
        values=("v",),
        mass="m",
        normalize=True,
    )
    # Assert
    np.testing.assert_allclose(uniform.measures.masses[1, :2], [0.5, 0.5])
    np.testing.assert_allclose(normalized.measures.masses[0, :3], [0.25, 0.25, 0.5])
    with pytest.raises(ValueError, match="sum to one"):
        measures_from_arrow(
            table.set_column(2, "m", pa.array([1.0] * 5)),
            row_id="id",
            values=("v",),
            mass="m",
        )


@pytest.mark.integration
@pytest.mark.contract
def test_long_measures_feed_adapters_only_with_exact_identities() -> None:
    """Table adapters accept a long-format column that equals the row ids."""
    # Arrange
    column = measures_from_arrow(_long_table(), row_id="id", values=("v",))
    wide = pa.table({"id": ["a", "b"], "x": [1.0, 2.0]})
    # Act
    data = from_arrow(wide, numeric=("x",), row_id="id", measures={"d": column})
    # Assert
    assert data.measure("d").n_observations == len(data.row_ids)
    with pytest.raises(ValueError, match="identities must equal"):
        from_arrow(
            wide.take([1, 0]), numeric=("x",), row_id="id", measures={"d": column}
        )
    with pytest.raises(ValueError, match="identities must equal"):
        from_polars(pl.from_arrow(wide), numeric=("x",), measures={"d": column})


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.contract
@examples(15)
@given(case=table_cases(), sliced=st.booleans(), order=st.permutations(tuple(range(8))))
def test_chunked_numeric_labels_and_selection_preserve_identity(
    case: TableCase, order: tuple[int, ...], *, sliced: bool
) -> None:
    """Copies preserve represented precision, labels and explicit selected ids."""
    table = arrow_source(case, sliced=sliced)
    assert isinstance(table, pa.Table)
    labels = tuple("a" if index % 2 else "b" for index in range(8))
    ids = tuple(f"row{index}" for index in range(8))
    table = table.append_column("label", pa.array(labels)).append_column(
        "id", pa.array(ids)
    )
    data = from_arrow(
        table, numeric=("x", "y", "z"), categorical=("label",), row_id="id", copy=True
    )
    expected = {name: values.copy() for name, values in case.columns.items()}
    scribble(case)
    selected = data.take(order)
    assert selected.row_ids == tuple(ids[index] for index in order)
    assert selected.categorical("label") == tuple(labels[index] for index in order)
    for name in ("x", "y", "z"):
        assert data.numeric(name).dtype == expected[name].dtype
        np.testing.assert_array_equal(
            selected.numeric(name), expected[name][list(order)]
        )
