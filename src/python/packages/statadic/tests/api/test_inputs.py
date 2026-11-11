"""Regression input adapters preserve roles and prediction semantics."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statadic.api.artifacts import load_predictor
from statadic.api.data import Observations
from statadic.api.design import Categorical, Intercept, Linear
from statadic.api.inputs import ArrowStream, as_observations
from statadic.api.regression import Observed, Poisson, PoissonResult, Regression

from .._budgets import examples
from .._typing import ill_typed
from .._validation.tables import as_source, scribble, table_cases

if TYPE_CHECKING:
    from pathlib import Path

    from .._validation.tables import TableCase


@pytest.mark.integration
def test_mapping_fit_and_prediction_roundtrip(tmp_path: Path) -> None:
    """Named mappings and scalar exposure retain prediction parity."""
    data = {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1, 2, 1, 3, 2, 4],
        "hours": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    result = Regression(
        Observed("y"), Intercept() + Linear("x"), exposure="hours", law=Poisson()
    ).estimate(data)
    assert isinstance(result, PoissonResult)
    result.save_predictor(tmp_path / "fit")
    loaded = load_predictor(tmp_path / "fit")
    for predictor in (result, loaded):
        expected = predictor.predict(
            Observations({"x": jnp.array([1.0, 2.0]), "hours": jnp.array([8.0, 8.0])}),
            kind="mean",
        )
        np.testing.assert_allclose(
            predictor.predict({"x": [1.0, 2.0], "hours": 8.0}, kind="mean"), expected
        )
        np.testing.assert_allclose(
            predictor.predict({"x": [1.0, 2.0]}, kind="rate"),
            predictor.predict({"x": [1.0, 2.0], "hours": 8.0}, kind="rate"),
        )
        with pytest.raises(ValueError, match=r"Unknown.*expected"):
            predictor.predict({"x": [1.0], "other": [1.0]})
        with pytest.raises(ValueError, match="length"):
            predictor.predict({"x": [1.0, 2.0], "hours": [1.0]})
    with pytest.raises(ValueError, match="scalar"):
        Regression("y", Intercept() + Linear("x")).estimate({"x": [1.0, 2.0], "y": 1.0})


@pytest.mark.component
@pytest.mark.contract
def test_mapping_categorical_and_precision() -> None:
    """Categorical levels and precision validation survive mapping input."""
    fit = Regression("y", Intercept() + Categorical("group")).estimate(
        {"y": [1.0, 2.0, 3.0, 4.0], "group": [1, 1, 2, 2]}
    )
    assert fit.predict({"group": [1, 2]}).shape == (2,)
    with pytest.raises(ValueError, match="Unseen"):
        fit.predict({"group": [3]})
    with jax.enable_x64(new_val=False), pytest.raises(ValueError, match=r"x.*x64"):
        as_observations(
            {"x": np.array([1.0], dtype=np.float64)},
            {"x": "numeric"},
            broadcast_scalars=True,
        )


@pytest.mark.integration
@pytest.mark.contract
def test_frame_input_filters_unrelated_and_lazy() -> None:
    """Only declared frame columns are used and lazy frames stay lazy."""
    pl = pytest.importorskip("polars")
    pa = pytest.importorskip("pyarrow")
    frame = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0], "unused": [None] * 4}
    ).cast({"x": pl.Float32, "y": pl.Float32})
    for table in (frame, frame.to_arrow()):
        fit = Regression("y", Intercept() + Linear("x")).estimate(table)
        np.testing.assert_allclose(fit.predict(table), [1.0, 2.0, 3.0, 4.0], rtol=1e-5)
    with pytest.raises(TypeError):
        Regression("y", Intercept() + Linear("x")).estimate(frame.lazy())
    table = pa.table({"x": pa.array([1.0, 2.0], type=pa.float32())})
    data = as_observations(table, {"x": "numeric"}, broadcast_scalars=True)
    assert (
        data.numeric("x").unsafe_buffer_pointer()
        == table.column("x").chunk(0).buffers()[1].address
    )


@pytest.mark.component
@pytest.mark.contract
def test_cluster_extras_and_column_named_kind() -> None:
    """Extra mapping labels survive fitting and names never become keywords."""
    from statadic.api.inference import Clustered  # noqa: PLC0415

    x = np.arange(8, dtype=np.float32)
    fit = Regression("y", Intercept() + Linear("kind")).estimate(
        {
            "kind": x,
            "y": x + np.array([0.0, 1.0] * 4),
            "firm": np.array(["a", "a", "b", "b", "c", "c", "d", "d"]),
        }
    )
    assert fit.infer(covariance=Clustered("firm")) is not None
    assert fit.predict({"kind": 1.0}).shape == (1,)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.regression
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "offset"])
def test_arrow_fit_owns_deferred_and_selected_backing_buffers(offset: int) -> None:
    """Caller mutation cannot change retained roles, diagnostics or inference."""
    pa = pytest.importorskip("pyarrow")
    from statadic.api.inference import Clustered  # noqa: PLC0415

    from .._validation.tables import aligned_array  # noqa: PLC0415

    x = aligned_array(np.arange(8, dtype=np.float32), offset=offset)
    y = aligned_array(x + np.asarray([0.0, 1.0] * 4, dtype=np.float32), offset=offset)
    group = aligned_array(np.repeat(np.arange(4, dtype=np.int32), 2), offset=offset)
    z = aligned_array(np.linspace(0.2, 1.0, 8, dtype=np.float32), offset=offset)
    fit = Regression("y", Intercept() + Linear("x")).estimate(
        pa.table({"x": x, "y": y, "group": group, "z": z})
    )
    retained = fit.context.observations
    before = fit.infer(covariance=Clustered("group")).uncertainty
    expected_y, expected_z = y.copy(), z.copy()
    x[:] = np.nan
    y[:] = 0
    group[:] = 0
    z[:] = np.nan
    assert retained.resolve_grouping("group") == (0, 0, 1, 1, 2, 2, 3, 3)
    np.testing.assert_array_equal(retained.numeric("y"), expected_y)
    np.testing.assert_array_equal(retained.resolve_numeric("z"), expected_z)
    after = fit.infer(covariance=Clustered("group")).uncertainty
    assert bool(before.valid)
    assert bool(after.valid)
    np.testing.assert_array_equal(before.covariance.values, after.covariance.values)


@pytest.mark.component
@pytest.mark.contract
def test_arrow_deferred_chunked_strings_own_their_bytes() -> None:
    """A string column can also alias mutable buffers despite Arrow's API."""
    pa = pytest.importorskip("pyarrow")
    from statadic.api.tables import ArrowColumns  # noqa: PLC0415

    payload = bytearray(b"ab")
    offsets = np.asarray([0, 1, 2], dtype=np.int32)
    strings = pa.Array.from_buffers(
        pa.string(), 2, [None, pa.py_buffer(offsets), pa.py_buffer(payload)]
    )
    source = ArrowColumns(
        pa.table({"firm": pa.chunked_array([strings.slice(0, 1), strings.slice(1, 1)])})
    )
    payload[:] = b"zz"
    assert source.labels("firm") == ("a", "b")


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("source", ["mapping", "arrow"])
def test_deferred_numeric_validates_only_retained_source_rows(source: str) -> None:
    """Excluded missing values cannot invalidate a selected numeric role."""
    columns = {
        "x": np.asarray([0, 1, 2, 3], dtype=np.float32),
        "z": np.asarray([np.nan, 10, 20, np.nan], dtype=np.float32),
    }
    raw = columns
    if source == "arrow":
        pa = pytest.importorskip("pyarrow")
        raw = pa.table(
            {
                "x": columns["x"],
                "z": pa.array(columns["z"], mask=np.isnan(columns["z"])),
            }
        )
    retained = as_observations(raw, {"x": "numeric"}, broadcast_scalars=False)
    selected = retained.take((2, 1, 3)).take((1, 0))
    resolved = selected.resolve_numeric("z")
    assert resolved.dtype == jnp.float32
    np.testing.assert_array_equal(resolved, [10, 20])
    with pytest.raises(ValueError, match=r"nonfinite|nulls"):
        retained.resolve_numeric("z")


@pytest.mark.component
@pytest.mark.contract
def test_labels_only_source_does_not_acquire_a_new_numeric_protocol() -> None:
    """An unrelated optional method does not change ColumnSource requirements."""

    class ExternalColumns:
        """Existing source extension implementing only the public label contract."""

        names = ("z",)
        n_rows = 4

        def labels(self, name: str) -> tuple[float | None, ...]:
            """Resolve the raw source values before the requested role is checked."""
            assert name == "z"
            return (None, 10.0, 20.0, None)

        def numeric(self, name: str) -> None:
            """Expose an unrelated preexisting operation that must not be called."""
            message = f"Unexpected extension numeric call: {name}"
            raise AssertionError(message)

    selected = Observations({"x": jnp.arange(4)}, source=ExternalColumns()).take((2, 1))
    np.testing.assert_array_equal(selected.resolve_numeric("z"), [20, 10])


@pytest.mark.component
@pytest.mark.contract
def test_deferred_numeric_source_preserves_precision_and_row_selection() -> None:
    """Numeric role resolution shares the owned source and its selected rows."""
    z = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    rows = (
        as_observations(
            {"x": np.ones(4, dtype=np.float32), "z": z},
            {"x": "numeric"},
            broadcast_scalars=False,
        )
        .take((3, 1, 2))
        .take((2, 0))
    )
    z[:] = 0
    with jax.enable_x64(new_val=False), pytest.raises(ValueError, match="x64"):
        rows.resolve_numeric("z")
    with jax.enable_x64(new_val=True):
        np.testing.assert_array_equal(rows.resolve_numeric("z"), [30.0, 40.0])


@pytest.mark.unit
@pytest.mark.contract
def test_explicit_identity_roles_and_conflicts() -> None:
    """Observations pass through and conflicting declared roles are rejected."""
    data = Observations({"x": jnp.array([1.0])})
    assert as_observations(data, {"x": "numeric"}, broadcast_scalars=True) is data
    with pytest.raises(ValueError, match="conflicting"):
        _ = (Linear("x") + Categorical("x")).input_columns


@pytest.mark.integration
@pytest.mark.contract
def test_empty_prediction_mapping_retains_one_row(tmp_path: Path) -> None:
    """Empty mappings support intercepts and preserve required-role diagnostics."""
    fit = Regression("y", Intercept()).estimate({"y": [1.0, 2.0, 3.0, 4.0]})
    fit.save_predictor(tmp_path / "intercept")
    for predictor in (fit, load_predictor(tmp_path / "intercept")):
        np.testing.assert_allclose(predictor.predict({}), [2.5])
    counts = Regression(
        Observed("y"), Intercept(), exposure="hours", law=Poisson()
    ).estimate({"y": [1, 2, 3, 4], "hours": [1.0] * 4})
    assert isinstance(counts, PoissonResult)
    counts.save_predictor(tmp_path / "counts")
    for predictor in (counts, load_predictor(tmp_path / "counts")):
        assert predictor.predict({}, kind="rate").shape == (1,)
        with pytest.raises(KeyError, match=r"hours.*exposure.*mean"):
            predictor.predict({}, kind="mean")


@pytest.mark.component
@pytest.mark.contract
def test_measure_column_retains_identity_and_does_not_broadcast() -> None:
    """Measure-valued columns preserve their observation identity and length."""
    from statadic.api.data import MeasureColumn  # noqa: PLC0415
    from statadic.sample.packed import PackedClouds  # noqa: PLC0415

    measures = MeasureColumn(
        ("a", "b"), PackedClouds.from_clouds([jnp.ones((1, 1)), jnp.ones((2, 1))])
    )
    data = as_observations(
        {"cloud": measures, "x": 2.0},
        {"cloud": "measure", "x": "numeric"},
        broadcast_scalars=True,
    )
    assert data.row_ids == ("a", "b")
    with pytest.raises(ValueError, match="length"):
        as_observations(
            {"cloud": measures, "x": [1.0]},
            {"cloud": "measure", "x": "numeric"},
            broadcast_scalars=True,
        )
    with pytest.raises(TypeError, match="numeric"):
        as_observations(
            {"cloud": measures}, {"cloud": "numeric"}, broadcast_scalars=True
        )


@pytest.mark.integration
@pytest.mark.contract
def test_table_precision_optional_dependencies_and_cluster_hint() -> None:
    """Frame adapters enforce precision, explicit measure and retained clusters."""
    pl = pytest.importorskip("polars")
    pa = pytest.importorskip("pyarrow")
    from statadic.api.inference import Clustered  # noqa: PLC0415
    from statadic.api.tables import from_polars  # noqa: PLC0415

    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0, 5.0],
            "firm": ["a", "a", "b", "b"],
        }
    )
    with jax.enable_x64(new_val=False), pytest.raises(ValueError, match="x64"):
        Regression("y", Intercept() + Linear("x")).estimate(frame)
    with jax.enable_x64(new_val=True):
        fit = Regression("y", Intercept() + Linear("x")).estimate(frame)
        framed = fit.infer(covariance=Clustered("firm")).uncertainty
        explicit = (
            Regression("y", Intercept() + Linear("x"))
            .estimate(from_polars(frame, numeric=("x", "y"), categorical=("firm",)))
            .infer(covariance=Clustered("firm"))
            .uncertainty
        )
        mapped = (
            Regression("y", Intercept() + Linear("x"))
            .estimate(frame.to_dict(as_series=False))
            .infer(covariance=Clustered("firm"))
            .uncertainty
        )
        for other in (explicit, mapped):
            np.testing.assert_allclose(
                framed.covariance.values, other.covariance.values, rtol=1e-12
            )
        assert "firm" in fit.context.available_columns
        with pytest.raises(KeyError, match=r"'frim'.*available: .*firm"):
            fit.infer(covariance=Clustered("frim"))
        assert from_polars(frame, numeric=("x",)).numeric("x").dtype == jnp.float64
    with pytest.raises(TypeError, match="measures_from_arrow"):
        as_observations(
            pa.table({"x": [1]}), {"cloud": "measure"}, broadcast_scalars=True
        )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['polars'] = None; import statadic.api.tables",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.integration
@pytest.mark.contract
def test_unused_frame_columns_are_resolved_only_when_inference_names_them() -> None:
    """Unsupported extra columns cannot break a fit; they fail where named."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    from statadic.api.inference import Clustered  # noqa: PLC0415

    frame = pl.DataFrame(
        {
            "x": pl.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float32),
            "y": pl.Series([1.0, 2.0, 3.0, 5.0, 4.0, 6.0], dtype=pl.Float32),
            "firm": [1, 1, 2, 2, 3, 3],
            "missing": [None, None, None, 1.0, None, None],
            "when": pl.date_range(pl.date(2026, 1, 1), pl.date(2026, 1, 6), eager=True),
            "wide": pl.Series([0.5] * 6, dtype=pl.Float64),
        }
    )
    with jax.enable_x64(new_val=False):
        fit = Regression("y", Intercept() + Linear("x")).estimate(frame)
        clustered = fit.infer(covariance=Clustered("firm"))
        assert dict(clustered.report().metadata)["inference_columns"] == {
            "firm": "grouping"
        }
        for unusable in ("missing", "when", "wide"):
            with pytest.raises(TypeError, match=unusable):
                fit.infer(covariance=Clustered(unusable))


@pytest.mark.component
@pytest.mark.contract
def test_mapping_extras_are_snapshotted_at_fit_time() -> None:
    """Later caller mutation cannot change what clustering resolves."""
    from statadic.api.inference import Clustered  # noqa: PLC0415

    x = np.arange(8, dtype=np.float32)
    firm = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    with jax.enable_x64(new_val=False):
        fit = Regression("y", Intercept() + Linear("x")).estimate(
            {
                "x": x,
                "y": x + np.asarray([0.0, 1.0] * 4, dtype=np.float32),
                "firm": firm,
                "wide": np.zeros(8, dtype=np.float64),
            }
        )
        before = fit.infer(covariance=Clustered("firm")).uncertainty.covariance.values
        firm[:] = "a"
        after = fit.infer(covariance=Clustered("firm")).uncertainty.covariance.values
    np.testing.assert_array_equal(before, after)
    assert "wide" in fit.context.available_columns
    assert "wide" not in fit.context.observations.columns
    with pytest.raises(ValueError, match="Unknown columns"):
        fit.predict({"x": 1.0, "firm": "a"})


@pytest.mark.integration
def test_optional_pandas_frame() -> None:
    """Arrow-compatible pandas frames use the same regression adapter."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {"x": np.arange(4, dtype=np.float32), "y": np.arange(4, dtype=np.float32)}
    )
    fit = Regression("y", Intercept() + Linear("x")).estimate(frame)
    np.testing.assert_allclose(fit.predict(frame), frame["y"], atol=1e-5)
    from statadic.api.inference import Clustered  # noqa: PLC0415

    clustered = frame.assign(
        y=np.asarray([0.0, 1.5, 2.0, 3.5], dtype=np.float32), firm=["a", "a", "b", "b"]
    )
    framed = Regression("y", Intercept() + Linear("x")).estimate(clustered)
    mapped = Regression("y", Intercept() + Linear("x")).estimate(
        {name: clustered[name].to_numpy() for name in ("x", "y", "firm")}
    )
    np.testing.assert_allclose(
        framed.infer(covariance=Clustered("firm")).uncertainty.covariance.values,
        mapped.infer(covariance=Clustered("firm")).uncertainty.covariance.values,
    )


@pytest.mark.component
@pytest.mark.contract
def test_iv_mapping_and_categorical_string_prediction() -> None:
    """IV and categorical models use the same named-input rules."""
    from statadic.api.inference import Clustered  # noqa: PLC0415
    from statadic.api.regression import Regression  # noqa: PLC0415

    z = np.arange(8, dtype=np.float32)
    model = Regression(
        "y", Intercept() + Linear("x"), endogenous=("x",), instruments=("z",)
    )
    noise = np.asarray([0.3, -0.2, 0.1, 0.4, -0.3, 0.2, -0.1, 0.0], dtype=np.float32)
    fit = model.estimate(
        {"x": z + 1 + noise, "z": z, "y": 2 * z + 1, "firm": list("aabbccdd")}
    )
    assert fit.predict({"x": 1.0}).shape == (1,)
    clustered = fit.infer(covariance=Clustered("firm")).report()
    assert dict(clustered.metadata)["inference_columns"] == {"firm": "grouping"}
    with pytest.raises(ValueError, match="scalar"):
        model.estimate({"x": 1.0, "z": z, "y": z})
    categorical = Regression("y", Intercept() + Categorical("group")).estimate(
        {"y": [1.0, 2.0, 3.0, 4.0], "group": ["a", "a", "b", "b"]}
    )
    assert categorical.predict({"group": "a"}).shape == (1,)


@pytest.mark.integration
@pytest.mark.contract
def test_iv_numeric_design_column_can_also_be_cluster() -> None:
    """Numeric design and grouping roles retain the same integer column."""
    from statadic.api.regression import Regression  # noqa: PLC0415

    year = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    z = np.arange(8, dtype=np.float32)
    values = {"year": year, "x": z + 1, "z": z, "y": 2 * z + 1}
    model = Regression(
        "y",
        Intercept() + Linear("year") + Linear("x"),
        clusters="year",
        endogenous=("x",),
        instruments=("z",),
    )
    explicit = Observations(values)
    prepared = [model.prepare(explicit), model.prepare(values)]
    pa = pytest.importorskip("pyarrow")
    prepared.append(model.prepare(pa.table(values)))
    n_years = len(set(year.tolist()))
    for item in prepared:
        assert item.problem.n_clusters == n_years
        np.testing.assert_array_equal(item.problem.groups, prepared[0].problem.groups)
    with pytest.raises(ValueError, match="integer keys"):
        model.prepare({**values, "year": year.astype(np.float32)})


@pytest.mark.component
@pytest.mark.contract
def test_numpy_categorical_scalars_are_host_labels() -> None:
    """Scalar integer labels broadcast like vectors; booleans remain invalid."""
    fit = Regression("y", Intercept() + Categorical("group")).estimate(
        {"y": [1.0, 2.0, 3.0, 4.0], "group": [1, 1, 2, 2]}
    )
    np.testing.assert_array_equal(
        fit.predict({"group": np.int32(1)}), fit.predict({"group": 1})
    )
    with pytest.raises(TypeError, match="integer/string"):
        fit.predict({"group": np.bool_(1)})


@pytest.mark.unit
@pytest.mark.contract
def test_scalar_broadcast_checks_once_and_broadcasts_in_jax() -> None:
    """A fixed covariate is checked once, then filled as one JAX vector."""
    data = as_observations(
        {"x": jnp.arange(4.0), "exposure": 2.5},
        {"x": "numeric", "exposure": "numeric"},
        broadcast_scalars=True,
    )
    assert isinstance(data.numeric("exposure"), jax.Array)
    np.testing.assert_array_equal(data.numeric("exposure"), [2.5] * 4)
    if not jax.config.x64_enabled:
        with pytest.raises(ValueError, match="x64"):
            as_observations(
                {"x": jnp.arange(4.0), "exposure": np.float64(2.5)},
                {"x": "numeric", "exposure": "numeric"},
                broadcast_scalars=True,
            )


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.contract
@examples(15)
@given(
    case=table_cases(),
    kind=st.sampled_from(
        ["mapping-numpy", "mapping-jax", "arrow", "arrow-sliced", "polars"]
    ),
)
def test_table_adapters_preserve_estimates_and_own_inputs(
    case: TableCase, kind: str
) -> None:
    """Chunking and caller mutation do not change fitted or deferred values."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    from statadic.api.inference import Clustered  # noqa: PLC0415

    expected = {name: values.copy() for name, values in case.columns.items()}
    model = Regression.ols("y", Intercept() + Linear("x"))
    reference = model.estimate(expected)
    source = as_source(case, kind)
    assert isinstance(source, (Mapping, ArrowStream))
    fit = model.estimate(source)
    before = np.asarray(
        fit.infer(covariance=Clustered("group")).uncertainty.covariance.values
    ).copy()
    np.testing.assert_allclose(
        fit.core.parameters, reference.core.parameters, rtol=3e-5, atol=3e-6
    )
    scribble(case)
    for name in ("x", "y"):
        np.testing.assert_array_equal(
            fit.context.observations.numeric(name), expected[name]
        )
    np.testing.assert_array_equal(
        fit.context.observations.resolve_numeric("z"), expected["z"]
    )
    np.testing.assert_array_equal(
        fit.infer(covariance=Clustered("group")).uncertainty.covariance.values, before
    )


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.contract
@examples(8)
@given(case=table_cases())
def test_lazy_table_requires_explicit_collection(case: TableCase) -> None:
    """Generated adapter coverage retains the explicit eager-input boundary."""
    pytest.importorskip("polars")
    with pytest.raises(TypeError):
        Regression.ols("y", Intercept() + Linear("x")).estimate(
            ill_typed(as_source(case, "polars-lazy"))
        )
    with pytest.raises(TypeError, match="collect"):
        as_observations(
            as_source(case, "polars-lazy"),
            {"x": "numeric", "y": "numeric"},
            broadcast_scalars=False,
        )
