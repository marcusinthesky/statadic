"""Panel host classes reach the native panel bricks' exact fits."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from inspect import unwrap
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.base import Model, RegressionResult, Result
from statadic.api.data import MeasureColumn, Observations, PanelLayout, RelativePrecision
from statadic.api.design import Categorical, DistanceToLandmarks, Intercept, Linear
from statadic.api.inputs import as_observations
from statadic.api.ordination import Ordination
from statadic.api.panel import (
    FamaMacBethCovariance,
    PanelCovariance,
    PanelResult,
    PreparedPanel,
    prepare_spatial_panel_data,
)
from statadic.api.regression import Regression
from statadic.api.representation import Energy
from statadic.api.spatial.graphs import PanelData
from statadic.estimation.linear import LeastSquares
from statadic.estimation.panel import (
    Between,
    FamaMacBeth,
    FamaMacBethFit,
    FirstDifference,
    PanelFit,
    PanelProblem,
    Pooled,
    RandomEffects,
    Within,
)
from statadic.inference.panel_covariance import fama_macbeth_covariance, panel_covariance
from statadic.operators.panel import prepare_panel_index
from statadic.operators.spatial import prepare_spatial_weights
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from statadic.api.design import Term
    from statadic.api.ordination import OrdinationResult, PreparedOrdination
    from statadic.api.panel import PanelEstimator


@pytest.mark.integration
def test_panel_layout_effects_covariance_and_known_effect_prediction() -> None:
    """Actual time gaps and absorbed degrees of freedom survive the facade."""
    rng = np.random.default_rng(38)
    entity, time = (np.repeat(np.arange(12), 4), np.tile([0, 1, 3, 6], 12))
    x = rng.normal(size=48)
    y = 0.4 + 0.6 * x + rng.normal(size=12)[entity] + rng.normal(size=48) * 0.3
    data = Observations(
        {"x": x, "y": y, "time": time},
        labels={"entity": tuple(f"unit-{value}" for value in entity)},
    )
    result = Regression(
        "y",
        Linear("x"),
        effects="unit",
        structure=PanelLayout("entity", "time"),
    ).estimate(data)
    native = Within().estimate(
        PanelProblem(
            result.design.transform(data),
            y,
            prepare_panel_index(entity, time),
            effects="unit",
            parameter_names=result.design.column_names,
        )
    )
    assert result.core.valid
    np.testing.assert_allclose(result.core.parameters, native.parameters)
    np.testing.assert_array_equal(result.core.df_resid, native.df_resid)
    np.testing.assert_allclose(
        result.predict(data, kind="conditional"), native.fitted_values + native.effects
    )
    uncertainty = result.infer(covariance=PanelCovariance(method="cluster-entity"))
    np.testing.assert_allclose(
        uncertainty.uncertainty.covariance.values,
        panel_covariance(native, method="cluster-entity").covariance.values,
    )
    targets = {row.kind: row for row in result.prediction_options().rows}
    assert targets["xb"].required_columns == ("x",)
    assert targets["conditional"].required_columns == ("x", "entity")
    unknown = Observations(
        {"x": jnp.asarray([0.4]), "time": jnp.asarray([7])}, labels={"entity": ["new"]}
    )
    assert np.isfinite(result.predict(unknown)).all()
    with pytest.raises(ValueError, match="known effect"):
        result.predict(unknown, kind="conditional")
    two_way = Regression(
        "y",
        Linear("x"),
        structure=PanelLayout("entity", "time"),
        effects="two-way",
    ).estimate(data)
    with pytest.raises(ValueError, match="known effect"):
        two_way.predict(
            Observations(
                {"x": jnp.asarray([0.4]), "time": jnp.asarray([7])},
                labels={"entity": ["unit-0"]},
            ),
            kind="conditional",
        )


def _panel_data() -> tuple[Observations, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(91)
    entity, time = np.repeat(np.arange(12), 5), np.tile(np.arange(5), 12)
    x = rng.normal(size=60)
    y = 0.4 + 0.6 * x + rng.normal(size=12)[entity] + rng.normal(size=60) * 0.3
    w = rng.uniform(0.5, 2.0, 60)
    # Zero-padded labels keep host level codes equal to the native entity codes.
    data = Observations(
        {"x": x, "y": y, "w": w, "time": time},
        labels={"entity": tuple(f"unit-{value:02d}" for value in entity)},
    )
    return data, entity, time, w


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["xb", "conditional"])
def test_panel_prediction_accepts_original_panel_data(
    kind: Literal["xb", "conditional"],
) -> None:
    """The estimate-time container preserves row predictions and fitted effects."""
    data, _, _, _ = _panel_data()
    layout = PanelLayout("entity", "time")
    original = PanelData(data, layout)
    result = Regression("y", Linear("x"), structure=layout, effects="unit").estimate(
        original
    )
    assert isinstance(result, PanelResult)
    np.testing.assert_allclose(
        result.predict(original, kind=kind), result.predict(data, kind=kind)
    )
    query = PanelData(data, layout, outcomes=data.take([7, 3, 1]))
    np.testing.assert_allclose(
        result.predict(query, kind=kind),
        result.predict(data, kind=kind)[jnp.array([7, 3, 1])],
    )
    with pytest.raises(ValueError, match="fitted entity/time layout"):
        result.predict(PanelData(data, PanelLayout("other", "time")), kind=kind)


def _panel_predictor_copies(
    role: str,
) -> tuple[PanelData, Term, Observations, Observations]:
    entity, time = np.tile(np.arange(4), 3), np.repeat(np.arange(3), 4)
    x = np.asarray([0, 1, 4, 2, 3, 5, 1, 6, 7, 2, 8, 4], dtype=np.float32)
    nodes = Observations(
        {"entity": entity, "time": time, "x": x},
        labels={"group": tuple(int(unit) % 2 for unit in entity)},
        measures={
            "cloud": MeasureColumn(
                tuple(range(len(x))),
                PackedClouds.from_array(
                    jnp.asarray(np.stack((x, x + 1), axis=1)[..., None])
                ),
            )
        },
    )
    positions = [9, 2, 5, 0, 10, 7, 3, 4]
    selected = nodes.take(positions)
    outcomes = Observations(
        {
            "entity": entity[positions],
            "time": time[positions],
            "y": x[positions]
            + np.asarray([0, 1, -1, 2, 1, 0, -2, 1], dtype=np.float32),
        },
        row_ids=tuple(f"outcome-{index}" for index in positions),
    )
    term: Term
    if role == "numeric":
        term = Linear("x")
        matching = replace(
            outcomes, columns={**outcomes.columns, "x": selected.numeric("x")}
        )
        conflicting = replace(
            outcomes, columns={**outcomes.columns, "x": selected.numeric("x") + 100}
        )
    elif role == "categorical":
        term = Categorical("group")
        matching = replace(outcomes, labels={"group": selected.categorical("group")})
        conflicting = replace(
            outcomes,
            labels={
                "group": tuple(
                    1 - int(value) for value in selected.categorical("group")
                )
            },
        )
    else:
        term = DistanceToLandmarks("cloud", Energy(), landmarks=(0,))
        clouds = selected.measure("cloud")
        matching = replace(
            outcomes, measures={"cloud": MeasureColumn(outcomes.row_ids, clouds)}
        )
        conflicting = replace(
            outcomes,
            measures={
                "cloud": MeasureColumn(
                    outcomes.row_ids,
                    clouds.with_mass_logits(
                        jnp.tile(jnp.asarray([0.0, 1.0]), (len(positions), 1))
                    ),
                )
            },
        )
    layout = PanelLayout("entity", "time")
    graph = prepare_spatial_weights(np.roll(np.eye(4), 1, axis=1))
    return (
        PanelData(nodes, layout, {"neighbors": graph}, outcomes),
        term,
        matching,
        conflicting,
    )


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.parametrize("role", ["numeric", "categorical", "measure"])
def test_panel_predictor_copies_match_full_nodes_at_fit_and_prediction(
    role: str,
) -> None:
    """Optional copies obey the same keyed roles in ordinary and spatial panels."""
    data, term, matching, conflicting = _panel_predictor_copies(role)
    model = Regression("y", Intercept() + term, structure=data.panel)
    prepared = model.prepare(data)
    assert isinstance(prepared, PreparedPanel)
    expected = prepared.design.transform(data.nodes)[
        jnp.asarray([9, 2, 5, 0, 10, 7, 3, 4])
    ]
    np.testing.assert_array_equal(prepared.values, expected)
    matched = model.prepare(replace(data, outcomes=matching))
    assert isinstance(matched, PreparedPanel)
    np.testing.assert_array_equal(matched.values, expected)
    with pytest.raises(ValueError, match="conflicts"):
        model.prepare(replace(data, outcomes=conflicting))
    result = model.estimate_prepared(prepared).require_valid()
    assert isinstance(result, PanelResult)
    for rows in (data.spatial.outcomes, matching):
        np.testing.assert_allclose(
            result.predict(replace(data, outcomes=rows)),
            expected @ result.core.parameters,
        )
    with pytest.raises(ValueError, match="conflicts"):
        result.predict(replace(data, outcomes=conflicting))
    # Spatial preparation must validate the same duplicates before any solve.
    for rows in (data.spatial.outcomes, matching):
        aligned = prepare_spatial_panel_data(
            replace(data, outcomes=rows).spatial,
            data.panel,
            data.period_graphs("neighbors"),
            prepared.design,
        )
        for values, indices in zip(aligned.values, aligned.node_rows, strict=True):
            np.testing.assert_array_equal(
                values, prepared.design.transform(data.nodes)[jnp.asarray(indices)]
            )
    with pytest.raises(ValueError, match="conflicts"):
        prepare_spatial_panel_data(
            replace(data, outcomes=conflicting).spatial,
            data.panel,
            data.period_graphs("neighbors"),
            prepared.design,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("brick", "intercept", "effects"),
    [
        (Pooled(), True, "pooled"),
        (Within(), False, "unit"),
        (Within(), False, "two-way"),
        (Between(), True, "pooled"),
        (Between(reweight=True), True, "pooled"),
        (FirstDifference(), False, "pooled"),
        (RandomEffects(), True, "random"),
        (RandomEffects(small_sample=True), True, "random"),
        (FamaMacBeth(), True, "pooled"),
    ],
)
def test_panel_regression_reaches_each_brick_exactly(
    brick: PanelEstimator,
    effects: Literal["pooled", "unit", "two-way", "random"],
    *,
    intercept: bool,
) -> None:
    """Each class/brick pair agrees exactly; covariance follows the native owner."""
    data, entity, time, w = _panel_data()
    terms = Intercept() + Linear("x") if intercept else Linear("x")
    model = Regression(
        "y",
        terms,
        structure=PanelLayout("entity", "time"),
        estimator=brick,
        effects=effects,
        precision=RelativePrecision("w"),
    )
    result = model.estimate(data)
    names = result.design.column_names
    native = brick.estimate(
        PanelProblem(
            result.design.transform(data),
            data.numeric("y"),
            prepare_panel_index(entity, time),
            w,
            effects=effects,
            intercept_column=0 if intercept else None,
            parameter_names=names,
        )
    )
    assert isinstance(model, Regression)
    assert isinstance(model, Model)
    assert isinstance(result, RegressionResult)
    assert isinstance(result, Result)
    assert result.estimator == brick
    assert result.core.valid
    np.testing.assert_array_equal(result.core.parameters, native.parameters)
    report = result.report()
    assert tuple(row.name for row in report.parameters) == names
    assert dict(report.metadata)["estimator"] == type(brick).__name__
    np.testing.assert_array_equal(
        [m.value for m in report.metrics if m.name == "df_resid"], [native.df_resid]
    )
    if isinstance(native, FamaMacBethFit):
        assert all(not row.available for row in result.prediction_options().rows)
        np.testing.assert_array_equal(
            result.infer(
                covariance=FamaMacBethCovariance(bandwidth=2)
            ).uncertainty.covariance.values,
            fama_macbeth_covariance(native, bandwidth=2).covariance.values,
        )
        with pytest.raises(ValueError, match="require FamaMacBethCovariance"):
            result.infer(covariance=PanelCovariance())
        with pytest.raises(ValueError, match="no supported panel prediction"):
            result.predict(data)
        with pytest.raises(ValueError, match="no supported panel prediction"):
            result.predict(data, kind="conditional")
        return
    np.testing.assert_array_equal(
        result.infer(
            covariance=PanelCovariance(method="heteroskedastic")
        ).uncertainty.covariance.values,
        panel_covariance(native, method="heteroskedastic").covariance.values,
    )
    with pytest.raises(ValueError, match="require PanelCovariance"):
        result.infer(covariance=FamaMacBethCovariance())
    np.testing.assert_array_equal(result.predict(data), native.fitted_values)
    if isinstance(brick, (Within, RandomEffects)):
        np.testing.assert_allclose(
            result.predict(data, kind="conditional"),
            native.fitted_values + native.effects,
        )
    else:
        with pytest.raises(ValueError, match="estimates no observation-aligned"):
            result.predict(data, kind="conditional")


@pytest.mark.integration
def test_panel_regression_rejects_fixed_effect_intercept() -> None:
    """Host fixed effects have no redundant explicit intercept normalization."""
    data, _, _, _ = _panel_data()
    layout = PanelLayout("entity", "time")
    for effects in ("unit", "time", "two-way"):
        with pytest.raises(ValueError, match="without an explicit intercept"):
            Regression(
                "y", Intercept() + Linear("x"), structure=layout, effects=effects
            ).prepare(data)
    result = Regression("y", Linear("x"), structure=layout, effects="unit").estimate(
        data
    )
    assert result.estimator == Within()
    assert result.design.intercept_columns == ()


@pytest.mark.component
@pytest.mark.contract
def test_structural_panel_estimator_preserves_capability_boundaries() -> None:
    """Custom bricks can fit without claiming known conditional effects."""
    problems: list[PanelProblem] = []

    class PooledBrick:
        def estimate(self, problem: PanelProblem) -> PanelFit:
            problems.append(problem)
            return Pooled().estimate(problem)

    data, _, _, _ = _panel_data()
    brick = PooledBrick()
    result = Regression(
        "y",
        Intercept() + Linear("x"),
        structure=PanelLayout("entity", "time"),
        estimator=brick,
    ).estimate(data)
    assert result.estimator is brick
    assert len(problems) == 1
    assert isinstance(result.core, PanelFit)
    np.testing.assert_array_equal(result.predict(data), result.core.fitted_values)
    metadata = dict(result.report().metadata)
    assert metadata["entity_effects"] is None
    assert metadata["time_effects"] is None
    with pytest.raises(ValueError, match="estimates no observation-aligned"):
        result.predict(data, kind="conditional")


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "model",
    [
        Regression("y", Intercept() + Linear("x"), estimator=LeastSquares()),
        Regression.ols("y", Intercept() + Linear("x")),
        Regression.wls(
            "y", Intercept() + Linear("x"), precision=RelativePrecision("w")
        ),
    ],
    ids=["explicit", "ols", "wls"],
)
def test_rowwise_least_squares_rejects_panel_preparation(model: Regression) -> None:
    """Resolving PanelData cannot reinterpret an explicitly selected row solver."""
    data, _, _, _ = _panel_data()
    with pytest.raises(
        TypeError, match="panel data requires a compatible panel estimator"
    ):
        model.prepare(PanelData(data, PanelLayout("entity", "time")))


@pytest.mark.integration
@pytest.mark.contract
def test_invalid_panel_native_result_is_rejected_without_annotation_hook() -> None:
    """Ordinary runtime validates custom result families before exposing reports."""
    script = """
import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.spatial.graphs import PanelData
from statadic.api.data import PanelLayout
from statadic.api.regression import Regression
from statadic.estimation.linear import LeastSquares, LinearProblem

rows = Observations({
    "x": jnp.array([0., 1., 2., 1., 3., 4.]),
    "y": jnp.array([1., 2., 2.5, 2., 4., 4.5]),
    "time": jnp.array([0, 1, 2, 0, 1, 2]),
}, labels={"unit": ["a", "a", "a", "b", "b", "b"]})
data = PanelData(rows, PanelLayout("unit", "time"))
terms = Intercept() + Linear("x")

try:
    Regression("y", terms, estimator=LeastSquares()).prepare(data)
except TypeError as error:
    assert "compatible panel estimator" in str(error)
else:
    raise AssertionError("row-wise solver accepted as panel estimator")

class WrongFamily:
    def estimate(self, problem):
        return LeastSquares().estimate(LinearProblem(
            problem.design, problem.outcome,
            parameter_names=problem.parameter_names,
        ))

try:
    Regression("y", terms, estimator=WrongFamily()).estimate(data)
except TypeError as error:
    assert "native PanelFit or FamaMacBethFit" in str(error)
else:
    raise AssertionError("LinearFit wrapped as PanelResult")
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


@pytest.mark.integration
@pytest.mark.parametrize(
    ("effects", "brick"),
    [
        ("pooled", Pooled()),
        ("unit", Within()),
        ("time", Within()),
        ("two-way", Within()),
    ],
)
def test_panel_ols_effects_match_composed_brick(
    effects: Literal["pooled", "unit", "time", "two-way"], brick: PanelEstimator
) -> None:
    """Every convenience option reaches the brick that owns its effect meaning."""
    data, _, _, _ = _panel_data()
    panel = PanelLayout("entity", "time")
    terms = Intercept() + Linear("x") if effects == "pooled" else Linear("x")
    convenience = Regression("y", terms, effects=effects, structure=panel).estimate(
        data
    )
    composed = Regression(
        "y", terms, structure=panel, effects=effects, estimator=brick
    ).estimate(data)
    assert convenience.estimator == composed.estimator
    np.testing.assert_array_equal(convenience.core.parameters, composed.core.parameters)
    np.testing.assert_array_equal(convenience.predict(data), composed.predict(data))
    if effects == "pooled":
        for result in (convenience, composed):
            with pytest.raises(ValueError, match="estimates no observation-aligned"):
                result.predict(data, kind="conditional")
        return
    np.testing.assert_allclose(
        convenience.predict(data, kind="conditional"),
        composed.predict(data, kind="conditional"),
    )


@pytest.mark.component
@pytest.mark.contract
def test_panel_rejects_unknown_effects_and_prediction_kind() -> None:
    """Unknown option names fail explicitly."""
    panel = PanelLayout("entity", "time")
    with pytest.raises(ValueError, match="Unknown panel effects"):
        unwrap(Regression.__init__)(
            object.__new__(Regression),
            "y",
            Linear("x"),
            structure=panel,
            effects="invalid",
        )
    data, _, _, _ = _panel_data()
    result = Regression("y", Linear("x"), effects="unit", structure=panel).estimate(
        data
    )
    with pytest.raises(ValueError, match="kind must be"):
        unwrap(type(result).predict)(result, data, kind="invalid")


@pytest.mark.component
@pytest.mark.contract
def test_panel_prepare_checks_entity_and_integer_time_roles() -> None:
    """Successful preparation means the layout's roles are usable for fitting."""
    data, entity, time, w = _panel_data()
    float_time = Observations(
        {
            "x": data.numeric("x"),
            "y": data.numeric("y"),
            "w": w,
            "time": time.astype(float),
        },
        labels={"entity": tuple(f"unit-{value:02d}" for value in entity)},
    )
    no_entity = Observations(
        {"x": data.numeric("x"), "y": data.numeric("y"), "time": time}
    )
    panel = PanelLayout("entity", "time")
    terms = Linear("x")
    for model in (
        Regression("y", terms, effects="unit", structure=panel),
        Regression("y", terms, effects="unit", structure=panel),
    ):
        with pytest.raises(ValueError, match="integer key distances"):
            model.prepare(float_time)
        with pytest.raises(KeyError):
            model.prepare(no_entity)
        assert model.prepare(data).context.observations is data


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("integer_entity", [False, True])
def test_projecting_declared_panel_columns_suffices_to_fit(
    *, integer_entity: bool
) -> None:
    """The advertised panel roles are enough to prepare and fit from a wide table."""
    data, entity, _, _ = _panel_data()
    model = Regression(
        "y",
        Linear("x"),
        precision=RelativePrecision("w"),
        effects="unit",
        structure=PanelLayout("entity", "time"),
    )
    assert model.input_columns == {
        "x": "numeric",
        "y": "numeric",
        "time": "numeric",
        "w": "numeric",
        "entity": "categorical",
    }
    wide: dict[str, object] = {
        **dict(data.columns),
        "entity": entity.tolist() if integer_entity else data.categorical("entity"),
        "unused": np.zeros(data.n_rows),
    }
    projected = {name: wide[name] for name in model.input_columns}
    restricted = as_observations(
        projected, model.input_columns, broadcast_scalars=False
    )
    expected = model.estimate(data)
    actual = model.estimate(restricted)
    np.testing.assert_allclose(actual.core.parameters, expected.core.parameters)


@pytest.mark.unit
@pytest.mark.contract
def test_declared_panel_effects_cannot_disagree_with_native_estimator() -> None:
    """A supplied estimator cannot silently discard the requested mean effects."""
    layout = PanelLayout("entity", "time")
    with pytest.raises(ValueError, match="Pooled requires"):
        Regression(
            "y", Linear("x"), structure=layout, effects="unit", estimator=Pooled()
        )
    with pytest.raises(ValueError, match="explicit"):
        Regression("y", Linear("x"), structure=layout, estimator=Within())


@pytest.mark.component
@pytest.mark.contract
def test_default_panel_effects_are_pooled() -> None:
    """An omitted model role is pooled and is not inferred from a solver."""
    data, _, _, _ = _panel_data()
    model = Regression(
        "y", Intercept() + Linear("x"), structure=PanelLayout("entity", "time")
    )
    prepared = model.prepare(data)
    assert model.inspect(prepared).effects == "pooled"


@pytest.mark.component
@pytest.mark.contract
def test_fixed_effect_panel_fits_training_ordination_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intercept validation inspects retained state and prediction never refits it."""
    data, _, _, _ = _panel_data()
    original = Ordination.estimate_prepared
    calls: list[tuple[int | str, ...]] = []

    def counted(self: Ordination, prepared: PreparedOrdination) -> OrdinationResult:
        calls.append(prepared.row_ids)
        return original(self, prepared)

    monkeypatch.setattr(Ordination, "estimate_prepared", counted)
    components = Ordination.pca(("x",), dim=1)
    model = Regression(
        "y", components, structure=PanelLayout("entity", "time"), effects="unit"
    )
    prepared = model.prepare(data)
    assert len(calls) == 1
    result = model.estimate_prepared(prepared)
    result.predict(data)
    _ = result.design.ordination("ordination").explained_variance
    assert len(calls) == 1
