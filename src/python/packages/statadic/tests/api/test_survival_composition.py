"""Named spatial survival uses the same graph, likelihood and prediction state."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statadic.api.survival as survival_family
from statadic.api.data import MeasureColumn, Observations, PanelLayout
from statadic.api.design import (
    Categorical,
    CyclicSpline,
    DistanceToLandmarks,
    Intercept,
    Linear,
)
from statadic.api.inference import ModelBased
from statadic.api.regression import Regression
from statadic.api.representation import Energy
from statadic.api.spatial.graphs import PanelData, SpatialError, SpatialLag, SpatialX
from statadic.api.survival import PreparedSurvival, RightCensored, SpatialSurvivalData
from statadic.estimation.survival import (
    SpatialSurvivalML,
    spatial_survival_log_likelihood,
)
from statadic.model.survival import LogNormalAFT, WeibullAFT
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from statadic.api.design import Term


def _panel() -> tuple[PanelData, tuple[IntegrationDesign, ...]]:
    rng = np.random.default_rng(1027)
    n, periods = 6, 4
    unit = np.tile(np.arange(n), periods)
    time = np.repeat(np.arange(periods), n)
    phase = rng.uniform(size=len(unit))
    x = rng.normal(size=len(unit))
    nodes = Observations({"unit": unit, "year": time, "phase": phase, "x": x})
    rows = rng.permutation(len(unit))[1:]
    events = np.ones(len(rows))
    events[0] = 0
    outcomes = Observations(
        {
            "unit": unit[rows],
            "year": time[rows],
            "duration": np.exp(0.2 * x[rows] + rng.normal(size=len(rows))),
            "event": events,
        },
        row_ids=rows.tolist(),
    )
    weights = prepare_spatial_weights(
        np.roll(np.eye(n), 1, axis=1), unit_ids=np.arange(n)
    )
    streams = tuple(
        IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=8)
        for t in range(periods)
    )
    return PanelData(
        nodes, PanelLayout("unit", "year"), {"neighbors": weights}, outcomes
    ), streams


@pytest.mark.integration
@pytest.mark.contract
def test_seasonal_two_way_spatial_aft_reuses_joint_likelihood() -> None:
    """Replay a missing node, a censored spell and a named WX in both interfaces."""
    data, streams = _panel()
    terms = Linear("x") + CyclicSpline("phase", df=3, period=1.0)
    response = RightCensored("duration", "event")
    ml = SpatialSurvivalML(rho_fixed=0.2)
    model = Regression(
        response,
        terms + SpatialX("x", "neighbors"),
        structure=data.panel,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
        effects="two-way",
        estimator=ml,
    )
    prepared = model.prepare(SpatialSurvivalData(data, streams))
    with pytest.raises(ValueError, match="another model"):
        replace(model).estimate_prepared(prepared)
    named = model.estimate_prepared(prepared).require_valid()
    assert prepared.problem.parameter_names is not None
    assert named.parameter_names == (*prepared.problem.parameter_names, "log_scale")
    np.testing.assert_allclose(
        named.core.log_likelihood,
        spatial_survival_log_likelihood(
            named.parameters,
            prepared.problem,
            rho_fixed=prepared.estimator.rho_fixed,
            integration=prepared.estimator.integration,
        ),
    )
    report = named.report()
    assert report.metadata["graph"] == "neighbors"
    assert report.metadata["integration_draws"] == (8,) * 4
    assert named.context.row_ids == data.spatial.outcomes.row_ids
    assert named.infer(covariance=ModelBased()).uncertainty.fit is named.core
    predicted = named.predict(data, kind="median")
    for period, prediction in zip(
        prepared.problem.periods, predicted.values, strict=True
    ):
        weights = np.asarray(period.graph.weights.values)
        design = np.asarray(period.design)
        columns = prepared.problem.process.slx_columns
        if columns:
            design = np.column_stack((design, weights @ design[:, columns]))
        mean = np.linalg.solve(
            np.eye(len(weights)) - 0.2 * weights,
            design @ np.asarray(named.parameters[:-1]),
        )
        np.testing.assert_allclose(prediction, np.exp(mean))


def _panel_with_predictor_roles() -> tuple[PanelData, tuple[IntegrationDesign, ...]]:
    data, streams = _panel()
    nodes = replace(
        data.nodes,
        labels={
            "group": tuple(int(unit) % 2 for unit in data.nodes.grouping_labels("unit"))
        },
        measures={
            "cloud": MeasureColumn(
                data.nodes.row_ids,
                PackedClouds.from_array(
                    jnp.arange(data.nodes.n_rows, dtype=float)[:, None, None]
                ),
            )
        },
    )
    return replace(data, nodes=nodes), streams


def _aligned_node_rows(data: PanelData) -> Observations:
    nodes, outcomes = data.nodes, data.spatial.outcomes
    keys = tuple(
        zip(nodes.grouping_labels("unit"), nodes.grouping_labels("year"), strict=True)
    )
    positions = {key: index for index, key in enumerate(keys)}
    expected = [
        positions[key]
        for key in zip(
            outcomes.grouping_labels("unit"),
            outcomes.grouping_labels("year"),
            strict=True,
        )
    ]
    return nodes.take(expected)


def _predictor_outcomes(
    data: PanelData,
    role: str,
) -> tuple[Term, Observations, Observations]:
    outcomes = data.spatial.outcomes
    aligned = _aligned_node_rows(data)
    if role == "numeric":
        return (
            Linear("x"),
            replace(outcomes, columns={**outcomes.columns, "x": aligned.numeric("x")}),
            replace(
                outcomes, columns={**outcomes.columns, "x": aligned.numeric("x") + 100}
            ),
        )
    if role == "categorical":
        return (
            Categorical("group"),
            replace(outcomes, labels={"group": aligned.categorical("group")}),
            replace(
                outcomes,
                labels={
                    "group": tuple(
                        1 - int(group) for group in aligned.categorical("group")
                    )
                },
            ),
        )
    clouds = aligned.measure("cloud")
    return (
        DistanceToLandmarks("cloud", Energy(), landmarks=(outcomes.row_ids[0],)),
        replace(outcomes, measures={"cloud": MeasureColumn(outcomes.row_ids, clouds)}),
        replace(
            outcomes,
            measures={
                "cloud": MeasureColumn(
                    outcomes.row_ids, clouds.with_coordinates(clouds.data + 100)
                )
            },
        ),
    )


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("role", ["numeric", "categorical", "measure"])
def test_independent_and_spatial_panels_use_node_predictors(role: str) -> None:
    """Match shuffled/subset spell keys and reject conflicts for every role."""
    data, streams = _panel_with_predictor_roles()
    terms, matching_outcomes, conflicting_outcomes = _predictor_outcomes(data, role)
    model = Regression(
        RightCensored("duration", "event"),
        Intercept() + terms,
        structure=data.panel,
        law=LogNormalAFT(),
    )
    with pytest.raises(ValueError, match="conflicts"):
        model.prepare(replace(data, outcomes=conflicting_outcomes))
    spatial_model = Regression(
        RightCensored("duration", "event"),
        Intercept() + terms,
        structure=data.panel,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
    )
    with pytest.raises(ValueError, match="conflicts"):
        spatial_model.prepare(
            SpatialSurvivalData(replace(data, outcomes=conflicting_outcomes), streams)
        )
    independent = model.prepare(replace(data, outcomes=matching_outcomes))
    expected_independent = model.prepare(data)
    assert isinstance(independent, PreparedSurvival)
    assert isinstance(expected_independent, PreparedSurvival)
    np.testing.assert_array_equal(independent.values, expected_independent.values)
    if role == "numeric":
        np.testing.assert_array_equal(
            independent.values[:, 1], _aligned_node_rows(data).numeric("x")
        )
    actual_spatial = spatial_model.prepare(
        SpatialSurvivalData(replace(data, outcomes=matching_outcomes), streams)
    )
    expected_spatial = spatial_model.prepare(SpatialSurvivalData(data, streams))
    for actual, expected_period in zip(
        actual_spatial.problem.periods,
        expected_spatial.problem.periods,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.design, expected_period.design)


@pytest.mark.component
@pytest.mark.contract
def test_survival_compiler_checks_family_graph_streams_and_confounding() -> None:
    """Reject an unsupported likelihood or an unidentified design before fitting."""
    data, streams = _panel()
    model = Regression(
        RightCensored("duration", "event"),
        Intercept(),
        structure=data.panel,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
    )
    with pytest.raises(ValueError, match="SpatialLag only"):
        replace(model, dependence=SpatialError("neighbors"))
    with pytest.raises(ValueError, match="LogNormalAFT"):
        replace(model, law=WeibullAFT())
    with pytest.raises(ValueError, match="layouts"):
        replace(model, structure=PanelLayout("unit", "period")).prepare(
            SpatialSurvivalData(data, streams)
        )
    with pytest.raises(KeyError, match="absent"):
        replace(model, dependence=SpatialLag("absent")).prepare(
            SpatialSurvivalData(data, streams)
        )
    with pytest.raises(ValueError, match="one explicit integration"):
        model.prepare(SpatialSurvivalData(data, streams[:-1]))
    with pytest.raises(ValueError, match="without an explicit intercept"):
        replace(
            model,
            terms=Intercept() + CyclicSpline("year", df=3, period=4.0),
            effects="two-way",
        ).prepare(SpatialSurvivalData(data, streams))


@pytest.mark.unit
@pytest.mark.contract
def test_legacy_survival_model_interfaces_are_absent() -> None:
    """Removed survival model doors stay absent from the public family."""
    assert not hasattr(survival_family, "SpatialSurvivalRegression")
    assert not hasattr(survival_family, "SpatialSurvivalInput")


@pytest.mark.component
@pytest.mark.contract
def test_independent_panel_survival_rejects_unused_integration_designs() -> None:
    """Supplying a numerical design must not imply it was consumed."""
    data, streams = _panel()
    model = Regression(
        RightCensored("duration", "event"),
        Intercept() + Linear("x"),
        structure=data.panel,
        law=LogNormalAFT(),
    )
    with pytest.raises(ValueError, match="does not consume"):
        model.prepare(SpatialSurvivalData(data, streams))
