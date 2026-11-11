"""Compositional Gaussian specifications agree with their specialized workflows."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api import spatial, survival
from statadic.api.data import Observations, PanelLayout, RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.dyadic import Dyads
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.inference import HC
from statadic.api.panel import PanelCovariance, PanelResult, PreparedPanel
from statadic.api.regression import Observed, Poisson, Regression, RegressionContext
from statadic.api.regression._model import _PreparedObserved, _PreparedStandard
from statadic.api.spatial.cross_section import (
    SpatialInformation,
    SpatialRegressionResult,
    _SpatialRegressionBackend,
)
from statadic.api.spatial.graph_regression import PreparedGraphRegression
from statadic.api.spatial.graphs import (
    PanelData,
    SpatialData,
    SpatialError,
    SpatialLag,
    SpatialLagError,
    SpatialProcesses,
    SpatialX,
)
from statadic.api.spatial.observed import (
    ObservedSpatialPanelResult,
    _ObservedSpatialPanelBackend,
)
from statadic.api.spatial.panel import SpatialPanelResult, _SpatialPanelBackend
from statadic.estimation.glm import BinomialLogitMLE
from statadic.estimation.iv import TwoStageLeastSquares
from statadic.estimation.linear import LinearFit
from statadic.estimation.panel import (
    Between,
    FamaMacBeth,
    FirstDifference,
    PanelFit,
    PanelProblem,
    Pooled,
)
from statadic.estimation.spatial import ML, SAC, SAR, SEM, SLXLeastSquares, SpatialFit
from statadic.operators.panel import absorb_panel_effects, prepare_panel_index
from statadic.operators.spatial import (
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)

from ..._typing import ill_typed

_LAYOUT = PanelLayout("unit", "time")
_GRAPH = "neighbors"


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "options",
    [
        {"estimator": BinomialLogitMLE()},
        {"estimator": TwoStageLeastSquares()},
        {"law": Poisson(), "estimator": Pooled()},
        {"law": Poisson(), "estimator": ML()},
        {"endogenous": ("d",), "instruments": ("z",), "estimator": Pooled()},
    ],
    ids=[
        "logit-for-gaussian",
        "iv-without-instruments",
        "panel-for-poisson",
        "spatial-for-poisson",
        "panel-for-iv",
    ],
)
def test_known_estimators_cannot_cross_scientific_problem_families(
    options: dict[str, object],
) -> None:
    """Reject incompatible built-in compositions before preparing any data."""
    with pytest.raises(TypeError, match=r"incompatible.*response, law and instrument"):
        ill_typed(Regression)("y", Linear("x"), **options)


@pytest.mark.unit
@pytest.mark.contract
def test_slx_estimator_requires_spatial_predictors() -> None:
    """The exact nonspatial composition fails before reaching a LinearProblem."""
    with pytest.raises(TypeError, match=r"SLXLeastSquares requires.*SpatialX"):
        Regression("y", Intercept() + Linear("x"), estimator=SLXLeastSquares())


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "estimator", [Pooled(), Between(), FirstDifference(), FamaMacBeth()]
)
def test_panel_bricks_require_panel_inputs(
    estimator: Pooled | Between | FirstDifference | FamaMacBeth,
) -> None:
    """A data-dependent panel brick cannot fall through to rowwise least squares."""
    model = Regression("y", Intercept() + Linear("x"), estimator=estimator)
    with pytest.raises(TypeError, match="requires PanelData or a panel layout"):
        model.prepare({"x": [0.0, 1.0, 2.0], "y": [0.3, 1.1, 2.5]})


@pytest.mark.component
@pytest.mark.contract
def test_slx_brick_cannot_receive_an_ordinary_panel_problem() -> None:
    """SpatialX panel means need a panel brick, rather than a spatial row brick."""
    rows = Observations(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.3, 1.1, 2.5, 3.4],
            "unit": [0, 1, 0, 1],
            "time": [0, 0, 1, 1],
        }
    )
    data = PanelData(
        rows,
        _LAYOUT,
        {_GRAPH: prepare_spatial_weights(np.array([[0.0, 1.0], [1.0, 0.0]]))},
    )
    model = Regression(
        "y", Linear("x") + SpatialX("x", _GRAPH), estimator=SLXLeastSquares()
    )
    with pytest.raises(
        TypeError, match="panel data requires a compatible panel estimator"
    ):
        model.prepare(data)


def _matrix(n: int) -> np.ndarray:
    """Build a directed cycle with an asymmetric second neighbour."""
    return 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), 2, axis=1)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(
    ("dependence", "brick"),
    [
        (None, SLXLeastSquares()),
        (SpatialLag(_GRAPH), ML()),
        (SpatialError(_GRAPH), ML()),
        (SpatialLag(_GRAPH) + SpatialError(_GRAPH), ML()),
    ],
)
def test_compiled_cross_section_matches_native_design_and_fit(
    dependence: SpatialLag | SpatialError | SpatialLagError | None,
    brick: ML | SLXLeastSquares,
) -> None:
    """Equal parameters, diagnostics and fitted predictions for each WX process."""
    rng = np.random.default_rng(91)
    n = 48
    w = _matrix(n)
    graph = prepare_spatial_weights(w)
    x = rng.normal(size=n)
    y = np.linalg.solve(
        np.eye(n) - 0.3 * w, 1 + 2 * x + 0.6 * w @ x + rng.normal(size=n)
    )
    rows = Observations({"x": x, "y": y})
    terms = Intercept() + Linear("x")
    model = Regression("y", terms + SpatialX("x", _GRAPH), dependence=dependence)
    result = model.estimate(SpatialData(rows, {_GRAPH: graph}))
    process = (
        None
        if dependence is None
        else SAC()
        if isinstance(dependence, SpatialLagError)
        else SAR()
        if isinstance(dependence, SpatialLag)
        else SEM()
    )
    direct = _SpatialRegressionBackend(
        "y",
        terms,
        graph,
        brick,
        process=process,
        graph_name=_GRAPH,
        lag_columns=("x",),
    ).estimate(rows)
    assert isinstance(result, SpatialRegressionResult)
    assert isinstance(result.context, RegressionContext)
    assert result.context.specification is model
    assert result.valid == direct.valid
    np.testing.assert_array_equal(result.core.parameters, direct.core.parameters)
    covariance = HC(kind="HC3") if dependence is None else SpatialInformation()
    np.testing.assert_array_equal(
        result.infer(covariance=covariance).uncertainty.covariance.values,
        direct.infer(covariance=covariance).uncertainty.covariance.values,
    )
    assert result.report() == direct.report()
    np.testing.assert_array_equal(result.predict(rows), direct.predict(rows))


@pytest.mark.component
@pytest.mark.reference
def test_panel_wx_uses_full_roster_before_outcome_selection() -> None:  # noqa: PLR0914
    """Missing responses and independently shuffled row axes preserve neighbour X."""
    rng = np.random.default_rng(31)
    n, t = 6, 5
    w = _matrix(n)
    graph = prepare_spatial_weights(w)
    x = rng.normal(size=(t, n))
    wx = x @ w.T
    y = 1 + 2 * x + 3 * wx
    unit = np.tile(np.arange(n), t)
    time = np.repeat(np.arange(t), n)
    nodes = Observations({"x": x.ravel(), "unit": unit, "time": time})
    keep = rng.permutation(n * t)[3:]
    outcomes = Observations(
        {"y": y.ravel()[keep], "unit": unit[keep], "time": time[keep]},
        row_ids=keep.tolist(),
    )
    order = rng.permutation(n * t)
    data = PanelData(nodes.take(order.tolist()), _LAYOUT, {_GRAPH: graph}, outcomes)
    terms = Intercept() + Linear("x") + SpatialX("x", _GRAPH)
    model = Regression("y", terms, structure=_LAYOUT)
    prepared = model.prepare(data)
    result = model.estimate_prepared(prepared)
    assert isinstance(result, PanelResult)
    np.testing.assert_allclose(result.parameters, [1, 2, 3], rtol=1e-5)
    np.testing.assert_allclose(result.predict(nodes), y.ravel(), atol=1e-5)
    assert result.context.row_ids == tuple(keep)
    with pytest.raises(ValueError, match="another model"):
        replace(model).estimate_prepared(prepared)
    with pytest.raises(ValueError, match="graphs exactly"):
        result.predict(nodes.take(list(range(n * t - 1))))


@pytest.mark.component
@pytest.mark.reference
def test_panel_backend_selection_and_native_effect_normalization() -> None:  # noqa: PLR0914
    """Complete and observed likelihoods agree with the same unit-effect coding."""
    rng = np.random.default_rng(13)
    n, t = 5, 8
    w = _matrix(n)
    graph = prepare_spatial_panel_weights(w, period_ids=np.arange(t))
    unit = np.tile(np.arange(n), t)
    time = np.repeat(np.arange(t), n)
    x = rng.normal(size=(t, n))
    y = np.linalg.solve(
        np.eye(n) - 0.4 * w, (0.3 * np.arange(n) + x + rng.normal(size=(t, n))).T
    ).T
    rows = Observations({"x": x.ravel(), "y": y.ravel(), "unit": unit, "time": time})
    data = PanelData(rows, _LAYOUT, {_GRAPH: graph})
    model = Regression("y", Linear("x"), effects="unit", dependence=SpatialLag(_GRAPH))
    balanced_state = model.prepare(data)
    assert isinstance(balanced_state, _PreparedStandard)
    assert isinstance(balanced_state.backend, _SpatialPanelBackend)
    repeated = model.prepare(replace(data, outcomes=rows))
    assert isinstance(repeated, _PreparedStandard)
    assert isinstance(repeated.backend, _SpatialPanelBackend)
    copied = replace(data, outcomes=rows.take(list(reversed(range(n * t)))))
    copied_state = model.prepare(copied)
    assert isinstance(copied_state, _PreparedStandard)
    assert isinstance(copied_state.backend, _SpatialPanelBackend)
    observed_model = replace(model, panel_likelihood="observed")
    observed_state = observed_model.prepare(copied)
    assert isinstance(observed_state, _PreparedObserved)
    assert isinstance(observed_state.backend, _ObservedSpatialPanelBackend)
    balanced = model.estimate_prepared(balanced_state)
    observed = observed_model.estimate_prepared(observed_state)
    assert isinstance(balanced, SpatialPanelResult)
    assert isinstance(observed, ObservedSpatialPanelResult)
    assert observed.report().title == "Observed spatial panel fit"
    assert balanced.valid
    assert observed.valid
    np.testing.assert_allclose(observed.parameters, balanced.parameters, atol=1e-5)
    np.testing.assert_allclose(
        observed.core.log_likelihood, balanced.core.log_likelihood, atol=1e-5
    )
    np.testing.assert_allclose(
        np.concatenate(balanced.predict(data).values),
        np.concatenate(observed.predict(data).values),
        atol=1e-5,
    )
    for panel_data in (data, replace(data, outcomes=rows)):
        with pytest.raises(ValueError, match=r"without.*intercept"):
            replace(model, terms=Intercept() + Linear("x")).prepare(panel_data)
    two_way = replace(model, effects="two-way").prepare(data)
    assert isinstance(two_way, _PreparedObserved)
    assert isinstance(two_way.backend, _ObservedSpatialPanelBackend)


@pytest.mark.unit
@pytest.mark.contract
def test_compiler_rejects_conflicting_graphs_and_processes() -> None:
    """Failed capabilities and identity checks cannot fall through to fitting."""
    graph = prepare_spatial_weights(_matrix(3))
    rows = Observations(
        {"x": [1, 2, 3], "y": [4, 6, 8], "unit": [1, 2, 0], "time": [0] * 3}
    )
    mixed = SpatialLag("a") + SpatialError("b")
    assert isinstance(mixed, SpatialProcesses)
    assert mixed.lag == ("a",)
    assert mixed.error == ("b",)
    # Process choice belongs to Regression; the solver carries only numerics.
    model = Regression(
        "y", Linear("x"), dependence=SpatialError(_GRAPH), estimator=ML()
    )
    assert model.dependence == SpatialError(_GRAPH)
    terms = Linear("x") + SpatialX("x", "other")
    with pytest.raises(ValueError, match="Missing named graphs"):
        Regression("y", terms, dependence=SpatialLag(_GRAPH)).prepare(
            SpatialData(rows, {_GRAPH: graph})
        )
    sac = SpatialError(_GRAPH) + SpatialLag(_GRAPH)
    with pytest.raises(TypeError, match="SAR and SEM only"):
        Regression("y", Linear("x"), dependence=sac).prepare(
            PanelData(rows, _LAYOUT, {_GRAPH: graph})
        )
    with pytest.raises(ValueError, match="ordered numeric unit IDs"):
        Regression("y", Linear("x"), dependence=SpatialLag(_GRAPH)).estimate(
            SpatialData(rows, {_GRAPH: graph}, unit="unit")
        )


@pytest.mark.unit
@pytest.mark.contract
def test_removed_spatial_facades_are_not_exported() -> None:
    """Pre-release models have one researcher-facing composition entry point."""
    for name in (
        "SpatialRegression",
        "SpatialLagML",
        "SpatialLagResult",
        "SpatialPanelRegression",
        "SpatialPanelML",
        "ObservedSpatialPanelRegression",
    ):
        assert not hasattr(spatial, name)
    for name in ("SpatialSurvivalRegression", "SpatialSurvivalInput"):
        assert not hasattr(survival, name)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(
    ("rho", "theta", "error", "expected"),
    [
        (0.0, 0.0, True, (1.0, 0.0)),
        (0.0, 0.5, False, (1.0, 0.5)),
        (0.5, 0.0, False, (4 / 3, 2 / 3)),
        (0.5, 0.5, False, (5 / 3, 4 / 3)),
    ],
)
def test_fixed_graph_mean_derivatives(
    rho: float,
    theta: float,
    error: bool,  # noqa: FBT001
    expected: tuple[float, float],
) -> None:
    """On disconnected neighbor pairs, only WX and outcome feedback change the mean."""
    w = np.kron(np.eye(8), [[0.0, 1.0], [1.0, 0.0]])
    rng = np.random.default_rng(42)
    rows = Observations({"x": rng.normal(size=len(w)), "y": rng.normal(size=len(w))})
    terms = Linear("x")
    spec = Regression(
        "y",
        terms + SpatialX("x", _GRAPH) if theta else terms,
        dependence=SpatialError(_GRAPH)
        if error
        else SpatialLag(_GRAPH)
        if rho
        else None,
    )
    prepared = spec.prepare(SpatialData(rows, {_GRAPH: prepare_spatial_weights(w)}))
    description = spec.inspect(prepared)
    assert description.local_columns == ("x",)
    assert description.spatial_x == ((SpatialX("x", _GRAPH),) if theta else ())
    assert description.outcome_graphs == ((_GRAPH,) if rho else ())
    assert description.error_graphs == ((_GRAPH,) if error else ())
    result = spec.estimate_prepared(prepared)
    assert isinstance(result, SpatialRegressionResult)
    coefficients = [1.0]
    if theta:
        coefficients.append(theta)
    if rho or error:
        coefficients.append(rho or 0.5)
    # Evaluate each model's retained impact map at the analytical witness.
    assert isinstance(result.core, (LinearFit, SpatialFit))
    result = replace(
        result, core=replace(result.core, parameters=jnp.asarray(coefficients))
    )
    np.testing.assert_allclose(
        result.impacts().values[0], (*expected, sum(expected)), rtol=1e-6
    )


@pytest.mark.unit
@pytest.mark.contract
def test_complete_panel_aligns_outcome_values_and_time_varying_graphs() -> None:
    """Complete and observed targets have separate graph-keyed inputs on each period."""
    units = [0, 1, 2, 3] * 2
    periods = [2] * 4 + [5] * 4
    w = np.roll(np.eye(4), 1, axis=1)
    graph = prepare_spatial_panel_weights(
        np.stack([w, w.T]), period_ids=np.array([2, 5])
    )
    nodes = Observations(
        {"unit": units, "time": periods, "x": np.arange(8), "y": [0.0] * 8}
    )
    outcomes = Observations(
        {"unit": units, "time": periods, "y": np.arange(8) + 1}
    ).take([6, 4, 2, 0, 1, 3, 5, 7])
    data = PanelData(
        nodes.take([7, 5, 3, 1, 0, 2, 4, 6]),
        _LAYOUT,
        {_GRAPH: graph},
        outcomes=outcomes,
    )
    spec = Regression(
        "y", Linear("x"), dependence=SpatialLag(_GRAPH), panel_likelihood="complete"
    )
    state = spec.prepare(data)
    description = spec.inspect(state)
    assert description.likelihood == "complete"
    assert "explicitly requests" in description.likelihood_reason
    assert description.node_count == nodes.n_rows
    assert description.outcome_row_ids == outcomes.row_ids
    assert isinstance(state, _PreparedStandard)
    assert isinstance(state.backend, _SpatialPanelBackend)
    # The likelihood must receive the separate outcome, aligned to the graph.
    assert hasattr(state.native, "context")
    np.testing.assert_array_equal(
        state.native.context.observations.numeric("y"), np.arange(8) + 1
    )
    np.testing.assert_array_equal(state.backend.weights.values, np.stack([w, w.T]))
    partial = replace(data, outcomes=outcomes.take([0, 1, 2]))
    with pytest.raises(ValueError, match="outcome for every graph node"):
        spec.prepare(partial)
    observed = replace(spec, panel_likelihood="observed")
    description = observed.inspect(observed.prepare(partial))
    assert description.likelihood == "observed"
    assert description.outcome_row_ids == outcomes.take([0, 1, 2]).row_ids
    automatic = replace(spec, panel_likelihood=None)
    assert "subset" in automatic.inspect(automatic.prepare(partial)).likelihood_reason


@pytest.mark.unit
@pytest.mark.reference
def test_changing_graph_exposure_precedes_within_unit_absorption() -> None:
    """Time-invariant own X has zero within variation but changing WX does not."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    unit = np.tile(np.arange(x.size), 2)
    time = np.repeat([2, 5], x.size)
    w = np.roll(np.eye(x.size), 1, axis=1)
    graphs = prepare_spatial_panel_weights(
        np.stack([w, w.T]), period_ids=np.array([2, 5])
    )
    rows = Observations(
        {"x": np.tile(x, 2), "y": np.ones(unit.size), "unit": unit, "time": time}
    )
    model = Regression("y", Linear("x") + SpatialX("x", _GRAPH), effects="unit")
    state = model.prepare(PanelData(rows, _LAYOUT, {_GRAPH: graphs}))
    assert not isinstance(state, PreparedGraphRegression)
    assert isinstance(state, PreparedPanel)
    assert state.design.column_names == ("x", f"W[{_GRAPH}]:x")
    projection = absorb_panel_effects(state.values, prepare_panel_index(unit, time))
    assert projection.converged
    exposure = np.stack([w @ x, w.T @ x])
    expected = exposure - exposure.mean(axis=0)
    assert np.any(expected != 0)
    np.testing.assert_allclose(projection.values[:, 0], 0)
    np.testing.assert_allclose(projection.values[:, 1], expected.ravel())


@pytest.mark.unit
@pytest.mark.contract
def test_spatial_precision_is_rejected_before_unweighted_estimation() -> None:
    """An unsupported precision role cannot disappear at a spatial lowering."""
    precision = RelativePrecision("weight")
    with pytest.raises(ValueError, match="precision"):
        Regression("y", Linear("x"), precision=precision, dependence=SpatialLag(_GRAPH))
    graph = prepare_spatial_weights(_matrix(4))
    rows = Observations(
        {"x": np.arange(4.0), "y": np.arange(4.0), "weight": np.ones(4)}
    )
    model = Regression(
        "y", Intercept() + Linear("x") + SpatialX("x", _GRAPH), precision=precision
    )
    with pytest.raises(ValueError, match="precision"):
        model.prepare(SpatialData(rows, {_GRAPH: graph}))


@pytest.mark.unit
@pytest.mark.contract
def test_trials_are_not_ignored_by_iv_or_dyadic_lowering() -> None:
    """Trial totals belong exclusively to binomial observations."""
    observed = Observed("y", trials="trials")
    with pytest.raises(ValueError, match="trials"):
        Regression(
            observed, Intercept() + Linear("x"), endogenous=("x",), instruments=("z",)
        )
    with pytest.raises(ValueError, match="trials"):
        Regression(observed, Intercept() + Linear("x"), structure=Dyads("i", "j"))


@pytest.mark.component
@pytest.mark.contract
def test_inspection_does_not_mislabel_conditional_preparation_as_least_squares() -> (
    None
):
    """Unsupported family inspection is explicit rather than a false descriptor."""
    model = Regression("y", Intercept(), law=Poisson())
    prepared = model.prepare({"y": [0.0, 1.0, 2.0]})
    with pytest.raises(TypeError, match="Gaussian"):
        model.inspect(prepared)


@pytest.mark.integration
@pytest.mark.reference
def test_panel_wx_retains_supported_precision_in_ordinary_panel_problem() -> None:  # noqa: PLR0914 -- matched design, restriction and row-alignment witness
    """Full-node WX is a mean-design transform; panel WLS remains supported."""
    units = np.tile(np.arange(4), 2)
    periods = np.repeat([0, 1], 4)
    x = np.array([1.0, 2.0, 4.0, 7.0, 2.0, 1.0, 8.0, 3.0])
    w = _matrix(4)
    wx = np.concatenate((w @ x[:4], w.T @ x[4:]))
    design = np.column_stack((np.ones(8), x, wx))
    outcome = design @ np.array([0.4, 0.7, -0.2]) + np.array([0.2, -0.1, 0.3, -0.2] * 2)
    precision = np.linspace(0.5, 2.0, 8)
    rows = Observations(
        {"x": x, "y": outcome, "weight": precision, "unit": units, "time": periods}
    )
    graphs = prepare_spatial_panel_weights(
        np.stack((w, w.T)), period_ids=np.array([0, 1])
    )
    model = Regression(
        "y",
        Intercept() + Linear("x") + SpatialX("x", _GRAPH),
        structure=_LAYOUT,
        precision=RelativePrecision("weight"),
        estimator=Pooled(),
    )
    prepared = model.prepare(PanelData(rows, _LAYOUT, {_GRAPH: graphs}))
    assert isinstance(prepared, PreparedPanel)
    np.testing.assert_allclose(prepared.values, design)
    assert prepared.design.column_names == ("intercept", "x", f"W[{_GRAPH}]:x")
    result = model.estimate_prepared(prepared)
    native = Pooled().estimate(
        PanelProblem(
            design,
            outcome,
            prepare_panel_index(units, periods),
            precision,
            effects="pooled",
            parameter_names=prepared.design.column_names,
        )
    )
    assert result.core.valid
    assert native.valid
    assert isinstance(result.core, PanelFit)
    np.testing.assert_allclose(result.parameters, native.parameters)
    np.testing.assert_allclose(
        result.core.linear.weighted_sse, native.linear.weighted_sse
    )
    assert result.parameter_names == prepared.design.column_names
    assert (
        tuple(row.name for row in result.report().parameters) == result.parameter_names
    )
    restriction = LinearRestriction.zero(f"W[{_GRAPH}]:x")
    matrix, values = restriction.compile(
        result.parameter_names, result.parameters.dtype
    )
    np.testing.assert_array_equal(matrix, [[0, 0, 1]])
    np.testing.assert_array_equal(values, 0)
    inferred = result.infer(covariance=PanelCovariance())
    tested = inferred.test(Hypothesis(restriction), reference="chi2")
    assert tested.available
    expected_statistic = (
        native.parameters[2] ** 2 / inferred.uncertainty.covariance.values[2, 2]
    )
    assert tested.statistic == pytest.approx(float(expected_statistic))
    order = [4, 1, 6, 3, 0, 5, 2, 7]
    np.testing.assert_allclose(
        result.predict(rows.take(order)), design[order] @ np.asarray(native.parameters)
    )
    np.testing.assert_allclose(
        result.predict(PanelData(rows.take(order), _LAYOUT, {_GRAPH: graphs})),
        design[order] @ np.asarray(native.parameters),
    )
    changed = prepare_spatial_panel_weights(
        np.stack((w.T, w)), period_ids=np.array([0, 1])
    )
    changed_wx = np.concatenate((w.T @ x[:4], w @ x[4:]))
    changed_design = np.column_stack((np.ones(8), x, changed_wx))
    observed = [6, 1, 4]
    query = PanelData(rows.take(order), _LAYOUT, {_GRAPH: changed}, rows.take(observed))
    np.testing.assert_allclose(
        result.predict(query), changed_design[observed] @ np.asarray(native.parameters)
    )
    with pytest.raises(ValueError, match="requires graph"):
        result.predict(PanelData(rows, _LAYOUT))
    with pytest.raises(ValueError, match="match the supplied graphs exactly"):
        result.predict(PanelData(rows.take([0, 1, 4, 5]), _LAYOUT, {_GRAPH: graphs}))
