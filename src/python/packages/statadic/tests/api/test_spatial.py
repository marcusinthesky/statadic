"""Spatial facades equal their native bricks and panel likelihood on matched inputs."""

from __future__ import annotations

from dataclasses import replace
from inspect import unwrap
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from statadic.api.data import Observations, PanelLayout, SpatialPanelData
from statadic.api.design import Intercept, Linear
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.hypotheses import LinearRestriction
from statadic.api.inference import HAC, HC, ModelBased
from statadic.api.panel import select_lag_columns
from statadic.api.regression import Regression
from statadic.api.spatial.cross_section import (
    SpatialInformation,
    SpatialMomentCovariance,
    _SpatialRegressionBackend,
)
from statadic.api.spatial.diagnostics import CommonFactor, CommonFactorRestriction
from statadic.api.spatial.observed import _ObservedSpatialPanelBackend
from statadic.api.spatial.panel import (
    KKPPanelResult,
    SpatialPanelResult,
    _SpatialPanelBackend,
)
from statadic.estimation.linear import LinearFit
from statadic.estimation.spatial import (
    KKP,
    ML,
    SAC,
    SAR,
    SEM,
    ObservedPanelML,
    PanelML,
    SLXLeastSquares,
    SpatialFit,
    SpatialGMM,
    SpatialIV,
    SpatialIVFit,  # noqa: TC002
    SpatialIVProblem,
    SpatialPanelFit,  # noqa: TC002
    SpatialPanelProblem,
    SpatialProblem,
)
from statadic.inference.linear_covariance import hac_covariance, hc_covariance
from statadic.inference.spatial import (
    SpatialUncertainty,
    spatial_covariance,
    spatial_impacts,
)
from statadic.operators.spatial import (
    SpatialPanelWeights,  # noqa: TC002
    SpatialWeights,  # noqa: TC002
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)

from .._typing import ill_typed

if TYPE_CHECKING:
    from statadic.api.capabilities import SummarizesLikelihood

UNITS, PERIODS = 12, 40
TOLERANCE_RHO = 0.1


def _panel(seed: int = 0) -> tuple[Observations, SpatialPanelWeights, Any, Any]:
    """Simulate a SAR panel with a known spatial coefficient."""
    rng = np.random.default_rng(seed)
    w = rng.random((UNITS, UNITS))
    np.fill_diagonal(w, 0.0)
    w /= w.sum(1, keepdims=True)
    x = rng.normal(size=(PERIODS, UNITS))
    noise = rng.normal(size=(PERIODS, UNITS)) * 0.3
    y = (0.4 + 0.8 * x + noise) @ np.linalg.inv(np.eye(UNITS) - 0.35 * w).T
    graph = prepare_spatial_panel_weights(
        w, period_ids=np.arange(PERIODS), unit_ids=np.arange(UNITS) + 100
    )
    data = Observations(
        {
            "y": y.reshape(-1),
            "x": x.reshape(-1),
            "unit": np.tile(np.arange(UNITS) + 100, PERIODS),
            "period": np.repeat(np.arange(PERIODS), UNITS),
        },
        row_ids=list(range(PERIODS * UNITS)),
    )
    return data, graph, x, y


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("effects", ["pooled", "unit"])
def test_spatial_panel_facade_equals_native(
    effects: Literal["pooled", "unit"],
) -> None:
    """The facade returns exactly the native fit and reports it."""
    data, graph, x, y = _panel()
    terms = Linear("x") if effects == "unit" else Intercept() + Linear("x")
    result = _SpatialPanelBackend(
        "y", terms, graph, PanelLayout("unit", "period"), PanelML(), SAR(), effects
    ).estimate(data)
    assert isinstance(result, SpatialPanelResult)
    design = (
        x[:, :, None] if effects == "unit" else np.stack([np.ones_like(x), x], axis=2)
    )
    native = PanelML().estimate(
        SpatialPanelProblem(
            design,
            y,
            graph,
            process=SAR(),
            effects=effects,
            parameter_names=result.design.column_names,
        )
    )
    assert result.core.valid
    np.testing.assert_array_equal(result.core.parameters, native.parameters)
    np.testing.assert_array_equal(result.core.log_likelihood, native.log_likelihood)
    assert abs(float(result.core.parameters[-1]) - 0.35) < TOLERANCE_RHO
    if effects == "unit":
        # Within projection profiles out one estimated effect per entity.
        assert (
            _statistics(result)["n_parameters"]
            == len(result.core.parameters) + 1 + UNITS
        )
    report = result.report()
    assert report.parameters[-1].name == "rho"
    assert {m.name for m in report.metrics} >= {"log_likelihood", "sigma2"}
    width = native.design.shape[-1]
    base = np.einsum("tnp,p->tn", native.design, native.parameters[:width])
    if effects == "unit":
        base = base + np.asarray(native.effects)[None, :]
    rho = float(native.parameters[width])
    structural = base + rho * (y @ np.asarray(graph.values).T)
    reduced = np.linalg.solve(np.eye(UNITS) - rho * np.asarray(graph.values), base.T).T
    np.testing.assert_allclose(
        np.concatenate(result.predict(data, kind="structural").values),
        structural.ravel(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.concatenate(result.predict(data).values),
        reduced.ravel(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.component
@pytest.mark.contract
def test_spatial_panel_rejects_misordered_rows() -> None:
    """Rows not in graph time-major order are rejected before fitting."""
    data, graph, _, _ = _panel()
    order = np.arange(PERIODS * UNITS)[::-1]
    shuffled = Observations(
        {n: np.asarray(data.numeric(n))[order] for n in ("y", "x", "unit", "period")},
        row_ids=list(range(PERIODS * UNITS)),
    )
    model = _SpatialPanelBackend(
        "y",
        Intercept() + Linear("x"),
        graph,
        PanelLayout("unit", "period"),
        PanelML(),
        SAR(),
        "pooled",
    )
    with pytest.raises(ValueError, match="time-major"):
        model.estimate(shuffled)


@pytest.mark.integration
def test_panel_regression_durbin_and_information() -> None:
    """The panel facade preserves the declared process and native information."""
    data, graph, x, y = _panel()
    estimator = PanelML()
    result = _SpatialPanelBackend(
        "y",
        Intercept() + Linear("x"),
        graph,
        PanelLayout("unit", "period"),
        estimator,
        SEM(),
        "pooled",
        lag_columns=("x",),
    ).estimate(data)
    assert isinstance(result, SpatialPanelResult)
    design = np.stack([np.ones_like(x), x], axis=2)
    native = estimator.estimate(
        SpatialPanelProblem(
            design,
            y,
            graph,
            process=SEM(),
            effects="pooled",
            slx_columns=(1,),
            parameter_names=result.design.column_names,
        )
    )
    np.testing.assert_array_equal(result.core.parameters, native.parameters)
    assert result.core.parameter_names == ("intercept", "x", "W:x", "lambda")
    np.testing.assert_array_equal(
        result.infer(covariance=SpatialInformation()).uncertainty.covariance.values,
        spatial_covariance(native).covariance.values,
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("graph_name", [None, "neighbors"])
def test_kkp_panel_result_reports_only_beta_and_exposes_plugin_covariance(
    graph_name: str | None,
) -> None:
    """Component estimates stay separate from coefficient inference."""
    _, graph, x, y = _panel(seed=22)
    alpha = np.random.default_rng(23).normal(size=UNITS)
    y = y + np.linalg.solve(np.eye(UNITS) - 0.35 * np.asarray(graph.values), alpha)
    data = Observations(
        {
            "y": y.reshape(-1),
            "x": x.reshape(-1),
            "unit": np.tile(np.arange(UNITS) + 100, PERIODS),
            "period": np.repeat(np.arange(PERIODS), UNITS),
        },
        row_ids=list(range(PERIODS * UNITS)),
    )
    result = _SpatialPanelBackend(
        "y",
        Intercept() + Linear("x"),
        graph,
        PanelLayout("unit", "period"),
        KKP(),
        SEM(),
        "random",
        graph_name=graph_name,
    ).estimate(data)
    assert isinstance(result, KKPPanelResult)
    assert [row.name for row in result.report().parameters] == ["intercept", "x"]
    expected_lambda_name = "lambda" if graph_name is None else f"lambda[{graph_name}]"
    assert expected_lambda_name in {metric.name for metric in result.report().metrics}
    metadata = dict(result.report().metadata)
    assert metadata["beta_solver_status"] == int(result.core.linear.solver.status)
    assert metadata["lambda_solver_status"] == int(result.core.lambda_solver.status)
    assert "solver_status" not in metadata
    assert result.core.valid
    # KKP is a moment/GLS estimator, so component estimates grant no likelihood.
    assert result.fit_statistics().metrics == ()
    assert result.likelihood_summary() is None
    assert result.infer(
        covariance=ModelBased()
    ).uncertainty.covariance.values.shape == (2, 2)


def _cross_section(
    seed: int = 3, units: int = 40
) -> tuple[Observations, SpatialWeights, Any, Any, Any]:
    """Simulate a SAR cross-section with an extra endogenous regressor."""
    rng = np.random.default_rng(seed)
    w = 0.6 * np.roll(np.eye(units), 1, axis=1) + 0.4 * np.roll(
        np.eye(units), -3, axis=1
    )
    x = rng.normal(size=(units, 2))
    excluded = rng.normal(size=units)
    endogenous = excluded + 0.5 * rng.normal(size=units)
    design = np.column_stack([np.ones(units), x])
    y = np.linalg.solve(
        np.eye(units) - 0.3 * w,
        design @ [0.4, 0.8, -0.5] + 0.6 * endogenous + rng.normal(size=units) * 0.3,
    )
    graph = prepare_spatial_weights(w, unit_ids=np.arange(units) + 500)
    data = Observations(
        {
            "y": y,
            "a": x[:, 0],
            "b": x[:, 1],
            "d": endogenous,
            "z": excluded,
            "wa": w @ x[:, 0],
            "wb": w @ x[:, 1],
            "unit": np.arange(units) + 500,
        },
        row_ids=list(range(units)),
    )
    return (
        data,
        graph,
        design,
        y,
        np.column_stack([endogenous, excluded, w @ x[:, 0], w @ x[:, 1]]),
    )


@pytest.mark.component
@pytest.mark.reference
def test_common_factor_keeps_the_actual_graph_qualified_rho() -> None:
    """The named proposition and independent Wald formula use one fitted schema."""
    data, graph, _, _, _ = _cross_section()
    result = _SpatialRegressionBackend(
        "y",
        Intercept() + Linear("a") + Linear("b"),
        graph,
        ML(),
        process=SAR(),
        graph_name="neighbors",
        unit="unit",
        lag_columns=("a", "b"),
    ).estimate(data)
    attached = result.infer(covariance=SpatialInformation()).uncertainty
    assert isinstance(attached, SpatialUncertainty)
    outcome = CommonFactor(covariance=attached).evaluate_spatial(result)
    proposition = outcome.hypothesis.null
    assert isinstance(proposition, CommonFactorRestriction)
    assert proposition.columns == ("a", "b")
    assert proposition.rho_name == "rho[neighbors]"
    assert "rho[neighbors]" in proposition.describe()
    assert proposition.metadata()["rho_name"] == "rho[neighbors]"
    beta = np.asarray(result.parameters)
    restriction = beta[3:5] + beta[5] * beta[1:3]
    gradient = np.zeros((2, 6))
    gradient[:, 1:3] = beta[5] * np.eye(2)
    gradient[:, 3:5] = np.eye(2)
    gradient[:, 5] = beta[1:3]
    variability = gradient @ np.asarray(attached.covariance.values) @ gradient.T
    expected = restriction @ np.linalg.solve(variability, restriction)
    assert outcome.available
    assert outcome.statistic == pytest.approx(expected, rel=3e-5)


@pytest.mark.component
@pytest.mark.contract
def test_custom_spatial_bricks_and_prediction_capability() -> None:
    """Custom estimators compose without silently assuming their prediction design."""

    class CustomSLX:
        slx_columns = (1,)

        def estimate(self, problem: SpatialProblem) -> LinearFit:
            return SLXLeastSquares().estimate(problem)

    class UnspecifiedSLX:
        def estimate(self, problem: SpatialProblem) -> LinearFit:
            return SLXLeastSquares().estimate(replace(problem, slx_columns=(1,)))

    class RecordingIV:
        received_roles = False
        supports_instrument_roles = True

        def estimate(self, problem: SpatialProblem) -> SpatialIVFit:
            self.received_roles = isinstance(problem, SpatialIVProblem)
            return SpatialIV(instrument_columns=(1, 2)).estimate(problem)

    data, graph, _, _, _ = _cross_section()
    terms = Intercept() + Linear("a") + Linear("b")
    result = _SpatialRegressionBackend(
        "y", terms, graph, CustomSLX(), process=None, lag_columns=("a",), unit="unit"
    ).estimate(data)
    assert isinstance(result.core, LinearFit)
    np.testing.assert_allclose(result.predict(data), result.core.fitted_values)
    other = _SpatialRegressionBackend(
        "y", terms, graph, UnspecifiedSLX(), process=None, unit="unit"
    ).estimate(data)
    with pytest.raises(ValueError, match="no qualified spatial prediction design"):
        other.predict(data)
    with pytest.raises(ValueError, match="instrument roles require"):
        _SpatialRegressionBackend(
            "y", terms, graph, CustomSLX(), process=None, unit="unit", endogenous=("d",)
        ).estimate(data)
    with pytest.raises(ValueError, match="declaring support"):
        _SpatialRegressionBackend(
            "y",
            terms,
            graph,
            CustomSLX(),
            process=None,
            unit="unit",
            endogenous=("d",),
            instrument_roles=True,
        ).estimate(data)
    iv = RecordingIV()
    _SpatialRegressionBackend(
        "y",
        terms,
        graph,
        iv,
        process=SAR(),
        unit="unit",
        endogenous=("d",),
        instruments=("z", "wa", "wb"),
        instrument_roles=True,
    ).estimate(data)
    assert iv.received_roles


@pytest.mark.integration
def test_custom_panel_brick() -> None:
    """A compatible custom panel estimator uses the ordinary lifecycle."""

    class CustomPanel:
        def estimate(self, problem: SpatialPanelProblem) -> SpatialPanelFit:
            return PanelML().estimate(problem)

    data, graph, _, _ = _panel()
    result = _SpatialPanelBackend(
        "y",
        Intercept() + Linear("x"),
        graph,
        PanelLayout("unit", "period"),
        CustomPanel(),
        SAR(),
        "pooled",
    ).estimate(data)
    assert isinstance(result, SpatialPanelResult)
    assert result.core.valid


@pytest.mark.integration
@pytest.mark.parametrize(
    ("estimator", "process", "columns"),
    [
        (ML(), SAR(), ()),
        (ML(), SEM(), ()),
        (ML(), SAC(), ()),
        (ML(), SAR(), (1, 2)),
        (SLXLeastSquares(), None, (1, 2)),
        (SpatialIV(instrument_columns=(1, 2)), SAR(), ()),
        (SpatialGMM(variance_model="homoskedastic"), SEM(), ()),
        (SpatialGMM(instrument_columns=(1, 2)), SAC(), ()),
    ],
    ids=repr,
)
def test_spatial_regression_equals_its_brick(
    estimator: ML | SLXLeastSquares | SpatialIV | SpatialGMM,
    process: SAR | SEM | SAC | None,
    columns: tuple[int, ...],
) -> None:
    """Host roles lower into the exact native problem independently of its brick."""
    data, graph, design, y, _ = _cross_section()
    terms = Intercept() + Linear("a") + Linear("b")
    result = _SpatialRegressionBackend(
        "y",
        terms,
        graph,
        estimator,
        process=process,
        graph_name="g",
        unit="unit",
        lag_columns=tuple(("intercept", "a", "b")[c] for c in columns),
    ).estimate(data)
    native = estimator.estimate(
        SpatialProblem(
            design,
            y,
            graph,
            process=process,
            slx_columns=columns,
            graph_name="g",
            parameter_names=result.design.column_names,
        )
    )
    assert bool(result.core.valid)
    assert type(result.core) is type(native)
    np.testing.assert_array_equal(result.core.parameters, native.parameters)
    assert result.core.parameter_names == native.parameter_names
    report = result.report()
    assert len(report.parameters) == len(native.parameters)
    assert dict(report.metadata)["estimator"] == type(estimator).__name__

    if isinstance(process, SAC) and isinstance(estimator, SpatialGMM):
        assert result.parameter_names[-2:] == ("rho[g]", "lambda[g]")
        matrix, values = LinearRestriction.zero("rho[g]", "lambda[g]").compile(
            result.parameter_names, result.parameters.dtype
        )
        np.testing.assert_array_equal(matrix[:, -2:], np.eye(2))
        np.testing.assert_array_equal(matrix[:, :-2], 0)
        np.testing.assert_array_equal(values, 0)


@pytest.mark.integration
def test_spatial_ml_result_predicts_infers_and_reports_impacts() -> None:
    """ML prediction, information covariance and impacts reuse the native fit."""
    data, graph, design, y, _ = _cross_section()
    estimator = ML()
    aligned = Observations(
        {name: data.numeric(name) for name in ("y", "a", "b")},
        row_ids=list(range(500, 540)),
    )
    terms = Intercept() + Linear("a") + Linear("b")
    result = _SpatialRegressionBackend(
        "y", terms, graph, estimator, process=SAR(), lag_columns=("a",)
    ).estimate(aligned)
    native = estimator.estimate(
        SpatialProblem(
            design,
            y,
            graph,
            process=SAR(),
            slx_columns=(1,),
            parameter_names=result.design.column_names,
        )
    )
    np.testing.assert_array_equal(result.predict(aligned), native.reduced_form)
    np.testing.assert_allclose(
        result.predict(aligned, kind="structural"), native.fitted_values
    )
    np.testing.assert_array_equal(result.structural_fitted(), native.fitted_values)
    np.testing.assert_array_equal(
        result.infer(covariance=SpatialInformation()).uncertainty.covariance.values,
        spatial_covariance(native).covariance.values,
    )
    np.testing.assert_array_equal(
        result.impacts().values, spatial_impacts(native).values
    )
    assert {m.name for m in result.report().metrics} >= {"log_likelihood", "sigma2"}
    with pytest.raises(TypeError, match="SpatialInformation"):
        result.infer(covariance=HC())
    changed = prepare_spatial_weights(np.asarray(graph.values), unit_ids=np.arange(40))
    with pytest.raises(ValueError, match="changed spatial unit roster"):
        result.predict(aligned, weights=changed)


@pytest.mark.integration
def test_spatial_error_structural_prediction_needs_only_predictors() -> None:
    """A disturbance graph changes covariance, without using neighbor responses."""
    data, graph, _, _, _ = _cross_section()
    new = Observations({"a": data.numeric("a")}, row_ids=graph.unit_ids.tolist())
    training = Observations(
        {"y": data.numeric("y"), "a": data.numeric("a")}, row_ids=new.row_ids
    )
    fit = _SpatialRegressionBackend(
        "y", Intercept() + Linear("a"), graph, ML(), process=SEM()
    ).estimate(training)
    targets = {row.kind: row for row in fit.prediction_options().rows}
    assert targets["structural"].required_columns == ("a",)
    assert targets["structural"].available
    np.testing.assert_allclose(
        fit.predict(new, kind="structural"), fit.predict(new), rtol=1e-6
    )


@pytest.mark.integration
def test_spatial_slx_result_predicts_and_attaches_linear_covariance() -> None:
    """SLX prediction rebuilds lagged columns; HC covariance matches the native."""
    data, graph, design, y, _ = _cross_section()
    estimator = SLXLeastSquares()
    result = _SpatialRegressionBackend(
        "y",
        Intercept() + Linear("a") + Linear("b"),
        graph,
        estimator,
        process=None,
        lag_columns=("a", "b"),
        unit="unit",
    ).estimate(data)
    native = estimator.estimate(
        SpatialProblem(
            design,
            y,
            graph,
            process=None,
            slx_columns=(1, 2),
            parameter_names=result.design.column_names,
        )
    )
    np.testing.assert_allclose(result.predict(data), native.fitted_values)
    with pytest.raises(ValueError, match="Prediction kind"):
        unwrap(type(result).predict)(result, data, kind="unknown")
    np.testing.assert_array_equal(
        result.infer(covariance=HC(kind="HC1")).uncertainty.covariance.values,
        hc_covariance(native, kind="HC1").covariance.values,
    )
    order = np.argsort(data.numeric("a"))
    ordered = result.infer(covariance=HAC(2, order_by="a"))
    assert ordered.uncertainty.fit is result.core
    np.testing.assert_allclose(
        ordered.uncertainty.covariance.values,
        hac_covariance(native, 2, row_order=order).covariance.values,
    )
    assert ordered.report().metadata["order_by"] == "a"
    impacts = result.impacts()
    beta, theta = native.parameters[:3], native.parameters[3:]
    np.testing.assert_allclose(impacts.values[:, 0], beta)
    np.testing.assert_allclose(
        impacts.values[1:, 2], beta[1:] + theta * graph.row_sums.mean()
    )
    assert impacts.valid
    assert not impacts.uncertainty_valid


@pytest.mark.integration
def test_spatial_iv_and_gmm_roles_covariance_and_unsupported_routes() -> None:
    """Endogenous/instrument roles reach the problem; native covariance attaches."""
    data, graph, design, y, extra = _cross_section(seed=4)
    terms = Intercept() + Linear("a") + Linear("b")
    problem = SpatialIVProblem(
        design,
        y,
        graph,
        process=SAR(),
        parameter_names=("intercept", "a", "b"),
        endogenous=extra[:, :1],
        excluded_instruments=extra[:, 1:],
    )
    for estimator in (
        SpatialIV(instrument_columns=(1, 2)),
        SpatialGMM(variance_model="homoskedastic"),
    ):
        result = _SpatialRegressionBackend(
            "y",
            terms,
            graph,
            estimator,
            process=SAR() if isinstance(estimator, SpatialIV) else SEM(),
            unit="unit",
            endogenous=("d",),
            instruments=("z", "wa", "wb"),
        ).estimate(data)
        native = estimator.estimate(
            replace(
                problem, process=SAR() if isinstance(estimator, SpatialIV) else SEM()
            )
        )
        assert bool(result.core.valid)
        np.testing.assert_array_equal(result.core.parameters, native.parameters)
        method = "homoskedastic"
        np.testing.assert_array_equal(
            result.infer(
                covariance=SpatialMomentCovariance(method)
            ).uncertainty.covariance.values,
            spatial_covariance(native, method=method).covariance.values,
        )
        assert len(result.report().parameters) == len(native.parameters)
        with pytest.raises(ValueError, match="no native prediction"):
            result.predict(data)
        with pytest.raises(ValueError, match="likelihood fit"):
            result.impacts()
        with pytest.raises(TypeError, match="SpatialMomentCovariance"):
            result.infer(covariance=ModelBased())
    with pytest.raises(ValueError, match="require SpatialIV or SpatialGMM"):
        _SpatialRegressionBackend(
            "y", terms, graph, ML(), process=SAR(), unit="unit", endogenous=("d",)
        ).estimate(data)
    with pytest.raises(ValueError, match="ordered numeric unit IDs"):
        _SpatialRegressionBackend("y", terms, graph, ML(), process=SAR()).estimate(data)


@pytest.mark.unit
@pytest.mark.contract
def test_ml_process_must_be_stated() -> None:
    """The process is required on a numerical problem, independent of its solver."""
    with pytest.raises(TypeError, match="process"):
        ill_typed(SpatialProblem)(np.ones((4, 1)), np.ones(4), object())


@pytest.mark.unit
@pytest.mark.contract
def test_named_lag_columns_resolve_and_reject_ambiguous_choices() -> None:
    """Named selectors use expanded design schema and reject the intercept."""
    names = ("intercept", "x", "group[b]")
    assert select_lag_columns(names, ("group[b]", "x"), ()) == (2, 1)
    with pytest.raises(ValueError, match="unique names"):
        select_lag_columns(names, ("absent",), ())
    with pytest.raises(ValueError, match="Intercept"):
        select_lag_columns(names, ("intercept",), ())
    with pytest.raises(ValueError, match="positional"):
        select_lag_columns(names, ("x",), (1,))


@pytest.mark.integration
@pytest.mark.numerical
def test_observed_spatial_panel_preserves_graph_and_prediction_semantics() -> None:  # noqa: PLR0914
    """Missing outcomes stay latent on the full graph; SEM lambda is not a lag."""
    rng = np.random.default_rng(42)
    n, periods = 6, 8
    matrix = np.ones((n, n)) - np.eye(n)
    matrix /= n - 1
    graphs = tuple(
        (t, prepare_spatial_weights(matrix, unit_ids=np.arange(n)))
        for t in range(periods)
    )
    x = rng.normal(size=(periods, n))
    y = 0.5 + 0.4 * x + rng.normal(size=(periods, n))
    node_unit = np.tile(np.arange(n), periods)
    node_period = np.repeat(np.arange(periods), n)
    keep = np.arange(periods * n) % 7 != 0
    nodes = Observations({"x": x.ravel(), "unit": node_unit, "period": node_period})
    outcomes = Observations(
        {"y": y.ravel()[keep], "unit": node_unit[keep], "period": node_period[keep]},
        row_ids=np.flatnonzero(keep).tolist(),
    )
    data = SpatialPanelData(nodes, outcomes)
    result = _ObservedSpatialPanelBackend(
        "y",
        Intercept() + Linear("x"),
        PanelLayout("unit", "period"),
        graphs,
        SEM(),
        "pooled",
        estimator=ObservedPanelML(),
    ).estimate(data)
    assert bool(result.core.valid)
    pred = result.predict(data)
    for value, values in zip(pred.values, result.panel.values, strict=True):
        np.testing.assert_allclose(
            value, values @ np.asarray(result.core.parameters[:-1]), rtol=1e-6
        )
    assert result.infer(covariance=SpatialInformation()).uncertainty.fit is result.core
    impacts = result.impacts()
    np.testing.assert_allclose(impacts.values[:, :, 1], 0.0, atol=1e-6)
    structural = result.predict(data, kind="structural")
    for a, b in zip(pred.values, structural.values, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6)
    # The same numerical parameter vector in an SAR model requires neighbours.
    lagged = replace(result, core=replace(result.core, lag=True))
    with pytest.raises(ValueError, match="neighbour outcome"):
        lagged.predict(data, kind="structural")
    complete = Observations({"y": y.ravel(), "unit": node_unit, "period": node_period})
    full_data = SpatialPanelData(nodes, complete)
    structural = lagged.predict(full_data, kind="structural")
    rho = float(lagged.core.parameters[-1])
    for t, actual in enumerate(structural.values):
        expected = (
            result.panel.values[t] @ np.asarray(result.core.parameters[:-1])
            + rho * matrix @ y[t]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _aligned(data: Observations) -> Observations:
    """Return the cross-section rows keyed by the graph's unit identifiers."""
    return Observations(
        {name: data.numeric(name) for name in ("y", "a", "b")},
        row_ids=list(range(500, 540)),
    )


def _summarizes(value: SummarizesLikelihood) -> SummarizesLikelihood:
    """Fail static type checking unless ``value`` conforms to the protocol."""
    return value


def _statistics(result: Any) -> dict[str, float]:  # noqa: ANN401
    """Return the shared information metrics of one fit by name."""
    return {m.name: float(m.value or 0.0) for m in result.fit_statistics().metrics}


def _assert_criteria(metrics: dict[str, float], ll: float, k: int, n: int) -> None:
    """Check the one set of formulas against an independent log likelihood."""
    assert metrics["n_parameters"] == k
    assert metrics["log_likelihood"] == pytest.approx(ll)
    assert metrics["aic"] == pytest.approx(-2 * ll + 2 * k)
    assert metrics["bic"] == pytest.approx(-2 * ll + k * np.log(n))
    assert metrics["hqic"] == pytest.approx(-2 * ll + 2 * k * np.log(np.log(n)))


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(
    ("process", "spatial_parameters"), [(SAR(), 1), (SEM(), 1), (SAC(), 2)]
)
def test_cross_section_fit_statistics_count_spatial_parameters_and_scale(
    process: SAR | SEM | SAC, spatial_parameters: int
) -> None:
    """K is design columns, rho and lambda, and the scale; n is the row count."""
    data, graph, _, _, _ = _cross_section()
    terms = Intercept() + Linear("a") + Linear("b")
    result = _SpatialRegressionBackend(
        "y", terms, graph, ML(), process=process, lag_columns=("a",)
    ).estimate(_aligned(data))
    assert isinstance(result.core, SpatialFit)
    assert bool(result.core.valid)
    # Intercept, a, b and W:a, then the spatial parameter(s) and the scale.
    k = 4 + spatial_parameters + 1
    _assert_criteria(_statistics(result), float(result.core.log_likelihood), k, 40)
    summary = _summarizes(result).likelihood_summary()
    assert summary == LikelihoodSummary(
        float(result.core.log_likelihood),
        k,
        40,
        "Spatial Gaussian log likelihood needs a valid fit.",
    )
    report = result.report()
    names = [m.name for m in report.metrics]
    assert names.count("log_likelihood") == 1
    assert names.index("n_parameters") > names.index("sigma2")
    assert result.fit_statistics().notes[0] in report.notes
    inference = result.infer(covariance=SpatialInformation())
    assert inference.fit_statistics == result.fit_statistics()
    assert result.fit_statistics().notes[0] in inference.report().notes


@pytest.mark.component
@pytest.mark.contract
def test_moment_spatial_estimator_reports_no_likelihood_statistics() -> None:
    """GMM/2SLS carries no likelihood criteria."""
    data, graph, _, _, _ = _cross_section(seed=4)
    terms = Intercept() + Linear("a") + Linear("b")
    gmm = _SpatialRegressionBackend(
        "y",
        terms,
        graph,
        SpatialGMM(variance_model="homoskedastic"),
        process=SEM(),
    ).estimate(_aligned(data))
    assert gmm.fit_statistics().metrics == ()
    assert gmm.likelihood_summary() is None
    assert "log_likelihood" not in {m.name for m in gmm.report().metrics}


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize(("effects", "extra"), [("pooled", 0), ("random", 1)])
def test_panel_fit_statistics_count_variance_ratio_and_use_time_by_unit_rows(
    effects: Literal["pooled", "random"], extra: int
) -> None:
    """K is coefficients, rho, the scale and any random-effect variance ratio."""
    data, graph, _, _ = _panel()
    result = _SpatialPanelBackend(
        "y",
        Intercept() + Linear("x"),
        graph,
        PanelLayout("unit", "period"),
        PanelML(),
        SAR(),
        effects,
    ).estimate(data)
    assert isinstance(result, SpatialPanelResult)
    assert bool(result.core.valid)
    _assert_criteria(
        _statistics(result),
        float(result.core.log_likelihood),
        2 + 1 + 1 + extra,
        UNITS * PERIODS,
    )
    panel_summary = _summarizes(result).likelihood_summary()
    assert panel_summary is not None
    assert (panel_summary.n_parameters, panel_summary.n_observations) == (
        2 + 1 + 1 + extra,
        UNITS * PERIODS,
    )
    names = [m.name for m in result.report().metrics]
    assert names.count("log_likelihood") == 1
    inference = result.infer(covariance=SpatialInformation())
    assert inference.fit_statistics == result.fit_statistics()


@pytest.mark.integration
@pytest.mark.parametrize("effects", ["pooled", "unit", "random"])
def test_observed_panel_fit_statistics_count_explicit_effects_and_observed_rows(  # noqa: PLR0914
    effects: Literal["pooled", "unit", "random"],
) -> None:
    """K is the mean design rank, lambda, the scale and any variance ratio."""
    rng = np.random.default_rng(42)
    n, periods = 6, 8
    matrix = np.ones((n, n)) - np.eye(n)
    matrix /= n - 1
    graphs = tuple(
        (t, prepare_spatial_weights(matrix, unit_ids=np.arange(n)))
        for t in range(periods)
    )
    x = rng.normal(size=(periods, n))
    y = 0.5 + 0.4 * x + rng.normal(size=(periods, n))
    node_unit = np.tile(np.arange(n), periods)
    node_period = np.repeat(np.arange(periods), n)
    keep = np.arange(periods * n) % 7 != 0
    nodes = Observations({"x": x.ravel(), "unit": node_unit, "period": node_period})
    outcomes = Observations(
        {"y": y.ravel()[keep], "unit": node_unit[keep], "period": node_period[keep]},
        row_ids=np.flatnonzero(keep).tolist(),
    )
    result = _ObservedSpatialPanelBackend(
        "y",
        Intercept() + Linear("x") if effects != "unit" else Linear("x"),
        PanelLayout("unit", "period"),
        graphs,
        SEM(),
        effects,
        estimator=ObservedPanelML(),
    ).estimate(SpatialPanelData(nodes, outcomes))
    assert bool(result.core.valid)
    coefficients = 1 if effects == "unit" else 2
    # Explicit unit dummies are mean coefficients; random effects are integrated out.
    k = coefficients + (n if effects == "unit" else 0) + 1 + 1 + (effects == "random")
    _assert_criteria(
        _statistics(result), float(result.core.log_likelihood), k, int(keep.sum())
    )
    observed_summary = _summarizes(result).likelihood_summary()
    assert observed_summary is not None
    assert (observed_summary.n_parameters, observed_summary.n_observations) == (
        k,
        int(keep.sum()),
    )
    names = [m.name for m in result.report().metrics]
    assert names.count("log_likelihood") == 1
    inference = result.infer(covariance=SpatialInformation())
    assert inference.fit_statistics == result.fit_statistics()


@pytest.mark.component
@pytest.mark.reference
def test_slx_least_squares_statistics_equal_the_ols_fit_on_the_lagged_design() -> None:
    """SLX least squares is Gaussian OLS, the non-spatial baseline for SAR/SEM."""
    data, graph, _, _, _ = _cross_section()
    aligned = _aligned(data)
    slx = _SpatialRegressionBackend(
        "y",
        Intercept() + Linear("a") + Linear("b"),
        graph,
        SLXLeastSquares(),
        process=None,
        lag_columns=("a",),
    ).estimate(aligned)
    w = np.asarray(graph.values)
    lagged = Observations(
        {
            "y": aligned.numeric("y"),
            "a": aligned.numeric("a"),
            "b": aligned.numeric("b"),
            "wa": w @ np.asarray(aligned.numeric("a")),
        }
    )
    ols = Regression(
        "y", Intercept() + Linear("a") + Linear("b") + Linear("wa")
    ).estimate(lagged)
    expected = ols.fit_statistics()
    assert slx.likelihood_summary() == ols.likelihood_summary()
    assert [m.name for m in slx.fit_statistics().metrics] == [
        m.name for m in expected.metrics
    ]
    for got, want in zip(slx.fit_statistics().metrics, expected.metrics, strict=True):
        assert got.value == pytest.approx(want.value, rel=1e-9)
    assert slx.fit_statistics().notes == expected.notes
    assert _statistics(slx)["n_parameters"] == 5  # noqa: PLR2004
    report = slx.report()
    assert slx.fit_statistics().notes[0] in report.notes
    assert "aic" in {m.name for m in report.metrics}
    assert slx.infer(covariance=HC()).fit_statistics == slx.fit_statistics()
