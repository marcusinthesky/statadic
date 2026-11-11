"""Host spell identities, censoring roles, and graph preparation."""

from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations, PanelLayout, SpatialPanelData
from statadic.api.design import Intercept, Linear
from statadic.api.inference import ModelBased
from statadic.api.panel import prepare_spatial_panel_data
from statadic.api.regression import Regression
from statadic.api.spatial.graphs import PanelData, SpatialLag
from statadic.api.survival import RightCensored, SpatialSurvivalData, SurvivalResult, refine
from statadic.api.survival._inputs import _SpatialSurvivalInput
from statadic.api.survival._spatial import _SpatialSurvivalBackend
from statadic.estimation.survival import SpatialSurvivalML, SurvivalML
from statadic.estimation.survival.refinement import IntegrationRefinement
from statadic.model.survival import LogLogisticAFT, LogNormalAFT, WeibullAFT
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign

from .._typing import ill_typed


def _sample() -> Observations:
    return Observations(
        {
            "x": np.array([0.0, 1.0, 2.0, 4.0]),
            "duration": np.array([1.0, 2.0, 3.0, 4.0]),
            "event": np.array([1, 0, 1, 1]),
            "period": np.array([1, 2, 1, 2]),
        },
        row_ids=["s1", "s2", "s3", "s4"],
        labels={"unit": ["a", "a", "b", "b"]},
    )


@pytest.mark.component
@pytest.mark.contract
def test_prepare_spells_and_protect_model_identity() -> None:
    """Retain spell keys and prevent prepared state crossing models."""
    model = Regression(
        RightCensored("duration", "event"),
        Linear("x"),
        estimator=SurvivalML(family=WeibullAFT()),
        structure=PanelLayout("unit", "period"),
        effects="unit",
    )
    prepared = model.prepare(_sample())
    assert prepared.values.shape == (4, 3)
    assert prepared.context.row_ids == ("s1", "s2", "s3", "s4")
    with pytest.raises(ValueError, match="another model"):
        Regression(
            RightCensored("duration", "event"),
            Intercept() + Linear("x"),
            estimator=SurvivalML(family=WeibullAFT()),
        ).estimate_prepared(prepared)


@pytest.mark.unit
@pytest.mark.contract
def test_family_and_estimator_must_be_stated() -> None:
    """Model-defining survival choices have no silent default."""
    with pytest.raises(TypeError, match="family"):
        ill_typed(SurvivalML)()
    with pytest.raises(TypeError, match="AFT law"):
        ill_typed(Regression)(RightCensored("duration", "event"), Intercept())


@pytest.mark.component
@pytest.mark.contract
def test_invalid_event_and_duplicate_unit_period_are_rejected() -> None:
    """Reject invalid events and ambiguous entity-period spells."""
    model = Regression(
        RightCensored("duration", "event"),
        Intercept(),
        estimator=SurvivalML(family=WeibullAFT()),
        structure=PanelLayout("unit", "period"),
    )
    data = Observations(
        {
            "duration": np.array([1.0, 2.0]),
            "event": np.array([1, 2]),
            "period": np.array([1, 2]),
        },
        labels={"unit": ["a", "a"]},
    )
    with pytest.raises(ValueError, match="indicators"):
        model.prepare(data)
    repeated = Observations(
        {
            "duration": np.array([1.0, 2.0]),
            "event": np.array([1, 0]),
            "period": np.array([1, 1]),
        },
        labels={"unit": ["a", "a"]},
    )
    with pytest.raises(ValueError, match="at most one"):
        model.prepare(repeated)


@pytest.mark.component
@pytest.mark.contract
def test_full_graph_and_observed_spells_use_separate_axes() -> None:
    """Align observed spells without dropping unobserved graph nodes."""
    nodes = Observations(
        {
            "x": np.array([10.0, 20.0, 30.0, 40.0]),
            "unit": np.array([1, 2, 1, 3]),
            "period": np.array([1, 1, 2, 2]),
        },
        row_ids=["n1", "n2", "n3", "n4"],
    )
    outcomes = Observations(
        {
            "unit": np.array([3, 1]),
            "period": np.array([2, 1]),
            "time": np.array([2.0, 1.0]),
            "event": np.array([0, 1]),
        },
        row_ids=["spell-b", "spell-a"],
    )
    g1 = prepare_spatial_weights(
        jnp.array([[0.0, 1.0], [1.0, 0.0]]), unit_ids=jnp.array([2, 1])
    )
    g2 = prepare_spatial_weights(
        jnp.array([[0.0, 1.0], [1.0, 0.0]]), unit_ids=jnp.array([1, 3])
    )
    prepared = prepare_spatial_panel_data(
        SpatialPanelData(nodes, outcomes),
        PanelLayout("unit", "period"),
        ((1, g1), (2, g2)),
        Linear("x"),
    )
    assert prepared.unit_ids == (1, 2, 3)
    assert prepared.periods[0].entity_indices == (1, 0)
    assert prepared.periods[0].observed_indices == (1,)
    assert prepared.outcome_rows == ((1,), (0,))
    np.testing.assert_array_equal(prepared.values[0][:, 0], [20.0, 10.0])


@pytest.mark.integration
def test_fitted_survival_prediction_and_report_use_native_fit() -> None:
    """Prediction uses the retained design and does not imply inference."""
    rng = np.random.default_rng(7)
    x = np.linspace(-1, 1, 40)
    durations = np.exp(0.4 + 0.3 * x) * rng.weibull(1.5, 40)
    censor_time = 2.0
    data = Observations(
        {
            "x": x,
            "duration": np.minimum(durations, censor_time),
            "event": (durations <= censor_time).astype(int),
        }
    )
    result = Regression(
        RightCensored("duration", "event"),
        Intercept() + Linear("x"),
        estimator=SurvivalML(family=WeibullAFT()),
    ).estimate(data)
    options = {row.kind: row for row in result.prediction_options().rows}
    assert options["mean"].available
    assert options["restricted_mean"].required_columns == ("x",)
    predictions = result.predict(data, np.array(1.0), kind="survival")
    assert bool(result.core.valid)
    assert dict(result.report().metadata)["family"] == "WeibullAFT"
    assert bool(jnp.all((predictions >= 0) & (predictions <= 1)))
    assert len(result.report().parameters) == len(result.core.parameters)
    assert result.infer(covariance=ModelBased()).uncertainty.fit is result.core
    grid = np.array([0.0, 0.5, 1.0])
    survival = result.predict(data, grid, kind="survival", grid=jnp.ndim(grid) == 1)
    hazard = result.predict(
        data, grid, kind="cumulative_hazard", grid=jnp.ndim(grid) == 1
    )
    assert survival.shape == (data.n_rows, 3)
    np.testing.assert_allclose(hazard, -np.log(survival), atol=1e-6)
    assert result.predict(data, kind="median").shape == (data.n_rows,)
    assert result.predict(
        data, grid, kind="restricted_mean", grid=jnp.ndim(grid) == 1
    ).shape == (data.n_rows, 3)
    assert result.predict(data, kind="mean").shape == (data.n_rows,)
    np.testing.assert_allclose(
        result.predict(data), result.predict(data, 0.5, kind="quantile")
    )
    with pytest.raises(ValueError, match="Mean prediction"):
        result.predict(data, np.array(1.0), kind="mean")
    with pytest.raises(ValueError, match="Mean prediction"):
        result.predict(data, kind="mean", grid=True)
    with pytest.raises(ValueError, match="Mean prediction"):
        result.predict(data, kind="mean", entry=np.array(0.0))
    with pytest.raises(ValueError, match="requires a value"):
        result.predict(data, kind="survival")


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("container", ["mapping", "arrow"])
@pytest.mark.parametrize("fixed_effects", [False, True])
def test_survival_named_prediction_inputs_match_independent_log_time_fit(  # noqa: PLR0914 -- independent dummy design, fitted state and query inputs
    container: Literal["mapping", "arrow"], *, fixed_effects: bool
) -> None:
    """Mappings and Arrow predictions reuse the fitted AFT design and effect keys."""
    from scipy.stats import norm  # noqa: PLC0415 -- independent test-only law

    dtype = np.float64 if jax.config.x64_enabled else np.float32
    count = 36
    unit = np.repeat(np.arange(3), 12)
    period = np.tile(np.arange(12), 3)
    x = np.sin(np.arange(count) * 0.7).astype(dtype)
    duration = np.exp(0.2 + 0.4 * x + 0.15 * unit + 0.2 * np.cos(period)).astype(dtype)
    training: dict[str, object] = {
        "x": x,
        "time": duration,
        "event": np.ones(count, dtype=np.int32),
    }
    if fixed_effects:
        training.update(unit=unit, period=period)
    model = Regression.aft(
        RightCensored("time", "event"),
        Linear("x") if fixed_effects else Intercept() + Linear("x"),
        law=LogNormalAFT(),
    )
    if fixed_effects:
        model = replace(model, structure=PanelLayout("unit", "period"), effects="unit")
    pa = pytest.importorskip("pyarrow") if container == "arrow" else None
    fit_input = training if pa is None else pa.table(training)
    result = model.estimate(fit_input)
    assert isinstance(result, SurvivalResult)
    assert result.valid
    design = (
        np.column_stack((x, np.eye(3)[unit]))
        if fixed_effects
        else np.column_stack((np.ones(count), x))
    )
    coefficients = np.linalg.lstsq(design, np.log(duration), rcond=None)[0]
    scale = np.sqrt(np.mean((np.log(duration) - design @ coefficients) ** 2))
    query_x = np.array([-0.5, 0.2, 0.8], dtype=dtype)
    query = {"x": query_x}
    query_design = np.column_stack((np.ones(3), query_x))
    if fixed_effects:
        query.update(unit=np.array([2, 0, 1]), period=np.array([3, 1, 5]))
        query_design = np.column_stack((query_x, np.eye(3)[query["unit"]]))
    query_input = query if pa is None else pa.table(query)
    expected_log_time = query_design @ coefficients
    np.testing.assert_allclose(
        result.predict(query_input), np.exp(expected_log_time), rtol=4e-4, atol=4e-4
    )
    times = np.array([0.5, 1.0, 2.0], dtype=dtype)
    expected = norm.sf((np.log(times)[None, :] - expected_log_time[:, None]) / scale)
    np.testing.assert_allclose(
        result.predict(query_input, times, kind="survival", grid=True),
        expected,
        rtol=4e-4,
        atol=4e-4,
    )
    if container == "mapping":
        scalar = {name: values[0] for name, values in query.items()}
        np.testing.assert_allclose(
            result.predict(scalar), np.exp(expected_log_time[:1]), rtol=4e-4
        )
        with pytest.raises(ValueError, match="Unknown columns"):
            result.predict({**query, "unrelated": np.zeros(3)})


@pytest.mark.integration
def test_log_logistic_fit_discloses_undefined_mean_before_prediction() -> None:
    """A valid heavy-tailed model can predict quantiles without a finite mean."""
    q = (np.arange(50) + 0.5) / 50
    durations = (q / (1 - q)) ** 2
    data = Observations({"time": durations, "event": np.ones_like(q)})
    result = Regression(
        RightCensored("time", "event"), Intercept(), law=LogLogisticAFT()
    ).estimate(data)
    assert result.valid
    options = {row.kind: row for row in result.prediction_options().rows}
    assert options["quantile"].available
    assert options["restricted_mean"].available
    assert not options["mean"].available
    assert np.isnan(result.predict(data, kind="mean")).all()
    assert np.isfinite(result.predict(data, kind="median")).all()
    np.testing.assert_allclose(
        result.predict(data), result.predict(data, 0.5, kind="quantile")
    )


@pytest.mark.component
@pytest.mark.contract
def test_fixed_effects_keep_names_and_reject_unseen_levels() -> None:
    """Retained reference coding cannot invent new conditional effects."""
    model = Regression(
        RightCensored("duration", "event"),
        Linear("x"),
        estimator=SurvivalML(family=WeibullAFT()),
        structure=PanelLayout("unit", "period"),
        effects="unit",
    )
    prepared = model.prepare(_sample())
    assert prepared.effects is not None
    assert prepared.effects.column_names == ("unit:str:'a'", "unit:str:'b'")
    unseen = Observations(
        {"x": np.array([1.0]), "period": np.array([3])}, labels={"unit": ["c"]}
    )
    with pytest.raises(ValueError, match="known fixed-effect"):
        prepared.effects.transform(unseen)


@pytest.mark.component
@pytest.mark.contract
def test_graph_input_rejects_outcomes_outside_graph() -> None:
    """A spell for an absent population member is not silently dropped."""
    nodes = Observations(
        {
            "x": np.array([1.0, 2.0]),
            "unit": np.array([1, 2]),
            "period": np.array([1, 1]),
        }
    )
    outcomes = Observations({"unit": np.array([3]), "period": np.array([1])})
    graph = prepare_spatial_weights(
        jnp.array([[0.0, 1.0], [1.0, 0.0]]), unit_ids=jnp.array([1, 2])
    )
    with pytest.raises(ValueError, match="map to a graph"):
        prepare_spatial_panel_data(
            SpatialPanelData(nodes, outcomes),
            PanelLayout("unit", "period"),
            ((1, graph),),
            Linear("x"),
        )


@pytest.mark.integration
@pytest.mark.reference
def test_spatial_survival_preparation_and_fit_preserve_graphs() -> None:  # noqa: PLR0914, PLR0915
    """Full-node prediction uses the retained graph while spells stay aligned."""
    rng = np.random.default_rng(241)
    n, nt = 6, 3
    w = np.ones((n, n)) - np.eye(n)
    w /= n - 1
    graphs = tuple(
        (t, prepare_spatial_weights(w, unit_ids=np.arange(n))) for t in range(nt)
    )
    x = rng.normal(size=(nt, n))
    times = np.exp(0.4 + 0.3 * x + 0.5 * rng.normal(size=(nt, n)))
    units = np.tile(np.arange(n), nt)
    periods = np.repeat(np.arange(nt), n)
    nodes = Observations({"x": x.ravel(), "unit": units, "period": periods})
    outcomes = Observations(
        {
            "time": times.ravel(),
            "event": np.ones(n * nt),
            "unit": units,
            "period": periods,
        }
    )
    integration = tuple(
        IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=16)
        for t in range(nt)
    )
    data = _SpatialSurvivalInput(SpatialPanelData(nodes, outcomes), integration)
    model = _SpatialSurvivalBackend(
        RightCensored("time", "event"),
        Intercept() + Linear("x"),
        PanelLayout("unit", "period"),
        graphs,
        SpatialSurvivalML(rho_fixed=0.0),
    )
    prepared = model.prepare(data)
    lagged = _SpatialSurvivalBackend(
        RightCensored("time", "event"),
        Intercept() + Linear("x"),
        PanelLayout("unit", "period"),
        graphs,
        SpatialSurvivalML(rho_fixed=0.0),
        lag_columns=("x",),
    ).prepare(data)
    assert lagged.problem.parameter_names == ("intercept", "x", "W:x")
    assert lagged.problem.process.slx_columns == (1,)
    # The shared WX selector gives the same contract as the Gaussian hosts.
    for lags, match in (
        (("intercept",), "Intercept cannot be selected"),
        (("x", "x"), "unique names"),
        (("z",), "unique names"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(model, lag_columns=lags).prepare(data)
    # Structural lag columns live on the problem, not on the solver.
    with pytest.raises(TypeError, match="process"):
        ill_typed(SpatialSurvivalML)(process=object())
    result = model.estimate_prepared(prepared)
    assert bool(result.core.valid)
    pred = result.predict(data.data, np.array(1.0), kind="survival")
    assert len(pred.values) == nt
    for values in pred.values:
        assert values.shape == (n,)
        assert bool(jnp.all((values >= 0) & (values <= 1)))
    assert len(result.report().parameters) == len(result.core.parameters)
    assert result.infer(covariance=ModelBased()).uncertainty.fit is result.core
    grid = np.array([0.0, 0.5, 1.0])
    survival = result.predict(
        data.data, grid, kind="survival", grid=jnp.ndim(grid) == 1
    )
    hazard = result.predict(
        data.data, grid, kind="cumulative_hazard", grid=jnp.ndim(grid) == 1
    )
    rmst = result.predict(
        data.data, grid, kind="restricted_mean", grid=jnp.ndim(grid) == 1
    )
    for sv, hz, rm in zip(survival.values, hazard.values, rmst.values, strict=True):
        assert sv.shape == (n, 3)
        np.testing.assert_allclose(hz, -np.log(sv), atol=1e-6)
        np.testing.assert_allclose(rm[:, 0], 0.0)
    assert result.predict(data.data, kind="median").values[0].shape == (n,)
    predicted_mean = result.predict(data.data, kind="mean").values
    beta = np.asarray(result.core.parameters[:2])
    sigma = np.exp(result.core.parameters[-1])
    for period, observed in enumerate(predicted_mean):
        np.testing.assert_allclose(
            observed, np.exp(beta[0] + beta[1] * x[period] + sigma**2 / 2)
        )
    nonzero = replace(result, core=replace(result.core, rho=jnp.asarray(0.4)))
    propagated = np.linalg.inv(np.eye(n) - 0.4 * w)
    expected = np.exp(
        propagated @ (beta[0] + beta[1] * x[0])
        + sigma**2 * np.sum(propagated**2, axis=1) / 2
    )
    np.testing.assert_allclose(
        nonzero.predict(data.data, kind="mean").values[0], expected
    )
    changed_graphs = tuple(
        (
            period,
            prepare_spatial_weights(
                3 * np.asarray(graph.values), unit_ids=np.asarray(graph.unit_ids)
            ),
        )
        for period, graph in graphs
    )
    with pytest.raises(ValueError, match="admissible spatial parameter"):
        nonzero.predict(data.data, graphs=changed_graphs, kind="mean")
    with pytest.raises(ValueError, match="Mean prediction"):
        result.predict(data.data, np.array(1.0), kind="mean")
    with pytest.raises(ValueError, match="Mean prediction"):
        result.predict(data.data, kind="mean", grid=True)
    for kind in ("survival", "restricted_mean"):
        invalid = result.predict(
            data.data, np.array([np.inf]), kind=kind, grid=True
        ).values[0]
        assert bool(jnp.all(jnp.isnan(invalid)))
    validation = tuple(
        IntegrationDesign.from_key(jax.random.key(t + 100), replicates=2, draws=16)
        for t in range(nt)
    )
    refined = refine(
        Regression(
            model.response,
            model.terms,
            estimator=model.estimator,
            structure=model.panel,
            law=LogNormalAFT(),
            dependence=SpatialLag("neighbors"),
        ).prepare(
            SpatialSurvivalData(
                PanelData(nodes, model.panel, {"neighbors": graphs}, outcomes),
                integration,
            )
        ),
        validation_integrations=validation,
        config=IntegrationRefinement(initial_draws=16, max_draws=32, replicates=2),
    )
    assert refined.result.core is refined.refinement.fit
    assert refined.refinement.validation_designs == validation
    assert refined.refinement.attempts


@pytest.mark.component
@pytest.mark.contract
def test_spatial_survival_rejects_positive_entry_and_missing_designs() -> None:
    """The host requires fixed integration designs and rejects spatial entry."""
    nodes = Observations(
        {
            "x": np.array([0.0, 1.0]),
            "unit": np.array([1, 2]),
            "period": np.array([1, 1]),
        }
    )
    outcomes = Observations(
        {
            "time": np.array([2.0, 3.0]),
            "event": np.array([1, 0]),
            "entry": np.array([0.0, 1.0]),
            "unit": np.array([1, 2]),
            "period": np.array([1, 1]),
        }
    )
    graph = prepare_spatial_weights(
        np.array([[0.0, 1.0], [1.0, 0.0]]), unit_ids=np.array([1, 2])
    )
    model = _SpatialSurvivalBackend(
        RightCensored("time", "event", "entry"),
        Intercept(),
        PanelLayout("unit", "period"),
        ((1, graph),),
    )
    base = SpatialPanelData(nodes, outcomes)
    with pytest.raises(ValueError, match="one explicit integration"):
        model.prepare(_SpatialSurvivalInput(base, ()))
    integration = IntegrationDesign.from_key(jax.random.key(0), replicates=2, draws=16)
    with pytest.raises(ValueError, match="Positive entry"):
        model.prepare(_SpatialSurvivalInput(base, (integration,)))
