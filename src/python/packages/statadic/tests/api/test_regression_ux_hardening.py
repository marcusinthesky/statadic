"""Regression tests for identity-preserving compositional UX contracts."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.association import Mantel, MantelInput
from statadic.api.base import PreparedRegression
from statadic.api.data import MeasureColumn, Observations, PanelLayout
from statadic.api.design import (
    Categorical,
    CyclicSpline,
    FittedDesign,
    Fourier,
    Intercept,
    Linear,
    RowDifference,
    Spline,
)
from statadic.api.inference import HC
from statadic.api.regression import Regression
from statadic.api.regression._model import _PreparedStandard
from statadic.api.representation import MMD
from statadic.api.spatial.graph_regression import PreparedGraphRegression
from statadic.api.spatial.graphs import PanelData, SpatialData, SpatialLag
from statadic.api.survival import PreparedSurvival, RightCensored
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.model.survival import LogNormalAFT
from statadic.operators.spatial import prepare_spatial_weights
from statadic.sample.packed import PackedClouds

from .._typing import ill_typed


@pytest.mark.component
@pytest.mark.contract
def test_renaming_a_covariate_intercept_does_not_remove_a_unit_effect() -> None:
    """The meaning of an intercept comes from the term, not its display name."""
    values = np.asarray([0.2, -1.0, 2.0, 0.4, 1.8, -0.8], dtype=np.float32)
    data = Observations(
        {
            "x": values,
            "intercept": values,
            "duration": np.exp(1 + values / 5),
            "event": np.ones(6, dtype=np.int32),
            "unit": [0, 1, 0, 1, 0, 1],
            "period": [0, 0, 1, 1, 2, 2],
        }
    )
    layout = PanelLayout("unit", "period")
    model = Regression(
        RightCensored("duration", "event"),
        Linear("x"),
        structure=layout,
        effects="unit",
        law=LogNormalAFT(),
    )
    original = model.prepare(data)
    renamed = replace(model, terms=Linear("intercept")).prepare(data)
    assert isinstance(original, PreparedSurvival)
    assert isinstance(renamed, PreparedSurvival)
    assert original.problem.parameter_names == ("x", "unit:int:0", "unit:int:1")
    assert renamed.problem.parameter_names == ("intercept", "unit:int:0", "unit:int:1")
    np.testing.assert_array_equal(original.values, renamed.values)


@pytest.mark.component
@pytest.mark.contract
def test_fourier_periods_are_distinct_column_identities() -> None:
    """Two frequencies on one source column do not collide as coefficients."""
    rows = Observations({"phase": jnp.arange(20.0)})
    design = FittedDesign(
        (
            Fourier("phase", harmonics=1, period=7)
            + Fourier("phase", harmonics=1, period=365)
        )._prepare_term(rows)
    )
    expected_columns = 4
    assert len(design.column_names) == len(set(design.column_names)) == expected_columns


@pytest.mark.component
@pytest.mark.contract
def test_measure_column_propagates_exact_reference_identities() -> None:
    """Labeled measures keep their declared order through preparation."""
    measures = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[2.0]])])
    column = MeasureColumn(("lab-b", "lab-a"), measures)
    prepared = MMD(LINEAR_KERNEL).prepare(column)
    assert prepared.row_ids == column.row_ids
    with pytest.raises(ValueError, match="must equal"):
        MMD(LINEAR_KERNEL).prepare(column, row_ids=("lab-a", "lab-b"))


@pytest.mark.component
@pytest.mark.contract
def test_mantel_control_belongs_to_the_single_input() -> None:
    """A second control keyword cannot bypass the declared input bundle."""
    square = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    request = MantelInput(square, square)
    with pytest.raises(TypeError, match="unexpected keyword argument 'control'"):
        ill_typed(Mantel().estimate)(request, control=square)


@pytest.mark.integration
@pytest.mark.contract
def test_contrast_uses_frozen_cyclic_basis_and_exact_parameter_schema() -> None:
    """Cyclic contrasts replay the fitted basis and validate coefficient order."""
    phase = jnp.arange(36.0) / 36.0
    response = (
        1.0 + 0.4 * jnp.sin(2 * jnp.pi * phase) + 0.1 * jnp.cos(4 * jnp.pi * phase)
    )
    term = CyclicSpline("phase", df=3, period=1.0)
    fit = (
        Regression("y", Intercept() + term)
        .estimate({"phase": phase, "y": response + 0.05 * phase})
        .require_valid()
    )
    contrast = fit.design.design_contrast(term, at=0.25, versus=0.75)
    query = Observations({"phase": jnp.asarray([0.25, 0.75])})
    design = fit.design.transform(query)
    np.testing.assert_allclose(contrast.coefficients, design[0] - design[1], atol=1e-7)
    result = fit.infer(covariance=HC("HC1")).contrast(contrast, reference="normal")
    assert result.valid
    expected = (design[0] - design[1]) @ fit.parameters
    assert result.estimate == pytest.approx(float(expected), rel=1e-6)
    with pytest.raises(ValueError, match="schema"):
        fit.infer(covariance=HC("HC1")).contrast(
            replace(contrast, parameter_names=contrast.parameter_names[::-1]),
            reference="normal",
        )


@pytest.mark.component
@pytest.mark.contract
def test_spatial_model_import_and_named_graph_compose() -> None:
    """A named fixed graph reaches the native regression preparation."""
    weights = prepare_spatial_weights(
        jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    )
    rows = Observations({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 1.0]})
    model = Regression("y", Intercept() + Linear("x"), dependence=SpatialLag("g"))
    state = model.prepare(SpatialData(rows, {"g": weights}))
    assert not isinstance(state, PreparedGraphRegression)
    assert isinstance(state, _PreparedStandard)
    assert isinstance(state.native, PreparedRegression)
    assert state.native.design.column_names == ("intercept", "x")


@pytest.mark.component
@pytest.mark.contract
def test_independent_aft_uses_full_node_covariates_with_missing_spells() -> None:
    """Switching off dependence preserves the keyed observed sample."""
    layout = PanelLayout("unit", "period")
    nodes = Observations(
        {"unit": [0, 1, 0, 1], "period": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    outcome = Observations(
        {
            "unit": [1, 0, 1],
            "period": [1, 0, 0],
            "duration": [2.0, 1.0, 3.0],
            "event": [1, 1, 0],
        },
        row_ids=("spell-3", "spell-0", "spell-1"),
    )
    data = PanelData(nodes, layout, outcomes=outcome)
    model = Regression(
        RightCensored("duration", "event"),
        Intercept() + Linear("x"),
        structure=layout,
        law=LogNormalAFT(),
    )
    prepared = model.prepare(data)
    assert isinstance(prepared, PreparedSurvival)
    assert prepared.context.row_ids == outcome.row_ids
    np.testing.assert_allclose(prepared.values[:, 1], [4.0, 1.0, 2.0])


@pytest.mark.integration
@pytest.mark.contract
def test_row_difference_recompiles_each_training_spline() -> None:
    """Equal coefficient names conceal different basis scales across two fits."""
    term = Spline("x", df=1, degree=1)
    model = Regression("y", Intercept() + term)
    question = RowDifference(at={"x": 10.0}, versus={"x": 0.0})
    fits = []
    for maximum in (10.0, 20.0):
        x = jnp.linspace(0, maximum, 31)
        fit = model.estimate({"x": x, "y": 2 * x + jnp.cos(x)})
        inference = fit.infer(covariance=HC("HC1"))
        delta = inference.contrast(question, reference="normal")
        expected = fit.predict({"x": [10.0, 0.0]})
        assert delta.valid
        assert delta.estimate == pytest.approx(
            float(expected[0] - expected[1]), rel=1e-6
        )
        fits.append(fit)
    compiled = fits[0].design.design_contrast(term, at=10.0, versus=0.0)
    assert fits[0].parameter_names == fits[1].parameter_names
    with pytest.raises(ValueError, match="different fitted design"):
        fits[1].infer(covariance=HC("HC1")).contrast(compiled, reference="normal")
    with pytest.raises(ValueError, match="exactly"):
        RowDifference({}, {"x": 2.0}).compile(fits[1].design)


@pytest.mark.component
@pytest.mark.contract
def test_row_difference_of_indicator_columns_is_signed() -> None:
    """Subtract categorical or intercept-only rows on the coefficient scale."""
    data = Observations({}, labels={"arm": ["control", "treated"]})
    design = FittedDesign((Intercept() + Categorical("arm"))._prepare_term(data))
    question = RowDifference(at={"arm": "control"}, versus={"arm": "treated"})
    np.testing.assert_array_equal(question.compile(design).coefficients, [0, -1])
    intercept = FittedDesign(Intercept()._prepare_term(data))
    np.testing.assert_array_equal(
        RowDifference({}, {}).compile(intercept).coefficients, [0]
    )
