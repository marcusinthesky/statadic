"""Retained named designs, safe formula compilation and distribution features."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import (
    Categorical,
    DistanceToLandmarks,
    FittedDesign,
    Intercept,
    Linear,
    LinearPredictorContrast,
    RowDifference,
    formula_to_terms,
    level_codes,
)
from statadic.api.inference import HC
from statadic.api.regression import Observed, Poisson, PoissonResult, Regression
from statadic.api.representation import Energy
from statadic.sample.packed import PackedClouds

from .._budgets import examples


@pytest.mark.component
@pytest.mark.contract
def test_categories_interactions_and_prediction_use_the_training_schema() -> None:
    """Prediction order never relearns categorical reference coding."""
    training = Observations(
        {"x": jnp.asarray([1.0, 2.0, 3.0, 4.0])}, labels={"g": ["b", "a", "c", "b"]}
    )
    terms = (
        Intercept() + Linear("x") + Categorical("g") + Linear("x") * Categorical("g")
    )
    design = FittedDesign(terms._prepare_term(training))
    prediction = Observations(
        {"x": jnp.asarray([5.0, 6.0, 7.0])}, labels={"g": ["c", "b", "a"]}
    )
    np.testing.assert_allclose(
        design.transform(prediction),
        [[1, 5, 0, 1, 0, 5], [1, 6, 1, 0, 6, 0], [1, 7, 0, 0, 0, 0]],
    )
    assert design.column_names == (
        "intercept",
        "x",
        "g[str:'b']",
        "g[str:'c']",
        "x:g[str:'b']",
        "x:g[str:'c']",
    )
    with pytest.raises(ValueError, match="Unseen"):
        design.transform(Observations({"x": jnp.asarray([1])}, labels={"g": ["new"]}))
    with pytest.raises(ValueError, match="unique"):
        FittedDesign((Linear("x") + Linear("x"))._prepare_term(training))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("precision", ["bfloat16", "float32", "float64"])
def test_indicator_terms_do_not_promote_numeric_feature_precision(
    precision: str,
) -> None:
    """Host convenience columns do not impose a new numerical dtype policy."""
    data = Observations(
        {"x": jnp.arange(5, dtype=precision)}, labels={"g": ["a", "b", "a", "b", "a"]}
    )
    design = FittedDesign(
        (Intercept() + Linear("x") + Categorical("g"))._prepare_term(data)
    )
    assert design.transform(data).dtype == jnp.dtype(precision)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("precision", ["bfloat16", "float32", "float64"])
def test_generated_only_design_uses_response_precision(precision: str) -> None:
    """A generated-only training design uses the response as floating context."""
    data = Observations(
        {"y": jnp.arange(5, dtype=precision)}, labels={"g": ["a", "b", "a", "b", "a"]}
    )
    prepared = Regression("y", Intercept() + Categorical("g")).prepare(data)
    assert prepared.values.dtype == jnp.dtype(precision)
    held_out = Observations({}, labels={"g": ["b", "a"]}, row_ids=(0, 1))
    generated = prepared.design.transform(held_out)
    assert jnp.result_type(generated, prepared.values) == jnp.dtype(precision)


@pytest.mark.component
@pytest.mark.contract
def test_bounded_formula_compiles_to_the_same_explicit_design() -> None:
    """Formula syntax only constructs canonical terms; it evaluates no user code."""
    data = Observations({"x": jnp.asarray([1.0, 2.0])}, labels={"g": ["a", "b"]})
    response, terms = formula_to_terms("y ~ 1 + x + C(g) + x:C(g)")
    expected = (
        Intercept() + Linear("x") + Categorical("g") + Linear("x") * Categorical("g")
    )
    assert response == "y"
    np.testing.assert_array_equal(
        FittedDesign(terms._prepare_term(data)).transform(data),
        FittedDesign(expected._prepare_term(data)).transform(data),
    )
    _, without_intercept = formula_to_terms("y ~ 0 + x")
    assert FittedDesign(without_intercept._prepare_term(data)).column_names == ("x",)
    for unsupported in (
        "y ~ x*x",
        "y ~ log(x)",
        "y ~ x - 1",
        "y ~ 0 + 1",
        "y ~ 0",
        "y ~ __import__('os')",
        "y ~ x +",
    ):
        with pytest.raises(ValueError, match=r"[Ff]ormula|Unsupported"):
            formula_to_terms(unsupported)


@pytest.mark.integration
def test_distribution_term_is_shared_by_ols_poisson_and_heldout_prediction() -> None:  # noqa: PLR0914
    """Empirical measures remain whole observations and landmarks stay frozen."""
    rng = np.random.default_rng(984)
    clouds = [rng.normal(size=(5 + index % 4, 2)) + index / 20 for index in range(36)]
    measures = PackedClouds.from_clouds(clouds)
    x = rng.normal(size=36)
    ids = tuple(range(36))
    column = MeasureColumn(ids, measures)
    base = Observations({"x": x}, measures={"cloud": column})
    geometry = Energy()
    landmark = DistanceToLandmarks("cloud", geometry, landmarks=(3, 21))
    features = landmark._prepare_term(base).transform(base)
    outcome = (
        0.4 + 0.3 * x + np.asarray(features) @ [0.5, -0.2] + rng.normal(size=36) * 0.1
    )
    counts = rng.poisson(np.exp(0.2 + 0.2 * x + 0.1 * np.asarray(features[:, 0])))
    training = Observations(
        {"x": x, "y": outcome, "count": counts}, measures={"cloud": column}
    )
    terms = Intercept() + Linear("x") + landmark
    linear = Regression("y", terms).estimate(training)
    poisson = Regression(Observed("count"), terms, law=Poisson()).estimate(training)
    assert isinstance(poisson, PoissonResult)
    assert linear.core.valid
    assert poisson.core.valid
    np.testing.assert_allclose(linear.core.design, poisson.core.design)
    heldout = Observations(
        {"x": jnp.asarray([-1.0, 0.2, 1.5])},
        row_ids=[100, 101, 102],
        measures={
            "cloud": MeasureColumn(
                (100, 101, 102),
                PackedClouds.from_clouds([rng.normal(size=(n, 2)) for n in (3, 7, 4)]),
            )
        },
    )
    transformed = linear.design.transform(heldout)
    np.testing.assert_allclose(
        transformed[:, -2:],
        geometry.prepare(measures.take(jnp.array([3, 21]))).cross(
            heldout.measure("cloud")
        ),
    )
    for fitted in (linear, poisson):
        prediction = fitted.predict(heldout)
        individual = jnp.concatenate(
            [fitted.predict(heldout.take([index])) for index in range(3)]
        )
        np.testing.assert_allclose(prediction, individual, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="training observation"):
        DistanceToLandmarks("cloud", geometry, landmarks=(100,))._prepare_term(training)


@pytest.mark.unit
@pytest.mark.contract
def test_default_level_order_is_sorted_and_mixed_labels_stay_distinct() -> None:
    """The reference is the smallest level; unlike label types are not ordered."""
    training = Observations(
        {"x": jnp.asarray([1.0, 2.0, 3.0])}, labels={"g": [10, 2, 7]}
    )
    fitted = Categorical("g")._prepare_term(training)
    assert fitted.levels == (2, 7, 10)
    assert fitted.reference == fitted.levels[0]
    assert level_codes(["b", "a", "b"])[0] == ("a", "b")
    np.testing.assert_array_equal(level_codes(["b", "a", "b"])[1], [1, 0, 1])
    levels, codes = level_codes(["1", 1, "a", 1])
    assert levels == (1, "1", "a")
    np.testing.assert_array_equal(codes, [1, 0, 2, 0])


@pytest.mark.unit
@pytest.mark.contract
def test_row_difference_compiles_to_a_typed_rows_estimand() -> None:
    """Row contrasts target a frozen rows-scope linear-predictor estimand."""
    training = Observations({"x": jnp.asarray([1.0, 2.0, 3.0])})
    design = FittedDesign((Intercept() + Linear("x"))._prepare_term(training))
    at = {"x": 2.0}
    contrast = RowDifference(at=at, versus={"x": 0.0}).compile(design)
    at["x"] = 9.0
    estimand = contrast.estimand
    assert isinstance(estimand, LinearPredictorContrast)
    assert estimand.scope == "rows"
    assert estimand.at == {"x": 2.0}
    assert estimand.describe() == ("linear predictor at {'x': 2.0} versus {'x': 0.0}")
    with pytest.raises(TypeError):
        estimand.at["x"] = 1.0  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.contract
def test_term_contrast_estimand_and_report_title() -> None:
    """Term contrasts hold single-entry mappings and title their reports."""
    x = jnp.arange(10.0)
    fit = Regression.ols("y", Linear("x")).estimate({"x": x, "y": x + jnp.sin(x)})
    term = Linear("x")
    contrast = fit.design.design_contrast(term, at=1.0, versus=0.0)
    estimand = contrast.estimand
    assert estimand.scope == "term"
    assert estimand.at == {"x": 1.0}
    assert estimand.versus == {"x": 0.0}
    assert estimand.describe() == (
        "structural linear-predictor contrast: x=1.0 versus 0.0"
    )
    with pytest.raises(TypeError):
        estimand.versus["x"] = 1.0  # type: ignore[index]
    result = fit.infer(covariance=HC("HC1")).contrast(contrast, reference="normal")
    assert result.estimand is estimand
    assert result.report().title == estimand.describe()


@pytest.mark.component
@pytest.mark.property
@examples(15)
@given(
    at=st.integers(-8, 8),
    versus=st.integers(-8, 8),
    scale=st.sampled_from([-2, -1, 2, 4]),
)
def test_contrasts_are_design_differences_and_linear(
    at: int, versus: int, scale: int
) -> None:
    """The public contrast targets the same linear predictor as prediction."""
    x = jnp.arange(10.0)
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
        {"x": x, "y": x + jnp.sin(x)}
    )
    inference = fit.infer(covariance=HC("HC1"))
    result = inference.contrast(
        RowDifference(at={"x": at}, versus={"x": versus}), reference="normal"
    )
    expected = (at - versus) * float(fit.core.parameters[1])
    assert result.estimate == pytest.approx(expected, rel=2e-5, abs=1e-6)
    scaled = inference.contrast(
        RowDifference(at={"x": scale * at}, versus={"x": scale * versus}),
        reference="normal",
    )
    assert scaled.estimate == pytest.approx(scale * expected, rel=2e-5, abs=1e-6)
