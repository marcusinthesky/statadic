"""Native regression facade parity, retained prediction roles and status reporting."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from inspect import unwrap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_predictor
from statadic.api.base import Model, RegressionResult, Result
from statadic.api.data import Observations, PanelLayout, RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.inference import HAC, HC, Clustered, ModelBased
from statadic.api.panel import PanelResult, PreparedPanel
from statadic.api.regression import (
    Binomial,
    LogisticResult,
    Observed,
    Poisson,
    PoissonResult,
    Regression,
)
from statadic.api.roles import InputRoleProvider  # noqa: TC002
from statadic.api.spatial.cross_section import SpatialInformation, SpatialRegressionResult
from statadic.api.spatial.graphs import PanelData, SpatialData, SpatialLag, SpatialX
from statadic.api.spatial.learning import GraphML
from statadic.api.survival import RightCensored
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    GLMFit,  # noqa: TC002  # runtime test annotation hook
    PoissonMLE,
    PoissonProblem,
)
from statadic.estimation.likelihood import ProbitMLE
from statadic.estimation.linear import (
    LeastSquares,
    LinearFit,  # noqa: TC002  # runtime test annotation hook
    LinearProblem,
)
from statadic.estimation.spatial import ML, SAR, SpatialProblem
from statadic.estimation.survival import SurvivalML
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.linear_covariance import (
    classical_covariance,
    cluster_covariance,
    hac_covariance,
    hc_covariance,
)
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.inference.spatial import spatial_covariance
from statadic.model.glm import BinomialLogitEquation, PoissonLogEquation  # noqa: TC002
from statadic.model.survival import LogNormalAFT, WeibullAFT
from statadic.operators.spatial import prepare_spatial_weights

from .._typing import ill_typed
from ._models import logit_model, poisson_model


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("shortcut", "explicit"),
    [
        (
            Regression.ols("y", Linear("x")),
            Regression("y", Linear("x"), estimator=LeastSquares()),
        ),
        (
            Regression.wls("y", Linear("x"), precision=RelativePrecision("w")),
            Regression(
                "y",
                Linear("x"),
                estimator=LeastSquares(),
                precision=RelativePrecision("w"),
            ),
        ),
        (
            Regression.logit(
                Observed("y", trials="n"),
                Linear("x"),
                offset="offset",
                estimator=BinomialLogitMLE(max_steps=31),
            ),
            Regression(
                Observed("y", trials="n"),
                Linear("x"),
                law=Binomial("logit"),
                offset="offset",
                estimator=BinomialLogitMLE(max_steps=31),
            ),
        ),
        (
            Regression.probit(
                Observed("y", trials="n"),
                Linear("x"),
                offset="offset",
                estimator=ProbitMLE(max_steps=31),
            ),
            Regression(
                Observed("y", trials="n"),
                Linear("x"),
                law=Binomial("probit"),
                offset="offset",
                estimator=ProbitMLE(max_steps=31),
            ),
        ),
        (
            Regression.poisson(
                "y",
                Linear("x"),
                offset="offset",
                exposure="hours",
                estimator=PoissonMLE(max_steps=31),
            ),
            Regression(
                "y",
                Linear("x"),
                law=Poisson(),
                offset="offset",
                exposure="hours",
                estimator=PoissonMLE(max_steps=31),
            ),
        ),
        (
            Regression.aft(
                RightCensored("y", "event", "entry"),
                Linear("x"),
                law=WeibullAFT(),
                estimator=SurvivalML(WeibullAFT(), max_steps=31),
            ),
            Regression(
                RightCensored("y", "event", "entry"),
                Linear("x"),
                law=WeibullAFT(),
                estimator=SurvivalML(WeibullAFT(), max_steps=31),
            ),
        ),
    ],
    ids=["ols", "wls", "logit", "probit", "poisson", "aft"],
)
def test_named_constructors_are_exact_canonical_specifications(
    shortcut: Regression, explicit: Regression
) -> None:
    """A familiar name supplies only the declared canonical components."""
    assert type(shortcut) is Regression
    assert shortcut is not explicit
    assert shortcut == explicit
    assert shortcut.input_columns == explicit.input_columns


@pytest.mark.unit
@pytest.mark.contract
def test_named_constructor_preparation_has_one_instance_owner() -> None:
    """Equal specifications own different preparations and add no intercept."""
    model = Regression.ols("y", Linear("x"))
    equivalent = Regression("y", Linear("x"), estimator=LeastSquares())
    data = Observations({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    prepared = model.prepare(data)
    assert model == equivalent
    assert not prepared.design.has_intercept
    assert model.law is None
    with pytest.raises(ValueError, match="another model instance"):
        equivalent.estimate_prepared(prepared)
    with pytest.raises(ValueError, match="another model instance"):
        replace(model).estimate_prepared(prepared)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("constructor", "options", "message"),
    [
        (Regression.logit, {"estimator": PoissonMLE()}, "BinomialLogitMLE"),
        (Regression.probit, {"estimator": BinomialLogitMLE()}, "ProbitMLE"),
        (Regression.poisson, {"estimator": ProbitMLE()}, "PoissonMLE"),
        (Regression.wls, {"precision": "w"}, "RelativePrecision"),
    ],
    ids=["logit", "probit", "poisson", "wls"],
)
def test_named_constructors_reject_incompatible_native_options(
    constructor: object, options: dict[str, object], message: str
) -> None:
    """The runtime guard also rejects inputs without the annotation hook."""
    with pytest.raises(TypeError, match=message):
        ill_typed(unwrap(ill_typed(constructor)))("y", Linear("x"), **options)


@pytest.mark.unit
@pytest.mark.contract
def test_named_constructors_keep_observation_and_structure_roles_explicit() -> None:
    """No shortcut supplies a new structure, trials owner or survival family."""
    terms = Intercept() + Linear("x")
    with pytest.raises(TypeError):
        ill_typed(Regression.ols)("y", terms, structure=PanelLayout("unit", "time"))
    with pytest.raises(TypeError):
        ill_typed(Regression.logit)("y", terms, trials="n")
    with pytest.raises(TypeError):
        ill_typed(Regression.wls)("y", terms)
    with pytest.raises(TypeError):
        ill_typed(Regression.aft)(RightCensored("y", "event"), terms)
    with pytest.raises(ValueError, match="matching SurvivalML family"):
        Regression.aft(
            RightCensored("y", "event"),
            terms,
            law=WeibullAFT(),
            estimator=SurvivalML(LogNormalAFT()),
        )
    with pytest.raises(ValueError, match="Only Binomial accepts observed trials"):
        Regression.ols(Observed("y", trials="n"), terms)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("model", "additional"),
    [
        (Regression("y", Linear("x")), {}),
        (
            Regression("y", Linear("x"), precision=RelativePrecision("weight")),
            {"weight": "numeric"},
        ),
        (
            poisson_model("y", Linear("x"), offset="offset", exposure="hours"),
            {"offset": "numeric", "hours": "numeric"},
        ),
        (
            logit_model("y", Linear("x"), offset="offset", trials="trials"),
            {"offset": "numeric", "trials": "numeric"},
        ),
    ],
)
def test_regression_models_declare_complete_input_roles(
    model: InputRoleProvider, additional: dict[str, str]
) -> None:
    """The public schema includes every family role without inspecting attributes."""
    assert dict(model.input_columns) == {"x": "numeric", "y": "numeric", **additional}


@pytest.mark.integration
@pytest.mark.contract
def test_injected_estimators_receive_the_retained_problem() -> None:
    """Structural estimator bricks participate without changing sample roles."""
    seen: list[LinearProblem | PoissonProblem | BinomialProblem] = []

    class LinearBrick:
        def estimate(self, problem: LinearProblem) -> LinearFit:
            seen.append(problem)
            return LeastSquares().estimate(problem)

    class PoissonBrick:
        def estimate(self, problem: PoissonProblem) -> GLMFit[PoissonLogEquation]:
            seen.append(problem)
            return PoissonMLE().estimate(problem)

    class LogisticBrick:
        def estimate(self, problem: BinomialProblem) -> GLMFit[BinomialLogitEquation]:
            seen.append(problem)
            return BinomialLogitMLE().estimate(problem)

    data = Observations(
        {
            "x": jnp.arange(8.0),
            "y": jnp.array([0, 1, 0, 1, 1, 0, 1, 0]),
            "w": jnp.arange(8.0) + 1,
        }
    )
    terms = Intercept() + Linear("x")
    Regression(
        "y", terms, estimator=LinearBrick(), precision=RelativePrecision("w")
    ).estimate(data)
    poisson_model("y", terms, estimator=PoissonBrick()).estimate(data)
    logit_model("y", terms, estimator=LogisticBrick()).estimate(data)
    assert len(seen) == 3  # noqa: PLR2004 -- one native problem for each injected brick
    assert isinstance(seen[0], LinearProblem)
    np.testing.assert_array_equal(seen[0].relative_precision, data.numeric("w"))
    assert isinstance(seen[1], PoissonProblem)
    assert isinstance(seen[2], BinomialProblem)


@pytest.mark.integration
@pytest.mark.contract
def test_linear_rejects_unsupported_covariance() -> None:
    """Production validation rejects unsupported specs before accessing fields."""
    script = """
import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept
from statadic.api.regression import Regression
from statadic.api.spatial.cross_section import SpatialInformation
data = Observations({"y": jnp.array([1., 2., 4.])})
result = Regression("y", Intercept()).estimate(data)
try:
    result.infer(covariance=SpatialInformation())
except TypeError as error:
    assert "Unsupported linear covariance" in str(error)
else:
    raise AssertionError("unsupported covariance accepted")
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


@pytest.mark.integration
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_linear_models_and_all_covariance_routes_reuse_the_native_state(  # noqa: PLR0914 -- explicit independent covariance controls
    precision: str,
) -> None:
    """Facade conveniences preserve estimates, weights, inference and fit identity."""
    rng = np.random.default_rng(51)
    x = jnp.asarray(rng.normal(size=40), dtype=precision)
    y = jnp.asarray(0.3 + 0.7 * x + rng.normal(size=40) * 0.4, dtype=precision)
    w = jnp.asarray(np.linspace(0.5, 2, 40), dtype=precision)
    data = Observations(
        {"x": x, "y": y, "w": w},
        labels={"group": tuple(np.repeat(["a", "b", "c", "d"], 10))},
    )
    terms = Intercept() + Linear("x")
    for model in (
        Regression("y", terms),
        Regression("y", terms, precision=RelativePrecision("w")),
    ):
        prepared = model.prepare(data)
        result = model.estimate_prepared(prepared)
        native = (
            LeastSquares().estimate(
                LinearProblem(
                    prepared.values, y, w, parameter_names=prepared.design.column_names
                )
            )
            if model.precision is not None
            else LeastSquares().estimate(
                LinearProblem(
                    prepared.values, y, parameter_names=prepared.design.column_names
                )
            )
        )
        assert result.core.valid == native.valid
        assert result.core.parameters.dtype == jnp.dtype(precision)
        np.testing.assert_allclose(result.core.parameters, native.parameters)
        np.testing.assert_allclose(result.predict(data), native.fitted_values)
        covariances = [
            ModelBased(),
            HC("HC0"),
            HC("HC1"),
            HC("HC2"),
            HC("HC3"),
            HAC(2, order_by=None),
            Clustered("group", 4 / 3),
        ]
        for spec in covariances:
            attached = result.infer(covariance=spec)
            assert attached.uncertainty.fit is result.core
            if isinstance(spec, ModelBased):
                expected = classical_covariance(native)
            elif isinstance(spec, HC):
                expected = hc_covariance(native, kind=spec.kind)
            elif isinstance(spec, HAC):
                expected = hac_covariance(native, 2)
            else:
                expected = cluster_covariance(
                    native,
                    jnp.repeat(jnp.arange(4), 10),
                    n_clusters=4,
                    finite_sample_correction=4 / 3,
                )
            np.testing.assert_allclose(
                attached.uncertainty.covariance.values, expected.covariance.values
            )
        inference = result.infer(covariance=ModelBased())
        interval = inference.intervals(
            reference="t", degrees_of_freedom=result.core.df_resid
        )
        expected_interval = coefficient_intervals(
            classical_covariance(native),
            reference="t",
            degrees_of_freedom=native.df_resid,
        )
        np.testing.assert_allclose(interval.estimate.ci, expected_interval.estimate.ci)
        tested = inference.test(
            Hypothesis(LinearRestriction.zero("x")),
            reference="f",
            degrees_of_freedom=result.core.df_resid,
        )
        expected_test = linear_restriction_test(
            classical_covariance(native),
            jnp.asarray([[0, 1]]),
            0,
            parameter_names=native.parameter_names,
            reference="f",
            degrees_of_freedom=native.df_resid,
        )
        assert tested.available
        assert tested.pvalue is not None
        np.testing.assert_allclose(tested.pvalue, expected_test.test.pvalue)
        assert all(
            row.pvalue is None and row.std_error is None
            for row in result.report().parameters
        )
        assert dict(inference.report().metadata)["reference"] is None
        assert (
            inference.report(reference="t", degrees_of_freedom=native.df_resid)
            .parameters[0]
            .ci_lower
            is not None
        )
        with pytest.raises(ValueError, match="another model"):
            Regression("y", terms).estimate_prepared(prepared)
        with pytest.raises(FrozenInstanceError):
            setattr(model, "response", "changed")  # noqa: B010


@pytest.mark.component
@pytest.mark.contract
def test_invalid_and_exact_fit_reports_preserve_unavailability() -> None:
    """Invalid coefficients and uncertainty remain explicitly unavailable."""
    data = Observations(
        {"x": jnp.asarray([1.0, 1.0, 1.0]), "y": jnp.asarray([1.0, 2.0, 4.0])}
    )
    invalid = Regression("y", Intercept() + Linear("x")).estimate(data)
    assert not invalid.core.valid
    assert all(
        row.estimate is None and not row.valid for row in invalid.report().parameters
    )
    inference = invalid.infer(covariance=ModelBased())
    assert all(
        row.estimate is None and row.std_error is None
        for row in inference.report().parameters
    )
    exact = Regression("y", Intercept() + Linear("x")).estimate(
        Observations(
            {
                "x": jnp.asarray([0.0, 1.0, 2.0, 3.0]),
                "y": jnp.asarray([1.0, 3.0, 5.0, 7.0]),
            }
        )
    )
    assert exact.core.valid
    report = exact.infer(covariance=ModelBased()).report(
        reference="t", degrees_of_freedom=exact.core.df_resid
    )
    assert all(
        row.estimate is not None and row.std_error is None and not row.valid
        for row in report.parameters
    )
    assert dict(report.metadata)["inference_status"] == "EXACT_FIT"


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.regression
def test_failed_interval_calibration_keeps_covariance_standard_errors() -> None:
    """A rejected reference withholds intervals only; HC3 SEs stay reported."""
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = Observations(
        {"x": x, "y": 1.0 + 0.5 * x + jnp.asarray([0.2, -0.1, 0.0, 0.3, -0.2, 0.1])}
    )
    inference = (
        Regression("y", Intercept() + Linear("x"))
        .estimate(data)
        .infer(covariance=HC("HC3"))
    )
    covariance_only = inference.report()
    failed = inference.report(reference="t", degrees_of_freedom=0)
    calibrated = inference.report(reference="normal")
    assert failed.require_available("std_error") is failed
    assert calibrated.require_available("interval", parameters=["x"]) is calibrated
    with pytest.raises(ValueError, match="INVALID_REFERENCE"):
        failed.require_available("interval")
    with pytest.raises(ValueError, match="not reported"):
        covariance_only.require_available("interval")
    with pytest.raises(ValueError, match="unknown"):
        calibrated.require_available("interval", parameters=["other"])
    for covariance_row, row in zip(
        covariance_only.parameters, failed.parameters, strict=True
    ):
        assert row.valid
        assert row.std_error == covariance_row.std_error
        assert row.std_error is not None
        assert row.ci_lower is None
        assert row.ci_upper is None
        assert row.interval_status == "unavailable"
        assert row.interval_unavailable_reason == "INVALID_REFERENCE"
    assert all(row.interval_status is None for row in covariance_only.parameters)
    assert all(
        row.interval_status == "available" and row.ci_lower is not None
        for row in calibrated.parameters
    )
    metadata = dict(failed.metadata)
    assert metadata["inference_status"] == "AVAILABLE"
    assert metadata["interval_status"] == "INVALID_REFERENCE"
    assert dict(covariance_only.metadata)["interval_status"] is None
    assert "interval: INVALID_REFERENCE" in failed.to_text()
    assert "interval: INVALID_REFERENCE" in failed.to_html()
    assert r"interval: INVALID\_REFERENCE" in failed.to_latex()


@pytest.mark.integration
def test_poisson_roles_predictions_and_covariance_match_the_native_fit() -> None:
    """Exposure/offset persist as roles and unsupported covariance stays explicit."""
    rng = np.random.default_rng(72)
    x = rng.normal(size=60)
    exposure = rng.uniform(0.5, 2.0, 60)
    offset = rng.normal(size=60) * 0.1
    y = rng.poisson(exposure * np.exp(0.2 + 0.3 * x + offset))
    data = Observations({"x": x, "count": y, "hours": exposure, "offset": offset})
    result = poisson_model(
        "count", Intercept() + Linear("x"), exposure="hours", offset="offset"
    ).estimate(data)
    assert isinstance(result, PoissonResult)
    options = {row.kind: row for row in result.prediction_options().rows}
    assert options["mean"].required_columns == ("x", "offset", "hours")
    assert options["link"].required_columns == options["mean"].required_columns
    assert options["rate"].required_columns == ("x", "offset")
    assert all(row.available for row in options.values())
    native = PoissonMLE().estimate(
        PoissonProblem(
            result.core.design,
            y,
            exposure=exposure,
            offset=offset,
            parameter_names=result.design.column_names,
        )
    )
    assert result.core.valid
    np.testing.assert_allclose(result.core.parameters, native.parameters)
    assert (
        tuple(row.name for row in result.report().parameters)
        == result.core.parameter_names
    )
    heldout = Observations(
        {
            "x": jnp.asarray([0.1, 0.7]),
            "hours": jnp.asarray([2.0, 5.0]),
            "offset": jnp.asarray([0.2, -0.1]),
        }
    )
    np.testing.assert_allclose(
        result.predict(heldout),
        native.predict(
            result.design.transform(heldout),
            exposure=jnp.asarray([2.0, 5.0]),
            offset=jnp.asarray([0.2, -0.1]),
            parameter_names=result.design.column_names,
        ),
    )
    np.testing.assert_allclose(
        result.predict(heldout, kind="rate"),
        result.predict(heldout) / heldout.numeric("hours"),
    )
    for spec in (ModelBased(), HC("HC0"), HC("HC1")):
        method = "model" if isinstance(spec, ModelBased) else spec.kind
        assert method in {"model", "HC0", "HC1"}
        np.testing.assert_allclose(
            result.infer(covariance=spec).uncertainty.covariance.values,
            glm_covariance(native, method=method).covariance.values,
        )
    with pytest.raises(TypeError, match="only model, HC0 and HC1"):
        result.infer(covariance=HC("HC2"))
    with pytest.raises(KeyError):
        result.predict(Observations({"x": jnp.asarray([1.0])}))


@pytest.mark.integration
def test_logistic_roles_predictions_and_covariance_match_the_native_fit(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Trials/offset persist as roles; the class reaches the brick's exact fit."""
    rng = np.random.default_rng(73)
    x = rng.normal(size=80)
    trials = rng.integers(1, 6, 80).astype(float)
    offset = rng.normal(size=80) * 0.1
    probability = 1 / (1 + np.exp(-(-0.2 + 0.6 * x + offset)))
    successes = rng.binomial(trials.astype(int), probability).astype(float)
    data = Observations(
        {"x": x, "successes": successes, "trials": trials, "offset": offset}
    )
    model = logit_model(
        "successes", Intercept() + Linear("x"), trials="trials", offset="offset"
    )
    result = model.estimate(data)
    assert isinstance(result, LogisticResult)
    options = {row.kind: row for row in result.prediction_options().rows}
    assert options["mean"].required_columns == ("x", "offset", "trials")
    assert options["probability"].required_columns == ("x", "offset")
    native = BinomialLogitMLE().estimate(
        BinomialProblem(
            result.core.design,
            successes,
            trials=trials,
            offset=offset,
            parameter_names=result.design.column_names,
        )
    )
    assert isinstance(model, Regression)
    assert isinstance(model, Model)
    assert isinstance(result, RegressionResult)
    assert isinstance(result, Result)
    assert result.core.valid
    np.testing.assert_array_equal(result.core.parameters, native.parameters)
    report = result.report()
    assert tuple(row.name for row in report.parameters) == result.core.parameter_names
    assert dict(report.metadata)["trials"] == "trials"
    heldout = Observations(
        {
            "x": jnp.asarray([0.1, 0.7]),
            "trials": jnp.asarray([2.0, 5.0]),
            "offset": jnp.asarray([0.2, -0.1]),
        }
    )
    result.save_predictor(tmp_path / "logistic")
    restored = load_predictor(tmp_path / "logistic")
    for kind in ("probability", "mean", "link"):
        np.testing.assert_allclose(
            restored.predict(heldout, kind=kind), result.predict(heldout, kind=kind)
        )
    np.testing.assert_allclose(restored.predict(heldout), result.predict(heldout))
    assert restored.capabilities == ("predict",)
    assert not hasattr(restored, "infer")
    assert not hasattr(restored, "estimate")
    assert not hasattr(restored, "fit")
    invalid = Observations(
        {
            "x": jnp.asarray([0.1, 0.7]),
            "trials": jnp.asarray([0.0, 1.5]),
            "offset": jnp.asarray([0.2, -0.1]),
        }
    )
    assert jnp.all(jnp.isnan(restored.predict(invalid, kind="mean")))
    with pytest.raises(KeyError):
        restored.predict(Observations({"x": jnp.asarray([0.1])}))
    with pytest.raises(ValueError, match="Prediction kind"):
        restored.predict(heldout, kind="linear")
    with pytest.raises(ValueError, match="nonfinite"):
        Observations(
            {
                "x": jnp.asarray([0.1]),
                "trials": jnp.asarray([2.0]),
                "offset": jnp.asarray([jnp.inf]),
            }
        )
    missing_offset = Observations(
        {"x": jnp.asarray([0.1]), "trials": jnp.asarray([2.0])}
    )
    for kind in ("probability", "mean", "link"):
        np.testing.assert_array_equal(
            jnp.isnan(restored.predict(invalid, kind=kind)),
            jnp.isnan(result.predict(invalid, kind=kind)),
        )
        with pytest.raises(KeyError):
            restored.predict(missing_offset, kind=kind)
    overflowing = replace(
        result, core=replace(result.core, parameters=jnp.asarray([0.0, 2.0]))
    )
    overflowing.save_predictor(tmp_path / "overflow")
    overflow_artifact = load_predictor(tmp_path / "overflow")
    extreme = Observations(
        {
            "x": jnp.asarray([jnp.finfo(float).max, -jnp.finfo(float).max]),
            "trials": jnp.asarray([2.0, 3.0]),
            "offset": jnp.asarray([0.0, 0.0]),
        }
    )
    for kind in ("probability", "mean", "link"):
        np.testing.assert_array_equal(
            overflow_artifact.predict(extreme, kind=kind),
            overflowing.predict(extreme, kind=kind),
        )
    design = result.design.transform(heldout)
    roles = {
        "trials": jnp.asarray([2.0, 5.0]),
        "offset": jnp.asarray([0.2, -0.1]),
        "parameter_names": result.design.column_names,
    }
    np.testing.assert_allclose(
        result.predict(heldout), native.predict(design, scale="response", **roles)
    )
    np.testing.assert_allclose(
        result.predict(heldout, kind="mean"),
        result.predict(heldout) * heldout.numeric("trials"),
    )
    np.testing.assert_allclose(
        result.predict(heldout, kind="link"),
        native.predict(design, scale="link", **roles),
    )
    for spec in (ModelBased(), HC("HC0"), HC("HC1")):
        method = "model" if isinstance(spec, ModelBased) else spec.kind
        assert method in {"model", "HC0", "HC1"}
        np.testing.assert_allclose(
            result.infer(covariance=spec).uncertainty.covariance.values,
            glm_covariance(native, method=method).covariance.values,
        )
    with pytest.raises(TypeError, match="only model, HC0 and HC1"):
        result.infer(covariance=HC("HC2"))
    with pytest.raises(ValueError, match="another model instance"):
        model.estimate_prepared(
            logit_model(
                "successes", Intercept() + Linear("x"), trials="trials"
            ).prepare(data)
        )


@pytest.mark.integration
@pytest.mark.reference
def test_sar_reuses_likelihood_and_rejects_changed_prediction_rosters() -> None:
    """Structural fitted values and reduced-form prediction retain distinct targets."""
    rng = np.random.default_rng(12)
    units = 32
    w = 0.7 * np.roll(np.eye(units), 1, axis=1) + 0.3 * np.roll(
        np.eye(units), -2, axis=1
    )
    x = rng.normal(size=units)
    design = np.column_stack([np.ones(units), x])
    y = np.linalg.solve(
        np.eye(units) - 0.2 * w, design @ [0.4, 0.8] + rng.normal(size=units) * 0.3
    )
    graph = prepare_spatial_weights(w, unit_ids=np.arange(units) + 200)
    data = Observations({"x": x, "y": y}, row_ids=list(range(200, 200 + units)))
    model = Regression(
        "y", Intercept() + Linear("x"), dependence=SpatialLag("neighbors")
    )
    result = model.estimate(SpatialData(data, {"neighbors": graph}))
    assert isinstance(result, SpatialRegressionResult)
    targets = {row.kind: row for row in result.prediction_options().rows}
    assert targets["structural"].required_columns == ("x", "y")
    assert targets["reduced-form"].required_columns == ("x",)
    native = ML().estimate(
        SpatialProblem(
            design,
            y,
            graph,
            process=SAR(),
            graph_name="neighbors",
            parameter_names=result.design.column_names,
        )
    )
    assert result.core.valid
    assert tuple(row.name for row in result.report().parameters) == (
        "intercept",
        "x",
        "rho[neighbors]",
    )
    np.testing.assert_allclose(result.core.parameters, native.parameters)
    np.testing.assert_allclose(result.predict(data), native.reduced_form)
    expected_reduced_form = np.linalg.solve(
        np.eye(units) - float(result.core.parameters[-1]) * w,
        design @ np.asarray(result.core.parameters[:-1]),
    )
    np.testing.assert_allclose(
        result.predict(data, kind="reduced-form"), expected_reduced_form
    )
    np.testing.assert_allclose(result.structural_fitted(), native.fitted_values)
    np.testing.assert_allclose(
        result.infer(covariance=SpatialInformation()).uncertainty.covariance.values,
        spatial_covariance(native).covariance.values,
    )
    assert result.impacts().valid
    with pytest.raises(ValueError, match="ordered numeric"):
        result.predict(data.take(list(reversed(range(units)))))
    changed = prepare_spatial_weights(w, unit_ids=np.arange(units) + 201)
    with pytest.raises(ValueError, match="changed spatial unit roster"):
        result.predict(data, weights=changed, kind="reduced-form")


@pytest.mark.unit
@pytest.mark.contract
def test_only_canonical_regression_specifications_are_exported() -> None:
    """Retired host constructors cannot compete with Regression."""
    from statadic.api import dyadic, panel, regression, survival  # noqa: PLC0415

    for name in (
        "OLS",
        "WLS",
        "Conditional",
        "IVRegression",
        "Probability",
        "LogMean",
        "LogRate",
    ):
        assert not hasattr(regression, name)
    assert not hasattr(panel, "PanelOLS")
    assert not hasattr(panel, "PanelRegression")
    assert not hasattr(dyadic, "DyadicRegression")
    assert not hasattr(survival, "SurvivalRegression")


@pytest.mark.component
@pytest.mark.contract
def test_panel_data_preserves_the_actual_prepared_result_family() -> None:
    """An inferred panel layout changes the native result and its capabilities."""
    rows = Observations(
        {
            "x": [0.0, 1.0, 2.0, 1.0, 3.0, 4.0],
            "y": [1.0, 2.0, 2.5, 2.0, 4.0, 4.5],
            "time": [0, 1, 2, 0, 1, 2],
        },
        labels={"unit": ["a", "a", "a", "b", "b", "b"]},
    )
    data = PanelData(rows, PanelLayout("unit", "time"))
    model = Regression("y", Intercept() + Linear("x"))
    prepared = model.prepare(data)
    assert isinstance(prepared, PreparedPanel)
    result = model.estimate_prepared(prepared)
    assert isinstance(result, PanelResult)
    assert not hasattr(result, "diagnose")
    np.testing.assert_array_equal(
        result.core.parameters, model.estimate(data).core.parameters
    )


@pytest.mark.unit
@pytest.mark.contract
def test_graph_learning_accepts_spatial_predictors_without_dependence() -> None:
    """A learned SLX graph is scientific even without a stochastic spatial lag."""
    terms = Intercept() + Linear("x") + SpatialX("x", "neighbors")
    model = Regression("y", terms, estimator=GraphML())
    assert model.dependence is None
    with pytest.raises(ValueError, match="dependence or SpatialX"):
        Regression("y", Intercept() + Linear("x"), estimator=GraphML())
    with pytest.raises(ValueError, match="require dependence"):
        Regression("y", terms, estimator=ML())


if TYPE_CHECKING:
    from typing import assert_type

    from statadic.api.base import PreparedRegression
    from statadic.api.regression import (
        IVResult,
        LikelihoodResult,
        LinearResult,
        RegressionFit,
    )
    from statadic.api.spatial.graph_regression import (
        GraphRegressionResult,
        PreparedGraphRegression,
    )
    from statadic.api.spatial.observed import ObservedSpatialPanelResult
    from statadic.api.spatial.panel import SpatialPanelResult
    from statadic.api.survival import PreparedSurvival, SurvivalResult

    def _named_constructor_types(data: Observations) -> None:
        """Every shortcut preserves the concrete family's prepared/result types."""
        terms = Intercept() + Linear("x")
        ols = Regression.ols("y", terms)
        assert_type(ols, Regression[LinearResult, PreparedRegression])
        assert_type(ols.estimate(data), LinearResult)
        assert_type(ols.estimate_prepared(ols.prepare(data)), LinearResult)
        wls = Regression.wls("y", terms, precision=RelativePrecision("w"))
        assert_type(wls, Regression[LinearResult, PreparedRegression])
        assert_type(wls.estimate(data), LinearResult)
        logit = Regression.logit(Observed("y", trials="n"), terms)
        assert_type(logit, Regression[LogisticResult, PreparedRegression])
        assert_type(logit.estimate(data), LogisticResult)
        probit = Regression.probit("y", terms)
        assert_type(probit, Regression[LikelihoodResult, PreparedRegression])
        assert_type(probit.estimate(data), LikelihoodResult)
        poisson = Regression.poisson("y", terms, exposure="hours")
        assert_type(poisson, Regression[PoissonResult, PreparedRegression])
        assert_type(poisson.estimate(data), PoissonResult)
        aft = Regression.aft(RightCensored("y", "event"), terms, law=WeibullAFT())
        assert_type(aft, Regression[SurvivalResult, PreparedSurvival])
        assert_type(aft.estimate(data), SurvivalResult)
        assert_type(aft.estimate_prepared(aft.prepare(data)), SurvivalResult)

    def _constructor_types(data: Observations) -> None:
        """Preserve ordinary family capabilities through the sole constructor."""
        linear = Regression("y", Intercept())
        assert_type(linear.prepare(data), PreparedRegression)
        assert_type(linear.estimate(data), LinearResult)
        assert_type(linear.estimate_prepared(linear.prepare(data)), LinearResult)
        assert_type(
            Regression("y", Intercept(), law=Poisson()).estimate(data), PoissonResult
        )
        poisson = Regression("y", Intercept(), law=Poisson())
        assert_type(poisson.estimate_prepared(poisson.prepare(data)), PoissonResult)
        assert_type(
            Regression(Observed("y"), Intercept(), law=Binomial()).estimate(data),
            LogisticResult,
        )
        logistic = Regression(Observed("y"), Intercept(), law=Binomial())
        assert_type(logistic.estimate_prepared(logistic.prepare(data)), LogisticResult)
        assert_type(
            Regression(
                RightCensored("y", "event"), Intercept(), law=WeibullAFT()
            ).estimate(data),
            SurvivalResult,
        )
        assert_type(
            Regression(
                RightCensored("y", "event"),
                Intercept(),
                estimator=SurvivalML(WeibullAFT()),
                dependence=None,
            ).estimate(data),
            SurvivalResult,
        )

    def _iv_role_types(
        data: Observations,
        endogenous: tuple[str, ...],
        instruments: tuple[str, ...],
    ) -> None:
        """Require nonempty role evidence before promising IV capabilities."""
        terms = Intercept() + Linear("x")
        assert_type(
            Regression("y", terms, endogenous=("d",), instruments=("z",)).estimate(
                data
            ),
            IVResult,
        )
        assert_type(
            Regression("y", terms, endogenous=(), instruments=()).estimate(data),
            RegressionFit,
        )
        assert_type(
            Regression(
                "y", terms, endogenous=endogenous, instruments=instruments
            ).estimate(data),
            RegressionFit,
        )

    def _erased_conditional_types(model: Regression, data: Observations) -> None:
        """Keep conditional possibilities when the model family was erased."""
        prepared = model.prepare(data)
        if isinstance(prepared, PreparedRegression):
            assert_type(model.estimate_prepared(prepared), RegressionFit)

    def _data_driven_types(panel: PanelData, spatial: SpatialData) -> None:
        """Retain data-driven Gaussian capabilities through both lifecycle paths."""
        terms = Intercept() + Linear("x")
        linear = Regression("y", terms)
        prepared = linear.prepare(panel)
        assert_type(prepared, PreparedPanel | PreparedGraphRegression)
        assert_type(linear.estimate(panel), PanelResult | GraphRegressionResult)
        assert_type(
            linear.estimate_prepared(prepared), PanelResult | GraphRegressionResult
        )
        if isinstance(prepared, PreparedPanel):
            assert_type(linear.estimate_prepared(prepared), PanelResult)
        else:
            assert_type(linear.estimate_prepared(prepared), GraphRegressionResult)
        structured = Regression("y", terms, structure=PanelLayout("unit", "time"))
        assert_type(structured.prepare(panel), PreparedPanel | PreparedGraphRegression)
        assert_type(structured.estimate(panel), PanelResult | GraphRegressionResult)
        dependence = Regression("y", terms, dependence=SpatialLag("neighbors"))
        assert_type(
            dependence.estimate(panel),
            SpatialPanelResult | ObservedSpatialPanelResult | GraphRegressionResult,
        )
        assert_type(
            dependence.estimate(spatial),
            SpatialRegressionResult | GraphRegressionResult,
        )
