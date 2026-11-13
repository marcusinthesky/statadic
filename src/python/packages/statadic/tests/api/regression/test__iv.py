"""IV host alignment and explicit inference requests."""

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.base import Estimator  # noqa: TC002
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import HansenJ, JarqueBera
from statadic.api.hypotheses import TestOutcome
from statadic.api.inference import HAC, HC, Clustered
from statadic.api.regression import IVResult, Regression
from statadic.core.equations import PlainLongRun
from statadic.estimation.iv import IVFit, IVProblem, TwoStepIVGMM  # noqa: TC002
from statadic.inference.iv import hansen_j, iv_covariance

from ..._typing import ill_typed


@pytest.mark.unit
@pytest.mark.contract
def test_iv_input_roles_preserve_numeric_cluster_columns() -> None:
    """An existing numeric cluster keeps its role; an extra cluster becomes labels."""
    numeric = Regression(
        "y",
        Linear("group") + Linear("x"),
        clusters="group",
        endogenous=("x",),
        instruments=("z",),
    )
    assert numeric.input_columns == {
        "y": "numeric",
        "group": "numeric",
        "x": "numeric",
        "z": "numeric",
    }
    labels = Regression(
        "y",
        Intercept() + Linear("x"),
        clusters="group",
        endogenous=("x",),
        instruments=("z",),
    )
    assert labels.input_columns == {
        "y": "numeric",
        "group": "categorical",
        "x": "numeric",
        "z": "numeric",
    }


@pytest.mark.integration
def test_host_retains_rows_predicts_structural_index_and_selects_cluster() -> None:
    """Check the retained IV contract against the selected witness."""
    rng = np.random.default_rng(15)
    z = rng.normal(size=80)
    x = z + rng.normal(size=80)
    y = 1 + 2 * x + rng.normal(size=80)
    data = Observations(
        {"x": x, "z": z, "y": y},
        row_ids=tuple(range(100, 180)),
        labels={"group": tuple(int(v) for v in np.repeat(np.arange(8), 10))},
    )
    model = Regression(
        "y", Intercept() + Linear("x"), endogenous=("x",), instruments=("z",)
    )
    prepared = model.prepare(data)
    fit = model.estimate_prepared(prepared)
    assert fit.context.row_ids == data.row_ids
    np.testing.assert_allclose(
        fit.predict(data),
        np.asarray(fit.parameters)[0] + np.asarray(fit.parameters)[1] * x,
        rtol=1e-5,
    )
    assert bool(fit.infer(covariance=HC()).uncertainty.valid)
    assert bool(fit.infer(covariance=Clustered("group")).uncertainty.valid)
    assert any("first_stage_partial_r2" in row.name for row in fit.report().metrics)
    with pytest.raises(ValueError, match="another model"):
        Regression(
            "y", Intercept() + Linear("x"), endogenous=("x",), instruments=("z",)
        ).estimate_prepared(prepared)


@pytest.mark.component
@pytest.mark.contract
def test_custom_iv_estimator_must_return_matching_native_schema() -> None:
    """A structurally unrelated or mismatched custom fit cannot enter the result."""
    from dataclasses import replace  # noqa: PLC0415

    from statadic.estimation.iv import TwoStageLeastSquares  # noqa: PLC0415

    data = Observations(
        {"x": np.arange(8.0), "z": np.arange(8.0) ** 2, "y": np.arange(8.0)}
    )

    class BadType:
        def estimate(self, _problem: IVProblem) -> object:
            return object()

    class BadNames:
        def estimate(self, problem: IVProblem) -> IVFit:
            return replace(
                TwoStageLeastSquares().estimate(problem),
                parameter_names=("bad", "names"),
            )

    class BadRows:
        def estimate(self, problem: IVProblem) -> IVFit:
            fit = TwoStageLeastSquares().estimate(problem)
            return replace(fit, design=fit.design[:-1])

    for estimator, error in (
        (BadType(), TypeError),
        (BadNames(), ValueError),
        (BadRows(), ValueError),
    ):
        with pytest.raises(error, match="IV estimator"):
            Regression(
                "y",
                Intercept() + Linear("x"),
                estimator=cast("Estimator[IVProblem, IVFit]", estimator),
                endogenous=("x",),
                instruments=("z",),
            ).estimate(data)


@pytest.mark.component
@pytest.mark.contract
def test_iv_hansen_diagnostic_and_ordered_hac_preserve_actual_fit() -> None:
    """J keeps native eligibility while post-fit HAC orders complete moment scores."""
    rng = np.random.default_rng(81)
    z = jnp.asarray(rng.normal(size=(50, 2)))
    x = z[:, 0] + 0.3 * z[:, 1] + jnp.asarray(rng.normal(size=50))
    y = 1 + 2 * x + jnp.asarray(rng.normal(size=50))
    time = jnp.asarray(rng.permutation(50))
    fitted = Regression(
        "y",
        Intercept() + Linear("x"),
        endogenous=("x",),
        instruments=("z1", "z2"),
        estimator=TwoStepIVGMM(PlainLongRun),
    ).estimate({"x": x, "y": y, "z1": z[:, 0], "z2": z[:, 1], "time": time})
    assert isinstance(fitted, IVResult)
    core, weight = fitted.core, fitted.core.affine.weight
    for acknowledge in (False, True):
        native = hansen_j(core, assume_correct_specification=acknowledge)
        outcome = fitted.diagnose(HansenJ(assume_correct_specification=acknowledge))[
            "hansen_j"
        ]
        assert isinstance(outcome, TestOutcome)
        assert outcome.available == bool(native.valid)
        if outcome.available:
            assert outcome.statistic == pytest.approx(float(native.statistic))
            assert outcome.pvalue == pytest.approx(float(native.pvalue))
        else:
            assert outcome.statistic is None
            assert outcome.unavailable_reason
    order = jnp.argsort(time)
    inferred = fitted.infer(covariance=HAC(2, order_by="time"))
    expected = iv_covariance(core, kind="HAC", bandwidth=2, row_order=order)
    np.testing.assert_allclose(
        inferred.uncertainty.covariance.values, expected.covariance.values
    )
    np.testing.assert_allclose(
        inferred.uncertainty.covariance.values,
        _numpy_iv_hac(core, order),
        rtol=5e-5 if core.parameters.dtype == jnp.float32 else 1e-9,
        atol=1e-9,
    )
    assert inferred.uncertainty.fit is core
    assert fitted.core is core
    np.testing.assert_array_equal(fitted.core.affine.weight, weight)
    assert inferred.report().metadata["order_by"] == "time"
    with pytest.raises(TypeError):
        ill_typed(fitted.diagnose)(JarqueBera())
    unknown = replace(core, method="unknown")
    assert not bool(
        jax.jit(hansen_j, static_argnames=("assume_correct_specification",))(
            unknown, assume_correct_specification=True
        ).valid
    )


@pytest.mark.component
@pytest.mark.contract
def test_iv_hansen_unavailable_reasons_name_the_failed_condition() -> None:
    """Each ineligible fit reports its own cause, matching native availability."""
    from statadic.estimation.iv import (  # noqa: PLC0415
        IteratedIVGMM,
        TwoStageLeastSquares,
    )

    rng = np.random.default_rng(82)
    z = rng.normal(size=(60, 2))
    x = z[:, 0] + 0.3 * z[:, 1] + rng.normal(size=60)
    y = 1 + 2 * x + rng.normal(size=60)
    data = {"x": x, "y": y, "z1": z[:, 0], "z2": z[:, 1]}

    def reason(
        instruments: tuple[str, ...],
        estimator: Estimator[IVProblem, IVFit],
        *,
        acknowledge: bool = True,
    ) -> tuple[str | None, IVFit]:
        fitted = Regression(
            "y",
            Intercept() + Linear("x"),
            estimator=estimator,
            endogenous=("x",),
            instruments=instruments,
        ).estimate(data)
        assert isinstance(fitted, IVResult)
        outcome = fitted.diagnose(HansenJ(assume_correct_specification=acknowledge))[
            "hansen_j"
        ]
        assert isinstance(outcome, TestOutcome)
        native = hansen_j(fitted.core, assume_correct_specification=acknowledge)
        assert outcome.available == bool(native.valid)
        return outcome.unavailable_reason, fitted.core

    overidentified = ("z1", "z2")
    assert reason(overidentified, TwoStepIVGMM(PlainLongRun))[0] is None
    assert (
        reason(overidentified, TwoStepIVGMM(PlainLongRun), acknowledge=False)[0]
        == "conventional-assumptions-not-acknowledged"
    )
    assert reason(overidentified, TwoStageLeastSquares())[0] == "weight-not-efficient"
    assert reason(("z1",), TwoStepIVGMM(PlainLongRun))[0] == "not-overidentified"
    stopped, core = reason(
        overidentified, IteratedIVGMM(PlainLongRun, max_iter=1, tolerance=1e-30)
    )
    assert not bool(core.converged)
    assert stopped == "weight-iteration-not-converged"


def _numpy_iv_hac(fit: IVFit, order: jax.Array) -> np.ndarray:
    """Independent Bartlett score variability with the actual retained moment map."""
    scores = np.asarray(fit.affine.scores.values, dtype=np.float64)[np.asarray(order)]
    omega = scores.T @ scores
    for lag in (1, 2):
        cross = scores[lag:].T @ scores[:-lag]
        omega += (1.0 - lag / 3.0) * (cross + cross.T)
    count = scores.shape[0]
    moment_map = np.asarray(fit.affine.moment_map, dtype=np.float64)
    return moment_map @ (omega / count) @ moment_map.T / count
