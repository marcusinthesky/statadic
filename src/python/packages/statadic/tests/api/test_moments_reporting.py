"""Host reports preserve native moment inference and availability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.diagnostics import HansenJ
from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.hypotheses import (
    Hypothesis,
    LinearRestriction,
    OveridentifyingRestrictions,
    TestOutcome,
)
from statadic.api.moments import (
    GELCalibration,
    GELInferenceResult,
    GMMSandwich,
    MomentData,
)
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.estimation.gel import FixedGEL
from statadic.estimation.gmm import IteratedGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gmm import gmm_covariance, hansen_j_from_fit

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path


def _identity(rows: tuple[int | str, ...] | list[int | str]) -> dict[str, object]:
    """Independently re-hash ordered row ids as reports promise."""
    payload = json.dumps(list(rows), separators=(",", ":"), ensure_ascii=False)
    return {"count": len(rows), "sha256": hashlib.sha256(payload.encode()).hexdigest()}


@pytest.mark.integration
def test_gmm_report_preserves_native_covariance_and_invalid_correction() -> None:
    """Keep native calculations and encode nonfinite correction metadata."""
    t = jnp.arange(30.0)
    moments = jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1)
    problem = AffineMomentProblem(moments, jnp.array([[1.0], [0.5]]))
    fit = Estimand(
        MomentConditions(),
        estimator=IteratedGMM(PlainLongRun, parameter_names=("theta",)),
    ).estimate(MomentData.from_arrays(problem, row_ids=tuple(range(30))))
    result = fit.infer(covariance=GMMSandwich(finite_sample_correction=1.5))
    native = gmm_covariance(fit.core, finite_sample_correction=1.5)
    np.testing.assert_allclose(
        result.uncertainty.covariance.values, native.covariance.values
    )
    assert result.uncertainty.fit is fit.core
    options = result.report().metadata["covariance_options"]
    assert isinstance(options, Mapping)
    assert options["finite_sample_correction"] == pytest.approx(1.5)
    assert result.report().metadata["scheme"] == "PlainLongRun"
    assert result.report().metadata["bandwidth"] == fit.core.bandwidth
    assert result.report(reference="normal").parameters[0].ci_lower is not None
    assert bool(
        result.test(
            Hypothesis(LinearRestriction.zero("theta")), reference="chi2"
        ).available
    )
    invalid = fit.infer(covariance=GMMSandwich(finite_sample_correction=float("nan")))
    assert not bool(invalid.uncertainty.valid)
    options = invalid.report().metadata["covariance_options"]
    assert isinstance(options, Mapping)
    assert options["finite_sample_correction"] is None
    assert options["finite_sample_correction_reason"] == "nonfinite-option"


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("acknowledge", [False, True])
def test_hansen_diagnostic_preserves_native_moment_reference(
    *, acknowledge: bool
) -> None:
    """The host retains actual efficient-weight and assumption availability evidence."""
    t = jnp.arange(30.0)
    problem = AffineMomentProblem(
        jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1), jnp.array([[1.0], [0.5]])
    )
    fit = Estimand(
        MomentConditions(),
        estimator=IteratedGMM(PlainLongRun),
    ).estimate(MomentData.from_arrays(problem, row_ids=tuple(range(30))))
    native = hansen_j_from_fit(fit.core, assume_correct_specification=acknowledge)
    outcome = fit.diagnose(HansenJ(assume_correct_specification=acknowledge))[
        "hansen_j"
    ]
    assert isinstance(outcome, TestOutcome)
    assert outcome.available == bool(native.valid)
    assert outcome.construction["efficient_weight"] == bool(native.efficient_weight)
    assert outcome.construction["native_status"] == int(native.status)
    assert outcome.construction["rows"] == _identity(fit.context.row_ids)
    if outcome.available:
        assert outcome.statistic == pytest.approx(float(native.test.statistic))
        assert outcome.pvalue == pytest.approx(float(native.test.pvalue))
    else:
        assert outcome.statistic is None
        assert outcome.pvalue is None
        assert outcome.unavailable_reason


@pytest.mark.component
@pytest.mark.contract
def test_invalid_gmm_fit_retains_unavailable_inference() -> None:
    """Unidentified fits keep native failure evidence and render without errors."""
    t = jnp.arange(30.0)
    moments = jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1)
    fit = Estimand(
        MomentConditions(), estimator=IteratedGMM(MeanCenteredLongRun)
    ).estimate(
        MomentData.from_arrays(
            AffineMomentProblem(moments, jnp.zeros((2, 1))),
            row_ids=tuple(range(30)),
        ),
    )
    inferred = fit.infer(covariance=GMMSandwich())
    assert not bool(inferred.uncertainty.valid)
    assert inferred.uncertainty.fit is fit.core
    report = inferred.report()
    assert report.parameters[0].std_error is None
    assert report.parameters[0].unavailable_reason == "INVALID_FIT"
    options = report.metadata["covariance_options"]
    assert isinstance(options, Mapping)
    assert report.metadata["scheme"] == "MeanCenteredLongRun"


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
@pytest.mark.subprocess
@pytest.mark.requires_program("pyrefly")
def test_gmm_host_retains_concrete_uncertainty_type(tmp_path: Path) -> None:
    """Native GMM state remains accessible without casts through the host API."""
    check_types(
        """
from typing import assert_type
from statadic.api.moments import GMMResult, GMMSandwich
from statadic.core.equations import PlainLongRun
from statadic.inference.gmm import GMMUncertainty

def native(result: GMMResult[PlainLongRun]) -> int:
    inferred = result.infer(covariance=GMMSandwich())
    assert_type(inferred.uncertainty, GMMUncertainty[PlainLongRun])
    return inferred.uncertainty.fit.bandwidth
""",
        "concrete_gmm_uncertainty.py",
        tmp_path,
    )


@pytest.mark.component
@pytest.mark.contract
def test_gel_report_preserves_metric_availability(  # noqa: PLR0914
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose each native quantity only when finite."""
    t = jnp.arange(30.0)
    moments = jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1)
    fit = Estimand(MomentConditions(), estimator=FixedGEL()).estimate(
        MomentData.from_arrays(FixedMomentProblem(moments), row_ids=tuple(range(30))),
    )
    bound = fit.infer(calibration=GELCalibration())
    outcome = bound.test(Hypothesis(OveridentifyingRestrictions(2, 0)))
    assert outcome.evidence is not None
    inference = GELInferenceResult(outcome.evidence, fit.context)

    def fail_test(*_args: object, **_kwargs: object) -> None:
        message = "report reran inference"
        raise AssertionError(message)

    monkeypatch.setattr("statadic.api.moments.gel_test", fail_test)
    report = inference.report()
    (row,) = report.tests
    assert row.name == "gel_lr"
    assert row.available
    assert row.statistic is not None
    assert (row.pvalue, row.law) == (None, None)
    assert report.metadata["degrees_of_freedom"] == inference.core.dof
    assert report.metadata["pvalue_unavailable_reason"] == "iid-reference-not-requested"
    assert not any(metric.name == "pvalue" for metric in report.metrics)
    monkeypatch.undo()
    tested = fit.infer(calibration=GELCalibration(assume_iid=True)).test(
        Hypothesis(OveridentifyingRestrictions(2, 0))
    )
    assert tested.evidence is not None
    calibrated = GELInferenceResult(tested.evidence, fit.context)
    (calibrated_row,) = calibrated.report().tests
    assert calibrated_row.pvalue == pytest.approx(calibrated.core.pvalue)
    assert calibrated_row.law == "chi-square"
    assert calibrated_row.law_parameters == (calibrated.core.dof,)
    assert type(calibrated_row.law_parameters[0]) is int
    zero = GELInferenceResult(
        calibrated.core._replace(dof=0, pvalue=float("nan")), fit.context
    )
    zero_report = zero.report()
    assert zero_report.metadata["degrees_of_freedom"] == 0
    assert zero_report.tests[0].pvalue is None
    assert (
        zero_report.metadata["pvalue_unavailable_reason"]
        == "nonpositive-degrees-of-freedom"
    )
    failed = GELInferenceResult(
        calibrated.core._replace(
            converged=False, stat=float("nan"), pvalue=float("nan")
        ),
        fit.context,
    )
    (failed_row,) = failed.report().tests
    assert not failed_row.available
    assert failed_row.unavailable_reason == "nonconverged"
    budget = 12_000
    assert len(failed._repr_html_()) < budget
