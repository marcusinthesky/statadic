"""Nested geometric regression contracts with explicit multivariate refit oracles."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from statadic.api.comparison import PermutationPlan
from statadic.api.data import Observations
from statadic.api.design import (  # noqa: TC002 -- patched method annotation
    Intercept,
    Linear,
    Spline,
    Term,
)
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.regression import (
    FreedmanLane,
    GeometricResponse,
    GeometricResult,
    Reduced,
    Regression,
    ResponsePermutation,
)
from statadic.api.representation import MMD
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.sample.packed import PackedClouds

from ..._typing import ill_typed

if TYPE_CHECKING:
    from statadic.api.regression._geometric import GeometricTestEvidence

pytestmark = pytest.mark.component


def _geometric_evidence(
    result: GeometricResult,
    calibration: ResponsePermutation | FreedmanLane,
    *,
    hypothesis: Hypothesis[Reduced] | None = None,
) -> GeometricTestEvidence:
    """Check the public test path before inspecting its complete numerical evidence."""
    question = (
        Hypothesis(
            Reduced(
                tuple(
                    result.design.column_names[index]
                    for index in result.design.intercept_columns
                )
            )
        )
        if hypothesis is None
        else hypothesis
    )
    outcome = result.infer(resampling=calibration).test(question)
    assert outcome.evidence is not None
    return outcome.evidence


@pytest.mark.contract
def test_binding_resampling_does_not_execute_hypothesis_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an explicit test evaluates the null; binding/reporting preserve the fit."""
    model, data, _ = _fixture()
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        message = "hypothesis work executed"
        raise AssertionError(message)

    monkeypatch.setattr(
        "statadic.api.regression._geometric._infer_geometric", forbidden
    )
    bound = result.infer(resampling=ResponsePermutation(PermutationPlan(n=3)))
    assert bound.observed is result
    assert bound.report().metadata["test_status"] == "not requested"
    with pytest.raises(AssertionError, match="hypothesis work"):
        bound.test(Hypothesis(Reduced(("intercept",))))


def _fixture() -> tuple[Regression, Observations, np.ndarray]:
    z = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5])
    x = np.asarray([0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, -2.0])
    response = np.column_stack((z + np.sin(x), x + np.cos(z)))
    row_ids = tuple(f"unit-{index}" for index in range(len(z)))
    geometry = MMD(LINEAR_KERNEL).prepare(
        PackedClouds.from_clouds([value[None, :] for value in response]),
        row_ids=row_ids,
    )
    return (
        Regression(
            response=GeometricResponse(geometry, distance_power=2),
            terms=Intercept() + Linear("z") + Linear("x"),
        ),
        Observations({"z": z, "x": x}, row_ids=row_ids),
        response,
    )


def _statistic(response: np.ndarray, full: np.ndarray, reduced: np.ndarray) -> float:
    full_error = response - full @ np.linalg.lstsq(full, response, rcond=None)[0]
    null_error = response - reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    within = np.sum(full_error**2)
    return float(
        ((np.sum(null_error**2) - within) / (full.shape[1] - reduced.shape[1]))
        / (within / (len(full) - full.shape[1]))
    )


@pytest.mark.reference
def test_freedman_lane_retains_exact_full_design_and_response_refit_evidence() -> None:
    """Every supplied schedule agrees with independently reconstructed responses."""
    model, data, response = _fixture()
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)
    full = np.column_stack(
        (
            np.ones(data.n_rows),
            np.asarray(data.numeric("z")),
            np.asarray(data.numeric("x")),
        )
    )
    reduced = full[:, :2]
    signal = reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    residuals = response - signal
    rows = np.asarray(
        [
            np.arange(data.n_rows),
            np.arange(data.n_rows)[::-1],
            np.roll(np.arange(data.n_rows), 3),
        ]
    )
    inferred = _geometric_evidence(
        result,
        FreedmanLane(PermutationPlan(n=len(rows), indices=rows, row_ids=data.row_ids)),
        hypothesis=Hypothesis(Reduced(("intercept", "z"))),
    )
    expected = [_statistic(signal + residuals[row], full, reduced) for row in rows]
    assert inferred.valid
    np.testing.assert_allclose(inferred.null, expected, rtol=3e-5)
    assert float(inferred.projection.statistic) == pytest.approx(
        _statistic(response, full, reduced), rel=3e-5
    )
    assert inferred.pvalue == pytest.approx(
        (1 + np.sum(np.asarray(expected) >= expected[0])) / (1 + len(expected))
    )
    np.testing.assert_array_equal(inferred.schedules, rows)
    assert inferred.arrangements is None
    assert inferred.hypothesis.null.columns == ("intercept", "z")
    assert result.design.column_names == ("intercept", "z", "x")
    assert result.context.values.shape[1] == full.shape[1]
    assert inferred.calibration_status == "approximate"
    assert inferred.calibrated is not None
    assert "approximate" in str(inferred.report())
    assert not hasattr(result, "parameters")
    assert not hasattr(result, "predict")


@pytest.mark.contract
def test_reduced_selects_retained_columns_without_refitting_learned_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spline is learned once with the full design and never learned by inference."""
    model, data, _ = _fixture()
    model = replace(model, terms=Intercept() + Linear("z") + Spline("x", df=3))
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)
    retained = result.design

    def forbid_prepare(self: Term, training: Observations) -> None:
        del self, training
        pytest.fail("Inference refitted a predictor term")

    monkeypatch.setattr(Spline, "_prepare_term", forbid_prepare)
    draws = 3
    inferred = _geometric_evidence(
        result,
        FreedmanLane(PermutationPlan(n=draws, seed=31)),
        hypothesis=Hypothesis(Reduced(("intercept", "z"))),
    )
    assert result.design is retained
    assert inferred.requested == draws
    for columns in (("z",), ("intercept", "missing"), retained.column_names):
        with pytest.raises(ValueError, match="proper subset"):
            _geometric_evidence(
                result, FreedmanLane(), hypothesis=Hypothesis(Reduced(columns))
            )
    with pytest.raises(ValueError, match="unique"):
        Reduced(("intercept", "intercept"))


@pytest.mark.contract
def test_raw_response_permutation_rejects_a_nuisance_adjustment_claim() -> None:
    """Nuisance-adjusted calibration must be an explicit reduced-residual request."""
    model, data, _ = _fixture()
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)
    with pytest.raises(ValueError, match="do not adjust nuisance"):
        _geometric_evidence(
            result,
            ResponsePermutation(),
            hypothesis=Hypothesis(Reduced(("intercept", "z"))),
        )


@pytest.mark.contract
def test_exhaustive_freedman_lane_schedules_remain_approximate() -> None:
    """Enumeration describes computational coverage, never exact nuisance validity."""
    model, data, _ = _fixture()
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)
    blocks = tuple(index // 2 for index in range(data.n_rows))
    inferred = _geometric_evidence(
        result,
        FreedmanLane(
            PermutationPlan(method="exact", blocks=blocks, row_ids=data.row_ids)
        ),
        hypothesis=Hypothesis(Reduced(("intercept", "z"))),
    )
    assert inferred.requested == 2 ** (data.n_rows // 2)
    assert inferred.calibration_status == "approximate"
    assert inferred.plan.method == "exact"
    assert inferred.valid
    assert inferred.pvalue == pytest.approx(
        float(
            np.mean(np.asarray(inferred.null) >= float(inferred.projection.statistic))
        )
    )
    assert inferred.report().metadata["pvalue_rule"] == "extreme / all"
    (row,) = inferred.report().tests
    assert row.name == "geometric_effect"
    assert row.pvalue == pytest.approx(inferred.pvalue)
    assert row.statistic == pytest.approx(float(inferred.projection.statistic))
    assert (row.law, row.law_parameters) == ("resampled", (inferred.requested,))
    assert "approximate" in row.reference
    assert "F" not in [metric.name for metric in inferred.report().metrics]
    assert inferred.key is None
    assert inferred.schedules is not None
    assert "approximate" in str(inferred.report())


@pytest.mark.contract
def test_geometric_regression_rejects_invalid_design_and_row_identity() -> None:
    """No inner join, rank-padding QR or inferred intercept changes the design."""
    model, data, _ = _fixture()
    with pytest.raises(ValueError, match="exactly match"):
        model.estimate(data.take(list(reversed(range(data.n_rows)))))
    duplicate = Observations(
        {"z": data.numeric("z"), "x": data.numeric("z")}, row_ids=data.row_ids
    )
    with pytest.raises(ValueError, match="full-rank"):
        model.estimate(duplicate)
    with pytest.raises(ValueError, match="Intercept"):
        replace(model, terms=Linear("x")).estimate(data)


@pytest.mark.contract
def test_observed_matrix_has_no_adjusted_geometry_or_inference_warrant() -> None:
    """One supplied matrix cannot turn a covariate adjustment into a qualified model."""
    model, data, response = _fixture()
    distances = np.linalg.norm(response[:, None] - response[None, :], axis=-1)
    observed = GeometricResponse.from_distances(
        distances, row_ids=data.row_ids, distance_power=2
    )
    with pytest.raises(ValueError, match="one-factor descriptive"):
        Regression(response=observed, terms=model.terms).estimate(data)


@pytest.mark.contract
def test_response_descriptor_rejects_implicit_functional_changes() -> None:
    """Only the two explicit raw/squared conventions enter this response contract."""
    model, _, _ = _fixture()
    assert isinstance(model.response, GeometricResponse)
    with pytest.raises(ValueError, match="integer 1 or 2"):
        GeometricResponse(model.response.source, distance_power=True)  # type: ignore[arg-type] -- rejected runtime input


@pytest.mark.contract
def test_geometric_hypothesis_wrapper_retains_default_and_explicit_question() -> None:
    """Default and explicit nulls share the wrapper without altering schedules."""
    model, data, _ = _fixture()
    result = model.estimate(data)
    assert isinstance(result, GeometricResult)
    plan = PermutationPlan(n=3, seed=31)
    default = _geometric_evidence(result, ResponsePermutation(plan))
    hypothesis = Hypothesis(Reduced(("intercept",)))
    explicit = _geometric_evidence(
        result, ResponsePermutation(plan), hypothesis=hypothesis
    )
    assert explicit.hypothesis is hypothesis
    assert default.hypothesis.null.columns == ("intercept",)
    np.testing.assert_array_equal(default.null, explicit.null)
    np.testing.assert_array_equal(default.schedules, explicit.schedules)
    assert explicit.report().metadata["hypothesis"] == hypothesis.metadata()
    with pytest.raises(TypeError):
        _geometric_evidence(
            result,
            ResponsePermutation(plan),
            hypothesis=ill_typed(Reduced(("intercept",))),
        )
    with pytest.raises(TypeError):
        _geometric_evidence(
            result,
            ResponsePermutation(plan),
            hypothesis=ill_typed(Hypothesis(LinearRestriction.zero("x"))),
        )
    with pytest.raises(ValueError, match="complement"):
        _geometric_evidence(
            result,
            ResponsePermutation(plan),
            hypothesis=Hypothesis(
                Reduced(("intercept",)), alternative=Reduced(("intercept", "z"))
            ),
        )
