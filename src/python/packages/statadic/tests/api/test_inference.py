"""Host reporting keeps numerical qualification separate from computation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy import stats
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear, RowDifference
from statadic.api.hypotheses import (
    ChiSquare,
    ConstantVariance,
    FisherF,
    Hypothesis,
    LinearRestriction,
)
from statadic.api.inference import HC, ModelBased
from statadic.api.regression import Regression
from statadic.core.results import InferenceStatus
from statadic.inference.parameters import linear_restriction_test

from .._budgets import examples
from .._typing import ill_typed
from .._validation.linear import wald_oracle

if TYPE_CHECKING:
    from jax import Array
    from statadic.api.inference import InferenceResult


def _identity(rows: tuple[int | str, ...] | list[int | str]) -> dict[str, object]:
    """Independently re-hash ordered row ids as reports promise."""
    payload = json.dumps(list(rows), separators=(",", ":"), ensure_ascii=False)
    return {"count": len(rows), "sha256": hashlib.sha256(payload.encode()).hexdigest()}


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(("degrees", "outside"), [(6.0, False), (100000.0, True)])
def test_report_preserves_computed_intervals_and_flags_reference_domain(
    degrees: float,
    *,
    outside: bool,
) -> None:
    """A successful numerical solve does not certify an untested reference law."""
    x = jnp.linspace(-2.0, 2.0, 8, dtype=jnp.float32)
    y = (
        1.2
        + 0.7 * x
        + jnp.array([0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.1, 0.1], dtype=x.dtype)
    )
    data = Observations({"x": x, "y": y})
    inference = (
        Regression("y", Intercept() + Linear("x"))
        .estimate(data)
        .infer(covariance=ModelBased())
    )

    intervals = inference.intervals(reference="t", degrees_of_freedom=degrees)
    report = inference.report(reference="t", degrees_of_freedom=degrees)

    assert bool(intervals.valid)
    assert bool(intervals.finite_reference_outside_validated_range) == outside
    assert dict(report.metadata)["finite_reference_outside_validated_range"] == outside
    assert bool(report.warnings) == outside
    for index, row in enumerate(report.parameters):
        assert row.valid
        assert row.ci_lower == pytest.approx(float(intervals.estimate.ci[0][index]))
        assert row.ci_upper == pytest.approx(float(intervals.estimate.ci[1][index]))


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    ("confidence", "level"),
    [
        (0.95, "95%"),
        (0.955, "95.5%"),
        (0.999, "99.9%"),
        (0.9999999999999999, "99.99999999999999%"),
    ],
)
def test_report_and_plot_show_supplied_confidence(
    confidence: float, level: str
) -> None:
    """Visible interval interpretations retain the selected confidence level."""
    x = jnp.arange(8.0)
    y = 1.0 + x + jnp.array([0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.1, 0.1])
    inference = (
        Regression("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": y})
        .infer(covariance=ModelBased())
    )
    report = inference.report(reference="normal", confidence=confidence)
    assert report.notes[0].startswith(f"{level} intervals;")
    meaning = report.coefficient_plot().interval_meaning
    assert meaning is not None
    assert meaning.startswith(f"{level} intervals;")


@pytest.mark.unit
@pytest.mark.reference
def test_extreme_contrast_confidence_retains_finite_bounds() -> None:
    """Contrast interval inversion shares the stable native normal reference."""
    x = jnp.arange(20.0)
    result = Regression("y", Linear("x")).estimate({"x": x, "y": 0.8 * x + jnp.sin(x)})
    confidence = float(jnp.nextafter(jnp.asarray(1.0), jnp.asarray(0.0)))
    actual = result.infer(covariance=HC("HC1")).contrast(
        RowDifference(at={"x": 1.0}, versus={"x": 0.0}),
        reference="normal",
        confidence=confidence,
    )
    assert actual.valid
    assert actual.estimate is not None
    assert actual.standard_error is not None
    assert actual.ci_lower is not None
    assert actual.ci_upper is not None
    expected_width = stats.norm.isf((1 - confidence) / 2) * actual.standard_error
    np.testing.assert_allclose(
        [actual.ci_lower, actual.ci_upper],
        [actual.estimate - expected_width, actual.estimate + expected_width],
        rtol=2e-6,
    )


@pytest.mark.component
@pytest.mark.contract
def test_contrast_report_preserves_point_when_covariance_is_unavailable() -> None:
    """A covariance failure does not erase an independently finite point estimate."""
    x = jnp.arange(10.0)
    fit = Regression.ols("y", Linear("x")).estimate({"x": x, "y": x + jnp.sin(x)})
    result = fit.infer(covariance=ModelBased(finite_sample_correction=0.0)).contrast(
        RowDifference(at={"x": 1.0}, versus={"x": 0.0}),
        reference="normal",
        confidence=0.9,
    )
    assert not result.valid
    metrics = {item.name: item for item in result.report().metrics}
    assert metrics["estimate"].available
    assert metrics["estimate"].value == pytest.approx(float(fit.parameters[0]))
    assert not metrics["standard_error"].available
    assert not metrics["ci_lower"].available
    assert result.report().metadata["confidence"] == pytest.approx(0.9)
    assert result.report().metadata["reference"] == "normal"


@pytest.mark.component
@pytest.mark.contract
def test_contrast_report_preserves_standard_error_when_interval_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interval inversion failure preserves the finite point and uncertainty."""
    x = jnp.arange(10.0)
    fit = Regression.ols("y", Linear("x")).estimate({"x": x, "y": x + jnp.sin(x)})

    def unavailable_critical_value(_: object) -> Array:
        return jnp.asarray(jnp.inf)

    monkeypatch.setattr(
        "statadic.api.inference.normal_critical_value", unavailable_critical_value
    )
    result = fit.infer(covariance=ModelBased()).contrast(
        RowDifference(at={"x": 1.0}, versus={"x": 0.0}), reference="normal"
    )
    metrics = {item.name: item for item in result.report().metrics}
    assert not result.valid
    assert metrics["estimate"].available
    assert metrics["standard_error"].available
    assert not metrics["ci_lower"].available
    assert not metrics["ci_upper"].available


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.parametrize("reference", ["chi2", "f"])
def test_named_hypothesis_matches_native_wald_and_independent_tail(
    reference: Literal["chi2", "f"],
) -> None:
    """Name resolution changes no restriction, covariance or tail calculation."""
    x = jnp.arange(12.0)
    data = {"x": x, "y": 0.8 * x + jnp.sin(x)}
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate(data)
        .infer(covariance=HC("HC3"))
    )
    hypothesis = Hypothesis(LinearRestriction(({"x": 2.0, "intercept": -1.0},), (0.3,)))
    degrees = 10.0 if reference == "f" else None
    actual = inference.test(hypothesis, reference=reference, degrees_of_freedom=degrees)
    native = linear_restriction_test(
        inference.uncertainty,
        jnp.asarray([[-1.0, 2.0]]),
        jnp.asarray([0.3]),
        parameter_names=inference.uncertainty.fit.parameter_names,
        reference=reference,
        degrees_of_freedom=degrees,
    )
    assert actual.available
    assert actual.statistic is not None
    assert actual.pvalue is not None
    assert actual.hypothesis is hypothesis
    assert actual.statistic == pytest.approx(float(native.test.statistic))
    assert actual.pvalue == pytest.approx(float(native.test.pvalue))
    expected = (
        stats.f.sf(actual.statistic, 1, 10)
        if reference == "f"
        else stats.chi2.sf(actual.statistic, 1)
    )
    assert actual.pvalue == pytest.approx(float(expected))
    assert actual.reference == (FisherF(1, 10) if reference == "f" else ChiSquare(1))
    assert actual.construction["rows"] == _identity(tuple(range(12)))
    assert actual.details["restriction_rank"] == pytest.approx(1.0)


@pytest.mark.integration
@pytest.mark.contract
def test_bound_hypothesis_rejects_foreign_design_and_preserves_portable_names() -> None:
    """Equal coefficient labels do not make independently fitted bases identical."""
    x = jnp.arange(10.0)
    data = {"x": x, "y": x + jnp.sin(x)}
    first = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    second = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    contrast = RowDifference(at={"x": 1.0}, versus={"x": 0.0}).compile(first.design)
    bound = Hypothesis(LinearRestriction.from_contrasts(contrast))
    assert first.infer(covariance=ModelBased()).test(bound, reference="chi2").available
    with pytest.raises(ValueError, match="different fitted design"):
        second.infer(covariance=ModelBased()).test(bound, reference="chi2")
    assert (
        second.infer(covariance=ModelBased())
        .test(Hypothesis(LinearRestriction.zero("x")), reference="chi2")
        .available
    )


@pytest.mark.integration
@pytest.mark.contract
def test_host_hypothesis_rejects_wrong_proposition_and_alternative() -> None:
    """The shared wrapper does not make every null suitable for every procedure."""
    x = jnp.arange(10.0)
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": x + jnp.sin(x)})
        .infer(covariance=ModelBased())
    )
    with pytest.raises(TypeError):
        inference.test(
            ill_typed(Hypothesis(ConstantVariance(("x",)))), reference="chi2"
        )
    with pytest.raises(ValueError, match="complement"):
        inference.test(
            Hypothesis(
                LinearRestriction.zero("x"),
                alternative=LinearRestriction.zero("intercept"),
            ),
            reference="chi2",
        )


@pytest.mark.integration
@pytest.mark.contract
def test_redundant_restrictions_preserve_native_unavailable_status() -> None:
    """The host keeps rank failure rather than deleting a restriction row."""
    x = jnp.arange(12.0)
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": x + jnp.sin(x)})
        .infer(covariance=ModelBased())
    )
    outcome = inference.test(
        Hypothesis(LinearRestriction(({"x": 1.0}, {"x": 2.0}))), reference="chi2"
    )
    assert not outcome.available
    assert outcome.statistic is None
    assert outcome.pvalue is None
    assert outcome.unavailable_reason == InferenceStatus.INVALID_RESTRICTION.name
    assert outcome.details["restriction_rank"] == 1
    assert type(outcome.details["restriction_rank"]) is int
    (row,) = outcome.report().tests
    assert row.unavailable_reason == outcome.unavailable_reason


@pytest.mark.integration
@pytest.mark.contract
def test_wald_report_preserves_reference_domain_warning() -> None:
    """Computed tails beyond the tested finite-reference range stay visibly bounded."""
    x = jnp.arange(10.0, dtype=jnp.float32)
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": x + jnp.sin(x)})
        .infer(covariance=ModelBased())
    )
    outcome = inference.test(
        Hypothesis(LinearRestriction.zero("x")),
        reference="f",
        degrees_of_freedom=100000.0,
    )
    assert outcome.available
    assert outcome.construction["finite_reference_outside_validated_range"] is True
    assert outcome.report().warnings


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("correction", [float("nan"), float("inf")])
def test_unavailable_correction_is_reported_without_nonfinite_metadata(
    correction: float,
) -> None:
    """An invalid covariance remains unavailable through both reportable paths."""
    x = jnp.arange(10.0)
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": x + jnp.sin(x)})
        .infer(covariance=ModelBased(finite_sample_correction=correction))
    )
    outcome = inference.test(Hypothesis(LinearRestriction.zero("x")), reference="chi2")
    assert not outcome.available
    assert outcome.statistic is None
    assert outcome.unavailable_reason == InferenceStatus.INVALID_COVARIANCE.name
    for metadata in (outcome.construction, inference.report().metadata):
        options = metadata["covariance_options"]
        assert isinstance(options, Mapping)
        assert options["finite_sample_correction"] is None
        assert options["finite_sample_correction_reason"] == "nonfinite-option"


def _ols_inference(
    covariance: ModelBased | HC | None = None,
) -> InferenceResult:
    """Fit a small OLS design whose coefficient tests are not extreme."""
    x = jnp.arange(12.0)
    data = {"x": x, "y": 0.1 * x + jnp.sin(3.0 * x)}
    return (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate(data)
        .infer(covariance=covariance or ModelBased())
    )


@pytest.mark.integration
@pytest.mark.contract
def test_report_without_null_has_no_coefficient_tests() -> None:
    """Nothing is tested unless the caller states a null."""
    report = _ols_inference().report(reference="normal")
    assert all(
        row.statistic is None and row.pvalue is None for row in report.parameters
    )
    assert "null" not in report.metadata
    assert not any("Two-sided" in note for note in report.notes)


@pytest.mark.integration
@pytest.mark.contract
def test_report_null_requires_reference_and_valid_values() -> None:
    """A null needs a reference, and must be a finite number naming real rows."""
    inference = _ols_inference()
    with pytest.raises(ValueError, match="reference"):
        inference.report(null=0.0)
    flag = bool(1)
    for bad in (float("nan"), float("inf"), "0", flag):
        with pytest.raises((TypeError, ValueError)):
            inference.report(reference="normal", null=ill_typed(bad))
    with pytest.raises(ValueError, match="unknown"):
        inference.report(reference="normal", null={"missing": 0.0})
    with pytest.raises(TypeError):
        inference.report(reference="normal", null={"x": ill_typed(flag)})
    with pytest.raises(ValueError, match="finite"):
        inference.report(reference="normal", null={"x": float("nan")})
    with pytest.raises(ValueError, match="at least one"):
        inference.report(reference="normal", null={})


@pytest.mark.integration
@pytest.mark.contract
def test_report_float_null_tests_every_row_and_mapping_only_named_rows() -> None:
    """A float covers all rows; a mapping covers only what it names."""
    inference = _ols_inference()
    every = inference.report(reference="normal", null=0.0)
    assert all(
        row.statistic is not None and row.pvalue is not None for row in every.parameters
    )
    assert every.metadata["null"] == pytest.approx(0.0)
    assert "Two-sided tests against 0; normal reference." in every.notes
    some = inference.report(reference="t", degrees_of_freedom=10.0, null={"x": 1.0})
    by_name = {row.name: row for row in some.parameters}
    assert by_name["intercept"].statistic is None
    assert by_name["intercept"].pvalue is None
    assert by_name["x"].statistic is not None
    assert by_name["x"].pvalue is not None
    assert dict(some.metadata["null"]) == {"x": 1.0}  # type: ignore[call-overload]
    assert "Two-sided tests of x = 1; t reference." in some.notes


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("kind", ["normal", "t"])
@pytest.mark.parametrize("null", [0.0, 0.25])
def test_report_pvalue_equals_single_restriction_test(
    kind: Literal["normal", "t"], null: float
) -> None:
    """The report row is the same Wald test that ``test`` performs."""
    inference = _ols_inference(HC("HC3"))
    degrees = 10.0 if kind == "t" else None
    report = inference.report(reference=kind, degrees_of_freedom=degrees, null=null)
    for row in report.parameters:
        outcome = inference.test(
            Hypothesis(LinearRestriction(({row.name: 1.0},), null)),
            reference="f" if kind == "t" else "chi2",
            degrees_of_freedom=degrees,
        )
        assert outcome.available
        assert row.pvalue == outcome.pvalue
        assert row.statistic is not None
        assert row.estimate is not None
        assert row.std_error is not None
        assert row.statistic == pytest.approx((row.estimate - null) / row.std_error)
        assert "effective_null" not in report.metadata
        assert not any("precision" in w for w in report.warnings)
        assert row.statistic**2 == pytest.approx(
            outcome.details["wald_statistic"], rel=1e-5
        )
    zero = inference.test(
        Hypothesis(LinearRestriction.zero("x")),
        reference="f" if kind == "t" else "chi2",
        degrees_of_freedom=degrees,
    )
    if not null:
        assert {r.name: r.pvalue for r in report.parameters}["x"] == zero.pvalue


def _float32_inference(scale: float = 1.0) -> InferenceResult:
    x = jnp.arange(12.0, dtype=jnp.float32)
    y = (0.1 * x + jnp.sin(3.0 * x)).astype(jnp.float32) * jnp.float32(scale)
    return (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": y})
        .infer(covariance=ModelBased())
    )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("kind", ["normal", "t"])
@pytest.mark.parametrize("offset", [3e-8, -3e-8, 0.0003, -0.0003])
def test_report_float32_statistic_and_pvalue_share_canonical_null(
    kind: Literal["normal", "t"], offset: float
) -> None:
    """Statistic and p-value come from the native test at the fit-dtype null."""
    inference = _float32_inference()
    assert inference.uncertainty.fit.parameters.dtype == jnp.float32
    names = inference.uncertainty.fit.parameter_names
    assert names is not None
    estimates = {
        n: float(v)
        for n, v in zip(
            names,
            inference.uncertainty.fit.parameters,
            strict=True,
        )
    }
    degrees = 10.0 if kind == "t" else None
    for name, estimate in estimates.items():
        null = estimate + offset
        effective = float(np.float32(null))
        report = inference.report(
            reference=kind, degrees_of_freedom=degrees, null={name: null}
        )
        row = {r.name: r for r in report.parameters}[name]
        outcome = inference.test(
            Hypothesis(LinearRestriction(({name: 1.0},), null)),
            reference="f" if kind == "t" else "chi2",
            degrees_of_freedom=degrees,
        )
        assert row.statistic is not None
        assert row.pvalue == outcome.pvalue
        assert row.statistic**2 == pytest.approx(
            outcome.details["wald_statistic"], rel=1e-5
        )
        if effective == estimate:
            assert row.statistic == pytest.approx(0.0)
        else:
            assert (row.statistic > 0) == (estimate > effective)
        if kind == "normal":
            expected = 2.0 * float(stats.norm.sf(abs(row.statistic)))
            assert row.pvalue == pytest.approx(expected, rel=1e-3, abs=1e-6)
        if effective != null:
            assert dict(report.metadata["effective_null"]) == {name: effective}  # type: ignore[call-overload]
            assert any(
                w
                == (
                    f"Null {null!r} for {name} is {effective!r} in the fit's "
                    f"float32 precision; the test uses {effective!r}."
                )
                for w in report.warnings
            )
        else:
            assert "effective_null" not in report.metadata


@pytest.mark.integration
@pytest.mark.contract
def test_report_float32_float_null_reports_float_effective_null() -> None:
    """A float null yields a float effective null and one warning per row."""
    requested = 0.1
    report = _float32_inference().report(reference="normal", null=requested)
    assert report.metadata["null"] == pytest.approx(requested)
    assert report.metadata["effective_null"] == pytest.approx(float(np.float32(0.1)))
    assert sum("float32 precision" in w for w in report.warnings) == len(
        report.parameters
    )


@pytest.mark.integration
@pytest.mark.contract
def test_report_float64_and_exact_nulls_have_no_effective_null() -> None:
    """Float64 keeps the null; an exactly representable float32 null is unchanged."""
    assert jax.config.x64_enabled
    report = _ols_inference().report(reference="normal", null=0.1)
    assert report.parameters[0].statistic is not None
    assert "effective_null" not in report.metadata
    assert not any("precision" in w for w in report.warnings)
    exact = _float32_inference().report(reference="normal", null=0.25)
    assert "effective_null" not in exact.metadata
    assert not any("precision" in w for w in exact.warnings)


@pytest.mark.integration
@pytest.mark.contract
def test_report_float32_small_variance_keeps_statistic_consistent() -> None:
    """A small positive variance still gives statistic**2 equal to the Wald value."""
    inference = _float32_inference(scale=3e-2)
    report = inference.report(reference="normal", null=0.1)
    for row in report.parameters:
        outcome = inference.test(
            Hypothesis(LinearRestriction(({row.name: 1.0},), 0.1)),
            reference="chi2",
        )
        assert row.statistic is not None
        assert row.pvalue == outcome.pvalue
        assert row.statistic**2 == pytest.approx(
            outcome.details["wald_statistic"], rel=1e-5
        )


@pytest.mark.integration
@pytest.mark.contract
def test_report_unavailable_standard_error_leaves_test_empty_and_warns() -> None:
    """No statistic or p-value is fabricated from an unavailable covariance."""
    x = jnp.arange(10.0)
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": x + jnp.sin(x)})
        .infer(covariance=ModelBased(finite_sample_correction=float("nan")))
    )
    report = inference.report(reference="normal", null=0.0)
    for row in report.parameters:
        assert row.statistic is None
        assert row.pvalue is None
        assert any(
            f"Coefficient test unavailable for {row.name}" in warning
            and InferenceStatus.INVALID_COVARIANCE.name in warning
            for warning in report.warnings
        )


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.contract
@examples(20)
@example(null=0.1, dtype="float32")
@given(
    null=st.floats(-2, 2, allow_subnormal=False),
    dtype=st.sampled_from(
        ["float32", "float64"] if jax.config.x64_enabled else ["float32"]
    ),
)
def test_arbitrary_coefficient_null_has_one_statistic_and_tail(
    null: float, dtype: str
) -> None:
    """Report rows and named Wald tests use the same represented null."""
    x = jnp.arange(12, dtype=dtype)
    inference = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate({"x": x, "y": 0.1 * x + jnp.sin(3 * x)})
        .infer(covariance=ModelBased())
    )
    report = inference.report(reference="normal", null={"x": null})
    row = next(row for row in report.parameters if row.name == "x")
    outcome = inference.test(
        Hypothesis(LinearRestriction(({"x": 1.0},), null)), reference="chi2"
    )
    assert outcome.available
    assert row.statistic is not None
    assert row.pvalue == outcome.pvalue
    assert row.statistic**2 == pytest.approx(
        outcome.details["wald_statistic"], rel=2e-5, abs=1e-7
    )
    effective = float(np.asarray(null, dtype=dtype))
    assert ("effective_null" in report.metadata) == (effective != null)
    assert any("precision" in warning for warning in report.warnings) == (
        effective != null
    )
    if effective != null:
        assert report.metadata["effective_null"] == {"x": effective}


@pytest.mark.integration
@pytest.mark.property
@examples(15)
@given(
    a=st.integers(-2, 2),
    b=st.sampled_from([-2, -1, 1, 2]),
    q=st.tuples(st.integers(-3, 3), st.integers(-3, 3)),
)
def test_wald_restrictions_match_oracle_and_reparameterization(
    a: int, b: int, q: tuple[int, int]
) -> None:
    """Full-rank restrictions agree under row scaling, order and term order."""
    inference = _float32_inference()
    restriction = np.asarray([[1, a], [0, b]], dtype=float)
    null = np.asarray(q, dtype=float) / 4
    beta = np.asarray(inference.uncertainty.fit.parameters, dtype=float)
    covariance = np.asarray(inference.uncertainty.covariance.values, dtype=float)
    expected = wald_oracle(restriction, null, beta, covariance)
    outcomes = []
    for matrix, target in (
        (restriction, null),
        (restriction[::-1] * np.asarray([[-2], [4]]), null[::-1] * [-2, 4]),
    ):
        rows = tuple({"x": float(row[1]), "intercept": float(row[0])} for row in matrix)
        actual = inference.test(
            Hypothesis(LinearRestriction(rows, tuple(map(float, target)))),
            reference="chi2",
        )
        assert actual.available
        assert actual.statistic == pytest.approx(expected, rel=3e-4, abs=2e-4)
        outcomes.append(actual)
    assert outcomes[0].pvalue == pytest.approx(outcomes[1].pvalue, rel=2e-4, abs=2e-6)
