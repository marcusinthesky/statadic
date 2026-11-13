"""Host questions own their inputs and retain finite, serializable evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_analysis, save_analysis
from statadic.api.data import Observations
from statadic.api.design import FittedDesign, Intercept, Linear, RowDifference
from statadic.api.hypotheses import (
    ChiSquare,
    ConstantVariance,
    FisherF,
    GaussianMoments,
    Hypothesis,
    LinearRestriction,
    NoAddedFittedPowerEffect,
    Normal,
    NoSerialCorrelation,
    OveridentifyingRestrictions,
    ResampledReference,
    TestOutcome,
)
from statadic.api.reporting import freeze_metadata

from .._typing import ill_typed

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.api.reporting import MetadataValue


@dataclass(frozen=True)
class _InvalidProposition:
    payload: object
    text: object = "A test proposition."

    def describe(self) -> str:
        return ill_typed(self.text)

    def metadata(self) -> Mapping[str, MetadataValue]:
        return ill_typed(self.payload)


def _design() -> FittedDesign:
    rows = Observations({"x": jnp.asarray([-1.0, 0.0, 2.0])})
    return FittedDesign((Intercept() + Linear("x"))._prepare_term(rows))


@pytest.mark.unit
@pytest.mark.contract
def test_normal_and_resampled_references_preserve_actual_conventions() -> None:
    """Locations, tails and approximation labels survive finite report serialization."""
    normal = Normal(location=2, scale=3)
    assert normal.metadata() == {"kind": "normal", "location": 2.0, "scale": 3.0}
    reference = ResampledReference("exact", 8, correction="none", approximate=True)
    evidence = jnp.asarray([0.0, 1.0])
    outcome = TestOutcome(
        "example",
        Hypothesis(GaussianMoments()),
        2,
        0.25,
        reference,
        available=True,
        evidence=evidence,
    )
    assert outcome.evidence is evidence
    retained_reference = outcome.report().metadata["reference"]
    assert isinstance(retained_reference, Mapping)
    assert retained_reference["approximate"] is True
    assert "Array" not in repr(outcome)
    assert "evidence" not in outcome.report().metadata
    with pytest.raises(ValueError, match="positive"):
        Normal(scale=0)
    with pytest.raises(ValueError, match="uncorrected"):
        ResampledReference("exact", 8, correction="add-one")
    with pytest.raises(ValueError, match="Invalid"):
        ResampledReference("random", 0, correction="add-one")


@pytest.mark.unit
@pytest.mark.contract
def test_named_restrictions_own_rows_and_resolve_actual_parameter_order() -> None:
    """Caller mutation and an extra nuisance parameter cannot change the question."""
    row = {"x": 2.0, "intercept": -1.0}
    restriction = LinearRestriction((row,), (3.0,))
    hypothesis = Hypothesis(restriction)
    snapshot = hypothesis.metadata()
    row["x"] = 99.0
    matrix, target = restriction.compile(
        ("intercept", "x", "log_sigma"), jnp.dtype("float32")
    )
    np.testing.assert_array_equal(matrix, [[-1.0, 2.0, 0.0]])
    np.testing.assert_array_equal(target, [3.0])
    assert matrix.dtype == jnp.float32
    assert hypothesis.metadata() == snapshot
    assert "2*x" in hypothesis.describe()
    with pytest.raises(TypeError):
        ill_typed(restriction.rows[0])["x"] = 7.0
    assert freeze_metadata(hypothesis.metadata()) == snapshot


@pytest.mark.unit
@pytest.mark.contract
def test_natural_integer_restriction_inputs_are_normalized_without_accepting_bool() -> (
    None
):
    """Integer mathematical coefficients remain valid under the runtime hook."""
    restriction = LinearRestriction(({"x": 2, "intercept": -1},), 3)
    matrix, target = restriction.compile(("intercept", "x"), jnp.dtype("float32"))
    np.testing.assert_array_equal(matrix, [[-1.0, 2.0]])
    np.testing.assert_array_equal(target, [3.0])
    assert type(restriction.rows[0]["x"]) is float
    with pytest.raises(ValueError, match="finite real"):
        LinearRestriction(({"x": True},))


@pytest.mark.unit
@pytest.mark.contract
def test_natural_integer_outcome_and_reference_inputs_are_normalized() -> None:
    """Finite report quantities accept integers while retaining strict bool guards."""
    outcome = TestOutcome(
        "wald",
        Hypothesis(LinearRestriction.zero("x")),
        2,
        0,
        ChiSquare(1),
        available=True,
        details={"rank": 1},
    )
    assert type(outcome.statistic) is float
    assert type(outcome.pvalue) is float
    assert type(outcome.details["rank"]) is int
    assert type(replace(outcome, details={"rank": 1.0}).details["rank"]) is float
    assert type(ChiSquare(2).df) is float
    assert type(FisherF(1, 10).df_denominator) is float
    with pytest.raises(ValueError, match="finite real"):
        replace(outcome, statistic=True)


@pytest.mark.unit
@pytest.mark.contract
def test_hypotheses_reject_invalid_extension_metadata_and_descriptions() -> None:
    """Structural extension promises are checked before snapshotting a question."""
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        Hypothesis(_InvalidProposition([]))
    with pytest.raises(ValueError, match="description"):
        Hypothesis(_InvalidProposition({}, ""))
    with pytest.raises(ValueError, match=r"alternative.*description"):
        Hypothesis(GaussianMoments(), alternative=_InvalidProposition({}, ""))


@pytest.mark.unit
@pytest.mark.contract
def test_contrasts_bind_one_actual_design_and_zero_pad_ancillary_parameters() -> None:
    """A learned-basis contrast keeps identity and excludes unrelated coefficients."""
    design = _design()
    contrast = RowDifference(at={"x": 2.0}, versus={"x": 0.0}).compile(design)
    restriction = LinearRestriction.from_contrasts(contrast, values=1.0)
    assert restriction.design is design
    matrix, target = restriction.compile(
        ("intercept", "x", "shape"), jnp.dtype("float64")
    )
    np.testing.assert_array_equal(matrix, [[0.0, 2.0, 0.0]])
    np.testing.assert_array_equal(target, [1.0])
    other = RowDifference(at={"x": 2.0}, versus={"x": 0.0}).compile(_design())
    with pytest.raises(ValueError, match="one fitted design"):
        LinearRestriction.from_contrasts(contrast, other)
    with pytest.raises(ValueError, match="at least one"):
        LinearRestriction.from_contrasts()


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("rows", "values"),
    [
        ((), 0.0),
        (({},), 0.0),
        (({"": 1.0},), 0.0),
        (({"x": float("nan")},), 0.0),
        (({"x": float("inf")},), 0.0),
        (({"x": 1.0},), float("nan")),
        (({"x": 1.0},), (0.0, 1.0)),
    ],
)
def test_restrictions_reject_malformed_statements(
    rows: tuple[dict[str, float], ...], values: tuple[float, ...] | float
) -> None:
    """Shape and finite-value mistakes fail before calling numerical inference."""
    with pytest.raises(ValueError, match=r"requires|Restriction|restriction"):
        LinearRestriction(rows, values)


@pytest.mark.unit
@pytest.mark.contract
def test_named_rows_preserve_redundant_restrictions_for_native_rank_checks() -> None:
    """The host neither guesses names nor silently deletes mathematical restrictions."""
    with pytest.raises(ValueError, match="unique"):
        LinearRestriction.zero("x", "x")
    with pytest.raises(ValueError, match="Unknown"):
        LinearRestriction.zero("missing").compile(("x",), jnp.dtype("float32"))
    restriction = LinearRestriction(({"x": 1.0}, {"x": 2.0}))
    matrix, _ = restriction.compile(("x",), jnp.dtype("float32"))
    np.testing.assert_array_equal(matrix, [[1.0], [2.0]])


@pytest.mark.unit
@pytest.mark.contract
def test_concrete_propositions_record_only_their_actual_scientific_scope() -> None:
    """Moment compatibility, finite variance designs and fitted-power spans differ."""
    propositions = (
        ConstantVariance(("intercept", "x")),
        GaussianMoments(),
        NoSerialCorrelation(2, "time"),
        NoAddedFittedPowerEffect((2, 2, 3)),
        OveridentifyingRestrictions(4, 2),
    )
    for proposition in propositions:
        hypothesis = Hypothesis(proposition)
        assert hypothesis.describe()
        assert hypothesis.metadata()["alternative"] is None
        assert freeze_metadata(hypothesis.metadata()) == hypothesis.metadata()
    assert "positional" in NoSerialCorrelation(2, "time").describe()
    assert "retained row" in NoSerialCorrelation(1, None).describe()
    assert OveridentifyingRestrictions(2, 2).metadata()["moment_count"] == len(
        ("m1", "m2")
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("degrees", [0.0, -1.0, float("nan"), float("inf"), True])
def test_actual_references_require_positive_finite_degrees(degrees: float) -> None:
    """Unavailable requests do not masquerade as calibrated reference values."""
    with pytest.raises(ValueError, match="degrees of freedom"):
        ChiSquare(degrees)
    with pytest.raises(ValueError, match="degrees of freedom"):
        FisherF(1.0, degrees)


@pytest.mark.unit
@pytest.mark.contract
def test_test_outcomes_preserve_descriptive_and_unavailable_evidence() -> None:
    """A DW statistic needs no p-value; failed tests retain finite components."""
    hypothesis = Hypothesis(NoSerialCorrelation(1, None))
    descriptive = TestOutcome("dw", hypothesis, 1.8, None, None, available=True)
    row = descriptive.test_row()
    assert (row.statistic, row.pvalue, row.law) == (1.8, None, None)
    assert row.hypothesis == hypothesis.describe()
    assert descriptive.components() == ()
    unavailable = TestOutcome(
        "bg",
        hypothesis,
        None,
        None,
        None,
        available=False,
        unavailable_reason="auxiliary-regression-saturated",
        details={"rank": 3.0},
    )
    row = unavailable.test_row()
    assert not row.available
    assert row.unavailable_reason == "auxiliary-regression-saturated"
    (component,) = unavailable.components()
    assert component.name == "bg.rank"
    assert component.value == pytest.approx(3.0)
    assert str(ChiSquare(2)) == "chi2(2)"
    assert str(FisherF(2, 97)) == "F(2, 97)"


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "changes",
    [
        {"statistic": None},
        {"statistic": float("nan")},
        {"pvalue": -0.1},
        {"pvalue": 1.1},
        {"reference": None},
        {"pvalue": None},
        {"available": False},
        {"unavailable_reason": "bad"},
        {"details": {"statistic": 1.0}},
        {"details": {"pvalue": 0.2}},
        {"details": {"rank": float("nan")}},
        {"construction": {"nonfinite": float("inf")}},
        {"hypothesis": "not-a-hypothesis"},
        {"reference": "chi2(1)"},
        {"unavailable_reason": 42},
    ],
)
def test_test_outcomes_reject_contradictory_or_nonfinite_fields(
    changes: dict[str, object],
) -> None:
    """No report can acquire fabricated calibration or duplicate metric identities."""
    outcome = TestOutcome(
        "wald",
        Hypothesis(LinearRestriction.zero("x")),
        2.0,
        0.15,
        ChiSquare(1),
        available=True,
    )
    with pytest.raises((TypeError, ValueError)):
        ill_typed(replace)(outcome, **changes)


@pytest.mark.integration
@pytest.mark.contract
def test_outcome_report_snapshot_owns_nested_metadata_and_restores_rendering_only(
    tmp_path: Path,
) -> None:
    """Saving a question preserves values, never live inference or fitted identity."""
    rows: list[MetadataValue] = ["unit-a", "unit-b"]
    construction: dict[str, MetadataValue] = {
        "row_ids": rows,
        "ordering": {"order_by": "time"},
    }
    outcome = TestOutcome(
        "wald",
        Hypothesis(LinearRestriction.zero("x")),
        2.0,
        0.15,
        ChiSquare(1),
        available=True,
        construction=construction,
        assumptions=("Caller-selected reference.",),
        sources=("hansen-1982",),
    )
    rows.reverse()
    report = outcome.report()
    retained = report.metadata["construction"]
    assert isinstance(retained, Mapping)
    assert retained["row_ids"] == ("unit-a", "unit-b")
    path = tmp_path / "outcome"
    save_analysis(path, report)
    restored = load_analysis(path)
    assert restored.report.metadata == report.metadata
    assert restored.report.metrics == report.metrics
    assert restored.capabilities == ("report", "render")
    assert not hasattr(outcome, "kernel")
    assert outcome._repr_html_() == report._repr_html_()
