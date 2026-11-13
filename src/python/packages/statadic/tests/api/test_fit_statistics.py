"""Fit statistics: the likelihood summary, information metrics and capabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import pytest
from hypothesis import given
from hypothesis import strategies as st
from statadic.api.base import Result
from statadic.api.fit_statistics import (
    FitStatistics,
    LikelihoodSummary,
    information_metrics,
)
from statadic.api.regression import (
    HurdleResult,
    LikelihoodResult,
    LinearResult,
)
from statadic.api.reporting import Report
from statadic.api.spatial.cross_section import SpatialRegressionResult
from statadic.api.spatial.graph_regression import GraphRegressionResult
from statadic.api.spatial.observed import ObservedSpatialPanelResult
from statadic.api.spatial.panel import SpatialPanelResult
from statadic.api.survival import SpatialSurvivalResult, SurvivalResult

from .._budgets import examples
from .._typing import ill_typed
from .._validation.linear import information_criteria

if TYPE_CHECKING:
    from statadic.api.capabilities import ReportsFitStatistics, SummarizesLikelihood
    from statadic.api.reporting import Metric

K_AIC = 3
_STATISTICS = ("n_parameters", "log_likelihood", "aic", "bic", "hqic")


def _by_name(metrics: tuple[Metric, ...]) -> dict[str, Metric]:
    return {metric.name: metric for metric in metrics}


@pytest.mark.unit
@pytest.mark.contract
def test_information_metrics_formulas_and_order() -> None:
    """The criteria follow -2 log L plus the stated penalties, in order."""
    metrics = information_metrics(-10.0, 3, 20, reason="r")
    assert tuple(m.name for m in metrics) == _STATISTICS
    values = {m.name: m.value for m in metrics}
    assert values["n_parameters"] == K_AIC
    assert isinstance(values["n_parameters"], int)
    assert values["aic"] == pytest.approx(20.0 + 6.0)
    assert values["bic"] == pytest.approx(20.0 + 3 * math.log(20))
    assert values["hqic"] == pytest.approx(20.0 + 6 * math.log(math.log(20)))
    assert all(m.definition == "" for m in metrics)


@pytest.mark.unit
@pytest.mark.contract
def test_information_metrics_unavailability() -> None:
    """Nonfinite likelihoods keep only the parameter count; HQIC needs n >= 3."""
    broken = _by_name(information_metrics(math.nan, 2, 10, reason="why"))
    assert broken["n_parameters"].available
    for name in _STATISTICS[1:]:
        assert not broken[name].available
        assert broken[name].unavailable_reason == "why"
    small = _by_name(information_metrics(-1.0, 2, 2, reason="r"))
    assert small["aic"].available
    assert small["bic"].available
    assert not small["hqic"].available


@pytest.mark.unit
@pytest.mark.contract
def test_default_fit_statistics_are_empty() -> None:
    """A result without a likelihood reports no fit statistics."""
    assert FitStatistics() == FitStatistics((), ())
    assert Result.fit_statistics.__qualname__ == "Result.fit_statistics"
    assert Result.likelihood_summary.__qualname__ == "Result.likelihood_summary"


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_parameters": -1}, "n_parameters"),
        ({"n_parameters": True}, "n_parameters"),
        ({"n_observations": 0}, "n_observations"),
        ({"n_observations": True}, "n_observations"),
        ({"unavailable_reason": ""}, "unavailable_reason"),
        ({"unavailable_reason": "  "}, "unavailable_reason"),
    ],
)
def test_likelihood_summary_rejects_malformed_fields(
    kwargs: dict[str, object], match: str
) -> None:
    """Counts must be non-bool integers in range and the reason nonempty."""
    fields: dict[str, object] = {
        "log_likelihood": -1.0,
        "n_parameters": 2,
        "n_observations": 5,
        "unavailable_reason": "why",
    }
    fields.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LikelihoodSummary(**ill_typed(fields))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("log_likelihood", [-10.0, math.nan])
def test_from_likelihood_matches_information_metrics_and_note(
    log_likelihood: float,
) -> None:
    """The summary yields the information metrics plus the information note."""
    summary = LikelihoodSummary(log_likelihood, 3, 20, "no likelihood")
    statistics = FitStatistics.from_likelihood(summary)
    expected = information_metrics(log_likelihood, 3, 20, reason="no likelihood")
    assert [m.name for m in statistics.metrics] == [m.name for m in expected]
    for got, want in zip(statistics.metrics, expected, strict=True):
        assert (got.value, got.available, got.unavailable_reason) == (
            want.value,
            want.available,
            want.unavailable_reason,
        )
    assert len(statistics.notes) == 1
    assert "Akaike 1974" in statistics.notes[0]


@dataclass(frozen=True)
class _Minimal(Result[None, None]):
    summary: LikelihoodSummary | None = None

    @override
    def report(self) -> Report:
        return Report(title="minimal")

    @override
    def likelihood_summary(self) -> LikelihoodSummary | None:
        return self.summary


@pytest.mark.unit
@pytest.mark.contract
def test_result_fit_statistics_derive_from_likelihood_summary() -> None:
    """The default derivation is empty without a summary, else from_likelihood."""
    assert _Minimal(None, None).fit_statistics() == FitStatistics()
    summary = LikelihoodSummary(-4.0, 2, 10, "r")
    result = _Minimal(None, None, summary)
    assert result.fit_statistics() == FitStatistics.from_likelihood(summary)


def _reports(value: ReportsFitStatistics) -> ReportsFitStatistics:
    return value


def _summarizes(value: SummarizesLikelihood) -> SummarizesLikelihood:
    return value


def _linear(value: LinearResult) -> ReportsFitStatistics:
    return value


def _likelihood(value: LikelihoodResult) -> ReportsFitStatistics:
    return value


def _hurdle(value: HurdleResult) -> ReportsFitStatistics:
    return value


def _spatial(value: SpatialRegressionResult) -> ReportsFitStatistics:
    return value


def _panel(value: SpatialPanelResult) -> ReportsFitStatistics:
    return value


def _observed(value: ObservedSpatialPanelResult) -> ReportsFitStatistics:
    return value


def _graph(value: GraphRegressionResult) -> ReportsFitStatistics:
    return value


def _survival(value: SurvivalResult) -> ReportsFitStatistics:
    return value


def _spatial_survival(value: SpatialSurvivalResult) -> ReportsFitStatistics:
    return value


def _minimal_reports(value: _Minimal) -> ReportsFitStatistics:
    return value


def _minimal_summarizes(value: _Minimal) -> SummarizesLikelihood:
    return value


@pytest.mark.unit
@pytest.mark.contract
def test_family_results_declare_fit_statistics_structurally() -> None:
    """Every family result has the method the static protocols require.

    Pyrefly checks the annotated parameters above; this asserts the runtime
    methods exist and that a minimal result satisfies both protocols.
    """
    for cls in (
        LinearResult,
        LikelihoodResult,
        HurdleResult,
        SpatialRegressionResult,
        SpatialPanelResult,
        ObservedSpatialPanelResult,
        GraphRegressionResult,
        SurvivalResult,
        SpatialSurvivalResult,
    ):
        assert callable(getattr(cls, "fit_statistics", None)), cls
    minimal = _Minimal(None, None)
    assert _reports(_minimal_reports(minimal)) is minimal
    assert _summarizes(_minimal_summarizes(minimal)) is minimal


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.contract
@examples(40)
@given(
    ll=st.floats(-1000, 1000, allow_subnormal=False),
    k=st.integers(0, 20),
    n=st.integers(16, 10000),
)
def test_information_criteria_match_independent_penalties(
    ll: float, k: int, n: int
) -> None:
    """All criteria use the same likelihood and declared count convention."""
    metrics = _by_name(information_metrics(ll, k, n, reason="unavailable"))
    for name, value in information_criteria(ll, k, n).items():
        assert metrics[name].available
        assert metrics[name].value == pytest.approx(value)
    bic, hqic, aic = (metrics[name].value for name in ("bic", "hqic", "aic"))
    assert bic is not None
    assert hqic is not None
    assert aic is not None
    assert bic >= hqic >= aic


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.contract
@examples(30)
@given(
    k=st.one_of(st.integers(-2, 10), st.booleans()),
    n=st.one_of(st.integers(-2, 10), st.booleans()),
    reason=st.sampled_from(["why", "", " "]),
)
def test_likelihood_summary_count_domain(k: int, n: int, reason: str) -> None:
    """Counts are integral quantities; booleans cannot stand for counts."""
    valid = (
        type(k) is int and k >= 0 and type(n) is int and n >= 1 and bool(reason.strip())
    )
    if valid:
        summary = LikelihoodSummary(-1.0, k, n, reason)
        assert summary.n_parameters == k
        assert summary.n_observations == n
    else:
        with pytest.raises(
            ValueError, match=r"n_parameters|n_observations|unavailable_reason"
        ):
            LikelihoodSummary(-1.0, k, n, reason)
