"""Positive and negative evidence for the supported structural host capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.contract,
    pytest.mark.typing,
    pytest.mark.subprocess,
    pytest.mark.requires_program("pyrefly"),
]


def test_structural_capabilities_preserve_real_family_signatures(
    tmp_path: Path,
) -> None:
    """Compatible methods satisfy protocols; wrong diagnostic signatures fail."""
    check_types(
        """
from statadic.api.capabilities import Reports, ResamplingInference
from statadic.api.diagnostics import LinearDiagnostic, RetainedLinearFit
from statadic.api.regression import (
    GeometricResult, GeometricInference, FreedmanLane, ResponsePermutation,
)
from statadic.api.reporting import Report
from collections.abc import Hashable
from statadic.api.dyadic import DyadicResult, NodeBootstrapResult
from statadic.api.hypotheses import Hypothesis, LinearRestriction, TestOutcome
from statadic.inference.dyadic import DyadicInference
from statadic.api.capabilities import Tests
from statadic.api.spatial.graph_regression import GraphRegressionResult
from statadic.api.spatial.resampling import GraphBootstrap, TimeBlockPairs

def report(value: GeometricResult) -> Reports:
    return value

def resamples(value: GeometricResult) -> ResamplingInference[
    ResponsePermutation | FreedmanLane, GeometricInference
]:
    return value

def dyadic_resamples(value: DyadicResult) -> ResamplingInference[
    DyadicInference[Hashable], NodeBootstrapResult
]:
    return value

def dyadic_tests(value: NodeBootstrapResult) -> Tests[
    Hypothesis[LinearRestriction], TestOutcome[LinearRestriction, NodeBootstrapResult]
]:
    return value

def graph_resamples(value: GraphRegressionResult) -> ResamplingInference[
    TimeBlockPairs, GraphBootstrap
]:
    return value

class IncorrectDiagnostic:
    def evaluate_linear(self, fitted: int) -> bool:
        return True

def needs_diagnostic(value: LinearDiagnostic) -> None: ...
needs_diagnostic(IncorrectDiagnostic())  # E: is not assignable
""",
        "capability_boundaries.py",
        tmp_path,
    )


def test_final_model_entry_cannot_be_overridden(tmp_path: Path) -> None:
    """The static checker rejects bypassing the shared preparation lifecycle."""
    check_types(
        """
from statadic.api.regression import Regression
from statadic.api.regression._model import RegressionFit, _Data

class InvalidRegression(Regression):
    def estimate(self, data: _Data, /) -> RegressionFit:  # E: final
        return super().estimate(data)
""",
        "final_lifecycle.py",
        tmp_path,
    )


def test_every_reporting_family_result_reports_fit_statistics(tmp_path: Path) -> None:
    """Each family result satisfies the fit-statistics protocol; others do not."""
    check_types(
        """
from statadic.api.capabilities import ReportsFitStatistics, SummarizesLikelihood
from statadic.api.regression import HurdleResult, LikelihoodResult, LinearResult
from statadic.api.spatial.cross_section import SpatialRegressionResult
from statadic.api.spatial.graph_regression import GraphRegressionResult
from statadic.api.spatial.observed import ObservedSpatialPanelResult
from statadic.api.spatial.panel import SpatialPanelResult
from statadic.api.survival import SpatialSurvivalResult, SurvivalResult

def linear(value: LinearResult) -> ReportsFitStatistics:
    return value

def likelihood(value: LikelihoodResult) -> ReportsFitStatistics:
    return value

def hurdle(value: HurdleResult) -> ReportsFitStatistics:
    return value

def spatial(value: SpatialRegressionResult) -> ReportsFitStatistics:
    return value

def panel(value: SpatialPanelResult) -> ReportsFitStatistics:
    return value

def observed(value: ObservedSpatialPanelResult) -> ReportsFitStatistics:
    return value

def graph(value: GraphRegressionResult) -> ReportsFitStatistics:
    return value

def survival(value: SurvivalResult) -> ReportsFitStatistics:
    return value

def spatial_survival(value: SpatialSurvivalResult) -> ReportsFitStatistics:
    return value

class NoStatistics:
    def fit_statistics(self, extra: int) -> int:
        return extra

def needs_statistics(value: ReportsFitStatistics) -> None: ...
needs_statistics(NoStatistics())  # E: is not assignable

class WrongSummary:
    def likelihood_summary(self) -> int:
        return 0

def needs_summary(value: SummarizesLikelihood) -> None: ...
needs_summary(WrongSummary())  # E: is not assignable
""",
        "fit_statistics_capabilities.py",
        tmp_path,
    )
