"""Structural capabilities of supported host workflows.

These protocols describe operations a result actually supports. They do not add
empty methods, choose scientific options or require third-party inheritance.
"""

from typing import Protocol

from statadic.api.fit_statistics import (  # noqa: TC001 -- public protocol annotation
    FitStatistics,
    LikelihoodSummary,
)
from statadic.api.reporting import Report  # noqa: TC001 -- public protocol annotation

__all__ = [
    "CovarianceInference",
    "Diagnoses",
    "Predicts",
    "Reports",
    "ReportsFitStatistics",
    "ResamplingInference",
    "SummarizesLikelihood",
    "Tests",
    "Transforms",
]


class Reports(Protocol):
    """An object whose retained evidence has a structured report."""

    def report(self) -> Report:
        """Describe retained evidence without recomputing an analysis."""
        ...


class Predicts[InputT, OutputT](Protocol):
    """A fitted result supporting prediction for its declared input family."""

    def predict(self, data: InputT, /) -> OutputT:
        """Predict from retained preparation."""
        ...


class Transforms[InputT, OutputT](Protocol):
    """A learned representation supporting frozen out-of-sample application."""

    def transform(self, data: InputT, /) -> OutputT:
        """Apply the retained representation."""
        ...


class Diagnoses[RequestT, OutputT](Protocol):
    """A result accepting explicit requests supported by its native family."""

    def diagnose(self, *requests: RequestT) -> OutputT:
        """Evaluate requests against the actual retained fit."""
        ...


class CovarianceInference[RequestT, OutputT](Protocol):
    """A result attaching explicitly requested sampling covariance."""

    def infer(self, *, covariance: RequestT) -> OutputT:
        """Attach covariance without choosing a hypothesis or reference law."""
        ...


class ResamplingInference[RequestT, OutputT](Protocol):
    """A result binding a supported explicit resampling design."""

    def infer(self, *, resampling: RequestT) -> OutputT:
        """Bind the schedule; the inference object receives the hypothesis."""
        ...


class Tests[HypothesisT, OutputT](Protocol):
    """A bound procedure accepting its supported hypothesis type."""

    def test(self, hypothesis: HypothesisT, /) -> OutputT:
        """Evaluate the stated hypothesis using the bound procedure."""
        ...


class ReportsFitStatistics(Protocol):
    """A result reporting fit-quality metrics and their convention notes."""

    def fit_statistics(self) -> FitStatistics:
        """Return fit statistics without refitting."""
        ...


class SummarizesLikelihood(Protocol):
    """A result exposing its maximised likelihood summary, or ``None``."""

    def likelihood_summary(self) -> LikelihoodSummary | None:
        """Return the typed likelihood summary."""
        ...
