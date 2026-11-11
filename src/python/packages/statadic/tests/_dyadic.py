"""Test composition of a dyadic point fit with its multinomial node bootstrap.

The retired ``fit_dyadic_model`` door is ``DyadicEstimator().estimate`` followed by
``node_bootstrap_report``; suites that assert on the complete
``DyadicFitResult`` call this two-step composition.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,  # runtime hint
)
from typing import TYPE_CHECKING

from statadic.estimation.dyadic import DyadicEstimator
from statadic.inference.dyadic import node_bootstrap_report

if TYPE_CHECKING:
    from statadic.inference.dyadic.contracts import DyadicFitRequest, DyadicFitResult


def fit_dyadic_report[NodeIdT: Hashable](
    request: DyadicFitRequest[NodeIdT],
) -> DyadicFitResult:
    """Fit one dyadic specification and its multinomial node-bootstrap schedule."""
    report, _ = node_bootstrap_report(
        DyadicEstimator().estimate(request.model), request.inference
    )
    return report
