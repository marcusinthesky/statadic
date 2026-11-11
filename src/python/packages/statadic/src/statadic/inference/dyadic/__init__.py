"""Multinomial node-bootstrap uncertainty for undirected dyadic regression.

Use NodeCountSchedule for caller-owned entity multiplicities and bootstrap_dyadic
for an already prepared design. DyadicInference controls the schedule and
coefficient requirements. node_bootstrap_report turns one DyadicEstimator fit
into its complete DyadicFitResult; statadic.api.dyadic.DyadicRegression is the
class-based workflow over the same composition.

Model and symmetric node-effect options belong to statadic.model.dyadic; the actual
DyadicFit and the DyadicEstimator brick belong to statadic.estimation.dyadic. Entity
counts induce dyad weights through endpoint-count products, not row resampling.
"""

from statadic.inference.dyadic.bootstrap import (
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
    draw_node_counts,
    node_bootstrap_coefficient,
)
from statadic.inference.dyadic.contracts import (
    BootstrapCoefficientSummary,
    DyadicBootstrapDesign,
    DyadicBootstrapResult,
    DyadicFitRequest,
    DyadicFitResult,
    DyadicInference,
    NodeCountSchedule,
)
from statadic.inference.dyadic.report import node_bootstrap_report

__all__ = [
    "BootstrapCoefficientSummary",
    "DyadicBootstrapDesign",
    "DyadicBootstrapResult",
    "DyadicFitRequest",
    "DyadicFitResult",
    "DyadicInference",
    "NodeCountSchedule",
    "bootstrap_coefficient_inference",
    "bootstrap_dyadic",
    "draw_node_counts",
    "node_bootstrap_coefficient",
    "node_bootstrap_report",
]
