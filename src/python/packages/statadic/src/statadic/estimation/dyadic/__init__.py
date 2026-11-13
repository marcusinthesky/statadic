"""Fit undirected dyadic regressions and retain their point-estimation state.

Use DyadicEstimator.estimate with a model.dyadic.DyadicModel for coefficients,
rank-cleaned columns, fitted/residual arrays, FWL checks and model-fit diagnostics.
The eager design builder and solve door own data-dependent rank selection and require
caller-enabled x64; they are not transformable rank-selection algorithms.

Node-count schedules, bootstrap uncertainty and the combined fit-and-bootstrap
coordinator belong to statadic.inference.dyadic.
"""

from statadic.estimation.dyadic.contracts import DyadicComputationError, DyadicFit
from statadic.estimation.dyadic.fit import DyadicEstimator, build_dyadic_design

__all__ = [
    "DyadicComputationError",
    "DyadicEstimator",
    "DyadicFit",
    "build_dyadic_design",
]
