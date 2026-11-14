"""Canonical regression specification and family-specific retained capabilities."""

from statadic.regression._conditional import (
    Binomial,
    Gamma,
    Gaussian,
    HurdlePoisson,
    HurdleResult,
    NegativeBinomial2,
    Observed,
    Poisson,
)
from statadic.regression._geometric import (
    FreedmanLane,
    GeometricInference,
    GeometricResponse,
    GeometricResult,
    GeometricTestEvidence,
    PreparedGeometric,
    Reduced,
    ResponsePermutation,
)
from statadic.regression._glm import LogisticResult, PoissonResult
from statadic.regression._iv import IVResult, PreparedIV
from statadic.regression._likelihood import LikelihoodResult
from statadic.regression._linear import LinearResult
from statadic.regression._model import (
    PreparedRegressionModel,
    Regression,
    RegressionContext,
    RegressionDescription,
    RegressionEstimator,
    RegressionFit,
)

__all__ = [
    "Binomial",
    "FreedmanLane",
    "Gamma",
    "Gaussian",
    "GeometricInference",
    "GeometricResponse",
    "GeometricResult",
    "GeometricTestEvidence",
    "HurdlePoisson",
    "HurdleResult",
    "IVResult",
    "LikelihoodResult",
    "LinearResult",
    "LogisticResult",
    "NegativeBinomial2",
    "Observed",
    "Poisson",
    "PoissonResult",
    "PreparedGeometric",
    "PreparedIV",
    "PreparedRegressionModel",
    "Reduced",
    "Regression",
    "RegressionContext",
    "RegressionDescription",
    "RegressionEstimator",
    "RegressionFit",
    "ResponsePermutation",
]
