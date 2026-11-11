"""Canonical regression specification and family-specific retained capabilities."""

from statadic.api.regression._conditional import (
    Binomial,
    Gamma,
    Gaussian,
    HurdlePoisson,
    HurdleResult,
    NegativeBinomial2,
    Observed,
    Poisson,
)
from statadic.api.regression._geometric import (
    FreedmanLane,
    GeometricInference,
    GeometricResponse,
    GeometricResult,
    GeometricTestEvidence,
    PreparedGeometric,
    Reduced,
    ResponsePermutation,
)
from statadic.api.regression._glm import LogisticResult, PoissonResult
from statadic.api.regression._iv import IVResult, PreparedIV
from statadic.api.regression._likelihood import LikelihoodResult
from statadic.api.regression._linear import LinearResult
from statadic.api.regression._model import (
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
