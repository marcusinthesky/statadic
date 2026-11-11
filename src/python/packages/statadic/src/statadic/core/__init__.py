# jaxtyping shape strings are runtime contracts.
"""Shared mathematical, statistical and numerical result contracts.

Axioms describes laws; domains describes spaces, evidence and authority; matrices
carries typed pairwise values. Equations defines estimating contributions,
Jacobians and variability. Results retains solver and inference evidence;
statistics gives estimates and tests their semantic provenance. Typing,
random and transforms provide shared composition policies.

Core depends on no statistical implementation family. Certified constructors
are producer contracts; numerical convergence and inferential validity remain
separate. Import the specific contract module when extending a family.
"""

from __future__ import annotations

__all__ = [
    "DIVERGENCE",
    "METRIC",
    "NEGATIVE_TYPE",
    "PSEUDOMETRIC",
    "QUASIMETRIC",
    "SEMIMETRIC",
    "STATIC",
    "STRONG_NEGATIVE_TYPE",
    "Array",
    "Axioms",
    "Bool",
    "DMat",
    "DivergenceLaw",
    "EquationContract",
    "EquationDescriptor",
    "EquationJacobian",
    "EstimateResult",
    "EstimatingEquationKind",
    "EstimatingFunction",
    "Float",
    "IndependentRows",
    "InferenceStatus",
    "Int",
    "Law",
    "LawProperties",
    "LawT",
    "LawfulCallable",
    "MeanCenteredLongRun",
    "MeanEquationJacobian",
    "Metric",
    "MetricLaw",
    "MomentCondition",
    "NegativeType",
    "NegativeTypeLaw",
    "Nonnegative",
    "Num",
    "OneWayCluster",
    "PRNGKey",
    "ParameterCovariance",
    "Params",
    "PlainLongRun",
    "Premetric",
    "PremetricLaw",
    "PseudometricLaw",
    "QuasimetricLaw",
    "Real",
    "ResultT",
    "SandwichCovariance",
    "Scalar",
    "ScalarLike",
    "Semimetric",
    "SemimetricLaw",
    "Separating",
    "SolverResult",
    "SolverStatus",
    "Static",
    "StrongNegativeTypeLaw",
    "Symmetric",
    "TestResult",
    "TriangleInequality",
    "VariabilityMatrix",
    "VariabilityScheme",
    "ZeroDiagonal",
    "cell_key",
    "mean_equation_jacobian",
    "parameter_covariance",
    "sandwich_covariance",
    "typed_jit",
    "typed_vmap",
    "unsafe_assume_mean_equation_jacobian",
]


# t46.1 — typing foundation.
from jaxtyping import Bool, Float, Int, Num, Real, Scalar

from statadic.core.axioms import (
    DIVERGENCE,
    METRIC,
    NEGATIVE_TYPE,
    PSEUDOMETRIC,
    QUASIMETRIC,
    SEMIMETRIC,
    STRONG_NEGATIVE_TYPE,
    Axioms,
    DivergenceLaw,
    Law,
    LawProperties,
    Metric,
    MetricLaw,
    NegativeType,
    NegativeTypeLaw,
    Nonnegative,
    Premetric,
    PremetricLaw,
    PseudometricLaw,
    QuasimetricLaw,
    Semimetric,
    SemimetricLaw,
    Separating,
    StrongNegativeTypeLaw,
    Symmetric,
    TriangleInequality,
    ZeroDiagonal,
)
from statadic.core.equations import (
    EquationContract,
    EquationDescriptor,
    EquationJacobian,
    EstimatingEquationKind,
    EstimatingFunction,
    IndependentRows,
    MeanCenteredLongRun,
    MeanEquationJacobian,
    MomentCondition,
    OneWayCluster,
    ParameterCovariance,
    PlainLongRun,
    SandwichCovariance,
    VariabilityMatrix,
    VariabilityScheme,
    mean_equation_jacobian,
    parameter_covariance,
    sandwich_covariance,
    unsafe_assume_mean_equation_jacobian,
)
from statadic.core.matrices import (
    DMat,
)
from statadic.core.random import cell_key
from statadic.core.results import (
    EstimateResult,
    InferenceStatus,
    SolverResult,
    SolverStatus,
    TestResult,
)
from statadic.core.transforms import typed_jit, typed_vmap
from statadic.core.typing import (
    STATIC,
    Array,
    LawfulCallable,
    LawT,
    Params,
    PRNGKey,
    ResultT,
    ScalarLike,
    Static,
)
