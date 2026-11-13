# ruff: noqa: TC001  # Public annotations resolve under the runtime import hook.
"""Explicit fixed-graph diagnostics around the existing numerical procedures."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from statadic.api.base import FitContext
from statadic.api.diagnostics import RetainedLinearFit
from statadic.api.hypotheses import (
    ChiSquare,
    Hypothesis,
    Normal,
    Proposition,
    Reference,
    TestOutcome,
)
from statadic.api.reporting import MetadataValue, _row_identity
from statadic.core.results import TestResult
from statadic.core.typing import Array
from statadic.estimation.linear import LinearFit
from statadic.estimation.spatial import (
    SpatialFit,
    SpatialGMMFit,
    SpatialIVFit,
    SpatialPanelFit,
)
from statadic.inference.spatial import (
    SpatialUncertainty,
    spatial_ak_test,
    spatial_common_factor_test,
    spatial_diagnostics,
    spatial_likelihood_ratio,
)
from statadic.operators.spatial import SpatialWeights

__all__ = [
    "AnselinKelejian",
    "CommonFactor",
    "CommonFactorRestriction",
    "NestedSpatialModel",
    "NoSpatialDependence",
    "ResidualMoran",
    "RetainedSpatialFit",
    "RetainedSpatialLikelihood",
    "SpatialDiagnostic",
    "SpatialLM",
    "likelihood_ratio",
]


@dataclass(frozen=True)
class NoSpatialDependence:
    """No specified spatial dependence, conditional on the supplied fixed graph."""

    component: Literal["lag", "error", "joint"]

    def describe(self) -> str:
        """Describe the model null, never equating residual Moran I with zero."""
        if self.component == "lag":
            return "No endogenous spatial outcome-lag effect on the fixed graph."
        if self.component == "error":
            return "No spatial dependence of innovations on the fixed graph."
        return "No spatial outcome-lag or innovation dependence on the fixed graph."

    def metadata(self) -> dict[str, MetadataValue]:
        """Retain the tested component independently of its procedure."""
        return {"kind": "no-spatial-dependence", "component": self.component}


@dataclass(frozen=True)
class CommonFactorRestriction:
    """The nonlinear SDM common-factor restrictions for selected coefficients."""

    columns: tuple[str, ...]
    rho_name: str = "rho"

    def describe(self) -> str:
        """State the nonlinear restriction actually evaluated by the kernel."""
        return f"theta[j] + {self.rho_name} * beta[j] = 0 for {self.columns}."

    def metadata(self) -> dict[str, MetadataValue]:
        """Retain the selected coefficient names."""
        return {
            "kind": "spatial-common-factor",
            "columns": self.columns,
            "rho_name": self.rho_name,
        }


@dataclass(frozen=True)
class NestedSpatialModel:
    """An explicit restricted model nested in the supplied unrestricted fit."""

    restrictions: int

    def describe(self) -> str:
        """State the declared nested-model comparison."""
        return f"The {self.restrictions} declared nested restrictions hold."

    def metadata(self) -> dict[str, MetadataValue]:
        """Retain the declared restriction count."""
        return {"kind": "nested-spatial-model", "restrictions": self.restrictions}


@runtime_checkable
class RetainedSpatialFit(Protocol):
    """Actual cross-sectional fit and the host sample to which it belongs."""

    @property
    def core(self) -> SpatialFit | SpatialIVFit | SpatialGMMFit | LinearFit:
        """Return the retained numerical fit without lowering another model."""
        ...

    @property
    def context(self) -> FitContext:
        """Return the actual observation identities."""
        ...

    @property
    def weights(self) -> SpatialWeights:
        """Return the exact graph used by the estimator."""
        ...


@runtime_checkable
class SpatialDiagnostic(Protocol):
    """A procedure supported by one of the retained spatial fit families."""

    def evaluate_spatial(self, fitted: RetainedSpatialFit) -> TestOutcome:
        """Evaluate without refitting or silently changing graph or covariance."""
        ...


def _graph_construction(
    context: FitContext, weights: SpatialWeights[Array], unit: str | None
) -> dict[str, MetadataValue]:
    identities = (
        context.row_ids if unit is None else context.observations.resolve_grouping(unit)
    )
    graph_ids = tuple(jax.device_get(weights.unit_ids).tolist())
    if identities != graph_ids:
        message = "Diagnostic rows must match the graph's ordered numeric unit IDs."
        raise ValueError(message)
    values = jax.device_get(weights.values)
    return {
        "rows": _row_identity(context.row_ids),
        "graph_units": _row_identity(graph_ids),
        "graph_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "unit": unit,
        "graph_conditioning": "supplied fixed exogenous graph",
    }


def _outcome[NullT: Proposition](
    name: str,
    null: NullT,
    native: TestResult[Array],
    reference: Reference,
    construction: dict[str, MetadataValue],
    *,
    reason: str,
    details: dict[str, float] | None = None,
) -> TestOutcome:
    statistic, pvalue = float(native.statistic), float(native.pvalue)
    available = math.isfinite(statistic) and math.isfinite(pvalue)
    return TestOutcome(
        name,
        Hypothesis(null),
        statistic if available else None,
        pvalue if available else None,
        reference if available else None,
        available,
        unavailable_reason=None if available else reason,
        details={} if details is None else details,
        construction={**construction, "native_valid": available},
        assumptions=(
            "The graph is fixed and exogenous; its construction is not tested.",
            "The stated procedure's model and reference assumptions are required.",
        ),
        evidence=native,
    )


def _ols_reason(fit: LinearFit, weights: SpatialWeights[Array]) -> str:
    if not bool(fit.valid):
        return "invalid-fit"
    if not bool(jnp.all(fit.relative_precision == 1)):
        return "weighted-fit"
    if not bool(jnp.any(jnp.all(fit.design == 1, axis=0))):
        return "missing-intercept"
    if not bool(jnp.all(jnp.diag(weights.values) == 0)):
        return "nonzero-graph-diagonal"
    return "spatial-reference-unavailable"


@dataclass(frozen=True)
class ResidualMoran:
    """Regression-adjusted residual Moran I with its two-sided normal tail."""

    weights: SpatialWeights[Array] = field(repr=False)
    unit: str | None = field(default=None, kw_only=True)
    name: str = "moran_residual"

    def evaluate_linear(self, fitted: RetainedLinearFit) -> TestOutcome:
        """Condition on the actual OLS design and supplied aligned graph."""
        native = spatial_diagnostics(fitted.core, self.weights)
        construction = _graph_construction(fitted.context, self.weights, self.unit)
        expectation, variance = (
            float(native.moran_expectation),
            float(native.moran_variance),
        )
        details = {}
        if math.isfinite(expectation):
            details["expectation"] = expectation
        if math.isfinite(variance):
            details["variance"] = variance
        valid_reference = (
            math.isfinite(variance) and variance > 0 and math.isfinite(expectation)
        )
        reference = (
            Normal(expectation, math.sqrt(variance)) if valid_reference else Normal()
        )
        return _outcome(
            self.name,
            NoSpatialDependence("error"),
            native.moran,
            reference,
            {
                **construction,
                "tail": "two-sided",
                "reference": "regression-adjusted normal approximation",
            },
            reason=_ols_reason(fitted.core, self.weights),
            details=details,
        )


@dataclass(frozen=True)
class SpatialLM:
    """One Gaussian OLS spatial score test on an explicit fixed graph.

    Robust lag/error adjusts for the competing local alternative. It does not
    supply heteroskedastic, weak-IV or learned-graph robustness.
    """

    weights: SpatialWeights[Array] = field(repr=False)
    kind: Literal["lag", "error", "robust-lag", "robust-error", "joint"] = field(
        kw_only=True
    )
    unit: str | None = field(default=None, kw_only=True)
    name: str | None = None

    def evaluate_linear(self, fitted: RetainedLinearFit) -> TestOutcome:
        """Select one native test without using the bundle's aggregate validity."""
        native = spatial_diagnostics(fitted.core, self.weights)
        tests = {
            "lag": native.lm_lag,
            "error": native.lm_error,
            "robust-lag": native.robust_lm_lag,
            "robust-error": native.robust_lm_error,
            "joint": native.lm_sarma,
        }
        if self.kind not in tests:
            message = "Unknown spatial LM kind."
            raise ValueError(message)
        component: Literal["lag", "error", "joint"] = (
            "joint"
            if self.kind == "joint"
            else "lag"
            if self.kind.endswith("lag")
            else "error"
        )
        return _outcome(
            self.name or f"lm_{self.kind.replace('-', '_')}",
            NoSpatialDependence(component),
            tests[self.kind],
            ChiSquare(2 if self.kind == "joint" else 1),
            {
                **_graph_construction(fitted.context, self.weights, self.unit),
                "kind": self.kind,
            },
            reason=_ols_reason(fitted.core, self.weights),
        )


@dataclass(frozen=True)
class AnselinKelejian:
    """Residual spatial-error test after the actual spatial 2SLS estimate."""

    name: str = "anselin_kelejian"

    def evaluate_spatial(self, fitted: RetainedSpatialFit) -> TestOutcome:
        """Use the retained instrument projection; accept no substitute residuals."""
        if not isinstance(fitted.core, SpatialIVFit):
            message = "Anselin-Kelejian requires a retained spatial IV fit."
            raise TypeError(message)
        return _outcome(
            self.name,
            NoSpatialDependence("error"),
            spatial_ak_test(fitted.core),
            ChiSquare(1),
            {
                "rows": _row_identity(fitted.context.row_ids),
                "case": "general-endogenous",
            },
            reason="spatial-iv-reference-unavailable",
        )


@dataclass(frozen=True)
class CommonFactor:
    """Nonlinear SDM Wald test using explicitly attached, fit-bound covariance."""

    covariance: SpatialUncertainty = field(kw_only=True, repr=False)
    name: str = "common_factor"

    def evaluate_spatial(self, fitted: RetainedSpatialFit) -> TestOutcome:
        """Consume existing uncertainty belonging to exactly this core fit."""
        fit = fitted.core
        if not isinstance(fit, SpatialFit):
            message = "Common-factor inference requires a retained SDM likelihood fit."
            raise TypeError(message)
        if self.covariance.fit is not fit:
            message = "Common-factor covariance belongs to another fitted model."
            raise ValueError(message)
        native = spatial_common_factor_test(self.covariance)
        names = fit.parameter_names or tuple(
            f"x{j}" for j in range(fit.design.shape[1])
        )
        columns = tuple(names[j] for j in fit.slx_columns)
        rho_name = "rho" if fit.parameter_names is None else names[fit.design.shape[1]]
        return _outcome(
            self.name,
            CommonFactorRestriction(columns, rho_name),
            native,
            ChiSquare(len(columns)),
            {
                "rows": _row_identity(fitted.context.row_ids),
                "covariance": self.covariance.method,
            },
            reason="common-factor-reference-unavailable",
        )


@runtime_checkable
class RetainedSpatialLikelihood(Protocol):
    """Already fitted Gaussian likelihood and its exact host observation IDs."""

    @property
    def core(self) -> SpatialFit | SpatialPanelFit:
        """Return the actual likelihood fit."""
        ...

    @property
    def context(self) -> FitContext:
        """Return its retained observations."""
        ...


def likelihood_ratio(
    unrestricted: RetainedSpatialLikelihood,
    restricted: RetainedSpatialLikelihood,
    *,
    degrees_of_freedom: int,
    assume_nested: bool = False,
) -> TestOutcome:
    """Compare two already-fitted matched interior models with declared nesting."""
    if not isinstance(
        unrestricted.core, (SpatialFit, SpatialPanelFit)
    ) or not isinstance(restricted.core, (SpatialFit, SpatialPanelFit)):
        message = "Likelihood comparisons require supported Gaussian spatial fits."
        raise TypeError(message)
    if unrestricted.context.row_ids != restricted.context.row_ids:
        message = "Likelihood comparisons require identical ordered observation IDs."
        raise ValueError(message)
    native = spatial_likelihood_ratio(
        unrestricted.core,
        restricted.core,
        degrees_of_freedom=degrees_of_freedom,
        assume_nested=assume_nested,
    )
    return _outcome(
        "likelihood_ratio",
        NestedSpatialModel(degrees_of_freedom),
        native,
        ChiSquare(degrees_of_freedom),
        {
            "rows": _row_identity(unrestricted.context.row_ids),
            "assume_nested": assume_nested,
        },
        reason="nested-likelihood-reference-unavailable",
    )
