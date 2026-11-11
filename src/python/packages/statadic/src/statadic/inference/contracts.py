# jaxtyping dimensions are runtime contracts.
"""Structural capabilities consumed by shared parameter inference.

An uncertainty owns its actual fit and covariance. Sample capacity is static
array capacity, while covariance.n_observations records actual contributions;
these differ for partially available period estimators. Providers implement
these read-only properties without inheriting a result hierarchy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from statadic.core.results import (
    SolverResult,  # noqa: TC001
)
from statadic.core.typing import (
    Array,  # noqa: TC001
)

__all__ = [
    "FittedUncertainty",
    "LikelihoodFit",
    "ParameterFit",
    "SamplingCovariance",
]


@runtime_checkable
class ParameterFit(Protocol):
    """Retained coefficient state required for intervals and restrictions."""

    @property
    def parameters(self) -> Array:
        """Return coefficients in their retained parameter order."""
        ...

    @property
    def parameter_names(self) -> tuple[str, ...] | None:
        """Return the optional ordered coefficient schema."""
        ...

    @property
    def solver(self) -> SolverResult[Array]:
        """Return numerical solver evidence for the actual fit."""
        ...


@runtime_checkable
class LikelihoodFit(Protocol):
    """Maximised log likelihood and local validity of a likelihood fit."""

    @property
    def log_likelihood(self) -> Array:
        """Return the maximised log likelihood, undefined where invalid."""
        ...

    @property
    def valid(self) -> Array:
        """Return whether the fit is locally numerically valid."""
        ...


@runtime_checkable
class SamplingCovariance(Protocol):
    """Parameter covariance capability required by shared summaries."""

    @property
    def values(self) -> Array:
        """Return the parameter sampling covariance matrix."""
        ...

    @property
    def n_observations(self) -> Array:
        """Return the actual contribution count, distinct from static capacity."""
        ...


@runtime_checkable
class FittedUncertainty(Protocol):
    """Actual-fit uncertainty accepted by shared interval/restriction methods."""

    @property
    def fit(self) -> ParameterFit:
        """Return the actual fitted coefficient state."""
        ...

    @property
    def covariance(self) -> SamplingCovariance:
        """Return covariance aligned with that fitted coefficient state."""
        ...

    @property
    def valid(self) -> Array:
        """Return inference availability independently of solver convergence."""
        ...

    @property
    def status(self) -> Array:
        """Return the provider's inference availability reason."""
        ...

    @property
    def sample_capacity(self) -> int:
        """Return static retained contribution capacity for summary metadata."""
        ...
