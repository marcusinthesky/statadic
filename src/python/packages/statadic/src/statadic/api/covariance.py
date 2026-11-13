"""Retained covariance results and aligned preparation for ``Estimand``.

Numerical estimators belong to ``statadic.estimation.covariance``. This owner adds
column identities and reporting without implementing another covariance formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import jax.numpy as jnp

from statadic.api.base import Result
from statadic.api.data import Observations
from statadic.api.reporting import Metric, Report
from statadic.estimation.covariance import (
    CovarianceProblem,
    LedoitWolf,
    SampleCovariance,
)
from statadic.estimation.covariance.estimators import LedoitWolfKernelResult

if TYPE_CHECKING:
    from statadic.api.base import Estimator
    from statadic.core.typing import Array, ArrayLike

__all__ = ["CovarianceContext", "CovarianceEstimate", "PreparedCovariance"]

_MATRIX_NDIM = 2


@dataclass(frozen=True)
class CovarianceContext:
    """The actual estimator, variable schema and ordered observation identities."""

    estimator: str
    n_observations: int
    n_variables: int
    columns: tuple[str, ...] = ()
    row_ids: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class PreparedCovariance:
    """Owned covariance inputs belonging to one quantity specification."""

    problem: CovarianceProblem
    context: CovarianceContext
    model_token: object


def _covariance_panel(
    data: Observations | ArrayLike,
    columns: tuple[str, ...] | None,
) -> tuple[Array, tuple[str, ...], tuple[int | str, ...]]:
    if isinstance(data, Observations):
        if columns is None:
            message = "Covariance on Observations requires explicit columns."
            raise ValueError(message)
        panel = jnp.column_stack([data.numeric(name) for name in columns])
        row_ids = data.row_ids
    else:
        panel = jnp.asarray(data)
        row_ids = tuple(range(panel.shape[0])) if panel.ndim else ()
    if panel.ndim != _MATRIX_NDIM or panel.shape[1] < 1:
        message = "Covariance data must have shape (n_observations, n_variables)."
        raise ValueError(message)
    if not jnp.issubdtype(panel.dtype, jnp.number) or jnp.iscomplexobj(panel):
        message = "Covariance data must contain real numeric values."
        raise TypeError(message)
    panel = panel.astype(jnp.result_type(panel, 1.0))
    if not bool(jnp.all(jnp.isfinite(panel))):
        message = "Covariance data must contain only finite values."
        raise ValueError(message)
    names = columns or tuple(f"variable_{i}" for i in range(panel.shape[1]))
    if len(names) != panel.shape[1]:
        message = "Covariance columns must match the variable axis."
        raise ValueError(message)
    return jnp.array(panel, copy=True), names, row_ids


def _covariance_domain(
    panel: Array,
    estimator: Estimator[CovarianceProblem, Array | LedoitWolfKernelResult],
) -> None:
    minimum = estimator.ddof + 1 if isinstance(estimator, SampleCovariance) else 1
    if isinstance(estimator, LedoitWolf):
        minimum = 2
    if panel.shape[0] < minimum:
        message = "Too few observations for the covariance estimator."
        raise ValueError(message)
    if (
        isinstance(estimator, LedoitWolf)
        and panel.shape[1] > 1
        and bool(jnp.any(jnp.var(panel, axis=0) <= 0))
    ):
        message = "LedoitWolf requires strictly positive sample variances."
        raise ValueError(message)


def _prepare_covariance(
    data: Observations | ArrayLike,
    columns: tuple[str, ...] | None,
    estimator: Estimator[CovarianceProblem, Array | LedoitWolfKernelResult],
    token: object,
) -> PreparedCovariance:
    panel, names, row_ids = _covariance_panel(data, columns)
    _covariance_domain(panel, estimator)
    return PreparedCovariance(
        CovarianceProblem(panel),
        CovarianceContext(
            type(estimator).__name__, len(panel), panel.shape[1], names, row_ids
        ),
        token,
    )


@dataclass(frozen=True)
class CovarianceEstimate(Result["Array | LedoitWolfKernelResult", CovarianceContext]):
    """A covariance estimate and the diagnostics its numerical producer defines.

    Singular covariance matrices are allowed. Numerical availability does not
    assert positive definiteness or provide parameter sampling uncertainty.
    """

    @property
    def sigma(self) -> Array:
        """Return the actual estimated covariance matrix."""
        return (
            self.core.sigma
            if isinstance(self.core, LedoitWolfKernelResult)
            else self.core
        )

    @property
    def valid(self) -> bool:
        """Whether the retained covariance matrix is numerically finite."""
        return bool(jnp.all(jnp.isfinite(self.sigma)))

    @property
    def estimator(self) -> str:
        """Return the actual numerical estimator's name."""
        return self.context.estimator

    @property
    def n_observations(self) -> int:
        """Return the retained observation count."""
        return self.context.n_observations

    @property
    def n_variables(self) -> int:
        """Return the retained variable count."""
        return self.context.n_variables

    @property
    def shrinkage(self) -> float | None:
        """Return constant-correlation shrinkage when that estimator supplied it."""
        return (
            float(self.core.shrinkage)
            if isinstance(self.core, LedoitWolfKernelResult)
            else None
        )

    @property
    def target(self) -> Array | None:
        """Return the retained shrinkage target, absent for sample covariance."""
        return (
            self.core.target if isinstance(self.core, LedoitWolfKernelResult) else None
        )

    @override
    def report(self) -> Report:
        """Describe the point estimator without fabricating sampling uncertainty."""
        return Report(
            "Covariance estimate",
            metrics=(
                Metric("n_observations", int(self.n_observations)),
                Metric("n_variables", int(self.n_variables)),
            ),
            metadata={
                "estimator": self.estimator,
                "columns": self.context.columns,
                "row_ids": self.context.row_ids,
                "fit_valid": self.valid,
            },
            warnings=()
            if self.valid
            else ("Covariance values are not numerically finite.",),
        )
