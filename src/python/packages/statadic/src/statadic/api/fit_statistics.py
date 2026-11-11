"""Fit statistics: the typed likelihood summary and the metrics derived from it.

A family supplies a ``LikelihoodSummary`` (its maximised log likelihood, ``k``
and ``n``) and this module builds ``n_parameters``, ``log_likelihood``, AIC,
BIC and HQIC once. ``linear_fit_statistics`` adds the least-squares R-squared
metrics. Nothing here refits a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from statadic.api.reporting import Metric, ReportNumber  # noqa: TC001

if TYPE_CHECKING:
    from statadic.estimation.linear import LinearFit

__all__ = [
    "FitStatistics",
    "LikelihoodSummary",
    "gaussian_least_squares_summary",
    "information_metrics",
    "linear_fit_statistics",
]

_MIN_HQIC_OBSERVATIONS = 3

_INFORMATION_NOTE = (
    "Information criteria use the maximised log likelihood and k estimated "
    "parameters, counting scale, dispersion, shape and spatial parameters: "
    "AIC (Akaike 1974), BIC (Schwarz 1978), HQIC (Hannan and Quinn 1979)."
)

_GAUSSIAN_REASON = "Gaussian log likelihood needs a valid fit with positive SSE."
_INVALID_FIT_REASON = "Needs a valid least-squares fit."
_NO_VARIATION_REASON = "The response has no variation about its weighted mean."
_ZERO_RESPONSE_REASON = "The response is zero in every row."
_NO_DEGREES_REASON = "Needs positive residual degrees of freedom (n > rank)."


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class LikelihoodSummary:
    """The maximised log likelihood with the counts the criteria penalise.

    ``log_likelihood`` is NaN when the fit is invalid; ``unavailable_reason``
    then explains the unavailable criteria. ``n_parameters`` is ``k`` under the
    counting rule of the information note and ``n_observations`` is the number
    of observations entering the likelihood.
    """

    log_likelihood: float
    n_parameters: int
    n_observations: int
    unavailable_reason: str

    def __post_init__(self) -> None:
        """Reject malformed counts and an empty unavailability reason."""
        if not _is_int(self.n_parameters) or self.n_parameters < 0:
            message = "n_parameters must be a nonnegative integer."
            raise ValueError(message)
        if not _is_int(self.n_observations) or self.n_observations < 1:
            message = "n_observations must be a positive integer."
            raise ValueError(message)
        if not isinstance(self.unavailable_reason, str) or not (
            self.unavailable_reason.strip()
        ):
            message = "unavailable_reason must be a nonempty string."
            raise ValueError(message)


def information_metrics(
    log_likelihood: float,
    n_parameters: int,
    n_observations: ReportNumber,
    *,
    reason: str,
) -> tuple[Metric, ...]:
    """Build ``n_parameters``, ``log_likelihood``, AIC, BIC and HQIC once.

    ``AIC = -2 log L + 2k``, ``BIC = -2 log L + k log n`` and
    ``HQIC = -2 log L + 2k log log n``. A nonfinite log likelihood makes it and
    the criteria unavailable with ``reason``; HQIC is also unavailable below
    three observations, where ``log log n`` is not positive.
    """
    k = int(n_parameters)
    n = float(n_observations)
    parameters = Metric("n_parameters", k)
    if not math.isfinite(log_likelihood):
        return (
            parameters,
            *(
                Metric(name, None, available=False, unavailable_reason=reason)
                for name in ("log_likelihood", "aic", "bic", "hqic")
            ),
        )
    deviance = -2.0 * float(log_likelihood)
    hqic = (
        Metric("hqic", deviance + 2.0 * k * math.log(math.log(n)))
        if n >= _MIN_HQIC_OBSERVATIONS
        else Metric(
            "hqic",
            None,
            available=False,
            unavailable_reason="HQIC needs at least three observations.",
        )
    )
    return (
        parameters,
        Metric("log_likelihood", float(log_likelihood)),
        Metric("aic", deviance + 2.0 * k),
        Metric("bic", deviance + k * math.log(n)),
        hqic,
    )


@dataclass(frozen=True)
class FitStatistics:
    """Fit-quality metrics and the notes that state their conventions.

    A family reports these beside its own metrics, and its ``infer`` carries
    them into the inference report so one table holds coefficients, tests and
    fit statistics.
    """

    metrics: tuple[Metric, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def from_likelihood(cls, summary: LikelihoodSummary) -> FitStatistics:
        """Return the information metrics of ``summary`` with the information note."""
        return cls(
            information_metrics(
                summary.log_likelihood,
                summary.n_parameters,
                summary.n_observations,
                reason=summary.unavailable_reason,
            ),
            (_INFORMATION_NOTE,),
        )


def _r_metric(name: str, value: float, reason: str) -> Metric:
    """Keep an undefined R-squared visible with its explicit reason."""
    if math.isfinite(value):
        return Metric(name, value)
    return Metric(name, None, available=False, unavailable_reason=reason)


def gaussian_least_squares_summary(core: LinearFit) -> LikelihoodSummary:
    """Return the Gaussian likelihood of a least-squares fit at its profiled scale.

    The log likelihood is evaluated at the maximum-likelihood scale with the
    analytic precision weights, and ``k`` counts the design rank plus that
    scale. It is NaN when the fit is invalid or the weighted SSE is not positive.
    """
    n = int(core.n_observations)
    weighted_sse = float(core.weighted_sse)
    usable = bool(core.valid) and n > 0 and weighted_sse > 0.0
    log_likelihood = math.nan
    if usable:
        weights = jnp.asarray(core.relative_precision, dtype=jnp.float64)
        log_likelihood = -0.5 * n * (
            math.log(2.0 * math.pi) + math.log(weighted_sse / n) + 1.0
        ) + 0.5 * float(jnp.sum(jnp.log(weights)))
    return LikelihoodSummary(
        log_likelihood, int(core.rank) + 1, max(n, 1), _GAUSSIAN_REASON
    )


def linear_fit_statistics(  # noqa: PLR0914
    core: LinearFit, *, weighted: bool
) -> FitStatistics:
    """Return Gaussian information criteria and R-squared of a least-squares fit.

    The log likelihood is evaluated at the maximum-likelihood scale with the
    analytic precision weights, and ``k`` counts the design rank plus that
    scale. R-squared is centred when the design spans a constant, found by
    appending a column of ones and comparing numerical ranks, so fixed-effect
    indicators count; otherwise it is uncentred. Nothing is refitted.
    """
    n = int(core.n_observations)
    rank = int(core.rank)
    weighted_sse = float(core.weighted_sse)
    weights = jnp.asarray(core.relative_precision, dtype=jnp.float64)
    outcome = jnp.asarray(core.outcome, dtype=jnp.float64)
    point_valid = bool(core.valid) and n > 0
    likelihood = FitStatistics.from_likelihood(gaussian_least_squares_summary(core))
    metrics = list(likelihood.metrics)
    design = jnp.asarray(core.design, dtype=jnp.float64)
    centred = jnp.linalg.matrix_rank(
        jnp.column_stack((design, jnp.ones(n)))
    ) == jnp.linalg.matrix_rank(design)
    if centred:
        mean = float(jnp.sum(weights * outcome) / jnp.sum(weights))
        total = float(jnp.sum(weights * (outcome - mean) ** 2))
        r_available = point_valid and total > 0.0
        r_squared = 1.0 - weighted_sse / total if r_available else math.nan
        r_reason = _INVALID_FIT_REASON if not point_valid else _NO_VARIATION_REASON
        adjusted = (
            1.0 - (1.0 - r_squared) * (n - 1) / (n - rank)
            if r_available and n > rank
            else math.nan
        )
        adjusted_reason = r_reason if not r_available else _NO_DEGREES_REASON
        r_metrics = (
            _r_metric("r_squared", r_squared, r_reason),
            _r_metric("adjusted_r_squared", adjusted, adjusted_reason),
        )
        convention = (
            "R² is centred and unadjusted; adjusted R² uses n - 1 and "
            "n - rank degrees of freedom."
        )
    else:
        total = float(jnp.sum(weights * outcome**2))
        r_available = point_valid and total > 0.0
        r_squared = 1.0 - weighted_sse / total if r_available else math.nan
        r_metrics = (
            _r_metric(
                "uncentred_r_squared",
                r_squared,
                _INVALID_FIT_REASON if not point_valid else _ZERO_RESPONSE_REASON,
            ),
        )
        convention = (
            "R² is uncentred because the design spans no constant; "
            "no adjusted R² is reported."
        )
    if weighted:
        convention += " Both use the precision weights."
    metrics[1:1] = r_metrics
    # Order: n_parameters, R-squared metrics, log likelihood and criteria.
    return FitStatistics(tuple(metrics), (*likelihood.notes, convention))
