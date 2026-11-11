"""Explicit host-side refinement of fixed-draw spatial-survival likelihoods.

Refinement diagnoses numerical sensitivity. It neither proves integration error
bounds nor establishes likelihood existence, global optimality or calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp

from statadic.estimation.survival.spatial import (
    SpatialSurvivalFit,  # noqa: TC001
    SpatialSurvivalML,  # noqa: TC001
    SpatialSurvivalProblem,  # noqa: TC001 -- runtime annotation
    spatial_survival_log_likelihood,
)
from statadic.optimize.integration import IntegrationDesign

_MIN_REPLICATES = 2
_MIN_DRAWS = 2
_SURVIVAL_SCORE = jax.grad(spatial_survival_log_likelihood, argnums=0)


@dataclass(frozen=True)
class IntegrationRefinement:
    """Bounded draw-doubling and independent-validation controls."""

    initial_draws: int = 1024
    max_draws: int = 65536
    replicates: int = 8
    log_probability_tolerance: float = 1e-3
    parameter_shift_se: float = 0.02

    def __post_init__(self) -> None:
        """Reject invalid budgets rather than silently changing them."""
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.initial_draws, self.max_draws, self.replicates)
        ):
            message = "Draw and replicate budgets must be integers, excluding bool."
            raise TypeError(message)
        invalid_budget = (
            self.initial_draws < _MIN_DRAWS
            or self.initial_draws % 2
            or self.max_draws < self.initial_draws
            or self.max_draws % 2
            or self.replicates < _MIN_REPLICATES
        )
        invalid_tolerance = (
            not math.isfinite(self.log_probability_tolerance)
            or not math.isfinite(self.parameter_shift_se)
            or self.log_probability_tolerance <= 0
            or self.parameter_shift_se <= 0
        )
        if invalid_budget or invalid_tolerance:
            message = (
                "Refinement requires positive tolerances, >=2 replicates "
                "and even ordered draw budgets."
            )
            raise ValueError(message)


@dataclass(frozen=True)
class RefinementAttempt:
    """One fit and its explicitly separate validation-stream diagnostics."""

    draws: int
    fit: SpatialSurvivalFit
    train_log_likelihood: float | None
    validation_log_likelihood: float | None
    max_log_probability_gap: float | None
    max_budget_log_probability_gap: float | None
    max_score_gap: float | None
    score_shift_in_se: float | None
    parameter_shift_in_se: float | None
    train_score: tuple[float, ...] | None
    validation_score: tuple[float, ...] | None
    parameter_shifts_in_se: tuple[float, ...] | None


@dataclass(frozen=True)
class RefinedSpatialSurvivalFit:
    """Final candidate and all attempted budgets, with a numerical stability status."""

    fit: SpatialSurvivalFit
    attempts: tuple[RefinementAttempt, ...]
    stable: bool
    status: Literal[
        "stable",
        "budget-exhausted",
        "invalid-fit",
        "information-unavailable",
        "integration-unavailable",
    ]
    validation_designs: tuple[IntegrationDesign, ...]
    config: IntegrationRefinement


_DEFAULT_REFINEMENT = IntegrationRefinement()


def _key(design: IntegrationDesign) -> jax.Array:
    if design.key is None:
        message = "Refinement requires keyed integration designs."
        raise ValueError(message)
    return design.key


def _with_draws(
    problem: SpatialSurvivalProblem, designs: tuple[IntegrationDesign, ...], draws: int
) -> SpatialSurvivalProblem:
    periods = tuple(
        replace(
            period,
            integration=IntegrationDesign.from_key(
                _key(design),
                replicates=design.replicates,
                draws=draws,
                proposal=design.proposal,
            ),
        )
        for period, design in zip(problem.periods, designs, strict=True)
    )
    return replace(problem, periods=periods)


def _score(
    estimator: SpatialSurvivalML, problem: SpatialSurvivalProblem, parameters: jax.Array
) -> jax.Array:
    return _SURVIVAL_SCORE(
        parameters,
        problem,
        rho_fixed=estimator.rho_fixed,
        integration=estimator.integration,
    )


def refine_spatial_survival(  # noqa: C901, PLR0912, PLR0914, PLR0915  # bounded retained attempts
    estimator: SpatialSurvivalML,
    problem: SpatialSurvivalProblem,
    *,
    validation_designs: tuple[IntegrationDesign, ...],
    config: IntegrationRefinement = _DEFAULT_REFINEMENT,
) -> RefinedSpatialSurvivalFit:
    """Refit fixed keyed draw prefixes and compare independent validation streams.

    All training and validation keys must be explicit and distinct, one of each
    per period. Supplied point arrays cannot be extended and are rejected. Entry
    truncation is outside the controller's initial release. Numerical stability
    requires two fits, positive-definite observed information, the per-period
    train/validation and budget log-probability gaps, score-implied parameter
    change and refit parameter change in standard errors.
    """
    if not problem.periods or len(validation_designs) != len(problem.periods):
        message = "Provide one validation design per nonempty problem period."
        raise ValueError(message)
    if not estimator.compute_information:
        message = "Refinement requires observed information to assess parameter shifts."
        raise ValueError(message)
    training = tuple(period.integration for period in problem.periods)
    designs = (*training, *validation_designs)
    if any(design.points is not None or design.key is None for design in designs):
        message = (
            "Refinement needs explicit keyed streams; point arrays cannot be extended."
        )
        raise ValueError(message)
    if any(
        design.replicates != config.replicates or design.draws != config.initial_draws
        for design in designs
    ):
        message = (
            "Every design must match the initial draw and replicate configuration."
        )
        raise ValueError(message)
    if any(period.entry_indices for period in problem.periods):
        message = "Positive-entry refinement is not yet supported."
        raise ValueError(message)
    keys = [
        tuple(jax.device_get(jax.random.key_data(_key(design))).tolist())
        for design in designs
    ]
    if len(keys) != len(set(keys)):
        message = (
            "Training and validation streams must use distinct keys for every period."
        )
        raise ValueError(message)
    attempts: list[RefinementAttempt] = []
    previous: SpatialSurvivalFit | None = None
    draws = config.initial_draws
    status: Literal[
        "stable",
        "budget-exhausted",
        "invalid-fit",
        "information-unavailable",
        "integration-unavailable",
    ] = "budget-exhausted"
    while True:
        train = _with_draws(problem, training, draws)
        validation = _with_draws(problem, validation_designs, draws)
        fit = estimator.estimate(train)
        train_log: float | None = None
        validation_log: float | None = None
        log_gap: float | None = None
        budget_gap: float | None = None
        score_gap: float | None = None
        score_shift: float | None = None
        shift: float | None = None
        train_score: tuple[float, ...] | None = None
        validation_score: tuple[float, ...] | None = None
        shifts: tuple[float, ...] | None = None
        if bool(fit.valid) and bool(fit.integration_valid):
            train_values = []
            validation_values = []
            for left, right in zip(train.periods, validation.periods, strict=True):
                lp = replace(problem, periods=(left,))
                rp = replace(problem, periods=(right,))
                train_values.append(
                    spatial_survival_log_likelihood(
                        fit.parameters,
                        lp,
                        rho_fixed=estimator.rho_fixed,
                        integration=estimator.integration,
                    )
                )
                validation_values.append(
                    spatial_survival_log_likelihood(
                        fit.parameters,
                        rp,
                        rho_fixed=estimator.rho_fixed,
                        integration=estimator.integration,
                    )
                )
            left_values, right_values = (
                jnp.stack(train_values),
                jnp.stack(validation_values),
            )
            train_log = float(jnp.sum(left_values))
            validation_log = float(jnp.sum(right_values))
            log_gap = float(jnp.max(jnp.abs(left_values - right_values)))
            if previous is not None:
                prior = _with_draws(problem, training, attempts[-1].draws)
                prior_values = jnp.stack(
                    [
                        spatial_survival_log_likelihood(
                            fit.parameters,
                            replace(problem, periods=(period,)),
                            rho_fixed=estimator.rho_fixed,
                            integration=estimator.integration,
                        )
                        for period in prior.periods
                    ]
                )
                budget_gap = float(jnp.max(jnp.abs(left_values - prior_values)))
            left_score = _score(estimator, train, fit.parameters)
            right_score = _score(estimator, validation, fit.parameters)
            train_score = tuple(float(item) for item in left_score)
            validation_score = tuple(float(item) for item in right_score)
            score_gap = float(jnp.max(jnp.abs(left_score - right_score)))
            if previous is not None and bool(fit.information_available):
                eig = jnp.linalg.eigvalsh(
                    (fit.observed_information + fit.observed_information.T) / 2
                )
                if bool(jnp.all(jnp.isfinite(eig)) & jnp.all(eig > 0)):
                    variance = jnp.diag(jnp.linalg.inv(fit.observed_information))
                    if bool(jnp.all(jnp.isfinite(variance)) & jnp.all(variance > 0)):
                        ratios = jnp.abs(
                            fit.parameters - previous.parameters
                        ) / jnp.sqrt(variance)
                        shifts = tuple(float(item) for item in ratios)
                        shift = float(jnp.max(ratios))
                        delta = jnp.linalg.solve(
                            fit.observed_information, left_score - right_score
                        )
                        score_shift = float(
                            jnp.max(jnp.abs(delta) / jnp.sqrt(variance))
                        )
        attempts.append(
            RefinementAttempt(
                draws,
                fit,
                train_log,
                validation_log,
                log_gap,
                budget_gap,
                score_gap,
                score_shift,
                shift,
                train_score,
                validation_score,
                shifts,
            )
        )
        if not bool(fit.integration_valid):
            status = "integration-unavailable"
            break
        if not bool(fit.valid):
            status = "invalid-fit"
            break
        if (
            log_gap is None
            or score_gap is None
            or not bool(jnp.isfinite(log_gap) & jnp.isfinite(score_gap))
        ):
            status = "integration-unavailable"
            break
        if previous is not None and (shift is None or score_shift is None):
            status = "information-unavailable"
            break
        if previous is not None and (
            budget_gap is None
            or not math.isfinite(budget_gap)
            or score_shift is None
            or not math.isfinite(score_shift)
        ):
            status = "integration-unavailable"
            break
        if shift is not None and budget_gap is not None and score_shift is not None:
            probability_stable = (
                max(log_gap, budget_gap) <= config.log_probability_tolerance
            )
            parameter_stable = max(shift, score_shift) <= config.parameter_shift_se
            if probability_stable and parameter_stable:
                status = "stable"
                break
        if draws >= config.max_draws:
            break
        previous = fit
        draws = min(draws * 2, config.max_draws)
    return RefinedSpatialSurvivalFit(
        fit, tuple(attempts), status == "stable", status, validation_designs, config
    )


__all__ = [
    "IntegrationRefinement",
    "RefinedSpatialSurvivalFit",
    "RefinementAttempt",
    "refine_spatial_survival",
]
