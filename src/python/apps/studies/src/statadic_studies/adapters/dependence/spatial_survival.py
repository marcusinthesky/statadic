"""Small joint Gaussian spatial log-time pilot with retained numerical status."""

from __future__ import annotations

import jax
import numpy as np
from statadic.estimation.survival import (
    SpatialSurvivalML,
    SpatialSurvivalProblem,
    prepare_spatial_survival_period,
)
from statadic.estimation.survival.refinement import (
    IntegrationRefinement,
    refine_spatial_survival,
)
from statadic.operators.spatial import ObservedSpatialPeriod, prepare_spatial_weights
from statadic.optimize.integration import GaussianIntegration, IntegrationDesign

from statadic_studies.adapters._random import rng
from statadic_studies.definitions.dependence.spatial_survival import (
    SPATIAL_SURVIVAL_PILOT_STUDIES,
)


def spatial_design():
    n = 8
    weights = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    design = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    return weights, design


def _key(study, scenario, replicate, seed, stream):
    generator = rng(study, scenario, replicate, seed=seed, stream=stream)
    return jax.random.key(int(generator.integers(0, 2**32, dtype=np.uint32)))


def make_problem(scenario, replicate, seed, draws):
    """Generate independent periods before creating their integration streams."""
    study = "survival.spatial-joint-pilot"
    rho = float(scenario.removeprefix("rho"))
    weights, design = spatial_design()
    graph = ObservedSpatialPeriod(
        prepare_spatial_weights(weights), tuple(range(8)), tuple(range(8))
    )
    generator = rng(study, scenario, replicate, seed=seed)
    periods, validations, censor_counts = [], [], []
    # Fixed administrative caps are declared independently of all event times.
    censor_time = np.full(8, np.inf)
    censor_time[:3] = 0.6
    for period_index in range(3):
        logs = np.linalg.solve(
            np.eye(8) - rho * weights,
            design @ np.array([0.1, 0.3]) + 0.6 * generator.normal(size=8),
        )
        event_time = np.exp(logs)
        event = event_time <= censor_time
        initial = IntegrationDesign.from_key(
            _key(study, scenario, replicate, seed, f"training-{period_index}"),
            replicates=2,
            draws=draws,
        )
        validation = IntegrationDesign.from_key(
            _key(study, scenario, replicate, seed, f"validation-{period_index}"),
            replicates=2,
            draws=draws,
        )
        periods.append(
            prepare_spatial_survival_period(
                graph,
                design,
                np.minimum(event_time, censor_time),
                event,
                integration=initial,
            )
        )
        validations.append(validation)
        censor_counts.append(int(np.sum(~event)))
    return (
        SpatialSurvivalProblem(tuple(periods), ("intercept", "slope")),
        tuple(validations),
        censor_counts,
    )


def _number(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _attempt(attempt):
    fit = attempt.fit
    return {
        "draws": attempt.draws,
        "fit_valid": bool(fit.valid),
        "solver_converged": bool(fit.solver.converged),
        "solver_status": int(fit.solver.status),
        "integration_valid": bool(fit.integration_valid),
        "integration_relative_error": _number(fit.integration_error),
        "information_available": bool(fit.information_available),
        "identified": bool(fit.identified),
        "input_valid": bool(fit.input_valid),
        "existence_status": int(fit.existence_status),
        "training_log_likelihood": _number(attempt.train_log_likelihood),
        "validation_log_likelihood": _number(attempt.validation_log_likelihood),
        "training_validation_log_gap": _number(attempt.max_log_probability_gap),
        "budget_log_gap": _number(attempt.max_budget_log_probability_gap),
        "score_gap": _number(attempt.max_score_gap),
        "score_shift_se": _number(attempt.score_shift_in_se),
        "parameter_shift_se": _number(attempt.parameter_shift_in_se),
        "training_score": None
        if attempt.train_score is None
        else [_number(v) for v in attempt.train_score],
        "validation_score": None
        if attempt.validation_score is None
        else [_number(v) for v in attempt.validation_score],
        "parameter_shifts_se": None
        if attempt.parameter_shifts_in_se is None
        else [_number(v) for v in attempt.parameter_shifts_in_se],
    }


def _outcome(fit, ok, failure, metadata):
    estimate = _number(fit.parameters[1])
    if not ok or estimate is None:
        return {"valid": False, "failure": failure, **metadata}
    return {"valid": True, "estimate": estimate, "truth": 0.3, **metadata}


def run_spatial_survival_pilot(study, scenario, replicate, seed, *, draws=999):
    """Run actual fixed and adaptive estimators on a common joint dataset."""
    if (
        study not in SPATIAL_SURVIVAL_PILOT_STUDIES
        or scenario not in SPATIAL_SURVIVAL_PILOT_STUDIES[study].scenarios
    ):
        raise ValueError("Unknown spatial survival study or scenario")
    if type(draws) is not int or draws < 2:
        raise ValueError("At least two integration draws are required")
    initial_draws = draws + draws % 2
    problem, validation, censor_counts = make_problem(
        scenario, replicate, seed, initial_draws
    )
    estimator = SpatialSurvivalML(
        max_steps=64, integration=GaussianIntegration(chunk_size=16)
    )
    refined = refine_spatial_survival(
        estimator,
        problem,
        validation_designs=validation,
        config=IntegrationRefinement(
            initial_draws=initial_draws, max_draws=2 * initial_draws, replicates=2
        ),
    )
    history = [_attempt(attempt) for attempt in refined.attempts]
    first = refined.attempts[0].fit
    metadata = {
        "censored_per_period": censor_counts,
        "requested_draws": draws,
        "initial_draws": initial_draws,
        "refinement_status": refined.status,
        "refinement_stable": bool(refined.stable),
        "attempts": history,
    }
    return {
        "fixed-draw-spatial-ML": _outcome(
            first,
            bool(first.valid) and bool(first.integration_valid),
            "fixed-fit-unavailable",
            metadata,
        ),
        "refined-spatial-ML": _outcome(
            refined.fit,
            bool(refined.stable)
            and bool(refined.fit.valid)
            and bool(refined.fit.integration_valid),
            f"refinement-{refined.status}",
            metadata,
        ),
    }
