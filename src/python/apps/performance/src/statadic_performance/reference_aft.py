"""Independent delayed-entry AFT likelihood, fit and information consumers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from statadic_performance.reference_models import lognormal_objective


def aft_objective(
    parameters, design, duration, event, entry, *, family, information=True
):
    """Conditional censored likelihood with analytic score and information."""
    active = entry > 0
    if family == "lognormal":
        value, score, hessian = lognormal_objective(
            parameters, design, duration, event, information=information
        )
        if active.any():
            entry_value, entry_score, entry_information = lognormal_objective(
                parameters,
                design[active],
                entry[active],
                np.zeros(active.sum(), dtype=bool),
                information=information,
            )
            value -= entry_value
            score -= entry_score
            if information:
                hessian -= entry_information
        return value, score, hessian
    if family != "weibull":
        raise ValueError(f"Unknown AFT family: {family}")
    shape = np.exp(parameters[-1])
    eta = design @ parameters[:-1]
    log_duration = np.log(duration)
    z = shape * (log_duration - eta)
    cumulative = np.exp(z)
    entry_z = shape * (np.log(np.where(active, entry, 1)) - eta)
    entry_cumulative = np.where(active, np.exp(entry_z), 0)
    rows = event * (parameters[-1] - log_duration + z) - cumulative + entry_cumulative
    score = np.r_[
        -shape * design.T @ (cumulative - event - entry_cumulative),
        np.sum(cumulative * z - event * (1 + z) - entry_cumulative * entry_z),
    ]
    if not information:
        return -rows.sum(), score, None
    information = np.empty((len(parameters), len(parameters)))
    information[:-1, :-1] = (
        shape**2 * design.T @ ((cumulative - entry_cumulative)[:, None] * design)
    )
    cross = (
        -shape
        * design.T
        @ (cumulative * (1 + z) - event - entry_cumulative * (1 + entry_z))
    )
    information[:-1, -1] = cross
    information[-1, :-1] = cross
    information[-1, -1] = np.sum(
        cumulative * (z**2 + z) - event * z - entry_cumulative * (entry_z**2 + entry_z)
    )
    return -rows.sum(), score, information


def prepare_aft(case: str, side: str, inputs: tuple) -> tuple[Callable, tuple]:
    """Keep conditional fitting and observed information within each call."""
    family = "weibull" if case.startswith("weibull") else "lognormal"
    if side == "statadic":
        import jax
        import jax.numpy as jnp
        from statadic.estimation.survival import SurvivalML, SurvivalProblem
        from statadic.model.survival import LogNormalAFT, WeibullAFT

        specification = WeibullAFT() if family == "weibull" else LogNormalAFT()

        def execute(design, duration, event, entry):
            fit = SurvivalML(specification, max_steps=256, tolerance=1e-8).estimate(
                SurvivalProblem(design, duration, event, entry=entry)
            )
            return {
                "values": jnp.r_[
                    fit.parameters, fit.log_likelihood, fit.observed_information.ravel()
                ],
                "valid": fit.valid,
            }

        return jax.jit(execute), tuple(map(jax.device_put, inputs))
    if side != "reference":
        raise ValueError(f"Unknown reference side: {side}")
    from scipy.optimize import minimize

    def execute(design, duration, event, entry):
        fit = minimize(
            lambda theta: aft_objective(
                theta, design, duration, event, entry, family=family, information=False
            )[:2],
            np.zeros(design.shape[1] + 1),
            method="BFGS",
            jac=True,
            options={"gtol": 1e-8, "maxiter": 256},
        )
        value, gradient, information = aft_objective(
            fit.x, design, duration, event, entry, family=family
        )
        return {
            "values": np.r_[fit.x, -value, information.ravel()],
            "valid": np.max(np.abs(gradient)) <= 1e-5
            and np.linalg.eigvalsh(information).min() > 0,
        }

    return execute, inputs
