"""Independent Gaussian period-block panel fit and information references."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np


def gaussian_expected_information(design, weights, beta, effects, rho, variance):
    """General normal mean/covariance information, then eliminate unit effects."""
    periods, nodes, p = design.shape
    width = p + nodes + 2
    information = np.zeros((width, width))
    identity = np.eye(nodes)
    for x, w in zip(design, weights, strict=True):
        inverse = np.linalg.inv(identity - rho * w)
        mean = inverse @ (x @ beta + effects)
        covariance = variance * inverse @ inverse.T
        precision = np.linalg.inv(covariance)
        change = inverse @ w
        mean_derivative = np.column_stack(
            [
                inverse @ x,
                inverse,
                change @ mean,
                np.zeros(nodes),
            ]
        )
        information += mean_derivative.T @ precision @ mean_derivative
        covariance_derivatives = [
            change @ covariance + covariance @ change.T,
            covariance / variance,
        ]
        for i, left in enumerate(covariance_derivatives):
            for j, right in enumerate(covariance_derivatives):
                information[-2 + i, -2 + j] += 0.5 * np.trace(
                    precision @ left @ precision @ right
                )
    keep = [*range(p), width - 2, width - 1]
    nuisance = list(range(p, p + nodes))
    cross = information[np.ix_(keep, nuisance)]
    return information[np.ix_(keep, keep)] - cross @ np.linalg.solve(
        information[np.ix_(nuisance, nuisance)], cross.T
    )


def balanced_profile(rho, design, outcome, weights):
    """SAR filtering precedes within-unit profiling for the actual W_t."""
    periods, nodes, p = design.shape
    filtered = np.asarray(
        [(np.eye(nodes) - rho * w) @ y for w, y in zip(weights, outcome, strict=True)]
    )
    within_x = design - design.mean(axis=0)
    within_y = filtered - filtered.mean(axis=0)
    beta = np.linalg.lstsq(within_x.reshape(-1, p), within_y.ravel(), rcond=None)[0]
    effects = (filtered - design @ beta).mean(axis=0)
    residual = filtered - design @ beta - effects
    variance = np.square(residual).sum() / outcome.size
    jacobian = sum(np.linalg.slogdet(np.eye(nodes) - rho * w)[1] for w in weights)
    likelihood = jacobian - outcome.size / 2 * (np.log(2 * np.pi * variance) + 1)
    return beta, effects, variance, likelihood


def observed_blocks(design, outcome, weights):
    """Both implementations receive identical active graphs and missing rows."""
    nodes = design.shape[1]
    blocks = []
    for period, (x, y, w) in enumerate(zip(design, outcome, weights, strict=True)):
        active = np.delete(np.arange(nodes), period % nodes)
        observed = np.delete(np.arange(len(active)), (period + 2) % len(active))
        blocks.append(
            (
                x[active],
                y[active][observed],
                w[np.ix_(active, active)],
                tuple(map(int, active)),
                tuple(map(int, observed)),
            )
        )
    return tuple(blocks)


def observed_profile(rho, blocks):
    """Direct Gaussian marginalization and GLS, independent of native reductions."""
    designs, outcomes, determinant = [], [], 0.0
    for x, y, w, _, observed in blocks:
        inverse = np.linalg.inv(np.eye(len(w)) - rho * w)
        covariance = (inverse @ inverse.T)[np.ix_(observed, observed)]
        root = np.linalg.cholesky(covariance)
        designs.append(np.linalg.solve(root, (inverse @ x)[list(observed)]))
        outcomes.append(np.linalg.solve(root, y))
        determinant += 2 * np.log(root.diagonal()).sum()
    design, outcome = np.vstack(designs), np.concatenate(outcomes)
    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residual = outcome - design @ beta
    variance = residual @ residual / len(outcome)
    likelihood = -0.5 * (
        determinant + len(outcome) * (np.log(2 * np.pi * variance) + 1)
    )
    return beta, variance, likelihood


def observed_objective(parameters, blocks):
    """Unprofiled marginal negative likelihood in beta/rho/log-variance."""
    beta, rho, log_variance = parameters[:-2], parameters[-2], parameters[-1]
    value = 0.0
    for x, y, w, _, observed in blocks:
        inverse = np.linalg.inv(np.eye(len(w)) - rho * w)
        mean = (inverse @ x @ beta)[list(observed)]
        covariance = (
            np.exp(log_variance) * (inverse @ inverse.T)[np.ix_(observed, observed)]
        )
        difference = y - mean
        value += 0.5 * (
            len(y) * np.log(2 * np.pi)
            + np.linalg.slogdet(covariance)[1]
            + difference @ np.linalg.solve(covariance, difference)
        )
    return value


def observed_information(parameters, blocks):
    """Independent central likelihood differences and log-variance Schur complement."""
    steps = 1e-4 * np.maximum(1.0, abs(parameters))
    directions = np.diag(steps)
    information = np.asarray(
        [
            [
                (
                    observed_objective(parameters + left + right, blocks)
                    - observed_objective(parameters + left - right, blocks)
                    - observed_objective(parameters - left + right, blocks)
                    + observed_objective(parameters - left - right, blocks)
                )
                / (4 * a * b)
                for right, b in zip(directions, steps, strict=True)
            ]
            for left, a in zip(directions, steps, strict=True)
        ]
    )
    return (
        information[:-1, :-1]
        - np.outer(information[:-1, -1], information[-1, :-1]) / information[-1, -1]
    )


def prepare_panel(case: str, side: str, inputs: tuple) -> tuple[Callable, tuple]:
    """Prepare graph identity once; full fitting/information remains measured."""
    if side == "statadic":
        import jax
        import jax.numpy as jnp
        from statadic.estimation.spatial import SAR

        if case == "spatial-panel-sar-within":
            from statadic.estimation.spatial.panel import PanelML, SpatialPanelProblem
            from statadic.operators.spatial import prepare_spatial_panel_weights

            weights = prepare_spatial_panel_weights(
                inputs[2], period_ids=np.arange(len(inputs[0]))
            )

            def execute(design, outcome, weights):
                fit = PanelML(tolerance=1e-08).estimate(
                    SpatialPanelProblem(
                        design,
                        outcome,
                        weights,
                        process=SAR(),
                        slx_columns=(),
                        effects="unit",
                    )
                )
                return {
                    "values": jnp.r_[
                        fit.parameters,
                        fit.sigma2,
                        fit.log_likelihood,
                        fit.effects,
                        fit.information.ravel(),
                    ],
                    "valid": fit.valid & jnp.all(jnp.isfinite(fit.information)),
                }

            return jax.jit(execute), (
                jax.device_put(inputs[0]),
                jax.device_put(inputs[1]),
                weights,
            )
        if case != "observed-spatial-panel-sar":
            raise ValueError(f"Unknown panel case: {case}")
        from statadic.estimation.spatial.observed_panel import (
            ObservedPanelML,
            ObservedSpatialPanelProblem,
        )
        from statadic.operators.spatial import (
            ObservedSpatialPeriod,
            prepare_spatial_weights,
        )

        blocks = observed_blocks(*inputs)
        periods = tuple(
            ObservedSpatialPeriod(prepare_spatial_weights(w), active, observed)
            for _, _, w, active, observed in blocks
        )
        problem = ObservedSpatialPanelProblem(
            periods,
            tuple(jax.device_put(block[0]) for block in blocks),
            tuple(jax.device_put(block[1]) for block in blocks),
            n_units=inputs[0].shape[1],
            process=SAR(),
            effects="pooled",
        )

        def execute(problem):
            fit = ObservedPanelML(tolerance=1e-08).estimate(
                replace(problem, process=SAR(), slx_columns=(), effects="pooled")
            )
            return {
                "values": jnp.r_[
                    fit.parameters,
                    fit.sigma2,
                    fit.log_likelihood,
                    fit.information.ravel(),
                ],
                "valid": fit.valid & fit.information_valid,
            }

        return jax.jit(execute), (problem,)
    if side != "reference":
        raise ValueError(f"Unknown reference side: {side}")
    from scipy.optimize import minimize_scalar

    if case == "spatial-panel-sar-within":

        def execute(design, outcome, weights):
            fit = minimize_scalar(
                lambda rho: -balanced_profile(rho, design, outcome, weights)[-1],
                bounds=(-0.999, 0.999),
                method="bounded",
                options={"xatol": 1e-10},
            )
            beta, effects, variance, likelihood = balanced_profile(
                fit.x, design, outcome, weights
            )
            information = gaussian_expected_information(
                design, weights, beta, effects, fit.x, variance
            )
            return {
                "values": np.r_[
                    beta, fit.x, variance, likelihood, effects, information.ravel()
                ],
                "valid": bool(fit.success)
                and np.linalg.eigvalsh(information).min() > 0,
            }

        return execute, inputs
    if case != "observed-spatial-panel-sar":
        raise ValueError(f"Unknown panel case: {case}")
    blocks = observed_blocks(*inputs)

    def execute(blocks):
        fit = minimize_scalar(
            lambda rho: -observed_profile(rho, blocks)[-1],
            bounds=(-0.999, 0.999),
            method="bounded",
            options={"xatol": 1e-10},
        )
        beta, variance, likelihood = observed_profile(fit.x, blocks)
        information = observed_information(np.r_[beta, fit.x, np.log(variance)], blocks)
        return {
            "values": np.r_[beta, fit.x, variance, likelihood, information.ravel()],
            "valid": bool(fit.success) and np.linalg.eigvalsh(information).min() > 0,
        }

    return execute, (blocks,)
