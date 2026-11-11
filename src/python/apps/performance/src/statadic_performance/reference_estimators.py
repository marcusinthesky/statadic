"""Matched regression, covariance and affine-GMM numerical consumers.

Peer branches use independent SciPy/statsmodels calculations and never import
STATADIC or JAX. These are explicit workload subcases, not family qualification.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def dispersion_objective(parameters, design, outcome, *, family):
    """Independent Gamma/NB2 negative likelihood and analytic log-scale score."""
    from scipy.special import digamma, gammaln

    eta = design @ parameters[:-1]
    mean = np.exp(eta)
    size = np.exp(-parameters[-1])
    if family == "gamma":
        rows = (
            size * np.log(size)
            - gammaln(size)
            + (size - 1) * np.log(outcome)
            - size * eta
            - size * outcome / mean
        )
        score_eta = size * (outcome / mean - 1)
        score_size = (
            np.log(size) + 1 - digamma(size) + np.log(outcome) - eta - outcome / mean
        )
    elif family == "nb2":
        rows = (
            gammaln(outcome + size)
            - gammaln(size)
            - gammaln(outcome + 1)
            + size * np.log(size)
            + outcome * eta
            - (outcome + size) * np.log(size + mean)
        )
        score_eta = size * (outcome - mean) / (size + mean)
        score_size = (
            digamma(outcome + size)
            - digamma(size)
            + np.log(size)
            + 1
            - np.log(size + mean)
            - (outcome + size) / (size + mean)
        )
    else:
        raise ValueError(f"Unknown dispersion family: {family}")
    return -rows.sum(), np.r_[-design.T @ score_eta, size * score_size.sum()]


def physical_information(parameters, design, outcome, *, family):
    """Differentiate the independent analytic score in physical dispersion."""

    def gradient(value):
        transformed = np.r_[value[:-1], np.log(value[-1])]
        result = dispersion_objective(transformed, design, outcome, family=family)[1]
        return np.r_[result[:-1], result[-1] / value[-1]]

    steps = 1e-5 * np.maximum(1.0, np.abs(parameters))
    directions = np.diag(steps)
    result = np.column_stack(
        [
            (gradient(parameters + direction) - gradient(parameters - direction))
            / (2 * step)
            for direction, step in zip(directions, steps, strict=True)
        ]
    )
    return (result + result.T) / 2


def constant_correlation_shrinkage(observations):
    """Independent constant-correlation Ledoit-Wolf moment calculation.

    This target differs from sklearn's identity-target LedoitWolf. Iterating
    observations independently avoids importing or calling the STATADIC formula.
    """
    centered = observations - observations.mean(axis=0)
    n, p = centered.shape
    sample = np.cov(observations, rowvar=False, ddof=0)
    variance = sample.diagonal()
    standard = np.sqrt(variance)
    mask = ~np.eye(p, dtype=bool)
    average = (sample / np.outer(standard, standard))[mask].mean()
    target = average * np.outer(standard, standard)
    np.fill_diagonal(target, variance)
    pi = np.zeros_like(sample)
    theta = np.zeros_like(sample)
    for row in centered:
        difference = np.outer(row, row) - sample
        pi += difference**2 / n
        theta += (row**2 - variance)[:, None] * difference / n
    rho = (
        np.trace(pi)
        + average * ((standard[None, :] / standard[:, None]) * theta)[mask].sum()
    )
    gamma = np.square(target - sample).sum()
    shrinkage = np.clip((pi.sum() - rho) / (n * gamma), 0, 1) if gamma > 0 else 0.0
    covariance = shrinkage * target + (1 - shrinkage) * sample
    return sample, covariance, target, shrinkage


def affine_gmm_peer(moments, derivative, weight, *, centered, updates):
    """SVD weighted solve and Bartlett covariance, retaining the last actual W."""
    from scipy.linalg import lstsq

    n = len(moments)
    mean_derivative = derivative.mean(axis=0)
    mean_moments = moments.mean(axis=0)

    def variability(rows):
        working = rows - rows.mean(axis=0) if centered else rows
        result = working.T @ working / n
        for lag in (1, 2):
            product = working[lag:].T @ working[:-lag] / n
            result += (1 - lag / 3) * (product + product.T)
        return result

    for iteration in range(updates + 1):
        root = np.linalg.cholesky(weight).T
        parameters = lstsq(root @ mean_derivative, -root @ mean_moments)[0]
        rows = moments + np.einsum("nmp,p->nm", derivative, parameters)
        omega = variability(rows)
        if iteration < updates:
            weight = np.linalg.inv(omega)
    normal = mean_derivative.T @ weight @ mean_derivative
    moment_map = np.linalg.solve(normal, mean_derivative.T @ weight)
    covariance = moment_map @ omega @ moment_map.T / n
    average = rows.mean(axis=0)
    return np.r_[
        parameters, weight.ravel(), covariance.ravel(), average @ weight @ average
    ]


def prepare_estimator(case: str, side: str, inputs: tuple) -> tuple[Callable, tuple]:
    """Prepare equal lifecycle outputs; fitting/inference stay in measured calls."""
    if side == "statadic":
        import jax
        import jax.numpy as jnp

        if case in {"binomial-logit", "poisson-offset"}:
            from statadic.estimation.glm import (
                BinomialLogitMLE,
                BinomialProblem,
                PoissonMLE,
                PoissonProblem,
            )
            from statadic.inference.glm_covariance import glm_covariance

            def execute(design, outcome, offset, role):
                fit = (
                    BinomialLogitMLE(tol=1e-9).estimate(
                        BinomialProblem(design, outcome, trials=role, offset=offset)
                    )
                    if case == "binomial-logit"
                    else PoissonMLE(tol=1e-9).estimate(
                        PoissonProblem(design, outcome, offset=offset, exposure=role)
                    )
                )
                covariance = glm_covariance(fit)
                return {
                    "values": jnp.r_[
                        fit.parameters,
                        covariance.covariance.values.ravel(),
                        fit.log_likelihood,
                        fit.fitted_means,
                    ],
                    "valid": fit.valid & covariance.valid,
                }

        elif case in {"gamma-estimated", "nb2-estimated"}:
            from statadic.estimation.likelihood import (
                EstimatedDispersion,
                GammaMLE,
                LikelihoodProblem,
                NegativeBinomialMLE,
            )

            estimator = (
                GammaMLE(EstimatedDispersion(), tolerance=1e-8)
                if case == "gamma-estimated"
                else NegativeBinomialMLE(EstimatedDispersion(), tolerance=1e-8)
            )

            def execute(design, outcome):
                fit = estimator.estimate(LikelihoodProblem(design, outcome))
                return {
                    "values": jnp.r_[
                        fit.parameters, fit.log_likelihood, fit.hessian.ravel()
                    ],
                    "valid": fit.valid,
                }

        elif case == "constant-correlation-shrinkage":
            from statadic.estimation.covariance import (
                ledoit_wolf_sample_kernel,
                sample_covariance_kernel,
            )

            def execute(observations):
                result = ledoit_wolf_sample_kernel(observations)
                return {
                    "values": jnp.r_[
                        sample_covariance_kernel(observations, ddof=0).ravel(),
                        result.sigma.ravel(),
                        result.target.ravel(),
                        result.shrinkage,
                    ],
                    "valid": jnp.all(jnp.isfinite(result.sigma)),
                }

        elif case in {"affine-gmm-actual-weight", "affine-gmm-iterated-hac"}:
            from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
            from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
            from statadic.estimation.moments import AffineMomentProblem
            from statadic.inference.gmm import gmm_covariance

            def execute(moments, derivative, weight):
                fit = (
                    FixedWeightGMM(variability=PlainLongRun, bandwidth=2).estimate(
                        AffineMomentProblem(moments, derivative, weight)
                    )
                    if case == "affine-gmm-actual-weight"
                    else IteratedGMM(
                        variability=MeanCenteredLongRun,
                        bandwidth=2,
                        max_iter=3,
                        tolerance=1e-30,
                    ).estimate(AffineMomentProblem(moments, derivative))
                )
                covariance = gmm_covariance(fit)
                return {
                    "values": jnp.r_[
                        fit.parameters,
                        fit.weight.ravel(),
                        covariance.covariance.values.ravel(),
                        fit.criterion,
                    ],
                    "valid": fit.valid & covariance.valid,
                }

        else:
            raise ValueError(f"Unknown estimator case: {case}")
        return jax.jit(execute), tuple(map(jax.device_put, inputs))
    if side != "reference":
        raise ValueError(f"Unknown benchmark side: {side}")
    if case in {"binomial-logit", "poisson-offset"}:
        import statsmodels.api as sm

        def execute(design, outcome, offset, role):
            if case == "binomial-logit":
                fit = sm.GLM(
                    np.column_stack([outcome, role - outcome]),
                    design,
                    family=sm.families.Binomial(),
                    offset=offset,
                ).fit(tol=1e-11, maxiter=256)
                means = fit.fittedvalues * role
            else:
                fit = sm.GLM(
                    outcome,
                    design,
                    family=sm.families.Poisson(),
                    offset=offset,
                    exposure=role,
                ).fit(tol=1e-11, maxiter=256)
                means = fit.fittedvalues
            return {
                "values": np.r_[fit.params, fit.cov_params().ravel(), fit.llf, means],
                "valid": bool(fit.converged),
            }

    elif case in {"gamma-estimated", "nb2-estimated"}:
        from scipy.optimize import minimize

        family = "gamma" if case == "gamma-estimated" else "nb2"

        def execute(design, outcome):
            fit = minimize(
                lambda theta: dispersion_objective(
                    theta, design, outcome, family=family
                ),
                np.r_[np.zeros(design.shape[1]), np.log(0.5)],
                jac=True,
                method="BFGS",
                options={"gtol": 1e-8, "maxiter": 256},
            )
            parameters = np.r_[fit.x[:-1], np.exp(fit.x[-1])]
            information = physical_information(
                parameters, design, outcome, family=family
            )
            return {
                "values": np.r_[parameters, -fit.fun, information.ravel()],
                "valid": np.max(np.abs(fit.jac)) <= 1e-5
                and np.linalg.eigvalsh(information).min() > 0,
            }

    elif case == "constant-correlation-shrinkage":

        def execute(observations):
            sample, covariance, target, shrinkage = constant_correlation_shrinkage(
                observations
            )
            return {
                "values": np.r_[
                    sample.ravel(), covariance.ravel(), target.ravel(), shrinkage
                ],
                "valid": np.isfinite(covariance).all(),
            }

    elif case in {"affine-gmm-actual-weight", "affine-gmm-iterated-hac"}:

        def execute(moments, derivative, weight):
            values = affine_gmm_peer(
                moments,
                derivative,
                weight,
                centered=case == "affine-gmm-iterated-hac",
                updates=3 if case == "affine-gmm-iterated-hac" else 0,
            )
            return {"values": values, "valid": np.isfinite(values).all()}

    else:
        raise ValueError(f"Unknown estimator case: {case}")
    return execute, inputs
