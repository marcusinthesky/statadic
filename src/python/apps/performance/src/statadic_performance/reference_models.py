"""Bounded spatial and ordinary AFT pilot adapters with matched fitted outputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def lognormal_objective(parameters, design, duration, event, *, information=True):
    """Independent censored-normal likelihood, analytic score and Hessian.

    The last parameter is log standard deviation. The original-time density
    Jacobian is retained. There is no delayed entry in this pilot case.
    """
    from scipy.special import log_ndtr

    sigma = np.exp(parameters[-1])
    z = (np.log(duration) - design @ parameters[:-1]) / sigma
    log_density = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    log_survival = log_ndtr(-z)
    mills = np.exp(log_density - log_survival)
    rows = np.where(event, log_density - np.log(sigma) - np.log(duration), log_survival)
    location_score = np.where(event, z, mills)
    scale_score = np.where(event, z**2 - 1, mills * z)
    gradient = -np.r_[design.T @ location_score / sigma, scale_score.sum()]
    if not information:
        return -rows.sum(), gradient, None
    mills_prime = mills * (mills - z)
    location_weight = np.where(event, 1.0, mills_prime)
    cross_weight = np.where(event, 2 * z, mills_prime * z + mills)
    scale_weight = np.where(event, 2 * z**2, mills_prime * z**2 + mills * z)
    hessian = np.zeros((len(parameters), len(parameters)))
    hessian[:-1, :-1] = design.T @ (location_weight[:, None] * design) / sigma**2
    hessian[:-1, -1] = design.T @ cross_weight / sigma
    hessian[-1, :-1] = hessian[:-1, -1]
    hessian[-1, -1] = scale_weight.sum()
    return -rows.sum(), gradient, hessian


def prepare_model(
    case: str, side: str, inputs: tuple
) -> tuple[Callable[..., dict], tuple]:
    """Prepare graph representations equally outside the measured fit call."""
    if side == "statadic":
        import jax
        import jax.numpy as jnp

        if case == "lognormal-aft":
            from statadic.estimation.survival import SurvivalML, SurvivalProblem
            from statadic.model.survival import LogNormalAFT

            def execute(design, duration, event):
                fit = SurvivalML(
                    LogNormalAFT(), max_steps=256, tolerance=1e-8
                ).estimate(SurvivalProblem(design, duration, event))
                return {
                    "values": jnp.concatenate(
                        [
                            fit.parameters,
                            fit.log_likelihood[None],
                            fit.observed_information.ravel(),
                        ]
                    ),
                    "valid": fit.valid,
                }

            return jax.jit(execute), tuple(map(jax.device_put, inputs))
        from statadic.estimation.spatial import ML, SAR, SLXLeastSquares, SpatialProblem
        from statadic.inference.linear_covariance import classical_covariance
        from statadic.inference.spatial.covariance import spatial_covariance
        from statadic.operators.spatial import prepare_spatial_weights

        prepared = prepare_spatial_weights(inputs[2])
        if case == "slx-classical":

            def execute(design, outcome, weights):
                fit = SLXLeastSquares().estimate(
                    SpatialProblem(
                        design, outcome, weights, process=None, slx_columns=(1, 2)
                    )
                )
                uncertainty = classical_covariance(fit)
                return {
                    "values": jnp.concatenate(
                        [
                            fit.parameters,
                            uncertainty.covariance.values.ravel(),
                            fit.fitted_values,
                        ]
                    ),
                    "valid": fit.valid & uncertainty.valid,
                }

        elif case == "sar-gaussian-ml":

            def execute(design, outcome, weights):
                fit = ML(max_steps=256, tolerance=1e-08).estimate(
                    SpatialProblem(
                        design, outcome, weights, process=SAR(), slx_columns=()
                    )
                )
                uncertainty = spatial_covariance(fit)
                return {
                    "values": jnp.concatenate(
                        [
                            fit.parameters,
                            fit.sigma2[None],
                            fit.log_likelihood[None],
                            uncertainty.covariance.values.ravel(),
                            fit.fitted_values,
                            fit.reduced_form,
                        ]
                    ),
                    "valid": fit.valid & uncertainty.valid,
                }

        else:
            raise ValueError(f"Unknown spatial reference case: {case}")
        return jax.jit(execute), (
            jax.device_put(inputs[0]),
            jax.device_put(inputs[1]),
            prepared,
        )
    if side != "reference":
        raise ValueError(f"Unknown reference side: {side}")
    if case == "lognormal-aft":
        from scipy.optimize import minimize

        def execute(design, duration, event):
            fitted = minimize(
                lambda parameters: lognormal_objective(
                    parameters, design, duration, event, information=False
                )[:2],
                np.zeros(design.shape[1] + 1),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-8, "maxiter": 256},
            )
            value, gradient, information = lognormal_objective(
                fitted.x, design, duration, event
            )
            return {
                "values": np.r_[fitted.x, -value, information.ravel()],
                "valid": bool(fitted.success)
                and np.max(np.abs(gradient)) <= 1e-5
                and np.linalg.eigvalsh(information).min() > 0,
            }

        return execute, inputs
    if case == "slx-classical":
        import statsmodels.api as sm

        def execute(design, outcome, weights):
            augmented = np.column_stack([design, weights @ design[:, 1:]])
            fit = sm.OLS(outcome, augmented).fit()
            return {
                "values": np.r_[fit.params, fit.cov_params().ravel(), fit.fittedvalues],
                "valid": True,
            }

        return execute, inputs
    if case == "sar-gaussian-ml":
        import libpysal
        import spreg

        prepared = libpysal.weights.full2W(inputs[2])

        def execute(design, outcome, weights):
            fit = spreg.ML_Lag(
                outcome[:, None], design[:, 1:], weights, method="full", epsilon=1e-9
            )
            return {
                "values": np.r_[
                    fit.betas.ravel(),
                    np.asarray(fit.sig2).ravel(),
                    fit.logll,
                    fit.vm.ravel(),
                    fit.predy.ravel(),
                    fit.predy_e.ravel(),
                ],
                "valid": True,
            }

        return execute, (inputs[0], inputs[1], prepared)
    raise ValueError(f"Unknown model reference case: {case}")
