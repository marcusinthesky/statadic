"""Larger deterministic adaptations of two branch reference fixtures, in CPU float64."""

import numpy as np


CASES = {
    "sar160": dict(
        method="SAR ML (larger workload)",
        reference="spreg.ML_Lag",
        fixture="160 nodes, 2 covariates, seed 126",
        work="Fit, log likelihood, coefficients and covariance",
        source="src/python/packages/statadic/tests/estimation/test_spatial_reference.py#L41-L98",
        note="Increased node count from 40 to 160. Identical directed ring design and full dense method.",
        rtol=1e-5,
        atol=2e-6,
    ),
    "weibull2400": dict(
        method="Weibull AFT (larger workload)",
        reference="lifelines.WeibullAFTFitter",
        fixture="2,400 right-censored spells, no delayed entry, seed 302601",
        work="Fit, log likelihood, coefficients and observed-information covariance",
        source="src/python/packages/statadic/tests/testing/test_survival_references.py#L39-L119",
        note="Increased spell count from 240 to 2,400. Formula parsing remains in the lifelines fit.",
        rtol=5e-4,
        atol=8e-5,
    ),
}


def _pack(parameters, likelihood, covariance):
    return np.concatenate(
        (
            np.asarray(parameters).ravel(),
            [float(likelihood)],
            np.asarray(covariance).ravel(),
        )
    )


def _sar(side):
    rng = np.random.default_rng(126)
    n = 160
    eye = np.eye(n)
    w = 0.8 * np.roll(eye, 1, axis=1) + 0.2 * np.roll(eye, -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = np.linalg.solve(eye - 0.35 * w, x @ [0.4, -0.5, 0.8] + rng.normal(size=n) * 0.4)

    if side == "statadic":
        from statadic.estimation.spatial import ML, SAR, SpatialProblem
        from statadic.inference.spatial import spatial_covariance
        from statadic.operators.spatial import prepare_spatial_weights

        problem = SpatialProblem(x, y, prepare_spatial_weights(w))
        estimator = ML(SAR())

        def evaluate():
            fit = estimator.estimate(problem)
            cov = spatial_covariance(fit)
            return _pack(fit.parameters, fit.loglik, cov.covariance.values)

        return evaluate

    from spreg import ML_Lag
    from libpysal.weights import full2W

    graph = full2W(w)

    def evaluate():
        fit = ML_Lag(
            y[:, None], x[:, 1:], graph, method="full", slx_lags=0, epsilon=1e-9
        )
        return _pack(fit.betas, fit.logll, fit.vm)

    return evaluate


def _weibull(side):
    rng = np.random.default_rng(302601)
    x = rng.normal(size=2_400)
    mu = 0.2 - 0.3 * x
    entry = np.zeros(x.size)
    u = rng.uniform(size=x.size)
    event_time = np.exp(mu) * ((entry / np.exp(mu)) ** 1.5 - np.log(u)) ** (1 / 1.5)
    censor_time = entry + rng.exponential(2.0, size=x.size)
    t = np.minimum(event_time, censor_time)
    event = event_time <= censor_time

    if side == "statadic":
        import jax.numpy as jnp
        from statadic.estimation.survival import SurvivalML, SurvivalProblem
        from statadic.inference.survival import observed_information
        from statadic.model.survival import WeibullAFT

        estimator = SurvivalML(WeibullAFT())
        problem = SurvivalProblem(
            jnp.asarray(np.column_stack([np.ones(x.size), x])),
            jnp.asarray(t),
            jnp.asarray(event),
            jnp.asarray(entry),
            ("intercept", "x"),
        )

        def evaluate():
            fit = estimator.estimate(problem)
            uncertainty = observed_information(fit)
            if not bool(fit.valid) or not bool(uncertainty.valid):
                raise ValueError("Invalid AFT fit or covariance")
            return _pack(
                fit.parameters, fit.log_likelihood, uncertainty.covariance.values
            )

        return evaluate

    import pandas as pd
    from lifelines import WeibullAFTFitter

    frame = pd.DataFrame(dict(x=x, t=t, event=event, entry=entry))
    order = [("lambda_", "Intercept"), ("lambda_", "x"), ("rho_", "Intercept")]

    def evaluate():
        fit = WeibullAFTFitter().fit(
            frame, "t", "event", entry_col="entry", formula="x"
        )
        return _pack(
            fit.params_.loc[order],
            fit.log_likelihood_,
            fit.variance_matrix_.loc[order, order],
        )

    return evaluate


def prepare(case, side):
    if case not in CASES or side not in {"statadic", "reference"}:
        raise ValueError((case, side))
    return _sar(side) if case == "sar160" else _weibull(side)
