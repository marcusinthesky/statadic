"""Native supplied-kappa or covariance-target numerical studies."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from statadic.estimation.covariance.cross_validation import cv_lambda_grid
from statadic.estimation.covariance.estimators import sample_covariance
from statadic.estimation.covariance.shrinkage import (
    optimal_lambda_dist,
    plug_in_lambda_dist,
    sample_cov_entry_variance,
    shrink_to_dist,
)
from statadic.optimize.psd import ridge_psd

from statadic_studies.adapters._random import rng

SIGMA = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
METHODS = ("sample", "fixed-target-optimal", "fixed-target-CV", "same-window-plugin")


def _record(estimate, truth, **extra):
    estimate, truth = float(estimate), float(truth)
    extra = {name: float(value) for name, value in extra.items()}
    if not np.isfinite([estimate, truth, *extra.values()]).all():
        return {"valid": False, "failure": "nonfinite-result"}
    return {"valid": True, "estimate": estimate, "truth": truth, **extra}


def _entry(scenario, replicate, seed):
    generator = rng("covariance.entry-variance", scenario, replicate, seed=seed)
    n = int(scenario[1:])
    panel = generator.multivariate_normal(np.zeros(4), SIGMA, size=n)
    result = sample_cov_entry_variance(panel)
    return {"fourth-moment-entry-variance": _record(result[0, 0], 2 * (n - 1) / n**2)}


def covariance_rules(scenario, replicate, seed):
    """Use shared data and exact native fixed-target and out-of-fold selectors."""
    study = "covariance.target-tuning"
    generator = rng(study, scenario, replicate, seed=seed)
    panel = generator.multivariate_normal(np.zeros(4), SIGMA, size=64)
    fixed = np.eye(4) * (1.0 if scenario == "fixed-identity" else 1.5)
    sample = np.asarray(sample_covariance(panel, ddof=0))
    same_window = np.diag(np.diag(sample))
    fixed_lambda, _, _ = optimal_lambda_dist(panel, fixed, assume_target_fixed=True)
    plugin_lambda, _, _ = plug_in_lambda_dist(panel, same_window)
    fold_seed = int(
        rng(study, scenario, replicate, seed=seed, stream="folds").integers(
            0, 2**31 - 1
        )
    )
    cv_lambda, _, _ = cv_lambda_grid(
        panel,
        fixed,
        grid=[0.0, 0.5, 1.0],
        n_folds=4,
        seed=fold_seed,
        assume_target_fixed=True,
    )
    matrices = (
        sample,
        np.asarray(shrink_to_dist(panel, fixed, fixed_lambda).sigma),
        np.asarray(shrink_to_dist(panel, fixed, cv_lambda).sigma),
        np.asarray(shrink_to_dist(panel, same_window, plugin_lambda).sigma),
    )
    return dict(zip(METHODS, matrices, strict=True))


def _covariance(scenario, replicate, seed):
    outputs = {}
    for method, matrix in covariance_rules(scenario, replicate, seed).items():
        loss = float(np.sum((matrix - SIGMA) ** 2))
        outputs[method] = _record(
            np.trace(matrix), np.trace(SIGMA), frobenius_loss=loss
        )
    return outputs


def _portfolio(scenario, replicate, seed):
    ones = np.ones(4)
    truth = 1 / (ones @ np.linalg.solve(SIGMA, ones))
    outputs = {}
    for method, matrix in covariance_rules(scenario, replicate, seed).items():
        regularized = np.asarray(ridge_psd(jnp.asarray(matrix), eps=1e-8))
        try:
            inverse_ones = np.linalg.solve(regularized, ones)
            weights = inverse_ones / np.sum(inverse_ones)
            value = weights @ SIGMA @ weights
            outputs[method] = _record(
                value, truth, loss_value=max(float(value - truth), 0.0)
            )
        except np.linalg.LinAlgError:
            outputs[method] = {"valid": False, "failure": "portfolio-solve-unavailable"}
    return outputs
