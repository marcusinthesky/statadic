"""Population-risk accounting under matched and misspecified shrinkage targets."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.covariance import (
    ledoit_wolf_sample_kernel,
    sample_covariance_kernel,
)
from scipy import stats

from .._validation.calibration import (
    DEFAULT_TRIALS,
    TrialOutcome,
    summarize_trials,
    trial_rng,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.slow, pytest.mark.qualification]


@pytest.mark.integration
@pytest.mark.statistical
@pytest.mark.parametrize(
    "matched", [False, True], ids=["misspecified-target", "matched-target"]
)
def test_shrinkage_population_loss_distribution(  # noqa: PLR0914
    *, matched: bool, record_property: Callable[[str, object], None]
) -> None:
    """Report paired loss uncertainty without assuming every target reduces risk."""
    n = 64
    scenario = 401 if matched else 402
    marginal = np.array([0.5, 1.0, 1.5, 2.0])
    if matched:
        correlation = 0.3 * np.ones((4, 4)) + 0.7 * np.eye(4)
    else:
        correlation = np.array(
            [
                [1.0, 0.7, 0.0, 0.0],
                [0.7, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.5],
                [0.0, 0.0, -0.5, 1.0],
            ]
        )
    truth = correlation * np.outer(marginal, marginal)
    root = np.linalg.cholesky(truth)
    panels = np.stack(
        [
            trial_rng(scenario, r).normal(size=(n, 4)) @ root.T + np.arange(4)
            for r in range(DEFAULT_TRIALS)
        ]
    )

    # Act: each dataset estimates both its sample covariance and shrinkage.
    def evaluate_sample_covariance_kernel(x: jax.Array) -> jax.Array:
        return sample_covariance_kernel(x, ddof=0)

    sample = np.asarray(
        jax.jit(jax.vmap(evaluate_sample_covariance_kernel))(jnp.asarray(panels))
    )
    shrunk = jax.jit(jax.vmap(ledoit_wolf_sample_kernel))(jnp.asarray(panels))
    sigma, intensity = np.asarray(shrunk.sigma), np.asarray(shrunk.shrinkage)

    valid = (
        np.isfinite(sample).all(axis=(1, 2))
        & np.isfinite(sigma).all(axis=(1, 2))
        & np.isfinite(intensity)
    )
    report = summarize_trials([TrialOutcome(valid=bool(value)) for value in valid])
    record_property("failure_report", json.dumps(asdict(report), sort_keys=True))
    failure_budget = 0.005
    assert report.failure_upper <= failure_budget, report
    assert valid.all(), "Loss summaries require explicit unavailable-trial handling."
    sample_loss = np.sum((sample - truth) ** 2, axis=(1, 2))
    shrinkage_loss = np.sum((sigma - truth) ** 2, axis=(1, 2))
    difference = sample_loss - shrinkage_loss
    # For demeaned Gaussian observations, n*S has Wishart(n-1, Sigma) law.
    frobenius_squared = np.sum(truth**2)
    expected_sample_loss = (
        (n - 1) * (frobenius_squared + np.trace(truth) ** 2) + frobenius_squared
    ) / n**2
    quantile = stats.t.ppf(0.9995, DEFAULT_TRIALS - 1)
    sample_halfwidth = quantile * stats.sem(sample_loss)
    improvement_halfwidth = quantile * stats.sem(difference)
    record_property("population_sample_risk", expected_sample_loss)
    record_property("mean_sample_loss", float(sample_loss.mean()))
    record_property("mean_shrinkage_loss", float(shrinkage_loss.mean()))
    record_property("paired_mean_improvement", float(difference.mean()))
    record_property(
        "paired_improvement_ci999",
        json.dumps(
            [
                float(difference.mean() - improvement_halfwidth),
                float(difference.mean() + improvement_halfwidth),
            ]
        ),
    )
    assert abs(sample_loss.mean() - expected_sample_loss) <= sample_halfwidth
