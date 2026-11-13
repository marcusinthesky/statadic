"""Independent finite-count likelihood and saturated-limit GLM witnesses."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from statadic.model.glm import glm_deviance, glm_log_likelihood, glm_mean

from .._budgets import examples
from .._validation.glm import count_data

if TYPE_CHECKING:
    from hypothesis import strategies as st

N_ROWS = 4
MAX_TRIALS = 8
BINOMIAL_EXAMPLE = (
    "binomial_logit",
    (-2.0, -0.5, 0.25, 2.0),
    (3.0, 5.0, 4.0, 6.0),
    (0.0, 2.0, 4.0, 1.0),
)
POISSON_EXAMPLE = (
    "poisson_log",
    (-2.0, -0.5, 0.25, 2.0),
    (3.0, 5.0, 4.0, 6.0),
    (0.0, 1.0, 4.0, 2.0),
)


def _count_data() -> st.SearchStrategy[
    tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]
]:
    """Use shared count cases with this numerical witness's fixed shape."""
    return count_data(n=N_ROWS, max_trials=MAX_TRIALS)


@pytest.mark.unit
@pytest.mark.reference
@examples(20)
@given(data=_count_data())
@example(data=BINOMIAL_EXAMPLE)
@example(data=POISSON_EXAMPLE)
def test_count_family_means_full_likelihood_and_zero_count_deviance(
    data: tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]],
) -> None:
    """Check factorial constants, grouped counts and exact zero-count limits.

    Args:
        data: Family, linear predictors, trial counts and admissible counts.

    """
    family, eta_values, trial_values, outcome_values = data
    eta = np.asarray(eta_values)
    trials = np.asarray(trial_values)
    outcome = np.asarray(outcome_values)
    if family == "binomial_logit":
        probability = 1 / (1 + np.exp(-eta))
        mean = trials * probability
        likelihood = 0.0
        saturated = 0.0
        for y, m, probability_i in zip(outcome, trials, probability, strict=True):
            likelihood += (
                math.log(math.comb(int(m), int(y)))
                + y * math.log(probability_i)
                + (m - y) * math.log1p(-probability_i)
            )
            saturated += math.log(math.comb(int(m), int(y)))
            for count in (y, m - y):
                if count > 0:
                    saturated += count * math.log(count / m)
    else:
        mean = np.exp(eta)
        likelihood = sum(
            y * log_mean - expected - math.lgamma(y + 1)
            for y, log_mean, expected in zip(outcome, eta, mean, strict=True)
        )
        saturated = sum(
            (y * math.log(y) if y else 0) - y - math.lgamma(y + 1) for y in outcome
        )

    if family == "binomial_logit":
        actual_mean = glm_mean(jnp.asarray(eta), jnp.asarray(trials), "binomial_logit")
        actual_likelihood = glm_log_likelihood(
            jnp.asarray(outcome),
            jnp.asarray(eta),
            jnp.asarray(trials),
            "binomial_logit",
        )
        actual_deviance = glm_deviance(
            jnp.asarray(outcome),
            jnp.asarray(eta),
            jnp.asarray(trials),
            "binomial_logit",
        )
    else:
        actual_mean = glm_mean(jnp.asarray(eta), jnp.asarray(trials), "poisson_log")
        actual_likelihood = glm_log_likelihood(
            jnp.asarray(outcome), jnp.asarray(eta), jnp.asarray(trials), "poisson_log"
        )
        actual_deviance = glm_deviance(
            jnp.asarray(outcome), jnp.asarray(eta), jnp.asarray(trials), "poisson_log"
        )

    np.testing.assert_allclose(actual_mean, mean, rtol=1e-12)
    np.testing.assert_allclose(actual_likelihood, likelihood, rtol=1e-12)
    np.testing.assert_allclose(
        actual_deviance, 2 * (saturated - likelihood), rtol=1e-12
    )
