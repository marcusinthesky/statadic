"""Preview binomial-logit and Poisson-log families with fixed dispersion one.

Binomial outcomes are integer success counts with positive integer trials;
Poisson outcomes are nonnegative integer counts. Offsets enter the linear
predictor additively and Poisson exposure contributes its natural logarithm.
No intercept, row deletion, frequency weights or survey weights are implicit.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy

from statadic.core.equations import EquationDescriptor
from statadic.core.typing import Array  # noqa: TC001

__all__ = [
    "BinomialLogitEquation",
    "GLMEquation",
    "GLMFamily",
    "PoissonLogEquation",
    "glm_deviance",
    "glm_log_likelihood",
    "glm_mean",
    "poisson_prediction",
]

type GLMFamily = Literal["binomial_logit", "poisson_log"]


class GLMEquation(EquationDescriptor):
    """Canonical mean score equation, with actual family/sample retained in a fit."""


class BinomialLogitEquation(GLMEquation):
    """One contribution is ``x_i * (successes_i - trials_i * p_i)``."""


class PoissonLogEquation(GLMEquation):
    """One contribution is ``x_i * (count_i - exp(eta_i))``."""


def poisson_prediction(
    base: Array, exposure: Array, kind: Literal["mean", "link", "rate"]
) -> Array:
    """Predict counts, their link, or the rate before exposure is applied."""
    if kind == "rate":
        return jnp.exp(base)
    eta = base + jnp.log(exposure)
    return eta if kind == "link" else jnp.exp(eta)


def glm_mean(eta: Array, trials: Array, family: GLMFamily) -> Array:
    """Return expected successes/counts, preserving the grouped-binomial scale."""
    return trials * jax.nn.sigmoid(eta) if family == "binomial_logit" else jnp.exp(eta)


def glm_log_likelihood(
    outcome: Array, eta: Array, trials: Array, family: GLMFamily
) -> Array:
    """Return full log likelihood, including count and binomial coefficient terms."""
    if family == "binomial_logit":
        constants = (
            gammaln(trials + 1) - gammaln(outcome + 1) - gammaln(trials - outcome + 1)
        )
        terms = (
            constants
            - outcome * jax.nn.softplus(-eta)
            - (trials - outcome) * jax.nn.softplus(eta)
        )
    else:
        terms = outcome * eta - jnp.exp(eta) - gammaln(outcome + 1)
    return jnp.sum(terms, axis=-1)


def glm_deviance(outcome: Array, eta: Array, trials: Array, family: GLMFamily) -> Array:
    """Return twice saturated-minus-fitted log likelihood, with zero-count limits."""
    if family == "binomial_logit":
        saturated = xlogy(outcome, outcome / trials) + xlogy(
            trials - outcome, (trials - outcome) / trials
        )
        fitted = -outcome * jax.nn.softplus(-eta) - (
            trials - outcome
        ) * jax.nn.softplus(eta)
        terms = saturated - fitted
    else:
        terms = xlogy(outcome, outcome) - outcome - outcome * eta + jnp.exp(eta)
    return 2 * jnp.sum(terms, axis=-1)
