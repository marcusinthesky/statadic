# ruff: noqa: TC001  # Runtime annotations are inspected by the import hook.
"""Schema checks for fits returned by injected regression estimators."""

from __future__ import annotations

import jax.numpy as jnp

from statadic.api.base import PreparedRegression
from statadic.inference.contracts import ParameterFit


def check_parameter_schema(fit: ParameterFit, names: tuple[str, ...]) -> None:
    """Require an injected estimator's fit to retain the prepared coefficient order.

    Reports, live prediction and persisted predictors pair the coefficients with
    the prepared design, so a custom estimator must return exactly its names.
    """
    if fit.parameter_names != names or tuple(fit.parameters.shape[-1:]) != (
        len(names),
    ):
        message = (
            "Estimator returned a different parameter schema: expected "
            f"{names}, got {fit.parameter_names}."
        )
        raise ValueError(message)


def check_binomial_observations(
    prepared: PreparedRegression, trials: str | None
) -> None:
    """Validate observed support before entering a binomial numerical solver."""
    rows = prepared.context.observations
    successes = rows.numeric(prepared.context.response)
    totals = 1 if trials is None else rows.numeric(trials)
    if not bool(jnp.all((totals > 0) & (totals == jnp.floor(totals)))):
        message = "Binomial trials must be positive integers."
        raise ValueError(message)
    if not bool(
        jnp.all(
            (successes >= 0)
            & (successes <= totals)
            & (successes == jnp.floor(successes))
        )
    ):
        message = "Binomial successes must be integer counts from zero through trials."
        raise ValueError(message)
