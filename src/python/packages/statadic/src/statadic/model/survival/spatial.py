"""Gaussian endogenous spatial lag of log spell durations."""

from dataclasses import dataclass, field

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LaggedLogTime:
    """Spatial log-time model with explicitly selected contextual design columns.

    Columns selected in ``slx_columns`` add ``W X`` predictors. They do not
    change the endogenous spatial lag of latent log duration.
    """

    slx_columns: tuple[int, ...] = field(default=(), metadata={"static": True})


__all__ = [
    "LaggedLogTime",
]
