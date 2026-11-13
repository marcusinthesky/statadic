# jaxtyping dimensions are runtime contracts.
"""Shared inference accepts a provider defined outside its concrete families."""

from __future__ import annotations

from typing import TYPE_CHECKING

from statadic.inference.parameters import coefficient_intervals

if TYPE_CHECKING:
    from statadic.inference.contracts import FittedUncertainty
    from statadic.inference.parameters import CoefficientIntervals

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from statadic.core.equations import (
    IndependentRows,  # noqa: TC002
    ParameterCovariance,  # noqa: TC002
)
from statadic.core.typing import Array  # noqa: TC002
from statadic.estimation.linear import (
    LeastSquares,
    LinearEquation,  # noqa: TC002
    LinearFit,  # noqa: TC002
    LinearProblem,
)
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.parameters import linear_restriction_test

from .._budgets import examples
from .._strategies import float_arrays


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExternalUncertainty:
    """An independent uncertainty provider implementing structural capabilities."""

    fit: LinearFit
    covariance: ParameterCovariance[LinearEquation, IndependentRows]
    valid: Array
    status: Array

    @property
    def sample_capacity(self) -> int:
        """Return the retained row capacity of this provider's fit."""
        return self.fit.outcome.shape[-1]


_X = jnp.column_stack((jnp.ones(9), jnp.arange(9.0)))
_ORIGINAL_Y = 0.4 + 0.7 * _X[:, 1] + 0.2 * jnp.sin(_X[:, 1])


def evaluate_coefficient_intervals(value: FittedUncertainty) -> CoefficientIntervals:
    """Return normal-reference intervals for any provider."""
    return coefficient_intervals(value, reference="normal")


def evaluate_restriction(provider: ExternalUncertainty, scale: Array) -> Array:
    """Test the slope restriction on a provider whose validity depends on ``scale``."""
    gated = ExternalUncertainty(
        provider.fit,
        provider.covariance,
        provider.valid & (scale > 0),
        provider.status,
    )
    return linear_restriction_test(
        gated, jnp.asarray([[0.0, 1.0]]), 0.0, reference="chi2"
    ).test.statistic


@pytest.mark.integration
@examples(20)
@example(y=_ORIGINAL_Y)
@given(y=float_arrays(9))
def test_external_uncertainty_composes_under_jit_and_vmap(y: Array) -> None:
    """A new provider needs no central type union or family registration.

    Args:
        y: Generated outcomes for a fixed design.

    """
    ordinary = classical_covariance(LeastSquares().estimate(LinearProblem(_X, y)))
    external = ExternalUncertainty(
        ordinary.fit, ordinary.covariance, ordinary.valid, ordinary.status
    )
    expected = jax.jit(evaluate_coefficient_intervals)(ordinary)
    actual = jax.jit(evaluate_coefficient_intervals)(external)
    np.testing.assert_array_equal(actual.estimate.ci, expected.estimate.ci)
    assert actual.estimate.n_obs == len(y)

    result = jax.jit(jax.vmap(evaluate_restriction, in_axes=(None, 0)))(
        external, jnp.asarray([1.0, 2.0])
    )
    np.testing.assert_array_equal(result[0], result[1])
