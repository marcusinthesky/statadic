"""Three public extension interfaces a downstream package can implement."""

# ruff: noqa: INP001  # Standalone executable documentation.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import jax.numpy as jnp
from statadic.api.design import Term
from statadic.api.representation import PreparedGeometry
from statadic.estimation.linear import LeastSquares

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import Array
    from statadic.api.data import Observations
    from statadic.api.representation import RowId
    from statadic.api.roles import InputRole
    from statadic.estimation.linear import LinearFit, LinearProblem
    from statadic.sample.packed import PackedClouds


@dataclass(frozen=True)
class FittedCentered:
    """Retain the training mean for all subsequent transformations."""

    column: str
    mean: Array

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Declare the prediction input."""
        return {self.column: "numeric"}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Name the coefficient in the learned design."""
        return (f"centered({self.column})",)

    def transform(self, data: Observations) -> Array:
        """Encode each new row independently of the other prediction rows."""
        return (data.numeric(self.column) - self.mean)[:, None]


@dataclass(frozen=True)
class Centered(Term):
    """Center a numeric column around its training sample mean."""

    column: str

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare the training input."""
        return {self.column: "numeric"}

    @override
    def _prepare_term(self, training: Observations) -> FittedCentered:
        """Learn the mean once on the aligned training rows."""
        return FittedCentered(self.column, jnp.mean(training.numeric(self.column)))


@dataclass(frozen=True)
class MeanDistance:
    """L1 distance between the mass-weighted means of two empirical measures."""

    def prepare(
        self, measures: PackedClouds, *, row_ids: Sequence[RowId] | None = None
    ) -> PreparedGeometry:
        """Retain the reference collection and its column order."""
        keys = range(measures.n_observations) if row_ids is None else row_ids
        return PreparedGeometry(self, measures, tuple(keys))

    @staticmethod
    def cross(left: PackedClouds, right: PackedClouds) -> Array:
        """Return the left-by-right distances between weighted means."""
        left_means = jnp.sum(left.coordinates * left.masses[..., None], axis=-2)
        right_means = jnp.sum(right.coordinates * right.masses[..., None], axis=-2)
        return jnp.abs(left_means[:, None, :] - right_means[None, :, :]).sum(axis=-1)

    @staticmethod
    def matrix(measures: PackedClouds) -> None:
        """Offer pairwise values without asserting a Hilbertian construction."""
        del measures


@dataclass(frozen=True)
class OverdeterminedLeastSquares:
    """Use statadic least squares only for designs with more rows than columns."""

    @staticmethod
    def estimate(problem: LinearProblem) -> LinearFit:
        """Enforce the downstream policy while preserving the native fit state."""
        n_rows, n_columns = jnp.shape(problem.design)[-2:]
        if n_rows <= n_columns:
            message = "This estimator requires more rows than design columns."
            raise ValueError(message)
        return LeastSquares().estimate(problem)
