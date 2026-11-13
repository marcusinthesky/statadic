"""Joint supervised graph fitting on aligned Gaussian spatial regressions."""

from __future__ import annotations

import math
from collections.abc import Mapping  # noqa: TC003 -- native runtime contract
from dataclasses import dataclass, field
from types import MappingProxyType

import jax.numpy as jnp

from statadic.api.spatial.graph_regression import (  # noqa: TC001 -- runtime annotations
    GraphRegressionResult,
    PreparedGraphRegression,
)
from statadic.api.spatial.graphs import (  # noqa: TC001
    BandwidthRef,
    DistanceGraph,
    ProjectionRef,
)
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.spatial.graph_learning import _estimate_joint_graph_ml
from statadic.estimation.spatial.graph_likelihood import estimate_graph_ml

__all__ = ["GraphML", "LearnBandwidth", "LearnLinear"]


@dataclass(frozen=True)
class LearnLinear:
    """Learn a d-by-r map, with optional fixed Frobenius norm and explicit start."""

    rank: int
    initial: Array | None = None
    norm: int | float | None = None

    def __post_init__(self) -> None:
        """Validate configuration before tracing."""
        if type(self.rank) is not int or self.rank <= 0:
            msg = "Projection rank must be a positive integer."
            raise ValueError(msg)
        if self.norm is not None and (not math.isfinite(self.norm) or self.norm <= 0):
            msg = "A fixed projection norm must be finite and positive."
            raise ValueError(msg)

    def initialize(self, width: int, dtype: jnp.dtype) -> Array:
        """Start on the first r coordinate axes unless a fitted start was supplied."""
        if self.rank > width:
            msg = "Projection rank exceeds the input coordinate width."
            raise ValueError(msg)
        value = (
            jnp.eye(width, self.rank, dtype=dtype)
            if self.initial is None
            else jnp.asarray(self.initial)
        )
        if (
            value.shape != (width, self.rank)
            or not jnp.issubdtype(value.dtype, jnp.floating)
            or not bool(jnp.all(jnp.isfinite(value)))
            or int(jnp.linalg.matrix_rank(value)) != self.rank
        ):
            msg = "Initial projection requires finite, full-rank matching components."
            raise ValueError(msg)
        return value


@dataclass(frozen=True)
class LearnBandwidth:
    """Positive initial bandwidth, fitted on the log scale."""

    initial: int | float

    def __post_init__(self) -> None:
        """Reject invalid bandwidth starts."""
        if not math.isfinite(self.initial) or self.initial <= 0:
            msg = "Initial graph bandwidth must be finite and positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class GraphML:
    """Joint Gaussian profile maximum likelihood for explicit graph parameters.

    Dependence and WX roles come from the regression. Each shared reference is
    optimized once across graphs and periods. Geometry uncertainty is not
    inferred from a fixed-W covariance matrix.
    """

    projections: Mapping[ProjectionRef, LearnLinear] = field(default_factory=dict)
    bandwidths: Mapping[BandwidthRef, LearnBandwidth] = field(default_factory=dict)
    max_steps: int = 256
    tolerance: float = 1e-7

    def __post_init__(self) -> None:
        """Own parameter declarations and reject colliding paths."""
        object.__setattr__(
            self, "projections", MappingProxyType(dict(self.projections))
        )
        object.__setattr__(self, "bandwidths", MappingProxyType(dict(self.bandwidths)))
        if {k.name for k in self.projections} & {k.name for k in self.bandwidths}:
            msg = "Projection and bandwidth names must be distinct."
            raise ValueError(msg)
        if (
            type(self.max_steps) is not int
            or self.max_steps < 1
            or not math.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            msg = "GraphML requires positive iterations and tolerance."
            raise ValueError(msg)

    def _initial(self, problem: PreparedGraphRegression) -> dict[str, Array]:
        widths: dict[ProjectionRef, int] = {}
        bandwidths = set()
        joint: set[ProjectionRef] = set()
        for period in problem.graphs:
            for src in period.values():
                if not isinstance(src, DistanceGraph):
                    continue
                ref, h = src.projection_ref, src.weighting.bandwidth
                if ref is not None and src.input_width is not None:
                    if ref in widths and widths[ref] != src.input_width:
                        msg = "A shared projection requires matching input dimensions."
                        raise ValueError(msg)
                    widths[ref] = src.input_width
                    if isinstance(h, BandwidthRef):
                        joint.add(ref)
                if isinstance(h, BandwidthRef):
                    bandwidths.add(h)
        if (
            set(widths) != self.projections.keys()
            or bandwidths != self.bandwidths.keys()
        ):
            msg = "Graph bindings must match all references used by the model."
            raise ValueError(msg)
        if any(self.projections[ref].norm is None for ref in joint):
            msg = "Fix the projection norm to jointly learn bandwidth."
            raise ValueError(msg)
        dtype = jnp.result_type(*problem.outcome, 1.0)
        return {
            **{
                ref.name: self.projections[ref].initialize(width, dtype)
                for ref, width in widths.items()
            },
            **{
                ref.name: jnp.log(jnp.asarray(spec.initial, dtype=dtype))
                for ref, spec in self.bandwidths.items()
            },
        }

    def estimate(self, problem: PreparedGraphRegression) -> GraphRegressionResult:
        """Estimate the native likelihood and bind its retained graph coordinates."""
        initial = self._initial(problem)
        if not initial:
            lag, error, design = problem.arrays()
            fit = estimate_graph_ml(
                lag,
                error,
                design,
                problem.outcome,
                problem.parameter_names,
                observed=problem.observed,
                max_steps=self.max_steps,
                tolerance=self.tolerance,
            )
            return problem.make_result(fit)
        fit, state, sensitivity, availability = _estimate_joint_graph_ml(
            initial,
            problem.arrays,
            problem.outcome,
            problem.parameter_names,
            n_lag=len(problem.lag_names),
            n_spatial=len(problem.lag_names) + len(problem.error_names),
            projection_ranks={
                ref.name: spec.rank for ref, spec in self.projections.items()
            },
            projection_norms={
                ref.name: spec.norm
                for ref, spec in self.projections.items()
                if spec.norm is not None
            },
            bandwidths=tuple(ref.name for ref in self.bandwidths),
            observed=problem.observed,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
        )
        free = {
            **{
                ref.name: int(state[ref.name].size)
                - (1 if spec.norm is not None else 0)
                for ref, spec in self.projections.items()
            },
            **{ref.name: 1 for ref in self.bandwidths},
        }
        return problem.make_result(
            fit,
            parameters=state,
            sensitivity=sensitivity,
            availability=availability,
            free_dimensions=free,
        )
