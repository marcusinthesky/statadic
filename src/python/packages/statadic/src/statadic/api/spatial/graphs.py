"""Named spatial graph sources with eager identity checks and pure numerical weights."""

from __future__ import annotations

import math
from collections.abc import (  # noqa: TC003 -- runtime public annotations
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, assert_never, override, runtime_checkable

import jax.numpy as jnp

from statadic.api.data import (
    MeasureColumn,  # noqa: TC001
    SpatialPanelData,
)
from statadic.api.design import SumTerm, Term
from statadic.operators.spatial import SpatialPanelWeights

if TYPE_CHECKING:
    from statadic.api.data import Observations, PanelLayout
    from statadic.api.design import FittedTerm
    from statadic.api.roles import InputRole
from statadic.api.representation import GeometrySpec, PairwiseGeometry  # noqa: TC001
from statadic.core.axioms import Law  # noqa: TC001
from statadic.core.domains import ConstructionOrigin, ObjectDomain  # noqa: TC001
from statadic.core.matrices import DMat, MatrixProvenance, checked_dmat  # noqa: TC001
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.operators.spatial.distance import rbf_weights
from statadic.operators.spatial.weights import (
    SpatialWeights,
    prepare_spatial_weights,
)
from statadic.sample.packed import PackedClouds  # noqa: TC001
from statadic.sample.reduction import Projection

__all__ = [
    "BandwidthRef",
    "DistanceGraph",
    "GraphPeriods",
    "GraphSource",
    "MeasureDistances",
    "PanelData",
    "PanelGraph",
    "PeriodGraphs",
    "ProjectionRef",
    "RBFWeights",
    "SpatialData",
    "SpatialError",
    "SpatialLag",
    "SpatialLagError",
    "SpatialProcesses",
    "SpatialX",
    "SquaredGeometry",
    "process_graphs",
    "split_spatial_terms",
]


_MIN_UNITS = 2
_MATRIX_AXES = 2
_MATRIX_NDIM = 2


@dataclass(frozen=True)
class ProjectionRef:
    """A shared, named input-by-output coordinate matrix."""

    name: str

    def __post_init__(self) -> None:
        """Require a nonempty stable parameter key."""
        if not self.name.strip():
            msg = "Projection parameter names must be nonempty."
            raise ValueError(msg)


@dataclass(frozen=True)
class BandwidthRef:
    """A shared positive bandwidth, expressed on the supplied cost scale."""

    name: str

    def __post_init__(self) -> None:
        """Require a nonempty stable parameter key."""
        if not self.name.strip():
            msg = "Bandwidth parameter names must be nonempty."
            raise ValueError(msg)


@dataclass(frozen=True)
class RBFWeights:
    """Off-diagonal row-normalized exp(-cost / bandwidth)."""

    bandwidth: int | float | BandwidthRef

    def __post_init__(self) -> None:
        """Check fixed bandwidths before numerical transformations."""
        h = self.bandwidth
        if not isinstance(h, BandwidthRef) and (
            isinstance(h, bool) or not math.isfinite(h) or h <= 0
        ):
            msg = "RBF bandwidth must be finite and positive."
            raise ValueError(msg)


@runtime_checkable
class SquaredGeometry(Protocol):
    """Optional direct cost implementation for differentiating projected measures."""

    def squared_cross(self, left: PackedClouds, right: PackedClouds) -> Array:
        """Evaluate squared distances without a zero-distance square root."""
        ...


@dataclass(frozen=True)
class MeasureDistances:
    """Ordered empirical measures and the coordinate map applied before comparison."""

    column: MeasureColumn
    geometry: GeometrySpec
    projection: ProjectionRef | Projection | Array | None

    def projected(self, parameters: Mapping[str, Array]) -> PackedClouds:
        """Transform the active coordinates while retaining counts and probabilities."""
        clouds = self.column.measures
        projection = self.projection
        if isinstance(projection, ProjectionRef):
            clouds = clouds.affine(parameters[projection.name])
        elif isinstance(projection, Projection):
            clouds = clouds.affine(
                projection.components, -projection.mean @ projection.components
            )
        elif projection is not None:
            clouds = clouds.affine(projection)
        return clouds

    def costs(self, parameters: Mapping[str, Array]) -> Array:
        """Compute pairwise costs, preferring a direct squared evaluation."""
        clouds = self.projected(parameters)
        if isinstance(self.geometry, SquaredGeometry):
            values = self.geometry.squared_cross(clouds, clouds)
        else:
            values = jnp.square(self.geometry.cross(clouds, clouds))
        # A finite self-cost is zero by construction, including roundoff.
        diagonal = jnp.diag(values)
        return (
            ((values + values.T) / 2)
            .at[jnp.diag_indices(values.shape[0])]
            .set(jnp.where(jnp.isfinite(diagonal), 0, diagonal))
        )


@dataclass(frozen=True)
class DistanceGraph:
    """One static firm roster with either retained costs or a measure recipe.

    Repeat a recipe across dates or bind an explicit (period, recipe) sequence
    in PanelData. Numerical leaves can vary without changing the roster.
    """

    source: (
        Array
        | DMat[ObjectDomain, MatrixProvenance[Law, ConstructionOrigin]]
        | MeasureDistances
    )
    unit_ids: tuple[int, ...]
    weighting: RBFWeights

    def __post_init__(self) -> None:
        """Validate the source and unit ordering at the eager boundary."""
        ids = tuple(self.unit_ids)
        if (
            len(ids) < _MIN_UNITS
            or any(type(i) is not int for i in ids)
            or len(set(ids)) != len(ids)
        ):
            msg = (
                "Distance graphs require at least two unique, ordered integer unit IDs."
            )
            raise ValueError(msg)
        object.__setattr__(self, "unit_ids", ids)
        if isinstance(self.source, MeasureDistances):
            if self.source.column.row_ids != ids:
                msg = "Measure rows must match graph unit IDs in order."
                raise ValueError(msg)
        else:
            values = (
                self.source.values if isinstance(self.source, DMat) else self.source
            )
            costs = checked_dmat(values).values
            if costs.shape[0] != len(ids):
                msg = "Graph cost shape must match unit IDs."
                raise ValueError(msg)

    @classmethod
    def from_costs(
        cls, costs: ArrayLike, *, unit_ids: Sequence[int], weighting: RBFWeights
    ) -> DistanceGraph:
        """Consume nonnegative symmetric costs directly in the exponential."""
        values = jnp.asarray(costs)
        if not (
            jnp.issubdtype(values.dtype, jnp.integer)
            or jnp.issubdtype(values.dtype, jnp.floating)
        ):
            msg = "Graph costs must have a real numeric dtype."
            raise TypeError(msg)
        return cls(
            values.astype(jnp.result_type(values, 1.0)), tuple(unit_ids), weighting
        )

    @classmethod
    def from_distances(
        cls,
        distances: ArrayLike
        | DMat[ObjectDomain, MatrixProvenance[Law, ConstructionOrigin]]
        | PairwiseGeometry,
        *,
        weighting: RBFWeights,
        unit_ids: Sequence[int] | None = None,
    ) -> DistanceGraph:
        """Square rooted distances once and retain producer row identities."""
        if isinstance(distances, PairwiseGeometry):
            ids = distances.row_ids if unit_ids is None else tuple(unit_ids)
            if ids != distances.row_ids:
                msg = "Supplied graph unit IDs disagree with the pairwise geometry."
                raise ValueError(msg)
            values = distances.values
        else:
            if unit_ids is None:
                msg = "Externally supplied distances require ordered unit IDs."
                raise ValueError(msg)
            ids = tuple(unit_ids)
            values = (
                distances.values
                if isinstance(distances, DMat)
                else jnp.asarray(distances)
            )
        if not all(type(identity) is int for identity in ids):
            msg = (
                "Spatial unit IDs require integer codes aligned with the observations."
            )
            raise ValueError(msg)
        checked = checked_dmat(values.astype(jnp.result_type(values, 1.0)))
        return cls(checked, tuple(int(i) for i in ids), weighting)

    @classmethod
    def from_measures(
        cls,
        column: MeasureColumn,
        *,
        geometry: GeometrySpec,
        weighting: RBFWeights,
        projection: Projection | ProjectionRef | Array | None = None,
    ) -> DistanceGraph:
        """Apply a fixed or shared map to the atoms before comparing the measures."""
        if not all(type(identity) is int for identity in column.row_ids):
            msg = "Supply integer-coded measure row IDs matching the graph roster."
            raise ValueError(msg)
        if isinstance(projection, ProjectionRef) and not isinstance(
            geometry, SquaredGeometry
        ):
            msg = "Learned projections require a direct squared_cross evaluator."
            raise TypeError(msg)
        if isinstance(projection, Projection) or (
            projection is not None and not isinstance(projection, ProjectionRef)
        ):
            matrix = (
                projection.components
                if isinstance(projection, Projection)
                else projection
            )
            if (
                matrix.ndim != _MATRIX_AXES
                or matrix.shape[0] != column.measures.data.shape[-1]
                or not matrix.shape[1]
                or not bool(jnp.all(jnp.isfinite(matrix)))
            ):
                msg = "Fixed projection expects finite input-by-output components."
                raise ValueError(msg)
        return cls(
            MeasureDistances(column, geometry, projection),
            tuple(int(i) for i in column.row_ids),
            weighting,
        )

    @property
    def projection_ref(self) -> ProjectionRef | None:
        """Return the jointly fitted projection reference, if present."""
        if isinstance(self.source, MeasureDistances) and isinstance(
            self.source.projection, ProjectionRef
        ):
            return self.source.projection
        return None

    @property
    def input_width(self) -> int | None:
        """Return the original atom dimension for initialization and validation."""
        return (
            self.source.column.measures.data.shape[-1]
            if isinstance(self.source, MeasureDistances)
            else None
        )

    def weights(self, parameters: Mapping[str, Array] | None = None) -> Array:
        """Evaluate dynamic parameters on the fixed roster, suitable for jax.grad."""
        state = {} if parameters is None else parameters
        if isinstance(self.source, MeasureDistances):
            cost = self.source.costs(state)
        elif isinstance(self.source, DMat):
            cost = jnp.square(self.source.values)
        elif isinstance(self.source, Array):
            cost = self.source
        else:
            assert_never(self.source)
        h = self.weighting.bandwidth
        bandwidth = state[h.name] if isinstance(h, BandwidthRef) else jnp.asarray(h)
        valid = jnp.all(jnp.isfinite(cost) & (cost >= 0))
        valid &= jnp.isfinite(bandwidth) & (bandwidth > 0)
        return jnp.where(valid, rbf_weights(cost, bandwidth), jnp.nan)

    def realize(
        self, parameters: Mapping[str, Array] | None = None
    ) -> SpatialWeights[Array]:
        """Materialize the fitted graph and verify finite weights and identity."""
        if parameters is None and (
            self.projection_ref is not None
            or isinstance(self.weighting.bandwidth, BandwidthRef)
        ):
            msg = "Unbound graph parameters require joint fitting or frozen values."
            raise ValueError(msg)
        if (
            isinstance(self.weighting.bandwidth, BandwidthRef)
            and parameters is not None
        ):
            h = parameters[self.weighting.bandwidth.name]
            if h.shape != () or not bool(jnp.isfinite(h) & (h > 0)):
                msg = "Resolved graph bandwidth must be a finite positive scalar."
                raise ValueError(msg)
        return prepare_spatial_weights(
            self.weights(parameters), unit_ids=jnp.asarray(self.unit_ids)
        )


type PeriodGraphs = tuple[tuple[int, SpatialWeights], ...]
type GraphSource = SpatialWeights | DistanceGraph
type GraphPeriods = tuple[tuple[int, GraphSource], ...]
type PanelGraph = GraphSource | SpatialPanelWeights | GraphPeriods


class _DependenceTerm:
    """Add declarations while preserving which graph supplies each process."""

    def __add__(self, other: _DependenceTerm) -> SpatialProcesses | SpatialLagError:
        if not isinstance(other, _DependenceTerm):
            return NotImplemented
        left_lag, left_error = process_graphs(self)
        right_lag, right_error = process_graphs(other)
        lag, error = left_lag + right_lag, left_error + right_error
        if len(lag) == len(error) == 1 and lag == error:
            return SpatialLagError(lag[0])
        return SpatialProcesses(lag, error)


@dataclass(frozen=True)
class SpatialLag(_DependenceTerm):
    """Declare endogenous outcome feedback on one named graph."""

    graph: str


@dataclass(frozen=True)
class SpatialError(_DependenceTerm):
    """Declare autoregressive disturbances on one named graph."""

    graph: str


@dataclass(frozen=True)
class SpatialLagError(_DependenceTerm):
    """Declare both spatial processes on the same operator (SAC)."""

    graph: str


@dataclass(frozen=True)
class SpatialProcesses(_DependenceTerm):
    """Ordered named operators, with independent coefficients for each role."""

    lag: tuple[str, ...]
    error: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate uniqueness independently for each process."""
        for names in (self.lag, self.error):
            if not isinstance(names, tuple) or any(
                not isinstance(name, str) or not name for name in names
            ):
                msg = "Each graph role requires a tuple of nonempty graph names."
                raise ValueError(msg)
            if len(set(names)) != len(names):
                msg = "A named graph may appear only once per process."
                raise ValueError(msg)
        if not self.lag and not self.error:
            msg = "Declare at least one spatial process."
            raise ValueError(msg)


def process_graphs(
    dependence: _DependenceTerm | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve ordered outcome and disturbance graph names."""
    if isinstance(dependence, SpatialProcesses):
        return dependence.lag, dependence.error
    lag = (
        (dependence.graph,)
        if isinstance(dependence, SpatialLag | SpatialLagError)
        else ()
    )
    error = (
        (dependence.graph,)
        if isinstance(dependence, SpatialError | SpatialLagError)
        else ()
    )
    return lag, error


@dataclass(frozen=True)
class SpatialX(Term):
    """Lag a named non-intercept design column on its full graph roster."""

    column: str
    graph: str

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """The selected column's underlying term declares its input roles."""
        return {}

    @override
    def _prepare_term(self, training: Observations) -> FittedTerm:
        msg = "SpatialX requires a top-level additive term and named graph data."
        raise ValueError(msg)


def split_spatial_terms(terms: Term, graph: str | None) -> tuple[Term, tuple[str, ...]]:
    """Separate design construction from a backend's name-based WX selection."""
    parts = terms.terms if isinstance(terms, SumTerm) else (terms,)
    lags = tuple(term for term in parts if isinstance(term, SpatialX))
    if not lags:
        return terms, ()
    if any(term.graph != graph for term in lags):
        msg = "SpatialX and dependence must reference the same named graph."
        raise ValueError(msg)
    base = tuple(term for term in parts if not isinstance(term, SpatialX))
    names = tuple(term.column for term in lags)
    if not base or "intercept" in names or len(set(names)) != len(names):
        msg = "SpatialX requires unique, non-intercept columns in the local design."
        raise ValueError(msg)
    return SumTerm(base), names


@dataclass(frozen=True)
class SpatialData:
    """Cross-sectional observations with a named graph registry and ordered IDs."""

    observations: Observations
    graphs: Mapping[str, GraphSource]
    unit: str | None = None

    def __post_init__(self) -> None:
        """Retain the mapping independently of subsequent caller mutations."""
        object.__setattr__(self, "graphs", MappingProxyType(dict(self.graphs)))


@dataclass(frozen=True)
class PanelData:
    """Full entity-period rows, optional observed records and named exogenous graphs.

    A single SpatialWeights repeats on the same units in every observed node
    period. SpatialPanelWeights has a balanced roster. A period graph tuple
    can describe changing populations. All identity checks precede estimation.
    """

    nodes: Observations
    panel: PanelLayout
    graphs: Mapping[str, PanelGraph] = field(default_factory=dict)
    outcomes: Observations | None = None

    def __post_init__(self) -> None:
        """Own the registry; period graph sequences are immutable tuples."""
        retained = {
            name: graph
            if isinstance(graph, (DistanceGraph, SpatialWeights, SpatialPanelWeights))
            else tuple((period, weights) for period, weights in graph)
            for name, graph in self.graphs.items()
        }
        object.__setattr__(self, "graphs", MappingProxyType(retained))

    @property
    def spatial(self) -> SpatialPanelData:
        """Preserve both row axes for the observed spatial likelihoods."""
        return SpatialPanelData(
            self.nodes, self.nodes if self.outcomes is None else self.outcomes
        )

    def graph_sources(self, name: str) -> GraphPeriods:
        """Bind named sources to their explicit integer periods before realization."""
        graph = self.graphs[name]
        if isinstance(graph, SpatialPanelWeights):
            periods = graph.period_ids.tolist()
            return tuple(
                (
                    period,
                    prepare_spatial_weights(
                        graph.values
                        if graph.values.ndim == _MATRIX_NDIM
                        else graph.values[t],
                        unit_ids=graph.unit_ids,
                        allow_isolates=True,
                        allow_diagonal=True,
                    ),
                )
                for t, period in enumerate(periods)
            )
        if isinstance(graph, (SpatialWeights, DistanceGraph)):
            times = self.nodes.grouping_labels(self.panel.time)
            if any(type(time) is not int for time in times):
                msg = "Spatial graph periods must use integer time keys."
                raise ValueError(msg)
            return tuple((int(t), graph) for t in sorted(set(times)))
        return graph

    def period_graphs(self, name: str) -> PeriodGraphs:
        """Realize a fixed graph source for each actual period."""
        return tuple(
            (period, source.realize() if isinstance(source, DistanceGraph) else source)
            for period, source in self.graph_sources(name)
        )
