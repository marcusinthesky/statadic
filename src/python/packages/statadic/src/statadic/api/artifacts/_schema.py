"""Closed, required-field wire records for version-7 artifacts.

These describe serialized values. Decoders at each owning boundary establish
their runtime validity before constructing retained numerical state.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from statadic.api.reporting import JsonValue  # noqa: TC001  # Runtime wire annotations.


class WriterRecord(TypedDict):
    package: str
    version: str


class ArrayEntry(TypedDict):
    shape: list[int]
    dtype: str
    offset: int
    bytes: int
    sha256: str


class ArrayInventory(TypedDict):
    filename: Literal["arrays.bin"]
    sha256: str
    bytes: int
    entries: dict[str, ArrayEntry]


class Bundle[FamilyT: str = str, StateT = object](TypedDict):
    schema_version: Literal[7]
    writer: WriterRecord
    purpose: Literal["prediction", "analysis"]
    family: FamilyT
    capabilities: list[str]
    state: StateT
    metadata: dict[str, JsonValue]
    arrays: ArrayInventory


class SimpleKernel(TypedDict):
    kind: Literal["linear", "cosine"]


class RBFKernelRecord(TypedDict):
    kind: Literal["rbf"]
    bandwidth: str
    weak_type: bool


class MMDConfig(TypedDict):
    kernel: SimpleKernel | RBFKernelRecord


class EnergyConfig(TypedDict):
    ground: Literal["euclidean", "angular", "cosine", "normalized_euclidean"]
    exponent: float


class WassersteinConfig(TypedDict):
    p: float


class SinkhornConfig(TypedDict):
    epsilon: float
    iterations: int
    tolerance: float


class SlicedConfig(TypedDict):
    p: float
    n_projections: int
    key: str
    key_implementation: str
    directions: str


class _GeometryState(TypedDict):
    data: str
    counts: str
    masses: str
    row_ids: list[int | str]


class MMDGeometry(_GeometryState):
    kind: Literal["mmd"]
    config: MMDConfig


class EnergyGeometry(_GeometryState):
    kind: Literal["energy"]
    config: EnergyConfig


class WassersteinGeometry(_GeometryState):
    kind: Literal["wasserstein_1d"]
    config: WassersteinConfig


class SinkhornGeometry(_GeometryState):
    kind: Literal["sinkhorn_divergence"]
    config: SinkhornConfig


class SlicedGeometry(_GeometryState):
    kind: Literal["sliced_wasserstein"]
    config: SlicedConfig


type GeometryRecord = (
    MMDGeometry
    | EnergyGeometry
    | WassersteinGeometry
    | SlicedGeometry
    | SinkhornGeometry
)


class CrossGramRecord(TypedDict):
    column_mean: str
    grand_mean: str
    axes: str


class LinearMapRecord(TypedDict):
    kind: Literal["linear"]
    feature_names: list[str]
    mean: str
    scale: str
    projection_mean: str
    components: str


class KernelMapRecord(TypedDict):
    kind: Literal["kernel"]
    feature_names: list[str]
    reference: str
    kernel: SimpleKernel | RBFKernelRecord
    projection: CrossGramRecord


class DistanceMapRecord(TypedDict):
    kind: Literal["distance"]
    row_ids: list[int | str]
    projection: CrossGramRecord
    geometry: GeometryRecord | None
    measure_column: str | None


class FeatureDistanceRecord(TypedDict):
    feature_names: list[str]
    reference: str
    ground: Literal["euclidean", "angular", "cosine", "normalized_euclidean"]


class GeodesicMapRecord(TypedDict):
    kind: Literal["geodesic"]
    distance_map: DistanceMapRecord
    training_paths: str
    neighbors: int
    tie_codes: str
    feature_distances: FeatureDistanceRecord | None


type CoordinateMapRecord = (
    LinearMapRecord | KernelMapRecord | DistanceMapRecord | GeodesicMapRecord
)


class OrdinationTerm(TypedDict):
    kind: Literal["ordination"]
    mapping: CoordinateMapRecord
    column_names: list[str]


class InterceptTerm(TypedDict):
    kind: Literal["intercept"]


class LinearTerm(TypedDict):
    kind: Literal["linear"]
    column: str


class CyclicSplineTerm(TypedDict):
    kind: Literal["cyclic_spline"]
    column: str
    df: int
    period: float


class FourierTerm(TypedDict):
    kind: Literal["fourier"]
    column: str
    harmonics: int
    period: float


class SplineTerm(TypedDict):
    kind: Literal["spline"]
    column: str
    df: int
    degree: int
    lower: float
    upper: float


class CategoricalTerm(TypedDict):
    kind: Literal["categorical"]
    column: str
    levels: list[int | str]
    reference: int | str


class SumTerm(TypedDict):
    kind: Literal["sum"]
    terms: list[TermRecord]


class InteractionTerm(TypedDict):
    kind: Literal["interaction"]
    left: TermRecord
    right: TermRecord


class LandmarkTerm(TypedDict):
    kind: Literal["distance_to_landmarks"]
    column: str
    geometry: GeometryRecord


type TermRecord = (
    InterceptTerm
    | LinearTerm
    | CyclicSplineTerm
    | FourierTerm
    | SplineTerm
    | CategoricalTerm
    | SumTerm
    | InteractionTerm
    | LandmarkTerm
    | OrdinationTerm
)


class DesignRecord(TypedDict):
    terms: TermRecord
    column_names: list[str]


class _RegressionState(TypedDict):
    design: DesignRecord
    parameters: str


class LinearState(_RegressionState):
    offset: None
    exposure: None


class PoissonState(_RegressionState):
    offset: str | None
    exposure: str | None


class LogisticState(_RegressionState):
    offset: str | None
    exposure: None
    trials: str | None


type RegressionState = LinearState | PoissonState | LogisticState


class LinearPayload(TypedDict):
    family: Literal["linear"]
    state: LinearState


class PoissonPayload(TypedDict):
    family: Literal["poisson"]
    state: PoissonState


class LogisticPayload(TypedDict):
    family: Literal["logistic"]
    state: LogisticState


type RegressionPayload = LinearPayload | PoissonPayload | LogisticPayload
"""One regression family together with the state that family requires."""

type RegressionBundle = (
    Bundle[Literal["linear"], LinearState]
    | Bundle[Literal["poisson"], PoissonState]
    | Bundle[Literal["logistic"], LogisticState]
)
