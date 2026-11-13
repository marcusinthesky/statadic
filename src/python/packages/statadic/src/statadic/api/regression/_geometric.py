"""Fixed-geometry regression and explicit nested permutation hypotheses.

The response functional is raw distance for DISCO or squared distance for
PERMANOVA. Adjusted inference requires a constructed PSD-eligible response and
full-rank, nested, unweighted designs. An observed matrix alone retains only
the existing one-factor descriptive/permutation interpretation. No ordination,
spectral truncation, coefficient vector or prediction model is manufactured.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 -- runtime proposition annotation
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, override

import jax
import jax.numpy as jnp

from statadic.api.base import Result
from statadic.api.comparison._shared.draws import _disco_draws
from statadic.api.comparison._shared.plan import PermutationPlan
from statadic.api.data import Observations  # noqa: TC001 -- retained host context
from statadic.api.design import (
    FittedCategorical,
    FittedDesign,
    FittedIntercept,
    FittedSumTerm,
    level_codes,
)
from statadic.api.hypotheses import Hypothesis, ResampledReference, TestOutcome
from statadic.api.inputs import as_observations
from statadic.api.reporting import (  # noqa: TC001
    MetadataValue,
    Metric,
    Report,
    TestRow,
    _row_identity,
)
from statadic.api.representation import (
    ObservedDissimilarity,
    PairwiseGeometry,
    PreparedGeometry,
)
from statadic.association.dispersion import disco_decomposition
from statadic.association.geometric import (
    GeometricProjection,
    centered_response_gram,
    geometric_projection,
)
from statadic.core.domains import ConstructionOrigin
from statadic.core.statistics import (
    CalibratedTest,
    Calibration,
    CalibrationProvenance,
    Design,
    ExactCalibration,
    Null,
    RealizedStatistic,
    ResampledCalibration,
    StatisticProvenance,
    VarianceRatioStatistic,
)
from statadic.inference.calibration import add_one_p_kernel
from statadic.inference.geometric import freedman_lane_null, response_permutation_null

if TYPE_CHECKING:
    from statadic.api.comparison._shared.design import ExchangeabilityDesign
    from statadic.api.design import Term
    from statadic.api.inputs import RegressionInput
    from statadic.core.typing import ArrayLike


@dataclass(frozen=True)
class GeometricResponse:
    """A fixed response geometry and its explicit distance functional.

    ``distance_power=1`` is the raw DISCO functional; ``2`` is the squared
    PERMANOVA functional. Row identities must exactly match the predictor data.
    """

    source: PairwiseGeometry | PreparedGeometry | ObservedDissimilarity = field(
        repr=False
    )
    distance_power: Literal[1, 2] = 1

    def __post_init__(self) -> None:
        """Reject an unrecognized source or an implicit change of functional."""
        if type(self.distance_power) is not int or self.distance_power not in (1, 2):
            message = "distance_power must be the integer 1 or 2."
            raise ValueError(message)
        if not isinstance(
            self.source, (PairwiseGeometry, PreparedGeometry, ObservedDissimilarity)
        ):
            message = "Use constructed geometry or GeometricResponse.from_distances."
            raise TypeError(message)

    @classmethod
    def from_distances(
        cls,
        values: ArrayLike,
        *,
        row_ids: tuple[int | str, ...],
        distance_power: Literal[1, 2] = 1,
    ) -> GeometricResponse:
        """Retain one observed matrix without asserting a population geometry law."""
        return cls(ObservedDissimilarity.from_array(values, row_ids), distance_power)


@dataclass(frozen=True)
class Reduced:
    """A nested null selecting scalar columns of the already-fitted full design.

    Names are the retained ``result.design.column_names``, not new terms to fit.
    Retain the explicit intercept and omit at least one full-design column.
    """

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Own an unambiguous ordered selection of retained coefficient columns."""
        object.__setattr__(self, "columns", tuple(self.columns))
        if (
            not self.columns
            or any(not isinstance(name, str) or not name for name in self.columns)
            or len(set(self.columns)) != len(self.columns)
        ):
            message = "Reduced requires nonempty, unique fitted column names."
            raise ValueError(message)

    def describe(self) -> str:
        """State the nested null in the full design's retained column basis."""
        return f"No added geometric effect beyond retained columns {self.columns}."

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Retain the selected column names without serializing a live design."""
        return {"kind": "reduced-geometric-design", "columns": self.columns}


@dataclass(frozen=True)
class ResponsePermutation:
    """Calibrate the strong response-exchangeability null with a retained plan.

    One-factor designs reuse the existing label arrangements and their exact
    schedule orientation. General designs jointly permute both response axes.
    Nuisance adjustment with non-intercept reduced columns uses FreedmanLane.
    """

    plan: PermutationPlan = field(default_factory=PermutationPlan)


@dataclass(frozen=True)
class FreedmanLane:
    """Permute reduced residuals, retaining approximate nuisance calibration.

    Exhaustive schedules are permitted as an enumeration of this approximation;
    they do not establish an exact test under arbitrary nuisance designs.
    """

    plan: PermutationPlan = field(default_factory=PermutationPlan)


@dataclass(frozen=True)
class PreparedGeometric:
    """Exact response, training-fitted design and identities owned by one model."""

    response: GeometricResponse
    geometry: PairwiseGeometry | ObservedDissimilarity = field(repr=False)
    observations: Observations = field(repr=False)
    design: FittedDesign = field(repr=False)
    values: jax.Array = field(repr=False)
    functional: jax.Array = field(repr=False)
    gram: jax.Array = field(repr=False)
    full_basis: jax.Array = field(repr=False)
    psd_eligible: bool
    group_codes: jax.Array | None = field(repr=False)
    group_levels: tuple[int | str, ...] = field(repr=False)
    model_token: object = field(repr=False)


def _basis(values: jax.Array) -> jax.Array:
    """Require identified columns before exposing a reduced orthonormal basis."""
    if values.shape[1] == 0:
        return jnp.empty((values.shape[0], 0), dtype=values.dtype)
    singular = jnp.linalg.svd(values, compute_uv=False)
    cutoff = jnp.max(singular) * max(values.shape) * jnp.finfo(values.dtype).eps
    rank = int(jnp.sum(singular > cutoff))
    if rank != values.shape[1]:
        message = "Geometric regression requires full-rank fitted design columns."
        raise ValueError(message)
    return jnp.linalg.qr(values, mode="reduced")[0]


def _groups(
    design: FittedDesign, data: Observations
) -> tuple[jax.Array | None, tuple[int | str, ...]]:
    """Recognize exactly one fitted categorical term plus the explicit intercept."""
    parts = (
        design.terms.terms
        if isinstance(design.terms, FittedSumTerm)
        else (design.terms,)
    )
    categorical = [part for part in parts if isinstance(part, FittedCategorical)]
    if len(categorical) != 1 or any(
        not isinstance(part, (FittedCategorical, FittedIntercept)) for part in parts
    ):
        return None, ()
    levels, codes = level_codes(data.categorical(categorical[0].column))
    return codes, levels


def prepare_geometric(
    response: GeometricResponse, terms: Term, data: RegressionInput, token: object
) -> PreparedGeometric:
    """Prepare only declared predictors and preserve exact response row alignment."""
    observations = as_observations(data, terms.input_columns, broadcast_scalars=False)
    source = response.source
    geometry = source.pairwise() if isinstance(source, PreparedGeometry) else source
    if geometry.row_ids != observations.row_ids:
        message = "Predictor row IDs must exactly match geometric response row order."
        raise ValueError(message)
    if isinstance(geometry, ObservedDissimilarity):
        geometry = ObservedDissimilarity.from_array(geometry.values, geometry.row_ids)
    design = FittedDesign(terms._prepare_term(observations))
    if not design.has_intercept:
        message = "Geometric regression requires an explicit Intercept term."
        raise ValueError(message)
    values = jnp.asarray(design.transform(observations))
    values = values.astype(jnp.result_type(values, geometry.values, 1.0))
    if values.shape[1] >= values.shape[0] or not bool(jnp.all(jnp.isfinite(values))):
        message = (
            "Geometric regression needs finite predictors and residual "
            "degrees of freedom."
        )
        raise ValueError(message)
    basis = _basis(values)
    functional = geometry.values**response.distance_power
    if not bool(jnp.all(jnp.isfinite(functional))):
        message = "The geometric response functional is not numerically representable."
        raise ValueError(message)
    gram = centered_response_gram(functional)
    codes, levels = _groups(design, observations)
    eligible = False
    if isinstance(geometry, PairwiseGeometry):
        warranted = (
            geometry.negative_type
            if response.distance_power == 1
            else geometry.squared_negative_type
        )
        if not warranted:
            message = (
                "Geometric regression requires the response functional's "
                "negative-type construction warrant."
            )
            raise ValueError(message)
        spectrum = jnp.linalg.eigvalsh(gram)
        slack = max(1e-8, 8 * gram.shape[0] * float(jnp.finfo(gram.dtype).eps))
        eligible = bool(
            jnp.all(jnp.isfinite(spectrum))
            & (
                jnp.min(spectrum)
                >= -slack * jnp.maximum(1.0, jnp.max(jnp.abs(spectrum)))
            )
        )
        if not eligible:
            message = (
                "The constructed response Gram is not numerically "
                "positive semidefinite."
            )
            raise ValueError(message)
    elif codes is None:
        message = (
            "Observed dissimilarities support the one-factor descriptive route only."
        )
        raise ValueError(message)
    return PreparedGeometric(
        response,
        geometry,
        observations,
        design,
        values,
        functional,
        gram,
        basis,
        eligible,
        codes,
        levels,
        token,
    )


def _reduced(
    prepared: PreparedGeometric, hypothesis: Reduced | None
) -> tuple[Reduced, jax.Array, bool]:
    """Select a nested basis from retained columns, without refitting any term."""
    names = prepared.design.column_names
    intercept = tuple(names[index] for index in prepared.design.intercept_columns)
    requested = Reduced(intercept) if hypothesis is None else hypothesis
    if not set(requested.columns) < set(names) or not set(intercept) <= set(
        requested.columns
    ):
        message = (
            "Reduced must retain the intercept and select a proper subset "
            "of fitted columns."
        )
        raise ValueError(message)
    indices = jnp.asarray([names.index(name) for name in requested.columns])
    basis = _basis(prepared.values[:, indices])
    return requested, basis, set(requested.columns) == set(intercept)


def _observed(
    prepared: PreparedGeometric, reduced_basis: jax.Array, *, intercept_only: bool
) -> GeometricProjection:
    """Keep the established one-factor arithmetic, including observed matrices."""
    if prepared.group_codes is not None and intercept_only:
        within, between, total, r_squared, statistic = disco_decomposition(
            prepared.functional, prepared.group_codes, len(prepared.group_levels)
        )
        return GeometricProjection(
            within,
            between,
            total,
            r_squared,
            statistic,
            jnp.asarray(len(prepared.group_levels) - 1, dtype=jnp.int32),
            jnp.asarray(
                len(prepared.geometry.row_ids) - len(prepared.group_levels),
                dtype=jnp.int32,
            ),
            jnp.all(
                jnp.isfinite(jnp.stack((within, between, total, r_squared, statistic)))
            ),
        )
    if not prepared.psd_eligible:
        message = (
            "Adjusted geometric regression requires constructed PSD-eligible geometry."
        )
        raise ValueError(message)
    return geometric_projection(prepared.gram, prepared.full_basis, reduced_basis)


def estimate_geometric(prepared: PreparedGeometric) -> GeometricResult:
    """Estimate the observed full-versus-intercept variation without inference."""
    _, reduced_basis, intercept_only = _reduced(prepared, None)
    return GeometricResult(
        _observed(prepared, reduced_basis, intercept_only=intercept_only), prepared
    )


class GeometricDesign(Design):
    """A fixed full/reduced explanatory design over aligned response objects."""


class NoAddedGeometricEffect(Null):
    """No response variation attributable to full-design additions to the null."""


class GeometricProjectionOrigin(ConstructionOrigin):
    """Nested projection of a producer-warranted fixed response functional."""


class ResponsePermutationOrigin(ConstructionOrigin):
    """Response/label permutations under the declared exchangeability design."""


class FreedmanLaneOrigin(ConstructionOrigin):
    """Approximate reduced-signal plus permuted reduced-residual calibration."""


class SuppliedGeometricSchedule(ConstructionOrigin):
    """Caller-supplied schedules, with no claim of uniform random generation."""


type GeometricTest = CalibratedTest[
    jax.Array,
    StatisticProvenance[VarianceRatioStatistic, GeometricProjectionOrigin],
    CalibrationProvenance[
        GeometricDesign,
        NoAddedGeometricEffect,
        Calibration,
        ConstructionOrigin,
    ],
]
type _GeometricCalibration = CalibrationProvenance[
    GeometricDesign,
    NoAddedGeometricEffect,
    Calibration,
    ConstructionOrigin,
]


@dataclass(frozen=True)
class GeometricResult(Result[GeometricProjection, PreparedGeometric]):
    """Observed variation with retained preparation, without invented coefficients."""

    @property
    def design(self) -> FittedDesign:
        """Return the training-fitted scalar column schema."""
        return self.context.design

    @property
    def row_ids(self) -> tuple[int | str, ...]:
        """Return the exact response/predictor row order."""
        return self.context.geometry.row_ids

    @property
    def geometry(self) -> PairwiseGeometry | ObservedDissimilarity:
        """Return the actual fixed response geometry and its construction evidence."""
        return self.context.geometry

    @property
    def group_codes(self) -> jax.Array | None:
        """Return one-factor label codes when that is the actual fitted design."""
        return self.context.group_codes

    @property
    def group_levels(self) -> tuple[int | str, ...]:
        """Return sorted one-factor levels, or no levels for a general design."""
        return self.context.group_levels

    @property
    def decomposition(self) -> tuple[jax.Array, ...]:
        """Return within, between, total, partial R-squared and pseudo-F."""
        return (
            self.core.within,
            self.core.between,
            self.core.total,
            self.core.r_squared,
            self.core.statistic,
        )

    @property
    def statistic(self) -> jax.Array:
        """Return the default full-versus-intercept pseudo-F."""
        return self.core.statistic

    @property
    def valid(self) -> bool:
        """Expose observed numerical availability, not inferential qualification."""
        return bool(self.core.valid)

    def infer(
        self,
        *,
        resampling: ResponsePermutation | FreedmanLane,
    ) -> GeometricInference:
        """Bind a resampling design; a hypothesis is evaluated only at ``test``."""
        if not isinstance(resampling, (ResponsePermutation, FreedmanLane)):
            message = "Use ResponsePermutation or FreedmanLane with an explicit plan."
            raise TypeError(message)
        return GeometricInference(self, resampling)

    @override
    def report(self) -> Report:
        """Describe observed variation and its actual geometry provenance."""
        return _projection_report(self.core, self.context)


def _projection_report(
    core: GeometricProjection, prepared: PreparedGeometric
) -> Report:
    return Report(
        "Geometric regression",
        metrics=tuple(
            Metric(
                name,
                # The native degrees of freedom are integer arrays by construction.
                (
                    int(value)
                    if jnp.issubdtype(value.dtype, jnp.integer)
                    else float(value)
                )
                if bool(jnp.isfinite(value))
                else None,
                available=bool(jnp.isfinite(value)),
                unavailable_reason=None
                if bool(jnp.isfinite(value))
                else "Numerical response variation is unavailable.",
            )
            for name, value in (
                ("within", core.within),
                ("between", core.between),
                ("total", core.total),
                ("partial_R2", core.r_squared),
                ("F", core.statistic),
                ("df_test", core.df_test),
                ("df_residual", core.df_residual),
            )
        ),
        metadata={
            "row_ids": prepared.geometry.row_ids,
            "group_levels": prepared.group_levels,
            "observation_unit": "whole empirical measure"
            if isinstance(prepared.geometry, PairwiseGeometry)
            else "row of a supplied dissimilarity matrix",
            "distance_power": prepared.response.distance_power,
            "geometry_provenance": "constructed response geometry"
            if isinstance(prepared.geometry, PairwiseGeometry)
            else "observed matrix only",
            "calibration": "not requested",
            "qualification": "not established by this numerical result",
        },
    )


@dataclass(frozen=True)
class GeometricTestEvidence:
    """Complete fixed-response permutation evidence for one retained hypothesis."""

    observed: GeometricResult
    hypothesis: Hypothesis[Reduced]
    projection: GeometricProjection
    null: jax.Array
    draw_valid: jax.Array
    arrangements: jax.Array | None
    schedules: jax.Array | None
    design: ExchangeabilityDesign
    admissible_arrangements: int
    plan: PermutationPlan
    calibration_status: Literal["exchangeability-required", "approximate", "supplied"]
    pvalue: float | None
    valid: bool
    calibrated: GeometricTest | None = field(repr=False)

    @property
    def requested(self) -> int:
        """Count all attempted draws, including unavailable statistics."""
        return self.null.shape[0]

    @property
    def failed(self) -> int:
        """Count unavailable draws without removing their positions."""
        return int(jnp.sum(~self.draw_valid))

    @property
    def supplied_schedules(self) -> bool:
        """Whether the caller supplied the index rows rather than a random stream."""
        return self.plan.supplied

    @property
    def key(self) -> jax.Array | None:
        """Return the actual random stream, absent for supplied/enumerated draws."""
        if self.plan.method != "random" or self.plan.supplied:
            return None
        return (
            self.plan.residual_key
            if self.calibration_status == "approximate"
            else self.plan.key
        )

    @property
    def seed(self) -> int | None:
        """Return a random seed only when a random stream was actually used."""
        return self.plan.seed if self.key is not None else None

    @property
    def prng_implementation(self) -> str | None:
        """Identify the retained stream's implementation rather than an assumed PRNG."""
        key = self.key
        return None if key is None else str(jax.random.key_impl(key))

    def report(self) -> Report:
        """Keep enumeration distinct from the scientific calibration assumption."""
        observed = _projection_report(self.projection, self.observed.context)
        if self.calibration_status == "approximate":
            calibration = "Freedman-Lane reduced-residual permutation (approximate)"
            null = (
                "reduced residuals exchangeable under the nuisance model (approximate)"
            )
        elif self.plan.supplied:
            calibration = (
                "caller-supplied outer-label schedules"
                if self.arrangements is not None
                else "caller-supplied response-row schedules"
            )
            null = (
                "caller-supplied schedules; exchangeability is asserted by their design"
            )
        else:
            calibration = (
                "outer-label permutation inference requested"
                if self.arrangements is not None
                else "response-row permutation inference requested"
            )
            null = (
                "outer-label exchangeability with fixed pooled geometry"
                if self.arrangements is not None
                else "response-row exchangeability with fixed geometry"
            )
        if not self.design.free:
            null += f"; permuting {self.design.description()}"
        label = "approximate " if self.calibration_status == "approximate" else ""
        row = (
            TestRow(
                "geometric_effect",
                float(self.projection.statistic),
                self.pvalue,
                "resampled",
                (int(self.requested),),
                f"{label}{self.plan.method} permutation ({int(self.requested)} draws)",
                self.hypothesis.describe(),
            )
            if self.valid
            else TestRow(
                "geometric_effect",
                None,
                hypothesis=self.hypothesis.describe(),
                available=False,
                unavailable_reason="Observed or null statistics unavailable.",
            )
        )
        return Report(
            "Geometric regression: permutation inference",
            tests=(row,),
            metrics=(
                *(metric for metric in observed.metrics if metric.name != "F"),
                Metric("requested_permutations", int(self.requested)),
                Metric("failed_permutations", int(self.failed)),
            ),
            metadata={
                **dict(observed.metadata),
                "reduced_columns": self.hypothesis.null.columns,
                "hypothesis": self.hypothesis.metadata(),
                "calibration": calibration,
                "calibration_status": self.calibration_status,
                "null": null,
                "random_seed": self.seed,
                "prng_implementation": self.prng_implementation,
                "schedule_method": self.plan.method,
                "supplied_schedules": self.plan.supplied,
                "schedule_orientation": "permuted label arrangements"
                if self.arrangements is not None
                else "response rows in both Gram axes",
                "pvalue_rule": "extreme / all"
                if self.plan.method == "exact"
                else "(extreme + 1) / (all + 1)",
                "tail": ">= observed (ties included)",
                "exchangeability": self.design.metadata(self.admissible_arrangements),
            },
        )


@dataclass(frozen=True)
class GeometricInference:
    """A retained fit and resampling choice, before any null draws are evaluated."""

    observed: GeometricResult = field(repr=False)
    resampling: ResponsePermutation | FreedmanLane

    def test(
        self, hypothesis: Hypothesis[Reduced]
    ) -> TestOutcome[Reduced, GeometricTestEvidence]:
        """Evaluate an explicit reduced-design question using the bound schedule."""
        if not isinstance(hypothesis, Hypothesis) or not isinstance(
            hypothesis.null, Reduced
        ):
            message = "Geometric inference requires Hypothesis[Reduced]."
            raise TypeError(message)
        if hypothesis.alternative is not None:
            message = "Geometric inference supports the complement alternative only."
            raise ValueError(message)
        evidence = _infer_geometric(self.observed, self.resampling, hypothesis)
        reference = None
        if evidence.valid:
            reference = ResampledReference(
                "supplied" if evidence.plan.supplied else evidence.plan.method,
                evidence.requested,
                correction="none" if evidence.plan.method == "exact" else "add-one",
                approximate=evidence.calibration_status == "approximate",
            )
        return TestOutcome(
            "geometric_effect",
            hypothesis,
            float(evidence.projection.statistic) if evidence.valid else None,
            evidence.pvalue,
            reference,
            evidence.valid,
            unavailable_reason=None
            if evidence.valid
            else "observed-or-null-statistic-unavailable",
            details={
                "requested_draws": evidence.requested,
                "failed_draws": evidence.failed,
            },
            construction={
                "rows": _row_identity(self.observed.row_ids),
                "distance_power": self.observed.context.response.distance_power,
                "calibration_status": evidence.calibration_status,
                "schedule_method": evidence.plan.method,
                "supplied_schedules": evidence.supplied_schedules,
                "random_seed": evidence.seed,
                "prng_implementation": evidence.prng_implementation,
                "tail": ">= observed (ties included)",
                "exchangeability": evidence.design.metadata(
                    evidence.admissible_arrangements
                ),
            },
            assumptions=(
                (
                    "The fixed response geometry and declared resampling design "
                    "are appropriate."
                ),
                "Freedman-Lane nuisance calibration remains approximate."
                if isinstance(self.resampling, FreedmanLane)
                else "Response or group-label exchangeability is required.",
            ),
            evidence=evidence,
        )

    def report(self) -> Report:
        """Describe the bound procedure without drawing schedules or testing a null."""
        return Report(
            "Geometric regression: bound resampling",
            metadata={
                "procedure": type(self.resampling).__name__,
                "schedule_method": self.resampling.plan.method,
                "test_status": "not requested",
            },
        )


def _infer_geometric(  # noqa: C901, PLR0912, PLR0914 -- explicit method/calibration evidence
    result: GeometricResult,
    calibration: ResponsePermutation | FreedmanLane,
    hypothesis: Hypothesis[Reduced] | None,
) -> GeometricTestEvidence:
    if not isinstance(calibration, (ResponsePermutation, FreedmanLane)):
        message = "Use ResponsePermutation or FreedmanLane with an explicit plan."
        raise TypeError(message)
    prepared, plan = result.context, calibration.plan
    requested, reduced_basis, intercept_only = _reduced(
        prepared, None if hypothesis is None else hypothesis.null
    )
    retained_hypothesis = Hypothesis(requested) if hypothesis is None else hypothesis
    observed = _observed(prepared, reduced_basis, intercept_only=intercept_only)
    arrangements = None
    schedules = None
    if isinstance(calibration, ResponsePermutation):
        if not intercept_only:
            message = (
                "Raw response permutations do not adjust nuisance effects; "
                "use FreedmanLane."
            )
            raise ValueError(message)
        if prepared.group_codes is not None:
            draws, design, admissible = plan.arrangements(
                prepared.group_codes, result.row_ids, prepared.observations
            )
            arrangements = draws
            null = _disco_draws(prepared.functional, draws, len(prepared.group_levels))
            draw_valid = jnp.isfinite(null)
        else:
            schedules, design, admissible = plan.schedules(
                result.row_ids, observations=prepared.observations
            )
            null, draw_valid = response_permutation_null(
                prepared.gram, prepared.full_basis, reduced_basis, schedules
            )
    else:
        if not prepared.psd_eligible:
            message = (
                "FreedmanLane requires constructed PSD-eligible response geometry."
            )
            raise ValueError(message)
        schedules, design, admissible = plan.schedules(
            result.row_ids, residual=True, observations=prepared.observations
        )
        null, draw_valid = freedman_lane_null(
            prepared.gram, prepared.full_basis, reduced_basis, schedules
        )
    valid = bool(observed.valid) and bool(jnp.all(draw_valid))
    pvalue = None
    calibrated: GeometricTest | None = None
    status: Literal["exchangeability-required", "approximate", "supplied"] = (
        "approximate"
        if isinstance(calibration, FreedmanLane)
        else "supplied"
        if plan.supplied
        else "exchangeability-required"
    )
    if valid:
        pvalue_array = (
            jnp.sum(null >= observed.statistic) / null.size
            if plan.method == "exact"
            else add_one_p_kernel(observed.statistic, null)
        )
        pvalue = float(pvalue_array)
        if prepared.psd_eligible:
            provenance: _GeometricCalibration
            if isinstance(calibration, FreedmanLane) and plan.method == "exact":
                # Enumerating approximate residual transforms is neither an
                # exact null-orbit test nor Monte Carlo/add-one calibration.
                provenance = CalibrationProvenance[
                    GeometricDesign,
                    NoAddedGeometricEffect,
                    Calibration,
                    FreedmanLaneOrigin,
                ]()
            elif isinstance(calibration, FreedmanLane):
                provenance = CalibrationProvenance[
                    GeometricDesign,
                    NoAddedGeometricEffect,
                    ResampledCalibration,
                    FreedmanLaneOrigin,
                ]()
            elif plan.supplied:
                provenance = CalibrationProvenance[
                    GeometricDesign,
                    NoAddedGeometricEffect,
                    ResampledCalibration,
                    SuppliedGeometricSchedule,
                ]()
            elif plan.method == "exact":
                provenance = CalibrationProvenance[
                    GeometricDesign,
                    NoAddedGeometricEffect,
                    ExactCalibration,
                    ResponsePermutationOrigin,
                ]()
            else:
                provenance = CalibrationProvenance[
                    GeometricDesign,
                    NoAddedGeometricEffect,
                    ResampledCalibration,
                    ResponsePermutationOrigin,
                ]()
            calibrated = CalibratedTest(
                RealizedStatistic(
                    observed.statistic,
                    StatisticProvenance[
                        VarianceRatioStatistic, GeometricProjectionOrigin
                    ](),
                ),
                pvalue_array,
                provenance,
            )
    return GeometricTestEvidence(
        result,
        retained_hypothesis,
        observed,
        null,
        draw_valid,
        arrangements,
        schedules,
        design,
        admissible,
        plan,
        status,
        pvalue,
        valid,
        calibrated,
    )
