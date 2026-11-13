"""Observed group comparisons and explicit permutation inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, override

import jax
import jax.numpy as jnp

from statadic.api.base import Result
from statadic.api.comparison._shared import (
    ComparisonStatistic,  # noqa: TC001
    ComparisonTest,  # noqa: TC001
    EnumerationCalibration,  # noqa: TC001
    ExactOuterLabelPermutation,
    ExchangeabilityDesign,  # noqa: TC001 -- runtime annotations
    NativeDecomposition,
    PermutationPlan,
    RandomCalibration,  # noqa: TC001
    RandomOuterLabelPermutation,
    RandomResidualPermutation,
    ResidualCalibration,  # noqa: TC001
    SpatialMedianDispersionOrigin,
    SquaredDistanceAnovaOrigin,
    SuppliedCalibration,  # noqa: TC001
    SuppliedOuterLabelSchedule,
    SuppliedResidualCalibration,  # noqa: TC001
    SuppliedResidualSchedule,
    _disco_draws,
)
from statadic.api.data import Observations  # noqa: TC001 -- runtime annotations
from statadic.api.hypotheses import Hypothesis, ResampledReference, TestOutcome
from statadic.api.ordination import OrdinationSpectrum
from statadic.api.reporting import (
    MetadataValue,  # noqa: TC001 -- proposition annotation
    Metric,
    PlotSpec,
    ReferenceLine,
    Report,
    TestRow,
    _row_identity,
)
from statadic.api.representation import (
    ObservedDissimilarity,  # noqa: TC001 -- runtime result annotations
    PairwiseGeometry,
)
from statadic.association.decomposition import (
    DiscoDecomposition,
)
from statadic.core.statistics import (
    CalibratedTest,
    CalibrationProvenance,
    ExactCalibration,
    GroupedDesign,
    NoGroupEffect,
    RealizedStatistic,
    ResampledCalibration,
    StatisticProvenance,
    VarianceRatioStatistic,
)
from statadic.core.typing import Array  # Runtime annotations.
from statadic.inference.dispersion import (
    dispersion_residual_null,
)

type ComparisonCore = (
    NativeDecomposition
    | tuple[Array, Array, Array, Array, Array]
    | Array
    | OrdinationSpectrum
)


@dataclass(frozen=True)
class GroupComparisonResult(
    Result[ComparisonCore, "PairwiseGeometry | ObservedDissimilarity"]
):
    """Observed analysis and retained geometry; no automatic inferential claims."""

    context: PairwiseGeometry | ObservedDissimilarity = field(repr=False)

    method: str
    group_codes: Array = field(repr=False)
    group_levels: tuple[int | str, ...] = field(repr=False)
    decomposition: tuple[Array, Array, Array, Array, Array] | None = field(repr=False)
    coordinates: Array | None = field(repr=False)
    dispersion_statistic: Array | None = field(default=None, repr=False)
    dispersion_radii: Array | None = field(default=None, repr=False)
    median_converged: bool | None = None
    bias_adjust: bool | None = None
    reason: str | None = None
    spectrum: OrdinationSpectrum | None = field(default=None, repr=False)
    groups: Observations | None = field(default=None, repr=False, compare=False)

    @property
    def geometry(self) -> PairwiseGeometry | ObservedDissimilarity:
        """Return the retained geometry."""
        return self.context

    @property
    def statistic(self) -> Array:
        """Return the observed F-ratio, independently of permutation availability."""
        if self.decomposition is not None:
            return self.decomposition[-1]
        if self.dispersion_statistic is None:
            message = "The comparison has no observed statistic."
            raise ValueError(message)
        return self.dispersion_statistic

    @property
    def valid(self) -> bool:
        """Require finite observed quantities rather than accepting infinities."""
        values = self.decomposition or (self.statistic,)
        return bool(jnp.all(jnp.isfinite(jnp.stack(values))))

    def infer(self, *, resampling: PermutationPlan) -> ComparisonInference:
        """Bind the declared residual-permutation design without testing a null."""
        if not isinstance(resampling, PermutationPlan):
            message = "Dispersion inference requires a PermutationPlan."
            raise TypeError(message)
        return ComparisonInference(self, resampling)

    def _resample(self, plan: PermutationPlan) -> ComparisonTestEvidence:  # noqa: PLR0914 -- retained evidence
        """Retain every requested draw and distinguish exact/random upper tails.

        DISCO and PERMANOVA evaluate label ``arrangements`` drawn from the plan's
        declared design (assignment units, exchangeability blocks). PERMDISP
        ``schedules`` index residuals rather than labels, admit only random
        draws and only ``blocks``: enumerating residual moves is not exact.
        """
        count = len(self.group_levels)
        arrangements: Array | None = None
        schedules: Array | None = None
        if self.method == "PERMDISP":
            if self.dispersion_radii is None:
                message = (
                    "PERMDISP inference requires retained radii"
                    f" (withheld: {self.reason})."
                    if self.reason
                    else "PERMDISP inference requires retained radii."
                )
                raise ValueError(message)
            if plan.method == "exact":
                message = (
                    "PERMDISP residual permutation is approximate; use a random plan."
                )
                raise ValueError(message)
            schedules, design, admissible = plan.schedules(
                self.geometry.row_ids, residual=True, observations=self.groups
            )
            draws = dispersion_residual_null(
                self.dispersion_radii, self.group_codes, count, schedules
            )
        else:
            arrangements, design, admissible = plan.arrangements(
                self.group_codes, self.geometry.row_ids, self.groups
            )
            values = self.geometry.values ** (2 if self.method == "PERMANOVA" else 1)
            draws = _disco_draws(values, arrangements, count)
        draw_valid = jnp.isfinite(draws)
        valid = self.valid and bool(jnp.all(draw_valid))
        requested = draws.shape[0]
        pvalue = None
        calibrated: ComparisonTest | None = None
        if valid:
            extreme = jnp.sum(draws >= self.statistic)
            pvalue_array = (
                extreme / requested
                if plan.method == "exact"
                else (extreme + 1) / (requested + 1)
            )
            pvalue = float(pvalue_array)
            provenance: (
                RandomCalibration
                | EnumerationCalibration
                | ResidualCalibration
                | SuppliedCalibration
                | SuppliedResidualCalibration
            ) = (
                CalibrationProvenance[
                    GroupedDesign,
                    NoGroupEffect,
                    ResampledCalibration,
                    SuppliedResidualSchedule,
                ]()
                if self.method == "PERMDISP" and plan.supplied
                else CalibrationProvenance[
                    GroupedDesign,
                    NoGroupEffect,
                    ResampledCalibration,
                    RandomResidualPermutation,
                ]()
                if self.method == "PERMDISP"
                else CalibrationProvenance[
                    GroupedDesign,
                    NoGroupEffect,
                    ResampledCalibration,
                    SuppliedOuterLabelSchedule,
                ]()
                if plan.supplied
                else CalibrationProvenance[
                    GroupedDesign,
                    NoGroupEffect,
                    ExactCalibration,
                    ExactOuterLabelPermutation,
                ]()
                if plan.method == "exact"
                else CalibrationProvenance[
                    GroupedDesign,
                    NoGroupEffect,
                    ResampledCalibration,
                    RandomOuterLabelPermutation,
                ]()
            )
            observed: RealizedStatistic[Array, ComparisonStatistic] | None
            if isinstance(self.core, DiscoDecomposition):
                observed = self.core.f_ratio
            elif self.method == "DISCO":
                observed = None
            elif self.method == "PERMANOVA":
                observed = RealizedStatistic(
                    value=self.statistic,
                    provenance=StatisticProvenance[
                        VarianceRatioStatistic, SquaredDistanceAnovaOrigin
                    ](),
                )
            else:
                observed = RealizedStatistic(
                    value=self.statistic,
                    provenance=StatisticProvenance[
                        VarianceRatioStatistic, SpatialMedianDispersionOrigin
                    ](),
                )
            if observed is not None:
                calibrated = CalibratedTest(
                    statistic=observed, pvalue=pvalue_array, provenance=provenance
                )
        return ComparisonTestEvidence(
            observed=self,
            arrangements=arrangements,
            schedules=schedules,
            design=design,
            admissible_arrangements=admissible,
            null=draws,
            draw_valid=draw_valid,
            method=plan.method,
            pvalue=pvalue,
            calibrated=calibrated,
            valid=valid,
            key=(plan.residual_key if self.method == "PERMDISP" else plan.key)
            if plan.method == "random" and plan.indices is None
            else None,
            seed=plan.seed
            if plan.method == "random" and plan.indices is None
            else None,
            prng_implementation=str(jax.config.jax_default_prng_impl)
            if plan.method == "random" and plan.indices is None
            else None,
            supplied_schedules=plan.supplied,
        )

    @override
    def report(self) -> Report:
        """Render only observed quantities and the retained interpretation."""
        available = self.valid
        metrics = [
            Metric("observations", len(self.geometry.row_ids)),
            Metric("groups", len(self.group_levels)),
            Metric(
                "F",
                float(self.statistic) if available else None,
                available=available,
                unavailable_reason=None
                if available
                else f"Withheld: {self.reason}."
                if self.reason
                else "Nonfinite observed statistic.",
                status="available" if available else "unavailable",
            ),
        ]
        if self.decomposition is not None:
            metrics.extend(
                Metric(
                    name,
                    float(value) if bool(jnp.isfinite(value)) else None,
                    available=bool(jnp.isfinite(value)),
                    unavailable_reason=None
                    if bool(jnp.isfinite(value))
                    else "Nonfinite decomposition.",
                    status="available" if bool(jnp.isfinite(value)) else "unavailable",
                )
                for name, value in zip(
                    ("within", "between", "total", "R2"),
                    self.decomposition[:-1],
                    strict=True,
                )
            )
        return Report(
            self.method,
            metrics=tuple(metrics),
            metadata={
                "observation_unit": "whole empirical measure"
                if isinstance(self.geometry, PairwiseGeometry)
                else "row of a supplied dissimilarity matrix",
                "row_ids": self.geometry.row_ids,
                "group_levels": self.group_levels,
                "geometry": repr(self.geometry.spec),
                "calibration": "not requested",
                "distance_power": 2 if self.method == "PERMANOVA" else 1,
                **(
                    {
                        "center": self._center_description(),
                        "bias_adjustment": "sqrt(n_g/(n_g-1))"
                        if self.bias_adjust
                        else None,
                    }
                    if self.method == "PERMDISP"
                    else {}
                ),
            },
        )

    def _center_description(self) -> str:
        """Describe the PERMDISP center that was actually estimated, if any."""
        if self.median_converged is None:
            withheld = f" (withheld: {self.reason})" if self.reason else ""
            return f"spatial median not estimated{withheld}"
        if self.median_converged:
            return "spatial median (stationarity-converged)"
        return "spatial median (not converged)"


@dataclass(frozen=True)
class ComparisonTestEvidence:
    """Complete permutation evidence, including unavailable draws and schedules."""

    observed: GroupComparisonResult
    arrangements: Array | None = field(repr=False)
    schedules: Array | None = field(repr=False)
    design: ExchangeabilityDesign = field(repr=False)
    admissible_arrangements: int
    null: Array = field(repr=False)
    draw_valid: Array = field(repr=False)
    method: Literal["random", "exact"]
    pvalue: float | None
    calibrated: ComparisonTest | None = field(repr=False)
    valid: bool
    key: Array | None = field(default=None, repr=False)
    seed: int | None = None
    prng_implementation: str | None = None
    supplied_schedules: bool = False

    def _repr_html_(self) -> str:
        """Render the report for notebook display."""
        return self.report()._repr_html_()

    @property
    def requested(self) -> int:
        """Return every attempted permutation, including failed numerical draws."""
        return self.null.shape[0]

    @property
    def failed(self) -> int:
        """Count nonfinite draws without removing their positions."""
        return int(jnp.sum(~self.draw_valid))

    def report(self) -> Report:
        """Keep failure-inclusive counts and the actual calibration rule explicit."""
        observed = self.observed.report()
        residual = self.observed.method == "PERMDISP"
        if self.supplied_schedules:
            calibration = (
                "caller-supplied residual schedules"
                if residual
                else "caller-supplied outer-label schedules"
            )
            null = (
                "caller-supplied schedules; exchangeability is asserted by their design"
            )
        else:
            calibration = (
                "full-model residual permutation inference requested"
                if residual
                else "outer-label permutation inference requested"
            )
            null = (
                "exchangeable one-way residuals of once-fitted radii"
                if residual
                else "outer-label exchangeability with fixed pooled geometry"
            )
        if not self.design.free:
            moved = self.design.description()
            null = (
                f"residuals of once-fitted radii exchangeable among {moved}"
                if residual
                else f"labels exchangeable among {moved}; fixed pooled geometry"
            )
        name = self.observed.method.lower()
        row = (
            TestRow(
                name,
                float(self.observed.statistic),
                self.pvalue,
                "resampled",
                (int(self.requested),),
                f"{self.method} permutation ({int(self.requested)} draws)",
            )
            if self.valid
            else TestRow(
                name,
                None,
                available=False,
                unavailable_reason=(
                    "Observed or requested null statistics are unavailable."
                ),
            )
        )
        return Report(
            f"{self.observed.method}: permutation inference",
            tests=(row,),
            metrics=(
                *(metric for metric in observed.metrics if metric.name != "F"),
                Metric("requested_permutations", int(self.requested)),
                Metric("failed_permutations", int(self.failed)),
                Metric("valid_permutations", int(self.requested - self.failed)),
            ),
            metadata={
                **dict(observed.metadata),
                "calibration": calibration,
                "permutation_method": self.method,
                "random_seed": self.seed,
                "prng_implementation": self.prng_implementation,
                "tail": ">= observed (ties included)",
                "pvalue_rule": "extreme / all"
                if self.method == "exact"
                else "(extreme + 1) / (all + 1)",
                "null": null,
                "exchangeability": self.design.metadata(self.admissible_arrangements),
                "qualification": "not established by this numerical result",
            },
            warnings=observed.warnings,
            plots=(
                PlotSpec(
                    "scatter",
                    tuple(range(self.requested)),
                    tuple(float(value) for value in self.null),
                    xlabel="Requested permutation",
                    ylabel="Null F-ratio",
                    title=f"{self.observed.method} permutation null",
                    reference_lines=(
                        ReferenceLine(
                            (0, self.requested - 1),
                            (float(self.observed.statistic),) * 2,
                            label="Observed F",
                        ),
                    ),
                ),
            )
            if self.valid
            else (),
        )


@dataclass(frozen=True)
class EqualGroupDispersion:
    """Equal dispersion about the fitted group centres under the retained scheme."""

    @staticmethod
    def describe() -> str:
        """State the group-dispersion question without replacing its resampling law."""
        return "Groups have equal dispersion about their group spatial medians."

    @staticmethod
    def metadata() -> dict[str, MetadataValue]:
        """Retain the supported hypothesis family."""
        return {"kind": "equal-group-dispersion"}


@dataclass(frozen=True)
class ComparisonInference:
    """A fitted group comparison and a bound, explicit residual schedule."""

    observed: GroupComparisonResult = field(repr=False)
    resampling: PermutationPlan

    def test(
        self, hypothesis: Hypothesis[EqualGroupDispersion]
    ) -> TestOutcome[EqualGroupDispersion, ComparisonTestEvidence]:
        """Evaluate the retained PERMDISP procedure without changing its centres."""
        if self.observed.method != "PERMDISP":
            message = (
                "This host supports PERMDISP; use geometric Regression "
                "for group effects."
            )
            raise TypeError(message)
        if not isinstance(hypothesis, Hypothesis) or not isinstance(
            hypothesis.null, EqualGroupDispersion
        ):
            message = "PERMDISP inference requires Hypothesis[EqualGroupDispersion]."
            raise TypeError(message)
        if hypothesis.alternative is not None:
            message = "PERMDISP supports its upper-tail complement alternative only."
            raise ValueError(message)
        evidence = self.observed._resample(self.resampling)  # noqa: SLF001 -- same-owner retained procedure
        return TestOutcome(
            "permdisp",
            hypothesis,
            float(self.observed.statistic) if evidence.valid else None,
            evidence.pvalue,
            ResampledReference(
                "supplied" if evidence.supplied_schedules else evidence.method,
                evidence.requested,
                correction="add-one",
                approximate=True,
            )
            if evidence.valid
            else None,
            evidence.valid,
            unavailable_reason=None
            if evidence.valid
            else "observed-or-null-dispersion-unavailable",
            details={
                "requested_draws": evidence.requested,
                "failed_draws": evidence.failed,
            },
            construction={
                "rows": _row_identity(self.observed.geometry.row_ids),
                "bias_adjust": self.observed.bias_adjust,
                "schedule_method": evidence.method,
                "supplied_schedules": evidence.supplied_schedules,
                "scheme": "full-model residual permutation of retained group radii",
                "random_seed": evidence.seed,
                "prng_implementation": evidence.prng_implementation,
                "tail": ">= observed (ties included)",
                "exchangeability": evidence.design.metadata(
                    evidence.admissible_arrangements
                ),
            },
            assumptions=(
                (
                    "The declared full-model residual permutation is appropriate; "
                    "its calibration is approximate."
                ),
            ),
            evidence=evidence,
        )

    def report(self) -> Report:
        """Describe the bound procedure without drawing or testing."""
        return Report(
            "PERMDISP: bound resampling",
            metadata={
                "test_status": "not requested",
                "schedule_method": self.resampling.method,
            },
        )
