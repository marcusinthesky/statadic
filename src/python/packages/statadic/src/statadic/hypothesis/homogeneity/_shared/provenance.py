"""Statistic and calibration provenance for group comparisons."""

from __future__ import annotations

from statadic.association.decomposition import (
    DiscoDecomposition,
    DiscoFRatioProvenance,
)
from statadic.core.domains import ConstructionOrigin, EmpiricalDistribution
from statadic.core.statistics import (
    CalibratedTest,
    CalibrationProvenance,
    ExactCalibration,
    GroupedDesign,
    NoGroupEffect,
    ResampledCalibration,
    StatisticProvenance,
    VarianceRatioStatistic,
)
from statadic.core.typing import Array  # Runtime annotations.

type NativeDecomposition = DiscoDecomposition[
    Array, EmpiricalDistribution, ConstructionOrigin
]


class SquaredDistanceAnovaOrigin(ConstructionOrigin):
    """One-factor ANOVA of the retained squared Hilbertian distance."""


class SpatialMedianDispersionOrigin(ConstructionOrigin):
    """ANOVA of distances to converged group spatial medians."""


class RandomOuterLabelPermutation(ConstructionOrigin):
    """Uniform draws from the declared relabelling group, add-one upper tail.

    The group is free relabelling of rows unless the plan declares assignment
    units (whole units move) or exchangeability blocks (labels stay in block).
    """


class ExactOuterLabelPermutation(ConstructionOrigin):
    """All distinct arrangements the declared design admits, observed included."""


class RandomResidualPermutation(ConstructionOrigin):
    """Uniform random full-model residual schedules with an add-one upper tail."""


class SuppliedOuterLabelSchedule(ConstructionOrigin):
    """Caller-supplied outer-label schedules with an add-one upper tail.

    The schedules' design, not STATADIC, asserts which relabellings are exchangeable.
    """


class SuppliedResidualSchedule(ConstructionOrigin):
    """Caller-supplied full-model residual schedules with an add-one upper tail.

    The schedules' design, not STATADIC, asserts which residual moves are exchangeable.
    """


type ComparisonStatistic = (
    DiscoFRatioProvenance[ConstructionOrigin]
    | StatisticProvenance[VarianceRatioStatistic, SquaredDistanceAnovaOrigin]
    | StatisticProvenance[VarianceRatioStatistic, SpatialMedianDispersionOrigin]
)
type RandomCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ResampledCalibration, RandomOuterLabelPermutation
]
type EnumerationCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ExactCalibration, ExactOuterLabelPermutation
]
type ResidualCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ResampledCalibration, RandomResidualPermutation
]
type SuppliedCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ResampledCalibration, SuppliedOuterLabelSchedule
]
type SuppliedResidualCalibration = CalibrationProvenance[
    GroupedDesign, NoGroupEffect, ResampledCalibration, SuppliedResidualSchedule
]
type ComparisonTest = CalibratedTest[
    Array,
    ComparisonStatistic,
    RandomCalibration
    | EnumerationCalibration
    | ResidualCalibration
    | SuppliedCalibration
    | SuppliedResidualCalibration,
]
