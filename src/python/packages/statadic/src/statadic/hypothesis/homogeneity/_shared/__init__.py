"""Aligned inputs, provenance and permutation plans for group comparisons.

``inputs`` aligns geometry and labels; ``provenance`` names what statistics and
calibrations mean; ``design`` resolves the declared relabelling group;
``draws`` counts, enumerates and samples its arrangements; ``plan`` is the
public ``PermutationPlan``.
"""

from statadic.hypothesis.homogeneity._shared.design import (
    DesignLabels,
    ExchangeabilityDesign,
    PlannedDraws,
)
from statadic.hypothesis.homogeneity._shared.draws import _disco_draws
from statadic.hypothesis.homogeneity._shared.inputs import (
    _MIN_GROUPS,
    ComparisonInput,
    GroupLabels,
    PreparedComparison,
    RowIds,
    _geometry,
    _grouping,
)
from statadic.hypothesis.homogeneity._shared.plan import PermutationPlan
from statadic.hypothesis.homogeneity._shared.provenance import (
    ComparisonStatistic,
    ComparisonTest,
    EnumerationCalibration,
    ExactOuterLabelPermutation,
    NativeDecomposition,
    RandomCalibration,
    RandomOuterLabelPermutation,
    RandomResidualPermutation,
    ResidualCalibration,
    SpatialMedianDispersionOrigin,
    SquaredDistanceAnovaOrigin,
    SuppliedCalibration,
    SuppliedOuterLabelSchedule,
    SuppliedResidualCalibration,
    SuppliedResidualSchedule,
)

__all__ = [
    "_MIN_GROUPS",
    "ComparisonInput",
    "ComparisonStatistic",
    "ComparisonTest",
    "DesignLabels",
    "EnumerationCalibration",
    "ExactOuterLabelPermutation",
    "ExchangeabilityDesign",
    "GroupLabels",
    "NativeDecomposition",
    "PermutationPlan",
    "PlannedDraws",
    "PreparedComparison",
    "RandomCalibration",
    "RandomOuterLabelPermutation",
    "RandomResidualPermutation",
    "ResidualCalibration",
    "RowIds",
    "SpatialMedianDispersionOrigin",
    "SquaredDistanceAnovaOrigin",
    "SuppliedCalibration",
    "SuppliedOuterLabelSchedule",
    "SuppliedResidualCalibration",
    "SuppliedResidualSchedule",
    "_disco_draws",
    "_geometry",
    "_grouping",
]
