"""Dispersion comparisons and reusable permutation plans over aligned geometry.

PERMDISP compares dispersion about group centres. Distance decomposition and
squared-distance ANOVA use geometric responses in ``api.regression``. Plans
declare the assignment unit and exchangeability blocks they relabel; supplied
distance matrices retain a weaker contract than constructed geometry.
"""

from statadic.api.comparison._models import PERMDISP
from statadic.api.comparison._results import (
    ComparisonCore,
    ComparisonInference,
    ComparisonTestEvidence,
    EqualGroupDispersion,
    GroupComparisonResult,
)
from statadic.api.comparison._shared import (
    ComparisonInput,
    ComparisonStatistic,
    ComparisonTest,
    DesignLabels,
    EnumerationCalibration,
    ExactOuterLabelPermutation,
    ExchangeabilityDesign,
    GroupLabels,
    NativeDecomposition,
    PermutationPlan,
    PlannedDraws,
    PreparedComparison,
    RandomCalibration,
    RandomOuterLabelPermutation,
    RandomResidualPermutation,
    ResidualCalibration,
    RowIds,
    SpatialMedianDispersionOrigin,
    SquaredDistanceAnovaOrigin,
    SuppliedCalibration,
    SuppliedOuterLabelSchedule,
    SuppliedResidualCalibration,
    SuppliedResidualSchedule,
)

__all__ = [
    "PERMDISP",
    "ComparisonCore",
    "ComparisonInference",
    "ComparisonInput",
    "ComparisonStatistic",
    "ComparisonTest",
    "ComparisonTestEvidence",
    "DesignLabels",
    "EnumerationCalibration",
    "EqualGroupDispersion",
    "ExactOuterLabelPermutation",
    "ExchangeabilityDesign",
    "GroupComparisonResult",
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
]
