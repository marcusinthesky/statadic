"""Permutation tests with explicit statistics, schedules and calibration.

Use ``test`` for generic or energy-specific tests and host report adapters.
Use ``kernel`` to assemble a statistic-plus-null result when the caller already
owns the permutation draws. Energy tests retain the selected ground geometry;
a test statistic alone does not warrant exchangeability.
"""

from statadic.inference.resampling.permutation.test import (
    EnergyPermutationOrigin,
    EnergyPermutationResult,
    EnergyStatisticOrigin,
    energy_permutation_test,
    energy_permutation_test_result,
    permutation_test_kernel,
    permutation_test_result,
)

__all__ = [
    "EnergyPermutationOrigin",
    "EnergyPermutationResult",
    "EnergyStatisticOrigin",
    "energy_permutation_test",
    "energy_permutation_test_result",
    "permutation_test_kernel",
    "permutation_test_result",
]
