# jaxtyping shape strings are runtime contracts.
"""Energy discrepancies induced by a declared ground geometry.

Use ``geometry`` to validate a ground/exponent pair, ``kernel`` to construct its
MMD kernel, ``scalar`` and ``matrix`` for energy functionals, ``statistics`` for
uncalibrated two-sample statistics, and ``mixture`` for fixed mixtures.
``balancing`` supplies surrogate-objective simplex weights and effective sizes.

Inputs are observation clouds and a ground distance. Outputs retain the energy
V-statistic convention and the supported law. Angular ground
cannot identify scale changes; select Euclidean geometry when scale matters.
Exact energy barycentres belong to ``statadic.geometry.barycentre.mmd.energy``;
resampling and test calibration belong to ``statadic.inference``.
"""

from __future__ import annotations

from statadic.discrepancy.mmd.energy.geometry import (
    ANOVA_EXPONENT,
    DEFAULT_DISTRIBUTION_EXPONENT,
    UNIT_EXPONENT,
    AnovaEnergyLaw,
    AnovaExponent,
    AnovaGroundLaw,
    DistributionEnergyGroundLaw,
    DistributionEnergyLaw,
    DistributionExponent,
    EnergyExponent,
    EnergyGeometry,
    FullFractionalPowerGroundLaw,
    MeanEnergyLaw,
    SeparatingDistributionEnergyGroundLaw,
    SeparatingDistributionEnergyLaw,
    SeparatingStrongNegativeTypeGroundLaw,
    SeparatingSubunitPowerGroundLaw,
    StrongFullFractionalPowerGroundLaw,
    SubunitExponent,
    SubunitPowerGroundLaw,
    SuperunitExponent,
    UnitExponent,
    UnitPowerMeanGroundLaw,
    energy_geometry,
    parse_distribution_exponent,
    parse_energy_exponent,
    unsafe_energy_geometry,
)
from statadic.discrepancy.mmd.energy.kernel import distance_induced_kernel
from statadic.discrepancy.mmd.energy.matrix import (
    empirical_mean_ground_distances,
    mean_distance_matrix,
)
from statadic.discrepancy.mmd.energy.matrix import (
    energy_distance_matrix as energy_distance_matrix_values,
)
from statadic.discrepancy.mmd.energy.matrix import (
    energy_functional_matrix as energy_distance_matrix,
)
from statadic.discrepancy.mmd.energy.mixture import (
    mixture_energy_distance,
    mixture_energy_distance_from_blocks,
)
from statadic.discrepancy.mmd.energy.scalar import (
    DECLARED_AXIOMS,
    DECLARED_BRANDS,
    IncompatibleEnergyGroundExponentError,
    energy_distance,
    energy_distance_with_geometry,
)
from statadic.discrepancy.mmd.energy.statistics import (
    energy_test_statistic,
    energy_test_statistic_from_blocks,
    energy_test_statistic_from_distances,
)

__all__ = [
    "ANOVA_EXPONENT",
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "DEFAULT_DISTRIBUTION_EXPONENT",
    "UNIT_EXPONENT",
    "AnovaEnergyLaw",
    "AnovaExponent",
    "AnovaGroundLaw",
    "DistributionEnergyGroundLaw",
    "DistributionEnergyLaw",
    "DistributionExponent",
    "EnergyExponent",
    "EnergyGeometry",
    "FullFractionalPowerGroundLaw",
    "IncompatibleEnergyGroundExponentError",
    "MeanEnergyLaw",
    "SeparatingDistributionEnergyGroundLaw",
    "SeparatingDistributionEnergyLaw",
    "SeparatingStrongNegativeTypeGroundLaw",
    "SeparatingSubunitPowerGroundLaw",
    "StrongFullFractionalPowerGroundLaw",
    "SubunitExponent",
    "SubunitPowerGroundLaw",
    "SuperunitExponent",
    "UnitExponent",
    "UnitPowerMeanGroundLaw",
    "distance_induced_kernel",
    "empirical_mean_ground_distances",
    "energy_distance",
    "energy_distance_matrix",
    "energy_distance_matrix_values",
    "energy_distance_with_geometry",
    "energy_geometry",
    "energy_test_statistic",
    "energy_test_statistic_from_blocks",
    "energy_test_statistic_from_distances",
    "mean_distance_matrix",
    "mixture_energy_distance",
    "mixture_energy_distance_from_blocks",
    "parse_distribution_exponent",
    "parse_energy_exponent",
    "unsafe_energy_geometry",
]
