# jaxtyping shape strings are runtime contracts.
"""Kernel mean discrepancies on empirical measures.

Use ``scalar`` for signed MMD-squared estimates and rooted distances, ``matrix``
for packed array reductions, ``branded`` for law-bearing matrices, and
``functional`` for checked float64 prototype comparisons. ``laws`` records
kernel-derived declarations; ``components``
exposes candidate reductions used by barycentre solvers. ``energy`` specializes
MMD to validated distance-induced kernels and energy V-statistics.

Inputs are sample clouds and a declared similarity strategy. Rooted outputs
require the nonnegative V-statistic; unbiased U-statistic estimates remain
signed. Feed valid distance matrices to geometry or association and use
``statadic.inference`` for calibration. Kernel laws and estimator conventions are
preserved by the curated exports below.
"""

from __future__ import annotations

from statadic.discrepancy.mmd.branded import mmd_squared_estimate_matrix
from statadic.discrepancy.mmd.branded import rooted_mmd_matrix as mmd_matrix
from statadic.discrepancy.mmd.contracts import (
    InsufficientUStatisticSampleError,
    NonfiniteMeanEmbeddingError,
    UnbiasedMmdDistanceError,
    ZeroNormMeanEmbeddingError,
)
from statadic.discrepancy.mmd.energy.kernel import distance_induced_kernel
from statadic.discrepancy.mmd.functional import cosine_mean_mmd_functional
from statadic.discrepancy.mmd.laws import (
    BUILTIN_MMD_KERNELS,
    DECLARED_AXIOMS,
    DECLARED_BRANDS,
    rooted_mmd_axioms,
    squared_mmd_axioms,
)
from statadic.discrepancy.mmd.matrix import mmd_matrix_values, mmd_squared_matrix
from statadic.discrepancy.mmd.scalar import cosine_mean_mmd, mmd, mmd_squared
from statadic.geometry.ground.similarities import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    CharacteristicKernelLaw,
    CharacteristicKernelProperties,
    PositiveSemidefiniteKernelLaw,
    PositiveSemidefiniteKernelProperties,
    SimilarityStrategy,
    declare_similarity,
    rbf_kernel,
)

__all__ = [
    "BUILTIN_MMD_KERNELS",
    "COSINE_KERNEL",
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "LINEAR_KERNEL",
    "CharacteristicKernelLaw",
    "CharacteristicKernelProperties",
    "InsufficientUStatisticSampleError",
    "NonfiniteMeanEmbeddingError",
    "PositiveSemidefiniteKernelLaw",
    "PositiveSemidefiniteKernelProperties",
    "SimilarityStrategy",
    "UnbiasedMmdDistanceError",
    "ZeroNormMeanEmbeddingError",
    "cosine_mean_mmd",
    "cosine_mean_mmd_functional",
    "declare_similarity",
    "distance_induced_kernel",
    "mmd",
    "mmd_matrix",
    "mmd_matrix_values",
    "mmd_squared",
    "mmd_squared_estimate_matrix",
    "mmd_squared_matrix",
    "rbf_kernel",
    "rooted_mmd_axioms",
    "squared_mmd_axioms",
]
