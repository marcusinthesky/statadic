# jaxtyping shape strings are runtime contracts.
"""Kernel-derived MMD laws and built-in executable declarations.

The rooted V-statistic law follows kernel capability. Squared estimates carry
no triangle guarantee, and signed U-statistics must not inherit a distance law.
Registries expose the same strategy objects used by the numeric entry points.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from statadic.core.axioms import (
    METRIC,
    PSEUDOMETRIC,
    Axioms,
    NegativeType,
    Premetric,
    Semimetric,
)
from statadic.discrepancy.mmd.energy.geometry import (
    UNIT_EXPONENT,
    energy_geometry,
)
from statadic.discrepancy.mmd.energy.kernel import distance_induced_kernel
from statadic.geometry.ground.metrics import (
    ANGULAR,
    EUCLIDEAN,
)
from statadic.geometry.ground.similarities import (  # runtime annotations
    COSINE_KERNEL,
    LINEAR_KERNEL,
    CharacteristicKernelLaw,
    PositiveSemidefiniteKernelLaw,  # noqa: TC001
    SimilarityStrategy,  # noqa: TC001
    rbf_kernel,
)

__all__ = [
    "BUILTIN_MMD_KERNELS",
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "rooted_mmd_axioms",
    "squared_mmd_axioms",
]


def rooted_mmd_axioms(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
) -> Axioms:
    """Derive the rooted MMD law from the kernel capability object."""
    if isinstance(kernel.law, CharacteristicKernelLaw):
        return METRIC
    return PSEUDOMETRIC


def squared_mmd_axioms(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
) -> Axioms:
    """Derive the squared-functional law without assigning triangle inequality."""
    result = Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY
    if isinstance(kernel.law, CharacteristicKernelLaw):
        result |= Axioms.SEPARATION
    return result


_RBF_KERNEL = rbf_kernel(1.0)
_EUCLIDEAN_ENERGY_KERNEL = distance_induced_kernel(
    energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
)
_ANGULAR_ENERGY_KERNEL = distance_induced_kernel(
    energy_geometry(ANGULAR, UNIT_EXPONENT)
)

BUILTIN_MMD_KERNELS: Final = MappingProxyType(
    {
        "rbf": _RBF_KERNEL,
        "distance_induced[euclidean]": _EUCLIDEAN_ENERGY_KERNEL,
        "distance_induced[angular]": _ANGULAR_ENERGY_KERNEL,
        "linear": LINEAR_KERNEL,
        "cosine": COSINE_KERNEL,
    }
)
"""Named documentation registry; values are the sole executable declarations."""

DECLARED_AXIOMS: Final = {
    **{
        f"mmd[{name}]": rooted_mmd_axioms(kernel)
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    **{
        f"mmd_squared[{name}]": squared_mmd_axioms(kernel)
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    "cosine_mean_mmd": METRIC,
}
"""Derived compatibility view used by the axiom battery; never hand-authored."""

DECLARED_BRANDS: Final = {
    **{
        f"mmd[{name}]": (
            NegativeType
            if isinstance(kernel.law, CharacteristicKernelLaw)
            else Premetric
        )
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    **{
        f"mmd_squared[{name}]": (
            Semimetric if isinstance(kernel.law, CharacteristicKernelLaw) else Premetric
        )
        for name, kernel in BUILTIN_MMD_KERNELS.items()
    },
    "cosine_mean_mmd": NegativeType,
}
"""Derived legacy marker view; typed producers use structural law protocols."""
