# jaxtyping shape strings are runtime contracts.
"""Wasserstein discrepancies between empirical measures.

Use ``exact`` for balanced equal-cardinality assignment distances and checked
eager entry points. Use ``univariate`` for exact weighted real-line transport
with unequal supports and ``sliced`` for finite-projection approximations with an
explicit projection design. The
algorithms retain different domains and order-validation exceptions; those
exceptions are qualified as ``ExactInvalidWassersteinOrderError`` and
``SlicedInvalidWassersteinOrderError`` at this family boundary.

Inputs are sample arrays or supplied costs; outputs are scalar distances. A
finite sliced design need not separate measures, and is not the exact transport
solver. Barycentre optimization belongs to ``statadic.geometry.barycentre``.
"""

from __future__ import annotations

from statadic.discrepancy.transport.exact import (
    EXACT_BALANCED_W1_AXIOMS,
    EXACT_BALANCED_W1_BRAND,
    EXACT_BALANCED_W2_AXIOMS,
    EXACT_BALANCED_W2_BRAND,
    EXACT_BALANCED_WASSERSTEIN_AXIOMS,
    EXACT_BALANCED_WASSERSTEIN_BRAND,
    EXPECTED_ARRAY_DIMENSIONS,
    UNIT_NORM_ATOL,
    WASSERSTEIN_ORDER_ONE,
    WASSERSTEIN_ORDER_TWO,
    EmptyOrNonMatrixSampleError,
    ExactTransportResult,
    InvalidCostMatrixError,
    NegativeCostError,
    NonfiniteCostError,
    NonfiniteSampleError,
    OffUnitSphereSampleError,
    UnbalancedSampleShapeError,
    ZeroNormSampleError,
    balanced_wasserstein1_from_cost,
    balanced_wasserstein1_from_cost_jax,
    balanced_wasserstein2_from_cost,
    balanced_wasserstein2_from_cost_jax,
    balanced_wasserstein_from_cost,
    balanced_wasserstein_from_cost_jax,
    balanced_wasserstein_from_cost_result,
    chord_cost_matrix,
    exact_balanced_wasserstein,
    exact_balanced_wasserstein1,
    exact_balanced_wasserstein1_jax,
    exact_balanced_wasserstein2,
    exact_balanced_wasserstein2_jax,
    exact_balanced_wasserstein_jax,
)
from statadic.discrepancy.transport.exact import (
    InvalidWassersteinOrderError as ExactInvalidWassersteinOrderError,
)
from statadic.discrepancy.transport.sliced import DECLARED_AXIOMS as SLICED_DECLARED_AXIOMS
from statadic.discrepancy.transport.sliced import DECLARED_BRANDS as SLICED_DECLARED_BRANDS
from statadic.discrepancy.transport.sliced import (
    InvalidProjectionCountError,
    ProjectionDesign,
    sliced_wasserstein,
    sliced_wasserstein_under_design,
    w1_1d,
    w2_1d,
)
from statadic.discrepancy.transport.sliced import (
    InvalidWassersteinOrderError as SlicedInvalidWassersteinOrderError,
)
from statadic.discrepancy.transport.univariate import wasserstein_1d, wasserstein_1d_power

__all__ = [
    "EXACT_BALANCED_W1_AXIOMS",
    "EXACT_BALANCED_W1_BRAND",
    "EXACT_BALANCED_W2_AXIOMS",
    "EXACT_BALANCED_W2_BRAND",
    "EXACT_BALANCED_WASSERSTEIN_AXIOMS",
    "EXACT_BALANCED_WASSERSTEIN_BRAND",
    "EXPECTED_ARRAY_DIMENSIONS",
    "SLICED_DECLARED_AXIOMS",
    "SLICED_DECLARED_BRANDS",
    "UNIT_NORM_ATOL",
    "WASSERSTEIN_ORDER_ONE",
    "WASSERSTEIN_ORDER_TWO",
    "EmptyOrNonMatrixSampleError",
    "ExactInvalidWassersteinOrderError",
    "ExactTransportResult",
    "InvalidCostMatrixError",
    "InvalidProjectionCountError",
    "NegativeCostError",
    "NonfiniteCostError",
    "NonfiniteSampleError",
    "OffUnitSphereSampleError",
    "ProjectionDesign",
    "SlicedInvalidWassersteinOrderError",
    "UnbalancedSampleShapeError",
    "ZeroNormSampleError",
    "balanced_wasserstein1_from_cost",
    "balanced_wasserstein1_from_cost_jax",
    "balanced_wasserstein2_from_cost",
    "balanced_wasserstein2_from_cost_jax",
    "balanced_wasserstein_from_cost",
    "balanced_wasserstein_from_cost_jax",
    "balanced_wasserstein_from_cost_result",
    "chord_cost_matrix",
    "exact_balanced_wasserstein",
    "exact_balanced_wasserstein1",
    "exact_balanced_wasserstein1_jax",
    "exact_balanced_wasserstein2",
    "exact_balanced_wasserstein2_jax",
    "exact_balanced_wasserstein_jax",
    "sliced_wasserstein",
    "sliced_wasserstein_under_design",
    "w1_1d",
    "w2_1d",
    "wasserstein_1d",
    "wasserstein_1d_power",
]
