"""Exact balanced transport costs for unequal finite probability supports.

The supplied rectangular matrix already contains the powered ground costs.
The result is their optimal integral, without a Wasserstein root. This keeps
the numerical transportation objective separate from its ground geometry and
order. Equal total probability mass does not require equal support counts.
"""

from __future__ import annotations

from statadic.core.typing import Array  # noqa: TC001
from statadic.numerics.optimize.transportation import (
    TransportationResult,
    solve_balanced_transport,
)

__all__ = ["TransportationResult", "balanced_transport_from_cost_result"]


def balanced_transport_from_cost_result(
    cost: Array,
    left_masses: Array,
    right_masses: Array,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-6,
) -> TransportationResult:
    """Solve a finite balanced transport objective and retain primal/dual evidence.

    ``cost`` has shape ``(n_left, n_right)`` and contains finite nonnegative
    costs. The marginal arrays have the corresponding lengths and each sum
    to one. Invalid numerical inputs or an unavailable solve produce NaN in
    ``value``; callers must inspect ``input_valid`` and ``converged``. The
    fixed-shape numerical door supports JIT and leading-axis vmap, without a
    derivative contract through its discrete pivot choices.
    """
    return solve_balanced_transport(
        cost,
        left_masses,
        right_masses,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
