"""Wasserstein barycentres of discrete empirical measures.

Use ``measure`` for constrained measure barycentres or alternating free-support
optimization. Inputs are source clouds, source weights and a declared ground
metric; results retain support, assignments, objective and convergence status.
These are optimization results. Exact pairwise transport distances live in
``statadic.discrepancy.transport`` and sampling uncertainty belongs to inference.
"""

from statadic.geometry.barycentre.wasserstein.measure import (
    WassersteinBarycentreResult,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)

__all__ = [
    "WassersteinBarycentreResult",
    "solve_wasserstein_free_support_barycentre",
    "solve_wasserstein_measure_barycentre",
]
