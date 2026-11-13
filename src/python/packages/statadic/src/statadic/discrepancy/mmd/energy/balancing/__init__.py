# jaxtyping shape strings are runtime contracts.
"""Surrogate energy balancing over candidate distributions.

Use ``weights`` for simplex weights and pairwise balancing, ``mixture`` to
evaluate a fixed mixture, ``statistics`` for uncalibrated energy statistics,
and ``effective_size`` for the inverse-variance mixture sample size.

Inputs are a target cloud, candidate clouds and a declared ground distance;
outputs are mixture weights or energy statistics. The surrogate balancing
objective differs from the exact barycentre objective in
``statadic.geometry.barycentre.mmd.energy``. Calibrate tests through inference.
"""

from __future__ import annotations

from statadic.discrepancy.mmd.energy.balancing.effective_size import (
    inverse_variance_m_eff,
)
from statadic.discrepancy.mmd.energy.balancing.mixture import (
    energy_distance_kernel_mixture,
)
from statadic.discrepancy.mmd.energy.balancing.statistics import (
    energy_test_kernel,
    energy_test_kernel_pairwise_max,
)
from statadic.discrepancy.mmd.energy.balancing.weights import (
    energy_distance_kernel_pairwise,
    kernel_balancing_weights,
)

__all__ = [
    "energy_distance_kernel_mixture",
    "energy_distance_kernel_pairwise",
    "energy_test_kernel",
    "energy_test_kernel_pairwise_max",
    "inverse_variance_m_eff",
    "kernel_balancing_weights",
]
