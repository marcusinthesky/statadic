"""Energy specialization of kernel-MMD barycentres.

Use the cloud, pairwise or precomputed-component doors in ``weights`` to
approximate a target distribution by a simplex mixture. Outputs retain weights
and optional convergence diagnostics. The generic MMD family supplies the
shared simplex solver; no second objective or optimizer is introduced here.
"""

from statadic.geometry.barycentre.mmd.energy.weights import (
    energy_barycentre_weights,
    energy_barycentre_weights_batched,
    energy_barycentre_weights_pairwise,
)
from statadic.geometry.barycentre.mmd.solve import EnergyBarycentreDiagnostics

__all__ = [
    "EnergyBarycentreDiagnostics",
    "energy_barycentre_weights",
    "energy_barycentre_weights_batched",
    "energy_barycentre_weights_pairwise",
]
