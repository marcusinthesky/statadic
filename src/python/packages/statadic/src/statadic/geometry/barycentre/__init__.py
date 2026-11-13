"""Barycentres minimizing discrepancies between measures.

Use ``mmd`` for target-directed simplex mixtures with a declared kernel,
``mmd.energy`` for the efficient distance-induced specialization, and
``wasserstein`` for measure and free-support transport barycentres. Each family
retains its objective-specific convergence certificate. Import the relevant
child; pointwise ground distances alone do not select a barycentre objective.
"""

__all__: list[str] = []
