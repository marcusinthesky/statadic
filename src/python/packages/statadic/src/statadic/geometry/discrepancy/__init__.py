"""Discrepancies between empirical measures.

Use ``mmd`` for kernel mean comparisons and its ``energy`` specialization,
``transport`` for exact or sliced Wasserstein distances, and ``metrize`` for
checked transformations of energy functionals. ``provenance`` describes the
sampling and origin contracts carried by discrepancy matrices.

Inputs are point clouds plus a declared ground distance or kernel. Outputs are
scalar estimates or matrices carrying the supported law and estimator scheme.
An unbiased signed MMD estimate is distinct from a rootable distance. Compose
these outputs with geometry, association or inference as appropriate.
"""

__all__: list[str] = []
