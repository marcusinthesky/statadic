"""Geometry of points, measures and distance matrices.

Use ``ground`` for pairwise distances, similarities and their declared laws;
``embedding`` for distance-preserving coordinates; ``median`` for geometric
centres; and ``neighbour`` for local structure. ``barycentre`` constructs
mixtures or measures minimizing a declared discrepancy.

These children consume arrays or law-bearing distance matrices and retain
geometric diagnostics. Sampling uncertainty and resampling belong to
``statadic.inference``. Import the required child to load its numerical machinery.
"""

__all__: list[str] = []
