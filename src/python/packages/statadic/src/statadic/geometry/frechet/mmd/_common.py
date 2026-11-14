"""Private numerical constants shared by MMD and energy barycentre solvers."""

from __future__ import annotations

from typing import Final

# Reduced-QP solver constants for the exact energy barycentre.
_QP_MAX_ITER: Final = (
    500  # ill-conditioned 51-candidate projections need a larger IP budget
)
_MAX_DISTANCE_EXPONENT: Final = 2.0
_DEGENERATE_CURVATURE: Final = 1e-14
_MATRIX_NDIM: Final = 2
_STACKED_CLOUD_NDIM: Final = 3
_MIN_CANDIDATES: Final = 2
