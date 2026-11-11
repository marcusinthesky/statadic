"""Kernel-MMD barycentres over candidate distributions.

Use ``weights`` to solve a target-directed simplex problem from a declared PSD
kernel; use ``energy`` for distance-induced reductions and batched energy doors.
The shared ``solve`` machinery retains curvature and convergence diagnostics.
Outputs are mixture weights, not sampling uncertainty; inference owns any
resampling or calibration applied to them.
"""

from __future__ import annotations

from statadic.geometry.barycentre.mmd.solve import KernelBarycentreDiagnostics
from statadic.geometry.barycentre.mmd.weights import kernel_barycentre_weights

__all__ = [
    "KernelBarycentreDiagnostics",
    "kernel_barycentre_weights",
]
