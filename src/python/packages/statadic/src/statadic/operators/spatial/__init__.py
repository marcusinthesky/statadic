"""Spatial operators for exogenous graphs and aligned panels.

Weights owns explicit dense/BCOO graph validation, spatial lags, checked solves,
and exact dense Jacobians. Moments owns the IV response map and quadratic
moment variability shared by spatial fitting and inference. Graph preparation
does not itself establish an econometric identification or calibration claim.
"""

from statadic.operators.spatial.weights import (
    ObservedSpatialPeriod,
    SpatialPanelWeights,
    SpatialWeights,
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
    spatial_lag,
    spatial_lags,
    spatial_logdet,
    spatial_panel_lag,
    spatial_solve,
)

__all__ = [
    "ObservedSpatialPeriod",
    "SpatialPanelWeights",
    "SpatialWeights",
    "prepare_spatial_panel_weights",
    "prepare_spatial_weights",
    "spatial_lag",
    "spatial_lags",
    "spatial_logdet",
    "spatial_panel_lag",
    "spatial_solve",
]
