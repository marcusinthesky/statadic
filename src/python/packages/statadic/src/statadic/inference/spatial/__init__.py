"""Uncertainty, covariate impacts and specification tests for actual spatial fits.

Use ``covariance`` to derive sampling uncertainty from retained ML, IV, GMM or
spatial-panel state, and to compute exact dense direct/indirect/total impacts.
Use ``diagnostics`` for residual Moran, LM/AK, common-factor and declared
nested-likelihood comparisons. Each diagnostic retains its own model-specific
null and availability requirements.

Fitting belongs to ``statadic.estimation.spatial``; graph preparation belongs to
``statadic.operators.spatial``. Parameter intervals and linear restrictions compose
through ``statadic.inference.parameters`` after covariance selection. General-panel
covariance lives in ``statadic.inference.panel_covariance``. Numerical validity,
reference assumptions and statistical calibration remain separate.
"""

from statadic.inference.spatial.covariance import (
    SpatialImpacts,
    SpatialPanelImpacts,
    SpatialUncertainty,
    spatial_covariance,
    spatial_impacts,
    spatial_panel_impacts,
)
from statadic.inference.spatial.diagnostics import (
    SpatialDiagnostics,
    spatial_ak_test,
    spatial_common_factor_test,
    spatial_diagnostics,
    spatial_likelihood_ratio,
)

__all__ = [
    "SpatialDiagnostics",
    "SpatialImpacts",
    "SpatialPanelImpacts",
    "SpatialUncertainty",
    "spatial_ak_test",
    "spatial_common_factor_test",
    "spatial_covariance",
    "spatial_diagnostics",
    "spatial_impacts",
    "spatial_likelihood_ratio",
    "spatial_panel_impacts",
]
