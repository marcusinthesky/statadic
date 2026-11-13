"""Fit spatial regressions and balanced static spatial panels.

Bricks fit a ``SpatialProblem``: ``SLXLeastSquares`` and ``ML(SAR() | SEM() |
SAC())`` with selected Durbin ``slx_columns`` (``likelihood``); ``SpatialIV`` for
instrumented spatial lags (``iv``); and ``SpatialGMM`` for declared quadratic
error moments (``gmm``). Extra endogenous arrays go on a ``SpatialIVProblem``.
``panel`` fits pooled, fixed or random effects with a constant or exogenous
time-varying graph. ``kkp`` is the distinct constant-W component
model, which spatially filters the entity effect as well as its innovation.

Inputs are aligned designs/outcomes and explicitly prepared spatial weights.
Fits retain the graph, parameter schema, actual solve state and separate
identification/availability evidence. General static panels belong to
``statadic.estimation.panel``. Compose an actual spatial fit with
``statadic.inference.spatial`` for covariance, impacts and diagnostics; solver
convergence alone does not authorize inference.
"""

from statadic.estimation.spatial.gmm import SpatialGMM, SpatialGMMFit
from statadic.estimation.spatial.iv import SpatialIV, SpatialIVFit, SpatialIVProblem
from statadic.estimation.spatial.kkp import KKP, KKPPanelFit
from statadic.estimation.spatial.likelihood import (
    ML,
    SAC,
    SAR,
    SEM,
    SLXLeastSquares,
    SpatialFit,
    SpatialProblem,
)
from statadic.estimation.spatial.observed_panel import (
    ObservedPanelML,
    ObservedSpatialPanelFit,
    ObservedSpatialPanelProblem,
    observed_panel_profile,
)
from statadic.estimation.spatial.panel import (
    PanelML,
    SpatialPanelFit,
    SpatialPanelProblem,
    spatial_panel_profile,
)

__all__ = [
    "KKP",
    "ML",
    "SAC",
    "SAR",
    "SEM",
    "KKPPanelFit",
    "ObservedPanelML",
    "ObservedSpatialPanelFit",
    "ObservedSpatialPanelProblem",
    "PanelML",
    "SLXLeastSquares",
    "SpatialFit",
    "SpatialGMM",
    "SpatialGMMFit",
    "SpatialIV",
    "SpatialIVFit",
    "SpatialIVProblem",
    "SpatialPanelFit",
    "SpatialPanelProblem",
    "SpatialProblem",
    "observed_panel_profile",
    "spatial_panel_profile",
]
