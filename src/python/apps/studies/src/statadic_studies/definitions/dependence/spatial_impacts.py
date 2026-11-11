"""Frozen native spatial spatial_impacts designs."""

from statadic_studies.definitions.dependence._spatial_design import (
    _CROSS_DESIGN,
    _IMPACT_METRICS,
    _OBSERVED_DESIGN,
)
from statadic_studies.study_types import Study

SPATIAL_IMPACTS_STUDIES = {
    "spatial.cross-section-impacts": Study(
        "spatial.cross-section-impacts",
        "spatial",
        "Evaluate native exact direct/indirect/total impact delta uncertainty.",
        _CROSS_DESIGN + "SAR rho=.25 with theta=0 or SDM theta=(.2,-.1). Native "
        "ML and spatial_covariance -> spatial_impacts. Report first-covariate "
        "impacts using actual native Jacobian covariance, normal95% intervals.",
        "Fixed-graph mean derivatives solve(I-.25W,-.5I+theta1 W). Direct=trace/n, "
        "total=sum/n, indirect=total-direct. Truth from independent NumPy full "
        "matrix solve; no causal interpretation or derivative of fitted "
        "parameters.",
        ("direct", "indirect", "total"),
        ("SAR", "Durbin"),
        _IMPACT_METRICS,
        ("LeSage Pace spatial impacts primary source review pending",),
    ),
    "spatial.observed-panel-impacts": Study(
        "spatial.observed-panel-impacts",
        "spatial",
        "Evaluate actual active-period panel impacts and observed-information "
        "delta uncertainty.",
        _OBSERVED_DESIGN + "Full y_t=solve(I-.25W_t,X_t beta+epsilon_t), actual "
        "ObservedPanelML SAR pooled fit, observed covariance and "
        "spatial_panel_impacts. "
        "Report first covariate/first period's full-active-graph derivative and "
        "SE.",
        "First-period fixed full-graph derivative solve(I-.25W0,-.5I): trace/n, "
        "off-diagonal sum/n, full sum/n. This is not an average of period SEs or "
        "an observed-only-subgraph impact; actual observed-information delta95% "
        "CI.",
        ("direct", "indirect", "total"),
        ("fixed-outcome-selection",),
        _IMPACT_METRICS,
        ("observed panel impact source review pending",),
    ),
}
