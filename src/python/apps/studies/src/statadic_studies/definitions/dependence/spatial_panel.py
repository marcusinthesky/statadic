"""Frozen native spatial spatial_panel designs."""

from statadic_studies.definitions.dependence._spatial_design import (
    _OBSERVED_DESIGN,
    _PARAMETER_METRICS,
)
from statadic_studies.study_types import Study

SPATIAL_PANEL_STUDIES = {
    "spatial.period-varying-panel": Study(
        "spatial.period-varying-panel",
        "spatial",
        "Evaluate genuine balanced period-varying PanelML pooled/fixed/random effects.",
        "Time-major6 periods by16 nodes. W_t=(.55+.04t)forward+(.45-.04t)backward. "
        "Two fixed Gaussian covariates, slope(-.5,.8); intercept.3 only pooled/RE. "
        "rho=.25, iid Gaussian innovation SD.5. Pooled alpha=0; fixed/random "
        "alpha_i~N(0,1) drawn once per whole panel and retained across periods. "
        "y_t=solve(I-rho W_t,X_t beta+alpha+epsilon_t). Actual PanelML effects "
        "match DGP. Filtering precedes effect profiling; no commuting W with "
        "demeaning.",
        "First slope -.5 and rho=.25, native information normal-Wald inference. "
        "RE effects independent of fixed covariates; fixed-effects study "
        "conditions "
        "on each replicate's generated effects. Variance-component boundary and "
        "information failures remain failures, not replaced with pooled inference.",
        ("beta", "rho"),
        ("pooled", "fixed", "random"),
        _PARAMETER_METRICS,
        ("period-varying spatial panel primary source review pending",),
    ),
    "spatial.observed-sem": Study(
        "spatial.observed-sem",
        "spatial",
        "Evaluate native observed-subvector pooled SEM likelihood.",
        _OBSERVED_DESIGN + "Full outcome X beta+solve(I-.3W_t,epsilon_t), select only "
        "declared observed indices. ObservedPanelML consumes a SEM/pooled problem "
        "with marginal observed-subvector Gaussian likelihood and observed "
        "covariance.",
        "First slope -.5 and structural lambda=.3. The omitted outcomes remain "
        "latent on the full graph; re-normalizing observed subgraphs would change "
        "the model and is not performed.",
        ("beta", "lambda"),
        ("fixed-outcome-selection",),
        _PARAMETER_METRICS,
        ("observed SEM primary source review pending",),
    ),
}
