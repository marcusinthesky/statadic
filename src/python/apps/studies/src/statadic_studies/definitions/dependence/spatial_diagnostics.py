"""Frozen native spatial spatial_diagnostics designs."""

from statadic_studies.definitions.dependence._spatial_design import (
    _CROSS_DESIGN,
    _TEST_METRICS,
)
from statadic_studies.study_types import Study

SPATIAL_DIAGNOSTICS_STUDIES = {
    "spatial.ols-diagnostics": Study(
        "spatial.ols-diagnostics",
        "spatial",
        "Evaluate native regression-adjusted Moran and lag/error specification "
        "diagnostics.",
        _CROSS_DESIGN + "OLS fit under iid null rho=lambda=0, SAR rho=.3 lambda=0, "
        "or SEM rho=0 lambda=.3. Native spatial_diagnostics on complete ordinary "
        "unweighted intercept OLS sample and exact W. All six actual diagnostics.",
        "Shared iid Gaussian null. Moran uses regression-adjusted normal "
        "reference; "
        "LM lag/error and competing-alternative-adjusted robust LM use chi- "
        "square1, "
        "joint SARMA chi-square2. Robust denotes competing lag/error adjustment, "
        "not heteroskedastic robustness. Lag/error alternatives examine distinct "
        "test sensitivities; not every test targets both departures.",
        (
            "Moran-normal",
            "lag-LM",
            "error-LM",
            "robust-lag-LM",
            "robust-error-LM",
            "joint-SARMA-LM",
        ),
        ("iid-null", "lag-alternative", "error-alternative"),
        _TEST_METRICS,
        ("Anselin LM/Moran diagnostic primary sources review pending",),
    ),
    "spatial.ak-diagnostic": Study(
        "spatial.ak-diagnostic",
        "spatial",
        "Evaluate actual Anselin-Kelejian test after spatial IV.",
        _CROSS_DESIGN + "SAR rho=.25; lambda0 null or .3 alternative, true x slopes "
        "unchanged. Actual SpatialIV with x1,x2 lag instruments, then "
        "spatial_ak_test "
        "retaining actual 2SLS instrument projection and residual correction.",
        "Null independent structural innovations, conditional fixed exogenous "
        "graph. Native chi-square1 AK test is not weak-instrument-robust or robust "
        "to arbitrary error dependence. No substituted OLS Moran statistic.",
        ("Anselin-Kelejian",),
        ("independent-errors", "error-alternative"),
        _TEST_METRICS,
        ("Anselin Kelejian 1997 primary source review pending",),
    ),
    "spatial.common-factor": Study(
        "spatial.common-factor",
        "spatial",
        "Evaluate native SDM nonlinear common-factor Wald test.",
        _CROSS_DESIGN + "rho=.25 and theta=-rho beta_slopes=(.125,-.2) under "
        "restriction-null; alternative theta=(.2,-.1). Actual SDM ML and "
        "information covariance then spatial_common_factor_test.",
        "Joint nonlinear restrictions theta_j+rho beta_j=0 for the two slopes; "
        "native gradient covariance and chi-square2. Intercept excluded from "
        "Durbin expansion. No claim that individually insignificant SLX terms "
        "establish common-factor equivalence.",
        ("common-factor-Wald",),
        ("restriction-null", "restriction-alternative"),
        _TEST_METRICS,
        ("spatial common-factor test primary source review pending",),
    ),
    "spatial.nested-likelihood-ratio": Study(
        "spatial.nested-likelihood-ratio",
        "spatial",
        "Evaluate explicitly nested interior SAR versus SDM likelihood ratio.",
        _CROSS_DESIGN + "rho=.25; theta=(0,0) under restriction-null or (.2,-.1) "
        "under alternative. Fit actual SAR restricted and SDM unrestricted on "
        "identical y,X,W. Native spatial_likelihood_ratio assumes nesting "
        "explicitly, degrees_of_freedom2, retaining boundary/data/rank failures.",
        "Two zero-SLX restrictions at interior rho; chi-square2 likelihood ratio "
        "calibration. Model labels alone are not evidence of nesting; designs "
        "here provide the declared actual restrictions.",
        ("interior-LR",),
        ("restriction-null", "restriction-alternative"),
        _TEST_METRICS,
        ("Wilks spatial LR formulation primary source review pending",),
    ),
}
