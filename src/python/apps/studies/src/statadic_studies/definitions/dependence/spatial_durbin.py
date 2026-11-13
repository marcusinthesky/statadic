"""Frozen native spatial spatial_durbin designs."""

from statadic_studies.definitions.dependence._spatial_design import (
    _CROSS_DESIGN,
    _PARAMETER_METRICS,
)
from statadic_studies.study_types import Study

SPATIAL_DURBIN_STUDIES = {
    "spatial.durbin-likelihood": Study(
        "spatial.durbin-likelihood",
        "spatial",
        "Evaluate actual SDM/SDEM/general-nesting Gaussian Durbin ML "
        "information inference.",
        _CROSS_DESIGN + "Add spatially lagged x1,x2 with theta=(.2,-.1). SDM rho=.25, "
        "lambda=0; SDEM rho=0,lambda=.3; GNS rho=.25,lambda=.3. Actual ML with "
        "a SAR, SEM or SAC process and slx_columns=(1,2) on SpatialProblem, "
        "not SLX regression substituted for Durbin ML.",
        "First slope -.5, first SLX coefficient .2 and last spatial parameter "
        "rho=.25 for SDM or lambda=.3 for SDEM/GNS. Actual expected-information "
        "normal-Wald intervals/tests; no finite-sample certification.",
        ("beta", "theta", "last-spatial"),
        ("SDM", "SDEM", "GNS"),
        _PARAMETER_METRICS,
        ("Durbin/general-nesting primary source review pending",),
    )
}
