"""Frozen population numerical study design."""

from statadic_studies.study_types import Study

LONGRUN_STUDIES = {
    "longrun.gaussian-ar1-variability": Study(
        "longrun.gaussian-ar1-variability",
        "longrun",
        "Measure finite-bandwidth long-run variance estimation bias and risk.",
        "Stationary zero-mean unit-variance Gaussian AR(1) with rho=0 or .5, "
        "n=128 or 512. Use actual plain and Hall-centered Bartlett estimators "
        "on the same data with bandwidth 4 or the native n-based automatic rule.",
        "Infinite-lag long-run variance (1+rho)/(1-rho); fixed bandwidth "
        "has truncation bias. A long-run variance is not variance of the mean.",
        ("plain-fixed4", "centered-fixed4", "plain-auto", "centered-auto"),
        ("rho0-n128", "rho0-n512", "rho0.5-n128", "rho0.5-n512"),
        ("bias", "mse", "failures"),
        (
            "newey-west-1987",
            "hall-1987-source-review-pending",
        ),
    )
}
