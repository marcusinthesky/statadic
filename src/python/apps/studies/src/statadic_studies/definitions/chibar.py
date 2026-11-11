"""Frozen population numerical study design."""

from statadic_studies.study_types import Study

CHIBAR_STUDIES = {
    "inequalities.chibar-weight-accuracy": Study(
        "inequalities.chibar-weight-accuracy",
        "moment-inequalities",
        "Measure the stochastic native chi-bar weight-vector approximation.",
        "Independent keyed inner Gaussian calibration with identity covariance "
        "of dimension three; default 999 draws, smoke 32; native constrained "
        "projection and its convergence indicator are retained.",
        "Binding-coordinate weights are Binomial(3,.5): (1,3,3,1)/8. "
        "Each outer replicate is one independent weight approximation.",
        ("weight-0", "weight-1", "weight-2", "weight-3"),
        ("identity-3",),
        ("bias", "mse", "failures"),
        ("wolak-1989-source-convention-review-pending",),
    )
}
