"""Frozen supplied-kappa or covariance-target designs."""

from statadic_studies.study_types import Study

KAPPA_STUDIES = {
    "kappa.supplied-delta": Study(
        "kappa.supplied-delta",
        "kappa",
        "Compare supplied-kappa delta variability with exact nonlinear variance.",
        "Fixed distance squared d2=1, marginal variances1; supplied kappa "
        "equals kappa0+U(-a,a), where kappa0=.5 or 1 and a=.2. Known "
        "predeclared SE=a/sqrt(3). Call the native covariance map and delta "
        "helper; this is not a fitted kappa estimator.",
        "Mapped covariance entry 1-.5*kappa0^2; exact entry variance "
        "(4*kappa0^2*a^2/3+4*a^4/45)/4. Delta variance uses supplied "
        "kappa_hat and is only a first-order approximation.",
        ("delta-covariance-entry",),
        ("kappa0.5", "kappa1"),
        ("bias", "mse", "se_accuracy", "failures"),
        ("kappa-delta-source-convention-review-pending",),
    ),
    "kappa.gaussian-concentration": Study(
        "kappa.gaussian-concentration",
        "kappa",
        "Describe actual moment-Jacobian sensitivity, not a weak-IV test.",
        "n128 iid N(0,1) moment observations, constant supplied derivative "
        "d=0 or .25, native uncentered Bartlett HAC bandwidth zero.",
        "Population plug-in sensitivity n*d^2; finite-sample inverse HAC "
        "makes actual concentration random. The threshold is descriptive.",
        ("moment-jacobian-concentration",),
        ("zero", "positive"),
        ("bias", "mse", "failures"),
        ("kappa-sensitivity-source-review-pending",),
    ),
}
