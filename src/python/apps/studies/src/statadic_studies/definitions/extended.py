"""Bounded descriptive designs for the additional regression workflows."""

from statadic_studies.study_types import Study

_LIKELIHOOD_METHODS = tuple(
    f"{dispersion}-{covariance}"
    for dispersion in ("estimated", "fixed")
    for covariance in ("model", "HC0", "HC1")
)
EXTENDED_STUDIES = {
    f"likelihood.{family}-inference": Study(
        f"likelihood.{family}-inference",
        "glm",
        f"Describe {family} slope inference and failures with explicit dispersion.",
        "Fixed intercept/linspace(-1,1) design, beta=(.2,.4), n=128; "
        "independent Gamma with variance .6*mean^2 or NB2 with variance "
        "mean+.6*mean^2. Fixed-dispersion methods receive the true .6. "
        "Each family is a separate study, not a comparison across outcomes.",
        "Log conditional-mean slope .4; normal-Wald intervals at .95.",
        _LIKELIHOOD_METHODS,
        ("ordinary",),
        ("bias", "mse", "coverage", "width", "failures"),
        ("statsmodels likelihood reference; convention review pending",),
        comparison_groups=(("same-log-mean-slope", _LIKELIHOOD_METHODS),),
        qualification="coverage-only",
    )
    for family in ("gamma", "nb2")
}
EXTENDED_STUDIES["likelihood.probit-inference"] = Study(
    "likelihood.probit-inference",
    "glm",
    "Describe binary and grouped-binomial probit slope inference and failures.",
    "n=128 fixed intercept/linspace(-1,1), beta=(.2,.4); independent "
    "binomial responses with trials1 or5 and probability Phi(X beta).",
    "Probit slope .4; normal-Wald intervals at .95.",
    ("model", "HC0", "HC1"),
    ("binary", "grouped"),
    ("bias", "mse", "coverage", "width", "failures"),
    ("statsmodels probit reference; convention review pending",),
    comparison_groups=(("same-probit-slope", ("model", "HC0", "HC1")),),
    qualification="coverage-only",
)
EXTENDED_STUDIES["survival.aft-mean-prediction"] = Study(
    "survival.aft-mean-prediction",
    "survival",
    "Describe fitted marginal means at a fixed predictor under censoring/entry.",
    "n=128, intercept/Uniform(-1,1), beta=(.2,.4); Weibull shape1.8, "
    "lognormal sigma .6 or log-logistic shape1.8. Independent censoring "
    "entry+Exponential(3); delayed-entry draws are conditional on T>entry "
    "with entry Uniform(0,.3). Spatial marginals and nonexistent means "
    "remain separate numerical tests rather than finite-target bias cells.",
    "Unconditional duration mean at predictor zero; family-specific analytic truth.",
    ("fitted-mean",),
    ("weibull", "lognormal", "loglogistic", "loglogistic-delayed"),
    ("bias", "mse", "failures"),
    ("flexsurv distributions reference; convention review pending",),
)
EXTENDED_STUDIES["iv.workflow-inference"] = Study(
    "iv.workflow-inference",
    "gmm",
    "Describe conventional IV slope inference across explicit dependence schemes.",
    "n=128, four independent standard-normal instruments; one endogenous "
    "regressor .8*z0+.4*z1+.5*s and outcome .7*d+u. Structural error "
    "u=.3*s+noise; independent, instrument-dependent heteroskedastic, "
    "stationary AR(.3), 16 independent clusters of8, or weak first stage "
    "with instrument loadings multiplied by .03. Cluster shocks are "
    "independent of instruments; HAC uses bandwidth4 in retained row order.",
    "First structural coefficient .7, with explicit normal-Wald approximation.",
    ("2SLS", "two-step", "iterated"),
    ("iid", "heteroskedastic", "hac", "clustered", "weak"),
    ("bias", "mse", "coverage", "width", "failures"),
    ("hansen-1982", "linearmodels IV reference; convention review pending"),
    comparison_groups=(("same-structural-slope", ("2SLS", "two-step", "iterated")),),
)
