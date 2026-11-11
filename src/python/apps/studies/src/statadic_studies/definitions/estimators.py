"""Additional descriptive estimator studies; source candidates require review."""

from statadic_studies.study_types import Study

ESTIMATOR_STUDIES = {
    study.id: study
    for study in (
        Study(
            "gmm.affine-mean",
            "gmm",
            "Measure parameter risk, Wald coverage and actual-fit J rejection, "
            "including misspecification.",
            "256 independent N(0,I3) rows minus (1,.5,-.25)*.4; misspecified "
            "scenario adds (0,.2,.2). Identity and iterated weights; centered "
            "bandwidth-zero variability.",
            "Correctly specified parameter .4; under misspecification .4 is the "
            "original structural benchmark, not an identified common moment root.",
            ("identity-GMM", "iterated-GMM", "Hansen-J"),
            ("correct", "misspecified"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("hansen-1982",),
        ),
        Study(
            "iv.structural-estimation",
            "gmm",
            "Compare strongly identified endogenous-regressor IV procedures and "
            "actual-fit J.",
            "n=256; four independent Gaussian instruments; one endogenous and one "
            "exogenous regressor with fixed full-rank loadings plus .5 Gaussian "
            "shocks; "
            "structural error shares .3 first shock. Independent noise scale 1 or "
            ".3+abs(z1).",
            "First structural coefficient .7; second -.4. Exclusion holds in both "
            "scenarios.",
            ("identity-GMM", "2SLS", "iterated-GMM", "Hansen-J"),
            ("homoskedastic", "heteroskedastic"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("hansen-1982",),
        ),
        Study(
            "glm.canonical-inference",
            "glm",
            "Evaluate canonical MLE and model/HC covariance normal-Wald inference.",
            "Fixed n=128 intercept/linspace(-1.5,1.5) design; beta=(.2,.4). "
            "Independent Binomial(5,logit inverse(X beta)) or Poisson(exp(X "
            "beta)); unit dispersion.",
            "Slope .4 on the canonical link scale; confidence level .95.",
            ("model", "HC0", "HC1"),
            ("binomial", "poisson"),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            (
                "nelder-wedderburn-1972",
                "white-1980",
            ),
        ),
        Study(
            "gel.fixed-moments",
            "gel",
            "Evaluate complete fixed-moment GEL LR null rejection and power with "
            "retained dual failures.",
            "Normal null: n=256,q=2 iid N(0,I); alternative mean=(.2,.2). Rare "
            "null: n=512,q=1 Bernoulli(.05)-.05. LR calibration explicitly assumes "
            "iid.",
            "Zero moment vector under both null scenarios; nonzero normal "
            "alternative. No fitted structural parameter.",
            ("cue", "el", "et"),
            ("normal-null", "normal-alternative", "rare-null"),
            ("size", "failures"),
            ("newey-smith-2004",),
        ),
        Study(
            "survival.independent-aft",
            "survival",
            "Evaluate independent AFT slope and observed-information normal-Wald "
            "coverage.",
            "n=128; intercept plus iid two Gaussian covariates; beta=(.2,.4,-.3). "
            "Weibull shape 1.5 or lognormal sigma .7; independent exponential "
            "censoring scale2 after entry. Entry zero or Uniform(0,1); event times "
            "sampled conditional on survival past entry.",
            "First covariate log-time coefficient .4 conditional on observed "
            "covariates and left truncation.",
            ("observed-information-Wald",),
            (
                "weibull-ordinary",
                "weibull-delayed",
                "lognormal-ordinary",
                "lognormal-delayed",
            ),
            ("bias", "mse", "coverage", "width", "size", "failures"),
            ("AFT source review pending",),
        ),
    )
}
