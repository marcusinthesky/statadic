"""Concrete moment-fitting/test variants and integration metadata overrides."""

from statadic_studies.study_types import Study

_AFFINE_METHODS = tuple(
    name
    for criterion in ("cue", "el", "et")
    for name in (criterion + "-parameter", criterion + "-profiled-LR")
)
_FIXED_METHODS = tuple(
    criterion + "-" + kind
    for criterion in ("cue", "el", "et")
    for kind in ("lr", "lm", "wald")
)
_GMM_METHODS = (
    "fixed-plain-Wald",
    "iterated-plain-Wald",
    "fixed-centered-Wald",
    "iterated-centered-Wald",
    "actual-fit-plain-J",
    "actual-fit-centered-J",
    "fixed-parameter-plain-J",
    "fixed-parameter-centered-J",
)

MOMENT_INFERENCE_STUDIES = {
    study.id: study
    for study in (
        Study(
            "gel.affine-profile",
            "gel",
            "Evaluate actual affine structural-parameter GEL fits and separately "
            "available profiled LR overidentification tests.",
            "n=128 independent Gaussian three-dimensional moment errors with "
            "covariance diag(1,2,.5); derivative d=(1,.5,-.25). g0=error-.4*d, or "
            "add(-.1,.2,0) under misspecification; shift orthogonal to d. "
            "AffineGEL cue/el/et uses native eager multiplier and outer "
            "minimization defaults; no fixed-moment substitution.",
            "Interior structural parameter .4 under correct specification; "
            "original structural benchmark under misspecification, where no common "
            "population root exists. Parameter point risk is separate from "
            "profiled LR chi-square2 testing. No parameter covariance is invented.",
            _AFFINE_METHODS,
            ("correct", "misspecified"),
            ("bias", "mse", "size", "power", "failures"),
            ("newey-smith-2004",),
        ),
        Study(
            "gel.fixed-test-forms",
            "gel",
            "Compare every public fixed GEL LR/LM/Wald form using the same "
            "complete dual fit per carrier.",
            "Normal null:128 iid N(0,I2) rows; alternative mean(.25,.25). Rare "
            "null:512 iid Bernoulli(.05)-.05 scalar rows. Actual FixedGEL "
            "cue/el/et fit followed by public gel_test lr,lm,wald with explicit "
            "iid acknowledgement. All solver/rank failures retained.",
            "Zero population moment vector for both null scenarios; shifted normal "
            "alternative. Effective moment rank controls degrees of freedom. "
            "Native LM and Wald share the same uncentered moment quadratic and are "
            "not claimed independent tests.",
            _FIXED_METHODS,
            ("normal-null", "normal-alternative", "rare-null"),
            ("size", "power", "failures"),
            ("newey-smith-2004",),
        ),
        Study(
            "gmm.variability-and-J",
            "gmm",
            "Evaluate actual fixed/iterated centered/plain GMM parameter "
            "inference, retained-fit efficient J and genuinely "
            "fixed-parameter J.",
            "Same n128 affine Gaussian model and orthogonal misspecification shift "
            "as affine GEL. Fixed weights identity; iterated native defaults; "
            "centered/plain bandwidth-zero variability. Wald normal intervals use "
            "actual-fit sandwich. Retained-fit J enforces efficient-weight "
            "provenance and fixed-point convergence; fixed-parameter hansen_j "
            "uses theta=.4 and k=0.",
            "Structural parameter .4 under correct specification; original "
            "benchmark under misspecification. Actual-fit J has two "
            "overidentifying df; genuinely fixed-parameter J has three. Correct- "
            "model acknowledgements describe the null reference and are "
            "intentionally violated in misspecification diagnostics. No fixed- "
            "weight fit receives efficient-J provenance.",
            _GMM_METHODS,
            ("correct", "misspecified"),
            ("bias", "mse", "coverage", "width", "size", "power", "failures"),
            ("hansen-1982", "newey-smith-2004"),
        ),
    )
}

# Older Study carriers do not yet have per-method metrics or null-scenario
# fields. The integration layer applies these explicit frozen overrides.
MOMENT_NULL_SCENARIOS = {
    "gel.affine-profile": ("correct",),
    "gel.fixed-test-forms": ("normal-null", "rare-null"),
    "gmm.variability-and-J": ("correct",),
}
MOMENT_METHOD_METRICS = {
    "gel.affine-profile": {
        method: (
            ("bias", "mse", "failures")
            if method.endswith("-parameter")
            else ("size", "power", "failures")
        )
        for method in _AFFINE_METHODS
    },
    "gel.fixed-test-forms": {
        method: ("size", "power", "failures") for method in _FIXED_METHODS
    },
    "gmm.variability-and-J": {
        method: (
            ("bias", "mse", "coverage", "width", "size", "power", "failures")
            if method.endswith("-Wald")
            else ("size", "power", "failures")
        )
        for method in _GMM_METHODS
    },
}


FIXED_MOMENT_HAC_STUDIES = {
    "gmm.fixed-moment-hac-wald": Study(
        "gmm.fixed-moment-hac-wald",
        "gmm",
        "Evaluate the actual fixed-moment multivariate HAC Wald procedure.",
        "256 ordered observations of a stationary Gaussian vector AR(1). Initial "
        "state N(0,Sigma), innovations N(0,(1-phi^2)Sigma), Sigma=((1,.4,0), "
        "(.4,1,.2),(0,.2,1)). phi0 or .6; population moment mean zero under "
        "null or shifted(.25,0,-.15). Shared data across native gmm_wald bandwidth "
        "0,8,16 and automatic sample-size rule. Plain, uncentered Bartlett "
        "Newey-West variability. Separate rank-deficient adverse null sets the "
        "third moment to the sum of the first two; host full-rank rejection "
        "is retained.",
        "H0 E[g_t]=0 for three supplied fixed moments, chi-square3 reference. "
        "No estimated-parameter degrees-of-freedom subtraction or Hansen-J "
        "substitution. Fixed finite bandwidth can underaccount serial correlation; "
        "b0 is an explicit adverse comparator under AR1. Full-rank variability "
        "required for native calibration; singular scenario is an "
        "availability stress.",
        ("NW-b0", "NW-b8", "NW-b16", "NW-auto"),
        ("iid-null", "AR1-null", "iid-shift", "AR1-shift", "rank-deficient-null"),
        ("size", "power", "failures"),
        ("hansen-1982", "newey-west-1987"),
    )
}
FIXED_MOMENT_HAC_NULL_SCENARIOS = {
    "gmm.fixed-moment-hac-wald": ("iid-null", "AR1-null", "rank-deficient-null")
}
