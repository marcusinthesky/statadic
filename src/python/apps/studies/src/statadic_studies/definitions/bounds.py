"""Frozen inequality, boundary-reference and whole-network bootstrap designs."""

from statadic_studies.study_types import Study

BOUNDS_DYADIC_STUDIES = {
    study.id: study
    for study in (
        Study(
            "inequalities.gaussian-operating-characteristics",
            "moment-inequalities",
            "Evaluate actual uncorrected indicator-GMS QLR/MMM and Wolak chi-bar "
            "tests under binding/slack/violated moment inequalities.",
            "n128 iid Gaussian moment rows with covariance [[1,.35],[.35,1]]. "
            "Means(0,0), (0,.35), (.35,.35) satisfy E[g]>=0; violated "
            "mean(-.2,.35). Native bandwidth0; inner calibration default999draws, "
            "explicit smoke32; independent named method streams on shared "
            "datasets. Wolak projection max_sweeps256; GMS eta0 is uncorrected, "
            "never Table-I RMS-certified.",
            "Null E[g]>=0; size/rejection under all binding/slack nulls, power "
            "under the violated alternative. Uses actual GMS reject decision and "
            "Wolak p<=.05; numerical projection/calibration failures count as "
            "attempted procedure failures.",
            ("indicator-GMS-QLR", "indicator-GMS-MMM", "Wolak-chi-bar"),
            ("binding-null", "partial-slack-null", "interior-null", "violated"),
            ("size", "power", "failures"),
            (
                "Andrews and Soares 2010",
                "Andrews and Barwick 2012",
                "Wolak 1989 source convention review pending",
            ),
        ),
        Study(
            "kappa.gaussian-boundary-tail",
            "kappa",
            "Calibrate the public positive-part boundary-tail helper in an "
            "explicitly known Gaussian experiment.",
            "n64 independent N(kappa,1) observations; constrained Gaussian mean "
            "estimate=max(samplemean,0); known preconstraint Gaussian SE1/sqrt64. "
            "Boundary null kappa0 or positive alternative .25. Invoke actual "
            "boundary_robust_pvalue at null0; inclusive boundary atom returns1.",
            "Nonnegative Gaussian mean kappa. This is reference-law calibration "
            "for the public helper, not a fit of STATADIC's correlation-gap/kappa "
            "estimator or evidence for arbitrary supplied standard errors.",
            ("positive-part-Gaussian-tail",),
            ("boundary-null", "positive-alternative"),
            ("bias", "mse", "size", "power", "failures"),
            ("Andrews 2001 boundary source review pending",),
        ),
        Study(
            "dyadic.node-bootstrap-inference",
            "dyadic",
            "Evaluate native multinomial node-bootstrap interval coverage and "
            "zero-slope decisions for pooled/node-fixed-effects fits.",
            "Complete undirected16/32-node networks, upper-triangle edges. "
            "Standard Gaussian node covariates and latent node shocks; "
            "focal=abs(c_i-c_j); y=.4+beta*focal+a_i+a_j+.3 Gaussian edge shock; "
            "beta0 or.6. Independent whole networks are replication units. Native "
            "DyadicEstimator then node_bootstrap_report, endpoint-count-product "
            "weights and required focal column. Default999draws; explicit smoke32; "
            "same node schedule across methods and null/alternative. x64 required.",
            "Structural dyadic slope beta, confidence level.95 and zero-slope "
            "p-value from actual bootstrap report. Require every requested draw "
            "solve-valid and focal-finite; retain counts otherwise. No edge- "
            "independent surrogate or bootstrap qualification from registration. "
            "At32draws minimum two-sided p=2/33>.05, so smoke cannot establish "
            "testing power.",
            ("pooled-node-bootstrap", "node-FE-bootstrap"),
            ("null-n16", "alternative-n16", "null-n32", "alternative-n32"),
            ("bias", "mse", "coverage", "width", "size", "power", "failures"),
            ("dyadic multinomial node-bootstrap seminal source review pending",),
        ),
    )
}
BOUNDS_DYADIC_NULL_SCENARIOS = {
    "inequalities.gaussian-operating-characteristics": (
        "binding-null",
        "partial-slack-null",
        "interior-null",
    ),
    "kappa.gaussian-boundary-tail": ("boundary-null",),
    "dyadic.node-bootstrap-inference": ("null-n16", "null-n32"),
}
BOUNDS_DYADIC_DRAW_STUDIES = (
    "inequalities.gaussian-operating-characteristics",
    "dyadic.node-bootstrap-inference",
)
BOUNDS_DYADIC_DECISION_CONVENTIONS = {
    "inequalities.gaussian-operating-characteristics": "GMS native reject; "
    "Wolak pvalue <= alpha",
    "kappa.gaussian-boundary-tail": "pvalue <= alpha",
    "dyadic.node-bootstrap-inference": "pvalue <= alpha",
}
