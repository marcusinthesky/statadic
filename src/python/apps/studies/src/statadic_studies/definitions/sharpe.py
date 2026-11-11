"""Descriptive Sharpe inference designs; citation candidates await source review."""

from statadic_studies.study_types import Study

SHARPE_INFERENCE_STUDIES = {
    study.id: study
    for study in (
        Study(
            "sharpe.selection-scores",
            "sharpe",
            "Measure PSR and DSR score exceedance with and without selection.",
            "n=128; eight independent Gaussian strategy series with unit variance, "
            "each mean zero under null or .2 under alternative. Select the maximum "
            "sample Sharpe or the first prespecified configuration; calculate DSR "
            "using the measured sample variance across all eight Sharpe estimates.",
            "Marginal per-observation Sharpe zero or .2; scores are asymptotic "
            "normal-CDF statistics, not posterior probabilities. The DSR benchmark "
            "is estimated and score exceedance is descriptive, not an exact p-value.",
            ("unselected-PSR", "selected-PSR", "selected-DSR"),
            ("null", "alternative"),
            ("size", "power", "failures"),
            ("bailey-lopez-de-prado-2014",),
        ),
        Study(
            "sharpe.paired-difference",
            "sharpe",
            "Evaluate paired circular-bootstrap Sharpe-difference test.",
            "n=128; paired stationary Gaussian AR(1) returns, rho=0 or .5; "
            "innovations have cross-correlation .5, unit marginal variance; "
            "first mean zero or .2, second mean zero.",
            "Difference in population Sharpe zero under null and .2 under alternative.",
            ("paired-circular",),
            ("iid-null", "iid-alternative", "ar-null", "ar-alternative"),
            ("size", "power", "failures"),
            ("ledoit-wolf-2008",),
        ),
        Study(
            "sharpe.multiple-inference",
            "sharpe",
            "Measure actual Romano-Wolf family decisions and Reality Check p-values.",
            "n=128; three strategies and benchmark share Gaussian shocks with "
            "pairwise correlation .5; stationary AR(1) rho=0 or .5; strategy means "
            "all zero under null, or all .2 under alternative.",
            "All strategies have unit marginal variance, benchmark mean zero; "
            "family null means no strategy has positive population Sharpe difference.",
            ("romano-wolf", "white-reality-check"),
            ("iid-null", "iid-alternative", "ar-null", "ar-alternative"),
            ("size", "power", "failures"),
            ("romano-wolf-2005", "white-2000"),
        ),
        Study(
            "sharpe.stationary-interval",
            "sharpe",
            "Measure stationary-bootstrap Sharpe interval coverage.",
            "n=128 Gaussian stationary AR(1) rho=0 or .5, unit marginal variance "
            "and mean 0 or .2; automatic block selection.",
            "Per-observation marginal Sharpe equals the Gaussian mean.",
            ("stationary-percentile",),
            ("iid-zero", "iid-positive", "ar-zero", "ar-positive"),
            ("bias", "mse", "coverage", "width", "failures"),
            ("politis-romano-1994", "politis-white-2004"),
        ),
    )
}

SHARPE_INFERENCE_NULL_SCENARIOS = {
    "sharpe.selection-scores": ("null",),
    "sharpe.paired-difference": ("iid-null", "ar-null"),
    "sharpe.multiple-inference": ("iid-null", "ar-null"),
    "sharpe.stationary-interval": (),
}
