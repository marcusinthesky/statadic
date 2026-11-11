"""Frozen population numerical study design."""

from statadic_studies.study_types import Study

SHARPE_MOMENT_STUDIES = {
    "sharpe.moment-risk": Study(
        "sharpe.moment-risk",
        "sharpe",
        "Measure bias and risk of native biased-moment skewness and kurtosis.",
        "Independent normal or standardized lognormal(sigma=.6) observations, "
        "n=64 or 256; native sample_skew_kurtosis on the same sample.",
        "Population skew and non-excess kurtosis, independently computed from "
        "normal or exact lognormal moments. Finite-sample bias is expected.",
        ("sample-skewness", "sample-kurtosis"),
        ("normal-n64", "normal-n256", "lognormal-n64", "lognormal-n256"),
        ("bias", "mse", "failures"),
        ("sample-moment-source-review-pending",),
    )
}
