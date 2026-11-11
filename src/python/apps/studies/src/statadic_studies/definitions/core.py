"""Baseline fixed statistical designs."""

from statadic_studies.study_types import Study

STUDIES = {
    "covariance.population-risk": Study(
        "covariance.population-risk",
        "covariance-estimation",
        "Compare covariance trace bias and Frobenius risk, retaining "
        "target misspecification.",
        "n=64,p=4 Gaussian observations; SD=(.5,1,1.5,2); constant correlation .3 "
        "or block correlations .7 and -.5; mean=(0,1,2,3).",
        "Known population covariance; trace target is sum of marginal variances.",
        ("sample-ddof0", "sample-ddof1", "ledoit-wolf-constant-correlation"),
        ("matched-target", "misspecified-target"),
        ("bias", "mse", "frobenius_loss", "failures"),
        ("Ledoit and Wolf 2004",),
        comparison_groups=(
            (
                "population-covariance",
                ("sample-ddof0", "sample-ddof1", "ledoit-wolf-constant-correlation"),
            ),
        ),
    ),
    "sharpe.point-estimation": Study(
        "sharpe.point-estimation",
        "sharpe",
        "Evaluate Sharpe point bias and MSE under known marginal moments.",
        "n=128 returns with population mean .2 and variance 1: iid Gaussian, "
        "standardized lognormal(sigma=.6), or stationary Gaussian AR(1,rho=.5).",
        "Population marginal mean divided by marginal SD = .2; no annualization.",
        ("sample-sharpe",),
        ("gaussian", "skewed", "dependent"),
        ("bias", "mse", "failures"),
        ("Sharpe 1966", "Lo 2002"),
    ),
    "linear.ols-covariance": Study(
        "linear.ols-covariance",
        "linear",
        "Compare fixed-design slope bias and coverage across covariance providers.",
        "n=64,256; intercept and fixed linspace(-2,2); Gaussian errors; "
        "SD=1 (homoskedastic) or 0.3+abs(x) (heteroskedastic). "
        "Student-t/F(df=n-2) and normal/chi2 references are separate scenarios; "
        "robust covariance does not imply an exact finite-sample reference law.",
        "Slope beta=0, conditional on a fixed full-rank design.",
        ("classical", "HC0", "HC1", "HC2", "HC3"),
        tuple(
            f"{law}-n{n}-{reference}"
            for law in ("homoskedastic", "heteroskedastic")
            for n in (64, 256)
            for reference in ("t", "normal")
        ),
        ("bias", "mse", "coverage", "width", "size", "failures"),
        ("White 1980", "MacKinnon and White 1985"),
        qualification="size-and-coverage",
        comparison_groups=(("ols-slope", ("classical", "HC0", "HC1", "HC2", "HC3")),),
    ),
    "mmd.finite-bias": Study(
        "mmd.finite-bias",
        "mmd",
        "Evaluate U/V squared-MMD bias with an enumerated population target.",
        "Independent samples from support {-1,0,2}; p=(.2,.5,.3); "
        "q=p (null) or (.5,.2,.3) (alternative); n,m=(8,8) or (8,13).",
        "Linear-kernel population MMD squared = (E_P X - E_Q Y)^2.",
        ("U", "V"),
        ("null-equal", "null-unequal", "alternative-equal", "alternative-unequal"),
        ("bias", "mse", "failures"),
        ("Gretton et al. 2012",),
        comparison_groups=(("linear-mmd-squared", ("U", "V")),),
    ),
}
