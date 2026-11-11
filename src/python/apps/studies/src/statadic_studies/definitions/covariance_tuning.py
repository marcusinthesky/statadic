"""Frozen supplied-kappa or covariance-target designs."""

from statadic_studies.study_types import Study

COVARIANCE_TUNING_STUDIES = {
    "covariance.entry-variance": Study(
        "covariance.entry-variance",
        "covariance-estimation",
        "Measure native fourth-moment sample covariance entry variance.",
        "n64 or256 iid N4(0,Sigma), Sigma has unit diagonal and all "
        "off-diagonal correlations .3. Native fourth-moment estimator on ddof0 "
        "sample covariance; record first diagonal entry.",
        "Var(S_11) = 2*(n-1)/n^2 for the ddof0 Gaussian sample covariance.",
        ("fourth-moment-entry-variance",),
        ("n64", "n256"),
        ("bias", "mse", "failures"),
        ("sample-covariance-variance-source-review-pending",),
    ),
    "covariance.target-tuning-risk": Study(
        "covariance.target-tuning-risk",
        "covariance-estimation",
        "Compare native sample weight rules using actual covariance risk.",
        "n64 iid N4 with unit diagonal and correlations .3. Fixed target is "
        "I4 or 1.5I4; compare sample ddof0, native fixed-target optimal "
        "plug-in, actual out-of-fold portfolio-variance CV (grid 0,.5,1; "
        "four folds), and same-window diagonal target with native plug-in. "
        "Same-window and fixed targets have different dependence assumptions.",
        "Population covariance Sigma; trace error and squared Frobenius loss. "
        "CV selects out-of-fold portfolio variance, not Frobenius risk.",
        ("sample", "fixed-target-optimal", "fixed-target-CV", "same-window-plugin"),
        ("fixed-identity", "fixed-scaled"),
        ("bias", "mse", "frobenius_loss", "failures"),
        ("covariance-target-tuning-source-review-pending",),
    ),
    "covariance.target-tuning-portfolio": Study(
        "covariance.target-tuning-portfolio",
        "covariance-estimation",
        "Compare selected covariance rules by known population portfolio risk.",
        "Same paired Gaussian panels and target rules as target-tuning-risk; "
        "build minimum-variance weights from each chosen covariance with "
        "native ridge_psd(eps=1e-8), then evaluate actual w.T Sigma w.",
        "Oracle minimum unit-sum portfolio variance 1/(1.T Sigma^-1 1); "
        "Nonnegative excess risk is a loss, not covariance-estimator bias or "
        "Frobenius risk; roundoff-negative excess is clipped to zero.",
        ("sample", "fixed-target-optimal", "fixed-target-CV", "same-window-plugin"),
        ("fixed-identity", "fixed-scaled"),
        ("loss", "failures"),
        ("covariance-target-tuning-source-review-pending",),
    ),
}
