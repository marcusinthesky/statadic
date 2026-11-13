"""Workload definitions for the tables. All timing outputs are coefficients + covariance."""

_COMMIT = "689d10a0e933cc8adb137289878f5bd31c931d28"
_PREFIX = f"https://github.com/marcusinthesky/statadic/blob/{_COMMIT}/src/python/packages/statadic/tests/"
_linear = "estimation/test_linear_reference.py#L75-L152"
_panel = "estimation/test_panel_reference.py#L66-L140"
_cov = "inference/test_panel_covariance_reference.py#L25-L110"


def _row(name, reference, workload, size, path, rtol, atol, category="Fixture"):
    return dict(
        name=name,
        reference=reference,
        workload=workload,
        size=size,
        source=_PREFIX + path,
        rtol=rtol,
        atol=atol,
        category=category,
    )


META = {
    "ols_hc3": _row(
        "OLS with HC3",
        "statsmodels.OLS",
        "Fresh fit and HC3 covariance",
        "18 rows × 3 coefficients",
        _linear,
        1e-10,
        1e-12,
    ),
    "ols_hc3_10000": _row(
        "OLS with HC3",
        "statsmodels.OLS",
        "Fresh fit and HC3 covariance, time grid on [0, 17]",
        "10,000 rows × 3 coefficients",
        _linear,
        1e-10,
        1e-12,
        "Scale check",
    ),
    "poisson_hc0": _row(
        "Poisson GLM with HC0",
        "statsmodels.GLM (Poisson)",
        "Offset + exposure, fresh fit and HC0 covariance",
        "12 rows × 2 coefficients",
        "estimation/test_glm.py#L150-L202",
        1e-5,
        2e-7,
    ),
    "iv_gmm": _row(
        "Iterated IV-GMM",
        "linearmodels.iv.IVGMM",
        "3 updates from identity, uncentered Bartlett HAC 2, coefficients + covariance",
        "120 rows × 2 slopes, 4 instruments",
        "estimation/test_gmm_reference.py#L28-L104",
        1e-10,
        1e-12,
    ),
    "panel_fe": _row(
        "Entity fixed effects",
        "linearmodels.panel.PanelOLS",
        "Fresh within fit and classical covariance",
        "200 rows, 20 × 10 balanced panel, 2 slopes",
        _cov,
        1e-7,
        1e-9,
    ),
    "panel_dk": _row(
        "Entity fixed effects with Driscoll–Kraay",
        "linearmodels.panel.PanelOLS",
        "Fresh within fit and Driscoll–Kraay covariance, Bartlett bandwidth 2",
        "200 rows, 20 × 10 balanced panel, 2 slopes",
        _cov,
        1e-7,
        1e-9,
    ),
    "panel_fe_2000": _row(
        "Entity fixed effects",
        "linearmodels.panel.PanelOLS",
        "Fresh within fit and classical covariance",
        "2,000 rows, 100 × 20 balanced panel, 2 slopes",
        _cov,
        1e-7,
        1e-9,
        "Scale check",
    ),
}
for key, label, method in [
    ("panel_pooled", "Pooled panel OLS", "PooledOLS"),
    ("panel_twoway", "Entity and time fixed effects", "PanelOLS"),
    ("panel_re", "Random effects", "RandomEffects"),
    ("panel_between", "Between estimator", "BetweenOLS"),
    ("panel_fd", "First differences", "FirstDifferenceOLS"),
    ("panel_fmb", "Fama–MacBeth", "FamaMacBeth"),
]:
    count = 2 if key == "panel_fd" else 3
    detail = (
        "Period-coefficient fit and Fama–MacBeth covariance"
        if key == "panel_fmb"
        else "Fresh fit and classical covariance"
    )
    META[key] = _row(
        label,
        "linearmodels.panel." + method,
        detail,
        f"122 input rows, 16 × 8 panel, 6 missing, {count} coefficients",
        _panel,
        2e-7,
        1e-9,
    )
for key, original in (
    ("ols_hc3_jit", "ols_hc3"),
    ("ols_hc3_10000_jit", "ols_hc3_10000"),
):
    META[key] = dict(
        META[original],
        category="User JIT sensitivity",
        name="OLS with HC3, explicit whole-function JIT",
        workload=(
            "Design and outcome are dynamic inputs to jax.jit. The compiled function calls "
            "LeastSquares.fit and hc_covariance(kind='HC3'), returning only coefficients and "
            "the covariance matrix. Host conversion and flattening are timed outside JIT. "
            "The reference calls the same statsmodels OLS as the default case."
        ),
    )
