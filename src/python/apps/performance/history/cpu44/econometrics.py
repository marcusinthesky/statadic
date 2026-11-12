"""Matched CPU fit + covariance workloads from statadic commit 689d10a0e9.

Inputs are prepared before timing. New estimator/problem objects are created on
every call. Outputs are coefficients followed by row-major covariance entries,
converted to float64 NumPy on both sides (synchronizes JAX). Use JAX_ENABLE_X64=1.
"""

import numpy as np


def _result(params, covariance):
    return np.concatenate(
        (
            np.asarray(params, dtype=np.float64).ravel(),
            np.asarray(covariance, dtype=np.float64).ravel(),
        )
    )


def _ols(side, n=18, compiled=False):
    # tests/estimation/test_linear_reference.py::_sample and test_public_linear_workflow...
    time = np.arange(n, dtype=float)
    if n != 18:
        time *= 17 / (n - 1)  # Preserve the design scale of the 18-row fixture.
    x = np.column_stack((np.ones_like(time), time / 7, np.cos(time * 0.7)))
    y = x @ np.array([0.7, 1.3, -0.4]) + (0.2 + time / 30) * np.sin(time * 1.9)
    if side == "statadic":
        import jax.numpy as jnp
        from statadic.estimation.linear import LeastSquares, LinearProblem
        from statadic.inference.linear_covariance import hc_covariance

        x, y = jnp.asarray(x), jnp.asarray(y)

        def fit_and_covariance(design, outcome):
            fit = LeastSquares().estimate(LinearProblem(design, outcome))
            covariance = hc_covariance(fit, kind="HC3").covariance.values
            return fit.parameters, covariance

        if compiled:
            import jax

            fit_and_covariance = jax.jit(fit_and_covariance)

        def run():
            # Pass arrays as dynamic arguments to prevent constant folding the fixture.
            return _result(*fit_and_covariance(x, y))
    else:
        import statsmodels.api as sm

        def run():
            fit = sm.OLS(y, x).fit(cov_type="HC3", use_t=True)
            return _result(fit.params, fit.cov_params())

    return run


def _poisson(side):
    # tests/estimation/test_glm.py::_data and test_full_fit_and_covariance_match_statsmodels
    x = np.column_stack((np.ones(12), np.linspace(-1.5, 1.5, 12)))
    counts = np.array([0, 1, 2, 1, 3, 2, 4, 3, 6, 4, 5, 7], dtype=float)
    offset, exposure = np.linspace(-0.15, 0.2, 12), np.linspace(0.75, 1.75, 12)
    if side == "statadic":
        import jax.numpy as jnp
        from statadic.estimation.glm import PoissonMLE, PoissonProblem
        from statadic.inference.glm_covariance import glm_covariance

        x, counts, offset, exposure = map(jnp.asarray, (x, counts, offset, exposure))

        def run():
            fit = PoissonMLE().estimate(
                PoissonProblem(x, counts, offset=offset, exposure=exposure)
            )
            return _result(
                fit.parameters, glm_covariance(fit, method="HC0").covariance.values
            )
    else:
        import statsmodels.api as sm

        def run():
            model = sm.GLM(
                counts,
                x,
                family=sm.families.Poisson(),
                offset=offset,
                exposure=exposure,
            )
            fit = model.fit(maxiter=200, tol=1e-12, cov_type="HC0")
            return _result(fit.params, fit.cov_params())

    return run


def _panel(side, driscoll_kraay, entities=20, periods=10):
    # tests/inference/test_panel_covariance_reference.py::test_panel_covariance_matches_linearmodels
    rng = np.random.default_rng(126)
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    n = len(entity)
    x = rng.normal(size=(n, 2))
    y = (
        x @ np.array([0.5, -0.3])
        + rng.normal(size=entities)[entity]
        + rng.normal(size=n)
    )
    weights = np.ones(n)
    if side == "statadic":
        import jax.numpy as jnp
        from statadic.estimation.panel import PanelProblem, Within
        from statadic.inference.panel_covariance import panel_covariance
        from statadic.operators.panel import prepare_panel_index

        index = prepare_panel_index(entity, time)
        x, y, weights = map(jnp.asarray, (x, y, weights))
        method = "driscoll-kraay" if driscoll_kraay else "classical"

        def run():
            fit = Within().estimate(PanelProblem(x, y, index, weights))
            cov = panel_covariance(fit, method=method, bandwidth=2)
            return _result(fit.parameters, cov.covariance.values)
    else:
        import pandas as pd
        from linearmodels.panel import PanelOLS

        index = pd.MultiIndex.from_arrays([entity, time])
        x, y = pd.DataFrame(x, index=index), pd.Series(y, index=index)
        weights = pd.Series(weights, index=index)
        opts = (
            dict(cov_type="kernel", kernel="bartlett", bandwidth=2)
            if driscoll_kraay
            else dict(cov_type="unadjusted")
        )

        def run():
            model = PanelOLS(y, x, weights=weights, entity_effects=True)
            fit = model.fit(debiased=True, auto_df=False, count_effects=True, **opts)
            return _result(fit.params, fit.cov)

    return run


def _unbalanced_panel(side, method):
    # tests/estimation/test_panel_reference.py explicit seed 126, unit weights.
    rng = np.random.default_rng(126)
    e = np.repeat(np.arange(16), 8)
    t = np.tile(np.arange(8), 16)
    keep = ~((e % 3 == 0) & (t == 2))
    e, t = e[keep], t[keep]
    x = np.column_stack([np.ones(len(e)), rng.normal(size=(len(e), 2))])
    y = (
        x @ np.array([0.3, -0.5, 0.8])
        + rng.normal(size=16)[e]
        + rng.normal(size=len(e)) * 0.4
    )
    w = np.ones(len(e))
    if method == "fd":
        x = x[:, 1:]  # First differences remove the explicit constant.
    if side == "statadic":
        import jax.numpy as jnp
        from statadic.estimation.panel import (
            Between,
            FamaMacBeth,
            FirstDifference,
            PanelProblem,
            Pooled,
            RandomEffects,
            Within,
        )
        from statadic.inference.panel_covariance import (
            fama_macbeth_covariance,
            panel_covariance,
        )
        from statadic.operators.panel import prepare_panel_index

        index = prepare_panel_index(e, t)
        x, y, w = map(jnp.asarray, (x, y, w))
        estimator = {
            "pooled": Pooled,
            "twoway": Within,
            "re": RandomEffects,
            "between": Between,
            "fd": FirstDifference,
            "fmb": FamaMacBeth,
        }[method]
        options = {"intercept_column": 0} if method in ("twoway", "re") else {}
        if method == "twoway":
            options["time_effects"] = True
        covariance = fama_macbeth_covariance if method == "fmb" else panel_covariance

        def run():
            fit = estimator(**options).estimate(PanelProblem(x, y, index, w))
            return _result(fit.parameters, covariance(fit).covariance.values)
    else:
        import pandas as pd
        from linearmodels import panel as lm

        index = pd.MultiIndex.from_arrays([e, t], names=["entity", "time"])
        x = pd.DataFrame(x, index=index).sort_index(level="time")
        y = pd.Series(y, index=index).sort_index(level="time")
        w = pd.Series(w, index=index).sort_index(level="time")
        estimator = {
            "pooled": lm.PooledOLS,
            "twoway": lm.PanelOLS,
            "re": lm.RandomEffects,
            "between": lm.BetweenOLS,
            "fd": lm.FirstDifferenceOLS,
            "fmb": lm.FamaMacBeth,
        }[method]
        options = (
            {"entity_effects": True, "time_effects": True} if method == "twoway" else {}
        )

        def run():
            fit = estimator(y, x, weights=w, **options).fit()
            return _result(fit.params, fit.cov)

    return run


def _iv(side):
    # tests/estimation/test_gmm_reference.py::_sample and test_affine_iv_fit_and_sampling_covariance...
    rng = np.random.default_rng(942)
    z = rng.normal(size=(120, 4))
    shocks = rng.normal(size=(120, 3))
    x = (
        z @ np.array([[1.0, 0.1], [0.3, 0.4], [0.1, 1.2], [0.5, -0.3]])
        + shocks[:, :2] * 0.5
    )
    error = shocks[:, 0] * 0.3 + shocks[:, 2] * (0.3 + np.abs(z[:, 1]))
    y = (x @ np.array([0.7, -0.4]) + error)[:, None]
    if side == "statadic":
        import jax.numpy as jnp
        from statadic.core.equations import PlainLongRun
        from statadic.estimation.gmm import IteratedGMM
        from statadic.estimation.moments import AffineMomentProblem
        from statadic.inference.gmm.covariance import gmm_covariance

        z, x, y = map(jnp.asarray, (z, x, y))

        def run():
            # Include materializing affine score inputs from raw IV rows in the timed call.
            moments = z * y
            derivative = -z[:, :, None] * x[:, None, :]
            model = IteratedGMM(
                variability=PlainLongRun, bandwidth=2, max_iter=3, tolerance=1e-30
            )
            fit = model.estimate(AffineMomentProblem(moments, derivative))
            return _result(fit.parameters, gmm_covariance(fit).covariance.values)
    else:
        from linearmodels.iv import IVGMM

        opts = dict(kernel="bartlett", bandwidth=2, center=False, debiased=False)

        def run():
            model = IVGMM(y, None, x, z, weight_type="kernel", **opts)
            fit = model.fit(
                initial_weight=np.eye(4),
                iter_limit=4,
                tol=0.0,
                cov_type="kernel",
                **opts,
            )
            return _result(fit.params, fit.cov)

    return run


def prepare(case, side):
    if side not in ("statadic", "reference"):
        raise ValueError(side)
    if case == "ols_hc3":
        return _ols(side)
    if case == "ols_hc3_jit":
        return _ols(side, compiled=True)
    if case == "ols_hc3_10000_jit":
        return _ols(side, n=10000, compiled=True)
    if case == "ols_hc3_10000":
        return _ols(side, 10000)
    if case == "panel_fe_2000":
        return _panel(side, False, entities=100, periods=20)
    if case == "poisson_hc0":
        return _poisson(side)
    if case in ("panel_fe", "panel_dk"):
        return _panel(side, case == "panel_dk")
    if case in (
        "panel_pooled",
        "panel_twoway",
        "panel_re",
        "panel_between",
        "panel_fd",
        "panel_fmb",
    ):
        return _unbalanced_panel(side, case.split("_", 1)[1])
    if case == "iv_gmm":
        return _iv(side)
    raise ValueError(case)
