"""Lifelines peers of tests/testing/test_survival_references.py at 689d10a."""
import numpy as np

SOURCE = "src/python/packages/statadic/tests/testing/test_survival_references.py"
CASES = {}
for family in ("weibull", "lognormal"):
    for variant in ("entry", "noentry", "unit", "pooled", "time", "two-way"):
        panel = variant not in ("entry", "noentry")
        name = "Weibull" if family == "weibull" else "lognormal"
        columns = {"unit": 9, "pooled": 2, "time": 31, "two-way": 38}.get(variant, 2)
        CASES[f"{family}-{variant}"] = {
            "method": f"{name} AFT" + (f", {variant} panel" if panel else (", delayed entry" if variant == "entry" else ", no entry")),
            "reference": f"lifelines.{'Weibull' if family == 'weibull' else 'LogNormal'}AFTFitter",
            "fixture": f"n={205 if panel else 240}, {columns} design columns + 1 ancillary parameter",
            "work": "Fit + log likelihood + observed-information covariance",
            "source": SOURCE,
            "note": "Delayed entry; unbalanced 8-entity/30-period spell design." if panel else "Seed 302601; right censored. Formula parsing inside lifelines fit. Predictions excluded.",
            "atol": 9e-4 if panel else 8e-5,
            "rtol": 2e-3 if panel else 5e-4,
        }


def _sample(family, delayed_entry):
    from scipy.stats import norm

    rng = np.random.default_rng(302601)
    x = rng.normal(size=240)
    mu = 0.2 - 0.3 * x
    entry = rng.uniform(0, 0.5, size=x.size) if delayed_entry else np.zeros(x.size)
    u = rng.uniform(size=x.size)
    if family == "weibull":
        event_time = np.exp(mu) * ((entry / np.exp(mu)) ** 1.5 - np.log(u)) ** (1 / 1.5)
    else:
        z_entry = (np.log(np.where(entry > 0, entry, 1)) - mu) / 0.7
        survival_at_entry = np.where(entry > 0, norm.sf(z_entry), 1)
        event_time = np.exp(mu + 0.7 * norm.isf(u * survival_at_entry))
    censor_time = entry + rng.exponential(2.0, size=x.size)
    return x, np.minimum(event_time, censor_time), event_time <= censor_time, entry


def _pack(parameters, likelihood, covariance):
    return np.concatenate((np.asarray(parameters), [float(likelihood)], np.asarray(covariance).ravel()))


def prepare(case, backend):
    family, variant = case.split("-", 1)
    panel = variant not in ("entry", "noentry")
    x, t, event, entry = _sample(family, variant != "noentry")
    if panel:
        entity, period = np.repeat(np.arange(8), 30), np.tile(np.arange(30), 8)
        keep = (entity + 2 * period) % 7 != 0
        x, t, event, entry, entity, period = (a[keep] for a in (x, t, event, entry, entity, period))

    if backend == "statadic":
        import jax.numpy as jnp
        from statadic.estimation.survival import SurvivalML, SurvivalProblem
        from statadic.inference.survival import observed_information
        from statadic.model.survival import WeibullAFT, LogNormalAFT

        estimator = SurvivalML(WeibullAFT() if family == "weibull" else LogNormalAFT())
        if panel:
            from statadic.api.data import Observations
            from statadic.api.design import Intercept, Linear
            from statadic.api.panel import PanelLayout
            from statadic.api.survival import RightCensored, SurvivalRegression

            data = Observations(dict(x=x, t=t, event=event.astype(np.int32), entry=entry, entity=entity, period=period))
            model = SurvivalRegression(
                RightCensored("t", "event", "entry"), Intercept() + Linear("x"),
                estimator, panel=PanelLayout("entity", "period"), effects=variant
            )
            prepared = model.prepare(data)
            def run_fit():
                return model.estimate_prepared(prepared).core
        else:
            problem = SurvivalProblem(jnp.asarray(np.column_stack([np.ones(x.size), x])),
                                      jnp.asarray(t), jnp.asarray(event), jnp.asarray(entry), ("intercept", "x"))
            def run_fit():
                return estimator.estimate(problem)

        def evaluate():
            result = run_fit()
            uncertainty = observed_information(result)
            if not bool(result.valid) or not bool(uncertainty.valid):
                raise ValueError("statadic AFT fit or inferred covariance invalid")
            return _pack(result.parameters, result.log_likelihood, uncertainty.covariance.values)
        return evaluate

    import pandas as pd
    from lifelines import WeibullAFTFitter, LogNormalAFTFitter

    if panel:
        # statadic's panel design sorts integer levels and drops the first with an intercept.
        columns = [x]
        if variant in ("unit", "two-way"):
            columns.extend(entity == e for e in np.unique(entity)[1:])
        if variant in ("time", "two-way"):
            columns.extend(period == p for p in np.unique(period)[1:])
        names = [f"c{i+1}" for i in range(len(columns))]
        frame = pd.DataFrame(np.column_stack(columns), columns=names)
    else:
        names = ["x"]
        frame = pd.DataFrame({"x": x})
    frame["t"], frame["event"], frame["entry"] = t, event, entry
    fitter, primary, ancillary = (
        (WeibullAFTFitter, "lambda_", "rho_") if family == "weibull"
        else (LogNormalAFTFitter, "mu_", "sigma_")
    )
    order = [(primary, "Intercept"), *((primary, name) for name in names), (ancillary, "Intercept")]
    formula = " + ".join(names)

    def evaluate():
        result = fitter().fit(frame, "t", "event", entry_col="entry", formula=formula)
        return _pack(result.params_.loc[order], result.log_likelihood_, result.variance_matrix_.loc[order, order])
    return evaluate
