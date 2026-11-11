from dataclasses import replace
from functools import partial
from pathlib import Path

import jax.numpy as jnp
from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.plots import prediction_line
from statadic.api.regression import Observed, Poisson, Regression

model = Regression(
    Observed("incidents"),
    Intercept() + Linear("load"),
    exposure="hours",
    law=Poisson(),
)
fit = model.estimate({"load": load, "hours": hours, "incidents": incidents})
report = fit.infer(covariance=HC("HC1")).report(reference="normal")
table = report.to_latex(digits=4)

grid = jnp.linspace(0.0, 5.0, 51)
figure = prediction_line(partial(fit.predict, kind="rate"), "load", grid,
                         xlabel="Facility load",
                         ylabel="Expected incidents per hour")
pgfplots = figure.to_pgfplots(width_cm=10, height_cm=5.5)

fit.save_predictor(Path("predictor"))       # prediction-only artifact
predictor = load_predictor(Path("predictor"))
save_analysis(Path("analysis"), replace(report, plots=(figure,)))
snapshot = load_analysis(Path("analysis"))  # report and render only
