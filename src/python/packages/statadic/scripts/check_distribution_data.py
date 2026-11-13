# ruff: noqa: INP001, S101  # standalone qualification checks
"""Check installed data adapters and analysis artifacts outside the checkout."""

import argparse
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import polars as pl
import pyarrow as pa
import statadic
from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.plots import prediction_line, residual_plot
from statadic.api.regression import Observed, Poisson, Regression


def _check_ols(output: Path) -> None:
    """Fit a Polars frame and retain its reports, plots and fitted state."""
    output.mkdir(parents=True)
    x = jnp.linspace(0.0, 5.0, 40)
    y = 1.0 + 0.8 * x + 0.4 * jax.random.normal(jax.random.key(17), x.shape)
    dtype = pl.Float64 if jax.config.x64_enabled else pl.Float32
    frame = pl.DataFrame(
        {"x": x.tolist(), "y": y.tolist()}, schema={"x": dtype, "y": dtype}
    )
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(frame).require_valid()
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    report.save_latex(output / "table.tex")
    plots = (residual_plot(fit), report.coefficient_plot())
    for name, plot in zip(("residual", "coefficients"), plots, strict=True):
        plot.save_pgfplots(output / f"{name}.tex", width_cm=10)
    fit.save_predictor(output / "predictor")
    new = {"x": [1.5, 2.5]}
    assert jnp.allclose(
        load_predictor(output / "predictor").predict(new), fit.predict(new)
    )
    save_analysis(output / "analysis", replace(report, plots=plots))
    assert load_analysis(output / "analysis").report.to_latex() == report.to_latex()
    for name in ("table.tex", "residual.tex", "coefficients.tex"):
        assert (output / name).stat().st_size > 0
    for name in ("predictor", "analysis"):
        assert (output / name / "manifest.json").stat().st_size > 0


def _check_poisson(output: Path) -> None:
    """Fit an Arrow table with exposure and retain its fitted artifacts."""
    output.mkdir(parents=True)
    load = jnp.linspace(0.0, 2.0, 80)
    exposure = 1.0 + load / 2.0
    mean = exposure * jnp.exp(0.5 + 0.4 * load)
    counts = jax.random.poisson(jax.random.key(23), mean)
    dtype = pa.float64() if jax.config.x64_enabled else pa.float32()
    table = pa.table(
        {
            "load": pa.array(load.tolist(), type=dtype),
            "exposure": pa.array(exposure.tolist(), type=dtype),
            "count": pa.array(counts.tolist(), type=pa.int32()),
        }
    )
    fit = (
        Regression(
            Observed("count"),
            Intercept() + Linear("load"),
            exposure="exposure",
            law=Poisson(),
        )
        .estimate(table)
        .require_valid()
    )
    report = fit.infer(covariance=HC("HC1")).report(reference="normal")
    report.save_latex(output / "table.tex")
    plot = prediction_line(
        partial(fit.predict, kind="rate"), "load", [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    plot.save_pgfplots(output / "rate.tex", width_cm=10)
    fit.save_predictor(output / "predictor")
    new = {"load": [0.5, 1.5], "exposure": [1.0, 2.0]}
    assert jnp.allclose(
        load_predictor(output / "predictor").predict(new), fit.predict(new)
    )
    save_analysis(output / "analysis", replace(report, plots=(plot,)))
    assert load_analysis(output / "analysis").report.to_latex() == report.to_latex()
    for name in ("table.tex", "rate.tex"):
        assert (output / name).stat().st_size > 0
    for name in ("predictor", "analysis"):
        assert (output / name / "manifest.json").stat().st_size > 0


def _main() -> None:
    """Check the installed wheel and write both workflows' artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert Path(statadic.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    _check_ols(args.output / "ols")
    _check_poisson(args.output / "poisson")


if __name__ == "__main__":
    _main()
