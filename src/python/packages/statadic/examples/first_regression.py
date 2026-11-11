"""Fit synthetic OLS data and export a reproducible analysis."""

import argparse
from tempfile import TemporaryDirectory

# tutorial-start
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import polars as pl
from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import RESET, BreuschPagan, Influence, JarqueBera
from statadic.api.inference import HC
from statadic.api.plots import residual_plot
from statadic.api.regression import Regression


def analyze(output: Path) -> None:
    """Estimate, inspect, diagnose, infer, export and reload the analysis."""
    output.mkdir(parents=True, exist_ok=True)
    x = jnp.linspace(0.0, 5.0, 40)
    y = 1.0 + 0.8 * x + 0.4 * jax.random.normal(jax.random.key(17), x.shape)
    dtype = pl.Float64 if jax.config.x64_enabled else pl.Float32
    frame = pl.DataFrame(
        {"x": x.tolist(), "y": y.tolist()}, schema={"x": dtype, "y": dtype}
    )
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(frame).require_valid()
    print(fit.report().to_text())

    diagnostics = fit.diagnose(JarqueBera(), BreuschPagan(), RESET(), Influence())
    print(diagnostics.report().to_text())

    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    print(report.to_text())
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


# tutorial-end


def main() -> None:
    """Write outputs to the requested directory or a temporary directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        with TemporaryDirectory() as directory:
            analyze(Path(directory))
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
