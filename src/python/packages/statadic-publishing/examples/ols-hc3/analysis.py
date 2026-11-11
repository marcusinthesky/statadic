"""Fit the synthetic OLS example and write its retained evidence."""

import argparse
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import jax
import polars as pl
from statadic.api.artifacts import save_analysis
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.plots import residual_plot
from statadic.api.regression import Regression
from statadic_evidence import write_json


def analyze(data: Path, seed: int, output: Path) -> None:
    frame = pl.read_csv(data)
    if not jax.config.x64_enabled:
        frame = frame.cast({"x": pl.Float32, "y": pl.Float32})
    fit = (
        Regression(response="y", terms=Intercept() + Linear("x"))
        .estimate(frame)
        .require_valid()
    )
    output.mkdir(parents=True, exist_ok=True)
    save_analysis(output / "fit-snapshot", fit.report())
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    figure = residual_plot(fit)
    save_analysis(output / "inference-snapshot", replace(report, plots=(figure,)))
    report.save_latex(output / "table.tex", digits=4)
    figure.save_pgfplots(output / "figure.tex", width_cm=10)
    write_json(
        output / "methods.json",
        {
            "model": "OLS",
            "terms": ["Intercept", "Linear(x)"],
            "covariance": "HC3",
            "reference": "normal",
            "observations": frame.height,
            "seed": seed,
            "statadic": version("statadic"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.seed, args.output)
