"""Fit synthetic counts with exposure and export a reproducible analysis."""

import argparse
from tempfile import TemporaryDirectory

# tutorial-start
from dataclasses import replace
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import pyarrow as pa
from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.plots import prediction_line
from statadic.api.regression import Regression, Observed, Poisson


def analyze(output: Path) -> None:
    """Fit counts, inspect inference, export and reload the analysis."""
    output.mkdir(parents=True, exist_ok=True)
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
        Regression(Observed('count'), Intercept() + Linear('load'), exposure='exposure', law=Poisson()).estimate(table).require_valid()
    )
    print(fit.report().to_text())

    report = fit.infer(covariance=HC("HC1")).report(reference="normal")
    print(report.to_text())
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
