"""One OLS analysis shared by the textbook chapter and native teaching lab."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.diagnostics import (
    RESET,
    BreuschGodfrey,
    BreuschPagan,
    DurbinWatson,
    Influence,
    JarqueBera,
    White,
)
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.inference import HAC, HC, ModelBased
from statadic.api.plots import residual_plot
from statadic.api.regression import Regression


# ols-data-start
def regression_data(noise=0.6, heterogeneity=0.0, outlier=False, seed=17):
    """Generate 80 fixed-design rows without changing NumPy's global RNG."""
    noise, heterogeneity = float(noise), float(heterogeneity)
    if not np.isfinite(noise) or not 0.1 <= noise <= 2:
        raise ValueError("noise must be finite and between 0.1 and 2.")
    if not np.isfinite(heterogeneity) or not 0 <= heterogeneity <= 2:
        raise ValueError("heterogeneity must be finite and between 0 and 2.")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    x = np.linspace(-2, 2, 80)
    errors = np.random.default_rng(seed).normal(size=len(x))
    y = 1 + 2 * x + noise * (1 + heterogeneity * np.abs(x)) * errors
    if outlier:
        y = y.copy()
        y[-1] += 12
    return x, y


# ols-data-end


def analyze(*, noise=0.6, heterogeneity=0.0, outlier=False, seed=17, bandwidth=3):
    """Return the actual fit, diagnostics and explicitly chosen inference objects.

    This function has no file outputs and does not change process precision.
    The synthetic time key declares positional order; it establishes no time
    series assumptions. Controls preserve the original regression lab's data.
    """
    # ols-fit-start
    x, y = regression_data(noise, heterogeneity, outlier, seed)
    observations = Observations(
        {"x": jnp.asarray(x), "y": jnp.asarray(y), "time": jnp.arange(len(x))}
    )
    fit = (
        Regression.ols("y", Intercept() + Linear("x"))
        .estimate(observations)
        .require_valid()
    )
    fit_report = fit.report()
    # ols-fit-end

    # ols-diagnose-start
    diagnostics = fit.diagnose(
        JarqueBera(),
        BreuschPagan(name="breusch_pagan_studentized"),
        BreuschPagan(studentized=False, name="breusch_pagan_gaussian"),
        White(),
        BreuschGodfrey(lags=2, order_by="time"),
        DurbinWatson(order_by="time"),
        RESET(reference="chi2", name="reset_chi2"),
        RESET(reference="F", name="reset_f"),
        Influence(),
    )
    diagnostics_report = diagnostics.report()
    # ols-diagnose-end

    # ols-infer-start
    inferences = {
        "model_based": fit.infer(covariance=ModelBased()),
        "hc3": fit.infer(covariance=HC("HC3")),
        "hac": fit.infer(covariance=HAC(bandwidth, order_by="time")),
    }
    reports = {
        "model_based": inferences["model_based"].report(
            reference="t", degrees_of_freedom=fit.df_resid
        ),
        "hc3": inferences["hc3"].report(reference="normal"),
        "hac": inferences["hac"].report(reference="normal"),
    }
    hypothesis = Hypothesis(LinearRestriction.zero("x"))
    tests = {
        "model_based": inferences["model_based"].test(
            hypothesis, reference="f", degrees_of_freedom=fit.df_resid
        ),
        "hc3": inferences["hc3"].test(hypothesis, reference="chi2"),
        "hac": inferences["hac"].test(hypothesis, reference="chi2"),
    }
    # ols-infer-end

    # ols-predict-start
    prediction_data = Observations({"x": jnp.asarray([-1.0, 0.0, 1.0])})
    predictions = fit.predict(prediction_data)
    # ols-predict-end

    observed_y = np.asarray(observations.numeric("y"))
    sse = float(fit.core.sse)
    # Centered R² is derived here because this design includes an intercept.
    centered_total = float(np.sum((observed_y - observed_y.mean()) ** 2))
    return {
        "observations": observations,
        "fit": fit,
        "fit_report": fit_report,
        "diagnostics": diagnostics,
        "diagnostics_report": diagnostics_report,
        "inferences": inferences,
        "reports": reports,
        "tests": tests,
        "prediction_data": prediction_data,
        "predictions": predictions,
        "summary": {
            "n": observations.n_rows,
            "rank": int(fit.core.rank),
            "df_resid": float(fit.df_resid),
            "sse": sse,
            "centered_r_squared": 1 - sse / centered_total,
        },
    }


# ols-report-start
def write_outputs(analysis, output: Path):
    """Persist selected reports and plots, a predictor and analysis snapshots."""
    output.mkdir(parents=True, exist_ok=True)
    fit = analysis["fit"]
    report = analysis["reports"]["hc3"]
    plots = (residual_plot(fit), report.coefficient_plot())
    report.save_latex(output / "table.tex")
    # Opt-in Wald tests of coefficient = 0 under the normal reference.
    tested = analysis["inferences"]["hc3"].report(reference="normal", null=0)
    tested.save_latex(output / "table-tests.tex")
    analysis["diagnostics_report"].save_latex(output / "diagnostics-table.tex")
    tested.save_values(output / "values.tex", prefix="ols")
    for name, plot in zip(("residual", "coefficients"), plots, strict=True):
        plot.save_pgfplots(output / f"{name}.tex", width_cm=10)
    fit.save_predictor(output / "predictor")
    save_analysis(output / "analysis", replace(report, plots=plots))
    save_analysis(output / "diagnostics", analysis["diagnostics_report"])
    for name in ("model_based", "hac"):
        save_analysis(output / f"inference-{name}", analysis["reports"][name])
    (output / "summary.json").write_text(
        json.dumps(analysis["summary"], indent=2) + "\n", encoding="utf-8"
    )
    (output / "tests.txt").write_text(
        "\n\n".join(test.report().to_text() for test in analysis["tests"].values()),
        encoding="utf-8",
    )
    restored = load_predictor(output / "predictor")
    np.testing.assert_allclose(
        restored.predict(analysis["prediction_data"]), analysis["predictions"]
    )
    assert load_analysis(output / "analysis").report.to_latex() == report.to_latex()


# ols-report-end


def main():
    """Run the default example; retain files only when --output is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = analyze()
    print(analysis["fit_report"].to_text())
    print(analysis["diagnostics_report"].to_text())
    print(analysis["reports"]["hc3"].to_text())
    if args.output is None:
        with TemporaryDirectory() as directory:
            write_outputs(analysis, Path(directory))
    else:
        write_outputs(analysis, args.output)


if __name__ == "__main__":
    main()
