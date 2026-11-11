"""Separate the fitted line, residual diagnostics and explicit uncertainty."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from statadic_docs_labs import presentation as view

    return mo, view


@app.cell(hide_code=True)
def _(view):
    view.header(
        "04",
        "A fitted line is not the whole story",
        "Add uneven noise or a contaminated outcome, then change covariance.",
        "Separate estimation, diagnostics and explicit reference-law choices.",
    )
    return


@app.cell
def _(mo):
    controls = mo.md("""
    {noise}

    {heterogeneity}

    {outlier}

    {covariance}

    {bandwidth}
    """).batch(
        noise=mo.ui.slider(start=0.1, stop=2, step=0.1, value=0.6, label="Noise level"),
        heterogeneity=mo.ui.slider(
            start=0, stop=2, step=0.25, value=0, label="Uneven error spread"
        ),
        outlier=mo.ui.checkbox(value=False, label="Contaminate the final outcome"),
        covariance=mo.ui.dropdown(
            ["HC3", "Model-based", "HAC"], value="HC3", label="Covariance convention"
        ),
        bandwidth=mo.ui.slider(
            start=0, stop=10, step=1, value=3, label="Bartlett HAC bandwidth"
        ),
    )
    mo.vstack(
        [
            controls,
            mo.md(
                "80 observations; seed=17. The generating mean is **1 + 2x**. "
                "Changing covariance should not change the OLS coefficients."
                " The declared unique time key is row position, not calendar time."
            ),
        ]
    )
    return (controls,)


@app.cell
def _():
    from statadic_docs_labs import native as backend

    return (backend,)


@app.cell
def _(backend, controls):
    regression = backend.regression_result(seed=17, **controls.value)
    return (regression,)


@app.cell(hide_code=True)
def _(mo, regression, view):
    mo.vstack(
        [
            view.metric_table(
                [
                    ("Estimated intercept", f"{regression['coefficients'][0]:.4f}"),
                    (
                        "Estimated slope (generating slope: 2)",
                        f"{regression['coefficients'][1]:.4f}",
                    ),
                    ("Residual degrees of freedom", f"{regression['df_resid']:.0f}"),
                    ("Design rank", f"{regression['summary']['rank']:.0f}"),
                    ("Residual SSE", f"{regression['summary']['sse']:.4f}"),
                    (
                        "Centered R² (derived with an intercept)",
                        f"{regression['summary']['centered_r_squared']:.4f}",
                    ),
                ]
            ),
            mo.ui.tabs(
                {
                    "Fit": view.regression_plot(regression),
                    "Residuals": view.residual_plot(regression),
                    "Requested diagnostics": mo.md(
                        "```text\n" + regression["diagnostics"] + "\n```"
                    ),
                    "Native inference report": mo.md(
                        "```text\n" + regression["report"] + "\n```"
                    ),
                    "Named slope hypothesis": mo.md(
                        "```text\n" + regression["test"] + "\n```"
                    ),
                }
            ),
            mo.md("**Reference:** " + regression["reference"]),
            mo.download(
                data=regression["latex"].encode("utf-8"),
                filename="statadic-regression.tex",
                label="Download this inference report as LaTeX",
            ),
            mo.md(
                "HC3 changes covariance; it does not make OLS outlier-resistant "
                "or identify a causal effect. Classical t/F inference needs "
                "independent homoskedastic Gaussian errors. HAC uses Bartlett "
                "weights in ascending time order; bandwidth counts observations. "
                "No diagnostic selects a covariance or validates exogeneity."
            ).callout(kind="warn"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    prediction = mo.ui.radio(
        [
            "The fitted slope changes",
            "Only uncertainty changes",
            "HC3 removes the outlier",
        ],
        label="Predict: with data fixed, what does switching to HC3 do?",
    )
    reveal = mo.ui.checkbox(label="Reveal the distinction")
    mo.vstack([prediction, reveal])
    return prediction, reveal


@app.cell(hide_code=True)
def _(mo, prediction, reveal):
    mo.md(
        (
            "Correct. "
            if prediction.value == "Only uncertainty changes"
            else "Keep fitting and inference separate. "
        )
        + "**The OLS point estimate stays the same.** HC3 changes covariance. "
        "This lab pairs HC3 with a normal reference and model-based covariance "
        "with a Student t reference and explicit residual degrees of freedom. "
        "HAC uses a normal reference with a declared ordering and bandwidth. "
        "These interval procedures differ in more than their covariance."
        if reveal.value
        else "Try both covariance settings without moving the data controls."
    )
    return


@app.cell(hide_code=True)
def _(backend, view):
    view.runtime_banner(backend.runtime_info())
    return


if __name__ == "__main__":
    app.run()
