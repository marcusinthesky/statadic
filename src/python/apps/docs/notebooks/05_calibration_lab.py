"""Does a test behave as advertised?. Run in the installed STATADIC project environment."""

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
        "06",
        "A good-looking p-value is not validation",
        "Repeat entire experiments, not just permutations of one dataset. Watch uncertainty around rejection frequencies.",
        "Distinguish a teaching simulation, numerical agreement and statistical qualification.",
    )
    return


@app.cell
def _(mo):
    study_form = (
        mo.md("""
    {trials}

    {n}

    {permutations}

    {seed}
    """)
        .batch(
            trials=mo.ui.dropdown(
                ["20", "50", "100"], value="20", label="Trials per scenario"
            ),
            n=mo.ui.dropdown(["16", "24", "32"], value="24", label="Points per sample"),
            permutations=mo.ui.dropdown(
                ["19", "49", "99"], value="49", label="Permutations"
            ),
            seed=mo.ui.number(
                start=0, stop=10000, step=1, value=123, label="Random seed"
            ),
        )
        .form(submit_button_label="Run the small study", show_clear_button=True)
    )
    mo.vstack(
        [
            study_form,
            mo.md(
                "Three predeclared location shifts: **0, 0.75, 1.5**. Alpha=0.05. "
                "The null scenario is shift=0; the others measure power for this specific design. "
                "Settings take effect only after submission."
            ),
        ]
    )
    return (study_form,)


@app.cell
def _():
    from statadic_docs_labs import native as backend

    return (backend,)


@app.cell
def _(mo, study_form, view):
    _request = view.request_or_snapshot(
        study_form.value, {"trials": 20, "n": 24, "permutations": 49, "seed": 123}
    )
    mo.stop(
        _request is None,
        mo.md("No simulation runs on page load. Submit the form to begin."),
    )
    study_settings = {
        **_request,
        "trials": int(_request["trials"]),
        "n": int(_request["n"]),
        "permutations": int(_request["permutations"]),
    }
    return (study_settings,)


@app.cell
def _(backend, mo, study_settings):
    with mo.status.progress_bar(
        total=3 * study_settings["trials"], title="Independent experiments"
    ) as _progress:
        study = backend.calibration_result(
            **study_settings, progress=lambda _done, _total: _progress.update()
        )
    return (study,)


@app.cell(hide_code=True)
def _(mo, study, study_settings, view):
    _rows = [
        (
            f"shift={row['shift']}",
            f"{row['rejections']}/{row['trials']} = {row['rate']:.3f}; "
            f"95% Wilson [{row['lower']:.3f}, {row['upper']:.3f}]",
        )
        for row in study
    ]
    mo.vstack(
        [
            mo.md(
                f"**Submitted study:** {study_settings['trials']} trials per scenario; "
                f"n={study_settings['n']}; B={study_settings['permutations']}; seed={study_settings['seed']}."
            ),
            view.calibration_plot(study),
            view.metric_table(_rows),
            mo.md(
                "**No pass/fail badge.** Small studies have wide Monte Carlo intervals. "
                "These are intervals for rejection frequencies, not confidence intervals for a model coefficient. "
                "All trials must complete; failed trials are not silently dropped."
            ).callout(kind="warn"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(backend, mo, study, study_settings):
    import json as _json

    _payload = {
        "purpose": "teaching simulation; not qualification evidence",
        "settings": study_settings,
        "scenarios": study,
        "runtime": backend.runtime_info(),
        "decision": "p <= 0.05",
        "interval": "95% Wilson; approximate",
    }
    mo.download(
        data=_json.dumps(_payload, indent=2).encode(),
        filename="statadic-teaching-study.json",
        mimetype="application/json",
        label="Save this study and its settings",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    reveal = mo.ui.checkbox(
        label="Reveal: is zero rejection in 20 null trials proof of a perfect test?"
    )
    reveal
    return (reveal,)


@app.cell(hide_code=True)
def _(mo, reveal):
    mo.md(
        "**No.** An observed count of zero still leaves substantial uncertainty about the underlying rejection probability. "
        "Repeated studies, predeclared designs, sufficient independent trials and the package's method-specific qualification "
        "suite address stronger claims. Numerical parity, JIT support and an attractive chart do not establish calibration."
        if reveal.value
        else "Read the interval, not only the point estimate. Try more trials, without searching for a favorable seed."
    )
    return


@app.cell(hide_code=True)
def _(backend, view):
    view.runtime_banner(backend.runtime_info())
    return


if __name__ == "__main__":
    app.run()
