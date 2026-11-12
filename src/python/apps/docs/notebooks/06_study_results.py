"""Explore U and V bias with the registered STATADIC study engine."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    from statadic_docs_labs import presentation as view
    from statadic_docs_labs.study_tutorial import run_study

    return json, mo, run_study, view


@app.cell(hide_code=True)
def _(view):
    view.header(
        "07",
        "U or V: where does bias come from?",
        "Compare two MMD estimators across independent repeated samples.",
        "Connect a checked mathematical result to empirical Monte Carlo bias.",
    )
    return


@app.cell
def _(mo):
    study_form = (
        mo.md("""{trials}\n\n{seed}""")
        .batch(
            trials=mo.ui.dropdown(
                ["20", "100", "1000"], value="20", label="Trials per scenario"
            ),
            seed=mo.ui.number(
                start=0, stop=10000, step=1, value=123, label="Random seed"
            ),
        )
        .form(submit_button_label="Run the study", show_clear_button=True)
    )
    mo.vstack(
        [
            study_form,
            mo.md(
                "Four fixed scenarios: null and alternative, each with equal and unequal sample sizes. Settings apply on submission."
            ),
        ]
    )
    return (study_form,)


@app.cell
def _(mo, study_form, view):
    _request = view.request_or_snapshot(study_form.value, {"trials": 20, "seed": 123})
    mo.stop(_request is None, mo.md("Submit the form to run the study."))
    settings = {"trials": int(_request["trials"]), "seed": int(_request["seed"])}
    return (settings,)


@app.cell
def _(mo, run_study, settings):
    with mo.status.spinner(title="Running the registered study"):
        study = run_study(**settings)
    return (study,)


@app.cell(hide_code=True)
def _(mo, settings, study):
    _rows = []
    for _scenario, _cell in study["summary"]["scenarios"].items():
        for _method, _result in _cell["methods"].items():
            _bias = _result["conditional_bias"]
            _fmt = lambda x: "unavailable" if x is None else f"{x:.4f}"
            _rows.append(
                f"| {_scenario} | {_method} | {_fmt(study['population_truth'].get(_scenario))} | "
                f"{_fmt(_bias['mean'])} | {_fmt(_bias['mcse'])} | {_result['valid']} | {_result['failed']} |"
            )
    mo.md(
        f"**Submitted study:** {settings['trials']} trials per scenario; seed={settings['seed']}.\n\n"
        "| Scenario | Estimator | Population MMD² | Bias | Monte Carlo SE | Valid | Failed |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(_rows)
        + "\n\nBias and its Monte Carlo SE are conditional on valid estimates. "
        "These source-only smoke results are teaching evidence and cannot qualify a method."
    )
    return


@app.cell(hide_code=True)
def _(json, mo, study):
    mo.download(
        data=json.dumps(study, indent=2).encode(),
        filename="statadic-mmd-teaching-summary.json",
        mimetype="application/json",
        label="Download summary and provenance",
    )
    return


if __name__ == "__main__":
    app.run()
