"""When each observation is a distribution. Run in the installed STATADIC project environment."""

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
        "05",
        "When a row is a whole distribution",
        "Each unit contains several atoms. Compare whole units without pretending the atoms are independent units.",
        "Prepare a geometry on training observations, extend an embedding, and retain training-only landmark features.",
    )
    return


@app.cell
def _(mo):
    workflow_form = (
        mo.md("""
    {geometry}

    {spread}
    """)
        .batch(
            geometry=mo.ui.dropdown(
                ["Energy", "RBF MMD", "1D W2"], value="Energy", label="Measure geometry"
            ),
            spread=mo.ui.slider(
                start=0.5, stop=2, step=0.25, value=1, label="Within-unit spread"
            ),
        )
        .form(submit_button_label="Compare, embed and predict", show_clear_button=True)
    )
    mo.vstack(
        [
            workflow_form,
            mo.md(
                "24 whole units, unequal support counts, uniform atom masses. "
                "The split is fixed: 18 training units and 6 held-out units. Seed=7."
            ),
        ]
    )
    return (workflow_form,)


@app.cell
def _():
    from statadic_docs_labs import native as backend

    return (backend,)


@app.cell
def _(mo, view, workflow_form):
    _request = view.request_or_snapshot(
        workflow_form.value, {"geometry": "Energy", "spread": 1.0}
    )
    mo.stop(
        _request is None, mo.md("Submit the form to prepare the training geometry.")
    )
    settings = dict(_request)
    return (settings,)


@app.cell
def _(backend, settings):
    workflow = backend.distribution_result(seed=7, **settings)
    return (workflow,)


@app.cell(hide_code=True)
def _(mo, view, workflow):
    _prediction_rows = [
        (unit, f"observed {observed:.3f}; predicted {predicted:.3f}")
        for unit, observed, predicted in zip(
            workflow["held_out_ids"],
            workflow["actual"],
            workflow["predicted"],
            strict=True,
        )
    ]
    mo.vstack(
        [
            mo.md(
                f"**Submitted geometry:** {workflow['geometry']}. "
                f"Training landmarks: `{workflow['landmarks'][0]}`, `{workflow['landmarks'][1]}`."
            ),
            mo.ui.tabs(
                {
                    "What is one observation?": view.measure_plot(workflow["clouds"]),
                    "Training map and held-out extension": view.embedding_plot(
                        workflow
                    ),
                    "Held-out predictions": view.metric_table(_prediction_rows),
                    "Native fit report": mo.md(
                        "```text\n" + workflow["report"] + "\n```"
                    ),
                }
            ),
            mo.md(
                f"**Exploratory held-out RMSE:** {workflow['rmse']:.4f}. "
                "This six-unit comparison is a workflow demonstration, not a performance claim."
            ),
            mo.md(
                "Changing geometry after inspecting held-out error makes that set a validation set. "
                "An unbiased final evaluation still needs untouched data. In cross-validation, prepare learned terms inside each fold."
            ).callout(kind="warn"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    reveal = mo.ui.checkbox(label="Reveal: why not pool all atoms before splitting?")
    reveal
    return (reveal,)


@app.cell(hide_code=True)
def _(mo, reveal):
    mo.md(
        "**The statistical unit is the outer observation.** An atom is part of its unit's empirical measure. "
        "Pooling atoms changes the question and can leak information across a split. "
        "The notebook retains the same training landmarks at prediction time and uses a fixed-axis PCoA extension, "
        "not a refit on held-out observations. Two PCoA axes may also omit distance information."
        if reveal.value
        else "Count units and atoms separately. Which count determines the train/held-out split?"
    )
    return


@app.cell(hide_code=True)
def _(backend, view):
    view.runtime_banner(backend.runtime_info())
    return


if __name__ == "__main__":
    app.run()
