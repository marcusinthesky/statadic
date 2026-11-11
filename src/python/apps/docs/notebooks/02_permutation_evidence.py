"""Is this difference surprising?. Run in the installed STATADIC project environment."""

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
        "03",
        "Is this difference surprising?",
        "Keep the points. Shuffle the labels. Compare the observed split with those shuffled worlds.",
        "Read a permutation null distribution and the finite resolution of a Monte Carlo p-value.",
    )
    return


@app.cell
def _(mo):
    experiment_form = (
        mo.md("""
    {shift}

    {n}

    {permutations}

    {seed}
    """)
        .batch(
            shift=mo.ui.slider(
                start=0, stop=2, step=0.25, value=0.75, label="Location shift"
            ),
            n=mo.ui.dropdown(["16", "32", "64"], value="32", label="Points per sample"),
            permutations=mo.ui.dropdown(
                ["19", "49", "99", "199"], value="99", label="Permutations"
            ),
            seed=mo.ui.number(
                start=0, stop=10000, step=1, value=42, label="Random seed"
            ),
        )
        .form(submit_button_label="Shuffle the labels", show_clear_button=True)
    )
    mo.vstack(
        [
            experiment_form,
            mo.md(
                "Edit the settings, then submit. Results describe the **last submitted settings**, "
                "not any unsubmitted edits. Euclidean ground and exponent one are fixed in advance."
            ),
        ]
    )
    return (experiment_form,)


@app.cell
def _():
    from statadic_docs_labs import native as backend

    return (backend,)


@app.cell
def _(experiment_form, mo, view):
    _request = view.request_or_snapshot(
        experiment_form.value, {"shift": 0.75, "n": 32, "permutations": 99, "seed": 42}
    )
    mo.stop(_request is None, mo.md("Submit the form to run the first experiment."))
    settings = {
        **_request,
        "n": int(_request["n"]),
        "permutations": int(_request["permutations"]),
    }
    return (settings,)


@app.cell
def _(backend, settings):
    permutation = backend.permutation_result(**settings)
    return (permutation,)


@app.cell(hide_code=True)
def _(mo, permutation, settings, view):
    mo.vstack(
        [
            mo.md(
                f"**Submitted:** n={settings['n']} per group; shift={settings['shift']}; "
                f"B={settings['permutations']}; seed={settings['seed']}."
            ),
            view.null_plot(permutation),
            view.metric_table(
                [
                    ("Observed scaled energy T", f"{permutation['observed']:.5g}"),
                    ("Upper-tail Monte Carlo p", f"{permutation['pvalue']:.4f}"),
                    (
                        "Smallest possible reported p",
                        f"{1 / (permutation['permutations'] + 1):.4f}",
                    ),
                    (
                        "Exceedances (including ties)",
                        int((permutation["null"] >= permutation["observed"]).sum()),
                    ),
                ]
            ),
            mo.md(
                "**Interpretation:** this compares the observed statistic with randomized label assignments. "
                "The p-value is not the probability that the null hypothesis is true, and non-rejection is not equivalence."
            ).callout(kind="info"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    reveal = mo.ui.checkbox(
        label="Challenge: can 19 random permutations yield p = 0.001? Reveal the answer."
    )
    reveal
    return (reveal,)


@app.cell(hide_code=True)
def _(mo, reveal):
    mo.md(
        "**No.** The add-one calculation is `(1 + exceedances) / (B + 1)`. With B=19, the minimum is 0.05. "
        "More permutations improve resolution; they do not fix a non-exchangeable design. "
        "Under this lab's null, all observations are IID from the same distribution. Clustered, time-series or paired "
        "data need a suitable design, not this unrestricted label shuffle."
        if reveal.value
        else "Try B=19 and B=199. Watch both the histogram and the minimum p-value."
    )
    return


@app.cell(hide_code=True)
def _(backend, view):
    view.runtime_banner(backend.runtime_info())
    return


if __name__ == "__main__":
    app.run()
