"""Can a distance miss a change in scale?. Run in the installed STATADIC project environment."""

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
        "02",
        "Can a distance miss a change in scale?",
        "Move or stretch one cloud. Then change what the ground metric is allowed to notice.",
        "Distinguish raw V-energy from rooted energy, without confusing either with a p-value.",
    )
    return


@app.cell
def _(mo):
    controls = mo.md("""
    {mode}

    {ground}

    {shift}

    {scale}

    {seed}
    """).batch(
        mode=mo.ui.dropdown(
            ["independent", "scaled-copy"], value="independent", label="Experiment"
        ),
        ground=mo.ui.dropdown(
            ["euclidean", "angular"], value="euclidean", label="Ground"
        ),
        shift=mo.ui.slider(
            start=-3, stop=3, step=0.25, value=1, label="Location shift"
        ),
        scale=mo.ui.slider(
            start=0.25, stop=3, step=0.25, value=1, label="Scale multiplier"
        ),
        seed=mo.ui.number(start=0, stop=10000, step=1, value=42, label="Random seed"),
    )
    mo.vstack(
        [
            controls,
            mo.md(
                "64 points per cloud. Underlying draws stay fixed as sliders move. "
                "For the scale challenge, choose **scaled-copy**, set **shift = 0**, then vary scale."
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
    energy = backend.energy_result(n=64, **controls.value)
    return (energy,)


@app.cell(hide_code=True)
def _(energy, mo, view):
    _warning = (
        "**Paired geometry demonstration:** Y is a transformed copy of X. Do not feed this setup into an IID two-sample test."
        if energy["mode"] == "scaled-copy"
        else "**Independent clouds:** finite samples from the same population need not have zero empirical discrepancy."
    )
    mo.vstack(
        [
            view.point_cloud_plot(energy["x"], energy["y"]),
            view.metric_table(
                [
                    ("Scalar API: raw V-energy", f"{energy['raw']:.6g}"),
                    ("Host Energy API: rooted V-energy", f"{energy['rooted']:.6g}"),
                    ("Ground / exponent", f"{energy['ground']} / 1"),
                ]
            ),
            mo.md(_warning).callout(kind="warn"),
            mo.md(
                "The scalar API requires an explicit ground metric; the host `Energy` API defaults to Euclidean. "
                "This lab passes the ground explicitly to both. It never clips a signed statistic into a distance."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    prediction = mo.ui.radio(
        ["Euclidean", "Angular", "Both must respond equally"],
        label="Predict: which ground notices Y=3X when shift is zero?",
    )
    reveal = mo.ui.checkbox(label="Reveal why")
    mo.vstack([prediction, reveal])
    return prediction, reveal


@app.cell(hide_code=True)
def _(mo, prediction, reveal):
    _answer = (
        "**Euclidean notices magnitude.** Positive scaling preserves every direction, so angular energy "
        "between a cloud and its scaled copy is zero up to numerical error. Euclidean energy can detect "
        "that change. Raw and rooted outputs are different conventions, not different sample sizes. "
        "A discrepancy describes separation; a significance test additionally needs a null and a valid resampling design."
    )
    mo.md(
        (
            "Correct. "
            if prediction.value == "Euclidean"
            else "Think about the previous vector experiment. "
        )
        + _answer
        if reveal.value
        else "Make a prediction before revealing the answer."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Mathematics and API": mo.md(r"""
    For exponent one, raw empirical V-energy is

    $$
    E_V=\frac{2}{nm}\sum_{i,j}d(x_i,y_j)
    -\frac{1}{n^2}\sum_{i,i'}d(x_i,x_{i'})
    -\frac{1}{m^2}\sum_{j,j'}d(y_j,y_{j'}).
    $$

    Self-pairs are retained. The host comparison returns the rooted V quantity under its
    qualified geometry. This is not the signed U-statistic convention.

    Calls: `statadic.discrepancy.mmd.energy.energy_distance`,
    `statadic.api.representation.Energy.cross`, `statadic.sample.packed.PackedClouds`.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _(backend, view):
    view.runtime_banner(backend.runtime_info())
    return


if __name__ == "__main__":
    app.run()
