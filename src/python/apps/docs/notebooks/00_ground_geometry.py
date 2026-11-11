"""What does distance notice?. Run in the installed STATADIC project environment."""

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
        "01",
        "What does distance notice?",
        "Make two vectors point in the same direction. Are they now the same object?",
        "Separate direction from magnitude, and recognize an invalid angular input.",
    )
    return


@app.cell
def _(mo):
    radius = mo.ui.slider(start=0, stop=3, step=0.25, value=2, label="Length of B")
    angle = mo.ui.slider(
        start=-180, stop=180, step=15, value=30, label="Direction of B (degrees)"
    )
    mo.vstack(
        [
            radius,
            angle,
            mo.md("Try **angle = 0**, then change the length. A stays at (1, 0)."),
        ]
    )
    return angle, radius


@app.cell
def _():
    from statadic_docs_labs import native as backend

    return (backend,)


@app.cell
def _(angle, backend, mo, radius):
    mo.stop(
        radius.value == 0,
        mo.md(
            "**No angular result:** the zero vector has no direction. Increase the length above zero. "
            "Euclidean distance is still defined, but this comparison requires both quantities."
        ).callout(kind="warn"),
    )
    geometry = backend.geometry_result(radius.value, angle.value)
    return (geometry,)


@app.cell(hide_code=True)
def _(geometry, mo, view):
    mo.vstack(
        [
            view.vectors_plot(geometry["x"], geometry["y"]),
            view.metric_table(
                [
                    ("Euclidean distance", f"{geometry['euclidean']:.4f}"),
                    ("Angular distance (radians)", f"{geometry['angular']:.4f}"),
                ]
            ),
            mo.md(
                "The numbers have different meanings and units. A smaller angular number does not mean a better metric."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    prediction = mo.ui.radio(
        [
            "Both distances must be zero",
            "Only angular distance is zero",
            "Neither distance can be zero",
        ],
        label="Predict: A=(1, 0), B=(3, 0). What happens?",
    )
    reveal = mo.ui.checkbox(label="Reveal the reasoning")
    mo.vstack([prediction, reveal])
    return prediction, reveal


@app.cell(hide_code=True)
def _(mo, prediction, reveal):
    _explanation = (
        "**Angular is zero; Euclidean is 2.** Angular ground discards positive radial scale. "
        "On raw vectors, zero angular distance does not establish equality. On a direction-only space, "
        "those two vectors intentionally represent the same direction."
    )
    _feedback = (
        "Your prediction matches the construction. "
        if prediction.value == "Only angular distance is zero"
        else "Compare direction and magnitude separately. "
    )
    mo.md(
        _feedback + _explanation
        if reveal.value
        else "Choose a prediction, try the controls, then reveal the explanation."
    )
    return


@app.cell(hide_code=True)
def _(backend, view):
    view.runtime_banner(backend.runtime_info())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "API and next experiment": mo.md(
                "Uses `statadic.geometry.ground.metrics.euclidean_distance` and `angular_distance`. "
                "Next: open **01_energy_distance.py** and ask what this geometry choice does to whole distributions."
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
