"""Presentation helpers with labels independent of color."""

from __future__ import annotations

import os
from collections.abc import Mapping

import marimo as mo
import numpy as np
from matplotlib.figure import Figure


def is_snapshot() -> bool:
    """Enable bounded defaults for an explicitly requested static snapshot."""
    return os.environ.get("STATADIC_LABS_SNAPSHOT") == "1"


def request_or_snapshot(value: Mapping | None, defaults: dict) -> dict | None:
    """Only a form submission or explicitly requested snapshot can start a job."""
    if value is not None:
        return dict(value)
    return dict(defaults) if is_snapshot() else None


def header(number: str, title: str, question: str, objective: str):
    """A consistent introduction with a single learning objective."""
    note = (
        "**Static-build mode:** default settings are being executed "
        "for an HTML snapshot. "
        if is_snapshot()
        else ""
    )
    return mo.vstack(
        [
            mo.md(
                f"**STATADIC LAB {number} · Predict → experiment → explain**\n\n"
                f"# {title}\n\n{question}"
            ),
            mo.md(f"**You will learn:** {objective}"),
            mo.md(
                note + "Synthetic data only. Numerical results below call the "
                "installed STATADIC package."
            ),
        ]
    )


def runtime_banner(info: Mapping[str, str]):
    """Show actual versions without falsely identifying the installed commit."""
    return mo.md(
        f"<small>Runtime: STATADIC {info['statadic']} · JAX {info['jax']} · "
        f"{info['dtype']} · {info['backend']} · marimo {info['marimo']}. "
        "</small>"
    )


def metric_table(rows: list[tuple[str, object]]):
    """Text-accessible values, also usable without inspecting a chart."""
    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return mo.md("| Quantity | Value |\n|:--|--:|\n" + body)


def point_cloud_plot(x, y) -> Figure:
    """Separate shapes and labels distinguish the two clouds."""
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    ax.scatter(x[:, 0], x[:, 1], label="Sample X", marker="o", alpha=0.7)
    ax.scatter(y[:, 0], y[:, 1], label="Sample Y", marker="x", alpha=0.8)
    ax.set(xlabel="Coordinate 1", ylabel="Coordinate 2", title="Two empirical clouds")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    return fig


def vectors_plot(x, y) -> Figure:
    """Do not use pyplot's process-global mutable figure registry."""
    fig = Figure(figsize=(6, 4), layout="constrained")
    ax = fig.subplots()
    ax.plot([0, x[0, 0]], [0, x[0, 1]], "-o", label="A: fixed unit vector")
    ax.plot([0, y[0, 0]], [0, y[0, 1]], "--s", label="B: your vector")
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set(
        xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), xlabel="Coordinate 1", ylabel="Coordinate 2"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    return fig


def null_plot(result: dict) -> Figure:
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    ax.hist(result["null"], bins=15, label="Permuted labels")
    ax.axvline(
        result["observed"], linestyle="--", linewidth=2, label="Observed statistic"
    )
    ax.set(
        xlabel="Scaled energy statistic T",
        ylabel="Number of permutations",
        title="How surprising is the observed split?",
    )
    ax.legend()
    return fig


def regression_plot(result: dict) -> Figure:
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    ax.scatter(result["x"], result["y"], marker="o", alpha=0.7, label="Observed")
    ax.plot(result["x"], result["predicted"], label="STATADIC OLS fit")
    ax.plot(
        result["x"],
        1 + 2 * result["x"],
        linestyle="--",
        label="Generating mean: 1 + 2x",
    )
    ax.set(xlabel="x", ylabel="y", title="Fit versus data-generating mean")
    ax.legend()
    return fig


def residual_plot(result: dict) -> Figure:
    fig = Figure(figsize=(7, 3.5), layout="constrained")
    ax = fig.subplots()
    ax.scatter(result["predicted"], result["residuals"], marker="o")
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel="Fitted value",
        ylabel="Residual",
        title="Residual spread is not a proof of assumptions",
    )
    return fig


def embedding_plot(result: dict) -> Figure:
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    train, query = result["coordinates"], result["query_coordinates"]
    ax.scatter(train[:, 0], train[:, 1], marker="o", label="Training units")
    ax.scatter(
        query[:, 0],
        query[:, 1],
        marker="x",
        s=70,
        label="Held-out units; fixed-axis extension",
    )
    ax.set(
        xlabel="PCoA axis 1",
        ylabel="PCoA axis 2",
        title="A point represents an entire empirical distribution",
    )
    ax.legend()
    return fig


def measure_plot(clouds) -> Figure:
    fig = Figure(figsize=(7, 3), layout="constrained")
    ax = fig.subplots()
    for i, cloud in enumerate(clouds[:6]):
        ax.scatter(cloud[:, 0], np.full(len(cloud), i), marker="|", s=130)
    ax.set(
        xlabel="Atom value",
        ylabel="Whole unit",
        yticks=list(range(6)),
        title="Six of the 24 observations; each row is a measure",
    )
    return fig


def calibration_plot(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    x = np.array([row["shift"] for row in rows])
    y = np.array([row["rate"] for row in rows])
    err = np.array(
        [
            [row["rate"] - row["lower"] for row in rows],
            [row["upper"] - row["rate"] for row in rows],
        ]
    )
    ax.errorbar(
        x,
        y,
        yerr=err,
        fmt="o",
        capsize=5,
        label="Estimated rejection rate; 95% Wilson interval",
    )
    ax.axhline(0.05, linestyle="--", label="Nominal alpha = 0.05 (null only)")
    ax.set(
        xlabel="Population location shift",
        ylabel="Rejection frequency",
        ylim=(-0.03, 1.03),
        title="A small simulation is evidence, not certification",
    )
    ax.legend(fontsize=8)
    return fig
