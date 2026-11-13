"""Declarative prediction artifacts and limited report snapshots.

Loading reconstructs an allowlisted predictor, never a checked native fitted
model, parameter covariance, identification certificate or inferential scheme.
"""

from __future__ import annotations

from collections.abc import Callable  # runtime factory annotation  # noqa: TC003
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from statadic.api.reporting import (
    JsonValue,  # noqa: TC001  # Runtime artifact annotations.
    Metric,
    ParameterRow,
    PlotSpec,
    ReferenceLine,
    Report,
    TestRow,
    freeze_metadata,
    metadata_to_json,
)

from ._store import Arrays, ArtifactError, read_bundle, record, write_bundle


@dataclass(frozen=True)
class AnalysisSnapshot:
    """A report snapshot; this is neither a fit nor restored sampling uncertainty."""

    report: Report

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Expose only formatting of already-computed report values."""
        return ("report", "render")


def _report_state(report: Report) -> dict[str, JsonValue]:
    def plot_state(plot: PlotSpec) -> dict[str, object]:
        return {
            "kind": plot.kind,
            "x": list(plot.x),
            "y": list(plot.y),
            "xlabel": plot.xlabel,
            "ylabel": plot.ylabel,
            "title": plot.title,
            "lower": None if plot.lower is None else list(plot.lower),
            "upper": None if plot.upper is None else list(plot.upper),
            "interval_meaning": plot.interval_meaning,
            "labels": list(plot.labels),
            "reference_lines": [asdict(line) for line in plot.reference_lines],
            "metadata": metadata_to_json(plot.metadata),
            "label": plot.label,
            "interval_style": plot.interval_style,
            "overlays": [plot_state(item) for item in plot.overlays],
            "markup": plot.markup,
        }

    plots = [plot_state(plot) for plot in report.plots]
    state: dict[str, object] = {
        "title": report.title,
        "parameters": [asdict(row) for row in report.parameters],
        "metrics": [asdict(metric) for metric in report.metrics],
        "tests": [asdict(row) for row in report.tests],
        "metadata": metadata_to_json(report.metadata),
        "warnings": list(report.warnings),
        "notes": list(report.notes),
        "plots": plots,
    }
    # Convert NumPy scalar subclasses through the same finite JSON policy as metadata.
    return metadata_to_json(freeze_metadata(state))


def save_analysis(path: Path, report: Report) -> None:
    """Persist a rendering-only analysis snapshot without claiming fit restoration."""
    write_bundle(
        Path(path),
        purpose="analysis",
        family="report",
        capabilities=("report", "render"),
        state=_report_state(report),
        arrays=Arrays(),
        metadata={"scope": "already-computed report snapshot"},
    )


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        message = "Report sections must be ordered lists."
        raise ArtifactError(message)
    return value


def _construct[T](factory: Callable[..., T], values: dict[str, object]) -> T:
    """Invoke records whose constructors check the dynamic serialized values."""
    return factory(**values)


def _restore_test(row: object) -> TestRow:
    state = record(row, {item.name for item in fields(TestRow)}, "test row")
    state["law_parameters"] = tuple(_items(state["law_parameters"]))
    return _construct(TestRow, state)


def _restore_report(value: object, *, trust_tex: bool) -> Report:
    saved = record(
        value,
        {
            "title",
            "parameters",
            "metrics",
            "tests",
            "metadata",
            "warnings",
            "plots",
            "notes",
        },
        "report",
    )
    for section in ("parameters", "metrics", "tests", "warnings", "plots", "notes"):
        if not isinstance(saved[section], list):
            message = f"Report {section} must be an ordered list."
            raise ArtifactError(message)
    parameters = [
        _construct(
            ParameterRow,
            record(row, {item.name for item in fields(ParameterRow)}, "parameter row"),
        )
        for row in _items(saved["parameters"])
    ]
    metrics = [
        _construct(
            Metric, record(row, {item.name for item in fields(Metric)}, "metric")
        )
        for row in _items(saved["metrics"])
    ]

    tests = [_restore_test(row) for row in _items(saved["tests"])]

    def restore_plot(plot: object) -> PlotSpec:
        state = record(plot, {item.name for item in fields(PlotSpec)}, "plot")
        if state["markup"] == "tex" and not trust_tex:
            msg = "TeX markup requires trust_tex=True."
            raise ArtifactError(msg)
        state["overlays"] = tuple(
            restore_plot(item) for item in _items(state["overlays"])
        )
        lines = state["reference_lines"]
        if not isinstance(lines, list):
            message = "Plot reference lines must be a list."
            raise ArtifactError(message)
        references = []
        for line in lines:
            endpoints = record(line, {"x", "y", "label"}, "reference line")
            for name in ("x", "y"):
                endpoints[name] = tuple(_items(endpoints[name]))
            references.append(_construct(ReferenceLine, endpoints))
        state["reference_lines"] = tuple(references)
        for name in ("x", "y", "labels", "lower", "upper"):
            if state[name] is not None:
                state[name] = tuple(_items(state[name]))
        return _construct(PlotSpec, state)

    plots = [restore_plot(plot) for plot in _items(saved["plots"])]
    saved["parameters"], saved["metrics"], saved["tests"], saved["plots"] = (
        tuple(parameters),
        tuple(metrics),
        tuple(tests),
        tuple(plots),
    )
    saved["warnings"] = tuple(_items(saved["warnings"]))
    saved["notes"] = tuple(_items(saved["notes"]))
    return _construct(Report, saved)


def load_analysis(path: Path, *, trust_tex: bool = False) -> AnalysisSnapshot:
    """Restore a rendering-only snapshot and reject any inferential capability claim."""
    manifest, arrays = read_bundle(Path(path), purpose="analysis")
    if manifest["family"] != "report" or manifest["capabilities"] != [
        "report",
        "render",
    ]:
        message = (
            "Analysis restoration supports only an explicitly limited report snapshot."
        )
        raise ArtifactError(message)
    arrays.require_consumed()
    try:
        report = _restore_report(manifest["state"], trust_tex=trust_tex)
    except (TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        message = "Malformed report snapshot."
        raise ArtifactError(message) from error
    return AnalysisSnapshot(report)
