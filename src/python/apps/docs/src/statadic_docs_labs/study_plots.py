"""Plot retained operating-characteristic bounds without recomputing evidence."""

from pathlib import Path
from textwrap import fill

from matplotlib.figure import Figure

AXES = (
    ("size", "size_failure_inclusive", "Rejection, failures included"),
    ("coverage", "coverage_all_attempts", "Coverage, failures uncovered"),
    ("failure", "failure_rate", "Unavailable procedures"),
)


def operating_points(report: dict) -> dict[str, list[dict]]:
    """Read exact retained endpoints; unavailable metrics never become zeros."""
    points = {axis: [] for axis, _, _ in AXES}
    for scenario, cell in report["scenarios"].items():
        for method, result in cell["methods"].items():
            bounds = result.get("qualification", {})
            for axis, metric, _ in AXES:
                lower = bounds.get(f"{axis}_lower")
                upper = bounds.get(f"{axis}_upper")
                mean = result.get(metric, {}).get("mean")
                if lower is None or upper is None or mean is None:
                    continue
                points[axis].append(
                    {
                        "label": f"{scenario} / {method}",
                        "mean": mean,
                        "lower": lower,
                        "upper": upper,
                        "status": bounds.get(f"{axis}_status", "see joint criterion"),
                        "valid": result["valid"],
                        "failed": result["failed"],
                        "requested": cell["requested"],
                    }
                )
    return {axis: rows for axis, rows in points.items() if rows}


def operating_plot(report: dict, study: dict, destination: Path) -> bool:
    """Render only supplied bounds and the frozen study's decision thresholds."""
    points = operating_points(report)
    if not points:
        return False
    limits = {
        "size": study["alpha"] + study["margin"],
        "coverage": 1 - study["alpha"] - study["margin"],
        "failure": study["failure_budget"],
    }
    panels = [(axis, title) for axis, _, title in AXES if axis in points]
    heights = [max(2, 0.42 * len(points[axis]) + 1) for axis, _ in panels]
    figure = Figure(figsize=(10, sum(heights) + 1), layout="constrained")
    axes = figure.subplots(len(panels), 1, squeeze=False, height_ratios=heights)
    for plot, (axis, title) in zip(axes[:, 0], panels, strict=True):
        rows = points[axis]
        for index, row in enumerate(rows):
            plot.plot([row["lower"], row["upper"]], [index, index], color="#1d3f8f")
            plot.plot(row["mean"], index, "o", color="#1d3f8f", markersize=4)
        plot.axvline(limits[axis], color="#181a20", linestyle="--", linewidth=1)
        plot.set_yticks(
            range(len(rows)),
            [
                fill(row["label"], 48)
                + f"\n{row['status']}; {row['valid']}/{row['requested']} valid, "
                + f"{row['failed']} failed"
                for row in rows
            ],
            fontsize=8,
        )
        plot.invert_yaxis()
        plot.set_title(f"{title}; frozen threshold {limits[axis]:g}", fontsize=11)
        plot.set_xlabel("Rate (point); retained one-sided 95% bounds (endpoints)")
        plot.grid(axis="x", color="#d4deef")
    figure.suptitle(f"{study['id']} · {report['profile']} profile", fontsize=13)
    figure.savefig(destination, format="svg", metadata={"Date": None})
    return True
