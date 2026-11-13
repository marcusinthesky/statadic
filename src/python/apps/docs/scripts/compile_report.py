"""Prepare and compile the documentation's booktabs report template."""

import argparse
import subprocess
import tempfile
from pathlib import Path

from statadic.api.reporting import ParameterRow, Report

APP = Path(__file__).resolve().parents[1]
BUNDLE = APP.parents[2] / "latex/bundle/tlextras-2022.0r0-statadic.zip"
TEMPLATE = APP / "templates/report.tex"
FIGURE_TEMPLATE = APP / "templates/figure.tex"


def _compile(
    content: str, output_dir: Path, *, only_cached: bool, template: Path, name: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}-content.tex").write_text(content, encoding="utf-8")
    source = output_dir / f"{name}.tex"
    source.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    command = [
        "tectonic",
        "--untrusted",
        "--bundle",
        str(BUNDLE),
        "--outdir",
        str(output_dir),
    ]
    if only_cached:
        command.append("--only-cached")
    completed = subprocess.run(
        [*command, str(source)], text=True, capture_output=True, check=False
    )
    output = completed.stdout + completed.stderr
    if completed.returncode or any(
        marker in output
        for marker in ("Overfull", "Missing character", "could not represent character")
    ):
        raise RuntimeError(output)


def compile_report(report: str, output_dir: Path, *, only_cached: bool) -> None:
    """Compile a report with the pinned bundle and the shared page layout."""
    _compile(
        report, output_dir, only_cached=only_cached, template=TEMPLATE, name="report"
    )


def compile_figure(figure: str, output_dir: Path, *, only_cached: bool) -> None:
    """Compile a native PGFplots figure with the pinned bundle."""
    _compile(
        figure,
        output_dir,
        only_cached=only_cached,
        template=FIGURE_TEMPLATE,
        name="figure",
    )


def main() -> None:
    """Download the pinned template dependencies into Tectonic's cache."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    with tempfile.TemporaryDirectory(prefix="statadic-tex-setup-") as temporary:
        report = Report(
            "Example",
            parameters=(
                ParameterRow("slope", 1.0, std_error=0.1, ci_lower=0.8, ci_upper=1.2),
            ),
        )
        compile_report(report.to_latex(), Path(temporary), only_cached=False)
        from statadic.api.reporting import PlotSpec

        compile_figure(
            PlotSpec("line", (0.0, 1.0), (0.0, 1.0)).to_pgfplots(width_cm=10),
            Path(temporary),
            only_cached=False,
        )


if __name__ == "__main__":
    main()
