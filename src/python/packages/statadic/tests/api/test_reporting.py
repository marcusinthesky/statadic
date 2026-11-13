"""Report snapshots preserve numerical evidence across escaped renderers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from html.parser import HTMLParser
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from matplotlib.collections import LineCollection
from statadic.api.reporting import (
    _METRIC_LABELS,
    FIT_STATISTIC_NAMES,
    Metric,
    ParameterRow,
    PlotSpec,
    ReferenceLine,
    Report,
    TestRow,
    _finite_metric,
    _law_number_text,
    _number_cell,
    _tex,
    freeze_metadata,
    metadata_to_json,
)

from .._budgets import examples
from .._typing import ill_typed
from .._validation.reporting import (
    expected_interval_state,
    expected_metric_state,
    report_numbers,
    reports,
)

if TYPE_CHECKING:
    from statadic.api.reporting import JsonValue

    from .._validation.reporting import ReportCase


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    @override
    def handle_data(self, data: str) -> None:
        self.values.append(data)


@pytest.mark.unit
@pytest.mark.contract
def test_report_owns_nested_metadata_and_retains_unrounded_values() -> None:
    """Canonical records own nested containers and retain full input precision."""
    ids: list[JsonValue] = [1, "1"]
    sample: dict[str, JsonValue] = {"ids": ids}
    metadata: dict[str, JsonValue] = {"sample": sample}
    number = 0.12345678901234567
    report = Report(
        "OLS",
        parameters=(ParameterRow("slope", number, pvalue=1e-100),),
        metadata=metadata,
    )
    ids.append(2)

    assert report.parameters[0].estimate == number
    assert isinstance(report.metadata, Mapping)
    retained = report.metadata["sample"]
    assert isinstance(retained, Mapping)
    assert retained["ids"] == (1, "1")
    with pytest.raises(FrozenInstanceError):
        ill_typed(report).title = "changed"
    with pytest.raises(TypeError):
        ill_typed(retained)["ids"] = ()
    assert "1e-100" in report.to_text()


@pytest.mark.unit
@pytest.mark.contract
def test_replaced_reports_and_plots_own_frozen_metadata() -> None:
    """Replacement revalidates metadata and JSON export owns mutable copies."""
    raw: dict[str, JsonValue] = {"ids": [1, "1"]}
    report = Report("sample", metadata=raw)
    plot = PlotSpec("line", (0.0, 1.0), (1.0, 2.0), metadata=raw)
    replacement = {"nested": raw}
    updated_report = replace(report, metadata=freeze_metadata(replacement))
    updated_plot = replace(plot, metadata=freeze_metadata(replacement))
    raw["ids"] = [2]
    assert report.metadata["ids"] == (1, "1")
    assert plot.metadata["ids"] == (1, "1")
    assert updated_report.metadata == updated_plot.metadata
    assert updated_report.metadata["nested"] == {"ids": (1, "1")}
    exported = metadata_to_json(updated_report.metadata)
    exported["nested"] = None
    assert updated_report.metadata["nested"] == {"ids": (1, "1")}
    with pytest.raises(ValueError, match="nonfinite"):
        replace(report, metadata={"invalid": float("nan")})


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "bad",
    [{1: "value"}, {"valid": 1, 2: "invalid"}, {"nested": {1: "value"}}, [{None: 1}]],
)
def test_metadata_converters_reject_nonstring_keys_at_every_depth(bad: object) -> None:
    """Raw-object conversion never emits non-JSON mapping keys."""
    for convert in (freeze_metadata, metadata_to_json):
        with pytest.raises(TypeError, match="keys must be strings"):
            convert(bad)


@pytest.mark.component
@pytest.mark.contract
def test_renderers_preserve_unavailability_and_escape_untrusted_labels() -> None:
    """Text, HTML and booktabs retain reasons while escaping ordinary labels."""
    name = r"α_1 & x%<script>\^~#${}"
    report = Report(
        "Example <script>",
        parameters=(
            ParameterRow(name, 1.25),
            ParameterRow(
                "failed",
                None,
                valid=False,
                status=4,
                unavailable_reason="rank deficient",
            ),
        ),
        metrics=(
            Metric(
                "log likelihood",
                None,
                available=False,
                unavailable_reason="not a likelihood estimator",
            ),
        ),
        metadata={"reference": "not requested"},
    )
    parser = _Text()
    parser.feed(report.to_html())

    assert name in parser.values
    assert "<script>" not in report.to_html()
    for rendering in (report.to_text(), report.to_html(), report.to_latex()):
        assert "rank deficient" in rendering
        assert "not a likelihood estimator" in rendering
    latex = report.to_latex()
    for escaped in (
        r"\_",
        r"\&",
        r"\%",
        r"\textbackslash{}",
        r"\textasciicircum{}",
        r"\textasciitilde{}",
        r"\#",
        r"\$",
        r"\{",
        r"\}",
    ):
        assert escaped in latex
    assert r"\begin{tabular}" in latex
    assert r"\toprule" in latex


@pytest.mark.unit
@pytest.mark.contract
def test_interval_availability_is_separate_from_estimate_availability() -> None:
    """An unavailable interval keeps its SE but cannot also report endpoints."""
    row = ParameterRow(
        "slope",
        1.0,
        std_error=0.5,
        interval_status="unavailable",
        interval_unavailable_reason="INVALID_REFERENCE",
    )
    assert row.valid
    assert row.std_error is not None
    assert row.ci_lower is None
    with pytest.raises(ValueError, match="cannot report endpoints"):
        ParameterRow(
            "slope",
            1.0,
            std_error=0.5,
            ci_lower=0.0,
            ci_upper=2.0,
            interval_status="unavailable",
            interval_unavailable_reason="INVALID_REFERENCE",
        )
    with pytest.raises(TypeError):
        ParameterRow(
            "slope",
            1.0,
            interval_status="unavailable",
            interval_unavailable_reason=ill_typed(3),
        )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_reports_reject_nonfinite_numbers(value: float) -> None:
    """Unavailable numerical evidence must use an explicit status and null."""
    with pytest.raises(ValueError, match="finite"):
        ParameterRow("slope", value)
    with pytest.raises(ValueError, match="finite"):
        Metric("likelihood", value)
    with pytest.raises(ValueError, match="nonfinite"):
        Report("invalid metadata", metadata={"df": value})


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finite_metric_retains_explicit_unavailability(value: float) -> None:
    """Nonfinite fitted summaries remain null with an explanation in reports."""
    metric = _finite_metric("scale", jnp.asarray(value), "Residual scale.")
    assert metric.value is None
    assert not metric.available
    assert metric.status == "unavailable"
    assert metric.unavailable_reason == "Unavailable in the retained fit."
    assert metric.unavailable_reason in Report("fit", metrics=(metric,)).to_text()


@pytest.mark.unit
def test_finite_metric_retains_value_and_definition() -> None:
    """Finite fitted evidence stays available with its definition."""
    metric = _finite_metric("scale", jnp.asarray(2.5), "Residual scale.")
    assert metric.value == pytest.approx(2.5)
    assert metric.available
    assert metric.definition == "Residual scale."
    assert metric.unavailable_reason is None


@pytest.mark.unit
@pytest.mark.contract
def test_unavailable_records_require_reasons_and_ordered_intervals() -> None:
    """Missing inference is explained and malformed intervals are refused."""
    with pytest.raises(ValueError, match="reason"):
        ParameterRow("slope", None, valid=False)
    with pytest.raises(ValueError, match="reason"):
        Metric("AIC", None, available=False)
    with pytest.raises(ValueError, match="ordered"):
        ParameterRow("slope", 1.0, ci_lower=2.0, ci_upper=0.0)
    assert (
        Metric("AIC", None, available=False, unavailable_reason="undefined").status
        == "unavailable"
    )


@pytest.mark.integration
def test_plot_renderers_use_identical_precomputed_coordinates() -> None:
    """Both renderers consume identical supplied points, intervals and references."""
    specification = PlotSpec(
        "coefficient_interval",
        x=(1.0, 2.0),
        y=(0.12345678901234567, -0.5),
        lower=(-0.2, -0.8),
        upper=(0.5, -0.1),
        interval_meaning="95% marginal normal intervals\r\\input{untrusted}",
        labels=("α_one", "β%two"),
        reference_lines=(ReferenceLine((0.5, 2.5), (0.0, 0.0), "zero"),),
    )
    figure = specification.to_matplotlib()
    try:
        axis = figure.axes[0]
        np.testing.assert_array_equal(
            axis.collections[0].get_offsets(),
            np.column_stack((specification.x, specification.y)),
        )
        intervals = axis.collections[1]
        assert isinstance(intervals, LineCollection)
        np.testing.assert_array_equal(
            intervals.get_segments(),
            [[[1.0, -0.2], [1.0, 0.5]], [[2.0, -0.8], [2.0, -0.1]]],
        )
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0.0, 0.0])
        legend = axis.get_legend()
        assert legend is not None
        assert [label.get_text() for label in legend.get_texts()] == ["zero"]
        pgf = specification.to_pgfplots()
        assert "(1,0.12345678901234566)" in pgf
        assert "(1,-0.2) (1,0.5)" in pgf
        assert r"$\alpha$\_one" in pgf
        assert r"$\beta$\%two" in pgf
        assert not any(line.startswith(r"\input") for line in pgf.splitlines())
    finally:
        plt.close(figure)


@pytest.mark.unit
@pytest.mark.contract
def test_interval_plot_requires_meaning_without_calculating_uncertainty() -> None:
    """Interval geometry alone never invents a confidence interpretation."""
    with pytest.raises(ValueError, match="meaning"):
        PlotSpec("coefficient_interval", x=(1.0,), y=(2.0,), lower=(1.0,), upper=(3.0,))


@pytest.mark.component
@pytest.mark.contract
def test_latex_compact_precision_and_notes() -> None:
    """Compact rendering preserves values and escapes retained notes."""
    report = Report(
        "Fit",
        parameters=(ParameterRow("x_1", 1.2e-9, ci_lower=1e-10, ci_upper=2e-9),),
        notes=("95% intervals & HC1",),
        metadata={"source": "example"},
    )
    latex = report.to_latex(digits=3)
    assert r"$1.20 \times 10^{-9}$" in latex
    assert "Statistic" not in latex
    assert "SE" not in latex
    assert r"95\% intervals \& HC1" in latex
    assert '% {"source": "example"}' in latex
    assert "95% intervals & HC1" in report.to_text()
    assert "95% intervals &amp; HC1" in report.to_html()
    assert report.parameters[0].estimate == pytest.approx(1.2e-9)
    with pytest.raises(ValueError, match="digits"):
        report.to_latex(digits=0)


def _worked_report() -> Report:
    return Report(
        "Poisson inference",
        parameters=(
            ParameterRow(
                "intercept",
                -0.49421,
                std_error=0.060761,
                ci_lower=-0.61330,
                ci_upper=-0.37512,
            ),
            ParameterRow(
                "load",
                0.29980,
                std_error=0.016360,
                ci_lower=0.26773,
                ci_upper=0.33187,
            ),
        ),
        metrics=(Metric("n_observations", 400),),
        metadata={"confidence": 0.95},
        notes=("95% intervals; normal reference; covariance HC1",),
    )


_WORKED_LATEX = r"""% Poisson inference
\begin{tabular}{@{}lS[table-format=-1.4,group-digits=integer]S[table-format=1.5,group-digits=integer]S[table-format=-1.4,group-digits=integer]S[table-format=-1.4,group-digits=integer]@{}}
\toprule
Parameter & {Estimate} & {SE} & \multicolumn{2}{c}{95\% CI} \\
\cmidrule(lr){4-5}
 & & & {Lower} & {Upper} \\
\midrule
intercept & -0.4942 & 0.06076 & -0.6133 & -0.3751 \\
Facility load & 0.2998 & 0.01636 & 0.2677 & 0.3319 \\
\midrule
Observations & {\num{400}} & & & \\
\bottomrule
\end{tabular}
\par 95\% intervals; normal reference; covariance HC1
% {"confidence": 0.95}
"""


@pytest.mark.unit
def test_latex_worked_example_is_exact() -> None:
    """The specification's worked example renders byte for byte."""
    latex = _worked_report().to_latex(
        digits=4, labels={"load": "Facility load", "n_observations": "Observations"}
    )
    assert latex == _WORKED_LATEX


@pytest.mark.unit
@pytest.mark.parametrize("digits", [1, 3, 6, 12])
def test_integer_values_are_exact_in_every_renderer(digits: int) -> None:
    """Integers are never rounded or put into an exponent."""
    count = 12_345_678_901
    report = Report(
        "Counts",
        parameters=(ParameterRow("n", count),),
        metrics=(Metric("n_observations", count),),
    )
    latex = report.to_latex(digits=digits)
    assert latex.count(str(count)) == 2  # noqa: PLR2004
    assert "\\times" not in latex
    assert str(count) in report.to_text()
    assert str(count) in report.to_html()
    assert "e+" not in report.to_text()


@pytest.mark.unit
def test_columns_keep_trailing_zeros_and_compute_table_format() -> None:
    """Columns align on kept zeros with a computed table-format."""
    report = Report(
        "Zeros",
        parameters=(
            ParameterRow("a", 1.975, std_error=2.0),
            ParameterRow("b", -12.5, std_error=0.25),
            ParameterRow("c", 123456.0, std_error=1.0),
        ),
        metrics=(Metric("r2", 0.5),),
    )
    latex = report.to_latex(digits=4)
    assert "S[table-format=-6.4,group-digits=integer]" in latex
    assert "S[table-format=1.4,group-digits=integer]" in latex
    assert "a & 1.975 & 2.000 \\\\" in latex
    assert "b & -12.50 & 0.2500 \\\\" in latex
    assert "c & 123456 & 1.000 \\\\" in latex
    assert "r2 & 0.5000 & \\\\" in latex


@pytest.mark.unit
def test_decimals_fix_places_and_digits_stay_significant() -> None:
    """Decimals fix places while digits stays the default."""
    report = Report("Places", parameters=(ParameterRow("a", 2.0, std_error=0.0123456),))
    fixed = report.to_latex(decimals=2)
    assert "a & 2.00 & 0.01 \\\\" in fixed
    assert "table-format=1.2" in fixed
    assert "a & 2.00000 & 0.0123456 \\\\" in report.to_latex()
    assert "a & 2.0 & 0.012 \\\\" in report.to_latex(digits=2)
    with pytest.raises(ValueError, match="decimals"):
        report.to_latex(decimals=-1)
    with pytest.raises(ValueError, match="digits"):
        report.to_latex(digits=0)


@pytest.mark.unit
def test_unavailable_cells_are_dashes_and_reasons_remain_notes() -> None:
    """A dash marks an unavailable cell and its reason stays a note."""
    report = Report(
        "Missing",
        parameters=(
            ParameterRow("a", 1.5, std_error=0.5),
            ParameterRow(
                "b",
                None,
                valid=False,
                unavailable_reason="rank deficient",
            ),
        ),
        metrics=(Metric("aic", None, available=False, unavailable_reason="undefined"),),
    )
    latex = report.to_latex(digits=2)
    assert "b & {---} & {---} \\\\" in latex
    assert "AIC & {---} & \\\\" in latex
    assert "unavailable &" not in latex
    assert r"\par b: rank deficient" in latex
    assert r"\par AIC: undefined" in latex


@pytest.mark.unit
def test_confidence_header_spans_both_interval_columns() -> None:
    """Retained confidence heads both interval columns once."""
    row = ParameterRow("a", 1.0, ci_lower=0.5, ci_upper=1.5)
    latex = Report("CI", (row,), metadata={"confidence": 0.9}).to_latex()
    assert r"Parameter & {Estimate} & \multicolumn{2}{c}{90\% CI} \\" in latex
    assert r"\cmidrule(lr){3-4}" in latex
    assert " & & {Lower} & {Upper} \\\\" in latex
    plain = Report("CI", (row,)).to_latex()
    assert "Parameter & {Estimate} & {Lower} & {Upper} \\\\" in plain
    assert "multicolumn" not in plain
    odd = Report("CI", (row,), metadata={"confidence": "high"}).to_latex()
    assert "multicolumn" not in odd


@pytest.mark.unit
def test_pvalues_use_three_decimals_with_a_floor() -> None:
    """Parameter p-values print three decimals with a floor."""
    report = Report(
        "p",
        parameters=(
            ParameterRow("a", 1.0, pvalue=0.0),
            ParameterRow("b", 1.0, pvalue=0.0004),
            ParameterRow("c", 1.0, pvalue=0.01234),
            ParameterRow("d", 1.0, pvalue=1.0),
            ParameterRow("e", 1.0),
        ),
    )
    latex = report.to_latex(digits=2, decimals=5)
    assert "$<0.001$" in latex
    assert "$0.012$" in latex
    assert "$1.000$" in latex
    assert "e & 1.00000 & --- \\\\" in latex
    assert "group-digits=integer]r@{}}" in latex


@pytest.mark.unit
def test_metrics_close_parameters_and_stand_alone_without_them() -> None:
    """Metrics become footer rows or a table of their own."""
    both = Report(
        "Fit",
        parameters=(ParameterRow("a", 1.5),),
        metrics=(Metric("n", 20), Metric("r2", 0.25), Metric("big", 1.5e-9)),
    )
    latex = both.to_latex(digits=3)
    assert latex.count(r"\begin{tabular}") == 1
    body = latex.split(r"\midrule")
    assert len(body) == 3  # noqa: PLR2004
    assert r"n & {\num{20}} \\" in latex
    assert "r2 & 0.250 \\\\" in latex
    assert r"big & {$1.50 \times 10^{-9}$} \\" in latex
    only = Report("M", metrics=(Metric("n", 20), Metric("r2", 0.25))).to_latex()
    assert latex.count("Metric") == 0
    assert only.startswith("% M\n\\begin{tabular}{@{}lS[table-format=")
    assert r"Metric & {Value} \\" in only
    assert "r2 & 0.250000 \\\\" in only


@pytest.mark.unit
def test_scientific_cells_are_braced_math() -> None:
    """Scientific cells are braced math inside S columns."""
    rows = (ParameterRow("a", 1.2e-9), ParameterRow("b", 1.5))
    latex = Report("S", rows).to_latex(digits=3)
    assert r"a & {$1.20 \times 10^{-9}$} \\" in latex
    assert "table-format=1.2," in latex
    alone = Report("S", rows[:1]).to_latex(digits=3)
    assert "@{}lc@{}" in alone
    assert r"a & $1.20 \times 10^{-9}$ \\" in alone


@pytest.mark.unit
def test_labels_are_plain_by_default_tex_on_request_and_ignore_extras() -> None:
    """Display labels are escaped unless declared TeX."""
    report = Report("L", (ParameterRow("x_1", 1.0), ParameterRow("y_2", 2.0)))
    labels = {"x_1": r"$\beta_1$ & more", "unused": "ignored"}
    plain = report.to_latex(labels=labels)
    assert r"\$\textbackslash{}beta\_1\$ \& more & 1.0" in plain
    assert r"y\_2 & 2.0" in plain
    tex = report.to_latex(labels=labels, label_markup="tex")
    assert r"$\beta_1$ & more & 1.0" in tex
    assert r"y\_2 & 2.0" in tex
    with pytest.raises(ValueError, match="newline"):
        report.to_latex(labels={"x_1": "a\nb"}, label_markup="tex")


@pytest.mark.unit
def test_tex_replaces_glyphs_the_fonts_lack() -> None:
    """Non-ASCII glyphs become math or ASCII without font sizes."""
    from statadic.api.reporting import _tex  # noqa: PLC0415

    assert _tex("χ² ≥ α — β…") == r"$\chi$$^{2}$ $\geq$ $\alpha$ --- $\beta$\ldots{}"
    symbols = "\u2212\u00d7\u00b7\u00b1\u2264\u2260\u2248\u221e\u2192\u2013\u00b3"
    assert _tex(symbols) == (
        r"$-$$\times$$\cdot$$\pm$$\leq$$\neq$$\approx$$\infty$$\to$--$^{3}$"
    )
    assert _tex("οΑΩωΓ") == r"o$\mathrm{A}$$\Omega$$\omega$$\Gamma$"
    for text in ("χ²≥—…", "Σσς"):
        for command in (r"\textsuperscript", r"\footnotesize", r"\small"):
            assert command not in _tex(text)
        assert _tex(text).isascii()
    latex = Report("t", (ParameterRow("χ", 1.0),), notes=("n ≥ 5",)).to_latex()
    assert r"$\chi$ & 1.0" in latex
    assert r"\par n $\geq$ 5" in latex


@pytest.mark.unit
def test_metrics_select_and_order_footer_rows(tmp_path: Path) -> None:
    """Selected metrics pick footer rows in order, in both tables; None is unchanged."""
    metrics = (Metric("n", 20), Metric("r2", 0.25), Metric("aic", 3.5))
    report = Report("Fit", parameters=(ParameterRow("a", 1.5),), metrics=metrics)
    default = report.to_latex()
    assert report.to_latex(metrics=None) == default
    assert report.to_latex(metrics=("n", "r2", "aic")) == default
    picked = report.to_latex(metrics=("aic", "n"))
    assert picked.index("AIC &") < picked.index("n &")
    assert "r2 &" not in picked
    assert report.to_latex(metrics=["r2"]).count(r"\midrule") == 2  # noqa: PLR2004
    bare = report.to_latex(metrics=())
    assert bare.count(r"\midrule") == 1
    assert "AIC" not in bare
    only = Report("M", metrics=metrics)
    assert only.to_latex(metrics=None) == only.to_latex()
    ordered = only.to_latex(metrics=("r2", "n"))
    assert ordered.index("r2 &") < ordered.index("n &")
    assert "AIC" not in ordered
    assert r"\begin{tabular}" not in only.to_latex(metrics=())
    assert only.to_text() == only.to_text()
    target = tmp_path / "t.tex"
    report.save_latex(target, metrics=("n",), notes=False)
    assert target.read_text(encoding="utf-8") == report.to_latex(
        metrics=("n",), notes=False
    )


@pytest.mark.unit
def test_metrics_selection_rejects_unknown_and_duplicate_names() -> None:
    """A bad name raises ValueError naming it."""
    report = Report("Fit", metrics=(Metric("n", 20), Metric("r2", 0.25)))
    with pytest.raises(ValueError, match="'bic'"):
        report.to_latex(metrics=("n", "bic"))
    with pytest.raises(ValueError, match="'n'"):
        report.to_latex(metrics=("n", "n"))
    with pytest.raises(TypeError, match="sequence"):
        report.to_latex(metrics="n")


@pytest.mark.unit
def test_notes_can_be_omitted() -> None:
    """notes=False drops every trailing note line."""
    report = _worked_report()
    report = replace(
        report,
        metrics=(Metric("n_observations", 400, definition="Row count."),),
        warnings=("careful",),
    )
    assert r"\par" in report.to_latex()
    quiet = report.to_latex(notes=False)
    assert r"\par" not in quiet
    assert quiet.startswith("% Poisson inference\n")
    assert quiet.endswith('% {"confidence": 0.95}\n')


@pytest.mark.unit
def test_save_latex_writes_the_exact_text(tmp_path: Path) -> None:
    """save_latex writes to_latex as UTF-8 and creates no directory."""
    report = Report("é", (ParameterRow("χ", 1.0),))
    target = Path(tmp_path) / "table.tex"
    report.save_latex(target, digits=3, notes=False)
    assert target.read_bytes() == report.to_latex(digits=3, notes=False).encode()
    report.save_latex(str(target))
    assert target.read_text(encoding="utf-8") == report.to_latex()
    assert b"\r" not in target.read_bytes()
    with pytest.raises(FileNotFoundError):
        report.save_latex(tmp_path / "missing" / "table.tex")
    with pytest.raises(ValueError, match="digits"):
        report.save_latex(target, digits=0)


@pytest.mark.unit
def test_pgfplots_numbers_options_styles_and_save(tmp_path: Path) -> None:
    """Compact numbers, empty options, cycle-list styling and direct saving."""
    plain = PlotSpec("line", (0.0, 1.0), (0.1, 3.0))
    text = plain.to_pgfplots()
    assert r"\begin{axis}[width=12cm,height=7cm]" in text
    assert r"\addplot+[no marks,forget plot] coordinates {(0,0.1) (1,3)};" in text
    assert "cycle list shift" not in text
    assert "xlabel" not in text
    assert "title" not in text
    titled = PlotSpec("line", (0.0,), (1.0,), xlabel="x", title="t")
    assert "xlabel={x},title={t}]" in titled.to_pgfplots()
    assert "ylabel" not in titled.to_pgfplots()

    unlabelled = PlotSpec(
        "scatter",
        (0.0, 1.0),
        (1.0, 2.0),
        overlays=(
            PlotSpec("line", (0.0, 1.0), (2.0, 3.0), label="b"),
            PlotSpec("line", (0.0, 1.0), (3.0, 4.0)),
            PlotSpec("line", (0.0, 1.0), (4.0, 5.0)),
        ),
        reference_lines=(ReferenceLine((0.0, 1.0), (0.0, 0.0), "ref"),),
    )
    lines = unlabelled.to_pgfplots().splitlines()
    assert lines[2:] == [
        r"\addplot+[only marks,forget plot] coordinates {(0,1) (1,2)};",
        r"\addplot[black,dashed,no marks] coordinates {(0,0) (1,0)};",
        r"\addlegendentry{ref}",
        r"\pgfplotsset{cycle list shift=1}",
        r"\addplot+[no marks] coordinates {(0,2) (1,3)};",
        r"\addlegendentry{b}",
        r"\addplot+[no marks,forget plot] coordinates {(0,3) (1,4)};",
        r"\pgfplotsset{cycle list shift=2}",
        r"\addplot+[no marks,forget plot] coordinates {(0,4) (1,5)};",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]

    bars = PlotSpec(
        "coefficient_interval",
        (0.0, 1.0),
        (0.5, 1.5),
        lower=(0.25, 1.0),
        upper=(0.75, 2.0),
        interval_meaning="m",
        labels=("a", "b"),
    ).to_pgfplots()
    assert "xtick={0,1}" in bars
    assert r"\addplot+[no marks,forget plot] coordinates {(0,0.25) (0,0.75)};" in bars

    target = tmp_path / "figure.tex"
    plain.save_pgfplots(target, width_cm=8, height_cm=5)
    assert target.read_text(encoding="utf-8") == plain.to_pgfplots(
        width_cm=8, height_cm=5
    )
    assert b"\r" not in target.read_bytes()
    with pytest.raises(FileNotFoundError):
        plain.save_pgfplots(tmp_path / "missing" / "figure.tex")
    with pytest.raises(ValueError, match="dimensions"):
        plain.save_pgfplots(target, width_cm=0)


@pytest.mark.integration
def test_ribbon_overlay_and_plain_labels() -> None:
    """Both renderers preserve overlays and plain labels."""
    overlay = PlotSpec("line", (0.0, 1.0), (2.0, 3.0), label="other")
    plot = PlotSpec(
        "line",
        (0.0, 1.0),
        (1.0, 2.0),
        lower=(0.0, 1.0),
        upper=(2.0, 3.0),
        interval_meaning="interval",
        interval_style="ribbon",
        label="$x$",
        overlays=(overlay,),
    )
    pgf = plot.to_pgfplots()
    assert r"-- cycle;" in pgf
    assert "draw=none,mark=none,fill=.,fill opacity=0.2,forget plot" in pgf
    assert pgf.count(r"\addplot+[no marks] ") == 2  # noqa: PLR2004
    assert "color=" not in pgf
    assert "cycle list shift" not in pgf
    assert r"\$x\$" in pgf
    assert pgf.count(r"\addlegendentry") == 2  # noqa: PLR2004
    fig = plot.to_matplotlib()
    assert len(fig.axes[0].collections) == 1
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert not legend.get_texts()[0].get_parse_math()
    plt.close(fig)
    with pytest.raises(ValueError, match="Ribbons"):
        PlotSpec(
            "line",
            (1.0, 0.0),
            (1.0, 2.0),
            lower=(0.0, 1.0),
            upper=(2.0, 3.0),
            interval_meaning="interval",
            interval_style="ribbon",
        )
    with pytest.raises(ValueError, match="newlines"):
        PlotSpec("line", (0.0,), (1.0,), markup="tex", label="bad\nlabel")
    with pytest.raises(ValueError, match="Overlays"):
        PlotSpec(
            "line",
            (0.0,),
            (1.0,),
            overlays=(PlotSpec("line", (0.0,), (1.0,), title="invalid"),),
        )


@pytest.mark.unit
def test_pgfplots_prints_integer_coordinates_exactly() -> None:
    """Integer coordinates, as permutation indices produce, print exactly."""
    plot = PlotSpec(
        "scatter",
        (0, 1, 2),
        (1.5, 2.0, 2**60),
        reference_lines=(ReferenceLine((0, 2), (1, 1), label="observed"),),
    )
    text = plot.to_pgfplots()
    assert "(0,1.5) (1,2) (2,1152921504606846976)" in text
    assert "(0,1) (2,1)" in text


def _tests_report() -> Report:
    return Report(
        "Tests",
        tests=(
            TestRow(
                "jarque_bera",
                1.39938,
                0.496739,
                "chi-square",
                (2,),
                "chi2(2)",
                "Residuals are normal.",
            ),
            TestRow("anova", 4.5, 1e-9, "f", (1, 78), "F(1, 78)"),
            TestRow(
                "perm",
                0.25,
                0.0123,
                "resampled",
                (999,),
                "random resampling (999 draws)",
            ),
            TestRow("durbin_watson", 1.9),
            TestRow(
                "reset",
                None,
                available=False,
                unavailable_reason="rank deficient",
            ),
        ),
    )


@pytest.mark.unit
def test_test_rows_render_as_latex_table() -> None:
    """Chi-square, F, resampled, descriptive and unavailable rows share a table."""
    latex = _tests_report().to_latex()
    assert r"Test & {Statistic} & Reference & $p$ \\" in latex
    assert r"jarque\_bera & 1.39938 & $\chi^2(2)$ & $0.497$ \\" in latex
    assert "anova & 4.50000 & $F(1, 78)$ & $<0.001$ \\" in latex
    assert "perm & 0.250000 & random resampling (999 draws) & $0.012$ \\" in latex
    assert "durbin\\_watson & 1.90000 & --- & --- \\" in latex
    assert "reset & {---} & --- & --- \\" in latex
    assert r"\par jarque\_bera: Residuals are normal." in latex
    assert r"\par reset: rank deficient" in latex
    assert "\\par jarque" not in _tests_report().to_latex(notes=False)


@pytest.mark.unit
def test_descriptive_tests_omit_reference_and_pvalue_columns() -> None:
    """Exercise this test-row behaviour."""
    latex = Report("D", tests=(TestRow("dw", 1.9),)).to_latex()
    assert "@{}lS[table-format=1.5,group-digits=integer]@{}" in latex
    assert "Test & {Statistic} \\\\" in latex
    assert "Reference" not in latex


@pytest.mark.unit
def test_law_cells_and_table_order() -> None:
    """Exercise this test-row behaviour."""
    normal = Report(
        "N",
        parameters=(ParameterRow("a", 1.0),),
        metrics=(Metric("n", 3),),
        tests=(
            TestRow("z0", 1.0, 0.3, "normal", (0, 1), "N(0, 1)"),
            TestRow("z1", 1.0, 0.3, "normal", (0.5, 2), "N(0.5, 4)"),
        ),
    ).to_latex()
    assert "$N(0, 1)$" in normal
    assert r"$N(\num{0.5}, \num{2}^2)$" in normal
    assert normal.index("Parameter") < normal.index("\\par\\medskip")
    assert normal.index("\\par\\medskip") < normal.index("Test &")
    only = Report("M", metrics=(Metric("n", 3),), tests=(TestRow("dw", 1.9),))
    text = only.to_latex()
    assert text.index("Test &") < text.index("\\par\\medskip") < text.index("Metric &")


@pytest.mark.unit
def test_test_rows_render_text_and_escaped_html() -> None:
    """Exercise this test-row behaviour."""
    report = _tests_report()
    text = report.to_text()
    assert "Test\tStatistic\tReference\tp-value" in text
    assert "jarque_bera\t1.39938\tchi2(2)\t0.496739" in text
    assert "durbin_watson\t1.9\tunavailable\tunavailable" in text
    assert "reset: rank deficient (status=unavailable)" in text
    assert "jarque_bera: Residuals are normal." in text
    page = Report("H", tests=(TestRow("a<b", 1.0, hypothesis="x & y"),)).to_html()
    assert "<th>Reference</th>" in page
    assert "<td>a&lt;b</td>" in page
    assert "<p>a&lt;b: x &amp; y</p>" in page


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reference", "head"), [("normal", "z"), ("t", "t"), ("other", "Statistic")]
)
def test_statistic_header_names_the_reference(reference: str, head: str) -> None:
    """Exercise this test-row behaviour."""
    report = Report(
        "I",
        parameters=(ParameterRow("a", 1.0, statistic=2.0),),
        metadata={"reference": reference},
    )
    expected = "Statistic" if head == "Statistic" else f"${head}$"
    assert f"{{{expected}}}" in report.to_latex()
    assert f"\t{head}\t" in report.to_text()
    assert f"<th>{head}</th>" in report.to_html()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"name": ""}, ValueError),
        ({"name": 1}, TypeError),
        ({"statistic": float("nan")}, ValueError),
        ({"statistic": None}, ValueError),
        ({"pvalue": 0.5}, ValueError),
        ({"law": "chi-square", "law_parameters": (2,)}, ValueError),
        ({"law": "chi-square", "pvalue": 1.5, "law_parameters": (2,)}, ValueError),
        ({"law": "chi-square", "pvalue": 0.5, "law_parameters": (2, 3)}, ValueError),
        ({"law": "chi-square", "pvalue": 0.5, "law_parameters": (0,)}, ValueError),
        ({"law": "f", "pvalue": 0.5, "law_parameters": (1,)}, ValueError),
        ({"law": "normal", "pvalue": 0.5, "law_parameters": (0, 0)}, ValueError),
        ({"law": "resampled", "pvalue": 0.5, "law_parameters": (9.5,)}, ValueError),
        ({"law_parameters": (1,)}, ValueError),
        ({"law_parameters": 1}, TypeError),
        ({"available": 1}, TypeError),
        ({"reference": 1}, TypeError),
        ({"unavailable_reason": "x"}, ValueError),
        ({"available": False, "statistic": None}, ValueError),
        (
            {"available": False, "unavailable_reason": "x"},
            ValueError,
        ),
    ],
)
def test_test_row_validation(kwargs: dict[str, object], error: type[Exception]) -> None:
    """Exercise this test-row behaviour."""
    values: dict[str, object] = {"name": "t", "statistic": 1.0, **kwargs}
    with pytest.raises(error):
        TestRow(**ill_typed(values))


@pytest.mark.unit
def test_unavailable_test_status_and_report_checks() -> None:
    """Exercise this test-row behaviour."""
    row = TestRow("t", None, available=False, unavailable_reason="why")
    assert row.status == "unavailable"
    with pytest.raises(ValueError, match="unique"):
        Report("R", tests=(TestRow("a", 1.0), TestRow("a", 2.0)))
    with pytest.raises(TypeError):
        Report("R", tests=ill_typed((Metric("a", 1.0),)))
    many = Report("R", tests=tuple(TestRow(f"t{i}", 1.0) for i in range(30)))
    preview = many._repr_html_()
    assert "t11" in preview
    assert "t12" not in preview


def _values_report() -> Report:
    return Report(
        "Values & check",
        parameters=(
            ParameterRow("intercept", -0.49421, 0.060761, -8.13, 0.497, -0.6, -0.4),
            ParameterRow("x", 0.2998, 0.05, 6.0, 1e-9, 0.2, 0.4),
            ParameterRow("rho", None, valid=False, unavailable_reason="singular"),
        ),
        metrics=(Metric("n_observations", 400),),
        tests=(
            TestRow("jarque_bera", 1.39938, 0.496739, "chi-square", (2,), "chi2(2)"),
        ),
    )


@pytest.mark.unit
@pytest.mark.contract
def test_to_values_declares_every_available_quantity() -> None:
    """Keys, order, value text, relations and unavailable comments are exact."""
    text = _values_report().to_values(prefix="ols", decimals=4)
    lines = text.splitlines()
    assert lines[0] == "% Generated by statadic Report.save_values from: Values & check"
    assert all(line.startswith(r"\providecommand") for line in lines[1:4])
    assert lines[4:] == [
        r"\jcDeclareValue{ols-intercept-estimate}{-0.4942}",
        r"\jcDeclareValue{ols-intercept-se}{0.0608}",
        r"\jcDeclareValue{ols-intercept-statistic}{-8.1300}",
        r"\jcDeclareValue{ols-intercept-p}{= 0.497}",
        r"\jcDeclareValue{ols-intercept-lower}{-0.6000}",
        r"\jcDeclareValue{ols-intercept-upper}{-0.4000}",
        r"\jcDeclareValue{ols-x-estimate}{0.2998}",
        r"\jcDeclareValue{ols-x-se}{0.0500}",
        r"\jcDeclareValue{ols-x-statistic}{6.0000}",
        r"\jcDeclareValue{ols-x-p}{< 0.001}",
        r"\jcDeclareValue{ols-x-lower}{0.2000}",
        r"\jcDeclareValue{ols-x-upper}{0.4000}",
        "% unavailable: ols-rho-estimate (singular)",
        r"\jcDeclareValue{ols-n-observations}{400}",
        r"\jcDeclareValue{ols-jarque-bera-statistic}{1.3994}",
        r"\jcDeclareValue{ols-jarque-bera-p}{= 0.497}",
        r"\jcDeclareValue{ols-jarque-bera-df}{2}",
    ]


@pytest.mark.unit
def test_to_values_significant_digits_and_f_degrees() -> None:
    """Small floats use a negative exponent; F laws declare both degrees."""
    report = Report(
        "t",
        parameters=(ParameterRow("a", 1.2e-9),),
        tests=(TestRow("f", 4.5, 0.01, "f", (1, 78.5), "F"),),
    )
    text = report.to_values(prefix="m", digits=3)
    assert r"{m-a-estimate}{1.20e-9}" in text
    assert r"{m-f-df1}{1}" in text
    assert r"{m-f-df2}{78.50}" in text
    assert r"{m-f-statistic}{4.50}" in text
    assert r"{m-f-p}{= 0.010}" in text


@pytest.mark.unit
def test_to_values_lists_unavailable_and_omits_unrequested() -> None:
    """Unavailable metrics, tests and intervals are comments; unrequested are absent."""
    report = Report(
        "t",
        parameters=(
            ParameterRow(
                "b",
                1.0,
                interval_status="unavailable",
                interval_unavailable_reason="bad\ncalibration",
            ),
        ),
        metrics=(Metric("aic", None, available=False, unavailable_reason="undefined"),),
        tests=(TestRow("reset", None, available=False, unavailable_reason="rank"),),
    )
    text = report.to_values(prefix="m")
    assert "% unavailable: m-b-lower (bad calibration)" in text
    assert "% unavailable: m-b-upper (bad calibration)" in text
    assert "% unavailable: m-aic (undefined)" in text
    assert "% unavailable: m-reset-statistic (rank)" in text
    declared = [x for x in text.splitlines() if x.startswith(r"\jcDeclareValue")]
    assert declared == [r"\jcDeclareValue{m-b-estimate}{1.00000}"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "report",
    [
        Report("t", parameters=(ParameterRow("x_1", 1.0), ParameterRow("x-1", 2.0))),
        Report(
            "t",
            parameters=(ParameterRow("x", 1.0),),
            metrics=(Metric("x-estimate", 1),),
        ),
    ],
)
def test_to_values_rejects_duplicate_keys(report: Report) -> None:
    """Both colliding sources are named."""
    with pytest.raises(ValueError, match=r"produced by both .* and ") as error:
        report.to_values(prefix="m")
    assert "m-x-1-estimate" in str(error.value) or "m-x-estimate" in str(error.value)
    assert "parameter" in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["", "Ols", "1a", "a_b"])
def test_to_values_rejects_bad_options(prefix: str) -> None:
    """Prefix, digits, decimals and empty slugs raise."""
    report = Report("t", parameters=(ParameterRow("x", 1.0),))
    with pytest.raises(ValueError, match="prefix"):
        report.to_values(prefix=prefix)
    with pytest.raises(ValueError, match="digits"):
        report.to_values(prefix="m", digits=0)
    with pytest.raises(ValueError, match="decimals"):
        report.to_values(prefix="m", decimals=-1)
    with pytest.raises(ValueError, match="value key"):
        Report("t", parameters=(ParameterRow("--", 1.0),)).to_values(prefix="m")


@pytest.mark.unit
def test_save_values_writes_the_exact_text(tmp_path: Path) -> None:
    """The file is exactly ``to_values`` with newline endings."""
    report = _values_report()
    path = tmp_path / "values.tex"
    report.save_values(path, prefix="ols", decimals=4)
    assert path.read_bytes() == report.to_values(prefix="ols", decimals=4).encode()
    assert b"\r" not in path.read_bytes()
    with pytest.raises(FileNotFoundError):
        report.save_values(tmp_path / "missing/values.tex", prefix="ols")


_FIT_LABELS = {
    "n_observations": "Observations",
    "n_parameters": "Parameters",
    "r_squared": "$R^2$",
    "adjusted_r_squared": "Adjusted $R^2$",
    "uncentred_r_squared": "Uncentred $R^2$",
    "log_likelihood": "Log likelihood",
    "aic": "AIC",
    "bic": "BIC",
    "hqic": "HQIC",
}


def _fit_report(*, parameters: bool = True) -> Report:
    return Report(
        "Fit",
        parameters=(ParameterRow("r_squared", 1.5, std_error=0.5),)
        if parameters
        else (),
        metrics=tuple(Metric(name, 2.5) for name in _FIT_LABELS),
    )


@pytest.mark.unit
@pytest.mark.parametrize("parameters", [True, False])
def test_latex_labels_fit_statistics_by_default(*, parameters: bool) -> None:
    """Standard fit-statistic names get TeX labels in footers and metric tables."""
    latex = _fit_report(parameters=parameters).to_latex()
    for name, label in _FIT_LABELS.items():
        assert f"\n{label} & " in latex
        if name != "r_squared" or not parameters:
            assert f"\n{name.replace('_', chr(92) + '_')} &" not in latex
    if parameters:
        assert "\nr\\_squared & 1.50000 & 0.500000 \\\\" in latex


@pytest.mark.unit
def test_latex_fit_label_override_and_markup() -> None:
    """A caller label wins and keeps its markup rule."""
    report = _fit_report(parameters=False)
    plain = report.to_latex(labels={"aic": "A & B", "r_squared": r"$R^2$"})
    assert "\nA \\& B & " in plain
    assert "\n\\$R\\textasciicircum{}2\\$ & " in plain or "\\$R" in plain
    assert "\nAIC & " not in plain
    tex = report.to_latex(labels={"aic": r"$\mathrm{AIC}$"}, label_markup="tex")
    assert "\n$\\mathrm{AIC}$ & " in tex
    assert "\nBIC & " in tex


@pytest.mark.unit
def test_latex_notes_use_metric_labels() -> None:
    """Notes naming a labelled metric use its label; other names stay escaped."""
    report = Report(
        "Notes",
        metrics=(
            Metric("log_likelihood", None, available=False, unavailable_reason="no_ll"),
            Metric("n_observations", 4, definition="Row_count."),
            Metric("custom_stat", None, available=False, unavailable_reason="x"),
        ),
    )
    latex = report.to_latex()
    assert "\\par Log likelihood: no\\_ll" in latex
    assert "\\par Observations: Row\\_count." in latex
    assert "\\par custom\\_stat: x" in latex
    assert "\\par Custom: x" in report.to_latex(labels={"custom_stat": "Custom"})


@pytest.mark.unit
def test_fit_labels_stay_out_of_other_renderers() -> None:
    """Text, HTML and values keep the retained names."""
    report = _fit_report()
    assert "n_observations" in report.to_text()
    assert "n_observations" in report.to_html()
    assert "Observations" not in report.to_text()
    assert "ols-n-observations" in report.to_values(prefix="ols")


def _state_report(**row: object) -> Report:
    return Report(
        "state",
        parameters=(ParameterRow("slope", 1.0, **row),),  # type: ignore[arg-type]
        metadata={"reference": "normal"},
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"interval_status": "unavailable", "ci_lower": 0.0, "ci_upper": 2.0},
            "cannot report endpoints",
        ),
        (
            {
                "interval_status": "unavailable",
                "interval_unavailable_reason": "why",
                "ci_upper": 2.0,
            },
            "cannot report endpoints",
        ),
        ({"interval_status": "unavailable"}, "requires a reason"),
        ({"interval_status": 3}, "requires a reason"),
        ({"interval_status": "available"}, "requires both endpoints"),
        (
            {"interval_status": "available", "ci_lower": 0.0},
            "(?i)both",
        ),
    ],
)
def test_contradictory_interval_states_are_rejected(
    kwargs: dict[str, object], match: str
) -> None:
    """Declared interval status must agree with endpoints and reason."""
    with pytest.raises(ValueError, match=match):
        ParameterRow("slope", 1.0, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.contract
def test_metric_unavailable_with_value_is_rejected() -> None:
    """An unavailable metric cannot also carry a value."""
    with pytest.raises(ValueError, match="report no value"):
        Metric("AIC", 1.0, available=False, unavailable_reason="why")


@pytest.mark.unit
@pytest.mark.contract
def test_interval_state_matrix_follows_declared_status() -> None:
    """Every output and require_available agree with the interval declaration."""
    available = _state_report(
        std_error=0.5, ci_lower=0.0, ci_upper=2.0, interval_status="available"
    )
    unrequested = _state_report(std_error=0.5, ci_lower=0.0, ci_upper=2.0)
    withheld = _state_report(
        std_error=0.5,
        interval_status="unavailable",
        interval_unavailable_reason="bad reference",
    )
    partial = Report(
        "partial",
        parameters=(
            ParameterRow("slope", 1.5, valid=False, unavailable_reason="no covariance"),
        ),
    )

    assert available.parameters[0].interval_available
    assert unrequested.parameters[0].interval_available
    assert not withheld.parameters[0].interval_available
    available.require_available("interval")
    unrequested.require_available("interval")
    with pytest.raises(ValueError, match=r"interval unavailable \(bad reference"):
        withheld.require_available("interval")

    assert "state-slope-lower" in available.to_values(prefix="state")
    assert "% unavailable: ols-slope-lower" in withheld.to_values(prefix="ols")
    assert "ols-slope-lower}" not in withheld.to_values(prefix="ols").replace(
        "% unavailable: ols-slope-lower", ""
    )
    for render in (withheld.to_text(), withheld.to_html()):
        assert "bad reference" in render
    assert "bad reference" in withheld.to_latex()
    assert "0.5" in withheld.to_text()

    # Partial result: the estimate stays visible and exported, SE is required.
    assert "1.5" in partial.to_text()
    assert "p-slope-estimate" in partial.to_values(prefix="p")
    partial.require_available("estimate")
    with pytest.raises(ValueError, match="std_error unavailable"):
        partial.require_available("std_error")


@pytest.mark.unit
def test_small_normal_reference_parameters_keep_significant_digits() -> None:
    """A positive scale never renders as zero in the LaTeX reference cell."""
    row = TestRow("shift", 1.5, 0.13, "normal", (-0.0001, 0.001), "N(-0.0001, 0.001^2)")
    report = Report("N", tests=(row,))
    assert r"$N(\num{-0.0001}, \num{0.001}^2)$" in report.to_latex()
    assert "N(-0.0001, 0.001^2)" in report.to_text()
    assert "N(-0.0001, 0.001^2)" in report.to_html()
    tiny = TestRow("t", 1.0, 0.5, "normal", (1e-5, 2e6), "tiny")
    assert r"$N(\num{1e-5}, \num{2000000}^2)$" in Report("N", tests=(tiny,)).to_latex()
    wide = TestRow("w", 1.0, 0.5, "normal", (0.5, 2.5e-7), "w")
    assert r"\num{2.5e-7}" in Report("N", tests=(wide,)).to_latex(digits=3)


@pytest.mark.unit
@pytest.mark.contract
def test_metric_labels_are_keyed_by_the_standard_fit_statistic_names() -> None:
    """The TeX label map covers exactly the named fit statistics, in order."""
    assert tuple(_METRIC_LABELS) == FIT_STATISTIC_NAMES
    assert FIT_STATISTIC_NAMES == (
        "n_observations",
        "n_parameters",
        "r_squared",
        "adjusted_r_squared",
        "uncentred_r_squared",
        "log_likelihood",
        "aic",
        "bic",
        "hqic",
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("available", "value", "reason"), product((False, True), repeat=3)
)
def test_metric_state_combinations(
    *, available: bool, value: bool, reason: bool
) -> None:
    """Availability determines whether a metric may carry a numerical value."""
    fields = {
        "available": available,
        "value": 1.0 if value else None,
        "unavailable_reason": "withheld" if reason else None,
    }
    if expected_metric_state(available=available, has_value=value, has_reason=reason):
        assert Metric("criterion", **ill_typed(fields)).available == available
    else:
        with pytest.raises(ValueError, match=r"metrics|Metric"):
            Metric("criterion", **ill_typed(fields))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("available", "endpoints", "reason"), product((False, True), repeat=3)
)
def test_requested_interval_state_combinations(
    *, available: bool, endpoints: bool, reason: bool
) -> None:
    """Requested intervals expose both endpoints or an unavailable reason."""
    fields = {
        "interval_status": "available" if available else "unavailable",
        "ci_lower": -1.0 if endpoints else None,
        "ci_upper": 1.0 if endpoints else None,
        "interval_unavailable_reason": "withheld" if reason else None,
    }
    if expected_interval_state(available=available, endpoints=endpoints, reason=reason):
        assert (
            ParameterRow("x", 0.0, **ill_typed(fields)).interval_available == available
        )
    else:
        with pytest.raises(ValueError, match="interval"):
            ParameterRow("x", 0.0, **ill_typed(fields))


@pytest.mark.component
@pytest.mark.property
@pytest.mark.contract
@examples(30)
@given(case=reports())
def test_generated_report_renderers_agree_on_availability(case: ReportCase) -> None:
    """Every renderer retains the reason for each withheld piece of evidence."""
    report = case.build()
    withheld = not report.metrics[0].available
    for rendering in (
        report.to_text(),
        report.to_html(),
        report.to_latex(),
        report.to_values(prefix="generated"),
    ):
        for reason in ("coefficient withheld", "criterion withheld", "test withheld"):
            assert (reason in rendering) == withheld
        assert ("interval withheld" in rendering) == (
            not report.parameters[0].interval_available
        )
    values = report.to_values(prefix="generated")
    assert (r"\jcDeclareValue{generated-criterion}" in values) == (not withheld)


@pytest.mark.unit
@pytest.mark.property
@examples(60)
@example(value=-0.0001, digits=3)
@example(value=0.001, digits=3)
@given(value=report_numbers(), digits=st.integers(1, 10))
def test_number_and_normal_parameter_formatting_preserves_magnitude(
    value: float, digits: int
) -> None:
    """Significant-digit output retains tiny scales and normalizes exponents."""
    cell = _number_cell(value, digits, None).text
    match = re.fullmatch(r"\$(.+) \\times 10\^\{(-?\d+)\}\$", cell)
    parsed = Decimal(f"{match[1]}e{match[2]}") if match else Decimal(cell)
    law_text = _law_number_text(value, digits)
    expected = Decimal(value)
    for actual in (parsed, Decimal(law_text)):
        assert abs(actual - expected) <= abs(expected) * Decimal(
            str(0.51 * 10 ** (1 - digits))
        ) + Decimal("5e-324")
    if "e" in law_text:
        exponent = law_text.split("e")[1]
        assert exponent == str(int(exponent))


@pytest.mark.unit
@pytest.mark.property
@examples(30)
@given(text=st.text(alphabet="&#%$_{}\\^~", min_size=1, max_size=20))
def test_tex_specials_are_escaped_compositionally(text: str) -> None:
    """Combining hostile label characters cannot bypass the escape mapping."""
    escapes = {
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "$": r"\$",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    assert _tex(text) == "".join(escapes[character] for character in text)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.contract
@examples(30)
@given(names=st.lists(st.sampled_from(FIT_STATISTIC_NAMES), unique=True))
def test_footer_subset_and_order_are_preserved(names: list[str]) -> None:
    """Both metric-only tables and coefficient footers honor selection order."""
    metrics = tuple(
        Metric(name, index + 1) for index, name in enumerate(FIT_STATISTIC_NAMES)
    )
    labels: dict[str, str] = {
        name: f"chosen{index}label" for index, name in enumerate(FIT_STATISTIC_NAMES)
    }
    for parameters in ((), (ParameterRow("coefficient", 1),)):
        table = Report("selection", parameters=parameters, metrics=metrics).to_latex(
            metrics=names, labels=labels, notes=False
        )
        positions = [table.index(labels[name]) for name in names]
        assert positions == sorted(positions)
        assert all(
            (label in table) == (name in names) for name, label in labels.items()
        )
        with pytest.raises(ValueError, match="unknown"):
            Report("selection", metrics=metrics).to_latex(metrics=[*names, "unknown"])
        if names:
            with pytest.raises(ValueError, match=r"Duplicate|duplicate"):
                Report("selection", metrics=metrics).to_latex(
                    metrics=[*names, names[0]]
                )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("available", "statistic", "pvalue", "law", "reason"),
    product((False, True), repeat=5),
)
def test_test_row_state_combinations(
    *, available: bool, statistic: bool, pvalue: bool, law: bool, reason: bool
) -> None:
    """A calibrated tail has a law; withheld tests have no numerical evidence."""
    valid = (pvalue == law) and (
        (available and statistic and not reason)
        or (not available and not statistic and not pvalue and reason)
    )
    fields = {
        "statistic": 1.0 if statistic else None,
        "pvalue": 0.5 if pvalue else None,
        "law": "normal" if law else None,
        "law_parameters": (0.0, 1.0) if law else (),
        "available": available,
        "unavailable_reason": "withheld" if reason else None,
    }
    if valid:
        assert TestRow("test", **ill_typed(fields)).available == available
    else:
        with pytest.raises(ValueError, match=r"test|statistic|law"):
            TestRow("test", **ill_typed(fields))


@pytest.mark.component
@pytest.mark.property
@examples(30)
@example(location=-0.0001, scale=0.001, digits=3)
@given(
    location=st.floats(-100, 100, allow_subnormal=False),
    scale=st.sampled_from([0.001, 5e-324, 1e-10, 10.0]),
    digits=st.integers(1, 8),
)
def test_normal_reference_cells_preserve_tiny_location_and_scale(
    location: float, scale: float, digits: int
) -> None:
    """Parse the actual emitted Normal reference rather than just its helper."""
    table = Report(
        "reference",
        tests=(
            TestRow(
                "test", 1.0, pvalue=0.5, law="normal", law_parameters=(location, scale)
            ),
        ),
    ).to_latex(digits=digits)
    emitted = re.findall(r"\\num\{([^}]+)\}", table)
    assert len(emitted) == len((location, scale))
    for text, original in zip(emitted, (location, scale), strict=True):
        expected = Decimal(original)
        assert abs(Decimal(text) - expected) <= abs(expected) * Decimal(
            str(0.51 * 10 ** (1 - digits))
        ) + Decimal("5e-324")
