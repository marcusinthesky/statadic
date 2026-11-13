"""Exercise the teaching workflow and compile its retained LaTeX report."""

import re
import runpy
import zlib
from pathlib import Path

import numpy as np
import pytest
from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.inference import HC
from statadic.api.reporting import (
    Metric,
    ParameterRow,
    PlotSpec,
    ReferenceLine,
    Report,
    TestRow,
)

APP = Path(__file__).resolve().parents[1]
NOTEBOOK = APP.parents[1] / "packages/statadic/docs/tutorials/ols/report.py"


@pytest.fixture(scope="module")
def chapter():
    return runpy.run_path(str(NOTEBOOK))


@pytest.mark.native
def test_ols_chapter_matches_independent_fit_and_hc3(chapter):
    data = chapter["make_data"]()
    fit = chapter["fit_data"](data)
    inference = fit.infer(covariance=HC("HC3"))
    x, y = (
        np.asarray(data[name], dtype=np.asarray(fit.parameters).dtype)
        for name in ("x", "y")
    )
    design = np.column_stack((np.ones_like(x), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    inverse = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    meat = design.T @ (design * (residual / (1 - leverage))[:, None] ** 2)
    hc3 = inverse @ meat @ inverse
    np.testing.assert_allclose(fit.parameters, beta, atol=2e-5, rtol=2e-5)
    rank = np.linalg.matrix_rank(design)
    assert len(y) == 80
    assert int(fit.core.rank) == rank
    assert float(fit.df_resid) == len(y) - rank
    assert float(fit.core.sse) == pytest.approx(float(residual @ residual), rel=2e-5)
    metrics = {metric.name: metric.value for metric in fit.report().metrics}
    r_squared = 1 - float(residual @ residual) / float(np.sum((y - y.mean()) ** 2))
    assert metrics["r_squared"] == pytest.approx(r_squared, rel=2e-5)
    np.testing.assert_allclose(
        inference.uncertainty.covariance.values, hc3, atol=2e-5, rtol=2e-5
    )
    report = inference.report(reference="normal")
    for row, expected, se in zip(
        report.parameters, beta, np.sqrt(np.diag(hc3)), strict=True
    ):
        np.testing.assert_allclose(
            [row.estimate, row.std_error], [expected, se], atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            [row.ci_lower, row.ci_upper],
            [expected - 1.95996398454 * se, expected + 1.95996398454 * se],
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.native
def test_ols_predictor_and_report_round_trip(chapter, tmp_path):
    fit = chapter["fit_data"](chapter["make_data"]())
    report = fit.infer(covariance=HC("HC3")).report(reference="normal")
    fit.save_predictor(tmp_path / "predictor")
    save_analysis(tmp_path / "analysis", report)
    np.testing.assert_allclose(
        load_predictor(tmp_path / "predictor").predict({"x": [0.0, 1.0]}),
        fit.predict({"x": [0.0, 1.0]}),
    )
    assert load_analysis(tmp_path / "analysis").report.to_latex() == report.to_latex()


def test_latex_compiles_with_escaped_labels_and_unavailability(chapter, tmp_path):
    fit = chapter["fit_data"](chapter["make_data"]())
    latex = fit.infer(covariance=HC("HC3")).report(reference="normal").to_latex()
    unavailable = Report(
        "Diagnostic",
        parameters=(
            ParameterRow(
                "slope_1 & cost%",
                None,
                valid=False,
                status="unavailable",
                unavailable_reason="rank deficient",
            ),
        ),
        warnings=("Check sample & model",),
    ).to_latex()
    assert r"slope\_1 \& cost\%" in unavailable
    assert "rank deficient" in unavailable and r"Check sample \& model" in unavailable
    compile_report = runpy.run_path(str(APP / "scripts/compile_report.py"))[
        "compile_report"
    ]
    compile_report(
        latex + "\n" + unavailable,
        tmp_path,
        only_cached=True,
    )
    assert (tmp_path / "report.pdf").is_file()


def test_report_table_compiles_every_rendering_case(tmp_path):
    """Counts, dashes, spans, floors, scientific cells and Greek labels compile."""
    report = Report(
        "Coverage \u2014 summary",
        parameters=(
            ParameterRow(
                "intercept",
                -0.49421,
                std_error=0.060761,
                pvalue=0.0,
                ci_lower=-0.6133,
                ci_upper=-0.37512,
            ),
            ParameterRow(
                "chi_stat",
                1.2e-9,
                std_error=3.4e-11,
                pvalue=0.0123,
                ci_lower=1.0e-9,
                ci_upper=1.5e-9,
            ),
            ParameterRow(
                "unobserved", None, valid=False, unavailable_reason="rank deficient"
            ),
            ParameterRow(
                "\u03c7\u00b2 load",
                2.0,
                std_error=0.5,
                pvalue=0.5,
                ci_lower=1.0,
                ci_upper=3.0,
            ),
        ),
        metrics=(
            Metric("n_observations", 1_234_567),
            Metric("r_squared", 0.25),
            Metric("aic", None, available=False, unavailable_reason="undefined"),
        ),
        metadata={"confidence": 0.95},
        notes=("n \u2265 5; 95% intervals",),
    )
    labels = {"chi_stat": "Statistic", "n_observations": "Observations"}
    plain = report.to_latex(digits=4, labels=labels)
    tex = report.to_latex(
        decimals=3, labels={"chi_stat": r"$\chi^2$ term"}, label_markup="tex"
    )
    metrics_only = Report(
        "Metrics", metrics=(Metric("n", 2_000_000), Metric("r2", 0.5))
    ).to_latex()
    assert "1234567" in plain and r"\multicolumn{2}{c}{95\% CI}" in plain
    assert "$<0.001$" in plain and "{---}" in plain
    assert r"$\chi$" in plain and "Observations" in plain
    compile_report = runpy.run_path(str(APP / "scripts/compile_report.py"))[
        "compile_report"
    ]
    compile_report(
        "\n".join((plain, r"\bigskip", tex, r"\bigskip", metrics_only)),
        tmp_path,
        only_cached=True,
    )
    assert (tmp_path / "report.pdf").is_file()
    missing = "\\par \u2203 no such glyph"
    with pytest.raises(RuntimeError):
        compile_report(missing, tmp_path / "bad", only_cached=True)


def test_fit_statistic_labels_compile(tmp_path):
    """A footer holding every default fit-statistic label compiles offline."""
    names = (
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
    report = Report(
        "Fit",
        parameters=(ParameterRow("x", 0.5, std_error=0.1, pvalue=0.01),),
        metrics=tuple(
            Metric(name, 1234.5 if name != "n_observations" else 400) for name in names
        ),
    )
    latex = report.to_latex(digits=4)
    assert "Adjusted $R^2$" in latex and "Uncentred $R^2$" in latex
    compile_report = runpy.run_path(str(APP / "scripts/compile_report.py"))[
        "compile_report"
    ]
    compile_report(latex, tmp_path, only_cached=True)
    assert (tmp_path / "report.pdf").is_file()


def test_selected_footer_compiles(tmp_path):
    """A chosen, reordered footer subset compiles offline."""
    report = Report(
        "Fit",
        parameters=(ParameterRow("x", 0.5, std_error=0.1, pvalue=0.01),),
        metrics=(
            Metric("n_observations", 400),
            Metric("r_squared", 0.25),
            Metric("log_likelihood", -123.4),
            Metric("aic", 250.8),
        ),
    )
    latex = report.to_latex(digits=4, metrics=("aic", "n_observations"))
    assert "R^2" not in latex
    assert latex.index("AIC") < latex.index("Observations")
    compile_report = runpy.run_path(str(APP / "scripts/compile_report.py"))[
        "compile_report"
    ]
    compile_report(latex, tmp_path, only_cached=True)
    assert (tmp_path / "report.pdf").is_file()


def test_test_table_compiles_every_law(tmp_path):
    """Chi-square, F, resampled, descriptive and unavailable test rows compile."""
    report = Report(
        "Tests",
        tests=(
            TestRow(
                "jarque_bera",
                1.39938,
                0.496739,
                "chi-square",
                (2,),
                "chi2(2)",
                "Residuals are normally distributed.",
            ),
            TestRow("anova_f", 4.5, 1e-9, "f", (1, 78), "F(1, 78)"),
            TestRow(
                "permutation",
                0.25,
                0.0123,
                "resampled",
                (999,),
                "random resampling (999 draws)",
            ),
            TestRow(
                "location shift",
                1.5,
                0.13,
                law="normal",
                law_parameters=(-0.0001, 0.001),
                reference="N(-0.0001, 0.001^2)",
            ),
            TestRow("durbin_watson", 1.9),
            TestRow(
                "reset", None, available=False, unavailable_reason="rank deficient"
            ),
        ),
    )
    inference = Report(
        "Coefficients",
        parameters=(ParameterRow("slope", 1.5, 0.5, 3.0, 0.003),),
        metadata={"reference": "normal"},
    )
    latex = report.to_latex()
    assert r"$\chi^2(2)$" in latex and "$F(1, 78)$" in latex
    assert "$<0.001$" in latex and r"\par jarque\_bera:" in latex
    assert r"$N(\num{-0.0001}, \num{0.001}^2)$" in latex
    compile_report = runpy.run_path(str(APP / "scripts/compile_report.py"))[
        "compile_report"
    ]
    compile_report(
        "\n".join((latex, r"\bigskip", inference.to_latex())),
        tmp_path,
        only_cached=True,
    )
    assert (tmp_path / "report.pdf").is_file()


def test_values_file_compiles_alone_and_after_the_manuscript_style(tmp_path):
    """The exported value file works with the docs preamble and with the style."""
    report = Report(
        "Values",
        parameters=(
            ParameterRow("intercept", -0.49421, 0.060761, -8.13, 0.497),
            ParameterRow("x", 1.2e-9, 3.4e-11, 35.0, 1e-9),
            ParameterRow("rho", None, valid=False, unavailable_reason="singular"),
        ),
        metrics=(Metric("n_observations", 1_234_567),),
        tests=(
            TestRow("jarque_bera", 1.39938, 0.496739, "chi-square", (2,), "chi2(2)"),
        ),
    )
    body = "\n".join(
        (
            r"\input{values.tex}",
            r"Estimate \jcnum{ols-intercept-estimate}, small \jcnum{ols-x-estimate},",
            r"count \jcnum{ols-n-observations}, raw \jcvalue{ols-intercept-se},",
            r"$p \jcvalue{ols-intercept-p}$, $p \jcvalue{ols-x-p}$,",
            r"$\chi^2(\jcvalue{ols-jarque-bera-df})$.",
        )
    )
    compile_module = runpy.run_path(str(APP / "scripts/compile_report.py"))

    first = tmp_path / "docs"
    first.mkdir()
    report.save_values(first / "values.tex", prefix="ols", decimals=4)
    compile_module["compile_report"](body, first, only_cached=True)
    assert (first / "report.pdf").is_file()

    second = tmp_path / "style"
    second.mkdir()
    report.save_values(second / "values.tex", prefix="ols", decimals=4)
    style = APP.parents[2] / "latex/templates/statadic-manuscript.sty"
    (second / style.name).write_text(
        style.read_text(encoding="utf-8"), encoding="utf-8"
    )
    template = tmp_path / "manuscript.tex"
    template.write_text(
        "\\documentclass{article}\n\\usepackage{statadic-manuscript}\n"
        "\\begin{document}\n\\input{report-content.tex}\n\\end{document}\n",
        encoding="utf-8",
    )
    compile_module["_compile"](
        body, second, only_cached=True, template=template, name="report"
    )
    assert (second / "report.pdf").is_file()

    undefined = tmp_path / "undefined"
    undefined.mkdir()
    report.save_values(undefined / "values.tex", prefix="ols")
    with pytest.raises(RuntimeError, match="ols-rho-estimate"):
        compile_module["compile_report"](
            r"\input{values.tex}\jcvalue{ols-rho-estimate}",
            undefined,
            only_cached=True,
        )


def test_native_figures_compile_from_cache(tmp_path):
    compile_figure = runpy.run_path(str(APP / "scripts/compile_report.py"))[
        "compile_figure"
    ]
    reference = ReferenceLine((0.0, 1.0), (0.0, 0.0), "reference")
    plots = (
        PlotSpec("line", (0.0, 1.0), (1.0, 2.0), label="line"),
        PlotSpec("scatter", (0.0, 1.0), (1.0, 2.0)),
        PlotSpec(
            "coefficient_interval",
            (0.0, 1.0),
            (1.0, 2.0),
            lower=(0.5, 1.5),
            upper=(1.5, 2.5),
            interval_meaning="95% intervals",
            labels=("a", "b"),
            reference_lines=(reference,),
        ),
        PlotSpec(
            "line",
            (0.0, 1.0),
            (1.0, 2.0),
            lower=(0.5, 1.5),
            upper=(1.5, 2.5),
            interval_meaning="95% intervals",
            interval_style="ribbon",
            markup="tex",
            label=r"$\hat\beta_1$",
            overlays=(
                PlotSpec("line", (0.0, 1.0), (2.0, 3.0), markup="tex", label="overlay"),
            ),
        ),
    )
    plots += (
        PlotSpec(
            "line",
            (0.0, 1.0, 2.0),
            (1.0, 2.0, 3.0),
            lower=(0.5, 1.5, 2.5),
            upper=(1.5, 2.5, 3.5),
            interval_meaning="95% intervals",
            interval_style="ribbon",
            label="base",
            overlays=(
                PlotSpec(
                    "line",
                    (0.0, 1.0, 2.0),
                    (2.0, 3.0, 4.0),
                    lower=(1.5, 2.5, 3.5),
                    upper=(2.5, 3.5, 4.5),
                    interval_meaning="95% intervals",
                    interval_style="ribbon",
                    label="first",
                ),
                PlotSpec(
                    "line",
                    (0.0, 1.0, 2.0),
                    (3.0, 4.0, 5.0),
                    lower=(2.5, 3.5, 4.5),
                    upper=(3.5, 4.5, 5.5),
                    interval_meaning="95% intervals",
                    interval_style="ribbon",
                    label="second",
                ),
            ),
        ),
        PlotSpec(
            "scatter",
            (0.0, 1.0),
            (1.0, 2.0),
            overlays=(PlotSpec("line", (0.0, 1.0), (2.0, 3.0), label="fit"),),
            reference_lines=(ReferenceLine((0.0, 1.0), (1.5, 1.5), "target"),),
        ),
    )
    for i, plot in enumerate(plots):
        compile_figure(
            plot.to_pgfplots(width_cm=10), tmp_path / str(i), only_cached=True
        )
        assert (tmp_path / str(i) / "figure.pdf").is_file()
    # The compiled page must paint the blue closed band, not merely compile its path.
    pdf = (tmp_path / "3/figure.pdf").read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.DOTALL)
    page = zlib.decompress(streams[0])
    coordinate = rb"[-.\d]+\s+[-.\d]+"
    assert re.search(
        rb"0 0 1 rg\s+/\S+ gs\s+"
        + coordinate
        + rb"\s+m\s+"
        + (coordinate + rb"\s+l\s+") * 3
        + rb"h\s+f\b",
        page,
    )
