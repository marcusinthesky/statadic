"""Displayed teaching outputs belong to the executable analysis being taught."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.reporting import ParameterRow, Report
from statadic_docs_labs import formal_extension

pytestmark = pytest.mark.native


def application():
    dependencies = []
    return SimpleNamespace(
        env=SimpleNamespace(note_dependency=dependencies.append),
        _statadic_formal={"claims": []},
    ), dependencies


def test_quick_start_displays_its_actual_data_and_saved_report():
    app, dependencies = application()
    text = ["<!-- statadic:regression-result -->"]
    formal_extension.source(app, "tutorials/first-regression", text)
    x = jnp.linspace(0.0, 5.0, 40)
    y = 1 + 0.8 * x + 0.4 * jax.random.normal(jax.random.key(17), x.shape)
    expected = np.linalg.lstsq(
        np.column_stack((np.ones(40), np.asarray(x))), np.asarray(y), rcond=None
    )[0]
    assert all(f"{coefficient:.4f}" in text[0] for coefficient in expected)
    assert "first_regression.py" in dependencies[0]
    assert "change_covariance.py" not in "\n".join(dependencies)
    assert "HC0" not in text[0]


def test_unavailable_parameter_evidence_stays_visible():
    report = Report(
        "Unavailable",
        parameters=(
            ParameterRow(
                "x",
                None,
                valid=False,
                status="unavailable",
                unavailable_reason="rank-deficient",
            ),
        ),
        warnings=("Inspect the retained design.",),
    )
    rendered = formal_extension._parameter_table((("HC3", report),))
    assert "rank-deficient" in rendered
    assert "unavailable" in rendered
    assert "Inspect the retained design." in rendered


def test_chapter_displays_one_retained_analysis_across_sections(monkeypatch):
    app, dependencies = application()
    original = formal_extension.runpy.run_path
    calls = []

    def load(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(formal_extension.runpy, "run_path", load)
    pages = {}
    for section in ("fit", "diagnose", "infer", "report"):
        text = [f"<!-- statadic:ols-result {section} -->"]
        formal_extension.source(app, f"tutorials/ols/{section}", text)
        pages[section] = text[0]
        assert "<!-- statadic:ols-result" not in text[0]
    assert len(calls) == 1
    assert all("ols_analysis.py" in path for path in dependencies)
    assert app._statadic_ols_analysis["fit_report"].to_text() in pages["fit"]
    assert app._statadic_ols_analysis["diagnostics_report"].to_text() in pages["diagnose"]
    assert app._statadic_ols_analysis["reports"]["hc3"].to_text() in pages["report"]
    assert "model_based" in pages["infer"] and "hc3" in pages["infer"]
