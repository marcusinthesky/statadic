"""Marimo notebooks render as native Sphinx pages with static outputs."""

import re
import shutil
from pathlib import Path

import pytest
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError

APP = Path(__file__).resolve().parents[1]
SCAFFOLD = APP.parents[1] / "packages/statadic/docs/_templates/notebook.py"

NOTEBOOK = '''\
import marimo

app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl

    plt.rcParams["savefig.format"] = "svg"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Zebrafish notebook

    A standfirst with math $a+b$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Narwhal heading

    Display math:

    $$
    c = d
    $$
    """)
    return


@app.cell
def _():
    shown_marker = 11
    shown_marker * 2
    return


@app.cell(hide_code=True)
def _():
    hidden_marker = 13
    mo.plain_text(f"hidden output {hidden_marker}")
    return


@app.cell
def _():
    _fig, _ax = plt.subplots()
    _ax.plot([0, 1], [0, 1])
    _ax
    return


@app.cell
def _():
    frame = pd.DataFrame({"alpha": [1, 2], "beta": [0.5, 1.5]})
    frame
    return (frame,)


@app.cell
def _(frame):
    mo.ui.table(frame)
    return


@app.cell
def _():
    pl.DataFrame({"gamma": [7, 8]})
    return


@app.cell
def _():
    knob = mo.ui.slider(0, 10, value=3, label="Knob")
    knob
    return (knob,)


@app.cell
def _(knob):
    mo.md(f"downstream sees {knob.value}")
    return


@app.cell
def _():
    mo.ui.tabs({
        "First report": mo.Html("<table><tr><td>first estimate</td></tr></table>"),
        "Second report": mo.Html("<table><tr><td>second estimate</td></tr></table>"),
    })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Quokka appendix heading
    """)
    return


if __name__ == "__main__":
    app.run()
'''

FAILING = """\
import marimo

app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r\"\"\"
    # Failing notebook
    \"\"\")
    return


@app.cell
def _():
    raise ValueError("deliberate failure")
    return


if __name__ == "__main__":
    app.run()
"""

PRECISION = '''\
import marimo

app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Precision
    """)
    return (mo,)


@app.cell
def _(mo):
    import jax

    mo.plain_text(f"x64-enabled={jax.config.jax_enable_x64}")
    return


if __name__ == "__main__":
    app.run()
'''


def visible_text(page: str) -> str:
    return re.sub(r"<[^>]+>", "", page)


def build(
    tmp_path: Path, *, reference: bool = False, **notebooks: str
) -> tuple[Sphinx, Path]:
    source = tmp_path / "src"
    (source / "tutorials").mkdir(parents=True)
    for name, text in notebooks.items():
        path = source / "tutorials" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (source / "tutorials" / "helper.py").write_text("VALUE = 1\n")
    (source / "_templates").mkdir()
    shutil.copy(SCAFFOLD, source / "_templates/notebook.py")
    names = "\n   ".join(f"tutorials/{name}" for name in notebooks)
    if reference:
        (source / "reference").mkdir()
        (source / "reference/regression.md").write_text(
            "# Regression\n\n## Evidence\n\nReference evidence.\n"
        )
        names += "\n   reference/regression"
    (tmp_path / "outside.py").write_text("VALUE = 2\n")
    (source / "index.md").write_text(
        "# Home\n\n[tests](../outside.py)\n\n"
        f"```{{toctree}}\n:hidden:\n\n{names}\n```\n"
    )
    (source / "conf.py").write_text(
        f"""
extensions = ["statadic_docs_pages.marimo_pages", "sphinx.ext.mathjax"]
source_suffix = {{".md": "markdown", ".py": "marimo"}}
exclude_patterns = ["_templates"]
myst_enable_extensions = ["dollarmath"]
html_static_path = [{str(APP / "static")!r}]
html_css_files = ["marimo.css"]
statadic_notebook_branch = "trunk"
statadic_repository_root = {str(source)!r}
statadic_notebook_cache = {str(tmp_path / "cache")!r}
statadic_molab_shield = {str(APP / "static/marimo/molab-shield.svg")!r}
"""
    )
    out = tmp_path / "html"
    app = Sphinx(
        str(source),
        str(source),
        str(out),
        str(tmp_path / "doctrees"),
        "html",
        warningiserror=True,
        freshenv=True,
    )
    app.build()
    return app, out


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build(tmp_path_factory.mktemp("site"), nb=NOTEBOOK)


@pytest.mark.notebook
def test_headings_reach_toc_and_search(site):
    app, out = site
    toc = app.env.tocs["tutorials/nb"].astext()
    assert "Zebrafish notebook" in toc
    assert "Narwhal heading" in toc
    index = (out / "searchindex.js").read_text()
    assert "narwhal" in index
    assert "zebrafish" in index


@pytest.mark.notebook
def test_math_reaches_mathjax(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert "MathJax" in page
    assert re.search(r"math notranslate nohighlight[^>]*>\\\(a\+b\\\)", page)
    assert "c = d" in page


@pytest.mark.notebook
def test_matplotlib_svg_is_an_image(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert '<img alt="Notebook figure"' in page
    assert "data:image/svg+xml;base64," in page


@pytest.mark.notebook
def test_code_is_shown_unless_hidden(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert "shown_marker" in visible_text(page)
    assert "hidden output 13" in visible_text(page)
    assert "hidden_marker" not in page


@pytest.mark.notebook
def test_tables_render_as_tables(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert page.count('class="dataframe"') == 3
    assert "<marimo-" not in page
    assert "_marimo_row_id" not in visible_text(page)
    for cell in ("alpha", "beta", "1.5", "gamma", "8"):
        assert re.search(rf"<t[dh][^>]*>{re.escape(cell)}</t[dh]>", page)


@pytest.mark.notebook
def test_report_tabs_keep_every_report_in_static_pages(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    for text in ("First report", "Second report", "first estimate", "second estimate"):
        assert text in visible_text(page)
    assert "Interactive element ui-element" not in visible_text(page)


def test_public_export_controls_code_visibility_and_preserves_markdown():
    from statadic_docs_pages.marimo_pages import compose

    document = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"marimo": {"name": "setup"}},
                "source": ["setup_secret = 1"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Lesson\n", "\nA short introduction."],
            },
            {
                "cell_type": "code",
                "metadata": {"jupyter": {"source_hidden": True}},
                "source": ["hidden = 3"],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"text/markdown": ["Computed **three** with $x=3$."]},
                    }
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": "## Appendix"},
            {"cell_type": "code", "metadata": {}, "source": "test_hidden = True"},
        ]
    }
    rendered = compose(document)
    assert "Computed **three** with $x=3$." in rendered
    assert "{marimo-launch}" in rendered
    assert not any(
        word in rendered
        for word in ("setup_secret", "hidden =", "Appendix", "test_hidden")
    )


@pytest.mark.notebook
def test_slider_degrades_to_default_state(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert "Slider Knob: 3" in page
    assert "downstream sees 3" in page


@pytest.mark.notebook
def test_appendix_is_dropped(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert "Appendix" not in page
    assert "Quokka" not in page


@pytest.mark.notebook
def test_launch_chrome_links_the_source(site):
    _, out = site
    page = (out / "tutorials/nb.html").read_text()
    assert (
        "https://molab.marimo.io/github/marcusinthesky/statadic/blob/trunk/"
        "tutorials/nb.py" in page
    )
    download = re.search(r'href="([^"]*_downloads/[^"]*/nb.py)"', page)
    assert download is not None
    assert (out / "tutorials" / download.group(1)).resolve().read_text() == NOTEBOOK
    assert (
        "uvx marimo edit --sandbox https://raw.githubusercontent.com/"
        "marcusinthesky/statadic/trunk/tutorials/nb.py" in visible_text(page)
    )


@pytest.mark.notebook
def test_helpers_and_scaffold_are_not_pages(site):
    app, _ = site
    assert set(app.env.found_docs) == {"index", "tutorials/nb"}


@pytest.mark.notebook
def test_a_failing_cell_fails_the_build(tmp_path):
    with pytest.raises(ExtensionError, match="deliberate failure|failed"):
        build(tmp_path, failing=FAILING)


@pytest.mark.notebook
@pytest.mark.parametrize("x64", ["0", "1"])
def test_execution_respects_jax_precision(tmp_path, monkeypatch, x64):
    monkeypatch.setenv("JAX_ENABLE_X64", x64)
    _, out = build(tmp_path, precision=PRECISION)
    page = (out / "tutorials/precision.html").read_text()
    assert f"x64-enabled={x64 == '1'}" in page


@pytest.mark.notebook
def test_execution_is_cached_by_content(tmp_path):
    from statadic_docs_pages.marimo_pages import execute

    path = tmp_path / "precision.py"
    path.write_text(PRECISION)
    execute(path, tmp_path / "cache")
    entry = next((tmp_path / "cache").glob("*.json"))
    stamp = entry.stat().st_mtime_ns
    execute(path, tmp_path / "cache")
    assert entry.stat().st_mtime_ns == stamp
    path.write_text(PRECISION + "\n# edited\n")
    execute(path, tmp_path / "cache")
    assert [p.name for p in (tmp_path / "cache").glob("*.json")] != [entry.name]


@pytest.mark.notebook
def test_the_scaffold_renders_as_a_page(tmp_path):
    _, out = build(tmp_path, scaffold=SCAFFOLD.read_text())
    page = (out / "tutorials/scaffold.html").read_text()
    text = visible_text(page)
    for heading in ("Introduction", "Step 1: Fit the line", "Next steps"):
        assert heading in text
    assert "Regression fit" in text
    assert "Appendix" not in text
    assert "test_slope_is_close_to_the_generator" not in text
    assert "Slider Noise: 0.4" in text


def test_package_digest_tracks_contents_and_relative_paths(tmp_path):
    from statadic_docs_pages.marimo_pages import _tree_digest

    source = tmp_path / "a" / "module.py"
    source.parent.mkdir()
    source.write_text("one")
    before = _tree_digest(tmp_path)
    stamp = source.stat().st_mtime_ns
    source.write_text("two")
    import os

    os.utime(source, ns=(stamp, stamp))
    assert _tree_digest(tmp_path) != before
    before = _tree_digest(tmp_path)
    source.rename(source.parent / "other.py")
    assert _tree_digest(tmp_path) != before


def test_cache_key_tracks_lock_and_renderer(tmp_path, monkeypatch):
    from statadic_docs_pages import marimo_pages

    notebook = tmp_path / "notebook.py"
    notebook.write_text("notebook")
    monkeypatch.setattr(marimo_pages, "_package_digest", lambda: "package")
    original = Path.read_bytes
    changed = set()

    def read_bytes(path):
        data = original(path)
        return data + b"changed" if path.resolve() in changed else data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    before = marimo_pages.cache_key(notebook)
    renderer = Path(marimo_pages.__file__).resolve()
    changed.add(renderer)
    assert marimo_pages.cache_key(notebook) != before
    changed.clear()
    changed.add(renderer.parents[4] / "uv.lock")
    assert marimo_pages.cache_key(notebook) != before


@pytest.mark.notebook
def test_output_links_resolve_from_nested_pages(tmp_path):
    marker = '@app.cell(hide_code=True)\ndef _():\n    mo.md(r"""\n    ## Appendix'
    links = """@app.cell(hide_code=True)
def _():
    mo.callout(mo.md("[Evidence](../../reference/regression.md#evidence)"))
    return


@app.cell(hide_code=True)
def _():
    mo.Html('<a href="../../reference/regression.md?view=all#evidence">HTML link</a>'
            '<a href="https://example.com/page.md#x">External</a>'
            '<a href="#here">Fragment</a>'
            '<a href="../../reference/file.pdf">Download</a>')
    return


"""
    notebook = NOTEBOOK.replace(marker, links + marker)
    _, out = build(tmp_path, reference=True, **{"ols/fit": notebook})
    page = (out / "tutorials/ols/fit.html").read_text()
    assert 'href="../../reference/regression.html#evidence"' in page
    assert 'href="../../reference/regression.html?view=all#evidence"' in page
    assert 'href="https://example.com/page.md#x"' in page
    assert 'href="#here"' in page
    assert 'href="../../reference/file.pdf"' in page
    assert (out / "reference/regression.html").is_file()


@pytest.mark.parametrize("mime", ["text/html", "text/markdown"])
def test_raw_output_resolves_document_links(mime):
    from types import SimpleNamespace

    from statadic_docs_pages.marimo_pages import render_output, resolve_output_links

    env = SimpleNamespace(
        docname="tutorials/ols/fit",
        found_docs={"reference/regression"},
        relfn2path=lambda path, docname: (
            "reference/regression.md",
            "/docs/reference/regression.md",
        ),
        path2doc=lambda path: "reference/regression",
    )
    builder = SimpleNamespace(
        get_relative_uri=lambda source, target: "../../reference/regression.html"
    )
    markup = '<a href="../../reference/regression.md#evidence">Evidence</a>'
    nodes = render_output(
        mime, markup, lambda text: resolve_output_links(text, env, builder)
    )
    assert 'href="../../reference/regression.html#evidence"' in nodes[0].astext()
