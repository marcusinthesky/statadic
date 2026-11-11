"""Local evidence links must resolve to exact rendered fragments."""

import importlib.util
from io import StringIO
from pathlib import Path


def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/check_browser.py"
    spec = importlib.util.spec_from_file_location("docs_browser_links", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.local_link_failures


def test_same_page_and_cross_page_fragments_are_checked(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="#missing">Missing</a><a href="claim.html#assumptions">Good</a>'
        '<a href="claim.html#absent">Bad</a><a href="absent.html">Absent</a>'
    )
    (tmp_path / "claim.html").write_text('<h2 id="assumptions">Assumptions</h2>')
    assert checker()(tmp_path) == [
        ("index.html", "#missing"),
        ("index.html", "claim.html#absent"),
        ("index.html", "absent.html"),
    ]


def test_decoded_legacy_and_directory_targets_preserve_downloads(tmp_path):
    (tmp_path / "section").mkdir()
    (tmp_path / "section/index.html").write_text('<a name="old anchor"></a>')
    (tmp_path / "proof.pdf").write_bytes(b"download")
    (tmp_path / "index.html").write_text(
        '<h2 id="here">Here</h2><a href="#here">Same</a>'
        '<a href="/section/#old%20anchor">Legacy</a>'
        '<a href="proof.pdf#page=2">PDF</a>'
        '<a href="https://example.org/#unavailable">External</a>'
        '<a href="#:~:text=Here">Text fragment</a>'
    )
    assert checker()(tmp_path) == []


def test_links_cannot_escape_the_built_site(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (tmp_path / "outside.html").write_text("outside")
    (site / "index.html").write_text('<a href="../outside.html">Outside</a>')
    assert checker()(site) == [("index.html", "../outside.html")]


def test_generated_research_disclosures_have_rendered_reference_targets(tmp_path):
    """Exercise MyST's label attachment and the actual HTML fragment checker."""
    from statadic_docs_labs.formal_extension import source
    from sphinx.application import Sphinx

    methods = (
        ("geometry.ground.metrics", "research-method-geometry-ground-metrics"),
        (
            "discrepancy.mmd.energy.scalar",
            "research-method-discrepancy-mmd-energy-scalar",
        ),
        ("api.data", "research-method-api-data"),
        ("api.diagnostics", "research-method-api-diagnostics"),
        ("mmd-u", "research-method-mmd-u"),
        ("mmd-v", "research-method-mmd-v"),
    )
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "validation").mkdir()
    (sources / "conf.py").write_text(
        'extensions = ["myst_parser"]\n'
        'source_suffix = {".md": "markdown"}\n'
        'root_doc = "index"\nnitpicky = True\n'
    )
    (sources / "index.md").write_text(
        "# Research links\n\n```{toctree}\nvalidation/research\n```\n\n"
        + "\n".join(
            f"- {{ref}}`Method {i} <research-method-{identifier}>`"
            for i, (identifier, _) in enumerate(methods)
        )
        + "\n"
    )
    (sources / "validation/research.md").write_text(
        "# Research catalogue\n\n<!-- statadic:research-catalogue -->\n"
    )
    output = tmp_path / "html"
    warnings = StringIO()
    app = Sphinx(
        srcdir=sources,
        confdir=sources,
        outdir=output,
        doctreedir=tmp_path / "doctrees",
        buildername="html",
        status=StringIO(),
        warning=warnings,
        warningiserror=True,
        freshenv=True,
    )
    app._statadic_research = [
        {
            "id": identifier,
            "title": f"Method {i}",
            "disposition": "ownership-review",
            "citation_review": "pending",
            "rationale": "Source correspondence remains explicit.",
            "studies": [],
        }
        for i, (identifier, _) in enumerate(methods)
    ]
    app._statadic_procedures = []
    app._statadic_formal = {"claims": []}
    app.connect("source-read", source)
    app.build(force_all=True)
    assert not warnings.getvalue(), warnings.getvalue()
    rendered = (output / "validation/research.html").read_text()
    links = (output / "index.html").read_text()
    for _, target in methods:
        assert f'<details id="{target}">' in rendered
        assert f"validation/research.html#{target}" in links
    assert checker()(output) == []
