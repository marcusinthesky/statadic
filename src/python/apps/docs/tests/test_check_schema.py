"""Page-type schemas, discovery and the shrink-only migration allowlist."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "check_schema", SCRIPTS / "check_schema.py"
)
check_schema = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_schema)

needs_schematter = pytest.mark.skipif(
    shutil.which("schematter") is None, reason="schematter is not on PATH"
)

CODE = "```python {.marimo}\nx = 1\n```\n"
FRONT = "---\ntitle: Example\nmarimo-version: 0.25.2\n---\n\n"


def step(n: int, code: bool = True) -> str:
    return f"## Step {n}: Do thing {n}\n\nWhy this matters.\n\n" + (
        CODE if code else ""
    )


def notebook(steps: int = 2, *, how_to: bool = False, drop: str = "") -> str:
    parts = {
        "Introduction": "## Introduction\n\nThe question.\n\n",
        "Before you start": "## Before you start\n\nKnow things.\n\n",
        "The data": "## The data\n\nSeeded rows.\n\n",
        "steps": "".join(step(n + 1) for n in range(steps)),
        "Explore": "## Explore\n\nMove a slider.\n\n",
        "What you learned": "## What you learned\n\n- One idea.\n\n",
        "Assumptions and evidence": "## Assumptions and evidence\n\nSee reference.\n\n",
        "Next steps": "## Next steps\n\n- Link.\n\n",
    }
    parts.pop(drop, None)
    if how_to:
        for optional in ("Before you start", "The data", "Explore"):
            parts.pop(optional, None)
    return FRONT + CODE + "\n# Title\n\nStandfirst.\n\n" + "".join(parts.values())


def page(kind: str, body: str, extra: str = "") -> str:
    return f"---\ntype: {kind}\n{extra}---\n\n{body}"


def passes(kind: str, text: str, tmp_path: Path) -> bool:
    path = tmp_path / "page.md"
    path.write_text(text)
    return check_schema.validate_group(kind, [path])


@needs_schematter
def test_tutorial_schema(tmp_path):
    assert passes("tutorial", notebook(2), tmp_path)
    assert not passes("tutorial", notebook(1), tmp_path)  # fewer than two steps
    assert not passes("tutorial", notebook(9), tmp_path)  # more than eight steps
    assert not passes("tutorial", notebook(2, drop="Introduction"), tmp_path)
    head, tail = notebook(2).rsplit("## Step 2", 1)
    codeless = head + "## Step 2" + tail.replace(CODE, "", 1)
    assert not passes("tutorial", codeless, tmp_path)


@needs_schematter
def test_tutorial_rejects_order_extra_sections_and_long_steps(tmp_path):
    text = notebook(2)
    intro = "## Introduction\n\nThe question.\n\n"
    before = "## Before you start\n\nKnow things.\n\n"
    swapped = text.replace(intro + before, before + intro)
    assert swapped != text
    assert not passes("tutorial", swapped, tmp_path)
    extra = notebook(2) + "## Appendix\n\n" + CODE + "\n## Surprise\n\ntext\n"
    assert not passes("tutorial", extra, tmp_path)
    long = notebook(2).replace("Why this matters.", "word " * 600, 1)
    assert not passes("tutorial", long, tmp_path)


@needs_schematter
def test_how_to_schema(tmp_path):
    assert passes("how-to", notebook(1, how_to=True), tmp_path)
    assert passes(
        "how-to", notebook(1, how_to=True) + "## Appendix\n\n" + CODE, tmp_path
    )
    assert not passes("how-to", notebook(1, how_to=True, drop="Next steps"), tmp_path)
    assert not passes("how-to", notebook(0, how_to=True), tmp_path)


REFERENCE = (
    "# regression\n\n## Usage\n\n" + CODE + "\n## Description\n\nWhat.\n\n"
    "## Options\n\nOpts.\n\n## Results\n\nOut.\n\n"
    "## Assumptions and limits\n\nLimits.\n\n## Evidence\n\nTests.\n\n"
    "## See also\n\n- Link.\n"
)


@needs_schematter
def test_reference_schema(tmp_path):
    assert passes("reference", page("reference", REFERENCE), tmp_path)
    optional = REFERENCE.replace(
        "## Assumptions", "## Methods and formulas\n\nMath.\n\n## Assumptions"
    )
    assert passes("reference", page("reference", optional), tmp_path)
    assert not passes(
        "reference",
        page("reference", REFERENCE.replace("## Evidence", "## Proof")),
        tmp_path,
    )
    assert not passes(
        "reference", page("reference", REFERENCE.replace(CODE, "")), tmp_path
    )
    assert not passes("reference", page("explanation", REFERENCE), tmp_path)


@needs_schematter
def test_other_types(tmp_path):
    assert passes(
        "landing",
        page("landing", "# Home\n\nIntro.\n\n```{toctree}\na\n```\n"),
        tmp_path,
    )
    assert not passes(
        "landing",
        page("landing", "# Home\n\n## Section\n\n```{toctree}\na\n```\n"),
        tmp_path,
    )
    assert passes(
        "explanation",
        page("explanation", "# Idea\n\nIntro.\n\n## One\n\ntext\n"),
        tmp_path,
    )
    assert not passes(
        "explanation",
        page("explanation", "# Idea\n\nIntro.\n\n## Step 1: Do\n\ntext\n"),
        tmp_path,
    )
    assert passes(
        "blog",
        page(
            "blog",
            "# Post\n\nAn Author · 12 November 2026\n\ntext\n",
            'description: "A post."\n',
        ),
        tmp_path,
    )
    assert not passes("blog", page("blog", "# Post\n\ntext\n"), tmp_path)
    assert passes(
        "evidence",
        page(
            "evidence",
            "# Facts\n\nIntro.\n\n<!-- statadic:reference-index -->\n\n"
            "## How to read this\n\ntext\n",
        ),
        tmp_path,
    )
    assert not passes("evidence", page("evidence", "# Facts\n\nIntro.\n"), tmp_path)
    assert passes(
        "glossary",
        page("glossary", "# Terms\n\n```{glossary}\nterm\n    Meaning.\n```\n"),
        tmp_path,
    )
    fixes = "# Help\n\n## Symptom\n\n**Cause** x.\n\n**Fix** y.\n"
    assert passes("troubleshooting", page("troubleshooting", fixes), tmp_path)
    assert not passes(
        "troubleshooting",
        page("troubleshooting", "# Help\n\n## Symptom\n\nx.\n"),
        tmp_path,
    )


def test_notebook_type_comes_from_the_directory(tmp_path):
    docs = tmp_path / "docs"
    for name in ("start", "tutorials/ols", "guides", "other"):
        (docs / name).mkdir(parents=True)
        (docs / name / "n.py").write_text("")
    kinds = {
        p.relative_to(docs).parts[0]: check_schema.notebook_type(p, docs)
        for p in docs.rglob("n.py")
    }
    assert kinds == {
        "start": "tutorial",
        "tutorials": "tutorial",
        "guides": "how-to",
        "other": None,
    }
    assert [p.name for p in check_schema.discover(docs)] == ["n.py", "n.py", "n.py"]


def test_discovery_excludes_templates_claims_and_rst(tmp_path):
    docs = tmp_path / "docs"
    for rel in (
        "_templates/n.py",
        "formal/claims/A.md",
        "reference/api.rst",
        "index.md",
    ):
        (docs / rel).parent.mkdir(parents=True, exist_ok=True)
        (docs / rel).write_text("")
    assert [p.name for p in check_schema.discover(docs)] == ["index.md"]


def test_allowlist_can_only_shrink(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "typed.md").write_text("---\ntype: landing\n---\n# T\n")
    (docs / "untyped.md").write_text("# T\n")
    stale = check_schema.stale_allowlist(
        ["docs/gone.md", "docs/typed.md", "docs/untyped.md"], tmp_path, docs
    )
    assert len(stale) == 2
    assert "no longer exists" in stale[0]
    assert "now declares a type" in stale[1]
    assert check_schema.stale_allowlist(["b.md", "a.md"], tmp_path, docs)


@needs_schematter
def test_missing_type_fails_unless_allowlisted(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "untyped.md").write_text("# T\n")
    assert check_schema.check(docs, tmp_path, [], scaffold=False) == 1
    assert check_schema.check(docs, tmp_path, ["docs/untyped.md"], scaffold=False) == 0
    # A stale entry is itself a failure.
    assert (
        check_schema.check(
            docs, tmp_path, ["docs/untyped.md", "docs/x.md"], scaffold=False
        )
        == 1
    )


@needs_schematter
def test_scaffold_validates_as_a_tutorial():
    assert check_schema.check(check_schema.DOCS, check_schema.ROOT, allowlist=None) == 0


def test_migration_allowlist_is_retired():
    assert not check_schema.ALLOWLIST.exists()
    assert check_schema.load_allowlist() == []


def test_semantic_directives_and_metadata_are_required():
    check = check_schema.complementary_issues
    assert check("landing", "# Home\n\nIntro\n\n```python\npass\n```\n")
    assert not check("landing", "# Home\n\nIntro\n\n```{toctree}\na\n```\n")
    assert check("glossary", "# Terms\n\n```python\npass\n```\n")
    assert check("glossary", "# Terms\n\n```{glossary}\n```\n")
    assert not check(
        "glossary", "# Terms\n\n```{glossary}\nterm\n    Definition.\n```\n"
    )
    assert check(
        "troubleshooting",
        "# Help\n\n## Crash\n\nFirst paragraph.\n\nSecond paragraph.\n",
    )
    assert not check(
        "troubleshooting", "# Help\n\n## Crash\n\n**Cause** One.\n\n**Fix** Two.\n"
    )
    assert check("blog", "# Post\n\nNo author or date.\n")
    assert not check("blog", "# Post\n\nAn Author · 12 November 2026\n")
    assert check("tutorial", "# Title\n\n## Introduction\n")
    assert check("evidence", "# Facts\n\nIntro.\n\n## How to read this\n\nWords.\n")
    assert not check(
        "evidence",
        "# Facts\n\nIntro.\n\n<!-- statadic:reference-index -->\n\n"
        "## How to read this\n\nWords.\n",
    )
    assert not check(
        "evidence",
        "# Facts\n\nIntro.\n\n:::{bibliography}\n:::\n\n"
        "## How to read this\n\nWords.\n",
    )
    assert check(
        "evidence",
        "# Facts\n\nIntro.\n\n<!-- statadic:arbitrary -->\n\n"
        "## How to read this\n\nWords.\n",
    )


def test_notebook_source_requires_preview_and_anchored_callout(tmp_path):
    path = tmp_path / "lesson.py"
    path.write_text("""
import marimo as mo
@app.cell
def _():
    mo.md("## The data")
    return
@app.cell
def _():
    mo.md("## Assumptions and evidence")
    return
@app.cell
def _():
    mo.callout(mo.md("[Limits](../reference/ref.md#assumptions-and-limits) "
                     "and [Evidence](../reference/ref.md#evidence)"))
    return
""")
    issues = check_schema.notebook_contract(path, "tutorial")
    assert any("table preview" in item for item in issues)
    assert any("raw-data figure" in item for item in issues)
    assert not any("callout" in item or "link to" in item for item in issues)
    path.write_text(
        path.read_text()
        .replace("mo.callout(mo.md(", "mo.md(")
        .replace('evidence)"))', 'evidence)")')
    )
    assert any(
        "one callout" in item for item in check_schema.notebook_contract(path, "how-to")
    )


@needs_schematter
def test_evidence_budget_counts_all_authored_sections(tmp_path):
    text = page(
        "evidence",
        "# Facts\n\n"
        + "word " * 1200
        + "\n\n<!-- statadic:reference-index -->\n\n## How to read this\n\nShort.\n",
    )
    assert not passes("evidence", text, tmp_path)


@needs_schematter
def test_only_known_generated_evidence_regions_are_excluded(tmp_path):
    prefix = "# Facts\n\nIntro.\n\n"
    suffix = "\n\n## How to read this\n\nRead it.\n"
    huge = "word " * 1200
    known = (
        prefix
        + "<!-- statadic:reference-index:start -->\n"
        + huge
        + "\n<!-- statadic:reference-index:end -->"
        + suffix
    )
    assert passes("evidence", page("evidence", known), tmp_path)
    unknown = known.replace("reference-index:start", "arbitrary:start").replace(
        "reference-index:end", "arbitrary:end"
    )
    assert not passes("evidence", page("evidence", unknown), tmp_path)
    assert not passes(
        "evidence",
        page(
            "evidence", known.replace("reference-index:end", "research-catalogue:end")
        ),
        tmp_path,
    )


def test_notebook_preview_checks_ignore_comments_and_unshown_values(tmp_path):
    notebook = tmp_path / "lesson.py"
    notebook.write_text("""
@app.cell
def _():
    mo.md("## The data")
    return
@app.cell
def _():
    # mo.ui.table(data), plt.subplots()
    table = mo.ui.table(data)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    return
@app.cell
def _():
    mo.md("## Assumptions and evidence")
    return
""")
    issues = check_schema.notebook_contract(notebook, "tutorial")
    assert any("table preview" in issue for issue in issues)
    assert any("raw-data figure" in issue for issue in issues)
    notebook.write_text(
        notebook.read_text().replace(
            "    ax.scatter(x, y)\n    return",
            "    ax.scatter(x, y)\n    mo.vstack([table, fig])\n    return",
        )
    )
    issues = check_schema.notebook_contract(notebook, "tutorial")
    assert not any(
        "table preview" in issue or "raw-data figure" in issue for issue in issues
    )


@needs_schematter
def test_hidden_appendix_does_not_consume_published_notebook_budget(tmp_path):
    body = notebook(2) + "## Appendix\n\n```python\n" + "hidden = 1\n" * 2000 + "```\n"
    projected = check_schema.published_notebook_export(body)
    assert "hidden = 1" not in projected
    assert passes("tutorial", projected, tmp_path)
    visible = notebook(2).replace("The question.", "word " * 5000)
    assert not passes(
        "tutorial", check_schema.published_notebook_export(visible), tmp_path
    )
    after = body + "\n## Unexpected section\n\nwords\n"
    assert not passes(
        "tutorial", check_schema.published_notebook_export(after), tmp_path
    )


def test_notebook_standfirst_is_one_sentence():
    assert not check_schema.complementary_issues(
        "tutorial", "# Title\n\nOne clear sentence.\n\n## Introduction\n"
    )
    assert check_schema.complementary_issues(
        "tutorial", "# Title\n\nFirst sentence. Second sentence.\n\n## Introduction\n"
    )


def test_generated_marker_alone_is_not_an_evidence_introduction():
    text = (
        "# Facts\n\n<!-- statadic:reference-index -->\n\n## How to read this\n\nRead.\n"
    )
    assert any(
        "introduction" in issue
        for issue in check_schema.complementary_issues("evidence", text)
    )


def test_official_marimo_markdown_round_trip_preserves_headings_and_dependencies(
    tmp_path,
):
    import re
    import subprocess
    import tomllib

    source = tmp_path / "original.py"
    source.write_text("""# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.2", "matplotlib>=3.8"]
# ///
import marimo
app = marimo.App()
@app.cell
def _():
    import marimo as mo
    mo.md("# Example\\n\\nA first sentence.")
    return (mo,)
@app.cell
def _(mo):
    mo.md("## Introduction\\n\\nA second sentence.")
    return
if __name__ == "__main__":
    app.run()
""")
    markdown, restored, again = (
        tmp_path / name for name in ("first.md", "restored.py", "again.md")
    )
    assert check_schema.export_notebook(source, markdown)
    subprocess.run(
        [sys.executable, "-m", "marimo", "convert", str(markdown), "-o", str(restored)],
        check=True,
    )
    assert check_schema.export_notebook(restored, again)

    def headings(value):
        return re.findall(r"(?m)^#{1,2} .+$", value)

    assert headings(markdown.read_text()) == headings(again.read_text())

    def script_metadata(value):
        match = re.search(r"(?ms)^# /// script\n(.*?)^# ///$", value)
        assert match
        return tomllib.loads(
            "\n".join(line.removeprefix("# ") for line in match[1].splitlines())
        )

    assert script_metadata(source.read_text()) == script_metadata(restored.read_text())
