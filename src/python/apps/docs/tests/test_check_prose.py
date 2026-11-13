"""Prose gates, advisory metrics and notebook string extraction."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("check_prose", SCRIPTS / "check_prose.py")
prose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prose)


def violations(text: str, page_type: str | None = None) -> list[dict]:
    paragraphs = prose.markdown_paragraphs(prose.markdown_lines(text))
    return prose.gate(paragraphs, page_type)


def rules(text: str, page_type: str | None = None) -> set[str]:
    return {v["rule"] for v in violations(text, page_type)}


def test_clean_prose_passes():
    assert not violations("Fit the line first. Then request an interval.\n")


def test_semicolons_are_flagged_in_prose_but_not_in_code():
    assert rules("The fit is stable; the interval is wide.\n") == {"semicolon"}
    assert not violations("Use `a; b` here.\n\n```python\nx = 1; y = 2\n```\n")


def test_sentence_length_limit():
    ok = " ".join(["word"] * 40) + "."
    long = " ".join(["word"] * 41) + "."
    assert not violations(ok)
    assert rules(long) == {"long-sentence"}


def test_banned_vocabulary_and_ids():
    for text in (
        "This was a campaign result here.",
        "The old lane is gone now.",
        "It is no longer supported here.",
        "See Paper 3 for details.",
        "The t12.3 item is open now.",
        "Wave W4 covers this part.",
        "The T2 task is done now.",
        "The estimator was retained from before.",
    ):
        assert "vocabulary" in rules(text), text


def test_evidence_pages_allow_process_words():
    text = "The campaign gave registration and qualification records.\n"
    assert "vocabulary" in rules(text)
    assert "vocabulary" not in rules(text, "evidence")
    assert "vocabulary" in rules("The lane was used for this check.\n", "evidence")


def test_negation_limit_per_paragraph():
    text = "It does not drift. It never stalls. It runs well.\n"
    assert rules(text) == {"negation"}
    assert not violations("It does not drift. It runs well.\n")
    assert not violations("It does not drift.\n\nIt never stalls here.\n")
    limits = "## Assumptions and limits\n\n" + text
    assert not violations(limits)
    assert rules("## Other\n\n" + text) == {"negation"}


def test_markdown_extraction_keeps_tables_and_note_prose():
    text = (
        "---\ntype: reference\n---\n"
        "# Heading; with semicolon\n\n"
        "<!-- note; hidden -->\n\n"
        "| a; b | c |\n|---|---|\n\n"
        "```python\nprint('x; y')\n```\n\n"
        ":::{note}\nA note; stripped here.\n:::\n\n"
        "See [the guide; text](https://example.com/a;b) for $x; y$ and `code; here`.\n"
    )
    paragraphs = prose.markdown_paragraphs(prose.markdown_lines(text))
    assert [p.text for p in paragraphs] == [
        "a; b",
        "c",
        "A note; stripped here.",
        "See the guide; text for MATH and CODE.",
    ]
    assert paragraphs[-1].line == 19


def test_line_numbers_follow_the_source():
    head = "---\ntype: x\n---\n\nFirst sentence here now.\n\n"
    text = head + "A bad; sentence follows here.\n"
    assert violations(text)[0]["line"] == 7


def test_short_fragments_are_ignored():
    assert prose.sentences("Yes. No way. Three words here.") == ["Three words here."]


def test_metrics():
    text = (
        "One two three four five six. Seven eight nine of ten in the box; done today.\n"
    )
    paragraphs = prose.markdown_paragraphs(prose.markdown_lines(text))
    values = prose.metrics(paragraphs)
    assert values["sentences"] == 2
    assert values["mean_words"] == 8.0
    assert values["sd_words"] == 2.0
    assert values["short_share"] == 1.0
    assert values["long_share"] == 0.0
    assert values["prepositions_per_100_words"] == round(100 * 2 / 16, 1)
    assert values["semicolons_per_100_sentences"] == 50.0
    assert values["negations_per_sentence"] == 0.0


def test_advisory_flags_do_not_gate():
    short = "Fit it now. Plot it now. Read it now. Save it now. Done now ok.\n"
    values = prose.metrics(prose.markdown_paragraphs(prose.markdown_lines(short)))
    assert prose.advisory_flags(values, None)
    assert not violations(short)


def test_glossary_uses_its_reference_variant_sentence_band():
    values = {
        "sentences": 8,
        "mean_words": 13.0,
        "sd_words": 2.0,
        "prepositions_per_100_words": 12.0,
    }
    assert prose.advisory_flags(values, "glossary") == []
    assert prose.advisory_flags(values, "explanation")


NOTEBOOK = '''
import marimo as mo

app = mo.App()


@app.cell
def _(slope):
    mo.md(f"""
    ## Result

    The slope is {slope:.2f}; it is close to the truth.
    """)
    mo.callout(mo.md("Nested call; flagged too."), kind="info")
    mo.md("Single line with a campaign.")
    print("A string; outside mo.md is ignored")
'''


def test_notebook_extraction_covers_fstrings_and_nested_calls():
    lines = prose.dynamic_notebook_markdown(NOTEBOOK)
    joined = "\n".join(line for _, line in lines)
    assert "The slope is VALUE; it is close to the truth." in joined
    assert "Nested call; flagged too." in joined
    assert "outside mo.md" not in joined
    paragraphs = prose.markdown_paragraphs(lines)
    found = prose.gate(paragraphs, "tutorial")
    assert {v["rule"] for v in found} == {"semicolon", "vocabulary"}
    slope = next(v for v in found if "slope" in v["sentence"])
    assert slope["line"] == 12  # the line of the sentence in the source


def test_allowlisted_and_exempt_pages_are_not_gated(tmp_path):
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    page = docs / "a.md"
    page.write_text("A bad; sentence follows here.\n")
    gated = prose.analyse(page, docs, root, allow=set())
    assert gated.gated and gated.violations
    skipped = prose.analyse(page, docs, root, allow={"docs/a.md"})
    assert not skipped.gated and skipped.violations
    claim = docs / "formal/claims/C.md"
    claim.parent.mkdir(parents=True)
    claim.write_text("A bad; sentence follows here.\n")
    assert not prose.analyse(claim, docs, root, allow=set()).gated


def test_main_exit_codes_and_json(tmp_path, capsys):
    good = tmp_path / "good.md"
    bad = tmp_path / "bad.md"
    good.write_text("Fit the line first. Then request an interval.\n")
    bad.write_text("The fit is stable; the interval is wide.\n")
    assert prose.main(["--paths", str(good), "--no-allowlist"]) == 0
    capsys.readouterr()
    assert prose.main(["--paths", str(bad), "--json", "--no-allowlist"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data[0]["violations"][0]["rule"] == "semicolon"


def test_author_owned_blog_is_measured_without_rewriting_its_voice(tmp_path):
    page = tmp_path / "essay.md"
    page.write_text(
        "---\ntype: blog\n---\n\n# An essay\n\nA bad; sentence follows here.\n"
    )
    result = prose.analyse(page, docs=tmp_path, root=tmp_path)
    assert result.violations
    assert not result.gated


def test_table_and_directive_bodies_are_gated():
    assert "semicolon" in rules("| A title | A bad; statement. |\n|---|---|\n")
    assert "vocabulary" in rules(":::{note}\nA campaign is here.\n:::\n")
    assert "semicolon" in rules(
        "```{glossary}\nterm\n    This is bad; another thought.\n```\n"
    )
    assert not violations('```{code-block} python\nprint("bad; code")\n```\n')


def test_notebook_analysis_uses_export_plus_dynamic_calls(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    path = docs / "tutorials" / "lesson.py"
    path.parent.mkdir(parents=True)
    path.write_text(NOTEBOOK)

    def fake_export(source, out):
        out.write_text("# Title\n\nExported prose has a campaign.\n")
        return True

    monkeypatch.setattr(prose.check_schema, "export_notebook", fake_export)
    result = prose.analyse(path, docs, tmp_path)
    assert {v["rule"] for v in result.violations} == {"semicolon", "vocabulary"}
    assert any("Exported prose" in v["sentence"] for v in result.violations)


def test_table_fragments_gate_but_do_not_distort_sentence_metrics():
    table = "| Short; fragment | A full sentence appears here. |\n|---|---|\n"
    paragraphs = prose.markdown_paragraphs(prose.markdown_lines(table))
    assert "semicolon" in {v["rule"] for v in prose.gate(paragraphs, None)}
    assert prose.metrics(paragraphs)["sentences"] == 1
    assert "vocabulary" not in rules(
        "| real-line W1 and W2 | Freedman–Lane test |\n|---|---|\n"
    )
    assert "vocabulary" in rules("The W13 task is open.\n")


def test_navigation_is_gated_but_excluded_from_narrative_metrics():
    text = (
        "A complete explanatory sentence appears here.\n\n"
        "## Next steps\n\n- A bad; navigation link.\n"
    )
    paragraphs = prose.markdown_paragraphs(prose.markdown_lines(text))
    assert prose.metrics(paragraphs)["sentences"] == 1
    assert "semicolon" in {item["rule"] for item in prose.gate(paragraphs, None)}


def test_executed_prose_catches_runtime_markdown_and_ignores_controls_and_appendix(
    tmp_path,
):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["text = 'A campaign appears.'\n", "mo.md(text)"],
                "outputs": [{"data": {"text/markdown": ["A campaign appears."]}}],
            },
            {
                "cell_type": "code",
                "source": "control",
                "outputs": [
                    {
                        "data": {
                            "text/markdown": (
                                "&lt;marimo-slider data-label='campaign'&gt;"
                            )
                        }
                    }
                ],
            },
            {"cell_type": "markdown", "source": ["## Appendix\n"]},
            {
                "cell_type": "code",
                "outputs": [{"data": {"text/markdown": "A campaign remains."}}],
            },
        ]
    }
    assert not prose.dynamic_notebook_markdown("mo.md(text)")
    findings = prose.check_executed_prose(tmp_path / "notebook.py", notebook)
    assert len(findings) == 1
    assert findings[0]["rule"] == "vocabulary"
    assert findings[0]["cell"] == 1


def test_curated_table_records_cannot_bypass_prose_gate(tmp_path, monkeypatch):
    family = tmp_path / "family-sources.json"
    limits = tmp_path / "known-limitations.json"
    family.write_text(
        json.dumps(
            {"header": "| Family | Scope |", "rows": ["| A | A brick was run. |"]}
        )
    )
    limits.write_text(
        json.dumps(
            {"tables": [{"columns": ["Issue"], "rows": [["A ledger records this."]]}]}
        )
    )
    monkeypatch.setattr(prose, "CURATED_TABLES", (family, limits))
    reports = prose.collect([family, limits], set())
    assert all(report.gated and report.violations for report in reports)


def test_executed_callout_and_mixed_widget_prose_are_gated(tmp_path):
    callout = (
        "<marimo-callout-output data-html='&quot;"
        "&lt;p&gt;A brick appears.&lt;/p&gt;&quot;'>"
        "</marimo-callout-output>"
    )
    mixed = (
        "&lt;p&gt;A ledger appears.&lt;/p&gt;"
        "&lt;marimo-slider data-label='brick'&gt;&lt;/marimo-slider&gt;"
    )
    pure = "<marimo-slider data-label='brick'></marimo-slider>"
    cells = [
        {
            "cell_type": "code",
            "source": "mo.callout(mo.md(variable))",
            "outputs": [{"data": {mime: value}}],
        }
        for mime, value in [
            ("text/html", callout),
            ("text/markdown", mixed),
            ("text/html", pure),
        ]
    ]
    found = prose.check_executed_prose(tmp_path / "example.py", {"cells": cells})
    assert {item["cell"] for item in found} == {1, 2}
    assert {item["term"] for item in found} == {"brick", "ledger"}
