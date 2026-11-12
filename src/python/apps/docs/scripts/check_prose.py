"""Gate and measure the prose of documentation pages and notebooks.

Hard gates (exit nonzero, non-allowlisted pages only): no semicolons, no sentence
over 40 words, no internal vocabulary, and at most one negation sentence per
paragraph outside ``Assumptions and limits``. Advisory metrics are always printed.
"""

import argparse
import ast
import html
import json
import re
import statistics
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_schema  # noqa: E402

ROOT = check_schema.ROOT
DOCS = check_schema.DOCS
EXEMPT = {"README.md"}

SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z“"\[(])')
FENCE = re.compile(r"^\s*(`{3,}|~{3,}|:{3,})(.*)$")
NEGATION = re.compile(r"\b(not|no|never|neither|nor|cannot|without)\b|n't", re.I)
PREPOSITIONS = frozenset(
    "of in to for with on at from by about as into through over between under "
    "without within across beyond after before during toward towards against "
    "among upon via per than like rather".split()
)
ID_PATTERNS = [r"\bt\d+(?:\.\d+)*\b", r"\bPaper \d", r"\bW\d+\b", r"\bT\d+\b"]
WORD_TERMS = [
    "qualification",
    "qualified",
    "qualify",
    "campaign",
    "witness",
    "lane",
    "ledger",
    "disposition",
    "registration",
    "retained",
    "brick",
    "door",
    "waist",
    "slice",
    "delivery",
    "publication hold",
    "former",
    "no longer",
    "has been removed",
]
EVIDENCE_ALLOWED = {
    "qualification",
    "qualified",
    "campaign",
    "registration",
    "provenance",
    "disposition",
}
BANNED = [(p, re.compile(p)) for p in ID_PATTERNS] + [
    (t, re.compile(rf"(?<!Freedman–)(?<!Freedman-)\b{re.escape(t)}s?\b", re.I))
    for t in WORD_TERMS
]
LIMITS = "assumptions and limits"
MAX_WORDS = 40
# Advisory bands from the style guide.
NARRATIVE_MEAN = (15, 22)
NARRATIVE_SD_MIN = 6
REFERENCE_MEAN = (12, 18)
PREPOSITIONS_MIN = 9


@dataclass
class Paragraph:
    line: int
    heading: str
    text: str
    measure: bool = True


@dataclass
class Report:
    path: str
    type: str | None
    gated: bool
    violations: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def clean_inline(text: str) -> str:
    """Reduce inline Markdown to words, keeping link text only."""
    text = re.sub(r"(\{[\w:.-]+\})?(`+)(?:.+?)\2", "CODE", text)
    text = re.sub(r"\$\$.+?\$\$|\$[^$\n]+\$", "MATH", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"<(?:https?|mailto):[^>]*>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<!--.*?-->", "", text)
    text = re.sub(r"(?<!\w)[*_]{1,3}|[*_]{1,3}(?!\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def markdown_paragraphs(lines: list[tuple[int, str]]) -> list[Paragraph]:
    """Split Markdown lines into prose paragraphs tagged with their heading."""
    paragraphs: list[Paragraph] = []
    heading = ""
    block: list[str] = []
    start = 0
    fence: tuple[str, int] | None = None
    comment = False
    display_math = False
    prose_fence: tuple[str, int] | None = None

    def flush() -> None:
        nonlocal block
        if block:
            text = clean_inline(" ".join(block))
            if text:
                paragraphs.append(Paragraph(start, heading, text))
        block = []

    for number, raw in lines:
        line = raw.rstrip()
        if fence:
            match = FENCE.match(line)
            if match and match[1][0] == fence[0] and len(match[1]) >= fence[1]:
                if not match[2].strip():
                    fence = None
            continue
        if comment:
            comment = "-->" not in line
            continue
        if display_math:
            display_math = line.count("$$") % 2 == 0
            continue
        match = FENCE.match(line)
        if (
            prose_fence
            and match
            and match[1][0] == prose_fence[0]
            and len(match[1]) >= prose_fence[1]
            and not match[2].strip()
        ):
            flush()
            prose_fence = None
            continue
        if match:
            flush()
            directive = match[2].strip()
            # Admonition and glossary bodies contain authored prose; executable
            # code, raw markup and ordinary fenced blocks do not.
            if directive.startswith(
                (
                    "{note}",
                    "{warning}",
                    "{tip}",
                    "{important}",
                    "{admonition}",
                    "{glossary}",
                )
            ) or (match[1][0] == ":" and not directive):
                if directive:
                    prose_fence = (match[1][0], len(match[1]))
                continue
            fence = (match[1][0], len(match[1]))
            continue
        stripped = line.strip()
        if stripped.startswith("<!--"):
            flush()
            comment = "-->" not in stripped
            continue
        if stripped.startswith("$$"):
            flush()
            display_math = stripped.count("$$") % 2 == 1
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("#").strip()
            continue
        if stripped.startswith("|"):
            flush()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r"[:\-\s]+", cell) for cell in cells):
                for cell in cells:
                    text = clean_inline(cell)
                    if text:
                        paragraphs.append(
                            Paragraph(
                                number,
                                heading,
                                text,
                                bool(re.search(r"[.!?][\"”']?$", text)),
                            )
                        )
            continue
        if re.match(r"^\[[^\]]+\]:\s", stripped):
            flush()
            continue
        item = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", stripped)
        if item:
            flush()
            stripped = item[1]
        stripped = re.sub(r"^>\s*", "", stripped)
        if not block:
            start = number
        block.append(stripped)
    flush()
    return paragraphs


def markdown_lines(text: str) -> list[tuple[int, str]]:
    """Number the lines of a page after removing its frontmatter."""
    lines = text.split("\n")
    first = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                first = index + 1
                break
    return [(n + 1, lines[n]) for n in range(first, len(lines))]


def _md_text(node: ast.expr) -> str | None:
    """Return the literal text of a string or f-string, with a token for values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("VALUE")
        return "".join(parts)
    return None


def dynamic_notebook_markdown(source: str) -> list[tuple[int, str]]:
    """Extract dynamic or nested mo.md calls omitted by marimo export."""
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "md"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "mo"
            and node.args
        ):
            text = _md_text(node.args[0])
            # Marimo exports a standalone literal mo.md cell as Markdown. Other
            # calls, including f-strings and callouts, remain inside code fences.
            expr = parents.get(node)
            owner = parents.get(expr) if expr is not None else None
            pure = (
                isinstance(expr, ast.Expr)
                and isinstance(owner, ast.FunctionDef)
                and len(owner.body) == 2
                and owner.body[0] is expr
                and isinstance(owner.body[1], ast.Return)
                and isinstance(node.args[0], ast.Constant)
            )
            if text is not None and not pure:
                calls.append((node.args[0].lineno, text))
    lines: list[tuple[int, str]] = []
    for lineno, text in sorted(calls):
        first, _, rest = text.partition("\n")
        body = [first, *textwrap.dedent(rest).split("\n")]
        lines.extend((lineno + i, line) for i, line in enumerate(body))
        lines.append((lineno, ""))
    return lines


def sentences(text: str) -> list[str]:
    parts = (s.strip() for s in SPLIT.split(text))
    return [s for s in parts if len(s.split()) >= 3]


def words(sentence: str) -> list[str]:
    return [w for w in sentence.split() if re.search(r"\w", w)]


def gate(paragraphs: list[Paragraph], page_type: str | None) -> list[dict]:
    """Return the hard-gate violations of the extracted paragraphs."""
    allowed = EVIDENCE_ALLOWED if page_type == "evidence" else set()
    found = []

    def add(rule: str, paragraph: Paragraph, detail: str, sentence: str) -> None:
        found.append(
            {
                "rule": rule,
                "line": paragraph.line,
                "term": detail,
                "sentence": sentence,
            }
        )

    for paragraph in paragraphs:
        negations = 0
        for sentence in (s.strip() for s in SPLIT.split(paragraph.text) if s.strip()):
            if ";" in sentence:
                add("semicolon", paragraph, ";", sentence)
            if len(words(sentence)) > MAX_WORDS:
                add(
                    "long-sentence",
                    paragraph,
                    f"{len(words(sentence))} words",
                    sentence,
                )
            for _term, pattern in BANNED:
                match = pattern.search(sentence)
                scientific_wasserstein = (
                    match
                    and match[0] in {"W1", "W2"}
                    and re.search(
                        r"wasserstein|real.line|transport", paragraph.text, re.I
                    )
                )
                if (
                    match
                    and not scientific_wasserstein
                    and match[0].lower() not in allowed
                ):
                    add("vocabulary", paragraph, match[0], sentence)
            negations += bool(NEGATION.search(sentence))
        if negations > 1 and paragraph.heading.lower() != LIMITS:
            add(
                "negation", paragraph, f"{negations} negation sentences", paragraph.text
            )
    return found


class ReadableMarimoHTML(HTMLParser):
    """Extract visible prose, excluding custom control attributes and bodies."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.control_depth = 0

    def handle_starttag(self, tag, attrs):
        if self.control_depth:
            self.control_depth += 1
        elif tag == "marimo-callout-output":
            value = dict(attrs).get("data-html", "")
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
            if isinstance(value, str):
                nested = ReadableMarimoHTML()
                nested.feed(value)
                self.parts.extend(nested.parts)
        elif tag.startswith("marimo-"):
            self.control_depth = 1
        elif tag in {"p", "div", "li", "br", "span"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.control_depth:
            self.control_depth -= 1
        elif tag in {"p", "div", "li", "br"}:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.control_depth:
            self.parts.append(data)


def check_executed_prose(path: Path, notebook: dict) -> list[dict]:
    """Gate rendered Markdown from executed code cells without rerunning them.

    The public Jupyter export carries Markdown as a string or list of lines.
    Static Markdown cells are already checked through the official Markdown
    export. Custom marimo controls are encoded HTML, not prose. Like the site,
    stop at the Appendix heading and ignore its hidden outputs.
    """
    page_type = (
        check_schema.notebook_type(path, DOCS)
        if path.resolve().is_relative_to(DOCS.resolve())
        else None
    )
    findings = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") == "markdown":
            first = source.strip().splitlines()
            if first and re.fullmatch(r"##\s+Appendix\s*", first[0]):
                break
            continue
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if not isinstance(data, dict):
                continue
            mime = "text/markdown" if "text/markdown" in data else "text/html"
            text = data.get(mime, "")
            text = "".join(text) if isinstance(text, list) else text
            if not isinstance(text, str):
                continue
            decoded = html.unescape(text)
            if "<marimo-" in decoded:
                parser = ReadableMarimoHTML()
                parser.feed(decoded)
                text = "".join(parser.parts)
            elif mime == "text/html":
                # Ordinary HTML tables and library reports are not authored prose.
                continue
            for problem in gate(markdown_paragraphs(markdown_lines(text)), page_type):
                findings.append(problem | {"cell": index})
    return findings


def metrics(paragraphs: list[Paragraph]) -> dict:
    """Compute the advisory style metrics of a page."""
    sents = [
        s
        for p in paragraphs
        if p.measure and p.heading.lower() not in {"next steps", "see also"}
        for s in sentences(p.text)
    ]
    lengths = [len(words(s)) for s in sents]
    total = sum(lengths)
    count = len(sents)
    if not count:
        return {"sentences": 0}
    prepositions = sum(
        w.lower().strip(".,:()\"'“”") in PREPOSITIONS for s in sents for w in words(s)
    )
    return {
        "sentences": count,
        "mean_words": round(statistics.fmean(lengths), 1),
        "sd_words": round(statistics.pstdev(lengths), 1),
        "short_share": round(sum(n <= 12 for n in lengths) / count, 2),
        "long_share": round(sum(n >= 30 for n in lengths) / count, 2),
        "prepositions_per_100_words": round(100 * prepositions / max(total, 1), 1),
        "semicolons_per_100_sentences": round(
            100 * sum(";" in s for s in sents) / count, 1
        ),
        "negations_per_sentence": round(
            sum(bool(NEGATION.search(s)) for s in sents) / count, 2
        ),
    }


def advisory_flags(values: dict, page_type: str | None) -> list[str]:
    """Name the advisory bands a page falls outside."""
    if values["sentences"] < 5:
        return []
    flags = []
    reference = page_type in {"reference", "glossary"}
    low, high = REFERENCE_MEAN if reference else NARRATIVE_MEAN
    if not low <= values["mean_words"] <= high:
        flags.append(f"mean words {values['mean_words']} outside {low}-{high}")
    if not reference and values["sd_words"] < NARRATIVE_SD_MIN:
        flags.append(f"sd {values['sd_words']} below {NARRATIVE_SD_MIN}")
    if values["prepositions_per_100_words"] < PREPOSITIONS_MIN:
        found = values["prepositions_per_100_words"]
        flags.append(f"prepositions {found}/100 below {PREPOSITIONS_MIN}")
    return flags


def analyse(
    path: Path, docs: Path = DOCS, root: Path = ROOT, allow: set[str] = frozenset()
) -> Report:
    rel = check_schema.relative(path, root)
    text = path.read_text()
    if path.suffix == ".py":
        page_type = check_schema.notebook_type(path, docs)
        with tempfile.TemporaryDirectory() as scratch:
            exported = Path(scratch) / "notebook.md"
            if not check_schema.export_notebook(path, exported):
                raise ValueError(f"Notebook Markdown export failed: {path}")
            lines = markdown_lines(exported.read_text())
        lines.extend(dynamic_notebook_markdown(text))
    else:
        page_type = check_schema.frontmatter(text).get("type")
        lines = markdown_lines(text)
    paragraphs = markdown_paragraphs(lines)
    in_docs = path.resolve().is_relative_to(docs.resolve())
    claim = in_docs and path.relative_to(docs).as_posix().startswith("formal/claims/")
    # Author-owned essays supply the voice reference rather than being rewritten
    # to satisfy the rules for generated documentation.
    gated = rel not in allow and rel not in EXEMPT and not claim and page_type != "blog"
    values = metrics(paragraphs)
    return Report(
        rel,
        page_type,
        gated,
        gate(paragraphs, page_type),
        values,
        advisory_flags(values, page_type) if values["sentences"] else [],
    )


def table(reports: list[Report]) -> str:
    head = (
        f"{'page':58} {'type':11} {'n':>4} {'mean':>5} {'sd':>5} {'<=12':>5} "
        f"{'>=30':>5} {'prep':>5} {';/100':>5} {'neg':>5} {'viol':>4}"
    )
    rows = [head, "-" * len(head)]
    for r in reports:
        m = r.metrics
        if m["sentences"] == 0:
            rows.append(f"{r.path[-58:]:58} {r.type or '-':11} {0:>4}")
            continue
        mark = "" if r.gated else "*"
        rows.append(
            f"{r.path[-58:]:58} {r.type or '-':11} {m['sentences']:>4} "
            f"{m['mean_words']:>5} {m['sd_words']:>5} {m['short_share']:>5} "
            f"{m['long_share']:>5} {m['prepositions_per_100_words']:>5} "
            f"{m['semicolons_per_100_sentences']:>5} {m['negations_per_sentence']:>5} "
            f"{len(r.violations):>3}{mark}"
        )
    rows.append("* violations present but not gated (allowlisted or exempt)")
    return "\n".join(rows)


CURATED_TABLES = (
    ROOT / "src/python/apps/docs/data/family-sources.json",
    ROOT / "src/python/apps/docs/evidence/known-limitations.json",
)


def analyse_curated_table(path: Path) -> Report:
    """Gate authored table records with the same rules as Markdown tables."""
    data = json.loads(path.read_text())
    if "tables" in data:
        lines = []
        for item in data["tables"]:
            lines.append("| " + " | ".join(item["columns"]) + " |")
            lines.extend("| " + " | ".join(row) + " |" for row in item["rows"])
    else:
        lines = [data["header"], *data["rows"]]
    paragraphs = markdown_paragraphs(list(enumerate(lines, 1)))
    values = metrics(paragraphs)
    return Report(
        check_schema.relative(path, ROOT),
        "evidence",
        True,
        gate(paragraphs, "evidence"),
        values,
        [],  # Records supply lookup tables rather than a narrative page.
    )


def collect(paths: list[Path] | None, allow: set[str]) -> list[Report]:
    if paths is None:
        targets = check_schema.discover(DOCS)
        targets = [p for p in targets if check_schema.relative(p, ROOT) not in allow]
        template = DOCS / "_templates/notebook.py"
        targets += [template] if template.exists() else []
    else:
        targets = paths
    if paths is None:
        targets += [p for p in CURATED_TABLES if p.exists()]
    curated = {p.resolve() for p in CURATED_TABLES}
    return [
        analyse_curated_table(p) if p.resolve() in curated else analyse(p, allow=allow)
        for p in targets
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path, help="check only these files")
    parser.add_argument("--json", action="store_true", help="print JSON, not text")
    parser.add_argument("--no-allowlist", action="store_true")
    args = parser.parse_args(argv)
    allow = set() if args.no_allowlist else set(check_schema.load_allowlist())
    paths = [p.resolve() for p in args.paths] if args.paths else None
    reports = collect(paths, allow)
    failures = [v for r in reports if r.gated for v in r.violations]
    if args.json:
        print(json.dumps([r.__dict__ for r in reports], indent=2))
    else:
        print(table(reports))
        for r in reports:
            for flag in r.flags:
                print(f"advisory {r.path}: {flag}")
            if r.gated:
                for v in r.violations:
                    print(
                        f"{r.path}:{v['line']}: {v['rule']} [{v['term']}] "
                        f"{v['sentence'][:140]}"
                    )
        print(f"{len(reports)} pages, {len(failures)} gate violations")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
