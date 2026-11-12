"""Validate documentation pages against their Schematter page-type schemas.

Markdown pages declare ``type:`` in YAML frontmatter. Notebooks under ``start/``,
``tutorials/`` and ``guides/`` take their type from the directory and are checked
through the output of ``marimo export md``. Pages listed in the migration
allowlist are skipped, and the allowlist can only shrink.
"""

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[3]
DOCS = ROOT / "src/python/packages/statadic/docs"
SCHEMAS = ROOT / "infra/schemas/documents"
ALLOWLIST = APP / "docs-migration-allowlist.json"

TYPES = {
    "landing",
    "tutorial",
    "how-to",
    "reference",
    "explanation",
    "evidence",
    "blog",
    "glossary",
    "troubleshooting",
}
NOTEBOOK_TYPES = {
    "_templates": "tutorial",
    "start": "tutorial",
    "tutorials": "tutorial",
    "guides": "how-to",
}
EXCLUDED_PARTS = ("_templates", "formal/claims")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(root.resolve()):
        return resolved.relative_to(root.resolve()).as_posix()
    return resolved.as_posix()


def frontmatter(text: str) -> dict:
    """Return the YAML frontmatter of a Markdown page, or an empty mapping."""
    match = FRONTMATTER.match(text)
    if not match:
        return {}
    data = yaml.safe_load(match[1])
    return data if isinstance(data, dict) else {}


def notebook_type(path: Path, docs: Path) -> str | None:
    """Return the page type implied by a notebook's directory."""
    parts = path.resolve().relative_to(docs.resolve()).parts
    if path.suffix == ".py" and len(parts) > 1:
        return NOTEBOOK_TYPES.get(parts[0])
    return None


def page_type(path: Path, docs: Path) -> str | None:
    """Return the declared (Markdown) or implied (notebook) page type."""
    if path.suffix == ".py":
        return notebook_type(path, docs)
    return frontmatter(path.read_text()).get("type")


def discover(docs: Path) -> list[Path]:
    """List the authored Markdown pages and typed notebooks under ``docs``."""
    pages = []
    for path in sorted(docs.rglob("*")):
        rel = path.relative_to(docs).as_posix()
        if any(rel == part or rel.startswith(part + "/") for part in EXCLUDED_PARTS):
            continue
        if path.suffix == ".md" or (
            path.suffix == ".py" and notebook_type(path, docs) is not None
        ):
            pages.append(path)
    return pages


def load_allowlist(path: Path = ALLOWLIST) -> list[str]:
    return json.loads(path.read_text()) if path.exists() else []


def stale_allowlist(allowlist: list[str], root: Path, docs: Path) -> list[str]:
    """Return one message per allowlist entry that no longer belongs there."""
    problems = []
    for entry in allowlist:
        path = root / entry
        if not path.exists():
            problems.append(f"{entry}: file no longer exists; remove the entry")
        elif page_type(path, docs) is not None:
            problems.append(f"{entry}: now declares a type; remove the entry")
    if allowlist != sorted(set(allowlist)):
        problems.append("allowlist must be a sorted list without duplicates")
    return problems


def schema_for(type_name: str) -> Path:
    return SCHEMAS / f"docs-{type_name}.document-schema.yaml"


def export_notebook(path: Path, out: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "md", str(path), "-o", str(out)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(f"{path}: marimo export failed\n{result.stderr.strip()}")
    return result.returncode == 0


def published_notebook_export(text: str) -> str:
    """Keep the Appendix heading, but omit its non-published body and code.

    Keep subsequent level-two headings so the schema can reject a section after
    the Appendix. Only headings outside fenced code are structural.
    """
    lines = text.splitlines(keepends=True)
    fence = None
    appendix = False
    selected = []
    for line in lines:
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not match.group(2).strip()
            ):
                fence = None
            if not appendix:
                selected.append(line)
            continue
        if fence is None and re.match(r"^[ \t]*##\s+Appendix\s*$", line):
            appendix = True
            selected.append(line)
        elif not appendix or (fence is None and re.match(r"^##\s+\S", line)):
            selected.append(line)
    return "".join(selected)


PROJECTIONS = (
    "formal-registry",
    "reference-index",
    "research-catalogue",
    "performance-catalogue",
    "family-sources",
    "known-limitations",
)


def complementary_issues(type_name: str, text: str) -> list[str]:
    """Check semantics that Schematter cannot express as block identity."""
    body = FRONTMATTER.sub("", text)
    issues = []
    headings = list(re.finditer(r"(?m)^## (.+?)\s*$", body))
    h1 = re.search(r"(?m)^# [^\n]+\n", body)
    intro = body[h1.end() :].split("\n## ", 1)[0] if h1 else ""
    if type_name == "landing":
        toctree = re.search(r"(?m)^\s*(?:`{3,}|:{3,})\{toctree\}", body)
        if not toctree:
            issues.append("landing requires a toctree directive")
        before = body[h1.end() : toctree.start()] if h1 and toctree else ""
        if not re.sub(r"<!--.*?-->|\([^\n]+\)=", "", before, flags=re.S).strip():
            issues.append("landing requires an introduction before its toctree")
    if type_name == "glossary":
        glossary = re.search(
            r"(?ms)^(?P<f>`{3,}|:{3,})\{glossary\}[^\n]*\n(?P<body>.*?)^(?P=f)\s*$",
            body,
        )
        if not glossary or not re.search(
            r"(?m)^\S[^\n]*\n[ \t]+\S", glossary.group("body")
        ):
            issues.append("glossary requires defined glossary entries")
    if type_name == "troubleshooting":
        for index, heading in enumerate(headings):
            end = (
                headings[index + 1].start() if index + 1 < len(headings) else len(body)
            )
            section = body[heading.end() : end]
            if not re.search(r"(?im)^\*\*Cause\*\*\s+\S", section) or not re.search(
                r"(?im)^\*\*Fix\*\*\s+\S", section
            ):
                issues.append(f"{heading.group(1)} requires Cause and Fix paragraphs")
    if type_name == "blog":
        if not re.search(
            r"(?im)^\s*\S[^\n]*[·|—]\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\s*$",
            intro,
        ):
            issues.append("blog requires an author and date line")
    if type_name in {"tutorial", "how-to"}:
        standfirst = intro.split("```", 1)[0].strip()
        if not standfirst:
            issues.append("notebook requires a standfirst")
        elif len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", standfirst)) != 1:
            issues.append("notebook standfirst must be one sentence")
    if type_name == "evidence":
        authored_intro = re.sub(r"<!--.*?-->", "", intro, flags=re.S)
        authored_intro = re.split(
            r"(?m)^\s*(?:`{3,}|:{3,})\{", authored_intro, maxsplit=1
        )[0]
        if not authored_intro.strip():
            issues.append("evidence requires an introduction")
        marker = re.search(
            r"<!--\s*statadic:(?:"
            + "|".join(PROJECTIONS)
            + r"|math\s+[A-Z0-9-]+)\s*-->",
            intro,
        )
        bibliography = re.search(r"(?m)^\s*(?:`{3,}|:{3,})\{bibliography\}", intro)
        if marker is None and bibliography is None:
            issues.append(
                "evidence requires a recognized generated projection "
                "before How to read this"
            )
    return issues


def notebook_contract(path: Path, type_name: str) -> list[str]:
    """Check source-only notebook obligations absent from Markdown export."""
    source = path.read_text()
    tree = ast.parse(source)
    headings = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "md"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                first = arg.value.strip().splitlines()
                if first and first[0].startswith("## "):
                    headings.append((node.lineno, first[0][3:]))
    headings.sort()
    lines = source.splitlines()
    issues = []
    if type_name == "tutorial":
        data_start = next((line for line, name in headings if name == "The data"), 0)
        data_end = next(
            (line for line, _ in headings if line > data_start), len(lines) + 1
        )
        cells = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and data_start < node.lineno < data_end
            and any(ast.unparse(d).startswith("app.cell") for d in node.decorator_list)
        ]
        table_visible = False
        figure_visible = False
        for cell in cells:
            expressions = [
                node for node in cell.body if not isinstance(node, ast.Return)
            ]
            output = (
                expressions[-1].value
                if expressions and isinstance(expressions[-1], ast.Expr)
                else None
            )
            if output is None:
                continue
            output_names = {
                node.id for node in ast.walk(output) if isinstance(node, ast.Name)
            }

            def displayed(call, output=output, cell=cell, output_names=output_names):
                if any(node is call for node in ast.walk(output)):
                    return True
                for statement in cell.body:
                    if (
                        isinstance(statement, (ast.Assign, ast.AnnAssign))
                        and statement.value is not None
                        and any(node is call for node in ast.walk(statement.value))
                    ):
                        targets = (
                            statement.targets
                            if isinstance(statement, ast.Assign)
                            else [statement.target]
                        )
                        if any(
                            isinstance(node, ast.Name) and node.id in output_names
                            for target in targets
                            for node in ast.walk(target)
                        ):
                            return True
                return False

            for call in (node for node in ast.walk(cell) if isinstance(node, ast.Call)):
                name = ast.unparse(call.func)
                if name in {
                    "mo.ui.table",
                    "pd.DataFrame",
                    "pl.DataFrame",
                } and displayed(call):
                    table_visible = True
                if (
                    name == "mo.md"
                    and displayed(call)
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                    and re.search(r"\|[^\n]+\|\s*\n\s*\|[ :|\-]+\|", call.args[0].value)
                ):
                    table_visible = True
                if (
                    name in {"plt.subplots", "plt.figure"}
                    and displayed(call)
                    and (
                        isinstance(output, ast.Name)
                        or (
                            isinstance(output, ast.Call)
                            and ast.unparse(output.func) in {"mo.vstack", "mo.hstack"}
                        )
                    )
                ):
                    drawing = any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr
                        in {
                            "scatter",
                            "plot",
                            "hist",
                            "bar",
                            "barh",
                            "boxplot",
                            "imshow",
                            "errorbar",
                        }
                        for node in ast.walk(cell)
                    )
                    figure_visible |= drawing
        if not table_visible:
            issues.append("The data requires a visible table preview")
        if not figure_visible:
            issues.append("The data requires a visible raw-data figure")
    start = next(
        (line for line, name in headings if name == "Assumptions and evidence"), 0
    )
    end = next((line for line, _ in headings if line > start), len(lines) + 1)
    callouts = []
    for cell in ast.walk(tree):
        if not isinstance(cell, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (start < cell.lineno < end) or not any(
            ast.unparse(d).startswith("app.cell") for d in cell.decorator_list
        ):
            continue
        expressions = [node for node in cell.body if not isinstance(node, ast.Return)]
        output = (
            expressions[-1].value
            if expressions and isinstance(expressions[-1], ast.Expr)
            else None
        )
        if output is not None:
            callouts.extend(
                node
                for node in ast.walk(output)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "callout"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mo"
            )
    if len(callouts) != 1:
        issues.append("Assumptions and evidence requires one callout")
    content = (
        "".join(
            node.value
            for node in ast.walk(callouts[0])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if len(callouts) == 1
        else ""
    )
    for anchor in ("#assumptions-and-limits", "#evidence"):
        link = r"\[[^\]]+\]\((?:\.\./)+reference/[^)#]+\.md" + re.escape(anchor) + r"\)"
        if not re.search(link, content):
            issues.append(
                f"Assumptions and evidence requires a reference link to {anchor}"
            )
    return issues


def strip_generated_evidence(text: str) -> tuple[str, list[str]]:
    """Remove only recognized paired projections, retaining their identity marker."""
    names = "|".join(re.escape(name) for name in PROJECTIONS)
    paired = re.compile(
        r"<!--\s*statadic:(?P<name>" + names + r"):start\s*-->.*?"
        r"<!--\s*statadic:(?P=name):end\s*-->",
        re.S,
    )
    clean = paired.sub(lambda match: f"<!-- statadic:{match.group('name')} -->", text)
    unmatched = re.search(
        r"<!--\s*statadic:(?:" + names + r"):(?:start|end)\s*-->", clean
    )
    return clean, (["unmatched generated projection marker"] if unmatched else [])


def validate_group(type_name: str, files: list[Path]) -> bool:
    """Run Schematter once for every file of one page type."""
    errors = []
    with tempfile.TemporaryDirectory() as scratch:
        selected = []
        for index, file in enumerate(files):
            text = file.read_text()
            if type_name == "evidence":
                text, markers = strip_generated_evidence(text)
                errors.extend((file, issue) for issue in markers)
                projected = Path(scratch) / f"{index}-{file.name}"
                projected.write_text(text)
                selected.append(projected)
            else:
                selected.append(file)
            errors.extend(
                (file, issue) for issue in complementary_issues(type_name, text)
            )
        for file, issue in errors:
            print(f"{file}: {issue}")
        result = subprocess.run(
            [
                "schematter",
                "validate",
                "--schema",
                str(schema_for(type_name)),
                *selected,
            ],
            capture_output=True,
            text=True,
        )
    output = (result.stdout + result.stderr).strip()
    if result.returncode:
        print(f"schema docs-{type_name}: FAILED\n{output}")
    return result.returncode == 0 and not errors


def check(
    docs: Path = DOCS,
    root: Path = ROOT,
    allowlist: list[str] | None = None,
    scaffold: bool = True,
    paths: list[Path] | None = None,
) -> int:
    """Validate pages, returning the number of failures."""
    if shutil.which("schematter") is None:
        print("schematter is not on PATH; run inside `devenv shell`")
        return 1
    allow = load_allowlist() if allowlist is None else allowlist
    failures = list(stale_allowlist(allow, root, docs)) if paths is None else []
    for message in failures:
        print(f"allowlist: {message}")
    selected = paths if paths is not None else discover(docs)
    groups: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path in selected:
            rel = relative(path, root)
            if rel in allow:
                skipped += 1
                continue
            type_name = page_type(path, docs)
            if type_name is None:
                failures.append(f"{rel}: no `type` in frontmatter")
                print(f"{rel}: missing page type (add `type:` to the frontmatter)")
            elif type_name not in TYPES:
                failures.append(f"{rel}: unknown type {type_name!r}")
                print(f"{rel}: unknown page type {type_name!r}")
            elif path.suffix == ".py":
                for issue in notebook_contract(path, type_name):
                    print(f"{rel}: {issue}")
                    failures.append(f"{rel}: {issue}")
                out = Path(tmp) / (rel.replace("/", "__") + ".md")
                if export_notebook(path, out):
                    out.write_text(published_notebook_export(out.read_text()))
                    groups[type_name].append(out)
                else:
                    failures.append(f"{rel}: export failed")
            else:
                groups[type_name].append(path)
        template = docs / "_templates/notebook.py"
        if scaffold and paths is None and template.exists():
            for issue in notebook_contract(template, "tutorial"):
                print(f"scaffold: {issue}")
                failures.append(f"scaffold: {issue}")
            out = Path(tmp) / "scaffold-notebook.md"
            if export_notebook(template, out):
                out.write_text(published_notebook_export(out.read_text()))
                groups["tutorial"].append(out)
            else:
                failures.append("scaffold: export failed")
        for type_name, files in sorted(groups.items()):
            if not validate_group(type_name, files):
                failures.append(f"schema docs-{type_name}")
    print(
        f"checked {sum(map(len, groups.values()))} files in {len(groups)} schemas, "
        f"skipped {skipped} allowlisted, {len(failures)} failures"
    )
    return len(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path, help="check only these files")
    parser.add_argument("--no-allowlist", action="store_true")
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.paths] if args.paths else None
    return 1 if check(allowlist=[] if args.no_allowlist else None, paths=paths) else 0


if __name__ == "__main__":
    sys.exit(main())
