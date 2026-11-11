# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["insitu-cli==0.2.0rc2", "pyyaml==6.0.3"]
# ///
"""Create project-owned plans and references; Insitu owns their derived views."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[4]
TERMINAL = ("done", "closed")
SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
CHILDREN_BLOCK = (
    '<!-- insitu:begin sql\nid = "planning-children"\nmodel = "children"\n'
    'template = "children.md.j2"\n-->\n<!-- insitu:end -->'
)
BACKLINKS_BLOCK = (
    '<!-- insitu:begin sql\nid = "reference-backlinks"\nmodel = "backlinks"\n'
    'template = "backlinks.md.j2"\n-->\n<!-- insitu:end -->'
)


class UniqueLoader(yaml.SafeLoader):
    """Reject repeated keys rather than silently changing canonical state."""


def unique_mapping(
    loader: UniqueLoader, node: Any, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"duplicate or non-string frontmatter key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


@dataclass(frozen=True)
class Record:
    path: Path
    data: dict[str, Any]
    body: str
    kind: str


def config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.load((root / ".copier-answers.yml").read_text(), Loader=UniqueLoader)


def read_document(path: Path) -> tuple[dict[str, Any], str]:
    match = re.fullmatch(r"---\n(.*?)\n---(?:\n|\Z)(.*)", path.read_text(), re.DOTALL)
    if not match:
        raise ValueError(f"{path}: expected YAML frontmatter")
    data = yaml.load(match[1], Loader=UniqueLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter must be an object")
    return data, match[2]


def path_kind(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    if len(parts) < 4 or parts[0] != ".context" or parts[-1] != "README.md":
        return None
    if parts[1] == "reference":
        return "reference" if len(parts) == 4 else None
    if parts[1] != "plan":
        return None
    if len(parts[3:-1]) % 2 or any(part != "tasks" for part in parts[3:-1:2]):
        raise ValueError(f"{path}: tasks must nest inside a parent's tasks/ directory")
    if any(not SLUG.fullmatch(part) for part in parts[2:-1:2]):
        raise ValueError(f"{path}: invalid plan/task slug")
    return "plan" if len(parts) == 4 else "task"


def read_records(root: Path = ROOT) -> list[Record]:
    records = []
    for path in sorted((root / ".context").rglob("README.md")):
        kind = path_kind(path, root)
        if kind is not None:
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ValueError(
                    f"{path}: records must use regular files and directories"
                )
            data, body = read_document(path)
            records.append(Record(path, data, body, kind))
    return records


def verified_acceptance(body: str) -> bool:
    """Require checked, substantive criteria in the real Acceptance section."""
    fence: tuple[str, int] | None = None
    in_comment = False
    in_acceptance = False
    criteria: list[bool] = []
    for line in body.splitlines():
        marker = re.match(r"^[ \t]*(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                if not marker[2].strip():
                    fence = None
            continue
        if marker and not in_comment:
            fence = marker[1][0], len(marker[1])
            continue
        # Comments inside examples remain examples; only prose comments are hidden.
        while in_comment or "<!--" in line:
            start = 0 if in_comment else line.index("<!--")
            end = line.find("-->", start if in_comment else start + 4)
            if end < 0:
                line = line[:start]
                in_comment = True
                break
            line = line[:start] + line[end + 3 :]
            in_comment = False
        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", line)
        if heading:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
            if len(heading[1]) <= 2:
                if in_acceptance:
                    break
                in_acceptance = len(heading[1]) == 2 and title == "Acceptance"
            continue
        if in_acceptance:
            criterion = re.match(
                r"^\s*(?:[-+*]|\d{1,9}[.)])[ \t]+\[([ xX])\](?:[ \t]+(.*))?$", line
            )
            if criterion:
                content = (criterion[2] or "").strip()
                if content and not re.fullmatch(r"~~(?:(?!~~).)+~~", content):
                    criteria.append(criterion[1].lower() == "x")
    return bool(criteria) and all(criteria)


def link_error(link: str, root: Path) -> str | None:
    """Keep declared edges identical to the paths queried by the SQL models."""
    guidance = (
        "use a plain repository-relative file path; put URLs and sections in body links"
    )
    try:
        parsed = urlsplit(link)
    except ValueError:
        return f"invalid link {link!r}; {guidance}"
    if parsed.scheme or parsed.netloc or not parsed.path:
        return f"invalid link {link!r}; {guidance}"
    try:
        target = (root / unquote(parsed.path).replace("\\", "/")).resolve()
        relative = target.relative_to(root.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return f"missing or escaping link {link!r}; {guidance}"
    if target.is_dir() and (target / "README.md").is_file():
        suggestion = "README.md" if relative == "." else relative + "/README.md"
        return f"link {link!r} names a directory; use {suggestion!r} instead"
    if not target.is_file():
        return f"missing or escaping link {link!r}; {guidance}"
    if link != parsed.path or link != relative:
        return f"noncanonical link {link!r}; use {relative!r} instead"
    return None


def validate_records(root: Path, records: list[Record] | None = None) -> list[str]:
    """Check relationships after schema validation establishes field shapes."""
    records = read_records(root) if records is None else records
    errors = []
    by_path = {record.path: record for record in records}
    for record in records:
        label, data = record.path.relative_to(root), record.data
        errors.extend(
            f"{label}: {error}"
            for link in data.get("links", [])
            if (error := link_error(link, root))
        )
        if "id" in data or "csl" in data or "depends_on" in data:
            errors.append(
                f"{label}: identity comes from paths; remove id, csl and depends_on"
            )
        if record.kind == "reference":
            continue
        if data.get("type") != record.kind:
            errors.append(f"{label}: type must assert path role {record.kind!r}")
        status = data.get("status")
        if (
            record.kind == "task"
            and record.path.parent.parent.parent / "README.md" not in by_path
        ):
            errors.append(f"{label}: containing plan/task is missing")
        children = [
            item
            for item in records
            if item.path.parent.parent == record.path.parent / "tasks"
        ]
        if status in TERMINAL and any(
            child.data.get("status") not in TERMINAL for child in children
        ):
            errors.append(f"{label}: terminal parent has open children")
        if status == "done" and not verified_acceptance(record.body):
            errors.append(
                f"{label}: done requires nonempty active checked acceptance criteria"
            )
        if (
            status == "closed"
            and not data.get("resolution", {}).get("reason", "").strip()
        ):
            errors.append(f"{label}: closed requires resolution.reason")
        markers = re.findall(
            r'<!-- insitu:begin sql\s+id = "planning-children".*?<!-- insitu:end -->',
            record.body,
            re.DOTALL,
        )
        if len(markers) != 1:
            errors.append(f"{label}: restore exactly one planning-children region")
    return errors


def run(root: Path, *command: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, check=True, text=True, **kwargs)


def schema_check(root: Path, records: list[Record], *, budget: bool = True) -> None:
    schemas = root / "infra" / "schemas" / "documents"
    entries = [
        root / "README.md",
        root / ".context" / "README.md",
        *sorted((root / ".context").glob("*/README.md")),
    ]
    entries.extend(
        root / name
        for name in ("ROADMAP.md", "AGENTS.md", ".agents/AGENTS.md")
        if (root / name).is_file()
    )
    for schema, paths in (
        ("budget", entries if budget else []),
        ("workitem", [item.path for item in records if item.kind != "reference"]),
        ("reference", [item.path for item in records if item.kind == "reference"]),
    ):
        if paths:
            run(
                root,
                "schematter",
                "validate",
                "--schema",
                str(schemas / f"{schema}.document-schema.yaml"),
                *map(str, paths),
            )


def serialize(data: dict[str, Any], body: str) -> str:
    metadata = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)
    return f"---\n{metadata}---\n\n{body.rstrip()}\n"


def create_record(
    kind: str, slug: str, title: str, root: Path = ROOT, parent: str | None = None
) -> Path:
    workflow = "reference" if kind == "reference" else "planning"
    if workflow not in config(root)["workflows"]:
        raise ValueError(f"{workflow} is not selected")
    if not SLUG.fullmatch(slug):
        raise ValueError(
            "slug must use lowercase words, digits and single hyphens, "
            "starting with a letter"
        )
    if not title.strip() or "\n" in title or "\r" in title or len(title.strip()) > 160:
        raise ValueError("title must occupy one line with between 1 and 160 characters")
    title = title.strip()
    directory = (
        root / ".context" / ("reference" if kind == "reference" else "plan") / slug
    )
    if kind == "task":
        container = (root / (parent or "")).resolve()
        if container.name == "README.md":
            container = container.parent
        if not container.is_relative_to(root.resolve() / ".context" / "plan"):
            raise ValueError(
                "parent must be an existing plan or task within .context/plan"
            )
        role = path_kind(container / "README.md", root)
        data, _ = read_document(container / "README.md")
        if (
            role not in {"plan", "task"}
            or data.get("type") != role
            or data.get("status") not in ("backlog", "in-progress", "blocked")
        ):
            raise ValueError("parent must be an open plan or task")
        directory = container / "tasks" / slug
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("record destination escapes the repository")
    data = {
        "type": "document" if kind == "reference" else kind,
        "title": title,
        "description": title,
        "links": [],
    }
    body = f"# {title}\n\n"
    if kind != "reference":
        data["status"] = "backlog"
        body += f"## Goal\n\n{title}\n\n## Acceptance\n\n"
        body += (
            "- [ ] Define and verify the intended outcome.\n\n"
            + CHILDREN_BLOCK
            + "\n\n"
        )
    else:
        body += "Summarize the source and why it matters here.\n\n"
    body += "## Backlinks\n\n" + BACKLINKS_BLOCK
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "README.md"
    with destination.open("x") as stream:
        stream.write(serialize(data, body))
    return destination


def bibliography(root: Path, records: list[Record], *, check: bool) -> None:
    settings = config(root)
    if "reference" not in settings["workflows"] or "latex" not in settings["languages"]:
        return
    references = [
        record.data | {"id": record.path.parent.name}
        for record in records
        if record.kind == "reference"
    ]
    for item in references:
        item.pop("description", None)
        item.pop("links", None)
    expected = ""
    if references:
        expected = run(
            root,
            "pandoc",
            "--from",
            "csljson",
            "--to",
            "bibtex",
            input=json.dumps(references, ensure_ascii=False),
            capture_output=True,
        ).stdout
    destination = root / "src" / "latex" / "refs.bib"
    if destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ValueError(
            "bibliography destination must use regular files and directories"
        )
    actual = destination.read_text() if destination.exists() else ""
    if actual != expected:
        if check:
            raise ValueError("src/latex/refs.bib is stale; run just sync")
        destination.write_text(expected)


def insitu(root: Path, command: str) -> None:
    executable = Path(sys.executable).with_name("insitu")
    if not executable.is_file():
        raise ValueError("insitu is missing; run just setup")
    run(
        root,
        str(executable),
        command,
        env={**os.environ, "INSITU_PYTHON": sys.executable},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-new", "task-new", "reference-new"):
        command = commands.add_parser(name)
        if name == "task-new":
            command.add_argument("parent")
        command.add_argument("slug")
        command.add_argument("--title", required=True)
    for name in ("prepare", "sync", "check"):
        commands.add_parser(name)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            run(
                ROOT,
                "bash",
                str(ROOT / "infra/nix/prepare-runtime.sh"),
                str(Path(sys.executable).with_name("insitu")),
            )
        elif args.command.endswith("-new"):
            kind = args.command.removesuffix("-new")
            parent = getattr(args, "parent", None)
            print(create_record(kind, args.slug, args.title, parent=parent))
        else:
            records = read_records()
            schema_check(ROOT, records, budget=args.command != "sync")
            errors = validate_records(ROOT, records)
            if errors:
                raise ValueError("\n".join(errors))
            bibliography(ROOT, records, check=args.command == "check")
            insitu(ROOT, args.command)
            if args.command == "sync":
                schema_check(ROOT, read_records())
    except (
        ValueError,
        OSError,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"workflow: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
