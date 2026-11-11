"""Build and audit all first-party Lean declarations without admission exceptions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOTLEAN = Path(__file__).resolve().parents[1]
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
MARKER = "SORRY_AUDIT_JSON "
COUNT = "SORRY_AUDIT_COUNT "


def lean_tokens(source: str) -> list[tuple[str, int]]:
    """Lex debug preflight tokens, ignoring nested comments, strings and raw strings.

    This is not a theorem parser: Lean itself classifies placeholders. Quoted identifiers
    remain visible, so `set_option «debug».skipKernelTC` cannot evade the preflight.
    """
    tokens: list[tuple[str, int]] = []
    i = 0
    line = 1
    while i < len(source):
        start = i
        if source.startswith("--", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/-", i):
            depth = 1
            i += 2
            while i < len(source) and depth:
                if source.startswith("/-", i):
                    depth += 1
                    i += 2
                elif source.startswith("-/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError("unterminated Lean block comment")
        elif source[i] == '"' or (source[i] == "r" and re.match(r'r#*"', source[i:])):
            raw = re.match(r'r(#+|)"', source[i:]) if source[i] == "r" else None
            if raw:
                i += raw.end()
                end = source.find('"' + raw[1], i)
                if end < 0:
                    raise ValueError("unterminated Lean raw string")
                i = end + 1 + len(raw[1])
            else:
                i += 1
                while i < len(source) and source[i] != '"':
                    i += 2 if source[i] == "\\" else 1
                if i >= len(source):
                    raise ValueError("unterminated Lean string")
                i += 1
        elif source[i] == "«":
            end = source.find("»", i + 1)
            if end < 0:
                raise ValueError("unterminated Lean quoted identifier")
            tokens.append((source[i + 1 : end], line))
            i = end + 1
        elif source[i] == "'" and (
            char := re.match(
                r"'(?:\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'", source[i:]
            )
        ):
            i += char.end()
        elif source[i].isalpha() or source[i] == "_":
            i += 1
            while i < len(source) and (source[i].isalnum() or source[i] in "_'!?"):
                i += 1
            tokens.append((source[start:i], line))
        else:
            if not source[i].isspace():
                tokens.append((source[i], line))
            i += 1
        line += source[start:i].count("\n")
    return tokens


def check_debug_options(sources: list[Path]) -> None:
    """Reject debug options before any source is built, including multiline commands."""
    for path in sources:
        tokens = lean_tokens(path.read_text())
        for i, (token, line) in enumerate(tokens):
            if token != "set_option" or i + 1 == len(tokens):
                continue
            name = tokens[i + 1][0]
            j = i + 2
            while j + 1 < len(tokens) and tokens[j][0] == ".":
                name += "." + tokens[j + 1][0]
                j += 2
            name = name.removeprefix("_root_.")
            if name == "debug" or name.startswith("debug."):
                raise ValueError(f"{path}:{line}: forbidden set_option {name}")


def discover(lean: Path) -> dict[str, Path]:
    """Discover source modules independently of any import roll-up."""
    modules: dict[str, Path] = {}
    for path in sorted(lean.rglob("*.lean")):
        rel = path.relative_to(lean)
        if any(part in {".lake"} or part.startswith(".") for part in rel.parts):
            continue
        if rel == Path("tools/Audit.lean"):
            continue
        if path.name == "lakefile.lean":
            continue
        parts = rel.with_suffix("").parts
        if parts[0] == "packages":
            if len(parts) < 3:
                raise ValueError(f"invalid package source path: {path}")
            parts = parts[2:]
        module = ".".join(parts)
        if not module or module in modules:
            raise ValueError(f"ambiguous source module: {path}")
        modules[module] = path
    if not modules:
        raise ValueError("no first-party Lean source modules found")
    return modules


def snapshot(lean: Path, modules: dict[str, Path]) -> dict[str, str]:
    paths = set(modules.values())
    paths.update(
        lean / name
        for name in (
            "lakefile.lean",
            "lean-toolchain",
            "lake-manifest.json",
            "check_setup.py",
            "tools/Audit.lean",
            "tools/integrity.py",
        )
    )
    paths.update(p for p in (lean / "packages").rglob("lakefile.lean"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"missing required audit inputs: {missing}")
    return {
        p.relative_to(lean).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def parse_records(stdout: str) -> list[dict[str, Any]]:
    records = [
        json.loads(line.split(MARKER, 1)[1])
        for line in stdout.splitlines()
        if MARKER in line
    ]
    counts = [
        int(line.split(COUNT, 1)[1]) for line in stdout.splitlines() if COUNT in line
    ]
    if len(counts) != 1 or counts[0] != len(records) or not records:
        raise ValueError("missing or incomplete compiled declaration census")
    if len({r["declaration"] for r in records}) != len(records):
        raise ValueError("duplicate compiled declaration")
    return records


def validate(records: list[dict[str, Any]], modules: dict[str, Path]) -> None:
    failures = []
    for record in records:
        if record["module"] not in modules:
            raise ValueError(f"unexpected census module: {record['module']}")
        bad = set(record["axioms"]) - ALLOWED_AXIOMS
        if record["axiom"] or bad or record["flags"] or record["premises"]:
            failures.append(
                f"{record['declaration']}: axioms={sorted(bad)} axiom={record['axiom']} flags={record['flags']} true premises={record['premises']}"
            )
    if failures:
        raise ValueError("Lean integrity violations:\n" + "\n".join(failures))


def collect(lean: Path, modules: dict[str, Path]) -> list[dict[str, Any]]:
    env = {**os.environ, "LAKE_NO_CACHE": "true"}
    result = subprocess.run(
        ["lake", "--no-cache", "--rehash", "--wfail", "build", *sorted(modules)],
        cwd=lean,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "fresh all-module Lean build failed:\n" + result.stdout + result.stderr
        )
    with tempfile.TemporaryDirectory(prefix="statadic-lean-audit-") as directory:
        snippet = Path(directory) / "Census.lean"
        imports = "".join(f"import all {module}\n" for module in sorted(modules))
        names = ", ".join("`" + module for module in sorted(modules))
        snippet.write_text(
            "module\n"
            + imports
            + (lean / "tools/Audit.lean").read_text()
            + f"\nset_option maxHeartbeats 0 in\nrun_cmd liftTermElabM do SorryAudit.run #[{names}]\n"
        )
        result = subprocess.run(
            ["lake", "env", "lean", str(snippet)],
            cwd=lean,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise ValueError("compiled census failed:\n" + result.stdout + result.stderr)
    return parse_records(result.stdout)


def certify(lean: Path = ROOTLEAN) -> dict[str, Any]:
    """Certify a stable source snapshot, or raise ValueError on any failed check."""
    lean = lean.resolve()
    modules = discover(lean)
    hashes = snapshot(lean, modules)
    check_debug_options([lean / p for p in hashes if p.endswith(".lean")])
    records = collect(lean, modules)
    if discover(lean) != modules or snapshot(lean, modules) != hashes:
        raise ValueError(
            "source files changed during the compiled audit; rerun on a stable tree"
        )
    validate(records, modules)
    return {
        "modules": sorted(modules),
        "declarations": records,
        "source_hashes": hashes,
    }
