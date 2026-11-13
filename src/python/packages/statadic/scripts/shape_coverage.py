# ruff: noqa: INP001  # standalone package-maintenance entry point
"""Report which declared return shapes Pyrefly actually checks.

A gradual value satisfies any annotation, so a clean shape report does not prove
that shapes are tracked. This corrupts the first return shape of each function
declared with ``@static_jaxtyping``, in a temporary copy of ``src``, and sorts the
functions by what the checker notices:

- inferred: the corruption is caught with local annotations removed, so the checker
  knows the producer's shape.
- annotation-only: it is caught only against a local annotation, which shows two
  declarations agree and not that either is true.
- unchecked: it goes unnoticed.

Corrupting a callee makes its callers gradual, so each function missed in a batch
is retried alone. Native-only signatures are not covered. Run from the package
directory with paths under ``src``. Exits 2 if the checker cannot give an answer.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_WRAPPERS = {"Float", "Int", "Bool", "Num", "Real", "Key", "UInt32", "Scalar", "Shaped"}
_PROBE = "zzz"
_CHECK = (
    "import subprocess, sys; sys.exit(subprocess.run(['pyrefly', 'check', '-c', "
    "'/dev/null', '--preset', 'strict', '--threads', '2', '--python-version', "
    "'3.13', '--python-interpreter-path', sys.executable, '--search-path', '.', "
    "'--search-path', 'src', '--output-format', 'min-text', *sys.argv[1:]])"
    ".returncode)"
)
_SUMMARY = re.compile(r"INFO \d+ errors?\b")
_CHECKER_COMPLETED = {0, 1}


class AuditError(RuntimeError):
    """The audit cannot give a trustworthy answer."""


@dataclass(frozen=True)
class Mutation:
    """One function whose declared return shape was corrupted."""

    file: str
    name: str
    start: int
    # Removing a local annotation can move the end of a function, never its start.
    end: int = field(compare=False)


def _declaration(function: ast.FunctionDef) -> ast.Constant | None:
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "static_jaxtyping"
            and isinstance(decorator.args[0], ast.Constant)
        ):
            return decorator.args[0]
    return None


def _return_shape(function: ast.FunctionDef) -> ast.Constant | None:
    if function.returns is None:
        return None
    for node in ast.walk(function.returns):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in _WRAPPERS
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2  # noqa: PLR2004  # dtype[array, shape]
        ):
            shape = node.slice.elts[1]
            if isinstance(shape, ast.Constant) and isinstance(shape.value, str):
                return shape
    return None


def _has_body(function: ast.FunctionDef) -> bool:
    return not (
        all(isinstance(line, (ast.Expr, ast.Pass)) for line in function.body)
        or any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in function.decorator_list
        )
    )


def _local_annotations(function: ast.FunctionDef) -> Iterator[ast.AnnAssign]:
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                yield node
        elif not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            pending.extend(ast.iter_child_nodes(node))


def _end(node: ast.expr | ast.stmt) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        message = "Parsed source has no end position."
        raise AuditError(message)
    return node.end_lineno, node.end_col_offset


def strip_local_annotations(source: bytes) -> bytes:
    """Remove annotated-assignment annotations in functions, keeping line numbers."""
    starts = [0]
    for line in source.split(b"\n"):
        starts.append(starts[-1] + len(line) + 1)

    def offset(position: tuple[int, int]) -> int:
        return starts[position[0] - 1] + position[1]

    edits: list[tuple[int, int, bytes]] = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in _local_annotations(function):
            first = offset(_end(node.target))
            equals = source.index(b"=", offset(_end(node.annotation)))
            edits.append((first, equals, b" "))
            statement_end = offset(_end(node))
            removed_lines = source.count(b"\n", first, equals)
            edits.append((statement_end, statement_end, b"\n" * removed_lines))
    for first, last, text in sorted(edits, reverse=True):
        source = source[:first] + text + source[last:]
    return source


def _mutate(
    path: Path, root: Path, only: set[Mutation] | None = None
) -> list[Mutation]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    edits: list[tuple[ast.Constant, str]] = []
    mutations: list[Mutation] = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef) or not _has_body(function):
            continue
        declaration, shape = _declaration(function), _return_shape(function)
        if declaration is None or shape is None:
            continue
        mutation = Mutation(
            str(path.relative_to(root)),
            function.name,
            function.lineno,
            function.end_lineno or function.lineno,
        )
        if only is not None and mutation not in only:
            continue
        corrupted = re.sub(
            r"(^|\s)([A-Za-z_]\w*)", rf"\1{_PROBE}", str(shape.value), count=1
        )
        edits.append((shape, corrupted if _PROBE in corrupted else _PROBE))
        edits.append((declaration, f"{declaration.value} {_PROBE}"))
        mutations.append(mutation)
    for node, text in sorted(
        edits, key=lambda edit: (edit[0].lineno, edit[0].col_offset), reverse=True
    ):
        raw = lines[node.lineno - 1].encode()
        lines[node.lineno - 1] = (
            raw[: node.col_offset]
            + json.dumps(text).encode()
            + raw[node.end_col_offset :]
        ).decode()
    if mutations:
        path.write_text("".join(lines), encoding="utf-8")
    return mutations


def run_checker(root: Path, package: Path, targets: list[str]) -> str:
    """Return checker output, raising unless the checker ran to completion."""
    result = subprocess.run(  # noqa: S603  # fixed locked toolchain command
        [  # noqa: S607
            "uv",
            "run",
            "--project",
            str(package),
            "--locked",
            "--isolated",
            "--group",
            "shapes",
            "python",
            "-c",
            _CHECK,
            *targets,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in _CHECKER_COMPLETED or not _SUMMARY.search(
        result.stderr
    ):
        message = (
            f"The checker did not complete (exit {result.returncode}):\n"
            f"{result.stderr}{result.stdout}"
        )
        raise AuditError(message)
    return result.stdout


def _caught(output: str, mutations: list[Mutation]) -> set[Mutation]:
    caught: set[Mutation] = set()
    for line in output.splitlines():
        match = re.match(r"ERROR (\S+?):(\d+):", line)
        if match is not None and _PROBE in line:
            caught.update(
                mutation
                for mutation in mutations
                if mutation.file == match[1]
                and mutation.start <= int(match[2]) <= mutation.end
            )
    return caught


def _fresh_copy(package: Path, root: Path) -> None:
    shutil.rmtree(root / "src", ignore_errors=True)
    shutil.copytree(
        package / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )


def selected_targets(package: Path, targets: list[str]) -> list[str]:
    """Return existing Python targets under ``src``, relative to the package."""
    base = package.resolve()
    selected: list[str] = []
    for target in targets or ["src/statadic"]:
        path = (base / target).resolve()
        if not path.is_relative_to(base / "src"):
            message = f"{target} is outside src"
            raise AuditError(message)
        if not (path.is_dir() or (path.is_file() and path.suffix == ".py")):
            message = f"{target} is not a directory or Python file"
            raise AuditError(message)
        selected.append(str(path.relative_to(base)))
    return selected


def _strip_file(file: Path) -> None:
    file.write_bytes(strip_local_annotations(file.read_bytes()))


def _run_pass(
    package: Path,
    root: Path,
    selected: list[str],
    *,
    strip: bool,
    only: set[Mutation] | None = None,
) -> tuple[list[Mutation], set[Mutation]]:
    """Corrupt the selected functions together, then retry each one missed alone."""
    _fresh_copy(package, root)
    mutations: list[Mutation] = []
    for target in selected:
        path = root / target
        for file in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            if strip:
                _strip_file(file)
            mutations.extend(_mutate(file, root, only))
    caught = _caught(run_checker(root, package, selected), mutations)
    for mutation in mutations:
        if mutation in caught:
            continue
        _fresh_copy(package, root)
        if strip:
            _strip_file(root / mutation.file)
        _mutate(root / mutation.file, root, {mutation})
        caught |= _caught(run_checker(root, package, [mutation.file]), [mutation])
    return mutations, caught


def _audit(package: Path, targets: list[str]) -> None:
    selected = selected_targets(package, targets)
    with tempfile.TemporaryDirectory(prefix="shape-coverage-") as directory:
        root = Path(directory)
        _fresh_copy(package, root)
        if _PROBE in run_checker(root, package, selected):
            message = f"The unmodified source already reports {_PROBE}."
            raise AuditError(message)
        mutations, caught = _run_pass(package, root, selected, strip=False)
        inferred = (
            _run_pass(package, root, selected, strip=True, only=caught)[1]
            if caught
            else set()
        )
    for label, members in (
        ("inferred", inferred),
        ("annotation-only", caught - inferred),
        ("unchecked", set(mutations) - caught),
    ):
        sys.stdout.write(f"{label} {len(members)} of {len(mutations)}\n")
    for label, members in (
        ("ANNOTATION-ONLY", caught - inferred),
        ("UNCHECKED", set(mutations) - caught),
    ):
        for mutation in mutations:
            if mutation in members:
                sys.stdout.write(
                    f"{label} {mutation.file}:{mutation.start} {mutation.name}\n"
                )


def main(targets: list[str]) -> int:
    """Classify each declared return shape; exit 2 if the audit cannot answer."""
    try:
        _audit(Path.cwd(), targets)
    except AuditError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
