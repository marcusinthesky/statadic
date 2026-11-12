"""Gate marimo notebooks: lint, unparsable cells, both precisions, tests, doctests.

Checks validate without repair; ``marimo check --fix`` belongs to ``just fmt``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
DOCS = APP.parents[1] / "packages/statadic/docs"
SECTIONS = ("start", "tutorials", "guides")
NO_TESTS_COLLECTED = 5
THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def discover() -> list[Path]:
    """Notebooks beside the pages, plus the authoring scaffold."""
    found = [
        path
        for section in SECTIONS
        for path in sorted((DOCS / section).rglob("*.py"))
        if "__marimo__" not in path.parts and path.name != "conftest.py"
    ]
    return [*found, DOCS / "_templates/notebook.py"]


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    allow_no_tests: bool = False,
) -> str | None:
    """Return None on success, else the tail of the combined output."""
    result = subprocess.run(
        command,
        env=os.environ | THREADS | (env or {}),
        capture_output=True,
        text=True,
        check=False,
    )
    ok = result.returncode == 0 or (
        result.returncode == NO_TESTS_COLLECTED and allow_no_tests
    )
    output = (result.stdout + result.stderr)[-3000:]
    return None if ok else output or f"command exited with status {result.returncode}"


def gates(notebook: Path):
    """Yield (name, thunk) in the order the plan lists them."""
    marimo = [sys.executable, "-m", "marimo"]
    pytest = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    pytest += ["-c", str(APP / "pyproject.toml")]
    yield (
        "lint",
        lambda: run(
            [*marimo, "check", "--strict", "--select", "MB,MR,MF", str(notebook)]
        ),
    )
    yield (
        "parse",
        lambda: (
            "contains _unparsable_cell"
            if "_unparsable_cell" in notebook.read_text()
            else None
        ),
    )
    for x64 in ("0", "1"):
        yield (
            f"x64={x64}",
            lambda x64=x64: run(
                [sys.executable, str(notebook)], env={"JAX_ENABLE_X64": x64}
            ),
        )
    published = any(
        notebook.resolve().is_relative_to(DOCS / section) for section in SECTIONS
    )
    yield "tests", lambda: run([*pytest, str(notebook)], allow_no_tests=not published)
    yield (
        "doctests",
        lambda: run([*pytest, "--doctest-modules", str(notebook)], allow_no_tests=True),
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path, help="Notebooks to check")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply marimo's automatic lint fixes instead of running the gates",
    )
    selected = parser.parse_args(arguments)
    notebooks = [p.resolve() for p in selected.paths] if selected.paths else discover()
    if selected.fix:
        command = [sys.executable, "-m", "marimo", "check", "--fix"]
        return subprocess.run([*command, *map(str, notebooks)], check=False).returncode
    failures: dict[str, dict[str, str]] = {}
    if not selected.paths:
        for path in sorted(DOCS.rglob("*.py")):
            if (
                "__marimo__" not in path.parts
                and "_unparsable_cell" in path.read_text()
            ):
                failures.setdefault(str(path.relative_to(DOCS)), {})["parse"] = (
                    "contains _unparsable_cell"
                )
    rows: list[tuple[str, list[tuple[str, bool]]]] = []
    for notebook in notebooks:
        label = (
            str(notebook.relative_to(DOCS))
            if notebook.is_relative_to(DOCS)
            else str(notebook)
        )
        results = []
        for name, gate in gates(notebook):
            print(f"Checking {label}: {name}", flush=True)
            problem = gate()
            results.append((name, problem is None))
            if problem is not None:
                failures.setdefault(label, {})[name] = problem
                print(f"\n--- {label}: {name}\n{problem}", file=sys.stderr, flush=True)
        rows.append((label, results))
    width = max((len(label) for label, _ in rows), default=8)
    names = [name for name, _ in rows[0][1]] if rows else []
    print(f"{'notebook':<{width}}  " + "  ".join(f"{n:<8}" for n in names))
    for label, results in rows:
        cells = "  ".join(f"{'pass' if ok else 'FAIL':<8}" for _, ok in results)
        print(f"{label:<{width}}  {cells}")
    for label, problems in failures.items():
        for name, output in problems.items():
            print(f"\n--- {label}: {name}\n{output}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
