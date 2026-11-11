# standalone package-maintenance entry point
"""Failure handling and annotation stripping in the shape-coverage audit."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    """Load the standalone script without treating ``scripts`` as a package."""
    path = Path(__file__).parents[2] / "scripts" / "shape_coverage.py"
    spec = importlib.util.spec_from_file_location("shape_coverage", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("target", ["../outside.py", "src/../outside.py", "/etc/hosts"])
def test_targets_cannot_escape_the_source_tree(
    audit: ModuleType, target: str, tmp_path: Path
) -> None:
    """A target outside ``src`` is refused before anything is read or rewritten."""
    (tmp_path / "src").mkdir()
    (tmp_path / "outside.py").write_text("value = 1\n")
    with pytest.raises(audit.AuditError, match="outside src"):
        audit.selected_targets(tmp_path, [target])


@pytest.mark.integration
@pytest.mark.contract
def test_missing_target_is_an_error(audit: ModuleType, tmp_path: Path) -> None:
    """A mistyped path cannot pass as a clean audit."""
    (tmp_path / "src").mkdir()
    with pytest.raises(audit.AuditError, match="not a directory or Python file"):
        audit.selected_targets(tmp_path, ["src/missing.py"])


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [(0, ""), (1, "error: no such group"), (2, " INFO 0 errors")],
    ids=["silent", "toolchain-failure", "checker-crash"],
)
def test_checker_failure_keeps_its_diagnostics(
    audit: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    stderr: str,
) -> None:
    """A run without a completed checker summary raises instead of finding nothing."""

    def finished(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, "", stderr)

    monkeypatch.setattr(subprocess, "run", finished)
    with pytest.raises(audit.AuditError, match=f"exit {returncode}") as caught:
        audit.run_checker(tmp_path, tmp_path, ["src"])
    assert stderr in str(caught.value)


@pytest.mark.integration
@pytest.mark.contract
def test_main_reports_infrastructure_failure_with_exit_two(
    audit: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The exit status separates "cannot answer" from unchecked functions."""
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path)
    assert audit.main(["../outside.py"]) == 2  # noqa: PLR2004  # documented status
    assert "outside src" in capsys.readouterr().err


@pytest.mark.component
def test_stripping_removes_only_function_local_annotations(audit: ModuleType) -> None:
    """Later lines stay put, and class fields and signatures keep their annotations."""
    source = b"""\
field: int = 1
def f(x: int) -> int:
    a: int = x
    b: tuple[
        int,
        int,
    ] = (x, x)
    c: (int) = x  # comment
    d: int
    class Inner:
        e: int = 1
    if x:
        g: int = 2
    def nested() -> int:
        h: int = x
        return h
    return a
marker = 1
"""
    stripped = audit.strip_local_annotations(source)
    tree = ast.parse(stripped)
    annotated = sorted(
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )
    assert annotated == ["d", "e", "field"]
    assert b"def f(x: int) -> int:" in stripped
    assert source.index(b"marker") != stripped.index(b"marker")
    assert source.count(b"\n") == stripped.count(b"\n")
    marker = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "marker"
    )
    assert marker.lineno == source.split(b"\n").index(b"marker = 1") + 1
