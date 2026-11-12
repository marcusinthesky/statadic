"""Reference runner evidence binds the imported package and test definitions."""

import runpy
from pathlib import Path

import pytest
import statadic

pytestmark = [pytest.mark.component, pytest.mark.contract, pytest.mark.regression]


def test_reference_inputs_detect_same_version_source_and_definition_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source bytes and newly added test definitions invalidate old evidence."""
    package = Path(__file__).resolve().parents[2]
    runner = runpy.run_path(str(package / "scripts/check_references.py"))
    source = tmp_path / "installed/statadic"
    source.mkdir(parents=True)
    module = source / "__init__.py"
    module.write_text("__version__ = 'unchanged'\n")
    monkeypatch.setattr(statadic, "__file__", str(module))
    fixture = tmp_path / "package"
    (fixture / "tests").mkdir(parents=True)
    (fixture / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    identity = runner["input_identity"](fixture)
    module.write_text("__version__ = 'unchanged'\nchanged = True\n")
    assert runner["input_identity"](fixture) != identity
    stable = runner["input_identity"](fixture)
    (fixture / "tests/test_added.py").write_text("def test_added(): pass\n")
    assert runner["input_identity"](fixture) != stable
    (fixture / "justfile").write_text(
        "test-external:\n    pytest -m external_reference\n"
    )
    recipe_identity = runner["input_identity"](fixture)
    (fixture / "justfile").write_text("test-external:\n    pytest -m unrelated\n")
    assert runner["input_identity"](fixture) != recipe_identity


def test_reference_inputs_bind_explicit_catalogue_bytes(tmp_path: Path) -> None:
    """An edited mapping cannot reuse execution evidence from another index."""
    package = Path(__file__).resolve().parents[2]
    runner = runpy.run_path(str(package / "scripts/check_references.py"))
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text('{"methods": []}\n')
    before = runner["input_identity"](package, catalogue)
    catalogue.write_text('{"methods": [], "scientific_procedures": []}\n')
    assert runner["input_identity"](package, catalogue) != before
