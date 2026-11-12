"""The notebook gate runner accepts the scaffold and rejects broken notebooks."""

import importlib.util
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
SCAFFOLD = APP.parents[1] / "packages/statadic/docs/_templates/notebook.py"
spec = importlib.util.spec_from_file_location(
    "check_notebooks", APP / "scripts/check_notebooks.py"
)
check_notebooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_notebooks)


@pytest.mark.notebook
def test_discovery_includes_the_scaffold():
    assert check_notebooks.discover()[-1] == SCAFFOLD


@pytest.mark.notebook
def test_the_scaffold_passes_every_gate(capsys):
    assert check_notebooks.main(["--paths", str(SCAFFOLD)]) == 0
    report = capsys.readouterr().out
    assert report.count("pass") == 6
    assert "FAIL" not in report


@pytest.mark.notebook
def test_a_failing_cell_is_reported(tmp_path, capsys):
    broken = tmp_path / "broken.py"
    broken.write_text(
        "import marimo\n\napp = marimo.App()\n\n\n"
        "@app.cell\ndef _():\n    raise ValueError('boom')\n    return\n\n\n"
        "if __name__ == '__main__':\n    app.run()\n"
    )
    assert check_notebooks.main(["--paths", str(broken)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_published_notebook_requires_collected_tests(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(check_notebooks, "DOCS", tmp_path)
    notebook = tmp_path / "tutorials" / "empty.py"
    notebook.parent.mkdir()
    notebook.write_text("")
    monkeypatch.setattr(
        check_notebooks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=5, stdout="no tests collected", stderr=""
        ),
    )
    gates = dict(check_notebooks.gates(notebook))
    assert gates["tests"]() == "no tests collected"
    assert gates["doctests"]() is None


def test_unparsable_gate_scans_all_docs(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    nested = docs / "reference" / "broken.py"
    nested.parent.mkdir(parents=True)
    nested.write_text('app._unparsable_cell("broken")\n')
    monkeypatch.setattr(check_notebooks, "DOCS", docs)
    monkeypatch.setattr(check_notebooks, "discover", lambda: [])
    assert check_notebooks.main([]) == 1
    assert "reference/broken.py" in capsys.readouterr().err


def test_e2e_marker_applies_only_to_canonical_notebook_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from statadic_docs_pages import pytest_notebooks

    root = tmp_path / "docs"
    monkeypatch.setattr(pytest_notebooks, "SECTIONS", (root / "tutorials",))
    marked = []
    inside = SimpleNamespace(
        path=root / "tutorials" / "lesson.py", add_marker=marked.append
    )
    outside = SimpleNamespace(
        path=root / "reference" / "lesson.py",
        add_marker=lambda marker: marked.append("wrong"),
    )
    pytest_notebooks.pytest_collection_modifyitems([inside, outside])
    assert len(marked) == 1 and marked[0].name == "e2e"
