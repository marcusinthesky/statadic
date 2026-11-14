"""The notebook gate runner validates notebooks and reports failures."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1]
SCAFFOLD = APP.parents[1] / "packages/statadic/docs/_templates/notebook.py"
spec = importlib.util.spec_from_file_location(
    "check_notebooks", APP / "scripts/check_notebooks.py"
)
check_notebooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_notebooks)


@pytest.mark.notebook
def test_scaffold_is_discovered_and_passes_lint_and_parse():
    assert check_notebooks.discover()[-1] == SCAFFOLD
    gates = dict(check_notebooks.gates(SCAFFOLD))
    assert gates["lint"]() is None
    assert gates["parse"]() is None


@pytest.mark.notebook
@pytest.mark.parametrize("x64", ["0", "1"])
def test_scaffold_executes_as_a_script(x64):
    """Retain the script entrypoint contract on one representative notebook."""
    assert (
        check_notebooks.run(
            [sys.executable, str(SCAFFOLD)], env={"JAX_ENABLE_X64": x64}
        )
        is None
    )


@pytest.mark.notebook
def test_a_failing_cell_is_reported(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text(
        "import marimo\n\napp = marimo.App()\n\n\n"
        "@app.cell\ndef _():\n    raise ValueError('boom')\n    return\n\n\n"
        "if __name__ == '__main__':\n    app.run()\n"
    )
    assert "boom" in dict(check_notebooks.gates(broken, extended=True))["x64=0"]()


@pytest.mark.parametrize("problem", [None, "gate failure"])
@pytest.mark.parametrize("extended", [False, True])
def test_main_reports_results_and_continues_after_failure(
    tmp_path, monkeypatch, capsys, problem, extended
):
    notebook = tmp_path / "notebook.py"
    visited = []

    def gate(name, result):
        def execute():
            visited.append(name)
            return result

        return execute

    def gates(path, **options):
        assert path == notebook
        assert options["extended"] is extended
        return [("first", gate("first", problem)), ("later", gate("later", None))]

    monkeypatch.setattr(check_notebooks, "gates", gates)
    arguments = [str(notebook)] if extended else ["--paths", str(notebook)]
    if extended:
        arguments.insert(0, "--extended")
    assert check_notebooks.main(arguments) == (problem is not None)
    output = capsys.readouterr()
    assert visited == ["first", "later"]
    assert "first" in output.out and "later" in output.out
    assert ("FAIL" in output.out) == (problem is not None)
    if problem:
        assert problem in output.err


def test_gate_commands_and_published_requirement(tmp_path, monkeypatch):
    monkeypatch.setattr(check_notebooks, "DOCS", tmp_path)
    notebook = tmp_path / "tutorials" / "lesson.py"
    notebook.parent.mkdir()
    notebook.write_text("")
    calls = []
    monkeypatch.setattr(
        check_notebooks,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    gates = dict(check_notebooks.gates(notebook))
    assert list(gates) == ["lint", "parse", "tests+doctests"]
    for gate in gates.values():
        gate()
    command, options = calls[1]
    assert command.count("--doctest-modules") == 1
    assert command[command.index("--require-notebook-tests") + 1] == str(notebook)
    assert command[-1] == str(notebook)
    assert options["allow_no_tests"] is False
    calls.clear()
    extended = dict(check_notebooks.gates(notebook, extended=True))
    extended["x64=0"]()
    extended["x64=1"]()
    assert [options["env"] for _, options in calls] == [
        {"JAX_ENABLE_X64": "0"},
        {"JAX_ENABLE_X64": "1"},
    ]
    calls.clear()
    dict(check_notebooks.gates(tmp_path / "other.py"))["tests+doctests"]()
    assert "--require-notebook-tests" not in calls[0][0]
    assert calls[0][1]["allow_no_tests"] is True


def test_only_empty_collection_can_be_allowed(monkeypatch):
    monkeypatch.setattr(
        check_notebooks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=5, stdout="empty", stderr=""
        ),
    )
    assert check_notebooks.run(["pytest"]) == "empty"
    assert check_notebooks.run(["pytest"], allow_no_tests=True) is None
    monkeypatch.setattr(
        check_notebooks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="error", stderr=""
        ),
    )
    assert check_notebooks.run(["pytest"], allow_no_tests=True) == "error"


@pytest.mark.parametrize(
    ("source", "required", "expected", "missing"),
    [
        ("def test_ok():\n    pass\n", True, 0, False),
        ('""">>> 1 + 1\n2\n"""\n', True, 1, True),
        ("", True, 1, True),
        ("", False, 5, False),
        ("def test_bad():\n    assert False\n", True, 1, False),
        ('""">>> 1 + 1\n3\n"""\ndef test_ok():\n    pass\n', True, 1, False),
        ("def invalid(:\n", True, 2, True),
    ],
)
def test_combined_pytest_collection(tmp_path, source, required, expected, missing):
    notebook = tmp_path / "notebook.py"
    notebook.write_text(source)
    result = run_notebook_pytest(notebook, required=required)
    assert result.returncode == expected, result.stdout + result.stderr
    assert ("No ordinary tests collected from notebook" in result.stdout) is missing


def run_notebook_pytest(notebook, *other_paths, required=True):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-c",
        os.devnull,
        "-p",
        "no:cacheprovider",
        "-p",
        "statadic_docs_pages.pytest_notebooks",
        "--doctest-modules",
    ]
    if required:
        command += ["--require-notebook-tests", str(notebook)]
    command.extend(map(str, (notebook, *other_paths)))
    env = os.environ | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)


def test_combined_pytest_collects_marimo_cell_test(tmp_path):
    notebook = tmp_path / "notebook.py"
    notebook.write_text(
        "import marimo\n\napp = marimo.App()\n\n"
        "@app.cell(hide_code=True)\ndef _():\n"
        "    def test_cell():\n        assert True\n    return\n"
    )
    result = run_notebook_pytest(notebook)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_combined_pytest_counts_only_target_and_runs_ordinary_once(tmp_path):
    notebook = tmp_path / "notebook.py"
    counter = tmp_path / "counter"
    notebook.write_text(
        '""">>> 1 + 1\n2\n"""\n'
        f"def test_once():\n    open({str(counter)!r}, 'a').write('x')\n"
    )
    other = tmp_path / "other.py"
    other.write_text("def test_other():\n    pass\n")
    result = run_notebook_pytest(notebook, other)
    assert result.returncode == 0, result.stdout + result.stderr
    assert counter.read_text() == "x"
    notebook.write_text('""">>> 1 + 1\n2\n"""\n')
    result = run_notebook_pytest(notebook, other)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No ordinary tests collected from notebook" in result.stdout


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
