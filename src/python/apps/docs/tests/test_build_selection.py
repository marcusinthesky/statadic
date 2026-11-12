"""Documentation inputs and evidence selections remain explicit and retained."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def build_module():
    path = Path(__file__).resolve().parents[1] / "scripts/build.py"
    spec = importlib.util.spec_from_file_location("docs_build_selection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection(arguments):
    return build_module().study_selection(arguments)


def test_unselected_local_exports_remain_absent(tmp_path, monkeypatch):
    publication = tmp_path / "build/studies/example-publication"
    publication.mkdir(parents=True)
    (publication / "evidence-manifest.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    selected = selection(["--release-evidence"])
    assert selected.release_evidence
    assert selected.study_export == []


def test_exports_are_explicit_absolute_and_unique(tmp_path):
    (tmp_path / "evidence-manifest.json").write_text("{}")
    selected = selection(["--study-export", str(tmp_path)])
    assert selected.study_export == [tmp_path.resolve()]
    assert not selected.release_evidence
    with pytest.raises(SystemExit):
        selection(["--study-export", str(tmp_path), "--study-export", str(tmp_path)])


def test_missing_export_is_rejected_before_a_build(tmp_path):
    with pytest.raises(SystemExit):
        selection(["--study-export", str(tmp_path)])


def test_build_provenance_binds_exact_mapped_test_sources(tmp_path, monkeypatch):
    """Changed published tests enter the existing before/after input comparison."""
    module = build_module()
    app = tmp_path / "src/python/apps/docs"
    python = tmp_path / "src/python"
    package = python / "packages/statadic"
    paths = {
        "src/python/apps/studies/catalogue.json": json.dumps(
            {
                "methods": [
                    {
                        "reference_mappings": [
                            {
                                "nodeids": [
                                    "tests/test_numerical.py::test_value[float64]"
                                ]
                            }
                        ]
                    }
                ],
                "scientific_procedures": [
                    {
                        "variants": [
                            {
                                "reference_mappings": [
                                    {
                                        "nodeids": [
                                            "tests/test_scientific.py::test_fit[weighted]"
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ],
            }
        ),
        "src/python/apps/performance/src/statadic_performance/identities.json": "{}",
        "src/python/apps/docs/conf.py": "",
        "src/python/apps/docs/pyproject.toml": "",
        "src/python/apps/docs/data/family-sources.json": "{}",
        "src/python/apps/docs/evidence/known-limitations.json": "{}",
        "src/python/uv.lock": "",
        "devenv.lock": "",
        "src/lean/build/certification.json": "{}",
        "src/latex/refs.bib": "",
        "src/python/packages/statadic/docs/tutorials/example.py": "# notebook source\n",
        "src/python/packages/statadic/docs/tutorials/__pycache__/example.pyc": "cache",
        "src/python/packages/statadic/docs/tutorials/__marimo__/session.json": "{}",
        "src/python/packages/statadic/tests/test_numerical.py": (
            "def test_value(): pass\n"
        ),
        "src/python/packages/statadic/tests/test_scientific.py": (
            "def test_fit(): pass\n"
        ),
        "src/python/packages/statadic/tests/test_unmapped.py": (
            "def test_other(): pass\n"
        ),
    }
    for relative, content in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(module, "APP", app)
    monkeypatch.setattr(module, "PYTHON", python)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DOCS", package / "docs")
    monkeypatch.setattr(module, "version", lambda name: "fixture")
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *args, **kwargs: "fixture"
    )
    monkeypatch.setattr(
        module,
        "jax",
        SimpleNamespace(
            default_backend=lambda: "cpu",
            devices=lambda: [],
            config=SimpleNamespace(x64_enabled=False),
        ),
    )
    provenance = module.provenance(study_exports=[], release_evidence=False)
    assert provenance["origin_revision"] == provenance["build_revision"] == "fixture"
    before = provenance["input_sha256"]
    for record in ("data/family-sources.json", "evidence/known-limitations.json"):
        source = app / record
        assert (
            before[str(source.relative_to(tmp_path))]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
    assert not any("__pycache__" in path or "__marimo__" in path for path in before)
    notebook = package / "docs/tutorials/example.py"
    assert (
        before[str(notebook.relative_to(tmp_path))]
        == hashlib.sha256(notebook.read_bytes()).hexdigest()
    )
    for name in ("test_numerical.py", "test_scientific.py"):
        path = package / "tests" / name
        relative = str(path.relative_to(tmp_path))
        assert before[relative] == hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text(path.read_text() + "# changed mapped source\n")
        after = module.provenance(study_exports=[], release_evidence=False)[
            "input_sha256"
        ]
        assert after[relative] != before[relative]
    assert "src/python/packages/statadic/tests/test_unmapped.py" not in before
    (package / "tests/test_scientific.py").unlink()
    with pytest.raises(ValueError, match="Invalid reference-test source"):
        module.provenance(study_exports=[], release_evidence=False)
