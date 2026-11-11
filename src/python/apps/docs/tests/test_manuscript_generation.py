"""Exercise actual generator control flow without executing its scientific campaigns."""

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / "src/latex/projects/01_statadic/analysis/generate.py"


def extracted(name, namespace):
    tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace[name]


def test_formal_copy_calls_canonical_freshness_before_reading_or_copying(tmp_path):
    def stale(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    function = extracted(
        "formal",
        {
            "Path": Path,
            "subprocess": SimpleNamespace(run=stale),
            "sys": sys,
            "ROOT": tmp_path,
        },
    )
    with pytest.raises(subprocess.CalledProcessError):
        function(tmp_path / "generated")
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("late_failure", [False, True])
def test_generation_replaces_only_complete_output(tmp_path, late_failure):
    target = tmp_path / "generated"
    target.mkdir()
    (target / "old.tex").write_text("last complete generation")

    def produce(stage):
        (stage / "new.tex").write_text("complete new generation")

    def finish(stage):
        if late_failure:
            raise RuntimeError("late numerical failure")

    namespace = {
        "Path": Path,
        "tempfile": tempfile,
        "ROOT": tmp_path,
        "sys": sys,
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: None),
        "generation_inputs": lambda: {"formal": "frozen"},
        "values": {},
        **{
            name: produce
            for name in (
                "provenance",
                "listings",
                "agreement",
                "coverage",
                "formal",
                "retained",
            )
        },
        "timing": finish,
    }
    function = extracted("generate", namespace)
    if late_failure:
        with pytest.raises(RuntimeError, match="late numerical failure"):
            function(target)
        assert (target / "old.tex").read_text() == "last complete generation"
        assert not (target / "new.tex").exists()
    else:
        function(target)
        assert not (target / "old.tex").exists()
        assert (target / "new.tex").read_text() == "complete new generation"
        assert (target / "values.tex").is_file()


def test_generation_refuses_output_symlink_before_resolving_it(tmp_path):
    destination = tmp_path / "prior-generation"
    destination.mkdir()
    (destination / "old.tex").write_text("preserve this result")
    link = tmp_path / "output-link"
    link.symlink_to(destination, target_is_directory=True)
    function = extracted("generate", {"Path": Path})
    with pytest.raises(ValueError, match="symbolic link"):
        function(link)
    assert link.is_symlink()
    assert (destination / "old.tex").read_text() == "preserve this result"


def input_fixture(root):
    package = root / "src/python/packages/statadic"
    script = root / "src/latex/projects/01_statadic/analysis/generate.py"
    paths = [
        script,
        root / "src/python/uv.lock",
        root / "src/python/pyproject.toml",
        package / "pyproject.toml",
        package / "docs/validation.md",
        package / "src/statadic/existing.py",
        root / "src/latex/refs.bib",
        root / "src/lean/build/certification.json",
        root / "src/lean/lean-toolchain",
        root / "src/lean/lake-manifest.json",
        root / "src/lean/registry/methods.json",
        root / "src/lean/registry/claims.json",
        root / "src/lean/tools/correspondence.py",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen input")
    namespace = {
        "Path": Path,
        "ROOT": root,
        "PACKAGE": package,
        "__file__": str(script),
        "LISTINGS": script.parents[1] / "src/listings",
        "hashlib": hashlib,
    }
    return extracted("generation_inputs", namespace), package


@pytest.mark.parametrize("change", ["added-source", "methods-registry"])
def test_input_change_aborts_publication_and_preserves_prior_output(tmp_path, change):
    inputs, package = input_fixture(tmp_path)
    target = tmp_path / "generated"
    target.mkdir()
    (target / "old.tex").write_text("last complete generation")

    def produce(stage):
        (stage / "new.tex").write_text("candidate generation")

    def change_inputs(stage):
        path = (
            package / "src/statadic/added.py"
            if change == "added-source"
            else tmp_path / "src/lean/registry/methods.json"
        )
        path.write_text("changed during scientific computation")

    namespace = {
        "Path": Path,
        "tempfile": tempfile,
        "ROOT": tmp_path,
        "sys": sys,
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: None),
        "generation_inputs": inputs,
        "values": {},
        **{
            name: produce
            for name in (
                "provenance",
                "listings",
                "agreement",
                "coverage",
                "formal",
                "retained",
            )
        },
        "timing": change_inputs,
    }
    function = extracted("generate", namespace)
    with pytest.raises(ValueError, match="inputs changed"):
        function(target)
    assert (target / "old.tex").read_text() == "last complete generation"
    assert not (target / "new.tex").exists()
