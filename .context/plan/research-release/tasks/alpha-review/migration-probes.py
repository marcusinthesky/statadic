"""Inspect removed-import consumers and isolate the manuscript formal-copy step."""

import ast
import hashlib
import importlib
import json
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def unavailable_imports(source):
    result = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "statadic.api.regression":
            module = importlib.import_module(node.module)
            result.extend(
                name.name for name in node.names if not hasattr(module, name.name)
            )
    return result


def run():
    distribution_path = ROOT / "src/python/packages/statadic/scripts/check_distribution.py"
    tree = ast.parse(distribution_path.read_text())
    embedded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_INSTALLED_CHECK"
            for target in node.targets
        )
    )
    generator_path = ROOT / "src/latex/projects/01_statadic/analysis/generate.py"
    generator = generator_path.read_text()
    extensions = ROOT / "src/python/apps/docs/tests/test_external_extensions.py"
    sys.path.insert(0, str(ROOT / "src/lean/tools"))
    bridge = runpy.run_path(str(ROOT / "src/lean/tools/bridge.py"))
    try:
        bridge["fresh_output"](ROOT)
        freshness = "accepted"
    except ValueError as error:
        freshness = str(error)
    formal = next(
        node
        for node in ast.parse(generator).body
        if isinstance(node, ast.FunctionDef) and node.name == "formal"
    )
    namespace = {
        "ROOT": ROOT,
        "Path": Path,
        "json": json,
        "runpy": runpy,
        "shutil": shutil,
        "put": lambda *args: None,
        "tex_escape": str,
    }
    exec(
        compile(
            ast.Module(body=[formal], type_ignores=[]), str(generator_path), "exec"
        ),
        namespace,
    )
    with tempfile.TemporaryDirectory(prefix="statadic-alpha-formal-copy-") as temporary:
        destination = Path(temporary)
        namespace["formal"](destination)
        copied = destination / "statadic-mathematics.tex"
        copied_hash = hashlib.sha256(copied.read_bytes()).hexdigest()
    return {
        "removed_distribution_imports": unavailable_imports(embedded),
        "removed_manuscript_imports": unavailable_imports(generator),
        "removed_docs_test_imports": unavailable_imports(extensions.read_text()),
        "canonical_freshness_check": freshness,
        "isolated_formal_step_copied_sha256": copied_hash,
        "limits": "Actual candidate namespaces and extracted formal() body; only temporary output. No wheel, docs collection, full generator or manuscript build executed.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
