"""Build the documentation site into ignored output."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import jax

APP = Path(__file__).resolve().parents[1]
PYTHON = APP.parents[1]
DOCS = PYTHON / "packages/statadic/docs"
OUTPUT = APP / "build"
HTML = OUTPUT / "html"
ROOT = PYTHON.parent.parent


def study_selection(arguments: list[str] | None = None) -> argparse.Namespace:
    """Select exports without discovering unrelated local campaign directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-export", action="append", type=Path, default=[])
    parser.add_argument(
        "--release-evidence",
        action="store_true",
        help="Render exports selected and verified by the release orchestrator",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Authoring build: reuse the existing Lean certification and skip the "
            "blueprint"
        ),
    )
    selected = parser.parse_args(arguments)
    selected.study_export = [path.resolve() for path in selected.study_export]
    if len(set(selected.study_export)) != len(selected.study_export):
        parser.error("A study export may only be selected once")
    for path in selected.study_export:
        if not (path / "evidence-manifest.json").is_file():
            parser.error(f"Not a retained publication export: {path}")
    return selected


def provenance(*, study_exports: list[Path], release_evidence: bool) -> dict:
    """Describe this build separately from the notebooks' source origin."""
    inputs = [
        PYTHON / "apps/studies/catalogue.json",
        PYTHON / "apps/performance/src/statadic_performance/identities.json",
        APP / "conf.py",
        APP / "pyproject.toml",
        PYTHON / "uv.lock",
        ROOT / "devenv.lock",
        ROOT / "src/lean/build/certification.json",
    ]
    catalogue = json.loads((PYTHON / "apps/studies/catalogue.json").read_text())
    owners = catalogue["methods"] + [
        variant
        for procedure in catalogue.get("scientific_procedures", [])
        for variant in procedure["variants"]
    ]
    package = (PYTHON / "packages/statadic").resolve()
    reference_tests = {
        node.split("::", 1)[0]
        for owner in owners
        for mapping in owner.get("reference_mappings", [])
        for node in mapping["nodeids"]
    }
    for module in sorted(reference_tests):
        source = (package / module).resolve()
        if not source.is_relative_to(package) or not source.is_file():
            raise ValueError(f"Invalid reference-test source: {module}")
        inputs.append(source)
    inputs.extend(
        p
        for p in DOCS.rglob("*")
        if p.is_file() and not {"__pycache__", "__marimo__"}.intersection(p.parts)
    )
    for directory in (
        APP / "scripts",
        APP / "src",
        PYTHON / "packages/statadic/src",
        PYTHON / "packages/statadic/examples",
    ):
        inputs.extend(directory.rglob("*.py"))
    for directory in (
        APP / "static",
        APP / "templates",
        APP / "data",
        APP / "evidence",
    ):
        inputs.extend(p for p in directory.rglob("*") if p.is_file())
    inputs.append(ROOT / "src/latex/refs.bib")
    inputs.extend((ROOT / "src/lean/registry").glob("*.json"))
    for directory in (ROOT / "src/lean/packages", ROOT / "src/lean/Example"):
        inputs.extend(directory.rglob("*.lean"))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "schema_version": 1,
        "origin_revision": revision,
        "build_revision": revision,
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        "built_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "statadic",
                "jax",
                "jaxlib",
                "numpy",
                "marimo",
                "sphinx",
                "shibuya",
            )
        },
        "backend_request": os.environ.get("JAX_PLATFORMS", "JAX default"),
        "x64_requested": os.environ.get("JAX_ENABLE_X64", "JAX default (false)"),
        "runtime": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64_enabled": jax.config.x64_enabled,
        },
        "delivery": (
            "Executed static snapshots; local marimo provides interactive computation"
        ),
        "release_evidence": release_evidence,
        "study_exports": [
            {
                "path": str(path),
                "manifest_sha256": hashlib.sha256(
                    (path / "evidence-manifest.json").read_bytes()
                ).hexdigest(),
            }
            for path in study_exports
        ],
        "input_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(inputs))
        },
    }


def validate_notebook_page(notebook: Path, session: dict) -> None:
    """Check publication requirements using the already executed export."""
    from statadic_docs_pages.marimo_pages import compose

    markdown = compose(session)
    for required in ("# ", "## Introduction", "{marimo-output}", "{marimo-launch}"):
        if required not in markdown:
            raise ValueError(f"{notebook}: rendered notebook is missing {required!r}")
    for hidden in ("## Appendix", "def test_"):
        if hidden in markdown:
            raise ValueError(f"{notebook}: rendered notebook exposes {hidden!r}")


def main() -> None:
    selected = study_selection()
    for checker in ("check_schema.py", "check_prose.py"):
        subprocess.run([sys.executable, str(APP / "scripts" / checker)], check=True)
    if selected.fast:
        reused = (
            ROOT / "src/lean/build/certification.json",
            ROOT / "src/lean/blueprint/web",
            ROOT / "src/lean/blueprint/print/print.pdf",
        )
        missing = [str(path) for path in reused if not path.exists()]
        if missing:
            sys.exit(
                "--fast reuses the existing Lean certification and blueprint, but "
                f"these are missing: {', '.join(missing)}. Run the build once "
                "without --fast, or `just lean::certify` and "
                "`just lean::blueprint-pdf`."
            )
    else:
        subprocess.run(
            [
                "just",
                "--justfile",
                str(PYTHON.parent.parent / "justfile"),
                "lean::certify",
            ],
            check=True,
        )
        subprocess.run(
            [
                "just",
                "--justfile",
                str(PYTHON.parent.parent / "justfile"),
                "lean::blueprint-pdf",
            ],
            check=True,
        )
    OUTPUT.mkdir(exist_ok=True)
    # Replace generated HTML and theme assets with the current build.
    if HTML.exists():
        shutil.rmtree(HTML)
    HTML.mkdir(exist_ok=True)
    selection = {
        "study_exports": selected.study_export,
        "release_evidence": selected.release_evidence,
    }
    metadata = provenance(**selection)
    sphinx_environment = os.environ | {
        "STATADIC_DOCS_STUDY_EXPORTS": json.dumps(
            [str(path) for path in selected.study_export]
        ),
        "STATADIC_DOCS_RELEASE_EVIDENCE": "1" if selected.release_evidence else "0",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-c",
            str(APP),
            "-n",
            "-W",
            "--keep-going",
            str(DOCS),
            str(HTML),
        ],
        check=True,
        env=sphinx_environment,
    )
    # The public export already contains executed prose, including text built
    # indirectly from variables that source-only Markdown cannot reveal.
    import check_prose
    from statadic_docs_pages.marimo_pages import execute

    prose_failures = []
    for notebook in check_prose.check_schema.discover(DOCS):
        if notebook.suffix == ".py":
            session = execute(notebook, OUTPUT / "notebook-cache")
            validate_notebook_page(notebook, session)
            problems = check_prose.check_executed_prose(notebook, session)
            prose_failures.extend((notebook, problem) for problem in problems)
    if prose_failures:
        for notebook, problem in prose_failures:
            print(
                f"{notebook}: cell {problem['cell']}: "
                f"{problem['rule']}: {problem['sentence']}"
            )
        raise ValueError("Executed notebook prose violates the documentation style")
    (HTML / "build-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    current = provenance(**selection)
    if any(metadata[key] != current[key] for key in ("input_sha256", "study_exports")):
        raise ValueError("Documentation inputs changed during this build")
    subprocess.run(
        [sys.executable, str(ROOT / "src/lean/tools/bridge.py"), "check"],
        check=True,
    )


if __name__ == "__main__":
    main()
