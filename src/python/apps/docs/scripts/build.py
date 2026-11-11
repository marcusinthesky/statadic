"""Build the site and static native-marimo snapshots into ignored output."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
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
    labs = json.loads((APP / "labs.json").read_text())
    inputs = [
        APP / "labs.json",
        PYTHON / "apps/studies/catalogue.json",
        PYTHON / "apps/performance/src/statadic_performance/identities.json",
        APP / "conf.py",
        APP / "pyproject.toml",
        PYTHON / "uv.lock",
        ROOT / "devenv.lock",
        ROOT / "src/lean/build/certification.json",
    ]
    inputs.extend(APP / "notebooks" / lab["notebook"] for lab in labs["labs"])
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
    inputs.extend(p for p in DOCS.rglob("*") if p.is_file())
    for directory in (
        APP / "scripts",
        APP / "src",
        PYTHON / "packages/statadic/src",
        PYTHON / "packages/statadic/examples",
    ):
        inputs.extend(directory.rglob("*.py"))
    for directory in (APP / "static", APP / "templates"):
        inputs.extend(p for p in directory.rglob("*") if p.is_file())
    inputs.append(ROOT / "src/latex/refs.bib")
    inputs.extend((ROOT / "src/lean/registry").glob("*.json"))
    for directory in (ROOT / "src/lean/packages", ROOT / "src/lean/Example"):
        inputs.extend(directory.rglob("*.lean"))
    return {
        "schema_version": 1,
        "origin_revision": labs["origin_revision"],
        "build_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.hrefs.update(value for key, value in attrs if key == "href" and value)


def main() -> None:
    selected = study_selection()
    subprocess.run(
        ["just", "--justfile", str(PYTHON.parent.parent / "justfile"), "lean::certify"],
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
    # Start from fresh generated HTML, rather than retaining a previous theme's assets.
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
    snapshots = HTML / "labs/snapshots"
    downloads = HTML / "labs/downloads"
    snapshots.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    (HTML / "build-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = os.environ | {"STATADIC_LABS_SNAPSHOT": "1", "MPLBACKEND": "Agg"}
    for lab in json.loads((APP / "labs.json").read_text())["labs"]:
        source = APP / "notebooks" / lab["notebook"]
        slug = lab["slug"]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                str(source),
                "-o",
                str(snapshots / f"{slug}.html"),
                "--no-include-code",
                "--force",
            ],
            env=env,
            check=True,
        )
        shutil.copy2(source, downloads / f"{slug}.py")
        (snapshots / f"{slug}.provenance.json").write_text(
            json.dumps(metadata | {"lab": slug}, indent=2) + "\n"
        )
        page = HTML / "labs" / f"{slug}.html"
        parser = Links()
        parser.feed(page.read_text())
        for relative in (f"snapshots/{slug}.html", f"downloads/{slug}.py"):
            if relative not in parser.hrefs or not (page.parent / relative).is_file():
                raise ValueError(f"Missing lab link or output: {page}: {relative}")
    current = provenance(**selection)
    if any(metadata[key] != current[key] for key in ("input_sha256", "study_exports")):
        raise ValueError("Documentation inputs changed during this build")
    subprocess.run(
        [sys.executable, str(ROOT / "src/lean/tools/bridge.py"), "check"],
        check=True,
    )


if __name__ == "__main__":
    main()
