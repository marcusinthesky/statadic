"""Prepare and invoke a retained, noneditable, locked local campaign environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from statadic_studies.catalogue import tutorial_paths

BUILD_EPOCH = "315532800"  # ZIP epoch used by uv's normalized sdist-to-wheel route.

PACKAGES = {
    "statadic": "packages/statadic",
    "statadic-evidence": "packages/statadic-evidence",
    "statadic-studies": "apps/studies",
}
MODULES = {
    "statadic": "statadic",
    "statadic-evidence": "statadic_evidence",
    "statadic-studies": "statadic_studies",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _environment(x64: str) -> dict:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PYTHON"):
        environment.pop(name, None)
    environment.update(
        JAX_ENABLE_X64=x64, JAX_PLATFORMS="cpu", SOURCE_DATE_EPOCH=BUILD_EPOCH
    )
    return environment


def _command(arguments: list[str], root: Path, x64: str) -> None:
    with (root / "preparation.log").open("a") as output:
        output.write(json.dumps(arguments) + "\n")
        output.flush()
        subprocess.run(
            arguments,
            check=True,
            env=_environment(x64),
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=root,
        )


def _wheel_identity(wheel: Path, prefix: Path) -> None:
    """Check every retained payload against the actual noneditable installation."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise ValueError("Wheel metadata is ambiguous")
        distribution_name = next(
            line[6:]
            for line in archive.read(metadata[0]).decode().splitlines()
            if line.startswith("Name: ")
        )
        distribution = importlib.metadata.distribution(distribution_name)
        direct = distribution.read_text("direct_url.json")
        if direct and json.loads(direct).get("dir_info", {}).get("editable"):
            raise ValueError("Editable installations cannot prepare wheel evidence")
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe wheel member")
            installed = Path(distribution.locate_file(name))
            if (
                not installed.resolve().is_relative_to(prefix.resolve())
                or installed.is_symlink()
                or not installed.is_file()
                or installed.read_bytes() != archive.read(name)
            ):
                raise ValueError(f"Installed wheel payload differs: {name}")


def _probe(root: Path) -> dict:
    prefix = root / "venv"
    if Path(sys.prefix).resolve() != prefix.resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("Use the retained isolated interpreter")
    imports = {}
    for distribution, module_name in MODULES.items():
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix.resolve()):
            raise ValueError(f"Nonisolated import: {module_name}")
        imports[distribution] = str(path.relative_to(prefix))
    for wheel in sorted((root / "wheels").glob("*.whl")):
        _wheel_identity(wheel, prefix)
        with zipfile.ZipFile(wheel) as archive:
            for module_name in MODULES.values():
                expected = {
                    name
                    for name in archive.namelist()
                    if name.startswith(module_name + "/") and name.endswith(".py")
                }
                if expected:
                    module = importlib.import_module(module_name)
                    parent = Path(module.__file__).parent
                    actual = {
                        module_name + "/" + path.relative_to(parent).as_posix()
                        for path in parent.rglob("*.py")
                    }
                    if actual != expected:
                        raise ValueError("Unexpected installed package source files")
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    return {"python": sys.version, "versions": versions, "imports": imports}


def verify(root: Path) -> dict:
    """Verify retained bytes and installed payloads independently of the checkout."""
    record = json.loads((root / "preparation.json").read_text())
    payload = record["payload"]
    if record["id"] != _digest(payload) or payload["schema_version"] != 1:
        raise ValueError("Preparation manifest changed")
    for name, expected in payload["files"].items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.is_file()
            or _hash(path) != expected
        ):
            raise ValueError(f"Retained preparation file changed: {name}")
    retained = {
        path.relative_to(root).as_posix()
        for directory in (root / "repository", root / "wheels")
        for path in directory.rglob("*")
        if path.is_file()
    } | {"requirements.txt"}
    if retained != set(payload["files"]):
        raise ValueError("Retained preparation inventory changed")
    if _probe(root) != payload["installation"]:
        raise ValueError("Installed dependency versions or imports changed")
    return payload


def _build_inputs(project: Path) -> dict:
    """Freeze package/build inputs, including added or removed nested source files."""
    ignored = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".git",
        ".venv",
        "build",
        "dist",
        "site",
        ".marimo",
    }
    return {
        path.relative_to(project).as_posix(): _hash(path)
        for package in PACKAGES.values()
        for path in sorted((project / package).rglob("*"))
        if path.is_file()
        and not ignored.intersection(path.relative_to(project / package).parts)
        and path.suffix != ".pyc"
    }


def prepare(root: Path, repository: Path, uv: Path, *, x64: str = "1") -> dict:
    """Build three wheels offline and install their locked third-party closure."""
    root, repository, uv = root.resolve(), repository.resolve(), uv.resolve()
    if x64 not in {"0", "1"}:
        raise ValueError("Precision must be selected explicitly")
    root.mkdir(parents=True, exist_ok=False)
    project = repository / "src/python"
    snapshot = root / "repository"
    files = [project / "uv.lock", project / "pyproject.toml"]
    files += [project / path / "pyproject.toml" for path in PACKAGES.values()]
    files += [
        repository / "src/lean/registry" / name
        for name in ("public-api.json", "methods.json")
    ]
    files += [project / "apps/studies/catalogue.json"]
    # The isolated suite validates discoverability against its frozen repository.
    # Retain these as data, without importing either documentation or performance.
    catalogue = json.loads((project / "apps/studies/catalogue.json").read_text())
    files += [
        project / "apps/performance/src/statadic_performance/identities.json",
        project / "packages/statadic/docs/reference/performance-workloads.md",
        *tutorial_paths(repository, catalogue).values(),
    ]
    files += sorted((repository / ".context/reference").glob("*/README.md"))
    before = {str(path): _hash(path) for path in files}
    build_inputs = _build_inputs(project)
    for source in files:
        target = snapshot / source.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    wheels = root / "wheels"
    wheels.mkdir()
    for package in PACKAGES:
        _command(
            [
                str(uv),
                "build",
                "--project",
                str(project),
                "--package",
                package,
                *(["--sdist", "--wheel"] if package == "statadic" else ["--wheel"]),
                "--out-dir",
                str(wheels),
                "--offline",
                "--no-build-isolation",
                "--python",
                sys.executable,
            ],
            root,
            x64,
        )
    if build_inputs != _build_inputs(project):
        raise ValueError("Package build inputs changed during wheel builds")
    if len(list(wheels.glob("*.whl"))) != len(PACKAGES):
        raise ValueError("Expected exactly three retained workspace wheels")
    requirements = root / "requirements.txt"
    _command(
        [
            str(uv),
            "export",
            "--project",
            str(project),
            "--package",
            "statadic-studies",
            "--offline",
            "--frozen",
            "--no-dev",
            "--no-emit-workspace",
            "--no-emit-local",
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(requirements),
        ],
        root,
        x64,
    )
    if any(
        line.startswith(("-e ", "./", "../", "file:")) or " @ file:" in line
        for line in requirements.read_text().splitlines()
    ):
        raise ValueError("Requirements must not contain workspace paths")
    if before != {str(path): _hash(path) for path in files}:
        raise ValueError("Lock/catalogue metadata changed during preparation")
    interpreter = root / "venv/bin/python"
    _command(
        [str(uv), "venv", "--offline", "--python", sys.executable, str(root / "venv")],
        root,
        x64,
    )
    _command(
        [
            str(uv),
            "pip",
            "install",
            "--offline",
            "--python",
            str(interpreter),
            "--require-hashes",
            "-r",
            str(requirements),
        ],
        root,
        x64,
    )
    _command(
        [
            str(uv),
            "pip",
            "install",
            "--offline",
            "--python",
            str(interpreter),
            "--no-deps",
            *map(str, sorted(wheels.glob("*.whl"))),
        ],
        root,
        x64,
    )
    _command(
        [str(uv), "pip", "check", "--python", str(interpreter), "--offline"], root, x64
    )
    installation = json.loads(
        subprocess.check_output(
            [
                str(interpreter),
                "-I",
                "-m",
                "statadic_studies.installation",
                "_probe",
                str(root),
            ],
            env=_environment(x64),
            cwd=root,
            text=True,
        )
    )
    retained = sorted(snapshot.rglob("*")) + sorted(wheels.rglob("*")) + [requirements]
    payload = {
        "schema_version": 1,
        "build_inputs": build_inputs,
        "build_policy": {
            "statadic_route": ["--sdist", "--wheel"],
            "source_date_epoch": BUILD_EPOCH,
            "archive_identity": "exact release-route wheel archive bytes",
        },
        "dependency_assurance": (
            "Hash-locked initial third-party install and recorded versions; "
            "later same-version third-party payload changes are not detected"
        ),
        "x64": x64,
        "installation": installation,
        "files": {
            path.relative_to(root).as_posix(): _hash(path)
            for path in retained
            if path.is_file()
        },
        "scope": (
            "noneditable-installed-wheels; frozen catalogue, "
            "not all-method qualification"
        ),
    }
    record = {"id": _digest(payload), "payload": payload}
    # Preparation is single-writer and created exclusively; publish only after success.
    temporary = root / "preparation.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(root / "preparation.json")
    return record


def invoke(root: Path, directory: Path, action: str, arguments: list[str]) -> None:
    """Delegate to the retained runner; current editable packages are never synced."""
    root, directory = root.resolve(), directory.resolve()
    record = json.loads((root / "preparation.json").read_text())
    if record["id"] != _digest(record["payload"]):
        raise ValueError("Preparation manifest changed")
    subprocess.run(
        [
            str(root / "venv/bin/python"),
            "-I",
            "-m",
            "statadic_studies.installation",
            "_launch",
            str(root),
            str(directory),
            action,
            *arguments,
        ],
        check=True,
        env=_environment(record["payload"]["x64"]),
        cwd=root,
    )


def main() -> None:
    """Prepare, run, resume or inspect one retained local installed environment."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "action", choices=("prepare", "run", "resume", "status", "_probe", "_launch")
    )
    parser.add_argument("preparation", type=Path)
    args, extra = parser.parse_known_args()
    root = args.preparation.resolve()
    if args.action == "prepare":
        options = argparse.ArgumentParser(allow_abbrev=False)
        options.add_argument("--repository", type=Path, required=True)
        options.add_argument("--uv", type=Path, required=True)
        options.add_argument("--x64", choices=("0", "1"), default="1")
        selected = options.parse_args(extra)
        print(
            json.dumps(
                prepare(root, selected.repository, selected.uv, x64=selected.x64),
                indent=2,
            )
        )
    elif args.action == "_probe":
        if extra:
            parser.error("Unexpected probe arguments")
        print(json.dumps(_probe(root)))
    elif args.action == "_launch":
        payload = verify(root)
        if len(extra) < 2:
            parser.error("Launch needs suite directory and action")
        directory, action, *arguments = extra
        if action not in {"run", "resume", "status"}:
            parser.error("Unknown installed suite action")
        forbidden = {"--repository", "--wheel", "--x64", "--resume", "--status"}
        if any(argument.split("=", 1)[0] in forbidden for argument in arguments):
            parser.error("Installed runs use frozen repository, wheel and precision")
        command = [
            str(root / "venv/bin/python"),
            "-I",
            "-m",
            "statadic_studies.cli",
            "suite",
            directory,
        ]
        if action == "run":
            wheel = next((root / "wheels").glob("statadic-*.whl"))
            command += [
                "--repository",
                str(root / "repository"),
                "--wheel",
                str(wheel),
                *arguments,
            ]
        else:
            if arguments:
                parser.error("Resume/status consume frozen selections only")
            command += ["--" + action]
        command += ["--x64", payload["x64"]]
        subprocess.run(command, check=True, env=_environment(payload["x64"]), cwd=root)
    else:
        if not extra:
            parser.error("Provide the suite directory")
        invoke(root, Path(extra[0]), args.action, extra[1:])


if __name__ == "__main__":
    main()
