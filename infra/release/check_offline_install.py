"""Prepare and verify a source-free install from the reviewed artifact lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "src/python"
PACKAGES = ("statadic", "statadic-evidence", "statadic-publishing")


def build_options(
    uv: str, python: Path, lock: Path, environment: dict[str, str]
) -> list[str]:
    """Build the reviewed source-only reference with existing locked backends."""
    packages = tomllib.loads(lock.read_text())["packages"]
    source_only = {item["name"] for item in packages if not item.get("wheels")}
    if not source_only:
        return ["--only-binary", ":all:"]
    if source_only != {"autograd-gamma"}:
        raise ValueError(
            f"Review the build backend for source-only dependencies: {sorted(source_only)}"
        )
    # Its hash-verified 0.5.0 sdist uses only setuptools.setup. Avoid an
    # independent build-isolation resolution by bootstrapping locked build tools.
    workspace = tomllib.loads((WORKSPACE / "uv.lock").read_text())
    requirements = lock.parent / "source-build-requirements.txt"
    lines = []
    for name in ("setuptools", "wheel", "packaging"):
        package = next(item for item in workspace["package"] if item["name"] == name)
        artifact = next(
            item
            for item in package["wheels"]
            if item["url"].endswith("py3-none-any.whl")
        )
        lines.append(f"{name} @ {artifact['url']} --hash={artifact['hash']}")
    requirements.write_text("\n".join(lines) + "\n")
    run(
        [
            uv,
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--require-hashes",
            "--python",
            str(python),
            "-r",
            str(requirements),
        ],
        lock.parent,
        environment,
    )
    return ["--no-build-isolation"]


def export_lock(
    uv: str, selection: str, output: Path, environment: dict[str, str]
) -> Path:
    """Retain artifact addresses rather than relying on cached index metadata."""
    output.mkdir(parents=True, exist_ok=True)
    lock = output / f"pylock.{selection}.toml"
    options = ["--package", "statadic-publishing", "--no-dev"]
    if selection != "publishing":
        options = ["--package", "statadic"]
        options += ["--group", "reference"] if selection == "tests" else ["--no-dev"]
        if selection in ("data", "tests"):
            options += ["--extra", "data"]
        options += ["--no-emit-package", "pytest-layout-enforcer"]
    run(
        [
            uv,
            "export",
            "--project",
            str(WORKSPACE),
            "--offline",
            "--frozen",
            *options,
            "--no-emit-workspace",
            "--format",
            "pylock.toml",
            "--output-file",
            str(lock),
            "--quiet",
        ],
        ROOT,
        environment,
    )
    return lock


def prepare_lock(
    uv: str, lock: Path, directory: Path, environment: dict[str, str]
) -> None:
    """Fetch only reviewed artifacts, then prove a second install works offline."""
    for name, offline in (("prepared", False), ("offline-probe", True)):
        target = directory / name
        execution_environment = environment | {"UV_OFFLINE": str(offline).lower()}
        run(
            [uv, "venv", "--python", sys.executable, str(target)],
            directory,
            environment,
        )
        run(
            [
                uv,
                "pip",
                "sync",
                *(["--offline"] if offline else []),
                "--no-index",
                *build_options(uv, target / "bin/python", lock, execution_environment),
                "--python",
                str(target / "bin/python"),
                str(lock),
            ],
            directory,
            execution_environment,
        )
        shutil.rmtree(target)


def run(command: list[str], directory: Path, environment: dict[str, str]) -> None:
    """Keep the exact failing command and installer diagnostics in the CI log."""
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=directory, env=environment, check=True)


def install_locked_environment(
    uv: str,
    python: Path,
    lock: Path,
    wheels: list[Path],
    directory: Path,
    environment: dict[str, str],
) -> None:
    """Install explicit locked artifacts, then validate fresh wheel metadata."""
    run(
        [
            uv,
            "pip",
            "sync",
            "--offline",
            "--no-index",
            *build_options(uv, python, lock, environment | {"UV_OFFLINE": "true"}),
            "--python",
            str(python),
            str(lock),
        ],
        directory,
        environment,
    )
    run(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--python",
            str(python),
            *map(str, wheels),
        ],
        directory,
        environment,
    )
    # --no-deps must not conceal a newly missing or incompatible dependency.
    run([uv, "pip", "check", "--python", str(python)], directory, environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare locked runtime/build inputs before verifying the install offline.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/ci/offline-install"
    )
    parser.add_argument("--python", type=Path, help="An existing isolated environment.")
    parser.add_argument("--lock", type=Path, help="An already exported artifact lock.")
    parser.add_argument(
        "--wheels", type=Path, nargs="+", help="Fresh workspace wheels."
    )
    parser.add_argument(
        "--selection",
        choices=("publishing", "runtime", "data", "tests"),
        default="publishing",
    )
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required; enter the pinned devenv shell")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "offline-install.json"
    receipt.unlink(missing_ok=True)
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    lock = (
        args.lock.resolve()
        if args.lock
        else export_lock(uv, args.selection, output, environment)
    )
    with tempfile.TemporaryDirectory(prefix="statadic-offline-install-") as temporary:
        directory = Path(temporary)
        if args.prepare:
            prepare_lock(uv, lock, directory, environment)
        wheels = [path.resolve() for path in args.wheels or []]
        build_constraints = None
        if not wheels:
            # The workspace dev group owns the backend and its transitive pins.
            # Export those constraints instead of resolving an ambient hatchling.
            build_constraints = output / "build-constraints.txt"
            run(
                [
                    uv,
                    "export",
                    "--project",
                    str(WORKSPACE),
                    "--offline",
                    "--frozen",
                    "--only-dev",
                    "--no-emit-workspace",
                    "--format",
                    "requirements.txt",
                    "--output-file",
                    str(build_constraints),
                    "--quiet",
                ],
                WORKSPACE,
                environment,
            )
            built = directory / "wheels"
            for package in (
                PACKAGES if args.selection == "publishing" else ("statadic",)
            ):
                run(
                    [
                        uv,
                        "build",
                        "--project",
                        str(WORKSPACE),
                        *([] if args.prepare else ["--offline"]),
                        "--build-constraints",
                        str(build_constraints),
                        "--require-hashes",
                        "--wheel",
                        "--package",
                        package,
                        "--out-dir",
                        str(built),
                    ],
                    WORKSPACE,
                    environment | {"UV_OFFLINE": "false"}
                    if args.prepare
                    else environment,
                )
            wheels = sorted(built.glob("*.whl"))
        if args.python:
            python = args.python.absolute()
        else:
            target = directory / "offline"
            run(
                [uv, "venv", "--python", sys.executable, str(target)],
                directory,
                environment,
            )
            python = target / "bin/python"
        install_locked_environment(uv, python, lock, wheels, directory, environment)
        names = (
            (
                "statadic",
                "statadic_evidence",
                "statadic_publishing",
                "jax",
                "numpy",
                "polars",
                "pyarrow",
            )
            if args.selection == "publishing"
            else ("statadic", "jax", "numpy")
        )
        run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "import importlib, sys; from pathlib import Path; "
                    f"names = {names!r}; "
                    "modules = [importlib.import_module(name) for name in names]; "
                    "assert all(Path(module.__file__).resolve().is_relative_to("
                    "Path(sys.prefix).resolve()) for module in modules); "
                    "print('Isolated offline imports passed:', ', '.join(names))"
                ),
            ],
            directory,
            environment,
        )
        receipt.write_text(
            json.dumps(
                {
                    "architecture": platform.machine(),
                    "python": platform.python_version(),
                    "lock_sha256": hashlib.sha256(
                        (WORKSPACE / "uv.lock").read_bytes()
                    ).hexdigest(),
                    "pylock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                    "build_constraints_sha256": (
                        hashlib.sha256(build_constraints.read_bytes()).hexdigest()
                        if build_constraints is not None
                        else None
                    ),
                    "wheels": {
                        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in wheels
                    },
                    "offline_install": "passed",
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
