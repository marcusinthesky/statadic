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

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "src/python"
PACKAGES = ("statadic", "statadic-evidence", "statadic-publishing")


def run(command: list[str], directory: Path, environment: dict[str, str]) -> None:
    """Keep the exact failing command and installer diagnostics in the CI log."""
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=directory, env=environment, check=True)


def sync_locked_dependencies(
    uv: str,
    python: Path,
    lock: Path,
    directory: Path,
    environment: dict[str, str],
    *,
    prepare: bool = False,
    build_constraints: Path | None = None,
) -> None:
    """Install locked artifacts, preparing constrained source builds when needed."""
    command = [uv, "pip", "sync"]
    if not prepare:
        command.extend(["--offline", "--no-index"])
    elif build_constraints is None:
        command.append("--no-index")
    if build_constraints is None:
        command.extend(["--only-binary", ":all:"])
    else:
        command.extend(
            ["--build-constraints", str(build_constraints), "--require-hashes"]
        )
    command.extend(["--python", str(python), str(lock)])
    run(
        command,
        directory,
        environment | {"UV_OFFLINE": "false"} if prepare else environment,
    )


def install_locked_environment(
    uv: str,
    python: Path,
    lock: Path,
    wheels: list[Path],
    directory: Path,
    environment: dict[str, str],
    *,
    build_constraints: Path | None = None,
) -> None:
    """Install explicit locked artifacts, then validate fresh wheel metadata."""
    sync_locked_dependencies(
        uv,
        python,
        lock,
        directory,
        environment,
        build_constraints=build_constraints,
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
        "--profile",
        choices=("publishing", "release"),
        default="publishing",
        help="Use publishing runtime dependencies or the complete release harness.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare locked runtime/build inputs before verifying the install offline.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/ci/offline-install"
    )
    parser.add_argument("--python", type=Path, help="An existing isolated environment.")
    parser.add_argument(
        "--wheels", type=Path, nargs="+", help="Fresh workspace wheels."
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
    if args.profile == "release":
        selection = [
            "--package",
            "statadic",
            "--group",
            "reference",
            # The source-free wheel harness needs no Git-only layout tool.
            "--no-emit-package",
            "pytest-layout-enforcer",
        ]
        packages = ("statadic",)
        imports = ("statadic", "jax", "numpy", "lifelines", "linearmodels", "spreg")
    else:
        selection = ["--package", "statadic-publishing", "--no-dev"]
        packages = PACKAGES
        imports = (
            "statadic",
            "statadic_evidence",
            "statadic_publishing",
            "jax",
            "numpy",
            "polars",
            "pyarrow",
        )
    export = [
        uv,
        "export",
        "--project",
        str(WORKSPACE),
        "--offline",
        "--frozen",
        *selection,
        "--no-emit-workspace",
        "--quiet",
    ]
    lock = output / f"pylock.{args.profile}.toml"
    run(
        [*export, "--format", "pylock.toml", "--output-file", str(lock)],
        ROOT,
        environment,
    )
    dependency_build_constraints = None
    if args.profile == "release":
        # The reference closure includes an sdist-only dependency. Its backend
        # must also come from the reviewed lock, with hashes, during preparation.
        dependency_build_constraints = output / "dependency-build-constraints.txt"
        run(
            [
                *export,
                "--format",
                "requirements.txt",
                "--output-file",
                str(dependency_build_constraints),
            ],
            ROOT,
            environment,
        )
    with tempfile.TemporaryDirectory(prefix="statadic-offline-install-") as temporary:
        directory = Path(temporary)
        if args.prepare:
            # Dependency artifacts use the lock's URLs and hashes. Only source
            # build backends need index lookup, constrained to the reviewed pins.
            prepared = directory / "prepared"
            run(
                [uv, "venv", "--python", sys.executable, str(prepared)],
                directory,
                environment,
            )
            sync_locked_dependencies(
                uv,
                prepared / "bin/python",
                lock,
                directory,
                environment,
                prepare=True,
                build_constraints=dependency_build_constraints,
            )
            shutil.rmtree(prepared)
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
            for package in packages:
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
        install_locked_environment(
            uv,
            python,
            lock,
            wheels,
            directory,
            environment,
            build_constraints=dependency_build_constraints,
        )
        run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "import importlib, sys; from pathlib import Path; "
                    f"names = {imports!r}; "
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
                    "profile": args.profile,
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
                    "dependency_build_constraints_sha256": (
                        hashlib.sha256(
                            dependency_build_constraints.read_bytes()
                        ).hexdigest()
                        if dependency_build_constraints is not None
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
