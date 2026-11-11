"""Build and validate local, portable research evidence bundles."""

import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import tomli_w
from statadic_evidence import inventory, sha256, verify_inventory, write_archive

from .manifest import (
    AnalysisRecord,
    Hash,
    InputHash,
    Manifest,
    OutputRecord,
    Runtime,
    Source,
    Versions,
    read_manifest,
    write_manifest,
)
from .project import Project, read_project

_BIB_KEY = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)


@dataclass(frozen=True)
class Bundle:
    """A successfully built evidence bundle."""

    path: Path


@dataclass(frozen=True)
class Verification:
    """Every failure found while verifying a bundle."""

    failures: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether verification found no failures."""
        return not self.failures


def _external(spec: Project) -> list[tuple[str, str]]:
    return [
        (spec.manuscript.bibliography.path, "assets/refs.bib"),
        (spec.manuscript.tex_bundle.path, "assets/tex-bundle.zip"),
        (spec.manuscript.check_log, "assets/check-log"),
        *[
            (path, f"assets/lock-{i}")
            for i, path in enumerate(spec.environment.lockfiles)
        ],
    ]


def _stage(source: Path, target: Path, spec: Project) -> None:
    # Reject links and special files before copying, including hidden sources.
    inventory(source, exclude=())
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    with (target / "publishing.toml").open("rb") as stream:
        document = tomllib.load(stream)
    for original, staged in _external(spec):
        source_file = source / original
        destination = target / staged
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_file.resolve() != destination.resolve():
            shutil.copyfile(source_file, destination)
    manuscript = document["manuscript"]
    manuscript["bibliography"]["path"] = "assets/refs.bib"
    manuscript["tex_bundle"]["path"] = "assets/tex-bundle.zip"
    manuscript["check_log"] = "assets/check-log"
    document["environment"]["lockfiles"] = [
        f"assets/lock-{i}" for i in range(len(spec.environment.lockfiles))
    ]
    (target / "publishing.toml").write_text(tomli_w.dumps(document), encoding="utf-8")


def _run(command: list[str], cwd: Path, log: Path) -> None:
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "TMPDIR",
            "JAX_ENABLE_X64",
            "LD_LIBRARY_PATH",
            "SSL_CERT_FILE",
            "FONTCONFIG_FILE",
            "FONTCONFIG_PATH",
        )
        if key in os.environ
    }
    env["JAX_ENABLE_X64"] = "1" if jax.config.x64_enabled else "0"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _hashes(root: Path) -> tuple[Hash, ...]:
    return tuple(
        Hash(path, digest) for path, digest in inventory(root, exclude=()).items()
    )


def _check_sources(project: Path, spec: Project) -> None:
    """Reject changed inputs or external files, and uncatalogued claim citations."""
    for item in spec.inputs:
        if item.path is not None and sha256(project / item.path) != item.sha256:
            raise ValueError(f"input hash mismatch: {item.id}")
    for pinned in (spec.manuscript.bibliography, spec.manuscript.tex_bundle):
        if sha256(project / pinned.path) != pinned.sha256:
            raise ValueError(f"external file hash mismatch: {pinned.path}")
    bibliography = project / spec.manuscript.bibliography.path
    keys = set(_BIB_KEY.findall(bibliography.read_text(encoding="utf-8")))
    cited = {key for claim in spec.claims for key in claim.citations}
    if cited - keys:
        raise ValueError(f"claim citations not in bibliography: {sorted(cited - keys)}")


def _analyse(staged: Path, spec: Project, generated: Path, log: Path) -> None:
    """Run the analysis on its single local input and require declared outputs."""
    local = [item.path for item in spec.inputs if item.path is not None]
    if len(local) != 1:
        raise ValueError("analysis requires exactly one local input")
    _run(
        [
            sys.executable,
            "-I",
            str(staged / spec.analysis.script),
            "--input",
            str(staged / local[0]),
            "--seed",
            str(spec.analysis.seed),
            "--output",
            str(generated),
        ],
        staged,
        log,
    )
    for item in spec.outputs:
        if item.required and not (generated / item.path).is_file():
            raise ValueError(f"required output missing: {item.id}")


def _compile(
    staged: Path, spec: Project, generated: Path, manuscript: Path, log: Path
) -> None:
    """Compile the manuscript offline against generated outputs and check its log."""
    shutil.copytree(staged / spec.manuscript.root, manuscript)
    shutil.copytree(generated, manuscript / "generated")
    shutil.copyfile(staged / spec.manuscript.bibliography.path, manuscript / "refs.bib")
    _run(
        [
            "tectonic",
            "-X",
            "compile",
            "--untrusted",
            "--only-cached",
            "--bundle",
            str(staged / spec.manuscript.tex_bundle.path),
            "--keep-logs",
            spec.manuscript.main,
        ],
        manuscript,
        log,
    )
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(
            [
                "bash",
                str(staged / spec.manuscript.check_log),
                str(manuscript / Path(spec.manuscript.main).with_suffix(".log")),
            ],
            cwd=manuscript,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _revision(project: Path) -> tuple[str, bool | None]:
    """Return the project's git revision and dirtiness, when git can tell."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=project, capture_output=True, text=True, check=True
        ).stdout

    try:
        return git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable", None


def _manifest(project: Path, output: Path, spec: Project) -> Manifest:
    """Write rebuild instructions, then describe the completed bundle."""
    staged, generated = output / "project", output / "generated"
    versions = Versions(
        platform.python_version(),
        version("statadic"),
        version("jax"),
        version("statadic-publishing"),
    )
    tectonic = subprocess.run(
        ["tectonic", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    x64 = int(jax.config.x64_enabled)
    (output / "REBUILD.md").write_text(
        f"# Rebuild\n\nInstall statadic {versions.statadic}, jax {versions.jax}, "
        f"statadic-evidence {version('statadic-evidence')} and "
        f"statadic-publishing {versions.statadic_publishing} "
        f"with Python {versions.python}, pinning every other dependency to the "
        "staged lockfiles in project/assets, because other releases can change "
        f"floating-point results. Install {tectonic} on PATH. "
        f"The original JAX x64 setting was {x64}.\n\n"
        f"```sh\nJAX_ENABLE_X64={x64} python -I -m statadic_publishing "
        "build project rebuilt\n"
        f"JAX_ENABLE_X64={x64} python -I -m statadic_publishing "
        "verify rebuilt project\n```\n",
        encoding="utf-8",
    )
    (output / "logs" / "manifest.log").write_text(
        "Preparing manifest.\n", encoding="utf-8"
    )
    outputs = []
    for item in spec.outputs:
        path = generated / item.path
        available = path.is_file()
        outputs.append(
            OutputRecord(
                item.id,
                item.media_type,
                f"generated/{item.path}",
                sha256(path) if available else None,
                "available" if available else "unavailable",
            )
        )
    analysis = spec.analysis
    record = AnalysisRecord(
        analysis.script,
        analysis.estimand,
        analysis.model,
        analysis.covariance,
        analysis.reference,
        tuple(item.id for item in spec.inputs),
        analysis.assumptions,
        analysis.snapshots,
        analysis.diagnostics,
    )
    return Manifest(
        1,
        spec.project.id,
        spec.project.id,
        datetime.now(UTC).isoformat(),
        Source(*_revision(project), _hashes(staged)),
        tuple(Hash(path, sha256(staged / path)) for path in spec.environment.lockfiles),
        tuple(
            InputHash(item.id, item.path or item.identity or "", item.sha256)
            for item in spec.inputs
        ),
        versions,
        Runtime(
            platform.platform(),
            jax.default_backend(),
            jax.config.x64_enabled,
            analysis.seed,
        ),
        (record,),
        tuple(outputs),
        spec.claims,
        inventory(output),
    )


def build(project: Path, output: Path) -> Bundle:
    """Stage, analyse, compile and record a project in a new or empty directory.

    A failed stage appends its error to that stage's log and writes a ``FAILED``
    marker; only a successful build writes ``manifest.json``.
    """
    project, output = Path(project).resolve(), Path(output).absolute()
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise ValueError("output must be a new or empty directory")
    if output.resolve().is_relative_to(project):
        raise ValueError("output must be outside the project")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    staged, generated = output / "project", output / "generated"
    stage = "stage"
    try:
        spec = read_project(project / "publishing.toml")
        _check_sources(project, spec)
        _stage(project, staged, spec)
        spec = read_project(staged / "publishing.toml")
        (logs / "stage.log").write_text(
            "Staged and checked inputs.\n", encoding="utf-8"
        )
        stage = "analysis"
        _analyse(staged, spec, generated, logs / "analysis.log")
        stage = "manuscript"
        _compile(
            staged, spec, generated, output / "manuscript", logs / "manuscript.log"
        )
        stage = "manifest"
        write_manifest(output / "manifest.json", _manifest(project, output, spec))
    except Exception as error:
        with (logs / f"{stage}.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{type(error).__name__}: {error}\n")
        (output / "FAILED").write_text(stage + "\n", encoding="utf-8")
        raise
    return Bundle(output)


def verify(bundle: Path, project: Path) -> Verification:
    """Check a bundle's manifest, inventory, sources, inputs and outputs."""
    bundle, project = Path(bundle), Path(project)
    failures = []
    try:
        manifest = read_manifest(bundle / "manifest.json")
    except Exception as error:
        return Verification((f"manifest: {error}",))
    try:
        verify_inventory(bundle, manifest.inventory)
    except Exception as error:
        failures.append(f"inventory: {error}")
    try:
        spec = read_project(project / "publishing.toml")
    except Exception as error:
        failures.append(f"project: {error}")
        spec = None
    if spec is not None:
        try:
            with TemporaryDirectory() as temporary:
                staged = Path(temporary) / "project"
                _stage(project, staged, spec)
                if _hashes(staged) != manifest.source.hashes:
                    failures.append("project sources or external files changed")
        except Exception as error:
            failures.append(f"project sources: {error}")
        for item in spec.inputs:
            if item.path is not None:
                try:
                    if sha256(project / item.path) != item.sha256:
                        failures.append(f"input changed: {item.id}")
                except Exception as error:
                    failures.append(f"input {item.id}: {error}")
        recorded = {item.id: item for item in manifest.outputs}
        for item in spec.outputs:
            found = recorded.get(item.id)
            if found is None:
                failures.append(f"output missing from manifest: {item.id}")
            elif (
                found.path != f"generated/{item.path}"
                or found.media_type != item.media_type
            ):
                failures.append(f"output identity mismatch: {item.id}")
            if item.required and (found is None or found.status != "available"):
                failures.append(f"required output unavailable: {item.id}")
    for item in manifest.outputs:
        if item.status == "available":
            try:
                if sha256(bundle / item.path) != item.sha256:
                    failures.append(f"output changed: {item.id}")
            except Exception as error:
                failures.append(f"output {item.id}: {error}")
    return Verification(tuple(failures))


def archive(bundle: Path, destination: Path) -> Path:
    """Verify a bundle against its staged project, then archive it deterministically."""
    bundle, destination = Path(bundle), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    result = verify(bundle, bundle / "project")
    if not result.valid:
        raise ValueError("; ".join(result.failures))
    write_archive(bundle, destination)
    return destination
