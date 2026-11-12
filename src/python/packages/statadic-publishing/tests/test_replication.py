"""Reconstruct the synthetic evidence with freshly built, isolated wheels."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from statadic.api.artifacts import load_analysis
from statadic_evidence import extract_archive, sha256
from statadic_publishing import archive, build, verify
from statadic_publishing.manifest import read_manifest
from statadic_publishing.operations import _stage
from statadic_publishing.project import read_project

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent.parent
EXAMPLE = PACKAGE / "examples" / "ols-hc3"


def assert_inference_available(bundle: Path) -> None:
    rows = load_analysis(bundle / "generated/inference-snapshot").report.parameters
    assert all(row.valid and row.std_error is not None for row in rows)


def test_fresh_wheel_replication(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _stage(EXAMPLE, source, read_project(EXAMPLE / "publishing.toml"))
    config = source / "publishing.toml"
    with config.open("a") as stream:
        stream.write(
            '\n[[outputs]]\nid = "optional"\npath = "optional.txt"\n'
            'media_type = "text/plain"\nrequired = false\n'
        )
    original = tmp_path / "original"
    build(source, original)
    assert verify(original, source).valid
    assert (
        next(
            item
            for item in read_manifest(original / "manifest.json").outputs
            if item.id == "optional"
        ).status
        == "unavailable"
    )
    assert_inference_available(original)
    archive_path = tmp_path / "bundle.tar"
    archive(original, archive_path)
    extracted = tmp_path / "extracted"
    extract_archive(archive_path, extracted)
    assert_inference_available(extracted)

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    uv = shutil.which("uv")
    assert uv is not None
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    for package in ("statadic", "statadic-evidence", "statadic-publishing"):
        subprocess.run(
            [
                uv,
                "build",
                "--offline",
                "--wheel",
                "--package",
                package,
                "--out-dir",
                str(wheels),
            ],
            cwd=WORKSPACE,
            env=env,
            check=True,
        )
    venv = tmp_path / "venv"
    subprocess.run(
        [uv, "venv", "--python", sys.executable, str(venv)],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    python = venv / "bin/python"
    # Install the lock's artifact URLs/hashes without resolving registry
    # candidates from incidental uv cache metadata. Keep the consumer isolated.
    subprocess.run(
        [
            sys.executable,
            str(WORKSPACE.parents[1] / "infra/release/check_offline_install.py"),
            "--python",
            str(python),
            "--output",
            str(tmp_path / "install"),
            "--wheels",
            *map(str, sorted(wheels.glob("*.whl"))),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    imported = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            "import statadic, statadic_evidence, statadic_publishing; "
            "print(statadic.__file__); print(statadic_evidence.__file__); "
            "print(statadic_publishing.__file__)",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
    )
    assert all(
        str(venv) in line and str(WORKSPACE) not in line
        for line in imported.splitlines()
    )
    rebuilt = tmp_path / "rebuilt"
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "statadic_publishing",
            "build",
            str(extracted / "project"),
            str(rebuilt),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "statadic_publishing",
            "verify",
            str(rebuilt),
            str(extracted / "project"),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    original_files = {
        p.relative_to(original / "generated")
        for p in (original / "generated").rglob("*")
        if p.is_file()
    }
    rebuilt_files = {
        p.relative_to(rebuilt / "generated")
        for p in (rebuilt / "generated").rglob("*")
        if p.is_file()
    }
    assert original_files == rebuilt_files
    assert {str(p): sha256(original / "generated" / p) for p in original_files} == {
        str(p): sha256(rebuilt / "generated" / p) for p in rebuilt_files
    }
    assert (rebuilt / "manuscript/main.pdf").is_file()
    assert (
        next(
            item
            for item in read_manifest(rebuilt / "manifest.json").outputs
            if item.id == "optional"
        ).status
        == "unavailable"
    )
    assert_inference_available(rebuilt)
