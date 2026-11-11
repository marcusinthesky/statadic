"""Locked installed preparation never restores editable workspace dependencies."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from statadic_studies import installation


@pytest.mark.parametrize(
    "flag", ["--repo", "--w", "--repo=/tmp/other", "--w=/tmp/other"]
)
def test_suite_rejects_abbreviated_frozen_inputs(monkeypatch, tmp_path, flag):
    from statadic_studies import cli

    arguments = [flag] if "=" in flag else [flag, "/tmp/other"]
    monkeypatch.setattr(sys, "argv", ["studies", "suite", str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as rejected:
        cli.main()
    assert rejected.value.code == 2


def test_preparation_rejects_changed_retained_requirements(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1 --hash=sha256:abc\n")
    payload = {
        "schema_version": 1,
        "files": {"requirements.txt": installation._hash(requirements)},
        "installation": {"versions": {"example": "1"}},
    }
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            {
                "id": installation._digest(payload),
                "payload": payload,
            }
        )
    )
    monkeypatch.setattr(installation, "_probe", lambda _: payload["installation"])
    assert installation.verify(tmp_path) == payload
    requirements.write_text("example==2\n")
    with pytest.raises(ValueError, match="file changed"):
        installation.verify(tmp_path)


def test_preparation_rejects_changed_installed_dependency(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("a==1")
    payload = {
        "schema_version": 1,
        "files": {
            "requirements.txt": installation._hash(tmp_path / "requirements.txt")
        },
        "installation": {"versions": {"a": "1"}},
    }
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            {
                "id": installation._digest(payload),
                "payload": payload,
            }
        )
    )
    monkeypatch.setattr(installation, "_probe", lambda _: {"versions": {"a": "2"}})
    with pytest.raises(ValueError, match="versions or imports changed"):
        installation.verify(tmp_path)


def test_invoke_uses_retained_isolated_interpreter(tmp_path, monkeypatch):
    payload = {"x64": "1"}
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            {
                "id": installation._digest(payload),
                "payload": payload,
            }
        )
    )
    calls = []
    monkeypatch.setenv("PYTHONPATH", "/unwanted/editable")
    monkeypatch.setenv("UV_PYTHON", "/unwanted/python")
    monkeypatch.setattr(
        installation.subprocess, "run", lambda *a, **k: calls.append((a, k))
    )
    installation.invoke(tmp_path, tmp_path / "suite", "resume", [])
    command = calls[0][0][0]
    assert command[:4] == [
        str(tmp_path / "venv/bin/python"),
        "-I",
        "-m",
        "statadic_studies.installation",
    ]
    assert command[-1] == "resume"
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert "UV_PYTHON" not in calls[0][1]["env"]
    assert calls[0][1]["env"]["JAX_ENABLE_X64"] == "1"


@pytest.mark.parametrize("mutate_source", [False, True])
def test_prepare_builds_offline_hash_locked_noneditable_closure(
    tmp_path, monkeypatch, mutate_source
):
    repository = tmp_path / "checkout"
    project = repository / "src/python"
    for name in ("uv.lock", "pyproject.toml", "apps/studies/catalogue.json"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    inputs = {
        "apps/docs/labs.json": json.dumps(
            {"labs": [{"slug": "example", "notebook": "example.py"}]}
        ),
        "apps/docs/notebooks/example.py": "# frozen notebook\n",
        "packages/statadic/docs/labs/example.md": "# frozen lab\n",
        "packages/statadic/docs/reference/performance-workloads.md": "# benchmarks\n",
        "apps/performance/src/statadic_performance/identities.json": "{}",
    }
    for name, content in inputs.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for path in installation.PACKAGES.values():
        target = project / path / "pyproject.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("[project]\n")
    for name in ("public-api.json", "methods.json"):
        target = repository / "src/lean/registry" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    nested = project / "apps/studies/src/statadic_studies/nested/example.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("value = 1\n")
    calls = []

    def command(arguments, root, x64):
        calls.append(arguments)
        if arguments[1] == "build":
            (root / "wheels/.gitignore").write_text("*\n")
            package = arguments[arguments.index("--package") + 1]
            if mutate_source and package == "statadic-evidence":
                nested.write_text("value = 2\n")
            (
                root / "wheels" / f"{package.replace('-', '_')}-0.1-py3-none-any.whl"
            ).write_bytes(b"test")
        if arguments[1] == "export":
            (root / "requirements.txt").write_text("numpy==2.4 --hash=sha256:abc\n")

    monkeypatch.setattr(installation, "_command", command)
    monkeypatch.setattr(installation.subprocess, "check_output", lambda *a, **k: "{}")
    root = tmp_path / "prepared"
    if mutate_source:
        with pytest.raises(ValueError, match="build inputs changed"):
            installation.prepare(root, repository, Path("/pinned/uv"))
        assert not (root / "preparation.json").exists()
        return
    record = installation.prepare(root, repository, Path("/pinned/uv"))
    builds = [call for call in calls if call[1] == "build"]
    assert len(builds) == 3
    statadic_build = next(
        call for call in builds if call[call.index("--package") + 1] == "statadic"
    )
    assert "--sdist" in statadic_build and "--wheel" in statadic_build
    assert record["payload"]["build_policy"]["source_date_epoch"] == "315532800"
    assert all(
        "--offline" in call and "--no-build-isolation" in call for call in builds
    )
    assert all(
        call[call.index("--python") + 1] == installation.sys.executable
        for call in builds
    )
    export = next(call for call in calls if call[1] == "export")
    assert {"--frozen", "--no-emit-workspace", "--no-emit-local"} <= set(export)
    assert any("--require-hashes" in call for call in calls)
    assert any("--no-deps" in call for call in calls)
    assert "repository/src/python/uv.lock" in record["payload"]["files"]
    for name in inputs:
        assert f"repository/src/python/{name}" in record["payload"]["files"]
        assert (root / "repository/src/python" / name).read_text() == inputs[name]
    assert "wheels/.gitignore" in record["payload"]["files"]
    with pytest.raises(FileExistsError):
        installation.prepare(root, repository, Path("/pinned/uv"))


def test_failed_command_preserves_logs_without_completion_manifest(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        installation._command(
            [installation.sys.executable, "-c", "raise SystemExit(1)"], tmp_path, "0"
        )
    assert (tmp_path / "preparation.log").is_file()
    assert not (tmp_path / "preparation.json").exists()


def test_wheel_identity_rejects_changed_payload_and_editable(tmp_path, monkeypatch):
    import zipfile

    site = tmp_path / "venv/lib/site-packages"
    source = site / "statadic/example.py"
    metadata = site / "statadic-0.1.dist-info/METADATA"
    source.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    metadata.write_text("Name: statadic\nVersion: 0.1\n")
    wheel = tmp_path / "statadic-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("statadic/example.py", source.read_bytes())
        archive.writestr("statadic-0.1.dist-info/METADATA", metadata.read_bytes())

    class Distribution:
        editable = False

        def read_text(self, name):
            return json.dumps({"dir_info": {"editable": self.editable}})

        def locate_file(self, name):
            return site / name

    distribution = Distribution()
    monkeypatch.setattr(
        installation.importlib.metadata, "distribution", lambda _: distribution
    )
    installation._wheel_identity(wheel, tmp_path / "venv")
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="payload differs"):
        installation._wheel_identity(wheel, tmp_path / "venv")
    distribution.editable = True
    with pytest.raises(ValueError, match="Editable"):
        installation._wheel_identity(wheel, tmp_path / "venv")


def test_build_epoch_overrides_ambient_timestamp(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1580601600")
    environment = installation._environment("1")
    assert environment["SOURCE_DATE_EPOCH"] == "315532800"
    assert environment["JAX_ENABLE_X64"] == "1"
