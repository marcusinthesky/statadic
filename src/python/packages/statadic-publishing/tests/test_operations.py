"""Build, inspect, and deliberately damage a real evidence bundle."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import pytest
from statadic_publishing import archive, build, verify
from statadic_publishing.operations import _run
from statadic_publishing.project import read_project


def _copy(source: Path, target: Path) -> Path:
    return Path(shutil.copytree(source, target))


def _failed_build(
    project: Path, output: Path, error: type[Exception], stage: str
) -> Path:
    with pytest.raises(error):
        build(project, output)
    assert (output / "FAILED").read_text().strip() == stage
    assert not (output / "manifest.json").exists()
    return output


@pytest.mark.parametrize("enabled", [False, True])
def test_child_uses_effective_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setenv("JAX_ENABLE_X64", "0" if enabled else "1")
    log = tmp_path / "child.log"
    with jax.enable_x64(new_val=enabled):
        _run(
            [sys.executable, "-I", "-c", "import jax; print(jax.config.x64_enabled)"],
            tmp_path,
            log,
        )
    assert log.read_text().strip() == str(enabled)


def test_build_verifies_and_archives_deterministically(
    project: Path, bundle: Path, tmp_path: Path
) -> None:
    assert verify(bundle, project).valid
    first, second = tmp_path / "first.tar", tmp_path / "second.tar"
    archive(bundle, first)
    archive(bundle, second)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        archive(bundle, first)
    with pytest.raises(ValueError, match="new or empty"):
        build(project, bundle)


def test_verify_reports_changed_input(
    project: Path, bundle: Path, tmp_path: Path
) -> None:
    changed = _copy(project, tmp_path / "source")
    data = changed / "data" / "synthetic.csv"
    data.write_bytes(data.read_bytes() + b"\n")
    assert "input changed: synthetic-data" in verify(bundle, changed).failures


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {
            "outputs": [
                {
                    "id": "x",
                    "media_type": "text/plain",
                    "path": "../escape",
                    "sha256": "x",
                    "status": "available",
                }
            ]
        },
    ],
)
def test_verify_rejects_invalid_manifest(
    project: Path, bundle: Path, tmp_path: Path, changes: dict[str, object]
) -> None:
    damaged = _copy(bundle, tmp_path / "bundle")
    manifest = damaged / "manifest.json"
    document = json.loads(manifest.read_text())
    document.update(changes)
    manifest.write_text(json.dumps(document))
    assert not verify(damaged, project).valid


def test_verify_reports_every_output_failure(
    project: Path, bundle: Path, tmp_path: Path
) -> None:
    damaged = _copy(bundle, tmp_path / "bundle")
    manifest = damaged / "manifest.json"
    document = json.loads(manifest.read_text())
    document["outputs"][0].update(status="unavailable", sha256=None)
    manifest.write_text(json.dumps(document))
    (damaged / "generated" / "figure.tex").unlink()
    (damaged / "generated" / "table.tex").write_text("tampered")
    failures = verify(damaged, project).failures
    assert any("required output unavailable" in failure for failure in failures)
    assert any("output figure:" in failure for failure in failures)
    assert "output changed: table" in failures


def test_failed_analysis_keeps_its_log(project: Path, tmp_path: Path) -> None:
    source = _copy(project, tmp_path / "source")
    (source / "analysis.py").write_text(
        'raise RuntimeError("analysis failure witness")\n'
    )
    failed = _failed_build(
        source, tmp_path / "failed", subprocess.CalledProcessError, "analysis"
    )
    assert "analysis failure witness" in (failed / "logs" / "analysis.log").read_text()


def test_missing_required_output_fails_analysis(project: Path, tmp_path: Path) -> None:
    source = _copy(project, tmp_path / "source")
    with (source / "publishing.toml").open("a") as stream:
        stream.write(
            '\n[[outputs]]\nid = "missing"\npath = "missing.txt"\n'
            'media_type = "text/plain"\nrequired = true\n'
        )
    failed = _failed_build(source, tmp_path / "failed", ValueError, "analysis")
    assert "required output missing" in (failed / "logs" / "analysis.log").read_text()


def test_analysis_requires_one_local_input(project: Path, tmp_path: Path) -> None:
    source = _copy(project, tmp_path / "source")
    spec = read_project(source / "publishing.toml")
    with (source / "publishing.toml").open("a") as stream:
        stream.write(
            f'\n[[inputs]]\nid = "copy"\npath = "{spec.inputs[0].path}"\n'
            f'sha256 = "{spec.inputs[0].sha256}"\nredistributable = true\n'
        )
    failed = _failed_build(source, tmp_path / "failed", ValueError, "analysis")
    assert "exactly one local input" in (failed / "logs" / "analysis.log").read_text()


def test_claim_citations_must_be_in_bibliography(project: Path, tmp_path: Path) -> None:
    source = _copy(project, tmp_path / "source")
    config = source / "publishing.toml"
    config.write_text(config.read_text().replace('"white-1980"', '"missing-key"'))
    failed = _failed_build(source, tmp_path / "failed", ValueError, "stage")
    assert "missing-key" in (failed / "logs" / "stage.log").read_text()


def test_missing_manuscript_citation_fails_manuscript(
    project: Path, tmp_path: Path
) -> None:
    source = _copy(project, tmp_path / "source")
    manuscript = source / "manuscript" / "main.tex"
    manuscript.write_text(
        manuscript.read_text().replace("white-1980", "missing-citation")
    )
    failed = _failed_build(
        source, tmp_path / "failed", subprocess.CalledProcessError, "manuscript"
    )
    assert "citation" in (failed / "logs" / "manuscript.log").read_text().lower()
