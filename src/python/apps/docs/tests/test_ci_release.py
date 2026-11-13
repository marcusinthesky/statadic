"""Release preflight only reuses complete evidence for the same execution."""

import copy
import json
import runpy
import sys
from pathlib import Path

import pytest
from statadic_evidence import extract_archive, write_archive

ROOT = Path(__file__).resolve().parents[5]
RELEASE = runpy.run_path(str(ROOT / "src/python/apps/docs/scripts/release.py"))


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    """Use the same complete synthetic receipt as the controller tests."""
    monkeypatch.syspath_prepend(str(ROOT / "infra/nix"))
    builder = runpy.run_path(str(ROOT / "infra/nix/test_ci_receipts.py"))
    directory = tmp_path / "input"
    identity = builder["make_receipt"](directory)
    return directory, identity


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")


def _manifest(identity):
    return {
        "mode": "preflight",
        "source_files": copy.deepcopy(identity["source_files"]),
        "revision": identity["revision"],
        "platform": {"machine": identity["architecture"]},
        "lanes": [],
    }


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--mode", "preflight", "--verify", "bundle"],
        ["--mode", "preflight", "--verify-archive", "bundle"],
    ],
)
def test_receipt_only_allowed_when_executing_preflight(monkeypatch, arguments):
    monkeypatch.setattr(
        sys, "argv", ["release", "--preflight-check-receipt", "receipt", *arguments]
    )
    with pytest.raises(SystemExit, match="2"):
        RELEASE["main"]()


@pytest.mark.parametrize(
    "missing", ["GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"]
)
def test_hosted_identity_required(receipt, hosted, monkeypatch, tmp_path, missing):
    monkeypatch.delenv(missing)
    directory, identity = receipt
    with pytest.raises(ValueError, match="GitHub run and attempt"):
        RELEASE["retain_preflight_check"](directory, tmp_path, _manifest(identity))


@pytest.mark.parametrize(
    "field", ["source_files", "revision", "architecture", "run_id", "run_attempt"]
)
def test_receipt_rejects_stale_identity(receipt, hosted, tmp_path, field):
    directory, identity = receipt
    path = directory / "receipt.json"
    record = json.loads(path.read_text())
    record["identity"][field] = (
        {"example.py": "changed"} if field == "source_files" else "changed"
    )
    _write(path, record)
    with pytest.raises(ValueError, match="identity differs"):
        RELEASE["retain_preflight_check"](directory, tmp_path, _manifest(identity))


def test_receipt_retained_and_verified_after_archive(receipt, hosted, tmp_path):
    directory, identity = receipt
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = _manifest(identity)
    RELEASE["retain_preflight_check"](directory, bundle, manifest)
    assert (bundle / "check-receipt/receipt.json").read_bytes() == (
        directory / "receipt.json"
    ).read_bytes()
    archive = tmp_path / "evidence.tar"
    write_archive(bundle, archive)
    extracted = tmp_path / "extracted"
    extract_archive(archive, extracted)
    RELEASE["verify_check_evidence"](extracted, manifest)
    (extracted / "check-receipt/shard-0.json").unlink()
    with pytest.raises((ValueError, OSError)):
        RELEASE["verify_check_evidence"](extracted, manifest)


@pytest.mark.parametrize("damage", ["missing", "tampered", "duplicate_check"])
def test_retained_evidence_rejects_damage(receipt, hosted, tmp_path, damage):
    directory, identity = receipt
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = _manifest(identity)
    RELEASE["retain_preflight_check"](directory, bundle, manifest)
    shard = bundle / "check-receipt/shard-0.json"
    if damage == "missing":
        shard.unlink()
    elif damage == "tampered":
        record = json.loads(shard.read_text())
        record["exitstatus"] = 1
        _write(shard, record)
    else:
        manifest["lanes"].append(copy.deepcopy(manifest["lanes"][0]))
    with pytest.raises((ValueError, OSError)):
        RELEASE["verify_check_evidence"](bundle, manifest)
