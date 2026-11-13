"""Persistent suite scope, interruption, stale provenance and real tiny computation."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from statadic_studies import campaign, suite
from statadic_studies.contracts import STUDIES

REPOSITORY = Path(__file__).resolve().parents[5]
GAPS = {"coverage_status": "incomplete", "missing_study": ["unimplemented"]}


@pytest.fixture
def coverage(monkeypatch):
    monkeypatch.setattr(suite, "check_catalogue", lambda repository: GAPS)


@pytest.fixture
def isolated(monkeypatch, coverage):
    monkeypatch.setattr(
        campaign, "environment", lambda: {"backend": "cpu", "x64": True}
    )


def test_scope_frozen_and_all_children_prepared(tmp_path, isolated):
    root = tmp_path / "suite"
    selected = ["mmd.finite-bias", "sharpe.point-estimation"]
    suite.create(root, REPOSITORY, selected, profile="smoke", trials=1)
    frozen = suite.manifest(root)["payload"]
    assert frozen["coverage"] == GAPS
    assert set(frozen["studies"]) == set(selected)
    assert all(
        (root / "campaigns" / name / "manifest.json").exists() for name in selected
    )
    assert suite.status(root)["all_methods_complete"] is False
    assert suite.status(root)["studies"] == dict.fromkeys(selected, "pending")


@pytest.mark.parametrize("selection", [[], ["missing"], ["mmd.finite-bias"] * 2])
def test_explicit_selection_required(tmp_path, isolated, selection):
    with pytest.raises(ValueError, match="Select unique"):
        suite.create(tmp_path / "suite", REPOSITORY, selection)


def test_evaluation_effort_and_stale_spec_rejected(tmp_path, monkeypatch, isolated):
    with pytest.raises(ValueError, match="10000"):
        suite.create(tmp_path / "small", REPOSITORY, ["mmd.finite-bias"], trials=2)
    root = tmp_path / "suite"
    suite.create(root, REPOSITORY, ["mmd.finite-bias"], profile="smoke", trials=1)
    monkeypatch.setitem(
        STUDIES, "mmd.finite-bias", replace(STUDIES["mmd.finite-bias"], aim="changed")
    )
    with pytest.raises(ValueError, match="definitions changed"):
        suite.run(root)


def test_stale_environment_rejected(tmp_path, monkeypatch, isolated):
    root = tmp_path / "suite"
    suite.create(root, REPOSITORY, ["mmd.finite-bias"], profile="smoke", trials=1)
    monkeypatch.setattr(
        campaign, "environment", lambda: {"backend": "cpu", "x64": False}
    )
    with pytest.raises(ValueError, match="environment"):
        suite.run(root)


def test_interruption_retained_and_incomplete_preparation_resumes(
    tmp_path, monkeypatch, isolated
):
    root = tmp_path / "suite"
    suite.create(root, REPOSITORY, ["mmd.finite-bias"], profile="smoke", trials=1)
    import shutil

    shutil.rmtree(root / "campaigns/mmd.finite-bias")
    child = SimpleNamespace(args=["worker"], poll=lambda: None, terminate=lambda: None)
    child.wait = lambda: (_ for _ in ()).throw(KeyboardInterrupt())

    # The second wait after termination must finish.
    def terminate():
        child.wait = lambda: 0

    child.terminate = terminate
    monkeypatch.setattr(suite.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(KeyboardInterrupt):
        suite.run(root)
    assert (root / "campaigns/mmd.finite-bias/manifest.json").exists()
    assert suite.status(root)["studies"]["mmd.finite-bias"] == "interrupted"


def test_failed_child_preserves_failure_and_pending_trials(
    tmp_path, monkeypatch, isolated
):
    root = tmp_path / "suite"
    suite.create(root, REPOSITORY, ["mmd.finite-bias"], profile="smoke", trials=1)
    monkeypatch.setattr(
        suite.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(args=["worker"], wait=lambda: 2),
    )
    with pytest.raises(suite.subprocess.CalledProcessError):
        suite.run(root)
    assert suite.status(root)["studies"]["mmd.finite-bias"] == "failed"
    assert campaign.verify(root / "campaigns/mmd.finite-bias") == []


def test_real_mmd_smoke_exports_resumes_missing_trial_and_skips_verified(
    tmp_path, monkeypatch, coverage
):
    root = tmp_path / "suite"
    suite.create(root, REPOSITORY, ["mmd.finite-bias"], profile="smoke", trials=1)
    report = suite.run(root)
    assert report["selected_execution_complete"]
    assert not report["all_methods_complete"]
    publication = root / "publications/mmd.finite-bias"
    assert suite.verify_export(publication)["profile"] == "smoke"
    child = root / "campaigns/mmd.finite-bias"
    (child / "trials/null-equal/00000000.json").unlink()
    assert suite.run(root)["selected_execution_complete"]
    assert len(campaign.verify(child)) == 4

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed verified work must not rerun")

    monkeypatch.setattr(suite.subprocess, "Popen", forbidden)
    assert suite.run(root)["selected_execution_complete"]
