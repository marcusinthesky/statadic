"""Independent scientific and campaign witnesses, including incomplete evidence."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
from statadic.testing.qualification import TrialOutcome, summarize_trials, trial_key
from statadic_evidence import extract_archive, write_archive
from statadic_studies import campaign
from statadic_studies.adapters import finite_mmd_truth, rng
from statadic_studies.analysis import export, mean_mcse, summarize, verify_export


def test_execution_identity_binds_numerical_dependency_and_effective_prng(monkeypatch):
    monkeypatch.setattr(campaign.source, "capture", lambda: {})
    first = campaign.environment()
    original = campaign.importlib.metadata.version
    monkeypatch.setattr(
        campaign.importlib.metadata,
        "version",
        lambda name: "changed" if name == "optimistix" else original(name),
    )
    changed = campaign.environment()
    assert first["versions"] != changed["versions"]
    assert changed["versions"]["optimistix"] == "changed"
    with jax.default_prng_impl("rbg"):
        assert campaign.environment()["jax_policy"]["jax_default_prng_impl"] == "rbg"
    assert campaign.environment()["jax_policy"] == first["jax_policy"]


def test_typed_trial_keys_are_independent_of_study_dgp_streams():
    first = jax.random.normal(trial_key(103, 7, stream=2), (20,))
    np.testing.assert_array_equal(
        first, jax.random.normal(trial_key(103, 7, stream=2), (20,))
    )
    assert not np.array_equal(
        first, jax.random.normal(trial_key(103, 7, stream=3), (20,))
    )


def test_changed_threefry_stream_cannot_resume_a_campaign(tmp_path):
    root = tmp_path / "partitionable"
    with jax.threefry_partitionable(True):
        campaign.create(root, "mmd.finite-bias", trials=1, profile="smoke")
        frozen = campaign.environment()
        first = np.asarray(jax.random.normal(jax.random.key(0), (8,)))
    with jax.threefry_partitionable(False):
        current = campaign.environment()
        second = np.asarray(jax.random.normal(jax.random.key(0), (8,)))
        assert not np.array_equal(first, second)
        assert frozen["jax_policy"] != current["jax_policy"]
        with pytest.raises(ValueError, match="environment"):
            campaign.execute(root)
    # Historical records remain inspectable; rejected resume executed no trial.
    assert campaign.verify(root) == []


def test_mcse_and_population_witnesses():
    result = mean_mcse([1, 2, 3, 4])
    assert result["mean"] == 2.5
    assert result["mcse"] == pytest.approx(np.sqrt(5 / 12))
    assert mean_mcse([2])["mcse"] is None
    assert finite_mmd_truth((0.2, 0.5, 0.3), (0.5, 0.2, 0.3)) == pytest.approx(0.09)
    assert finite_mmd_truth((0.2, 0.5, 0.3), (0.2, 0.5, 0.3)) == pytest.approx(0)


def test_named_streams_ignore_scheduling():
    first = rng("a", "b", 17, seed=8).integers(0, 1000, size=10)
    _ = rng("a", "b", 2, seed=8).integers(0, 1000, size=10)
    np.testing.assert_array_equal(
        first, rng("a", "b", 17, seed=8).integers(0, 1000, size=10)
    )
    assert not np.array_equal(
        first, rng("a", "b", 17, seed=8, stream="method").integers(0, 1000, size=10)
    )


@pytest.fixture
def fixed_environment(monkeypatch):
    monkeypatch.setattr(campaign, "environment", lambda: {"test": "fixed"})


def test_resume_merge_and_reject_overlap(tmp_path, fixed_environment):
    full, left, right = (tmp_path / name for name in ("full", "left", "right"))
    for path in (full, left, right):
        campaign.create(path, "mmd.finite-bias", trials=4, profile="smoke")
    campaign.execute(full)
    campaign.execute(left, stop=2)
    assert summarize(left)["scenarios"]["null-equal"]["pending"] == 2
    campaign.execute(right, start=2)
    merged = tmp_path / "merged"
    campaign.merge(merged, [left, right])
    assert campaign.verify(full) == campaign.verify(merged)
    campaign.execute(left)
    assert campaign.verify(full) == campaign.verify(left)
    with pytest.raises(ValueError, match="Overlapping"):
        campaign.merge(tmp_path / "bad", [full, left])


def test_corrupt_or_mismatched_records_rejected(tmp_path, fixed_environment):
    root = tmp_path / "campaign"
    campaign.create(root, "mmd.finite-bias", trials=2)
    campaign.execute(root, stop=1)
    path = next((root / "trials").rglob("*.json"))
    record = json.loads(path.read_text())
    record["payload"]["outcomes"]["U"]["estimate"] = 999
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        campaign.verify(root)


def test_unexpected_interruption_is_pending(tmp_path, fixed_environment, monkeypatch):
    root = tmp_path / "campaign"
    campaign.create(root, "mmd.finite-bias", trials=2)

    def interrupted(*args, **kwargs):
        raise MemoryError("infrastructure stopped")

    monkeypatch.setattr(campaign, "run_trial", interrupted)
    with pytest.raises(MemoryError):
        campaign.execute(root)
    assert campaign.verify(root) == []
    assert summarize(root)["scenarios"]["null-equal"]["pending"] == 2


def test_failure_inclusive_qualification():
    failed = [TrialOutcome(valid=False) for _ in range(10)]
    report = summarize_trials(failed, coverage=True)
    assert report.failed == 10
    assert report.covered == 0
    assert report.conditional_rejection is None
    assert report.size_status == "excess-error"
    assert report.coverage_status == "excess-error"


def test_manifest_and_environment_changes_stop_resume(
    tmp_path, fixed_environment, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.create(root, "mmd.finite-bias", trials=2)
    monkeypatch.setattr(campaign, "environment", lambda: {"test": "changed"})
    with pytest.raises(ValueError, match="environment"):
        campaign.execute(root)
    path = root / "manifest.json"
    record = campaign.read(path)
    record["payload"]["trials"] = 200
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest changed"):
        campaign.verify(root)


def test_shard_incompatible_design_is_rejected(tmp_path, fixed_environment):
    left, right = tmp_path / "left", tmp_path / "right"
    campaign.create(left, "mmd.finite-bias", trials=2, seed=1)
    campaign.create(right, "mmd.finite-bias", trials=2, seed=2)
    with pytest.raises(ValueError, match="Incompatible"):
        campaign.merge(tmp_path / "merged", [left, right])


def test_historical_design_verifies_but_cannot_resume(
    tmp_path, fixed_environment, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.create(root, "mmd.finite-bias", trials=2)
    campaign.execute(root)
    monkeypatch.delitem(campaign.STUDIES, "mmd.finite-bias")
    assert len(campaign.verify(root)) == 8
    assert summarize(root)["scenarios"]["null-equal"]["execution"] == "complete"
    with pytest.raises(ValueError, match="installed adapter"):
        campaign.execute(root)


@pytest.mark.parametrize(
    "field,value", [("estimate", True), ("width", "wide"), ("pvalue", -0.1)]
)
def test_rehashed_invalid_outcomes_rejected(tmp_path, fixed_environment, field, value):
    root = tmp_path / "campaign"
    campaign.create(root, "linear.ols-covariance", trials=2)
    campaign.execute(root, stop=1)
    path = next((root / "trials").rglob("*.json"))
    record = campaign.read(path)
    record["payload"]["outcomes"]["HC0"][field] = value
    record["sha256"] = campaign.fingerprint(record["payload"])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        campaign.verify(root)


def test_installed_adapters_smoke(tmp_path):
    for study in (
        "linear.ols-covariance",
        "mmd.finite-bias",
        "covariance.population-risk",
        "sharpe.point-estimation",
    ):
        root = tmp_path / study
        campaign.create(root, study, trials=2, profile="smoke")
        campaign.execute(root)
        report = summarize(root)
        for scenario in report["scenarios"].values():
            assert scenario["execution"] == "complete"
            assert all(value["valid"] == 2 for value in scenario["methods"].values())
            assert all(
                value.get("qualification", {}).get("size_status", "smoke-only")
                == "smoke-only"
                for value in scenario["methods"].values()
            )


def test_write_once_does_not_overwrite(tmp_path):
    path = tmp_path / "record.json"
    campaign.write_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        campaign.write_once(path, {"original": False})
    assert campaign.read(path) == {"original": True}


def test_export_archive_roundtrip_retains_nested_manifests(tmp_path, fixed_environment):
    root = tmp_path / "campaign"
    campaign.create(root, "mmd.finite-bias", trials=2, profile="smoke")
    campaign.execute(root)
    publication = tmp_path / "publication"
    export(root, publication)
    expected = verify_export(publication)
    archive = tmp_path / "publication.tar"
    write_archive(publication, archive)
    reconstructed = tmp_path / "downloaded"
    extract_archive(archive, reconstructed)
    assert verify_export(reconstructed) == expected
    (reconstructed / "campaign" / "manifest.json").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_export(reconstructed)


def test_nested_source_capsule_tampering_is_rejected(tmp_path):
    root = tmp_path / "captured"
    campaign.create(root, "mmd.finite-bias", trials=1, profile="smoke")
    captured = root / "source/statadic_studies/adapters/discrepancy/finite_bias.py"
    assert captured.is_file()
    captured.write_text(captured.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="Source capsule"):
        campaign.verify(root)


def test_nested_engine_change_invalidates_resume(tmp_path, monkeypatch):
    from statadic_studies import source

    root = tmp_path / "captured"
    campaign.create(root, "mmd.finite-bias", trials=1, profile="smoke")
    original = source.capture

    def changed():
        files = original()
        files["statadic_studies/adapters/discrepancy/finite_bias.py"] += (
            b"\n# changed\n"
        )
        return files

    monkeypatch.setattr(source, "capture", changed)
    with pytest.raises(ValueError, match="environment"):
        campaign.execute(root)
