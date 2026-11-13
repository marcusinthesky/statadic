"""Adversarial retained-record checks; these are not benchmark measurements."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile

import pytest
from statadic_evidence import fingerprint, inventory, sha256, write_json
from statadic_performance import reference
from statadic_performance.reference_workloads import fixture


def campaign(tmp_path, monkeypatch, *, qualification=False, reporting=False):
    path = tmp_path / "campaign"
    (path / "inputs").mkdir(parents=True)
    wheel = tmp_path / "statadic.whl"
    payload = b'"""Synthetic wheel used only to test evidence validation."""\n'
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("statadic/__init__.py", payload)
    shutil.copyfile(wheel, path / "inputs/candidate.whl")
    lock = tmp_path / "uv.lock"
    versions = {
        "numpy": "2.4.6",
        "jax": "0.11.0",
        "jaxlib": "0.11.0",
        "statadic": "0.4.1",
        "statsmodels": "0.14.6",
    }
    lock.write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
            for name, version in versions.items()
        )
    )
    shutil.copyfile(lock, path / "inputs/uv.lock")
    scope = {
        "fingerprint": "a" * 64,
        "request": {"performance_workloads": ["linear-fit-and-inference"]},
    }
    if reporting:
        scope["request"]["performance_policy"] = "report"
    write_json(path / "inputs/scope.json", scope)
    case = "ols-hc3"
    description, _ = fixture(case, pilot=not qualification)
    sessions = 3 if qualification else 1
    samples, warmups = (30, 5) if qualification else (3, 1)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "qualification" if qualification else "pilot",
        "wheel_sha256": sha256(wheel),
        "lock_sha256": sha256(lock),
        "scope_fingerprint": scope["fingerprint"],
        "harness_fingerprint": reference.harness_fingerprint(),
        "cases": [case],
        "sessions": sessions,
        "samples": samples,
        "warmups": warmups,
        "fixtures": {case: fingerprint(description)},
        "records": [],
    }
    for session in range(sessions):
        for side in reference.SIDES:
            record = {
                "schema_version": 1,
                "case": case,
                "side": side,
                "session": session,
                "process_identity": f"{side}-{session}",
                "python": "3.13.5",
                "dtype": "float64",
                "backend": "cpu",
                "hardware": {"cpu": "synthetic-test-only", "affinity": [1]},
                "fixture": description,
                "installed_distributions": versions,
                "installed_wheel": {
                    "wheel_sha256": sha256(wheel),
                    "version": "0.4.1",
                    "payload_files": {
                        "statadic/__init__.py": hashlib.sha256(payload).hexdigest()
                    },
                }
                if side == "statadic"
                else None,
                "compiler_settings": {
                    "jax_disable_jit": False,
                    "jax_disable_most_optimizations": False,
                    "jax_default_matmul_precision": "highest",
                }
                if side == "statadic"
                else None,
                "jax_imported": side == "statadic",
                "cache_policy": "fresh-process;persistent-disabled",
                "warmups": warmups,
                "warm_seconds": [1.0 if side == "statadic" else 2.0] * samples,
                "first_call_seconds": 1.0,
                "process_peak_rss_bytes": 90 if side == "statadic" else 100,
                "timing_scope": "same consumer outputs",
                "rss_scope": "whole fresh process",
                "threads": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1",
                    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
                },
                "values": [1.0, 2.0],
                "shape": [2],
            }
            name = f"{case}-{side}-{session}.json"
            write_json(path / name, record)
            manifest["records"].append(name)
    if qualification:
        # This tiny roster tests gate arithmetic, not real adapter coverage.
        monkeypatch.setattr(
            reference,
            "QUALIFICATION_CASES",
            {
                "linear-fit-and-inference": (case,),
            },
        )
    seal(path, manifest)
    return path, wheel, scope, lock, manifest


def seal(path, manifest):
    manifest["files"] = inventory(path)
    write_json(path / "manifest.json", manifest)


def verify(path, wheel, scope, lock, *, qualification=True):
    return reference.verify_reference_campaign(
        path,
        wheel=wheel,
        scope=scope,
        lock=lock,
        require_qualification=qualification,
    )


def test_pilot_reports_observations_but_cannot_mint_release_evidence(
    tmp_path, monkeypatch
):
    path, wheel, scope, lock, _ = campaign(tmp_path, monkeypatch)
    result = verify(path, wheel, scope, lock, qualification=False)
    assert result["contract_evidence"] == {}
    assert result["observations"][0]["total_peak_rss_ratio"] == 0.90
    with pytest.raises(ValueError, match="Pilot evidence"):
        verify(path, wheel, scope, lock)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warm_seconds", [1.81] * 30),
        ("process_peak_rss_bytes", 91),
    ],
)
def test_each_of_three_sessions_must_win_both_resources(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    path, wheel, scope, lock, manifest = campaign(
        tmp_path, monkeypatch, qualification=True
    )
    result = verify(path, wheel, scope, lock)
    assert len(result["contract_evidence"]["linear-fit-and-inference"]["sessions"]) == 3
    last = path / "ols-hc3-statadic-2.json"
    record = reference.load(last)
    record[field] = value
    write_json(last, record)
    seal(path, manifest)
    with pytest.raises(ValueError, match="win failed"):
        verify(path, wheel, scope, lock)


def test_explicit_report_policy_retains_adverse_completed_pilot(tmp_path, monkeypatch):
    path, wheel, scope, lock, manifest = campaign(tmp_path, monkeypatch, reporting=True)
    worker = path / "ols-hc3-statadic-0.json"
    record = reference.load(worker)
    record["warm_seconds"] = [5.0] * manifest["samples"]
    record["process_peak_rss_bytes"] = 700
    write_json(worker, record)
    seal(path, manifest)
    result = verify(path, wheel, scope, lock)
    assert result["observations"][0]["passed"] is False
    evidence = result["contract_evidence"]["linear-fit-and-inference"]
    assert evidence["status"] == "reported"
    assert evidence["performance_policy"] == "report"
    assert evidence["measurement_purpose"] == "pilot"
    assert evidence["cases"] == ["ols-hc3"]
    assert evidence["sessions"] == [
        {
            "id": "0",
            "warm_time_ratio": 2.5,
            "total_peak_rss_ratio": 7.0,
        }
    ]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("values", [1.0, 2.1], "output mismatch"),
        ("process_peak_rss_bytes", 0, "positive measurement"),
        ("warm_seconds", [float("nan")] * 3, "positive measurement"),
    ],
)
def test_report_policy_still_requires_numerical_agreement_and_actual_measurements(
    tmp_path,
    monkeypatch,
    field,
    value,
    match,
):
    path, wheel, scope, lock, manifest = campaign(tmp_path, monkeypatch, reporting=True)
    worker = path / "ols-hc3-statadic-0.json"
    record = reference.load(worker)
    record[field] = value
    # This invalid raw-record fixture intentionally permits nonfinite JSON;
    # real workers reject nonfinite outputs and measurements before writing.
    import json

    worker.write_text(json.dumps(record))
    seal(path, manifest)
    with pytest.raises(ValueError, match=match):
        verify(path, wheel, scope, lock)


def test_report_policy_cannot_hide_an_unmeasured_required_workload(
    tmp_path, monkeypatch
):
    path, wheel, scope, lock, manifest = campaign(tmp_path, monkeypatch, reporting=True)
    scope["request"]["performance_workloads"].append("ordination-fit-transform")
    write_json(path / "inputs/scope.json", scope)
    seal(path, manifest)
    with pytest.raises(ValueError, match="Missing required performance observations"):
        verify(path, wheel, scope, lock)


def test_unknown_performance_policy_cannot_disable_legacy_superiority(
    tmp_path, monkeypatch
):
    path, wheel, scope, lock, _ = campaign(tmp_path, monkeypatch)
    scope["request"]["performance_policy"] = "anything-goes"
    with pytest.raises(ValueError, match="performance policy"):
        verify(path, wheel, scope, lock)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("values", [1.0, 2.1], "output mismatch"),
        ("process_identity", "reference-0", "process identity"),
        ("installed_wheel", {"wheel_sha256": "b" * 64}, "wheel binding"),
        ("installed_distributions", {"numpy": "0.0"}, "dependency differs"),
        ("warm_seconds", [1.0], "timing distribution"),
        ("process_peak_rss_bytes", 0, "positive measurement"),
        ("threads", {}, "thread or compiler"),
        ("hardware", {"cpu": "different"}, "hardware or affinity"),
    ],
)
def test_resealed_semantically_invalid_worker_is_rejected(
    tmp_path,
    monkeypatch,
    field,
    value,
    match,
):
    path, wheel, scope, lock, manifest = campaign(tmp_path, monkeypatch)
    worker = path / "ols-hc3-statadic-0.json"
    record = reference.load(worker)
    record[field] = value
    write_json(worker, record)
    seal(path, manifest)
    with pytest.raises(ValueError, match=match):
        verify(path, wheel, scope, lock, qualification=False)


def test_whole_scope_cannot_inherit_one_pilot_family(tmp_path, monkeypatch):
    path, wheel, scope, lock, manifest = campaign(
        tmp_path, monkeypatch, qualification=True
    )
    scope["request"]["performance_workloads"].append("joint-censored-sar-aft")
    write_json(path / "inputs/scope.json", scope)
    seal(path, manifest)
    with pytest.raises(ValueError, match="Incomplete required performance"):
        verify(path, wheel, scope, lock)


def test_every_required_workload_subcase_must_be_measured(tmp_path, monkeypatch):
    path, wheel, scope, lock, _ = campaign(tmp_path, monkeypatch, qualification=True)
    monkeypatch.setattr(
        reference,
        "QUALIFICATION_CASES",
        {"linear-fit-and-inference": ("ols-hc3", "wls-hc3")},
    )
    with pytest.raises(ValueError, match="Incomplete required performance"):
        verify(path, wheel, scope, lock)


def test_registered_subcases_are_complete_for_exactly_fourteen_workload_contracts():
    pairs = [
        (workload, case)
        for workload, cases in reference.QUALIFICATION_CASES.items()
        for case in cases
    ]
    assert len(reference.QUALIFICATION_CASES) == 14
    assert "joint-censored-sar-aft" not in reference.QUALIFICATION_CASES
    assert len({case for _, case in pairs}) == len(pairs)
    assert {case for _, case in pairs} <= set(reference.PILOT_CASES)
    assert set(reference.PILOT_CASES) - {case for _, case in pairs} == {
        "kernel-pca-fit-transform",
        "isomap-fit-transform",
        "geometric-freedman-lane",
    }
    for workload, case in pairs:
        assert reference.PILOT_CASES[case] == workload


def test_measured_wheel_and_harness_are_freshness_requirements(tmp_path, monkeypatch):
    path, wheel, scope, lock, _ = campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(reference, "harness_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="harness changed"):
        verify(path, wheel, scope, lock, qualification=False)


def test_runner_preserves_virtual_environment_interpreter_symlink(
    tmp_path, monkeypatch
):
    _, wheel, scope, lock, _ = campaign(tmp_path, monkeypatch)
    base = tmp_path / "base-python"
    base.write_text("not executed")
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(base)
    observed = []

    def stop(command, **kwargs):
        observed.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(reference.subprocess, "run", stop)
    with pytest.raises(subprocess.CalledProcessError):
        reference.run(
            tmp_path / "new-run",
            wheel=wheel,
            scope=scope,
            lock=lock,
            python=interpreter,
            reference_python=interpreter,
            cases=["ols-hc3"],
        )
    assert observed[0][0] == str(interpreter.absolute())
    assert observed[0][0] != str(base)
