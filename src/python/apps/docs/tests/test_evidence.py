"""Reject altered release evidence and incomplete scientific execution lanes."""

import json
import os
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from statadic_evidence import extract_archive, inventory, write_archive

APP = Path(__file__).resolve().parents[1]
RELEASE = runpy.run_path(str(APP / "scripts/release.py"))


def test_release_lane_streams_bytes_before_exit(tmp_path, capfdbinary):
    log = tmp_path / "lane.log"
    proceed = tmp_path / "proceed"
    command = [
        sys.executable,
        "-c",
        "import os, pathlib, sys, time; "
        "os.write(1, b'first\\xff'); "
        "p = pathlib.Path(sys.argv[1]); "
        "exec('while not p.exists():\\n time.sleep(0.01)'); "
        "os.write(2, b'second'); sys.exit(7)",
        str(proceed),
    ]
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            RELEASE["run_logged"](command, cwd=tmp_path, env=os.environ, log_path=log)
        )
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while (not log.exists() or log.read_bytes() != b"first\xff") and (
            time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert log.read_bytes() == b"first\xff"
        assert capfdbinary.readouterr().out == b"first\xff"
    finally:
        proceed.touch()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert results[0].returncode == 7
    assert log.read_bytes() == b"first\xffsecond"
    assert capfdbinary.readouterr().out == b"second"


def test_evidence_rejects_tampering_and_stale_sources(tmp_path, monkeypatch):
    artifact = tmp_path / "example.whl"
    artifact.write_bytes(b"qualified bytes")
    (tmp_path / "site").mkdir()
    (tmp_path / "site/build-provenance.json").write_text(
        json.dumps({"release_evidence": True, "study_exports": []})
    )
    manifest = {
        "schema_version": 1,
        "revision": "revision",
        "status": "passed",
        "source_files": {"source.py": "original"},
        "files": inventory(tmp_path),
        "documentation_evidence": {"release_evidence": True, "exports": {}},
    }
    manifest.update(_reference_manifest(tmp_path))
    manifest["files"] = inventory(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify = RELEASE["verify"]
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: "revision\n"
    )
    monkeypatch.setitem(
        verify.__globals__, "source_snapshot", lambda: manifest["source_files"]
    )
    verify(tmp_path, require_qualification=False)
    artifact.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="evidence changed"):
        verify(tmp_path, require_qualification=False)
    monkeypatch.setitem(
        verify.__globals__, "source_snapshot", lambda: {"source.py": "new"}
    )
    with pytest.raises(ValueError, match="Source changed"):
        verify(tmp_path, require_qualification=False)


@pytest.mark.parametrize("outcome", ["<skipped/>", "<failure/>", "<error/>", "empty"])
def test_incomplete_required_lanes_fail(tmp_path, outcome):
    report = tmp_path / "results.xml"
    case = "" if outcome == "empty" else f"<testcase>{outcome}</testcase>"
    report.write_text(f"<testsuites><testsuite>{case}</testsuite></testsuites>")
    with pytest.raises(ValueError, match="Incomplete required"):
        RELEASE["junit_summary"](report)


def test_release_documentation_binds_the_exact_retained_export(tmp_path):
    export = tmp_path / "method-evaluations/example"
    export.mkdir(parents=True)
    evidence = export / "evidence-manifest.json"
    evidence.write_text('{"study": "example"}')
    site = tmp_path / "site"
    site.mkdir()
    metadata = {
        "release_evidence": True,
        "study_exports": [
            {
                "path": "/original/build/location",
                "manifest_sha256": RELEASE["digest"](evidence),
            }
        ],
    }
    path = site / "build-provenance.json"
    path.write_text(json.dumps(metadata))
    receipt = RELEASE["documentation_evidence"](tmp_path, [{"study": "example"}])
    assert receipt["exports"] == {"example": RELEASE["digest"](evidence)}
    # Original machine paths are informational; the retained bytes own identity.
    command = RELEASE["documentation_command"]([export])
    assert command[-3:] == [
        "--release-evidence",
        "--study-export",
        str(export.resolve()),
    ]
    evidence.write_text('{"study": "different"}')
    with pytest.raises(ValueError, match="different study exports"):
        RELEASE["documentation_evidence"](tmp_path, [{"study": "example"}])
    with pytest.raises(ValueError, match="different study exports"):
        RELEASE["documentation_evidence"](tmp_path, [])
    path.write_text(json.dumps(metadata | {"release_evidence": False}))
    with pytest.raises(ValueError, match="explicit evidence selection"):
        RELEASE["documentation_evidence"](tmp_path, [])


def test_scientific_disposition_is_distinct_from_execution_success(tmp_path):
    report = tmp_path / "results.xml"
    report.write_text("""<testsuites><testsuite><testcase name="study">
      <properties><property name="qualification_report"
      value='{ "coverage_status": "inconclusive", "requested": 10000 }'/></properties>
      </testcase></testsuite></testsuites>""")
    results = RELEASE["junit_summary"](report)
    assert results["selected"] == 1
    assert results["studies"][0]["report"]["coverage_status"] == "inconclusive"
    assert results["studies"][0]["report"]["requested"] == 10000


def _reference_manifest(tmp_path):
    (tmp_path / "reference-catalogue.json").write_text('{"methods": []}')
    runner = runpy.run_path(str(RELEASE["PACKAGE"] / "scripts/check_references.py"))
    manifest = {"reference_index_version": 1, "lanes": []}
    for name in ("external", "precision-float32", "precision-float64"):
        lane = "external-float64" if name == "external" else name
        report = tmp_path / f"{name}.xml"
        report.write_text(
            '<testsuites><testcase name="fit"><properties>'
            '<property name="reference_dependencies" value=\'["spreg"]\'/>'
            '<property name="reference_nodeid" value="tests/reference.py::test_fit"/>'
            "</properties></testcase></testsuites>"
        )
        receipt = runner["reference_receipt"](
            tmp_path / "reference-catalogue.json", report, lane
        )
        path = tmp_path / "reference-index" / lane / "reference-index.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
        manifest["lanes"].append(
            {
                "name": f"{name}-reference-index",
                "status": "passed",
                "reference_index": receipt,
            }
        )
    (tmp_path / "check.log").write_text("Completed checkout checks\n")
    manifest["lanes"].append(
        {
            "name": "check",
            "command": ["just", "check"],
            "status": "passed",
            "returncode": 0,
        }
    )
    return manifest


@pytest.mark.parametrize("version", [None, 0, 2])
def test_reference_indexes_cannot_be_disabled_by_manifest_version(tmp_path, version):
    with pytest.raises(ValueError, match="requires reference index version 1"):
        RELEASE["verify_reference_indexes"](
            tmp_path, {"reference_index_version": version}
        )


def test_reference_indexes_are_reconstructed_from_retained_reports(tmp_path):
    """Matching manifest and receipt forgery cannot replace executed reference nodes."""
    manifest = _reference_manifest(tmp_path)
    RELEASE["verify_reference_indexes"](tmp_path, manifest)
    # Both summaries can agree while lying about the underlying report.
    manifest["lanes"][0]["reference_index"]["mapped_nodes"] = ["invented"]
    receipt_path = tmp_path / "reference-index/external-float64/reference-index.json"
    receipt_path.write_text(json.dumps(manifest["lanes"][0]["reference_index"]))
    with pytest.raises(ValueError, match="differs from retained execution"):
        RELEASE["verify_reference_indexes"](tmp_path, manifest)


def test_failed_command_still_writes_manifest_and_log(tmp_path, monkeypatch):
    main = RELEASE["main"]
    monkeypatch.setitem(main.__globals__, "configure_execution_policy", lambda: None)
    monkeypatch.setitem(main.__globals__, "OUTPUT", tmp_path / "bundle")
    monkeypatch.setitem(main.__globals__, "source_snapshot", lambda: {})
    monkeypatch.setattr(sys, "argv", ["release.py", "--mode", "preflight"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "revision")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k one_case --deselect required_case")
    monkeypatch.setenv("PYTEST_PLUGINS", "caller_collection_plugin")
    environments = []

    def failed(command, **kwargs):
        environments.append(kwargs["env"])
        kwargs["log_path"].touch()
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setitem(main.__globals__, "run_logged", failed)
    assert main() == 1
    manifest = json.loads((tmp_path / "bundle/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["lanes"][0]["returncode"] == 7
    assert manifest["lanes"][0]["name"] == "prepare-dependencies"
    assert "prepare-dependencies.log" in manifest["files"]
    assert "check.log" not in manifest["files"]
    assert len(environments) == 1
    assert environments[0]["SOURCE_DATE_EPOCH"] == "315532800"
    assert "PYTEST_ADDOPTS" not in environments[0]
    assert "PYTEST_PLUGINS" not in environments[0]
    assert manifest["build_policy"]["source_date_epoch"] == "315532800"
    reviews = manifest["formal_scope"]["source_review"]
    assert "human semantic review pending" in reviews["STATADIC-MMD-002"]
    assert "human semantic review pending" in reviews["STATADIC-MMD-001"]
    assert manifest["formal_scope"]["implementation_verification"] == (
        "Not established by Lean certification"
    )


def test_archive_preserves_hidden_and_nested_manifests(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "site").mkdir(parents=True)
    (bundle / "site/.buildinfo").write_text("hidden build identity")
    (bundle / "site/manifest.json").write_text("nested manifest")
    files = inventory(bundle)
    assert set(files) == {"site/.buildinfo", "site/manifest.json"}
    (bundle / "manifest.json").write_text(json.dumps({"files": files}))
    archive = tmp_path / "evidence.tar"
    write_archive(bundle, archive)
    extracted = tmp_path / "downloaded"
    extract_archive(archive, extracted)
    assert inventory(extracted) == files
    assert (extracted / "manifest.json").read_bytes() == (
        bundle / "manifest.json"
    ).read_bytes()


def test_preflight_cannot_claim_scientific_qualification(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "mode": "preflight",
            }
        )
    )
    with pytest.raises(ValueError, match="does not establish scientific"):
        RELEASE["verify"](tmp_path)


def test_interruption_retains_running_lane_without_passing(tmp_path, monkeypatch):
    main = RELEASE["main"]
    monkeypatch.setitem(main.__globals__, "configure_execution_policy", lambda: None)
    monkeypatch.setitem(main.__globals__, "OUTPUT", tmp_path / "bundle")
    monkeypatch.setitem(main.__globals__, "source_snapshot", lambda: {})
    monkeypatch.setattr(sys, "argv", ["release.py", "--mode", "preflight"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "revision")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setitem(main.__globals__, "run_logged", interrupted)
    assert main() == 1
    manifest = json.loads((tmp_path / "bundle/manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["lanes"][0]["status"] == "running"


def test_release_policy_is_explicit_before_inspection(monkeypatch):
    updates = []
    policy = SimpleNamespace(
        config=SimpleNamespace(
            update=lambda name, value: updates.append((name, value))
        ),
        default_backend=lambda: "cpu",
    )
    monkeypatch.setitem(sys.modules, "jax", policy)
    monkeypatch.setenv("JAX_PLATFORMS", "caller-policy")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    RELEASE["configure_execution_policy"]()
    assert RELEASE["os"].environ["JAX_PLATFORMS"] == "cpu"
    assert RELEASE["os"].environ["JAX_ENABLE_X64"] == "1"
    assert updates == [("jax_enable_x64", True)]


def test_release_workflow_provisions_references_before_preflight():
    workflow = (APP.parents[3] / ".github/workflows/release-validation.yml").read_text()
    assert workflow.count("devenv shell -- just setup-references") == 1
    assert (
        workflow.index("devenv shell -- just setup")
        < workflow.index("devenv shell -- just setup-references")
        < workflow.index("devenv shell -- just release-preflight")
    )
