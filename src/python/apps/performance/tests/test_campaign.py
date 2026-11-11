"""Comparison rejects incompatible evidence before reporting ratios."""

import pytest
from statadic_evidence import inventory, write_json
from statadic_performance.cli import compare, main, report, source_inventory, verify


def test_source_inventory_detects_nested_additions(tmp_path):
    owner = tmp_path / "src/python/apps/performance/src/statadic_performance"
    owner.mkdir(parents=True)
    (owner / "identities.json").write_text("{}")
    before = source_inventory(tmp_path)
    nested = owner / "nested/added.py"
    nested.parent.mkdir()
    nested.write_text("value = 1\n")
    assert source_inventory(tmp_path) != before
    nested.unlink()
    assert source_inventory(tmp_path) == before


def campaign(path, *, dtype="float32"):
    path.mkdir()
    record = {
        "workload_fingerprint": "fixed-design",
        "harness_fingerprint": "fixed-harness-code",
        "dtype": dtype,
        "backend": "cpu",
        "hardware": {"cpu": "test"},
        "devices": ["cpu"],
        "versions": {},
        "python": "3.13",
        "cache_policy": "disabled",
        "compiler_settings": {"jax_disable_most_optimizations": False},
        "threads": {"OMP_NUM_THREADS": "1"},
        "memory_only": False,
        "variants": [
            {
                "variant": "test",
                "warm_seconds": [1.0, 2.0],
                "transfer_seconds": 0.1,
                "lowering_seconds": 0.1,
                "compilation_seconds": 0.1,
                "first_call_seconds": 0.1,
                "output_host_materialization_seconds": 0.1,
                "numerical_witness": {"status": "passed"},
                "compiler_memory": {"status": "unavailable"},
            }
        ],
    }
    write_json(path / "ols-float32-0.json", record)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "lock_sha256": "locked",
        "revision": "test",
        "workloads": ["ols"],
        "dtypes": ["float32"],
        "sessions": 1,
        "samples": 2,
        "warmups": 1,
        "smoke": True,
        "records": ["ols-float32-0.json"],
        "memory_records": ["ols-float32-memory.json"],
        "cold_records": ["ols-float32-cold-test-0.json"],
        "variant_names": {"ols": ["test"]},
    }
    memory = record | {"memory_only": True, "process_peak_rss_bytes": 1024}
    write_json(path / "ols-float32-memory.json", memory)
    cold = record | {
        "measurement_mode": "cold-jit-call",
        "variants": [
            {
                "variant": "test",
                "cold_jit_call_seconds": 0.3,
                "numerical_witness": {"status": "passed"},
            }
        ],
    }
    write_json(path / "ols-float32-cold-test-0.json", cold)
    manifest["files"] = inventory(path)
    write_json(path / "manifest.json", manifest)
    return manifest, record


def test_comparison_and_report(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    campaign(before)
    campaign(after)
    assert compare(before, after)["comparisons"][0]["candidate_over_baseline"] == 1
    assert "Median warm seconds" in report(before)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float64"),
        ("hardware", {}),
        ("cache_policy", "changed"),
        ("threads", {}),
        ("compiler_settings", {"jax_disable_most_optimizations": True}),
        ("workload_fingerprint", "new"),
        ("harness_fingerprint", "changed-code-with-the-same-description"),
    ],
)
def test_comparison_rejects_mismatch(tmp_path, field, value):
    before, after = tmp_path / "before", tmp_path / "after"
    campaign(before)
    manifest, record = campaign(after)
    record[field] = value
    write_json(after / manifest["records"][0], record)
    manifest["files"] = inventory(after)
    write_json(after / "manifest.json", manifest)
    with pytest.raises(ValueError, match=field):
        compare(before, after)


def test_evidence_tampering_and_incomplete_distribution(tmp_path):
    path = tmp_path / "campaign"
    manifest, record = campaign(path)
    record["variants"][0]["warm_seconds"] = []
    write_json(path / manifest["records"][0], record)
    with pytest.raises(ValueError, match="inventory"):
        verify(path)
    manifest["files"] = inventory(path)
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="distribution"):
        verify(path)


@pytest.mark.parametrize("side", ["before", "after", "alias"])
@pytest.mark.parametrize("filename", ["manifest.json", "nested/comparison.json"])
def test_comparison_output_preserves_campaigns(tmp_path, monkeypatch, side, filename):
    before, after = tmp_path / "before", tmp_path / "after"
    campaign(before)
    campaign(after)
    alias = tmp_path / "alias"
    alias.symlink_to(after, target_is_directory=True)
    original = {path: inventory(path) for path in (before, after)}
    manifests = {path: (path / "manifest.json").read_bytes() for path in original}
    monkeypatch.setattr(
        "sys.argv",
        [
            "performance",
            "compare",
            str(before),
            str(after),
            "--output",
            str(tmp_path / side / filename),
        ],
    )
    with pytest.raises(ValueError, match="immutable"):
        main()
    for path, files in original.items():
        assert inventory(path) == files
        assert (path / "manifest.json").read_bytes() == manifests[path]
        verify(path)


def test_comparison_requires_recorded_compiler_settings(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    for path in (before, after):
        manifest, record = campaign(path)
        del record["compiler_settings"]
        write_json(path / manifest["records"][0], record)
        manifest["files"] = inventory(path)
        write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="compiler_settings"):
        compare(before, after)


def test_report_output_does_not_truncate_hard_linked_evidence(tmp_path, monkeypatch):
    path = tmp_path / "campaign"
    campaign(path)
    manifest = path / "manifest.json"
    original = manifest.read_bytes()
    output = tmp_path / "report.md"
    output.hardlink_to(manifest)
    monkeypatch.setattr(
        "sys.argv", ["performance", "report", str(path), "--output", str(output)]
    )
    main()
    assert manifest.read_bytes() == original
    assert "Median warm seconds" in output.read_text()
    verify(path)
