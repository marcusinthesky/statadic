"""Scientific registration gaps cannot silently become release qualification."""

import pytest
from statadic_docs_labs.campaign_gate import (
    check_execution_identity,
    check_retained_sources,
    inspect_campaigns,
    match_wheel,
)
from statadic_evidence import sha256


def test_hash_only_history_cannot_qualify_a_release():
    for payload in (
        {},
        {"source_capsule": {}},
        {"source_capsule": {"schema_version": 1, "files": {}}},
    ):
        with pytest.raises(ValueError, match="retained source capsules"):
            check_retained_sources(payload)
    check_retained_sources(
        {
            "source_capsule": {
                "schema_version": 1,
                "files": {"statadic/__init__.py": "digest"},
            }
        }
    )


def test_incomplete_catalogue_stops_before_campaign_execution(monkeypatch, tmp_path):
    import statadic_studies.catalogue

    monkeypatch.setattr(
        statadic_studies.catalogue,
        "check_catalogue",
        lambda repository: {
            "coverage_status": "incomplete",
            "missing_study": ["spatial.iv"],
            "source_review_pending": ["equivalence"],
            "ownership_review_pending": [],
        },
    )
    with pytest.raises(ValueError, match="1 missing study mappings"):
        inspect_campaigns(tmp_path, [])


def test_campaign_must_evaluate_exact_release_distribution(tmp_path):
    wheel = tmp_path / "statadic.whl"
    wheel.write_bytes(b"release distribution")
    match_wheel([{"wheel_sha256": sha256(wheel)}], wheel)
    with pytest.raises(ValueError, match="built release wheel"):
        match_wheel([{"wheel_sha256": "previous release"}], wheel)


def test_campaign_runner_changes_and_unsupported_scope_cannot_qualify():
    identity = {
        "backend": "cpu",
        "x64": True,
        "statadic_source": "library",
        "engine_source": "runner",
        "evidence_source": "evidence",
        "versions": {"jax": "locked"},
        "jax_policy": {
            "jax_default_prng_impl": "threefry2x32",
            "jax_threefry_partitionable": True,
        },
        "python": "3.13",
    }
    check_execution_identity(identity, identity)
    for field in (
        "engine_source",
        "evidence_source",
        "versions",
        "statadic_source",
        "jax_policy",
    ):
        with pytest.raises(ValueError, match="execution identity changed"):
            check_execution_identity(identity, identity | {field: "changed"})
    with pytest.raises(ValueError, match="CPU float64"):
        check_execution_identity(identity | {"x64": False}, identity)


def test_unknown_or_changed_threefry_policy_cannot_qualify():
    import jax
    from statadic_studies.campaign import environment

    with jax.enable_x64(True), jax.threefry_partitionable(True):
        frozen = environment()
    with jax.enable_x64(True), jax.threefry_partitionable(False):
        with pytest.raises(ValueError, match="jax_policy"):
            check_execution_identity(frozen, environment())
    legacy_policy = dict(frozen["jax_policy"])
    del legacy_policy["jax_threefry_partitionable"]
    with pytest.raises(ValueError, match="recorded Threefry partitionability"):
        check_execution_identity(frozen | {"jax_policy": legacy_policy}, frozen)
