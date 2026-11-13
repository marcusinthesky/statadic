"""Require retained, complete scientific campaigns before release qualification."""

from __future__ import annotations

from pathlib import Path

from statadic_evidence import fingerprint, sha256


def check_execution_identity(frozen: dict, current: dict) -> None:
    """Require the evaluated implementation and runner in the supported CPU scope."""
    if frozen.get("backend") != "cpu" or frozen.get("x64") is not True:
        raise ValueError("Release evaluations require the supported CPU float64 scope")
    policy = frozen.get("jax_policy")
    if (
        not isinstance(policy, dict)
        or type(policy.get("jax_threefry_partitionable")) is not bool
    ):
        raise ValueError(
            "Release evaluations require recorded Threefry partitionability"
        )
    for field in (
        "statadic_source",
        "engine_source",
        "evidence_source",
        "versions",
        "jax_policy",
        "python",
    ):
        if frozen.get(field) != current.get(field):
            raise ValueError(f"Scientific campaign execution identity changed: {field}")


def check_retained_sources(payload: dict) -> None:
    """Historical hash-only campaigns remain readable, but cannot qualify releases."""
    capsule = payload.get("source_capsule")
    if (
        not isinstance(capsule, dict)
        or capsule.get("schema_version") != 1
        or not isinstance(capsule.get("files"), dict)
        or not capsule["files"]
    ):
        raise ValueError("Release evaluations require retained source capsules")


def inspect_campaigns(
    repository: Path, paths: list[Path], *, scope: dict | None = None
) -> list[dict]:
    """Validate scope while preserving adverse scientific results."""
    from statadic_studies.analysis import verify_export
    from statadic_studies.campaign import environment, manifest
    from statadic_studies.catalogue import check_catalogue
    from statadic_studies.contracts import STUDIES

    coverage = check_catalogue(repository)
    resolved = None
    if scope is not None:
        from statadic_docs_labs.qualification_scope import resolve_scope

        resolved = resolve_scope(repository, scope)
        coverage = resolved
        if coverage["missing_study"] or coverage["source_review_pending"]:
            raise ValueError(
                "Scoped qualification is incomplete: "
                f"{len(coverage['missing_study'])} missing study mappings, "
                f"{len(coverage['source_review_pending'])} source reviews pending"
            )
    elif coverage["coverage_status"] == "incomplete":
        raise ValueError(
            "All-method qualification is incomplete: "
            f"{len(coverage['missing_study'])} missing study mappings, "
            f"{len(coverage['source_review_pending'])} source reviews and "
            f"{len(coverage['ownership_review_pending'])} ownership reviews pending"
        )
    required = set(coverage["required_studies"])
    records = {}
    for path in paths:
        summary = verify_export(path)
        campaign = manifest(path / "campaign")
        payload = campaign["payload"]
        check_retained_sources(payload)
        check_execution_identity(payload["environment"], environment())
        identity = payload["study"]["id"]
        if identity in records:
            raise ValueError(f"Duplicate evaluation campaign: {identity}")
        if identity not in STUDIES or fingerprint(payload["study"]) != fingerprint(
            STUDIES[identity].record()
        ):
            raise ValueError(f"Study definition changed: {identity}")
        if (
            payload["profile"] != "evaluation"
            or payload["trials"] < 10000
            or payload.get("execution_options", {}).get("draws", 0) < 999
            or payload["installation_scope"] != "wheel-supplied"
            or any(
                cell["execution"] != "complete"
                for cell in summary["scenarios"].values()
            )
        ):
            raise ValueError(
                f"Incomplete evaluation profile or installed wheel: {identity}"
            )
        wheel = path / "campaign/installation/statadic.whl"
        if sha256(wheel) != payload["wheel_sha256"]:
            raise ValueError(f"Retained wheel changed: {identity}")
        records[identity] = {
            "study": identity,
            "campaign": campaign["id"],
            "wheel_sha256": payload["wheel_sha256"],
            "path": str(path.resolve()),
            "summary": summary,
        }
    if missing := required - records.keys():
        raise ValueError(f"Missing retained evaluation campaigns: {sorted(missing)}")
    if resolved is not None:
        from statadic_docs_labs.qualification_scope import require_scientific_results

        require_scientific_results(resolved, list(records.values()))
    return list(records.values())


def match_wheel(records: list[dict], wheel: Path) -> None:
    """Prevent campaigns for a different distribution qualifying this release."""
    expected = sha256(wheel)
    if any(record["wheel_sha256"] != expected for record in records):
        raise ValueError("Scientific campaigns do not evaluate the built release wheel")
