"""Check package extraction retains fail-closed repository and package gates."""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(ci: dict, package: dict) -> None:
    """Reject a detached gate, missing architecture or reduced shard matrix."""
    jobs = ci["jobs"]
    required = {"package", "check-nontest"}
    require(
        set(jobs["check"]["needs"]) == required,
        "Repository aggregate needs both scopes",
    )
    require(
        jobs["check"].get("if") == "always()",
        "Repository aggregate must report dependency failures",
    )
    require(
        any(
            "DEPENDENCY_RESULTS" in str(step) and '"$result" != success' in str(step)
            for step in jobs["check"]["steps"]
        ),
        "Repository aggregate must reject every failed dependency",
    )
    require(
        jobs["package"].get("uses") == "./.github/workflows/package-validation.yml",
        "Package workflow must be called",
    )
    require(
        any(
            "just check-ci-nontest" in step.get("run", "")
            for step in jobs["check-nontest"]["steps"]
        ),
        "Full workspace gate must remain",
    )
    require(
        jobs["check-nontest"]["strategy"]["matrix"]["runner"] == ["ubuntu-24.04"],
        "Architecture-independent workspace checks have one owner",
    )
    workspace_alias = jobs["check-nontest-arm"]
    require(
        workspace_alias.get("needs") == "check-nontest"
        and workspace_alias.get("if") == "always()"
        and any(
            '"$WORKSPACE_RESULT" != success' in step.get("run", "")
            and step.get("env", {}).get("WORKSPACE_RESULT")
            == "${{ needs.check-nontest.result }}"
            for step in workspace_alias["steps"]
        ),
        "The former ARM workspace status must require the shared gate",
    )
    references = jobs["reference-validation"]
    require(
        references.get("needs") == "package"
        and references.get("if") == "always()"
        and any(
            '"$PACKAGE_RESULT" != success' in step.get("run", "")
            and step.get("env", {}).get("PACKAGE_RESULT")
            == "${{ needs.package.result }}"
            for step in references["steps"]
        ),
        "Reference statuses must require installed package comparisons",
    )
    jobs = package["jobs"]
    require(
        jobs["timings"].get("continue-on-error") is True,
        "Advisory timing retention cannot block package validation",
    )
    for name, dependencies in (
        ("regressions", {"build"}),
        ("plan", {"regressions"}),
        ("shard", {"plan"}),
        ("quality", {"build"}),
        ("installed", {"build"}),
        ("aggregate", {"quality", "plan", "shard", "installed"}),
        ("candidate", {"aggregate"}),
    ):
        needs = jobs[name].get("needs", [])
        require(
            set([needs] if isinstance(needs, str) else needs) == dependencies,
            f"Detached package gate: {name}",
        )
    for name in ("regressions", "plan", "shard", "quality", "installed", "aggregate"):
        require(
            set(jobs[name]["strategy"]["matrix"]["runner"])
            == {"ubuntu-24.04", "ubuntu-24.04-arm"},
            f"Both architectures required: {name}",
        )
    require(
        jobs["shard"]["strategy"]["matrix"]["shard"] == [0, 1, 2, 3],
        "Retain all four shards",
    )
    for name, command in (
        ("build", " build --output"),
        ("regressions", "ci_shards.py regressions"),
        ("plan", "ci_shards.py plan"),
        ("shard", "ci_shards.py shard"),
        ("quality", "just python::check-package"),
        ("installed", " qualify --candidate"),
        ("aggregate", "ci_shards.py aggregate"),
        ("candidate", " seal --candidate"),
    ):
        require(
            any(command in step.get("run", "") for step in jobs[name]["steps"]),
            f"Missing validation: {name}",
        )
    require(
        package.get("permissions") == {"contents": "read"},
        "Candidate workflow cannot publish",
    )
    require(
        all("permissions" not in job for job in jobs.values()),
        "Candidate jobs cannot elevate permissions",
    )
