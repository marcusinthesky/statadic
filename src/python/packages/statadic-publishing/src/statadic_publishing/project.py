"""Strict reader for researcher-authored publishing.toml."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from statadic.api.reporting import FrozenMetadataValue, MetadataValue, freeze_metadata

from ._schema import decode, relative_path

Status = Literal["pending", "unavailable", "not-run", "passed", "failed", "approved"]
HumanReview = Literal["pending", "approved"]


@dataclass(frozen=True)
class ProjectInfo:
    """Project identity and title."""

    id: str
    title: str


@dataclass(frozen=True, init=False)
class Analysis:
    """The authored analysis and the output ids it designates as evidence.

    ``model`` accepts authored metadata, such as lists, and retains an
    immutable copy.
    """

    script: str
    seed: int
    estimand: str
    assumptions: tuple[str, ...]
    model: Mapping[str, FrozenMetadataValue]
    covariance: str
    reference: str
    snapshots: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def __init__(  # noqa: PLR0913, PLR0917 -- one authored record
        self,
        script: str,
        seed: int,
        estimand: str,
        assumptions: tuple[str, ...],
        model: Mapping[str, MetadataValue],
        covariance: str,
        reference: str,
        snapshots: tuple[str, ...],
        diagnostics: tuple[str, ...],
    ) -> None:
        relative_path(script)
        for name, value in (
            ("script", script),
            ("seed", seed),
            ("estimand", estimand),
            ("assumptions", assumptions),
            ("model", freeze_metadata(model)),
            ("covariance", covariance),
            ("reference", reference),
            ("snapshots", snapshots),
            ("diagnostics", diagnostics),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Input:
    """One input, either local and hashed or identified externally."""

    id: str
    path: str | None
    identity: str | None
    sha256: str
    redistributable: bool
    acquisition: str | None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.identity is None):
            raise ValueError("input must specify exactly one of path or identity")
        if self.path is not None:
            relative_path(self.path)
        if not self.redistributable and not self.acquisition:
            raise ValueError(
                "non-redistributable input requires acquisition instructions"
            )


@dataclass(frozen=True)
class Output:
    """One file the analysis writes under its output directory."""

    id: str
    path: str
    media_type: str
    required: bool

    def __post_init__(self) -> None:
        relative_path(self.path)


@dataclass(frozen=True)
class HashedFile:
    """An external file pinned by its SHA-256 digest."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        relative_path(self.path, external=True)


@dataclass(frozen=True)
class Manuscript:
    """The manuscript source, bibliography and offline TeX toolchain."""

    root: str
    main: str
    bibliography: HashedFile
    tex_bundle: HashedFile
    check_log: str

    def __post_init__(self) -> None:
        relative_path(self.root)
        relative_path(self.main)
        relative_path(self.check_log, external=True)


@dataclass(frozen=True)
class Environment:
    """Lockfiles that pin the analysis environment."""

    lockfiles: tuple[str, ...]

    def __post_init__(self) -> None:
        for path in self.lockfiles:
            relative_path(path, external=True)


@dataclass(frozen=True)
class Claim:
    """A manuscript claim and its explicit evidence and review dispositions."""

    id: str
    text: str
    outputs: tuple[str, ...]
    citations: tuple[str, ...]
    registry_claims: tuple[str, ...]
    formal: Status
    source_fidelity: Status
    implementation: Status
    qualification: Status
    human_review: HumanReview


@dataclass(frozen=True)
class Project:
    """A researcher-authored publishing.toml with unique, resolvable ids."""

    project: ProjectInfo
    analysis: Analysis
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    manuscript: Manuscript
    environment: Environment
    claims: tuple[Claim, ...]

    def __post_init__(self) -> None:
        for kind, items in (
            ("input", self.inputs),
            ("output", self.outputs),
            ("claim", self.claims),
        ):
            ids = [item.id for item in items]
            duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
            if duplicates:
                raise ValueError(f"duplicate {kind} ids: {duplicates}")
        referenced = {
            *self.analysis.snapshots,
            *self.analysis.diagnostics,
            *(output for claim in self.claims for output in claim.outputs),
        }
        unknown = referenced - {item.id for item in self.outputs}
        if unknown:
            raise ValueError(f"unknown output ids: {sorted(unknown)}")


def read_project(path: Path) -> Project:
    """Read an authored project, defaulting optional analysis, input and claim keys."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    analysis = data.get("analysis")
    if isinstance(analysis, dict):
        analysis.setdefault("snapshots", [])
        analysis.setdefault("diagnostics", [])
    for item in data.get("inputs", []):
        for name in ("path", "identity", "acquisition"):
            item.setdefault(name, None)
    for item in data.get("claims", []):
        item.setdefault("registry_claims", [])
    return decode(Project, data)
