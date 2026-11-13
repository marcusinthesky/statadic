"""Version 1 evidence manifest; successful builds write this last."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal, NoReturn

from statadic.api.reporting import (
    FrozenMetadataValue,
    MetadataValue,
    freeze_metadata,
    metadata_to_json,
)
from statadic_evidence import write_json

from ._schema import decode, relative_path
from .project import Claim


@dataclass(frozen=True)
class Hash:
    """A bundle-relative file and its SHA-256 digest."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        relative_path(self.path)


@dataclass(frozen=True)
class InputHash:
    """An input's id, path or external identity, and digest."""

    id: str
    identity: str
    sha256: str


@dataclass(frozen=True)
class Source:
    """The project revision and the staged source digests."""

    revision: str
    dirty: bool | None
    hashes: tuple[Hash, ...]


@dataclass(frozen=True)
class Versions:
    """Package versions that produced the bundle."""

    python: str
    statadic: str
    jax: str
    statadic_publishing: str


@dataclass(frozen=True)
class Runtime:
    """Platform, JAX backend, precision and seed of the build."""

    platform: str
    backend: str
    x64: bool
    seed: int


@dataclass(frozen=True, init=False)
class AnalysisRecord:
    """The authored analysis as recorded in the manifest.

    ``model`` accepts authored metadata and retains an immutable copy.
    """

    script: str
    estimand: str
    model: Mapping[str, FrozenMetadataValue]
    covariance: str
    reference: str
    inputs: tuple[str, ...]
    assumptions: tuple[str, ...]
    snapshots: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def __init__(  # noqa: PLR0913, PLR0917 -- one recorded analysis
        self,
        script: str,
        estimand: str,
        model: Mapping[str, MetadataValue],
        covariance: str,
        reference: str,
        inputs: tuple[str, ...],
        assumptions: tuple[str, ...],
        snapshots: tuple[str, ...],
        diagnostics: tuple[str, ...],
    ) -> None:
        relative_path(script)
        for name, value in (
            ("script", script),
            ("estimand", estimand),
            ("model", freeze_metadata(model)),
            ("covariance", covariance),
            ("reference", reference),
            ("inputs", inputs),
            ("assumptions", assumptions),
            ("snapshots", snapshots),
            ("diagnostics", diagnostics),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class OutputRecord:
    """A declared output and whether the build produced it."""

    id: str
    media_type: str
    path: str
    sha256: str | None
    status: Literal["available", "unavailable"]

    def __post_init__(self) -> None:
        relative_path(self.path)
        if (self.status == "available") != (self.sha256 is not None):
            raise ValueError(
                "available output requires a hash; unavailable output has none"
            )


@dataclass(frozen=True)
class Manifest:
    """Version 1 evidence manifest."""

    schema_version: Literal[1]
    bundle_id: str
    project_id: str
    created_utc: str
    source: Source
    lockfiles: tuple[Hash, ...]
    inputs: tuple[InputHash, ...]
    versions: Versions
    runtime: Runtime
    analyses: tuple[AnalysisRecord, ...]
    outputs: tuple[OutputRecord, ...]
    claims: tuple[Claim, ...]
    inventory: dict[str, str]


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def read_manifest(path: Path) -> Manifest:
    """Read a strict v1 JSON manifest, rejecting duplicate keys and nonfinite values."""

    def invalid_constant(value: str) -> NoReturn:
        raise ValueError(f"invalid JSON constant: {value}")

    with path.open(encoding="utf-8") as stream:
        data = json.load(
            stream, object_pairs_hook=_object, parse_constant=invalid_constant
        )
    return decode(Manifest, data)


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return metadata_to_json(value)


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Validate and write canonical JSON using the evidence package."""
    value = _plain(manifest)
    # Decode the JSON representation to validate direct dataclass construction too.
    value = json.loads(json.dumps(value, allow_nan=False))
    decode(Manifest, value)
    write_json(path, value)
