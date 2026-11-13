"""Version-7 declarative JSON and little-endian numeric artifact storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping  # runtime annotations  # noqa: TC003
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp

from statadic.api.data import _check_array_precision
from statadic.api.reporting import (
    JsonValue,  # noqa: TC001  # Runtime artifact annotations.
    freeze_metadata,
    metadata_to_json,
)

from ._schema import (  # noqa: TC001  # Runtime artifact annotations.
    ArrayEntry,
    ArrayInventory,
    Bundle,
    WriterRecord,
)

_FORMATS = {
    "float16": "e",
    "float32": "f",
    "float64": "d",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
}
_SCHEMA_VERSION = 7
_DEFAULT_WIDTH = 4
_CODEC_CHUNK = 4096


def _encode(array: jax.Array) -> bytes:
    """Encode finite numeric scalars without a NumPy persistence bridge."""
    code = _FORMATS[array.dtype.name]
    flattened = array.reshape(-1)
    chunks = []
    for start in range(0, flattened.size, _CODEC_CHUNK):
        values = flattened[start : start + _CODEC_CHUNK].tolist()
        chunks.append(struct.pack("<" + code * len(values), *values))
    return b"".join(chunks)


class ArtifactError(ValueError):
    """Raised when an artifact cannot support its declared capabilities safely."""


def record(value: object, keys: set[str], context: str) -> dict[str, object]:
    """Require an exact record schema rather than ignoring unknown fields."""
    if not isinstance(value, dict) or set(value) != keys:
        message = f"Invalid {context} schema; expected fields {sorted(keys)}."
        raise ArtifactError(message)
    return {key: value[key] for key in keys}


def optional_text(value: object, context: str) -> str | None:
    """Decode an optional nonempty declarative name."""
    return None if value is None else text(value, context)


def integer(value: object, context: str) -> int:
    """Decode an integer without accepting booleans or negative inventory sizes."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        message = f"{context} must be a nonnegative integer."
        raise ArtifactError(message)
    return value


def number(value: object, context: str) -> float:
    """Decode one finite real JSON number, excluding booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        message = f"{context} must be a finite real number."
        raise ArtifactError(message)
    return float(value)


def items(value: object, context: str) -> list[object]:
    """Decode an ordered JSON list without asserting its element schema."""
    if not isinstance(value, list):
        message = f"{context} must be an ordered list."
        raise ArtifactError(message)
    return list(value)


def string_record(value: object, context: str) -> dict[str, object]:
    """Decode a record with dynamic string keys, preserving empty metadata keys."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        message = f"{context} must be a record with string keys."
        raise ArtifactError(message)
    return {key: child for key, child in value.items() if isinstance(key, str)}


def text(value: object, context: str) -> str:
    """Require one nonempty declarative string."""
    if not isinstance(value, str) or not value:
        message = f"{context} must be a nonempty string."
        raise ArtifactError(message)
    return value


def labels(value: object, context: str) -> tuple[int | str, ...]:
    """Preserve integer versus string identities and their exact order."""
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, str)) for item in value
    ):
        message = f"{context} must contain integer/string identities."
        raise ArtifactError(message)
    result = tuple(value)
    if len(set(result)) != len(result):
        message = f"{context} identities must be unique."
        raise ArtifactError(message)
    return result


def _require_artifact_precision(value: object) -> None:
    """Reject precision loss at the artifact boundary."""
    try:
        _check_array_precision(value, "artifact")
    except (TypeError, ValueError) as exc:
        message = "Artifact requires caller-enabled JAX x64; saving never downcasts."
        raise ArtifactError(message) from exc


class Arrays:
    """Own numeric payloads and assign declarative local array references."""

    def __init__(self, values: Mapping[str, jax.Array] | None = None) -> None:
        """Start a writer or a reader using already-verified array values."""
        self.values = {} if values is None else dict(values)
        self.used: set[str] = set()

    def add(self, value: object) -> str:
        """Copy a concrete numeric array at the explicit persistence boundary."""
        _require_artifact_precision(value)
        array = jnp.array(value, copy=True)
        if array.dtype.name not in _FORMATS or not bool(jnp.all(jnp.isfinite(array))):
            message = (
                "Artifact arrays must be finite real numeric arrays, never objects."
            )
            raise ArtifactError(message)
        name = f"array{len(self.values):04d}"
        self.values[name] = array
        return name

    def get(self, name: object) -> jax.Array:
        """Resolve one allowlisted array and preserve caller-owned precision."""
        key = text(name, "array reference")
        if key not in self.values:
            message = f"Missing artifact array {key!r}."
            raise ArtifactError(message)
        value = self.values[key]
        if value.dtype.itemsize > _DEFAULT_WIDTH and not jax.config.x64_enabled:
            message = (
                "Artifact requires caller-enabled JAX x64; loading never downcasts."
            )
            raise ArtifactError(message)
        self.used.add(key)
        return value

    def require_consumed(self) -> None:
        """Reject arrays unrelated to the declared prediction schema."""
        if self.used != set(self.values):
            message = "Artifact contains unreferenced numeric state."
            raise ArtifactError(message)


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            message = f"Duplicate JSON field {key!r}."
            raise ArtifactError(message)
        result[key] = value
    return result


def _constant(value: str) -> object:
    message = f"Nonfinite JSON constant {value!r} is unsupported."
    raise ArtifactError(message)


def write_bundle(
    path: Path,
    *,
    purpose: Literal["prediction", "analysis"],
    family: str,
    capabilities: tuple[str, ...],
    state: object,
    arrays: Arrays,
    metadata: dict[str, JsonValue],
) -> None:
    """Write arrays first and atomically publish the validated manifest last."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists():
        message = "Refusing to overwrite an existing artifact manifest."
        raise FileExistsError(message)
    chunks = []
    entries: dict[str, ArrayEntry] = {}
    offset = 0
    for name, array in sorted(arrays.values.items()):
        chunk = _encode(array)
        entries[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "offset": offset,
            "bytes": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(),
        }
        chunks.append(chunk)
        offset += len(chunk)
    payload = b"".join(chunks)
    document: Bundle = {
        "schema_version": _SCHEMA_VERSION,
        "writer": {"package": "statadic", "version": version("statadic")},
        "purpose": purpose,
        "family": family,
        "capabilities": list(capabilities),
        "state": state,
        "metadata": metadata,
        "arrays": {
            "filename": "arrays.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "entries": entries,
        },
    }
    encoded = (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n"
    ).encode()
    for filename, content in (("arrays.bin", payload), ("manifest.json", encoded)):
        descriptor, temporary_name = tempfile.mkstemp(dir=target, prefix=".artifact-")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
            temporary.replace(target / filename)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _check_schema(document: object) -> None:
    """Report incompatible versions even when older manifests lack new fields."""
    if (
        isinstance(document, dict)
        and "schema_version" in document
        and (
            type(document["schema_version"]) is not int
            or document["schema_version"] != _SCHEMA_VERSION
        )
    ):
        message = (
            f"Artifact schema version {document['schema_version']!r} is not supported; "
            f"statadic {version('statadic')} reads schema {_SCHEMA_VERSION} only. "
            "Re-save it from the original analysis."
        )
        raise ArtifactError(message)


def _decode_entry(value: object) -> ArrayEntry:
    saved = record(
        value, {"shape", "dtype", "offset", "bytes", "sha256"}, "array entry"
    )
    dtype = text(saved["dtype"], "array dtype")
    if dtype not in _FORMATS:
        message = "Unsupported artifact dtype."
        raise ArtifactError(message)
    return {
        "shape": [
            integer(n, "array dimension") for n in items(saved["shape"], "array shape")
        ],
        "dtype": dtype,
        "offset": integer(saved["offset"], "array offset"),
        "bytes": integer(saved["bytes"], "array bytes"),
        "sha256": text(saved["sha256"], "array digest"),
    }


def _decode_manifest(
    document: object, purpose: Literal["prediction", "analysis"]
) -> Bundle:
    """Establish the envelope and inventory schemas before any payload is decoded."""
    _check_schema(document)
    saved = record(
        document,
        {
            "schema_version",
            "writer",
            "purpose",
            "family",
            "capabilities",
            "state",
            "metadata",
            "arrays",
        },
        "manifest",
    )
    if (
        saved["schema_version"] != _SCHEMA_VERSION
        or type(saved["schema_version"]) is not int
    ):
        message = "Artifact schema version must be 7."
        raise ArtifactError(message)
    if saved["purpose"] != purpose:
        message = f"This loader requires an artifact with purpose={purpose!r}."
        raise ArtifactError(message)
    writer = record(saved["writer"], {"package", "version"}, "writer")
    decoded_writer: WriterRecord = {
        "package": text(writer["package"], "writer package"),
        "version": text(writer["version"], "writer version"),
    }
    inventory = record(
        saved["arrays"], {"filename", "sha256", "bytes", "entries"}, "array inventory"
    )
    if inventory["filename"] != "arrays.bin":
        message = "Artifact payload must be the local arrays.bin file."
        raise ArtifactError(message)
    decoded_inventory: ArrayInventory = {
        "filename": "arrays.bin",
        "sha256": text(inventory["sha256"], "payload digest"),
        "bytes": integer(inventory["bytes"], "payload bytes"),
        "entries": {
            text(name, "array name"): _decode_entry(value)
            for name, value in string_record(
                inventory["entries"], "array entries"
            ).items()
        },
    }
    metadata = metadata_to_json(
        freeze_metadata(string_record(saved["metadata"], "artifact metadata"))
    )
    return {
        "schema_version": 7,
        "writer": decoded_writer,
        "purpose": purpose,
        "family": text(saved["family"], "artifact family"),
        "capabilities": [
            text(item, "capability")
            for item in items(saved["capabilities"], "capabilities")
        ],
        "state": saved["state"],
        "metadata": metadata,
        "arrays": decoded_inventory,
    }


def read_bundle(
    path: Path, *, purpose: Literal["prediction", "analysis"]
) -> tuple[Bundle, Arrays]:
    """Check schema, whole-file integrity and every array header before decoding."""
    target = Path(path)
    try:
        document = json.loads(
            (target / "manifest.json").read_text(encoding="utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        message = "Artifact manifest is missing or malformed."
        raise ArtifactError(message) from error
    try:
        manifest = _decode_manifest(document, purpose)
    except (TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        message = "Artifact manifest contains invalid metadata."
        raise ArtifactError(message) from error
    inventory = manifest["arrays"]
    try:
        payload = (target / "arrays.bin").read_bytes()
    except OSError as error:
        message = "Artifact numeric payload is missing or unreadable."
        raise ArtifactError(message) from error
    if (
        inventory["bytes"] != len(payload)
        or inventory["sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        message = "Artifact numeric integrity check failed."
        raise ArtifactError(message)
    try:
        values = _read_arrays(payload, inventory["entries"])
    except (OSError, ValueError, struct.error) as error:
        if isinstance(error, ArtifactError):
            raise
        message = "Artifact numeric payload is malformed or requires unsafe decoding."
        raise ArtifactError(message) from error
    return manifest, Arrays(values)


def _read_arrays(
    payload: bytes, entries: dict[str, ArrayEntry]
) -> dict[str, jax.Array]:
    """Check bounds, contiguity, scalar formats and integrity before decoding."""
    values = {}
    offset = 0
    for name in sorted(entries):
        entry = entries[name]
        shape = entry["shape"]
        dtype = entry["dtype"]
        code = _FORMATS[dtype]
        width = struct.calcsize("<" + code)
        length = math.prod(shape) * width
        if (
            entry["offset"] != offset
            or entry["bytes"] != length
            or length > len(payload) - offset
        ):
            msg = "Artifact array inventory disagrees with the payload"
            raise ArtifactError(msg)
        chunk = payload[offset : offset + length]
        if entry["sha256"] != hashlib.sha256(chunk).hexdigest():
            msg = "Artifact array integrity mismatch"
            raise ArtifactError(msg)
        if width > _DEFAULT_WIDTH and not jax.config.x64_enabled:
            msg = "Artifact requires caller-enabled JAX x64; loading never downcasts"
            raise ArtifactError(msg)
        arrays = [
            jnp.asarray(
                [
                    item[0]
                    for item in struct.iter_unpack(
                        "<" + code, chunk[start : start + _CODEC_CHUNK * width]
                    )
                ],
                dtype=dtype,
            )
            for start in range(0, length, _CODEC_CHUNK * width)
        ]
        flat = jnp.concatenate(arrays) if arrays else jnp.asarray([], dtype=dtype)
        array = flat.reshape(tuple(shape))
        if not bool(jnp.all(jnp.isfinite(array))):
            msg = "Artifact array is nonfinite"
            raise ArtifactError(msg)
        values[name] = array
        offset += length
    if offset != len(payload):
        msg = "Artifact contains unreferenced payload bytes"
        raise ArtifactError(msg)
    return values
