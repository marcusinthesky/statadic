"""Deterministic evidence serialization and exact, safe file inventories.

This package uses only the standard library. Inventories exclude exact relative
paths, never basename matches: a nested ``manifest.json`` remains evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    """Hash a regular file without following a symbolic link."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: object) -> str:
    """Hash the canonical, finite JSON representation of a value."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically replace a JSON file, retaining actual newlines and UTF-8."""
    payload = _json_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"Unsafe evidence path: {value!r}")
    return value


def _entries(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected evidence directory: {root}")
    entries = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    for path in entries:
        _relative(path.relative_to(root).as_posix())
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Unsupported evidence entry: {path}")
    return entries


def inventory(
    root: Path, exclude: tuple[str, ...] = ("manifest.json",)
) -> dict[str, str]:
    """Hash all files, including hidden files, except exact root-relative names."""
    root = Path(root)
    exclusions = {_relative(name) for name in exclude}
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in _entries(root)
        if path.is_file() and path.relative_to(root).as_posix() not in exclusions
    }


def verify_inventory(
    root: Path, expected: dict[str, str], exclude: tuple[str, ...] = ("manifest.json",)
) -> None:
    """Reject changed, missing or unlisted files and unsafe inventory paths."""
    for name, digest in expected.items():
        _relative(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Invalid evidence digest: {name}")
    if inventory(root, exclude) != expected:
        raise ValueError("Evidence file inventory changed")


def write_archive(root: Path, destination: Path) -> None:
    """Write deterministic uncompressed tar, including the root manifest.

    Callers verify scientific completeness and their manifest before archiving.
    The archive destination must be outside the evidence directory.
    """
    root, destination = Path(root), Path(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive must be outside evidence directory")
    entries = _entries(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Archive destination is a symbolic link")
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".archive.")
    try:
        with (
            os.fdopen(descriptor, "wb") as stream,
            tarfile.open(
                fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive,
        ):
            for path in entries:
                info = tarfile.TarInfo(path.relative_to(root).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                if path.is_dir():
                    info.type, info.mode = tarfile.DIRTYPE, 0o755
                    archive.addfile(info)
                else:
                    payload = path.read_bytes()
                    info.mode, info.size = 0o644, len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def extract_archive(source: Path, destination: Path) -> None:
    """Validate an entire archive before extracting into a new directory.

    Reject links, traversal, duplicate members and special files. Extraction into
    an existing directory is prohibited so existing links cannot redirect writes.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Extraction destination must not exist")
    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        files: set[str] = set()
        for member in members:
            name = _relative(member.name)
            if name in names or not (member.isdir() or member.isfile()):
                raise ValueError(f"Duplicate or unsupported archive member: {name}")
            names.add(name)
            if member.isfile():
                files.add(name)
        for name in names:
            if any(
                parent.as_posix() in files for parent in PurePosixPath(name).parents
            ):
                raise ValueError(f"Archive file is a parent directory: {name}")
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Missing archive payload: {member.name}")
                with extracted, target.open("xb") as output:
                    import shutil

                    shutil.copyfileobj(extracted, output)
