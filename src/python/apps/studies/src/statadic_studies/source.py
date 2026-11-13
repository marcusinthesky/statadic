"""Captured imported Python sources for reproducible scientific runner provenance."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath

import statadic
import statadic_evidence
from statadic_evidence import fingerprint


def capture() -> dict[str, bytes]:
    """Read imported runtime Python sources recursively, excluding configuration."""
    roots = {
        "statadic": Path(statadic.__file__).parent,
        "statadic_studies": Path(__file__).parent,
        "statadic_evidence": Path(statadic_evidence.__file__).parent,
    }
    return {
        f"{name}/{path.relative_to(root).as_posix()}": path.read_bytes()
        for name, root in roots.items()
        for path in sorted(root.rglob("*.py"))
    }


def hashes(files: dict[str, bytes]) -> dict[str, str]:
    """Fingerprint exact retained bytes, not regenerated representations."""
    return {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }


def identities(files: dict[str, str]) -> dict[str, str]:
    """Use the same per-package relative path aggregation as execution provenance."""
    return {
        key: fingerprint(
            {
                name.split("/", 1)[1]: digest
                for name, digest in files.items()
                if name.startswith(package + "/")
            }
        )
        for package, key in (
            ("statadic", "statadic_source"),
            ("statadic_studies", "engine_source"),
            ("statadic_evidence", "evidence_source"),
        )
    }


def retain(root: Path, files: dict[str, bytes]) -> None:
    """Publish each captured source atomically without modifying existing evidence."""
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)


def verify(root: Path, capsule: dict, environment: dict) -> None:
    """Reject missing, altered, additional or unsafe retained Python source bytes."""
    if capsule.get("schema_version") != 1:
        raise ValueError("Unsupported source capsule")
    expected = capsule["files"]
    for name in expected:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[0]
            not in {"statadic", "statadic_studies", "statadic_evidence"}
        ):
            raise ValueError("Unsafe source capsule path")
    paths = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]
    if any(path.is_symlink() for path in paths) or set(expected) != {
        path.relative_to(root).as_posix() for path in paths
    }:
        raise ValueError("Source capsule inventory changed")
    actual = hashes({name: (root / name).read_bytes() for name in expected})
    if actual != expected or any(
        environment.get(key, value) != value
        for key, value in identities(actual).items()
    ):
        raise ValueError("Source capsule bytes or execution identities changed")
