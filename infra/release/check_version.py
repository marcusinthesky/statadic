"""Check release versions without importing STATADIC or installing dependencies."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
VERSION = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:(a|b|rc)(?:0|[1-9]\d*))?"
)


def check(root: Path, tag: str | None = None) -> bool:
    """Reject inconsistent versions; return whether the release is a prerelease."""
    project = tomllib.loads(
        (root / "src/python/packages/statadic/pyproject.toml").read_text()
    )["project"]
    version = project["version"]
    match = VERSION.fullmatch(version)
    if project["name"] != "statadic" or match is None:
        raise ValueError(f"Unsupported STATADIC release version: {version}")
    lock = tomllib.loads((root / "src/python/uv.lock").read_text())
    packages = [p for p in lock["package"] if p["name"] == "statadic"]
    if len(packages) != 1 or packages[0]["version"] != version:
        raise ValueError("STATADIC version in uv.lock differs from pyproject.toml")
    registry = json.loads((root / "src/lean/registry/public-api.json").read_text())
    if registry["package_version"] != version:
        raise ValueError("Lean public API registry version differs from pyproject.toml")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} must be exactly 'v{version}'")
    return match.group(1) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require this exact v-prefixed package version")
    args = parser.parse_args()
    tag = args.tag
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ["GITHUB_REF_NAME"]
    prerelease = check(ROOT, tag)
    value = f"prerelease={str(prerelease).lower()}\n"
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(value)
    print(value, end="")


if __name__ == "__main__":
    main()
