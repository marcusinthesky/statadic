"""Fail before Lake can fetch dependencies during an ordinary local check."""

import json
import subprocess
import sys
from pathlib import Path


def output(*command: str) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("Enter the pinned devenv shell and run just lean::setup.") from None


def main() -> None:
    root = Path(__file__).resolve().parent
    version = (root / "lean-toolchain").read_text().strip().rsplit(":v", 1)[1]
    if not output("lean", "--version").startswith(f"Lean (version {version},"):
        raise SystemExit(f"Lean {version} is required; enter the pinned devenv shell.")
    if "--toolchain-only" in sys.argv:
        return
    manifest = json.loads((root / "lake-manifest.json").read_text())
    for package in manifest["packages"]:
        path = root / manifest["packagesDir"] / package["name"]
        if (
            not (path / ".git").exists()
            or output("git", "-C", str(path), "rev-parse", "HEAD") != package["rev"]
        ):
            raise SystemExit(f"{package['name']} is missing or changed; run just lean::setup.")


if __name__ == "__main__":
    main()
