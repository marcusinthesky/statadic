# ruff: noqa: INP001  # standalone package-maintenance entry point
"""Retain the installed-wheel interpreter and complete compatibility resolution."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import statadic


def main() -> None:
    """Reject checkout imports and write a small machine-readable environment record."""
    location = Path(statadic.__file__).resolve()
    if "site-packages" not in location.parts:
        message = f"Compatibility checks require an installed wheel: {location}"
        raise RuntimeError(message)
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package": str(location),
        "precision": os.environ.get("JAX_ENABLE_X64"),
        "platforms": os.environ.get("JAX_PLATFORMS"),
        "distributions": dict(
            sorted(
                (distribution.metadata["Name"].lower(), distribution.version)
                for distribution in metadata.distributions()
            )
        ),
    }
    Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
