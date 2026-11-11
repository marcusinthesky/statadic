# standalone package-maintenance entry point
"""Run static contract fixtures with Pyrefly's native diagnostic expectations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def ill_typed(value: object) -> Any:  # noqa: ANN401  # the purpose of this helper
    """Return ``value`` unchanged so a test can reach a runtime rejection.

    Use only where a test deliberately breaks a static contract, to show the
    runtime also refuses it. A call that is valid needs a better type, not this.

    Args:
        value: The deliberately invalid value, callable or object.

    Returns:
        The same object, typed as unknown to the checker.

    """
    return value


def check_types(fixture: str, name: str, tmp_path: Path) -> None:
    """Require exactly the diagnostics marked with inline ``# E:`` comments.

    Args:
        fixture: Source text, with no expectations for an accepted fixture.
        name: Temporary Python module name.
        tmp_path: Directory in which to check the fixture.

    """
    path = tmp_path / name
    path.write_text(fixture, encoding="utf-8")
    result = subprocess.run(  # fixed checker and interpreter  # noqa: S603
        [
            str(Path(sys.executable).with_name("pyrefly")),
            "check",
            "-c",
            "/dev/null",
            "--python-interpreter-path",
            sys.executable,
            "--expectations",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
