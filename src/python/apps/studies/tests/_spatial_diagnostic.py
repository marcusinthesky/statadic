"""Failure-only diagnostics for the float32/float64 panel subprocess test."""

import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
REPLAY = Path(__file__).with_name("_panel_replay.py")


def run_with_diagnostic(
    code, environment, *, primary_timeout=60, diagnostic_timeout=45
):
    """Preserve the original subprocess failure after one bounded diagnostic."""
    command = [sys.executable, "-c", code]
    try:
        subprocess.run(command, env=environment, check=True, timeout=primary_timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as primary:
        print("Panel precision test failed; collecting diagnostic rerun", flush=True)
        try:
            x64 = environment["JAX_ENABLE_X64"]
            path = (
                REPOSITORY / "build/ci/panel-diagnostic" / f"x64-{x64}" / "inputs.npz"
            )
            diagnostic_environment = environment | {
                "PANEL_PRIMARY_FAILURE": f"{type(primary).__name__}: {primary}",
            }
            subprocess.run(
                [sys.executable, str(REPLAY), "capture", str(path), x64],
                env=diagnostic_environment,
                check=True,
                timeout=diagnostic_timeout,
            )
        except Exception as error:
            # Preserve the primary failure even if diagnostic capture fails.
            print(f"Panel diagnostic failed: {error}", flush=True)
        raise
