"""Execute each real notebook in an isolated process against installed STATADIC."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
LABS = json.loads((APP / "labs.json").read_text())["labs"]
# Dynamic rendered values prove the submit-gated computation ran in snapshot mode.
SUBMITTED_RESULTS = {
    "permutation-evidence": "n=32 per group",
    "distributional-workflow": "Exploratory held-out RMSE",
    "calibration": "20 trials per scenario",
    "study-results": "20 trials per scenario",
}
pytestmark = pytest.mark.notebook


@pytest.mark.parametrize("lab", LABS, ids=[lab["slug"] for lab in LABS])
def test_notebook_snapshot(lab, tmp_path):
    environment = {**os.environ, "STATADIC_LABS_SNAPSHOT": "1", "MPLBACKEND": "Agg"}
    notebook = APP / "notebooks" / lab["notebook"]
    output = tmp_path / "snapshot.html"
    if lab["slug"] in SUBMITTED_RESULTS:
        command = [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(notebook),
            "--no-include-code",
            "-o",
            str(output),
        ]
    else:
        command = [sys.executable, str(notebook)]
    completed = subprocess.run(
        command,
        cwd=APP,
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    if lab["slug"] in SUBMITTED_RESULTS:
        assert SUBMITTED_RESULTS[lab["slug"]] in output.read_text()
