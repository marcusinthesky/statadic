"""Exercise the CPU44 verifier CLI on deterministic synthetic worker records."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1]
FIXTURES = APP / "history" / "cpu44"


@pytest.fixture
def replay(tmp_path):
    run = tmp_path / "run"
    results = run / "results"
    results.mkdir(parents=True)
    hashes = {}
    for src in [*FIXTURES.glob("*.py"), FIXTURES / "lock-constraints.txt"]:
        shutil.copyfile(src, run / src.name)
        hashes[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    for side in ("main", "peer"):
        shutil.copyfile(run / "lock-constraints.txt", run / f"{side}-environment.txt")
    with (APP / "results/cpu44/summary.csv").open() as f:
        ids = [r["case_id"] for r in csv.DictReader(f)]
    manifest = dict(
        revision="a" * 40,
        fixture_sha256=hashes,
        lanes={"2": ids},
        platform="test",
        cpu_model="test",
        python="test",
        trials=3,
        warm_calls=5,
    )
    (run / "manifest.json").write_text(json.dumps(manifest))
    for case_id in ids:
        module, case = case_id.split("/")
        for trial in range(3):
            for side in ("statadic", "reference"):
                stem = f"{module}.{case}.{side}.{trial}"
                ms = 1 if side == "statadic" else 2
                if case_id == ids[1] and trial == 2 and side == "statadic":
                    ms = 3
                row = dict(
                    module=module,
                    case=case,
                    side=side,
                    trial=trial,
                    cpu=[2],
                    ms=[4, *([ms] * 5)],
                    baseline_kib=100,
                    rss_kib=120,
                )
                (results / f"{stem}.json").write_text(json.dumps(row))
                np.save(results / f"{stem}.npy", np.array([1.0]))
    return run


def summarize(run, output):
    return subprocess.run(
        [
            sys.executable,
            str(APP / "scripts/summarize_cpu44.py"),
            "--parent",
            str(run),
            "--candidate",
            str(run),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )


def test_consistent_and_mixed_trials(replay, tmp_path):
    output = tmp_path / "tables"
    proc = summarize(replay, output)
    assert proc.returncode == 0, proc.stderr
    with (output / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 44
    assert (rows[0]["warm_vs_reference"], rows[1]["warm_vs_reference"]) == (
        "faster",
        "mixed",
    )
    assert float(rows[1]["candidate_over_reference_warm"]) == 0.5


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing", "Missing or unexpected worker results"),
        ("nan", "finite, positive timings"),
        ("output", "Numerical comparison failed"),
    ],
)
def test_corrupt_replay_cannot_publish_tables(replay, tmp_path, corruption, message):
    target = replay / "results/spatial_cases.sar.statadic.0.json"
    if corruption == "missing":
        target.unlink()
    elif corruption == "nan":
        row = json.loads(target.read_text())
        row["ms"][0] = float("nan")
        target.write_text(json.dumps(row))
    else:
        np.save(target.with_suffix(".npy"), np.array([99.0]))
    output = tmp_path / "tables"
    proc = summarize(replay, output)
    assert proc.returncode != 0
    assert message in proc.stderr
    assert not output.exists()
