import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from statadic_publishing.manifest import read_manifest, write_manifest
from statadic_publishing.project import read_project

PROJECT = """
[project]
id = "paper"
title = "Paper"
[analysis]
script = "analysis.py"
seed = 7
estimand = "slope"
assumptions = ["independent observations"]
model = { family = "OLS" }
covariance = "HC3"
reference = "normal"
[[inputs]]
id = "data"
path = "data.csv"
sha256 = "abc"
redistributable = true
[[outputs]]
id = "snapshot"
path = "snapshot.json"
media_type = "application/json"
required = true
[manuscript]
root = "manuscript"
main = "main.tex"
bibliography = { path = "../refs.bib", sha256 = "abc" }
tex_bundle = { path = "../bundle.zip", sha256 = "abc" }
check_log = "../check-log"
[environment]
lockfiles = ["../uv.lock"]
[[claims]]
id = "claim"
text = "A claim"
outputs = ["snapshot"]
citations = []
formal = "pending"
source_fidelity = "unavailable"
implementation = "not-run"
qualification = "passed"
human_review = "pending"
"""


def test_project_strict(tmp_path: Path) -> None:
    path = tmp_path / "publishing.toml"
    path.write_text(PROJECT)
    project = read_project(path)
    assert project.analysis.assumptions == ("independent observations",)
    assert project.analysis.snapshots == ()
    assert project.claims[0].human_review == "pending"
    duplicate = (
        '[[outputs]]\nid = "snapshot"\npath = "other.json"\n'
        'media_type = "application/json"\nrequired = false\n'
    )
    path.write_text(PROJECT + duplicate)
    with pytest.raises(ValueError, match="duplicate output ids"):
        read_project(path)
    for old, new in [
        ('outputs = ["snapshot"]', 'outputs = ["missing"]'),
        ('reference = "normal"', 'reference = "normal"\ndiagnostics = ["missing"]'),
        ('script = "analysis.py"', 'script = "../analysis.py"'),
        ('path = "snapshot.json"', 'path = "/snapshot.json"'),
        ('human_review = "pending"', 'human_review = "passed"'),
        ('formal = "pending"', ""),
        ('title = "Paper"', 'title = "Paper"\nextra = 1'),
        ("redistributable = true", "redistributable = false"),
        ('path = "../refs.bib"', 'path = "/refs.bib"'),
        ('path = "../bundle.zip"', 'path = "/bundle.zip"'),
        ('check_log = "../check-log"', 'check_log = "/check-log"'),
        ('lockfiles = ["../uv.lock"]', 'lockfiles = ["/uv.lock"]'),
    ]:
        path.write_text(PROJECT.replace(old, new))
        with pytest.raises(ValueError):
            read_project(path)
    model: Any = project.analysis.model  # Deliberately violate the immutable contract.
    with pytest.raises(TypeError):
        model["family"] = "changed"


def manifest_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "inventory": {},
        "bundle_id": "b",
        "project_id": "p",
        "created_utc": "2026-01-01T00:00:00Z",
        "source": {"revision": "unavailable", "dirty": None, "hashes": []},
        "lockfiles": [],
        "inputs": [],
        "versions": {
            "python": "3.13",
            "statadic": "x",
            "jax": "x",
            "statadic_publishing": "x",
        },
        "runtime": {"platform": "test", "backend": "cpu", "x64": True, "seed": 7},
        "analyses": [
            {
                "script": "analysis.py",
                "estimand": "slope",
                "model": {},
                "covariance": "HC3",
                "reference": "normal",
                "inputs": [],
                "assumptions": ["authored"],
                "snapshots": ["snapshot"],
                "diagnostics": [],
            }
        ],
        "outputs": [
            {
                "id": "snapshot",
                "media_type": "application/json",
                "path": "snapshot.json",
                "sha256": "abc",
                "status": "available",
            }
        ],
        "claims": [
            {
                "id": "c",
                "text": "text",
                "outputs": ["snapshot"],
                "citations": [],
                "registry_claims": [],
                "formal": "pending",
                "source_fidelity": "unavailable",
                "implementation": "not-run",
                "qualification": "failed",
                "human_review": "pending",
            }
        ],
    }


def test_manifest_roundtrip_and_rejections(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    data = manifest_data()
    path.write_text(json.dumps(data))
    manifest = read_manifest(path)
    write_manifest(path, manifest)
    assert read_manifest(path) == manifest
    assert manifest.analyses[0].assumptions == ("authored",)
    assert manifest.claims[0].qualification == "failed"
    model: Any = manifest.analyses[0].model  # Deliberately attempt mutation.
    with pytest.raises(TypeError):
        model["family"] = "changed"
    changes: tuple[Callable[[dict[str, Any]], object], ...] = (
        lambda d: d.update(schema_version=2),
        lambda d: d.update(extra=1),
        lambda d: d.pop("runtime"),
        lambda d: d["outputs"][0].pop("status"),
        lambda d: d["outputs"][0].update(status="unknown"),
        lambda d: d["outputs"][0].update(path="../outside"),
    )
    for change in changes:
        data = manifest_data()
        change(data)
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            read_manifest(path)


def test_authored_model_metadata_is_accepted_and_frozen() -> None:
    """Records accept authored lists and retain immutable metadata."""
    from statadic_publishing.manifest import AnalysisRecord  # noqa: PLC0415
    from statadic_publishing.project import Analysis  # noqa: PLC0415

    analysis = Analysis(
        "analysis.py", 1, "slope", (), {"terms": ["x", "z"]}, "HC3", "normal", (), ()
    )
    record = AnalysisRecord(
        "analysis.py", "slope", {"terms": ["x", "z"]}, "HC3", "normal", (), (), (), ()
    )
    assert analysis.model["terms"] == ("x", "z")
    assert record.model == analysis.model
