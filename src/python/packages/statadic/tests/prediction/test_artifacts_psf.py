"""Closed artifact contracts for Wasserstein geometry and orthogonal coordinate maps."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.measure.clouds import PackedClouds
from statadic.observation.geometry import Wasserstein
from statadic.ordination.fit import Ordination
from statadic.prediction.artifacts import ArtifactError, load_ordination_predictor
from statadic.prediction.artifacts._ordination import (
    decode_coordinate_map,
    decode_coordinate_record,
)
from statadic.prediction.artifacts._store import Arrays

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.prediction.artifacts._schema import LinearMapRecord

pytestmark = [pytest.mark.integration, pytest.mark.contract]


def _record(document: dict[str, object], *keys: str) -> dict[str, object]:
    """Traverse known JSON records without changing their declared structure."""
    for key in keys:
        value = document[key]
        assert isinstance(value, dict)
        document = value
    return document


@pytest.fixture(scope="module")
def rotated_wasserstein_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save one real rotated map for manifest validation independent of refitting."""
    clouds = PackedClouds.from_clouds(
        [
            [[0.0, 0.0], [0.5, 0.0]],
            [[1.0, 0.0], [1.5, 0.0]],
            [[0.0, 1.0], [0.5, 1.0]],
        ]
    )
    fit = Ordination.pcoa(
        Wasserstein(p=2, ground=EUCLIDEAN),
        dim=2,
        negative_inertia="positive-subspace",
    ).estimate(clouds)
    path = tmp_path_factory.mktemp("wasserstein-artifact")
    fit.rotate(jnp.asarray([[0.0, -1.0], [1.0, 0.0]])).save_predictor(path)
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("p", 0.5),
        ("p", -2.0),
        ("p", float("nan")),
        ("p", float("inf")),
        ("p", "2"),
        ("p", True),
        ("p", None),
        ("max_iterations", 0),
        ("max_iterations", -1),
        ("max_iterations", 2.5),
        ("max_iterations", "100"),
        ("max_iterations", True),
        ("tolerance", 0.0),
        ("tolerance", -1e-6),
        ("tolerance", float("inf")),
        ("tolerance", "0.000001"),
        ("tolerance", False),
        ("unexpected", 1),
    ],
)
def test_rotated_ordination_rejects_invalid_wasserstein_settings(
    tmp_path: Path,
    rotated_wasserstein_artifact: Path,
    field: str,
    value: object,
) -> None:
    """Loaded numerical domains, JSON types and extra fields fail as ArtifactError."""
    path = tmp_path / "corrupted"
    shutil.copytree(rotated_wasserstein_artifact, path)
    manifest = path / "manifest.json"
    document = json.loads(manifest.read_text())
    config = _record(document, "state", "base", "geometry", "config")
    config[field] = value
    manifest.write_text(json.dumps(document))
    with pytest.raises(ArtifactError):
        load_ordination_predictor(path)


def _linear_record(arrays: Arrays) -> LinearMapRecord:
    """Construct a valid two-dimensional base independently of the wire decoder."""
    return {
        "kind": "linear",
        "feature_names": ["x", "z"],
        "mean": arrays.add(jnp.zeros(2)),
        "scale": arrays.add(jnp.ones(2)),
        "projection_mean": arrays.add(jnp.zeros(2)),
        "components": arrays.add(jnp.eye(2)),
    }


@pytest.mark.parametrize(
    "rotation",
    [
        [[2.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.25], [0.0, 1.0]],
        [[1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0],
    ],
    ids=["scale", "shear", "wrong-width", "wrong-square-width", "vector"],
)
def test_coordinate_decoder_rejects_nonorthogonal_or_misaligned_rotation(
    rotation: object,
) -> None:
    """Valid numeric payload references cannot authorize a corrupt coordinate map."""
    arrays = Arrays()
    wire = {
        "kind": "orthogonal",
        "base": _linear_record(arrays),
        "rotation": arrays.add(jnp.asarray(rotation)),
    }
    saved = decode_coordinate_record(wire)
    with pytest.raises(ArtifactError, match="orthogonal rotation"):
        decode_coordinate_map(saved, arrays)


def test_coordinate_wire_rejects_nested_orthogonal_wrappers() -> None:
    """The flattened orthogonal wire contract cannot be bypassed by recursion."""
    arrays = Arrays()
    rotation = arrays.add(jnp.eye(2))
    base = {"kind": "orthogonal", "base": _linear_record(arrays), "rotation": rotation}
    nested = {"kind": "orthogonal", "base": base, "rotation": rotation}
    with pytest.raises(ArtifactError, match="Nested orthogonal"):
        decode_coordinate_record(nested)
