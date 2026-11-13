"""Schema-six frozen ordination and nested regression prediction transport."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import (
    ArtifactError,
    load_ordination_predictor,
    load_predictor,
    save_ordination_predictor,
    save_predictor,
)
from statadic.api.data import Observations
from statadic.api.design import FittedDesign, Intercept
from statadic.api.ordination import (
    DistanceGram,
    KernelGram,
    LinearGram,
    Ordination,
    PrincipalCoordinates,
    ShortestPathDistances,
)
from statadic.api.ordination_terms import (
    GeodesicCoordinateMap,
    KernelCoordinateMap,
    LinearCoordinateMap,
)
from statadic.api.regression import LinearResult, Regression
from statadic.api.representation import MMD, Energy, ObservedDissimilarity
from statadic.geometry.ground.similarities import LINEAR_KERNEL, RBFKernel, gram
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration
_SCHEMA_VERSION = 7


def _model(kind: str) -> Ordination:
    features = ("x", "z")
    geometry: LinearGram | KernelGram | DistanceGram
    if kind == "linear":
        geometry = LinearGram(features, scale=True)
    elif kind == "kernel":
        geometry = KernelGram(RBFKernel(1.2), features)
    else:
        geometry = DistanceGram(ShortestPathDistances(2, features))
    return Ordination(geometry, PrincipalCoordinates(2, "positive-subspace"))


def _training() -> Observations:
    return Observations(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "z": [0.0, 0.2, 0.8, 1.8, 3.2, 5.0]},
        row_ids=("a", "b", "c", "d", "e", "f"),
    )


@pytest.mark.contract
@pytest.mark.parametrize("kind", ["linear", "kernel", "isomap"])
def test_frozen_ordination_roundtrip_preserves_transform_only_capabilities(
    tmp_path: Path, kind: str
) -> None:
    """Every supported retained map restores identical held-out coordinates."""
    fit = _model(kind).estimate(_training())
    query = Observations({"z": [0.4, 2.5], "x": [1.5, 3.5]})
    path = tmp_path / kind
    fit.save_predictor(path)
    restored = load_ordination_predictor(path)
    assert restored.capabilities == fit.capabilities
    assert not hasattr(restored, "estimate")
    assert not hasattr(restored, "infer")
    np.testing.assert_array_equal(restored.transform(query), fit.transform(query))
    np.testing.assert_allclose(
        restored.transform(query.take([1])), fit.transform(query)[1:2], atol=1e-12
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["schema_version"] == _SCHEMA_VERSION


@pytest.mark.contract
@pytest.mark.parametrize("kind", ["linear", "kernel", "isomap"])
def test_nested_ordination_regression_roundtrip_keeps_design_and_predictions(
    tmp_path: Path, kind: str
) -> None:
    """A frozen nested design carries its actual fitted map, not a refit recipe."""
    training = _training()
    specification = Intercept() + _model(kind)
    design = FittedDesign(specification._prepare_term(training))
    y = np.asarray([1.0, 2.0, 1.5, 3.2, 4.1, 5.7])
    coefficients = np.linalg.lstsq(design.transform(training), y, rcond=None)[0]
    path = tmp_path / kind
    save_predictor(path, design=design, parameters=coefficients, family="linear")
    restored = load_predictor(path)
    query = Observations({"x": [1.5, 3.5], "z": [0.4, 2.5]})
    np.testing.assert_allclose(
        restored.predict(query), design.transform(query) @ coefficients, atol=1e-12
    )
    assert restored.design.column_names == design.column_names


@pytest.mark.contract
@pytest.mark.parametrize("kind", ["kernel", "isomap"])
def test_unknown_map_fields_and_false_capability_are_rejected(
    tmp_path: Path, kind: str
) -> None:
    """Valid numerical hashes never authorize extra executable schema fields."""
    fit = _model(kind).estimate(_training())
    path = tmp_path / kind
    fit.save_predictor(path)
    manifest = path / "manifest.json"
    original = json.loads(manifest.read_text())
    corrupted = json.loads(manifest.read_text())
    corrupted["state"]["import"] = "module.Class"
    manifest.write_text(json.dumps(corrupted))
    with pytest.raises(ArtifactError, match="schema"):
        load_ordination_predictor(path)
    original["capabilities"].append("infer")
    manifest.write_text(json.dumps(original))
    with pytest.raises(ArtifactError, match="capabilit"):
        load_ordination_predictor(path)


@pytest.mark.contract
def test_geodesic_codec_rejects_invalid_neighbor_budget(tmp_path: Path) -> None:
    """A valid payload cannot grant an impossible retained graph attachment."""
    fit = _model("isomap").estimate(_training())
    assert isinstance(fit.mapping, GeodesicCoordinateMap)
    path = tmp_path / "graph"
    fit.save_predictor(path)
    manifest = path / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["state"]["neighbors"] = len(fit.row_ids)
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ArtifactError, match="neighbours"):
        load_ordination_predictor(path)


@pytest.mark.contract
def test_fitted_kernel_snapshots_a_mutable_bandwidth_buffer() -> None:
    """Later host mutation cannot change the scientific kernel of a fitted map."""
    bandwidth = np.asarray(1.2)
    model = Ordination(
        KernelGram(RBFKernel(bandwidth), ("x", "z")), PrincipalCoordinates(2)
    )
    fitted = model.estimate(_training())
    assert isinstance(fitted.mapping, KernelCoordinateMap)
    original = fitted.transform(_training())
    bandwidth[...] = 100.0
    np.testing.assert_array_equal(fitted.transform(_training()), original)


@pytest.mark.contract
def test_feature_artifact_retains_scale_and_reports_unrepresentable_queries(
    tmp_path: Path,
) -> None:
    """Finite retained state does not make an overflowing query available."""
    fit = _model("linear").estimate(_training())
    assert isinstance(fit.mapping, LinearCoordinateMap)
    mapping = replace(fit.mapping, mean=jnp.full(2, -1e308))
    path = tmp_path / "overflow"
    save_ordination_predictor(path, mapping=mapping, metadata={})
    restored = load_ordination_predictor(path)
    with pytest.raises(ValueError, match="representable"):
        restored.transform(np.full((1, 2), 1e308))


@pytest.mark.contract
@pytest.mark.parametrize("constructed", [True, False])
def test_distance_artifact_preserves_geometry_reference_order_and_query_domain(
    tmp_path: Path, *, constructed: bool
) -> None:
    """Observed matrices grant only labelled cross-distance extension."""
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.2], [0.5, 2.0], [3.0, 4.0], [2.0, -1.0]])
    query = jnp.asarray([[1.1, 2.2], [-4.0, 7.0]])
    ids = (0, "a", 1, "b", 2)
    measures = PackedClouds.from_clouds([row[None] for row in points])
    geometry = MMD(LINEAR_KERNEL).prepare(measures, row_ids=ids)
    distances = jnp.linalg.norm(points[:, None] - points[None], axis=-1)
    fit = Ordination(DistanceGram(), PrincipalCoordinates(2)).estimate(
        geometry if constructed else ObservedDissimilarity.from_array(distances, ids)
    )
    path = tmp_path / "distance"
    fit.save_predictor(path)
    restored = load_ordination_predictor(path)
    cross = jnp.linalg.norm(query[:, None] - points[None], axis=-1)
    np.testing.assert_array_equal(
        restored.transform_distances(cross, reference_ids=ids),
        fit.transform_distances(cross, reference_ids=ids),
    )
    with pytest.raises(ValueError, match="reference IDs"):
        restored.transform_distances(cross, reference_ids=ids[::-1])
    with pytest.raises(ValueError, match="representable"):
        restored.transform_distances(np.full((1, len(ids)), 1e200), reference_ids=ids)
    integer_cross = np.full((1, len(ids)), 10**10, dtype=np.int64)
    np.testing.assert_array_equal(
        restored.transform_distances(integer_cross, reference_ids=ids),
        restored.transform_distances(integer_cross.astype(float), reference_ids=ids),
    )
    query_measures = PackedClouds.from_clouds([row[None] for row in query])
    if constructed:
        np.testing.assert_array_equal(
            restored.transform(query_measures), fit.transform(query_measures)
        )
        assert restored.capabilities == ("transform", "transform_distances")
    else:
        assert restored.capabilities == ("transform_distances",)
        with pytest.raises(ValueError, match="cross distances"):
            restored.transform(query_measures)
    with pytest.raises(ArtifactError, match="approved regression"):
        load_predictor(path)


@pytest.mark.contract
def test_linear_map_codec_rejects_nonpositive_retained_scale(tmp_path: Path) -> None:
    """A declarative map still needs valid training-scale state."""
    fit = _model("linear").estimate(_training())
    assert isinstance(fit.mapping, LinearCoordinateMap)
    invalid = replace(fit.mapping, scale=jnp.asarray([1.0, 0.0]))
    with pytest.raises(ArtifactError, match="scale"):
        save_ordination_predictor(tmp_path / "invalid", mapping=invalid, metadata={})


@pytest.mark.contract
@pytest.mark.parametrize("bandwidth_dtype", [jnp.float32, jnp.float64])
def test_kernel_artifact_preserves_strong_bandwidth_precision(
    tmp_path: Path, bandwidth_dtype: type
) -> None:
    """A narrow feature matrix does not narrow a retained strong scalar kernel."""
    with jax.enable_x64(new_val=True):
        training = jnp.asarray(
            [[0.0, 0.0], [1.0, 0.2], [0.5, 2.0], [3.0, 4.0], [2.0, -1.0]],
            dtype=jnp.float32,
        )
        query = jnp.asarray([[1.1, 2.2], [-4.0, 7.0], [4.0, -2.0]], dtype=jnp.float32)
        bandwidth = jnp.asarray(1.3, dtype=bandwidth_dtype)
        fit = Ordination(
            KernelGram(RBFKernel(bandwidth)), PrincipalCoordinates(2)
        ).estimate(training)
        assert isinstance(fit.mapping, KernelCoordinateMap)
        fit.save_predictor(tmp_path)
        restored = load_ordination_predictor(tmp_path)
        assert isinstance(restored.mapping, KernelCoordinateMap)
        assert isinstance(restored.mapping.kernel, RBFKernel)
        retained = jnp.asarray(restored.mapping.kernel.bandwidth)
        assert retained.dtype == bandwidth.dtype
        assert not retained.weak_type
        np.testing.assert_array_equal(retained, bandwidth)
        original_gram = gram(query, training, kernel=fit.mapping.kernel)
        restored_gram = gram(query, training, kernel=restored.mapping.kernel)
        assert restored_gram.dtype == original_gram.dtype
        np.testing.assert_array_equal(restored_gram, original_gram)
        np.testing.assert_array_equal(restored.transform(query), fit.transform(query))
        np.testing.assert_array_equal(
            jax.jit(restored.mapping.project_values)(query), fit.transform(query)
        )


@pytest.mark.contract
def test_kernel_artifact_preserves_weak_float32_under_wide_default(
    tmp_path: Path,
) -> None:
    """Loading retains weak dtype even when it differs from the reader default."""
    with jax.enable_x64(new_val=False):
        bandwidth = jnp.asarray(1.3)
    assert bandwidth.dtype == jnp.float32
    assert bandwidth.weak_type
    with jax.enable_x64(new_val=True):
        training = jnp.asarray([[0.0], [1.0], [2.0], [4.0]], dtype=jnp.float32)
        fit = Ordination(
            KernelGram(RBFKernel(bandwidth)), PrincipalCoordinates(2)
        ).estimate(training)
        fit.save_predictor(tmp_path)
        restored = load_ordination_predictor(tmp_path)
        assert isinstance(restored.mapping, KernelCoordinateMap)
        assert isinstance(restored.mapping.kernel, RBFKernel)
        retained = jnp.asarray(restored.mapping.kernel.bandwidth)
        assert retained.dtype == bandwidth.dtype
        assert retained.weak_type
        # Strongifying the stored scalar would change promotion against a
        # narrower floating query, even though its numerical value is unchanged.
        narrow = jnp.asarray(1.0, dtype=jnp.float16)
        assert jnp.result_type(narrow, retained) == jnp.result_type(narrow, bandwidth)
        np.testing.assert_array_equal(
            restored.transform(training), fit.transform(training)
        )


@pytest.mark.contract
def test_kernel_artifact_accepts_bandwidth_outside_reference_dtype(
    tmp_path: Path,
) -> None:
    """A valid promoted kernel need not fit its scalar into the feature dtype."""
    with jax.enable_x64(new_val=True):
        training = jnp.zeros((3, 1), dtype=jnp.float32)
        bandwidth = jnp.asarray(1e-100, dtype=jnp.float64)
        fit = Ordination(
            KernelGram(RBFKernel(bandwidth)), PrincipalCoordinates(None)
        ).estimate(training)
        assert fit.coordinates.shape == (3, 0)
        fit.save_predictor(tmp_path)
        restored = load_ordination_predictor(tmp_path)
        assert isinstance(restored.mapping, KernelCoordinateMap)
        assert isinstance(restored.mapping.kernel, RBFKernel)
        retained = jnp.asarray(restored.mapping.kernel.bandwidth)
        assert retained.dtype == bandwidth.dtype
        np.testing.assert_array_equal(retained, bandwidth)
        np.testing.assert_array_equal(
            restored.transform(training), fit.transform(training)
        )


@pytest.mark.contract
def test_pcr_artifact_preserves_mixed_precision_kernel_predictions(
    tmp_path: Path,
) -> None:
    """The fitted regression uses the same scalar semantics after nested loading."""
    with jax.enable_x64(new_val=True):
        training = Observations(
            {
                "x": jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
                "z": jnp.asarray([0.0, 0.2, 0.8, 1.8, 3.2, 5.0], dtype=jnp.float32),
                "y": jnp.asarray([1.0, 2.0, 1.5, 3.2, 4.1, 5.7], dtype=jnp.float32),
            }
        )
        components = Ordination(
            KernelGram(RBFKernel(jnp.asarray(1.3, dtype=jnp.float64)), ("x", "z")),
            PrincipalCoordinates(2),
        )
        fit = Regression("y", Intercept() + components).estimate(training)
        assert isinstance(fit, LinearResult)
        query = Observations(
            {
                "x": jnp.asarray([1.5, 3.5], dtype=jnp.float32),
                "z": jnp.asarray([0.4, 2.5], dtype=jnp.float32),
            }
        )
        fit.save_predictor(tmp_path)
        restored = load_predictor(tmp_path)
        np.testing.assert_array_equal(
            restored.design.transform(query), fit.design.transform(query)
        )
        np.testing.assert_array_equal(restored.predict(query), fit.predict(query))


@pytest.mark.contract
def test_measure_geometry_artifact_preserves_mixed_precision_kernel(
    tmp_path: Path,
) -> None:
    """Distance maps and feature maps share the same retained kernel codec."""
    with jax.enable_x64(new_val=True):
        training = PackedClouds.from_clouds(
            [
                jnp.asarray([[value]], dtype=jnp.float32)
                for value in (0.0, 1.0, 2.0, 4.0)
            ]
        )
        geometry = MMD(RBFKernel(jnp.asarray(1.3, dtype=jnp.float64))).prepare(training)
        fit = Ordination(DistanceGram(), PrincipalCoordinates(2)).estimate(geometry)
        query = PackedClouds.from_clouds(
            [jnp.asarray([[value]], dtype=jnp.float32) for value in (0.4, 2.5)]
        )
        fit.save_predictor(tmp_path)
        restored = load_ordination_predictor(tmp_path)
        np.testing.assert_array_equal(restored.transform(query), fit.transform(query))


@pytest.mark.contract
def test_ordination_loader_wraps_reconstructed_geometry_domain_error(
    tmp_path: Path,
) -> None:
    """Malformed geometry options preserve a uniform artifact exception boundary."""
    training = PackedClouds.from_clouds(
        [jnp.asarray([[value]]) for value in (0.0, 1.0, 2.0, 4.0)]
    )
    fit = Ordination(DistanceGram(Energy()), PrincipalCoordinates(2)).estimate(training)
    fit.save_predictor(tmp_path)
    manifest = tmp_path / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["state"]["geometry"]["config"]["exponent"] = 0
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ArtifactError, match="retained ordination") as error:
        load_ordination_predictor(tmp_path)
    assert type(error.value.__cause__) is ValueError


@pytest.mark.contract
@pytest.mark.parametrize("weak_type", [0, 1, "false", None])
def test_kernel_codec_rejects_nonboolean_promotion_flag(
    tmp_path: Path, weak_type: object
) -> None:
    """Array integrity does not authorize an ambiguous scalar promotion policy."""
    fit = _model("kernel").estimate(_training())
    fit.save_predictor(tmp_path)
    manifest = tmp_path / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["state"]["kernel"]["weak_type"] = weak_type
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ArtifactError, match="promotion flag"):
        load_ordination_predictor(tmp_path)


@pytest.mark.contract
def test_kernel_codec_rejects_nonscalar_bandwidth_payload(tmp_path: Path) -> None:
    """A real feature matrix is not valid scalar state even with correct hashes."""
    fit = _model("kernel").estimate(_training())
    fit.save_predictor(tmp_path)
    manifest = tmp_path / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["state"]["kernel"]["bandwidth"] = saved["state"]["reference"]
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ArtifactError, match="positive real scalar"):
        load_ordination_predictor(tmp_path)
