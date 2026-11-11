"""Independent spectral, held-out and nested-design ordination witnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations
from statadic.api.design import FittedDesign, Intercept, Linear
from statadic.api.ordination import (
    DistanceGram,
    KernelGram,
    LinearGram,
    Ordination,
    PrincipalCoordinates,
    ShortestPathDistances,
)
from statadic.api.ordination_terms import (
    FittedOrdination,
    KernelCoordinateMap,
    LinearCoordinateMap,
)
from statadic.api.regression import Regression
from statadic.api.representation import Energy, ObservedDissimilarity, SlicedWasserstein
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.geometry.embedding import NonEuclideanEmbeddingError
from statadic.geometry.ground.similarities import LINEAR_KERNEL, RBFKernel
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from .._validation.linear import LinearCase
from hypothesis import given
from hypothesis import strategies as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .._budgets import examples
from .._validation.linear import linear_cases

pytestmark = pytest.mark.component


def _distances(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0) - values.mean(axis=1)[:, None] + values.mean()


def _reference_paths(training: np.ndarray) -> np.ndarray:
    """Build independent SciPy paths from an explicit union-neighbor adjacency."""
    distances = _distances(training)
    adjacency = np.zeros_like(distances, dtype=bool)
    for index, row in enumerate(distances):
        order = np.lexsort((np.arange(len(row)), row, np.arange(len(row)) == index))[:2]
        adjacency[index, order] = True
    adjacency |= adjacency.T
    rows, columns = np.nonzero(adjacency)
    return shortest_path(
        csr_matrix((distances[rows, columns], (rows, columns)), shape=distances.shape),
        directed=False,
    )


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_feature_factor_matches_covariance_and_linear_kernel(dtype: type) -> None:
    """SVD and linear-kernel coordinates agree through invariant Gram products."""
    values = np.random.default_rng(84).normal(size=(11, 4)).astype(dtype)
    factor = Ordination(LinearGram(), PrincipalCoordinates(3)).estimate(values)
    kernel = Ordination(KernelGram(LINEAR_KERNEL), PrincipalCoordinates(3)).estimate(
        values
    )
    tolerance = 2e-5 if dtype is np.float32 else 2e-12
    expected = np.linalg.eigvalsh(np.cov(values, rowvar=False))[::-1]
    np.testing.assert_allclose(
        factor.explained_variance, expected[:3], rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        factor.coordinates @ factor.coordinates.T,
        kernel.coordinates @ kernel.coordinates.T,
        rtol=tolerance,
        atol=tolerance,
    )
    assert factor.coordinates.dtype == dtype
    assert factor.eigenvalues.shape == (len(values),)
    assert factor.explained_variance.shape == (3,)
    np.testing.assert_allclose(
        factor.explained_variance_ratio,
        expected[:3] / expected.sum(),
        rtol=tolerance,
        atol=tolerance,
    )
    assert isinstance(factor.mapping, LinearCoordinateMap)


@pytest.mark.reference
def test_kernel_centering_and_heldout_projection_match_numpy() -> None:
    """Independent kernel eigenpairs catch query-batch centering and n factors."""
    training = np.asarray([[0, 0], [1, 0.2], [0.5, 2], [3, 4], [2, -1]], dtype=float)
    query = np.asarray([[1.1, 2.2], [-4, 7], [4, -2]], dtype=float)
    bandwidth = 1.3
    kernel = np.exp(-(_distances(training) ** 2) / (2 * bandwidth**2))
    centered = _center(kernel)
    eigenvalues, vectors = np.linalg.eigh(centered)
    vectors = vectors[:, ::-1][:, :2]
    eigenvalues = eigenvalues[::-1][:2]
    cross = np.exp(
        -np.sum((query[:, None] - training[None]) ** 2, axis=-1) / (2 * bandwidth**2)
    )
    cross -= cross.mean(axis=1)[:, None] + kernel.mean(axis=0) - kernel.mean()
    fit = Ordination(
        KernelGram(RBFKernel(bandwidth)), PrincipalCoordinates(2)
    ).estimate(training)
    actual = fit.transform(query)
    np.testing.assert_allclose(
        fit.coordinates @ fit.coordinates.T,
        (vectors * eigenvalues) @ vectors.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual @ fit.coordinates.T, cross @ vectors @ vectors.T, atol=1e-12
    )
    np.testing.assert_allclose(fit.transform(query[1:2]), actual[1:2], atol=1e-12)
    assert isinstance(fit.mapping, KernelCoordinateMap)
    np.testing.assert_allclose(
        jax.jit(fit.mapping.project_values)(jnp.asarray(query)), actual, atol=1e-12
    )


@pytest.mark.contract
def test_prepared_sources_snapshot_inputs_and_reject_other_model() -> None:
    """Caller mutation cannot alter retained features or observed geometry."""
    values = np.asarray([[0, 0], [1, 0], [0, 2], [3, 4]], dtype=float)
    original = values.copy()
    model = Ordination(LinearGram(), PrincipalCoordinates(2))
    prepared = model.prepare(values)
    observed = ObservedDissimilarity.from_array(_distances(values))
    values[:] = 0
    np.testing.assert_array_equal(prepared.values, original)
    np.testing.assert_allclose(observed.values, _distances(original))
    with pytest.raises(ValueError, match="another model"):
        Ordination(LinearGram(), PrincipalCoordinates(2)).estimate_prepared(prepared)


@pytest.mark.reference
def test_distance_ordination_retains_gower_reference_order_and_geometry() -> None:
    """Held-out Euclidean distances reproduce the training coordinate frame."""
    training = np.asarray([[0, 0], [1, 0.2], [0.5, 2], [3, 4], [2, -1]], dtype=float)
    query = np.asarray([[1.1, 2.2], [-4, 7], [4, -2]], dtype=float)
    ids = ("a", "b", "c", "d", "e")
    observed = ObservedDissimilarity.from_array(_distances(training), ids)
    fit = Ordination(DistanceGram(), PrincipalCoordinates(2)).estimate(observed)
    rotation = np.linalg.lstsq(
        training - training.mean(axis=0), fit.coordinates, rcond=None
    )[0]
    cross = np.linalg.norm(query[:, None] - training[None], axis=-1)
    predicted = fit.transform_distances(cross, reference_ids=ids)
    np.testing.assert_allclose(
        predicted, (query - training.mean(axis=0)) @ rotation, atol=2e-12
    )
    np.testing.assert_allclose(
        fit.transform_distances(cross[1:2], reference_ids=ids),
        predicted[1:2],
        atol=2e-12,
    )
    assert fit.capabilities == ("transform_distances",)
    with pytest.raises(ValueError, match="reference IDs"):
        fit.transform_distances(cross, reference_ids=ids[::-1])
    clouds = PackedClouds.from_clouds([point[None] for point in training])
    constructed = Ordination(DistanceGram(Energy()), PrincipalCoordinates(2)).estimate(
        clouds
    )
    assert constructed.context.construction_warrant is not None
    assert "transform" in constructed.capabilities


@pytest.mark.contract
def test_signed_spectrum_is_explicit_and_never_promoted_to_a_law() -> None:
    """Positive-subspace approximation preserves the adverse full spectrum."""
    values = np.abs(np.arange(4.0)[:, None] - np.arange(4.0)[None]) ** 2
    observed = ObservedDissimilarity.from_array(values)
    with pytest.raises(NonEuclideanEmbeddingError):
        Ordination(DistanceGram(), PrincipalCoordinates(None)).estimate(observed)
    descriptive = Ordination(
        DistanceGram(), PrincipalCoordinates(None, "positive-subspace")
    ).estimate(observed)
    expected = np.linalg.eigvalsh(_center(-0.5 * values**2))[::-1]
    np.testing.assert_allclose(descriptive.eigenvalues, expected, atol=1e-12)
    assert descriptive.spectrum.material_negative_inertia
    assert descriptive.context.construction_warrant is None
    assert descriptive.report().warnings
    zero = Ordination(
        DistanceGram(), PrincipalCoordinates(None, "positive-subspace")
    ).estimate(ObservedDissimilarity.from_array(np.zeros((3, 3))))
    assert zero.coordinates.shape == (3, 0)


@pytest.mark.contract
def test_unqualified_constructed_geometry_and_path_laws_remain_distinct() -> None:
    """A construction or graph operation cannot manufacture a Hilbertian warrant."""
    points = np.column_stack((np.arange(5.0), np.arange(5.0) ** 2))
    clouds = PackedClouds.from_clouds([point[None] for point in points])
    unsupported = SlicedWasserstein(ProjectionDesign(jax.random.PRNGKey(3), p=3))
    with pytest.raises(ValueError, match="qualified squared-negative-type"):
        Ordination(DistanceGram(), PrincipalCoordinates(None)).estimate(
            unsupported.prepare(clouds)
        )
    descriptive = Ordination(
        DistanceGram(), PrincipalCoordinates(None, "positive-subspace")
    ).estimate(unsupported.prepare(clouds))
    assert descriptive.context.construction_warrant is None
    graph = Ordination(
        DistanceGram(ShortestPathDistances(2, distances=Energy())),
        PrincipalCoordinates(None, "positive-subspace"),
    ).estimate(clouds)
    assert graph.context.construction_warrant is None
    assert "no Hilbertian" in graph.context.origin


@pytest.mark.reference
def test_isomap_graph_and_frozen_query_extension_match_independent_paths() -> None:
    """SciPy paths and explicit Nyström products check graph and extension math."""
    training = np.column_stack((np.arange(6.0), np.arange(6.0) ** 2 / 5))
    expected_paths = _reference_paths(training)
    model = Ordination(
        DistanceGram(ShortestPathDistances(2, features=("x", "z"))),
        PrincipalCoordinates(2, "positive-subspace"),
    )
    fit = model.estimate(Observations({"x": training[:, 0], "z": training[:, 1]}))
    query = np.asarray([[1.5, 0.4], [3.5, 2.2]])
    cross = np.linalg.norm(query[:, None] - training[None], axis=-1)
    query_paths = []
    for row in cross:
        order = np.lexsort((np.arange(len(row)), row))[:2]
        query_paths.append(np.min(expected_paths[order] + row[order, None], axis=0))
    raw = -0.5 * expected_paths**2
    eigenvalues, vectors = np.linalg.eigh(_center(raw))
    vectors = vectors[:, ::-1][:, :2]
    centered_cross = -0.5 * np.asarray(query_paths) ** 2
    centered_cross -= (
        centered_cross.mean(axis=1)[:, None] + raw.mean(axis=0) - raw.mean()
    )
    transformed = fit.transform(query)
    np.testing.assert_allclose(
        transformed @ fit.coordinates.T,
        centered_cross @ vectors @ vectors.T,
        atol=1e-11,
    )
    np.testing.assert_allclose(fit.transform(query[1:2]), transformed[1:2], atol=1e-12)
    np.testing.assert_allclose(fit.transform(training), fit.coordinates, atol=1e-11)
    assert eigenvalues[-1] > 0


@pytest.mark.contract
def test_disconnected_graphs_are_rejected_without_bridges_or_dropped_rows() -> None:
    """Two isolated pairs retain a genuine failure instead of invented paths."""
    points = np.asarray([[0.0], [1.0], [100.0], [101.0]])
    model = Ordination(
        DistanceGram(ShortestPathDistances(1, features=("x",))), PrincipalCoordinates(1)
    )
    with pytest.raises(ValueError, match="disconnected"):
        model.estimate(Observations({"x": points[:, 0]}))


@pytest.mark.reference
def test_direct_term_fits_only_training_features_and_freezes_scaling() -> None:
    """A direct nested term uses held-out predictors without refitting its axes."""
    training = Observations(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "z": [3.0, 1.0, 5.0, -1.0, 2.0],
            "y": [0.0, 1.0, 4.0, 9.0, 16.0],
        }
    )
    query = Observations({"x": [100.0, 101.0], "z": [-30.0, 80.0]})
    spec = Ordination(
        LinearGram(("x", "z"), scale=True), PrincipalCoordinates(1), name="components"
    )
    design = FittedDesign((Intercept() + spec)._prepare_term(training))
    separate = spec.estimate(training)
    np.testing.assert_allclose(
        design.transform(query)[:, 1:], separate.transform(query), atol=1e-12
    )
    assert design.column_names == ("intercept", "components_1")
    np.testing.assert_allclose(
        design.transform(query.take([1])), design.transform(query)[1:2], atol=1e-12
    )
    retained = design.ordination("components")
    np.testing.assert_array_equal(retained.spectrum.eigenvalues, separate.eigenvalues)
    np.testing.assert_array_equal(retained.loadings, separate.loadings)
    np.testing.assert_allclose(
        retained.explained_variance_ratio, separate.explained_variance_ratio
    )
    assert retained.inspection.context.row_ids == training.row_ids


@pytest.mark.contract
def test_named_ordination_lookup_never_refits_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested inspection retains actual evidence even when fitting is disabled."""
    rows = Observations(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [3.0, 0.0, 4.0, 1.0],
            "y": [1.0, 2.0, 4.0, 5.0],
        }
    )
    components = Ordination.pca(("x", "z"), dim=1, name="features")
    fit = Regression.ols("y", Intercept() + components).estimate(rows)
    interaction = FittedDesign((components * Linear("x"))._prepare_term(rows))

    def fail_fit(*_args: object, **_kwargs: object) -> None:
        message = "inspection refitted the ordination"
        raise AssertionError(message)

    monkeypatch.setattr(Ordination, "_estimate_prepared", fail_fit)
    retained = fit.design.ordination("features")
    assert retained.inspection.mapping is retained.mapping
    assert retained.spectrum.eigenvalues.shape == (rows.n_rows,)
    assert retained.loadings.shape == (2, 1)
    assert retained.explained_variance.shape == (1,)
    assert interaction.ordination("features").loadings.shape == (2, 1)
    with pytest.raises(KeyError, match="No fitted ordination"):
        fit.design.ordination("unknown")


@pytest.mark.contract
def test_predictive_only_ordination_does_not_claim_live_inspection() -> None:
    """A decoded-style map can predict without fabricating training fit evidence."""
    rows = Observations({"x": [0.0, 1.0, 2.0], "z": [1.0, 3.0, 0.0]})
    fitted = Ordination.pca(("x", "z"), dim=1).estimate(rows)
    term = FittedOrdination(fitted.mapping, ("ordination_1",))
    np.testing.assert_allclose(term.transform(rows), fitted.coordinates, atol=1e-12)
    with pytest.raises(ValueError, match="predictive-only"):
        _ = term.inspection
    with pytest.raises(ValueError, match="Predictive-only"):
        FittedDesign(term).ordination("ordination")


@pytest.mark.contract
def test_factories_preserve_components_and_signed_inertia_policy() -> None:
    """Convenient spellings retain the same mathematical model and defaults."""
    pca = Ordination.pca(("x", "z"), dim=1, scale=True, name="scores")
    assert pca == Ordination(
        LinearGram(("x", "z"), scale=True), PrincipalCoordinates(1), "scores"
    )
    strict = Ordination.pcoa(dim=None)
    assert strict.geometry == DistanceGram()
    assert strict.coordinates == PrincipalCoordinates(None)
    descriptive = Ordination.pcoa(dim=1, negative_inertia="positive-subspace")
    assert descriptive.coordinates.negative_inertia == "positive-subspace"


@pytest.mark.contract
def test_multiple_unnamed_ordinations_explain_name_collision() -> None:
    """Independent decompositions require explicit stable names in the design."""
    rows = Observations({"x": [0.0, 1.0, 2.0], "z": [1.0, 3.0, 0.0]})
    with pytest.raises(ValueError, match="distinct name="):
        FittedDesign(
            (
                Ordination.pca(("x",), dim=1) + Ordination.pca(("z",), dim=1)
            )._prepare_term(rows)
        )


@pytest.mark.contract
def test_integer_cross_distances_promote_before_squaring() -> None:
    """Large integer cross distances must not silently wrap and remain finite."""
    distances = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    fit = Ordination(DistanceGram(), PrincipalCoordinates(1)).estimate(
        ObservedDissimilarity.from_array(distances)
    )
    cross = np.asarray([[10_000_000_000, 10_000_000_001, 10_000_000_002]])
    np.testing.assert_allclose(
        fit.transform_distances(cross, reference_ids=(0, 1, 2)),
        fit.transform_distances(cross.astype(float), reference_ids=(0, 1, 2)),
        atol=1e-12,
    )


@pytest.mark.contract
def test_scaled_feature_query_map_has_jit_and_vmap_numeric_boundary() -> None:
    """The retained numerical map transforms arrays without host concretization."""
    training = np.asarray([[0.0, 2.0], [1.0, 4.0], [2.0, 1.0], [3.0, 8.0]])
    fit = Ordination(LinearGram(scale=True), PrincipalCoordinates(2)).estimate(training)
    assert isinstance(fit.mapping, LinearCoordinateMap)
    query = jnp.asarray([[9.0, 10.0], [-1.0, 5.0]])
    expected = fit.transform(query)
    mapping = fit.mapping

    def project_row(row: jax.Array) -> jax.Array:
        return mapping.project_values(row[None])[0]

    np.testing.assert_allclose(
        jax.jit(fit.mapping.project_values)(query), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.vmap(project_row)(query),
        expected,
        atol=1e-12,
    )


@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-learn")
@pytest.mark.parametrize("kind", ["linear", "kernel", "isomap"])
def test_ordination_matches_sklearn_training_and_heldout_gram_products(
    optional_reference: Callable[[str], ModuleType], kind: str
) -> None:
    """Live independent methods check shared normalization and frozen extension."""
    training = np.column_stack((np.arange(7.0), np.arange(7.0) ** 2 / 5))
    query = np.asarray([[1.5, 0.4], [3.5, 2.2]])
    geometry: LinearGram | KernelGram | DistanceGram
    if kind == "linear":
        peer = optional_reference("sklearn.decomposition").PCA(
            n_components=2, svd_solver="full"
        )
        geometry = LinearGram()
    elif kind == "kernel":
        peer = optional_reference("sklearn.decomposition").KernelPCA(
            n_components=2, kernel="rbf", gamma=1 / (2 * 1.2**2), eigen_solver="dense"
        )
        geometry = KernelGram(RBFKernel(1.2))
    else:
        peer = optional_reference("sklearn.manifold").Isomap(
            n_components=2, n_neighbors=2, eigen_solver="dense"
        )
        geometry = DistanceGram(ShortestPathDistances(2, features=("x", "z")))
    reference = peer.fit_transform(training)
    model = Ordination(geometry, PrincipalCoordinates(2, "positive-subspace"))
    fitted = model.estimate(
        Observations({"x": training[:, 0], "z": training[:, 1]})
        if kind == "isomap"
        else training
    )
    np.testing.assert_allclose(
        fitted.coordinates @ fitted.coordinates.T,
        reference @ reference.T,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fitted.transform(query) @ fitted.coordinates.T,
        peer.transform(query) @ reference.T,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.property
@examples(12)
@given(
    case=linear_cases(n=8, p=4, intercept=True),
    shift=st.integers(-3, 3),
    order=st.permutations(tuple(range(8))),
)
def test_ordination_gram_equivalence_translation_and_permutation(
    case: LinearCase, shift: int, order: tuple[int, ...]
) -> None:
    """Compare full Gram matrices, including repeated or zero eigenvalues."""
    values = jnp.asarray(case.design[:, 1:])
    factor = Ordination(LinearGram(), PrincipalCoordinates(3)).estimate(values)
    distances = _distances(np.asarray(values))
    observed = ObservedDissimilarity.from_array(distances, tuple(range(8)))
    distance = Ordination(DistanceGram(), PrincipalCoordinates(3)).estimate(observed)
    expected = np.asarray(values) - np.asarray(values).mean(axis=0)
    gram = expected @ expected.T
    tolerance = 3e-4 if values.dtype == jnp.float32 else 3e-10
    for result in (factor, distance):
        np.testing.assert_allclose(
            result.coordinates @ result.coordinates.T,
            gram,
            rtol=tolerance,
            atol=tolerance,
        )
    np.testing.assert_allclose(
        factor.transform(values), factor.coordinates, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        distance.transform_distances(distances, reference_ids=tuple(range(8))),
        distance.coordinates,
        rtol=tolerance,
        atol=tolerance,
    )
    indices = np.asarray(order)
    changed = Ordination(LinearGram(), PrincipalCoordinates(3)).estimate(
        (values + shift)[indices]
    )
    np.testing.assert_allclose(
        changed.coordinates @ changed.coordinates.T,
        gram[np.ix_(indices, indices)],
        rtol=tolerance,
        atol=tolerance,
    )
