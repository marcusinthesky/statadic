"""Frozen measure geometry serves square analyses and rectangular prediction."""

from __future__ import annotations

from dataclasses import replace
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.representation import (
    MMD,
    Energy,
    GeometrySpec,
    HilbertianMeasureMatrix,
    PairwiseGeometry,
    PreparedGeometry,
    SlicedWasserstein,
    Wasserstein1D,
)
from statadic.core.domains import FiniteValues
from statadic.core.matrices import checked_dmat
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.discrepancy.transport.univariate import wasserstein_1d
from statadic.geometry.ground.similarities import (
    COSINE_KERNEL,
    PositiveSemidefiniteKernelProperties,
    RBFKernel,
    declare_similarity,
)
from statadic.sample.packed import PackedClouds

from .._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from statadic.api.representation import RowId


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "spec",
    [
        MMD(RBFKernel(jnp.asarray(1.2))),
        Energy(),
        Wasserstein1D(p=2),
        SlicedWasserstein(
            ProjectionDesign(jax.random.PRNGKey(7), n_projections=3, p=2)
        ),
    ],
)
def test_cross_is_new_by_frozen_reference(spec: GeometrySpec) -> None:
    """Every column refers to its retained measure, with genuine law evidence."""
    reference = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[5.0]])]
    )
    other = PackedClouds.from_clouds(
        [jnp.asarray([[1.0]]), jnp.asarray([[3.0], [4.0]]), jnp.asarray([[6.0]])]
    )
    prepared = spec.prepare(reference, row_ids=("first", "second"))
    pairwise = prepared.pairwise()
    actual = prepared.cross(other)
    assert actual.shape == (3, 2)
    assert pairwise.row_ids == ("first", "second")
    assert pairwise.spec is spec
    assert pairwise.negative_type
    assert pairwise.squared_negative_type
    assert pairwise.matrix is not None
    assert pairwise.construction_warrant
    witness = spec.matrix(reference)
    assert witness is not None
    np.testing.assert_array_equal(pairwise.values, witness.values)
    for row in range(other.n_observations):
        for column in range(reference.n_observations):
            expected = spec.prepare(reference.take([column])).cross(other.take([row]))
            np.testing.assert_allclose(
                actual[row, column], expected[0, 0], atol=1e-13, rtol=1e-13
            )


@pytest.mark.integration
def test_cross_transport_matches_independent_pair_call() -> None:
    """Rectangular evaluation retains unequal masses and the pairwise order."""
    train = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[4.0]])], [[0.25, 0.75], [1.0]]
    )
    new = PackedClouds.from_clouds(
        [jnp.asarray([[1.0], [3.0], [5.0]])], [[0.5, 0.25, 0.25]]
    )
    actual = Wasserstein1D(p=2).prepare(train).cross(new)
    expected = wasserstein_1d(
        PackedClouds.from_array(
            jnp.asarray(new.data[0, :, 0])[..., None],
            jnp.asarray(new.masses[0])
            / jnp.sum(jnp.asarray(new.masses[0]), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(train.data[0, :, 0])[..., None],
            jnp.asarray(train.masses[0])
            / jnp.sum(jnp.asarray(train.masses[0]), axis=-1, keepdims=True),
        ),
        p=2,
    )
    assert float(actual[0, 0]) == pytest.approx(float(expected), rel=1e-13)


@pytest.mark.component
@pytest.mark.contract
def test_unsupported_laws_and_arbitrary_pairwise_construction_fail() -> None:
    """A caller cannot promote a p=3 transport matrix to Hilbertian geometry."""
    measures = PackedClouds.from_clouds([jnp.ones((1, 1)), jnp.zeros((2, 1))])
    spec = SlicedWasserstein(ProjectionDesign(jax.random.PRNGKey(1), p=3))
    pairwise = spec.prepare(measures).pairwise()
    assert not pairwise.negative_type
    assert not pairwise.squared_negative_type
    assert pairwise.matrix is None
    with pytest.raises(TypeError, match="produced"):
        PairwiseGeometry(pairwise.values, pairwise.row_ids, spec, None)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    "factory",
    [
        lambda: Energy(exponent=2.0),
        lambda: Wasserstein1D(p=1),
    ],
)
def test_distance_specs_reject_changed_conventions(
    factory: Callable[[], object],
) -> None:
    """Reject unqualified energy exponents and transport orders."""
    with pytest.raises(ValueError, match=r"range|p=2"):
        factory()


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("spec", [MMD, Energy])
def test_measure_distances_fix_the_rooted_v_convention(
    spec: type[MMD | Energy],
) -> None:
    """The estimator convention is reported on the spec but cannot be chosen."""
    parameters = signature(spec).parameters
    assert "estimator" not in parameters
    assert "rooted" not in parameters


@pytest.mark.unit
@pytest.mark.contract
def test_wasserstein_order_is_required() -> None:
    """The name gives the dimension, so the order must be stated."""
    assert signature(Wasserstein1D).parameters["p"].default is Parameter.empty


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("ids", [("a",), ("a", "a"), (True, "a")])
def test_reference_id_validation(ids: tuple[RowId, ...]) -> None:
    """Reference IDs are unique, aligned and distinct from Boolean flags."""
    measures = PackedClouds.from_clouds([jnp.ones((1, 1)), jnp.zeros((2, 1))])
    with pytest.raises(ValueError, match="row_ids"):
        Energy().prepare(measures, row_ids=ids)


@pytest.mark.component
@pytest.mark.contract
def test_eager_geometry_retains_material_kernel_failure() -> None:
    """An impossible negative discrepancy fails analysis and prediction eagerly."""

    def negative_dot(
        x: jax.Array, y: jax.Array, *, reference: jax.Array | None = None
    ) -> jax.Array:
        del reference
        return -(x @ y.T)

    kernel = declare_similarity(
        negative_dot,
        law=PositiveSemidefiniteKernelProperties(),
        required_evidence=FiniteValues,
    )
    measures = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[2.0]])])
    prepared = MMD(kernel=kernel).prepare(measures)
    with pytest.raises(ValueError, match="undefined"):
        prepared.pairwise()
    with pytest.raises(ValueError, match="undefined"):
        prepared.cross(measures)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("spec", [MMD(kernel=COSINE_KERNEL), Energy(ground="angular")])
def test_singleton_undefined_domain_is_not_hidden_by_zero_diagonal(
    spec: GeometrySpec,
) -> None:
    """Zero-norm angular/cosine atoms remain unavailable even with only one row."""
    measures = PackedClouds.from_clouds([jnp.asarray([[0.0, 0.0]])])
    prepared = spec.prepare(measures)
    native = spec.matrix(measures)
    assert native is not None
    assert np.isnan(native.values[0, 0])
    with pytest.raises(ValueError, match="undefined"):
        prepared.pairwise()
    with pytest.raises(ValueError, match="undefined"):
        prepared.cross(measures)


@pytest.mark.component
@pytest.mark.contract
def test_sliced_query_precision_cannot_redraw_frozen_geometry() -> None:
    """The bounded initial interface rejects a changed projection dtype."""
    reference = PackedClouds.from_clouds(
        [
            jnp.asarray([[0.0, 0.0]], dtype=jnp.float32),
            jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        ]
    )
    query = PackedClouds.from_clouds([jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)])
    spec = SlicedWasserstein(
        ProjectionDesign(jax.random.PRNGKey(2), n_projections=1, p=2)
    )
    prepared = spec.prepare(reference)
    with pytest.raises(ValueError, match="matching effective"):
        prepared.cross(query)
    with pytest.raises(ValueError, match="matching effective"):
        spec.cross(reference, query)
    with pytest.raises(ValueError, match="matching effective"):
        spec.cross(query, reference)


class _L1OnMeans:
    """Minimal external geometry: L1 distance between measure means."""

    def prepare(
        self, measures: PackedClouds, *, row_ids: Sequence[RowId] | None = None
    ) -> PreparedGeometry:
        ids = tuple(range(measures.n_observations)) if row_ids is None else row_ids
        return PreparedGeometry(self, measures, tuple(ids))

    def cross(self, left: PackedClouds, right: PackedClouds) -> jax.Array:
        def means(measures: PackedClouds) -> jax.Array:
            return jnp.sum(measures.data * measures.masses[..., None], axis=1)

        return jnp.sum(
            jnp.abs(means(left)[:, None, :] - means(right)[None, :, :]), axis=-1
        )

    def matrix(
        self,
        measures: PackedClouds,  # noqa: ARG002
    ) -> HilbertianMeasureMatrix | None:
        return None


@pytest.mark.component
@pytest.mark.contract
def test_custom_geometry_plugs_in_without_a_hilbertian_claim() -> None:
    """A structural GeometrySpec prepares and evaluates, but earns no law."""
    spec = _L1OnMeans()
    assert isinstance(spec, GeometrySpec)
    reference = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[5.0]])]
    )
    other = PackedClouds.from_clouds([jnp.asarray([[3.0]])])
    prepared = spec.prepare(reference, row_ids=("a", "b"))
    np.testing.assert_allclose(prepared.cross(other), [[2.0, 2.0]])
    pairwise = prepared.pairwise()
    assert pairwise.spec is spec
    assert pairwise.matrix is None
    assert not pairwise.squared_negative_type
    with pytest.raises(ValueError, match="squared-negative-type"):
        Ordination(DistanceGram(), PrincipalCoordinates()).estimate(pairwise)
    with pytest.raises(TypeError, match="GeometrySpec"):
        PreparedGeometry(ill_typed(object()), reference, ("a", "b"))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("raw_matrix", ["checked", "object"])
def test_custom_geometry_requires_explicit_law_warrant(raw_matrix: str) -> None:
    """A raw DMat or arbitrary object cannot claim a Hilbertian producer law."""

    class RawGeometry(_L1OnMeans):
        @override
        def matrix(self, measures: PackedClouds) -> Any:
            if raw_matrix == "checked":
                return checked_dmat(jnp.zeros((measures.n_observations,) * 2))
            return object()

    measures = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[1.0]])])
    with pytest.raises(TypeError, match="HilbertianMeasureMatrix warrant"):
        RawGeometry().prepare(measures).pairwise()


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.regression
def test_custom_geometry_cross_must_return_query_by_reference_shape() -> None:
    """A finite but transposed extension matrix names expected and actual shapes."""

    class TransposedGeometry(_L1OnMeans):
        @override
        def cross(self, left: PackedClouds, right: PackedClouds) -> jax.Array:
            return super().cross(left, right).T

    reference = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[5.0]])]
    )
    other = PackedClouds.from_clouds([jnp.asarray([[3.0]])])
    prepared = TransposedGeometry().prepare(reference, row_ids=("a", "b"))
    with pytest.raises(ValueError, match=r"shape \(1, 2\).*got \(2, 1\)"):
        prepared.cross(other)


@pytest.mark.component
@pytest.mark.contract
def test_custom_geometry_can_explicitly_assert_a_construction_warrant() -> None:
    """A third-party wrapper may name the actual construction it delegates to."""

    class ExternalEnergy(_L1OnMeans):
        @override
        def cross(self, left: PackedClouds, right: PackedClouds) -> jax.Array:
            return Energy().cross(left, right)

        @override
        def matrix(self, measures: PackedClouds) -> HilbertianMeasureMatrix:
            return HilbertianMeasureMatrix(
                Energy().matrix(measures).matrix, "Delegated rooted V-energy."
            )

    measures = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[1.0]])])
    result = ExternalEnergy().prepare(measures).pairwise()
    assert result.negative_type
    assert result.squared_negative_type
    assert result.construction_warrant == "Delegated rooted V-energy."
    np.testing.assert_array_equal(result.values, Energy().matrix(measures).values)
    with pytest.raises(ValueError, match="construction warrant"):
        HilbertianMeasureMatrix(Energy().matrix(measures).matrix, "  ")


@pytest.mark.unit
@pytest.mark.contract
def test_pairwise_geometry_cannot_be_reauthorized_by_dataclass_replace() -> None:
    """Replacement cannot carry authorization or edit a producer's law evidence."""
    measures = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[1.0]])])
    unqualified = _L1OnMeans().prepare(measures).pairwise()
    qualified = Energy().prepare(measures).pairwise()
    with pytest.raises(TypeError):
        replace(ill_typed(unqualified), construction_warrant="claimed")
    with pytest.raises(TypeError, match="produced"):
        replace(unqualified, matrix=qualified.matrix)
    with pytest.raises(TypeError, match="produced"):
        replace(qualified, values=jnp.zeros_like(qualified.values))
    assert qualified.matrix is not None
    np.testing.assert_array_equal(qualified.values, qualified.matrix.values)
    assert qualified.construction_warrant == Energy().matrix(measures).construction
