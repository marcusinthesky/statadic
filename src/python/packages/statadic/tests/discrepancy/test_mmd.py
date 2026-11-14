# standalone package-maintenance entry point
"""Typed MMD strategies, estimator semantics, and packed-matrix parity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from statadic.core.domains import (
    FiniteValues,
)
from statadic.core.matrices import (
    DMat,
)
from statadic.discrepancy.mmd import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    PositiveSemidefiniteKernelProperties,
    ZeroNormMeanEmbeddingError,
    cosine_mean_mmd,
    cosine_mean_mmd_functional,
    declare_similarity,
    distance_induced_kernel,
    mmd,
    mmd_matrix,
    mmd_squared,
    mmd_squared_matrix,
    rbf_kernel,
)
from statadic.discrepancy.mmd.energy import (
    energy_distance,
    energy_geometry,
    parse_distribution_exponent,
)
from statadic.geometry.ground.metrics import (
    ANGULAR,
    EUCLIDEAN,
)
from statadic.geometry.ground.similarities import (
    distance_induced_gram,
    linear_gram,
)
from statadic.sample.packed import PackedClouds, pack

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import ill_typed

if TYPE_CHECKING:
    from statadic.core.axioms import (
        Law,
    )
    from statadic.discrepancy.mmd import SimilarityStrategy
    from statadic.geometry.ground.metrics import (
        GroundDistance,
    )
    from statadic.geometry.ground.similarities import PositiveSemidefiniteKernelLaw

_PARITY_RTOL = 1e-11
_RAGGED_ATOL = 1e-7


def _energy_kernel(
    metric: GroundDistance[Law] = EUCLIDEAN,
    exponent: float = 1.0,
) -> SimilarityStrategy[PositiveSemidefiniteKernelLaw]:
    geometry = energy_geometry(  # type: ignore[no-matching-overload]
        metric,
        parse_distribution_exponent(exponent),
    )
    return distance_induced_kernel(geometry)


def _linear_plus_quadratic_gram(
    left: jax.Array,
    right: jax.Array,
) -> jax.Array:
    inner = left @ right.T
    return inner + inner**2


CUSTOM_KERNEL = declare_similarity(
    _linear_plus_quadratic_gram,
    law=PositiveSemidefiniteKernelProperties(),
    name="linear-plus-quadratic-test-kernel",
    required_evidence=FiniteValues,
)


_RBF_RNG = np.random.default_rng(7)
_RBF_X = jnp.asarray(_RBF_RNG.normal(size=(12, 5)))
_RBF_Y = jnp.asarray(_RBF_RNG.normal(size=(9, 5)) + 0.5)


@pytest.fixture
def clouds() -> tuple[jax.Array, jax.Array]:
    """Return deterministic unequal-size samples for scalar parity checks."""
    rng = np.random.default_rng(20260731)
    return (
        jnp.asarray(rng.normal(size=(40, 12))),
        jnp.asarray(rng.normal(size=(33, 12)) + 0.4),
    )


@pytest.fixture
def x64() -> None:
    """Skip strict parity checks when JAX float64 is unavailable."""
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("requires the package test session's float64 policy")


@pytest.mark.usefixtures("x64")
class TestEnergyParity:
    """The unhalved negative-type kernel has MMD²/energy constant one."""

    @pytest.mark.component
    @pytest.mark.reference
    @pytest.mark.parametrize(
        ("metric", "exponent"),
        [(EUCLIDEAN, 1.0), (EUCLIDEAN, 0.5), (ANGULAR, 1.0)],
    )
    def test_constant_is_one(
        self,
        clouds: tuple[jax.Array, jax.Array],
        metric: GroundDistance[Law],
        exponent: float,
    ) -> None:
        """Match energy V-statistics without an undocumented scale factor."""
        x, y = clouds
        expected = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=exponent,
            metric=metric,
        )
        observed = mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=_energy_kernel(metric, exponent),
        )
        assert jnp.allclose(observed, expected, rtol=_PARITY_RTOL)

    @pytest.mark.component
    @pytest.mark.property
    def test_half_normalized_kernel_has_factor_two(
        self,
        clouds: tuple[jax.Array, jax.Array],
    ) -> None:
        """Expose the factor induced by the alternate half-kernel convention."""

        def half(left: jax.Array, right: jax.Array) -> jax.Array:
            return 0.5 * distance_induced_gram(left, right, metric=EUCLIDEAN)

        kernel = declare_similarity(
            half,
            law=PositiveSemidefiniteKernelProperties(),
            required_evidence=FiniteValues,
        )
        x, y = clouds
        expected = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric=EUCLIDEAN,
        )
        assert jnp.allclose(
            2.0
            * mmd_squared(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                kernel=kernel,
            ),
            expected,
            rtol=_PARITY_RTOL,
        )

    @pytest.mark.unit
    @pytest.mark.property
    def test_reference_does_not_move_v_statistic(
        self,
        clouds: tuple[jax.Array, jax.Array],
    ) -> None:
        """Keep the V-statistic invariant to the induced-kernel reference."""
        x, y = clouds
        kernel = _energy_kernel()
        base = mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=kernel,
        )
        shifted = mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=kernel,
            reference=jnp.full((12,), -20.0),
        )
        assert jnp.allclose(base, shifted, rtol=1e-9)


class TestScalarMmd:
    """Certify scalar estimator laws and declared-strategy boundaries."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_default_linear_is_zero_on_self(self) -> None:
        """Return zero for identical empirical measures."""
        x = jnp.arange(30.0).reshape(10, 3)
        assert jnp.allclose(
            mmd(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(x)),
            ),
            0.0,
        )

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=_RBF_X, y=_RBF_Y)
    @given(x=float_arrays((12, 5)), y=float_arrays((9, 5)))
    def test_rbf_is_symmetric_and_rooted(self, x: jax.Array, y: jax.Array) -> None:
        """Root the nonnegative characteristic-kernel V-statistic."""
        kernel = rbf_kernel(1.0)
        squared = mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=kernel,
        )
        assert jnp.allclose(
            mmd(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                kernel=kernel,
            ),
            jnp.sqrt(jnp.maximum(squared, 0)),
        )
        assert jnp.allclose(
            mmd(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                kernel=kernel,
            ),
            mmd(
                PackedClouds.from_array(jnp.asarray(y)),
                PackedClouds.from_array(jnp.asarray(x)),
                kernel=kernel,
            ),
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_u_statistic_remains_signed_and_cannot_be_rooted(self) -> None:
        """Preserve signed U-estimates instead of branding their square root."""
        x = jnp.array([[0.0], [1.0]])
        y = jnp.array([[0.0], [2.0]])
        estimate = mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=rbf_kernel(1.0),
            unbiased=True,
        )
        assert float(estimate) < 0.0
        with pytest.raises(TypeError, match=r"Literal\[False\]"):
            mmd(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                unbiased=ill_typed(value=True),
            )

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("bare", ["linear", linear_gram, 42])
    def test_bare_dispatch_is_rejected(self, bare: object) -> None:
        """Require law-bearing strategies at every numerical entry point."""
        x = jnp.ones((2, 2))
        with pytest.raises(TypeError, match="declared SimilarityStrategy"):
            mmd_squared(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(x)),
                kernel=ill_typed(bare),
            )

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("bandwidth", [True, 0.0, -1.0, np.inf, np.nan])
    def test_rbf_parameters_fail_at_factory(self, bandwidth: object) -> None:
        """Reject invalid bandwidths before numerical evaluation."""
        with pytest.raises((TypeError, ValueError), match="bandwidth"):
            rbf_kernel(ill_typed(bandwidth))

    @pytest.mark.unit
    @pytest.mark.contract
    def test_incompatible_ground_power_fails_before_kernel_construction(self) -> None:
        """Reject a ground/exponent pair lacking a proved energy law."""
        with pytest.raises(TypeError, match="fractional-power compatibility"):
            _energy_kernel(ANGULAR, 1.5)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_cosine_domain_violation_stays_visible(self) -> None:
        """Do not disguise an input outside the cosine-kernel carrier."""
        bad = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        good = jnp.eye(2)
        assert bool(
            jnp.isnan(
                mmd_squared(
                    PackedClouds.from_array(jnp.asarray(bad)),
                    PackedClouds.from_array(jnp.asarray(good)),
                    kernel=COSINE_KERNEL,
                )
            )
        )


class TestPrototypeChord:
    """Pin the explicitly unit-sphere prototype chord boundary."""

    @pytest.mark.component
    @pytest.mark.numerical
    def test_checked_host_and_jax_forms_agree(self) -> None:
        """Keep checked host and transformable implementations aligned."""
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 1.0])
        expected = np.sqrt(2.0)
        assert cosine_mean_mmd_functional(left, right) == pytest.approx(expected)
        assert cosine_mean_mmd(jnp.asarray(left), jnp.asarray(right)) == pytest.approx(
            expected
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_host_door_rejects_zero_and_off_sphere_prototypes(self) -> None:
        """Reject prototype inputs without unit-sphere evidence."""
        with pytest.raises(ZeroNormMeanEmbeddingError):
            cosine_mean_mmd_functional(np.zeros(2), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="unit-normalized"):
            cosine_mean_mmd_functional(np.array([2.0, 0.0]), np.array([1.0, 0.0]))


@pytest.fixture
def ragged_sets() -> list[jax.Array]:
    """Return deterministic clouds with unequal observation counts."""
    rng = np.random.default_rng(20260802)
    return [
        jnp.asarray(rng.normal(size=(rows, 4)) + index * 0.3)
        for index, rows in enumerate((3, 7, 5))
    ]


_RAGGED_KERNELS = [
    pytest.param(LINEAR_KERNEL, 1e-12, id="linear"),
    pytest.param(COSINE_KERNEL, 1e-12, id="cosine"),
    pytest.param(rbf_kernel(1.3), 1e-12, id="rbf"),
    pytest.param(_energy_kernel(EUCLIDEAN), 1e-7, id="energy-euclidean"),
    pytest.param(_energy_kernel(ANGULAR), 1e-7, id="energy-angular"),
    pytest.param(CUSTOM_KERNEL, 1e-12, id="third-party"),
]


@pytest.mark.usefixtures("x64")
class TestRaggedMmdMatrix:
    """Certify packed matrix estimators against scalar definitions."""

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.parametrize(("kernel", "rtol"), _RAGGED_KERNELS)
    def test_v_statistic_matches_every_scalar_cell(
        self,
        ragged_sets: list[jax.Array],
        kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
        rtol: float,
    ) -> None:
        """Match every V-statistic cell for each declared kernel law."""
        values = mmd_squared_matrix(
            pack(ragged_sets, require_positive_counts=True), kernel=kernel
        )
        np.testing.assert_array_equal(np.diag(np.asarray(values)), 0.0)
        np.testing.assert_allclose(values, values.T)
        for left in range(len(ragged_sets)):
            for right in range(left + 1, len(ragged_sets)):
                expected = mmd_squared(
                    PackedClouds.from_array(jnp.asarray(ragged_sets[left])),
                    PackedClouds.from_array(jnp.asarray(ragged_sets[right])),
                    kernel=kernel,
                )
                np.testing.assert_allclose(
                    values[left, right],
                    expected,
                    rtol=rtol,
                    atol=1e-12,
                )

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.parametrize(
        "kernel",
        [LINEAR_KERNEL, rbf_kernel(1.3), _energy_kernel(EUCLIDEAN)],
    )
    def test_u_statistic_matches_scalar_including_signed_diagonal(
        self,
        kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    ) -> None:
        """Match signed U-statistics, including within-cloud diagonals."""
        rng = np.random.default_rng(4242)
        sets = [jnp.asarray(rng.normal(size=(rows, 4))) for rows in (2, 3, 5)]
        reference = jnp.zeros((4,))
        values = mmd_squared_matrix(
            pack(sets, require_positive_counts=True),
            kernel=kernel,
            reference=reference,
            unbiased=True,
        )
        for left in range(len(sets)):
            for right in range(len(sets)):
                expected = mmd_squared(
                    PackedClouds.from_array(jnp.asarray(sets[left])),
                    PackedClouds.from_array(jnp.asarray(sets[right])),
                    kernel=kernel,
                    reference=reference,
                    unbiased=True,
                )
                np.testing.assert_allclose(
                    values[left, right],
                    expected,
                    rtol=_RAGGED_ATOL,
                    atol=1e-12,
                )

    @pytest.mark.component
    @pytest.mark.contract
    def test_rooted_matrix_is_the_typed_canonical_surface(
        self,
        ragged_sets: list[jax.Array],
    ) -> None:
        """Return the canonical typed distribution-distance carrier."""
        kernel = _energy_kernel(EUCLIDEAN)
        result = mmd_matrix(
            pack(ragged_sets, require_positive_counts=True), kernel=kernel
        )
        assert isinstance(result, DMat)
        squared = mmd_squared_matrix(
            pack(ragged_sets, require_positive_counts=True), kernel=kernel
        )
        expected = jnp.sqrt(jnp.maximum(squared, 0.0))
        expected = expected.at[jnp.diag_indices(expected.shape[0])].set(0.0)
        assert jnp.allclose(result.values, expected)

    @pytest.mark.component
    @pytest.mark.contract
    def test_packed_core_is_traceable(self, ragged_sets: list[jax.Array]) -> None:
        """Trace the packed numerical core with the strategy static."""
        packed = pack(ragged_sets, require_positive_counts=True)
        kernel = _energy_kernel(EUCLIDEAN)
        reference = ragged_sets[0][0]

        def traced(clouds: PackedClouds, anchor: jax.Array) -> jax.Array:
            return mmd_squared_matrix(clouds, kernel=kernel, reference=anchor)

        jaxpr = jax.make_jaxpr(traced)(packed, reference)
        # Narrow before reading `.shape`: `out_avals` is typed as the abstract
        # `AbstractValue` base, which carries no shape. The isinstance also
        # asserts the traced output is a shaped array rather than a token.
        out_aval = jaxpr.out_avals[0]
        assert isinstance(out_aval, jax.core.ShapedArray)
        assert out_aval.shape == (3, 3)
