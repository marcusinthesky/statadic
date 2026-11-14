# standalone package-maintenance entry point
"""``statadic.discrepancy.mmd.energy`` — the energy statistics and their ground metric.

The energy distance is parameterised by a ground metric, which is what carries
its axioms: ``DECLARED_AXIOMS`` is keyed ``energy_distance[euclidean]`` /
``energy_distance[angular]``, not by the statistic. Whether each of those two
configurations satisfies what it declares is measured in
``tests/testing/test_axioms.py``; that the *default* is the scale-blind angular
one, and what that costs, is pinned in
``tests/migration/test_ground_discrepancy_migration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import random
from statadic.core.axioms import (
    METRIC,
    NEGATIVE_TYPE,
    SEMIMETRIC,
)
from statadic.core.domains import (
    FiniteValues,
)
from statadic.discrepancy.mmd.energy import (
    ANOVA_EXPONENT,
    UNIT_EXPONENT,
    AnovaEnergyLaw,
    AnovaExponent,
    DistributionEnergyLaw,
    EnergyGeometry,
    IncompatibleEnergyGroundExponentError,
    MeanEnergyLaw,
    SeparatingDistributionEnergyLaw,
    SubunitExponent,
    SuperunitExponent,
    UnitExponent,
    energy_distance,
    energy_distance_with_geometry,
    energy_geometry,
    energy_test_statistic,
    mean_distance_matrix,
    mixture_energy_distance,
    parse_distribution_exponent,
    parse_energy_exponent,
    unsafe_energy_geometry,
)
from statadic.discrepancy.mmd.energy import (
    energy_distance_matrix_values as energy_distance_matrix,
)
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    EuclideanLaw,
    angular_distance,
    cosine_distance,
    declare_ground_distance,
    euclidean_distance,
    normalized_euclidean_distance,
)
from statadic.sample.packed import PackedClouds, pack

from ..._budgets import examples
from ..._strategies import float_arrays, nonzero_rows
from ..._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from jax import Array
    from statadic.core.axioms import (
        Axioms,
        LawProperties,
    )
    from statadic.geometry.ground.config import (
        GroundDistanceName,
    )
    from statadic.geometry.ground.metrics import (
        AngularLaw,
        GroundDistance,
    )


@dataclass(frozen=True, slots=True)
class _FakeEnergyGeometry:
    """Structural lookalike that did not cross either construction factory."""

    ground: GroundDistance[AngularLaw]
    exponent: SuperunitExponent
    law: MeanEnergyLaw


@dataclass(frozen=True, slots=True)
class _SquaredEuclideanLaw:
    """Squared Euclidean is negative type and mean-only at unit power."""

    axioms: Axioms = field(default=SEMIMETRIC, init=False)
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)
    unit_power_mean_only: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class _ManhattanLaw:
    """Manhattan is a negative-type metric but not strong in dimension two."""

    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)


class TestEnergyDistanceMatrix:
    """Tests for the batched energy_distance_matrix primitive."""

    @pytest.mark.component
    @pytest.mark.property
    @examples(20)
    @example(
        metric="angular",
        sets=[
            random.normal(random.fold_in(random.PRNGKey(0), i), (32, 4))
            for i in range(7)
        ],
    )
    @example(
        metric="euclidean",
        sets=[
            random.normal(random.fold_in(random.PRNGKey(0), i), (32, 4))
            for i in range(7)
        ],
    )
    @given(
        metric=st.sampled_from(["angular", "euclidean"]),
        sets=st.lists(nonzero_rows(8, 4), min_size=4, max_size=4),
    )
    def test_matches_naive_loop(
        self,
        metric: GroundDistanceName,
        sets: list[jax.Array],
    ) -> None:
        """Batched matrix agrees with per-pair energy_distance loop."""
        d_batched = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric=metric
        )
        n = len(sets)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = energy_distance(
                    PackedClouds.from_array(jnp.asarray(sets[i])),
                    PackedClouds.from_array(jnp.asarray(sets[j])),
                    exponent=1.0,
                    metric=metric,
                )
                assert jnp.allclose(d_batched[i, j], d_ij, atol=1e-9)

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(
        sets=[
            random.normal(random.fold_in(random.PRNGKey(1), i), (16, 3))
            for i in range(5)
        ],
    )
    @given(sets=st.lists(float_arrays((16, 3)), min_size=5, max_size=5))
    def test_symmetric_zero_diagonal(self, sets: list[jax.Array]) -> None:
        """Matrix is exactly symmetric with an exactly-zero diagonal."""
        d = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric="euclidean"
        )
        assert jnp.array_equal(d, d.T)
        assert jnp.array_equal(jnp.diagonal(d), jnp.zeros(len(sets)))

    @pytest.mark.unit
    @pytest.mark.contract
    def test_invalid_exponent_raises(self) -> None:
        """Exponent outside (0, 2) raises ValueError."""
        sets = [jnp.ones((4, 2)), jnp.zeros((4, 2))]
        with pytest.raises(ValueError, match="exponent"):
            energy_distance_matrix(
                pack(sets, require_positive_counts=True), exponent=2.0, metric="angular"
            )

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("metric", ["angular", "cosine"])
    def test_superunit_ground_is_rejected_by_both_matrix_surfaces(
        self,
        metric: GroundDistanceName,
    ) -> None:
        """Both eager matrix doors validate the ground/exponent theorem."""
        sets = [
            jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            jnp.array([[-1.0, 0.0], [0.0, -1.0]]),
        ]
        with pytest.raises(IncompatibleEnergyGroundExponentError):
            energy_distance_matrix(
                pack(sets, require_positive_counts=True), exponent=1.5, metric=metric
            )
        with pytest.raises(IncompatibleEnergyGroundExponentError):
            mean_distance_matrix(
                pack(sets, require_positive_counts=True), exponent=1.5, metric=metric
            )


class TestRaggedEnergyDistanceMatrix:
    """t62: unequal row counts route through the masked packed core.

    Tolerance rationale (``rtol=0`` throughout — the stated absolute bound is
    the enforced one, not a relative bound in disguise). ``tests/conftest.py``
    enables x64 and the clouds are O(1)-valued with counts <= 9, so the only
    drift between a matrix entry and the scalar per-pair call comes from the
    packed block reducing an ``m_max x m_max`` ``jnp.where``-masked grid where
    the scalar core reduces a dense ``n x m`` one.

    * Euclidean: ``atol=1e-6``. An earlier ``1e-12`` was calibrated on the
      pinned seeds 7/11/13 alone (max ``|Δ|`` 8.9e-16 there) and is **not** a
      general bound: sweeping 200 seeds through ``_ragged_sets`` exceeds it on
      17 of them, worst ``|Δ| = 5.9605e-08`` at seed 32. The cause is not
      masking — at seed 32 the padded and unpadded self-diagonals are
      bit-identical. It is ``cdist``'s expand-and-sqrt form losing half its
      precision on a *self* distance: the true ``cdist(x, x)`` diagonal is
      exactly zero, and ``sqrt`` of a ~1e-16 cancellation residual is ~1e-8.
      Every failing pair involves the ``n=1`` cloud, its cross terms agree to
      ``0.0``, and the deviation equals that cloud's spurious scalar-side self
      distance exactly. The batched matrix path is the *more* accurate of the
      two here; the tolerance exists to absorb the scalar reference's own
      artifact.
    * Angular: measured max ``|Δ|`` 1.4901e-8 over the same 200-seed sweep
      (0 exceedances at ``1e-7``). That is ``arccos`` conditioning, **not** a
      masking defect: a 1e-16 perturbation of the cosine near 1 maps to
      ``sqrt(2e-16) ~ 1.5e-8`` in angle. ``atol=1e-6`` keeps a decade of margin
      rather than the 6.7x that ``1e-7`` left.
    """

    _RAGGED_COUNTS = (1, 2, 5, 9)
    #: Per-ground absolute tolerance; see the class docstring for measurements.
    _PARITY_ATOL: ClassVar[dict[str, float]] = {"euclidean": 1e-6, "angular": 1e-6}
    #: Below this the pair is treated as degenerate for the vacuity guard. Kept
    #: three decades ABOVE ``_PARITY_ATOL`` on purpose: a floor below the
    #: tolerance would let the parity assertions pass on values the tolerance
    #: alone already covers. Observed pair energies are O(1) (1.03--7.09), so
    #: this floor has ample margin while still being load-bearing.
    _NONZERO_FLOOR = 1e-3

    def _ragged_sets(self, seed: int = 7) -> list[jnp.ndarray]:
        """Build clouds with counts (1, 2, 5, 9) and a shared feature width."""
        key = random.PRNGKey(seed)
        return [
            random.normal(random.fold_in(key, index), (count, 3))
            for index, count in enumerate(self._RAGGED_COUNTS)
        ]

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.parametrize("metric", ["euclidean", "angular"])
    def test_ragged_entries_match_scalar_pairs(
        self,
        metric: GroundDistanceName,
    ) -> None:
        """Every off-diagonal entry equals the scalar energy_distance call."""
        sets = self._ragged_sets()
        matrix = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric=metric
        )

        assert matrix.shape == (len(sets), len(sets))
        nonzero = 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                scalar = energy_distance(
                    PackedClouds.from_array(jnp.asarray(sets[i])),
                    PackedClouds.from_array(jnp.asarray(sets[j])),
                    exponent=1.0,
                    metric=metric,
                )
                assert jnp.allclose(
                    matrix[i, j],
                    scalar,
                    rtol=0.0,
                    atol=self._PARITY_ATOL[metric],
                )
                nonzero += int(abs(float(scalar)) > self._NONZERO_FLOOR)
        # The angular ground must not collapse to an all-zero matrix, or the
        # parity assertions above would hold vacuously.
        assert nonzero > 0

    @pytest.mark.unit
    @pytest.mark.property
    def test_ragged_matrix_is_exactly_symmetric_with_zero_diagonal(self) -> None:
        """The packed path keeps exact symmetry and an exactly-zero diagonal."""
        sets = self._ragged_sets(seed=11)
        matrix = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric="euclidean"
        )
        assert jnp.array_equal(matrix, matrix.T)
        assert jnp.array_equal(jnp.diagonal(matrix), jnp.zeros(len(sets)))

    @pytest.mark.unit
    @pytest.mark.property
    def test_ragged_entries_are_finite_under_zero_padding(self) -> None:
        """Angular NaNs on padded rows are contained by the masked reduction."""
        sets = self._ragged_sets(seed=13)
        matrix = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric="angular"
        )
        assert bool(jnp.all(jnp.isfinite(matrix)))

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_single_cloud_matrix_is_one_by_one_zero(self) -> None:
        """A one-cloud list returns the 1x1 zero matrix."""
        sets = [random.normal(random.PRNGKey(3), (5, 2))]
        matrix = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric="euclidean"
        )
        assert matrix.shape == (1, 1)
        assert jnp.array_equal(matrix, jnp.zeros((1, 1)))

    @pytest.mark.unit
    @pytest.mark.contract
    def test_packed_core_stays_traceable(self) -> None:
        """The packed core remains a pure jaxpr over its PackedClouds pytree."""
        packed = pack(self._ragged_sets(seed=5), require_positive_counts=True)

        def evaluate_energy_distance_matrix_packed_core(
            clouds: PackedClouds,
        ) -> jax.Array:
            return energy_distance_matrix(clouds, 1.0, metric=EUCLIDEAN)

        jaxpr = jax.make_jaxpr(evaluate_energy_distance_matrix_packed_core)(packed)
        # Narrow before reading `.shape`: `out_avals` is typed as the abstract
        # `AbstractValue` base, which carries no shape. The isinstance also
        # asserts the traced output is a shaped array rather than a token.
        out_aval = jaxpr.out_avals[0]
        assert isinstance(out_aval, jax.core.ShapedArray)
        assert out_aval.shape == (len(self._RAGGED_COUNTS),) * 2

    @pytest.mark.unit
    @pytest.mark.property
    def test_ragged_padding_is_inert(self) -> None:
        """Arbitrary inactive storage cannot change the empirical functional."""
        clouds = pack(self._ragged_sets(), require_positive_counts=True)
        changed = clouds.with_coordinates(
            jnp.where(clouds.active_mask[..., None], clouds.data, jnp.nan)
        )
        np.testing.assert_array_equal(
            energy_distance_matrix(clouds, exponent=1.0, metric="euclidean"),
            energy_distance_matrix(changed, exponent=1.0, metric="euclidean"),
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_mixed_dtypes_are_promoted_before_packing(self) -> None:
        """A float32 cloud cannot truncate its float64 neighbours."""
        sets = [
            jnp.array([[1.0, 2.0]], dtype=jnp.float32),
            jnp.array([[3.5, 4.25], [0.5, 1.5]], dtype=jnp.float64),
        ]
        matrix = energy_distance_matrix(
            pack(sets, require_positive_counts=True), exponent=1.0, metric="euclidean"
        )
        assert matrix.dtype == jnp.float64
        assert jnp.allclose(
            matrix[0, 1],
            energy_distance(
                PackedClouds.from_array(jnp.asarray(sets[0].astype(jnp.float64))),
                PackedClouds.from_array(jnp.asarray(sets[1])),
                exponent=1.0,
                metric="euclidean",
            ),
            rtol=0.0,
            atol=self._PARITY_ATOL["euclidean"],
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_equal_shape_mixed_dtypes_keep_the_stacked_result(self) -> None:
        """Promotion produces the same result as pre-promoted packed inputs."""
        narrow = jnp.array([[1.0, 2.0], [0.5, 1.5]], dtype=jnp.float32)
        wide = jnp.array([[3.5, 4.25], [0.25, 6.5]], dtype=jnp.float64)

        mixed = energy_distance_matrix(
            pack([narrow, wide], require_positive_counts=True),
            exponent=1.0,
            metric="euclidean",
        )
        promoted = energy_distance_matrix(
            pack([narrow.astype(jnp.float64), wide], require_positive_counts=True),
            exponent=1.0,
            metric="euclidean",
        )

        assert mixed.dtype == jnp.float64
        assert jnp.array_equal(mixed, promoted)
        assert jnp.allclose(
            mixed[0, 1],
            energy_distance(
                PackedClouds.from_array(jnp.asarray(narrow.astype(jnp.float64))),
                PackedClouds.from_array(jnp.asarray(wide)),
                exponent=1.0,
                metric="euclidean",
            ),
            rtol=0.0,
            atol=self._PARITY_ATOL["euclidean"],
        )


class TestEnergyDistance:
    """Tests for energy_distance function."""

    @pytest.mark.unit
    @pytest.mark.contract
    def test_metric_is_required(self) -> None:
        """Omitting the ground metric raises instead of selecting one."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(TypeError, match="metric"):
            ill_typed(energy_distance)(x, x)

    @pytest.mark.unit
    @pytest.mark.property
    def test_identical_samples(self) -> None:
        """Test energy distance between identical samples is near zero."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = x.copy()
        dist = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=1.0,
            metric="angular",
        )
        assert jnp.allclose(dist, 0.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    def test_different_samples(self) -> None:
        """Test energy distance between different samples is positive."""
        x = jnp.array([[1.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[10.0, 10.0], [10.0, 10.0]])
        dist = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=1.0,
            metric="angular",
        )
        assert dist > 0.0

    @pytest.mark.unit
    @pytest.mark.contract
    def test_exponent_validation(self) -> None:
        """Test that exponent must be in (0, 2)."""
        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0, 4.0]])

        with pytest.raises(ValueError, match="exponent must be in the range"):
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=0.0,
                metric="angular",
            )

        with pytest.raises(ValueError, match="exponent must be in the range"):
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=2.0,
                metric="angular",
            )

        with pytest.raises(ValueError, match="exponent must be in the range"):
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=-0.5,
                metric="angular",
            )

        with pytest.raises(ValueError, match="exponent must be in the range"):
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=2.5,
                metric="angular",
            )

    @pytest.mark.unit
    @pytest.mark.property
    def test_valid_exponent_boundaries(self) -> None:
        """Test exponents near boundaries work correctly."""
        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0, 4.0]])

        # Just inside boundaries should work
        dist1 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=0.01,
            metric="angular",
        )
        assert dist1 > 0.0

        dist2 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=1.99,
            metric="euclidean",
        )
        assert dist2 > 0.0

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(
        x=random.normal(random.PRNGKey(42), (10, 5)),
        y=random.normal(random.PRNGKey(43), (10, 5)),
    )
    @given(x=nonzero_rows(10, 5), y=nonzero_rows(10, 5))
    def test_symmetry(self, x: jax.Array, y: jax.Array) -> None:
        """Test energy distance is symmetric on nonzero rows."""
        dist_xy = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric="angular",
        )
        dist_yx = energy_distance(
            PackedClouds.from_array(jnp.asarray(y)),
            PackedClouds.from_array(jnp.asarray(x)),
            metric="angular",
        )

        assert jnp.allclose(dist_xy, dist_yx, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    @pytest.mark.slow
    def test_non_negativity(self) -> None:
        """Test energy distance is non-negative."""
        key = random.PRNGKey(42)
        for i in range(10):
            x = random.normal(key, (20, 3))
            y = random.normal(random.PRNGKey(43 + i), (20, 3))
            dist = energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                metric="angular",
            )
            assert dist >= 0.0

    @pytest.mark.unit
    @pytest.mark.property
    def test_different_exponents(self) -> None:
        """Test that different exponents give different results."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[5.0, 5.0], [6.0, 6.0]])

        dist1 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=0.5,
            metric="euclidean",
        )
        dist2 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=1.0,
            metric="euclidean",
        )
        dist3 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=1.5,
            metric="euclidean",
        )

        # Different exponents should produce different values
        assert not jnp.allclose(dist1, dist2)
        assert not jnp.allclose(dist2, dist3)

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("metric", ["angular", "cosine"])
    def test_superunit_ground_is_rejected(
        self,
        metric: GroundDistanceName,
    ) -> None:
        """A valid float cannot bypass the selected ground's CND theorem."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(IncompatibleEnergyGroundExponentError):
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(-x)),
                exponent=1.5,
                metric=metric,
            )

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("metric", ["euclidean", "normalized_euclidean"])
    def test_supported_superunit_ground_remains_available(
        self,
        metric: GroundDistanceName,
    ) -> None:
        """Both declared full-fractional compatibility branches remain valid."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        value = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(-x)),
            exponent=1.5,
            metric=metric,
        )
        assert bool(jnp.isfinite(value))

    @pytest.mark.unit
    @pytest.mark.property
    def test_scale_sensitivity(self) -> None:
        """Test energy distance changes with scale of separation."""
        x = jnp.array([[1.0, 0.0], [0.9, 0.1]])

        # Small separation
        y1 = jnp.array([[1.1, 0.1], [1.2, 0.2]])
        dist1 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y1)),
            metric="normalized_euclidean",
        )

        # Large separation
        y2 = jnp.array([[0.0, 1.0], [-0.1, 0.9]])  # Orthogonal direction
        dist2 = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y2)),
            metric="normalized_euclidean",
        )

        # Larger separation (orthogonal) should give larger distance
        assert dist2 > dist1

    @pytest.mark.component
    @pytest.mark.contract
    def test_jit_and_vmap_return_jax_scalars(self) -> None:
        """The validated wrapper remains composable with fixed static options."""
        x = jnp.array([[0.0], [1.0]])
        y = jnp.array([[2.0], [3.0]])

        def evaluate_energy_distance(a: jax.Array, b: jax.Array) -> Array:
            return energy_distance(
                PackedClouds.from_array(jnp.asarray(a)),
                PackedClouds.from_array(jnp.asarray(b)),
                metric="euclidean",
            )

        jitted = jax.jit(evaluate_energy_distance)
        vmapped: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = jax.vmap(
            lambda a, b: energy_test_statistic(a, b, metric="euclidean")
        )

        assert jnp.allclose(
            jitted(x, y),
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                metric="euclidean",
            ),
        )
        assert jnp.allclose(vmapped(jnp.stack([x, y]), jnp.stack([y, x])), 3.0)


class TestEnergyTestStatistic:
    """Tests for energy_test_statistic function."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_identical_samples(self) -> None:
        """Test statistic for identical samples is near zero."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = x.copy()
        stat = energy_test_statistic(x, y, exponent=1.0, metric="angular")
        assert jnp.allclose(stat, 0.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    def test_different_samples(self) -> None:
        """Test statistic for different samples is positive."""
        x = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
        stat = energy_test_statistic(x, y, exponent=1.0, metric="angular")
        assert stat > 0.0

    @pytest.mark.unit
    @pytest.mark.property
    def test_sample_size_scaling(self) -> None:
        """Statistic grows with sample size via the (n·m)/(n+m) prefactor.

        For a fixed shift the plug-in energy distance is roughly stable, so the
        test statistic T = (n·m)/(n+m)·Ê inherits the near-linear growth of its
        size prefactor. A larger balanced sample must yield a strictly larger
        statistic than a small subsample.
        """
        key = random.PRNGKey(42)
        x = random.normal(key, (100, 5))
        y = random.normal(random.PRNGKey(43), (100, 5)) + 1.0  # Shifted distribution

        stat_small = energy_test_statistic(x[:20], y[:20], metric="angular")
        stat_large = energy_test_statistic(x, y, metric="angular")

        # The (n·m)/(n+m) prefactor is ~5x larger at n=100 vs n=20, so the
        # statistic must scale up markedly (not merely be positive).
        assert stat_large > 2.0 * stat_small

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_relationship_to_energy_distance(self) -> None:
        """Test statistic computation matches expected formula."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        stat = energy_test_statistic(x, y, exponent=1.0, metric="angular")
        ed = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            exponent=1.0,
            metric="angular",
        )

        n, m = len(x), len(y)
        expected_stat = (n * m) / (n + m) * ed

        assert jnp.allclose(stat, expected_stat, rtol=1e-5)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_exponent_validation(self) -> None:
        """Test that exponent must be in (0, 2)."""
        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0, 4.0]])

        with pytest.raises(ValueError, match="exponent must be in the range"):
            energy_test_statistic(x, y, exponent=0.0, metric="angular")

        with pytest.raises(ValueError, match="exponent must be in the range"):
            energy_test_statistic(x, y, exponent=2.0, metric="angular")

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("metric", ["angular", "cosine"])
    def test_superunit_ground_is_rejected(
        self,
        metric: GroundDistanceName,
    ) -> None:
        """The test-statistic compatibility door shares theorem validation."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(IncompatibleEnergyGroundExponentError):
            energy_test_statistic(x, -x, exponent=1.5, metric=metric)

    @pytest.mark.unit
    @pytest.mark.property
    @pytest.mark.slow
    def test_non_negativity(self) -> None:
        """Test statistic is non-negative."""
        key = random.PRNGKey(42)
        for i in range(10):
            x = random.normal(key, (15, 4))
            y = random.normal(random.PRNGKey(43 + i), (15, 4))
            stat = energy_test_statistic(x, y, metric="angular")
            assert stat >= 0.0

    @pytest.mark.unit
    @pytest.mark.property
    def test_unequal_sample_sizes(self) -> None:
        """Test statistic works with unequal sample sizes."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

        stat = energy_test_statistic(x, y, metric="angular")
        assert stat > 0.0
        assert jnp.isfinite(stat)


class TestMixtureEnergyDistance:
    """Test target distance from a weighted candidate mixture."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_nonnegative(self) -> None:
        """Verify nonnegative."""
        x = random.normal(random.PRNGKey(0), (60, 4)) + 3.0
        close = random.normal(random.PRNGKey(1), (30, 4)) + 3.0
        far = random.normal(random.PRNGKey(2), (50, 4)) + 9.0
        e = mixture_energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            pack([close, far], require_positive_counts=True),
            jnp.array([0.5, 0.5]),
            metric="euclidean",
        )
        numerical_floor = -1e-4
        assert e >= numerical_floor

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("metric", ["angular", "cosine"])
    def test_superunit_ground_is_rejected(
        self,
        metric: GroundDistanceName,
    ) -> None:
        """The mixture compatibility door shares theorem validation."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(IncompatibleEnergyGroundExponentError):
            mixture_energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                pack([x, -x], require_positive_counts=True),
                jnp.array([0.5, 0.5]),
                exponent=1.5,
                metric=metric,
            )


# --- custom (callable) ground metrics -------------------------------------
#
# The `metric=` argument accepts a callable as well as a name. Each assertion
# below compares the custom-metric result against the built-in Euclidean one
# with a strict inequality that only holds if the callable actually drove the
# computation — "is finite" would pass just as well if the argument were
# silently ignored. (t49 B2: from `test_scipy_api.py`.)


class TestCustomMetricInStatistics:
    """Tests for using custom metrics in energy statistics."""

    @pytest.mark.component
    @pytest.mark.property
    def test_energy_distance_with_callable(self) -> None:
        """Test energy_distance with custom metric."""

        def squared_euclidean(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Squared Euclidean distance (without sqrt)."""
            diff = x[:, None, :] - y[None, :, :]
            return jnp.sum(diff**2, axis=-1)

        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[10.0, 0.0], [11.0, 0.0]])

        declared = declare_ground_distance(
            squared_euclidean,
            law=_SquaredEuclideanLaw(),
            name="squared_euclidean",
            required_evidence=FiniteValues,
        )
        result = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric=declared,
            exponent=1.0,
        )

        # The callable metric must actually drive the computation: squared-
        # Euclidean inflates cross-distances vs plain Euclidean, so the energy
        # distance under the custom metric strictly exceeds the built-in one.
        euclid = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric="euclidean",
            exponent=1.0,
        )
        assert jnp.isfinite(result)
        assert float(result) > float(euclid) > 0.0

    @pytest.mark.component
    @pytest.mark.property
    def test_energy_test_statistic_with_callable(self) -> None:
        """Test energy_test_statistic with custom metric."""

        def manhattan(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Manhattan distance."""
            return jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)

        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[10.0, 10.0], [11.0, 11.0]])

        declared = declare_ground_distance(
            manhattan,
            law=_ManhattanLaw(),
            name="manhattan",
            required_evidence=FiniteValues,
        )
        result = energy_test_statistic(x, y, metric=declared, exponent=1.0)

        # Manhattan ≥ Euclidean coordinate-wise, so the well-separated groups
        # yield a strictly larger statistic under the custom metric — proving
        # the callable is actually used rather than silently ignored.
        euclid = energy_test_statistic(x, y, metric="euclidean", exponent=1.0)
        assert jnp.isfinite(result)
        assert float(result) > float(euclid) > 0.0

    @pytest.mark.component
    @pytest.mark.property
    def test_consistency_across_metrics(self) -> None:
        """Test that built-in and callable metrics give same results."""
        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[10.0, 0.0], [11.0, 0.0]])

        # Using string metric
        result_str = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric="euclidean",
            exponent=1.0,
        )

        # Using callable metric
        declared = declare_ground_distance(
            euclidean_distance,
            law=EuclideanLaw(),
            name="euclidean_callable",
            required_evidence=FiniteValues,
        )
        result_callable = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric=declared,
            exponent=1.0,
        )

        assert jnp.allclose(result_str, result_callable, atol=1e-5)


class TestEnergyGeometry:
    """Theorem-sensitive geometry construction and deterministic witnesses."""

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            (0.5, SubunitExponent),
            (1.0, UnitExponent),
            (1.5, SuperunitExponent),
            (2.0, AnovaExponent),
        ],
    )
    def test_exponent_parser_preserves_theorem_regime(
        self,
        value: float,
        expected_type: type[object],
    ) -> None:
        """A free float is converted to one of four nominal branches."""
        parsed = parse_energy_exponent(value)
        assert isinstance(parsed, expected_type)
        assert parsed.value == value
        assert jax.tree_util.tree_leaves(parsed) == []

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("value", [0.0, -1.0, 2.1, float("inf"), float("nan")])
    def test_energy_exponent_rejects_values_outside_open_closed_domain(
        self,
        value: float,
    ) -> None:
        """Only ``0 < alpha <= 2`` has a semantic regime."""
        with pytest.raises(ValueError, match="exponent must be in the range"):
            parse_energy_exponent(value)

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("value", [True, False, "1", object()])
    def test_energy_exponent_rejects_boolean_and_non_real_values(
        self,
        value: object,
    ) -> None:
        """Boolean equality with one cannot enter the unit theorem branch."""
        with pytest.raises(TypeError, match="exponent must be in the range"):
            parse_energy_exponent(value)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_energy_exponent_accepts_and_normalizes_numpy_real(self) -> None:
        """A NumPy real scalar is normalized to a static Python float."""
        parsed = parse_energy_exponent(np.float64(0.5))
        assert isinstance(parsed, SubunitExponent)
        assert type(parsed.value) is float

    @pytest.mark.unit
    @pytest.mark.contract
    def test_distribution_parser_rejects_anova_boundary(self) -> None:
        """The compatibility float API cannot enter the mean-only branch."""
        with pytest.raises(ValueError, match=r"range \(0, 2\)"):
            parse_distribution_exponent(2.0)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_safe_geometry_matrix_distinguishes_law_and_estimand(self) -> None:
        """Each branch returns only the law its ground theorem supports."""
        euclidean_sub = energy_geometry(EUCLIDEAN, SubunitExponent(0.5))
        euclidean_unit = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
        euclidean_super = energy_geometry(EUCLIDEAN, SuperunitExponent(1.5))
        normalized_super = energy_geometry(
            NORMALIZED_EUCLIDEAN,
            SuperunitExponent(1.5),
        )
        angular_unit = energy_geometry(ANGULAR, UNIT_EXPONENT)
        cosine_sub = energy_geometry(COSINE, SubunitExponent(0.5))
        cosine_unit = energy_geometry(COSINE, UNIT_EXPONENT)
        anova = energy_geometry(EUCLIDEAN, ANOVA_EXPONENT)

        assert isinstance(euclidean_sub.law, SeparatingDistributionEnergyLaw)
        assert isinstance(euclidean_unit.law, SeparatingDistributionEnergyLaw)
        assert isinstance(euclidean_super.law, SeparatingDistributionEnergyLaw)
        assert type(normalized_super.law) is DistributionEnergyLaw
        assert type(angular_unit.law) is DistributionEnergyLaw
        assert type(cosine_sub.law) is DistributionEnergyLaw
        assert type(cosine_unit.law) is MeanEnergyLaw
        assert isinstance(anova.law, AnovaEnergyLaw)
        assert jax.tree_util.tree_leaves(euclidean_super) == []

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("ground", [ANGULAR, COSINE])
    def test_superunit_rejects_ground_with_explicit_cnd_counterexample(
        self,
        ground: object,
    ) -> None:
        """Angular and cosine carry no full-power capability."""
        with pytest.raises(TypeError, match="full fractional-power"):
            energy_geometry(
                ill_typed(ground),
                SuperunitExponent(1.5),
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_anova_rejects_non_euclidean_quotient_ground(self) -> None:
        """A metric-like quotient is not Euclidean ANOVA geometry."""
        with pytest.raises(TypeError, match="ANOVA"):
            energy_geometry(
                ill_typed(NORMALIZED_EUCLIDEAN),
                ANOVA_EXPONENT,
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_unvalidated_exponent_object_is_not_treated_as_anova(self) -> None:
        """An arbitrary object cannot reach the final theorem branch."""
        with pytest.raises(TypeError):
            energy_geometry(EUCLIDEAN, ill_typed(object()))

    @pytest.mark.unit
    @pytest.mark.contract
    def test_public_geometry_protocol_cannot_bypass_factories(self) -> None:
        """The structural contract itself is not a concrete constructor."""
        with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
            ill_typed(EnergyGeometry)()

    @pytest.mark.unit
    @pytest.mark.contract
    def test_unsafe_factory_is_explicit_and_preserves_exact_exponent(self) -> None:
        """Unsupported assertions require the conspicuous unsafe door."""
        assumed = MeanEnergyLaw()
        geometry = unsafe_energy_geometry(
            ANGULAR,
            SuperunitExponent(1.5),
            assume_law=assumed,
        )
        assert geometry.exponent.value == pytest.approx(1.5)
        assert geometry.law is assumed
        assert geometry.ground.requires_nonzero_rows
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        assert bool(
            jnp.isfinite(
                energy_distance_with_geometry(
                    PackedClouds.from_array(jnp.asarray(x)),
                    PackedClouds.from_array(jnp.asarray(x)),
                    geometry,
                )
            )
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_unsafe_factory_still_validates_static_container_parts(self) -> None:
        """Unsafe means an assumed law, not arbitrary exponent/law objects."""
        with pytest.raises(TypeError):
            unsafe_energy_geometry(
                ANGULAR,
                ill_typed(object()),
                assume_law=MeanEnergyLaw(),
            )
        with pytest.raises(TypeError):
            unsafe_energy_geometry(
                ANGULAR,
                SuperunitExponent(1.5),
                assume_law=ill_typed(object()),
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_structural_lookalike_cannot_bypass_factory_provenance(self) -> None:
        """A duck object needs the unsafe factory before evaluation."""
        fake = _FakeEnergyGeometry(
            ground=ANGULAR,
            exponent=SuperunitExponent(1.5),
            law=MeanEnergyLaw(),
        )
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(TypeError, match="must be constructed"):
            energy_distance_with_geometry(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(x)),
                fake,
            )

    @pytest.mark.component
    @pytest.mark.contract
    def test_typed_evaluator_matches_compatibility_surface(self) -> None:
        """The typed evaluator preserves V-statistic arithmetic."""
        x = jnp.array([[0.0], [1.0]])
        y = jnp.array([[2.0], [4.0]])
        geometry = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
        assert jnp.allclose(
            energy_distance_with_geometry(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                geometry,
            ),
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                exponent=1.0,
                metric="euclidean",
            ),
        )

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_circle_geodesic_and_cosine_fail_negative_type_above_one(self) -> None:
        """The four-point CND witnesses have positive quadratic forms."""
        points = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        weights = jnp.array([1.0, 1.0, -1.0, -1.0])
        exponent = 1.5
        angular_q = weights @ (angular_distance(points, points) ** exponent) @ weights
        cosine_q = weights @ (cosine_distance(points, points) ** exponent) @ weights

        assert float(angular_q) == pytest.approx(
            4.0 * jnp.pi**exponent * (1.0 - 2.0 ** (1.0 - exponent)),
            rel=1e-6,
        )
        assert float(cosine_q) == pytest.approx(
            4.0 * 2.0**exponent - 8.0,
            rel=1e-6,
        )
        assert angular_q > 0.0
        assert cosine_q > 0.0

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_angular_unit_energy_is_zero_for_distinct_diameter_laws(self) -> None:
        """Circle geodesic is negative type but not strong negative type."""
        x = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
        y = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        assert x.shape != y.shape
        assert float(
            energy_distance(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                metric=ANGULAR,
            )
        ) == pytest.approx(
            0.0,
            abs=1e-6,
        )

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize(
        "distance",
        [angular_distance, cosine_distance, normalized_euclidean_distance],
    )
    def test_zero_rows_expose_t55_3_domain_handoff(
        self, distance: Callable[[Array, Array], Array]
    ) -> None:
        """Directional declarations remain conditional pending checked carriers."""
        zero = jnp.zeros((1, 2))
        values = distance(zero, zero)
        assert bool(jnp.all(jnp.isnan(values)))
