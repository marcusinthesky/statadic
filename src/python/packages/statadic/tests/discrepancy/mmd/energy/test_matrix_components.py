# standalone package-maintenance entry point
"""Public raw empirical components: cardinality, diagonal, and transform contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.matrices import (
    PairwiseEstimateMatrix,
)
from statadic.discrepancy.mmd.energy import (
    IncompatibleEnergyGroundExponentError,
    empirical_mean_ground_distances,
    mean_distance_matrix,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
)
from statadic.sample.packed import PackedClouds, pack

from ...._typing import check_types

if TYPE_CHECKING:
    from pathlib import Path

    from jax.typing import (
        DTypeLike,
    )
    from statadic.geometry.ground.config import (
        GroundDistanceName,
    )


def _clouds(dtype: DTypeLike = jnp.float64) -> list[jax.Array]:
    return [
        jnp.asarray(values, dtype=dtype)[:, None] for values in ([0, 2], [4], [1, 3, 5])
    ]


def _explicit_means(clouds: list[jax.Array], exponent: float = 1.0) -> np.ndarray:
    """Independent host calculation over the observations, with no packing."""
    arrays = [np.asarray(cloud) for cloud in clouds]
    return np.asarray(
        [
            [
                np.mean(
                    [np.linalg.norm(x - y) ** exponent for x in left for y in right]
                )
                for right in arrays
            ]
            for left in arrays
        ]
    )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ragged_components_match_independent_pairwise_means(dtype: DTypeLike) -> None:
    """True counts and live self-pairs distinguish components from energy distances."""
    clouds = _clouds(dtype)
    result = empirical_mean_ground_distances(pack(clouds, require_positive_counts=True))
    expected = np.asarray([[1, 3, 7 / 3], [3, 0, 5 / 3], [7 / 3, 5 / 3, 16 / 9]])
    tolerance = 8 * np.finfo(np.asarray(result.values).dtype).eps

    assert isinstance(result, PairwiseEstimateMatrix)
    assert result.values.dtype == clouds[0].dtype
    np.testing.assert_allclose(_explicit_means(clouds), expected, rtol=tolerance)
    np.testing.assert_allclose(result.values, expected, rtol=tolerance, atol=tolerance)
    assert float(result.values[0, 0]) > 0


@pytest.mark.component
@pytest.mark.regression
def test_equal_size_agreement_single_cloud_and_ordering() -> None:
    """The new eager boundary keeps the old stacked arithmetic and caller order."""
    equal = [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[4.0], [7.0]])]
    np.testing.assert_array_equal(
        empirical_mean_ground_distances(
            pack(equal, require_positive_counts=True)
        ).values,
        mean_distance_matrix(
            pack(equal, require_positive_counts=True), metric="euclidean"
        ),
    )
    singleton = empirical_mean_ground_distances(
        pack([jnp.asarray([[4.0]])], require_positive_counts=True)
    )
    np.testing.assert_array_equal(singleton.values, [[0.0]])

    order = [2, 0, 1]
    reordered = empirical_mean_ground_distances(
        pack([_clouds()[index] for index in order], require_positive_counts=True)
    )
    expected = _explicit_means(_clouds())[np.ix_(order, order)]
    np.testing.assert_allclose(reordered.values, expected, rtol=1e-14)


@pytest.mark.component
@pytest.mark.reference
def test_ragged_mixed_dtypes_promote_without_truncation() -> None:
    """A narrow first cloud cannot downcast a later caller-owned wide cloud."""
    clouds = [
        jnp.asarray([[0.0], [2.0]], dtype=jnp.float32),
        jnp.asarray([[4.0 + 2**-30]], dtype=jnp.float64),
    ]
    result = empirical_mean_ground_distances(pack(clouds, require_positive_counts=True))
    assert result.values.dtype == jnp.float64
    np.testing.assert_allclose(result.values, _explicit_means(clouds), rtol=1e-14)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("metric", ["angular", "cosine"])
def test_directional_padding_is_ignored_and_raw_power_needs_no_energy_theorem(
    metric: GroundDistanceName,
) -> None:
    """Inactive zero directions cannot inject NaNs or padded-length denominators."""
    clouds = [
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[-1.0, 0.0]]),
    ]
    exponent = 1.5
    expected: list[list[float]] = []
    for left in clouds:
        row: list[float] = []
        for right in clouds:
            cosine = np.asarray(left @ right.T) / (
                np.linalg.norm(np.asarray(left), axis=1)[:, None]
                * np.linalg.norm(np.asarray(right), axis=1)[None, :]
            )
            cosine = np.clip(cosine, -1.0, 1.0)
            distances = np.arccos(cosine) if metric == "angular" else 1 - cosine
            row.append(float(np.mean(distances**exponent)))
        expected.append(row)
    result = empirical_mean_ground_distances(
        pack(clouds, require_positive_counts=True), exponent=exponent, metric=metric
    )
    assert np.isfinite(np.asarray(result.values)).all()
    np.testing.assert_allclose(result.values, expected, atol=1e-14)
    with pytest.raises(IncompatibleEnergyGroundExponentError):
        mean_distance_matrix(
            pack([clouds[0], clouds[0]], require_positive_counts=True),
            exponent=exponent,
            metric=metric,
        )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("exponent", [0.0, 2.0, -1.0, float("nan"), float("inf")])
def test_invalid_exponents_are_rejected(exponent: float) -> None:
    """The raw operation retains the declared open numeric exponent interval."""
    with pytest.raises(ValueError, match="exponent must be in the range"):
        empirical_mean_ground_distances(
            pack(_clouds(), require_positive_counts=True), exponent=exponent
        )


@pytest.mark.component
@pytest.mark.reference
def test_fixed_shape_packed_kernel_supports_jit_and_vmap() -> None:
    """Batching promises attach to packed arrays, not dynamically ragged lists."""
    first = pack(_clouds(), require_positive_counts=True)
    # Same fixed capacity with different counts and deliberately invalid padding.
    second_clouds = [jnp.asarray([[1.0]]), jnp.asarray([[2.0], [5.0]]), _clouds()[2]]
    second = pack(second_clouds, require_positive_counts=True)
    mask = jnp.arange(first.data.shape[1])[None, :, None] >= first.counts[:, None, None]
    first = PackedClouds(jnp.where(mask, jnp.nan, first.data), first.counts)
    batched = PackedClouds(
        jnp.stack([first.data, second.data]), jnp.stack([first.counts, second.counts])
    )

    def kernel(clouds: PackedClouds) -> jax.Array:
        return empirical_mean_ground_distances(clouds, 1.0, metric=EUCLIDEAN).values

    np.testing.assert_allclose(jax.jit(kernel)(first), _explicit_means(_clouds()))
    expected = np.stack([_explicit_means(_clouds()), _explicit_means(second_clouds)])
    np.testing.assert_allclose(jax.jit(jax.vmap(kernel))(batched), expected, rtol=1e-14)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.typing
def test_public_result_type_cannot_be_substituted_for_a_distance(
    tmp_path: Path,
) -> None:
    """Raw dispersion preserves the existing estimate carrier's semantic boundary."""
    check_types(
        """\
from typing import assert_type
import jax.numpy as jnp
from statadic.core.axioms import PremetricLaw
from statadic.core.domains import DirectOrigin, EmpiricalDistribution
from statadic.core.matrices import MatrixProvenance, DMat, PairwiseEstimateMatrix
from statadic.core.statistics import VStatisticEstimator
from statadic.discrepancy.mmd.energy import empirical_mean_ground_distances
from statadic.discrepancy.provenance import ExpectedGroundDistance
from statadic.sample.packed import pack

raw = empirical_mean_ground_distances(pack([jnp.ones((2, 1)), jnp.ones((3, 1))]))
assert_type(raw, PairwiseEstimateMatrix[
    EmpiricalDistribution, ExpectedGroundDistance, VStatisticEstimator, DirectOrigin
])
def needs_distance(
    value: DMat[EmpiricalDistribution, MatrixProvenance[PremetricLaw, DirectOrigin]]
) -> None: ...
needs_distance(raw)  # E: is not assignable
""",
        "raw_ground_means.py",
        tmp_path,
    )
