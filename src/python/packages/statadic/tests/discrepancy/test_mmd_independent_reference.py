"""Independent Gram-block witnesses for raw, rooted and packed MMD estimates."""

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.mmd import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    distance_induced_kernel,
    mmd,
    mmd_squared,
    mmd_squared_matrix,
    rbf_kernel,
)
from statadic.discrepancy.mmd.energy import UNIT_EXPONENT, energy_geometry
from statadic.geometry.ground.metrics import ANGULAR, EUCLIDEAN
from statadic.sample.packed import PackedClouds, pack

from .mmd._reference import mmd_from_blocks, rbf_block


@pytest.mark.unit
@pytest.mark.reference
def test_zero_mass_atoms_do_not_select_an_invalid_angular_reference() -> None:
    """Direct angular expectations ignore zero-probability support storage."""
    x, y = np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([[-1.0, 0.0], [0.0, -1.0]])
    p, q = np.asarray([0.3, 0.7]), np.asarray([0.5, 0.5])
    expected = (
        2 * p @ np.arccos(x @ y.T) @ q
        - p @ np.arccos(x @ x.T) @ p
        - q @ np.arccos(y @ y.T) @ q
    )
    clouds = PackedClouds.from_clouds(
        [jnp.concatenate([jnp.full((1, 2), jnp.nan), jnp.asarray(x)]), jnp.asarray(y)],
        [jnp.asarray([0.0, 0.3, 0.7]), jnp.asarray(q)],
    )
    kernel = distance_induced_kernel(energy_geometry(ANGULAR, UNIT_EXPONENT))
    np.testing.assert_allclose(
        mmd_squared(clouds[0], clouds[1], kernel=kernel), expected, atol=1e-13
    )
    matrix = mmd_squared_matrix(clouds, kernel=kernel)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix[0, 1], expected, atol=1e-13)


def _block(left: np.ndarray, right: np.ndarray, family: str) -> np.ndarray:
    """Build every oracle block from raw vectors without STATADIC geometry."""
    if family == "linear":
        return left @ right.T
    if family == "cosine":
        return (left @ right.T) / (
            np.linalg.norm(left, axis=1)[:, None]
            * np.linalg.norm(right, axis=1)[None, :]
        )
    if family == "rbf":
        return rbf_block(left, right, 1.3)
    return (
        np.linalg.norm(left, axis=1)[:, None]
        + np.linalg.norm(right, axis=1)[None, :]
        - np.linalg.norm(left[:, None] - right[None, :], axis=-1)
    )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("family", ["linear", "cosine", "rbf", "energy"])
@pytest.mark.parametrize("unbiased", [False, True])
def test_scalar_and_packed_mmd_match_independent_u_v_blocks(
    family: str,
    *,
    unbiased: bool,
) -> None:
    """Unequal counts and repeats exercise n(n-1), n squared and signed diagonals."""
    clouds = [
        np.asarray([[1.0, 0.0], [0.0, 2.0]]),
        np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 1.0]]),
        np.asarray([[2.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [3.0, 0.0]]),
    ]
    kernel = {
        "linear": LINEAR_KERNEL,
        "cosine": COSINE_KERNEL,
        "rbf": rbf_kernel(1.3),
        "energy": distance_induced_kernel(energy_geometry(EUCLIDEAN, UNIT_EXPONENT)),
    }[family]
    expected = np.asarray(
        [
            [
                mmd_from_blocks(
                    _block(x, x, family),
                    _block(y, y, family),
                    _block(x, y, family),
                    unbiased=unbiased,
                )
                for y in clouds
            ]
            for x in clouds
        ]
    )
    if not unbiased:
        np.fill_diagonal(expected, 0.0)
    reference = jnp.zeros(2)

    scalar = mmd_squared(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(clouds[0]))),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(clouds[1]))),
        kernel=kernel,
        reference=reference,
        unbiased=unbiased,
    )
    matrix = mmd_squared_matrix(
        pack([jnp.asarray(x) for x in clouds], require_positive_counts=True),
        kernel=kernel,
        reference=reference,
        unbiased=unbiased,
    )

    assert np.isfinite(matrix).all()
    assert matrix.shape == expected.shape
    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)
    assert float(scalar) == pytest.approx(expected[0, 1], rel=1e-12, abs=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("family", ["linear", "cosine", "rbf", "energy"])
def test_rooted_v_mmd_matches_independent_nonnegative_functional(family: str) -> None:
    """The legal square root is applied once, after the V-statistic reduction."""
    left = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    right = np.asarray([[2.0, 1.0], [-1.0, -1.0], [3.0, 0.0]])
    kernel = {
        "linear": LINEAR_KERNEL,
        "cosine": COSINE_KERNEL,
        "rbf": rbf_kernel(1.3),
        "energy": distance_induced_kernel(energy_geometry(EUCLIDEAN, UNIT_EXPONENT)),
    }[family]
    expected = mmd_from_blocks(
        _block(left, left, family),
        _block(right, right, family),
        _block(left, right, family),
        unbiased=False,
    )

    actual = mmd(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(left))),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(right))),
        kernel=kernel,
        reference=jnp.zeros(2),
    )

    assert float(actual) == pytest.approx(np.sqrt(expected), rel=1e-12, abs=1e-12)


@pytest.mark.unit
@pytest.mark.numerical
def test_raw_u_estimate_retains_negative_squared_value() -> None:
    """Two identical two-point samples have linear-kernel U estimate minus two."""
    points = jnp.asarray([[-1.0], [1.0]])

    actual = mmd_squared(
        PackedClouds.from_array(jnp.asarray(points)),
        PackedClouds.from_array(jnp.asarray(points)),
        kernel=LINEAR_KERNEL,
        unbiased=True,
    )

    assert float(actual) == pytest.approx(-2.0)
