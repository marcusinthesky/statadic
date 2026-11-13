"""Differentiable parameter-bearing strategies retain scientific contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.domains import (
    FiniteValues,
)
from statadic.discrepancy.mmd import mmd_squared, mmd_squared_matrix
from statadic.discrepancy.mmd.components import (
    candidate_energy_components,
    candidate_kernel_components,
)
from statadic.discrepancy.mmd.energy import (
    UNIT_EXPONENT,
    energy_distance,
    energy_distance_matrix_values,
    energy_geometry,
)
from statadic.discrepancy.mmd.energy.kernel import distance_induced_kernel
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.geometry.ground.similarities import RBFKernel, gram, rbf_gram, rbf_kernel
from statadic.sample.packed import PackedClouds, pack
from statadic.sample.reduction import Projection

if TYPE_CHECKING:
    from jax import Array
    from statadic.core.domains import ClaimAuthority, SpaceDescriptor
    from statadic.core.transforms import TransformDescriptor
    from statadic.geometry.ground.metrics import EuclideanLaw
    from statadic.geometry.ground.similarities import (
        PositiveSemidefiniteKernelLaw,
        SimilarityStrategy,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ScaledEuclidean:
    """Test-local positive scale parameter under an unchanged Euclidean law."""

    scale: jax.Array

    @property
    def name(self) -> str:
        return "scaled_euclidean"

    @property
    def backend(self) -> Literal["jax"]:
        return "jax"

    @property
    def law(self) -> EuclideanLaw:
        return EUCLIDEAN.law

    @property
    def space(self) -> SpaceDescriptor:
        return EUCLIDEAN.space

    @property
    def transform(self) -> TransformDescriptor:
        return EUCLIDEAN.transform

    @property
    def authority(self) -> ClaimAuthority:
        return EUCLIDEAN.authority

    @property
    def required_evidence(self) -> type[FiniteValues]:
        return FiniteValues

    @property
    def requires_nonzero_rows(self) -> bool:
        return False

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.scale * EUCLIDEAN(x, y)


def _mmd_reference(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Independent broadcast-distance Gaussian V-statistic calculation."""

    def block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
        return np.exp(-squared / (2.0 * bandwidth**2))

    return float(block(x, x).mean() + block(y, y).mean() - 2 * block(x, y).mean())


def _bandwidth_gradient_reference(
    x: np.ndarray, y: np.ndarray, bandwidth: float
) -> float:
    """Analytic Gaussian bandwidth derivative under the same V normalization."""

    def block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
        return np.exp(-squared / (2 * bandwidth**2)) * squared / bandwidth**3

    return float(block(x, x).mean() + block(y, y).mean() - 2 * block(x, y).mean())


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_rbf_coincident_coordinate_gradient(dtype: jnp.dtype) -> None:
    """Self-Grams and duplicate rows have finite analytic coordinate gradients."""
    x = jnp.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, -0.5]], dtype=dtype)
    bandwidth = 1.3

    def evaluate_total(rows: jax.Array) -> Array:
        return jnp.sum(rbf_gram(rows, rows, bandwidth=bandwidth))

    gradient = jax.jit(jax.grad(evaluate_total))(x)
    host = np.asarray(x)
    difference = host[:, None, :] - host[None, :, :]
    weights = np.exp(-np.sum(difference**2, axis=-1) / (2 * bandwidth**2))
    expected = -2 * np.sum(weights[:, :, None] * difference, axis=1) / bandwidth**2
    tolerance = 20 * np.finfo(host.dtype).eps
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(gradient, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dynamic_rbf_parameter_transforms(dtype: jnp.dtype) -> None:
    """Bandwidth remains an array leaf through JIT, batching, JVP and VJP."""
    x = jnp.asarray([[0.0], [1.0], [1.0]], dtype=dtype)
    y = jnp.asarray([[0.0], [2.0]], dtype=dtype)
    bandwidth = jnp.asarray(1.2, dtype=dtype)

    def objective(kernel: RBFKernel) -> jax.Array:
        return mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=kernel,
        )

    kernel = RBFKernel(bandwidth)

    def evaluate_jvp(item: RBFKernel) -> tuple[jax.Array, jax.Array]:
        return jax.jvp(objective, (item,), (RBFKernel(jnp.ones_like(item.bandwidth)),))

    value, tangent = jax.jit(evaluate_jvp)(kernel)
    gradient = jax.jit(jax.grad(objective))(kernel).bandwidth
    batched = jax.jit(jax.vmap(objective))(
        RBFKernel(jnp.asarray([0.8, 1.2, 2.0], dtype=dtype))
    )
    tolerance = 40 * np.finfo(np.asarray(x).dtype).eps
    np.testing.assert_allclose(
        value,
        _mmd_reference(np.asarray(x), np.asarray(y), 1.2),
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(tangent, gradient, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(batched[1], value, atol=tolerance, rtol=tolerance)
    expected = _bandwidth_gradient_reference(np.asarray(x), np.asarray(y), 1.2)
    np.testing.assert_allclose(gradient, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        gram(x, y, rbf_kernel(1.2)), gram(x, y, kernel), atol=tolerance, rtol=tolerance
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("bandwidth", [-1.0, 0.0, float("inf"), float("nan")])
def test_invalid_traced_bandwidth_is_visible(bandwidth: float) -> None:
    """A traced inadmissible bandwidth cannot acquire a valid Gaussian Gram."""
    rows = jnp.asarray([[0.0], [1.0]])

    def evaluate_gram(value: jax.Array) -> Array:
        return gram(rows, rows, RBFKernel(value))

    result = jax.jit(evaluate_gram)(jnp.asarray(bandwidth))
    assert bool(jnp.all(jnp.isnan(result)))


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_projected_unequal_mmd_coordinate_and_parameter_gradients(
    dtype: jnp.dtype,
) -> None:
    """Packed unequal samples and candidate components retain a learned map."""
    x = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    y = jnp.asarray([[0.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=dtype)
    projection = Projection(
        jnp.zeros(2, dtype=dtype), jnp.asarray([[1.0], [0.5]], dtype=dtype), "supplied"
    )
    kernel = RBFKernel(jnp.asarray(0.9, dtype=dtype))

    def objective(mapping: Projection, strategy: RBFKernel) -> jax.Array:
        left, right = mapping.transform(x), mapping.transform(y)
        matrix = mmd_squared_matrix(
            pack([left, right], require_positive_counts=True), kernel=strategy
        )
        components, cross, target = candidate_kernel_components(
            PackedClouds.from_array(jnp.asarray(left)),
            pack([left, right], require_positive_counts=True),
            strategy,
        )
        return matrix[0, 1] + components[1, 1] - 2 * cross[1] + target

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        projection, kernel
    )
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradients[0].mean)))
    assert bool(jnp.all(jnp.isfinite(gradients[0].components)))
    assert bool(jnp.isfinite(gradients[1].bandwidth))
    tolerance = 40 * np.finfo(np.asarray(x).dtype).eps
    np.testing.assert_allclose(
        value,
        2
        * _mmd_reference(
            np.asarray(projection.transform(x)),
            np.asarray(projection.transform(y)),
            0.9,
        ),
        atol=tolerance,
        rtol=tolerance,
    )
    # Centering translates both clouds, so this stationary kernel must be invariant.
    np.testing.assert_allclose(gradients[0].mean, 0.0, atol=tolerance)
    mapping = np.asarray(projection.components)

    def component_gradient(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        difference = left[:, None, :] - right[None, :, :]
        projected = (difference @ mapping)[..., 0]
        weights = np.exp(-(projected**2) / (2 * 0.9**2))
        return (
            -np.mean((weights * projected)[..., None] * difference, axis=(0, 1))
            / 0.9**2
        )

    host_x, host_y = np.asarray(x), np.asarray(y)
    expected = 2 * (
        component_gradient(host_x, host_x)
        + component_gradient(host_y, host_y)
        - 2 * component_gradient(host_x, host_y)
    )
    np.testing.assert_allclose(
        gradients[0].components[:, 0], expected, atol=tolerance, rtol=tolerance
    )


@pytest.mark.integration
@pytest.mark.numerical
def test_dynamic_ground_composes_through_energy_geometry() -> None:
    """Ground leaves survive geometry, induced kernels and packed reductions."""
    x = jnp.asarray([[0.0], [1.0]])
    y = jnp.asarray([[2.0], [3.0], [5.0]])

    def objectives(strategy: _ScaledEuclidean) -> jax.Array:
        geometry = energy_geometry(strategy, UNIT_EXPONENT)
        kernel = distance_induced_kernel(geometry)
        components, cross = candidate_energy_components(
            PackedClouds.from_array(jnp.asarray(x)),
            pack([x, y], require_positive_counts=True),
            strategy,
        )
        energy = energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            metric=geometry.ground,
        )
        matrix = energy_distance_matrix_values(
            pack([x, y], require_positive_counts=True), metric=geometry.ground
        )
        return jnp.stack(
            (
                energy,
                matrix[0, 1],
                mmd_squared(
                    PackedClouds.from_array(jnp.asarray(x)),
                    PackedClouds.from_array(jnp.asarray(y)),
                    kernel=kernel,
                ),
                2 * cross[1] - components[0, 0] - components[1, 1],
            )
        )

    scale = jnp.asarray(1.7)
    values = jax.jit(objectives)(_ScaledEuclidean(scale))
    derivatives = jax.jit(jax.jacfwd(objectives))(_ScaledEuclidean(scale)).scale

    def evaluate_total(strategy: _ScaledEuclidean) -> Array:
        return jnp.sum(objectives(strategy))

    reverse_derivative = jax.jit(jax.grad(evaluate_total))(
        _ScaledEuclidean(scale)
    ).scale
    reference = energy_distance(
        PackedClouds.from_array(jnp.asarray(x)),
        PackedClouds.from_array(jnp.asarray(y)),
        metric=EUCLIDEAN,
    )
    np.testing.assert_allclose(values, jnp.repeat(scale * reference, 4), atol=1e-12)
    np.testing.assert_allclose(derivatives, jnp.repeat(reference, 4), atol=1e-12)
    np.testing.assert_allclose(reverse_derivative, 4 * reference, atol=1e-12)
    kernel = distance_induced_kernel(
        energy_geometry(_ScaledEuclidean(scale), UNIT_EXPONENT)
    )
    tangent_kernel = distance_induced_kernel(
        energy_geometry(_ScaledEuclidean(jnp.ones_like(scale)), UNIT_EXPONENT)
    )

    def induced_objective(
        strategy: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    ) -> jax.Array:
        return mmd_squared(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(y)),
            kernel=strategy,
        )

    def evaluate_jvp(
        strategy: jax.Array, tangent: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.jvp(induced_objective, (strategy,), (tangent,))

    induced_value, induced_tangent = jax.jit(evaluate_jvp)(kernel, tangent_kernel)
    np.testing.assert_allclose(induced_value, scale * reference, atol=1e-12)
    np.testing.assert_allclose(induced_tangent, reference, atol=1e-12)
