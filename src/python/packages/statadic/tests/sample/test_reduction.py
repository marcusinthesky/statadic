"""Tests for dimension-reduction maps (PCA / random projection)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from jax import random
from statadic.sample.reduction import Projection, fit_projection

from .._budgets import examples
from .._strategies import float_arrays

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import jax


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-learn")
def test_pca_heldout_reconstruction_matches_train_fitted_sklearn(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """A shifted test sample must use training centering and the fitted subspace."""
    peer = optional_reference("sklearn.decomposition")
    training = np.asarray(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 1.0],
            [4.0, 2.0, -1.0],
            [7.0, 5.0, 2.0],
            [-1.0, 1.0, 0.0],
            [3.0, 0.0, 2.0],
            [2.0, 4.0, 1.0],
            [5.0, 2.0, 3.0],
        ]
    )
    heldout = np.asarray([[20.0, -4.0, 7.0], [25.0, -2.0, 4.0], [18.0, 0.0, 9.0]])
    reference = peer.PCA(n_components=2, svd_solver="full").fit(training)
    expected_projector = reference.components_.T @ reference.components_
    expected_reconstruction = reference.transform(heldout) @ reference.components_

    fitted = fit_projection(jnp.asarray(training), d_out=2, kind="pca")
    transformed = fitted.transform(jnp.asarray(heldout))

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(
        fitted.mean, training.mean(axis=0), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        fitted.components @ fitted.components.T,
        expected_projector,
        rtol=1e-12,
        atol=1e-12,
    )
    # Reconstruction removes arbitrary SVD signs without using the test mean.
    np.testing.assert_allclose(
        transformed @ fitted.components.T,
        expected_reconstruction,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transformed @ fitted.components.T,
        (heldout - training.mean(axis=0)) @ expected_projector,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.component
@pytest.mark.numerical
def test_supplied_random_projection_uses_frozen_mean_and_matrix() -> None:
    """A common explicit map tests application independently of RNG algorithms."""
    mean = np.asarray([2.0, -1.0, 3.0])
    components = np.asarray([[1.0, 0.0], [0.5, -1.0], [2.0, 1.0]])
    heldout = np.asarray([[10.0, -3.0, 4.0], [-2.0, 7.0, 1.0]])
    projection = Projection(jnp.asarray(mean), jnp.asarray(components), "random")
    expected = (heldout - mean) @ components

    actual = projection.transform(jnp.asarray(heldout))

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def _normal_matrix(rows: int, cols: int) -> jax.Array:
    return random.normal(random.PRNGKey(0), (rows, cols))


class TestFitProjection:
    """Check fitted dimension-reduction maps and validation."""

    @pytest.mark.component
    @pytest.mark.contract
    @example(x=_normal_matrix(40, 500))
    @given(x=float_arrays((40, 60)))
    @examples(20)
    def test_pca_projection_shape(self, x: jax.Array) -> None:
        """Verify pca projection shape."""
        proj = fit_projection(x, d_out=15, kind="pca")
        assert proj.components.shape == (x.shape[1], 15)
        assert proj.transform(x).shape == (x.shape[0], 15)

    @pytest.mark.component
    @pytest.mark.contract
    @example(x=_normal_matrix(40, 500))
    @given(x=float_arrays((40, 60)))
    @examples(20)
    def test_random_projection_shape(self, x: jax.Array) -> None:
        """Verify random projection shape."""
        proj = fit_projection(x, d_out=15, kind="random", seed=0)
        assert proj.transform(x).shape == (x.shape[0], 15)

    @pytest.mark.component
    @pytest.mark.contract
    @example(x=_normal_matrix(10, 500))
    @given(x=float_arrays((10, 60)))
    @examples(20)
    def test_projection_clamps_dim(self, x: jax.Array) -> None:
        """Verify projection clamps dim."""
        proj = fit_projection(x, d_out=50, kind="pca")
        # d_out clamped to <= n-1.
        maximum_components = x.shape[0] - 1
        assert proj.components.shape[1] <= maximum_components
