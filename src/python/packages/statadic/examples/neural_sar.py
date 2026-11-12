"""Learn an atom-level NNX projection through a Gaussian SAR likelihood.

Run with the locked documentation environment, which supplies Flax. The final
Regression report conditions on the trained graph; it does not include neural
training uncertainty.
"""

# ruff: noqa: INP001, T201  # Standalone executable documentation.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.representation import Energy
from statadic.api.spatial.graphs import (
    DistanceGraph,
    PanelData,
    RBFWeights,
    SpatialLag,
)
from statadic.estimation.spatial.graph_likelihood import graph_profile
from statadic.operators.spatial.distance import rbf_weights
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    import jax
    from statadic.api.spatial.observed import ObservedSpatialPanelResult


class NeuralSAR(nnx.Module):
    """Project six-dimensional atoms to two coordinates and learn one SAR rho."""

    def __init__(self, seed: int = 7) -> None:
        """Initialize a small MLP and a nonzero spatial coefficient."""
        rngs = nnx.Rngs(seed)
        self.hidden = nnx.Linear(6, 8, rngs=rngs)
        self.projection = nnx.Linear(8, 2, rngs=rngs, use_bias=False)
        self.raw_rho = nnx.Param(jnp.asarray(0.2))

    def __call__(self, vectors: jax.Array) -> jax.Array:
        """Return a bounded lower-dimensional representation of each atom."""
        return jnp.tanh(self.projection(jnp.tanh(self.hidden(vectors))))

    @property
    def rho(self) -> jax.Array:
        """Keep rho strictly inside the row-normalized graph's stable region."""
        return 0.95 * jnp.tanh(self.raw_rho[...])

    def costs(self, clouds: PackedClouds) -> jax.Array:
        """Compare projected distributions, preserving their atom masses."""
        projected = clouds.with_coordinates(self(clouds.coordinates))
        energy = Energy(ground="euclidean").squared_cross(projected, projected)
        # Mirror once to satisfy the host graph's exact symmetry contract.
        upper = jnp.triu(energy, k=1)
        return upper + upper.T

    def weights(self, clouds: PackedClouds) -> jax.Array:
        """Use raw V-energy as the RBF cost, with fixed unit bandwidth."""
        return rbf_weights(self.costs(clouds), jnp.asarray(1.0))


def simulate() -> tuple[PackedClouds, jax.Array, jax.Array]:
    """Generate eight units with three atoms each and 24 independent SAR periods."""
    rng = np.random.default_rng(1978)
    vectors = rng.normal(size=(8, 3, 6))
    # An independent NumPy teacher: linear projection and empirical V-energy.
    latent = np.tanh(vectors[..., :2])
    ground = np.linalg.norm(
        latent[:, None, :, None, :] - latent[None, :, None, :, :], axis=-1
    ).mean(axis=(-1, -2))
    cost = 2 * ground - ground.diagonal()[:, None] - ground.diagonal()[None, :]
    weights = np.exp(-cost)
    np.fill_diagonal(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    x = rng.normal(size=(24, 8))
    innovations = rng.normal(scale=0.15, size=x.shape)
    y = np.linalg.solve(np.eye(8) - 0.35 * weights, (0.4 + 0.7 * x + innovations).T).T
    clouds = PackedClouds.from_array(jnp.asarray(vectors))
    return clouds, jnp.asarray(x), jnp.asarray(y)


def profile(
    model: NeuralSAR, clouds: PackedClouds, x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Profile beta and innovation variance while retaining all graph derivatives."""
    n_periods, n_units = x.shape
    lag = (model.weights(clouds)[None],) * n_periods
    error = (jnp.empty((0, n_units, n_units), dtype=x.dtype),) * n_periods
    design = tuple(jnp.column_stack((jnp.ones_like(row), row)) for row in x)
    return graph_profile(lag, error, design, tuple(y), model.rho[None])


def loss(
    model: NeuralSAR, clouds: PackedClouds, x: jax.Array, y: jax.Array
) -> jax.Array:
    """Minimize negative Gaussian profile log likelihood per observation."""
    return -profile(model, clouds, x, y)[2] / y.size


@nnx.jit
def train_step(
    model: NeuralSAR,
    optimizer: nnx.Optimizer,
    clouds: PackedClouds,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Differentiate through energy, graph weights and the SAR log determinant."""
    objective, gradients = nnx.value_and_grad(loss)(model, clouds, x, y)
    optimizer.update(model, gradients)
    return objective


def conditional_fit(
    model: NeuralSAR, clouds: PackedClouds, x: jax.Array, y: jax.Array
) -> ObservedSpatialPanelResult:
    """Freeze the learned costs, then estimate and report a conditional SAR fit."""
    graph = DistanceGraph.from_costs(
        model.costs(clouds),
        unit_ids=tuple(range(x.shape[1])),
        weighting=RBFWeights(1.0),
    )
    rows = Observations(
        {
            "unit": jnp.tile(jnp.arange(x.shape[1]), x.shape[0]),
            "period": jnp.repeat(jnp.arange(x.shape[0]), x.shape[1]),
            "x": x.ravel(),
            "y": y.ravel(),
        }
    )
    panel = PanelData(rows, PanelLayout("unit", "period"), graphs={"learned": graph})
    return (
        Regression("y", Intercept() + Linear("x"), dependence=SpatialLag("learned"))
        .estimate(panel)
        .require_valid()
    )


def run(
    steps: int = 120,
) -> tuple[NeuralSAR, ObservedSpatialPanelResult]:
    """Train on deterministic simulated data and return the conditional fit."""
    clouds, x, y = simulate()
    model = NeuralSAR()
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    initial = float(loss(model, clouds, x, y))
    for _ in range(steps):
        train_step(model, optimizer, clouds, x, y)
    final = float(loss(model, clouds, x, y))
    print(
        f"Negative profile log likelihood / observation: {initial:.4f} -> {final:.4f}"
    )
    print(f"Jointly trained rho: {float(model.rho):.4f}")
    fit = conditional_fit(model, clouds, x, y)
    print(fit.report().to_text())
    return model, fit


if __name__ == "__main__":
    run()
