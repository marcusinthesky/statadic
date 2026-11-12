"""Exercise NNX training through energy geometry and the real SAR likelihood."""

import re
import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

pytestmark = pytest.mark.native
ROOT = Path(__file__).resolve().parents[5]
EXAMPLE = ROOT / "src/python/packages/statadic/examples/neural_sar.py"
NEURAL = runpy.run_path(str(EXAMPLE))


@pytest.fixture
def sample():
    """Prepare the deterministic arrays in the caller's JAX precision."""
    clouds, x, y = NEURAL["simulate"]()
    return NEURAL["NeuralSAR"](), clouds, x, y


def numpy_reference(model, clouds, x, y):
    """Compute V-energy, graph weights and the SAR profile independently."""
    atoms = np.asarray(model(clouds.coordinates), dtype=np.float64)
    masses = np.asarray(clouds.masses, dtype=np.float64)
    ground = np.linalg.norm(atoms[:, None, :, None] - atoms[None, :, None, :], axis=-1)
    mean = np.einsum("ijab,ia,jb->ij", ground, masses, masses)
    cost = 2 * mean - mean.diagonal()[:, None] - mean.diagonal()[None, :]
    weights = np.exp(-cost)
    np.fill_diagonal(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    a = np.eye(x.shape[1]) - float(model.rho) * weights
    design = np.column_stack((np.ones(x.size), np.asarray(x).ravel()))
    response = (np.asarray(y) @ a.T).ravel()
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    sigma2 = np.mean((response - design @ beta) ** 2)
    likelihood = len(y) * np.linalg.slogdet(a)[1]
    likelihood -= y.size * (np.log(2 * np.pi * sigma2) + 1) / 2
    return cost, weights, beta, sigma2, likelihood


def test_energy_graph_and_sar_profile_match_independent_numpy(sample):
    model, clouds, x, y = sample
    cost, weights, beta, sigma2, likelihood = numpy_reference(*sample)
    actual_cost = np.asarray(model.costs(clouds))
    actual_weights = np.asarray(model.weights(clouds))
    np.testing.assert_allclose(actual_cost, cost, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(actual_cost, actual_cost.T)
    np.testing.assert_array_equal(np.diag(actual_cost), 0)
    np.testing.assert_allclose(actual_weights, weights, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(actual_weights.sum(axis=1), 1, atol=2e-6)
    np.testing.assert_array_equal(np.diag(actual_weights), 0)
    assert np.all(actual_weights >= 0)
    actual = NEURAL["profile"](model, clouds, x, y)
    np.testing.assert_allclose(actual[0], beta, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(actual[1], sigma2, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(actual[2], likelihood, atol=2e-4, rtol=2e-5)
    assert model(clouds.coordinates).shape == (8, 3, 2)


def test_profile_gradients_reach_mlp_and_rho(sample):
    model, clouds, x, y = sample
    objective = NEURAL["loss"]
    value, gradients = nnx.value_and_grad(objective)(model, clouds, x, y)
    assert np.isfinite(value)
    leaves = jax.tree.leaves(gradients)
    assert all(np.all(np.isfinite(leaf)) for leaf in leaves)
    assert float(jnp.linalg.norm(gradients["hidden"]["kernel"][...])) > 1e-6
    assert float(jnp.linalg.norm(gradients["projection"]["kernel"][...])) > 1e-6
    assert abs(float(gradients["raw_rho"][...])) > 1e-6
    # Include an MLP weight and rho so a detached graph cannot pass this check.
    kernel = model.hidden.kernel[...]
    raw_rho = model.raw_rho[...]
    direction = jnp.zeros_like(kernel).at[0, 0].set(1)
    step = 1e-3 if kernel.dtype == jnp.float32 else 1e-5

    def evaluate(offset):
        model.hidden.kernel[...] = kernel + offset * direction
        model.raw_rho[...] = raw_rho + offset
        return float(objective(model, clouds, x, y))

    difference = (evaluate(step) - evaluate(-step)) / (2 * step)
    model.hidden.kernel[...] = kernel
    model.raw_rho[...] = raw_rho
    derivative = float(gradients["hidden"]["kernel"][0, 0] + gradients["raw_rho"][...])
    np.testing.assert_allclose(derivative, difference, atol=2e-4, rtol=3e-3)


def test_readme_training_changes_network_and_produces_valid_conditional_sar(sample):
    model, clouds, x, y = sample
    initial_kernel = np.asarray(model.hidden.kernel[...]).copy()
    initial_projection = np.asarray(model.projection.kernel[...]).copy()
    initial_rho = float(model.rho)
    initial_weights = np.asarray(model.weights(clouds))
    initial_loss = float(NEURAL["loss"](*sample))
    readme = (ROOT / "README.md").read_text()
    section = readme.split("<!-- statadic:readme-neural-sar:start -->", 1)[1]
    section = section.split("<!-- statadic:readme-neural-sar:end -->", 1)[0]
    source = re.search(r"```python\n(.*?)\n```", section, re.S)
    assert source is not None
    namespace = {
        "NeuralSAR": NEURAL["NeuralSAR"],
        "clouds": clouds,
        "y": y,
        "design": tuple(jnp.column_stack((jnp.ones_like(row), row)) for row in x),
        "no_errors": (jnp.empty((0, 8, 8), dtype=x.dtype),) * len(y),
    }
    exec(compile(source[1], "README.md:neural-sar", "exec"), namespace)
    trained = namespace["model"]
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    for _ in range(120):
        NEURAL["train_step"](model, optimizer, clouds, x, y)
    assert float(NEURAL["loss"](model, clouds, x, y)) < initial_loss - 0.01
    # Compare the mathematical objectives at one state: tiny compiler rounding
    # differences can accumulate over nonconvex Adam trajectories.
    documented_value, documented_gradients = nnx.value_and_grad(namespace["objective"])(
        trained
    )
    example_value, example_gradients = nnx.value_and_grad(NEURAL["loss"])(
        trained, clouds, x, y
    )
    np.testing.assert_allclose(documented_value, example_value, atol=2e-6, rtol=2e-5)
    for documented, example in zip(
        jax.tree.leaves(documented_gradients),
        jax.tree.leaves(example_gradients),
        strict=True,
    ):
        np.testing.assert_allclose(documented, example, atol=2e-6, rtol=2e-4)
    final_loss = float(NEURAL["loss"](trained, clouds, x, y))
    assert final_loss < initial_loss - 0.01
    assert not np.allclose(trained.hidden.kernel[...], initial_kernel)
    assert not np.allclose(trained.projection.kernel[...], initial_projection)
    assert abs(float(trained.rho) - initial_rho) > 0.01
    assert abs(float(trained.rho)) < 0.95
    assert not np.allclose(trained.weights(clouds), initial_weights)
    fit = NEURAL["conditional_fit"](trained, clouds, x, y)
    assert bool(fit.valid)
    assert np.all(np.isfinite(fit.parameters))
    assert np.isfinite(float(fit.core.log_likelihood))
    assert fit.report().parameters
    # The named Regression refit agrees with the independent Gaussian profile
    # evaluated at its conditional rho, rather than assuming joint convergence.
    trained.raw_rho[...] = jnp.arctanh(fit.parameters[-1] / 0.95)
    _, _, beta, sigma2, likelihood = numpy_reference(trained, clouds, x, y)
    np.testing.assert_allclose(fit.parameters[:-1], beta, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(fit.core.sigma2, sigma2, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(
        fit.core.log_likelihood, likelihood, atol=2e-3, rtol=2e-4
    )
