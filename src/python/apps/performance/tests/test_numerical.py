"""Independent witnesses reject finite but scientifically incorrect results."""

from types import SimpleNamespace

import numpy as np
import pytest
from statadic_performance.numerical import linear_reference, rbf_reference, witness


def test_analytic_mmd_gradient_matches_central_difference():
    x = np.array([[0.0, 1.0], [1.0, 2.0], [-1.0, 0.0]])
    y = np.array([[0.5, 1.0], [2.0, -1.0]])
    _, gradient = rbf_reference(x, y)
    epsilon = 1e-5
    numerical = np.zeros_like(x)
    for i, j in np.ndindex(x.shape):
        high, low = x.copy(), x.copy()
        high[i, j] += epsilon
        low[i, j] -= epsilon
        numerical[i, j] = (rbf_reference(high, y)[0] - rbf_reference(low, y)[0]) / (
            2 * epsilon
        )
    np.testing.assert_allclose(gradient, numerical, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("variant", ["mmd-value-gradient", "energy-euclidean"])
def test_wrong_but_finite_discrepancy_is_rejected(variant):
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[1.0], [3.0]])
    value, gradient = rbf_reference(x, y)
    wrong = (value + 1, gradient) if variant == "mmd-value-gradient" else 100.0
    with pytest.raises(ValueError, match="witness disagreement"):
        witness("discrepancy", variant, (x, y), wrong, "float64")


def test_wrong_but_finite_ols_is_rejected():
    x = np.column_stack([np.ones(8), np.arange(8)])
    y = 1 + 2 * x[:, 1] + np.array([0.0, 0.1, -0.2, 0.1, 0.2, -0.1, 0.1, 0.2])
    with pytest.raises(ValueError, match="witness disagreement"):
        witness("ols", "parameters-projection", (x, y), np.zeros(2), "float64")


def test_wrong_but_finite_refits_are_rejected():
    x = np.column_stack([np.ones(8), np.arange(8)])
    y = 1 + 2 * x[:, 1]
    rows = np.array([np.arange(8), np.arange(7, -1, -1)])
    wrong = SimpleNamespace(values=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="witness disagreement"):
        witness("row-refits", "chunk-1", (x, y, rows), wrong, "float64")


def test_witness_rejects_unqualified_design_domain():
    with pytest.raises(ValueError, match="condition"):
        linear_reference(np.ones((8, 2)), np.arange(8))
