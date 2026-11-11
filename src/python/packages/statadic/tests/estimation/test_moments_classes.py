"""Native moment estimator bricks preserve dynamic weights under JIT."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import PlainLongRun
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem


@pytest.mark.component
@pytest.mark.contract
def test_gmm_classes_match_existing_solve_and_dynamic_weight() -> None:
    """Preserve iterative solves and trace fixed weights as data."""
    t = jnp.arange(30.0)
    g = jnp.stack((jnp.sin(t), jnp.cos(t * 0.7)), axis=1)
    d = jnp.array([[1.0], [0.5]])
    fixed = FixedWeightGMM(PlainLongRun)
    weight = jnp.array([[2.0, 0.1], [0.1, 1.0]])
    result = jax.jit(fixed.estimate)(AffineMomentProblem(g, d, weight))
    np.testing.assert_array_equal(result.weight, weight)
    assert bool(result.valid)
