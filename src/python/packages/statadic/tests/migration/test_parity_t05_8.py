# standalone package-maintenance entry point
"""Compare minimization parity and the corrected maximization objective.

Both implementations receive the same inputs.

Not compared against the frozen t05 oracle here (that cross-check runs in the
orchestrator's consolidated pass via the mixture-test ``solver="pgd"`` path);
Minimization guards the t05.8 JIT numerics. Maximization now additionally
considers a best-diagonal vertex, correcting stationary uniform minima on PSD
inputs; its contract is objective dominance over the historical ascent and
global PSD optimality, rather than iterate parity with that defective path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from optax.projections import projection_simplex
from statadic.optimize.simplex import (
    pgd_maximize_quadratic_form,
    pgd_minimize_quadratic_form,
)

from ._parity import DEFAULT_ATOL


def _reference_pgd_simplex_quadratic(
    a: jnp.ndarray, n_steps: int, sign: float
) -> jnp.ndarray:
    """Eager-Python mirror of ``statadic.optimize._pgd_simplex_quadratic``."""
    k = a.shape[0]
    eigvals = jnp.linalg.eigvalsh(a)
    lam_max = jnp.maximum(jnp.max(eigvals), 1e-10)
    step = 1.0 / (2.0 * lam_max)
    w = jnp.ones(k) / k
    for _ in range(n_steps):
        grad = 2.0 * a @ w
        w = projection_simplex(w + sign * step * grad)
    return w


def _psd_matrix(rng: np.random.Generator, k: int) -> jnp.ndarray:
    m = rng.standard_normal((k, k))
    return jnp.asarray(m @ m.T + 1e-3 * np.eye(k))


@pytest.mark.component
@pytest.mark.regression
def test_pgd_minimize_quadratic_form_matches_eager_reference() -> None:
    """Verify pgd minimize quadratic form matches eager reference."""
    rng = np.random.default_rng(0)
    a = _psd_matrix(rng, k=6)
    n_steps = 200

    jitted = pgd_minimize_quadratic_form(a, n_steps=n_steps)
    reference = _reference_pgd_simplex_quadratic(a, n_steps, sign=-1.0)

    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(reference), atol=DEFAULT_ATOL
    )
    np.testing.assert_allclose(float(jnp.sum(jitted)), 1.0, atol=DEFAULT_ATOL)


@pytest.mark.component
@pytest.mark.regression
def test_pgd_maximize_quadratic_form_dominates_historical_ascent() -> None:
    """The PSD vertex safeguard improves or preserves the old ascent objective."""
    rng = np.random.default_rng(1)
    a = _psd_matrix(rng, k=5)
    n_steps = 200

    jitted = pgd_maximize_quadratic_form(a, n_steps=n_steps)
    reference = _reference_pgd_simplex_quadratic(a, n_steps, sign=+1.0)

    objective = float(jitted @ a @ jitted)
    historical_objective = float(reference @ a @ reference)
    assert objective >= historical_objective - DEFAULT_ATOL
    np.testing.assert_allclose(objective, np.max(np.diag(a)), atol=DEFAULT_ATOL)
    np.testing.assert_allclose(float(jnp.sum(jitted)), 1.0, atol=DEFAULT_ATOL)
