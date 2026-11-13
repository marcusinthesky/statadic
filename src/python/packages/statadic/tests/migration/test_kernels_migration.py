# jaxtyping shape strings are runtime contracts.
"""Migration tests: jaxopt BoxOSQP → optax PGD.

Verifies the PGD simplex solver (_pgd_simplex) against analytic optima on
three QP instances with known closed-form solutions, plus constraint-satisfaction
checks on all problem instances used for regression.

Independent analytic assertions avoid comparing identical solver paths.
"""

import jax.numpy as jnp
import pytest
from jax import config, random
from statadic.discrepancy.mmd.energy.balancing import kernel_balancing_weights
from statadic.optimize.simplex import pgd_simplex_affine as _pgd_simplex

config.update("jax_enable_x64", val=True)

_ANALYTIC_TOLERANCE = 1e-4
_SIMPLEX_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Problem instances (seeded, varied K and cluster structure) — used for
# constraint-satisfaction regression only (non-circular).
# ---------------------------------------------------------------------------


def _make_instances() -> list[tuple[str, jnp.ndarray, list[jnp.ndarray]]]:
    """Build ~5 problem instances as (name, target, candidates)."""
    instances = []

    # 1. K=2, target near cand1
    key = random.PRNGKey(0)
    target = random.normal(key, (20, 3))
    cand1 = target + random.normal(random.PRNGKey(1), (20, 3)) * 0.05
    cand2 = random.normal(random.PRNGKey(2), (20, 3)) + 10.0
    instances.append(("K2_near", target, [cand1, cand2]))

    # 2. K=3, symmetric cluster
    target = random.normal(random.PRNGKey(10), (30, 2))
    cands = [
        random.normal(random.PRNGKey(10 + i), (30, 2)) + jnp.array([i * 2.0, 0.0])
        for i in range(3)
    ]
    instances.append(("K3_symmetric", target, cands))

    # 3. K=5, target equidistant
    target = jnp.zeros((10, 2))
    cands = [
        jnp.ones((10, 2))
        * jnp.array([jnp.cos(i * 2 * jnp.pi / 5), jnp.sin(i * 2 * jnp.pi / 5)])
        * 3.0
        for i in range(5)
    ]
    instances.append(("K5_equidistant", target, cands))

    # 4. K=10, random cloud
    target = random.normal(random.PRNGKey(99), (50, 4))
    cands = [random.normal(random.PRNGKey(100 + i), (50, 4)) for i in range(10)]
    instances.append(("K10_random", target, cands))

    # 5. K=3, degenerate (two identical candidates)
    target = random.normal(random.PRNGKey(42), (15, 2))
    base = random.normal(random.PRNGKey(43), (15, 2)) + 5.0
    instances.append(("K3_degenerate", target, [base, base, target + 0.01]))

    return instances


_INSTANCES = _make_instances()


# ---------------------------------------------------------------------------
# Analytic assertions for _pgd_simplex
# ---------------------------------------------------------------------------


class TestPGDSimplexAnalytic:
    """Check ``_pgd_simplex`` against independent analytic optima.

    Every problem minimizes ``½ wᵀ quadratic w + cᵀ w`` on the probability
    simplex.
    The analytic solutions are derived in the docstring of each test.
    """

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_min_variance_diag(self) -> None:
        """Min-variance on simplex: quadratic = 2·diag(1, 4), c = 0.

        Analytic derivation:
            Objective on simplex (w2 = 1 - w1):
                f(w1) = w1² + 4(1-w1)² = 5w1² - 8w1 + 4
            Unconstrained minimum: df/dw1 = 10w1 - 8 = 0  →  w1* = 0.8
            w1* ∈ [0,1], so the interior point is feasible.
            Optimum: w* = [0.8, 0.2].
        """
        quadratic = 2.0 * jnp.diag(jnp.array([1.0, 4.0]))
        c = jnp.zeros(2)
        w = _pgd_simplex(quadratic, c)
        expected = jnp.array([0.8, 0.2])
        assert jnp.allclose(w, expected, atol=_ANALYTIC_TOLERANCE)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_active_lower_bound_corner(self) -> None:
        """Verify an active lower bound produces the expected simplex corner.

        QP: quadratic = 2·I₂, c = [10, 0].
        Analytic derivation:
            The unconstrained minimiser is ``[-5, 0]``, violating ``w1 ≥ 0``.
            On the simplex with w2 = 1 - w1:
                f(w1) = w1² + (1-w1)² + 10·w1 = 2w1² + 8w1 + 1
            df/dw1 = 4w1 + 8 > 0 for all w1 ≥ 0  →  f is increasing on [0,1].
            Constrained minimum at w1 = 0, i.e. w* = [0, 1].
        """
        quadratic = 2.0 * jnp.eye(2)
        c = jnp.array([10.0, 0.0])
        w = _pgd_simplex(quadratic, c)
        expected = jnp.array([0.0, 1.0])
        assert jnp.allclose(w, expected, atol=_ANALYTIC_TOLERANCE)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_correlated_symmetric(self) -> None:
        """Correlated Σ = [[1, 0.5], [0.5, 1]]: optimum [0.5, 0.5] by symmetry.

        QP: quadratic = 2·Σ, c = 0.
        Analytic derivation:
            On simplex (w2 = 1 - w1):
                f(w1) = w1² + (1-w1)² + w1(1-w1) = w1² - w1 + 1
            df/dw1 = 2w1 - 1 = 0  →  w1* = 0.5  (interior, feasible).
            Optimum: w* = [0.5, 0.5].
        """
        covariance = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        quadratic = 2.0 * covariance
        c = jnp.zeros(2)
        w = _pgd_simplex(quadratic, c)
        expected = jnp.array([0.5, 0.5])
        assert jnp.allclose(w, expected, atol=_ANALYTIC_TOLERANCE)


# ---------------------------------------------------------------------------
# Constraint-satisfaction regression (non-circular: checks properties, not
# solver-vs-solver parity).
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSimplexConstraints:
    """Check simplex constraints across all regression instances."""

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.parametrize(
        ("name", "target", "candidates"),
        _INSTANCES,
        ids=[instance[0] for instance in _INSTANCES],
    )
    def test_simplex_constraint(
        self,
        name: str,
        target: jnp.ndarray,
        candidates: list[jnp.ndarray],
    ) -> None:
        """Weights sum to 1 and are non-negative for every instance."""
        w = kernel_balancing_weights(target, candidates, solver="pgd")
        assert abs(float(jnp.sum(w)) - 1.0) < _SIMPLEX_TOLERANCE, (
            f"{name}: sum(w)={jnp.sum(w)}"
        )
        assert float(jnp.min(w)) >= -_SIMPLEX_TOLERANCE, f"{name}: min(w)={jnp.min(w)}"
