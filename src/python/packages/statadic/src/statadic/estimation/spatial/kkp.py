# jaxtyping numerical carrier contracts.
"""Fixed-W Kapoor-Kelejian-Prucha random-effects spatial error GMM.

The KKP component model spatially filters both the entity and idiosyncratic
errors. It differs from the marginal ML model with unfiltered entity effects in
``spatial_panel``. KKP's temporal moment projections require constant W; use the
time-varying-W marginal likelihood for changing networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from statadic.core.results import (
    SolverResult,  # noqa: TC001
)
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
)
from statadic.estimation.linear import (  # noqa: TC001
    LeastSquares,
    LinearFit,
    LinearProblem,
)
from statadic.estimation.spatial.likelihood import SEM
from statadic.estimation.spatial.panel import (
    SpatialPanelProblem,  # noqa: TC001 -- runtime annotation
)
from statadic.operators.spatial import (  # noqa: TC001
    SpatialPanelWeights,
    spatial_panel_lag,
)
from statadic.optimize.backends import bounded_scalar_minimise

__all__ = ["KKP", "KKPPanelFit"]


_DESIGN_DIMENSIONS = 2
_PANEL_DIMENSIONS = 3
_MIN_PERIODS = 2


def _full_weight_tau(w: Array) -> Array:
    """KKP (2007), equation 26: Gaussian quadratic-form covariance.

    For zero-diagonal W, the centered forms have symmetric matrices I,
    W'W and (W+W')/2. Their covariance is 2*tr(A_i A_j)/N.
    """
    n = w.shape[0]
    gram = w.T @ w
    cross = jnp.sum(gram * (w + w.T)) / n
    return jnp.array(
        [
            [2.0, 2 * jnp.sum(w * w) / n, 0.0],
            [2 * jnp.sum(w * w) / n, 2 * jnp.sum(gram * gram) / n, cross],
            [0.0, cross, (jnp.sum(w * w.T) + jnp.sum(w * w)) / n],
        ],
        dtype=w.dtype,
    )


@dataclass(frozen=True)
class KKP:
    """Constant zero-diagonal-graph KKP component estimator.

    Full weighting uses the normal-error covariance in KKP (2007), equation
    26. Some peer implementations use different trace normalizations; their
    full-weight estimates are not an equivalent reference for this option.
    """

    full_weights: bool = False
    tolerance: float = 1e-7

    def estimate(self, problem: SpatialPanelProblem) -> KKPPanelFit:
        """Estimate the retained KKP component model."""
        if (
            not isinstance(problem.process, SEM)
            or problem.effects != "random"
            or problem.slx_columns
        ):
            message = "KKP requires a SEM random-component problem without SLX columns."
            raise ValueError(message)
        return self._estimate_compiled(problem)

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_compiled(self, problem: SpatialPanelProblem) -> KKPPanelFit:
        return _kkp_panel_fit(
            problem.design,
            problem.outcome,
            problem.weights,
            full_weights=self.full_weights,
            parameter_names=problem.parameter_names,
            tolerance=self.tolerance,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class KKPPanelFit:
    """Actual transformed coefficient fit and separately estimated error components.

    ``linear`` supplies conventional plug-in beta covariance. This API does not
    fabricate standard errors for lambda or variance components; those require
    a separately qualified joint GMM covariance, which spreg also does not expose
    through its KKP beta covariance matrix.
    """

    linear: LinearFit
    design: Array
    outcome: Array
    weights: SpatialPanelWeights
    spatial_lambda: Array
    variance_components: Array
    theta: Array
    residuals: Array
    lambda_solver: SolverResult[Array]
    valid: Array

    @property
    def parameters(self) -> Array:
        """Return regression coefficients; lambda is separately named."""
        return jnp.where(self.valid[..., None], self.linear.parameters, jnp.nan)

    @property
    def parameter_names(self) -> tuple[str, ...] | None:
        """Return the regression coefficient names of the transformed fit."""
        return self.linear.parameter_names

    @property
    def solver(self) -> SolverResult[Array]:
        """Return the lambda solve, the only iterative step of the KKP fit."""
        return self.lambda_solver


def _moments(u: Array, w: Array, *, between: bool) -> tuple[Array, Array]:
    """Assemble KKP's three within or between quadratic equations."""
    t, n = u.shape
    wu = spatial_panel_lag(w, u)
    wwu = spatial_panel_lag(w, wu)
    if between:
        qu = jnp.broadcast_to(jnp.mean(u, axis=0), u.shape)
        qw = jnp.broadcast_to(jnp.mean(wu, axis=0), u.shape)
        qww = jnp.broadcast_to(jnp.mean(wwu, axis=0), u.shape)
        divisor = n
    else:
        qu = u - jnp.mean(u, axis=0)
        qw = wu - jnp.mean(wu, axis=0)
        qww = wwu - jnp.mean(wwu, axis=0)
        divisor = n * (t - 1)
    g = jnp.stack([jnp.sum(u * qu), jnp.sum(wu * qw), jnp.sum(u * qw)]) / divisor
    linear = (
        jnp.stack(
            [
                2 * jnp.sum(u * qw),
                2 * jnp.sum(wwu * qw),
                jnp.sum(u * qww) + jnp.sum(wu * qw),
            ]
        )
        / divisor
    )
    quadratic = (
        -jnp.stack([jnp.sum(wu * qw), jnp.sum(wwu * qww), jnp.sum(wu * qww)]) / divisor
    )
    variance = jnp.stack(
        [
            jnp.asarray(1.0, dtype=u.dtype),
            jnp.sum(w * w) / n,
            jnp.asarray(0.0, dtype=u.dtype),
        ]
    )
    return g, jnp.column_stack([linear, quadratic, variance])


def _variance_profile(
    lam: Array, g: Array, system: Array, inverse: Array
) -> tuple[Array, Array]:
    """Profile nonnegative scale components in a small quadratic criterion."""
    target = g - system[:, :2] @ jnp.stack([lam, lam * lam])
    columns = system[:, 2:]
    gram = columns.T @ inverse @ columns
    rhs = columns.T @ inverse @ target
    if columns.shape[1] == 1:
        components = jnp.maximum(rhs / gram[0, 0], 0)
    else:
        unconstrained = jnp.linalg.solve(gram, rhs)
        candidates = jnp.stack(
            [
                unconstrained,
                jnp.array([jnp.maximum(rhs[0] / gram[0, 0], 0), 0]),
                jnp.array([0, jnp.maximum(rhs[1] / gram[1, 1], 0)]),
                jnp.zeros(2, dtype=g.dtype),
            ]
        )
        residuals = target[None, :] - candidates @ columns.T
        values = jnp.einsum("bi,ij,bj->b", residuals, inverse, residuals)
        values = jnp.where(jnp.all(candidates >= 0, axis=1), values, jnp.inf)
        components = candidates[jnp.argmin(values)]
    residual = target - columns @ components
    return residual @ inverse @ residual, components


def _kkp_panel_fit(  # named scientific fit/covariance state  # noqa: PLR0914
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialPanelWeights,
    *,
    full_weights: bool = False,
    parameter_names: tuple[str, ...] | None = None,
    tolerance: float = 1e-7,
) -> KKPPanelFit:
    """Fit the balanced constant-W KKP estimator using within/between moments.

    Positive component estimates and a nonnegative implied entity variance are
    required. Time-varying W is rejected because the KKP Q projections cannot be
    commuted with a changing graph. The beta solve uses explicit spatial then
    temporal filtering without constructing an NT-square covariance.
    """
    if weights.values.ndim != _DESIGN_DIMENSIONS:
        message = "KKP requires fixed W; use PanelML for time-varying W_t."
        raise ValueError(message)
    x, y = jnp.asarray(design), jnp.asarray(outcome)
    dtype = jnp.result_type(x, y, weights.values, 1.0)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(dtype)
    if not jnp.issubdtype(y.dtype, jnp.floating):
        y = y.astype(dtype)
    if (
        x.ndim != _PANEL_DIMENSIONS
        or y.shape != x.shape[:2]
        or y.shape != (weights.period_ids.shape[0], weights.unit_ids.shape[0])
        or x.shape[0] < _MIN_PERIODS
    ):
        message = (
            "KKP requires a complete balanced time-major panel with at "
            "least two periods."
        )
        raise ValueError(message)
    t, n, p = x.shape
    w = weights.values
    initial = LeastSquares().estimate(LinearProblem(x.reshape(t * n, p), y.reshape(-1)))
    u = initial.residuals.reshape(t, n)
    g0, system0 = _moments(u, w, between=False)
    limit = jnp.asarray(0.99, dtype=dtype) * weights.limit
    first = bounded_scalar_minimise(
        lambda lam: _variance_profile(lam, g0, system0, jnp.eye(3, dtype=dtype))[0],
        jnp.stack([-limit, limit]),
        tolerance=tolerance,
    )
    _, v0 = _variance_profile(first.value[0], g0, system0, jnp.eye(3, dtype=dtype))
    filtered = u - first.value[0] * spatial_panel_lag(w, u)
    v1 = t * jnp.sum(jnp.mean(filtered, axis=0) ** 2) / n
    g1, system1 = _moments(u, w, between=True)
    system = jnp.concatenate(
        [
            jnp.column_stack([system0, jnp.zeros(3, dtype=dtype)]),
            jnp.column_stack(
                [system1[:, :2], jnp.zeros(3, dtype=dtype), system1[:, 2]]
            ),
        ],
        axis=0,
    )
    g = jnp.concatenate([g0, g1])
    tau = jnp.eye(3, dtype=dtype)
    if full_weights:
        tau = _full_weight_tau(w)
    variability = jnp.kron(jnp.diag(jnp.stack([v0[0] ** 2 / (t - 1), v1**2])), tau)
    inverse = jnp.linalg.inv(variability)
    solver = bounded_scalar_minimise(
        lambda lam: _variance_profile(lam, g, system, inverse)[0],
        jnp.stack([-limit, limit]),
        tolerance=tolerance,
    )
    _, components = _variance_profile(solver.value[0], g, system, inverse)
    theta = 1 - jnp.sqrt(components[0] / components[1])
    sx = x - solver.value[0] * spatial_panel_lag(w, x)
    sy = y - solver.value[0] * spatial_panel_lag(w, y)
    sx = sx - theta * jnp.mean(sx, axis=0)
    sy = sy - theta * jnp.mean(sy, axis=0)
    linear = LeastSquares().estimate(
        LinearProblem(
            sx.reshape(t * n, p), sy.reshape(-1), parameter_names=parameter_names
        )
    )
    valid = (
        initial.valid
        & linear.valid
        & first.converged
        & solver.converged
        & jnp.all(components > 0)
        & (components[1] >= components[0])
        & (jnp.abs(solver.value[0]) < limit - 2 * tolerance)
        & jnp.all(jnp.diag(w) == 0)
    )
    return KKPPanelFit(
        linear,
        x,
        y,
        weights,
        solver.value[0],
        components,
        theta,
        jnp.where(valid, y - jnp.einsum("tip,p->ti", x, linear.parameters), jnp.nan),
        solver,
        valid,
    )
