# jaxtyping numerical carrier contracts.
"""Exact dense static spatial panels with exogenous fixed or time-varying W_t.

Model: A_t y_t = X_t beta + alpha + B_t^-1 epsilon_t, where A_t=I-rho W_t
and B_t=I-lambda W_t. Alpha is zero, a fixed entity effect, or an independent
Gaussian random entity effect. Epsilon_t are independent spherical Gaussian
innovations. W_t and the full balanced unit roster are conditioned on.

Filtering precedes effect profiling. In particular, W_t is never commuted with
within transformation, and log-Jacobians are summed over actual period graphs.
Random effects use a Woodbury reduction with sum_t B_t'B_t; no NT-square
covariance or constant-W eigenvector assumption is used.
"""

from __future__ import annotations

from collections.abc import (
    Callable,  # runtime annotation hook  # noqa: TC003
)
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from statadic.core.results import (
    SolverResult,
    SolverStatus,
)
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
)
from statadic.estimation._validation import validate_parameter_names
from statadic.estimation.spatial._design import graph_parameter_names
from statadic.estimation.spatial.likelihood import (
    SAC,
    SAR,
    SEM,
    _fixed_graph_information,
    _information_rank,
)
from statadic.operators.spatial import (  # noqa: TC001
    SpatialPanelWeights,
    spatial_panel_lag,
)
from statadic.optimize.backends import (
    _newton_root_result,
    bounded_scalar_minimise,
    minimise_bfgs_result,
)

__all__ = [
    "FixedGraphCache",
    "PanelML",
    "SpatialPanelFit",
    "SpatialPanelProblem",
    "spatial_panel_profile",
]


_PANEL_DIMENSIONS = 3
_FIXED_GRAPH_DIMENSIONS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialPanelProblem:
    """Aligned complete time-major design, outcome, and spatial graph."""

    design: ArrayLike
    outcome: ArrayLike
    weights: SpatialPanelWeights
    process: SAR | SEM | SAC = field(kw_only=True, metadata={"static": True})
    effects: Literal["pooled", "unit", "random"] = field(
        kw_only=True, metadata={"static": True}
    )
    slx_columns: tuple[int, ...] = field(
        default=(), kw_only=True, metadata={"static": True}
    )
    graph_name: str | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )
    parameter_names: tuple[str, ...] | None = field(
        default=None, kw_only=True, metadata={"static": True}
    )


@dataclass(frozen=True)
class PanelML:
    """Exact Gaussian spatial panel likelihood with declared process and effects."""

    max_steps: int = 256
    tolerance: float = 1e-7

    def estimate(self, problem: SpatialPanelProblem) -> SpatialPanelFit:
        """Estimate the retained likelihood without altering its numerical contract."""
        if not isinstance(problem.process, (SAR, SEM, SAC)):
            message = "Spatial panel ML requires a SAR, SEM or SAC process."
            raise TypeError(message)
        spatial = (
            ("rho",)
            if isinstance(problem.process, SAR)
            else ("lambda",)
            if isinstance(problem.process, SEM)
            else ("rho", "lambda")
        )
        graph_parameter_names(
            None,
            problem.parameter_names,
            problem.slx_columns,
            problem.graph_name,
            spatial,
        )
        fit = _spatial_panel_ml_fit(
            problem.design,
            problem.outcome,
            problem.weights,
            model=problem.process.model,
            effects=problem.effects,
            slx_columns=problem.slx_columns,
            parameter_names=problem.parameter_names,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
        )
        return replace(
            fit,
            parameter_names=graph_parameter_names(
                fit.parameter_names,
                problem.parameter_names,
                problem.slx_columns,
                problem.graph_name,
                spatial,
            ),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialPanelFit:
    """Retained balanced panel likelihood, effects and complete fit diagnostics.

    Parameters contain design coefficients followed by active rho/lambda.
    Random-effect variance ratio is retained separately, never confused with a
    regression coefficient. Expected fixed/pooled information eliminates entity
    effects; random-effect information is the observed marginal Hessian and
    includes variance-ratio and log innovation-scale nuisances last.
    For pooled SAC, solver residual is the infinity norm of the gradient of
    average negative profile likelihood in normalized spatial coordinates
    (spatial parameters divided by their search limit).
    """

    parameters: Array
    design: Array
    outcome: Array
    weights: SpatialPanelWeights
    effects: Array
    residuals: Array
    innovations: Array
    sigma2: Array
    variance_ratio: Array
    variance_boundary: Array
    log_likelihood: Array
    information: Array
    solver: SolverResult[Array]
    identified: Array
    boundary: Array
    valid: Array
    lag: bool = field(metadata={"static": True})
    error: bool = field(metadata={"static": True})
    effect_type: str = field(metadata={"static": True})
    information_method: str = field(metadata={"static": True})
    slx_columns: tuple[int, ...] = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})


def _filters(
    weights: SpatialPanelWeights,
    spatial: Array,
    *,
    lag: bool,
    error: bool,
) -> tuple[Array, Array, Array]:
    """Build each period's two filters and exact summed Jacobian."""
    n = weights.values.shape[-1]
    t = weights.period_ids.shape[0]
    w = jnp.broadcast_to(weights.values, (t, n, n))
    identity = jnp.eye(n, dtype=w.dtype)
    rho = spatial[0] if lag else jnp.asarray(0, dtype=w.dtype)
    lam = spatial[-1] if error else jnp.asarray(0, dtype=w.dtype)
    a = identity - rho * w
    b = identity - lam * w
    logdet = jnp.sum(jnp.linalg.slogdet(a)[1] + jnp.linalg.slogdet(b)[1])
    return a, b, logdet


type FixedGraphCache = tuple[Array, Array, Array, Array]


def _fixed_graph_cache(w: Array, design: Array, outcome: Array) -> FixedGraphCache:
    """Precompute every rho/lambda-free product of one fixed W with the data.

    Returns ``W y``, ``W^2 y``, ``W X`` and W's complex eigenvalues. Each filter
    ``(I - lam W)(I - rho W) y`` is then a linear combination of these, so an
    optimiser evaluation needs no matrix product with W.
    """
    wy = outcome @ w.T
    return (
        wy,
        wy @ w.T,
        jnp.einsum("ij,tjk->tik", w, design),
        jnp.linalg.eigvals(w),
    )


def _pooled_fixed_graph_filters(
    cache: FixedGraphCache,
    spatial: Array,
    design: Array,
    outcome: Array,
    *,
    lag: bool,
    error: bool,
) -> tuple[Array, Array, Array]:
    """Filter a pooled panel with one fixed W, without a (T,N,N) broadcast.

    Algebraically identical to the per-period path: the log-Jacobian is
    ``T * sum_k log|1 - rho * eigenvalue_k|`` (complex eigenvalues keep their
    imaginary part) and every filter is a combination of ``cache``. The branch is
    decided by the static graph rank, not by data.
    """
    wy, w2y, wx, eigenvalues = cache
    t = design.shape[0]
    rho = spatial[0] if lag else jnp.asarray(0, dtype=wy.dtype)
    lam = spatial[-1] if error else jnp.asarray(0, dtype=wy.dtype)
    by = outcome - rho * wy - lam * (wy - rho * w2y)
    bx = design - lam * wx
    logdet = t * (
        jnp.sum(jnp.log(jnp.abs(1 - rho * eigenvalues)))
        + jnp.sum(jnp.log(jnp.abs(1 - lam * eigenvalues)))
    )
    return by, bx, logdet


def _random_profile(
    by: Array, bx: Array, b: Array, ratio: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Use Woodbury GLS without differentiating degenerate eigenspaces."""
    n = by.shape[1]
    qaa = jnp.einsum("tji,tjk->ik", b, b)
    matrix = jnp.eye(n, dtype=by.dtype) + ratio * qaa
    data = jnp.concatenate([by[:, :, None], bx], axis=2)
    sums = jnp.einsum("tji,tjk->ik", b, data)
    gram = jnp.einsum("tip,tiq->pq", data, data) - ratio * sums.T @ jnp.linalg.solve(
        matrix, sums
    )
    beta = jnp.linalg.solve(gram[1:, 1:], gram[1:, 0])
    residual = by - jnp.einsum("tip,p->ti", bx, beta)
    score = jnp.linalg.solve(matrix, jnp.einsum("tji,tj->i", b, residual))
    alpha = ratio * score
    innovations = residual - jnp.einsum("tij,j->ti", b, alpha)
    sse = jnp.sum(innovations**2) + ratio * jnp.sum(score**2)
    return beta, alpha, sse, jnp.linalg.slogdet(matrix)[1] / 2, innovations


def _fixed_profile(  # noqa: PLR0914
    by: Array, bx: Array, b: Array
) -> tuple[Array, Array, Array, Array]:  # named scientific fit/covariance state
    """Project filtered entity effects before the stable QR coefficient solve."""
    t, n, p = bx.shape
    qaa = jnp.einsum("tji,tjk->ik", b, b)
    qax = jnp.einsum("tji,tjk->ik", b, bx)
    qay = jnp.einsum("tji,tj->i", b, by)
    # Project onto the orthogonal complement of the filtered effect design.
    # Qaa is only N-by-N, never NT-by-NT.
    effect_x = jnp.linalg.solve(qaa, qax)
    effect_y = jnp.linalg.solve(qaa, qay)
    residual_x = bx - jnp.einsum("tij,jk->tik", b, effect_x)
    residual_y = by - jnp.einsum("tij,j->ti", b, effect_y)
    q, r = jnp.linalg.qr(residual_x.reshape(t * n, p), mode="reduced")
    beta = solve_triangular(r, q.T @ residual_y.reshape(-1), lower=False)
    alpha = effect_y - effect_x @ beta
    innovations = (
        by - jnp.einsum("tij,j->ti", b, alpha) - jnp.einsum("tip,p->ti", bx, beta)
    )
    sse = jnp.sum(innovations**2)
    return beta, alpha, sse, innovations


def spatial_panel_profile(  # named scientific fit/covariance state  # noqa: PLR0914
    spatial: Array,
    design: Array,
    outcome: Array,
    weights: SpatialPanelWeights,
    *,
    lag: bool = True,
    error: bool = False,
    effects: Literal["pooled", "unit", "random"] = "unit",
    variance_ratio: Array | None = None,
    fixed_graph_cache: FixedGraphCache | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return beta, entity effects, scale, log-likelihood and filtered residuals.

    This public objective-state door permits independent grid and derivative
    checks without certifying the fitted-solution derivative. The random-effect
    ratio tau=Var(alpha)/Var(epsilon) is nonnegative and supplied explicitly.
    """
    t, n, p = design.shape
    if effects == "pooled" and weights.values.ndim == _FIXED_GRAPH_DIMENSIONS:
        cache = (
            _fixed_graph_cache(weights.values, design, outcome)
            if fixed_graph_cache is None
            else fixed_graph_cache
        )
        by, bx, logdet = _pooled_fixed_graph_filters(
            cache, spatial, design, outcome, lag=lag, error=error
        )
        b = jnp.zeros((0, n, n), dtype=design.dtype)  # unused by pooled effects
    else:
        a, b, logdet = _filters(weights, spatial, lag=lag, error=error)
        ay = jnp.einsum("tij,tj->ti", a, outcome)
        by = jnp.einsum("tij,tj->ti", b, ay)
        bx = jnp.einsum("tij,tjk->tik", b, design)
    if effects == "unit":
        beta, alpha, sse, innovations = _fixed_profile(by, bx, b)
    elif effects == "random":
        if variance_ratio is None:
            message = "Random effects require a nonnegative variance_ratio."
            raise ValueError(message)
        beta, alpha, sse, adjustment, innovations = _random_profile(
            by, bx, b, variance_ratio
        )
        logdet -= adjustment
    elif effects == "pooled":
        flat = bx.reshape(t * n, p)
        q, r = jnp.linalg.qr(flat, mode="reduced")
        beta = solve_triangular(r, q.T @ by.reshape(-1), lower=False)
        alpha = jnp.zeros(n, dtype=design.dtype)
        innovations = by - jnp.einsum("tip,p->ti", bx, beta)
        sse = jnp.sum(innovations**2)
    else:
        message = "effects must be 'pooled', 'unit' or 'random'."
        raise ValueError(message)
    sigma2 = sse / (t * n)
    log_likelihood = logdet - (t * n) / 2 * (jnp.log(2 * jnp.pi * sigma2) + 1)
    admissible = jnp.all(jnp.abs(spatial) < weights.limit)
    if effects == "random" and variance_ratio is not None:
        admissible &= variance_ratio >= 0
    return (
        jnp.where(admissible, beta, jnp.nan),
        jnp.where(admissible, alpha, jnp.nan),
        jnp.where(admissible, sigma2, jnp.nan),
        jnp.where(admissible, log_likelihood, jnp.nan),
        jnp.where(admissible, innovations, jnp.nan),
    )


def _fixed_information(
    design: Array,
    weights: SpatialPanelWeights,
    beta: Array,
    alpha: Array,
    spatial: Array,
    sigma2: Array,
    *,
    lag: bool,
    error: bool,
    fixed: bool,
) -> Array:
    """Sum conditional expected information, then eliminate fixed effects."""
    if weights.values.ndim == _FIXED_GRAPH_DIMENSIONS:
        # With a common B, the whitened entity-effect design spans exactly
        # the time-constant vectors. Its Schur complement demeans the mean
        # derivatives; the covariance/scale information is unchanged.
        if fixed:
            design = design - jnp.mean(design, axis=0)
        return _fixed_graph_information(
            design, weights.values, beta, spatial, sigma2, lag=lag, error=error
        )
    t, n, p = design.shape
    width = p + (n if fixed else 0) + spatial.shape[0] + 1
    dtype = jnp.result_type(design, weights.values, beta, alpha, spatial, sigma2)
    identity = jnp.eye(n, dtype=dtype)
    w = jnp.broadcast_to(weights.values, (t, n, n))

    def add(period: Array, information: Array) -> Array:
        x = (
            jnp.concatenate([design[period], identity], axis=1)
            if fixed
            else design[period]
        )
        coefficient = jnp.concatenate([beta, alpha]) if fixed else beta
        return information + _fixed_graph_information(
            x[None], w[period], coefficient, spatial, sigma2, lag=lag, error=error
        )

    joint = jax.lax.fori_loop(0, t, add, jnp.zeros((width, width), dtype=dtype))
    if not fixed:
        return joint
    keep = jnp.concatenate([jnp.arange(p), jnp.arange(p + n, width)])
    nuisance = jnp.arange(p, p + n)
    cross = joint[jnp.ix_(keep, nuisance)]
    return joint[jnp.ix_(keep, keep)] - cross @ jnp.linalg.solve(
        joint[jnp.ix_(nuisance, nuisance)], cross.T
    )


def _random_information(
    design: Array,
    outcome: Array,
    weights: SpatialPanelWeights,
    parameters: Array,
    spatial: Array,
    ratio: Array,
    sigma2: Array,
    *,
    lag: bool,
    error: bool,
) -> Array:
    """Differentiate the full marginal likelihood including scale nuisances."""
    p = design.shape[-1]
    q = spatial.shape[0]
    n = outcome.size

    def objective(state: Array) -> Array:
        beta = state[:p]
        s = state[p : p + q]
        tau = state[-2]
        scale = jnp.exp(state[-1])
        a, b, logdet = _filters(weights, s, lag=lag, error=error)
        residual = jnp.einsum(
            "tij,tj->ti",
            b,
            jnp.einsum("tij,tj->ti", a, outcome)
            - jnp.einsum("tip,p->ti", design, beta),
        )
        qaa = jnp.einsum("tji,tjk->ik", b, b)
        total = jnp.einsum("tji,tj->i", b, residual)
        matrix = jnp.eye(qaa.shape[0], dtype=qaa.dtype) + tau * qaa
        sse = jnp.sum(residual**2) - tau * total @ jnp.linalg.solve(matrix, total)
        return (
            0.5
            * (
                n * jnp.log(2 * jnp.pi * scale)
                + sse / scale
                + jnp.linalg.slogdet(matrix)[1]
            )
            - logdet
        )

    state = jnp.concatenate([parameters, jnp.stack([ratio, jnp.log(sigma2)])])
    return jax.hessian(objective)(state)


def _panel_score_tolerance(dtype: jnp.dtype, tolerance: float) -> Array:
    """Allow rounding in the per-observation, normalized-coordinate score."""
    # This floor depends on precision, not the likelihood's arbitrary offset.
    return jnp.maximum(tolerance, 16 * jnp.finfo(dtype).eps)


def _recover_stalled_panel_search(
    objective: Callable[[Array], Array],
    average_objective: Callable[[Array], Array],
    selected: SolverResult[Array],
    limit: Array,
    *,
    dimension: int,
    max_steps: int,
    tolerance: float,
) -> SolverResult[Array]:
    """Try one fresh BFGS metric using only an available half-budget allowance.

    The first solve retains its entire original budget. Recovery is limited to
    a finite interior stalled iterate with enough unused steps for this static
    allowance, preserving JIT and the original total iteration cap. Success
    still requires the independently evaluated average profile score.
    """
    allowance = max_steps // 2
    if allowance == 0:
        return selected

    def interior(value: Array) -> Array:
        return jnp.all(
            jnp.abs(limit * jnp.tanh(value[:dimension])) < limit - 2 * tolerance
        )

    def score_coordinates(value: Array) -> Array:
        # Bounded coordinates enter the score through tanh; a log variance
        # ratio is already unconstrained.
        return jnp.concatenate([jnp.tanh(value[:dimension]), value[dimension:]])

    def recover(previous: SolverResult[Array]) -> SolverResult[Array]:
        candidate = minimise_bfgs_result(
            objective,
            previous.value,
            max_steps=allowance,
            rtol=tolerance,
            atol=tolerance,
        )
        score = jnp.max(
            jnp.abs(jax.grad(average_objective)(score_coordinates(candidate.value)))
        )
        value = objective(candidate.value)
        improve = (
            jnp.all(jnp.isfinite(candidate.value))
            & jnp.isfinite(value)
            & jnp.isfinite(score)
            & interior(candidate.value)
            & (value <= objective(previous.value))
        )
        converged = (
            improve
            & candidate.converged
            & (score <= _panel_score_tolerance(candidate.value.dtype, tolerance))
        )
        candidate_status = jnp.where(
            candidate.converged & ~converged, SolverStatus.STALLED, candidate.status
        )
        return SolverResult(
            jnp.where(improve, candidate.value, previous.value),
            converged,
            previous.iterations + candidate.iterations,
            jnp.where(improve, candidate_status, previous.status),
            jnp.where(improve, score, previous.residual_norm),
            jnp.where(improve, candidate.rank, previous.rank),
            previous.max_steps,
        )

    retry = (
        ~selected.converged
        & (selected.status == SolverStatus.STALLED)
        & (selected.iterations <= max_steps - allowance)
        & jnp.all(jnp.isfinite(selected.value))
        & jnp.isfinite(objective(selected.value))
        & jnp.isfinite(selected.residual_norm)
        & interior(selected.value)
    )

    def retain(previous: SolverResult[Array]) -> SolverResult[Array]:
        return previous

    return jax.lax.cond(retry, recover, retain, selected)


def _recover_panel_score_root(
    selected: SolverResult[Array],
    objective: Callable[[Array], Array],
    average_objective: Callable[[Array], Array],
    limit: Array,
    *,
    dimension: int,
    max_steps: int,
    tolerance: float,
) -> SolverResult[Array]:
    """Polish an interior random-effect score after likelihood rounding stalls.

    Newton solves the normalized score rather than comparing nearly identical
    likelihood values. A candidate still needs positive full-rank curvature,
    the original score threshold, and at most one ULP of objective rounding.
    That allowance concerns the computed likelihood, not a proof of decrease.
    All certificates are evaluated at the encoded point returned to the caller.
    """
    allowance = 8
    if max_steps < allowance:
        return selected

    score_function = jax.grad(average_objective)
    hessian_function = jax.hessian(average_objective)

    def coordinates(value: Array) -> Array:
        return jnp.concatenate([jnp.tanh(value[:dimension]), value[dimension:]])

    def admissible(value: Array) -> Array:
        ratio = jnp.exp(value[-1])
        return (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.abs(limit * value[:dimension]) < limit - 2 * tolerance)
            & jnp.isfinite(ratio)
            & (ratio > 0)
        )

    def positive_curvature(value: Array) -> Array:
        hessian = hessian_function(value)
        symmetric = (hessian + hessian.T) / 2
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        cutoff = jnp.finfo(value.dtype).eps * value.size * jnp.max(jnp.abs(eigenvalues))
        symmetry_tolerance = (
            8 * jnp.finfo(value.dtype).eps * value.size * jnp.max(jnp.abs(hessian))
        )
        return (
            jnp.all(jnp.isfinite(hessian))
            & (jnp.max(jnp.abs(hessian - hessian.T)) <= symmetry_tolerance)
            & jnp.all(jnp.isfinite(eigenvalues))
            & jnp.all(eigenvalues > cutoff)
        )

    def recover(previous: SolverResult[Array]) -> SolverResult[Array]:
        initial = coordinates(previous.value)
        candidate = _newton_root_result(
            score_function,
            initial,
            max_steps=allowance,
            rtol=tolerance,
            atol=_panel_score_tolerance(initial.dtype, tolerance),
        )
        encoded = jnp.concatenate(
            [jnp.arctanh(candidate.value[:dimension]), candidate.value[dimension:]]
        )
        returned = coordinates(encoded)
        score = jnp.max(jnp.abs(score_function(returned)))
        old_score = jnp.max(jnp.abs(score_function(initial)))
        value = objective(encoded)
        ceiling = jnp.nextafter(objective(previous.value), jnp.inf)
        accepted = (
            candidate.converged
            & (candidate.status == SolverStatus.SUCCESS)
            & admissible(candidate.value)
            & admissible(returned)
            & jnp.all(jnp.isfinite(encoded))
            & positive_curvature(returned)
            & jnp.isfinite(value)
            & jnp.isfinite(ceiling)
            & (value <= ceiling)
            & jnp.isfinite(score)
            & (score <= _panel_score_tolerance(returned.dtype, tolerance))
            & (score <= old_score)
        )
        return SolverResult(
            jnp.where(accepted, encoded, previous.value),
            accepted | previous.converged,
            previous.iterations
            + jnp.asarray(candidate.iterations, dtype=previous.iterations.dtype),
            jnp.where(accepted, SolverStatus.SUCCESS, previous.status),
            jnp.where(accepted, score, previous.residual_norm),
            jnp.where(accepted, returned.size, previous.rank),
            previous.max_steps,
        )

    def retain(previous: SolverResult[Array]) -> SolverResult[Array]:
        return previous

    def qualify(previous: SolverResult[Array]) -> SolverResult[Array]:
        initial = coordinates(previous.value)
        qualified = (
            jnp.isfinite(objective(previous.value))
            & jnp.all(jnp.isfinite(score_function(initial)))
            & positive_curvature(initial)
        )
        return jax.lax.cond(qualified, recover, retain, previous)

    retry = (
        ~selected.converged
        & (selected.status == SolverStatus.STALLED)
        & (selected.iterations <= max_steps - allowance)
        & admissible(coordinates(selected.value))
        & jnp.isfinite(selected.residual_norm)
    )
    return jax.lax.cond(retry, qualify, retain, selected)


def _multistart_panel_search(
    profile: Callable[[Array, Array | None], tuple[Array, Array, Array, Array, Array]],
    dimension: int,
    limit: Array,
    *,
    random_effects: bool,
    max_steps: int,
    tolerance: float,
    sequential_starts: bool = False,
    guard_unconverged: bool = False,
    score_sample_size: int | None = None,
) -> tuple[SolverResult[Array], Array]:
    """Search bounded spatial coordinates and, optionally, a positive variance ratio."""
    seed = jnp.array([-0.5, 0.0, 0.5], dtype=limit.dtype)
    starts = (
        seed[:, None]
        if dimension == 1
        else jnp.stack(jnp.meshgrid(seed, seed), axis=-1).reshape(-1, dimension)
    )
    if random_effects:
        starts = jnp.concatenate(
            [
                jnp.repeat(starts, 2, axis=0),
                jnp.tile(
                    jnp.array([[-2.0], [0.0]], dtype=limit.dtype), (starts.shape[0], 1)
                ),
            ],
            axis=1,
        )

    def objective(z: Array) -> Array:
        ratio = jnp.exp(z[-1]) if random_effects else None
        return -profile(limit * jnp.tanh(z[:dimension]), ratio)[3]

    def solve(initial: Array) -> SolverResult[Array]:
        return minimise_bfgs_result(
            objective, initial, max_steps=max_steps, rtol=tolerance, atol=tolerance
        )

    solutions = (
        jax.lax.map(solve, starts) if sequential_starts else jax.vmap(solve)(starts)
    )
    values = (
        jax.lax.map(objective, solutions.value)
        if sequential_starts
        else jax.vmap(objective)(solutions.value)
    )

    def select_solution(index: Array) -> SolverResult[Array]:
        def select_leaf(value: Array) -> Array:
            return value[index]

        return jax.tree_util.tree_map(select_leaf, solutions)

    if score_sample_size is not None:

        def average_objective(u: Array) -> Array:
            ratio = jnp.exp(u[-1]) if random_effects else None
            return -profile(limit * u[:dimension], ratio)[3] / score_sample_size

        def score_coordinates(value: Array) -> Array:
            return jnp.concatenate([jnp.tanh(value[:dimension]), value[dimension:]])

        scores = jax.vmap(jax.grad(average_objective))(
            jax.vmap(score_coordinates)(solutions.value)
        )
        residuals = jnp.max(jnp.abs(scores), axis=1)
        finite = (
            jnp.isfinite(values)
            & jnp.isfinite(residuals)
            & jnp.all(jnp.isfinite(solutions.value), axis=1)
        )
        eligible = (
            finite
            & solutions.converged
            & (residuals <= _panel_score_tolerance(solutions.value.dtype, tolerance))
        )
        any_eligible = jnp.any(eligible)
        candidates = jnp.where(
            any_eligible,
            eligible,
            jnp.isfinite(values) & jnp.all(jnp.isfinite(solutions.value), axis=1),
        )
        best = jnp.argmin(jnp.where(candidates, values, jnp.inf))
        selected = select_solution(best)
        failed_status = jnp.where(
            selected.converged, SolverStatus.STALLED, selected.status
        )
        selected = SolverResult(
            selected.value,
            any_eligible,
            selected.iterations,
            jnp.where(
                any_eligible,
                SolverStatus.SUCCESS,
                jnp.where(
                    ~jnp.any(candidates),
                    SolverStatus.NONFINITE,
                    failed_status,
                ),
            ),
            residuals[best],
            selected.rank,
            selected.max_steps,
        )
        selected = _recover_stalled_panel_search(
            objective,
            average_objective,
            selected,
            limit,
            dimension=dimension,
            max_steps=max_steps,
            tolerance=tolerance,
        )
        if random_effects:
            selected = _recover_panel_score_root(
                selected,
                objective,
                average_objective,
                limit,
                dimension=dimension,
                max_steps=max_steps,
                tolerance=tolerance,
            )
    else:
        eligible = jnp.isfinite(values) if guard_unconverged else solutions.converged
        best = jnp.argmin(jnp.where(eligible, values, jnp.inf))
        selected = select_solution(best)
    spatial = limit * jnp.tanh(selected.value[:dimension])
    ratio = (
        jnp.exp(selected.value[-1])
        if random_effects
        else jnp.asarray(0, dtype=limit.dtype)
    )
    return SolverResult(
        spatial,
        selected.converged,
        selected.iterations,
        selected.status,
        selected.residual_norm,
        selected.rank,
        selected.max_steps,
    ), ratio


def _panel_search(
    profile: Callable[[Array, Array | None], tuple[Array, Array, Array, Array, Array]],
    dimension: int,
    limit: Array,
    *,
    effects: str,
    max_steps: int,
    tolerance: float,
    sequential_grid: bool = False,
    guard_unconverged: bool = False,
    comparison_tolerance: float | None = None,
    score_sample_size: int | None = None,
) -> tuple[SolverResult[Array], Array]:
    """Compare positive random-effect fits with their exact zero-variance boundary."""
    zero = jnp.asarray(0, dtype=limit.dtype)
    comparison_tolerance = (
        tolerance if comparison_tolerance is None else comparison_tolerance
    )
    if dimension == 1 and effects != "random":
        solver = bounded_scalar_minimise(
            lambda s: -profile(s[None], None)[3],
            jnp.stack([-limit, limit]),
            max_steps=max_steps,
            tolerance=tolerance,
            grid_strategy="sequential" if sequential_grid else "vectorized",
        )
        return solver, zero
    solver, ratio = _multistart_panel_search(
        profile,
        dimension,
        limit,
        random_effects=effects == "random",
        max_steps=max_steps,
        tolerance=tolerance,
        sequential_starts=sequential_grid,
        guard_unconverged=guard_unconverged,
        score_sample_size=score_sample_size,
    )
    if effects != "random":
        return solver, ratio
    if dimension == 1:
        boundary = bounded_scalar_minimise(
            lambda s: -profile(s[None], zero)[3],
            jnp.stack([-limit, limit]),
            max_steps=max_steps,
            tolerance=tolerance,
            grid_strategy="sequential" if sequential_grid else "vectorized",
        )
    else:
        boundary, _ = _multistart_panel_search(
            lambda spatial, _: profile(spatial, zero),
            dimension,
            limit,
            random_effects=False,
            max_steps=max_steps,
            tolerance=tolerance,
            sequential_starts=sequential_grid,
            guard_unconverged=guard_unconverged,
        )
    choose = boundary.converged & (
        profile(boundary.value, zero)[3]
        >= profile(solver.value, ratio)[3] - comparison_tolerance
    )

    def pick(a: Array, b: Array) -> Array:
        return jnp.where(choose, a, b)

    return jax.tree_util.tree_map(pick, boundary, solver), jnp.where(
        choose, zero, ratio
    )


@partial(
    jax.jit,
    static_argnames=("lag", "error", "effects", "max_steps", "tolerance"),
)
def _search_panel_likelihood(
    x: Array,
    y: Array,
    weights: SpatialPanelWeights,
    limit: Array,
    cache: FixedGraphCache | None,
    *,
    lag: bool,
    error: bool,
    effects: Literal["pooled", "unit", "random"],
    max_steps: int,
    tolerance: float,
) -> tuple[SolverResult[Array], Array]:
    """Reuse the search executable while sample and graph values remain dynamic."""
    profile_effects = effects
    if effects == "unit" and weights.values.ndim == _FIXED_GRAPH_DIMENSIONS:
        # Only a time-invariant filter commutes with time demeaning. The
        # concentrated likelihood then has the pooled form on centered data,
        # with the original NT scale divisor and T log-Jacobian contributions.
        x = x - jnp.mean(x, axis=0)
        y = y - jnp.mean(y, axis=0)
        cache = _fixed_graph_cache(weights.values, x, y)
        profile_effects = "pooled"

    def profile(
        spatial: Array, ratio: Array | None
    ) -> tuple[Array, Array, Array, Array, Array]:
        return spatial_panel_profile(
            spatial,
            x,
            y,
            weights,
            lag=lag,
            error=error,
            effects=profile_effects,
            variance_ratio=ratio,
            fixed_graph_cache=cache,
        )

    return _panel_search(
        profile,
        int(lag) + int(error),
        limit,
        effects=effects,
        max_steps=max_steps,
        tolerance=tolerance,
        score_sample_size=(
            y.size
            if effects == "random" or (effects == "pooled" and lag and error)
            else None
        ),
    )


@partial(
    jax.jit,
    static_argnames=(
        "model",
        "effects",
        "slx_columns",
        "parameter_names",
        "max_steps",
        "tolerance",
    ),
)
def _spatial_panel_ml_fit(  # named scientific fit/covariance state  # noqa: C901, PLR0914
    design: ArrayLike,
    outcome: ArrayLike,
    weights: SpatialPanelWeights,
    *,
    model: Literal["sar", "sem", "sac"] = "sar",
    effects: Literal["pooled", "unit", "random"] = "unit",
    slx_columns: tuple[int, ...] = (),
    parameter_names: tuple[str, ...] | None = None,
    max_steps: int = 256,
    tolerance: float = 1e-7,
) -> SpatialPanelFit:
    """Fit a balanced static panel, retaining every actual W_t in its likelihood.

    Fixed effects require a design without absorbed columns, including a common
    intercept. Random/pooled effects can include an explicit intercept. Missing
    units require a separately specified unbalanced spatial model; zero padding
    is not interpreted as observed data. No fitted-solution derivative is promised.
    """
    x, y = jnp.asarray(design), jnp.asarray(outcome)
    dtype = jnp.result_type(x, y, weights.values, 1.0)
    if not jnp.issubdtype(dtype, jnp.floating):
        message = "Spatial panels require real arrays."
        raise ValueError(message)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(dtype)
    if not jnp.issubdtype(y.dtype, jnp.floating):
        y = y.astype(dtype)
    if (
        x.ndim != _PANEL_DIMENSIONS
        or y.shape != x.shape[:2]
        or y.shape != (weights.period_ids.shape[0], weights.unit_ids.shape[0])
    ):
        message = "Use aligned complete time-major design (T,N,p) and outcome (T,N)."
        raise ValueError(message)
    if model not in ("sar", "sem", "sac") or effects not in (
        "pooled",
        "unit",
        "random",
    ):
        message = "Unknown spatial panel model or effect specification."
        raise ValueError(message)
    validate_parameter_names(parameter_names, x.shape[2])
    if len(set(slx_columns)) != len(slx_columns) or any(
        c < 0 or c >= x.shape[2] for c in slx_columns
    ):
        message = "slx_columns must select unique explicit design columns."
        raise ValueError(message)
    names = parameter_names
    if names is not None:
        names = (*names, *("W:" + names[c] for c in slx_columns))
        names = (
            *names,
            *(("rho",) if model != "sem" else ()),
            *(("lambda",) if model != "sar" else ()),
        )
        validate_parameter_names(
            names, x.shape[2] + len(slx_columns) + (2 if model == "sac" else 1)
        )
    if slx_columns:
        x = jnp.concatenate(
            [x, spatial_panel_lag(weights.values, x[:, :, jnp.asarray(slx_columns)])],
            axis=2,
        )
    lag, error = model != "sem", model != "sar"
    limit = weights.limit * jnp.asarray(0.999, dtype=dtype)
    cache = (
        _fixed_graph_cache(weights.values, x, y)
        if effects == "pooled" and weights.values.ndim == _FIXED_GRAPH_DIMENSIONS
        else None
    )

    def profile(
        s: Array, ratio: Array | None = None
    ) -> tuple[Array, Array, Array, Array, Array]:
        return spatial_panel_profile(
            s,
            x,
            y,
            weights,
            lag=lag,
            error=error,
            effects=effects,
            variance_ratio=ratio,
            fixed_graph_cache=cache,
        )

    solver, ratio = _search_panel_likelihood(
        x,
        y,
        weights,
        limit,
        cache,
        lag=lag,
        error=error,
        effects=effects,
        max_steps=max_steps,
        tolerance=tolerance,
    )
    spatial = solver.value
    beta, alpha, sigma2, log_likelihood, innovations = profile(spatial, ratio)
    parameters = jnp.concatenate([beta, spatial])
    if effects == "random":
        information = _random_information(
            x, y, weights, parameters, spatial, ratio, sigma2, lag=lag, error=error
        )
        information_method = "observed-marginal"
    else:
        information = _fixed_information(
            x,
            weights,
            beta,
            alpha,
            spatial,
            sigma2,
            lag=lag,
            error=error,
            fixed=effects == "unit",
        )
        information_method = "expected-conditional"
    safe_info = jnp.where(jnp.isfinite(information), information, 0)
    identified = _information_rank(safe_info) == information.shape[0]
    slope_design = x - jnp.mean(x, axis=0) if effects == "unit" else x
    safe_design = jnp.where(jnp.isfinite(slope_design), slope_design, 0).reshape(
        -1, x.shape[-1]
    )
    identified &= jnp.linalg.matrix_rank(safe_design) == x.shape[-1]
    boundary = jnp.any(jnp.abs(spatial) >= limit - tolerance * 2)
    valid = (
        identified
        & solver.converged
        & ~boundary
        & jnp.isfinite(log_likelihood)
        & (sigma2 > 0)
        & jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(y))
    )
    rho = spatial[0] if lag else jnp.asarray(0, dtype=dtype)
    residual = (
        y
        - rho * spatial_panel_lag(weights.values, y)
        - jnp.einsum("tip,p->ti", x, beta)
        - alpha[None, :]
    )
    return SpatialPanelFit(
        jnp.where(valid, parameters, jnp.nan),
        x,
        y,
        weights,
        jnp.where(valid, alpha, jnp.nan),
        jnp.where(valid, residual, jnp.nan),
        jnp.where(valid, innovations, jnp.nan),
        jnp.where(valid, sigma2, jnp.nan),
        ratio,
        (ratio <= tolerance) & (effects == "random"),
        jnp.where(valid, log_likelihood, jnp.nan),
        information,
        solver,
        identified,
        boundary,
        valid,
        lag,
        error,
        effects,
        information_method,
        slx_columns,
        names,
    )
