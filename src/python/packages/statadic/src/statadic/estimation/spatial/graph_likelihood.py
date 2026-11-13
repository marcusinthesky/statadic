"""Dense Gaussian likelihood on several named graph roles, by period.

The host aligns graph rosters and builds WX columns. These functions accept
ordinary arrays, including graphs and covariates recomputed inside a JAX trace.
The profile assumes a common innovation variance and independent periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from statadic.core.results import SolverResult, SolverStatus
from statadic.core.typing import Array  # noqa: TC001 -- runtime annotation
from statadic.optimize.backends import minimise_bfgs_result_args

__all__ = [
    "GraphMLFit",
    "estimate_graph_ml",
    "graph_coefficients",
    "graph_diagnostics",
    "graph_filter",
    "graph_log_density",
    "graph_norms",
    "graph_profile",
]


def graph_filter(graphs: Array, coefficients: Array) -> Array:
    """Return I minus the weighted sum of a (roles, nodes, nodes) stack."""
    return jnp.eye(graphs.shape[-1], dtype=graphs.dtype) - jnp.einsum(
        "g,gij->ij", coefficients, graphs
    )


def _graph_mean(
    graphs: Array,
    coefficients: Array,
    design: Array,
    beta: Array,
    *,
    outcome: Array | None = None,
) -> Array:
    """Evaluate a reduced-form mean or a structural mean at supplied outcomes."""
    mean = design @ beta
    if not graphs.shape[0]:
        return mean
    if outcome is not None:
        return mean + jnp.einsum("g,gij,j->i", coefficients, graphs, outcome)
    return jnp.linalg.solve(graph_filter(graphs, coefficients), mean)


class _Whitened(NamedTuple):
    x: Array
    y: Array
    jacobian: Array


def _whiten(
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    spatial: Array,
    observed: tuple[Array, ...] | None,
) -> _Whitened:
    n_lag = lag[0].shape[0]
    if not n_lag and not error[0].shape[0]:
        # WX terms have already entered the design. Without SAR/SEM roles the
        # innovation covariance is identity, including for observed subsets.
        blocks_x = (
            design
            if observed is None
            else tuple(
                x[positions] for x, positions in zip(design, observed, strict=True)
            )
        )
        return _Whitened(
            jnp.concatenate(blocks_x),
            jnp.concatenate(outcome),
            jnp.zeros((), dtype=jnp.result_type(*design, *outcome, spatial)),
        )
    if observed is None and all(x.shape == design[0].shape for x in design):
        a = jax.vmap(graph_filter, in_axes=(0, None))(jnp.stack(lag), spatial[:n_lag])
        b = jax.vmap(graph_filter, in_axes=(0, None))(jnp.stack(error), spatial[n_lag:])
        bx = b @ jnp.stack(design)
        bay = (b @ (a @ jnp.stack(outcome)[..., None])).reshape(-1)
        jac = jnp.sum(jnp.linalg.slogdet(a)[1] + jnp.linalg.slogdet(b)[1])
        return _Whitened(bx.reshape(-1, bx.shape[-1]), bay, jac)
    blocks_x = []
    blocks_y = []
    jacobian = jnp.zeros((), dtype=spatial.dtype)
    for t, (w, v, x, y) in enumerate(zip(lag, error, design, outcome, strict=True)):
        a = graph_filter(w, spatial[:n_lag])
        b = graph_filter(v, spatial[n_lag:])
        if observed is None:
            blocks_x.append(b @ x)
            blocks_y.append(b @ (a @ y))
            jacobian += jnp.linalg.slogdet(a)[1] + jnp.linalg.slogdet(b)[1]
        else:
            positions = observed[t]
            if not positions.shape[0]:
                continue
            # Marginalize unobserved graph nodes under the full joint Gaussian.
            # Select rows of (BA)^-1 by solving against a rectangular RHS.
            select = (positions[:, None] == jnp.arange(x.shape[0])[None, :]).astype(
                x.dtype
            )
            root = jnp.linalg.solve((b @ a).T, select.T).T
            chol = jnp.linalg.cholesky(root @ root.T)
            mean_design = jnp.linalg.solve(a, x)[positions]
            blocks_x.append(solve_triangular(chol, mean_design, lower=True))
            blocks_y.append(solve_triangular(chol, y, lower=True))
            jacobian -= jnp.sum(jnp.log(jnp.diag(chol)))
    return _Whitened(jnp.concatenate(blocks_x), jnp.concatenate(blocks_y), jacobian)


def graph_profile(
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    spatial: Array,
    *,
    observed: tuple[Array, ...] | None = None,
) -> tuple[Array, Array, Array]:
    """Return profiled coefficients, variance and log likelihood on admissible W.

    In each period, stacks have shape (number of roles, N, N), including a
    zero-length first axis for absent roles. Coefficients are SAR then SEM in
    declaration order within each role. For partial observations, each outcome
    vector is in the order of its observed integer indices into the full roster.
    """
    x, y, jacobian = _whiten(lag, error, design, outcome, spatial, observed)
    q, r = jnp.linalg.qr(x, mode="reduced")
    beta = solve_triangular(r, q.T @ y, lower=False)
    sigma2 = jnp.mean(jnp.square(y - x @ beta))
    log_likelihood = jacobian - y.size * (jnp.log(2 * jnp.pi * sigma2) + 1) / 2
    return beta, sigma2, log_likelihood


def graph_log_density(
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    beta: Array,
    spatial: Array,
    sigma2: Array,
    *,
    observed: tuple[Array, ...] | None = None,
) -> Array:
    """Sum period joint Gaussian log densities at a fixed parameter vector."""
    x, y, jacobian = _whiten(lag, error, design, outcome, spatial, observed)
    return (
        jacobian
        - (
            y.size * jnp.log(2 * jnp.pi * sigma2)
            + jnp.sum((y - x @ beta) ** 2) / sigma2
        )
        / 2
    )


class _GraphProblem(NamedTuple):
    lag: tuple[Array, ...]
    error: tuple[Array, ...]
    design: tuple[Array, ...]
    outcome: tuple[Array, ...]
    observed: tuple[Array, ...] | None
    norms: Array


def graph_coefficients(raw: Array, norms: Array, n_lag: int) -> Array:
    """Map unconstrained coefficients into separate admissible lag and error regions."""

    # Map each process independently into its open norm-weighted L1 ball.
    # hypot smooths the denominator at zero while preserving the strict bound.
    def bound(values: Array) -> Array:
        return values / (1 + jnp.hypot(values, jnp.finfo(values.dtype).eps).sum())

    return jnp.concatenate((bound(raw[:n_lag]), bound(raw[n_lag:]))) / norms


def _objective(raw: Array, problem: _GraphProblem) -> Array:
    return -graph_profile(
        problem.lag,
        problem.error,
        problem.design,
        problem.outcome,
        graph_coefficients(raw, problem.norms, problem.lag[0].shape[0]),
        observed=problem.observed,
    )[2]


def graph_norms(lag: tuple[Array, ...], error: tuple[Array, ...]) -> Array:
    """Maximum absolute row sum per role over all supplied periods."""

    def norms(stack: tuple[Array, ...]) -> Array:
        return jnp.max(
            jnp.stack([jnp.max(jnp.sum(jnp.abs(w), axis=-1), axis=-1) for w in stack]),
            axis=0,
        )

    row_norms = jnp.concatenate((norms(lag), norms(error)))
    return jnp.where(row_norms > 0, row_norms, 1)


def graph_diagnostics(
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    spatial: Array,
    norms: Array,
    *,
    observed: tuple[Array, ...] | None,
    tolerance: float,
) -> tuple[Array, Array]:
    """Check observed-design rank, conditional profile curvature and admissibility.

    Curvature varies only the original SAR and SEM coefficients. Every graph
    matrix and WX column is held at its candidate value.
    """
    x = _whiten(lag, error, design, outcome, spatial, observed).x
    finite = jnp.all(jnp.isfinite(x))
    identified = finite & (
        jnp.linalg.matrix_rank(jnp.where(finite, x, 0)) == x.shape[1]
    )
    if not spatial.size:
        return identified, jnp.zeros((), dtype=bool)

    def negative_profile(v: Array) -> Array:
        return -graph_profile(lag, error, design, outcome, v, observed=observed)[2]

    curvature = jax.hessian(negative_profile)(spatial)
    safe = jnp.where(jnp.isfinite(curvature), curvature, 0)
    eigenvalues = jnp.linalg.eigvalsh((safe + safe.T) / 2)
    threshold = jnp.finfo(safe.dtype).eps * spatial.size * jnp.max(jnp.abs(eigenvalues))
    identified &= jnp.all(jnp.isfinite(curvature)) & jnp.all(eigenvalues > threshold)
    n_lag = lag[0].shape[0]
    leverage = jnp.abs(spatial * norms)
    margin = 1 - jnp.maximum(leverage[:n_lag].sum(), leverage[n_lag:].sum())
    boundary = ~jnp.isfinite(margin) | (
        margin <= max(2 * tolerance, 8 * jnp.finfo(safe.dtype).eps)
    )
    return identified, boundary


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GraphMLFit:
    """Gaussian point fit with conditional structural and numerical diagnostics."""

    parameters: Array
    sigma2: Array
    log_likelihood: Array
    converged: Array
    identified: Array
    boundary: Array
    valid: Array
    parameter_names: tuple[str, ...] = field(metadata={"static": True})
    solver: SolverResult[Array]


def estimate_graph_ml(  # noqa: PLR0914 -- separate fit diagnostics
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    parameter_names: tuple[str, ...],
    *,
    observed: tuple[Array, ...] | None = None,
    max_steps: int = 256,
    tolerance: float = 1e-7,
) -> GraphMLFit:
    """Fit a pooled dense model, including explicitly supplied fixed-effect columns."""
    n_obs = sum(y.size for y in outcome)
    if n_obs <= design[0].shape[1]:
        msg = "Graph ML requires more observed outcomes than mean coefficients."
        raise ValueError(msg)
    norms = graph_norms(lag, error)
    problem = _GraphProblem(lag, error, design, outcome, observed, norms)
    raw = jnp.zeros_like(norms)
    if norms.size:
        solver = minimise_bfgs_result_args(
            _objective,
            raw,
            problem,
            max_steps=max_steps,
            rtol=tolerance,
            atol=tolerance,
        )
        spatial = graph_coefficients(solver.value, norms, lag[0].shape[0])

        converged = solver.converged
    else:
        spatial = raw
        converged = jnp.ones((), dtype=bool)
        # Linear coefficients use a separate direct QR solve. No spatial
        # variables remain for the outer optimizer in a pure SLX model.
        zero = jnp.asarray(0, dtype=jnp.int32)
        solver = SolverResult(
            raw,
            converged,
            zero,
            jnp.asarray(SolverStatus.SUCCESS),
            jnp.zeros((), dtype=raw.dtype),
            zero,
            zero,
        )
    beta, sigma2, log_likelihood = graph_profile(
        lag, error, design, outcome, spatial, observed=observed
    )
    identified, boundary = graph_diagnostics(
        lag,
        error,
        design,
        outcome,
        spatial,
        norms,
        observed=observed,
        tolerance=tolerance,
    )
    estimates = jnp.concatenate((beta, spatial))
    valid = (
        identified
        & converged
        & ~boundary
        & (sigma2 > 0)
        & jnp.isfinite(log_likelihood)
        & jnp.all(jnp.isfinite(estimates))
    )
    return GraphMLFit(
        jnp.where(valid, estimates, jnp.nan),
        sigma2,
        log_likelihood,
        converged,
        identified,
        boundary,
        valid,
        parameter_names,
        solver,
    )
