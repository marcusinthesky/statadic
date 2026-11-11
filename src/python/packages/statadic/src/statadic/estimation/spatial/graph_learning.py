"""Joint Gaussian graph-profile search on prepared numerical graph builders.

Named roles, rosters and graph recipes are prepared by the API. This owner
decodes numerical coordinates, profiles coefficients and retains search evidence.
Recipe builders are static callables; fitted-solution derivatives and compile
reuse across different builders are outside this internal contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping  # noqa: TC003

import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from statadic.core.typing import Array
from statadic.estimation.spatial.graph_likelihood import (
    GraphMLFit,
    graph_coefficients,
    graph_diagnostics,
    graph_norms,
    graph_profile,
)
from statadic.optimize.backends import minimise_bfgs_result_args

__all__: list[str] = []

type _GraphArrays = tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]


def _profile_sensitivity(
    raw: Array,
    coordinates: dict[str, Array],
    ll: Array,
    n_obs: int,
    score: Callable[[Array], Array],
) -> dict[str, bool]:
    """Screen for profile movement, without certifying local identification."""
    sensitivity: dict[str, bool] = {}
    for index, key in enumerate(sorted(coordinates)):
        value = coordinates[key]
        direction = random.normal(
            random.fold_in(random.key(0), index), value.shape, dtype=value.dtype
        )
        direction /= jnp.linalg.norm(direction)
        step = 0.05 * (jnp.linalg.norm(value) if value.ndim else 1.0)
        scores = []
        for sign in (-1, 1):
            trial, _ = ravel_pytree(
                (raw, {**coordinates, key: value + sign * step * direction})
            )
            scores.append(score(trial))
        delta = jnp.abs(jnp.stack(scores) - ll)
        floor = 16 * jnp.finfo(ll.dtype).eps * jnp.maximum(n_obs, jnp.abs(ll))
        sensitivity[key] = bool(jnp.any(jnp.isfinite(delta) & (delta > floor)))
    return sensitivity


def _estimate_joint_graph_ml(  # noqa: PLR0914 -- explicit numerical configuration
    initial: dict[str, Array],
    arrays: Callable[[Mapping[str, Array]], _GraphArrays],
    outcome: tuple[Array, ...],
    parameter_names: tuple[str, ...],
    *,
    n_lag: int,
    n_spatial: int,
    projection_ranks: Mapping[str, int],
    projection_norms: Mapping[str, int | float],
    bandwidths: tuple[str, ...],
    observed: tuple[Array, ...] | None,
    max_steps: int,
    tolerance: float,
) -> tuple[GraphMLFit, dict[str, Array], dict[str, bool], dict[str, bool]]:
    """Fit a prepared Gaussian profile while retaining actual graph coordinates."""
    dtype = jnp.result_type(*initial.values(), *outcome)
    flat, unravel = ravel_pytree((jnp.zeros(n_spatial, dtype=dtype), initial))
    n_obs = sum(y.size for y in outcome)
    if n_obs <= len(parameter_names) - n_spatial:
        msg = "Graph ML requires more observed outcomes than mean coefficients."
        raise ValueError(msg)

    def decode(vector: Array) -> tuple[Array, dict[str, Array]]:
        raw, state = unravel(vector)
        for name, norm in projection_norms.items():
            matrix = state[name]
            state[name] = matrix * (norm / jnp.linalg.norm(matrix))
        for name in bandwidths:
            state[name] = jnp.exp(state[name])
        return raw, state

    def evaluate(vector: Array) -> tuple[Array, Array, Array, Array]:
        raw, state = decode(vector)
        lag, error, design = arrays(state)
        spatial = graph_coefficients(raw, graph_norms(lag, error), n_lag)
        beta, variance, ll = graph_profile(
            lag, error, design, outcome, spatial, observed=observed
        )
        return beta, spatial, variance, ll

    def objective(vector: Array, _: None) -> Array:
        return -evaluate(vector)[-1] / n_obs

    solution = minimise_bfgs_result_args(
        objective,
        flat,
        None,
        max_steps=max_steps,
        rtol=tolerance,
        atol=tolerance,
        limited_memory=True,
    )
    beta, spatial, variance, ll = evaluate(solution.value)
    _, state = decode(solution.value)
    lag, error, design = arrays(state)
    identified, boundary = graph_diagnostics(
        lag,
        error,
        design,
        outcome,
        spatial,
        graph_norms(lag, error),
        observed=observed,
        tolerance=tolerance,
    )
    joined = jnp.concatenate((beta, spatial))
    valid = (
        solution.converged
        & jnp.isfinite(solution.residual_norm)
        & identified
        & ~boundary
        & jnp.isfinite(ll)
        & jnp.isfinite(variance)
        & (variance > 0)
        & jnp.all(jnp.isfinite(joined))
    )
    availability = {
        "finite-graph-arrays": bool(
            all(
                jnp.all(jnp.isfinite(block))
                for stacks in (lag, error, design)
                for block in stacks
            )
        ),
    }
    for name, rank in projection_ranks.items():
        matrix = state[name]
        finite = jnp.all(jnp.isfinite(matrix))
        available = finite & (
            jnp.linalg.matrix_rank(jnp.where(finite, matrix, 0)) == rank
        )
        availability[f"full-rank-projection:{name}"] = bool(available)
        valid &= available
    for name in bandwidths:
        available = jnp.isfinite(state[name]) & (state[name] > 0)
        availability[f"positive-finite-bandwidth:{name}"] = bool(available)
        valid &= available

    def score(trial: Array) -> Array:
        _, trial_state = decode(trial)
        w, v, x = arrays(trial_state)
        return graph_profile(w, v, x, outcome, spatial, observed=observed)[2]

    raw, coordinates = unravel(solution.value)
    sensitivity = _profile_sensitivity(raw, coordinates, ll, n_obs, score)
    valid &= all(sensitivity.values())
    fit = GraphMLFit(
        jnp.where(valid, joined, jnp.nan),
        variance,
        ll,
        solution.converged,
        identified,
        boundary,
        valid,
        parameter_names,
        solution,
    )
    return fit, state, sensitivity, availability
