"""Fixed-draw Gaussian orthant integration for differentiable objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp, ndtri

from statadic.core.results import SolverResult, SolverStatus
from statadic.optimize.backends import minimise_bfgs_result_args

__all__ = [
    "GaussianIntegration",
    "GaussianIntegrationResult",
    "GaussianProposal",
    "IntegrationDesign",
    "InverseNormalResult",
    "gaussian_log_survival",
    "gaussian_tilt_proposal",
    "inverse_log_normal_result",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import DTypeLike


def _inverse_log_normal_solve(logp: jax.Array) -> jax.Array:
    """Invert log Phi by fixed-iteration safeguarded Newton steps."""
    reflect = logp > -jnp.log(jnp.asarray(2, dtype=logp.dtype))
    tail_logp = jnp.where(reflect, jnp.log(-jnp.expm1(logp)), logp)
    lower = -2 * jnp.sqrt(jnp.maximum(-tail_logp, 0)) - 10
    upper = jnp.zeros_like(logp)

    x = jnp.where(
        tail_logp > -20,  # noqa: PLR2004  # Conservative representable CDF.
        ndtri(jnp.exp(jnp.maximum(tail_logp, -20))),
        -jnp.sqrt(-2 * tail_logp),
    )

    def step(
        _: jax.Array, state: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        lo, hi, current = state
        logcdf = log_ndtr(current)
        error = logcdf - tail_logp
        lo = jnp.where(error < 0, current, lo)
        hi = jnp.where(error > 0, current, hi)
        log_density = -0.5 * current * current - 0.5 * jnp.log(
            jnp.asarray(2 * jnp.pi, dtype=current.dtype)
        )
        proposal = current - error * jnp.exp(logcdf - log_density)
        proposal = jnp.where(
            jnp.isfinite(proposal) & (proposal >= lo) & (proposal <= hi),
            proposal,
            lo + (hi - lo) / 2,
        )
        return lo, hi, proposal

    _, _, x = jax.lax.fori_loop(0, 16, step, (lower, upper, x))
    x = jnp.where(reflect, -x, x)
    valid = jnp.isfinite(logp) & (logp < 0) & jnp.isfinite(x)
    return jnp.where(valid, x, jnp.nan)


def _inverse_log_normal(logp: jax.Array) -> jax.Array:
    """Differentiate the converged log-CDF equation using native implicit AD."""
    return jax.lax.custom_root(
        partial(_log_normal_residual, logp=logp),
        -jnp.sqrt(jnp.maximum(-2 * logp, 1)),
        partial(_solve_log_normal_root, logp=logp),
        _solve_diagonal_tangent,
    )


def _log_normal_residual(value: jax.Array, *, logp: jax.Array) -> jax.Array:
    """Canonical equation differentiated by the implicit-root primitive."""
    return log_ndtr(value) - logp


def _solve_log_normal_root(
    _function: Callable[[jax.Array], jax.Array], _initial: jax.Array, *, logp: jax.Array
) -> jax.Array:
    """Use the stable safeguarded primal solve without differentiating its steps."""
    return _inverse_log_normal_solve(logp)


def _solve_diagonal_tangent(
    linear: Callable[[jax.Array], jax.Array], rhs: jax.Array
) -> jax.Array:
    """Solve the elementwise tangent equation using its native linearization."""
    return rhs / linear(jnp.ones_like(rhs))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InverseNormalResult:
    """Inverse log-CDF with numerical convergence and resolution evidence."""

    value: jax.Array
    valid: jax.Array
    residual: jax.Array
    iterations: jax.Array


def inverse_log_normal_result(logp: jax.Array) -> InverseNormalResult:
    """Invert a scalar or array of log probabilities without exponentiating tails."""
    logp = jnp.asarray(logp)
    value = _inverse_log_normal(logp)
    reflect = logp > -jnp.log(jnp.asarray(2, dtype=logp.dtype))
    target = jnp.where(reflect, jnp.log(-jnp.expm1(logp)), logp)
    residual = jnp.abs(log_ndtr(-jnp.abs(value)) - target)
    tolerance = 64 * jnp.finfo(logp.dtype).eps * jnp.maximum(1, jnp.abs(target))
    valid = jnp.isfinite(value) & jnp.isfinite(residual) & (residual <= tolerance)
    return InverseNormalResult(
        jnp.where(valid, value, jnp.nan),
        valid,
        residual,
        jnp.asarray(16, dtype=jnp.int32),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussianProposal:
    """Frozen variable order and sequential normal tilts, fitted at a pilot point.

    ``valid`` checks preparation arithmetic and its saddle residual, not the
    integration error. ``solver`` retains the actual numerical optimization.
    No minimax optimality, deterministic error bound or rare-tail efficiency
    theorem is inferred from these numerical diagnostics.
    """

    permutation: jax.Array
    tilt: jax.Array
    solver: SolverResult[jax.Array]
    saddle_residual: jax.Array
    valid: jax.Array

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        """Expose the frozen proposal and its preparation evidence as array leaves."""
        return tuple(getattr(self, item.name) for item in fields(self)), None

    @classmethod
    def tree_unflatten(
        cls, auxiliary: None, children: tuple[object, ...]
    ) -> GaussianProposal:
        """Allow structural Boolean/None filters without invoking array validation."""
        del auxiliary
        result = object.__new__(cls)
        for item, child in zip(fields(cls), children, strict=True):
            object.__setattr__(result, item.name, child)
        return result


def _tilt_residual(value: jax.Array, args: tuple[jax.Array, jax.Array]) -> jax.Array:
    """Evaluate the one-sided specialization of Botev's saddle equations (8)."""
    strict, lower = args
    width = lower.size - 1
    location = jnp.concatenate((value[:width], jnp.zeros(1, dtype=value.dtype)))
    tilt = jnp.concatenate((value[width:], jnp.zeros(1, dtype=value.dtype)))
    threshold = lower - strict @ location - tilt
    log_density = -0.5 * threshold**2 - 0.5 * jnp.log(
        jnp.asarray(2 * jnp.pi, dtype=value.dtype)
    )
    mills = jnp.exp(log_density - log_ndtr(-threshold))
    return jnp.concatenate(
        ((strict.T @ mills - tilt)[:width], (tilt - location + mills)[:width])
    )


def _tilt_loss(value: jax.Array, args: tuple[jax.Array, jax.Array]) -> jax.Array:
    residual = _tilt_residual(value, args)
    return 0.5 * jnp.sum(residual**2)


def gaussian_tilt_proposal(  # noqa: PLR0914  # pilot Gaussian, saddle and solver evidence
    mean: jax.Array,
    covariance: jax.Array,
    lower: jax.Array,
    *,
    max_steps: int = 256,
    tolerance: float = 1e-5,
) -> GaussianProposal:
    """Prepare a fixed tilted sequential proposal at explicit pilot parameters.

    Bounds are ordered by descending marginal standardized threshold. This is
    a deterministic heuristic, not Genz's greedy conditional reordering.
    Maintained L-BFGS minimizes squared saddle residuals from Botev (2017),
    equations (5), (6) and (8), restricted to the first ``d-1`` coordinates.
    The last coordinate is integrated analytically. A small independently
    recomputed saddle residual is required; solver termination alone is
    insufficient. This is a numerical proposal fit, not certified minimax
    tilting. See https://arxiv.org/abs/1603.04166, sections 2 and 3.

    Prepare once outside the likelihood optimization and retain the proposal
    with the draw design. Likelihood gradients/Hessians then differentiate the
    fixed-proposal approximation. Differentiating this preparation algorithm
    to obtain derivatives of a refitted proposal is outside its contract.
    No covariance ridge, probability clipping or change of target is applied.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        message = "max_steps must be a positive integer"
        raise ValueError(message)
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        message = "tolerance must be finite and positive"
        raise ValueError(message)
    mean, covariance, lower = _gaussian_inputs(mean, covariance, lower)
    if mean.size < 1:
        message = "A tilted proposal requires a nonempty Gaussian vector"
        raise ValueError(message)
    marginal = (lower - mean) / jnp.sqrt(jnp.diag(covariance))
    order = jnp.argsort(-marginal, stable=True)
    ordered_covariance = covariance[jnp.ix_(order, order)]
    chol = jnp.linalg.cholesky(ordered_covariance)
    diagonal = jnp.diag(chol)
    strict = chol / diagonal[:, None] - jnp.eye(mean.size, dtype=mean.dtype)
    bounds = (lower[order] - mean[order]) / diagonal
    finite = (
        jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(covariance == covariance.T)
        & jnp.all(jnp.isfinite(chol))
        & jnp.all(diagonal > 0)
    )
    safe_strict = jnp.where(finite, strict, jnp.zeros_like(strict))
    safe_bounds = jnp.where(finite, bounds, jnp.zeros_like(bounds))
    width = mean.size - 1
    if width == 0:
        solver = SolverResult(
            value=jnp.zeros(0, dtype=mean.dtype),
            converged=finite,
            iterations=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.asarray(
                jnp.where(finite, SolverStatus.SUCCESS, SolverStatus.NONFINITE),
                dtype=jnp.int32,
            ),
            residual_norm=jnp.asarray(0, dtype=mean.dtype),
            rank=jnp.asarray(-1, dtype=jnp.int32),
            max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
        )
        return GaussianProposal(
            order,
            jnp.zeros(1, dtype=mean.dtype),
            solver,
            jnp.asarray(0, dtype=mean.dtype),
            finite,
        )
    args = (safe_strict, safe_bounds)

    def conditional_mean(i: jax.Array, values: jax.Array) -> jax.Array:
        threshold = safe_bounds[i] - safe_strict[i] @ values
        log_density = -0.5 * threshold**2 - 0.5 * jnp.log(
            jnp.asarray(2 * jnp.pi, dtype=mean.dtype)
        )
        return values.at[i].set(jnp.exp(log_density - log_ndtr(-threshold)))

    pilot_mean = jax.lax.fori_loop(
        0, mean.size, conditional_mean, jnp.zeros(mean.size, dtype=mean.dtype)
    )
    solver = minimise_bfgs_result_args(
        _tilt_loss,
        jnp.concatenate((pilot_mean[:width], jnp.zeros(width, dtype=mean.dtype))),
        args,
        max_steps=max_steps,
        rtol=tolerance**2,
        atol=tolerance**2,
        limited_memory=True,
    )
    residual = jnp.max(jnp.abs(_tilt_residual(solver.value, args)))
    tilt = jnp.concatenate((solver.value[width:], jnp.zeros(1, dtype=mean.dtype)))
    valid = finite & jnp.all(jnp.isfinite(tilt)) & (residual <= tolerance)
    return GaussianProposal(order, tilt, solver, residual, valid)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IntegrationDesign:
    """A replayable key or explicit point array with fixed draw budget."""

    key: jax.Array | None
    points: jax.Array | None
    replicates: int = field(metadata={"static": True})
    draws: int = field(metadata={"static": True})
    proposal: GaussianProposal | None = None

    def tree_flatten(self) -> tuple[tuple[object, ...], tuple[int, int]]:
        """Keep keys, points and proposal dynamic, and draw dimensions static."""
        return (self.key, self.points, self.proposal), (self.replicates, self.draws)

    @classmethod
    def tree_unflatten(
        cls, auxiliary: tuple[int, int], children: tuple[object, ...]
    ) -> IntegrationDesign:
        """Reconstruct structural masks while preserving the actual static budgets."""
        result = object.__new__(cls)
        for name, child in zip(("key", "points", "proposal"), children, strict=True):
            object.__setattr__(result, name, child)
        object.__setattr__(result, "replicates", auxiliary[0])
        object.__setattr__(result, "draws", auxiliary[1])
        return result

    @classmethod
    def from_key(
        cls,
        key: jax.Array,
        *,
        replicates: int = 8,
        draws: int = 1024,
        proposal: GaussianProposal | None = None,
    ) -> IntegrationDesign:
        """Construct fixed keyed antithetic draw streams."""
        if replicates < 1 or draws < 2 or draws % 2:  # noqa: PLR2004
            message = "replicates must be positive and draws must be positive and even"
            raise ValueError(message)
        return cls(key, None, replicates, draws, proposal)

    @classmethod
    def from_points(
        cls, points: jax.Array, *, proposal: GaussianProposal | None = None
    ) -> IntegrationDesign:
        """Construct a design from caller-supplied open-unit points."""
        points = jnp.asarray(points)
        if points.ndim != 3 or points.shape[0] < 1 or points.shape[1] < 1:  # noqa: PLR2004
            message = "points must have shape (replicates, draws, dimensions)"
            raise ValueError(message)
        return cls(None, points, points.shape[0], points.shape[1], proposal)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GaussianIntegration:
    """Static integration execution controls."""

    chunk_size: int = field(default=128, metadata={"static": True})

    def __post_init__(self) -> None:
        """Validate the static chunk budget."""
        if self.chunk_size < 1:
            message = "chunk_size must be positive"
            raise ValueError(message)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GaussianIntegrationResult:
    """Approximate log probability and per-replicate integration diagnostics.

    Effective counts describe importance-weight concentration, not independent
    draws; keyed draws are antithetic pairs. Relative standard error is a
    between-replicate diagnostic and assumes independent replicate designs.
    Neither quantity certifies approximation accuracy.
    """

    log_probability: jax.Array
    replicate_log_probabilities: jax.Array
    effective_counts: jax.Array
    relative_standard_error: jax.Array
    valid: jax.Array


def _keyed_points(
    key: jax.Array, replicate: jax.Array, indices: jax.Array, dim: int, dtype: DTypeLike
) -> jax.Array:
    repkey = jax.random.fold_in(key, replicate)
    pair = indices // 2

    def fold(i: jax.Array) -> jax.Array:
        return jax.random.fold_in(repkey, i)

    keys = jax.vmap(fold)(pair)
    # Use integer midpoint bins: unlike adding a floating midpoint to a full
    # mantissa, neither endpoint can round to 0 or 1.
    bits = 23 if dtype == jnp.float32 else 52
    uint = jnp.uint32 if bits == 23 else jnp.uint64  # noqa: PLR2004

    def draw_bits(k: jax.Array) -> jax.Array:
        return jax.random.bits(k, (dim,), dtype=uint)

    values = jax.vmap(draw_bits)(keys)
    values = jax.lax.shift_right_logical(
        values,
        jnp.asarray((32 if bits == 23 else 64) - bits, dtype=uint),  # noqa: PLR2004
    )
    u = (values.astype(dtype) + jnp.asarray(0.5, dtype=dtype)) * jnp.asarray(
        2.0**-bits, dtype=dtype
    )
    return jnp.where((indices % 2)[:, None] == 0, u, 1 - u)


_DEFAULT_INTEGRATION = GaussianIntegration()


def _gaussian_inputs(
    mean: jax.Array, covariance: jax.Array, lower: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Check the common fixed-shape Gaussian input contract and caller precision."""
    mean, covariance, lower = (
        jnp.asarray(mean),
        jnp.asarray(covariance),
        jnp.asarray(lower),
    )
    if (
        mean.ndim != 1
        or lower.shape != mean.shape
        or covariance.shape != (mean.size, mean.size)
    ):
        message = "mean, lower and covariance shapes must align"
        raise ValueError(message)
    dtype = jnp.result_type(mean, covariance, lower, 1.0)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        message = "Gaussian integration requires float32 or float64 inputs."
        raise TypeError(message)
    return mean.astype(dtype), covariance.astype(dtype), lower.astype(dtype)


def gaussian_log_survival(  # noqa: C901, PLR0914, PLR0915  # Integration state.
    mean: jax.Array,
    covariance: jax.Array,
    lower: jax.Array,
    design: IntegrationDesign,
    spec: GaussianIntegration = _DEFAULT_INTEGRATION,
) -> GaussianIntegrationResult:
    """Approximate a Gaussian joint upper tail with fixed-draw sequential integration.

    With no proposal this is ordinary zero-tilt GHK. A supplied frozen proposal
    changes the sequential importance law and variable order; the exact
    likelihood-ratio correction retains the same Gaussian upper-tail target.
    Gradients/Hessians in the Gaussian inputs are of that fixed-proposal,
    fixed-draw approximation. Validity describes numerical representability,
    not an integration-error bound or a converged fitted solution derivative.
    """
    mean, covariance, lower = _gaussian_inputs(mean, covariance, lower)
    dtype = mean.dtype
    dim = mean.size
    if design.points is not None and design.points.shape[2] != dim:
        message = "point dimension must match the Gaussian dimension"
        raise ValueError(message)
    if dim == 0:
        return GaussianIntegrationResult(
            jnp.asarray(0, dtype=dtype),
            jnp.zeros((design.replicates,), dtype=dtype),
            jnp.full((design.replicates,), design.draws, dtype=dtype),
            jnp.asarray(0, dtype=dtype),
            jnp.asarray(True),  # noqa: FBT003
        )
    proposal_valid = jnp.asarray(True)  # noqa: FBT003  # scalar numerical evidence
    tilt = jnp.zeros(dim, dtype=dtype)
    if design.proposal is not None:
        proposal = design.proposal
        if proposal.permutation.shape != (dim,) or proposal.tilt.shape != (dim,):
            message = "Gaussian proposal dimensions must match the target"
            raise ValueError(message)
        if not jnp.issubdtype(proposal.permutation.dtype, jnp.integer):
            message = "Gaussian proposal permutation must contain integers"
            raise TypeError(message)
        proposal_valid = (
            proposal.valid
            & jnp.all(jnp.sort(proposal.permutation) == jnp.arange(dim))
            & jnp.all(jnp.isfinite(proposal.tilt))
            & (proposal.tilt[-1] == 0)
        )
        order = jnp.where(proposal_valid, proposal.permutation, jnp.arange(dim))
        mean, lower = mean[order], lower[order]
        covariance = covariance[jnp.ix_(order, order)]
        tilt = jnp.where(proposal_valid, proposal.tilt.astype(dtype), tilt)
    chol = jnp.linalg.cholesky(covariance)
    base_valid = (
        proposal_valid
        & jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.isfinite(chol))
        & jnp.all(jnp.diag(chol) > 0)
        & jnp.all(covariance == covariance.T)
    )
    nchunks = (design.draws + spec.chunk_size - 1) // spec.chunk_size

    def one_rep(rep: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        def chunk_step(
            state: tuple[jax.Array, jax.Array, jax.Array], chunk: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
            sumw, sumw2, good = state
            ids = chunk * spec.chunk_size + jnp.arange(
                spec.chunk_size, dtype=jnp.uint32
            )
            mask = ids < design.draws
            if design.points is None:
                if design.key is None:
                    message = "a keyed design requires a PRNG key"
                    raise ValueError(message)
                points = _keyed_points(design.key, rep, ids, dim, dtype)
            else:
                points = jnp.asarray(
                    design.points[rep, jnp.minimum(ids, design.draws - 1), :],
                    dtype=dtype,
                )
            point_valid = jnp.all(
                (points > 0) & (points < 1) & jnp.isfinite(points), axis=1
            )
            safe_points = jnp.where(
                point_valid[:, None], points, jnp.asarray(0.5, dtype=dtype)
            )
            z = jnp.zeros((spec.chunk_size, dim), dtype=dtype)
            logw = jnp.zeros((spec.chunk_size,), dtype=dtype)

            def coordinate(
                i: jax.Array, state2: tuple[jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array]:
                latents, weights = state2
                a = (lower[i] - mean[i] - latents @ chol[i]) / chol[i, i]
                logtail = log_ndtr(tilt[i] - a)
                draw = jax.lax.cond(
                    i == dim - 1,
                    lambda: jnp.zeros_like(logtail),
                    lambda: (
                        tilt[i]
                        - inverse_log_normal_result(
                            logtail + jnp.log1p(-safe_points[:, i])
                        ).value
                    ),
                )
                # phi(z) / phi(z-tilt) = exp(tilt^2/2 - tilt*z).
                # The terminal tilt is zero because its tail is integrated out.
                correction = 0.5 * tilt[i] ** 2 - tilt[i] * draw
                return latents.at[:, i].set(draw), weights + logtail + correction

            z, logw = jax.lax.fori_loop(0, dim, coordinate, (z, logw))
            ok = point_valid & jnp.isfinite(logw) & jnp.all(jnp.isfinite(z), axis=1)
            good = good & jnp.all(jnp.where(mask, ok, True))  # noqa: FBT003
            logw = jnp.where(mask & ok, logw, -jnp.inf)
            return (
                jnp.logaddexp(sumw, logsumexp(logw)),
                jnp.logaddexp(sumw2, logsumexp(2 * logw)),
                good,
            ), None

        init = (
            jnp.asarray(-jnp.inf, dtype=dtype),
            jnp.asarray(-jnp.inf, dtype=dtype),
            jnp.asarray(True),  # noqa: FBT003
        )
        (s, s2, ok), _ = jax.lax.scan(
            jax.checkpoint(chunk_step), init, jnp.arange(nchunks)
        )
        return (
            s - jnp.log(jnp.asarray(design.draws, dtype=dtype)),
            jnp.exp(2 * s - s2),
            ok,
        )

    logs, ess, valid_reps = jax.lax.map(
        one_rep, jnp.arange(design.replicates, dtype=jnp.uint32)
    )
    valid = base_valid & jnp.all(valid_reps) & jnp.all(jnp.isfinite(logs))
    value = logsumexp(logs) - jnp.log(jnp.asarray(design.replicates, dtype=dtype))
    if design.replicates > 1:
        scaled = jnp.exp(logs - jnp.max(logs))
        relative_se = jnp.std(scaled, ddof=1) / (
            jnp.mean(scaled) * jnp.sqrt(jnp.asarray(design.replicates, dtype=dtype))
        )
    else:
        relative_se = jnp.asarray(jnp.nan, dtype=dtype)
    return GaussianIntegrationResult(
        jnp.where(valid, value, jnp.nan), logs, ess, relative_se, valid
    )
