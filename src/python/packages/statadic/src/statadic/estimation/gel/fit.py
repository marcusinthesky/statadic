# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Fixed and affine GEL fitting with retained multiplier and rank evidence.

These eager fitting boundaries preserve caller-owned precision and the original
carrier, multiplier cap and frozen-subspace profile objective. Calibration and
the iid reference are selected separately by :mod:`statadic.inference.gel`.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import optimistix as ox
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, ArrayLike, as_index, static_jaxtyping  # noqa: TC001
from statadic.estimation.gel._dual import (
    _GEL_RCOND,
    GelCriterion,
    _gel_identified_subspace,
    _gel_inner_sup,
    _gel_profile_objective_kernel,
    _require_host_gel,
)
from statadic.estimation.moments import (  # noqa: TC001
    AffineMomentProblem,
    FixedMomentProblem,
    _validate_moment_panel,
)

if TYPE_CHECKING:
    import jax

__all__ = ["AffineGEL", "AffineGelFit", "FixedGEL", "FixedGelFit", "GelCriterion"]


class FixedGelFit(NamedTuple):
    """Fixed moments and their actual multiplier supremum, without calibration.

    ``basis`` and ``keep`` retain the numerical row space whose rank licenses
    later test dimensions. ``converged`` concerns the multiplier solve only.
    """

    moments: Float[Array, "t m"]
    lam: Float[Array, " m"]
    objective: float
    criterion: GelCriterion
    basis: Float[Array, "m m"]
    keep: Float[Array, " m"]
    effective_rank: int
    converged: bool
    rank_cutoff: float


class AffineGelFit(NamedTuple):
    """Affine parameter fit and separate solver and retained-rank evidence.

    The profile uses the basis frozen at its least-squares starting value.
    ``rank_stable`` means only that final and starting numerical ranks match;
    it does not claim that their eigenspaces coincide or stay fixed en route.
    No test form or reference distribution is selected by this carrier.
    """

    moments0: Float[Array, "t m"]
    jacobian: Float[Array, "t m k"]
    moments: Float[Array, "t m"]
    theta: Float[Array, " k"]
    lam: Float[Array, " m"]
    objective: float
    criterion: GelCriterion
    basis: Float[Array, "m m"]
    keep: Float[Array, " m"]
    effective_rank: int
    rank_at_parameters: int
    rank_stable: bool
    inner_converged: bool
    outer_converged: bool
    converged: bool
    rank_cutoff: float


# Eager API validation is local to this fitting contract. Keeping these small
# checks here avoids making fitting depend on the later inference stage.
_MATRIX_NDIM = 2
_JACOBIAN_PANEL_NDIM = 3


def _validate_positive_integer(value: int, *, name: str) -> int:
    """Validate a non-boolean positive integer control."""
    index = as_index(value)
    if index is None or index < 1:
        message = f"{name} must be a positive integer."
        raise ValueError(message)
    return index


@static_jaxtyping("t m k")
def _coerce_affine_jacobian(
    moments0: ArrayLike,
    jacobian: ArrayLike,
) -> Float[Array, "t m k"]:
    """Validate and broadcast a common/per-observation affine Jacobian.

    A common ``(m, k)`` Jacobian is broadcast to the per-observation
    ``(n_observations, m, k)`` panel the linear GEL and Hansen J paths both
    consume; a panel of that shape passes through.

    Args:
        moments0: Validated moment panel, shape ``(n_observations, m)``. Only
            its shape is read.
        jacobian: Either a common ``(m, k)`` Jacobian or a per-observation
            ``(n_observations, m, k)`` panel.

    Returns:
        The ``(n_observations, m, k)`` derivative panel at the caller's dtype.

    Raises:
        ValueError: When the Jacobian is neither shape, disagrees with
            ``moments0``, or carries a nonfinite entry.

    """
    moments = jnp.asarray(moments0)
    derivative = jnp.asarray(jacobian)
    n_observations, m = moments.shape
    if derivative.ndim == _MATRIX_NDIM:
        if derivative.shape[0] != m:
            message = "a common jacobian must have shape (m, k)."
            raise ValueError(message)
        derivative_panel = jnp.broadcast_to(
            derivative[None, :, :], (n_observations, *derivative.shape)
        )
    elif derivative.ndim == _JACOBIAN_PANEL_NDIM:
        if derivative.shape[:2] != moments.shape:
            message = "a varying jacobian must have shape (n_observations, m, k)."
            raise ValueError(message)
        derivative_panel = derivative
    else:
        message = "jacobian must have shape (m, k) or (n_observations, m, k)."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(derivative_panel))):
        message = "jacobian must contain only finite values."
        raise ValueError(message)
    return derivative_panel


def _fixed_gel_fit(
    g: ArrayLike,
    criterion: GelCriterion = "cue",
    max_iter: int = 100,
) -> FixedGelFit:
    """Fit the GEL multiplier for a fixed moment panel.

    No structural parameter is fitted and no iid reference is assumed. Inputs
    need at least two finite rows and one identified stochastic direction.
    The host boundary rejects JAX tracing; private fixed-shape dual kernels
    remain transformable.
    """
    _require_host_gel(g, name="fixed_gel_fit")
    moments, _n_observations, _m = _validate_moment_panel(g)
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    if criterion not in ("cue", "el", "et"):
        message = f"unknown GEL criterion {criterion!r}"
        raise ValueError(message)
    basis, keep, rank = _gel_identified_subspace(moments, _GEL_RCOND)
    lam, objective, converged, effective_rank = _gel_inner_sup(
        moments, criterion, max_iter=max_iter, subspace=(basis, keep, rank)
    )
    if effective_rank < 1:
        message = "GEL requires at least one identified stochastic moment direction."
        raise ValueError(message)
    return FixedGelFit(
        moments=moments,
        lam=lam,
        objective=objective,
        criterion=criterion,
        basis=basis,
        keep=keep,
        effective_rank=effective_rank,
        converged=converged,
        rank_cutoff=_GEL_RCOND,
    )


@static_jaxtyping("t m k")
def _validate_profile_setup(
    moments0: Float[Array, "t m"],
    derivative_panel: Float[Array, "t m k"],
    criterion: GelCriterion,
    tol: float,
) -> int:
    """Validate identification, carrier, statistic, and solver controls."""
    m = moments0.shape[1]
    n_params = derivative_panel.shape[2]
    if m <= n_params:
        message = (
            "profiled GEL requires over-identification: the number of moments "
            f"m={m} must exceed the number of parameters k={n_params}."
        )
        raise ValueError(message)
    if jnp.linalg.matrix_rank(derivative_panel.mean(axis=0)) < n_params:
        message = "the mean moment jacobian does not identify every parameter."
        raise ValueError(message)
    if criterion not in ("cue", "el", "et"):
        message = f"unknown GEL criterion {criterion!r}"
        raise ValueError(message)
    if not math.isfinite(tol) or tol <= 0.0:
        message = "tol must be finite and positive."
        raise ValueError(message)
    return n_params


class _AffineGelSetup(NamedTuple):
    """Validated inputs and solver controls for :func:`affine_gel_fit`.

    Attributes:
        moments0: Validated moment panel at ``theta=0``, shape ``(t, m)``.
        derivative_panel: Coerced affine derivative, shape ``(t, m, k)``.
        n_observations: Panel length ``t``.
        n_params: Parameter count ``k``.
        max_iter: Validated inner-solve iteration cap.
        outer_max_iter: Validated outer-minimization iteration cap.

    """

    moments0: Float[Array, "t m"]
    derivative_panel: Float[Array, "t m k"]
    n_observations: int
    n_params: int
    max_iter: int
    outer_max_iter: int


class _ProfileStart(NamedTuple):
    """Outer starting parameter and the identified subspace frozen there.

    Attributes:
        theta0: Least-squares starting parameter, shape ``(k,)``.
        basis: Orthonormal ``(m, m)`` rotation of ``Ω(theta0)``.
        keep: ``(m,)`` 0/1 retention mask of identified directions.
        rank: Number of retained stochastic moment directions.

    """

    theta0: Float[Array, " k"]
    basis: Float[Array, "m m"]
    keep: Float[Array, " m"]
    rank: int


class _ProfileScore(NamedTuple):
    """Inner-sup score of the profiled fit in the frozen subspace.

    Attributes:
        lam: Inner-sup multiplier at the profiled parameter, shape ``(m,)``.
        objective: Inner-sup objective value at the profiled parameter.
        inner_converged: Whether the final inner multiplier solve converged.
        effective_rank: Retained stochastic moment rank.
        rank_at_parameters: Numerical rank evaluated at the final parameter.
        rank_stable: Whether the identified rank at the profiled parameter
            still matches the rank frozen at the starting value.

    """

    lam: Float[Array, " m"]
    objective: float
    inner_converged: bool
    effective_rank: int
    rank_at_parameters: int
    rank_stable: bool


def _prepare_affine_gel(
    g0: ArrayLike,
    jacobian: ArrayLike,
    criterion: GelCriterion,
    max_iter: int,
    outer_max_iter: int,
    tol: float,
) -> _AffineGelSetup:
    """Validate the profiled entry point's panels, carrier and solver controls."""
    _require_host_gel(g0, name="affine_gel_fit")
    _require_host_gel(jacobian, name="affine_gel_fit")
    moments0, n_observations, _m = _validate_moment_panel(g0, name="g0")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    outer_max_iter = _validate_positive_integer(
        outer_max_iter,
        name="outer_max_iter",
    )
    derivative_panel = _coerce_affine_jacobian(moments0, jacobian)
    # Kept at the call site, not in the shared helper: the parameter-count floor
    # is this entry point's contract, and `_validate_profile_setup` cannot catch
    # it (`m <= 0` and `rank < 0` are both false for an empty parameter axis).
    if derivative_panel.shape[2] < 1:
        message = "affine_gel_fit requires at least one parameter."
        raise ValueError(message)
    n_params = _validate_profile_setup(
        moments0,
        derivative_panel,
        criterion,
        tol,
    )
    return _AffineGelSetup(
        moments0=moments0,
        derivative_panel=derivative_panel,
        n_observations=n_observations,
        n_params=n_params,
        max_iter=max_iter,
        outer_max_iter=outer_max_iter,
    )


@static_jaxtyping("t m k")
def _affine_moments(
    setup: _AffineGelSetup,
    theta: Float[Array, " k"],
) -> Float[Array, "t m"]:
    """Evaluate the affine moment panel ``g_t(theta) = g0_t + D_t @ theta``."""
    moments0: Float[Array, "t m"] = setup.moments0
    derivative_panel: Float[Array, "t m k"] = setup.derivative_panel
    return moments0 + jnp.einsum("tmk,k->tm", derivative_panel, theta)


def _gel_profile_start(setup: _AffineGelSetup) -> _ProfileStart:
    """Least-squares starting parameter and the subspace frozen there.

    The identified subspace is decided once, at the starting parameter. Deciding
    it inside the objective instead would make the profile discontinuous in
    theta — see :func:`_gel_profile_objective_kernel` — and would put the
    non-differentiable ``eigh`` on the outer solver's tape.
    """
    dbar = setup.derivative_panel.mean(axis=0)
    g0bar = setup.moments0.mean(axis=0)
    theta0 = jnp.linalg.lstsq(dbar, -g0bar)[0]
    basis, keep, reference_rank = _gel_identified_subspace(
        _affine_moments(setup, theta0), _GEL_RCOND
    )
    # Use the original parameter scale: relative rank on the projection alone
    # can certify roundoff in a parameter direction annihilated by the mask.
    columns = jnp.max(jnp.abs(dbar), axis=0)
    normalized = dbar / jnp.where(columns > 0, columns, 1.0)
    projected = keep[:, None] * (basis.T @ normalized)
    cutoff = _GEL_RCOND * jnp.linalg.norm(normalized, ord=2)
    parameter_rank = int(jnp.sum(jnp.linalg.svd(projected, compute_uv=False) > cutoff))
    if parameter_rank != setup.n_params:
        message = "the frozen GEL moment coordinates do not identify every parameter."
        raise ValueError(message)
    return _ProfileStart(theta0=theta0, basis=basis, keep=keep, rank=reference_rank)


@static_jaxtyping("k")
def _minimise_profile(
    setup: _AffineGelSetup,
    start: _ProfileStart,
    criterion: GelCriterion,
    tol: float,
) -> tuple[Float[Array, " k"], bool]:
    """Minimize the profiled GEL objective over ``theta``.

    Returns:
        Tuple ``(theta, outer_converged)`` with ``theta`` the profiled parameter
        and ``outer_converged`` true only for a successful solve at a finite
        profiled objective.

    """
    max_iter = setup.max_iter
    panel_arg = jnp.asarray(setup.derivative_panel)
    moments_arg = jnp.asarray(setup.moments0)
    # The retention decision and basis both respect caller-owned precision.
    basis_arg = jnp.asarray(start.basis, dtype=moments_arg.dtype)
    keep_arg = jnp.asarray(start.keep, dtype=moments_arg.dtype)

    def profile_objective(
        theta: jax.Array,
        args: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        moments, derivative, subspace, mask = args
        return _gel_profile_objective_kernel(
            moments,
            derivative,
            theta,
            subspace,
            mask,
            criterion,
            max_iter,
        )

    solver_args = (moments_arg, panel_arg, basis_arg, keep_arg)
    fit = ox.minimise(
        profile_objective,
        ox.BFGS(rtol=tol, atol=tol),
        jnp.asarray(start.theta0),
        args=solver_args,
        max_steps=setup.outer_max_iter,
        throw=False,
    )
    theta: Float[Array, " k"] = jnp.asarray(fit.value)
    profile_value = float(profile_objective(jnp.asarray(theta), solver_args))
    outer_converged = bool(
        fit.result == ox.RESULTS.successful and math.isfinite(profile_value)
    )
    return theta, outer_converged


@static_jaxtyping("k")
def _score_profile(
    setup: _AffineGelSetup,
    start: _ProfileStart,
    theta: Float[Array, " k"],
    criterion: GelCriterion,
) -> _ProfileScore:
    """Score the profiled fit in the subspace the profile was minimized over.

    The final multiplier is solved in exactly the frozen profile coordinates.
    A separate final numerical-rank check supplies equality evidence only;
    no eigenspace or entire-path stability claim follows from it.

    Returns:
        :class:`_ProfileScore`.

    """
    fitted = _affine_moments(setup, theta)
    lam, objective, inner_converged, effective_rank = _gel_inner_sup(
        fitted,
        criterion,
        max_iter=setup.max_iter,
        subspace=(start.basis, start.keep, start.rank),
    )
    _, _, rank_at_theta = _gel_identified_subspace(fitted, _GEL_RCOND)
    return _ProfileScore(
        lam=lam,
        objective=objective,
        inner_converged=inner_converged,
        effective_rank=effective_rank,
        rank_at_parameters=rank_at_theta,
        rank_stable=bool(rank_at_theta == start.rank),
    )


def _affine_gel_fit(
    g0: ArrayLike,
    jacobian: ArrayLike,
    criterion: GelCriterion = "cue",
    max_iter: int = 100,
    outer_max_iter: int = 500,
    tol: float = 1e-8,
) -> AffineGelFit:
    """Fit ``g_t(theta) = g0_t + D_t @ theta`` inside a GEL criterion.

    The multiplier basis is frozen at the least-squares start. Inner and outer
    convergence and final numerical-rank equality are recorded separately.
    This entry point preserves the original overidentification and mean
    Jacobian rank guards; it does not select a test or assume iid observations.
    """
    setup = _prepare_affine_gel(g0, jacobian, criterion, max_iter, outer_max_iter, tol)
    start = _gel_profile_start(setup)
    theta, outer_converged = _minimise_profile(setup, start, criterion, tol)
    score = _score_profile(setup, start, theta, criterion)
    converged = bool(
        outer_converged and score.inner_converged and math.isfinite(score.objective)
    )
    if score.effective_rank - setup.n_params < 1:
        message = (
            "profiled GEL requires the identified moment rank "
            f"r={score.effective_rank} to exceed the parameter count "
            f"k={setup.n_params}."
        )
        raise ValueError(message)
    return AffineGelFit(
        moments0=setup.moments0,
        jacobian=setup.derivative_panel,
        moments=_affine_moments(setup, theta),
        theta=theta,
        lam=score.lam,
        objective=score.objective,
        criterion=criterion,
        basis=start.basis,
        keep=start.keep,
        effective_rank=score.effective_rank,
        rank_at_parameters=score.rank_at_parameters,
        rank_stable=score.rank_stable,
        inner_converged=score.inner_converged,
        outer_converged=outer_converged,
        converged=converged,
        rank_cutoff=_GEL_RCOND,
    )


@dataclass(frozen=True)
class FixedGEL:
    """Fit a multiplier to supplied moments at the eager host boundary."""

    criterion: GelCriterion = "cue"
    max_iter: int = 100

    def estimate(self, problem: FixedMomentProblem) -> FixedGelFit:
        """Retain the multiplier and numerical rank evidence."""
        return _fixed_gel_fit(problem.moments, self.criterion, self.max_iter)


@dataclass(frozen=True)
class AffineGEL:
    """Fit affine structural parameters and their GEL multiplier eagerly."""

    criterion: GelCriterion = "cue"
    max_iter: int = 100
    outer_max_iter: int = 500
    tol: float = 1e-8

    def estimate(self, problem: AffineMomentProblem) -> AffineGelFit:
        """Retain the profile fit and separate convergence evidence."""
        if problem.weight is not None:
            msg = "AffineGEL does not accept a GMM weight."
            raise ValueError(msg)
        return _affine_gel_fit(
            problem.moments0,
            problem.jacobian,
            self.criterion,
            self.max_iter,
            self.outer_max_iter,
            self.tol,
        )
