# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Shared thresholds and validators for the inference tests.

This module is the sink of the :mod:`statadic.inference` package: it imports
no sibling, so every other module may depend on it without creating a cycle.
(:class:`~statadic.optimize.simplex.NonnegativeQPSolution` is imported for typing
only and lives outside the package, so it does not weaken that property.)

Shared checks serve multiple inference families. Finite moment-row validation
belongs to the shared numerical problem owner in ``estimation.moments``.
The nonnegative-QP status
boundary is shared by GMS and Wolak. Affine GEL derivative validation belongs
in its fitting owner; the retired affine-J path no longer needs a duplicate.

Precision
---------
t65.1 removed the ``Float = NDArray[np.float64]`` alias and, with it, the
``np.asarray(x, dtype=np.float64)`` coercion every validator performed. That
coercion was a silent upcast: it forced float64 arithmetic regardless of what
the caller had configured, which is the same ownership inversion t56.7 deleted
the 51 ``jax.enable_x64`` doors to fix. These helpers now evaluate at the
caller's dtype and return :class:`jax.Array`; a stage that needs float64 opens
the scope itself (``pipeline.precision.own_float64``).

The validators remain **eager**: each closes a device sync with ``bool()`` or
``float()`` to reach a Python ``raise``, so none is jittable and none is
intended to be. That is the same shape as
:func:`statadic.discrepancy.metrize.sqrt_energy_functional`, and it is why they sit
at API doors rather than inside kernels.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.stats import chi2
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, ArrayLike, as_index, static_jaxtyping  # noqa: TC001

if TYPE_CHECKING:
    from statadic.optimize.simplex import NonnegativeQPSolution


class ProjectionUnavailableError(RuntimeError):
    """A completed numerical projection reported unusable solver lanes.

    Backend/runtime failures retain their original exception types. Counts describe
    actual native solver status flags; they do not establish statistical validity.
    """

    def __init__(
        self, message: str, *, lane_counts: dict[str, int], max_projected_kkt: float
    ) -> None:
        super().__init__(message)
        self.lane_counts = dict(lane_counts)
        self.max_projected_kkt = max_projected_kkt


_BINDING_TOLERANCE = 1e-10
_WEAK_IDENTIFICATION_THRESHOLD = 10.0
_MATRIX_NDIM = 2
_PSD_TOLERANCE = 1e-10


def _chi2_survival(stat: float, dof: int) -> float:
    """Upper-tail chi-square survival ``P(X > stat)`` for ``X ~ chi2(dof)``.

    Replaces the ``scipy.stats.chi2.sf`` host boundary that ``gel.py`` and
    ``wolak.py`` each owned. ``jax.scipy.stats.chi2.sf`` reaches the same tail
    through ``gammaincc`` rather than SciPy's Cephes ``chdtrc``.

    **Precision is the caller's to configure.** This routine does not mutate
    the JAX x64 config; it evaluates at whatever the caller has set. Under x64
    it tracks SciPy to ``2.5e-14`` relative; under the default float32 config
    the far tail degrades to roughly ``4e-9`` relative. Both are immaterial for
    the way these p-values are consumed — they are compared against alpha and
    reported to six decimals, never log-combined or used in a stopping rule —
    so neither warrants forcing a precision here.

    Args:
        stat: Observed nonnegative statistic.
        dof: Positive integer degrees of freedom.

    Returns:
        The survival probability as a host float.

    """
    return float(chi2.sf(jnp.asarray(stat), dof))


def _validate_positive_integer(value: int, *, name: str) -> int:
    """Validate a non-boolean positive integer control."""
    index = as_index(value)
    if index is None or index < 1:
        message = f"{name} must be a positive integer."
        raise ValueError(message)
    return index


def _validate_bandwidth(bandwidth: int | None, n_observations: int) -> int | None:
    """Validate an optional HAC lag against its observation count."""
    if bandwidth is None:
        return None
    index = as_index(bandwidth)
    if index is None or index < 0 or index >= n_observations:
        message = "bandwidth must be an integer in [0, n_observations)."
        raise ValueError(message)
    return index


@static_jaxtyping("d")
def _validate_full_rank_covariance(
    covariance: ArrayLike,
    *,
    dimension: int | None = None,
) -> Float[Array, "d d"]:
    """Validate a finite symmetric positive-definite covariance matrix.

    Returns the symmetrized matrix at the caller's dtype. The eigenvalue and
    rank tests are evaluated eagerly and reach a Python ``raise``, so this is
    an API door rather than a kernel.
    """
    matrix = jnp.asarray(covariance)
    if (
        matrix.ndim != _MATRIX_NDIM
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 1
    ):
        message = "covariance must be a nonempty square matrix."
        raise ValueError(message)
    if dimension is not None and matrix.shape != (dimension, dimension):
        message = f"covariance must have shape ({dimension}, {dimension})."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        message = "covariance must contain only finite values."
        raise ValueError(message)
    if not bool(jnp.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)):
        message = "covariance must be symmetric."
        raise ValueError(message)
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(jnp.max(jnp.abs(eigenvalues))))
    if float(eigenvalues.min()) < -_PSD_TOLERANCE * scale:
        message = "covariance must be positive semidefinite."
        raise ValueError(message)
    rank = int(jnp.linalg.matrix_rank(symmetric))
    if rank < symmetric.shape[0]:
        message = (
            "covariance must be full rank for the requested reference law; "
            f"observed rank={rank} for dimension={symmetric.shape[0]}."
        )
        raise ValueError(message)
    return symmetric


def _require_projection_solution(
    solution: NonnegativeQPSolution,
    *,
    context: str,
) -> None:
    """Reject unusable compiled solver lanes at the eager inference boundary.

    Shared by the two batched nonnegative-QP consumers — ``wolak.py``'s
    chi-bar-squared weight simulation and ``gms.py``'s QLR statistics. Both
    reached the same verdict from byte-identical bodies before t65.5; ``context``
    is the only thing that ever differed.

    Args:
        solution: Batched solver result whose per-lane status flags are read.
        context: Caller name interpolated into the failure message.

    Raises:
        RuntimeError: When any lane is nonfinite, nonconvex, unbounded, or
            unconverged. The message reports each count and the worst projected
            KKT residual over the rejected lanes.

    """
    input_finite = jnp.asarray(solution.input_finite, dtype=bool)
    convex = jnp.asarray(solution.convex, dtype=bool)
    bounded = jnp.asarray(solution.bounded, dtype=bool)
    converged = jnp.asarray(solution.converged, dtype=bool)
    usable = input_finite & convex & bounded & converged
    if bool(jnp.all(usable)):
        return
    residual = jnp.asarray(solution.projected_gradient_norm)
    # Eager boolean-mask selection: `usable` is concrete here, so the
    # data-dependent output length is legal. It would not be under `jit`.
    failed_residual = residual[~usable]
    max_residual = float(jnp.max(failed_residual)) if failed_residual.size else 0.0
    message = (
        f"{context} nonnegative-QP solver rejected "
        f"{int(jnp.count_nonzero(~usable))}/{usable.size} lanes "
        f"(nonfinite={int(jnp.count_nonzero(~input_finite))}, "
        f"nonconvex={int(jnp.count_nonzero(~convex))}, "
        f"unbounded_like={int(jnp.count_nonzero(~bounded))}, "
        f"unconverged={int(jnp.count_nonzero(~converged))}, "
        f"max_projected_kkt={max_residual:.6g})."
    )
    raise ProjectionUnavailableError(
        message,
        lane_counts={
            "rejected": int(jnp.count_nonzero(~usable)),
            "total": int(usable.size),
            "nonfinite": int(jnp.count_nonzero(~input_finite)),
            "nonconvex": int(jnp.count_nonzero(~convex)),
            "unbounded_like": int(jnp.count_nonzero(~bounded)),
            "unconverged": int(jnp.count_nonzero(~converged)),
        },
        max_projected_kkt=max_residual,
    )
