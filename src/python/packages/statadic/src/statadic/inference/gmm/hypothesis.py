# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""GMM equality-restriction tests: Wald and Hansen J.

The Wald uses plain Newey-West; the legacy ``J`` defaults to Hall centered
variability. Centering records a variability convention, and does not itself
license a misspecified or profiled GMM reference distribution. Conventional
Hansen-J calibration requires correct moment specification and efficient
interior GMM estimation with strong identification. The preview
:func:`hansen_j_from_fit` checks the actual fit's convergence and final weight.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import chi2
from jaxtyping import Bool, Float, Int  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    VariabilityT,
)
from statadic.core.results import (
    InferenceStatus,
    TestResult,
)
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
    Static,
    static_jaxtyping,
)
from statadic.estimation.gmm import (  # noqa: TC001
    AffineGMMFit,
)
from statadic.estimation.moments import _validate_moment_panel
from statadic.inference._common import (
    as_index,
)
from statadic.operators.longrun import (
    hall_centered_hac_kernel,
    newey_west_kernel,
    optimal_bandwidth,
)

__all__ = [
    "GMMHansenJResult",
    "HansenJKernelResult",
    "HansenJResult",
    "WaldKernelResult",
    "WaldResult",
    "gmm_wald",
    "gmm_wald_kernel",
    "hansen_j",
    "hansen_j_from_fit",
    "hansen_j_kernel",
]


class WaldKernelResult(NamedTuple):
    """Fixed-structure JAX result for a GMM Wald statistic."""

    stat: Float[Array, ""]
    pvalue: Float[Array, ""]
    gbar: Float[Array, " m"]
    covariance: Float[Array, "m m"]


class HansenJKernelResult(NamedTuple):
    """Fixed-structure JAX result for a Hansen-J statistic."""

    stat: Float[Array, ""]
    pvalue: Float[Array, ""]
    covariance: Float[Array, "m m"]


@static_jaxtyping("t m")
@partial(jax.jit, static_argnames=("bandwidth",))
def gmm_wald_kernel(
    g: Float[Array, "t m"],
    bandwidth: Static[int],
) -> WaldKernelResult:
    """Evaluate the complete Wald statistic and χ² tail on the JAX graph."""
    moments = jnp.asarray(g)
    n_observations, n_moments = moments.shape
    gbar = jnp.mean(moments, axis=0)
    covariance = newey_west_kernel(moments, bandwidth)
    inverse = jnp.linalg.pinv(covariance)
    stat = n_observations * gbar @ inverse @ gbar
    return WaldKernelResult(
        stat=stat,
        pvalue=chi2.sf(stat, n_moments),
        gbar=gbar,
        covariance=covariance,
    )


@static_jaxtyping("t m")
@partial(jax.jit, static_argnames=("n_params", "bandwidth", "centered"))
def hansen_j_kernel(
    g: Float[Array, "t m"],
    n_params: Static[int],
    bandwidth: Static[int],
    *,
    centered: Static[bool],
) -> HansenJKernelResult:
    """Evaluate a Hansen-J statistic and χ² tail on the JAX graph."""
    moments = jnp.asarray(g)
    n_observations, n_moments = moments.shape
    covariance = (
        hall_centered_hac_kernel(moments, bandwidth)
        if centered
        else newey_west_kernel(moments, bandwidth)
    )
    gbar = jnp.mean(moments, axis=0)
    stat = n_observations * gbar @ jnp.linalg.pinv(covariance) @ gbar
    return HansenJKernelResult(
        stat=stat,
        pvalue=chi2.sf(stat, n_moments - n_params),
        covariance=covariance,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GMMHansenJResult:
    """Fit-derived conventional Hansen J with explicit availability evidence.

    Exactly identified, fixed arbitrary-weight and unconverged/non-efficient
    fits retain unavailable NaN statistics and p-values. ``test`` composes the
    existing analytic-test result with an empty resampling distribution.
    """

    test: TestResult[Float[Array, "*batch"]]
    degrees_of_freedom: Int[Array, "*batch"]
    status: Int[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    overidentified: Bool[Array, "*batch"]
    efficient_weight: Bool[Array, "*batch"]
    conventional_assumptions: Bool[Array, "*batch"]


def hansen_j_from_fit(
    fit: AffineGMMFit[VariabilityT],
    *,
    assume_correct_specification: bool,
) -> GMMHansenJResult:
    """Calibrate an efficiently converged overidentified fit as chi-square.

    The explicit acknowledgement covers correct moment specification, a
    strongly identified interior true parameter, a valid dependence/bandwidth
    asymptotic regime and the conventional GMM reference. These assumptions
    cannot be established from arrays. Neither centered variability nor an
    arbitrary stationary weighted criterion establishes them.

    In addition to coefficient convergence, ``||W Omega-I||_F/||I||_F`` must
    be at most ``sqrt(machine epsilon)`` at the final coefficient, and Omega
    must be positive definite/full rank. Supplied fixed weights are rejected
    even if numerically close: their provenance does not establish efficient
    estimation. Fixed supplied moments remain a separate testing contract.
    """
    count, moments = fit.scores.values.shape
    width = fit.parameters.shape[0]
    degrees = moments - width
    omega = fit.variability.values
    safe_omega = jnp.where(jnp.isfinite(omega), omega, 0.0)
    full_rank = jnp.linalg.matrix_rank(safe_omega) == moments
    positive = jnp.all(jnp.linalg.eigvalsh((safe_omega + safe_omega.T) / 2) > 0)
    efficient = (
        ~jnp.asarray(fit.fixed_weight)
        & fit.solver.converged
        & (fit.weight_fixed_point_error <= jnp.sqrt(jnp.finfo(omega.dtype).eps))
    )
    valid = (
        fit.valid
        & (degrees > 0)
        & efficient
        & full_rank
        & positive
        & jnp.asarray(assume_correct_specification)
    )
    gbar = jnp.mean(fit.scores.values, axis=0)
    statistic = count * gbar @ fit.effective_weight @ gbar
    pvalue = chi2.sf(statistic, max(degrees, 1))
    valid = valid & jnp.isfinite(statistic) & jnp.isfinite(pvalue)
    status = jnp.where(
        valid, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_REFERENCE
    )
    status = jnp.where(full_rank & positive, status, InferenceStatus.INVALID_COVARIANCE)
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return GMMHansenJResult(
        test=TestResult(
            statistic=jnp.where(valid, statistic, jnp.nan),
            pvalue=jnp.where(valid, pvalue, jnp.nan),
            null_distribution=jnp.empty((0,), dtype=fit.parameters.dtype),
            n_resamples=0,
        ),
        degrees_of_freedom=jnp.asarray(degrees, dtype=jnp.int32),
        status=jnp.asarray(status, dtype=jnp.int32),
        valid=valid,
        overidentified=jnp.asarray(degrees > 0),
        efficient_weight=efficient,
        conventional_assumptions=jnp.asarray(assume_correct_specification),
    )


def _resolved_bandwidth(bandwidth: int | None, n_observations: int) -> int:
    """Resolve and validate the static HAC bandwidth for a numerical kernel."""
    resolved = optimal_bandwidth(n_observations) if bandwidth is None else bandwidth
    index = as_index(resolved)
    if index is None or index < 0 or index >= n_observations:
        message = "bandwidth must be an integer in [0, n_observations)."
        raise ValueError(message)
    return index


class WaldResult(NamedTuple):
    """Return type of :func:`gmm_wald`.

    Attributes:
        stat: Wald statistic ``T·ḡᵀ Ŝ⁻¹ ḡ``.
        dof: Degrees of freedom (number of moments ``m``).
        pvalue: Upper-tail χ²(dof) p-value.
        gbar: Sample-mean moment vector, shape ``(m,)``.

    """

    stat: float
    dof: int
    pvalue: float
    gbar: Float[Array, " m"]


def gmm_wald(g: ArrayLike, bandwidth: int | None = None) -> WaldResult:
    """GMM Wald test of ``H0: E[g_t] = 0`` with HAC weighting.

    Uses the plain Newey-West HAC (moments are mean-zero under the null for a
    Wald test).  Statistic ``W = T · ḡᵀ Ŝ⁻¹ ḡ ~ χ²(m)`` under H0.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        bandwidth: HAC truncation lag; ``None`` → data-driven.

    Returns:
        :class:`WaldResult`.

    """
    moments, n_observations, m = _validate_hansen_dimensions(g, n_params=0)
    resolved_bandwidth = _resolved_bandwidth(bandwidth, n_observations)
    result = gmm_wald_kernel(jnp.asarray(moments), resolved_bandwidth)
    covariance = result.covariance
    _require_full_rank_moment_covariance(covariance, m, statistic="GMM Wald")
    return WaldResult(
        stat=float(result.stat),
        dof=m,
        pvalue=float(result.pvalue),
        gbar=result.gbar,
    )


class HansenJResult(NamedTuple):
    """Return type of :func:`hansen_j`.

    Attributes:
        stat: Hansen ``J`` statistic.
        dof: Over-identifying degrees of freedom (``m - k``).
        pvalue: Upper-tail χ²(dof) p-value.
        centered: Whether Hall centering was applied.

    """

    stat: float
    dof: int
    pvalue: float
    centered: bool


@static_jaxtyping("t m")
def _validate_hansen_dimensions(
    g: ArrayLike,
    n_params: int,
) -> tuple[Float[Array, "t m"], int, int]:
    """Validate the moment panel and over-identification dimensions.

    The panel half is :func:`statadic.estimation.moments._validate_moment_panel`;
    only the over-identification pair below is specific to Hansen J.
    """
    moments, n_observations, m = _validate_moment_panel(g)
    if n_params < 0:
        message = "n_params must be nonnegative."
        raise ValueError(message)
    if m <= n_params:
        message = (
            "Hansen J requires over-identification: the number of moments "
            f"m={m} must exceed n_params={n_params}."
        )
        raise ValueError(message)
    return moments, n_observations, m


def _require_full_rank_moment_covariance(
    covariance: ArrayLike,
    m: int,
    *,
    statistic: str,
) -> None:
    """Fail closed when a chi-square reference has fewer than `m` directions."""
    covariance_rank = int(jnp.linalg.matrix_rank(jnp.asarray(covariance)))
    if covariance_rank < m:
        message = (
            f"{statistic} requires a full-rank moment covariance; "
            f"observed rank={covariance_rank} for m={m}."
        )
        raise ValueError(message)


def hansen_j(
    g: ArrayLike,
    n_params: int = 0,
    bandwidth: int | None = None,
    *,
    centered: bool = True,
    estimated_by_gmm: bool = False,
) -> HansenJResult:
    """Hansen ``J`` over-identification test with Hall centered-HAC weighting.

    ``J = n_observations · ḡᵀ Ŝ⁻¹ ḡ`` where ``Ŝ`` is the Hall centered HAC when
    ``centered=True``, or the plain Newey-West HAC when ``centered=False``
    (the existing comparator). ``J ~ χ²(m - k)`` under correct specification
    only when the ``k`` parameters were estimated by the corresponding GMM
    criterion.  Merely evaluating moments at an external estimate and
    subtracting ``k`` does not have that reference distribution, so this
    function rejects that call unless ``estimated_by_gmm=True``. Mean centering
    alone does not establish misspecification-robust calibration. This legacy
    acknowledgement cannot inspect the fit; prefer ``hansen_j_from_fit`` for
    the new retained-fit API.

    Args:
        g: Moment process, shape ``(n_observations, m)``.
        n_params: Number of estimated parameters ``k`` (e.g. 1 when κ is
            estimated).  Over-identifying dof is ``m - k``.
        bandwidth: HAC truncation lag; ``None`` → data-driven.
        centered: Use Hall centering (default ``True``).
        estimated_by_gmm: Explicit confirmation that any ``n_params > 0``
            parameters were estimated with the same GMM moments/criterion.
            Prefer :func:`hansen_j_from_fit` for estimated affine moments.

    Returns:
        :class:`HansenJResult`.

    """
    moments, n_observations, m = _validate_hansen_dimensions(g, n_params)
    if n_params > 0 and not estimated_by_gmm:
        message = (
            "Hansen J with n_params > 0 requires parameters estimated by the "
            "same GMM criterion; use hansen_j_from_fit for affine moments or pass "
            "estimated_by_gmm=True only for an already-fitted GMM estimate."
        )
        raise ValueError(message)
    resolved_bandwidth = _resolved_bandwidth(bandwidth, n_observations)
    result = hansen_j_kernel(
        jnp.asarray(moments),
        n_params,
        resolved_bandwidth,
        centered=centered,
    )
    covariance = result.covariance
    _require_full_rank_moment_covariance(covariance, m, statistic="Hansen J")
    dof = m - n_params
    return HansenJResult(
        stat=float(result.stat),
        dof=dof,
        pvalue=float(result.pvalue),
        centered=centered,
    )
