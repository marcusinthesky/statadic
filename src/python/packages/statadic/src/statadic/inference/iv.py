"""Actual-weight instrumental-variable sandwich and overidentification inference."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.scipy.stats import chi2

from statadic.core.equations import (
    ParameterCovariance,  # noqa: TC001
    VariabilityScheme,  # noqa: TC001
    _parameter_covariance_from_limiting,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.estimation.gmm import AffineGMMEquation  # noqa: TC001
from statadic.estimation.iv import IVFit  # noqa: TC001
from statadic.inference._row_order import _validated_row_order
from statadic.operators.longrun import newey_west_kernel
from statadic.operators.parameters import cluster_parameter_covariance

__all__ = ["IVHansenJ", "IVUncertainty", "hansen_j", "iv_covariance"]

_MIN_CLUSTERS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IVUncertainty:
    """Uncertainty tied to the exact IV fit and requested meat."""

    fit: IVFit
    covariance: ParameterCovariance[AffineGMMEquation, VariabilityScheme]
    status: Array
    valid: Array
    method: str = field(metadata={"static": True})
    n_clusters: Array

    @property
    def sample_capacity(self) -> int:
        """Return the retained observation count."""
        return self.fit.outcome.shape[0]


def iv_covariance(
    fit: "IVFit",
    *,
    kind: str = "HC0",
    bandwidth: int = 0,
    row_order: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    n_clusters: int = 0,
    finite_sample_correction: float = 1.0,
) -> IVUncertainty:
    """Full sandwich using retained W and the explicitly requested meat."""
    n, p = fit.design.shape
    scores = fit.affine.scores.values
    if row_order is not None and kind != "HAC":
        message = "row_order applies only to HAC covariance."
        raise ValueError(message)
    if kind == "model":
        omega = (fit.residuals @ fit.residuals / n) * (
            fit.instruments.T @ fit.instruments / n
        )
    elif kind in {"HC0", "HC1"}:
        omega = scores.T @ scores / n
        if kind == "HC1":
            finite_sample_correction *= n / (n - p) if n > p else float("nan")
    elif kind == "HAC":
        omega = _hac_variability(scores, bandwidth, row_order)
    elif kind == "cluster":
        if (
            groups is None
            or isinstance(n_clusters, bool)
            or not isinstance(n_clusters, int)
            or n_clusters < _MIN_CLUSTERS
        ):
            msg = "Cluster covariance requires at least two clusters"
            raise ValueError(msg)
        groups = jnp.asarray(groups)
        if groups.shape != (n,) or not jnp.issubdtype(groups.dtype, jnp.integer):
            message = "Cluster codes must be aligned integers"
            raise ValueError(message)
        valid_codes = jnp.all((groups >= 0) & (groups < n_clusters))
        safe = jnp.clip(groups, 0, n_clusters - 1)
        occupied = jnp.bincount(safe, length=n_clusters)
        omega = cluster_parameter_covariance(scores, safe, n_clusters=n_clusters) / n
        omega = jnp.where(valid_codes & jnp.all(occupied > 0), omega, jnp.nan)
    else:
        msg = "Unsupported IV covariance"
        raise ValueError(msg)
    limiting = fit.affine.moment_map @ omega @ fit.affine.moment_map.T
    limiting = (limiting + limiting.T) / 2
    covariance = _parameter_covariance_from_limiting(
        limiting,
        fit.affine.n_observations,
        jnp.asarray(finite_sample_correction),
        fit.affine.weighted_rank,
        fit.affine.identified,
        fit.valid & jnp.all(jnp.isfinite(omega)),
    )
    valid = (
        fit.valid
        & covariance.numerically_valid
        & jnp.all(jnp.diag(covariance.values) > 0)
    )
    status = jnp.where(
        valid, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return IVUncertainty(fit, covariance, status, valid, kind, jnp.asarray(n_clusters))


def _hac_variability(
    scores: Array, bandwidth: int, row_order: ArrayLike | None
) -> Array:
    """Apply Bartlett variability after validating the complete score-row order."""
    if (
        isinstance(bandwidth, bool)
        or not isinstance(bandwidth, int)
        or not 0 <= bandwidth < scores.shape[0]
    ):
        message = "Invalid HAC bandwidth"
        raise ValueError(message)
    order, valid = _validated_row_order(row_order, scores.shape[0])
    omega = newey_west_kernel(scores[order], bandwidth)
    return jnp.where(valid, omega, jnp.nan)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IVHansenJ:
    """Conventional overidentification test and weight provenance."""

    statistic: Array
    pvalue: Array
    degrees_of_freedom: int = field(metadata={"static": True})
    valid: Array
    provenance: str = field(metadata={"static": True})


def hansen_j(fit: "IVFit", *, assume_correct_specification: bool) -> IVHansenJ:
    """Conventional J only for estimated efficient weights and explicit assumptions."""
    df = fit.instruments.shape[1] - fit.design.shape[1]
    available = (
        fit.valid
        & (df > 0)
        & assume_correct_specification
        & (fit.method in {"two-step", "iterated"})
    )
    omega = fit.preliminary_variability
    symmetric = (omega + omega.T) / 2
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    positive = jnp.all(jnp.isfinite(eigenvalues)) & (
        eigenvalues[0] > jnp.finfo(omega.dtype).eps * eigenvalues[-1] * omega.shape[0]
    )
    available = available & positive
    if fit.method == "iterated":
        omega = fit.final_variability
        symmetric = (omega + omega.T) / 2
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        positive = jnp.all(jnp.isfinite(eigenvalues)) & (
            eigenvalues[0]
            > jnp.finfo(omega.dtype).eps * eigenvalues[-1] * omega.shape[0]
        )
        error = jnp.linalg.norm(
            fit.affine.weight @ omega - jnp.eye(omega.shape[0])
        ) / jnp.sqrt(omega.shape[0])
        available = (
            available
            & positive
            & fit.converged
            & (error <= jnp.sqrt(jnp.finfo(omega.dtype).eps))
        )
    stat = fit.outcome.shape[0] * fit.affine.criterion
    return IVHansenJ(
        jnp.where(available, stat, jnp.nan),
        jnp.where(available, chi2.sf(stat, max(df, 1)), jnp.nan),
        df,
        available,
        fit.method,
    )
