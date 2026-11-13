# ruff: noqa: F722  # jaxtyping numerical carrier contracts.
"""Preview affine GMM sandwich covariance from the actual retained fitting W.

One contribution is one retained time row. The fit's Bartlett long-run
variability divides every lag by n; the mean Jacobian and actual last solve
weight produce limiting covariance, then explicit division by n produces
parameter sampling covariance. Any finite-sample correction is separate.
Conventional covariance requires correct moment specification, suitable
dependence conditions, strong identification and an interior parameter.
Centering alone does not license misspecification-robust GMM inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Int  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    ParameterCovariance,
    VariabilityScheme,
    VariabilityT,
    _parameter_covariance_from_limiting,
    _symmetric_psd,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.gmm import (  # noqa: TC001
    AffineGMMEquation,
    AffineGMMFit,
)

__all__ = ["GMMUncertainty", "gmm_covariance"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GMMUncertainty(
    Generic[VariabilityT]  # noqa: UP046
):  # retain scheme variance
    """Actual-fit sampling covariance with distinct inference availability.

    Finite-budget fits may supply covariance despite an unconverged iterated
    weight sequence: the formula uses the actual weight of their last exact
    coefficient solve. This is not the efficient inverse-bread shortcut.
    Positive marginal variances are required here; joint restrictions retain
    their own covariance-rank checks. No reference distribution is selected.
    """

    fit: AffineGMMFit[VariabilityT]
    covariance: ParameterCovariance[AffineGMMEquation, VariabilityT]
    status: Int[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    method: str = field(metadata={"static": True})
    n_clusters: Int[Array, "*batch"]

    @property
    def sample_capacity(self) -> int:
        """Return fixed row capacity separately from retained contribution count."""
        return self.fit.scores.values.shape[-2]


def gmm_covariance[SchemeT: VariabilityScheme](
    fit: AffineGMMFit[SchemeT],
    *,
    finite_sample_correction: float | Array = 1.0,
) -> GMMUncertainty[SchemeT]:
    """Compose the full actual-W sandwich at the fit's final coefficient.

    This conventional covariance is justified by correctly specified moments,
    an interior strongly identified parameter, and the selected long-run
    estimator's regularity conditions. Iteration convergence is separately
    available in ``fit.solver``; no validity claim follows from a matching seed,
    a successful pseudoinverse or mean centering alone.
    """
    correction = jnp.asarray(finite_sample_correction)
    if correction.ndim != 0:
        message = "finite_sample_correction must be scalar within one fit."
        raise ValueError(message)
    omega = fit.variability.values
    safe_omega = jnp.where(jnp.isfinite(omega), omega, 0.0)
    omega_valid = jnp.all(jnp.isfinite(omega)) & _symmetric_psd(safe_omega)
    limiting = fit.moment_map @ safe_omega @ fit.moment_map.T
    limiting = (limiting + limiting.T) / 2
    covariance: ParameterCovariance[AffineGMMEquation, SchemeT] = (
        _parameter_covariance_from_limiting(
            limiting,
            fit.n_observations,
            correction,
            fit.weighted_rank,
            fit.identified,
            fit.valid & omega_valid & jnp.all(jnp.isfinite(fit.moment_map)),
        )
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
    return GMMUncertainty(
        fit=fit,
        covariance=covariance,
        status=jnp.asarray(status, dtype=jnp.int32),
        valid=valid,
        method="GMM-HAC",
        n_clusters=jnp.asarray(0, dtype=jnp.int32),
    )
