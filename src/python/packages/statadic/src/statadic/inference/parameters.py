# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Fit-aware linear restrictions and coefficient intervals.

Covariance construction, finite-sample correction and reference distribution
remain separate choices. These adapters consume one uncertainty object that
already owns its actual fit; they never refit parameters or combine an
independently supplied coefficient vector with a covariance. A selected
reference law remains the caller's statistical assumption, not a guarantee
created by numerical rank or a covariance method name.

Finite Student-t/F references beyond independently tested degrees of freedom
are flagged separately from numerical convergence and validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.scipy.stats import chi2
from jaxtyping import Bool, Float, Int  # noqa: TC002

from statadic.core.results import (
    EstimateResult,
    InferenceStatus,
    TestResult,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.inference._parameter_reference import (
    _MAX_VALIDATED_DEGREES,
    _f_survival,
    _outside_validated_range,
    _student_critical_value,
)
from statadic.inference.contracts import (
    FittedUncertainty,  # noqa: TC001
)

__all__ = [
    "CoefficientIntervals",
    "LinearRestrictionResult",
    "coefficient_intervals",
    "linear_restriction_test",
    "normal_critical_value",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearRestrictionResult:
    """A Wald restriction calculation and its explicitly selected calibration.

    ``test.statistic`` is the Wald quadratic for ``reference='chi2'`` and the
    quadratic divided by the number of restrictions for ``reference='f'``.
    ``wald_statistic`` preserves the undivided quadratic in both cases.
    ``numerator_df`` is the full restriction count; ``denominator_df`` is the
    explicitly supplied F degree of freedom, or infinity for chi-square.

    No pseudoinverse calibration is given to redundant restrictions or a
    singular restricted covariance. Invalid lanes expose NaN statistics and
    p-values together with ``valid=False`` and their reason in ``status``.
    The composed :class:`TestResult` has an empty null sample and zero resamples.
    """

    test: TestResult[Float[Array, "*batch"]]  # jaxtyping batch axes
    wald_statistic: Float[Array, "*batch"]
    restriction_rank: Int[Array, "*batch"]
    covariance_rank: Int[Array, "*batch"]
    numerator_df: Int[Array, "*batch"]
    denominator_df: Float[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]
    finite_reference_outside_validated_range: Bool[Array, "*batch"]
    reference: Literal["chi2", "f"] = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoefficientIntervals:
    """Coefficient estimates, standard errors and marginal symmetric intervals.

    The composed ``estimate.converged`` retains the fit's numerical convergence
    flag. ``valid`` separately states whether the covariance and chosen
    reference permit these intervals; a converged exact fit need not permit
    ordinary inference. ``degrees_of_freedom`` is explicitly supplied for a
    Student-t reference and infinity for a normal reference. Intervals are
    marginal, not a simultaneous confidence region.
    """

    estimate: EstimateResult[Float[Array, "*batch parameters"]]
    critical_value: Float[Array, "*batch"]
    degrees_of_freedom: Float[Array, "*batch"]
    confidence: Float[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]
    finite_reference_outside_validated_range: Bool[Array, "*batch"]
    reference: Literal["normal", "t"] = field(metadata={"static": True})
    parameter_names: tuple[str, ...] | None = field(metadata={"static": True})


def _batch_array(
    value: ArrayLike, batch_shape: tuple[int, ...], dtype: jnp.dtype, name: str
) -> Array:
    """Reject new batch axes before broadcasting a numeric reference argument."""
    array = jnp.asarray(value) + jnp.zeros((), dtype=dtype)
    if jnp.iscomplexobj(array):
        message = f"{name} must be real-valued."
        raise ValueError(message)
    if jnp.broadcast_shapes(array.shape, batch_shape) != batch_shape:
        message = f"{name} must be scalar or match the fit's batch axes."
        raise ValueError(message)
    return jnp.broadcast_to(array, batch_shape)


def _reference_degrees(
    degrees_of_freedom: ArrayLike | None,
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Keep a finite-reference degree choice independent from covariance names."""
    if required:
        if degrees_of_freedom is None:
            message = "degrees_of_freedom is required for the selected reference."
            raise ValueError(message)
        degrees = _batch_array(
            degrees_of_freedom, batch_shape, dtype, "degrees_of_freedom"
        )
        return degrees, jnp.isfinite(degrees) & (degrees > 0.0)
    if degrees_of_freedom is not None:
        message = "The selected reference does not use degrees_of_freedom."
        raise ValueError(message)
    return jnp.full(batch_shape, jnp.inf, dtype=dtype), jnp.ones(
        batch_shape, dtype=bool
    )


def _restriction_arrays(
    uncertainty: FittedUncertainty,
    restriction: ArrayLike,
    value: ArrayLike,
    parameter_names: tuple[str, ...] | None,
) -> tuple[Array, Array]:
    """Check positional width and, where available, exact parameter ordering."""
    fit = uncertainty.fit
    if fit.parameter_names is not None and parameter_names != fit.parameter_names:
        message = "Restriction parameter_names must exactly match the fit's order."
        raise ValueError(message)
    if fit.parameter_names is None and parameter_names is not None:
        message = "A positional fit cannot verify restriction parameter_names."
        raise ValueError(message)
    matrix = jnp.asarray(restriction)
    if jnp.iscomplexobj(matrix):
        message = "restriction must be real-valued."
        raise ValueError(message)
    matrix = matrix + jnp.zeros((), dtype=fit.parameters.dtype)
    expected_rank = 2
    if (
        matrix.ndim != expected_rank
        or matrix.shape[0] < 1
        or matrix.shape[1] != fit.parameters.shape[-1]
    ):
        message = "restriction must have shape (nonempty restrictions, parameters)."
        raise ValueError(message)
    target = jnp.asarray(value)
    if jnp.iscomplexobj(target):
        message = "The restriction value must be real-valued."
        raise ValueError(message)
    if target.ndim == 0:
        target = jnp.broadcast_to(target, (matrix.shape[0],))
    if target.shape != (matrix.shape[0],):
        message = "The restriction value must be scalar or have one entry per row."
        raise ValueError(message)
    return matrix, target


def _rank(matrix: Array) -> Array:
    """Use a declared dtype-relative SVD cutoff for restriction diagnostics."""
    cutoff = 10.0 * max(matrix.shape[-2:]) * jnp.finfo(matrix.dtype).eps
    return jnp.asarray(jnp.linalg.matrix_rank(matrix, rtol=cutoff), dtype=jnp.int32)


def _restricted_covariance(
    matrix: Array, covariance: Array
) -> tuple[Array, Array, Array]:
    """Return a safe solve matrix, actual rank and positive-definite validity."""
    restricted = jnp.einsum("qi,...ij,rj->...qr", matrix, covariance, matrix)
    finite = jnp.all(jnp.isfinite(restricted), axis=(-2, -1))
    safe = jnp.where(jnp.isfinite(restricted), restricted, 0.0)
    transpose = jnp.swapaxes(safe, -1, -2)
    tolerance = (
        10.0
        * safe.shape[-1]
        * jnp.finfo(safe.dtype).eps
        * jnp.max(jnp.abs(safe), axis=(-2, -1))
    )
    symmetric = jnp.max(jnp.abs(safe - transpose), axis=(-2, -1)) <= tolerance
    symmetrized = 0.5 * (safe + transpose)
    rank = _rank(symmetrized)
    valid = (
        finite
        & symmetric
        & (rank == safe.shape[-1])
        & (jnp.linalg.eigvalsh(symmetrized)[..., 0] > 0.0)
    )
    solve_matrix = jnp.where(
        valid[..., None, None], symmetrized, jnp.eye(safe.shape[-1], dtype=safe.dtype)
    )
    return solve_matrix, jnp.where(finite, rank, -1), valid


def _restriction_quadratic(
    uncertainty: FittedUncertainty, matrix: Array, target: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Compute the Wald quadratic only through a checked full-rank solve."""
    finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(target))
    safe_matrix = jnp.where(jnp.isfinite(matrix), matrix, 0.0)
    restriction_rank = _rank(safe_matrix)
    restriction_valid = finite & (restriction_rank == matrix.shape[0])
    restricted, covariance_rank, covariance_valid = _restricted_covariance(
        safe_matrix, uncertainty.covariance.values
    )
    delta = (
        jnp.einsum("qp,...p->...q", safe_matrix, uncertainty.fit.parameters) - target
    )
    quadratic = jnp.einsum(
        "...q,...q->...", delta, jnp.linalg.solve(restricted, delta[..., None])[..., 0]
    )
    return (
        quadratic,
        jnp.broadcast_to(jnp.where(finite, restriction_rank, -1), quadratic.shape),
        covariance_rank,
        restriction_valid,
        covariance_valid,
    )


def _wald_calibration(
    quadratic: Array,
    numerator_df: Array,
    denominator_df: Array,
    reference: Literal["chi2", "f"],
    reference_valid: Array,
) -> tuple[Array, Array]:
    """Keep the Wald quadratic distinct from its chosen chi-square or F scale."""
    if reference == "f":
        statistic = quadratic / numerator_df
        return statistic, _f_survival(
            statistic, numerator_df, jnp.where(reference_valid, denominator_df, 1.0)
        )
    if reference == "chi2":
        return quadratic, chi2.sf(quadratic, numerator_df)
    message = "reference must be 'chi2' or 'f'."
    raise ValueError(message)


def _availability(
    uncertainty: FittedUncertainty,
    numeric_valid: Array,
    covariance_valid: Array,
    reference_valid: Array,
    *,
    restriction_valid: Array | bool = True,
) -> tuple[Array, Array]:
    """Preserve upstream reasons; finite output alone cannot permit inference."""
    valid = (
        uncertainty.valid
        & restriction_valid
        & covariance_valid
        & reference_valid
        & numeric_valid
    )
    status = jnp.where(
        numeric_valid, InferenceStatus.AVAILABLE, InferenceStatus.NUMERICAL_FAILURE
    )
    status = jnp.where(covariance_valid, status, InferenceStatus.INVALID_COVARIANCE)
    status = jnp.where(restriction_valid, status, InferenceStatus.INVALID_RESTRICTION)
    status = jnp.where(reference_valid, status, InferenceStatus.INVALID_REFERENCE)
    return valid, jnp.asarray(
        jnp.where(uncertainty.valid, status, uncertainty.status), dtype=jnp.int32
    )


def linear_restriction_test(
    uncertainty: FittedUncertainty,
    restriction: ArrayLike,
    value: ArrayLike,
    *,
    parameter_names: tuple[str, ...] | None = None,
    reference: Literal["chi2", "f"],
    degrees_of_freedom: ArrayLike | None = None,
) -> LinearRestrictionResult:
    """Check and calibrate ``restriction @ beta = value`` without refitting.

    Named fits require ``parameter_names`` to exactly match the fitted order;
    positional fits require the correct matrix width. Both the restriction
    matrix and its induced covariance must have full row rank. ``reference``
    is mandatory. An F reference additionally requires an explicit finite,
    positive denominator degree of freedom, such as ``fit.df_resid`` when that
    law is justified. Robust covariance alone does not justify an exact F law.
    Finite-reference degrees beyond the independently tested range are flagged.

    Shape/alignment errors raise. Numeric failures retain per-lane unavailable
    status under eager execution, JIT and vmap.
    """
    matrix, target = _restriction_arrays(
        uncertainty, restriction, value, parameter_names
    )
    batch_shape = uncertainty.fit.parameters.shape[:-1]
    denominator_df, reference_valid = _reference_degrees(
        degrees_of_freedom,
        required=reference == "f",
        batch_shape=batch_shape,
        dtype=uncertainty.fit.parameters.dtype,
    )
    (
        quadratic,
        restriction_rank,
        covariance_rank,
        restriction_valid,
        covariance_valid,
    ) = _restriction_quadratic(uncertainty, matrix, target)
    numerator_df = jnp.full(batch_shape, matrix.shape[0], dtype=jnp.int32)
    statistic, pvalue = _wald_calibration(
        quadratic, numerator_df, denominator_df, reference, reference_valid
    )
    valid, status = _availability(
        uncertainty,
        jnp.isfinite(statistic)
        & (statistic >= 0.0)
        & jnp.isfinite(pvalue)
        & (pvalue >= 0.0)
        & (pvalue <= 1.0),
        covariance_valid,
        reference_valid,
        restriction_valid=restriction_valid,
    )
    return LinearRestrictionResult(
        test=TestResult(
            statistic=jnp.where(valid, statistic, jnp.nan),
            pvalue=jnp.where(valid, pvalue, jnp.nan),
            null_distribution=jnp.empty(
                (*batch_shape, 0), dtype=uncertainty.fit.parameters.dtype
            ),
            n_resamples=0,
        ),
        wald_statistic=jnp.where(valid, quadratic, jnp.nan),
        restriction_rank=restriction_rank,
        covariance_rank=covariance_rank,
        numerator_df=numerator_df,
        denominator_df=denominator_df,
        valid=valid,
        status=jnp.asarray(status, dtype=jnp.int32),
        finite_reference_outside_validated_range=(
            (numerator_df > _MAX_VALIDATED_DEGREES)
            | _outside_validated_range(denominator_df)
            if reference == "f"
            else jnp.zeros(batch_shape, dtype=bool)
        ),
        reference=reference,
    )


def normal_critical_value(confidence: ArrayLike) -> Array:
    """Return a two-sided normal critical value, or NaN outside ``(0, 1)``.

    Invert the lower tail to avoid rounding ``(1 + confidence) / 2`` to one
    at an otherwise representable confidence. Floating inputs retain their
    precision; nonfinite inversion remains unavailable.
    """
    confidence_array = jnp.asarray(confidence)
    confidence_array = confidence_array + 0.0
    if jnp.iscomplexobj(confidence_array):
        message = "confidence must be real-valued."
        raise ValueError(message)
    valid = (
        jnp.isfinite(confidence_array)
        & (confidence_array > 0.0)
        & (confidence_array < 1.0)
    )
    safe = jnp.where(valid, confidence_array, 0.95)
    critical = -ndtri((1.0 - safe) / 2.0)
    return jnp.where(valid & jnp.isfinite(critical), critical, jnp.nan)


def coefficient_intervals(  # noqa: PLR0914  # separate reference and quantity availability
    uncertainty: FittedUncertainty,
    *,
    confidence: ArrayLike = 0.95,
    reference: Literal["normal", "t"],
    degrees_of_freedom: ArrayLike | None = None,
) -> CoefficientIntervals:
    """Return marginal coefficient intervals for an explicitly selected reference.

    Student-t intervals require explicit denominator degrees of freedom.
    A normal reference accepts no degrees-of-freedom argument. Confidence must
    lie strictly inside ``(0, 1)``. All coefficient variances must be finite and
    positive; exact fits and unavailable covariance preserve their recorded
    status. Failed critical-value inversion produces unavailable intervals.

    The original coefficients and solver convergence remain visible in the
    composed estimate, independently of inference availability.
    Student-t degrees beyond the independently tested range are flagged.
    """
    if reference not in ("normal", "t"):
        message = "reference must be 'normal' or 't'."
        raise ValueError(message)
    fit = uncertainty.fit
    batch_shape = fit.parameters.shape[:-1]
    confidence_array = _batch_array(
        confidence, batch_shape, fit.parameters.dtype, "confidence"
    )
    degrees, reference_valid = _reference_degrees(
        degrees_of_freedom,
        required=reference == "t",
        batch_shape=batch_shape,
        dtype=fit.parameters.dtype,
    )
    reference_valid = (
        reference_valid
        & jnp.isfinite(confidence_array)
        & (confidence_array > 0.0)
        & (confidence_array < 1.0)
    )
    safe_confidence = jnp.where(reference_valid, confidence_array, 0.95)
    if reference == "t":
        critical, quantile_valid = _student_critical_value(
            safe_confidence, jnp.where(reference_valid, degrees, 1.0)
        )
    else:
        critical = normal_critical_value(safe_confidence)
        quantile_valid = jnp.isfinite(critical)
    variances = jnp.diagonal(uncertainty.covariance.values, axis1=-2, axis2=-1)
    variance_valid = jnp.all(jnp.isfinite(variances) & (variances > 0.0), axis=-1)
    standard_errors = jnp.sqrt(jnp.maximum(variances, 0.0))
    width = critical[..., None] * standard_errors
    lower, upper = fit.parameters - width, fit.parameters + width
    finite_bounds = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper), axis=-1)
    valid, status = _availability(
        uncertainty, quantile_valid & finite_bounds, variance_valid, reference_valid
    )
    return CoefficientIntervals(
        estimate=EstimateResult(
            estimate=fit.parameters,
            se=jnp.where(valid[..., None], standard_errors, jnp.nan),
            ci=(
                jnp.where(valid[..., None], lower, jnp.nan),
                jnp.where(valid[..., None], upper, jnp.nan),
            ),
            converged=fit.solver.converged,
            n_obs=uncertainty.sample_capacity,
        ),
        critical_value=jnp.where(valid, critical, jnp.nan),
        degrees_of_freedom=degrees,
        confidence=confidence_array,
        valid=valid,
        status=jnp.asarray(status, dtype=jnp.int32),
        finite_reference_outside_validated_range=(
            _outside_validated_range(degrees)
            if reference == "t"
            else jnp.zeros(batch_shape, dtype=bool)
        ),
        reference=reference,
        parameter_names=fit.parameter_names,
    )
