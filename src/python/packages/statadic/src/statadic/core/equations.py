"""Estimating-equation contracts — ``psi``, its Jacobian, and its variability.

What this module is for
-----------------------
:mod:`statadic.core.matrices` carries what a *dissimilarity* is and
:mod:`statadic.core.statistics` carries what a *number* is. Neither says what an
**estimator** is, and an estimator is identified by the equation it solves
rather than by the model family it belongs to. OLS, a Poisson GLM, GEE, 2SLS,
quantile regression and M-estimation differ only in ``psi``; everything
downstream of ``psi`` is shared::

    psi     the estimating function, evaluated per observation   (T, m)
    J       its Jacobian in the parameters — the *bread*         (T, m, k)
    Omega   the variability matrix of psi — the *meat*           (m, m)

Writing that triple once lets :func:`sandwich_covariance` share covariance
composition across model families. An estimating equation alone establishes
neither a likelihood-ratio statistic nor a valid score-test reference
distribution; those need additional model-specific assumptions and contracts.

Shared contracts, separate producers
-------------------------------------
The contracts name equations and variability without importing a concrete
statistical family. Estimation retains fitted contributions and derivatives;
:mod:`statadic.operators.longrun` constructs variability; inference composes them
under explicit assumptions. Shared contracts let those producers depend on the
same vocabulary without reversing the dependency DAG. The analogous matrix
boundary joins :mod:`statadic.geometry.ground` to discrepancy and geometry consumers.

Omega's scheme is a soundness fact, not a label
------------------------------------------------
:mod:`statadic.estimation.gmm` retains the selected variability scheme. Legacy
fixed-moment tests in :mod:`statadic.inference.gmm.hypothesis` also expose a
``centered: Static[bool]`` option. Plain Newey-West retains the raw moment
cross-products; Hall's (1987) mean-centered form subtracts the sample moment
mean first. A boolean cannot express that a particular consumer requires one
form rather than the other. Neither centering choice alone establishes valid
profiled GMM inference under arbitrary misspecification.

:class:`PlainLongRun` and :class:`MeanCenteredLongRun` are therefore nominal
**siblings** under :class:`VariabilityScheme`, neither usable where the other is
required — exactly as :class:`statadic.core.statistics.VStatisticEstimator` and
:class:`statadic.core.statistics.UStatisticEstimator` are siblings, and for the same
reason: a consumer that type-checks against one having been handed the other is
silently wrong with no static symptom.

Nothing semantic is a runtime leaf
----------------------------------
The axis markers are plain empty classes and the descriptors are zero-field
frozen dataclasses, exactly as in :mod:`statadic.core.domains`. Equation and
limiting-covariance carriers have one array leaf and a static producer token.
:class:`ParameterCovariance` additionally carries numerical diagnostics and
explicit normalization as array leaves, so batches may have different retained
observation counts without changing compilation keys.

Limits of the contract
-----------------------
Typed fits and variability producers carry nominal schemes, while legacy
array-taking test kernels retain their established interfaces. The producer
token restricts construction; it does not prove that supplied values implement
the claimed estimating equation or that separately built components share a
sample. Fit-attached inference retains those alignments explicitly.

Only :class:`MomentCondition` currently names a produced equation family.
Additional families should be introduced with a real producer. No current path
reads persisted estimating contributions or variability back into a checked
carrier; adding such a door requires its own value and provenance contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar, final

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int, Shaped  # noqa: TC002

from statadic.core.domains import OriginT  # runtime: `EquationContract` Generic base
from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001

__all__ = [
    "EquationContract",
    "EquationDescriptor",
    "EquationJacobian",
    "EquationKindT",
    "EquationT",
    "EstimatingEquationKind",
    "EstimatingFunction",
    "IndependentRows",
    "MeanCenteredLongRun",
    "MeanEquationJacobian",
    "MomentCondition",
    "OneWayCluster",
    "ParameterCovariance",
    "PlainLongRun",
    "SandwichCovariance",
    "VariabilityMatrix",
    "VariabilityScheme",
    "VariabilityT",
    "mean_equation_jacobian",
    "parameter_covariance",
    "sandwich_covariance",
    "unsafe_assume_equation_jacobian",
    "unsafe_assume_estimating_function",
    "unsafe_assume_mean_equation_jacobian",
    "unsafe_assume_variability_matrix",
]


# --- The equation axis. Which equation `psi` is, never how it is solved. ---


class EstimatingEquationKind:
    """Root of the estimating-equation axis; names no equation.

    Deliberately *not* an estimator axis. ``2SLS``, ``LIML`` and ``IV-GMM``
    solve the same moment condition by different criteria, so the estimator is a
    fact about the algorithm, while what travels through the sandwich is the
    equation. :class:`statadic.core.statistics.EstimatorScheme` carries the former.
    """


class MomentCondition(EstimatingEquationKind):
    """``E[g(Z, theta)] = 0`` — a moment restriction, possibly overidentified.

    The only inhabitant with a producer in statadic today (``inference/gmm``,
    ``gel``, ``gms``, ``wolak``). Overidentification is a property of the
    ``(m, k)`` shape rather than a separate marker: ``m > k`` is visible on the
    Jacobian and is what gives the Hansen ``J`` its ``m - k`` degrees of
    freedom.
    """


# --- The variability scheme axis. How Omega was formed; a soundness fact. ---


class VariabilityScheme:
    """Root of the variability-scheme axis; the uninformative scheme.

    The widest brand, and therefore what any future checked door must return: a
    persisted ``(m, m)`` array shows squareness, finiteness and symmetry, and
    none of those recover whether the sample mean was subtracted before the
    autocovariances were formed. A result at this brand satisfies neither
    :class:`PlainLongRun` nor :class:`MeanCenteredLongRun`.
    """


@final
class PlainLongRun(VariabilityScheme):
    """Long-run covariance of ``psi`` formed **without** mean centering.

    Correct where the estimating function is mean-zero by construction — a Wald
    statistic on a correctly specified moment condition. Produced by
    :func:`statadic.operators.longrun.newey_west_variability`.
    """


@final
class MeanCenteredLongRun(VariabilityScheme):
    """Hall (1987) long-run covariance, formed after subtracting ``gbar``.

    An unrelated **sibling** of :class:`PlainLongRun`, not a refinement of it.
    Targets the long-run variance of ``g_t - E[g_t]`` even when
    ``E[g_t] != 0``, under the applicable dependence and bandwidth assumptions.
    The brand records that mean centering occurred; it does not certify a
    profiled Hansen ``J`` or universal misspecification-robust GMM inference.
    Produced by
    :func:`statadic.operators.longrun.hall_centered_variability`.
    """


@final
class IndependentRows(VariabilityScheme):
    """Independent-row parameter variability, with correction recorded separately."""


@final
class OneWayCluster(VariabilityScheme):
    """One-way independent-cluster variability from summed row influences."""


# --- Axis parameters. Variance names the flow it blocks; phantom otherwise. ---

#: **Covariant** equation kind. Explicit ``TypeVar`` for the reason given in
#: :mod:`statadic.core.domains`: these are phantom parameters, so PEP 695 has
#: nothing to infer variance from and settles on invariant.
EquationKindT = TypeVar(  # noqa: PLC0105
    "EquationKindT",
    bound=EstimatingEquationKind,
    covariant=True,
)

#: **Covariant** variability scheme. Blocks: a plain long-run covariance
#: supplied where a mean-centered one is required, and conversely. Named
#: ``VariabilityT`` rather than ``SchemeT`` because
#: :mod:`statadic.core.matrices` already binds that name to the *energy* estimation
#: scheme, and the two axes are unrelated.
VariabilityT = TypeVar(  # noqa: PLC0105
    "VariabilityT",
    bound="VariabilityScheme",
    covariant=True,
)


# --- The descriptor. Many axes here, so the carriers can take one token. ---


@dataclass(frozen=True, slots=True)
class EquationDescriptor:
    """Non-generic root for a complete static estimating-equation descriptor."""


class EquationContract(
    EquationDescriptor,
    Generic[EquationKindT, OriginT],  # noqa: UP046  # PEP 695 loses variance
):
    """Zero-field descriptor tying an equation kind to its construction origin.

    Descriptors carry axes; **carriers carry descriptors**. A call site names
    the combination once with a ``type`` alias and then spells one token, the
    same discipline :data:`statadic.core.transforms.L2NormalizationContract` and
    :data:`statadic.core.domains.L2UnitSphereSpace` already use. That is what keeps
    the carriers below at one and two parameters while the number of semantic
    axes grows.
    """


#: **Invariant** equation descriptor: composition must not widen distinct
#: equations to their common root merely to pair a bread and a meat.
EquationT = TypeVar(
    "EquationT",
    bound=EquationDescriptor,
)


# --- The carriers. One array leaf each; every semantic axis is phantom. ---


@final
@dataclass(frozen=True, slots=True)
class _ProducerToken:
    """Private constructor token carried as static pytree metadata.

    Deliberately a second token rather than an import of
    :mod:`statadic.core.matrices`'s: a distance-matrix producer must not be able to
    mint an estimating function, and keeping this module a leaf of the intra-
    ``core`` import order is what lets the fence rank rank-0 modules at all.
    """


_PRODUCER_TOKEN = _ProducerToken()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EstimatingFunction(Generic[EquationT]):  # noqa: UP046  # PEP 695 loses variance
    """``psi`` evaluated at every observation — the estimating function itself.

    Attributes:
        values: ``psi(Z_t, theta)`` per observation, shape
            ``(observations, moments)`` — or ``(*batch, observations, moments)``
            once a transform has batched it. Rows are observations and columns
            are equation components, matching the ``g`` argument
            :mod:`statadic.inference.gmm` already takes.

    Examples:
        >>> import jax.numpy as jnp
        >>> psi = unsafe_assume_estimating_function(jnp.zeros((8, 3)))
        >>> psi.values.shape
        (8, 3)

    """

    values: Float[Array, "*batch observations moments"]  # noqa: F722  # jaxtyping shape
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EquationJacobian(Generic[EquationT]):  # noqa: UP046  # PEP 695 loses variance
    """``J`` — the bread: the derivative of ``psi`` in the parameters.

    Retains the per-observation panel when it is available. Covariance
    composition takes its mean. Producers with an aggregate mean derivative
    can instead provide :class:`MeanEquationJacobian`, without constructing a
    redundant observation panel.

    Attributes:
        values: ``d psi / d theta'`` per observation, shape
            ``(observations, moments, parameters)`` — or with a leading
            ``*batch``. Affine GMM retains these rows when supplied, while
            producers such as native IV retain only their aggregate derivative.

    """

    values: Float[
        Array,
        "*batch observations moments parameters",  # noqa: F722  # jaxtyping shape
    ]
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MeanEquationJacobian(Generic[EquationT]):  # noqa: UP046
    """Mean-equation derivative ``J = T^-1 sum_t d psi_t / d theta'``.

    ``values`` has shape ``(*batch, moments, parameters)``. This is a distinct
    carrier from the derivative panel so a leading batch axis cannot be
    mistaken for an observation axis. It contains neither a sum derivative
    nor an observation count; the caller supplies the retained count explicitly
    to :func:`parameter_covariance`.
    """

    values: Float[Array, "*batch moments parameters"]  # noqa: F722
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VariabilityMatrix(
    Generic[EquationT, VariabilityT],  # noqa: UP046  # PEP 695 loses variance
):
    """``Omega`` — the meat: the variability matrix of ``psi``.

    The scheme is the load-bearing parameter; see the module docstring. HAC,
    cluster-robust, two-way, Driscoll-Kraay and spatial-robust forms are all
    *meat variants* and belong on this carrier at their own scheme brand, not on
    separate containers.

    Attributes:
        values: ``Omega``, shape ``(moments, moments)`` — or with a leading
            ``*batch``.

    Examples:
        >>> import jax.numpy as jnp
        >>> meat = unsafe_assume_variability_matrix(jnp.eye(3))
        >>> meat.values.shape
        (3, 3)

    """

    values: Float[Array, "*batch moments moments"]  # noqa: F722  # jaxtyping shape
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SandwichCovariance(
    Generic[EquationT, VariabilityT],  # noqa: UP046  # PEP 695 loses variance
):
    """Limiting covariance of ``sqrt(T) * (theta_hat - theta)``.

    A **sibling** of :class:`VariabilityMatrix`, never a subtype: this is
    ``(parameters, parameters)`` and that is ``(moments, moments)``, and only the
    just-identified case makes them the same shape. Keeping the scheme means a
    consumer can demand that a reported standard error came from the centered
    meat.

    Attributes:
        values: Limiting covariance, shape ``(parameters, parameters)`` — or
            with a leading ``*batch``. No division by ``T`` and no finite-sample
            correction has been applied. See :class:`ParameterCovariance` for
            the explicitly scaled sampling-covariance approximation.

    """

    values: Float[Array, "*batch parameters parameters"]  # noqa: F722  # jaxtyping
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParameterCovariance(
    Generic[EquationT, VariabilityT],  # noqa: UP046
):
    """Sampling-covariance approximation with explicit scaling and diagnostics.

    ``values = limiting_covariance * finite_sample_correction / n_observations``.
    The count is the retained number of contribution units, never a generic
    effective sample size. For the HAC providers that unit is one time row.

    ``rank`` is the producer's identification rank. Generic equation composition
    ranks ``J' W J`` at JAX's default pseudoinverse cutoff; linear-fit producers
    use the directly factored design's SVD rank retained in their fit. Fit-bound
    affine GMM uses the equilibrated weighted design rank from its coefficient
    solve. Those rank conventions need not agree on an ill-conditioned design.
    ``identified`` requires full parameter rank. ``numerically_valid`` checks
    finite inputs/output, symmetric positive-semidefinite composition and valid
    scaling. Invalid compositions return NaN values, including rank failures.

    These flags establish numerical composition, not statistical validity.
    With explicit ``W`` a singular positive-semidefinite meat is allowed, and
    the resulting covariance may be singular or zero. Positive variances,
    restriction rank, dependence assumptions and reference distributions still
    require their own inference checks. No evaluation-point or retained-row
    identity is inferred from a static descriptor or a matching count.
    """

    values: Float[Array, "*batch parameters parameters"]  # noqa: F722
    n_observations: Int[Array, "*batch"]  # noqa: F722
    finite_sample_correction: Float[Array, "*batch"]  # noqa: F722
    rank: Int[Array, "*batch"]  # noqa: F722
    identified: Bool[Array, "*batch"]  # noqa: F722
    numerically_valid: Bool[Array, "*batch"]  # noqa: F722
    _token: _ProducerToken = field(repr=False, metadata={"static": True})


# --- Doors. One private producer for in-package code; loud external doors. ---


@static_jaxtyping("*batch moments")
def _variability_matrix[
    EquationA: EquationDescriptor,
    VariabilityA: VariabilityScheme,
](
    values: Float[Array, "*batch moments moments"],  # noqa: F722  # jaxtyping shape
) -> VariabilityMatrix[EquationA, VariabilityA]:
    """Construct ``Omega`` from a producer that knows which scheme it formed.

    The private door in-package meat producers use, so a producer writes its
    scheme once, in its return type. :mod:`statadic.operators.longrun` is the only
    caller today.

    Args:
        values: The variability matrix the producer computed.

    Returns:
        The matrix branded with the equation and scheme the call site requires.

    """
    return VariabilityMatrix(values=values, _token=_PRODUCER_TOKEN)


@static_jaxtyping("*batch parameters")
def _parameter_covariance_from_limiting[
    EquationA: EquationDescriptor,
    VariabilityA: VariabilityScheme,
](
    limiting: Float[Array, "*batch parameters parameters"],  # noqa: F722
    n_observations: Int[Array, "*batch"],  # noqa: F722
    finite_sample_correction: Float[Array, "*batch"],  # noqa: F722
    rank: Int[Array, "*batch"],  # noqa: F722
    identified: Bool[Array, "*batch"],  # noqa: F722
    numerically_valid: Bool[Array, "*batch"],  # noqa: F722
) -> ParameterCovariance[EquationA, VariabilityA]:
    """Scale a producer's limiting covariance without refactoring normal equations.

    The producer owns contribution units, fitted-state alignment and its rank
    evidence. This door checks scaling, finite symmetric PSD output and full
    reported rank; it never replaces a stable QR influence map with ``J' W J``.
    """
    if (
        limiting.ndim < 2  # noqa: PLR2004  # matrix axes
        or limiting.shape[-1] == 0
        or limiting.shape[-2] != limiting.shape[-1]
    ):
        message = "Parameter covariance needs nonempty square parameter axes."
        raise ValueError(message)
    if not jnp.issubdtype(n_observations.dtype, jnp.integer):
        message = "n_observations must be an integer retained contribution count."
        raise ValueError(message)
    batch_shape = limiting.shape[:-2]
    diagnostics = (
        n_observations,
        finite_sample_correction,
        rank,
        identified,
        numerically_valid,
    )
    if (
        jnp.broadcast_shapes(batch_shape, *(value.shape for value in diagnostics))
        != batch_shape
    ):
        message = "Covariance diagnostics must match the covariance batch axes."
        raise ValueError(message)
    count, correction, rank, identified, valid = (
        jnp.broadcast_to(value, batch_shape) for value in diagnostics
    )
    identified = identified & (rank == limiting.shape[-1])
    values = limiting * (correction / jnp.maximum(count, 1))[..., None, None]
    finite = jnp.all(jnp.isfinite(limiting), axis=(-2, -1))
    safe_limiting = jnp.where(jnp.isfinite(limiting), limiting, 0.0)
    valid = (
        valid
        & identified
        & (count > 0)
        & jnp.isfinite(correction)
        & (correction > 0)
        & finite
        & _symmetric_psd(safe_limiting)
        & jnp.all(jnp.isfinite(values), axis=(-2, -1))
    )
    return ParameterCovariance(
        values=jnp.where(valid[..., None, None], values, jnp.nan),
        n_observations=count,
        finite_sample_correction=correction,
        rank=rank,
        identified=identified,
        numerically_valid=valid,
        _token=_PRODUCER_TOKEN,
    )


@static_jaxtyping("*batch observations moments")
def unsafe_assume_estimating_function[EquationA: EquationDescriptor](
    values: Float[Array, "*batch observations moments"],  # noqa: F722  # jaxtyping
) -> EstimatingFunction[EquationA]:
    """Assert an equation descriptor that nothing in this call verifies.

    The name is the review trigger, as for
    :func:`statadic.core.matrices.unsafe_assume_dmat`. Every use is a claim the
    *caller* owes evidence for.

    Args:
        values: The ``(observations, moments)`` array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return EstimatingFunction(values=values, _token=_PRODUCER_TOKEN)


@static_jaxtyping("*batch observations moments parameters")
def unsafe_assume_equation_jacobian[EquationA: EquationDescriptor](
    values: Float[
        Array,
        "*batch observations moments parameters",  # noqa: F722  # jaxtyping
    ],
) -> EquationJacobian[EquationA]:
    """Assert an equation descriptor for a derivative panel, unverified.

    Args:
        values: The ``(observations, moments, parameters)`` array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return EquationJacobian(values=values, _token=_PRODUCER_TOKEN)


@static_jaxtyping("*batch moments parameters")
def unsafe_assume_mean_equation_jacobian[EquationA: EquationDescriptor](
    values: Float[Array, "*batch moments parameters"],  # noqa: F722
) -> MeanEquationJacobian[EquationA]:
    """Assert a descriptor and **mean**, rather than sum, derivative convention.

    This door verifies neither the declared equation nor alignment, retained
    rows, parameter ordering, or the parameter at which the derivative was
    evaluated. The caller owes those facts, as for the panel door.

    Args:
        values: Aggregate mean derivative, shape ``(*batch, moments, parameters)``.

    Returns:
        The branded mean derivative.

    """
    return MeanEquationJacobian(values=values, _token=_PRODUCER_TOKEN)


def mean_equation_jacobian[EquationA: EquationDescriptor](
    panel: EquationJacobian[EquationA],
) -> MeanEquationJacobian[EquationA]:
    """Average a derivative panel over its observation axis exactly once.

    Args:
        panel: Per-observation derivatives at one common evaluation point.

    Returns:
        Mean-equation Jacobian with the same equation descriptor.

    """
    if panel.values.ndim < 3 or panel.values.shape[-3] == 0:  # noqa: PLR2004
        message = "A derivative panel needs at least one observation."
        raise ValueError(message)
    return MeanEquationJacobian(
        values=jnp.mean(panel.values, axis=-3), _token=_PRODUCER_TOKEN
    )


@static_jaxtyping("*batch moments")
def unsafe_assume_variability_matrix[
    EquationA: EquationDescriptor,
    VariabilityA: VariabilityScheme,
](
    values: Float[Array, "*batch moments moments"],  # noqa: F722  # jaxtyping shape
) -> VariabilityMatrix[EquationA, VariabilityA]:
    """Assert an equation and a **scheme** that nothing in this call verifies.

    The scheme is the dangerous half: no inspection of a finished ``(m, m)``
    array recovers whether the sample mean was subtracted before the
    autocovariances were formed, so this door is the only place that claim can
    be written without a producer standing behind it.

    Args:
        values: The ``(moments, moments)`` array to brand.

    Returns:
        The array wrapped and branded exactly as the call site declares.

    """
    return VariabilityMatrix(values=values, _token=_PRODUCER_TOKEN)


# --- The composition rule. Written once; this is the whole point of the waist. ---


@static_jaxtyping("*batch moments")
def sandwich_covariance[EquationA: EquationDescriptor, VariabilityA: VariabilityScheme](
    bread: EquationJacobian[EquationA] | MeanEquationJacobian[EquationA],
    meat: VariabilityMatrix[EquationA, VariabilityA],
    weight: Float[Array, "*batch moments moments"] | None = None,  # noqa: F722
) -> SandwichCovariance[EquationA, VariabilityA]:
    """Combine mean-equation bread/meat into a **limiting** covariance.

    With ``J`` the mean Jacobian and ``W`` the weighting matrix,

    ``Avar = (J' W J)^-1 J' W Omega W J (J' W J)^-1``

    which collapses to the efficient ``(J' Omega^-1 J)^-1`` at ``W = Omega^-1``
    and to the classical ``J^-1 Omega J^-T`` when the equation is just
    identified. Passing ``weight=None`` selects the efficient form directly
    rather than computing the general expression at ``W = pinv(Omega)``.

    There is no division by observation count and no finite-sample correction:
    this is the covariance of ``sqrt(T) * (theta_hat - theta)`` under the
    declared mean-equation convention. The legacy pseudoinverse arithmetic is
    preserved, including finite results for singular inputs; those results do
    not authorize ordinary inference. :func:`parameter_covariance` adds
    explicit sampling scale and numerical diagnostics.

    Generic estimating-equation covariance composition uses this rule. Concrete
    linear-fit providers retain stable QR influence maps and share the
    :class:`ParameterCovariance` output contract without refactoring normal
    equations. HAC,
    cluster-robust, two-way, Driscoll-Kraay and spatial-robust covariance are
    *meat variants* rather than per-family code. Note what the signature keeps
    and what it refuses:

    * ``EquationA`` is shared, so a bread and a meat from unrelated equations
      have no common descriptor below :class:`EquationDescriptor` to unify on;
    * ``VariabilityA`` **propagates into the result**, so a consumer written
      against ``SandwichCovariance[..., MeanCenteredLongRun]`` rejects a
      standard error built from the plain long-run meat. That rejection is the
      static replacement for ``inference/gmm.py``'s ``centered`` boolean.

    Shape checks are static and survive tracing. Descriptors are invariant so
    unrelated equations cannot unify through descriptor widening in checked
    callers. They do not establish dataset/evaluation-point identity or
    parameter labels; unsafe construction leaves those obligations with the
    producer. No host value comparison occurs.

    Args:
        bread: Per-observation derivatives, averaged here, or an aggregate
            **mean** derivative with the same parameter order as the estimate.
        meat: The variability matrix ``Omega`` at its declared scheme.
        weight: The GMM weighting matrix ``W``. ``None`` selects the efficient
            weighting ``W = Omega^-1``.

    Returns:
        Limiting covariance, branded with the meat's scheme.

    Examples:
        >>> import jax.numpy as jnp
        >>> panel = jnp.tile(jnp.eye(2)[None], (5, 1, 1))
        >>> bread = unsafe_assume_equation_jacobian(panel)
        >>> meat = unsafe_assume_variability_matrix(jnp.eye(2) * 4.0)
        >>> bool(jnp.allclose(sandwich_covariance(bread, meat).values, 4 * jnp.eye(2)))
        True

    """
    jacobian = (
        mean_equation_jacobian(bread).values
        if isinstance(bread, EquationJacobian)
        else bread.values
    )
    omega = meat.values
    _check_composition_shapes(jacobian, omega, weight)
    transposed = jnp.swapaxes(jacobian, -1, -2)
    if weight is None:
        normal = transposed @ jnp.linalg.pinv(omega) @ jacobian
        covariance = jnp.linalg.pinv(normal)
    else:
        normal = jnp.linalg.pinv(transposed @ weight @ jacobian)
        middle = transposed @ weight @ omega @ weight @ jacobian
        covariance = normal @ middle @ normal
    symmetric = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    return SandwichCovariance(values=symmetric, _token=_PRODUCER_TOKEN)


def _check_composition_shapes(
    jacobian: Array, omega: Array, weight: Array | None
) -> None:
    """Reject mismatched moment/parameter axes without comparing traced values."""
    if jacobian.ndim < 2 or min(jacobian.shape[-2:]) == 0:  # noqa: PLR2004
        message = "A mean Jacobian needs nonempty moment and parameter axes."
        raise ValueError(message)
    moments = jacobian.shape[-2]
    if omega.ndim < 2 or omega.shape[-2:] != (moments, moments):  # noqa: PLR2004
        message = "Meat dimensions must match the Jacobian's moment axis."
        raise ValueError(message)
    shapes = [jacobian.shape[:-2], omega.shape[:-2]]
    if weight is not None:
        if weight.ndim < 2 or weight.shape[-2:] != (moments, moments):  # noqa: PLR2004
            message = "Weight dimensions must match the Jacobian's moment axis."
            raise ValueError(message)
        shapes.append(weight.shape[:-2])
    jnp.broadcast_shapes(*shapes)


@static_jaxtyping("*batch m n k")
def _pinv_rank(matrix: Shaped[Array, "*batch m n"]) -> Int[Array, "*batch"]:  # noqa: F722
    """Count singular values retained by the existing JAX pinv convention."""
    singular_values: Float[Array, "*batch k"] = jnp.linalg.svd(  # noqa: F722
        matrix, compute_uv=False
    )
    relative_cutoff = 10.0 * max(matrix.shape[-2:]) * jnp.finfo(matrix.dtype).eps
    return jnp.sum(
        singular_values > relative_cutoff * singular_values[..., :1], axis=-1
    )


@static_jaxtyping("*batch n")
def _symmetric_psd(matrix: Shaped[Array, "*batch n n"]) -> Bool[Array, "*batch"]:  # noqa: F722
    """Allow roundoff-scale symmetry/PSD noise without accepting indefiniteness."""
    transposed: Shaped[Array, "*batch n n"] = jnp.swapaxes(  # noqa: F722
        matrix, -1, -2
    )
    scale = jnp.max(jnp.abs(matrix), axis=(-2, -1))
    tolerance = 10.0 * matrix.shape[-1] * jnp.finfo(matrix.dtype).eps * scale
    symmetric = jnp.max(jnp.abs(matrix - transposed), axis=(-2, -1)) <= tolerance
    eigenvalues: Float[Array, "*batch n"] = jnp.linalg.eigvalsh(  # noqa: F722
        0.5 * (matrix + transposed)
    )
    return symmetric & (eigenvalues[..., 0] >= -tolerance)


@static_jaxtyping("*batch")
def _limiting_covariance_diagnostics(
    jacobian: Array, omega: Array, weight: Array | None
) -> tuple[Array, Int[Array, "*batch"], Bool[Array, "*batch"], Bool[Array, "*batch"]]:  # noqa: F722
    """Compose safe numeric arrays and retain pre-sanitization validity evidence."""
    finite_jacobian = jnp.all(jnp.isfinite(jacobian), axis=(-2, -1))
    finite_omega = jnp.all(jnp.isfinite(omega), axis=(-2, -1))
    safe_jacobian = jnp.where(jnp.isfinite(jacobian), jacobian, 0.0)
    safe_omega = jnp.where(jnp.isfinite(omega), omega, 0.0)
    safe_bread: MeanEquationJacobian[EquationDescriptor] = (
        unsafe_assume_mean_equation_jacobian(safe_jacobian)
    )
    safe_meat: VariabilityMatrix[EquationDescriptor, VariabilityScheme] = (
        unsafe_assume_variability_matrix(safe_omega)
    )
    valid_matrices = finite_jacobian & finite_omega & _symmetric_psd(safe_omega)
    if weight is None:
        effective_weight = jnp.linalg.pinv(safe_omega)
        valid_matrices = valid_matrices & (_pinv_rank(safe_omega) == omega.shape[-1])
        limiting = sandwich_covariance(safe_bread, safe_meat).values
    else:
        effective_weight = jnp.where(jnp.isfinite(weight), weight, 0.0)
        valid_matrices = (
            valid_matrices
            & jnp.all(jnp.isfinite(weight), axis=(-2, -1))
            & _symmetric_psd(effective_weight)
        )
        limiting = sandwich_covariance(safe_bread, safe_meat, effective_weight).values
    normal = jnp.swapaxes(safe_jacobian, -1, -2) @ effective_weight @ safe_jacobian
    rank = _pinv_rank(normal)
    identified = finite_jacobian & (rank == jacobian.shape[-1])
    return limiting, rank, identified, valid_matrices


@static_jaxtyping("*batch moments")
def parameter_covariance[
    EquationA: EquationDescriptor,
    VariabilityA: VariabilityScheme,
](
    bread: EquationJacobian[EquationA] | MeanEquationJacobian[EquationA],
    meat: VariabilityMatrix[EquationA, VariabilityA],
    n_observations: int | Int[Array, "*batch"],  # noqa: F722
    weight: Float[Array, "*batch moments moments"] | None = None,  # noqa: F722
    *,
    finite_sample_correction: float | Float[Array, "*batch"] = 1.0,  # noqa: F722
) -> ParameterCovariance[EquationA, VariabilityA]:
    """Scale mean-equation limiting covariance by the retained contribution count.

    ``Omega`` must estimate the variability of the square-root-count scaled
    mean equation. For row contributions and the HAC producers in
    :mod:`statadic.operators.longrun`, this is the long-run covariance of one row,
    formed with denominator ``T`` at every lag. The result is
    ``sandwich_covariance(...).values / T * finite_sample_correction``.

    Supply the **actual fitting weight** for a non-efficient GMM solve. Omitting
    it explicitly chooses efficient weighting and requires nonsingular meat.
    A symmetric PSD explicit weight may be singular only when ``J' W J`` still
    identifies every parameter. There is no generic misspecification-robust GMM
    validity claim.

    Complete derivative panels require ``n_observations`` to equal their row
    count. Aggregate producers must establish retained-row alignment, parameter
    order and a common evaluation parameter themselves; matching dimensions or
    a static descriptor cannot establish those facts. The fit-aware providers
    in :mod:`statadic.inference.linear_covariance` consume the actual
    :class:`statadic.estimation.linear.LinearFit` that owns its complete retained
    arrays. Do not pair newly prepared scores with stale masks.

    Shape/layout errors raise ``ValueError``. Invalid numeric inputs (including
    nonpositive counts/corrections) produce NaNs and false validity flags under
    eager execution, JIT and batching. Precision remains caller-owned.

    Args:
        bread: Panel or aggregate mean-equation Jacobian.
        meat: Contribution-unit variability at its declared scheme.
        n_observations: Positive integer retained count, scalar or batch-shaped.
        weight: Actual symmetric PSD fitting weight, or efficient weighting.
        finite_sample_correction: Positive multiplier, separate from count scaling.

    Returns:
        Sampling covariance with numerical identification/validity diagnostics.

    """
    jacobian = (
        mean_equation_jacobian(bread).values
        if isinstance(bread, EquationJacobian)
        else bread.values
    )
    omega = meat.values
    _check_composition_shapes(jacobian, omega, weight)
    limiting, rank, identified, valid_matrices = _limiting_covariance_diagnostics(
        jacobian, omega, weight
    )
    count = jnp.asarray(n_observations)
    if not jnp.issubdtype(count.dtype, jnp.integer):
        message = "n_observations must be an integer retained contribution count."
        raise ValueError(message)
    correction = jnp.asarray(finite_sample_correction, dtype=limiting.dtype)
    batch_shape = limiting.shape[:-2]
    if jnp.broadcast_shapes(count.shape, correction.shape, batch_shape) != batch_shape:
        message = "Count and correction must be scalar or match covariance batch axes."
        raise ValueError(message)
    count = jnp.broadcast_to(count, batch_shape)
    correction = jnp.broadcast_to(correction, batch_shape)
    if isinstance(bread, EquationJacobian):
        valid_matrices = valid_matrices & (count == bread.values.shape[-3])
    return _parameter_covariance_from_limiting(
        limiting, count, correction, rank, identified, valid_matrices
    )
