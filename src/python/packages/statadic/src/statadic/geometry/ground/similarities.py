# jaxtyping shape strings are runtime contracts.
"""ground similarities — ``k: X x X -> R`` on points of ``R^d``.

Position
--------
Consumes point arrays and produces a pointwise **Gram** matrix.

Sibling of :mod:`statadic.geometry.ground.metrics`, which produces distances. Both
compare points; Gram matrices are similarities rather than ``DMat`` distance
carriers. Declared similarity strategies carry their PSD or characteristic law.

PSD is not uniform across this module
--------------------------------------
:func:`linear_gram`, :func:`cosine_gram` and :func:`rbf_gram` are positive
semidefinite on every input. :func:`distance_induced_gram` is PSD **only where
the powered ground is of negative type**. Lean proves the Euclidean unit-power
instance as ``StatadicFormal.Geometry.distNegativeType_of_innerProductSpace``; the
shipped law descriptors additionally declare the literature-backed angular and
cosine guarantees through exponent
one and normalized Euclidean through exponents below two. Angular and cosine
have explicit CND counterexamples above one. Feeding an indefinite Gram to
kernel PCA or a QP is the failure :mod:`statadic.geometry.embedding` was written to
stop on the distance side, so the ground/exponent condition is carried by typed
energy/MMD construction rather than inferred from this raw Gram helper.

The reference point is explicit, and one reference must serve every block
--------------------------------------------------------------------------
``distance_induced_gram`` implements ``k(x, y) = d(x, z0) + d(y, z0) - d(x, y)``,
the kernel ``EnergyStatistics/MeanEmbedding.lean:318`` (``inner_energyPointEmbed``)
proves is an honest ``L2`` inner product. ``z0`` is a keyword argument rather
than an implicit origin because ``metric="angular"`` normalises its inputs, so
``d(x, 0)`` is **NaN**, not a large number: the origin is not merely a poor
reference on the sphere, it is not in the space at all.

The **Gram** moves with ``z0``; the MMD built from it does not, because the
``z0`` terms cancel against a total mass of zero (measured: relative deviation
``< 9e-13`` across ``z0`` in ``{0, 5, -20}``). That cancellation is why
:mod:`statadic.discrepancy.mmd` may pick the reference for the caller — **and why it
must pick the same one for all three blocks**. Writing
``mean(K_xx) + mean(K_yy) - 2 mean(K_xy)`` with references ``a``, ``b``, ``c``
leaves ``2(A_a - A_c) + 2(B_b - B_c)`` behind, which vanishes only when
``a = b = c``. A per-call default resolved inside this function would therefore
be silently wrong at the statistic; the default here is the origin, and the
statistic resolves its own shared reference before calling in.

Note the **unhalved** convention: the manuscript's ``(1/2)(d + d - d)`` kernel is
this one scaled by ``1/2``, which is where its ``sqrt(2)`` comes from. See
:mod:`statadic.discrepancy.mmd` for the constant that pairs with each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import Final

import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Float  # noqa: TC002

from statadic.core.axioms import (
    Law,  # runtime contract  # noqa: TC001
)
from statadic.core.domains import (
    DirectOrigin,
    FiniteValues,
    GovernedReferenceAuthority,
    L2UnitSphereSpace,
    MathematicalSpace,
    NonzeroNormalizationDomain,
    NonzeroVectorCarrierSpace,
    NormalizationPullbackOrigin,
    RawVectorSpace,
)
from statadic.core.transforms import (
    IdentityMap,
    L2Normalization,
    NonzeroInputRequired,
    TotalOnDeclaredSource,
    TransformContract,
)
from statadic.core.typing import Array, ScalarLike, static_jaxtyping  # noqa: TC001
from statadic.geometry.ground._similarity_strategy import (
    CharacteristicKernelLaw,
    CharacteristicKernelProperties,
    PositiveSemidefiniteKernelLaw,
    PositiveSemidefiniteKernelProperties,
    SimilarityFunction,
    SimilarityStrategy,
    declare_similarity,
)
from statadic.geometry.ground.config import (  # runtime annotations  # noqa: TC001
    GroundDistanceSelection,
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (  # runtime annotations  # noqa: TC001
    GroundDistance,
    cdist,
    squared_euclidean_distance,
)

__all__ = [
    "COSINE_KERNEL",
    "LINEAR_KERNEL",
    "CharacteristicKernelLaw",
    "CharacteristicKernelProperties",
    "PositiveSemidefiniteKernelLaw",
    "PositiveSemidefiniteKernelProperties",
    "RBFKernel",
    "RawIdentityTransform",
    "SimilarityFunction",
    "SimilarityStrategy",
    "cosine_gram",
    "declare_similarity",
    "distance_induced_gram",
    "gram",
    "linear_gram",
    "rbf_gram",
    "rbf_kernel",
]


@jit
@static_jaxtyping("n d m")
def linear_gram(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute the linear Gram matrix ``<x_i, y_j>``.

    PSD: unconditionally, with feature map ``phi(x) = x``.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.

    Returns:
        Array of shape (n, m) of pairwise inner products.

    Examples:
        >>> x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> linear_gram(x, x)
        Array([[1., 0.],
               [0., 1.]], dtype=float32)

    """
    return jnp.dot(x, y.T)


@jit
@static_jaxtyping("n d m")
def cosine_gram(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute the cosine Gram matrix ``<x_i, y_j> / (||x_i|| ||y_j||)``.

    PSD: unconditionally, with feature map ``phi(x) = x / ||x||``.

    Note that this normalizes **per row**, before any pooling. It is therefore
    *not* the prototype chord in
    :func:`statadic.discrepancy.mmd.cosine_mean_mmd`; that function consumes one
    already-normalized prototype per object and performs no normalization. On
    already-unit-norm rows this function coincides with :func:`linear_gram`.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.

    Returns:
        Array of shape (n, m) of pairwise cosine similarities. Zero-norm rows
        yield NaN, matching
        :func:`statadic.geometry.ground.metrics.normalized_euclidean_distance`.

    Examples:
        >>> x = jnp.array([[3.0, 4.0]])  # magnitude 5
        >>> y = jnp.array([[0.6, 0.8]])  # same direction, magnitude 1
        >>> bool(jnp.allclose(cosine_gram(x, y), 1.0))
        True

    """
    x_unit = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    y_unit = y / jnp.linalg.norm(y, axis=1, keepdims=True)
    return jnp.dot(x_unit, y_unit.T)


@jit
@static_jaxtyping("n d m")
def rbf_gram(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    *,
    bandwidth: ScalarLike,
) -> Float[Array, "n m"]:
    """Compute the Gaussian RBF Gram ``exp(-||x_i - y_j||^2 / (2 bandwidth^2))``.

    PSD: unconditionally, and characteristic on ``R^d`` for any positive
    bandwidth — which is why
    :data:`statadic.discrepancy.mmd.DECLARED_BRANDS` gives ``mmd[rbf]`` the
    ``NegativeType`` brand.

    ``bandwidth`` is **required**. Bandwidth selection, the median heuristic and
    MMD-Agg are out of scope for this module and carry their own bug history
    (``CHANGELOG.md:212``); a default here would hand callers an implicit choice
    that no battery gates.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.
        bandwidth: Kernel bandwidth (sigma). Must be positive. Traced, not
            static: this function is ``jit``-decorated, so the annotation admits
            a scalar tracer as well as a Python float.

    Returns:
        Array of shape (n, m) of pairwise RBF similarities in ``(0, 1]``.

    Examples:
        >>> x = jnp.array([[0.0, 0.0]])
        >>> rbf_gram(x, x, bandwidth=1.0)
        Array([[1.]], dtype=float32)

    """
    scale = jnp.asarray(bandwidth)
    if scale.ndim != 0 or not jnp.issubdtype(scale.dtype, jnp.number):
        message = "RBF bandwidth must be a real numeric scalar"
        raise TypeError(message)
    if jnp.issubdtype(scale.dtype, jnp.complexfloating):
        message = "RBF bandwidth must be a real numeric scalar"
        raise TypeError(message)
    squared = squared_euclidean_distance(x, y)
    valid = jnp.isfinite(scale) & (scale > 0)
    return jnp.where(valid, jnp.exp(-squared / (2.0 * scale**2)), jnp.nan)


@partial(jit, static_argnames=("exponent",))
@static_jaxtyping("n d m")
def _distance_induced_gram_core(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    *,
    metric: GroundDistance[Law],
    exponent: float = 1.0,
    reference: Float[Array, " d"] | None = None,
) -> Float[Array, "n m"]:
    """JIT core for :func:`distance_induced_gram` with a declared strategy."""
    origin = jnp.zeros((1, x.shape[-1]), dtype=x.dtype)
    ref = origin if reference is None else jnp.asarray(reference)[None, :]
    dxr = cdist(x, ref, metric=metric)[:, 0] ** exponent
    dyr = cdist(y, ref, metric=metric)[:, 0] ** exponent
    dxy = cdist(x, y, metric=metric) ** exponent
    return dxr[:, None] + dyr[None, :] - dxy


@static_jaxtyping("n d m")
def distance_induced_gram(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    *,
    metric: GroundDistanceSelection,
    exponent: float = 1.0,
    reference: Float[Array, " d"] | None = None,
) -> Float[Array, "n m"]:
    """Compute the distance-induced Gram ``d(x, z0) + d(y, z0) - d(x, y)``.

    This is the **unhalved** kernel of ``MeanEmbedding.lean:318``
    (``inner_energyPointEmbed``: ``<Phi x, Phi y> = ||x|| + ||y|| - dist x y``),
    the one whose MMD-squared equals ``energy_distance`` at constant **1**. The
    manuscript's half-normalized variant is this matrix times ``1/2``, and its
    constant is 2 — the source of the ``sqrt(2)`` in
    ``E = sqrt(2) * MMD``.

    PSD: **conditional**. It holds where the powered ground is of negative type,
    proven for the Euclidean unit-power case by
    ``StatadicFormal.Geometry.distNegativeType_of_innerProductSpace`` and
    declared by shipped law descriptors for their supported exponent regimes.
    This raw helper does not validate that pairing. Typed energy geometry and
    distance-induced kernel construction carry the required law.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.
        metric: Declared strategy or closed serialized built-in selection.
        exponent: Distance exponent alpha applied before the kernel algebra
            (static). Its negative-type domain depends on ``metric``; this raw
            helper does not check the pairing because the precondition belongs
            to the statistic — see :func:`statadic.discrepancy.mmd.energy.energy_distance`.
        reference: The point ``z0``, shape (d,). Defaults to the origin, which
            is correct for ``metric="euclidean"`` (it is then exactly the Lean
            kernel) and **invalid** for ``metric="angular"``, whose
            normalization excludes zero rows. Such metrics require an explicit
            reference point in their domain. The Gram depends on this choice;
            every MMD built from it does not, provided one reference serves every
            block.

    Returns:
        Array of shape (n, m) of pairwise kernel values.

    Examples:
        >>> x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        >>> distance_induced_gram(x, x, metric="euclidean")
        Array([[0., 0.],
               [0., 2.]], dtype=float32)

    """
    resolved_metric = resolve_ground_distance(metric)
    if reference is None and resolved_metric.requires_nonzero_rows:
        message = "reference is required when the ground distance excludes zero rows"
        raise ValueError(message)
    return _distance_induced_gram_core(
        x,
        y,
        metric=resolved_metric,
        exponent=exponent,
        reference=reference,
    )


type RawIdentityTransform = TransformContract[
    RawVectorSpace,
    RawVectorSpace,
    IdentityMap,
    TotalOnDeclaredSource,
    DirectOrigin,
]
type _NormalizationTransform = TransformContract[
    NonzeroVectorCarrierSpace,
    L2UnitSphereSpace,
    L2Normalization,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
]

_RAW_VECTOR_SPACE: RawVectorSpace = MathematicalSpace()
_NONZERO_VECTOR_SPACE: NonzeroVectorCarrierSpace = MathematicalSpace()
_RAW_IDENTITY: RawIdentityTransform = TransformContract()
_NORMALIZATION_PULLBACK: _NormalizationTransform = TransformContract()
_GOVERNED_REFERENCE = GovernedReferenceAuthority()
_PSD_KERNEL = PositiveSemidefiniteKernelProperties()
_CHARACTERISTIC_KERNEL = CharacteristicKernelProperties()


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class RBFKernel:
    """Characteristic Gaussian kernel with a dynamic scalar bandwidth leaf.

    Unlike the fixed configuration returned by :func:`rbf_kernel`, this callable
    can be passed through JIT, batching and differentiation with respect to its
    bandwidth. Nonpositive or nonfinite numerical bandwidths yield NaN; their
    characteristic-kernel law applies only on the positive finite domain.
    """

    bandwidth: Array | ScalarLike

    @property
    def name(self) -> str:
        """Diagnostic family name independent of traced parameter values."""
        return "rbf"

    @property
    def law(self) -> CharacteristicKernelProperties:
        """Characteristic law conditional on finite positive bandwidth."""
        return _CHARACTERISTIC_KERNEL

    @property
    def space(self) -> RawVectorSpace:
        """Finite-dimensional raw vector input space."""
        return _RAW_VECTOR_SPACE

    @property
    def transform(self) -> RawIdentityTransform:
        """Direct kernel construction on the declared input space."""
        return _RAW_IDENTITY

    @property
    def authority(self) -> GovernedReferenceAuthority:
        """Governed reference supporting the Gaussian kernel law."""
        return _GOVERNED_REFERENCE

    @property
    def required_evidence(self) -> type[FiniteValues]:
        """Finite input rows required by checked carrier consumers."""
        return FiniteValues

    @static_jaxtyping("n d m")
    def __call__(
        self,
        x: Float[Array, "n d"],
        y: Float[Array, "m d"],
        *,
        reference: Float[Array, " d"] | None = None,
    ) -> Float[Array, "n m"]:
        """Evaluate the Gram block, ignoring the optional shared reference."""
        del reference
        return rbf_gram(x, y, bandwidth=self.bandwidth)


LINEAR_KERNEL: Final[
    SimilarityStrategy[
        PositiveSemidefiniteKernelProperties,
        RawVectorSpace,
        RawIdentityTransform,
        GovernedReferenceAuthority,
        FiniteValues,
    ]
] = declare_similarity(
    linear_gram,
    law=_PSD_KERNEL,
    name="linear",
    space=_RAW_VECTOR_SPACE,
    transform=_RAW_IDENTITY,
    authority=_GOVERNED_REFERENCE,
    required_evidence=FiniteValues,
)
"""Declared linear kernel; PSD but not characteristic on probability laws."""

COSINE_KERNEL: Final[
    SimilarityStrategy[
        PositiveSemidefiniteKernelProperties,
        NonzeroVectorCarrierSpace,
        _NormalizationTransform,
        GovernedReferenceAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_similarity(
    cosine_gram,
    law=_PSD_KERNEL,
    name="cosine",
    space=_NONZERO_VECTOR_SPACE,
    transform=_NORMALIZATION_PULLBACK,
    authority=_GOVERNED_REFERENCE,
    required_evidence=NonzeroNormalizationDomain,
)
"""Declared cosine kernel; the normalization pullback remains explicit."""


def rbf_kernel(
    bandwidth: ScalarLike,
) -> SimilarityStrategy[
    CharacteristicKernelProperties,
    RawVectorSpace,
    RawIdentityTransform,
    GovernedReferenceAuthority,
    FiniteValues,
]:
    """Construct a characteristic RBF strategy with validated bandwidth."""
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, Real):
        message = f"RBF bandwidth must be finite and > 0, got {bandwidth!r}"
        raise TypeError(message)
    value = float(bandwidth)
    if not math.isfinite(value) or value <= 0.0:
        message = f"RBF bandwidth must be finite and > 0, got {bandwidth!r}"
        raise ValueError(message)
    return declare_similarity(
        partial(rbf_gram, bandwidth=value),
        law=_CHARACTERISTIC_KERNEL,
        name=f"rbf[bandwidth={value}]",
        space=_RAW_VECTOR_SPACE,
        transform=_RAW_IDENTITY,
        authority=_GOVERNED_REFERENCE,
        required_evidence=FiniteValues,
    )


@static_jaxtyping("n d m")
def gram(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw] = LINEAR_KERNEL,
    *,
    reference: Float[Array, " d"] | None = None,
) -> Float[Array, "n m"]:
    """Compute a Gram matrix through one declared kernel strategy.

    Dispatch is structural: the strategy is the executable kernel and carries
    the PSD/characteristic law, space, transform, authority, and evidence axes.

    Args:
        x: Array of shape (n, d) where each row is a sample.
        y: Array of shape (m, d) where each row is a sample.
        kernel: Declared PSD strategy. Third parties can use a fixed
            :func:`declare_similarity` declaration or a registered callable
            pytree. Use :func:`rbf_kernel` for fixed bandwidth configuration
            and :class:`RBFKernel` for dynamic parameters.
        reference: Shared reference used only by strategies that declare one.

    Notes:
        The finite-support relation between a generating kernel and energy is
        described at :ref:`formal-STATADIC-ENERGY-001`; see also
        :cite:p:`sejdinovic-equivalence-2013`. The result assumes the stated
        kernel properties and does not certify numerical evaluation.

    Returns:
        Array of shape (n, m) containing pairwise kernel values.

    Raises:
        TypeError: If ``kernel`` is not a declared strategy.

    Examples:
        >>> x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> gram(x, x, kernel=LINEAR_KERNEL)
        Array([[1., 0.],
               [0., 1.]], dtype=float32)

    """
    if not isinstance(kernel, SimilarityStrategy):
        message = "kernel must be a declared SimilarityStrategy"
        raise TypeError(message)
    return kernel(x, y, reference=reference)
