# jaxtyping shape strings are runtime contracts.
"""ground metrics — ``d: X x X -> R`` on points of ``R^d``.

Position
--------
Consumes point arrays and produces a pointwise distance matrix.

What each callable guarantees is a **declared property**, not part of its name
(t46 taxonomy): see :data:`DECLARED_AXIOMS` below and
:mod:`statadic.core.axioms` for why preconditions carry flags rather than alias
names.

Three of the four dissimilarities here normalise before comparing
-----------------------------------------------------------------
``angular_distance``, ``cosine_distance`` and ``normalized_euclidean_distance``
all divide by ``||x||``, so ``d(x, 2x) == 0`` and the **identity of
indiscernibles fails on R^d** — measured by t46.1's battery, not assumed. Their
``DMat`` brand is therefore :class:`~statadic.core.axioms.Premetric`, not
``Metric``; anything downstream that reads ``d(x, y) == 0`` as ``x == y``
(deduplication, an injective embedding, "same distribution") is unsound on
scale-equivalent inputs and should pass ``metric="euclidean"``.

Domain handoff (T55.3)
----------------------
The law descriptors below are conditional on finite rows, and the three
normalizing strategies additionally require every row to be nonzero. The
current ``GroundDistance`` call signature carries array shape and law but not
that value-domain evidence: zero rows intentionally propagate ``NaN``. T55.3
owns the finite/nonzero point-cloud and raw-vector/direction-quotient carrier
types. Until then, a declared law must not be read as a proof of totality over
all values admitted by ``Float[Array, "n d"]``.

Why jaxtyping is imported at runtime here
-----------------------------------------
Under ``if TYPE_CHECKING:`` the names are unresolvable when beartype evaluates
the annotation, and beartype **skips** rather than fails — so a guarded shape
string is decoration in both planes (pyrefly does not verify shape algebra
either). See :mod:`statadic.core.typing`, which re-exports the aliases so no stage
imports ``jaxtyping`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Float  # noqa: TC002

from statadic.core.axioms import (
    METRIC,
    NEGATIVE_TYPE,
    PSEUDOMETRIC,
    SEMIMETRIC,
    STRONG_NEGATIVE_TYPE,
    Axioms,
    Law,
    LawProperties,
    Metric,
    NegativeType,
    Premetric,
    Semimetric,
)
from statadic.core.domains import (
    ClaimAuthority,
    DeclaredAuthority,
    DirectOrigin,
    EvidenceOrigin,
    FiniteValues,
    GovernedReferenceAuthority,
    L2UnitSphereSpace,
    MathematicalSpace,
    NonzeroNormalizationDomain,
    NonzeroVectorCarrierSpace,
    NormalizationPullbackOrigin,
    ObjectDomain,
    PositiveRaySpace,
    ProjectiveSpace,
    RawVectorSpace,
    SpaceDescriptor,
    UnitNormSupport,
    ValueEvidence,
)
from statadic.core.matrices import _space_dmat
from statadic.core.transforms import (
    IdentityMap,
    L2Normalization,
    NonzeroInputRequired,
    PositiveRayQuotient,
    ProjectiveQuotient,
    TotalOnDeclaredSource,
    TransformContract,
    TransformDescriptor,
)
from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.geometry.ground._strategy import (
    GroundDistance,
    GroundDistanceFunction,
    GroundLawDeclaration,
    declare_ground_distance,
)

if TYPE_CHECKING:
    from statadic.core.matrices import MatrixContract, SpaceDMat
    from statadic.sample.evidence import CheckedClouds

__all__ = [
    "ANGULAR",
    "COSINE",
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "EUCLIDEAN",
    "NORMALIZED_EUCLIDEAN",
    "PROJECTIVE_ANGULAR",
    "RAY_ANGULAR",
    "RAY_CHORD",
    "SPHERE_ANGULAR",
    "SPHERE_CHORD",
    "AngularLaw",
    "CosineLaw",
    "EuclideanLaw",
    "GroundDistance",
    "GroundDistanceFunction",
    "GroundLawDeclaration",
    "NormalizedEuclideanLaw",
    "ProjectiveAngularLaw",
    "SphereAngularLaw",
    "angular_distance",
    "cdist",
    "checked_cloud_pdist",
    "cosine_distance",
    "declare_ground_distance",
    "euclidean_distance",
    "normalized_euclidean_distance",
    "pdist",
    "projective_angular_distance",
    "sphere_angular_distance",
    "sphere_chord_distance",
    "squared_euclidean_distance",
]


@dataclass(frozen=True, slots=True)
class EuclideanLaw:
    """Literature-declared Euclidean law on finite-dimensional vectors.

    Lean proves finite Euclidean energy nonnegativity in
    :ref:`formal-STATADIC-ENERGY-003` (ordinary negative type). Strong negative
    type, and hence measure separation, follows from Lyons (2013) for separable
    Hilbert spaces but does not yet have a Euclidean/Hilbert instance in the
    Lean development.
    """

    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=STRONG_NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)
    strong_negative_type: Literal[True] = field(default=True, init=False)
    subunit_power_strong_on_quotient: Literal[True] = field(default=True, init=False)
    full_fractional_power_negative_type: Literal[True] = field(default=True, init=False)
    full_fractional_power_strong: Literal[True] = field(default=True, init=False)
    squared_distance_anova: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class AngularLaw:
    """Negative-type pseudometric law for direction-only angular distance."""

    axioms: Axioms = field(default=PSEUDOMETRIC, init=False)
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)
    subunit_power_strong_on_quotient: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class CosineLaw:
    """Reflexive symmetric negative-type law for cosine dissimilarity."""

    axioms: Axioms = field(
        default=Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY,
        init=False,
    )
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)
    subunit_power_strong_on_quotient: Literal[True] = field(default=True, init=False)
    unit_power_mean_only: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class NormalizedEuclideanLaw:
    """Negative-type pseudometric law for normalized Euclidean distance."""

    axioms: Axioms = field(default=PSEUDOMETRIC, init=False)
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)
    subunit_power_strong_on_quotient: Literal[True] = field(default=True, init=False)
    full_fractional_power_negative_type: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class SphereAngularLaw:
    """Metric negative-type angular law on the L2 unit sphere."""

    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class ProjectiveAngularLaw:
    """Metric law of acute angular distance on real projective space.

    No negative-type capability is declared: metricity and conditional
    negative definiteness are independent facts, and statadic has no governed
    authority establishing the latter in arbitrary dimension.
    """

    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=LawProperties(0), init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)


@jit
@static_jaxtyping("n d m")
def angular_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute pairwise angular distance (in radians) between row vectors.

    Angular distance is the angle between vectors measured in radians, computed
    as arccos of the cosine similarity. It ranges from 0 (identical direction)
    to π (opposite directions).

    Axioms: declares ``PSEUDOMETRIC`` (``NONNEGATIVE | SYMMETRY | TRIANGLE``);
    brand :class:`~statadic.core.axioms.Premetric`. ``IDENTITY`` does **not** hold —
    the vectors are normalised first, so ``d(x, 2x) == 0``.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.

    Returns:
        Array of shape (n, m) containing pairwise angular distances in radians.

    Notes:
        Zero vectors return NaN distances as they have no defined direction.
        The ground and its assumptions matter for energy comparisons; see
        :ref:`formal-STATADIC-ENERGY-001`.

    Examples:
        >>> x = jnp.array([[1.0, 0.0]])
        >>> y = jnp.array([[0.0, 1.0]])  # Orthogonal
        >>> angular_distance(x, y)
        Array([[1.5707964]], dtype=float32)  # π/2 radians

    """
    # Compute norms
    x_norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    y_norms = jnp.linalg.norm(y, axis=1, keepdims=True)

    # Normalize vectors
    x_norm = x / x_norms
    y_norm = y / y_norms

    # Compute cosine similarity
    cos_sim = jnp.dot(x_norm, y_norm.T)

    # Clip to valid range for arccos due to numerical errors
    cos_sim = jnp.clip(cos_sim, -1.0, 1.0)

    # Return angle in radians
    return jnp.arccos(cos_sim)


@jit
@static_jaxtyping("n d m")
def cosine_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute pairwise cosine distance between row vectors.

    Cosine distance is defined as 1 - cosine_similarity. It ranges from 0
    (identical direction) to 2 (opposite directions).

    Axioms: declares ``NONNEGATIVE | SYMMETRY`` only; brand
    :class:`~statadic.core.axioms.Premetric`. It fails ``TRIANGLE`` (witness
    ``x=(1,0), y=(1,1), z=(0,1)``) and ``IDENTITY`` (normalisation). It *is* of
    negative type — a condition the linear marker chain cannot express, so it
    travels through :func:`statadic.testing.axioms.assert_negative_type`.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.

    Returns:
        Array of shape (n, m) containing pairwise cosine distances.

    Notes:
        Zero vectors return NaN distances as they have no defined direction.

    Examples:
        >>> x = jnp.array([[1.0, 0.0]])
        >>> y = jnp.array([[0.0, 1.0]])  # Orthogonal
        >>> cosine_distance(x, y)
        Array([[1.]], dtype=float32)

    """
    # Compute norms
    x_norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    y_norms = jnp.linalg.norm(y, axis=1, keepdims=True)

    # Normalize vectors
    x_norm = x / x_norms
    y_norm = y / y_norms

    # Compute cosine similarity
    cos_sim = jnp.dot(x_norm, y_norm.T)

    # Return 1 - cosine_similarity
    return 1.0 - cos_sim


def _sqrt_with_zero_selection(squared: Array) -> Array:
    """Keep exact zero distances with a selected zero derivative at coincidence.

    Guard the square root itself: masking its output still forms an infinite
    derivative at zero. This is a computational selection at a nonsmooth norm,
    not a classical differentiability claim. Nonfinite operands propagate.
    """
    zero = squared == 0
    return jnp.where(zero, 0.0, jnp.sqrt(jnp.where(zero, 1.0, squared)))


@jit
@static_jaxtyping("n d m")
def normalized_euclidean_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute pairwise Euclidean distance between normalized vectors.

    Each vector is first normalized to unit length before computing Euclidean
    distances. This makes the metric invariant to vector magnitude.

    Axioms: declares ``PSEUDOMETRIC``; brand
    :class:`~statadic.core.axioms.Premetric`. Chord length on the unit sphere, so
    ``TRIANGLE`` holds but ``IDENTITY`` does not.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.

    Returns:
        Array of shape (n, m) containing pairwise normalized Euclidean distances.

    Notes:
        Related to cosine distance by: normalized_euclidean = sqrt(2 * cosine_distance)
        Zero vectors return NaN distances as they have no defined direction.

    Examples:
        >>> x = jnp.array([[3.0, 4.0]])  # magnitude 5
        >>> y = jnp.array([[0.6, 0.8]])  # same direction, magnitude 1
        >>> normalized_euclidean_distance(x, y)
        Array([[0.]], dtype=float32)  # Zero distance after normalization

    """
    # Compute norms
    x_norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    y_norms = jnp.linalg.norm(y, axis=1, keepdims=True)

    # Normalize vectors
    x_norm = x / x_norms
    y_norm = y / y_norms

    # Compute pairwise Euclidean distances
    # Using ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    # Since vectors are normalized: ||a|| = ||b|| = 1
    dot_products = jnp.dot(x_norm, y_norm.T)
    squared_distances = 2.0 - 2.0 * dot_products

    # Ensure non-negative due to numerical errors
    squared_distances = jnp.maximum(squared_distances, 0.0)

    return _sqrt_with_zero_selection(squared_distances)


@jit
@static_jaxtyping("n d m")
def sphere_angular_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Angular metric on already checked L2-unit rows; performs no normalization."""
    cosine = jnp.clip(jnp.dot(x, y.T), -1.0, 1.0)
    return jnp.arccos(cosine)


@jit
@static_jaxtyping("n d m")
def sphere_chord_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Euclidean chord metric on already checked L2-unit rows."""
    squared = jnp.maximum(2.0 - 2.0 * jnp.dot(x, y.T), 0.0)
    return _sqrt_with_zero_selection(squared)


@jit
@static_jaxtyping("n d m")
def projective_angular_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Acute angular metric ``acos(|<N(x), N(y)>|)`` on projective classes."""
    x_unit = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    y_unit = y / jnp.linalg.norm(y, axis=1, keepdims=True)
    cosine = jnp.clip(jnp.abs(jnp.dot(x_unit, y_unit.T)), 0.0, 1.0)
    return jnp.arccos(cosine)


@jit
@static_jaxtyping("n d m")
def squared_euclidean_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute squared pairwise L2 distances without a square-root intermediate.

    This array primitive does not declare a metric law. It is suitable for
    smooth Gaussian kernels, including coordinate gradients at coincident rows.
    Direct differences preserve resolved local separations in broad-spread
    clouds; the sum of squares is nonnegative by construction.
    """
    # The norm/dot identity loses resolved local differences at large offsets
    # and broad spreads. The reduction fuses direct coordinate differences;
    # coincident rows keep their exact zero derivative.
    differences = x[:, None, :] - y[None, :, :]
    return jnp.sum(jnp.square(differences), axis=-1)


@jit
@static_jaxtyping("n d m")
def euclidean_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Compute pairwise Euclidean distance between row vectors.

    Standard L2 distance metric. Computes ||x_i - y_j||_2 for all pairs.

    At coincident rows the norm is nonsmooth; JAX uses the explicit zero
    derivative selection. Away from coincidence the ordinary norm derivative
    applies. This does not assert a classical Hessian at a collision.

    Axioms: declares ``METRIC``; brand
    :class:`~statadic.core.axioms.NegativeType` — the only ground metric here that
    sees scale, and the one to pass when identification actually matters.

    Args:
        x: Array of shape (n, d) where each row is a d-dimensional vector.
        y: Array of shape (m, d) where each row is a d-dimensional vector.

    Returns:
        Array of shape (n, m) containing pairwise Euclidean distances.

    Examples:
        >>> x = jnp.array([[0.0, 0.0]])
        >>> y = jnp.array([[3.0, 4.0]])
        >>> euclidean_distance(x, y)
        Array([[5.]], dtype=float32)

    """
    return _sqrt_with_zero_selection(squared_euclidean_distance(x, y))


type _RawDirectTransform = TransformContract[
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
type _SphereDirectTransform = TransformContract[
    L2UnitSphereSpace,
    L2UnitSphereSpace,
    IdentityMap,
    TotalOnDeclaredSource,
    DirectOrigin,
]
type _ProjectiveRepresentativeTransform = TransformContract[
    NonzeroVectorCarrierSpace,
    ProjectiveSpace,
    ProjectiveQuotient,
    NonzeroInputRequired,
    DirectOrigin,
]
type _PositiveRayRepresentativeTransform = TransformContract[
    NonzeroVectorCarrierSpace,
    PositiveRaySpace,
    PositiveRayQuotient,
    NonzeroInputRequired,
    NormalizationPullbackOrigin,
]

_RAW_VECTOR_SPACE: RawVectorSpace = MathematicalSpace()
_NONZERO_VECTOR_SPACE: NonzeroVectorCarrierSpace = MathematicalSpace()
_L2_SPHERE_SPACE: L2UnitSphereSpace = MathematicalSpace()
_PROJECTIVE_SPACE: ProjectiveSpace = MathematicalSpace()
_POSITIVE_RAY_SPACE: PositiveRaySpace = MathematicalSpace()
_RAW_DIRECT: _RawDirectTransform = TransformContract()
_NORMALIZATION_PULLBACK: _NormalizationTransform = TransformContract()
_SPHERE_DIRECT: _SphereDirectTransform = TransformContract()
_PROJECTIVE_REPRESENTATIVE: _ProjectiveRepresentativeTransform = TransformContract()
_POSITIVE_RAY_REPRESENTATIVE: _PositiveRayRepresentativeTransform = TransformContract()
_GOVERNED_REFERENCE = GovernedReferenceAuthority()
_DECLARED_AUTHORITY = DeclaredAuthority()


EUCLIDEAN: Final[
    GroundDistance[
        EuclideanLaw,
        RawVectorSpace,
        _RawDirectTransform,
        GovernedReferenceAuthority,
        FiniteValues,
    ]
] = declare_ground_distance(
    euclidean_distance,
    law=EuclideanLaw(),
    name="euclidean",
    space=_RAW_VECTOR_SPACE,
    transform=_RAW_DIRECT,
    authority=_GOVERNED_REFERENCE,
    required_evidence=FiniteValues,
)
"""Declared Euclidean strategy; the exact strong-negative metric default."""

ANGULAR: Final[
    GroundDistance[
        AngularLaw,
        NonzeroVectorCarrierSpace,
        _NormalizationTransform,
        GovernedReferenceAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_ground_distance(
    angular_distance,
    law=AngularLaw(),
    name="angular",
    requires_nonzero_rows=True,
    space=_NONZERO_VECTOR_SPACE,
    transform=_NORMALIZATION_PULLBACK,
    authority=_GOVERNED_REFERENCE,
    required_evidence=NonzeroNormalizationDomain,
)
"""Declared direction-only angular pseudometric strategy."""

COSINE: Final[
    GroundDistance[
        CosineLaw,
        NonzeroVectorCarrierSpace,
        _NormalizationTransform,
        GovernedReferenceAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_ground_distance(
    cosine_distance,
    law=CosineLaw(),
    name="cosine",
    requires_nonzero_rows=True,
    space=_NONZERO_VECTOR_SPACE,
    transform=_NORMALIZATION_PULLBACK,
    authority=_GOVERNED_REFERENCE,
    required_evidence=NonzeroNormalizationDomain,
)
"""Declared cosine dissimilarity strategy."""

NORMALIZED_EUCLIDEAN: Final[
    GroundDistance[
        NormalizedEuclideanLaw,
        NonzeroVectorCarrierSpace,
        _NormalizationTransform,
        GovernedReferenceAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_ground_distance(
    normalized_euclidean_distance,
    law=NormalizedEuclideanLaw(),
    name="normalized_euclidean",
    requires_nonzero_rows=True,
    space=_NONZERO_VECTOR_SPACE,
    transform=_NORMALIZATION_PULLBACK,
    authority=_GOVERNED_REFERENCE,
    required_evidence=NonzeroNormalizationDomain,
)
"""Declared normalized-Euclidean pseudometric strategy."""

RAY_ANGULAR: Final[
    GroundDistance[
        SphereAngularLaw,
        PositiveRaySpace,
        _PositiveRayRepresentativeTransform,
        DeclaredAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_ground_distance(
    angular_distance,
    law=SphereAngularLaw(),
    name="ray_angular",
    requires_nonzero_rows=True,
    space=_POSITIVE_RAY_SPACE,
    transform=_POSITIVE_RAY_REPRESENTATIVE,
    authority=_DECLARED_AUTHORITY,
    required_evidence=NonzeroNormalizationDomain,
)
"""Angular metric on positive-ray equivalence classes."""

RAY_CHORD: Final[
    GroundDistance[
        EuclideanLaw,
        PositiveRaySpace,
        _PositiveRayRepresentativeTransform,
        DeclaredAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_ground_distance(
    normalized_euclidean_distance,
    law=EuclideanLaw(),
    name="ray_chord",
    requires_nonzero_rows=True,
    space=_POSITIVE_RAY_SPACE,
    transform=_POSITIVE_RAY_REPRESENTATIVE,
    authority=_DECLARED_AUTHORITY,
    required_evidence=NonzeroNormalizationDomain,
)
"""Chord metric on positive-ray equivalence classes."""

SPHERE_ANGULAR: Final[
    GroundDistance[
        SphereAngularLaw,
        L2UnitSphereSpace,
        _SphereDirectTransform,
        GovernedReferenceAuthority,
        UnitNormSupport,
    ]
] = declare_ground_distance(
    sphere_angular_distance,
    law=SphereAngularLaw(),
    name="sphere_angular",
    requires_nonzero_rows=True,
    space=_L2_SPHERE_SPACE,
    transform=_SPHERE_DIRECT,
    authority=_GOVERNED_REFERENCE,
    required_evidence=UnitNormSupport,
)
"""Direct angular metric on checked L2-unit rows."""

SPHERE_CHORD: Final[
    GroundDistance[
        EuclideanLaw,
        L2UnitSphereSpace,
        _SphereDirectTransform,
        GovernedReferenceAuthority,
        UnitNormSupport,
    ]
] = declare_ground_distance(
    sphere_chord_distance,
    law=EuclideanLaw(),
    name="sphere_chord",
    requires_nonzero_rows=True,
    space=_L2_SPHERE_SPACE,
    transform=_SPHERE_DIRECT,
    authority=_GOVERNED_REFERENCE,
    required_evidence=UnitNormSupport,
)
"""Direct Euclidean chord metric on checked L2-unit rows."""

PROJECTIVE_ANGULAR: Final[
    GroundDistance[
        ProjectiveAngularLaw,
        ProjectiveSpace,
        _ProjectiveRepresentativeTransform,
        DeclaredAuthority,
        NonzeroNormalizationDomain,
    ]
] = declare_ground_distance(
    projective_angular_distance,
    law=ProjectiveAngularLaw(),
    name="projective_angular",
    requires_nonzero_rows=True,
    space=_PROJECTIVE_SPACE,
    transform=_PROJECTIVE_REPRESENTATIVE,
    authority=_DECLARED_AUTHORITY,
    required_evidence=NonzeroNormalizationDomain,
)
"""Acute angular metric on real-projective equivalence classes."""

# Compatibility views for the property battery. Values are derived from the
# executable strategy objects; no law fact is authored a second time here.
_DECLARED_GROUND_CASES: Final = {
    "euclidean_distance": EUCLIDEAN,
    "angular_distance": ANGULAR,
    "cosine_distance": COSINE,
    "normalized_euclidean_distance": NORMALIZED_EUCLIDEAN,
}

DECLARED_AXIOMS: Final[dict[str, Axioms]] = {
    name: strategy.law.axioms for name, strategy in _DECLARED_GROUND_CASES.items()
}
"""Derived property-battery view; strategy laws are the source of truth."""


def _legacy_marker_for_ground(strategy: GroundDistance[Law]) -> type[Premetric]:
    """Derive the retired one-axis marker used only by migration tests."""
    if strategy.law.axioms & METRIC == METRIC:
        if getattr(strategy.law, "negative_type", False):
            return NegativeType
        return Metric
    if strategy.law.axioms & SEMIMETRIC == SEMIMETRIC:
        return Semimetric
    return Premetric


DECLARED_BRANDS: Final[dict[str, type[Premetric]]] = {
    name: _legacy_marker_for_ground(strategy)
    for name, strategy in _DECLARED_GROUND_CASES.items()
}
"""Derived migration-only marker view; numerical producers never consume it."""


def checked_cloud_pdist[
    DomainKind: ObjectDomain,
    SupportKind: ValueEvidence,
    PooledKind: ValueEvidence,
    EvidenceSourceKind: EvidenceOrigin,
    GroundLawKind: Law,
    GroundSpaceKind: SpaceDescriptor,
    GroundTransformKind: TransformDescriptor,
    GroundAuthorityKind: ClaimAuthority,
](
    checked: CheckedClouds[
        DomainKind,
        SupportKind,
        PooledKind,
        EvidenceSourceKind,
    ],
    metric: GroundDistance[
        GroundLawKind,
        GroundSpaceKind,
        GroundTransformKind,
        GroundAuthorityKind,
        SupportKind,
    ],
) -> SpaceDMat[
    GroundSpaceKind,
    MatrixContract[
        GroundLawKind,
        GroundTransformKind,
        GroundAuthorityKind,
        SupportKind,
        EvidenceSourceKind,
    ],
]:
    """Build a fully indexed matrix from one checked empirical point cloud.

    ``SupportKind`` is shared invariantly with the strategy's weakest evidence
    requirement. Stronger carrier evidence widens through its nominal evidence
    lattice; insufficient evidence is a static error. Runtime checking repeats
    the marker test for untyped callers and never dispatches on a strategy name.

    The five claim-and-warrant axes travel as one
    :class:`~statadic.core.matrices.MatrixContract`; the strategy's own parameters
    are what fill it, so adding an axis to the contract changes this signature
    in one place rather than in its return type, its local annotation, and every
    caller's.
    """
    if checked.clouds.data.ndim != 3 or checked.clouds.counts.ndim != 1:  # noqa: PLR2004
        message = "checked_cloud_pdist requires one unbatched packed carrier"
        raise ValueError(message)
    if checked.clouds.data.shape[0] != 1:
        message = (
            "checked_cloud_pdist requires exactly one cloud, got "
            f"{checked.clouds.data.shape[0]}"
        )
        raise ValueError(message)
    if not isinstance(checked.support, metric.required_evidence):
        message = (
            f"ground strategy {metric.name!r} requires "
            f"{metric.required_evidence.__name__}, got "
            f"{type(checked.support).__name__}"
        )
        raise TypeError(message)
    if isinstance(checked.clouds.counts, jax.core.Tracer):
        message = "checked_cloud_pdist is an eager carrier boundary"
        raise TypeError(message)
    count = int(jax.device_get(checked.clouds.counts[0]))
    if count <= 0:
        message = "checked_cloud_pdist requires a nonempty cloud"
        raise ValueError(message)
    rows = checked.clouds.data[0, :count]
    return _space_dmat(metric(rows, rows))


@static_jaxtyping("n d m")
def cdist(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    metric: GroundDistance[Law] = EUCLIDEAN,
) -> Float[Array, "n m"]:
    """Compute pairwise distances using a declared strategy.

    Similar to scipy.spatial.distance.cdist but for JAX arrays.
    Built-ins are immutable singleton strategies; third-party callables cross
    :func:`declare_ground_distance` so their exact signature and law survive.

    Axioms: exactly what ``metric`` declares — see :data:`DECLARED_AXIOMS`. The
    dispatcher promises nothing of its own, which is why the axioms live on the
    callables rather than here.

    Args:
        x: Array of shape (n, d) where each row is a sample.
        y: Array of shape (m, d) where each row is a sample.
        metric: Declared built-in or third-party ground-distance strategy.

    Returns:
        Array of shape (n, m) containing pairwise distances.

    Raises:
        TypeError: If an undeclared string or bare callable bypasses typing.

    Examples:
        >>> x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> y = jnp.array([[1.0, 1.0]])
        >>> cdist(x, y, metric=EUCLIDEAN)
        Array([[1.       ],
               [1.       ]], dtype=float32)

        >>> # Custom metrics require a law declaration.
        >>> def scaled_euclidean(x, y):
        ...     return 2.0 * euclidean_distance(x, y)
        >>> custom = declare_ground_distance(
        ...     scaled_euclidean, law=EuclideanLaw(), name="scaled_euclidean"
        ... )
        >>> cdist(x, y, metric=custom)
        Array([[2.],
               [2.]], dtype=float32)

    """
    if (
        not callable(metric)
        or not hasattr(metric, "law")
        or getattr(metric, "backend", None) != "jax"
    ):
        message = (
            "metric must be a declared GroundDistance; parse serialized names "
            "with statadic.geometry.ground.config or wrap a callable with "
            "declare_ground_distance"
        )
        raise TypeError(message)
    return metric(x, y)


@static_jaxtyping("n d")
def pdist(
    x: Float[Array, "n d"],
    metric: GroundDistance[Law] = EUCLIDEAN,
) -> Float[Array, "n n"]:
    """Compute pairwise distances within a single sample.

    Similar to scipy.spatial.distance.pdist but for JAX arrays.
    Computes distances between all pairs of observations in x.

    Args:
        x: Array of shape (n, d) where each row is a sample.
        metric: Declared built-in or third-party ground-distance strategy.

    Returns:
        Array of shape (n, n) containing pairwise distances.

    Examples:
        >>> x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> pdist(x, metric=EUCLIDEAN)
        Array([[0., 1., 1.],
               [1., 0., 1.4142135],
               [1., 1.4142135, 0.]], dtype=float32)

    """
    return cdist(x, x, metric=metric)
