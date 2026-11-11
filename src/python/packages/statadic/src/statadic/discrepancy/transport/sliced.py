# jaxtyping shape strings are runtime contracts.
"""optimal-transport discrepancies — exact 1-D Wasserstein and sliced variants.

Position
--------
Consumes sample clouds and produces a scalar discrepancy.

No POT/OTT dependency: 1-D distances use the closed-form sorted-quantile
formula; multi-dimensional distances use random-projection (sliced Wasserstein).

Axioms
------
The exact one-dimensional callables declare ``METRIC``. The implemented
multivariate sliced-Wasserstein approximation uses finitely many shared
projections, so it declares only ``PSEUDOMETRIC``: two distinct empirical
measures may agree on every selected projection. Reusing one
:class:`ProjectionDesign` makes symmetry and triangle inequality meaningful;
it cannot recover separation lost by finite projection.

Sliced-Wasserstein aggregation convention
-----------------------------------------
This module implements the standard sliced-Wasserstein distance:

.. code-block:: text

    SW_p(μ, nu) = ( E_θ[ W_p^p(θ#μ, θ#nu) ] )^{1/p}

i.e. the p-th root of the **mean over projections of W_p^p** (not the mean of
W_p values).  This is the definition that makes SW_p a proper metric and matches
the literature (see Bonneel et al. 2015; Kolouri et al. 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int  # noqa: TC002

from statadic.core.axioms import (  # runtime annotations  # noqa: TC001
    METRIC,
    PSEUDOMETRIC,
    Axioms,
    Metric,
    Premetric,
)
from statadic.core.transforms import typed_vmap
from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.transport.univariate import wasserstein_1d, wasserstein_1d_power
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from jax.typing import (
        DTypeLike,
    )

__all__ = [
    "DECLARED_AXIOMS",
    "DECLARED_BRANDS",
    "InvalidProjectionCountError",
    "InvalidWassersteinOrderError",
    "ProjectionDesign",
    "sliced_wasserstein",
    "sliced_wasserstein_power_under_design",
    "sliced_wasserstein_under_design",
    "w1_1d",
    "w2_1d",
]

_MINIMUM_WASSERSTEIN_ORDER = 1.0
_SUPPORT_AXES = 2

#: Axioms guaranteed on every input in the documented domain. The finite-design
#: sliced approximation cannot promise separation on multivariate measures.
DECLARED_AXIOMS: Final[dict[str, Axioms]] = {
    "w1_1d": METRIC,
    "w2_1d": METRIC,
    "sliced_wasserstein": PSEUDOMETRIC,
}

#: The temporary linear marker chain cannot express a pseudometric, so sliced
#: Wasserstein brands down to ``Premetric`` until that compatibility surface is
#: removed. The exact one-dimensional distances remain metrics.
DECLARED_BRANDS: Final[dict[str, type[Premetric]]] = {
    "w1_1d": Metric,
    "w2_1d": Metric,
    "sliced_wasserstein": Premetric,
}


class InvalidProjectionCountError(ValueError):
    """Raised when a projection design would draw no directions."""

    def __init__(self, n_projections: int) -> None:
        """Initialize the diagnostic naming the rejected count.

        Args:
            n_projections: The rejected number of projection directions.

        """
        super().__init__(
            "a projection design must draw at least one direction, got "
            f"{n_projections}; the empty design defines no distance at all"
        )


class InvalidWassersteinOrderError(ValueError):
    """Raised when a Wasserstein order below one is requested."""

    def __init__(self, p: float) -> None:
        """Initialize the diagnostic naming the rejected order.

        Args:
            p: The rejected Wasserstein order.

        """
        super().__init__(
            f"the Wasserstein order must satisfy p >= 1, got {p}; "
            "below one the p-th root of a mean of p-th powers is not a metric"
        )


@final
@dataclass(frozen=True, slots=True)
class ProjectionDesign:
    """One reusable set of random projection directions, validated eagerly.

    Node defect 2. A finite projection set makes the sliced distance at best a
    **pseudometric** on multivariate empirical measures, and — worse — drawing a
    fresh set inside each pair call does not define one common function on
    pairs at all, so symmetry and the triangle inequality have no meaning across
    a matrix. This carrier is what a matrix producer holds so that *every* pair
    is projected through the same directions; the design is the object shared,
    not the key, so no caller can accidentally reseed between pairs.

    Not a pytree and not registered: ``n_projections`` and ``p`` are static
    (``p`` selects a Python branch in the projected reduction) and the key is
    consumed eagerly. Thread the design into a compiled function as a closure or
    a static argument, never as a traced leaf.

    Attributes:
        key: PRNG key the directions are drawn from.
        n_projections: Number of directions; must be at least one.
        p: Wasserstein order; must be at least one.

    Raises:
        InvalidProjectionCountError: If ``n_projections`` is not positive.
        InvalidWassersteinOrderError: If ``p < 1`` or is not finite.

    Examples:
        >>> design = ProjectionDesign(key=jax.random.PRNGKey(0), n_projections=8)
        >>> design.directions(3, jnp.float32).shape
        (8, 3)

    """

    key: jax.Array
    n_projections: int = 128
    # `int | float`, not `float`: beartype does not honour the implicit numeric
    # tower by default, and the shipped callers pass `p=2`.
    p: int | float = 2.0

    def __post_init__(self) -> None:
        """Reject an empty design or an order below one.

        Raises:
            InvalidProjectionCountError: If ``n_projections`` is not positive.
            InvalidWassersteinOrderError: If ``p < 1`` or is not finite.

        """
        if self.n_projections < 1:
            raise InvalidProjectionCountError(self.n_projections)
        order = float(self.p)
        if not jnp.isfinite(order) or order < _MINIMUM_WASSERSTEIN_ORDER:
            raise InvalidWassersteinOrderError(self.p)

    @static_jaxtyping("projections d")
    def directions(self, d: int, dtype: DTypeLike) -> Float[Array, "projections d"]:
        """Draw the shared unit directions for a ``d``-dimensional carrier.

        Args:
            d: Feature width of the samples to project.
            dtype: Floating dtype of those samples.

        Returns:
            Unit rows of shape ``(n_projections, d)``, identical for every pair
            evaluated under this design.

        """
        raw: Float[Array, "projections d"] = jax.random.normal(
            self.key, shape=(self.n_projections, d), dtype=dtype
        )
        return raw / jnp.linalg.norm(raw, axis=1, keepdims=True)


@static_jaxtyping("n m")
def w2_1d(
    x: PackedClouds,
    y: PackedClouds,
) -> Array:
    """2-Wasserstein distance between weighted 1-D empirical samples.

    Integrates squared quantile gaps. For equal uniform supports this gives::

        W₂(x, y) = sqrt( mean( (sort(x) - sort(y))² ) )

    Axioms: declares ``METRIC``; brand :class:`~statadic.core.axioms.Metric`.

    Args:
        x: 1-D sample, shape ``(n,)``.
        y: 1-D sample, shape ``(m,)``; cardinalities may differ.

    Returns:
        Non-negative scalar W₂ distance.

    """
    return wasserstein_1d(x, y, p=2)


@static_jaxtyping("n m")
def w1_1d(
    x: PackedClouds,
    y: PackedClouds,
) -> Array:
    """1-Wasserstein distance between weighted 1-D empirical samples.

    Integrates the empirical CDF difference. For equal uniform supports this gives::

        W₁(x, y) = mean( |sort(x) - sort(y)| )

    Axioms: declares ``METRIC``; brand :class:`~statadic.core.axioms.Metric`.

    Args:
        x: 1-D sample, shape ``(n,)``.
        y: 1-D sample, shape ``(m,)``; cardinalities may differ.

    Returns:
        Non-negative scalar W₁ distance.

    """
    return wasserstein_1d(x, y, p=1)


@static_jaxtyping("n d m")
def sliced_wasserstein(
    key: jax.Array,
    x: PackedClouds,
    y: PackedClouds,
    n_projections: int = 128,
    p: int = 2,
) -> Array:
    """Sliced p-Wasserstein distance between multivariate empirical samples.

    Approximates the p-Wasserstein distance by averaging 1-D Wasserstein
    distances over random linear projections:

    .. code-block:: text

        SW_p(x, y) = ( (1/L) Σ_l W_p^p(θ_l·x, θ_l·y) )^{1/p}

    where θ_l are uniform random unit vectors (drawn via normalised Gaussian
    rows) and W_p on the projected 1-D samples is computed via sorted quantiles.

    Aggregation convention: ``SW_p = (E_θ[W_p^p])^{1/p}`` — the standard
    population sliced-Wasserstein definition. This differs from the simpler
    ``E_θ[W_p]`` aggregation.

    Axioms: the shipped finite-projection implementation declares
    ``PSEUDOMETRIC``. A finite direction set need not separate multivariate
    empirical measures, and the declaration is meaningful only for a **fixed**
    design shared by every pair being compared. Building a matrix by calling
    this function per pair with different keys does not define one common
    symmetric function, which is what
    :class:`ProjectionDesign` and :func:`sliced_wasserstein_under_design` exist
    to make explicit.

    Args:
        key: JAX PRNGKey for drawing projection directions.
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``; cardinalities may differ.
        n_projections: Number of random projection directions.
        p: Wasserstein order; must satisfy ``p >= 1``.
        x_weights: Optional atom-weight vector, normalized to probabilities.
        y_weights: Optional second-support atom-weight vector.

    Returns:
        Non-negative scalar sliced W_p distance.

    Raises:
        InvalidProjectionCountError: If ``n_projections`` is not positive.
        InvalidWassersteinOrderError: If ``p < 1``.

    """
    return sliced_wasserstein_under_design(
        ProjectionDesign(key=key, n_projections=n_projections, p=p),
        x,
        y,
    )


@static_jaxtyping("n d m projections")
def sliced_wasserstein_under_design(
    design: ProjectionDesign,
    x: PackedClouds,
    y: PackedClouds,
) -> Array:
    """Sliced p-Wasserstein distance under one **shared** projection design.

    The pair-level form node defect 2 asks for: the directions come from the
    design rather than from a key drawn inside the call, so a caller evaluating
    a whole matrix passes one design and every pair is projected identically.
    That is what makes symmetry and the triangle inequality meaningful across
    the matrix — with a finite direction set the result is still, in general,
    only a **pseudometric** on multivariate empirical measures, so a matrix
    producer must not upgrade the claim merely because the design is shared.

    :func:`sliced_wasserstein` is this function with the design built from its
    ``key``/``n_projections``/``p`` arguments, which is where the eager
    ``p >= 1`` and nonempty-design validation now lives for both doors.

    Args:
        design: The validated, reusable projection design.
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``; cardinalities may differ.
        x_weights: Optional atom-weight vector, normalized to probabilities.
        y_weights: Optional second-support atom-weight vector.

    Returns:
        Non-negative scalar sliced W_p distance.

    Raises:
        ValueError: If atom weights have additional batch axes. Use ``vmap`` for
            separate weighted measures under the same design.

    """
    if x.data.ndim != _SUPPORT_AXES or y.data.ndim != _SUPPORT_AXES:
        return x.map_aligned(
            y, lambda a, b: sliced_wasserstein_under_design(design, a, b)
        )
    return sliced_wasserstein_power_under_design(design, x, y) ** (1.0 / design.p)


def sliced_wasserstein_power_under_design(
    design: ProjectionDesign, x: PackedClouds, y: PackedClouds
) -> Array:
    """Mean unrooted projected transport costs on a shared finite design."""
    if x.data.ndim != _SUPPORT_AXES or y.data.ndim != _SUPPORT_AXES:
        return x.map_aligned(
            y, lambda a, b: sliced_wasserstein_power_under_design(design, a, b)
        )
    left, right = x, y
    x_weights, y_weights = x.masses, y.masses
    left_points, right_points = x.coordinates, y.coordinates
    p = design.p
    _n, d = left_points.shape
    directions: Float[Array, "projections d"] = design.directions(
        d, left_points.dtype
    )  # (n_projections, d)

    # Project samples: (n, n_proj).
    x_proj = left_points @ directions.T
    y_proj = right_points @ directions.T

    # Exact per-projection costs retain both samples' atom masses.
    # `k` is the vmap-traced projection index, not a Python int (t46.1).
    def wp_p_1d(k: Int[Array, ""]) -> Array:
        return wasserstein_1d_power(
            PackedClouds(x_proj[:, k, None], left.counts, x_weights),
            PackedClouds(y_proj[:, k, None], right.counts, y_weights),
            p=p,
        )

    # vmap over projection index.
    # The count comes from the design, not from a local: `sliced_wasserstein`
    # no longer keeps one in scope now that both doors route through
    # `ProjectionDesign`.
    wp_p_vals: Float[Array, " projections"] = typed_vmap(wp_p_1d)(
        jnp.arange(design.n_projections)
    )

    # SW_p = (mean W_p^p)^{1/p}.
    return jnp.mean(wp_p_vals)
