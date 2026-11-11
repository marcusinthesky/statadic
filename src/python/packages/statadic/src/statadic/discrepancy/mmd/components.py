# jaxtyping shape strings are runtime contracts.
"""Candidate reductions for kernel-MMD and distance-induced energy objectives.

``candidate_kernel_components`` returns candidate Gram means, target cross
means and the target self mean. ``candidate_energy_components`` returns raw
candidate distance means and target cross means, including self pairs. These
are components rather than calibrated estimates or law-bearing distances.

Clouds must be finite, nonempty and share a feature width. The caller owns the
ground/kernel value domain and exponent validity. Unequal cloud sizes retain
their empirical probabilities.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from jax import vmap
from jaxtyping import Float  # noqa: TC002

from statadic.core.axioms import (
    Law,  # runtime annotations  # noqa: TC001
)
from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001
from statadic.discrepancy.mmd.scalar import _shared_reference
from statadic.geometry.ground.metrics import (  # runtime annotations  # noqa: TC001
    EUCLIDEAN,
    GroundDistance,
    cdist,
)
from statadic.geometry.ground.similarities import (
    gram,
)

if TYPE_CHECKING:
    from statadic.geometry.ground.similarities import (
        PositiveSemidefiniteKernelLaw,
        SimilarityStrategy,
    )
    from statadic.sample.packed import (
        PackedClouds,
    )

__all__ = ["candidate_energy_components", "candidate_kernel_components"]


@static_jaxtyping("n d k")
def candidate_energy_components(
    target: PackedClouds,
    candidates: PackedClouds,
    metric: GroundDistance[Law] = EUCLIDEAN,
    exponent: ArrayLike = 1.0,
) -> tuple[Float[Array, "k k"], Float[Array, " k"]]:
    """Compute candidate Gram and target-cross mean-distance components.

    Packed candidate probabilities preserve unequal cloud sizes.
    """

    def costs(a: Array, b: Array) -> Array:
        return cdist(a, b, metric=metric) ** exponent

    def row(a: PackedClouds) -> Array:
        return a.expectation(candidates, costs)

    return vmap(row)(candidates), target.expectation(candidates, costs)


@static_jaxtyping("n d k")
def candidate_kernel_components(
    target: PackedClouds,
    candidates: PackedClouds,
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
    reference: Float[Array, " d"] | None = None,
) -> tuple[
    Float[Array, "k k"],
    Float[Array, " k"],
    Float[Array, ""],
]:
    """Compute kernel Gram means for a target-directed MMD barycentre.

    The returned values are the candidate--candidate Gram means ``G``, the
    target--candidate Gram means ``c``, and the target self mean ``b``. The
    same reference is passed through every block so distance-induced kernels
    retain the MMD cancellation convention.
    """
    shared_reference = _shared_reference(target.coordinates, reference, target.masses)
    values = partial(gram, kernel=kernel, reference=shared_reference)

    def row(a: PackedClouds) -> Array:
        return a.expectation(candidates, values)

    candidate_gram = vmap(row)(candidates)
    candidate_gram = (candidate_gram + candidate_gram.T) / 2.0
    target_cross = target.expectation(candidates, values)
    target_self = target.expectation(target, values)
    return candidate_gram, target_cross, target_self
