# jaxtyping shape strings are runtime contracts.
"""Dimension-reduction maps for high-dimensional two-sample tests.

High-dimensional energy/MMD tests lose power polynomially in ``d`` at fixed ``n``
(Ramdas et al. 2015). Reducing to ``d' ≈ 10-50`` before a two-sample test restores
power. :func:`fit_projection` is intended to be fit on a SELECTION-SPLIT-A ONLY (no
look-ahead into the test split).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import (  # noqa: TC001  # runtime annotation vocabulary
    DimensionVar,
    Elements,
    Shape,
    ShapedArray,
)

__all__ = [
    "Projection",
    "fit_projection",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Projection[
    In: DimensionVar = int,
    Out: DimensionVar = int,
    Batch: Shape = Shape,
]:
    """A fitted linear dimension-reduction map (PCA or random projection).

    ``fit_projection`` returns this; apply with :meth:`transform`. The projection
    is fit on SELECTION-SPLIT-A ONLY to avoid look-ahead / double-dipping into the
    test split.
    """

    mean: Float[ShapedArray[[*Elements["Batch"], "In"]], "*batch d_in"]
    components: Float[
        ShapedArray[[*Elements["Batch"], "In", "Out"]], "*batch d_in d_out"
    ]
    kind: str = field(metadata={"static": True})

    def transform[Rows: DimensionVar](
        self: Projection[In, Out, tuple[()]],
        z: Float[ShapedArray[["Rows", "In"]], "n d_in"],
    ) -> Float[ShapedArray[["Rows", "Out"]], "n d_out"]:
        """Project observations using the fitted centering and components."""
        return (z - self.mean) @ self.components


def fit_projection[Rows: DimensionVar, In: DimensionVar](
    data: Float[ShapedArray[["Rows", "In"]], "n d_in"],
    d_out: int = 32,
    kind: str = "pca",
    seed: int | None = None,
) -> Projection[In, int, tuple[()]]:
    """Fit a dimension-reduction map on ``data`` (SELECTION-SPLIT-A ONLY).

    High-dimensional energy/MMD tests lose power polynomially in ``d`` at fixed
    ``n`` (Ramdas et al. 2015). Reducing to ``d' ≈ 10-50`` before the two-sample
    test restores power. Fitting on split A only keeps the test split
    independent of the reduction (no look-ahead).

    Args:
        data: Selection-split-A data, shape (n_a, d).
        d_out: Target dimension d' (clamped to min(d, n_a) as needed).
        kind: ``"pca"`` (top principal directions) or ``"random"`` (Gaussian
            random projection; JL).
        seed: RNG seed for ``kind="random"``.

    Returns:
        A fitted :class:`Projection`. The input width is retained statically;
        the output width is gradual because it is clamped at preparation time.

    """
    mean = jnp.mean(data, axis=0)
    d = data.shape[1]
    d_eff = int(min(d_out, d, max(1, data.shape[0] - 1)))
    if kind == "pca":
        centered = data - mean
        # Right singular vectors are the principal directions.
        _, _, vt = jnp.linalg.svd(centered, full_matrices=False)
        components = vt[:d_eff].T  # (d, d')
    elif kind == "random":
        key = random.PRNGKey(seed if seed is not None else 0)
        proj = random.normal(key, (d, d_eff)) / jnp.sqrt(d_eff)
        components = proj
    else:
        message = f"kind must be 'pca' or 'random', got {kind!r}"
        raise ValueError(message)
    return Projection(mean=mean, components=components, kind=kind)
