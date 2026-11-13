# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Mantel association coefficients for aligned dissimilarity structures.

``mantel_correlation_kernel`` computes Pearson or tie-aware Spearman correlation
of supplied pair vectors. For distance matrices, pass matching strict upper
triangles. Permutation calibration and node-bootstrap intervals belong to
:mod:`statadic.inference.mantel`; the coefficient alone establishes no null law.
"""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata
from jaxtyping import Float as JaxFloat  # noqa: TC002

from statadic.core.typing import (  # noqa: TC001
    Array,
    Static,
    static_jaxtyping,
)

MantelMethod = Literal["pearson", "spearman"]

__all__ = [
    "MantelMethod",
    "mantel_correlation_kernel",
    "partial_mantel_correlation_kernel",
]


def _require_method(method: MantelMethod) -> None:
    """Reject an unsupported correlation method at every public entry.

    Every public door calls this. Until the bootstrap loops became traceable the
    private ``_correlate`` helper carried the only runtime spelling of this
    guard, and the retired eager Mantel door — which never called it — had none:
    an untyped
    caller passing ``method="kendall"`` silently received a Pearson coefficient,
    because the kernel's ``if method == "spearman"`` simply fell through. The
    guard now lives where the contract is documented instead of where an
    implementation detail happened to put it.
    """
    if method not in ("pearson", "spearman"):
        message = f"Unknown method '{method}'. Choose 'pearson' or 'spearman'."
        raise ValueError(message)


@partial(jax.jit, static_argnames=("method",))
@static_jaxtyping("p")
def mantel_correlation_kernel(
    x: JaxFloat[Array, " p"],
    y: JaxFloat[Array, " p"],
    method: Static[MantelMethod] = "pearson",
) -> JaxFloat[Array, ""]:
    """Return one Pearson or tie-aware Spearman coefficient on graph."""
    _require_method(method)
    x_values = jnp.asarray(x)
    y_values = jnp.asarray(y)
    if method == "spearman":
        x_values = rankdata(x_values, method="average")
        y_values = rankdata(y_values, method="average")
    x_centered = x_values - jnp.mean(x_values)
    y_centered = y_values - jnp.mean(y_values)
    denominator = jnp.linalg.norm(x_centered) * jnp.linalg.norm(y_centered)
    return jnp.sum(x_centered * y_centered) / denominator


@jax.jit
@static_jaxtyping("p")
def partial_mantel_correlation_kernel(
    x: JaxFloat[Array, " p"],
    y: JaxFloat[Array, " p"],
    control: JaxFloat[Array, " p"],
) -> JaxFloat[Array, ""]:
    """Return the Pearson correlation of ``x`` and ``y`` net of ``control``.

    Both pair vectors are least-squares residualized on ``[1, control]`` and the
    residuals correlated. Descriptive only: the coefficient carries no null law.
    """
    design = jnp.stack([jnp.ones_like(control), control], axis=1)

    def residual(values: JaxFloat[Array, " p"]) -> JaxFloat[Array, " p"]:
        return values - design @ jnp.linalg.lstsq(design, values)[0]

    return mantel_correlation_kernel(residual(x), residual(y), "pearson")
