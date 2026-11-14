# jaxtyping shape strings are runtime contracts.
"""Familywise corrections for equivalence verdicts."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["equivalence_multiplicity_control"]


def equivalence_multiplicity_control(
    equivalence_pvalues: jnp.ndarray,
    alpha: float = 0.05,
    method: str = "holm",
) -> jnp.ndarray:
    """Select equivalence verdicts using Holm or Bonferroni correction.

    Both procedures control familywise error at most ``alpha`` under arbitrary
    dependence, provided each true-null p-value satisfies P(p <= u) <= u for
    every u in [0, 1]. See Holm (1979), Theorem 1 and its Bonferroni argument.
    The bootstrap p-value produced by ``equivalence_test`` has not been shown
    to satisfy this condition. NaN p-values are never rejected. An empty family
    returns an empty Boolean mask.

    Benjamini-Hochberg is refused: it controls false discovery rate under
    particular assumptions, rather than familywise error in general.

    Args:
        equivalence_pvalues: Per-verdict p-values, shape (M,).
        alpha: Familywise error level.
        method: ``"holm"`` (default) or ``"bonferroni"``.

    Returns:
        Boolean mask of rejected hypotheses, shape (M,).

    """
    p = jnp.asarray(equivalence_pvalues)
    if method == "bh":
        message = "Benjamini-Hochberg controls FDR, not FWER in general."
        raise ValueError(message)
    if method == "bonferroni":
        if p.shape[0] == 0:
            return jnp.zeros(p.shape, dtype=bool)
        return p <= alpha / p.shape[0]
    if method == "holm":
        order = jnp.argsort(p)
        sorted_p = p[order]
        passed = sorted_p <= alpha / (p.shape[0] - jnp.arange(p.shape[0]))
        rejected = jnp.cumprod(passed).astype(bool)
        return jnp.zeros(p.shape, dtype=bool).at[order].set(rejected)
    message = f"Unknown method {method!r}."
    raise ValueError(message)
