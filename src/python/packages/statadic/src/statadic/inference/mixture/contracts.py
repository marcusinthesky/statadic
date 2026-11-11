# jaxtyping shape strings are runtime contracts.
"""Retained report contract for selected-mixture inference.

``MixtureTestResult`` records point discrepancy, studentized permutation null,
bootstrap interval, selection-solver evidence and optional equivalence framing.
The difference-test p-value and the equivalence p-value address different nulls;
only the latter is an input to equivalence multiplicity adjustment. The contract is
shared by selection, split execution and downstream reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from statadic.core.typing import Array  # noqa: TC001  # retained array contract

__all__ = ["MixtureTestResult"]


@dataclass(frozen=True)
class MixtureTestResult:
    """Result of a studentized / re-optimized mixture two-sample test.

    Attributes:
        energy_point_estimate: Ê(X, Y_{w*}) — the PRIMARY hedge-quality metric
            (magnitude; monotone in hedge quality). Reported *instead of* an
            inverted p-value.
        energy_ci: (low, high) bootstrap CI on Ê.
        studentized_statistic: The studentized (Chung-Romano) test statistic.
        pvalue: **Difference** permutation p-value of the studentized statistic.
            Tests H0: F_X = F_{Y_w} (one-sided, greater). SMALL p ⇒ the
            distributions DIFFER ⇒ NOT equivalent. This is NOT an equivalence
            p-value; do not feed it to
            :func:`statadic.inference.multiplicity.equivalence_multiplicity_control`.
        equivalence_pvalue: **Equivalence / relevance** p-value for the relevance
            null H0: E(X, Y_w) ≥ δ vs H1: E(X, Y_w) < δ. Derived from the
            bootstrap distribution of Ê as ``(1 + #{Ê* ≥ δ}) / (1 + B)``. SMALL p
            ⇒ evidence FOR equivalence (Ê is confidently below δ). Consistent with
            the CI-based ``equivalence`` flag. This is a bootstrap approximation;
            non-finite replicates or margin give NaN. ``None`` outside equivalence
            framing. Holm (default) or Bonferroni familywise correction accepts
            these p-values but requires valid true-null p-values, which this
            bootstrap approximation is not shown to provide.
        equivalence: Bootstrap interval verdict, true when all replicates, both
            endpoints and δ are finite and ``ci_hi < δ``.
        delta: The equivalence margin used (None if not an equivalence test).
        weights: The selected mixture weights w*.
        m_eff: Inverse-variance effective size at w*.
        null_distribution: Permutation null of the studentized statistic.
        bootstrap_distribution: Bootstrap distribution of Ê (source of the CI and
            the equivalence p-value).
        reoptimized_in_permutation: True iff w* was re-optimized per replicate.
        studentized: Always True for this constructor (documents composition).
        equivalence_framed: True iff built via equivalence_test.
        num_permutations: Number of permutation replicates.
        solver_used: Numerical solver actually used for weight selection.
        solver_converged: Exact-solver convergence flag when available.
        solver_frank_wolfe_gap: Exact-solver KKT gap when available.
        solver_cnd_tangent_max_eigenvalue: Largest tangent-space Gram
            eigenvalue used by the CND check when available.
        solver_simplex_sum_error: Absolute simplex-sum error when available.
        solver_min_weight: Minimum selected weight when available.
        equivalence_curve: Optional list of (delta, rejected) over a δ-grid.

    """

    energy_point_estimate: float
    energy_ci: tuple[float, float]
    studentized_statistic: float
    pvalue: float
    weights: Array
    m_eff: float
    null_distribution: Array
    reoptimized_in_permutation: bool
    studentized: bool
    equivalence_framed: bool
    num_permutations: int
    weight_method: str = "kernel_balance"
    solver_used: str = ""
    solver_converged: bool | None = None
    solver_frank_wolfe_gap: float | None = None
    solver_cnd_tangent_max_eigenvalue: float | None = None
    solver_simplex_sum_error: float | None = None
    solver_min_weight: float | None = None
    bootstrap_distribution: Array = field(default_factory=lambda: jnp.array([]))
    equivalence_pvalue: float | None = None
    equivalence: bool | None = None
    delta: float | None = None
    equivalence_curve: list[tuple[float, bool]] = field(default_factory=list)
