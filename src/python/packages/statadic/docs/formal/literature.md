---
type: explanation
description: Mathematical statements, hypotheses and comparisons for STATADIC methods.
---

# Reading the mathematical foundations

## How to read this

Several mathematical questions sit behind a statistical method: a definition says what is computed, while a theorem gives a conclusion under stated assumptions. The algorithm and its implementation need their own analysis, and the [claim registry](index.md) records these distinctions alongside the assumptions for each result.

## Linear estimation

For a fixed design, the Gauss–Markov theorem compares linear unbiased estimators
under its stated error and rank conditions. That comparison does not assert that arbitrary estimators are unbiased or that a numerical QR solve is exact. The formal statement for the fixed design is linked at
{ref}`formal-STATADIC-LINEAR-004`, alongside the bias–variance identity at
{ref}`formal-STATADIC-MOMENTS-001`. For the underlying result, including its hypotheses and the scope of the comparison, see {cite:p}`stepniak-gauss-markov-2010`.

## Energy and kernels

<!-- statadic:math STATADIC-ENERGY-001 -->

An energy functional compares cross-sample distances with distances within each sample, and for a compatible generating kernel the expression on finite support equals
twice the quadratic form of the kernel, whose positive semidefiniteness
supplies nonnegativity. See {ref}`formal-STATADIC-ENERGY-001` and
{cite:p}`sejdinovic-equivalence-2013`. Separation requires further assumptions, and the scalar STATADIC function also requires a ground distance. Angular geometry
discards positive radial scale, so select the ground using the differences
that matter for your analysis.

## MMD estimator reductions

The unequal-sample MMD² U reduction removes within-group diagonals and keeps
all cross-group pairs. Under supplied common pair expectations, its finite-law
expectation is the corresponding population-moment expression. See
{ref}`formal-STATADIC-MMD-001` and {cite:p}`gretton-etal-2012`, Lemma 6,
equation (3), pp. 728–729. IID sampling, the RKHS interpretation and numerical
implementation refinement remain separate obligations.

The supporting V-bias identity counts the contribution from diagonal terms as $(\delta_X-\mu_X)/m+(\delta_Y-\mu_Y)/n$, without supplying a sign guarantee on its own.
STATADIC averages both orientations of the cross-kernel calculation, and
symmetry relates that formula to the single cross block in the proof.

A U estimate can be negative, so its value cannot be interpreted directly as a rooted distance between the samples, even when the population discrepancy is nonnegative.

## Randomization

A permutation test compares an observed statistic with statistics under a specified randomization scheme, whose validity depends on exchangeability and the schedule used to randomize observations. The add-one arithmetic addresses the
Monte Carlo calculation rather than establishing those assumptions. See {ref}`formal-STATADIC-RANDOMIZATION-001` for the exact claim and
its boundaries.

## Cross-checks in other Lean libraries

| Library | Where to compare | Scope to check |
| --- | --- | --- |
| Mathlib | [Variance of finite sums](https://github.com/leanprover-community/mathlib4/blob/56a005855ea17e0b0958f73ec801ae140b0a1679/Mathlib/Probability/Moments/Variance.lean) | STATADIC-LINEAR-010 uses `ProbabilityTheory.variance_fun_sum` from STATADIC's pinned dependency. Check square integrability and whether the theorem states covariance or raw second moments. |
| StatLean | [OLS prediction error](https://github.com/StatLean/Stat-Lean/blob/e436d8a184421c2a0ed8db0b2688c6cd26a7f5a2/StatLean/HighDimensionalStatistics/OLS/MSEExpectation.lean) and [randomization level](https://github.com/StatLean/Stat-Lean/blob/e436d8a184421c2a0ed8db0b2688c6cd26a7f5a2/StatLean/HypothesisTesting/Randomization/ExactLevel.lean) | `mse_ols_expectation_le` concerns prediction risk. Compare its loss and sampling model with STATADIC's contrast variance. For `randTest_exact_level`, check the tie randomization and invariance hypotheses against STATADIC's inclusive rank rule. |
| MathEconLib | [Finite-distribution expectations](https://github.com/matheconlib/matheconlib/blob/9207b5b65a2a54172d8dd3a3f23b82e9e072ac8b/MathEconLib/Probability/FinDist/Expect.lean) | Compare `expect_eq_sum` and `expect_mixture` with STATADIC's finite-law expectation and its normalized weights. An equivalence would require a map between the two representations. |

The StatLean and MathEconLib links are reference snapshots for contributors.
Their declarations are outside STATADIC's dependency graph.
