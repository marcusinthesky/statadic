# Reading the mathematical foundations

A statistical method has several distinct mathematical questions. A definition
says what quantity is being computed. A theorem gives a conclusion under stated
assumptions. An algorithm and its implementation still need their own analysis.
The [claim registry](index.md) records these distinctions.

## Linear estimation

For a fixed design, the Gauss–Markov theorem compares linear unbiased estimators
under its stated error and rank conditions. It does not assert that arbitrary
estimators are unbiased or that a numerical QR solve is exact. The finite-design
formal statement is linked at {ref}`formal-STATADIC-LINEAR-004`; the
bias–variance identity is at {ref}`formal-STATADIC-MOMENTS-001`. For the underlying
result and its hypotheses, see {cite:p}`stepniak-gauss-markov-2010`.

<!-- statadic:math STATADIC-ENERGY-001 -->

## Energy and kernels

An energy functional compares cross-sample distances with distances within each
sample. Under a compatible generating kernel, its finite-support expression
equals twice the kernel quadratic form; positive semidefiniteness supplies
nonnegativity. See {ref}`formal-STATADIC-ENERGY-001` and
{cite:p}`sejdinovic-equivalence-2013`. This does not give separation without
further assumptions. The scalar STATADIC function requires a ground; angular geometry
discards positive radial scale, so the selected ground matters.

## MMD estimator reductions

The unequal-sample MMD² U reduction removes within-group diagonals and keeps
all cross-group pairs. Under supplied common pair expectations, its finite-law
expectation is the corresponding population-moment expression. See
{ref}`formal-STATADIC-MMD-001` and {cite:p}`gretton-etal-2012`, Lemma 6,
equation (3), pp. 728–729. IID sampling, the RKHS interpretation and numerical
implementation refinement remain separate obligations.

The supporting V-bias identity counts the retained diagonal contribution:
$(\delta_X-\mu_X)/m+(\delta_Y-\mu_Y)/n$. It alone supplies no sign guarantee.
STATADIC averages both cross-kernel orientations; symmetry relates that exact-real
formula to the proof's single cross block. The signed U estimate may be negative
and does not define a rooted distance.

## Randomization

A permutation test compares an observed statistic with statistics under a
specified randomization scheme. Validity requires the relevant exchangeability
assumptions and an appropriate schedule; add-one arithmetic alone cannot prove
calibration. See {ref}`formal-STATADIC-RANDOMIZATION-001` for the exact claim and
its boundaries.

## Cross-checks in other Lean libraries

| Library | Where to compare | Scope to check |
| --- | --- | --- |
| Mathlib | [Variance of finite sums](https://github.com/leanprover-community/mathlib4/blob/56a005855ea17e0b0958f73ec801ae140b0a1679/Mathlib/Probability/Moments/Variance.lean) | STATADIC-LINEAR-010 uses `ProbabilityTheory.variance_fun_sum` from STATADIC's pinned dependency. Check square integrability and whether the theorem states covariance or raw second moments. |
| StatLean | [OLS prediction error](https://github.com/StatLean/Stat-Lean/blob/e436d8a184421c2a0ed8db0b2688c6cd26a7f5a2/StatLean/HighDimensionalStatistics/OLS/MSEExpectation.lean) and [randomization level](https://github.com/StatLean/Stat-Lean/blob/e436d8a184421c2a0ed8db0b2688c6cd26a7f5a2/StatLean/HypothesisTesting/Randomization/ExactLevel.lean) | `mse_ols_expectation_le` concerns prediction risk. Compare its loss and sampling model with STATADIC's contrast variance. For `randTest_exact_level`, check the tie randomization and invariance hypotheses against STATADIC's inclusive rank rule. |
| MathEconLib | [Finite-distribution expectations](https://github.com/matheconlib/matheconlib/blob/9207b5b65a2a54172d8dd3a3f23b82e9e072ac8b/MathEconLib/Probability/FinDist/Expect.lean) | Compare `expect_eq_sum` and `expect_mixture` with STATADIC's finite-law expectation and its normalized weights. An equivalence would require a map between the two representations. |

The StatLean and MathEconLib links are reference snapshots for contributors.
Their declarations are outside STATADIC's dependency graph.
