module

public import Architect
public import Mathlib.Basic.Real.Basic
public import Mathlib.LinearAlgebra.Matrix.Trace

/-! # Direct and total effects conditional on a fixed spatial graph

Halleck Vega and Elhorst (2015), Section 2, equations (8)–(10) and Table 1.
The reduced-form multiplier is supplied as a matrix. This keeps the algebra
valid for asymmetric weights and for each panel period's own roster. -/

@[expose] public section
noncomputable section

open scoped BigOperators
open Matrix

namespace StatadicFormal.Estimation.Spatial

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The average row sum of a square matrix on the active roster. -/
def averageRowSum (M : Matrix ι ι ℝ) : ℝ :=
  (∑ i, ∑ j, M i j) / Fintype.card ι

/-- A finite covariate change in the fixed-graph reduced mean has exactly
its impact-matrix response; the covariate-independent offset cancels. -/
theorem durbin_mean_change (S W : Matrix ι ι ℝ) (β θ : ℝ) (a x v : ι → ℝ) :
    S *ᵥ (a + (β • (1 : Matrix ι ι ℝ) + θ • W) *ᵥ (x + v)) -
      S *ᵥ (a + (β • (1 : Matrix ι ι ℝ) + θ • W) *ᵥ x) =
      (S * (β • (1 : Matrix ι ι ℝ) + θ • W)) *ᵥ v := by
  simp only [Matrix.mulVec_add, ← Matrix.mulVec_mulVec, ← add_assoc, add_sub_cancel_left]

/-- Trace and row-sum multipliers for a covariate with a spatially lagged term. -/
@[blueprint "STATADIC-SPATIAL-003"
  (statement := /-- For a fixed graph $W$ and multiplier $S$, the marginal effect
  matrix is $D=S(\beta I+\theta W)$. Its mean diagonal is
  $\beta\operatorname{tr}(S)/n+\theta\operatorname{tr}(SW)/n$. Its mean row
  sum is the same linear combination of the mean row sums of $S$ and $SW$. -/)
  (proof := /-- Distribute $S$ over the sum and commute scalar multiplication
  through the matrix product. Apply linearity of the trace and of both finite
  sums, and divide by the roster size. -/)]
theorem durbin_impact_averages (S W : Matrix ι ι ℝ) (β θ : ℝ) :
    let D := S * (β • (1 : Matrix ι ι ℝ) + θ • W)
    D.trace / Fintype.card ι =
        β * (S.trace / Fintype.card ι) + θ * ((S * W).trace / Fintype.card ι) ∧
      averageRowSum D = β * averageRowSum S + θ * averageRowSum (S * W) := by
  dsimp only
  rw [Matrix.mul_add, Matrix.mul_smul, Matrix.mul_smul, Matrix.mul_one]
  constructor
  · simp only [Matrix.trace_add, Matrix.trace_smul, smul_eq_mul, add_div, mul_div_assoc]
  · simp only [averageRowSum, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      Finset.sum_add_distrib, ← Finset.mul_sum, add_div, mul_div_assoc]

end StatadicFormal.Estimation.Spatial
