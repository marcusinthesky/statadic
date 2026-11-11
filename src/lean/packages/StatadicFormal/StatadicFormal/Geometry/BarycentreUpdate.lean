module

public import Architect
public import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Ring

/-! Squared-Euclidean free support update conditional on fixed matchings. -/

public section
open scoped BigOperators InnerProductSpace
namespace StatadicFormal.Geometry
variable {H ι : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [Fintype ι]

/-- The weighted matched centroid has an exact, strictly positive excess cost for any other point. -/
@[blueprint "STATADIC-BARYCENTRE-001"
  (statement := /-- For a fixed support row and fixed matchings let $x_i$ be the
  matched source locations. With nonnegative weights summing to one, put
  $m=\sum_i w_i x_i$. At any $y$, the fixed-assignment squared cost satisfies
  $\sum_i w_i\|x_i-y\|^2=\sum_i w_i\|x_i-m\|^2+\|y-m\|^2$.
  Thus the centroid is the unique minimizing location for this row. -/)
  (proof := /-- Write $x_i-y=(x_i-m)-(y-m)$ and expand the squared norms.
  The cross terms sum to zero since $\sum_i w_i(x_i-m)=0$.
  The remaining excess cost is strictly positive unless $y=m$. -/)]
theorem barycentre_row_gap (w : ι → ℝ) (hsum : ∑ i, w i = 1) (x : ι → H) (y : H) :
    let m := ∑ i, w i • x i
    let cost := fun z => ∑ i, w i * ‖x i - z‖ ^ 2
    cost y = cost m + ‖y - m‖ ^ 2 ∧ (cost y ≤ cost m ↔ y = m) := by
  let m := ∑ i, w i • x i
  have hc : ∑ i, w i • (x i - m) = 0 := by
    simp [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul, hsum, m]
  have hc' := congrArg (fun v : H => ⟪v, y - m⟫_ℝ) hc
  simp only [sum_inner, real_inner_smul_left, inner_zero_left] at hc'
  have exp (i) : w i * ‖x i - y‖ ^ 2 = w i * ‖x i - m‖ ^ 2 +
      w i * ‖y - m‖ ^ 2 - 2 * (w i * ⟪x i - m, y - m⟫_ℝ) := by
    rw [show x i - y = (x i - m) - (y - m) by abel, norm_sub_sq_real]
    ring
  have heq : (∑ i, w i * ‖x i - y‖ ^ 2) = (∑ i, w i * ‖x i - m‖ ^ 2) + ‖y - m‖ ^ 2 := by
    simp only [exp, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hc',
      ← Finset.sum_mul, hsum, one_mul, mul_zero, sub_zero]
  refine ⟨heq, ?_⟩
  change (∑ i, w i * ‖x i - y‖ ^ 2) ≤ (∑ i, w i * ‖x i - m‖ ^ 2) ↔ _
  rw [heq, add_le_iff_nonpos_right]
  constructor
  · intro h
    exact sub_eq_zero.mp (norm_eq_zero.mp (sq_eq_zero_iff.mp (le_antisymm h (sq_nonneg _))))
  · rintro rfl
    simp [m]

end StatadicFormal.Geometry
