module

public import Architect
public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Linarith

/-! Diagonal moment-inequality statistic and its nonnegative orthant projection. -/
@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Inference

/-- A diagnostic interpretation of the squared negative-part statistic. -/
@[blueprint "STATADIC-INEQUALITY-001"
  (statement := /-- For standardized moments $z_j$, the squared negative-part
  statistic $M(z)=\sum_j\min(z_j,0)^2$ is the minimum of
  $\sum_j(z_j-t_j)^2$ over $t_j\ge0$, attained at $t_j=\max(z_j,0)$.
  It vanishes exactly when every standardized sample moment is nonnegative. -/)
  (proof := /-- When $z_j\ge0$, choosing $t_j=z_j$ gives zero.
  When $z_j<0$, the closest admissible value is zero, since moving right
  increases its distance to $z_j$. Sum these coordinatewise bounds.
  Vanishing of the sum of squares forces each negative part to vanish. -/)]
theorem mmm_orthant_distance {J : Type*} [Fintype J] (z : J → ℝ) :
    let m := ∑ j, (min (z j) 0) ^ 2
    (∑ j, (z j - max (z j) 0) ^ 2) = m ∧
    (∀ t : J → ℝ, (∀ j, 0 ≤ t j) → m ≤ ∑ j, (z j - t j) ^ 2) ∧
    (m = 0 ↔ ∀ j, 0 ≤ z j) := by
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · apply Finset.sum_congr rfl
    intro j _
    rcases le_total (z j) 0 with hj | hj <;> simp [hj]
  · intro t ht
    apply Finset.sum_le_sum
    intro j _
    rcases le_total (z j) 0 with hj | hj
    · rw [min_eq_left hj]
      nlinarith [mul_nonpos_of_nonpos_of_nonneg hj (ht j), sq_nonneg (t j)]
    · simpa [hj] using sq_nonneg (z j - t j)
  · constructor
    · intro h j
      have hz := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => sq_nonneg (min (z i) 0))).mp h
      have hj := sq_eq_zero_iff.mp (hz j (Finset.mem_univ _))
      exact hj ▸ min_le_left (z j) 0
    · intro h
      simp [h]

end StatadicFormal.Inference
