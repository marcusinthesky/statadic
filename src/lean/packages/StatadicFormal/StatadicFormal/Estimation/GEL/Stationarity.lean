module

public import StatadicFormal.Estimation.Linear.FiniteLaw
import Mathlib.Tactic.Positivity

/-! Implied probabilities and the separating-direction diagnostic for EL/ET. -/
@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Estimation.GEL

/-- A positive stationary carrier yields balancing probabilities and excludes
a direction weakly separating all moments and strictly separating even one. -/
@[blueprint "STATADIC-GEL-001"
  (statement := /-- For finite nonempty moment rows $g_i$ and positive
  carrier weights $a_i$, suppose $\sum_i a_i g_i=0$. Normalizing by
  $A=\sum_i a_i$ gives positive probabilities $p_i=a_i/A$ with
  $\sum_i p_i g_i=0$. Any projection $b^Tg_i$ that is nonnegative on
  every row must vanish on every row. -/)
  (proof := /-- The total carrier weight is strictly positive. Divide the
  normalization and stationarity sums by it. Sum the stationarity equations
  against $b$ and exchange the two finite sums. The sum of the nonnegative
  numbers $a_i b^Tg_i$ is zero, forcing each projection to vanish. -/)]
theorem positive_stationary_weights {I J : Type*} [Fintype I] [Nonempty I] [Fintype J]
    (a : I → ℝ) (g : I → J → ℝ) (ha : ∀ i, 0 < a i)
    (hstat : ∀ j, (∑ i, a i * g i j) = 0) :
    let p := fun i => a i / ∑ k, a k
    (∀ i, 0 < p i) ∧ (∑ i, p i) = 1 ∧ (∀ j, (∑ i, p i * g i j) = 0) ∧
      ∀ b : J → ℝ, (∀ i, 0 ≤ ∑ j, b j * g i j) → ∀ i, (∑ j, b j * g i j) = 0 := by
  have hA : 0 < ∑ i, a i := Finset.sum_pos (fun i _ => ha i) Finset.univ_nonempty
  refine ⟨fun i => div_pos (ha i) hA, ?_, ?_, ?_⟩
  · simp only [div_eq_mul_inv, ← Finset.sum_mul, mul_inv_cancel₀ hA.ne']
  · intro j
    change (∑ i, (a i / ∑ k, a k) * g i j) = 0
    calc
      _ = (∑ i, a i * g i j) * (∑ k, a k)⁻¹ := by
        rw [Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro i _
        rw [div_eq_mul_inv, mul_right_comm]
      _ = 0 := by rw [hstat, zero_mul]
  · intro b hb i
    have hz : (∑ i, a i * ∑ j, b j * g i j) = 0 := by
      simp_rw [Finset.mul_sum, mul_left_comm (a _) (b _)]
      rw [Finset.sum_comm]
      simp_rw [← Finset.mul_sum, hstat, mul_zero, Finset.sum_const_zero]
    have h := (Finset.sum_eq_zero_iff_of_nonneg (fun k _ => mul_nonneg (ha k).le (hb k))).mp hz
    exact (mul_eq_zero.mp (h i (Finset.mem_univ i))).resolve_left (ha i).ne'

end StatadicFormal.Estimation.GEL
