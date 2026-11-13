module

public import Architect
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring

/-! Primal and dual witnesses for finite, balanced, unregularized transport. -/

public section
open scoped BigOperators
namespace StatadicFormal.Discrepancy.Transport
variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Costs above a feasible dual lower bound are exactly the transported reduced costs. -/
@[blueprint "STATADIC-TRANSPORT-001"
  (statement := /-- A nonnegative transport matrix $P$ has row masses $a_i$ and
  column masses $b_j$. If potentials satisfy $u_i+v_j\le C_{ij}$, then
  $\sum P_{ij}C_{ij}-\sum a_iu_i-\sum b_jv_j=
  \sum P_{ij}(C_{ij}-u_i-v_j)\ge0$.
  Equality on the support of $P$ certifies minimality among all feasible plans. -/)
  (proof := /-- Insert the row and column mass constraints into the potential sums.
  Collect terms by cell. Each reduced-cost contribution is nonnegative and vanishes
  when its mass is zero or its potential constraint is tight. Hence a tight plan
  attains a lower bound shared by every other feasible plan. -/)]
theorem transport_gap (C P : ι → κ → ℝ) (a u : ι → ℝ) (b v : κ → ℝ)
    (hp : ∀ i j, 0 ≤ P i j) (hr : ∀ i, ∑ j, P i j = a i)
    (hc : ∀ j, ∑ i, P i j = b j) (hd : ∀ i j, u i + v j ≤ C i j) :
    let primal := ∑ i, ∑ j, P i j * C i j
    let dual := (∑ i, a i * u i) + ∑ j, b j * v j
    primal - dual = ∑ i, ∑ j, P i j * (C i j - (u i + v j)) ∧ dual ≤ primal ∧
      ((∀ i j, P i j ≠ 0 → C i j = u i + v j) → primal = dual) := by
  have hu : (∑ i, ∑ j, P i j * u i) = ∑ i, a i * u i := by
    simp only [← Finset.sum_mul, hr]
  have hv : (∑ i, ∑ j, P i j * v j) = ∑ j, b j * v j := by
    rw [Finset.sum_comm]
    simp only [← Finset.sum_mul, hc]
  have he : (∑ i, ∑ j, P i j * C i j) - ((∑ i, a i * u i) + ∑ j, b j * v j) =
      ∑ i, ∑ j, P i j * (C i j - (u i + v j)) := by
    simp only [mul_sub, mul_add, Finset.sum_sub_distrib, Finset.sum_add_distrib, hu, hv]
  refine ⟨he, sub_nonneg.mp (he.ge.trans' ?_), ?_⟩
  · exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ =>
      mul_nonneg (hp i j) (sub_nonneg.mpr (hd i j))
  · intro ht
    apply sub_eq_zero.mp
    rw [he]
    apply Finset.sum_eq_zero
    intro i _
    apply Finset.sum_eq_zero
    intro j _
    by_cases h : P i j = 0
    · rw [h, zero_mul]
    · rw [ht i j h, sub_self, mul_zero]

end StatadicFormal.Discrepancy.Transport
