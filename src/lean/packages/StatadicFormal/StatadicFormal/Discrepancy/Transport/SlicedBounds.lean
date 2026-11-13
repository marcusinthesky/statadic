module

public import StatadicFormal.Discrepancy.Transport.AssignmentDuality
public import Mathlib.Algebra.Order.Rearrangement
public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Sorted real-line quadratic transport and fixed-design projection bounds. -/
public section
open scoped BigOperators InnerProductSpace
namespace StatadicFormal.Discrepancy.Transport

/-- Matching ascending ranks minimizes quadratic cost among all balanced equal-mass plans. -/
@[blueprint "STATADIC-TRANSPORT-003"
  (statement := /-- For sorted real observations $x_i,y_i$ on the same finite nonempty
  rank axis, matching equal ranks minimizes $\sum P_{ij}(x_i-y_j)^2$ over all
  doubly stochastic $P$. Dividing by the sample size gives squared uniform $W_2$. -/)
  (proof := /-- In each permutation cost the two squared marginal sums are
  invariant. The remaining cross product is maximized by the rank matching via
  the rearrangement inequality. Birkhoff decomposition extends that optimum
  from permutations to every doubly stochastic matrix. -/)]
theorem sorted_quadratic_transport {n : ℕ} (x y : Fin n → ℝ) (hx : Monotone x) (hy : Monotone y)
    (P : Matrix (Fin n) (Fin n) ℝ) (hP : P ∈ doublyStochastic ℝ (Fin n)) :
    (∑ i, (x i - y i) ^ 2) ≤ assignmentCost (fun i j => (x i - y j) ^ 2) P := by
  apply fractional_bound _ _ hP
  intro σ
  have hs := (hx.monovary hy).sum_mul_comp_perm_le_sum_mul (σ := σ)
  have he : (∑ i, y (σ i) ^ 2) = ∑ i, y i ^ 2 := Equiv.sum_comp σ (fun i => y i ^ 2)
  simp_rw [sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib, mul_assoc, ← Finset.mul_sum]
  linarith

/-- Every fixed unit-direction design has powered sliced cost at most full
powered Euclidean assignment cost at all real orders at least one. -/
@[blueprint "STATADIC-TRANSPORT-004"
  (statement := /-- For two uniform equal-size samples in a real inner product space,
  let the directions $\theta_l$ have norm at most one and weights $q_l\ge0$
  summing to one. At $p\ge1$, their weighted mean of optimal one-dimensional
  $p$-power costs is at most the full Euclidean $p$-power assignment cost. -/)
  (proof := /-- Use the same full-space matching as a candidate for every
  projected problem. Cauchy--Schwarz bounds each projected paired distance by
  the ambient distance. Raise to the power $p$, average over rows, and then
  over directions. Optimal projection costs can only be smaller. -/)]
theorem sliced_power_le_assignment {H ι L : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℝ H] [Fintype ι] [Fintype L] (x y : ι → H) (θ : L → H)
    (q cost : L → ℝ) (w : Equiv.Perm ι) (p : ℝ) (hp : 1 ≤ p)
    (hθ : ∀ l, ‖θ l‖ ≤ 1) (hq : ∀ l, 0 ≤ q l) (htot : ∑ l, q l = 1)
    (hc : ∀ l (σ : Equiv.Perm ι), cost l ≤
      (∑ i, |⟪θ l, x i⟫_ℝ - ⟪θ l, y (σ i)⟫_ℝ| ^ p) / Fintype.card ι) :
    (∑ l, q l * cost l) ≤ (∑ i, ‖x i - y (w i)‖ ^ p) / Fintype.card ι := by
  let B := (∑ i, ‖x i - y (w i)‖ ^ p) / Fintype.card ι
  have hB (l) : cost l ≤ B := by
    apply (hc l w).trans
    apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
    refine Finset.sum_le_sum fun i _ => Real.rpow_le_rpow (abs_nonneg _) ?_ (by linarith)
    rw [← inner_sub_right]
    apply (abs_real_inner_le_norm _ _).trans
    simpa using mul_le_mul_of_nonneg_right (hθ l) (norm_nonneg (x i - y (w i)))
  calc
    _ ≤ ∑ l, q l * B := Finset.sum_le_sum fun l _ => mul_le_mul_of_nonneg_left (hB l) (hq l)
    _ = _ := by rw [← Finset.sum_mul, htot, one_mul]

end StatadicFormal.Discrepancy.Transport
