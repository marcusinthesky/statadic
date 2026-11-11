module

public import StatadicFormal.Probability.FiniteExtensions
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-! Distinguish weight concentration from Monte Carlo accuracy. -/
@[expose] public section
noncomputable section
namespace StatadicFormal.Optimization

/-- The effective count used by a Gaussian integration replicate lies in [1,n]. -/
@[blueprint "STATADIC-INTEGRATION-001"
  (statement := /-- If $n$ simulation weights are nonnegative and
  $S_2=\sum w_i^2>0$, then $1\le (\sum w_i)^2/S_2\le n$.
  Equality at $n$ requires all weights to coincide. -/)
  (proof := /-- Each $w_i$ is at most the total weight, so
  $\sum w_i^2\le(\sum w_i)^2$. Cauchy--Schwarz against the constant vector
  gives the upper bound. At equality the sum of squared deviations about
  the mean vanishes, hence each deviation is zero. -/)]
theorem effective_count_bounds {ι : Type*} [Fintype ι] [Nonempty ι]
    (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) (hs : 0 < ∑ i, w i ^ 2) :
    let s := ∑ i, w i
    let n : ℝ := Fintype.card ι
    1 ≤ s ^ 2 / (∑ i, w i ^ 2) ∧ s ^ 2 / (∑ i, w i ^ 2) ≤ n ∧
      (s ^ 2 / (∑ i, w i ^ 2) = n → ∀ i, w i = s / n) := by
  let s := ∑ i, w i
  let n : ℝ := Fintype.card ι
  have hn : 0 < n := Nat.cast_pos.mpr Fintype.card_pos
  have lower : ∑ i, w i ^ 2 ≤ s ^ 2 := by
    calc
      _ ≤ ∑ i, w i * s := Finset.sum_le_sum fun i _ => by
        rw [pow_two]
        exact mul_le_mul_of_nonneg_left (Finset.single_le_sum (fun j _ => hw j)
          (Finset.mem_univ i)) (hw i)
      _ = _ := by rw [← Finset.sum_mul, pow_two]
  have upper : s ^ 2 ≤ n * ∑ i, w i ^ 2 := by
    simpa [s, n] using Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ : ι => (1 : ℝ)) w
  refine ⟨(one_le_div hs).mpr lower, (div_le_iff₀ hs).mpr upper, ?_⟩
  intro heq
  have hvar : (∑ i, (w i - s / n) ^ 2) = 0 := by
    have he : s ^ 2 = n * ∑ i, w i ^ 2 := (div_eq_iff hs.ne').mp heq
    simp_rw [sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib,
      ← Finset.sum_mul, ← Finset.mul_sum]
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    change (∑ i, w i ^ 2) - 2 * s * (s / n) + n * (s / n) ^ 2 = 0
    field_simp [hn.ne']
    nlinarith [congrArg (fun t : ℝ => t * n) he]
  intro i
  exact sub_eq_zero.mp (sq_eq_zero_iff.mp ((Finset.sum_eq_zero_iff_of_nonneg
    (fun j _ => sq_nonneg (w j - s / n))).mp hvar i (Finset.mem_univ i)))

end StatadicFormal.Optimization
