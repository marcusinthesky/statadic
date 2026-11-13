module

public import StatadicFormal.Probability.Finite
public import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Finite expectation identities used as supporting algebra for linear models. -/
open scoped BigOperators
@[expose] public section

noncomputable section
namespace StatadicFormal.Probability.FiniteLaw
variable {Ω : Type*} [Fintype Ω] (P : FiniteLaw Ω)
@[simp] theorem mean_const (c : ℝ) : P.mean (fun _ => c) = c := by
  unfold mean
  rw [← Finset.sum_mul, P.total, one_mul]

/-- Additivity of finite expectation. -/
theorem mean_add (f g : Ω → ℝ) :
    P.mean (fun ω => f ω + g ω) = P.mean f + P.mean g := by
  simp [mean, mul_add, Finset.sum_add_distrib]

/-- Subtractivity of finite expectation. -/
theorem mean_sub (f g : Ω → ℝ) :
    P.mean (fun ω => f ω - g ω) = P.mean f - P.mean g := by
  simp [mean, mul_sub, Finset.sum_sub_distrib]

/-- Pull a deterministic coefficient outside expectation. -/
theorem mean_smul (c : ℝ) (f : Ω → ℝ) :
    P.mean (fun ω => c * f ω) = c * P.mean f := by
  unfold mean
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro ω _
  ring

/-- Exchange two finite sums; no independence is required. -/
theorem mean_sum {ι : Type*} [Fintype ι] (f : ι → Ω → ℝ) :
    P.mean (fun ω => ∑ i, f i ω) = ∑ i, P.mean (f i) := by
  classical
  simp only [mean, Finset.mul_sum]
  rw [Finset.sum_comm]

theorem variance_add_const (f : Ω → ℝ) (c : ℝ) :
    P.variance (fun ω => f ω + c) = P.variance f := by
  unfold variance
  rw [mean_add, mean_const]
  congr 1
  funext ω
  ring

end StatadicFormal.Probability.FiniteLaw
