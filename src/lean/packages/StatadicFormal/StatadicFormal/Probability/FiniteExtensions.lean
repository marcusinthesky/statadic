module

public import StatadicFormal.Estimation.Linear.FiniteLaw
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
# Finite-law expectation and variance support

Reuses the repository's `FiniteLaw`; no replacement probability model.
Exact real arithmetic and finite sample spaces only. These are supporting
identities, not asymptotic or implementation-correctness guarantees.
-/
@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Probability.FiniteLaw
variable {Ω : Type*} [Fintype Ω] (P : FiniteLaw Ω)

/-- Expectation preserves nonnegativity. -/
theorem research_mean_nonneg (f : Ω → ℝ) (hf : ∀ ω, 0 ≤ f ω) :
    0 ≤ P.mean f := by
  unfold mean
  exact Finset.sum_nonneg (fun ω _ => mul_nonneg (P.nonneg ω) (hf ω))

/-- Expectation preserves pointwise order. -/
theorem research_mean_mono (f g : Ω → ℝ) (h : ∀ ω, f ω ≤ g ω) :
    P.mean f ≤ P.mean g := by
  unfold mean
  exact Finset.sum_le_sum (fun ω _ => mul_le_mul_of_nonneg_left (h ω) (P.nonneg ω))

/-- Every finite-law variance is nonnegative. -/
theorem research_variance_nonneg (f : Ω → ℝ) : 0 ≤ P.variance f := by
  exact P.research_mean_nonneg _ (fun ω => sq_nonneg (f ω - P.mean f))

/-- Centering has zero expectation. -/
theorem research_mean_center (f : Ω → ℝ) :
    P.mean (fun ω => f ω - P.mean f) = 0 := by
  rw [P.mean_sub, P.mean_const]
  ring

/-- Scaling a statistic scales its variance quadratically. -/
theorem research_variance_scale (c : ℝ) (f : Ω → ℝ) :
    P.variance (fun ω => c * f ω) = c ^ 2 * P.variance f := by
  unfold variance
  rw [P.mean_smul]
  have heq : (fun ω => (c * f ω - c * P.mean f) ^ 2) =
      (fun ω => c ^ 2 * (f ω - P.mean f) ^ 2) := by
    funext ω
    ring
  rw [heq, P.mean_smul]

/-- For a centered statistic variance is its second moment. -/
theorem research_variance_of_centered (f : Ω → ℝ) (h : P.mean f = 0) :
    P.variance f = P.mean (fun ω => f ω ^ 2) := by
  simp only [variance, h, sub_zero]

/-- Affine expectation, useful for bias calculations. -/
theorem research_mean_affine (a b : ℝ) (f : Ω → ℝ) :
    P.mean (fun ω => a * f ω + b) = a * P.mean f + b := by
  rw [P.mean_add, P.mean_smul, P.mean_const]

/-- A deterministic additive shift does not affect variance. -/
theorem research_variance_affine (a b : ℝ) (f : Ω → ℝ) :
    P.variance (fun ω => a * f ω + b) = a ^ 2 * P.variance f := by
  rw [P.variance_add_const, P.research_variance_scale]

end StatadicFormal.Probability.FiniteLaw
