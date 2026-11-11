module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Architect
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Identifiability and the zero-count boundary in an intercept-only Poisson fit with offsets. -/
@[expose] public section
noncomputable section
namespace StatadicFormal.Estimation

/-- Intercept-dependent terms of the Poisson log likelihood after aggregating positive exposures. -/
def poissonRateLogLik (Y E b : ℝ) := Y * b - E * Real.exp b

/-- Aggregate the row contributions without changing differences in the intercept likelihood. -/
theorem poisson_exposure_reduction {ι : Type*} [Fintype ι] (y e k : ι → ℝ)
    (he : ∀ i, 0 < e i) (b : ℝ) :
    (∑ i, (y i * (b + Real.log (e i)) - Real.exp (b + Real.log (e i)) - k i)) =
      poissonRateLogLik (∑ i, y i) (∑ i, e i) b + ∑ i, (y i * Real.log (e i) - k i) := by
  simp_rw [Real.exp_add, Real.exp_log (he _), mul_add]
  unfold poissonRateLogLik
  rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- A zero total count drives the optimum to the boundary even when all exposures are positive. -/
@[blueprint "STATADIC-POISSON-001"
  (statement := /-- In an intercept-only Poisson model with positive exposures, write
  $Y=\sum y_i$, $E=\sum e_i>0$ and $\ell(b)=Yb-Ee^b$ up to an additive constant.
  For $Y>0$, $b_*=\log(Y/E)$ is the unique maximizer. If $Y=0$, lowering any
  finite intercept by one strictly increases the likelihood. -/)
  (proof := /-- The log offset converts $\exp(b+\log e_i)$ to $e_i\exp b$.
  At $b_*$ the likelihood difference for a change $h$ is
  $Y(h+1-\exp h)$, strictly negative for $h\ne0$.
  With no observed counts the remaining term $-E\exp b$ is strictly decreasing. -/)]
theorem poisson_intercept_boundary (Y E : ℝ) (he : 0 < E) :
    (0 < Y → ∀ h : ℝ, h ≠ 0 →
      poissonRateLogLik Y E (Real.log (Y / E) + h) < poissonRateLogLik Y E (Real.log (Y / E))) ∧
    (Y = 0 → ∀ b : ℝ, poissonRateLogLik Y E b < poissonRateLogLik Y E (b - 1)) := by
  constructor
  · intro hy h hh
    have hnorm : E * (Y / E) = Y := mul_div_cancel₀ _ he.ne'
    simp only [poissonRateLogLik, Real.exp_add, Real.exp_log (div_pos hy he), ← mul_assoc, hnorm]
    nlinarith [mul_lt_mul_of_pos_left (Real.add_one_lt_exp hh) hy]
  · rintro rfl b
    have h := mul_lt_mul_of_pos_left (Real.exp_lt_exp.mpr (show b - 1 < b by linarith)) he
    simpa [poissonRateLogLik] using neg_lt_neg h

end StatadicFormal.Estimation
