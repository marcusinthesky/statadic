module

public import Architect
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Ring

/-!
# GLM working normal equations

At a fixed parameter, use the current inverse-link derivative and variance for
each row. The weighted least-squares normal equations in the proposed predictor
increment are the Fisher scoring equations. This is algebra at one iterate.
-/

@[expose] public section
noncomputable section

open scoped BigOperators

namespace StatadicFormal.Estimation.GLM

variable {ι : Type*} [Fintype ι]

/-- The working weight at unit dispersion, given the mean derivative and row variance. -/
def workingWeight (d v : ℝ) : ℝ := d ^ 2 / v

/-- The adjusted response on the link scale. -/
def workingResponse (η y μ d : ℝ) : ℝ := η + (y - μ) / d

/-- The normal equation for a design column is the score minus its information update. -/
@[blueprint "STATADIC-GLM-001"
  (statement := /-- Write $d_i=d\mu_i/d\eta_i\ne0$, $V_i>0$,
  $w_i=d_i^2/V_i$, and $z_i=\eta_i+(y_i-\mu_i)/d_i$ at the current iterate.
  For any design column $x$ and predictor increment $s$,
  $\sum_i x_i w_i(z_i-\eta_i-s_i)=
  \sum_i x_i(y_i-\mu_i)d_i/V_i-\sum_i x_i w_i s_i$. -/)
  (proof := /-- Subtract the old linear predictor from the working response.
  Multiplying $(y_i-\mu_i)/d_i$ by $d_i^2/V_i$ gives
  $(y_i-\mu_i)d_i/V_i$. Subtract the increment contribution $w_i s_i$,
  multiply by the design entry and sum. With $s=X\Delta$ this matches
  each weighted normal equation to the corresponding Fisher scoring equation. -/)]
theorem working_normal_eq_score (x η y μ d v s : ι → ℝ)
    (hd : ∀ i, d i ≠ 0) (hv : ∀ i, 0 < v i) :
    (∑ i, x i * workingWeight (d i) (v i) * (workingResponse (η i) (y i) (μ i) (d i)
      - η i - s i)) =
      (∑ i, x i * (y i - μ i) * d i / v i) - ∑ i, x i * workingWeight (d i) (v i) * s i := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  unfold workingWeight workingResponse
  field_simp [hd i, (hv i).ne']
  ring

end StatadicFormal.Estimation.GLM
