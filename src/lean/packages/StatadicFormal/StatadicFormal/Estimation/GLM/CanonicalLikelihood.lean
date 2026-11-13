module

public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Probability.Moments.Variance
public import Architect
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! Canonical row likelihoods in STATADIC's grouped-binomial and Poisson GLMs. -/
@[expose] public section
noncomputable section
open MeasureTheory ProbabilityTheory
namespace StatadicFormal.Estimation.GLM

/-- Probability of success at a finite canonical predictor. -/
def successProbability (η : ℝ) : ℝ := Real.exp η / (1 + Real.exp η)

/-- Negative softplus evaluated as a real identity; the Python likelihood uses both tails. -/
theorem softplus_neg (η : ℝ) :
    Real.log (1 + Real.exp (-η)) = Real.log (1 + Real.exp η) - η := by
  have he := Real.exp_pos η
  rw [Real.exp_neg, show 1 + (Real.exp η)⁻¹ = (1 + Real.exp η) / Real.exp η by field_simp; ring]
  rw [Real.log_div (by positivity) he.ne', Real.log_exp]

private theorem softplus_derivative (η : ℝ) :
    HasDerivAt (fun t => Real.log (1 + Real.exp t)) (successProbability η) η := by
  simpa [successProbability] using ((Real.hasDerivAt_exp η).const_add (1 : ℝ)).log (by positivity)

private theorem success_derivative (η : ℝ) :
    HasDerivAt successProbability (successProbability η * (1 - successProbability η)) η := by
  have h := (Real.hasDerivAt_exp η).div ((Real.hasDerivAt_exp η).const_add (1 : ℝ))
    (by positivity)
  convert h using 1
  · rfl
  · unfold successProbability
    have hn : 0 < 1 + Real.exp η := by positivity
    field_simp

/-- Score and curvature for the actual grouped-binomial logit criterion. -/
@[blueprint "STATADIC-GLM-002"
  (statement := /-- For a grouped count $y$ on $m$ trials let
  $p=\exp(\eta)/(1+\exp(\eta))$. For the implemented log likelihood
  $C-y\log(1+e^{-\eta})-(m-y)\log(1+e^\eta)$ the score is $y-mp$
  and its derivative is $-mp(1-p)$. -/)
  (proof := /-- Replace the negative-predictor softplus by the positive one
  minus the predictor. The likelihood becomes $C+y\eta-m\log(1+e^\eta)$.
  Differentiate it, then apply the quotient rule to $p$. -/)]
theorem binomial_score_curvature (y m C η : ℝ) :
    HasDerivAt (fun t => C - y * Real.log (1 + Real.exp (-t)) -
      (m - y) * Real.log (1 + Real.exp t)) (y - m * successProbability η) η ∧
    HasDerivAt (fun t => y - m * successProbability t)
      (-(m * successProbability η * (1 - successProbability η))) η := by
  have hrewrite : (fun t => C - y * Real.log (1 + Real.exp (-t)) -
      (m-y) * Real.log (1+Real.exp t)) =
      fun t => C + y*t - m*Real.log (1+Real.exp t) := by
    funext t
    rw [softplus_neg]
    ring
  rw [hrewrite]
  constructor
  · simpa [Pi.sub_def] using (((hasDerivAt_id η).const_mul y).const_add C).sub
      ((softplus_derivative η).const_mul m)
  · convert ((success_derivative η).const_mul m).const_sub y using 1
    ring

/-- The native log rate already includes the log-exposure offset. -/
@[blueprint "STATADIC-GLM-003"
  (statement := /-- For $\ell(\eta)=y\eta-\exp(\eta)-C$ the score is
  $y-\exp(\eta)$ and its derivative is $-\exp(\eta)$.
  An offset held fixed changes the mean through $\eta$ and has no coefficient derivative. -/)
  (proof := /-- Differentiate the linear count term and the exponential
  cumulant. The log-factorial term is constant in the predictor. -/)]
theorem poisson_score_curvature (y C η : ℝ) :
    HasDerivAt (fun t => y*t - Real.exp t - C) (y - Real.exp η) η ∧
    HasDerivAt (fun t => y - Real.exp t) (-Real.exp η) η := by
  exact ⟨by simpa using ((((hasDerivAt_id η).const_mul y).sub (Real.hasDerivAt_exp η)).sub_const C),
    (Real.hasDerivAt_exp η).const_sub y⟩

/-- The variance of a score contrast differs from working information by the
weighted sum of row variance discrepancies. -/
@[blueprint "STATADIC-GLMCOV-002"
  (statement := /-- On a fixed design, let $a_i=x_i^Tu$ be a contrast direction
  and $S=\sum_i a_i(Y_i-\mu_i)$. If the row residuals have mean zero, are
  square integrable and uncorrelated across rows, with actual variances $v_i$,
  then $E S=0$ and $\mathrm{Var}(S)=\sum_i a_i^2 v_i$.
  A working variance $w_i$ leaves discrepancy $\sum_i a_i^2(v_i-w_i)$. -/)
  (proof := /-- Integrate the finite weighted sum. Expand its variance as a
  double sum of covariances, take each scalar factor outside and discard
  off-diagonal terms. Subtract the working information in the same direction. -/)]
theorem score_information_gap {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι] [DecidableEq ι]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (ε : ι → Ω → ℝ) (a v w : ι → ℝ)
    (hm : ∀ i, MemLp (ε i) 2 μ) (hz : ∀ i, ∫ x, ε i x ∂μ = 0)
    (hc : ∀ i j, covariance (ε i) (ε j) μ = if i = j then v i else 0) :
    let S := fun x => ∑ i, a i * ε i x
    (∫ x, S x ∂μ) = 0 ∧ variance S μ - ∑ i, a i ^ 2 * w i = ∑ i, a i ^ 2 * (v i - w i) := by
  dsimp only
  constructor
  · rw [integral_finsetSum _ fun i _ => ((hm i).integrable (by norm_num)).const_mul (a i)]
    simp [integral_const_mul, hz]
  · rw [variance_fun_sum (fun i => (hm i).const_mul (a i)), ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    simp [covariance_const_mul_left, covariance_const_mul_right, hc, mul_ite, pow_two]
    ring

end StatadicFormal.Estimation.GLM
