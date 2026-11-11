module

public import StatadicFormal.Estimation.Survival.DelayedEntry
public import Mathlib.Probability.Distributions.Exponential
public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity

/-! The two parametric survivor and original-duration event-density formulas in the registry. -/
@[expose] public section
noncomputable section
open MeasureTheory ProbabilityTheory Set
namespace StatadicFormal.Estimation.Survival

/-- Evaluate the Weibull survivor from the normalized unit-exponential cumulative hazard,
then differentiate it in the original duration coordinate. -/
@[blueprint "STATADIC-SURVIVAL-004"
  (statement := /-- Let $t>0$, log scale $\eta$, log shape $a$, and
  $z=e^a(\log t-\eta)$. A unit-exponential cumulative-hazard threshold
  gives $S(t)=1-F_{\rm Exp(1)}(e^z)=\exp(-e^z)$.
  The derivative of this survivor is the negative of the density with
  log value $a-\log t+z-e^z$. -/)
  (proof := /-- Evaluate the unit-exponential survival at its positive
  cumulative hazard $e^z$. Differentiate through the hazard and log time,
  using $z'(t)=e^a/t$, then collect the exponential factors. -/)]
theorem weibull_aft_factors (t η a : ℝ) (ht : 0 < t) :
    let z := Real.exp a * (Real.log t - η)
    let S := fun u => Real.exp (-Real.exp (Real.exp a * (Real.log u - η)))
    1 - cdf (expMeasure 1) (Real.exp z) = S t ∧
      HasDerivAt S (-Real.exp (a - Real.log t + z - Real.exp z)) t := by
  dsimp only
  constructor
  · simp [cdf_expMeasure_eq (by norm_num : (0 : ℝ) < 1), (Real.exp_pos _).le]
  · let z := Real.exp a * (Real.log t - η)
    have hz := ((Real.hasDerivAt_log ht.ne').sub_const η).const_mul (Real.exp a)
    have h := hz.exp.neg.exp
    change HasDerivAt (fun u => Real.exp (-Real.exp (Real.exp a * (Real.log u - η))))
      (Real.exp (-Real.exp z) * -(Real.exp z * (Real.exp a * t⁻¹))) t at h
    apply h.congr_deriv
    rw [Real.exp_sub, Real.exp_add, Real.exp_sub, Real.exp_log ht, Real.exp_neg]
    ring

private theorem normal_cdf_deriv (z : ℝ) :
    HasDerivAt (fun y => cdf (gaussianReal 0 1) y) (gaussianPDFReal 0 1 z) z := by
  let f := gaussianPDFReal 0 1
  have cont : Continuous f := by rw [show f = _ from gaussianPDFReal_def 0 1]; fun_prop
  have repr (y : ℝ) : cdf (gaussianReal 0 1) y = ∫ t in Iic y, f t := by
    rw [cdf_eq_real, measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero]
    exact ENNReal.toReal_ofReal (integral_nonneg (fun x => gaussianPDFReal_nonneg 0 1 x))
  have eqn : (fun y => cdf (gaussianReal 0 1) y) =
      (fun y => (∫ t in Iic z, f t) + ∫ t in z..y, f t) := by
    funext y
    rw [repr, ← intervalIntegral.integral_Iic_sub_Iic
      (integrable_gaussianPDFReal 0 1).integrableOn (integrable_gaussianPDFReal 0 1).integrableOn]
    ring
  rw [eqn]
  exact (intervalIntegral.integral_hasDerivAt_right (cont.intervalIntegrable _ _)
    cont.stronglyMeasurable.stronglyMeasurableAtFilter cont.continuousAt).const_add _

/-- The lognormal law is obtained by exponentiating a Gaussian. The reflected
standard normal parameterization gives the native lower-CDF call directly. -/
@[blueprint "STATADIC-SURVIVAL-005"
  (statement := /-- For standard normal $Z$, write $T=\exp(\eta-e^a Z)$,
  which has lognormal location $\eta$ and log standard deviation $a$.
  At $t>0$ let $z=(\log t-\eta)e^{-a}$. Then
  $P(T\ge t)=\Phi(-z)$, and $-S'(t)$ is positive with log value
  $-\log t-a-z^2/2-\log(2\pi)/2$. -/)
  (proof := /-- The event $T\ge t$ is $Z\le -z$ by monotonicity of the
  exponential. Differentiate the normal CDF by the fundamental theorem of
  calculus, then differentiate its log-time argument. Insert the standard
  Gaussian density and split the logarithms of its normalizing constant. -/)]
theorem lognormal_aft_factors (t η a : ℝ) (ht : 0 < t) :
    let z := (Real.log t - η) * Real.exp (-a)
    let S := fun u => cdf (gaussianReal 0 1) (-((Real.log u - η) * Real.exp (-a)))
    (gaussianReal 0 1).real {v | t ≤ Real.exp (η - Real.exp a * v)} = S t ∧
      HasDerivAt S (-Real.exp (-Real.log t - a - z ^ 2 / 2 - Real.log (2 * Real.pi) / 2)) t := by
  let z := (Real.log t - η) * Real.exp (-a)
  dsimp only
  constructor
  · rw [cdf_eq_real]
    congr 1
    ext v
    change t ≤ Real.exp (η - Real.exp a * v) ↔ v ≤ -((Real.log t - η) * Real.exp (-a))
    rw [← Real.log_le_iff_le_exp ht, Real.exp_neg]
    constructor
    · intro h
      have h' : v ≤ -(Real.log t - η) / Real.exp a :=
        (le_div_iff₀ (Real.exp_pos a)).mpr (by linarith [mul_comm v (Real.exp a)])
      simpa only [div_eq_mul_inv, neg_mul] using h'
    · intro h
      have h' : v ≤ -(Real.log t - η) / Real.exp a := by simpa only [div_eq_mul_inv, neg_mul] using h
      have := (le_div_iff₀ (Real.exp_pos a)).mp h'
      linarith [mul_comm v (Real.exp a)]
  · have hz := (((Real.hasDerivAt_log ht.ne').sub_const η).mul_const (Real.exp (-a))).neg
    have h := (normal_cdf_deriv (-z)).comp t hz
    let den := gaussianPDFReal 0 1 (-z) * (t⁻¹ * Real.exp (-a))
    have hpos : 0 < den := mul_pos (gaussianPDFReal_pos _ _ _ one_ne_zero)
      (mul_pos (inv_pos.mpr ht) (Real.exp_pos _))
    have logden : Real.log den = -Real.log t - a - z ^ 2 / 2 - Real.log (2 * Real.pi) / 2 := by
      dsimp only [den]
      rw [Real.log_mul (gaussianPDFReal_pos _ _ _ one_ne_zero).ne' (by positivity),
        Real.log_mul (inv_ne_zero ht.ne') (Real.exp_pos _).ne', Real.log_inv, Real.log_exp,
        gaussianPDFReal_def]
      simp only [NNReal.coe_one, mul_one, sub_zero, neg_sq]
      rw [Real.log_mul (by positivity) (Real.exp_pos _).ne', Real.log_inv,
        Real.log_sqrt (by positivity), Real.log_exp]
      ring
    change HasDerivAt _ (-Real.exp (-Real.log t - a - z ^ 2 / 2 - Real.log (2 * Real.pi) / 2)) t
    rw [← logden, Real.exp_log hpos]
    simpa only [Function.comp_def, den, mul_neg, Pi.neg_apply] using h

end StatadicFormal.Estimation.Survival
