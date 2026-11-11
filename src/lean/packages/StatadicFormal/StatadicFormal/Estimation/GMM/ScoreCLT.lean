module

public import Mathlib.Probability.CentralLimitTheorem
public import Mathlib.Probability.CramerWold
public import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Distributions.Gaussian.Fernique
public import Architect

/-! Sufficient sampling conditions for the multivariate GMM score CLT. -/
@[expose] public section
noncomputable section
open Matrix MeasureTheory ProbabilityTheory Filter
open scoped RealInnerProductSpace
namespace StatadicFormal.Estimation.GMM

/-- A primitive iid score CLT, retaining arbitrary covariance between moments in a row. -/
@[blueprint "STATADIC-GMM-011"
  (statement := /-- Suppose finite-dimensional true-parameter moment vectors
  $g_i$ are iid, have finite second moment and mean zero. Let $S$ be their
  covariance, expressed as $\operatorname{Var}(t^Tg_i)=t^T S t$ for all
  contrasts $t$. Then $n^{-1/2}\sum_{i=1}^n g_i$ converges in law to
  $N(0,S)$, including singular $S$. -/)
  (proof := /-- Project every row onto a fixed contrast. The scalar projected
  variables remain iid, square integrable and centered, with the specified
  variance. Apply the scalar iid CLT and identify the contrast law of the
  centered multivariate Gaussian by its covariance form. Cramér–Wold gives
  the vector limit. -/)]
theorem iid_score_limit {Ω m : Type*} [MeasurableSpace Ω] [Fintype m] [DecidableEq m]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → EuclideanSpace ℝ m)
    (S : Matrix m m ℝ) (hs : S.PosSemidef) (h2 : MemLp (X 0) 2 μ)
    (h0 : ∫ ω, X 0 ω ∂μ = 0) (hind : iIndepFun X μ)
    (hid : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (hcov : ∀ t : EuclideanSpace ℝ m, Var[fun ω => ⟪t, X 0 ω⟫; μ] = t ⬝ᵥ (S *ᵥ t)) :
    TendstoInDistribution (fun (n : ℕ) ω => (Real.sqrt n)⁻¹ • ∑ i ∈ Finset.range n, X i ω)
      atTop id (fun _ => μ) (multivariateGaussian 0 S) := by
  let ν := multivariateGaussian (0 : EuclideanSpace ℝ m) S
  have hm n := (hid n).aemeasurable_fst
  apply TendstoInDistribution.of_inner (by fun_prop)
  · intro n
    exact (Finset.aemeasurable_fun_sum (Finset.range n) fun i _ => hm i).const_smul (Real.sqrt n)⁻¹
  intro t
  let f := fun u : EuclideanSpace ℝ m => ⟪t, u⟫
  have hf : Measurable f := by dsimp [f]; fun_prop
  have hcent : ∫ ω, f (X 0 ω) ∂μ = 0 := by
    rw [show (∫ ω, f (X 0 ω) ∂μ) = _ from integral_inner (h2.integrable (by norm_num)) t, h0,
      inner_zero_right]
  have hmean : ∫ u, f u ∂ν = 0 := by
    rw [show (∫ u, f u ∂ν) = _ from integral_inner (IsGaussian.integrable_fun_id (μ := ν)) t]
    simp [ν]
  have hv : Var[f;ν] = t ⬝ᵥ (S *ᵥ t) := by
    rw [show Var[f;ν] = _ from (covarianceBilin_self (μ := ν) IsGaussian.memLp_two_id t).symm,
      covarianceBilin_multivariateGaussian hs]
  have hg : HasLaw f (gaussianReal 0 (Var[fun ω => f (X 0 ω);μ]).toNNReal) ν := by
    refine ⟨hf.aemeasurable, ?_⟩
    rw [show ν.map f = _ from IsGaussian.map_eq_gaussianReal (μ := ν) (innerSL ℝ t)]
    change gaussianReal (∫ u, f u ∂ν) (Var[f;ν]).toNNReal = _
    rw [hmean, hv, hcov t]
  let Y (i : ℕ) := f ∘ X i
  have hi : iIndepFun Y μ := hind.comp (fun _ => f) (fun _ => hf)
  have ident : ∀ i, IdentDistrib (Y i) (Y 0) μ μ := fun i => (hid i).comp hf
  have h := tendstoInDistribution_inv_sqrt_mul_sum_sub (X := Y) hg (h2.const_inner t) hi ident
  convert h using 1
  · funext n ω
    rw [real_inner_comm t _, inner_smul_right, inner_sum]
    change _ = (Real.sqrt n)⁻¹ * (_ - _ * (∫ ω, f (X 0 ω) ∂μ))
    rw [hcent, mul_zero, sub_zero]
    rfl
  · funext x
    exact real_inner_comm _ _

end StatadicFormal.Estimation.GMM
