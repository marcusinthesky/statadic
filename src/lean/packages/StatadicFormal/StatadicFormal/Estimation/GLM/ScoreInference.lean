module

public import StatadicFormal.Estimation.GMM.AffineInference
public import Mathlib.Analysis.Matrix.Normed
public import Mathlib.Analysis.Matrix.MeasurableSpace
public import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Architect

/-! Random-bread and score conditions for canonical GLM sandwich inference. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory Matrix ProbabilityTheory
open scoped Topology Matrix.Norms.Elementwise

namespace StatadicFormal.Estimation.GLM

variable {p : Type*} [Fintype p] [DecidableEq p]

omit [DecidableEq p] in
/-- The same offset in both predictors cancels from the natural-parameter displacement. -/
theorem score_secant {ι : Type*} [Fintype ι] [DecidableEq ι] (X : Matrix ι p ℝ)
    (y m₀ m₁ w : ι → ℝ) (b₀ b₁ : p → ℝ) (c : ℝ)
    (hmean : ∀ i, m₁ i - m₀ i = w i * (X *ᵥ (b₁ - b₀)) i) :
    (c • (Xᵀ * diagonal w * X)) *ᵥ (b₁ - b₀) =
      c • (Xᵀ *ᵥ (y - m₀)) - c • (Xᵀ *ᵥ (y - m₁)) := by
  rw [← smul_sub, ← Matrix.mulVec_sub, Matrix.smul_mulVec, ← Matrix.mulVec_mulVec,
    ← Matrix.mulVec_mulVec]
  congr 2
  ext i
  simp [Matrix.mulVec_diagonal, ← hmean, sub_sub_sub_cancel_left]

variable {Ω : Type*} [MeasurableSpace Ω]

private lemma inv_continuous {A : Matrix p p ℝ} (hr : A.det ≠ 0) :
    ContinuousAt (fun H : Matrix p p ℝ => H⁻¹) A :=
  continuousAt_matrix_inv _ (by simpa only [Ring.inverse_eq_inv'] using continuousAt_inv₀ hr)

/-- Fitted roots can be approximate, provided the scaled residual vanishes. -/
@[blueprint "STATADIC-GLMCOV-003"
  (statement := /-- Let $S_n,R_n$ be the mean scores at the true and fitted
  parameters, and $H_n$ the secant sensitivity between them. Assume full-rank
  sample sensitivities, $H_n\to H$ in probability for nonsingular $H$,
  $\sqrt N S_n\Rightarrow Z$, and $\sqrt N R_n\to0$ in probability.
  Then $\sqrt N(\hat\beta-\beta_0)\Rightarrow H^{-1}Z$. -/)
  (proof := /-- Rowwise secants give $H_n(\hat\beta-\beta_0)=S_n-R_n$.
  Slutsky removes the negligible fitted-score error. Inversion is continuous
  at the nonsingular sensitivity limit. Apply Slutsky again to the inverse
  sensitivity and the scaled score difference. -/)]
theorem glm_score_limit (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ν : Measure (p → ℝ)) [IsProbabilityMeasure ν] (b : ℕ → Ω → p → ℝ) (b₀ : p → ℝ)
    (S R : ℕ → Ω → p → ℝ) (Hn : ℕ → Ω → Matrix p p ℝ) (H : Matrix p p ℝ)
    (hH : H.det ≠ 0) (hr : ∀ n, ∀ᵐ ω ∂μ, (Hn n ω).det ≠ 0)
    (hm : ∀ n, AEMeasurable (fun ω i j => (Hn n ω)⁻¹ i j) μ)
    (hmR : ∀ n : ℕ, AEMeasurable (fun ω => Real.sqrt ((n : ℝ)+1) • R n ω) μ)
    (heq : ∀ n, ∀ᵐ ω ∂μ, Hn n ω *ᵥ (b n ω - b₀) = S n ω - R n ω)
    (hlln : TendstoInMeasure μ (fun n ω i j => Hn n ω i j) atTop (fun _ i j => H i j))
    (hres : TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n : ℝ)+1) • R n ω)
      atTop (fun _ => 0))
    (hclt : TendstoInDistribution (fun (n : ℕ) ω => Real.sqrt ((n : ℝ)+1) • S n ω)
      atTop id (fun _ => μ) ν) :
    TendstoInDistribution (fun (n : ℕ) ω => Real.sqrt ((n : ℝ)+1) • (b n ω - b₀))
      atTop (fun z => H⁻¹ *ᵥ z) (fun _ => μ) ν := by
  have hi : TendstoInMeasure μ (fun n ω i j => (Hn n ω)⁻¹ i j) atTop
      (fun _ i j => H⁻¹ i j) := StatadicFormal.Probability.tendstoInMeasure_map_at_constant
    (E := p → p → ℝ) (F := p → p → ℝ) μ (fun n ω i j => Hn n ω i j) H
    (fun q i j => (Matrix.of q)⁻¹ i j) hlln (inv_continuous hH)
    (fun n => (hm n).aestronglyMeasurable)
  have hd := TendstoInDistribution.continuous_comp_prodMk_of_tendstoInMeasure_const
    (continuous_fst.sub continuous_snd : Continuous (fun v : (p → ℝ) × (p → ℝ) => v.1 - v.2))
    hclt hres hmR
  have hcont : Continuous (fun q : (p → ℝ) × (p → p → ℝ) => Matrix.of q.2 *ᵥ q.1) :=
    continuous_snd.matrix_mulVec continuous_fst
  have h := TendstoInDistribution.continuous_comp_prodMk_of_tendstoInMeasure_const hcont hd hi hm
  apply h.congr _ (Filter.Eventually.of_forall fun z => by
    change H⁻¹ *ᵥ (z - 0) = _
    rw [sub_zero])
  intro n
  filter_upwards [hr n, heq n] with ω hunit he
  change (Hn n ω)⁻¹ *ᵥ (_ • S n ω - _ • R n ω) = _
  rw [← smul_sub, ← he, Matrix.mulVec_smul, Matrix.mulVec_mulVec,
    Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr hunit), Matrix.one_mulVec]

/-- Both HC0 and corrections tending to one give the same normalized plug-in limit. -/
@[blueprint "STATADIC-GLMCOV-004"
  (statement := /-- Suppose the reported sensitivity $A_n$, fitted-score meat $B_n$
  and multiplier $c_n$ converge jointly in probability to $A,B,1$ with nonsingular $A$.
  Then $c_n A_n^{-1} B_n A_n^{-T}\to A^{-1}BA^{-T}$ in probability.
  For the model-based choice $B=A=A^T$ the limit reduces to $A^{-1}$. -/)
  (proof := /-- Matrix inversion is continuous at $A$; multiplication,
  transposition and scalar scaling are continuous. Apply continuous mapping
  at the joint limit. For model information use symmetry and cancel the
  inverse with $A$. -/)]
theorem normalized_glm_sandwich_limit (μ : Measure Ω) [IsProbabilityMeasure μ]
    (An Bn : ℕ → Ω → Matrix p p ℝ) (c : ℕ → Ω → ℝ) (A B : Matrix p p ℝ) (ha : A.det ≠ 0)
    (hlln : TendstoInMeasure μ
      (fun n ω => ((fun i j => An n ω i j), (fun i j => Bn n ω i j), c n ω))
      atTop (fun _ => ((fun i j => A i j), (fun i j => B i j), (1 : ℝ))))
    (hm : ∀ n, AEStronglyMeasurable
      (fun ω i j => (c n ω • ((An n ω)⁻¹ * Bn n ω * ((An n ω)⁻¹)ᵀ)) i j) μ) :
    TendstoInMeasure μ
      (fun n ω i j => (c n ω • ((An n ω)⁻¹ * Bn n ω * ((An n ω)⁻¹)ᵀ)) i j)
      atTop (fun _ i j => (A⁻¹ * B * (A⁻¹)ᵀ) i j) ∧
      (Aᵀ = A → B = A → A⁻¹ * B * (A⁻¹)ᵀ = A⁻¹) := by
  let data := (p → p → ℝ) × (p → p → ℝ) × ℝ
  let f : data → (p → p → ℝ) := fun q i j =>
    (q.2.2 • ((Matrix.of q.1)⁻¹ * Matrix.of q.2.1 * ((Matrix.of q.1)⁻¹)ᵀ)) i j
  have hi := (inv_continuous ha).tendsto.comp
    (continuous_fst.tendsto ((A,B,(1 : ℝ)) : data))
  have hc : ContinuousAt f (A,B,(1 : ℝ)) := by
    have htr : Tendsto (fun q : data => ((Matrix.of q.1)⁻¹)ᵀ) (𝓝 (A,B,(1 : ℝ))) (𝓝 (A⁻¹)ᵀ) :=
      (continuous_id.matrix_transpose.tendsto A⁻¹).comp hi
    have hprod : Tendsto (fun q : data => (Matrix.of q.1)⁻¹ * Matrix.of q.2.1 *
        ((Matrix.of q.1)⁻¹)ᵀ) (𝓝 (A,B,(1 : ℝ))) (𝓝 (A⁻¹ * B * (A⁻¹)ᵀ)) :=
      (hi.mul ((continuous_fst.comp continuous_snd).tendsto _)).mul htr
    apply tendsto_pi_nhds.mpr
    intro i
    apply tendsto_pi_nhds.mpr
    intro j
    exact ((continuous_snd.comp continuous_snd).tendsto _).mul
      (tendsto_pi_nhds.mp (tendsto_pi_nhds.mp hprod i) j)
  constructor
  · have hl := StatadicFormal.Probability.tendstoInMeasure_map_at_constant
      (E := data) (F := p → p → ℝ) μ _ (A,B,(1 : ℝ)) f hlln hc hm
    change TendstoInMeasure μ
      (fun n ω i j => (c n ω • ((An n ω)⁻¹ * Bn n ω * ((An n ω)⁻¹)ᵀ)) i j) atTop
      (fun _ i j => ((1 : ℝ) • (A⁻¹ * B * (A⁻¹)ᵀ) : Matrix p p ℝ) i j) at hl
    simpa only [one_smul] using hl
  · rintro h rfl
    rw [Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr ha), Matrix.one_mul,
      Matrix.transpose_nonsing_inv, h]

/-- The Gaussian reference covariance for a regular fitted GLM root. -/
@[blueprint "STATADIC-GLMCOV-005"
  (statement := /-- For a Gaussian limiting score $Z$ with covariance $B$ and
  nonsingular sensitivity $A$, the coefficient limit $A^{-1}Z$ is Gaussian
  with covariance $A^{-1} B A^{-T}$. Together with the fitted sandwich limit
  this justifies the covariance used for asymptotic contrasts. -/)
  (proof := /-- Apply the Gaussian linear-response identity with response
  matrix $-A^{-1}$. The sign in the score response cancels, and the two minus
  signs in its covariance cancel as well. -/)]
theorem gaussian_root_covariance (A B : Matrix p p ℝ) (ν : Measure (p → ℝ))
    [IsGaussian ν] (hs : ∀ i, MemLp (fun z : p → ℝ => z i) 2 ν)
    (hcov : ∀ i j, covariance (fun z => z i) (fun z => z j) ν = B i j) :
    IsGaussian (ν.map (fun z => A⁻¹ *ᵥ z)) ∧
      ∀ i j, covariance (fun z => (A⁻¹ *ᵥ z) i) (fun z => (A⁻¹ *ᵥ z) j) ν =
        (A⁻¹ * B * (A⁻¹)ᵀ) i j := by
  simpa only [Matrix.neg_mulVec, Pi.neg_apply, neg_neg, Matrix.transpose_neg,
    neg_mul, mul_neg] using GMM.gaussian_response (-A⁻¹) ν B hs hcov

end StatadicFormal.Estimation.GLM
