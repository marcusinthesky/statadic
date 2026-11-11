module

public import StatadicFormal.Estimation.GMM.AffineLimit
public import StatadicFormal.Probability.ContinuousMapping
public import Mathlib.Analysis.Matrix.Normed
public import Mathlib.Analysis.Matrix.MeasurableSpace
public import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
public import Mathlib.Probability.Distributions.Gaussian.Basic
public import Mathlib.Probability.Moments.Covariance
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.LinearAlgebra.Matrix.ToLin

/-! Statistical limits for the finite-dimensional affine GMM problem. -/

@[expose] public section
noncomputable section

open Matrix MeasureTheory ProbabilityTheory Filter
open scoped Topology Matrix.Norms.Elementwise
namespace StatadicFormal.Estimation.GMM

variable {Ω m p : Type*} [MeasurableSpace Ω] [Fintype m] [Fintype p] [DecidableEq p]

/-- For affine moments the LLN acts on a finite list of coefficients. -/
@[blueprint "STATADIC-GMM-006"
  (statement := /-- Let $\bar g_n(b)=a_n+D_n b$. Suppose the finite arrays
  $(a_n,D_n,W_n)$ converge jointly in probability to $(a,D,W)$, the
  population moment satisfies $a+Db_0=0$, and $D^TWD$ is invertible.
  Then the measurable affine GMM coefficient $-L_n a_n$ converges in probability to $b_0$.
  With a symmetric positive definite weight this is the identified weighted normal-equation solution. -/)
  (proof := /-- Matrix products and transposes are continuous, and inversion
  is continuous at the nonsingular limiting Gram matrix. The continuous
  mapping principle at the constant joint limit gives convergence to $-La$.
  The identity $LD=I$ and the population moment identify this limit as $b_0$. -/)]
theorem coefficient_consistent (μ : Measure Ω) [IsProbabilityMeasure μ]
    (an : ℕ → Ω → m → ℝ) (Dn : ℕ → Ω → Matrix m p ℝ) (Wn : ℕ → Ω → Matrix m m ℝ)
    (a : m → ℝ) (D : Matrix m p ℝ) (W : Matrix m m ℝ) (b₀ : p → ℝ)
    (hlln : TendstoInMeasure μ (fun n ω => (an n ω, (fun i j => Dn n ω i j), (fun i j => Wn n ω i j))) atTop
      (fun _ => (a, (fun i j => D i j), (fun i j => W i j))))
    (hm : ∀ n, AEStronglyMeasurable (fun ω => coefficient (an n ω) (Dn n ω) (Wn n ω)) μ)
    (hr : (Dᵀ * W * D).det ≠ 0) (hex : a + D *ᵥ b₀ = 0) :
    TendstoInMeasure μ (fun n ω => coefficient (an n ω) (Dn n ω) (Wn n ω)) atTop
      (fun _ => b₀) := by
  let data := (m → ℝ) × (m → p → ℝ) × (m → m → ℝ)
  let f := fun q : data => coefficient q.1 q.2.1 q.2.2
  have hcont : ContinuousAt f (a,D,W) := by
    have hd : Tendsto (fun q : data => q.2.1) (𝓝 (a,D,W)) (𝓝 D) :=
      (continuous_fst.comp continuous_snd).tendsto _
    have hw : Tendsto (fun q : data => q.2.2) (𝓝 (a,D,W)) (𝓝 W) :=
      (continuous_snd.comp continuous_snd).tendsto _
    have hrsp := response_tendsto _ _ _ _ hd hw hr
    have join : Continuous (fun q : Matrix p m ℝ × (m → ℝ) => -(q.1 *ᵥ q.2)) :=
      (continuous_fst.matrix_mulVec continuous_snd).neg
    exact (join.tendsto _).comp (hrsp.prodMk_nhds (continuous_fst.tendsto _))
  have hid := (affine_identification D W a b₀ hr).1
  rw [hex, Matrix.mulVec_zero, neg_zero, sub_eq_zero] at hid
  rw [← hid]
  let X : ℕ → Ω → data := fun n ω => (an n ω, Dn n ω, Wn n ω)
  exact StatadicFormal.Probability.tendstoInMeasure_map_at_constant (E := data) (F := p → ℝ) μ X (a,D,W) f hlln hcont hm

/-- Weak limit of the actual scaled coefficient errors, not just a covariance formula. -/
@[blueprint "STATADIC-GMM-007"
  (statement := /-- Suppose $\sqrt{N}\,\bar g_n(b_0)$ converges in law to $Z$
  and $(D_n,W_n)$ converges in probability to $(D,W)$ with nonsingular
  $D^TWD$. Assume measurable response maps and almost surely full-rank
  sample weighted Grams. Then $\sqrt{N}(\hat b_n-b_0)$ converges in law
  to $-LZ$, $L=(D^TWD)^{-1}D^TW$, with $N=n+1$. -/)
  (proof := /-- With probability one the exact normal equations give
  $\hat b_n-b_0=-L_n\bar g_n(b_0)$. The matrix LLN and continuity at the
  population Gram imply $L_n\to L$ in probability. Slutsky's theorem and
  continuity of matrix-vector multiplication transfer the assumed score CLT.
  A Gaussian score limit consequently yields a Gaussian coefficient limit. -/)]
theorem coefficient_limit_law (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ν : Measure (m → ℝ)) [IsProbabilityMeasure ν]
    (an : ℕ → Ω → m → ℝ) (Dn : ℕ → Ω → Matrix m p ℝ) (Wn : ℕ → Ω → Matrix m m ℝ)
    (D : Matrix m p ℝ) (W : Matrix m m ℝ) (b₀ : p → ℝ)
    (hr : (Dᵀ * W * D).det ≠ 0)
    (hrn : ∀ n, ∀ᵐ ω ∂μ, ((Dn n ω)ᵀ * Wn n ω * Dn n ω).det ≠ 0)
    (hm : ∀ n, AEMeasurable (fun ω i j => response (Dn n ω) (Wn n ω) i j) μ)
    (hlln : TendstoInMeasure μ (fun n ω => ((fun i j => Dn n ω i j), (fun i j => Wn n ω i j))) atTop (fun _ => ((fun i j => D i j), (fun i j => W i j))))
    (hclt : TendstoInDistribution (fun (n : ℕ) ω => Real.sqrt ((n : ℝ)+1) • (an n ω + Dn n ω *ᵥ b₀))
      atTop id (fun _ => μ) ν) :
    TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ)+1) • (coefficient (an n ω) (Dn n ω) (Wn n ω) - b₀))
      atTop (fun z => -(response D W *ᵥ z)) (fun _ => μ) ν := by
  let data := (m → p → ℝ) × (m → m → ℝ)
  let X : ℕ → Ω → data := fun n ω => (Dn n ω, Wn n ω)
  let f : data → (p → m → ℝ) := fun q i j => response q.1 q.2 i j
  have hc : ContinuousAt (fun q : (m → p → ℝ) × (m → m → ℝ) => response q.1 q.2) (D,W) :=
    response_tendsto _ _ _ _ (continuous_fst.tendsto _) (continuous_snd.tendsto _) hr
  have hl : TendstoInMeasure μ (fun n ω i j => response (Dn n ω) (Wn n ω) i j) atTop
      (fun _ i j => response D W i j) :=
    StatadicFormal.Probability.tendstoInMeasure_map_at_constant (E := data) (F := p → m → ℝ) μ X (D,W) f hlln hc
      (fun n => (hm n).aestronglyMeasurable)
  have hg : Continuous (fun q : (m → ℝ) × (p → m → ℝ) => -((Matrix.of q.2) *ᵥ q.1)) :=
    (continuous_snd.matrix_mulVec continuous_fst).neg
  have h := TendstoInDistribution.continuous_comp_prodMk_of_tendstoInMeasure_const hg hclt hl hm
  apply h.congr _ (Filter.Eventually.of_forall (fun _ => rfl))
  intro n
  filter_upwards [hrn n] with ω hω
  simp only [(affine_identification _ _ (an n ω) b₀ hω).1, Matrix.mulVec_smul, smul_neg]
  rfl

omit [DecidableEq p] in
/-- A matrix transforms a Gaussian score law to a Gaussian law with sandwich covariance. -/
@[blueprint "STATADIC-GMM-008"
  (statement := /-- If $Z$ has a Gaussian law with square-integrable coordinates
  and covariance $S$, the law of $-LZ$ is Gaussian. Its coordinate covariance
  is $LSL^T$, including when the limit is singular. -/)
  (proof := /-- Gaussian laws are closed under continuous linear maps.
  Expand each output coordinate as a finite linear combination and use bilinearity
  of covariance; the two negative signs cancel. -/)]
theorem gaussian_response (L : Matrix p m ℝ) (ν : Measure (m → ℝ)) [IsGaussian ν]
    (S : Matrix m m ℝ) (hs : ∀ i, MemLp (fun z : m → ℝ => z i) 2 ν)
    (hcov : ∀ i j, covariance (fun z => z i) (fun z => z j) ν = S i j) :
    IsGaussian (ν.map (fun z => -(L *ᵥ z))) ∧
      ∀ i j, covariance (fun z => -(L *ᵥ z) i) (fun z => -(L *ᵥ z) j) ν =
        (L * S * Lᵀ) i j := by
  let T : (m → ℝ) →L[ℝ] (p → ℝ) :=
    ⟨-L.mulVecLin, by change Continuous (fun z => -(L *ᵥ z)); fun_prop⟩
  refine ⟨isGaussian_map (μ := ν) T, ?_⟩
  intro i j
  rw [covariance_fun_neg_left, covariance_fun_neg_right, neg_neg]
  simp only [Matrix.mulVec, dotProduct]
  rw [covariance_fun_sum_fun_sum (fun k => (hs k).const_mul (L i k))
    (fun k => (hs k).const_mul (L j k))]
  simp only [covariance_const_mul_left, covariance_const_mul_right, hcov, Matrix.mul_apply,
    Matrix.transpose_apply, Finset.sum_mul]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => by ring

end StatadicFormal.Estimation.GMM
