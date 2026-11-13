module

public import StatadicFormal.Estimation.GMM.AffineLimit
public import Mathlib.Probability.StrongLaw

/-! Primitive sampling conditions for affine GMM consistency. -/
@[expose] public section
noncomputable section

open Matrix Filter MeasureTheory ProbabilityTheory
open scoped Topology
namespace StatadicFormal.Estimation.GMM

/-- With rowwise iid affine coefficients, the required LLN follows from first moments. -/
@[blueprint "STATADIC-GMM-010"
  (statement := /-- Each observation supplies the intercept $a_i$ and derivative
  $D_i$ of its affine moments. Assume the joint rows have a common law, are
  pairwise independent across observations, and have a finite first moment.
  Let $(a,D)=E[(a_i,D_i)]$, $a+Db_0=0$, and $D^T W D$ nonsingular.
  For any weight sequence $W_n\to W$ almost surely, the GMM coefficients of
  the row averages converge to $b_0$ almost surely. -/)
  (proof := /-- Apply the strong law to the joint row vector, which allows
  dependence between its entries. The mean intercept and mean derivative
  therefore converge on the same probability-one set. On its intersection
  with weight convergence, use continuity of the response map and the
  population moment equation. -/)]
theorem iid_coefficient_strong_consistency {Ω m p : Type*} [MeasurableSpace Ω]
    [Fintype m] [Fintype p] [DecidableEq p] (μ : Measure Ω)
    (X : ℕ → Ω → ((m → ℝ) × (m → p → ℝ))) (Wn : ℕ → Ω → Matrix m m ℝ)
    (W : Matrix m m ℝ) (b₀ : p → ℝ)
    (hi : Integrable (X 0) μ) (hind : Pairwise (fun i j => IndepFun (X i) (X j) μ))
    (hid : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (hw : ∀ᵐ ω ∂μ, Tendsto (fun n => Wn n ω) atTop (𝓝 W)) :
    let q := ∫ ω, X 0 ω ∂μ
    let D := Matrix.of q.2
    let means := fun (n : ℕ) ω => (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, X i ω
    (Dᵀ * W * D).det ≠ 0 → q.1 + D *ᵥ b₀ = 0 →
      ∀ᵐ ω ∂μ, Tendsto (fun n => coefficient (means n ω).1 (Matrix.of (means n ω).2)
        (Wn n ω)) atTop (𝓝 b₀) := by
  let q := ∫ ω, X 0 ω ∂μ
  let D := Matrix.of q.2
  let means := fun (n : ℕ) ω => (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, X i ω
  change (Dᵀ * W * D).det ≠ 0 → q.1 + D *ᵥ b₀ = 0 → _
  intro hr hex
  filter_upwards [strong_law_ae X hi hind hid, hw] with ω hrow hweight
  have hq : Tendsto (fun n => means n ω) atTop (𝓝 q) := hrow
  have hd : Tendsto (fun n => Matrix.of (means n ω).2) atTop (𝓝 D) :=
    (continuous_snd.tendsto q).comp hq
  have ha : Tendsto (fun n => (means n ω).1) atTop (𝓝 q.1) := (continuous_fst.tendsto q).comp hq
  have h := (affine_identification D W q.1 b₀ hr).1
  rw [hex, Matrix.mulVec_zero, neg_zero, sub_eq_zero] at h
  rw [← h]
  exact coefficient_tendsto (fun n => (means n ω).1) (fun n => Matrix.of (means n ω).2)
    (fun n => Wn n ω) q.1 D W ha hd hweight hr

end StatadicFormal.Estimation.GMM
