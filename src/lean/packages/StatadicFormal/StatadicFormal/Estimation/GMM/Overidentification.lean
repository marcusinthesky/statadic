module

public import StatadicFormal.Estimation.GMM.AffineInference
import Mathlib.Tactic.Ring

/-! Residual moment projection and the weak limit of the overidentification criterion. -/
@[expose] public section
noncomputable section
open Matrix MeasureTheory ProbabilityTheory Filter
open scoped Topology Matrix.Norms.Elementwise
namespace StatadicFormal.Estimation.GMM

variable {m p : Type*} [Fintype m] [Fintype p] [DecidableEq m] [DecidableEq p]

/-- Remove from the moments the directions fitted by the estimand. -/
def residualMap (D : Matrix m p ℝ) (W : Matrix m m ℝ) : Matrix m m ℝ := 1 - D * response D W

/-- The quadratic applied to the true-parameter score vector for the retained GMM weight. -/
def residualWeight (D : Matrix m p ℝ) (W : Matrix m m ℝ) : Matrix m m ℝ :=
  (residualMap D W)ᵀ * W * residualMap D W

private theorem fitted_moment (D : Matrix m p ℝ) (W : Matrix m m ℝ) (a : m → ℝ) (b₀ : p → ℝ)
    (hr : (Dᵀ * W * D).det ≠ 0) :
    a + D *ᵥ coefficient a D W = residualMap D W *ᵥ (a + D *ᵥ b₀) := by
  have hzero : residualMap D W * D = 0 := by
    rw [residualMap, Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, response_mul_derivative _ _ hr,
      Matrix.mul_one, sub_self]
  rw [Matrix.mulVec_add, Matrix.mulVec_mulVec, hzero, Matrix.zero_mulVec, add_zero, residualMap,
    Matrix.sub_mulVec, Matrix.one_mulVec, ← Matrix.mulVec_mulVec, coefficient, Matrix.mulVec_neg,
    sub_eq_add_neg]

private theorem residual_weight_limit {α : Type*} {l : Filter α} (Dn : α → Matrix m p ℝ)
    (Wn : α → Matrix m m ℝ) (D : Matrix m p ℝ) (W : Matrix m m ℝ)
    (hd : Tendsto Dn l (𝓝 D)) (hw : Tendsto Wn l (𝓝 W)) (hr : (Dᵀ * W * D).det ≠ 0) :
    Tendsto (fun n => residualWeight (Dn n) (Wn n)) l (𝓝 (residualWeight D W)) := by
  have hl := response_tendsto Dn Wn D W hd hw hr
  have hp : Tendsto (fun n => residualMap (Dn n) (Wn n)) l (𝓝 (residualMap D W)) :=
    tendsto_const_nhds.sub (mul_limit hd hl)
  have hpt : Tendsto (fun n => (residualMap (Dn n) (Wn n))ᵀ) l (𝓝 (residualMap D W)ᵀ) :=
    (continuous_id.matrix_transpose.tendsto (residualMap D W)).comp hp
  exact mul_limit (mul_limit hpt hw) hp

/-- The sample criterion is a projected quadratic; its weak limit retains the actual weight. -/
@[blueprint "STATADIC-JTEST-001"
  (statement := /-- Let $P_n=I-D_n L_n$ and assume almost-sure sample rank.
  Suppose the normalized true-parameter moments $\sqrt{N}\bar g_n(b_0)$
  converge in law to $Z$ and $(D_n,W_n)$ converge in probability to $(D,W)$,
  with $D^TWD$ nonsingular. With measurable residual weight maps, the
  statistic $N\bar g_n(\hat b_n)^TW_n\bar g_n(\hat b_n)$ converges in law
  to $Z^T P^T W P Z$, where $P=I-DL$. -/)
  (proof := /-- The identity $L_n D_n=I$ gives
  $\bar g_n(\hat b_n)=P_n\bar g_n(b_0)$. The mapping from
  $(D,W)$ to $P^T W P$ is continuous at any nonsingular weighted Gram.
  Slutsky's theorem then transfers the score limit through the quadratic form. -/)]
theorem criterion_limit_law {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (ν : Measure (m → ℝ)) [IsProbabilityMeasure ν]
    (an : ℕ → Ω → m → ℝ) (Dn : ℕ → Ω → Matrix m p ℝ) (Wn : ℕ → Ω → Matrix m m ℝ)
    (D : Matrix m p ℝ) (W : Matrix m m ℝ) (b₀ : p → ℝ)
    (hr : (Dᵀ * W * D).det ≠ 0)
    (hrn : ∀ n, ∀ᵐ ω ∂μ, ((Dn n ω)ᵀ * Wn n ω * Dn n ω).det ≠ 0)
    (hm : ∀ n, AEMeasurable (fun ω i j => residualWeight (Dn n ω) (Wn n ω) i j) μ)
    (hlln : TendstoInMeasure μ (fun n ω => ((fun i j => Dn n ω i j), (fun i j => Wn n ω i j)))
      atTop (fun _ => ((fun i j => D i j), (fun i j => W i j))))
    (hclt : TendstoInDistribution (fun (n : ℕ) ω => Real.sqrt ((n : ℝ)+1) •
      (an n ω + Dn n ω *ᵥ b₀)) atTop id (fun _ => μ) ν) :
    TendstoInDistribution (fun (n : ℕ) ω =>
      let g := an n ω + Dn n ω *ᵥ coefficient (an n ω) (Dn n ω) (Wn n ω)
      ((n : ℝ)+1) * (g ⬝ᵥ (Wn n ω *ᵥ g))) atTop
      (fun z => z ⬝ᵥ (residualWeight D W *ᵥ z)) (fun _ => μ) ν := by
  let data := (m → p → ℝ) × (m → m → ℝ)
  let X : ℕ → Ω → data := fun n ω => (Dn n ω, Wn n ω)
  let g : data → (m → m → ℝ) := fun q i j => residualWeight q.1 q.2 i j
  have hg : ContinuousAt g (D,W) :=
    residual_weight_limit (fun q : data => Matrix.of q.1) (fun q : data => Matrix.of q.2)
      D W (continuous_fst.tendsto (D,W)) (continuous_snd.tendsto (D,W)) hr
  have hl := StatadicFormal.Probability.tendstoInMeasure_map_at_constant
    (E := data) (F := m → m → ℝ) μ X (D,W) g hlln hg (fun n => (hm n).aestronglyMeasurable)
  have hf : Continuous (fun q : (m → ℝ) × (m → m → ℝ) => q.1 ⬝ᵥ ((Matrix.of q.2) *ᵥ q.1)) :=
    continuous_fst.dotProduct (continuous_snd.matrix_mulVec continuous_fst)
  have h := TendstoInDistribution.continuous_comp_prodMk_of_tendstoInMeasure_const hf hclt hl hm
  apply h.congr _ (Filter.Eventually.of_forall (fun _ => rfl))
  intro n
  filter_upwards [hrn n] with ω hω
  let z := an n ω + Dn n ω *ᵥ b₀
  let R := residualMap (Dn n ω) (Wn n ω)
  let w := Wn n ω
  let k := Real.sqrt ((n : ℝ) + 1)
  have hk : k ^ 2 = n + 1 := Real.sq_sqrt (by positivity)
  rw [fitted_moment _ _ _ b₀ hω]
  change (k • z) ⬝ᵥ ((Rᵀ * w * R) *ᵥ (k • z)) = _ * ((R *ᵥ z) ⬝ᵥ (w *ᵥ (R *ᵥ z)))
  rw [Matrix.mulVec_smul, dotProduct_smul, smul_dotProduct, smul_eq_mul, smul_eq_mul,
    ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Matrix.dotProduct_transpose_mulVec,
    dotProduct_comm (w *ᵥ _), ← mul_assoc, ← sq, hk]

end StatadicFormal.Estimation.GMM
