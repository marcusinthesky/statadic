module

public import Architect
public import Mathlib.Analysis.Normed.Field.Lemmas
public import Mathlib.Topology.Instances.Matrix
import Mathlib.Tactic.Ring

/-! The identified affine GMM functional and its behavior at a population limit. -/

@[expose] public section
noncomputable section

open Matrix Filter
open scoped Topology
namespace StatadicFormal.Estimation.GMM

variable {m p : Type*} [Fintype m] [Fintype p] [DecidableEq p]

/-- Moment response map; STATADIC returns the negative of its action on the moment intercept. -/
def response (D : Matrix m p ℝ) (W : Matrix m m ℝ) : Matrix p m ℝ :=
  (Dᵀ * W * D)⁻¹ * (Dᵀ * W)

/-- Exact coefficient for affine mean moments a + D b. An interpretation as a unique minimizer
requires the weighted design rank and appropriate positive weight assumptions. -/
def coefficient (a : m → ℝ) (D : Matrix m p ℝ) (W : Matrix m m ℝ) : p → ℝ :=
  -(response D W *ᵥ a)

/-- An invertible weighted Gram gives a left inverse of the derivative. -/
theorem response_mul_derivative (D : Matrix m p ℝ) (W : Matrix m m ℝ)
    (hr : (Dᵀ * W * D).det ≠ 0) : response D W * D = 1 := by
  unfold response
  rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr hr)]

/-- Identification is a population statement about the zero of the affine moments. -/
@[blueprint "STATADIC-GMM-004"
  (statement := /-- Write population moments as $g(b)=a+Db$. Suppose
  $g(b_0)=0$ and $D^T W D$ is nonsingular. Then $g(b)=0$ exactly when
  $b=b_0$. For any $a$ and $b_0$, the coefficient defined by the weighted
  normal equations obeys $\hat b-b_0=-L(a+Db_0)$ with
  $L=(D^T W D)^{-1}D^T W$. -/)
  (proof := /-- Invertibility gives $LD=I$. Expand $-L(a+Db_0)$ to
  obtain the error identity. Two moment zeros therefore coincide. For spatial
  IV this calculation needs the population instrument-error moment to vanish
  as well as rank of the projected population regressors. -/)]
theorem affine_identification (D : Matrix m p ℝ) (W : Matrix m m ℝ) (a : m → ℝ)
    (b₀ : p → ℝ) (hr : (Dᵀ * W * D).det ≠ 0) :
    coefficient a D W - b₀ = -(response D W *ᵥ (a + D *ᵥ b₀)) ∧
    (a + D *ᵥ b₀ = 0 → ∀ b, (a + D *ᵥ b = 0 ↔ b = b₀)) := by
  have err (b : p → ℝ) : coefficient a D W - b = -(response D W *ᵥ (a + D *ᵥ b)) := by
    rw [Matrix.mulVec_add, Matrix.mulVec_mulVec, response_mul_derivative _ _ hr, Matrix.one_mulVec]
    simp [coefficient, sub_eq_add_neg, add_comm]
  refine ⟨err b₀, fun h₀ b => ⟨fun hb => ?_, fun h => h ▸ h₀⟩⟩
  have h := err b
  have h' := err b₀
  rw [hb, Matrix.mulVec_zero, neg_zero, sub_eq_zero] at h
  rw [h₀, Matrix.mulVec_zero, neg_zero, sub_eq_zero] at h'
  exact h.symm.trans h'

omit [Fintype p] [DecidableEq p] in
/-- A product of convergent finite matrix arrays converges entrywise. -/
theorem mul_limit {q α : Type*} {l : Filter α}
    {A : α → Matrix p m ℝ} {B : α → Matrix m q ℝ} {a : Matrix p m ℝ} {b : Matrix m q ℝ}
    (ha : Tendsto A l (𝓝 a)) (hb : Tendsto B l (𝓝 b)) :
    Tendsto (fun n => A n * B n) l (𝓝 (a * b)) := by
  have h : Continuous (fun x : Matrix p m ℝ × Matrix m q ℝ => x.1 * x.2) :=
    continuous_fst.matrix_mul continuous_snd
  exact (h.tendsto (a,b)).comp (ha.prodMk_nhds hb)

/-- Continuity of the response map only at a nonsingular population Gram. -/
theorem response_tendsto {α : Type*} {l : Filter α}
    (Dn : α → Matrix m p ℝ) (Wn : α → Matrix m m ℝ)
    (D : Matrix m p ℝ) (W : Matrix m m ℝ)
    (hd : Tendsto Dn l (𝓝 D)) (hw : Tendsto Wn l (𝓝 W)) (hr : (Dᵀ * W * D).det ≠ 0) :
    Tendsto (fun n => response (Dn n) (Wn n)) l (𝓝 (response D W)) := by
  have hdt : Tendsto (fun n => (Dn n)ᵀ) l (𝓝 Dᵀ) :=
    (continuous_id.matrix_transpose.tendsto D).comp hd
  have hP := mul_limit hdt hw
  have hG := mul_limit hP hd
  have hi : ContinuousAt Ring.inverse (Dᵀ * W * D).det := by
    simpa only [Ring.inverse_eq_inv'] using continuousAt_inv₀ hr
  have hinv : Tendsto (fun n => ((Dn n)ᵀ * Wn n * Dn n)⁻¹) l (𝓝 (Dᵀ * W * D)⁻¹) :=
    (continuousAt_matrix_inv _ hi).tendsto.comp hG
  exact mul_limit hinv hP

/-- Continuity of the coefficient functional at a population Gram of full rank. -/
theorem coefficient_tendsto {α : Type*} {l : Filter α} (an : α → m → ℝ)
    (Dn : α → Matrix m p ℝ) (Wn : α → Matrix m m ℝ)
    (a : m → ℝ) (D : Matrix m p ℝ) (W : Matrix m m ℝ)
    (ha : Tendsto an l (𝓝 a)) (hd : Tendsto Dn l (𝓝 D))
    (hw : Tendsto Wn l (𝓝 W)) (hr : (Dᵀ * W * D).det ≠ 0) :
    Tendsto (fun n => coefficient (an n) (Dn n) (Wn n)) l (𝓝 (coefficient a D W)) := by
  have hrsp := response_tendsto Dn Wn D W hd hw hr
  apply tendsto_pi_nhds.mpr
  intro i
  change Tendsto (fun n => -(∑ j, response (Dn n) (Wn n) i j * an n j))
    l (𝓝 (-(∑ j, response D W i j * a j)))
  apply Filter.Tendsto.neg
  apply tendsto_finsetSum
  intro j _
  have hrow := tendsto_pi_nhds.mp (tendsto_pi_nhds.mp hrsp i) j
  exact hrow.mul (tendsto_pi_nhds.mp ha j)

/-- Consistency of the covariance on the square-root sample size scale. -/
@[blueprint "STATADIC-GMM-005"
  (statement := /-- If $D_n\to D$, $W_n\to W$ and the estimated moment covariance
  $S_n\to S$ on a sample path, and $D^TWD$ is nonsingular, then
  $L_nS_nL_n^T\to LSL^T$ for the affine GMM response map $L_n$. -/)
  (proof := /-- Finite matrix products and transposes preserve limits. Inversion
  is continuous at the nonsingular limiting Gram matrix, giving $L_n\to L$.
  Apply continuity again to the two outer products. -/)]
theorem sandwich_tendsto {α : Type*} {l : Filter α}
    (Dn : α → Matrix m p ℝ) (Wn Sn : α → Matrix m m ℝ)
    (D : Matrix m p ℝ) (W S : Matrix m m ℝ)
    (hd : Tendsto Dn l (𝓝 D)) (hw : Tendsto Wn l (𝓝 W)) (hs : Tendsto Sn l (𝓝 S))
    (hr : (Dᵀ * W * D).det ≠ 0) :
    Tendsto (fun n => response (Dn n) (Wn n) * Sn n * (response (Dn n) (Wn n))ᵀ)
      l (𝓝 (response D W * S * (response D W)ᵀ)) := by
  have hl := response_tendsto Dn Wn D W hd hw hr
  have hlt : Tendsto (fun n => (response (Dn n) (Wn n))ᵀ) l (𝓝 (response D W)ᵀ) :=
    (continuous_id.matrix_transpose.tendsto (response D W)).comp hl
  exact mul_limit (mul_limit hl hs) hlt

end StatadicFormal.Estimation.GMM
