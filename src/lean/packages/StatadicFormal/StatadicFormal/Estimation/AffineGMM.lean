module

public import StatadicFormal.Estimation.Linear.OLS
import Mathlib.Tactic.Ring

/-! Finite-sample normal equations for fixed-weight affine moments and spatial IV. -/
@[expose] public section
noncomputable section
open Matrix StatadicFormal.Estimation.Linear

namespace StatadicFormal.Estimation

variable {m p n : Type*} [Fintype m] [Fintype p] [DecidableEq p]

/-- The norm gap for an affine moment criterion after applying a weight factor. -/
@[blueprint "STATADIC-GMM-003"
  (statement := /-- Write the weighted affine moments as $a+B\theta$,
  where $a=R\bar g_0$ and $B=R D$ for a fixed weight factor $R$.
  Suppose $B^TB$ is invertible and set
  $b=(B^TB)^{-1}B^T(-a)$. For every $h$,
  \[
    \|a+B(b+h)\|^2=\|a+Bb\|^2+\|Bh\|^2 .
  \]
  In particular the computed coefficient minimizes the quadratic criterion. -/)
  (proof := /-- The normal equations give $B^T(a+Bb)=0$. Expand
  the squared norm of $(a+Bb)+Bh$. Its mixed inner product is
  $h^TB^T(a+Bb)=0$. The remaining sum of squares is nonnegative. -/)]
theorem affine_gmm_gap (B : Matrix m p ℝ) (a : m → ℝ)
    (hgram : IsUnit (Bᵀ * B).det) (h : p → ℝ) :
    let b := olsCoefficients B (-a)
    let e := a + B *ᵥ b
    let next := a + B *ᵥ (b + h)
    next ⬝ᵥ next = e ⬝ᵥ e + (B *ᵥ h) ⬝ᵥ (B *ᵥ h) ∧
      e ⬝ᵥ e ≤ next ⬝ᵥ next := by
  let b := olsCoefficients B (-a)
  let e := a + B *ᵥ b
  have hnormal : Bᵀ *ᵥ e = 0 := by
    dsimp only [e]
    rw [Matrix.mulVec_add, Matrix.mulVec_mulVec, ols_normal_equations B (-a) hgram,
      Matrix.mulVec_neg, add_neg_cancel]
  have horth : e ⬝ᵥ (B *ᵥ h) = 0 := by
    rw [← Matrix.dotProduct_transpose_mulVec B, hnormal, dotProduct_zero]
  have heq : a + B *ᵥ (b + h) = e + B *ᵥ h := by rw [Matrix.mulVec_add, ← add_assoc]
  change (a + B *ᵥ (b + h)) ⬝ᵥ _ = e ⬝ᵥ e + _ ∧ e ⬝ᵥ e ≤ (a + B *ᵥ (b + h)) ⬝ᵥ _
  rw [heq]
  simp only [add_dotProduct, dotProduct_add, dotProduct_comm (B *ᵥ h) e, horth, add_zero,
    zero_add, true_and, le_add_iff_nonneg_right]
  exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)

/-- Spatial 2SLS orthogonality after projecting the structural design on instruments. -/
@[blueprint "STATADIC-IV-001"
  (statement := /-- For instruments $H$ with nonsingular $H^TH$, write
  $P=H(H^TH)^{-1}H^T$ and $D=PX$ for the projected structural regressors.
  If $D^TD$ is nonsingular and $b=(D^TD)^{-1}D^Ty$, then
  $D^T(y-Xb)=0$. Here $X$ may include the endogenous column $Wy$. -/)
  (proof := /-- Symmetry and idempotence of $P$ give
  $D^TD=X^TP^TPX=X^TP^T X=D^TX$. Substitute this identity
  into the normal equations $D^TD b=D^Ty$ and subtract. -/)]
theorem spatial_iv_orthogonality [Fintype n] [DecidableEq m]
    (H : Matrix n m ℝ) (X : Matrix n p ℝ) (y : n → ℝ)
    (hH : IsUnit (Hᵀ * H).det) :
    let P := H * (Hᵀ * H)⁻¹ * Hᵀ
    let D := P * X
    IsUnit (Dᵀ * D).det → Dᵀ *ᵥ (y - X *ᵥ olsCoefficients D y) = 0 := by
  let P := H * (Hᵀ * H)⁻¹ * Hᵀ
  let D := P * X
  change IsUnit (Dᵀ * D).det → _
  intro hD
  have hs : Pᵀ = P := by
    simp [P, Matrix.transpose_mul, Matrix.transpose_nonsing_inv, Matrix.mul_assoc]
  have hi : P * P = P := by
    have hinv := Matrix.nonsing_inv_mul (Hᵀ * H) hH
    calc
      P * P = H * ((Hᵀ * H)⁻¹ * (Hᵀ * H)) * (Hᵀ * H)⁻¹ * Hᵀ := by
        simp only [P, Matrix.mul_assoc]
      _ = P := by rw [hinv, Matrix.mul_one]
  have hDX : Dᵀ * X = Dᵀ * D := by
    simp only [D, Matrix.transpose_mul, hs, Matrix.mul_assoc]
    rw [← Matrix.mul_assoc P P, hi]
  change Dᵀ *ᵥ (y - X *ᵥ olsCoefficients D y) = 0
  rw [Matrix.mulVec_sub, Matrix.mulVec_mulVec, hDX, ols_normal_equations D y hD, sub_self]

end StatadicFormal.Estimation
