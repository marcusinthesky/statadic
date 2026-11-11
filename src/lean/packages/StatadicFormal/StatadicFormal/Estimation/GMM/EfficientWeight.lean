module

public import StatadicFormal.Estimation.GMM.AffineLimit
import Mathlib.Tactic.Abel

/-! The efficient moment weight minimizes asymptotic contrast variance. -/

@[expose] public section
noncomputable section
open Matrix

namespace StatadicFormal.Estimation.GMM

/-- The response matrix using inverse moment covariance is optimal among left inverses. -/
@[blueprint "STATADIC-GMM-009"
  (statement := /-- Let the moment covariance $S$ be symmetric positive definite,
  with $D^T S^{-1}D$ nonsingular. Put $E=(D^T S^{-1}D)^{-1}D^TS^{-1}$.
  For every alternative response $A$ satisfying $AD=I$,
  \[
    ASA^T-ESE^T=(A-E)S(A-E)^T .
  \]
  Thus each contrast variance under $E$ is at most its variance under $A$. -/)
  (proof := /-- Both $A$ and $E$ reproduce the coefficients, hence
  $(A-E)D=0$. Since $ES=(D^TS^{-1}D)^{-1}D^T$, the two mixed terms
  in the expansion vanish. The remaining quadratic form is nonnegative
  from the moment covariance assumption. -/)]
theorem efficient_weight_ordering {m p : Type*} [Fintype m] [Fintype p]
    [DecidableEq m] [DecidableEq p] (D : Matrix m p ℝ) (S : Matrix m m ℝ)
    (hs : Sᵀ = S) (hi : S.det ≠ 0) (hpos : ∀ v, 0 ≤ v ⬝ᵥ (S *ᵥ v))
    (hr : (Dᵀ * S⁻¹ * D).det ≠ 0) (A : Matrix p m ℝ) (ha : A * D = 1) :
    let E := response D S⁻¹
    A * S * Aᵀ - E * S * Eᵀ = (A - E) * S * (A - E)ᵀ ∧
      ∀ t : p → ℝ, t ⬝ᵥ ((E * S * Eᵀ) *ᵥ t) ≤ t ⬝ᵥ ((A * S * Aᵀ) *ᵥ t) := by
  let E := response D S⁻¹
  let H := A - E
  have hh : H * D = 0 := by
    dsimp only [H, E]
    rw [Matrix.sub_mul, ha, response_mul_derivative _ _ hr, sub_self]
  have he : E * S = (Dᵀ * S⁻¹ * D)⁻¹ * Dᵀ := by
    dsimp only [E, response]
    simp only [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ (isUnit_iff_ne_zero.mpr hi),
      Matrix.mul_one]
  have hcross : E * S * Hᵀ = 0 := by
    rw [he, Matrix.mul_assoc, ← Matrix.transpose_mul, hh, Matrix.transpose_zero, Matrix.mul_zero]
  have hcross' : H * S * Eᵀ = 0 := by
    have h := congrArg Matrix.transpose hcross
    simpa only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_zero,
      hs, Matrix.mul_assoc] using h
  have gap : A * S * Aᵀ - E * S * Eᵀ = H * S * Hᵀ := by
    have heq : A = H + E := sub_add_cancel A E |>.symm
    rw [heq, Matrix.transpose_add, Matrix.add_mul, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add,
      hcross, hcross']
    abel
  refine ⟨gap, fun t => sub_nonneg.mp ?_⟩
  change 0 ≤ t ⬝ᵥ ((A * S * Aᵀ) *ᵥ t) - t ⬝ᵥ ((E * S * Eᵀ) *ᵥ t)
  rw [← dotProduct_sub, ← Matrix.sub_mulVec, gap, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
    ← Matrix.dotProduct_transpose_mulVec, dotProduct_comm]
  exact hpos _

end StatadicFormal.Estimation.GMM
