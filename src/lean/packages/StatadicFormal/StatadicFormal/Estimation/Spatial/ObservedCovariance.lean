module

public import Architect
public import Mathlib.Basic.Real.Basic
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-! # Covariance of observed outcomes in a spatial graph model

For one period let `F = B * A` be the full-roster innovation filter, with
innovation covariance proportional to the identity. `P` selects the observed
outcomes. Solving the transposed system yields the observed rows of the full
inverse, including the contributions of unobserved nodes. -/

@[expose] public section
noncomputable section

open Matrix

namespace StatadicFormal.Estimation.Spatial

variable {ι ο : Type*} [Fintype ι] [DecidableEq ι]

/-- The rectangular solve in the observed-outcome likelihood produces a root
of the selected full covariance per unit innovation variance. -/
@[blueprint "STATADIC-SPATIAL-002"
  (statement := /-- Let $F$ be an invertible real innovation filter and let $P$
  select observations. If $F^T Z=P^T$, then
  $Z^T Z=P(F^{-1}F^{-T})P^T$. The identity holds for any rectangular $P$. -/)
  (proof := /-- Transpose the solve to obtain $Z^T F=P$. Multiply on the right
  by $F^{-1}$, yielding $Z^T=PF^{-1}$. Substitute this in the Gram matrix
  and associate the products. -/)]
theorem observed_covariance_of_transposed_solve (F : Matrix ι ι ℝ)
    (P : Matrix ο ι ℝ) (Z : Matrix ι ο ℝ) (hF : IsUnit F.det) (hsolve : Fᵀ * Z = Pᵀ) :
    Zᵀ * Z = P * (F⁻¹ * (F⁻¹)ᵀ) * Pᵀ := by
  have heq : Zᵀ * F = P := by
    simpa only [Matrix.transpose_mul, Matrix.transpose_transpose] using
      congrArg Matrix.transpose hsolve
  have hroot : Zᵀ = P * F⁻¹ := by
    rw [← heq, Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hF, Matrix.mul_one]
  have hz : Z = (F⁻¹)ᵀ * Pᵀ := by
    simpa only [Matrix.transpose_mul, Matrix.transpose_transpose] using
      congrArg Matrix.transpose hroot
  rw [hroot, hz]
  simp only [Matrix.mul_assoc]

end StatadicFormal.Estimation.Spatial
