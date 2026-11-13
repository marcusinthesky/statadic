module

public import Architect
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

/-!
# Jacobian of a fully observed spatial panel

Each period has a nonsingular outcome filter `A` and error filter `B` on a
common finite roster. The Jacobian maps outcomes to innovations. Covariate
offsets do not enter it. Marginalising missing outcomes requires a different
calculation and is outside this statement.
-/

@[expose] public section
noncomputable section

open scoped BigOperators

namespace StatadicFormal.Estimation.Spatial

variable {τ ι : Type*} [Fintype τ] [DecidableEq τ] [Fintype ι] [DecidableEq ι]

/-- Full-panel log absolute Jacobian for period-varying lag and error filters. -/
@[blueprint "STATADIC-SPATIAL-001"
  (statement := /-- The log absolute determinant of the block diagonal map
  with period blocks $B_t A_t$ equals the sum over periods of the log absolute
  determinants of the lag and error filters, when each is nonsingular. -/)
  (proof := /-- The determinant of a block diagonal matrix is the product of
  the block determinants. Apply multiplicativity to each block and sum logs
  of their nonzero determinants. No commutation of the filters is needed. -/)]
theorem panel_log_jacobian (A B : τ → Matrix ι ι ℝ)
    (ha : ∀ t, (A t).det ≠ 0) (hb : ∀ t, (B t).det ≠ 0) :
    Real.log |(Matrix.blockDiagonal (fun t ↦ B t * A t)).det| =
      ∑ t, (Real.log |(A t).det| + Real.log |(B t).det|) := by
  simp only [Real.log_abs, Matrix.det_blockDiagonal, Matrix.det_mul]
  rw [Real.log_prod (fun t _ ↦ mul_ne_zero (hb t) (ha t))]
  apply Finset.sum_congr rfl
  intro t _
  rw [Real.log_mul (hb t) (ha t), add_comm]

/-- A common pair of filters contributes the same log Jacobian in every period. -/
theorem fixed_panel_log_jacobian (A B : Matrix ι ι ℝ) (ha : A.det ≠ 0) (hb : B.det ≠ 0) :
    Real.log |(Matrix.blockDiagonal (fun _ : τ ↦ B * A)).det| =
      (Fintype.card τ : ℝ) * (Real.log |A.det| + Real.log |B.det|) := by
  simpa [nsmul_eq_mul, mul_add] using
    (panel_log_jacobian (fun _ : τ ↦ A) (fun _ : τ ↦ B) (fun _ ↦ ha) (fun _ ↦ hb))

end StatadicFormal.Estimation.Spatial
