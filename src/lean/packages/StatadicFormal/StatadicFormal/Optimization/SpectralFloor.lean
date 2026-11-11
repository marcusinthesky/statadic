module

public import Mathlib.LinearAlgebra.Matrix.Trace
public import Mathlib.Basic.Real.Basic
public import Architect
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Trace distortion and the squared-error projection in spectral coordinates. -/
@[expose] public section
noncomputable section
open Matrix
namespace StatadicFormal.Optimization

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Total positive floor correction equals the distortion of total variance. -/
@[blueprint "STATADIC-PSD-001"
  (statement := /-- In an orthonormal eigenbasis $U$, replace $d_i$ by
  $c_i=\max(\delta,d_i)$. The change of trace is exactly
  $\sum_i\max(\delta-d_i,0)$. In particular it is zero precisely when every
  original eigenvalue meets the floor. -/)
  (proof := /-- Move $U^TU=I$ across the trace. Subtract the two diagonal
  sums term by term. Every summand is a nonnegative shortfall, so a zero sum
  forces every shortfall to vanish. -/)]
theorem spectral_floor_trace (U : Matrix ι ι ℝ) (hU : Uᵀ * U = 1)
    (d : ι → ℝ) (δ : ℝ) :
    let A := U * diagonal d * Uᵀ
    let B := U * diagonal (fun i => max δ (d i)) * Uᵀ
    B.trace - A.trace = ∑ i, max (δ - d i) 0 ∧ (B.trace = A.trace ↔ ∀ i, δ ≤ d i) := by
  have tr (v : ι → ℝ) : (U * diagonal v * Uᵀ).trace = ∑ i, v i := by
    rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, hU, Matrix.one_mul, Matrix.trace_diagonal]
  dsimp only
  rw [tr, tr]
  have gap : (∑ i, max δ (d i)) - ∑ i, d i = ∑ i, max (δ - d i) 0 := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun i _ => by rcases le_total δ (d i) with h | h <;> simp [*])
  refine ⟨gap, ?_⟩
  rw [← sub_eq_zero, gap, Finset.sum_eq_zero_iff_of_nonneg (fun i _ => le_max_right _ _)]
  simp

/-- Orthogonal-coordinate reduction of the Frobenius projection onto a spectral floor. -/
@[blueprint "STATADIC-PSD-002"
  (statement := /-- In an eigenbasis of the input, let $D=\mathrm{diag}(d)$
  and $C=\mathrm{diag}(\max(d_i,\delta))$. For every real matrix $B$ with
  $B_{ii}\ge\delta$, $\|B-D\|_F^2\ge\|C-D\|_F^2+\|B-C\|_F^2$.
  Every symmetric competitor above the spectral floor has these diagonal bounds. -/)
  (proof := /-- Off-diagonal terms agree on the two sides since $D$ and $C$
  are diagonal. When $d_i$ already meets the floor its correction is zero.
  Otherwise the cross term $2(B_{ii}-\delta)(\delta-d_i)$ is nonnegative.
  Sum the entrywise inequalities. -/)]
theorem spectral_floor_frobenius (d : ι → ℝ) (δ : ℝ) (B : Matrix ι ι ℝ)
    (hb : ∀ i, δ ≤ B i i) :
    let D := diagonal d
    let C := diagonal (fun i => max δ (d i))
    (∑ i, ∑ j, ((C i j - D i j) ^ 2 + (B i j - C i j) ^ 2)) ≤
      ∑ i, ∑ j, (B i j - D i j) ^ 2 := by
  dsimp only
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  by_cases h : i = j
  · subst j
    simp only [diagonal_apply_eq]
    rcases le_total δ (d i) with hd | hd
    · simp [hd]
    · rw [max_eq_left hd]
      nlinarith [mul_nonneg (sub_nonneg.mpr (hb i)) (sub_nonneg.mpr hd)]
  · simp [diagonal_apply_ne _ h]

/-- Squared Frobenius length expressed as a trace. -/
private def frobSq (M : Matrix ι ι ℝ) : ℝ := (Mᵀ * M).trace

omit [DecidableEq ι] in
private theorem frob_entries (M : Matrix ι ι ℝ) :
    frobSq M = ∑ i, ∑ j, M i j ^ 2 := by
  simp only [frobSq, Matrix.trace, Matrix.diag_apply, Matrix.mul_apply,
    Matrix.transpose_apply, ← sq]
  rw [Finset.sum_comm]

private theorem frob_orthogonal (U M : Matrix ι ι ℝ) (h : Uᵀ * U = 1) :
    frobSq (U * M * Uᵀ) = frobSq M := by
  simp only [frobSq, Matrix.transpose_mul, Matrix.transpose_transpose]
  calc
    _ = (U * (Mᵀ * ((Uᵀ * U) * M)) * Uᵀ).trace := by simp only [Matrix.mul_assoc]
    _ = (Uᵀ * (U * (Mᵀ * M))).trace := by rw [h, Matrix.one_mul, Matrix.trace_mul_comm]
    _ = _ := by rw [← Matrix.mul_assoc, h, Matrix.one_mul]

/-- Orthogonal reconstruction satisfies the spectral floor and minimizes the same
entrywise squared error in the original coordinates. -/
@[blueprint "STATADIC-PSD-003"
  (statement := /-- Suppose $A=U\operatorname{diag}(d)U^T$ with orthogonal $U$.
  Replacing $d_i$ by $c_i=\max(d_i,\delta)$ gives a symmetric matrix $C$
  with $z^TCz\ge\delta z^Tz$ for every $z$. Every competing symmetric $B$
  obeying that same floor satisfies $\|C-A\|_F^2\le\|B-A\|_F^2$. -/)
  (proof := /-- Put $v=U^Tz$. Orthogonality preserves length, and the
  clipped quadratic form is $\sum_i c_i v_i^2\ge\delta\sum_i v_i^2$.
  For the competitor, test its floor on each unit column of $U$ to bound the
  diagonal of $U^TBU$. Apply the entrywise floor projection inequality in that
  basis. Cyclic invariance of the trace preserves squared Frobenius error
  when returning to the original coordinates. -/)]
theorem spectral_floor_projection (U : Matrix ι ι ℝ) (hu : Uᵀ * U = 1) (hu' : U * Uᵀ = 1)
    (d : ι → ℝ) (δ : ℝ) :
    let A := U * diagonal d * Uᵀ
    let C := U * diagonal (fun i => max δ (d i)) * Uᵀ
    Cᵀ = C ∧ (∀ z, δ * (z ⬝ᵥ z) ≤ z ⬝ᵥ (C *ᵥ z)) ∧
      ∀ B : Matrix ι ι ℝ, (∀ z, δ * (z ⬝ᵥ z) ≤ z ⬝ᵥ (B *ᵥ z)) →
        ((C - A)ᵀ * (C - A)).trace ≤ ((B - A)ᵀ * (B - A)).trace := by
  let c := fun i => max δ (d i)
  let D := diagonal d
  let E := diagonal c
  have adj (x y : ι → ℝ) : x ⬝ᵥ (U *ᵥ y) = (Uᵀ *ᵥ x) ⬝ᵥ y := by
    rw [← Matrix.dotProduct_transpose_mulVec, dotProduct_comm]
  have isom (z : ι → ℝ) : (Uᵀ *ᵥ z) ⬝ᵥ (Uᵀ *ᵥ z) = z ⬝ᵥ z := by
    rw [← adj, Matrix.mulVec_mulVec, hu', Matrix.one_mulVec]
  refine ⟨by simp [Matrix.transpose_mul, Matrix.mul_assoc], ?_, ?_⟩
  · intro z
    let v := Uᵀ *ᵥ z
    change δ * (z ⬝ᵥ z) ≤ z ⬝ᵥ ((U * E * Uᵀ) *ᵥ z)
    rw [← isom z, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, adj]
    change δ * (v ⬝ᵥ v) ≤ v ⬝ᵥ (diagonal c *ᵥ v)
    simp only [dotProduct, Matrix.mulVec_diagonal, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    nlinarith [mul_nonneg (sub_nonneg.mpr (le_max_left δ (d i))) (sq_nonneg (v i)),
      show c i = max δ (d i) from rfl]
  · intro B hb
    let T := Uᵀ * B * U
    have hdiag (i : ι) : δ ≤ T i i := by
      let z := fun j => U j i
      have he : z ⬝ᵥ z = 1 := by
        have hentry := congrFun (congrFun hu i) i
        simpa [z, dotProduct, Matrix.mul_apply] using hentry
      have hentry : z ⬝ᵥ (B *ᵥ z) = T i i := by
        rw [show T = Uᵀ * (B * U) by simp [T, Matrix.mul_assoc]]
        simp [Matrix.mul_apply, Matrix.mulVec, dotProduct, z]
      simpa [he, hentry] using hb z
    have recover : U * T * Uᵀ = B := by
      dsimp only [T]
      rw [← Matrix.mul_assoc U (Uᵀ * B), ← Matrix.mul_assoc U Uᵀ, hu', Matrix.one_mul,
        Matrix.mul_assoc, hu', Matrix.mul_one]
    have hbound : frobSq (E - D) ≤ frobSq (T - D) := by
      rw [frob_entries, frob_entries]
      calc
        _ ≤ ∑ i, ∑ j, ((E i j - D i j) ^ 2 + (T i j - E i j) ^ 2) :=
          Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => le_add_of_nonneg_right
            (sq_nonneg (T i j - E i j))))
        _ ≤ _ := spectral_floor_frobenius d δ T hdiag
    change frobSq (U * E * Uᵀ - U * D * Uᵀ) ≤ frobSq (B - U * D * Uᵀ)
    rw [← recover, ← Matrix.sub_mul, ← Matrix.mul_sub, ← Matrix.sub_mul, ← Matrix.mul_sub,
      frob_orthogonal U _ hu, frob_orthogonal U _ hu]
    exact hbound

end StatadicFormal.Optimization
