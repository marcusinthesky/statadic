module

public import StatadicFormal.Optimization.SpectralFloor
public import Mathlib.LinearAlgebra.Matrix.PosDef
public import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! Exact primal certificates for correlation repair and its final diagonal scaling. -/
@[expose] public section
noncomputable section
open Matrix
namespace StatadicFormal.Optimization
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

private def loss (B A : Matrix ι ι ℝ) : ℝ := ∑ i, ∑ j, (B i j - A i j) ^ 2

omit [DecidableEq ι] in
private theorem trace_loss (B A : Matrix ι ι ℝ) :
    ((B - A)ᵀ * (B - A)).trace = loss B A := by
  simp only [loss, Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply,
    Matrix.sub_apply, ← sq]
  rw [Finset.sum_comm]

/-- A diagonal shift changes objective differences by zero on a fixed-diagonal slice. -/
private theorem shift_difference (B C A : Matrix ι ι ℝ) (y : ι → ℝ) (hd : ∀ i, B i i = C i i) :
    loss B (A + diagonal y) - loss C (A + diagonal y) = loss B A - loss C A := by
  simp only [loss, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  by_cases h : i = j
  · subst j
    simp [hd]
  · simp [Matrix.diagonal_apply_ne _ h]

/-- An exactly unit-diagonal positive-part projection at a dual shift is Frobenius-nearest. -/
@[blueprint "STATADIC-PSD-004"
  (statement := /-- Let $A+\operatorname{diag}(y)=U\operatorname{diag}(d)U^T$ with $U$
  orthogonal. If its positive part $C=U\operatorname{diag}(\max(0,d))U^T$
  has unit diagonal, then $C$ is positive semidefinite and minimizes
  $\|B-A\|_F^2$ among positive semidefinite matrices $B$ with unit diagonal. -/)
  (proof := /-- Clipping at zero yields the PSD Frobenius projection of the shifted
  target. A diagonal shift changes the squared loss by the same quantity for all
  matrices on the unit-diagonal slice. Subtract the losses to cancel that quantity. -/)]
theorem nearest_correlation_dual_certificate (A U : Matrix ι ι ℝ) (y d : ι → ℝ)
    (hu : Uᵀ * U = 1) (hu' : U * Uᵀ = 1)
    (ha : A + diagonal y = U * diagonal d * Uᵀ) :
    let C := U * diagonal (fun i => max 0 (d i)) * Uᵀ
    (∀ i, C i i = 1) → C.PosSemidef ∧ ∀ B : Matrix ι ι ℝ, B.PosSemidef → (∀ i, B i i = 1) →
      (∑ i, ∑ j, (C i j - A i j) ^ 2) ≤ ∑ i, ∑ j, (B i j - A i j) ^ 2 := by
  let C := U * diagonal (fun i => max 0 (d i)) * Uᵀ
  dsimp only
  intro hC
  obtain ⟨hs, hq, hproj⟩ := spectral_floor_projection U hu hu' d 0
  refine ⟨Matrix.posSemidef_iff_dotProduct_mulVec.mpr ⟨?_, ?_⟩, ?_⟩
  · simpa [Matrix.IsHermitian, Matrix.conjTranspose_eq_transpose_of_trivial] using hs
  · intro z
    simpa using hq z
  · intro B hB hb
    have hh := hproj B (fun z => by simpa using hB.dotProduct_mulVec_nonneg z)
    rw [← ha, trace_loss, trace_loss] at hh
    change loss C A ≤ loss B A
    apply sub_nonneg.mp
    rw [← shift_difference B C A y (fun i => (hb i).trans (hC i).symm)]
    exact sub_nonneg.mpr hh

/-- Diagonal congruence after flooring preserves positivity and puts back the requested variances. -/
@[blueprint "STATADIC-PSD-005"
  (statement := /-- Let $F$ be positive definite with $F_{ii}>0$ and desired variances
  $v_i>0$. Define $s_i=\sqrt{v_i}/\sqrt{F_{ii}}$ and $B_{ij}=s_iF_{ij}s_j$.
  Then $B$ is positive definite and $B_{ii}=v_i$. Choosing $v_i=1$ gives the
  final correlation normalization. -/)
  (proof := /-- The diagonal scaling matrix $S$ has no zero diagonal entry, so
  $Sz\ne0$ for nonzero $z$. Hence $z^T SFSz=(Sz)^TF(Sz)>0$. On the diagonal,
  substitute the square roots to obtain $s_i^2F_{ii}=v_i$. -/)]
theorem correlation_rescale (F : Matrix ι ι ℝ) (v : ι → ℝ) (hF : F.PosDef)
    (hdiag : ∀ i, 0 < F i i) (hv : ∀ i, 0 < v i) :
    let s := fun i => Real.sqrt (v i) / Real.sqrt (F i i)
    let B := diagonal s * F * diagonal s
    B.PosDef ∧ (∀ i, B i i = v i) ∧ ∀ i j, B i j = s i * F i j * s j := by
  let s := fun i => Real.sqrt (v i) / Real.sqrt (F i i)
  have hs (i) : s i ≠ 0 := (div_pos (Real.sqrt_pos.mpr (hv i))
    (Real.sqrt_pos.mpr (hdiag i))).ne'
  have inj : Function.Injective (diagonal s).mulVec := by
    intro x y h
    funext i
    have he := congrFun h i
    simpa [Matrix.mulVec_diagonal, hs i] using he
  refine ⟨?_, ?_, ?_⟩
  · simpa [Matrix.conjTranspose_eq_transpose_of_trivial] using
      hF.conjTranspose_mul_mul_same (B := diagonal s) inj
  · intro i
    simp only [Matrix.mul_diagonal, Matrix.diagonal_mul]
    change (s i * F i i) * s i = v i
    dsimp only [s]
    have h := Real.sq_sqrt (hdiag i).le
    have h' := Real.sq_sqrt (hv i).le
    have hn : Real.sqrt (F i i) ≠ 0 := (Real.sqrt_pos.mpr (hdiag i)).ne'
    field_simp [hn]
    rw [h, h', mul_comm]
  · intro i j
    simp [Matrix.mul_diagonal, Matrix.diagonal_mul]

end StatadicFormal.Optimization
