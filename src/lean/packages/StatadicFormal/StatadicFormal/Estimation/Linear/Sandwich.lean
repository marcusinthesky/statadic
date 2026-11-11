module

public import StatadicFormal.Estimation.Linear.OLS
public import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Tactic.Ring

/-! Exact finite-law variance for heteroskedastic uncorrelated errors. -/
open scoped BigOperators
open Matrix StatadicFormal.Probability
@[expose] public section
noncomputable section
namespace StatadicFormal.Estimation.Linear

variable {ι κ Ω : Type*} [Fintype ι] [Fintype κ] [Fintype Ω]
  [DecidableEq ι] [DecidableEq κ]

theorem ols_contrast_variance_diagonal (P : FiniteLaw Ω) (X : Matrix ι κ ℝ)
    (t : κ → ℝ) (s : ι → ℝ) (ε : ι → Ω → ℝ) (σ2 : ι → ℝ)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 i else 0) :
    P.variance (fun ω => t ⬝ᵥ olsCoefficients X (fun i => s i + ε i ω)) =
      ∑ i, σ2 i * (olsContrastWeights X t i) ^ 2 := by
  simp_rw [ols_contrast_eq]
  change P.variance (estimate (olsContrastWeights X t) (fun i ω => s i + ε i ω)) = _
  rw [variance_shifted_estimate]
  exact variance_diagonal P _ ε σ2 hzero hpair

/-- The diagonal sum in the variance theorem is the usual matrix sandwich. -/
theorem ols_diagonal_sandwich (X : Matrix ι κ ℝ) (t : κ → ℝ) (σ2 : ι → ℝ) :
    (∑ i, σ2 i * (olsContrastWeights X t i) ^ 2) =
      t ⬝ᵥ ((Xᵀ * X)⁻¹ * Xᵀ * Matrix.diagonal σ2 * X * (Xᵀ * X)⁻¹) *ᵥ t := by
  let G := (Xᵀ * X)⁻¹
  have hs : Gᵀ = G := by
    dsimp [G]
    rw [Matrix.transpose_nonsing_inv, Matrix.transpose_mul]
    simp
  have ht : ∀ v : κ → ℝ, t ⬝ᵥ G *ᵥ v = (G *ᵥ t) ⬝ᵥ v := by
    intro v
    simpa [hs, dotProduct_comm] using (Matrix.dotProduct_transpose_mulVec G t v)
  have hx : ∀ v : ι → ℝ, (G *ᵥ t) ⬝ᵥ Xᵀ *ᵥ v = (X *ᵥ (G *ᵥ t)) ⬝ᵥ v := by
    intro v
    simpa [dotProduct_comm] using (Matrix.dotProduct_transpose_mulVec X (G *ᵥ t) v)
  rw [show (Xᵀ * X)⁻¹ = G from rfl]
  simp only [← Matrix.mulVec_mulVec, ht, hx]
  simp [olsContrastWeights, ← show (Xᵀ * X)⁻¹ = G from rfl,
    Matrix.mulVec_diagonal, dotProduct, pow_two, mul_left_comm]

/-- Finite-law diagonal-error sandwich, without an asymptotic claim. -/
@[blueprint "STATADIC-LINEAR-007"
  (statement := /-- Let $X$ be a fixed real matrix and let $Y=s+\varepsilon$
  on a finite probability space, where $\mathbb E\varepsilon=0$ and
  $\mathbb E(\varepsilon_i\varepsilon_j)=\mathbf 1_{i=j}\sigma_i^2$.
  With $\hat\beta=(X^TX)^{-1}X^TY$ and $D=\operatorname{diag}(\sigma_i^2)$,
  \[\operatorname{Var}(t^T\hat\beta)=
  t^T(X^TX)^{-1}X^TDX(X^TX)^{-1}t.\]
  Matrix inverses here use Lean's total nonsingular inverse. -/)
  (proof := /-- Expand the finite variance of the linear contrast, use the
  diagonal pair moments, then rearrange the resulting sum into the sandwich.
  The deterministic signal cancels under centering. -/)]
theorem finite_ols_sandwich (P : FiniteLaw Ω) (X : Matrix ι κ ℝ)
    (t : κ → ℝ) (s : ι → ℝ) (ε : ι → Ω → ℝ) (σ2 : ι → ℝ)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 i else 0) :
    P.variance (fun ω => t ⬝ᵥ olsCoefficients X (fun i => s i + ε i ω)) =
      t ⬝ᵥ ((Xᵀ * X)⁻¹ * Xᵀ * Matrix.diagonal σ2 * X * (Xᵀ * X)⁻¹) *ᵥ t := by
  rw [ols_contrast_variance_diagonal P X t s ε σ2 hzero hpair, ols_diagonal_sandwich]

/-- Any nonnegative diagonal meat, including squared residuals and their
positive leverage corrections, yields a nonnegative contrast quadratic. -/
theorem ols_sandwich_nonnegative (X : Matrix ι κ ℝ) (t : κ → ℝ) (d : ι → ℝ)
    (hd : ∀ i, 0 ≤ d i) :
    0 ≤ t ⬝ᵥ ((Xᵀ * X)⁻¹ * Xᵀ * Matrix.diagonal d * X * (Xᵀ * X)⁻¹) *ᵥ t := by
  rw [← ols_diagonal_sandwich]
  exact Finset.sum_nonneg (fun i _ => mul_nonneg (hd i) (sq_nonneg _))

/-- The OLS fitted-value matrix. -/
def hatMatrix (X : Matrix ι κ ℝ) : Matrix ι ι ℝ :=
  X * (Xᵀ * X)⁻¹ * Xᵀ

omit [DecidableEq ι] in
theorem hat_symmetric (X : Matrix ι κ ℝ) : (hatMatrix X)ᵀ = hatMatrix X := by
  unfold hatMatrix
  simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_nonsing_inv]
  simp [Matrix.mul_assoc]

omit [DecidableEq ι] in
theorem hat_idempotent (X : Matrix ι κ ℝ) (h : IsUnit (Xᵀ * X).det) :
    hatMatrix X * hatMatrix X = hatMatrix X := by
  unfold hatMatrix
  calc
    (X * (Xᵀ * X)⁻¹ * Xᵀ) * (X * (Xᵀ * X)⁻¹ * Xᵀ) =
        X * ((Xᵀ * X)⁻¹ * (Xᵀ * X) * (Xᵀ * X)⁻¹) * Xᵀ := by simp [Matrix.mul_assoc]
    _ = X * (Xᵀ * X)⁻¹ * Xᵀ := by rw [Matrix.nonsing_inv_mul _ h]; simp

omit [DecidableEq ι] in
/-- Each diagonal entry of a symmetric projection lies between zero and one. -/
theorem hat_leverage_bounds (X : Matrix ι κ ℝ) (h : IsUnit (Xᵀ * X).det)
    (i : ι) : 0 ≤ hatMatrix X i i ∧ hatMatrix X i i ≤ 1 := by
  let H := hatMatrix X
  have hs : ∀ i j, H i j = H j i := by
    intro i j
    have ht := congrArg (fun M : Matrix ι ι ℝ => M i j) (hat_symmetric X)
    exact ht.symm
  have hi : ∑ j, H i j ^ 2 = H i i := by
    have ht := congrArg (fun M : Matrix ι ι ℝ => M i i) (hat_idempotent X h)
    change (∑ j, H i j * H j i) = H i i at ht
    simpa only [← hs i, ← pow_two] using ht
  have hn : 0 ≤ H i i := by
    rw [← hi]
    exact Finset.sum_nonneg (fun j _ => sq_nonneg _)
  have hsq : H i i ^ 2 ≤ H i i := by
    calc
      H i i ^ 2 ≤ ∑ j, H i j ^ 2 :=
        Finset.single_le_sum (fun j _ => sq_nonneg (H i j)) (Finset.mem_univ i)
      _ = H i i := hi
  constructor
  · exact hn
  · nlinarith

omit [DecidableEq ι] in
/-- Full rank makes the projection trace equal the number of columns. -/
theorem hat_leverage_sum (X : Matrix ι κ ℝ) (h : IsUnit (Xᵀ * X).det) :
    (∑ i, hatMatrix X i i) = Fintype.card κ := by
  change Matrix.trace (hatMatrix X) = _
  unfold hatMatrix
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, Matrix.mul_nonsing_inv _ h]
  simp [Matrix.trace_one]

omit [DecidableEq ι] in
theorem hat_fixes_design (X : Matrix ι κ ℝ) (h : IsUnit (Xᵀ * X).det) :
    hatMatrix X * X = X := by
  unfold hatMatrix
  rw [Matrix.mul_assoc, Matrix.mul_assoc, Matrix.nonsing_inv_mul _ h]
  simp

/-- Each residual is a linear combination of the errors with coefficients
from the complementary projection; its second moment is σ²(1 − hᵢ). -/
@[blueprint "STATADIC-LINEAR-008"
  (statement := /-- Let $X$ have invertible $X^TX$ and let $Y=X\beta+\varepsilon$
  on a finite probability space, with $\mathbb E\varepsilon=0$ and
  $\mathbb E(\varepsilon_i\varepsilon_j)=\mathbf 1_{i=j}\sigma^2$.
  Put $\hat\beta=(X^TX)^{-1}X^TY$ and
  $h_i=[X(X^TX)^{-1}X^T]_{ii}$. Then, for each $i$,
  \[\mathbb E\bigl(Y_i-(X\hat\beta)_i\bigr)^2=\sigma^2(1-h_i).\]
  -/)
  (proof := /-- The residual is the corresponding row of $(I-H)\varepsilon$.
  Symmetry and idempotence of $H$ make $I-H$ a symmetric projection, whose
  row sum of squares is its diagonal. Apply spherical second moments. -/)]
theorem ols_residual_second_moment (P : FiniteLaw Ω) (X : Matrix ι κ ℝ)
    (β : κ → ℝ) (ε : ι → Ω → ℝ) (σ2 : ℝ)
    (h : IsUnit (Xᵀ * X).det) (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0) (i : ι) :
    P.mean (fun ω => ((∑ j, X i j * β j) + ε i ω -
      (X *ᵥ olsCoefficients X (fun k => (∑ j, X k j * β j) + ε k ω)) i) ^ 2) =
      σ2 * (1 - hatMatrix X i i) := by
  let R : Matrix ι ι ℝ := 1 - hatMatrix X
  have hR : R * R = R := by
    dsimp [R]
    simp only [sub_mul, mul_sub, one_mul, mul_one, hat_idempotent X h]
    abel
  have hsym : Rᵀ = R := by simp [R, Matrix.transpose_sub, hat_symmetric]
  have hrow : ∑ j, R i j ^ 2 = R i i := by
    have he := congrArg (fun M : Matrix ι ι ℝ => M i i) hR
    change (∑ j, R i j * R j i) = R i i at he
    have hs : ∀ j, R j i = R i j := by
      intro j
      exact congrArg (fun M : Matrix ι ι ℝ => M i j) hsym
    simpa only [hs, ← pow_two] using he
  have hr : ∀ ω, (∑ j, X i j * β j) + ε i ω -
      (X *ᵥ olsCoefficients X (fun k => (∑ j, X k j * β j) + ε k ω)) i =
      estimate (R i) ε ω := by
    intro ω
    have hsignal : R * X = 0 := by
      simp [R, Matrix.sub_mul, hat_fixes_design X h]
    have hvec : (fun k => (∑ j, X k j * β j) + ε k ω) = X *ᵥ β + fun k => ε k ω := by
      ext k
      simp [Matrix.mulVec, dotProduct]
    have hcoef : ∀ y, X *ᵥ olsCoefficients X y = hatMatrix X *ᵥ y := by
      intro y
      simp [olsCoefficients, hatMatrix, Matrix.mulVec_mulVec, Matrix.mul_assoc]
    rw [hcoef, hvec]
    have hzero' : R *ᵥ (X *ᵥ β) = 0 := by
      rw [Matrix.mulVec_mulVec, hsignal]
      simp
    have heq : (R *ᵥ (X *ᵥ β + fun k => ε k ω)) i = (R *ᵥ fun k => ε k ω) i := by
      rw [Matrix.mulVec_add, hzero']
      simp
    have heq' : (X *ᵥ β + fun k => ε k ω) i -
        (hatMatrix X *ᵥ (X *ᵥ β + fun k => ε k ω)) i =
        (R *ᵥ fun k => ε k ω) i := by
      simpa only [R, Matrix.sub_mulVec, Matrix.one_mulVec, Pi.sub_apply] using heq
    simpa [Matrix.mulVec, dotProduct, estimate] using heq'
  have hm : P.mean (estimate (R i) ε) = 0 := by rw [mean_estimate]; simp [hzero]
  have hv := variance_spherical P (R i) ε σ2 hzero hpair
  unfold FiniteLaw.variance at hv
  rw [hm] at hv
  simp only [sub_zero] at hv
  simp_rw [hr]
  rw [hv, hrow]
  simp [R]

/-- Under spherical errors the HC2-adjusted squared residual has mean σ²,
provided the leverage is strictly below one. -/
theorem ols_hc2_residual_mean (P : FiniteLaw Ω) (X : Matrix ι κ ℝ)
    (β : κ → ℝ) (ε : ι → Ω → ℝ) (σ2 : ℝ)
    (h : IsUnit (Xᵀ * X).det) (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0) (i : ι) (hi : hatMatrix X i i < 1) :
    P.mean (fun ω => ((∑ j, X i j * β j) + ε i ω -
      (X *ᵥ olsCoefficients X (fun k => (∑ j, X k j * β j) + ε k ω)) i) ^ 2 /
        (1 - hatMatrix X i i)) = σ2 := by
  have he : (fun ω => ((∑ j, X i j * β j) + ε i ω -
      (X *ᵥ olsCoefficients X (fun k => (∑ j, X k j * β j) + ε k ω)) i) ^ 2 /
        (1 - hatMatrix X i i)) =
      (fun ω => (1 - hatMatrix X i i)⁻¹ * ((∑ j, X i j * β j) + ε i ω -
      (X *ᵥ olsCoefficients X (fun k => (∑ j, X k j * β j) + ε k ω)) i) ^ 2) := by
    funext ω
    simp [div_eq_mul_inv, mul_comm]
  rw [he, P.mean_smul, ols_residual_second_moment P X β ε σ2 h hzero hpair i]
  have hn : 1 - hatMatrix X i i ≠ 0 := ne_of_gt (sub_pos.mpr hi)
  field_simp

end StatadicFormal.Estimation.Linear
