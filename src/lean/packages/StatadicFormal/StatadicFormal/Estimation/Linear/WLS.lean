module

public import StatadicFormal.Estimation.Linear.OLS
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! Weighted least squares through an explicitly whitened design. -/
open scoped BigOperators
open Matrix StatadicFormal.Probability
@[expose] public section
noncomputable section
namespace StatadicFormal.Estimation.Linear

variable {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

/-- Design with each row scaled by the square root of its observation weight. -/
def whitenedDesign (X : Matrix ι κ ℝ) (w : ι → ℝ) : Matrix ι κ ℝ :=
  Matrix.diagonal (fun i => Real.sqrt (w i)) * X

/-- Response scaled by the square roots of the observation weights. -/
def whitenedResponse (y w : ι → ℝ) : ι → ℝ :=
  Matrix.diagonal (fun i => Real.sqrt (w i)) *ᵥ y

/-- Weighted inverse-Gram coefficient vector; later results state the rank and weight assumptions. -/
def wlsCoefficients (X : Matrix ι κ ℝ) (y w : ι → ℝ) : κ → ℝ :=
  (Xᵀ * Matrix.diagonal w * X)⁻¹ *ᵥ (Xᵀ *ᵥ (Matrix.diagonal w *ᵥ y))

private lemma sqrt_diagonal_mul_self (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) :
    Matrix.diagonal (fun i => Real.sqrt (w i)) *
      Matrix.diagonal (fun i => Real.sqrt (w i)) = Matrix.diagonal w := by
  rw [Matrix.diagonal_mul_diagonal]
  congr 1
  funext i
  exact Real.mul_self_sqrt (hw i)

omit [Fintype κ] [DecidableEq κ] in
theorem whitened_gram (X : Matrix ι κ ℝ) (w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) :
    (whitenedDesign X w)ᵀ * whitenedDesign X w = Xᵀ * Matrix.diagonal w * X := by
  simp only [whitenedDesign, Matrix.transpose_mul, Matrix.diagonal_transpose]
  rw [← Matrix.mul_assoc, Matrix.mul_assoc (Xᵀ), sqrt_diagonal_mul_self w hw]

omit [Fintype κ] [DecidableEq κ] in
theorem whitened_rhs (X : Matrix ι κ ℝ) (y w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) :
    (whitenedDesign X w)ᵀ *ᵥ whitenedResponse y w =
      Xᵀ *ᵥ (Matrix.diagonal w *ᵥ y) := by
  simp only [whitenedDesign, whitenedResponse, Matrix.transpose_mul, Matrix.diagonal_transpose,
    Matrix.mulVec_mulVec]
  rw [Matrix.mul_assoc, sqrt_diagonal_mul_self w hw]

theorem wls_eq_whitened_ols (X : Matrix ι κ ℝ) (y w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) :
    wlsCoefficients X y w = olsCoefficients (whitenedDesign X w) (whitenedResponse y w) := by
  unfold wlsCoefficients olsCoefficients
  rw [whitened_gram X w hw, whitened_rhs X y w hw]

theorem wls_normal_equations (X : Matrix ι κ ℝ) (y w : ι → ℝ)
    (h : IsUnit (Xᵀ * Matrix.diagonal w * X).det) :
    (Xᵀ * Matrix.diagonal w * X) *ᵥ wlsCoefficients X y w =
      Xᵀ *ᵥ (Matrix.diagonal w *ᵥ y) := by
  unfold wlsCoefficients
  rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ h]
  simp

omit [DecidableEq κ] in
/-- Weighted residual sum of squares is the ordinary residual sum after whitening. -/
theorem whitened_residuals (X : Matrix ι κ ℝ) (y w : ι → ℝ) (b : κ → ℝ)
    (hw : ∀ i, 0 ≤ w i) :
    residualSumSquares (whitenedDesign X w) (whitenedResponse y w) b =
      ∑ i, w i * (y i - (X *ᵥ b) i) ^ 2 := by
  unfold residualSumSquares whitenedDesign whitenedResponse
  rw [← Matrix.mulVec_mulVec]
  apply Finset.sum_congr rfl
  intro i _
  simp only [Matrix.mulVec_diagonal]
  rw [← mul_sub, mul_pow, Real.sq_sqrt (hw i)]

theorem wls_unique_minimizer (X : Matrix ι κ ℝ) (y w : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (h : IsUnit (Xᵀ * Matrix.diagonal w * X).det) :
    (∀ v, (∑ i, w i * (y i - (X *ᵥ wlsCoefficients X y w) i) ^ 2) ≤
      ∑ i, w i * (y i - (X *ᵥ v) i) ^ 2) ∧
    (∀ b, (∀ v, (∑ i, w i * (y i - (X *ᵥ b) i) ^ 2) ≤
      ∑ i, w i * (y i - (X *ᵥ v) i) ^ 2) → b = wlsCoefficients X y w) := by
  have hg : IsUnit ((whitenedDesign X w)ᵀ * whitenedDesign X w).det := by
    rwa [whitened_gram X w hw]
  constructor
  · intro v
    rw [← whitened_residuals X y w _ hw, ← whitened_residuals X y w _ hw,
      wls_eq_whitened_ols X y w hw]
    exact normal_equations_minimize_residuals _ _ _ (ols_normal_equations _ _ hg) v
  · intro b hb
    have hb' : ∀ v, residualSumSquares (whitenedDesign X w) (whitenedResponse y w) b ≤
        residualSumSquares (whitenedDesign X w) (whitenedResponse y w) v := by
      intro v
      simpa only [whitened_residuals X y w _ hw] using hb v
    have hn := (normal_equations_iff_minimizes _ _ hg b).mpr hb'
    rw [wls_eq_whitened_ols X y w hw]
    exact ols_normal_equations_unique _ _ hg b hn

/-- Aitken's ordering obtained by applying the existing full-rank OLS theorem
to the whitened observations. Competitors are expressed in whitened units. -/
@[blueprint "STATADIC-LINEAR-006"
  (statement := /-- Let $W=\operatorname{diag}(w_i)$ with $w_i>0$ and let
  $X^TWX$ be invertible. On a finite probability space suppose
  $Y=X\beta+\varepsilon$, $\mathbb E\varepsilon=0$, and
  $\mathbb E(\varepsilon_i\varepsilon_j)=\mathbf 1_{i=j}\sigma^2/w_i$
  for $\sigma^2>0$. Put $\hat\beta_W=(X^TWX)^{-1}X^TWY$,
  $\widetilde X=W^{1/2}X$, and $\widetilde Y=W^{1/2}Y$.
  For any $a$ with $\widetilde X^Ta=t$,
  \[\mathbb E(t^T\hat\beta_W)=t^T\beta,\qquad
  \operatorname{Var}(t^T\hat\beta_W)\leq\operatorname{Var}(a^T\widetilde Y).\]
  -/)
  (proof := /-- Whitening gives spherical errors and the same inverse-Gram
  coefficient vector. Apply the full-rank finite-law OLS contrast theorem to
  the whitened design and observations. -/)]
theorem finite_wls_contrast {Ω : Type*} [Fintype Ω]
    (P : FiniteLaw Ω) (X : Matrix ι κ ℝ) (w : ι → ℝ) (a : ι → ℝ)
    (t β : κ → ℝ) (ε : ι → Ω → ℝ) (σ2 : ℝ) (hσ : 0 < σ2)
    (hw : ∀ i, 0 < w i) (hgram : IsUnit (Xᵀ * Matrix.diagonal w * X).det)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 / w i else 0)
    (hrep : ∀ j, ∑ i, a i * whitenedDesign X w i j = t j) :
    P.mean (fun ω => t ⬝ᵥ wlsCoefficients X
      (fun i => (∑ j, X i j * β j) + ε i ω) w) = ∑ j, t j * β j ∧
    P.variance (fun ω => t ⬝ᵥ wlsCoefficients X
      (fun i => (∑ j, X i j * β j) + ε i ω) w) ≤
    P.variance (estimate a (fun i ω => Real.sqrt (w i) *
      ((∑ j, X i j * β j) + ε i ω))) := by
  let e : ι → Ω → ℝ := fun i ω => Real.sqrt (w i) * ε i ω
  have hz : ∀ i, P.mean (e i) = 0 := by
    intro i
    simp [e, P.mean_smul, hzero]
  have hp : ∀ i j, P.mean (fun ω => e i ω * e j ω) =
      if i = j then σ2 else 0 := by
    intro i j
    have he : (fun ω => e i ω * e j ω) =
        (fun ω => (Real.sqrt (w i) * Real.sqrt (w j)) * (ε i ω * ε j ω)) := by
      funext ω
      dsimp [e]
      ring
    rw [he, P.mean_smul, hpair]
    by_cases hij : i = j
    · subst j
      simp only [ite_true]
      rw [Real.mul_self_sqrt (le_of_lt (hw i))]
      field_simp [ne_of_gt (hw i)]
    · simp [hij]
  have hg : IsUnit ((whitenedDesign X w)ᵀ * whitenedDesign X w).det := by
    rwa [whitened_gram X w (fun i => le_of_lt (hw i))]
  have ht := finite_fullrank_ols_contrast P (whitenedDesign X w) a t β e σ2 hσ hg hz hp hrep
  have hy : ∀ ω, (fun i => (∑ j, whitenedDesign X w i j * β j) + e i ω) =
      whitenedResponse (fun i => (∑ j, X i j * β j) + ε i ω) w := by
    intro ω
    funext i
    simp [whitenedDesign, whitenedResponse, e, Matrix.diagonal_mul, Matrix.mulVec_diagonal,
      Finset.mul_sum, mul_add, mul_assoc]
  have he : (fun i ω => (∑ j, whitenedDesign X w i j * β j) + e i ω) =
      (fun i ω => Real.sqrt (w i) * ((∑ j, X i j * β j) + ε i ω)) := by
    funext i ω
    have hi := congrFun (hy ω) i
    simpa [whitenedResponse, Matrix.mulVec_diagonal] using hi
  rw [he] at ht
  simp_rw [hy, ← wls_eq_whitened_ols X _ w (fun i => le_of_lt (hw i))] at ht
  exact ht

end StatadicFormal.Estimation.Linear
