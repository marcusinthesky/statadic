module

public import StatadicFormal.Estimation.Linear.GaussMarkov
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-! Exact real full-column-rank contrast construction; Stępniak (2010),
Theorem 3.2 and Theorem 4.1, pp. 3–5. -/
open scoped BigOperators
open Matrix StatadicFormal.Probability
@[expose] public section

noncomputable section
namespace StatadicFormal.Estimation.Linear
variable {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq κ]

/-- Observation weights for an OLS contrast, using the inverse Gram matrix. -/
def olsContrastWeights (X : Matrix ι κ ℝ) (t : κ → ℝ) : ι → ℝ :=
  X *ᵥ ((Xᵀ * X)⁻¹ *ᵥ t)

theorem ols_weights_reproduce (X : Matrix ι κ ℝ) (t : κ → ℝ)
    (h : IsUnit (Xᵀ * X).det) :
    Xᵀ *ᵥ olsContrastWeights X t = t := by
  unfold olsContrastWeights
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ h]
  simp


/-- Inverse-Gram OLS coefficient vector; normal-equation results assume an invertible Gram matrix. -/
def olsCoefficients (X : Matrix ι κ ℝ) (y : ι → ℝ) : κ → ℝ :=
  (Xᵀ * X)⁻¹ *ᵥ (Xᵀ *ᵥ y)

/-- The inverse-Gram coefficient vector solves the normal equations. -/
theorem ols_normal_equations (X : Matrix ι κ ℝ) (y : ι → ℝ)
    (h : IsUnit (Xᵀ * X).det) :
    (Xᵀ * X) *ᵥ olsCoefficients X y = Xᵀ *ᵥ y := by
  unfold olsCoefficients
  rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ h]
  simp

/-- A contrast of the normal-equation solution equals the corresponding
linear combination of observations. -/
theorem ols_contrast_eq (X : Matrix ι κ ℝ) (t : κ → ℝ) (y : ι → ℝ) :
    t ⬝ᵥ olsCoefficients X y = olsContrastWeights X t ⬝ᵥ y := by
  unfold olsCoefficients olsContrastWeights
  have hs : ((Xᵀ * X)⁻¹)ᵀ = (Xᵀ * X)⁻¹ := by
    rw [Matrix.transpose_nonsing_inv, Matrix.transpose_mul]
    simp
  calc
    t ⬝ᵥ (Xᵀ * X)⁻¹ *ᵥ (Xᵀ *ᵥ y) =
        (Xᵀ *ᵥ y) ⬝ᵥ (Xᵀ * X)⁻¹ *ᵥ t := by
      simpa only [hs] using (Matrix.dotProduct_transpose_mulVec ((Xᵀ * X)⁻¹) t (Xᵀ *ᵥ y))
    _ = y ⬝ᵥ X *ᵥ ((Xᵀ * X)⁻¹ *ᵥ t) := by
      simpa only [dotProduct_comm (Xᵀ *ᵥ y)] using (Matrix.dotProduct_transpose_mulVec X ((Xᵀ * X)⁻¹ *ᵥ t) y)
    _ = (X *ᵥ ((Xᵀ * X)⁻¹ *ᵥ t)) ⬝ᵥ y := dotProduct_comm _ _


/-- The explicitly constructed full-rank OLS contrast is unbiased and BLUE
in a finite spherical model; this is a scalar finite-law specialization of
Stępniak (2010), Theorem 4.1, pp. 4–5. -/
@[blueprint "STATADIC-LINEAR-005"
  (statement := /-- Let $X\in\mathbb R^{n\times p}$ have invertible $X^TX$,
  $t,\beta\in\mathbb R^p$, and $Y=X\beta+\varepsilon$ on a finite probability
  space, with $\mathbb E\varepsilon=0$ and
  $\mathbb E(\varepsilon\varepsilon^T)=\sigma^2 I_n$, $\sigma^2>0$. Put
  \[ \hat\beta=(X^TX)^{-1}X^TY. \]
  For any $a\in\mathbb R^n$ satisfying $X^Ta=t$, the exact-real OLS contrast
  satisfies
  \[ \mathbb E(t^T\hat\beta)=t^T\beta,\qquad
     \operatorname{Var}(t^T\hat\beta)\leq\operatorname{Var}(a^TY). \]
  -/)
  (proof := /-- As in Stępniak (2010), Theorem 4.1, pp. 4--5,
  set $b=X(X^TX)^{-1}t$. The Gram construction gives $X^Tb=t$ and
  $b\in\operatorname{col}(X)$.
  Hence $b^T(a-b)=0$, so the spherical moment assumptions imply
  \[ \operatorname{Var}(a^TY)-\operatorname{Var}(b^TY)
     =\sigma^2\sum_i(a_i-b_i)^2\geq0. \]
  The normal equations identify $b^TY=t^T\hat\beta$. -/)]
theorem finite_fullrank_ols_contrast {Ω : Type*} [Fintype Ω] [DecidableEq ι]
    (P : FiniteLaw Ω) (X : Matrix ι κ ℝ) (a : ι → ℝ) (t β : κ → ℝ)
    (ε : ι → Ω → ℝ) (σ2 : ℝ) (hσ : 0 < σ2)
    (hgram : IsUnit (Xᵀ * X).det)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0)
    (hrepA : ∀ j, ∑ i, a i * X i j = t j) :
    P.mean (fun ω => t ⬝ᵥ olsCoefficients X (fun i => (∑ j, X i j * β j) + ε i ω)) = ∑ j, t j * β j ∧
    P.variance (fun ω => t ⬝ᵥ olsCoefficients X (fun i => (∑ j, X i j * β j) + ε i ω)) ≤
    P.variance (estimate a (fun i ω => (∑ j, X i j * β j) + ε i ω)) := by
  simp_rw [ols_contrast_eq]
  change P.mean (estimate (olsContrastWeights X t) (fun i ω => (∑ j, X i j * β j) + ε i ω)) = _ ∧
    P.variance (estimate (olsContrastWeights X t) (fun i ω => (∑ j, X i j * β j) + ε i ω)) ≤ _
  apply finite_gauss_markov_contrast P X a (olsContrastWeights X t)
    ((Xᵀ * X)⁻¹ *ᵥ t) t β ε σ2 hσ hzero hpair
  · intro i
    simp [olsContrastWeights, Matrix.mulVec, dotProduct, mul_comm]
  · intro j
    have heq := congrFun (ols_weights_reproduce X t hgram) j
    simpa [Matrix.mulVec, dotProduct, mul_comm] using heq
  · exact hrepA


/-- Under invertibility, the normal equations have the unique inverse-Gram solution.
This is the full-column-rank case of Stępniak (2010), Theorem 3.2(d), p. 3. -/
theorem ols_normal_equations_unique (X : Matrix ι κ ℝ) (y : ι → ℝ)
    (h : IsUnit (Xᵀ * X).det) (b : κ → ℝ)
    (hb : (Xᵀ * X) *ᵥ b = Xᵀ *ᵥ y) : b = olsCoefficients X y := by
  have heq := congrArg (fun v ↦ (Xᵀ * X)⁻¹ *ᵥ v) hb
  simpa [olsCoefficients, Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ h] using heq

/-- The minimum-variance contrast weights are unique when the spherical error
variance is positive. This is the scalar finite-law full-rank specialization
of uniqueness in Stępniak (2010), Theorem 4.1, pp. 4–5. -/
theorem finite_fullrank_ols_weights_unique {Ω : Type*} [Fintype Ω] [DecidableEq ι]
    (P : FiniteLaw Ω) (X : Matrix ι κ ℝ) (a : ι → ℝ) (t : κ → ℝ)
    (ε : ι → Ω → ℝ) (σ2 : ℝ) (hσ : 0 < σ2)
    (hgram : IsUnit (Xᵀ * X).det)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω ↦ ε i ω * ε j ω) =
      if i = j then σ2 else 0)
    (hrepA : ∀ j, ∑ i, a i * X i j = t j)
    (heq : P.variance (estimate a ε) =
      P.variance (estimate (olsContrastWeights X t) ε)) :
    a = olsContrastWeights X t := by
  have hnormal : ∀ i, olsContrastWeights X t i =
      ∑ j, ((Xᵀ * X)⁻¹ *ᵥ t) j * X i j := by
    intro i
    simp [olsContrastWeights, Matrix.mulVec, dotProduct, mul_comm]
  have hsame : ∀ j, ∑ i, a i * X i j =
      ∑ i, olsContrastWeights X t i * X i j := by
    intro j
    have h := congrFun (ols_weights_reproduce X t hgram) j
    rw [hrepA j]
    simpa [Matrix.mulVec, dotProduct, mul_comm] using h.symm
  have hgap := gauss_markov_gap P X a (olsContrastWeights X t)
    ((Xᵀ * X)⁻¹ *ᵥ t) ε σ2 hzero hpair hnormal hsame
  rw [heq, sub_self] at hgap
  have hsum : (∑ i, (a i - olsContrastWeights X t i) ^ 2) = 0 := by
    exact (mul_eq_zero.mp hgap.symm).resolve_left (ne_of_gt hσ)
  funext i
  have hi : (a i - olsContrastWeights X t i) ^ 2 = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun i _ ↦ sq_nonneg _)).mp hsum i (Finset.mem_univ i)
  exact sub_eq_zero.mp (sq_eq_zero_iff.mp hi)


/-- Residual sum of squares for an exact-real finite linear design. -/
def residualSumSquares (X : Matrix ι κ ℝ) (y : ι → ℝ) (b : κ → ℝ) : ℝ :=
  ∑ i, (y i - (X *ᵥ b) i) ^ 2

omit [DecidableEq κ] in
/-- A normal-equation solution minimizes the residual sum of squares, the
forward direction of Stępniak (2010), Theorem 3.2(b), p. 3. -/
theorem normal_equations_minimize_residuals (X : Matrix ι κ ℝ) (y : ι → ℝ)
    (b : κ → ℝ) (hnormal : (Xᵀ * X) *ᵥ b = Xᵀ *ᵥ y) (v : κ → ℝ) :
    residualSumSquares X y b ≤ residualSumSquares X y v := by
  let r : ι → ℝ := fun i ↦ y i - (X *ᵥ b) i
  let d : κ → ℝ := fun j ↦ v j - b j
  have hr : Xᵀ *ᵥ r = 0 := by
    change Xᵀ *ᵥ (y - X *ᵥ b) = 0
    rw [Matrix.mulVec_sub, Matrix.mulVec_mulVec, hnormal]
    exact sub_self _
  have hort : ∑ i, r i * (X *ᵥ d) i = 0 := by
    have h := Matrix.dotProduct_transpose_mulVec X d r
    rw [hr] at h
    simpa [dotProduct, mul_comm] using h.symm
  have hdecomp : residualSumSquares X y v = residualSumSquares X y b +
      (∑ i, (X *ᵥ d) i ^ 2) - 2 * ∑ i, r i * (X *ᵥ d) i := by
    unfold residualSumSquares
    have hx : ∀ i, (X *ᵥ v) i = (X *ᵥ b) i + (X *ᵥ d) i := by
      intro i
      change (X *ᵥ v) i = (X *ᵥ b) i + (X *ᵥ (v - b)) i
      rw [Matrix.mulVec_sub]
      simp
    simp_rw [hx]
    simp only [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    dsimp [r]
    ring
  rw [hort] at hdecomp
  have hnon : 0 ≤ ∑ i, (X *ᵥ d) i ^ 2 :=
    Finset.sum_nonneg (fun i _ ↦ sq_nonneg _)
  linarith

omit [DecidableEq κ] in
/-- Pythagorean residual identity in Stępniak (2010), equation (3.5), p. 3. -/
theorem normal_equations_residual_gap (X : Matrix ι κ ℝ) (y : ι → ℝ)
    (b v : κ → ℝ) (hnormal : (Xᵀ * X) *ᵥ b = Xᵀ *ᵥ y) :
    residualSumSquares X y v = residualSumSquares X y b + ∑ i, (X *ᵥ (v - b)) i ^ 2 := by
  let r : ι → ℝ := fun i ↦ y i - (X *ᵥ b) i
  let d : κ → ℝ := fun j ↦ v j - b j
  have hr : Xᵀ *ᵥ r = 0 := by
    change Xᵀ *ᵥ (y - X *ᵥ b) = 0
    rw [Matrix.mulVec_sub, Matrix.mulVec_mulVec, hnormal]
    exact sub_self _
  have hort : ∑ i, r i * (X *ᵥ d) i = 0 := by
    have h := Matrix.dotProduct_transpose_mulVec X d r
    rw [hr] at h
    simpa [dotProduct, mul_comm] using h.symm
  have hdecomp : residualSumSquares X y v = residualSumSquares X y b +
      (∑ i, (X *ᵥ d) i ^ 2) - 2 * ∑ i, r i * (X *ᵥ d) i := by
    unfold residualSumSquares
    have hx : ∀ i, (X *ᵥ v) i = (X *ᵥ b) i + (X *ᵥ d) i := by
      intro i
      change (X *ᵥ v) i = (X *ᵥ b) i + (X *ᵥ (v - b)) i
      rw [Matrix.mulVec_sub]
      simp
    simp_rw [hx]
    simp only [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    dsimp [r]
    ring
  rw [hort] at hdecomp
  change residualSumSquares X y v = residualSumSquares X y b +
      (∑ i, (X *ᵥ (v - b)) i ^ 2) - 2 * 0 at hdecomp
  simpa using hdecomp

/-- Full-column-rank form of the normal-equation/minimizer equivalence in
Stępniak (2010), Theorem 3.2(b), p. 3. -/
theorem normal_equations_iff_minimizes (X : Matrix ι κ ℝ) (y : ι → ℝ)
    (hgram : IsUnit (Xᵀ * X).det) (b : κ → ℝ) :
    ((Xᵀ * X) *ᵥ b = Xᵀ *ᵥ y) ↔
      (∀ v, residualSumSquares X y b ≤ residualSumSquares X y v) := by
  constructor
  · intro h v
    exact normal_equations_minimize_residuals X y b h v
  · intro hmin
    let z := olsCoefficients X y
    have hz : (Xᵀ * X) *ᵥ z = Xᵀ *ᵥ y := ols_normal_equations X y hgram
    have hgap := normal_equations_residual_gap X y z b hz
    have hsum : (∑ i, (X *ᵥ (b - z)) i ^ 2) = 0 := by
      have hnon : 0 ≤ ∑ i, (X *ᵥ (b - z)) i ^ 2 :=
        Finset.sum_nonneg (fun i _ ↦ sq_nonneg _)
      have horder := hmin z
      linarith
    have hzero : X *ᵥ (b - z) = 0 := by
      funext i
      have hi := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ ↦ sq_nonneg _)).mp hsum i (Finset.mem_univ i)
      exact sq_eq_zero_iff.mp hi
    have hxb : X *ᵥ b = X *ᵥ z := by
      rw [Matrix.mulVec_sub, sub_eq_zero] at hzero
      exact hzero
    calc
      (Xᵀ * X) *ᵥ b = Xᵀ *ᵥ (X *ᵥ b) := (Matrix.mulVec_mulVec _ _ _).symm
      _ = Xᵀ *ᵥ (X *ᵥ z) := by rw [hxb]
      _ = (Xᵀ * X) *ᵥ z := Matrix.mulVec_mulVec _ _ _
      _ = Xᵀ *ᵥ y := hz

end StatadicFormal.Estimation.Linear
