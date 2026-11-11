module

public import StatadicFormal.Estimation.Linear.OLS
public import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# OLS unbiasedness from first moments

The design is fixed. Errors may have arbitrary dependence and need only be
integrable, so unequal or nonexistent second moments do not affect the result.
-/

@[expose] public section
noncomputable section

open scoped BigOperators
open Matrix MeasureTheory

namespace StatadicFormal.Estimation.Linear

/-- Every coefficient is unbiased under integrable mean-zero errors on a probability space. -/
@[blueprint "STATADIC-LINEAR-009"
  (statement := /-- Let $X$ be fixed with $X^T X$ invertible, and let
  $Y=X\beta+\varepsilon$ with each $\varepsilon_i$ integrable and of mean zero.
  Under any probability law, $\mathbb E[\hat\beta_j]=\beta_j$ for every
  coefficient of $\hat\beta=(X^T X)^{-1}X^T Y$. -/)
  (proof := /-- Write $L=(X^TX)^{-1}X^T$. Invertibility gives $LX=I$,
  so $\hat\beta_j=\beta_j+\sum_i L_{ji}\varepsilon_i$.
  Integrate term by term over the finite row index. The constant integrates
  to $\beta_j$ and each integrable error has zero expectation. -/)]
theorem ols_mean_unbiased {Ω ι κ : Type*} [MeasurableSpace Ω] [Fintype ι] [Fintype κ]
    [DecidableEq κ] (P : Measure Ω) [IsProbabilityMeasure P] (X : Matrix ι κ ℝ)
    (β : κ → ℝ) (ε : ι → Ω → ℝ) (hgram : IsUnit (Xᵀ * X).det)
    (hint : ∀ i, Integrable (ε i) P) (hzero : ∀ i, ∫ ω, ε i ω ∂P = 0) (j : κ) :
    (∫ ω, olsCoefficients X (X *ᵥ β + fun i ↦ ε i ω) j ∂P) = β j := by
  let L := (Xᵀ * X)⁻¹ * Xᵀ
  have hLX : L * X = 1 := by
    unfold L
    rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hgram]
  have estimate_at (ω : Ω) :
      olsCoefficients X (X *ᵥ β + fun i ↦ ε i ω) j = β j + ∑ i, L j i * ε i ω := by
    unfold olsCoefficients
    rw [Matrix.mulVec_mulVec]
    change (L *ᵥ (X *ᵥ β + fun i ↦ ε i ω)) j = _
    rw [Matrix.mulVec_add, Matrix.mulVec_mulVec, hLX, Matrix.one_mulVec]
    rfl
  simp_rw [estimate_at]
  have hi (i : ι) : Integrable (fun ω ↦ L j i * ε i ω) P := (hint i).const_mul _
  rw [integral_add (integrable_const _) (integrable_finsetSum _ fun i _ ↦ hi i),
    integral_finsetSum _ fun i _ ↦ hi i]
  simp [integral_const_mul, hzero]

end StatadicFormal.Estimation.Linear
