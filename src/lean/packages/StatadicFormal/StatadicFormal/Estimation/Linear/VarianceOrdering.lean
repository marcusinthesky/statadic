module

public import StatadicFormal.Estimation.Linear.OLS
public import Mathlib.Probability.Moments.Variance

/-!
# Gauss–Markov variance ordering on a probability space

The design and the finite list of observations are fixed in size. Their joint
sampling distribution may be continuous. Only second moments enter the ordering.
-/

@[expose] public section
noncomputable section

open scoped BigOperators
open Matrix MeasureTheory ProbabilityTheory

namespace StatadicFormal.Estimation.Linear

variable {Ω ι κ : Type*} [MeasurableSpace Ω] [Fintype ι] [DecidableEq ι]

/-- Spherical covariance reduces the variance of a linear estimator to a sum of squares. -/
theorem variance_dot_spherical (P : Measure Ω) [IsProbabilityMeasure P] (Y : ι → Ω → ℝ)
    (a : ι → ℝ) (v : ℝ) (hY : ∀ i, MemLp (Y i) 2 P)
    (hc : ∀ i j, covariance (Y i) (Y j) P = if i = j then v else 0) :
    variance (fun ω ↦ a ⬝ᵥ (fun i ↦ Y i ω)) P = v * ∑ i, a i ^ 2 := by
  unfold dotProduct
  rw [variance_fun_sum (fun i ↦ (hY i).const_mul (a i)), Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  simp only [covariance_const_mul_left, covariance_const_mul_right, hc, mul_ite, mul_zero,
    Finset.sum_ite_eq, Finset.mem_univ, ite_true]
  ring

variable [Fintype κ] [DecidableEq κ]

/-- Exact improvement from projecting any unbiased contrast weights onto the design space. -/
@[blueprint "STATADIC-LINEAR-010"
  (statement := /-- Let $X$ be fixed with $X^TX$ invertible. Let the response
  vector $Y$ have mean $X\beta$, square-integrable coordinates and covariance $vI$, $v\ge0$.
  For any contrast $t$ and any weights $a$ with $X^Ta=t$, put $b=X(X^TX)^{-1}t$.
  Then $\operatorname{Var}(a^TY)-\operatorname{Var}(t^T\hat\beta)
  = v\sum_i(a_i-b_i)^2\ge0$. In the linear mean model $a^TY$ and
  $t^T\hat\beta$ estimate the same contrast without bias. -/)
  (proof := /-- The OLS contrast equals $b^TY$, and $X^Tb=t$. Taking expectations
  gives $b^TX\beta=a^TX\beta=t^T\beta$.
  Hence $X^T(a-b)=0$ while $b$ lies in the column space of $X$, giving
  $b^T(a-b)=0$. Square integrability permits expansion of each variance as
  a finite double sum of covariances. The spherical covariance leaves only
  $v\sum_i a_i^2$ or $v\sum_i b_i^2$. The Pythagorean identity gives the gap. -/)]
theorem ols_variance_ordering (P : Measure Ω) [IsProbabilityMeasure P]
    (X : Matrix ι κ ℝ) (t β : κ → ℝ) (a : ι → ℝ) (Y : ι → Ω → ℝ) (v : ℝ)
    (hgram : IsUnit (Xᵀ * X).det) (ha : Xᵀ *ᵥ a = t) (hv : 0 ≤ v)
    (hY : ∀ i, MemLp (Y i) 2 P) (hmean : ∀ i, ∫ ω, Y i ω ∂P = (X *ᵥ β) i)
    (hc : ∀ i j, covariance (Y i) (Y j) P = if i = j then v else 0) :
    let Vₐ := variance (fun ω ↦ a ⬝ᵥ (fun i ↦ Y i ω)) P
    let Vₒ := variance (fun ω ↦ t ⬝ᵥ olsCoefficients X (fun i ↦ Y i ω)) P
    (∫ ω, a ⬝ᵥ (fun i ↦ Y i ω) ∂P) = t ⬝ᵥ β ∧
    (∫ ω, t ⬝ᵥ olsCoefficients X (fun i ↦ Y i ω) ∂P) = t ⬝ᵥ β ∧
    Vₐ - Vₒ = v * ∑ i, (a i - olsContrastWeights X t i) ^ 2 ∧ Vₒ ≤ Vₐ := by
  dsimp only
  simp_rw [ols_contrast_eq]
  have mean_repro (u : ι → ℝ) (hu : Xᵀ *ᵥ u = t) :
      (∫ ω, u ⬝ᵥ (fun i ↦ Y i ω) ∂P) = t ⬝ᵥ β := by
    calc
      _ = u ⬝ᵥ (X *ᵥ β) := by
        unfold dotProduct
        rw [integral_finsetSum _ (fun i _ ↦ ((hY i).integrable (by norm_num)).const_mul (u i))]
        simp only [integral_const_mul, hmean]
      _ = β ⬝ᵥ (Xᵀ *ᵥ u) := (Matrix.dotProduct_transpose_mulVec X β u).symm
      _ = t ⬝ᵥ β := by rw [hu, dotProduct_comm]
  refine ⟨mean_repro a ha, mean_repro _ (ols_weights_reproduce X t hgram), ?_⟩
  rw [variance_dot_spherical P Y a v hY hc,
    variance_dot_spherical P Y (olsContrastWeights X t) v hY hc]
  have hort := normal_orthogonality X a (olsContrastWeights X t) ((Xᵀ * X)⁻¹ *ᵥ t)
    (fun i ↦ by simp [olsContrastWeights, Matrix.mulVec, dotProduct, mul_comm]) (fun j ↦ by
      have hj := congrFun (ha.trans (ols_weights_reproduce X t hgram).symm) j
      simpa [Matrix.mulVec, dotProduct, mul_comm] using hj)
  rw [squares_decomposition _ _ hort, mul_add]
  have hnon := mul_nonneg hv (Finset.sum_nonneg (s := Finset.univ)
    (fun i _ ↦ sq_nonneg (a i - olsContrastWeights X t i)))
  exact ⟨by ring, le_add_of_nonneg_right hnon⟩

end StatadicFormal.Estimation.Linear
