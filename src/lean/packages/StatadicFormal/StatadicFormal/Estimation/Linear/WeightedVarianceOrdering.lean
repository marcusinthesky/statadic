module

public import StatadicFormal.Estimation.Linear.VarianceOrdering
public import StatadicFormal.Estimation.Linear.WLS

/-! # Variance ordering for known diagonal precision weights -/

@[expose] public section
noncomputable section

open Matrix MeasureTheory ProbabilityTheory

namespace StatadicFormal.Estimation.Linear

/-- Precision-weighted least squares is unbiased and efficient among linear unbiased contrasts. -/
@[blueprint "STATADIC-LINEAR-011"
  (statement := /-- Let positive precisions $w_i$ be fixed and
  $W=\operatorname{diag}(w)$. Assume $X^TWX$ is invertible, $E[Y]=X\beta$
  and $\operatorname{Cov}(Y)=vW^{-1}$, $v\ge0$, with square-integrable rows.
  For any $t$ and deterministic $a$ with $X^Ta=t$, the WLS contrast
  $t^T\hat\beta_W$ has expectation $t^T\beta$ and variance at most that of $a^TY$. -/)
  (proof := /-- Put $D=\operatorname{diag}(\sqrt{w_i})$. Transform the design
  and response to $DX,DY$, and transform the competitor to $D^{-1}a$.
  These changes preserve its contrast estimate and reproduction condition.
  The transformed covariance is $vI$. Apply the OLS variance ordering and use
  the weighted Gram identity to identify the whitened OLS fit with WLS. -/)]
theorem wls_variance_ordering {Ω ι κ : Type*} [MeasurableSpace Ω] [Fintype ι] [DecidableEq ι]
    [Fintype κ] [DecidableEq κ] (P : Measure Ω) [IsProbabilityMeasure P]
    (X : Matrix ι κ ℝ) (t β : κ → ℝ) (a w : ι → ℝ) (Y : ι → Ω → ℝ) (v : ℝ)
    (hw : ∀ i, 0 < w i) (hgram : IsUnit (Xᵀ * Matrix.diagonal w * X).det)
    (ha : Xᵀ *ᵥ a = t) (hv : 0 ≤ v) (hY : ∀ i, MemLp (Y i) 2 P)
    (hmean : ∀ i, ∫ ω, Y i ω ∂P = (X *ᵥ β) i)
    (hc : ∀ i j, covariance (Y i) (Y j) P = if i = j then v / w i else 0) :
    (∫ ω, t ⬝ᵥ wlsCoefficients X (fun i ↦ Y i ω) w ∂P) = t ⬝ᵥ β ∧
    variance (fun ω ↦ t ⬝ᵥ wlsCoefficients X (fun i ↦ Y i ω) w) P ≤
      variance (fun ω ↦ a ⬝ᵥ (fun i ↦ Y i ω)) P := by
  let Z := whitenedDesign X w
  let U (i : ι) (ω : Ω) := √(w i) * Y i ω
  let b (i : ι) := a i / √(w i)
  have hroot (i : ι) : √(w i) ≠ 0 := (Real.sqrt_pos.2 (hw i)).ne'
  have hDb : Matrix.diagonal (fun i ↦ √(w i)) *ᵥ b = a := by
    funext i
    simp only [Matrix.mulVec_diagonal, b]
    field_simp [hroot i]
  have hrep : Zᵀ *ᵥ b = t := by
    rw [← ha]
    funext j
    simp only [Z, whitenedDesign, Matrix.transpose_mul, Matrix.diagonal_transpose,
      ← Matrix.mulVec_mulVec, hDb]
  have hm (i : ι) : ∫ ω, U i ω ∂P = (Z *ᵥ β) i := by
    simp [U, integral_const_mul, hmean, Z, whitenedDesign, ← Matrix.mulVec_mulVec,
      Matrix.mulVec_diagonal]
  have hs (i j : ι) : covariance (U i) (U j) P = if i = j then v else 0 := by
    simp only [U, covariance_const_mul_left, covariance_const_mul_right, hc]
    by_cases e : i = j
    · subst j
      simp only [ite_true, ← mul_assoc, Real.mul_self_sqrt (hw i).le]
      field_simp [(hw i).ne']
    · simp [e]
  have hg : IsUnit (Zᵀ * Z).det := by rwa [whitened_gram X w (fun i ↦ (hw i).le)]
  have h := ols_variance_ordering P Z t β b U v hg hrep hv
    (fun i ↦ (hY i).const_mul _) hm hs
  have hz (ω : Ω) : (fun i ↦ U i ω) = whitenedResponse (fun i ↦ Y i ω) w := by
    funext i
    simp [U, whitenedResponse, Matrix.mulVec_diagonal]
  have hb (ω : Ω) : b ⬝ᵥ (fun i ↦ U i ω) = a ⬝ᵥ (fun i ↦ Y i ω) := by
    unfold dotProduct
    apply Finset.sum_congr rfl
    intro i _
    dsimp [b, U]
    field_simp [hroot i]
  refine ⟨?_, ?_⟩
  · simpa only [hz, Z, ← wls_eq_whitened_ols X _ w (fun i ↦ (hw i).le)] using h.2.1
  · have horder := h.2.2.2
    simp_rw [hb] at horder
    simpa only [hz, Z, ← wls_eq_whitened_ols X _ w (fun i ↦ (hw i).le)] using horder

end StatadicFormal.Estimation.Linear
