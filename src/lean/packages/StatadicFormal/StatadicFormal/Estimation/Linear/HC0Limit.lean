module

public import Architect
public import Mathlib.Analysis.Normed.Field.Lemmas
public import Mathlib.Topology.Instances.Matrix
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-! # HC0 covariance along convergent sample paths

White (1980), Theorem 1(i)–(ii). We consider the fixed-limit case and
separate the moment laws of large numbers from residual replacement and
continuity of the normalized sandwich. -/

@[expose] public section
noncomputable section

open Filter Matrix
open scoped Topology BigOperators

namespace StatadicFormal.Estimation.Linear

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Normalized outer products of the score rows of one sample. -/
def hc0Meat {n : ℕ} (X : Matrix (Fin (n+1)) ι ℝ) (u : Fin (n+1) → ℝ) : Matrix ι ι ℝ :=
  fun i j ↦ (∑ a, X a i * X a j * (u a)^2) / (n+1)

/-- Sample Gram per observation. Each triangular array has at least one row. -/
def hc0Gram {n : ℕ} (X : Matrix (Fin (n+1)) ι ℝ) : Matrix ι ι ℝ :=
  ((n+1 : ℝ)⁻¹) • (Xᵀ * X)

/-- Scaling the unnormalized Gram and meat by the sample count converts the
coefficient covariance to the covariance on the square-root-sample-size scale. -/
theorem hc0_scaling (G B : Matrix ι ι ℝ) (c : ℝ) (hc : c ≠ 0) (hg : IsUnit G.det) :
    ((c⁻¹ • G)⁻¹ * (c⁻¹ • B) * (c⁻¹ • G)⁻¹) = c • (G⁻¹ * B * G⁻¹) := by
  let _ := invertibleOfNonzero (inv_ne_zero hc)
  rw [Matrix.inv_smul _ _ hg]
  simp [invOf_eq_inv, smul_smul, hc]

private theorem sandwich_tendsto (G B : ℕ → Matrix ι ι ℝ) (A V : Matrix ι ι ℝ)
    (hg : Tendsto G atTop (𝓝 A)) (hb : Tendsto B atTop (𝓝 V)) (ha : A.det ≠ 0) :
    Tendsto (fun n ↦ (G n)⁻¹ * B n * (G n)⁻¹) atTop (𝓝 (A⁻¹ * V * A⁻¹)) := by
  have hinv : ContinuousAt Ring.inverse A.det := by
    simpa only [Ring.inverse_eq_inv'] using (continuousAt_inv₀ ha)
  have hi := (continuousAt_matrix_inv A hinv).tendsto.comp hg
  exact (hi.mul hb).mul hi

omit [DecidableEq ι] in
/-- A uniform design bound turns coefficient error into a prediction error
bound common to every row, even when the number of rows changes. -/
theorem hc0_prediction_bound {n : ℕ} (X : Matrix (Fin (n+1)) ι ℝ)
    (b β : ι → ℝ) (K : ℝ) (hx : ∀ a i, |X a i| ≤ K) (a : Fin (n+1)) :
    |(X *ᵥ (b - β)) a| ≤ K * ∑ i, |b i - β i| := by
  change |∑ i, X a i * (b i - β i)| ≤ _
  rw [Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
  intro i _
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_right (hx a i) (abs_nonneg _)

omit [Fintype ι] [DecidableEq ι] in
private theorem hc0_meat_bound {n : ℕ} (X : Matrix (Fin (n+1)) ι ℝ)
    (r u : Fin (n+1) → ℝ) (i j : ι) (δ C : ℝ) (hδ : 0 ≤ δ)
    (herr : ∀ a, |r a - u a| ≤ δ)
    (hmoment : (∑ a, |X a i * X a j| * |r a + u a|) / (n+1) ≤ C) :
    |hc0Meat X r i j - hc0Meat X u i j| ≤ C * δ := by
  have hn : (0 : ℝ) < n + 1 := by positivity
  unfold hc0Meat
  rw [← sub_div, ← Finset.sum_sub_distrib, abs_div, abs_of_pos hn]
  calc
    _ ≤ (∑ a, |X a i * X a j * (r a)^2 - X a i * X a j * (u a)^2|) / (n+1) :=
      div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _) hn.le
    _ ≤ (∑ a, (|X a i * X a j| * |r a + u a|) * δ) / (n+1) := by
      apply div_le_div_of_nonneg_right _ hn.le
      apply Finset.sum_le_sum
      intro a _
      rw [← mul_sub, sq_sub_sq, ← mul_assoc, abs_mul, abs_mul]
      exact mul_le_mul_of_nonneg_left (herr a) (by positivity)
    _ = ((∑ a, |X a i * X a j| * |r a + u a|) / (n+1)) * δ := by
      rw [← Finset.sum_mul]
      ring
    _ ≤ C * δ := mul_le_mul_of_nonneg_right hmoment hδ

/-- The fitted-residual normalized sandwich has the oracle limit under
pathwise sample-moment convergence and a vanishing maximum prediction error. -/
@[blueprint "STATADIC-LINEAR-012"
  (statement := /-- Write $G_n=X_n^T X_n/N$, $B_n(v)=X_n^T\operatorname{diag}(v^2)X_n/N$
  where $N=n+1$. Suppose $G_n\to A$, $B_n(u)\to V$ for true errors $u$,
  and $A$ is nonsingular. Suppose the maximum absolute difference between
  fitted residuals $r$ and $u$ tends to zero, while the sample means of
  $|x_i x_j||r+u|$ are bounded. Then $G_n^{-1}B_n(r)G_n^{-1}\to A^{-1}VA^{-1}$. -/)
  (proof := /-- Factor $r^2-u^2=(r-u)(r+u)$. For each meat entry, its
  replacement error is bounded by the residual difference bound times a
  bounded sample moment. Squeeze each entry to zero, add the oracle limit,
  and use continuity of inversion at $A$ and of matrix multiplication. -/)]
theorem hc0_sandwich_tendsto (X : (n : ℕ) → Matrix (Fin (n+1)) ι ℝ)
    (r u : (n : ℕ) → Fin (n+1) → ℝ) (A V : Matrix ι ι ℝ)
    (hg : Tendsto (fun n ↦ hc0Gram (X n)) atTop (𝓝 A))
    (hu : Tendsto (fun n ↦ hc0Meat (X n) (u n)) atTop (𝓝 V))
    (ha : A.det ≠ 0) (δ : ℕ → ℝ) (C : ℝ)
    (hδ : Tendsto δ atTop (𝓝 0)) (hδ0 : ∀ n, 0 ≤ δ n)
    (herr : ∀ n a, |r n a - u n a| ≤ δ n)
    (hmoment : ∀ n i j, (∑ a, |X n a i * X n a j| * |r n a + u n a|) / (n+1) ≤ C) :
    Tendsto (fun n ↦ (hc0Gram (X n))⁻¹ * hc0Meat (X n) (r n) * (hc0Gram (X n))⁻¹)
      atTop (𝓝 (A⁻¹ * V * A⁻¹)) := by
  apply sandwich_tendsto _ _ _ _ hg _ ha
  have hdiff : Tendsto (fun n ↦ hc0Meat (X n) (r n) - hc0Meat (X n) (u n))
      atTop (𝓝 0) := by
    apply tendsto_pi_nhds.mpr
    intro i
    apply tendsto_pi_nhds.mpr
    intro j
    apply squeeze_zero_norm (a := fun n ↦ C * δ n)
    · intro n
      exact hc0_meat_bound (X n) (r n) (u n) i j (δ n) C (hδ0 n) (herr n) (hmoment n i j)
    · simpa using (tendsto_const_nhds (x := C)).mul hδ
  simpa using hdiff.add hu

end StatadicFormal.Estimation.Linear
