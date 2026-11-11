module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Tactic.Ring
public import Architect

public section
open scoped BigOperators InnerProductSpace
namespace StatadicFormal.Discrepancy.MMD

/-- Exact-real identity for the weighted linear-kernel V-statistic. -/
@[blueprint "STATADIC-MMD-002"
  (statement := /-- For finite families of vectors $x_i,y_j$ in a real inner
  product space, real weights $w_i,v_j$ of equal total mass, and any reference
  $r$, the linear-kernel V-statistic satisfies
  \[
  \sum_{i,k}w_iw_k\langle x_i,x_k\rangle+
  \sum_{j,l}v_jv_l\langle y_j,y_l\rangle-
   2\sum_{i,j}w_iv_j\langle x_i,y_j\rangle=
  \left\|\sum_i w_i(x_i-r)-\sum_jv_j(y_j-r)\right\|^2.
  \] -/)
  (proof := /-- Equal total mass cancels the reference. Expand the inner
  product of the difference of weighted sums and distribute both finite sums. -/)]
theorem linear_mmd_reduction {ι κ E : Type*} [Fintype ι] [Fintype κ]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x : ι → E) (y : κ → E) (w : ι → ℝ) (v : κ → ℝ) (r : E)
    (hmass : (∑ i, w i) = ∑ j, v j) :
    (∑ i, ∑ k, w i * w k * ⟪x i, x k⟫_ℝ) +
    (∑ j, ∑ l, v j * v l * ⟪y j, y l⟫_ℝ) -
    2 * (∑ i, ∑ j, w i * v j * ⟪x i, y j⟫_ℝ) =
    ‖(∑ i, w i • (x i - r)) - (∑ j, v j • (y j - r))‖ ^ 2 := by
  have href : (∑ i, w i • (x i - r)) - (∑ j, v j • (y j - r)) =
      (∑ i, w i • x i) - (∑ j, v j • y j) := by
    simp only [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul, hmass]
    abel
  rw [href, ← real_inner_self_eq_norm_sq]
  simp only [inner_sub_left, inner_sub_right, sum_inner, inner_sum,
    real_inner_smul_left, real_inner_smul_right]
  simp only [Finset.sum_sub_distrib]
  simp_rw [← mul_assoc]
  have hxx : (∑ i, ∑ k, w i * w k * ⟪x k, x i⟫_ℝ) =
      ∑ i, ∑ k, w i * w k * ⟪x i, x k⟫_ℝ := by
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro k _
    rw [real_inner_comm]
  have hyy : (∑ j, ∑ l, v j * v l * ⟪y l, y j⟫_ℝ) =
      ∑ j, ∑ l, v j * v l * ⟪y j, y l⟫_ℝ := by
    apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro l _
    rw [real_inner_comm]
  have hxy : (∑ i, ∑ j, w i * v j * ⟪y j, x i⟫_ℝ) =
      ∑ i, ∑ j, w i * v j * ⟪x i, y j⟫_ℝ := by
    simp_rw [real_inner_comm (y _)]
  have hyx : (∑ j, ∑ i, v j * w i * ⟪x i, y j⟫_ℝ) =
      ∑ i, ∑ j, w i * v j * ⟪x i, y j⟫_ℝ := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hxx, hyy, hxy, hyx]
  ring

end StatadicFormal.Discrepancy.MMD
