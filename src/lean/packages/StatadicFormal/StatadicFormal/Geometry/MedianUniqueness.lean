module

public import StatadicFormal.Geometry.MedianCertificate
public import Mathlib.Analysis.InnerProductSpace.Convex
public import Mathlib.LinearAlgebra.AffineSpace.FiniteDimensional
import Mathlib.Tactic.Abel
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-! The geometric median is unique for data not contained in an affine line. -/
public section
open scoped BigOperators
namespace StatadicFormal.Geometry

variable {H ι : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [Fintype ι]

omit [Fintype ι] in
private theorem all_same_ray_collinear (x : ι → H) {a b : H} (hab : a ≠ b)
    (hr : ∀ i, SameRay ℝ (a - x i) (b - x i)) : Collinear ℝ (Set.range x) := by
  rw [collinear_iff_exists_forall_eq_smul_vadd]
  refine ⟨a, b - a, ?_⟩
  rintro _ ⟨i, rfl⟩
  by_cases h : a - x i = 0
  · exact ⟨0, by simpa using (sub_eq_zero.mp h).symm⟩
  obtain ⟨r, _, he⟩ := (hr i).exists_nonneg_left h
  have hr1 : r - 1 ≠ 0 := by
    intro heq
    have r1 : r = 1 := sub_eq_zero.mp heq
    rw [r1, one_smul] at he
    exact hab (sub_left_inj.mp he)
  have hx : (r - 1) • (a - x i) = b - a := by
    rw [sub_smul, he, one_smul]
    abel
  refine ⟨-((r - 1)⁻¹), ?_⟩
  change x i = -((r - 1)⁻¹) • (b - a) + a
  rw [← hx, smul_smul, neg_mul, inv_mul_cancel₀ hr1, neg_one_smul]
  abel

/-- Strictness of one term at the midpoint rules out two distinct minimizers. -/
@[blueprint "STATADIC-MEDIAN-004"
  (statement := /-- For a nonempty finite unweighted sample in a proper real inner
  product space, if the observations are not all on one affine line, their
  sum of Euclidean distances has a unique minimizer. -/)
  (proof := /-- Existence follows by minimizing on a compact ball. Suppose two
  different points $a,b$ minimize. At their midpoint the triangle inequality
  bounds each distance by its endpoint average. At least one inequality is strict:
  equality for every observation would put $a-x_i$ and $b-x_i$ on the same ray
  for every $i$, hence each $x_i$ on the affine line through $a,b$. Summing gives
  a value below the common minimum, a contradiction. -/)]
theorem geometric_median_unique [ProperSpace H] [Nonempty ι] (x : ι → H)
    (hn : ¬ Collinear ℝ (Set.range x)) :
    ∃! z : H, ∀ y, (∑ i, dist z (x i)) ≤ ∑ i, dist y (x i) := by
  classical
  obtain ⟨a, ha⟩ := geometric_median_exists x
  refine ⟨a, ha, ?_⟩
  intro b hb
  by_contra hba
  have hr : ∃ i, ¬ SameRay ℝ (a - x i) (b - x i) := by
    by_contra! hall
    exact hn (all_same_ray_collinear x (Ne.symm hba) hall)
  obtain ⟨k, hk⟩ := hr
  let m := (2 : ℝ)⁻¹ • (a + b)
  have hm (i : ι) :
      2 * dist m (x i) = ‖(a - x i) + (b - x i)‖ := by
    have heq : (a - x i) + (b - x i) = (2 : ℝ) • (m - x i) := by
      simp only [m, smul_sub, smul_smul, mul_inv_cancel₀ (by norm_num : (2 : ℝ) ≠ 0),
        one_smul, two_smul]
      abel
    rw [heq, norm_smul, Real.norm_of_nonneg (by norm_num : 0 ≤ (2 : ℝ)), dist_eq_norm]
  have hsum : ∑ i, 2 * dist m (x i) < ∑ i, (dist a (x i) + dist b (x i)) := by
    apply Finset.sum_lt_sum
    · intro i _
      rw [hm, dist_eq_norm, dist_eq_norm]
      exact norm_add_le _ _
    · exact ⟨k, Finset.mem_univ _, by rw [hm, dist_eq_norm, dist_eq_norm]; exact norm_add_lt_of_not_sameRay hk⟩
  rw [← Finset.mul_sum, Finset.sum_add_distrib] at hsum
  linarith [ha m, hb m]

/-- For two distinct collinear observations both endpoints and the entire segment minimize. -/
@[blueprint "STATADIC-MEDIAN-005"
  (statement := /-- For the two observations $-1$ and $1$ on the real line,
  the complete set of minimizers of $|t+1|+|t-1|$ is $[-1,1]$. -/)
  (proof := /-- On the interval the two absolute deviations sum to $2$.
  Below $-1$ they sum to $-2t>2$, and above $1$ to $2t>2$.
  A stationarity certificate therefore need not imply uniqueness on collinear data. -/)]
theorem two_point_medians (t : ℝ) :
    (∀ y : ℝ, |t + 1| + |t - 1| ≤ |y + 1| + |y - 1|) ↔ -1 ≤ t ∧ t ≤ 1 := by
  have lb (y : ℝ) : 2 ≤ |y + 1| + |y - 1| := by linarith [le_abs_self (y+1), neg_le_abs (y-1)]
  constructor
  · intro h
    have ht := h 0
    norm_num at ht
    constructor <;> linarith [le_abs_self (t+1), le_abs_self (t-1), neg_le_abs (t+1), neg_le_abs (t-1)]
  · rintro ⟨hlo, hhi⟩ y
    rw [abs_of_nonneg (by linarith : 0 ≤ t + 1), abs_of_nonpos (by linarith : t - 1 ≤ 0)]
    linarith [lb y]

end StatadicFormal.Geometry
