module

public import Architect
public import Mathlib.Topology.MetricSpace.ProperSpace
public import Mathlib.Topology.Order.Compact
public import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! A posteriori optimality bound, including ties at a sample point. -/

public section
open scoped BigOperators InnerProductSpace
namespace StatadicFormal.Geometry
variable {H ι : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [Fintype ι]

/-- The weighted noncoincident directions and the coincident mass certify a spatial median. -/
@[blueprint "STATADIC-MEDIAN-001"
  (statement := /-- Translate the proposed median to zero. Let the noncoincident residuals
  be $r_i\ne0$ with weights $w_i\ge0$, and let $M\ge0$ be the mass at zero.
  Write $R=\sum_i w_i r_i/\|r_i\|$ and $F(h)=M\|h\|+\sum_i w_i\|r_i-h\|$.
  Then $F(h)-F(0)\ge (M-\|R\|)\|h\|$ for every $h$.
  In particular $\|R\|\le M$ certifies global optimality. -/)
  (proof := /-- Each unit vector $u_i=r_i/\|r_i\|$ supports its norm:
  $\|r_i-h\|\ge\langle u_i,r_i-h\rangle=\|r_i\|-\langle u_i,h\rangle$.
  Sum with nonnegative weights, add the coincident contribution $M\|h\|$,
  and use $\langle R,h\rangle\le\|R\|\|h\|$. -/)]
theorem median_force_bound (r : ι → H) (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i)
    (hr : ∀ i, r i ≠ 0) (M : ℝ) (h : H) :
    let R := ∑ i, w i • (‖r i‖⁻¹ • r i)
    let gap := M * ‖h‖ + ∑ i, w i * ‖r i - h‖ - ∑ i, w i * ‖r i‖
    (M - ‖R‖) * ‖h‖ ≤ gap ∧ (‖R‖ ≤ M → 0 ≤ gap) := by
  let u := fun i => ‖r i‖⁻¹ • r i
  let R := ∑ i, w i • u i
  have hunit (i) : ‖u i‖ = 1 := by simp [u, norm_smul, norm_ne_zero_iff.mpr (hr i)]
  have ht (i) : ⟪u i, r i⟫_ℝ = ‖r i‖ := by
    simp [u, real_inner_smul_left, pow_two,
      norm_ne_zero_iff.mpr (hr i)]
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
    mul_le_mul_of_nonneg_left (real_inner_le_norm (u i) (r i - h)) (hw i))
  simp only [inner_sub_right, ht, hunit, one_mul, mul_sub, Finset.sum_sub_distrib] at hs
  have hinner := real_inner_le_norm R h
  dsimp only [R] at hinner
  rw [sum_inner] at hinner
  simp only [real_inner_smul_left] at hinner
  have bound : (M - ‖R‖) * ‖h‖ ≤
      M * ‖h‖ + ∑ i, w i * ‖r i - h‖ - ∑ i, w i * ‖r i‖ := by
    rw [sub_mul]
    linarith
  exact ⟨bound, fun hM => (mul_nonneg (sub_nonneg.mpr hM) (norm_nonneg h)).trans bound⟩

/-- Every nonempty finite group of observations has a geometric median in a proper metric space. -/
@[blueprint "STATADIC-MEDIAN-002"
  (statement := /-- A nonempty finite sample in a proper metric space, including every
  finite-dimensional Euclidean group, has a point minimizing its sum of distances. -/)
  (proof := /-- Choose an observed point $x_0$ and let $R$ be its objective value.
  Outside the closed ball about $x_0$ of radius $R$, the term $d(y,x_0)$ alone exceeds
  $R$. Inside, the continuous objective attains its minimum on the compact ball,
  which contains $x_0$. -/)]
theorem geometric_median_exists {E : Type*} [MetricSpace E] [ProperSpace E] [Nonempty ι]
    (x : ι → E) : ∃ z : E, ∀ y, (∑ i, dist z (x i)) ≤ ∑ i, dist y (x i) := by
  classical
  let i := Classical.arbitrary ι
  let F := fun y => ∑ j, dist y (x j)
  have hR : 0 ≤ F (x i) := Finset.sum_nonneg fun _ _ => dist_nonneg
  have hcenter := Metric.mem_closedBall_self hR (x := x i)
  have hcont : Continuous F := continuous_finsetSum _ (fun j _ => continuous_id.dist continuous_const)
  obtain ⟨z, _, hz⟩ := (isCompact_closedBall (x i) (F (x i))).exists_isMinOn
    ⟨x i, hcenter⟩ hcont.continuousOn
  refine ⟨z, fun y => ?_⟩
  by_cases hy : y ∈ Metric.closedBall (x i) (F (x i))
  · exact hz hy
  · have hfar := lt_of_not_ge (mt Metric.mem_closedBall.mpr hy)
    exact (hz hcenter).trans (hfar.le.trans (Finset.single_le_sum
      (f := fun j => dist y (x j)) (fun _ _ => dist_nonneg) (Finset.mem_univ i)))

private theorem norm_majorant (r h : H) (hr : r ≠ 0) :
    ‖r - h‖ - ‖r‖ ≤ -⟪‖r‖⁻¹ • r, h⟫_ℝ + ‖h‖ ^ 2 / (2 * ‖r‖) := by
  have hn : 0 < ‖r‖ := norm_pos_iff.mpr hr
  have hs := norm_sub_sq_real r h
  have hsq := sq_nonneg (‖r - h‖ - ‖r‖)
  rw [real_inner_smul_left]
  field_simp
  nlinarith

/-- The coincident-mass condition is also necessary for the unconstrained objective. -/
@[blueprint "STATADIC-MEDIAN-003"
  (statement := /-- For residuals $r_i\ne0$, weights $w_i\ge0$, and coincident mass
  $M\ge0$, zero globally minimizes $F(h)=M\|h\|+\sum_i w_i\|r_i-h\|$
  exactly when $\|R\|\le M$, where $R=\sum_i w_i r_i/\|r_i\|$. -/)
  (proof := /-- The supporting-plane bound proves sufficiency. For necessity,
  $(\|r-h\|-\|r\|)^2\ge0$ yields the upper bound
  $\|r-h\|-\|r\|\le-\langle r/\|r\|,h\rangle+\|h\|^2/(2\|r\|)$.
  Set $h=tR$. If $\|R\|>M$, the linear term in the total objective change
  is strictly negative, while the remainder is quadratic in $t$, giving descent. -/)]
theorem median_force_iff (r : ι → H) (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i)
    (hr : ∀ i, r i ≠ 0) (M : ℝ) (hM : 0 ≤ M) :
    let R := ∑ i, w i • (‖r i‖⁻¹ • r i)
    (∀ h : H, (∑ i, w i * ‖r i‖) ≤ M * ‖h‖ + ∑ i, w i * ‖r i - h‖) ↔ ‖R‖ ≤ M := by
  let u := fun i => ‖r i‖⁻¹ • r i
  let R := ∑ i, w i • u i
  let L := ∑ i, w i / (2 * ‖r i‖)
  constructor
  · intro hmin
    by_contra! hbad
    change M < ‖R‖ at hbad
    have hn : 0 < ‖R‖ := hM.trans_lt hbad
    have hL : 0 ≤ L := Finset.sum_nonneg fun i _ => div_nonneg (hw i) (by positivity)
    let t := (‖R‖ - M) / ((L + 1) * ‖R‖)
    have ht : 0 < t := div_pos (sub_pos.mpr hbad) (by positivity)
    have hteq : t * ((L + 1) * ‖R‖) = ‖R‖ - M :=
      div_mul_cancel₀ _ (by positivity)
    let h := t • R
    have ub (i) : w i * ‖r i - h‖ - w i * ‖r i‖ ≤
        -(w i * ⟪u i, h⟫_ℝ) + (w i / (2 * ‖r i‖)) * ‖h‖ ^ 2 := by
      have hh := mul_le_mul_of_nonneg_left (norm_majorant (r i) h (hr i)) (hw i)
      dsimp only [u]
      convert hh using 1 <;> ring
    have hb := Finset.sum_le_sum (s := Finset.univ) (fun i _ => ub i)
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_neg_distrib,
      ← Finset.sum_mul] at hb
    have hi : (∑ i, w i * ⟪u i, h⟫_ℝ) = t * ‖R‖ ^ 2 := by
      calc _ = ⟪R, h⟫_ℝ := by simp [R, sum_inner, real_inner_smul_left]
           _ = _ := by rw [show h = t • R from rfl, real_inner_smul_right, real_inner_self_eq_norm_sq]
    have hnorm : ‖h‖ = t * ‖R‖ := by simp [h, norm_smul, Real.norm_of_nonneg ht.le]
    change _ ≤ -_ + L * ‖h‖ ^ 2 at hb
    rw [hi, hnorm] at hb
    have hm := hmin h
    rw [hnorm] at hm
    have hstrict : L * (t * ‖R‖) < ‖R‖ - M := by nlinarith [mul_pos ht hn]
    nlinarith [mul_lt_mul_of_pos_right hstrict (mul_pos ht hn)]
  · intro hm h
    exact sub_nonneg.mp ((median_force_bound r w hw hr M h).2 hm)

end StatadicFormal.Geometry
