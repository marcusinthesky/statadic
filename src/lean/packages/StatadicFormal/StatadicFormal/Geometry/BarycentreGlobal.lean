module

public import StatadicFormal.Geometry.BarycentreUpdate
public import StatadicFormal.Optimization.SimplexCertificate
public import Mathlib.Analysis.Convex.SpecificFunctions.Basic
public import Mathlib.Analysis.Normed.Affine.AddTorsor
public import Mathlib.Topology.Order.Compact
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-! Global existence and objective certificates for STATADIC's finite barycentre targets. -/
@[expose] public section
noncomputable section
open Matrix StatadicFormal.Optimization
namespace StatadicFormal.Geometry

variable {K I H : Type*} [Fintype K] [Fintype I] [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The global free-support W2 minimum is attained among the finitely many matched centroids. -/
@[blueprint "STATADIC-BARYCENTRE-002"
  (statement := /-- For equal-cardinality uniform finite clouds in Euclidean space
  with fixed nonnegative source weights summing to one, the equal-cardinality
  free-support $W_2^2$ barycentre problem has a global minimizer. One minimizer
  is the rowwise weighted centroid of a choice of source permutations: its joint
  squared cost is no larger than the cost of any support and any matching pattern. -/)
  (proof := /-- For each tuple of permutations, compute the matched row centroids.
  Their conditional cost is minimal by the exact squared-distance excess formula.
  The set of tuples of permutations is finite and nonempty. Minimize the conditional
  minima over that set; the selected centroids then beat every competing support
  and permutation tuple. -/)]
theorem finite_wasserstein_barycentre_exists [DecidableEq I] (w : K → ℝ)
    (hs : ∑ k, w k = 1) (x : K → I → H) :
    let center := fun (σ : K → Equiv.Perm I) i => ∑ k, w k • x k (σ k i)
    let cost := fun (z : I → H) (σ : K → Equiv.Perm I) =>
      ∑ i, ∑ k, w k * ‖x k (σ k i) - z i‖ ^ 2
    ∃ σ, ∀ z τ, cost (center σ) σ ≤ cost z τ := by
  classical
  let center := fun (σ : K → Equiv.Perm I) i => ∑ k, w k • x k (σ k i)
  let cost := fun (z : I → H) (σ : K → Equiv.Perm I) =>
    ∑ i, ∑ k, w k * ‖x k (σ k i) - z i‖ ^ 2
  obtain ⟨σ, _, hσ⟩ := Finset.exists_min_image Finset.univ (fun σ => cost (center σ) σ)
    Finset.univ_nonempty
  refine ⟨σ, fun z τ => (hσ τ (Finset.mem_univ τ)).trans ?_⟩
  apply Finset.sum_le_sum
  intro i _
  have h := (barycentre_row_gap w hs (fun k => x k (τ k i)) (z i)).1
  change (∑ k, w k * ‖x k (τ k i) - center τ i‖ ^ 2) ≤ _
  dsimp only at h
  linarith [sq_nonneg ‖z i - center τ i‖]

/-- Existence and a posteriori objective error for target-directed MMD mixture fitting. -/
@[blueprint "STATADIC-BARYCENTRE-003"
  (statement := /-- On the candidate probability simplex, let
  $F(s)=t+s^TQs/2+c^Ts$, where $Q$ is symmetric positive semidefinite on
  zero-sum directions. The empirical squared-MMD mixture objective has this
  form with $Q=2G$. A minimizing mixture exists. At any feasible mixture $s$
  with gradient $g=Qs+c$, $0\le F(s)-\min F\le g^Ts-\min_i g_i$.
  A zero Frank--Wolfe gap therefore certifies a global optimum. -/)
  (proof := /-- The simplex is a nonempty closed subset of a compact coordinate
  cube and the quadratic is continuous. Every feasible mixture averages its
  gradient coordinates and so has linearized cost at least their minimum.
  Apply the tangent-space quadratic gap bound at any attained minimizer. -/)]
theorem target_mixture_gap [Nonempty K] (Q : Matrix K K ℝ) (hQ : Qᵀ = Q)
    (hp : ∀ h, (∑ i, h i) = 0 → 0 ≤ h ⬝ᵥ (Q *ᵥ h)) (c : K → ℝ) (t : ℝ) :
    let F := fun s => t + quadraticObjective Q c s
    let S := {s : K → ℝ | (∀ i, 0 ≤ s i) ∧ (∑ i, s i) = 1}
    ∃ u ∈ S, ∀ s ∈ S,
      0 ≤ F s - F u ∧ F s - F u ≤
        (Q *ᵥ s + c) ⬝ᵥ s - Finset.univ.inf' Finset.univ_nonempty (Q *ᵥ s + c) := by
  classical
  let F := fun s => t + quadraticObjective Q c s
  let S := {s : K → ℝ | (∀ i, 0 ≤ s i) ∧ (∑ i, s i) = 1}
  have closed : IsClosed S := (isClosed_Ici (a := (0 : K → ℝ))).inter
    (isClosed_eq (continuous_finsetSum _ fun i _ => continuous_apply i) continuous_const)
  have compact : IsCompact S := isCompact_Icc.of_isClosed_subset closed
    (show S ⊆ Set.Icc 0 (1 : K → ℝ) from fun s hs =>
      ⟨hs.1, fun i => (Finset.single_le_sum (fun j _ => hs.1 j) (Finset.mem_univ i)).trans_eq hs.2⟩)
  let k := Classical.arbitrary K
  have ne : S.Nonempty := ⟨fun i => if i = k then 1 else 0, by simp [S]; intro i; positivity⟩
  have cont : Continuous F := by
    simp only [F, quadraticObjective, Matrix.mulVec, dotProduct]
    fun_prop
  obtain ⟨u, hu, hmin⟩ := compact.exists_isMinOn ne cont.continuousOn
  refine ⟨u, hu, fun s hs => ⟨sub_nonneg.mpr (hmin hs), ?_⟩⟩
  let g := Q *ᵥ s + c
  let L := Finset.univ.inf' Finset.univ_nonempty g
  have hl : L ≤ g ⬝ᵥ u := by
    calc L = ∑ i, L * u i := by rw [← Finset.mul_sum, hu.2, mul_one]
         _ ≤ _ := Finset.sum_le_sum fun i _ =>
           mul_le_mul_of_nonneg_right (Finset.inf'_le _ (Finset.mem_univ i)) (hu.1 i)
  have hgap := quadratic_gap_bound Q hQ hp c s u L (hu.2.trans hs.2.symm) hl
  simpa [F, L, g] using hgap

/-- A midpoint on a constant-speed geodesic minimizes the two-source equally weighted
p-power objective at every p ≥ 1. This also describes the p = 1 nonunique case. -/
@[blueprint "STATADIC-BARYCENTRE-004"
  (statement := /-- In any metric space, if $m$ is at equal half-distance from
  endpoints $a,b$, it globally minimizes the equal-weight sum of $p$-powered
  distances for every real $p\ge1$. -/)
  (proof := /-- For any candidate $z$, the average of its two endpoint distances
  bounds $d(a,b)/2$ by the triangle inequality. Apply the increasing $p$th power
  and then convexity to bound this by the average of the two powers.
  The midpoint attains both inequalities. -/)]
theorem midpoint_power_min {E : Type*} [PseudoMetricSpace E] (a b m z : E) (p : ℝ) (hp : 1 ≤ p)
    (ha : dist a m = dist a b / 2) (hb : dist m b = dist a b / 2) :
    (dist a m ^ p + dist m b ^ p) / 2 ≤ (dist a z ^ p + dist z b ^ p) / 2 := by
  have htri : dist a b / 2 ≤ (1/2 : ℝ) * dist a z + (1/2 : ℝ) * dist z b := by
    linarith [dist_triangle a z b]
  have hconv := (convexOn_rpow hp).2 (show dist a z ∈ Set.Ici (0 : ℝ) from dist_nonneg)
    (show dist z b ∈ Set.Ici (0 : ℝ) from dist_nonneg)
    (show 0 ≤ (1/2 : ℝ) by norm_num) (show 0 ≤ (1/2 : ℝ) by norm_num)
    (show (1/2 : ℝ) + 1/2 = 1 by norm_num)
  simp only [smul_eq_mul] at hconv
  rw [ha, hb]
  linarith [Real.rpow_le_rpow (by positivity) htri (by linarith : 0 ≤ p)]

private theorem two_cloud_gap (a b z : H) (t : ℝ) :
    (1 - t) * ‖a - z‖ ^ 2 + t * ‖b - z‖ ^ 2 =
      t * (1 - t) * ‖a - b‖ ^ 2 + ‖z - ((1 - t) • a + t • b)‖ ^ 2 := by
  simp only [norm_sub_sq_real, norm_add_sq_real, inner_add_right, real_inner_smul_left,
    real_inner_smul_right, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs,
    real_inner_comm z a, real_inner_comm z b]
  ring

/-- The Euclidean W2 displacement interpolation actually selected by an optimal
two-source assignment is globally optimal among all equal-size competing supports. -/
@[blueprint "STATADIC-BARYCENTRE-005"
  (statement := /-- Suppose $\pi$ minimizes squared assignment cost between two equal-size
  uniform Euclidean clouds $a,b$. With source weights $1-t,t$ in $[0,1]$, the support
  $z_i=(1-t)a_i+t b_{\pi(i)}$ achieves the global minimum: the weighted squared cost
  at every competing support and pair of matchings is at least
  $t(1-t)\sum_i\|a_i-b_{\pi(i)}\|^2$, attained at the stated support. -/)
  (proof := /-- Complete the square for each pair of matched atoms. The remainder
  is a nonnegative squared deviation from the paired centroid. Relabel the
  competitor's two matchings to obtain a single permutation between the sources;
  its source-to-source cost is no smaller than the optimal one. The interpolated
  support makes all squared deviations vanish. -/)]
theorem two_source_w2_interpolation (a b : I → H) (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1)
    (π : Equiv.Perm I) (hπ : ∀ ρ : Equiv.Perm I,
      (∑ i, ‖a i - b (π i)‖ ^ 2) ≤ ∑ i, ‖a i - b (ρ i)‖ ^ 2) :
    let F := fun (z : I → H) (σ τ : Equiv.Perm I) =>
      ∑ i, ((1 - t) * ‖a (σ i) - z i‖ ^ 2 + t * ‖b (τ i) - z i‖ ^ 2)
    let z := fun i => (1 - t) • a i + t • b (π i)
    F z (Equiv.refl I) π = t * (1-t) * (∑ i, ‖a i - b (π i)‖ ^ 2) ∧
      ∀ y σ τ, F z (Equiv.refl I) π ≤ F y σ τ := by
  dsimp only
  have exactCost : (∑ i, ((1 - t) * ‖a i - ((1-t) • a i + t • b (π i))‖ ^ 2 +
      t * ‖b (π i) - ((1-t) • a i + t • b (π i))‖ ^ 2)) =
      t * (1-t) * (∑ i, ‖a i - b (π i)‖ ^ 2) := by
    simp only [two_cloud_gap, sub_self, norm_zero, zero_pow, ne_eq, OfNat.ofNat_ne_zero,
      not_false_eq_true, add_zero, Finset.mul_sum]
  refine ⟨exactCost, fun y σ τ => ?_⟩
  simp only [Equiv.refl_apply]
  rw [exactCost]
  have relabel : (∑ i, ‖a (σ i) - b (τ i)‖ ^ 2) =
      ∑ i, ‖a i - b ((σ.symm.trans τ) i)‖ ^ 2 := by
    simpa using Equiv.sum_comp σ (fun j => ‖a j - b (τ (σ.symm j))‖ ^ 2)
  calc
    _ ≤ t * (1-t) * (∑ i, ‖a (σ i) - b (τ i)‖ ^ 2) := by
      rw [relabel]
      exact mul_le_mul_of_nonneg_left (hπ _) (mul_nonneg ht.1 (sub_nonneg.mpr ht.2))
    _ ≤ _ := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun i _ => by
        rw [two_cloud_gap]
        exact le_add_of_nonneg_right (sq_nonneg _)

/-- For every supported order, matching on the powered cost and taking atomwise
midpoints gives a global two-source barycentre when the weights are equal. -/
@[blueprint "STATADIC-BARYCENTRE-006"
  (statement := /-- For two uniform Euclidean clouds with equal counts and exactly equal
  source weights, match atoms with a minimum $p$-powered assignment, $p\ge1$. The midpoints
  of those pairs globally minimize the equal-weight two-source powered objective over
  all competing equal-count supports and their assignments. -/)
  (proof := /-- The average powered distance to a pair is minimized at its Euclidean
  midpoint, at value $2^{-p}d(a,b)^p$. Sum that inequality over competing pairs.
  Their paired endpoint cost is at least the optimum after relabeling one assignment.
  The proposed midpoint support attains the resulting lower bound. -/)]
theorem two_source_midpoint_interpolation (a b : I → H) (p : ℝ) (hp : 1 ≤ p)
    (π : Equiv.Perm I) (hπ : ∀ ρ : Equiv.Perm I,
      (∑ i, dist (a i) (b (π i)) ^ p) ≤ ∑ i, dist (a i) (b (ρ i)) ^ p) :
    let F := fun (z : I → H) (σ τ : Equiv.Perm I) =>
      ∑ i, (dist (a (σ i)) (z i) ^ p + dist (z i) (b (τ i)) ^ p) / 2
    let z := fun i => midpoint ℝ (a i) (b (π i))
    F z (Equiv.refl I) π = (∑ i, dist (a i) (b (π i)) ^ p) / 2 ^ p ∧
      ∀ y σ τ, F z (Equiv.refl I) π ≤ F y σ τ := by
  dsimp only
  have left (x y : H) : dist x (midpoint ℝ x y) = dist x y / 2 := by
    simp only [dist_left_midpoint, Real.norm_of_nonneg zero_le_two, div_eq_inv_mul]
  have right (x y : H) : dist (midpoint ℝ x y) y = dist x y / 2 := by
    simp only [dist_midpoint_right, Real.norm_of_nonneg zero_le_two, div_eq_inv_mul]
  have half (x y : H) : ((dist x y / 2) ^ p + (dist x y / 2) ^ p) / 2 =
      dist x y ^ p / 2 ^ p := by rw [Real.div_rpow dist_nonneg zero_le_two]; ring
  simp only [Equiv.refl_apply, left, right, half, ← Finset.sum_div]
  refine ⟨True.intro, fun y σ τ => ?_⟩
  have relabel : (∑ i, dist (a (σ i)) (b (τ i)) ^ p) =
      ∑ i, dist (a i) (b ((σ.symm.trans τ) i)) ^ p := by
    simpa using Equiv.sum_comp σ (fun j => dist (a j) (b (τ (σ.symm j))) ^ p)
  calc
    _ ≤ (∑ i, dist (a (σ i)) (b (τ i)) ^ p) / 2 ^ p := by
      rw [relabel]
      exact div_le_div_of_nonneg_right (hπ _) (by positivity)
    _ ≤ _ := by
      rw [Finset.sum_div, Finset.sum_div]
      refine Finset.sum_le_sum fun i _ => ?_
      have h := midpoint_power_min _ _ _ (y i) p hp (left (a (σ i)) (b (τ i))) (right _ _)
      simpa only [left, right, half] using h

end StatadicFormal.Geometry
