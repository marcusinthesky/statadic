module

public import StatadicFormal.Discrepancy.Transport.DualCertificate
public import Mathlib.Analysis.Convex.Birkhoff
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-! Strong duality for the finite unregularized, equal-mass assignment problem. -/

@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Discrepancy.Transport
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Cost at unit row and column masses. Divide by the number of rows for probabilities. -/
def assignmentCost (C P : Matrix ι ι ℝ) : ℝ := ∑ i, ∑ j, P i j * C i j

private theorem perm_cost (C : Matrix ι ι ℝ) (σ : Equiv.Perm ι) :
    assignmentCost C (σ.permMatrix ℝ) = ∑ i, C i (σ i) := by
  simp [assignmentCost, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply, ite_mul]

/-- Birkhoff's decomposition upgrades an assignment lower bound to every fractional plan. -/
theorem fractional_bound (C P : Matrix ι ι ℝ) (hP : P ∈ doublyStochastic ℝ ι)
    (z : ℝ) (hz : ∀ σ : Equiv.Perm ι, z ≤ ∑ i, C i (σ i)) : z ≤ assignmentCost C P := by
  obtain ⟨w, hw, htot, he⟩ := exists_eq_sum_perm_of_mem_doublyStochastic hP
  have point (i j : ι) : P i j = ∑ σ, w σ * (σ.permMatrix ℝ) i j := by
    rw [← he]
    simp [Matrix.sum_apply]
  have expand : assignmentCost C P = ∑ σ, w σ * assignmentCost C (σ.permMatrix ℝ) := by
    unfold assignmentCost
    simp_rw [point, Finset.sum_mul, mul_assoc]
    simp_rw [Finset.mul_sum]
    simp_rw [Finset.sum_comm (s := (Finset.univ : Finset ι))
      (t := (Finset.univ : Finset (Equiv.Perm ι)))]
  rw [expand, ← one_mul z, ← htot, Finset.sum_mul]
  apply Finset.sum_le_sum
  intro σ _
  rw [perm_cost]
  exact mul_le_mul_of_nonneg_left (hz σ) (hw σ)

private def balance (f : Matrix ι ι ℝ) (k : ι) : ℝ := (∑ a, f a k) - ∑ b, f k b

private def edge (i j : ι) : Matrix ι ι ℝ :=
  fun a b => if a = i then (if b = j then 1 else 0) else 0

private theorem edge_cost (C : Matrix ι ι ℝ) (i j : ι) :
    assignmentCost C (edge i j) = C i j := by
  simp [assignmentCost, edge, ite_mul, Finset.sum_ite_irrel]

private theorem edge_balance (i j k : ι) :
    balance (edge i j) k = (if k = j then 1 else 0) - (if k = i then 1 else 0) := by
  simp [balance, edge, Finset.sum_ite_irrel]

omit [DecidableEq ι] in
private theorem cost_add (C f g : Matrix ι ι ℝ) :
    assignmentCost C (f + g) = assignmentCost C f + assignmentCost C g := by
  simp [assignmentCost, add_mul, Finset.sum_add_distrib]

omit [DecidableEq ι] in
private theorem balance_add (f g : Matrix ι ι ℝ) (k : ι) :
    balance (f + g) k = balance f k + balance g k := by
  simp [balance, Finset.sum_add_distrib]
  ring

/-- When the zero diagonal is an optimal permutation, every nonnegative
circulation has nonnegative cost. Pad the flow on that diagonal and normalize. -/
private theorem circulation_nonneg (C : Matrix ι ι ℝ) (hd : ∀ i, C i i = 0)
    (hmin : ∀ σ : Equiv.Perm ι, 0 ≤ ∑ i, C i (σ i))
    (f : Matrix ι ι ℝ) (hp : ∀ i j, 0 ≤ f i j) (hb : ∀ k, balance f k = 0) :
    0 ≤ assignmentCost C f := by
  let r := fun i => ∑ j, f i j
  let T := 1 + ∑ i, r i
  have hr (i) : 0 ≤ r i := Finset.sum_nonneg fun j _ => hp i j
  have ht : 0 < T := add_pos_of_pos_of_nonneg zero_lt_one (Finset.sum_nonneg fun i _ => hr i)
  have hle (i) : r i ≤ T := by
    have := Finset.single_le_sum (f := r) (fun j _ => hr j) (Finset.mem_univ i)
    dsimp only [T]
    linarith
  let P : Matrix ι ι ℝ := fun i j => (f i j + if i = j then T - r i else 0) / T
  have hP : P ∈ doublyStochastic ℝ ι := by
    apply mem_doublyStochastic_iff_sum.mpr
    refine ⟨fun i j => div_nonneg (add_nonneg (hp i j) (by split <;> simp [hle])) ht.le, ?_, ?_⟩
    · intro i
      simp [P, ← Finset.sum_div, Finset.sum_add_distrib, ← show r i = ∑ j, f i j from rfl, ht.ne']
    · intro j
      have hcol : ∑ i, f i j = r j := sub_eq_zero.mp (hb j)
      simp [P, ← Finset.sum_div, Finset.sum_add_distrib, hcol, ht.ne']
  have hcost : assignmentCost C P = assignmentCost C f / T := by
    have hh (i j) : P i j * C i j = (f i j * C i j) / T := by
      by_cases h : i = j
      · subst j
        simp [hd]
      · simp [P, h, div_mul_eq_mul_div]
    simp_rw [assignmentCost, hh, ← Finset.sum_div]
  have hn := fractional_bound C P hP 0 hmin
  rw [hcost] at hn
  simpa [ht.ne'] using mul_nonneg hn ht.le

private def flowCosts (C : Matrix ι ι ℝ) (s t : ι) : Set ℝ :=
  {z | ∃ f : Matrix ι ι ℝ, (∀ i j, 0 ≤ f i j) ∧
    (∀ k, balance f k = (if k = t then 1 else 0) - (if k = s then 1 else 0)) ∧
    assignmentCost C f = z}

private theorem extend_flow (C : Matrix ι ι ℝ) (s t u : ι) {z : ℝ} (hz : z ∈ flowCosts C s t) :
    z + C t u ∈ flowCosts C s u := by
  obtain ⟨f,hf,hb,rfl⟩ := hz
  refine ⟨f + edge t u, fun i j => add_nonneg (hf i j) (by simp [edge]; positivity), ?_, ?_⟩
  · intro k
    rw [balance_add, hb, edge_balance]
    ring
  · rw [cost_add, edge_cost]

theorem potential_exists [Nonempty ι] (C : Matrix ι ι ℝ) (hd : ∀ i, C i i = 0)
    (hmin : ∀ σ : Equiv.Perm ι, 0 ≤ ∑ i, C i (σ i)) :
    ∃ v : ι → ℝ, ∀ i j, v j - v i ≤ C i j := by
  let s := Classical.arbitrary ι
  have ne (i : ι) : (flowCosts C s i).Nonempty :=
    ⟨C s i, edge s i, fun a b => by simp [edge]; positivity, edge_balance s i, edge_cost C s i⟩
  have lower (i : ι) : BddBelow (flowCosts C s i) := by
    refine ⟨-C i s, ?_⟩
    intro z hz
    obtain ⟨f,hf,hb,he⟩ := extend_flow C s i s hz
    have hh := circulation_nonneg C hd hmin f hf (fun k => by simpa using hb k)
    linarith
  refine ⟨fun i => sInf (flowCosts C s i), fun i j => ?_⟩
  have h : sInf (flowCosts C s j) - C i j ≤ sInf (flowCosts C s i) := by
    apply le_csInf (ne i)
    intro z hz
    exact sub_le_iff_le_add.mpr (csInf_le (lower j) (extend_flow C s i j hz))
  linarith

/-- Both sides attain the same optimum for finite square uniform assignment costs. -/
@[blueprint "STATADIC-TRANSPORT-002"
  (statement := /-- For a finite nonempty square cost matrix $C$ there exist a permutation
  $\sigma$ and potentials $u,v$ with $u_i+v_j\le C_{ij}$. The assignment minimizes cost
  among all doubly stochastic couplings and attains $\sum_i u_i+\sum_j v_j$.
  Dividing by the common number of rows gives the uniform probability-mass objective. -/)
  (proof := /-- Minimize over the finite permutations and relabel the columns by an
  optimum. Subtract the diagonal cost from each row to form residual costs $D$.
  Every nonnegative circulation has nonnegative residual cost: pad its diagonal to
  equal row and column totals, normalize, and decompose the resulting doubly stochastic
  matrix into permutation matrices. From a fixed origin let $v_j$ be the infimum
  cost of sending one unit to $j$. A direct edge makes this set nonempty; adding
  a return edge bounds it below. Extending a flow by one edge gives
  $v_j-v_i\le D_{ij}$. Undo the relabeling and row shift; the matched edges are tight.
  The resulting feasible dual lower bound is the assignment value. -/)]
theorem assignment_strong_duality [Nonempty ι] (C : Matrix ι ι ℝ) :
    ∃ (σ : Equiv.Perm ι) (u v : ι → ℝ),
      (∀ i j, u i + v j ≤ C i j) ∧
      (∑ i, C i (σ i)) = (∑ i, u i) + ∑ j, v j ∧
      ∀ P : Matrix ι ι ℝ, P ∈ doublyStochastic ℝ ι → (∑ i, C i (σ i)) ≤ assignmentCost C P := by
  classical
  obtain ⟨σ, _, hσ⟩ := Finset.exists_min_image Finset.univ
    (fun e : Equiv.Perm ι => ∑ i, C i (e i)) Finset.univ_nonempty
  have hm (e : Equiv.Perm ι) := hσ e (Finset.mem_univ e)
  let D : Matrix ι ι ℝ := fun i j => C i (σ j) - C i (σ i)
  have hd (i : ι) : D i i = 0 := sub_self _
  have hD (e : Equiv.Perm ι) : 0 ≤ ∑ i, D i (e i) := by
    simpa [D, Finset.sum_sub_distrib] using sub_nonneg.mpr (hm (e.trans σ))
  obtain ⟨v, hv⟩ := potential_exists D hd hD
  refine ⟨σ, fun i => C i (σ i) - v i, fun j => v (σ.symm j), ?_, ?_, ?_⟩
  · intro i j
    have h := hv i (σ.symm j)
    simp only [D, Equiv.apply_symm_apply] at h
    linarith
  · rw [Finset.sum_sub_distrib, Equiv.sum_comp σ.symm v, sub_add_cancel]
  · exact fun P hP => fractional_bound C P hP _ hm

end StatadicFormal.Discrepancy.Transport
