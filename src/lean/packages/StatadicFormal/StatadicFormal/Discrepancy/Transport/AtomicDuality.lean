module

public import StatadicFormal.Discrepancy.Transport.AssignmentDuality
public import Mathlib.Topology.Order.Compact
public import Mathlib.Topology.Instances.Real.Lemmas
import Mathlib.Tactic.Linarith

/-! Primal and dual attainment for arbitrary finite atomic probability marginals. -/
@[expose] public section
noncomputable section
namespace StatadicFormal.Discrepancy.Transport
variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Admissibility in the finite Kantorovich problem. -/
def IsCoupling (a : ι → ℝ) (b : κ → ℝ) (P : ι → κ → ℝ) : Prop :=
  (∀ i j, 0 ≤ P i j) ∧ (∀ i, ∑ j, P i j = a i) ∧ ∀ j, ∑ i, P i j = b j

private theorem attained (a : ι → ℝ) (b : κ → ℝ) (ha : ∀ i, 0 ≤ a i) (hb : ∀ j, 0 ≤ b j)
    (hA : ∑ i, a i = 1) (hB : ∑ j, b j = 1) (C : ι → κ → ℝ) :
    ∃ P, IsCoupling a b P ∧ ∀ Q, IsCoupling a b Q →
      (∑ i, ∑ j, P i j * C i j) ≤ ∑ i, ∑ j, Q i j * C i j := by
  let B := Set.Icc (0 : ι → κ → ℝ) 1
  let S := {P : ι → κ → ℝ | (∀ i, ∑ j, P i j = a i) ∧ ∀ j, ∑ i, P i j = b j}
  have admissible (P : ι → κ → ℝ) : P ∈ B ∩ S ↔ IsCoupling a b P := by
    constructor
    · rintro ⟨h, hrow, hcol⟩
      exact ⟨h.1, hrow, hcol⟩
    · rintro ⟨hn, hrow, hcol⟩
      refine ⟨⟨hn, ?_⟩, hrow, hcol⟩
      intro i j
      calc P i j ≤ ∑ k, P i k := Finset.single_le_sum (fun k _ => hn i k) (Finset.mem_univ j)
           _ = a i := hrow i
           _ ≤ 1 := hA ▸ Finset.single_le_sum (fun k _ => ha k) (Finset.mem_univ i)
  have ev (i : ι) (j : κ) : Continuous (fun P : ι → κ → ℝ => P i j) :=
    (continuous_apply j).comp (continuous_apply i)
  have closed : IsClosed S := by
    apply IsClosed.inter
    · change IsClosed {P : ι → κ → ℝ | ∀ i, ∑ j, P i j = a i}
      rw [Set.ofPred_forall]
      apply isClosed_iInter
      intro i
      exact isClosed_eq (continuous_finsetSum _ (fun j _ => ev i j)) continuous_const
    · change IsClosed {P : ι → κ → ℝ | ∀ j, ∑ i, P i j = b j}
      rw [Set.ofPred_forall]
      apply isClosed_iInter
      intro j
      exact isClosed_eq (continuous_finsetSum _ (fun i _ => ev i j)) continuous_const
  have witness : (B ∩ S).Nonempty := by
    refine ⟨fun i j => a i * b j, (admissible _).mpr ⟨fun i j => mul_nonneg (ha i) (hb j), ?_, ?_⟩⟩
    · simp [← Finset.mul_sum, hB]
    · simp [← Finset.sum_mul, hA]
  let cost := fun P : ι → κ → ℝ => ∑ i, ∑ j, P i j * C i j
  have cont : Continuous cost := by
    apply continuous_finsetSum; intro i _
    exact continuous_finsetSum _ (fun j _ => (ev i j).mul continuous_const)
  obtain ⟨P, hp, hmin⟩ := (isCompact_Icc.inter_right closed).exists_isMinOn witness cont.continuousOn
  exact ⟨P, (admissible _).mp hp, fun Q hQ => hmin ((admissible _).mpr hQ)⟩

/-- Dual attainment at an optimal plan. A row can send a small amount of mass
from an occupied cell to any other row of the same column. -/
private theorem dual_at_plan [DecidableEq ι] [Nonempty ι] [DecidableEq κ]
    (a : ι → ℝ) (b : κ → ℝ) (ha : ∀ i, 0 < a i) (C P : ι → κ → ℝ) (hP : IsCoupling a b P)
    (hm : ∀ Q, IsCoupling a b Q → (∑ i, ∑ j, P i j * C i j) ≤ ∑ i, ∑ j, Q i j * C i j) :
    ∃ (u : ι → ℝ) (v : κ → ℝ), (∀ i j, u i + v j ≤ C i j) ∧
      ∀ i j, P i j ≠ 0 → C i j = u i + v j := by
  classical
  let S := fun i => Finset.univ.filter (fun j => 0 < P i j)
  have hne (i) : (S i).Nonempty := by
    by_contra! hn
    have zero (j : κ) : P i j = 0 := by
      have hj : j ∉ S i := by simp [hn]
      exact le_antisymm (by simpa [S] using hj) (hP.1 i j)
    have h := hP.2.1 i
    simp [zero] at h
    linarith [ha i]
  choose J hJ best using fun i k => Finset.exists_min_image (S i)
    (fun j => C k j - C i j) (hne i)
  let D := fun i k => C k (J i k) - C i (J i k)
  have cycles (σ : Equiv.Perm ι) : 0 ≤ ∑ i, D i (σ i) := by
    let j := fun i => J i (σ i)
    obtain ⟨i₀, _, small⟩ := Finset.exists_min_image Finset.univ
      (fun i => P i (j i)) Finset.univ_nonempty
    let ε := P i₀ (j i₀)
    have he : 0 < ε := (Finset.mem_filter.mp (hJ i₀ (σ i₀))).2
    let T := fun i k => if k = j i then ε else 0
    let Q := fun i k => P i k - T i k + T (σ.symm i) k
    have hrow (i) : ∑ k, T i k = ε := by simp [T]
    have hQ : IsCoupling a b Q := by
      refine ⟨?_, ?_, ?_⟩
      · intro i k
        have hle : T i k ≤ P i k := by
          by_cases hk : k = j i
          · simpa [T, hk, ε] using small i (Finset.mem_univ i)
          · simpa [T, hk] using hP.1 i k
        exact add_nonneg (sub_nonneg.mpr hle) (by simp [T]; positivity)
      · intro i
        simp [Q, Finset.sum_add_distrib, Finset.sum_sub_distrib, hrow, hP.2.1]
      · intro k
        simp only [Q, Finset.sum_add_distrib, Finset.sum_sub_distrib, hP.2.2]
        rw [Equiv.sum_comp σ.symm (fun i => T i k), sub_add_cancel]
    have h := hm Q hQ
    have exchange :
        (∑ i, ∑ k, T (σ.symm i) k * C i k) = ∑ i, ε * C (σ i) (j i) := by
      rw [← Equiv.sum_comp σ (fun i => ∑ k, T (σ.symm i) k * C i k)]
      simp [T, ite_mul]
    have removed (i) : ∑ k, T i k * C i k = ε * C i (j i) := by simp [T, ite_mul]
    simp only [Q, add_mul, sub_mul, Finset.sum_add_distrib, Finset.sum_sub_distrib] at h
    rw [exchange] at h
    simp_rw [removed] at h
    have hex : 0 ≤ ε * (∑ i, D i (σ i)) := by
      simp only [D, ← show ∀ i, j i = J i (σ i) from fun _ => rfl, Finset.mul_sum,
        mul_sub, Finset.sum_sub_distrib]
      linarith
    exact nonneg_of_mul_nonneg_right hex he
  obtain ⟨u, hu⟩ := potential_exists D (fun i => sub_self _) cycles
  choose I _ hI using fun j => Finset.exists_min_image Finset.univ (fun i => C i j - u i)
    Finset.univ_nonempty
  let v := fun j => C (I j) j - u (I j)
  refine ⟨u, v, fun i j => by dsimp [v]; linarith [hI j i (Finset.mem_univ i)], ?_⟩
  intro i j hij
  have hj : j ∈ S i := Finset.mem_filter.mpr ⟨Finset.mem_univ _, lt_of_le_of_ne (hP.1 _ _) hij.symm⟩
  have h := (hu i (I j)).trans (best i (I j) j hj)
  dsimp [v]
  linarith [hI j i (Finset.mem_univ i)]

/-- Both finite weighted optima are attained, including for a rectangular cost array. -/
@[blueprint "STATADIC-TRANSPORT-005"
  (statement := /-- On finite nonempty supports let $a_i>0$, $b_j\ge0$ be
  two unit-mass vectors. For every finite real cost array $C$, there exist
  a coupling $P$ and potentials $u_i+v_j\le C_{ij}$ with
  $\langle P,C\rangle=a^Tu+b^Tv$. Thus the primal minimum and dual maximum
  agree and both are attained. Zero source masses can be omitted from the support. -/)
  (proof := /-- The feasible set is closed in a compact cube and contains the
  independent coupling, so choose a minimizer. For each source row $i$, define
  the cost of sending mass to row $k$ as the smallest $C_{kj}-C_{ij}$ among
  its occupied cells. Every permutation of rows has nonnegative total such cost:
  move a common small mass from the corresponding occupied cells according
  to the permutation, which preserves both marginals. The shortest-flow
  potential argument gives row potentials bounded by these differences.
  Set $v_j=\min_i(C_{ij}-u_i)$. Every occupied cell is tight, and the
  reduced-cost identity gives primal-dual equality. -/)]
theorem finite_atomic_strong_duality [Nonempty ι] [DecidableEq ι] [DecidableEq κ]
    (a : ι → ℝ) (b : κ → ℝ) (ha : ∀ i, 0 < a i) (hb : ∀ j, 0 ≤ b j)
    (hA : ∑ i, a i = 1) (hB : ∑ j, b j = 1) (C : ι → κ → ℝ) :
    ∃ (P : ι → κ → ℝ) (u : ι → ℝ) (v : κ → ℝ),
      IsCoupling a b P ∧ (∀ i j, u i + v j ≤ C i j) ∧
      (∑ i, ∑ j, P i j * C i j) = (∑ i, a i * u i) + ∑ j, b j * v j := by
  obtain ⟨P, hp, hm⟩ := attained a b (fun i => (ha i).le) hb hA hB C
  obtain ⟨u, v, hd, ht⟩ := dual_at_plan a b ha C P hp hm
  exact ⟨P, u, v, hp, hd, (transport_gap C P a u b v hp.1 hp.2.1 hp.2.2 hd).2.2 ht⟩

end StatadicFormal.Discrepancy.Transport
