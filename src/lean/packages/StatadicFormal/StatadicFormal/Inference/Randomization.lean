module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Lattice
public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.GroupTheory.GroupAction.Defs
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum
public import Architect

public section

open scoped BigOperators

namespace StatadicFormal.Inference

/-- Finite form of the averaging argument in Hemerik and Goeman (2018),
Theorem 1, second proof. The null is expressed as invariance of rejection
probability under each transformation. The pointwise bound is the paper's
order-statistic counting step, stated explicitly as a premise. -/
theorem finite_randomization_level
    {Ω G : Type*} [Fintype Ω] [Fintype G] [Nonempty G]
    (w : Ω → ℝ) (transform : G → Ω → Ω) (reject : Ω → Prop)
    [DecidablePred reject] (α : ℝ)
    (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hinv : ∀ g, (∑ ω, w ω * if reject (transform g ω) then 1 else 0) =
      ∑ ω, w ω * if reject ω then 1 else 0)
    (hcount : ∀ ω, (∑ g, if reject (transform g ω) then (1 : ℝ) else 0) ≤
      α * (Fintype.card G : ℝ)) :
    (∑ ω, w ω * if reject ω then 1 else 0) ≤ α := by
  have hcard : (0 : ℝ) < Fintype.card G := by exact_mod_cast Fintype.card_pos
  have htotal : (Fintype.card G : ℝ) *
      (∑ ω, w ω * if reject ω then 1 else 0) ≤ α * (Fintype.card G : ℝ) := by
    calc
      (Fintype.card G : ℝ) * (∑ ω, w ω * if reject ω then 1 else 0) =
          ∑ g : G, ∑ ω, w ω * if reject (transform g ω) then 1 else 0 := by
            simp only [hinv, Finset.sum_const, Finset.card_univ,
              nsmul_eq_mul]
      _ = ∑ ω, w ω * (∑ g, if reject (transform g ω) then (1 : ℝ) else 0) := by
            rw [Finset.sum_comm]
            apply Finset.sum_congr rfl
            intro ω _
            rw [Finset.mul_sum]
      _ ≤ ∑ ω, w ω * (α * (Fintype.card G : ℝ)) := by
            apply Finset.sum_le_sum
            intro ω _
            exact mul_le_mul_of_nonneg_left (hcount ω) (hw ω)
      _ = α * (Fintype.card G : ℝ) := by
            rw [← Finset.sum_mul, hsum, one_mul]
  nlinarith



noncomputable section

lemma upper_rank_count {G : Type*} [Fintype G] (f : G → ℝ) (t : ℝ) (ht : 0 ≤ t) :
    ((Finset.univ.filter (fun x : G =>
      (((Finset.univ.filter (fun y : G => f x ≤ f y)).card : ℕ) : ℝ) ≤ t)).card : ℝ) ≤ t := by
  classical
  let A : Finset G := Finset.univ.filter (fun x : G =>
      ((Finset.univ.filter (fun y : G => f x ≤ f y)).card : ℝ) ≤ t)
  by_cases hA : A.Nonempty
  · obtain ⟨x, hx, hmin⟩ := Finset.exists_min_image A f hA
    have hsub : A ⊆ Finset.univ.filter (fun y : G => f x ≤ f y) := by
      intro y hy
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      exact hmin y hy
    have hcard := Finset.card_le_card hsub
    have hx' : ((Finset.univ.filter (fun y : G => f x ≤ f y)).card : ℝ) ≤ t := by
      simpa [A] using hx
    exact le_trans (by exact_mod_cast hcard) hx'
  · change (A.card : ℝ) ≤ t
    rw [Finset.not_nonempty_iff_eq_empty.mp hA]
    simpa using ht

variable {G Ω : Type*} [Group G] [Fintype G] [MulAction G Ω]

/-- Number of group transforms whose statistic is at least the observed statistic, including ties. -/
def rank (T : Ω → ℝ) (ω : Ω) : ℝ :=
  ((Finset.univ.filter (fun g : G => T ω ≤ T (g • ω))).card : ℝ)

lemma rank_smul (T : Ω → ℝ) (ω : Ω) (g : G) :
    rank (G := G) T (g • ω) = ((Finset.univ.filter (fun h : G => T (g • ω) ≤ T (h • ω))).card : ℝ) := by
  classical
  unfold rank
  apply congrArg (fun n : ℕ => (n : ℝ))
  apply Finset.card_bij (fun h _ => h * g)
  · intro h hh
    simpa only [Finset.mem_filter, Finset.mem_univ, true_and, mul_smul] using hh
  · intro a ha b hb he
    exact mul_right_cancel he
  · intro b hb
    refine ⟨b * g⁻¹, ?_, by simp⟩
    simpa only [Finset.mem_filter, Finset.mem_univ, true_and, mul_smul, inv_smul_smul] using hb

lemma count_rank (T : Ω → ℝ) (ω : Ω) (α : ℝ) (hα : 0 ≤ α) :
    (∑ g : G, if rank (G := G) T (g • ω) ≤ α * (Fintype.card G : ℝ) then (1 : ℝ) else 0) ≤
      α * (Fintype.card G : ℝ) := by
  classical
  simp_rw [rank_smul]
  rw [Finset.sum_boole]
  exact upper_rank_count (fun g : G => T (g • ω)) _
    (mul_nonneg hα (Nat.cast_nonneg _))

/-- Finite-group specialization of Hemerik and Goeman (2018), Theorem 1.
The test uses an upper-tail rank rule and counts ties conservatively.
Equivalence to the paper’s order-statistic rule has not been formalized.
The finite data law itself is group-invariant, stronger than the paper's joint
statistic invariance. The rank count is derived, not assumed. -/
@[blueprint "STATADIC-RANDOMIZATION-001"
  (statement := /-- Let a finite group $G$ act on a finite sample space with
  probability law $P$ satisfying $P(gx)=P(x)$ for all $g,x$. For a real statistic
  $T$, count upper-tail ties in
  \[
    R(x)=\sum_{g\in G}\mathbf{1}\{T(gx)\ge T(x)\}.
  \]
  For $\alpha\ge0$, rejection when $R(x)\le\alpha |G|$ satisfies
  \[
    P\{R(X)\le\alpha |G|\}\le\alpha.
  \]
  This is a finite-law rank formulation; equivalence to the paper's
  order-statistic rule remains to be formalized. -/)
  (proof := /-- Group multiplication permutes the transformation indices. On
  each orbit, the number of rejected transformations is at most $\alpha|G|$:
  choose a rejected statistic of minimum value and include all rejected
  transformations in its tie-inclusive upper tail. Invariance and finite-sum
  exchange then give
  \[
    P\{R(X)\le\alpha|G|\}
      =\mathbb{E}\left[\frac{1}{|G|}\sum_{g\in G}
        \mathbf{1}\{R(gX)\le\alpha|G|\}\right]\le\alpha.
  \] -/)]
theorem finite_group_randomization_level [Fintype Ω]
    (T : Ω → ℝ) (w : Ω → ℝ) (α : ℝ) (hα : 0 ≤ α)
    (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hinvariant : ∀ (g : G) ω, w (g • ω) = w ω) :
    (∑ ω, w ω * if rank (G := G) T ω ≤ α * (Fintype.card G : ℝ) then 1 else 0) ≤ α := by
  classical
  apply finite_randomization_level w (fun (g : G) ω => g • ω)
    (fun ω => rank (G := G) T ω ≤ α * (Fintype.card G : ℝ)) α hw hsum
  · intro g
    calc
      (∑ ω, w ω * if rank (G := G) T (g • ω) ≤ α * (Fintype.card G : ℝ) then 1 else 0) =
          ∑ ω, w (g • ω) * if rank (G := G) T (g • ω) ≤ α * (Fintype.card G : ℝ) then 1 else 0 := by
            simp only [hinvariant]
      _ = ∑ ω, w ω * if rank (G := G) T ω ≤ α * (Fintype.card G : ℝ) then 1 else 0 :=
            Equiv.sum_comp (MulAction.toPerm g) (fun ω => w ω * if rank (G := G) T ω ≤ α * (Fintype.card G : ℝ) then 1 else 0)
  · intro ω
    exact count_rank T ω α hα

end
end StatadicFormal.Inference

