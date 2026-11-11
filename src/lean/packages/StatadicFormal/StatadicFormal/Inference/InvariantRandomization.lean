module

public import StatadicFormal.Inference.Randomization
public import Mathlib.Probability.IdentDistrib
public import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.MeasureTheory.Integral.Bochner.Set

/-!
# Full-group randomization on a general probability space

Hemerik and Goeman (2018), Section 2, Theorem 1, second proof. The joint
law of the orbit statistics is invariant; the data law itself need not be.
-/

public section

open MeasureTheory ProbabilityTheory

namespace StatadicFormal.Inference

variable {G Ω : Type*} [Group G] [Fintype G] [MulAction G Ω]

/-- Omitting just the identity from the full group makes the add-one numerator the inclusive rank. -/
lemma rank_eq_one_add_nonidentity (T : Ω → ℝ) (x : Ω) [DecidableEq G] :
    rank (G := G) T x = 1 + ∑ g ∈ Finset.univ.erase (1 : G),
      if T x ≤ T (g • x) then (1 : ℝ) else 0 := by
  have hr := rank_smul T x (1 : G)
  simp only [one_smul] at hr
  rw [hr, ← Finset.sum_boole]
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ (1 : G))]
  simp [add_comm]

/-- The inclusive rank cutoff is the strict order-statistic cutoff in Theorem 1.
`e` lists all transforms in ascending order, and `k` is its zero-based critical index. -/
lemma rank_le_iff_gt_order_stat (T : Ω → ℝ) (x : Ω) (α : ℝ)
    (e : Fin (Fintype.card G) ≃ G) (he : Monotone fun j => T (e j • x))
    (k : Fin (Fintype.card G)) (hk : k.val + 1 = ⌈(1 - α) * Fintype.card G⌉₊) :
    rank (G := G) T x ≤ α * Fintype.card G ↔ T (e k • x) < T x := by
  classical
  let s := Finset.univ.filter fun j => T x ≤ T (e j • x)
  have hr : rank (G := G) T x = s.card := by
    have h := rank_smul T x (1 : G)
    simp only [one_smul] at h
    rw [h, ← Finset.sum_boole, ← (e.sum_comp (fun g => if T x ≤ T (g • x) then (1 : ℝ) else 0))]
    simp [s, Finset.sum_boole]
  have hfloor : (k.val : ℝ) < (1 - α) * Fintype.card G := by
    apply Nat.lt_ceil.mp
    omega
  have hceil : (1 - α) * Fintype.card G ≤ (k.val : ℝ) + 1 := by
    simpa only [← hk, Nat.cast_add, Nat.cast_one] using Nat.le_ceil ((1 - α) * Fintype.card G)
  rw [hr]
  constructor
  · intro h
    by_contra! hnot
    have hsub : Finset.Ici k ⊆ s := fun j hj =>
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, hnot.trans (he (Finset.mem_Ici.mp hj))⟩
    have hn : Fintype.card G - k.val ≤ s.card := by simpa using Finset.card_le_card hsub
    have hn' : Fintype.card G ≤ s.card + k.val := by omega
    have hreal : (Fintype.card G : ℝ) ≤ s.card + k.val := by exact_mod_cast hn'
    linarith
  · intro h
    have hsub : s ⊆ Finset.Ioi k := by
      intro j hj
      apply Finset.mem_Ioi.mpr
      by_contra! hle
      exact (not_lt_of_ge (Finset.mem_filter.mp hj).2) ((he hle).trans_lt h)
    have hn : s.card ≤ Fintype.card G - 1 - k.val := by simpa using Finset.card_le_card hsub
    have hn' : s.card + k.val + 1 ≤ Fintype.card G := by have := k.isLt; omega
    have hreal : (s.card : ℝ) + k.val + 1 ≤ Fintype.card G := by exact_mod_cast hn'
    linarith

variable [MeasurableSpace Ω]

/-- Full-group level for the inclusive upper-tail rank on a general data space. -/
@[blueprint "STATADIC-RANDOMIZATION-003"
  (statement := /-- Let a finite group $G$ act on a probability space of
  outcomes. Assume all transformed statistics are measurable and the joint
  distribution of $(T(hX))_{h\in G}$ agrees with that of $(T(hgX))_{h\in G}$
  for each $g$. Write $r(x)=\#\{h:T(hx)\ge T(x)\}$. For $\alpha\ge0$,
  the full-group test satisfies $P(r(X)\le\alpha |G|)\le\alpha$. -/)
  (proof := /-- Let $A$ be the inclusive rank rejection event. It is
  measurable as a threshold on a finite sum of comparisons. Joint orbit-law
  invariance gives $P(gX\in A)=P(X\in A)$ for each $g$. For every $x$,
  at most $\alpha |G|$ transformed observations reject, by the upper-rank
  count including ties. Integrate this bound, exchange the finite sum and
  integral, and divide by the positive cardinality $|G|$. -/)]
theorem invariant_group_randomization_level (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω → ℝ) (hT : ∀ g : G, Measurable fun x : Ω => T (g • x))
    (hinv : ∀ g : G, IdentDistrib (fun x : Ω => fun h : G => T (h • (g • x)))
      (fun x : Ω => fun h : G => T (h • x)) μ μ)
    (α : ℝ) (hα : 0 ≤ α) :
    μ.real {x | rank (G := G) T x ≤ α * Fintype.card G} ≤ α := by
  classical
  let m := (Fintype.card G : ℝ)
  let reject : Set (G → ℝ) := {v | (∑ g : G, if v 1 ≤ v g then (1 : ℝ) else 0) ≤ α * m}
  have hmeas : MeasurableSet reject := by
    apply measurableSet_le _ measurable_const
    exact Finset.measurable_fun_sum _ fun g _ =>
      Measurable.ite (measurableSet_le (measurable_pi_apply 1) (measurable_pi_apply g))
        measurable_const measurable_const
  let A : Set Ω := {x | rank (G := G) T x ≤ α * m}
  have hvec (x : Ω) : (fun h : G => T (h • x)) ∈ reject ↔ x ∈ A := by
    have hr := rank_smul T x (1 : G)
    simp only [one_smul] at hr
    simp [reject, A, hr, Finset.sum_boole]
  have hpre (g : G) : MeasurableSet {x : Ω | g • x ∈ A} := by
    have h : Measurable (fun x : Ω => fun h : G => T (h • (g • x))) :=
      Measurable.of_eval fun h => by simpa only [← mul_smul] using hT (h * g)
    exact (hmeas.preimage h).congr (Set.ext fun x => hvec (g • x))
  have hprob (g : G) : μ.real {x : Ω | g • x ∈ A} = μ.real A := by
    have h := (hinv g).measure_mem_eq hmeas
    have h₁ : (fun x : Ω => fun h : G => T (h • (g • x))) ⁻¹' reject =
        {x | g • x ∈ A} := Set.ext fun x => hvec (g • x)
    have h₂ : (fun x : Ω => fun h : G => T (h • x)) ⁻¹' reject = A := Set.ext hvec
    simpa only [measureReal_def, h₁, h₂] using congrArg ENNReal.toReal h
  let f (g : G) := ({x : Ω | g • x ∈ A}).indicator (1 : Ω → ℝ)
  have hf (g : G) : Integrable (f g) μ := (integrable_const _).indicator (hpre g)
  have hcount (x : Ω) : (∑ g : G, f g x) ≤ α * m := by
    simpa [f, A, Set.indicator, m] using count_rank (G := G) T x α hα
  have hsum : m * μ.real A ≤ α * m := calc
    _ = ∑ g : G, ∫ x, f g x ∂μ := by
      simp only [f, integral_indicator_one (hpre _), hprob, Finset.sum_const,
        Finset.card_univ, nsmul_eq_mul, m]
    _ = ∫ x, ∑ g : G, f g x ∂μ := (integral_finsetSum _ fun g _ => hf g).symm
    _ ≤ ∫ _ : Ω, α * m ∂μ := integral_mono (integrable_finsetSum _ fun g _ => hf g)
      (integrable_const _) hcount
    _ = α * m := by simp
  change μ.real A ≤ α
  rw [mul_comm m] at hsum
  have hm : 0 < m := by
    dsimp [m]
    exact_mod_cast Fintype.card_pos (α := G)
  exact le_of_mul_le_mul_right hsum hm

end StatadicFormal.Inference
