module

public import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
public import Mathlib.Order.Interval.Finset.Fin
public import Architect

/-!
# Strong familywise error control

Holm (1979), Section 2, pp. 66–67. Each true null supplies a marginal
super-uniformity bound under the same joint law. A step-down rejection is a
prefix of successful tests in an outcome-dependent order, as in the sorted
comparison and cumulative product in `equivalence_multiplicity_control`.
-/

public section

open MeasureTheory

namespace StatadicFormal.Inference

variable {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}

/-- Marginal validity for the subset of true hypotheses under the common law. -/
@[expose] def MarginallyValid (μ : Measure Ω) (p : Fin n → Ω → ℝ) (T : Finset (Fin n)) : Prop :=
  ∀ i ∈ T, ∀ u ∈ Set.Icc (0 : ℝ) 1, μ {ω | p i ω ≤ u} ≤ ENNReal.ofReal u

/-- Boole's inequality, together with the marginal bound at the chosen cutoff. -/
private theorem cutoff_bound (μ : Measure Ω) (p : Fin n → Ω → ℝ) (T : Finset (Fin n))
    (hvalid : MarginallyValid μ p T) (α : ℝ) (hα : α ∈ Set.Icc (0 : ℝ) 1)
    (k : ℕ) (hk : T.card ≤ k) :
    μ (⋃ i ∈ T, {ω | p i ω ≤ α / k}) ≤ ENNReal.ofReal α := by
  by_cases hT : T.Nonempty
  · have hkpos : 0 < k := lt_of_lt_of_le (Finset.card_pos.mpr hT) hk
    have hkreal : (1 : ℝ) ≤ k := by exact_mod_cast hkpos
    have hu : α / k ∈ Set.Icc (0 : ℝ) 1 :=
      ⟨div_nonneg hα.1 (Nat.cast_nonneg k),
        (div_le_one (by positivity)).mpr (hα.2.trans hkreal)⟩
    calc
      _ ≤ ∑ i ∈ T, μ {ω | p i ω ≤ α / k} := measure_biUnion_finset_le T _
      _ ≤ ∑ _i ∈ T, ENNReal.ofReal (α / k) :=
        Finset.sum_le_sum fun i hi => hvalid i hi _ hu
      _ = ENNReal.ofReal ((T.card : ℝ) * (α / k)) := by
        rw [Finset.sum_const, ← ENNReal.ofReal_nsmul, nsmul_eq_mul]
      _ ≤ ENNReal.ofReal α := by
        apply ENNReal.ofReal_le_ofReal
        calc
          _ ≤ (k : ℝ) * (α / k) := mul_le_mul_of_nonneg_right (by exact_mod_cast hk) hu.1
          _ = α := mul_div_cancel₀ _ (by positivity)
  · simp [Finset.not_nonempty_iff_eq_empty.mp hT]

/-- Bonferroni FWER control for any subset of true nulls. -/
@[blueprint "STATADIC-MULTIPLICITY-001"
  (statement := /-- Let $T$ be the true hypotheses among $n$ tests. Suppose
  $0\le\alpha\le1$ and $P(p_i\le u)\le u$ for every $i\in T$ and
  $0\le u\le1$. Reject each $i$ with $p_i\le\alpha/n$. The probability of
  rejecting any member of $T$ is at most $\alpha$, under arbitrary dependence. -/)
  (proof := /-- For nonempty $T$, also $n>0$. Apply the union bound over $T$,
  then marginal validity at $u=\alpha/n$. This bounds the rejection probability
  by $|T|\alpha/n\le\alpha$. The empty union has probability zero. -/)]
theorem bonferroni_fwer (μ : Measure Ω) (p : Fin n → Ω → ℝ) (T : Finset (Fin n))
    (hvalid : MarginallyValid μ p T) (α : ℝ) (hα : α ∈ Set.Icc (0 : ℝ) 1) :
    μ (⋃ i ∈ T, {ω | p i ω ≤ α / n}) ≤ ENNReal.ofReal α := by
  apply cutoff_bound μ p T hvalid α hα n
  simpa using Finset.card_le_univ T

/-- At a zero-based rank all tests up to and including the rank must pass.
The permutation maps hypothesis labels to their ranks for this outcome. -/
@[expose] def StepDownRejects (p : Fin n → Ω → ℝ) (α : ℝ) (rank : Ω → Equiv.Perm (Fin n))
    (i : Fin n) (ω : Ω) : Prop :=
  ∀ j ≤ rank ω i, p ((rank ω).symm j) ω ≤ α / ((n - j.val : ℕ) : ℝ)

/-- Strong FWER of Holm's sorted step-down rule; the bound holds for any tie order. -/
@[blueprint "STATADIC-MULTIPLICITY-002"
  (statement := /-- Under the marginal validity and level assumptions above,
  rank the hypotheses for each outcome. Reject a hypothesis at zero-based
  rank $s$ only if every rank $j\le s$ passes its comparison to
  $\alpha/(n-j)$. The probability of rejecting any true null is at most
  $\alpha$. This includes Holm's ascending order with arbitrary ties. -/)
  (proof := /-- If any true null is rejected, let $r$ be the first rank of
  a true null. The prefix test must have passed at $r$. All $|T|$ true
  nulls occupy distinct ranks in the suffix from $r$ onward, hence
  $|T|\le n-r$. Thus the first true null has $p_i\le\alpha/(n-r)
  \le\alpha/|T|$. Take the union over true nulls, use marginal validity,
  and sum $|T|$ contributions of at most $\alpha/|T|$. If $T$ is empty
  there are no false rejections. -/)]
theorem holm_fwer (μ : Measure Ω) (p : Fin n → Ω → ℝ) (T : Finset (Fin n))
    (hvalid : MarginallyValid μ p T) (α : ℝ) (hα : α ∈ Set.Icc (0 : ℝ) 1)
    (rank : Ω → Equiv.Perm (Fin n)) :
    μ (⋃ i ∈ T, {ω | StepDownRejects p α rank i ω}) ≤ ENNReal.ofReal α := by
  apply le_trans (measure_mono _) (cutoff_bound μ p T hvalid α hα T.card le_rfl)
  intro ω hω
  simp only [Set.mem_iUnion, exists_prop, Set.mem_ofPred_eq] at hω ⊢
  obtain ⟨i, hi, hrejected⟩ := hω
  let ranks := T.map (rank ω).toEmbedding
  have hi' : rank ω i ∈ ranks := Finset.mem_map.mpr ⟨i, hi, rfl⟩
  let r := ranks.min' ⟨_, hi'⟩
  have hr : r ∈ ranks := Finset.min'_mem _ _
  obtain ⟨i₀, hi₀, hri⟩ := Finset.mem_map.mp hr
  have hbound : T.card ≤ n - r.val := by
    have hsub : ranks ⊆ Finset.Ici r := fun j hj => Finset.mem_Ici.mpr (Finset.min'_le _ _ hj)
    simpa [ranks] using Finset.card_le_card hsub
  have hpass := hrejected r (Finset.min'_le _ _ hi')
  have hindex : (rank ω).symm r = i₀ := (rank ω).symm_apply_eq.mpr hri.symm
  rw [hindex] at hpass
  refine ⟨i₀, hi₀, hpass.trans (div_le_div_of_nonneg_left hα.1 ?_ ?_)⟩
  · exact_mod_cast Finset.card_pos.mpr ⟨i, hi⟩
  · exact_mod_cast hbound

end StatadicFormal.Inference
