module

public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Basic
public import Architect

/-! Failure-inclusive operating characteristics for attempted simulation trials. -/
@[expose] public section
noncomputable section
namespace StatadicFormal.Diagnostics

/-- Sharp finite bounds over all unobserved outcomes of failed trials. -/
@[blueprint "STATADIC-QUALIFICATION-001"
  (statement := /-- Of $n>0$ attempted simulation trials let $V$ be the valid trials
  and $R\subseteq V$ the observed rejections. If $A$ completes the unobserved outcomes
  with $A\cap V=R$, the overall rejection rate satisfies
  $|R|/n\le |A|/n\le(|R|+n-|V|)/n$. Both endpoints are attainable. -/)
  (proof := /-- Every completed set contains $R$ and lies in the disjoint union
  $R\cup V^c$. Compare the three counts, divide by the same positive $n$, then
  take the completions $A=R$ and $A=R\cup V^c$ to attain the endpoints. -/)]
theorem qualification_completion_bounds {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (V R : Finset ι) (hr : R ⊆ V) :
    let U := R ∪ Vᶜ
    let rate := fun A : Finset ι => (A.card : ℝ) / Fintype.card ι
    U.card = R.card + (Fintype.card ι - V.card) ∧ R ∩ V = R ∧ U ∩ V = R ∧
      ∀ A : Finset ι, A ∩ V = R → rate R ≤ rate A ∧ rate A ≤ rate U := by
  classical
  have hn : 0 < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr Fintype.card_pos
  have hd : Disjoint R Vᶜ := Finset.disjoint_left.mpr fun i hi hc =>
    (Finset.mem_compl.mp hc) (hr hi)
  refine ⟨by rw [Finset.card_union_of_disjoint hd, Finset.card_compl],
    Finset.inter_eq_left.mpr hr, ?_, ?_⟩
  · ext i
    simp only [Finset.mem_inter, Finset.mem_union, Finset.mem_compl]
    exact ⟨fun h => h.1.resolve_right (not_not_intro h.2), fun h => ⟨Or.inl h, hr h⟩⟩
  · intro A hA
    have lower : R ⊆ A := hA ▸ Finset.inter_subset_left
    have upper : A ⊆ R ∪ Vᶜ := by
      intro i hi
      by_cases hv : i ∈ V
      · exact Finset.mem_union_left _ (hA ▸ Finset.mem_inter.mpr ⟨hi, hv⟩)
      · exact Finset.mem_union_right _ (Finset.mem_compl.mpr hv)
    exact ⟨div_le_div_of_nonneg_right (by exact_mod_cast Finset.card_le_card lower)
        hn.le,
      div_le_div_of_nonneg_right (by exact_mod_cast Finset.card_le_card upper) hn.le⟩

end StatadicFormal.Diagnostics
