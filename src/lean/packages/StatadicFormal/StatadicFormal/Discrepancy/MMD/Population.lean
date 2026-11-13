module

public import StatadicFormal.Discrepancy.Energy.Population
public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Tactic.Ring

/-! # Nonnegativity and separation of population MMD

The input kernel is represented by a real Hilbert feature map. Characteristicness
means injectivity of the mean embedding on the chosen class of laws. -/

@[expose] public section
noncomputable section

open MeasureTheory StatadicFormal.Discrepancy.Energy
open scoped InnerProductSpace

namespace StatadicFormal.Discrepancy.MMD

variable {α H : Type*} [MeasurableSpace α] [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]

/-- Population squared MMD written in terms of independent pair draws. -/
def populationMmdSq (k : α → α → ℝ) (P Q : ProbabilityMeasure α) : ℝ :=
  pairMean P P k + pairMean Q Q k - 2 * pairMean P Q k

/-- The embedding exists under first-moment Bochner integrability. -/
theorem populationMmdSq_eq (φ : α → H) (k : α → α → ℝ)
    (hk : ∀ x y, k x y = ⟪φ x, φ y⟫_ℝ) (P Q : ProbabilityMeasure α)
    (hp : Integrable φ (P : Measure α)) (hq : Integrable φ (Q : Measure α)) :
    populationMmdSq k P Q = ‖(∫ x, φ x ∂P) - (∫ x, φ x ∂Q)‖ ^ 2 := by
  have hkernel : k = (fun x y ↦ ⟪φ x, φ y⟫_ℝ) := funext (fun x ↦ funext (hk x))
  rw [populationMmdSq, hkernel, pairMean_inner P P φ hp hp, pairMean_inner Q Q φ hq hq,
    pairMean_inner P Q φ hp hq, norm_sub_sq_real]
  simp only [real_inner_self_eq_norm_sq]
  ring

/-- Population MMD is nonnegative; it separates laws in precisely the classes
on which the Hilbert mean embedding is injective. -/
@[blueprint "STATADIC-MMD-007"
  (statement := /-- Let $k(x,y)=\langle\phi(x),\phi(y)\rangle$, where $\phi$ is
  Bochner integrable under every law in a class $C$. Then the within-plus-within
  minus twice cross kernel expectation $M(P,Q)$ is nonnegative on $C$.
  Furthermore, $M(P,Q)=0$ exactly for equal laws on $C$ if and only if
  $P\mapsto\int\phi\,dP$ is injective on $C$. -/)
  (proof := /-- Factor the three pair expectations into inner products of
  feature means and expand their squared norm difference. A squared norm is
  nonnegative and vanishes exactly at zero. Thus a zero discrepancy amounts
  to equality of the two embeddings. This is equivalent to equality of the
  laws throughout the class exactly when the embedding is injective there. -/)]
theorem population_mmd_separation (C : Set (ProbabilityMeasure α)) (φ : α → H)
    (k : α → α → ℝ) (hk : ∀ x y, k x y = ⟪φ x, φ y⟫_ℝ)
    (hi : ∀ P ∈ C, Integrable φ (P : Measure α)) :
    (∀ P ∈ C, ∀ Q ∈ C, 0 ≤ populationMmdSq k P Q) ∧
      (Set.InjOn (fun P : ProbabilityMeasure α ↦ ∫ x, φ x ∂P) C ↔
        ∀ P ∈ C, ∀ Q ∈ C, (populationMmdSq k P Q = 0 ↔ P = Q)) := by
  have hz (P : ProbabilityMeasure α) (hp : P ∈ C) (Q : ProbabilityMeasure α) (hq : Q ∈ C) :
      populationMmdSq k P Q = 0 ↔ (∫ x, φ x ∂P) = ∫ x, φ x ∂Q := by
    simp [populationMmdSq_eq φ k hk P Q (hi P hp) (hi Q hq), sub_eq_zero]
  refine ⟨fun P hp Q hq ↦ ?_, ?_, ?_⟩
  · rw [populationMmdSq_eq φ k hk P Q (hi P hp) (hi Q hq)]
    exact sq_nonneg _
  · intro hinj P hp Q hq
    exact (hz P hp Q hq).trans ⟨fun h ↦ hinj hp hq h, fun h ↦ by rw [h]⟩
  · intro h P hp Q hq heq
    exact (h P hp Q hq).mp ((hz P hp Q hq).mpr heq)

end StatadicFormal.Discrepancy.MMD
