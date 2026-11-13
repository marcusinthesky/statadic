module

public import Mathlib.MeasureTheory.Measure.WithDensity
public import Mathlib.MeasureTheory.Measure.Prod
public import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
public import Architect

/-! Observation probabilities before discarding the independent censoring factor. -/
@[expose] public section
noncomputable section
open MeasureTheory Set
open scoped ENNReal
namespace StatadicFormal.Estimation.Survival

/-- A recorded event has its failure time in the requested window and is observable at entry. -/
def eventWindow (a : ℝ) (B : Set ℝ) : Set (ℝ × ℝ) :=
  {z | z.1 ∈ B ∧ a ≤ z.1 ∧ z.1 ≤ z.2}

/-- Strict inequality allocates tied failure/censor times to the event branch. -/
def censorWindow (a : ℝ) (B : Set ℝ) : Set (ℝ × ℝ) :=
  {z | z.2 ∈ B ∧ a ≤ z.2 ∧ z.2 < z.1}

/-- Conditioning the independent product law on observation at entry derives both
event and censor contributions, including the censor survival at a recorded event. -/
@[blueprint "STATADIC-SURVIVAL-003"
  (statement := /-- Let $T,C$ have independent laws $F,G$. Enter at fixed $a$
  only when $T,C\ge a$, with positive probability $H=F[a,\infty)G[a,\infty)$.
  For a measurable time window $B$, the conditional event probability is
  $H^{-1}\int_{B\cap[a,\infty)}G[t,\infty)\,dF(t)$.
  The conditional censor probability is
  $H^{-1}\int_{B\cap[a,\infty)}F(c,\infty)\,dG(c)$. Ties are assigned to events. -/)
  (proof := /-- Independence gives a product measure. Slice the event set
  at fixed failure time and the censor set at fixed censor time. The surviving
  opposite coordinate lies in the corresponding upper interval; integrate each
  slice by Tonelli. Both events imply entry. Divide by the product probability
  of the two entry conditions. -/)]
theorem delayed_entry_observation_law (F G : Measure ℝ) [IsProbabilityMeasure F]
    [IsProbabilityMeasure G] (a : ℝ) (B : Set ℝ) (hB : MeasurableSet B)
    (hF : F (Ici a) ≠ 0) (hG : G (Ici a) ≠ 0) :
    let admitted := Ici a ×ˢ Ici a
    let H := F (Ici a) * G (Ici a)
    (F.prod G) admitted ≠ 0 ∧
    (F.prod G) (eventWindow a B) / (F.prod G) admitted =
      (∫⁻ t in B ∩ Ici a, G (Ici t) ∂F) / H ∧
    (F.prod G) (censorWindow a B) / (F.prod G) admitted =
      (∫⁻ c in B ∩ Ici a, F (Ioi c) ∂G) / H := by
  classical
  dsimp only
  rw [Measure.prod_prod]
  refine ⟨mul_ne_zero hF hG, ?_, ?_⟩
  · congr 1
    have hm : MeasurableSet (eventWindow a B) :=
      (hB.preimage measurable_fst).inter ((measurableSet_le measurable_const measurable_fst).inter
        (measurableSet_le measurable_fst measurable_snd))
    rw [Measure.prod_apply hm, ← lintegral_indicator (hB.inter measurableSet_Ici)]
    apply lintegral_congr
    intro t
    by_cases h : t ∈ B ∩ Ici a
    · rw [indicator_of_mem h]
      congr 1
      ext c
      simp [eventWindow, h.1, show a ≤ t from h.2]
    · rw [indicator_of_notMem h]
      have hempty : Prod.mk t ⁻¹' eventWindow a B = ∅ := by
        ext c
        simp only [eventWindow, mem_preimage, mem_ofPred_eq, notMem_empty, iff_false]
        exact fun ⟨hb, ha, _⟩ => h ⟨hb, ha⟩
      rw [hempty, measure_empty]
  · congr 1
    have hm : MeasurableSet (censorWindow a B) :=
      (hB.preimage measurable_snd).inter ((measurableSet_le measurable_const measurable_snd).inter
        (measurableSet_lt measurable_snd measurable_fst))
    rw [Measure.prod_apply_symm hm, ← lintegral_indicator (hB.inter measurableSet_Ici)]
    apply lintegral_congr
    intro c
    by_cases h : c ∈ B ∩ Ici a
    · rw [indicator_of_mem h]
      congr 1
      ext t
      simp [censorWindow, h.1, show a ≤ c from h.2]
    · rw [indicator_of_notMem h]
      have hempty : (fun t : ℝ => (t,c)) ⁻¹' censorWindow a B = ∅ := by
        ext t
        simp only [censorWindow, mem_preimage, mem_ofPred_eq, notMem_empty, iff_false]
        exact fun ⟨hb, ha, _⟩ => h ⟨hb, ha⟩
      rw [hempty, measure_empty]

end StatadicFormal.Estimation.Survival
