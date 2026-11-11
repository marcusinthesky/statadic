module

public import Mathlib.MeasureTheory.Measure.Prod
public import Architect
import Mathlib.Tactic.NormNum

/-! What an honest split contributes to a selected-method error guarantee. -/
@[expose] public section
noncomputable section
open MeasureTheory Set
open scoped ENNReal
namespace StatadicFormal.Inference

/-- Integrating a held-out null guarantee over arbitrary training-data choices. -/
@[blueprint "STATADIC-SELECTION-001"
  (statement := /-- Let training and evaluation data have joint law $P\otimes Q$.
  Let $N$ be the measurable set of training outcomes whose selected hypothesis
  is true. If the evaluation rejection probability for almost every $t\in N$
  is at most $\alpha$, then the probability of reporting a false rejection is
  at most $\alpha P(N)$. The selection rule may be continuous-valued. -/)
  (proof := /-- Restrict the rejection set to training outcomes in $N$.
  Tonelli expresses its probability as the integral over $N$ of the held-out
  rejection probabilities. Apply their almost-sure bound and integrate
  the constant. -/)]
theorem heldout_selection_level {T E : Type*} [MeasurableSpace T] [MeasurableSpace E]
    (P : Measure T) (Q : Measure E) [IsProbabilityMeasure Q]
    (N : Set T) (hN : MeasurableSet N) (R : Set (T × E)) (hR : MeasurableSet R)
    (α : ℝ≥0∞) (hlevel : ∀ᵐ t ∂P, t ∈ N → Q (Prod.mk t ⁻¹' R) ≤ α) :
    (P.prod Q) ((N ×ˢ univ) ∩ R) ≤ α * P N := by
  classical
  rw [Measure.prod_apply ((hN.prod MeasurableSet.univ).inter hR)]
  calc
    _ ≤ ∫⁻ t, N.indicator (fun _ => α) t ∂P := by
      apply lintegral_mono_ae
      filter_upwards [hlevel] with t ht
      by_cases h : t ∈ N
      · rw [indicator_of_mem h]
        have hs : Prod.mk t ⁻¹' ((N ×ˢ univ) ∩ R) = Prod.mk t ⁻¹' R := by ext e; simp [h]
        rw [hs]
        exact ht h
      · rw [indicator_of_notMem h]
        have hs : Prod.mk t ⁻¹' ((N ×ˢ univ) ∩ R) = ∅ := by ext e; simp [h]
        rw [hs, measure_empty]
    _ = _ := by rw [lintegral_indicator hN, lintegral_const, Measure.restrict_apply_univ]

/-- Exact enumeration of selecting the smaller of two valid null p-values on the same data. -/
@[blueprint "STATADIC-SELECTION-002"
  (statement := /-- For two independent p-values uniform on
  $\{1/20,\ldots,1\}$, either one individually rejects at $1/20$ in
  exactly $20$ of $400$ equally likely pairs. Reporting the smaller p-value
  rejects in $39$ of those pairs. -/)
  (proof := /-- In the $20$ by $20$ grid the first row or the first column
  rejects. Their intersection consists of the single pair of minimum ranks. -/)]
theorem selected_minimum_null_grid :
    let grid := (Finset.univ : Finset (Fin 20 × Fin 20))
    (grid.filter (fun z => z.1 = 0)).card = 20 ∧
    (grid.filter (fun z => z.2 = 0)).card = 20 ∧
    (grid.filter (fun z => min z.1 z.2 = 0)).card = 39 := by
  decide

end StatadicFormal.Inference
