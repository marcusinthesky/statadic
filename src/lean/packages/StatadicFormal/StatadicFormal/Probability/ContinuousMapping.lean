module

public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure

/-! Continuous maps at a constant probability limit; behavior away from the limit is unrestricted. -/

public section
open MeasureTheory Filter
open scoped Topology
namespace StatadicFormal.Probability

/-- Continuous mapping at the constant limit; the function may be discontinuous elsewhere. -/
theorem tendstoInMeasure_map_at_constant {Ω E F : Type*} [MeasurableSpace Ω] [PseudoMetricSpace E] [PseudoMetricSpace F]
    (μ : Measure Ω) [IsFiniteMeasure μ] (X : ℕ → Ω → E) (x : E) (f : E → F)
    (hX : TendstoInMeasure μ X atTop (fun _ => x)) (hf : ContinuousAt f x)
    (hm : ∀ n, AEStronglyMeasurable (fun ω => f (X n ω)) μ) :
    TendstoInMeasure μ (fun n ω => f (X n ω)) atTop (fun _ => f x) := by
  apply (exists_seq_tendstoInMeasure_atTop_iff hm).mpr
  intro ns hns
  obtain ⟨ks, hks, hconv⟩ := (hX.comp hns.tendsto_atTop).exists_seq_tendsto_ae
  refine ⟨ks, hks, ?_⟩
  filter_upwards [hconv] with ω hω
  exact hf.tendsto.comp hω

end StatadicFormal.Probability
