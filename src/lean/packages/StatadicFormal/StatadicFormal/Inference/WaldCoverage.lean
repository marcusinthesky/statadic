module

public import StatadicFormal.Probability.ContinuousMapping
public import Mathlib.MeasureTheory.Measure.Portmanteau
public import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
public import Architect
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-! Transfer a scalar contrast CLT and a consistent scale into interval coverage. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory Set
open scoped Topology
namespace StatadicFormal.Inference

/-- A continuity set of a studentized limiting distribution determines asymptotic coverage. -/
@[blueprint "STATADIC-WALD-001"
  (statement := /-- Let $X_n$ be a scaled contrast error and $D_n$ its positive
  estimated scale in the same units. If $X_n\Rightarrow\sigma Z$, $D_n\to\sigma$
  in probability, and $\sigma>0$, then $X_n/D_n\Rightarrow Z$.
  For every measurable acceptance region $K$ whose boundary has zero limiting
  probability, $P(X_n/D_n\in K)\to P(Z\in K)$. Taking $K=[-c,c]$ gives
  the limiting coverage of the corresponding two-sided interval. -/)
  (proof := /-- The reciprocal is continuous at the positive scale limit.
  Slutsky gives convergence of the product of the contrast error and the
  reciprocal scale. Apply the continuity-set implication of the portmanteau
  theorem to the laws of these studentized errors. -/)]
theorem studentized_coverage {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (X D : ℕ → Ω → ℝ) (σ : ℝ) (hσ : 0 < σ)
    (hD : TendstoInMeasure μ D atTop (fun _ => σ)) (hm : ∀ n, AEMeasurable (D n) μ)
    (hX : TendstoInDistribution X atTop (fun z : ℝ => σ * z) (fun _ => μ) ν)
    (K : Set ℝ) (hK : MeasurableSet K) (hbd : ν (frontier K) = 0) :
    TendstoInDistribution (fun n ω => X n ω / D n ω) atTop id (fun _ => μ) ν ∧
      Tendsto (fun n => μ {ω | X n ω / D n ω ∈ K}) atTop (𝓝 (ν K)) := by
  have hi := StatadicFormal.Probability.tendstoInMeasure_map_at_constant μ D σ Inv.inv hD
    (continuousAt_inv₀ hσ.ne') (fun n => (hm n).inv.aestronglyMeasurable)
  have hp := TendstoInDistribution.continuous_comp_prodMk_of_tendstoInMeasure_const
    (continuous_fst.mul continuous_snd : Continuous (fun q : ℝ × ℝ => q.1 * q.2))
    hX hi (fun n => (hm n).inv)
  have h : TendstoInDistribution (fun n ω => X n ω / D n ω) atTop id (fun _ => μ) ν := by
    apply hp.congr (fun _ => Filter.Eventually.of_forall fun _ => (div_eq_mul_inv ..).symm)
    exact Filter.Eventually.of_forall fun z => by change σ * z * σ⁻¹ = z; field_simp
  refine ⟨h, ?_⟩
  have hlim := ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto'
    h.tendsto (E := K) (by simpa using hbd)
  change Tendsto (fun n => (μ.map fun ω => X n ω / D n ω) K) atTop (𝓝 ((ν.map id) K)) at hlim
  change Tendsto (fun n => μ ((fun ω => X n ω / D n ω) ⁻¹' K)) atTop (𝓝 (ν K))
  simpa only [Measure.map_id, Measure.map_apply_of_aemeasurable (h.forall_aemeasurable _) hK]
    using hlim

/-- The interval event is the standardized event when the reported standard error is positive. -/
theorem interval_event_iff (b b₀ se c : ℝ) (hs : 0 < se) :
    b - c*se ≤ b₀ ∧ b₀ ≤ b + c*se ↔ (b-b₀)/se ∈ Set.Icc (-c) c := by
  rw [Set.mem_Icc, le_div_iff₀ hs, div_le_iff₀ hs]
  constructor <;> rintro ⟨h₁,h₂⟩ <;> constructor <;> linarith

end StatadicFormal.Inference
