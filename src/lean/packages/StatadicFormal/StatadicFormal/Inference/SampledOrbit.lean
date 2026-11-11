module

public import StatadicFormal.Inference.InvariantRandomization
public import StatadicFormal.Inference.SampledRandomization
public import Mathlib.Data.Finset.Sort
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! Exact null-level control for independent random transformations and the role of ties. -/
@[expose] public section
noncomputable section
open MeasureTheory
open scoped BigOperators
namespace StatadicFormal.Inference

variable {G Ω : Type*} [Group G] [Fintype G] [MulAction G Ω]

omit [Fintype G] in
private lemma measurable_sampleP [MeasurableSpace Ω] {B : ℕ}
    (T : Ω → ℝ) (hT : Measurable T) (hact : ∀ g : G, Measurable fun x : Ω => g • x)
    (s : Fin B → G) : Measurable fun x => sampleP T x s := by
  classical
  have hs : Measurable fun x : Ω => ∑ i : Fin B, if T x ≤ T (s i • x) then (1 : ℝ) else 0 :=
    Finset.measurable_fun_sum _ fun i _ => Measurable.ite
      (measurableSet_le hT (hT.comp (hact (s i)))) measurable_const measurable_const
  simpa [sampleP, Finset.sum_boole] using
    ((measurable_const.add hs).div_const (B + 1 : ℝ) :
      Measurable fun x : Ω => (1 + ∑ i : Fin B, if T x ≤ T (s i • x) then (1 : ℝ) else 0) /
        (B + 1 : ℝ))

/-- Sampling with replacement from the finite group is represented by the average
over all ordered tuples. The data law can be continuous or discrete. -/
@[blueprint "STATADIC-RANDOMIZATION-004"
  (statement := /-- Let a finite group $G$ act by probability-preserving transformations
  under the null. Independently of the data, draw $B$ iid uniform transformations with
  replacement. With an inclusive upper tail and $p=(1+b)/(B+1)$,
  $\Pr(p\le\alpha)\le\alpha$ for every $\alpha\ge0$, including ties. -/)
  (proof := /-- For fixed data $x$, choose a uniform origin $h$ on its orbit.
  Apply the finite sampled-rank counting theorem to the statistic $g\mapsto T(gx)$
  on the group itself. It bounds the average over origins and ordered schedules.
  Integrate the bound over $x$. Each origin preserves the null law, so the resulting
  average equals the rejection probability at the observed origin. -/)]
theorem sampled_orbit_level [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {B : ℕ} (T : Ω → ℝ) (hT : Measurable T)
    (hact : ∀ g : G, MeasurePreserving (fun x : Ω => g • x) μ μ) (α : ℝ) (hα : 0 ≤ α) :
    (∑ s : Fin B → G, μ.real {x | sampleP T x s ≤ α}) / (Fintype.card G : ℝ) ^ B ≤ α := by
  classical
  let m : ℝ := Fintype.card G
  have hm : 0 < m := Nat.cast_pos.mpr Fintype.card_pos
  let A (s : Fin B → G) : Set Ω := {x | sampleP T x s ≤ α}
  have hA (s) : MeasurableSet (A s) := measurableSet_le
    (measurable_sampleP T hT (fun g => (hact g).measurable) s) measurable_const
  let f (h : G) (s : Fin B → G) := (fun x : Ω => (A s).indicator (1 : Ω → ℝ) (h • x))
  have meas (h : G) (s) : MeasurableSet ((fun x : Ω => h • x) ⁻¹' A s) :=
    (hA s).preimage (hact h).measurable
  have hf (h : G) (s) : Integrable (f h s) μ :=
    (integrable_const (1 : ℝ)).indicator (meas h s) |>.congr
      (Filter.Eventually.of_forall fun _ => by simp [f, Set.indicator, Set.preimage])
  have hfint (h : G) (s) : ∫ x, f h s x ∂μ = μ.real (A s) := by
    change (∫ x, (A s).indicator (1 : Ω → ℝ) (h • x) ∂μ) = _
    have eqf : (fun x : Ω => (A s).indicator (1 : Ω → ℝ) (h • x)) =
        ((fun x : Ω => h • x) ⁻¹' A s).indicator (1 : Ω → ℝ) := by
      ext x
      simp [Set.indicator, Set.preimage]
    rw [eqf, integral_indicator_one (meas h s), measureReal_def,
      (hact h).measure_preimage (hA s).nullMeasurableSet]
    rfl
  have hpoint (x : Ω) : ∑ h : G, ∑ s : Fin B → G, (1 / m / m ^ B) * f h s x ≤ α := by
    have horb : ∀ (h : G) (s : Fin B → G),
        sampleP (fun z : G => T (z • x)) h s = sampleP T (h • x) s := by
      intro h s
      simp [sampleP, smul_eq_mul, mul_smul]
    have bound := sampleP_level (G := G) (B := B) (fun z : G => T (z • x)) (fun _ => 1 / m) α hα
      (fun _ => by positivity) (by simp [nsmul_eq_mul, ← show (Fintype.card G : ℝ) = m from rfl]) (fun _ _ => rfl)
    simpa only [horb, f, A, Set.indicator, Set.mem_ofPred_eq, Pi.one_apply, m] using bound
  have htotal := integral_mono (μ := μ) (integrable_finsetSum _ fun h _ =>
    integrable_finsetSum _ fun s _ => (hf h s).const_mul (1 / m / m ^ B))
    (integrable_const α) hpoint
  simp only [integral_finsetSum _ fun h _ => integrable_finsetSum _ fun s _ =>
    (hf h s).const_mul (1 / m / m ^ B), integral_finsetSum _ fun s _ =>
    (hf _ s).const_mul (1 / m / m ^ B), integral_const_mul, hfint, integral_const,
    probReal_univ, one_smul, Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at htotal
  rw [← Finset.mul_sum] at htotal
  change (∑ s, μ.real (A s)) / m ^ B ≤ α
  convert htotal using 1
  change _ = m * ((1 / m / m ^ B) * _)
  field_simp

omit [Fintype G] in
/-- The extreme tie case of the implemented inclusive count. -/
@[blueprint "STATADIC-RANDOMIZATION-005"
  (statement := /-- For a constant statistic all transformations tie, and the
  inclusive add-one p-value equals one for every schedule and every draw count. -/)
  (proof := /-- Every transformed statistic equals the observed value. All $B$
  indicators are one, giving $(1+B)/(B+1)=1$. -/)]
theorem constant_sample_p {B : ℕ} (x : Ω) (s : Fin B → G) (c : ℝ) :
    sampleP (fun _ => c) x s = 1 := by
  simp [sampleP, add_comm]
  positivity

/-- Under no ties every possible upper rank occurs exactly once on the transformed roster. -/
private lemma untied_orbit_count (T : Ω → ℝ) (x : Ω)
    (ht : Function.Injective fun g : G => T (g • x)) (k : ℕ) (hk : k ≤ Fintype.card G) :
    (∑ g : G, if rank (G := G) T (g • x) ≤ k then (1 : ℝ) else 0) = k := by
  classical
  let _ := LinearOrder.lift' (fun g : G => T (g • x)) ht
  let n := Fintype.card G
  let e := Fintype.orderIsoFinOfCardEq G (k := n) rfl
  have order (i j : Fin n) : T (e i • x) ≤ T (e j • x) ↔ i ≤ j := e.le_iff_le
  have hr (i : Fin n) : rank (G := G) T (e i • x) = (n - i.val : ℕ) := by
    rw [rank_smul, ← Finset.sum_boole, ← e.toEquiv.sum_comp
      (fun g : G => if T (e i • x) ≤ T (g • x) then (1 : ℝ) else 0)]
    change (∑ j : Fin n, if T (e i • x) ≤ T (e j • x) then (1 : ℝ) else 0) = _
    simp only [order, Finset.sum_boole]
    have he : Finset.univ.filter (fun j : Fin n => i ≤ j) = Finset.Ici i := by ext; simp
    rw [he, Fin.card_Ici]
  rw [← e.toEquiv.sum_comp (fun g : G => if rank (G := G) T (g • x) ≤ k then (1 : ℝ) else 0)]
  change (∑ j : Fin n, if rank (G := G) T (e j • x) ≤ k then (1 : ℝ) else 0) = _
  simp only [hr, Nat.cast_le, Finset.sum_boole]
  have hk' : k ≤ n := hk
  by_cases hk0 : k = 0
  · subst k
    have hn (i : Fin n) : ¬n - i.val ≤ 0 := by have := i.isLt; omega
    simp [hn]
  · let a : Fin n := ⟨n - k, by omega⟩
    have he : Finset.univ.filter (fun i : Fin n => n - i.val ≤ k) = Finset.Ici a := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Ici]
      change n - i.val ≤ k ↔ n - k ≤ i.val
      omega
    rw [he, Fin.card_Ici]
    congr 1
    change n - (n - k) = k
    omega

/-- Only a tie-free transformed roster yields exact rejection probability at each grid point. -/
@[blueprint "STATADIC-RANDOMIZATION-006"
  (statement := /-- On an invariant null probability space let all transformed
  statistics be distinct almost surely. For every integer $0\le k\le |G|$,
  the inclusive full-group p-value satisfies $P(p\le k/|G|)=k/|G|$. -/)
  (proof := /-- Sort the transformed scores at each tie-free observation. The upper
  ranks along that ordering are $|G|,\ldots,1$, so exactly $k$ transformed origins
  reject. Integrate this count, exchange the finite sum with the integral and use
  invariance of the null law to give every origin the same rejection probability. -/)]
theorem untied_full_group_grid [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T : Ω → ℝ) (hT : Measurable T) (hact : ∀ g : G, MeasurePreserving (fun x : Ω => g • x) μ μ)
    (ht : ∀ᵐ x ∂μ, Function.Injective fun g : G => T (g • x)) (k : ℕ) (hk : k ≤ Fintype.card G) :
    μ.real {x | rank (G := G) T x ≤ k} = k / (Fintype.card G : ℝ) := by
  classical
  let A := {x : Ω | rank (G := G) T x ≤ k}
  have hA : MeasurableSet A := by
    have h : Measurable fun x : Ω => ∑ g : G, if T x ≤ T (g • x) then (1 : ℝ) else 0 :=
      Finset.measurable_fun_sum _ fun g _ =>
      Measurable.ite (measurableSet_le hT (hT.comp (hact g).measurable))
        (measurable_const (a := (1 : ℝ))) (measurable_const (a := (0 : ℝ)))
    have hr (x : Ω) : rank (G := G) T x =
        ∑ g : G, if T x ≤ T (g • x) then (1 : ℝ) else 0 := by
      simpa [Finset.sum_boole] using rank_smul T x (1 : G)
    simpa only [A, hr] using measurableSet_le h (measurable_const (a := (k : ℝ)))
  let E (g : G) := (fun x : Ω => g • x) ⁻¹' A
  have hm (g : G) : MeasurableSet (E g) := hA.preimage (hact g).measurable
  have he (g : G) : μ.real (E g) = μ.real A :=
    congrArg ENNReal.toReal ((hact g).measure_preimage hA.nullMeasurableSet)
  have hcount : (fun x : Ω => ∑ g : G, (E g).indicator (1 : Ω → ℝ) x) =ᵐ[μ]
      (fun _ => (k : ℝ)) := ht.mono fun x h => by
    simpa [Set.indicator, E, A] using untied_orbit_count T x h k hk
  have hint (g : G) : Integrable ((E g).indicator (1 : Ω → ℝ)) μ :=
    (integrable_const (1 : ℝ)).indicator (hm g)
  have htotal : (Fintype.card G : ℝ) * μ.real A = k := calc
    _ = ∑ g : G, ∫ x, (E g).indicator (1 : Ω → ℝ) x ∂μ := by
      simp only [integral_indicator_one (hm _), he, Finset.sum_const, Finset.card_univ,
        nsmul_eq_mul]
    _ = ∫ x, ∑ g : G, (E g).indicator (1 : Ω → ℝ) x ∂μ :=
      (integral_finsetSum _ fun g _ => hint g).symm
    _ = ∫ _ : Ω, (k : ℝ) ∂μ := integral_congr_ae hcount
    _ = _ := by simp
  exact (eq_div_iff (Nat.cast_ne_zero.mpr Fintype.card_pos.ne')).mpr
    ((mul_comm _ _).trans htotal)

end StatadicFormal.Inference
