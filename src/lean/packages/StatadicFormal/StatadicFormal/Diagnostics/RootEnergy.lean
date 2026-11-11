module

public import StatadicFormal.Discrepancy.MMD.Population
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Population conditions justifying the metric returned by energy metrization. -/
@[expose] public section
noncomputable section
open MeasureTheory StatadicFormal.Discrepancy.Energy
open scoped InnerProductSpace
namespace StatadicFormal.Diagnostics

/-- Square rooting the population energy yields a scaled Hilbert distance. -/
@[blueprint "STATADIC-ROOTENERGY-001"
  (statement := /-- Let $d(x,y)=\|\phi(x)-\phi(y)\|^2$, with both first and second
  feature moments integrable for every law in a class $C$. If the feature mean
  $m(P)$ is injective on $C$, the root of unrooted energy satisfies the triangle
  inequality and is zero exactly for equal laws in $C$. -/)
  (proof := /-- The population energy identity gives
  $E(P,Q)=2\|m(P)-m(Q)\|^2$. Its square root is $\sqrt 2$ times
  the Hilbert distance. Apply the triangle inequality and injectivity of $m$. -/)]
theorem root_energy_metric {α H : Type*} [MeasurableSpace α] [NormedAddCommGroup H]
    [InnerProductSpace ℝ H] [CompleteSpace H] (C : Set (ProbabilityMeasure α)) (φ : α → H)
    (h1 : ∀ P ∈ C, Integrable φ (P : Measure α))
    (h2 : ∀ P ∈ C, Integrable (fun x => ‖φ x‖ ^ 2) (P : Measure α))
    (hinj : Set.InjOn (fun P : ProbabilityMeasure α => ∫ x, φ x ∂P) C) :
    let d := fun x y => ‖φ x - φ y‖ ^ 2
    let D := fun P Q : ProbabilityMeasure α =>
      Real.sqrt (2 * pairMean P Q d - pairMean P P d - pairMean Q Q d)
    (∀ P ∈ C, ∀ Q ∈ C, D P Q = 0 ↔ P = Q) ∧
      ∀ P ∈ C, ∀ Q ∈ C, ∀ R ∈ C, D P R ≤ D P Q + D Q R := by
  let d := fun x y => ‖φ x - φ y‖ ^ 2
  let D := fun P Q : ProbabilityMeasure α =>
    Real.sqrt (2 * pairMean P Q d - pairMean P P d - pairMean Q Q d)
  let m := fun P : ProbabilityMeasure α => ∫ x, φ x ∂P
  have distForm (P Q : ProbabilityMeasure α) (hp : P ∈ C) (hq : Q ∈ C) :
      D P Q = Real.sqrt 2 * dist (m P) (m Q) := by
    have h := population_energy_kernel_identity (P : Measure α) (Q : Measure α)
      φ d (fun x y => ⟪φ x, φ y⟫_ℝ) (h1 P hp) (h1 Q hq) (h2 P hp) (h2 Q hq)
      (fun _ _ => rfl) (fun x y => by simp [d, norm_sub_sq_real]; ring)
    dsimp only [D]
    rw [h.2.1, Real.sqrt_mul (by norm_num), Real.sqrt_sq (norm_nonneg _), dist_eq_norm]
  change (∀ P ∈ C, ∀ Q ∈ C, D P Q = 0 ↔ P = Q) ∧ _
  constructor
  · intro P hp Q hq
    rw [distForm P Q hp hq, mul_eq_zero, dist_eq_zero]
    simp only [show Real.sqrt 2 ≠ 0 by positivity, false_or]
    exact ⟨fun h => hinj hp hq h, fun h => congrArg m h⟩
  · intro P hp Q hq R hr
    change D P R ≤ D P Q + D Q R
    rw [distForm P R hp hr, distForm P Q hp hq, distForm Q R hq hr, ← mul_add]
    exact mul_le_mul_of_nonneg_left (dist_triangle _ _ _) (Real.sqrt_nonneg _)

end StatadicFormal.Diagnostics
