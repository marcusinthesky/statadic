module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Topology.MetricSpace.Basic

/-!
# Negative type of a metric

The finite zero-sum predicate is independent of probability laws and energy statistics.
Its definition is shared with pricing-perspective; the Hilbert-space instance lives in
`StatadicFormal.Geometry.NegativeType.Basic`.
-/

public section

open scoped BigOperators
namespace StatadicFormal.Geometry

/-- A metric has negative type when every zero-sum finite distance form is nonpositive. -/
@[expose] def DistNegativeType (α : Type*) [PseudoMetricSpace α] : Prop :=
  ∀ (k : ℕ) (x : Fin k → α) (w : Fin k → ℝ), (∑ i, w i) = 0 →
    ∑ i, ∑ j, w i * w j * dist (x i) (x j) ≤ 0

end StatadicFormal.Geometry
