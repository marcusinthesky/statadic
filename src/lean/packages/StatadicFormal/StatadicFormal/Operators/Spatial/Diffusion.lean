module

public import Architect
public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Row-normalised diffusion on a finite graph

The input `cost` is the exponent numerator: for ordinary distances `D`, it is
`D²`. A roster must contain another unit for every row to carry probability
mass. The proofs use exact real arithmetic.
-/

@[expose] public section
noncomputable section

open scoped BigOperators

namespace StatadicFormal.Operators.Spatial

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Off-diagonal affinity at a bandwidth on the same scale as `cost`. -/
def affinity (cost : ι → ι → ℝ) (bandwidth : ℝ) (i j : ι) : ℝ :=
  if i = j then 0 else Real.exp (-(cost i j / bandwidth))

/-- One row of the off-diagonal RBF diffusion operator. -/
def diffusion (cost : ι → ι → ℝ) (bandwidth : ℝ) (i j : ι) : ℝ :=
  affinity cost bandwidth i j / ∑ k, affinity cost bandwidth i k

omit [Fintype ι] in
private theorem affinity_nonneg (cost : ι → ι → ℝ) (h : ℝ) (i j : ι) :
    0 ≤ affinity cost h i j := by
  unfold affinity
  split
  · exact le_rfl
  · exact (Real.exp_pos _).le

/-- A finite row with a peer is a probability vector, with zero mass on itself. -/
@[blueprint "STATADIC-DIFFUSION-001"
  (statement := /-- On a finite roster, an off-diagonal exponential kernel normalised
  by its row sum has nonnegative entries, zero diagonal and unit row sum for every
  row with at least one peer. -/)
  (proof := /-- The peer contributes strictly positive exponential mass and every
  term is nonnegative. Dividing by the positive row total preserves signs; summing
  the quotients divides that total by itself. -/)]
theorem diffusion_row (cost : ι → ι → ℝ) (h : ℝ) (i : ι) (peer : ∃ j, i ≠ j) :
    diffusion cost h i i = 0 ∧ (∀ j, 0 ≤ diffusion cost h i j) ∧
      ∑ j, diffusion cost h i j = 1 := by
  obtain ⟨j, hij⟩ := peer
  have positive : 0 < ∑ k, affinity cost h i k := by
    apply Finset.sum_pos' (fun k _ ↦ affinity_nonneg cost h i k)
    refine ⟨j, Finset.mem_univ j, ?_⟩
    simpa [affinity, hij] using Real.exp_pos (-(cost i j / h))
  refine ⟨by simp [diffusion, affinity], ?_, ?_⟩
  · exact fun k ↦ div_nonneg (affinity_nonneg cost h i k) positive.le
  · simp only [diffusion, ← Finset.sum_div, div_self positive.ne']

/-- Simultaneously rescaling exponent costs and bandwidth preserves the entire graph. -/
@[blueprint "STATADIC-DIFFUSION-002"
  (statement := /-- Multiplying every finite exponent cost and its bandwidth by
  the same positive scalar leaves each normalised diffusion weight unchanged. -/)
  (proof := /-- Cancel the common factor in the exponent quotient. Both the
  unnormalised affinities and their row sums are unchanged. -/)]
theorem diffusion_scale (cost : ι → ι → ℝ) (h c : ℝ) (hc : 0 < c) (i j : ι) :
    diffusion (fun a b ↦ c * cost a b) (c * h) i j = diffusion cost h i j := by
  simp only [diffusion, affinity, mul_div_mul_left _ _ hc.ne']

end StatadicFormal.Operators.Spatial
