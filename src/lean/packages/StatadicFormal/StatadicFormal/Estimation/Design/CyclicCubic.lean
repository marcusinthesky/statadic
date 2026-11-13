module

public import Architect
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.Ring

/-!
# Uniform cyclic cubic design

These exact-real local identities describe the four weights in
`statadic.api.design.CyclicSpline.transform`. Shared equally spaced cells use the
same coordinate scale. At a join their polynomial extensions differ by a
multiple of the cube of displacement, including at the cyclic seam.
-/

@[expose] public section
noncomputable section
namespace StatadicFormal.Estimation.Design.CyclicCubic

/-- One unit cell of the uniform cubic spline with four neighbouring controls. -/
def cell (a b c d u : ℝ) : ℝ :=
  (a * (1 - u) ^ 3 + b * (3 * u ^ 3 - 6 * u ^ 2 + 4) +
    c * (-3 * u ^ 3 + 3 * u ^ 2 + 3 * u + 1) + d * u ^ 3) / 6

/-- The four untrimmed cardinal weights form a partition of unity. -/
theorem partition_unity (u : ℝ) : cell 1 1 1 1 u = 1 := by
  unfold cell
  ring

/-- Moving a common coefficient into the intercept preserves the predictor.
Taking the omitted coefficient as this reference sets its new value to zero. -/
theorem reference_coding (a b c d r u : ℝ) :
    r + cell (a - r) (b - r) (c - r) (d - r) u = cell a b c d u := by
  unfold cell
  ring

/-- Adjacent cells have identical constant, linear, and quadratic Taylor terms. -/
@[blueprint "STATADIC-SPLINE-001"
  (statement := /-- For neighboring unit cells of a uniform cubic cardinal
  spline, the extensions of their polynomials differ at displacement $h$
  from their join by $(-a+4b-6c+4d-e)h^3/6$, where $a,b,c,d,e$
  are the consecutive control values. Their Taylor coefficients through
  order two therefore agree, including at a cyclic join. -/)
  (proof := /-- Evaluate the two cubic polynomials at $1+h$ and $h$
  respectively. Expand and collect powers of $h$. The terms of degrees
  zero, one, and two cancel. -/)]
theorem adjacent_cell_jet (a b c d e h : ℝ) :
    cell a b c d (1 + h) - cell b c d e h =
      ((-a + 4 * b - 6 * c + 4 * d - e) / 6) * h ^ 3 := by
  unfold cell
  ring

/-- Evaluate a cell after identifying control indices modulo the knot count. -/
def wrappedCell {n : ℕ} (controls : ZMod n → ℝ) (i : ZMod n) (u : ℝ) : ℝ :=
  cell (controls (i - 1)) (controls i) (controls (i + 1)) (controls (i + 2)) u

/-- Advancing the cell index by one cycle leaves every coefficient unchanged. -/
theorem wrapped_periodic {n : ℕ} (controls : ZMod n → ℝ) (i : ZMod n) (u : ℝ) :
    wrappedCell controls (i + (n : ZMod n)) u = wrappedCell controls i u := by
  simp

end StatadicFormal.Estimation.Design.CyclicCubic
