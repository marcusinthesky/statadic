module

public import Mathlib.Data.Rat.Defs
public import Mathlib.Data.Fintype.Fin
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! Exact-arithmetic model of the leading-block mask in `statadic.sample.packed.masked_sum`.
This does not specify JAX floating-point reduction order or prove Python correspondence. -/
@[expose] public section
open scoped BigOperators
namespace StatadicFormal.Discrepancy.MMD

/-- Zero padded entries before summing, as in the JAX source operation. -/
def maskedReduction {m n : ℕ} (a : Fin m → Fin n → ℚ) (rows cols : ℕ) : ℚ :=
  ∑ i, ∑ j, if i.val < rows ∧ j.val < cols then a i j else 0

/-- Sum only the declared leading block. -/
def leadingReduction {m n : ℕ} (a : Fin m → Fin n → ℚ) (rows cols : ℕ) : ℚ :=
  ∑ i ∈ Finset.univ.filter (fun i : Fin m => i.val < rows),
    ∑ j ∈ Finset.univ.filter (fun j : Fin n => j.val < cols), a i j

/-- Padding exclusion preserves the exact rational leading-block sum. -/
theorem maskedReduction_eq_leading {m n : ℕ} (a : Fin m → Fin n → ℚ)
    (rows cols : ℕ) : maskedReduction a rows cols = leadingReduction a rows cols := by
  simp [maskedReduction, leadingReduction, Finset.sum_filter, ite_and]

end StatadicFormal.Discrepancy.MMD
