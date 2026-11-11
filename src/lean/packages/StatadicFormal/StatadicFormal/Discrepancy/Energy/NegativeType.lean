module

public import StatadicFormal.Discrepancy.Energy.Finite
public import StatadicFormal.Geometry.NegativeType.Basic

/-!
# Finite Euclidean energy nonnegativity

The ordinary norm has negative type by the shared Hilbert-space geometry theorem.
This module specializes that result to the finite probability energy functional.
-/

public section

open scoped BigOperators
namespace StatadicFormal.Discrepancy.Energy

/-- Finite probability energy for the ordinary norm is nonnegative. -/
@[blueprint "STATADIC-ENERGY-003"
  (statement := /-- Let $x_1,\ldots,x_n$ lie in a real inner product space and let
  $p_i,q_i$ be nonnegative weights each summing to one. Then
  \[
    2\sum_{i,j}p_iq_j\lVert x_i-x_j\rVert-
    \sum_{i,j}p_ip_j\lVert x_i-x_j\rVert-
    \sum_{i,j}q_iq_j\lVert x_i-x_j\rVert\geq 0.
  \] -/)
  (proof := /-- The signed weights $p_i-q_i$ sum to zero. Negative type of the
  norm makes their distance quadratic nonpositive; symmetry identifies its
  negative with the stated energy. -/)]
theorem finite_euclidean_energy_nonneg {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {n : ℕ} (x : Fin n → E)
    (P Q : StatadicFormal.Probability.FiniteLaw (Fin n)) :
    0 ≤ rawEnergy (fun i j ↦ ‖x i - x j‖) P.weight Q.weight := by
  have hzero : ∑ i, (P.weight i - Q.weight i) = 0 := by
    rw [Finset.sum_sub_distrib, P.total, Q.total, sub_self]
  have h := StatadicFormal.Geometry.distNegativeType_of_innerProductSpace E n x
    (fun i ↦ P.weight i - Q.weight i) hzero
  simp_rw [dist_eq_norm] at h
  rw [rawEnergy_eq_neg_quadratic _ _ _ (fun i j ↦ norm_sub_rev _ _)]
  exact neg_nonneg.mpr h

end StatadicFormal.Discrepancy.Energy
