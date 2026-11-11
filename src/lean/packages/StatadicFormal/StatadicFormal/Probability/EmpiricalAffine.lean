module

public import Architect
public import Mathlib.Algebra.Module.LinearMap.Basic
public import Mathlib.Algebra.Module.BigOperators
public import Mathlib.Basic.Real.Basic

/-! Finite empirical weights and affine pushforwards. -/

public section
open scoped BigOperators
namespace StatadicFormal.Probability
variable {ι V W : Type*} [Fintype ι] [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W]

/-- An affine map commutes with an empirical mean, with the precise offset defect for any mass. -/
@[blueprint "STATADIC-SAMPLE-001"
  (statement := /-- Fix row weights $p_i$, points $x_i$, a linear map $A$ and
  an offset $b$. The first moment of the pushforward is
  $\sum_i p_i(Ax_i+b)=A(\sum_i p_i x_i)+(\sum_i p_i)b$.
  With unit mass the offset appears exactly once. Zero-mass rows have no contribution. -/)
  (proof := /-- Distribute scalar multiplication across the translated point.
  Commute the linear map with each scalar multiplication and the finite sum.
  Collect the constant terms to obtain the total mass times the offset. -/)]
theorem affine_empirical_moment (p : ι → ℝ) (x : ι → V) (A : V →ₗ[ℝ] W) (b : W) :
    (∑ i, p i • (A (x i) + b)) = A (∑ i, p i • x i) + (∑ i, p i) • b ∧
      ((∑ i, p i) = 1 → (∑ i, p i • (A (x i) + b)) = A (∑ i, p i • x i) + b) := by
  have h : (∑ i, p i • (A (x i) + b)) = A (∑ i, p i • x i) + (∑ i, p i) • b := by
    simp [smul_add, Finset.sum_add_distrib, Finset.sum_smul]
  exact ⟨h, fun hp => by rw [h, hp, one_smul]⟩

end StatadicFormal.Probability
