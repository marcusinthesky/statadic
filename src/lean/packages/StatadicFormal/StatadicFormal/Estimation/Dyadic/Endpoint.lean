module

public import Architect
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Tactic.Ring

/-! Identification of intercept and node effects in an undirected dyadic design. -/
@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Estimation.Dyadic

/-- The intercept absorbs the common endpoint offset when a reference node is omitted. -/
@[blueprint "STATADIC-DYADIC-001"
  (statement := /-- In an undirected dyad $(i,j)$ the node-effect design
  has entries $D_{ijk}=1_{i=k}+1_{j=k}$. With any base node $b$,
  \[
    c+u_i+u_j=(c+2u_b)+\sum_{k\ne b}D_{ijk}(u_k-u_b).
  \]
  Consequently changing the base preserves the fitted linear predictor.
  Including every effect and the intercept leaves a common-shift ambiguity. -/)
  (proof := /-- Insert the omitted base term, whose contrast is zero.
  The endpoint indicator sums select $u_i-u_b$ and $u_j-u_b$.
  Adding the adjusted intercept recovers the original predictor. -/)]
theorem endpoint_rebase {V : Type*} [Fintype V] [DecidableEq V]
    (c : ℝ) (u : V → ℝ) (b i j : V) :
    (c + 2 * u b) + ∑ k, (if k = b then 0 else
      ((if k = i then (1 : ℝ) else 0) + (if k = j then 1 else 0)) * (u k - u b)) =
      c + u i + u j := by
  have hz (k : V) : (if k = b then 0 else
      ((if k = i then (1 : ℝ) else 0) + (if k = j then 1 else 0)) * (u k - u b)) =
      ((if k = i then (1 : ℝ) else 0) + (if k = j then 1 else 0)) * (u k - u b) := by
    split <;> simp_all
  simp_rw [hz]
  simp only [add_mul, ite_mul, one_mul, zero_mul, Finset.sum_add_distrib,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  ring

end StatadicFormal.Estimation.Dyadic
