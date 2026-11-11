module

public import StatadicFormal.Discrepancy.Energy.Finite
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith

/-! DISCO between-group dispersion from empirical group distance means. -/
@[expose] public section
noncomputable section
open scoped BigOperators
open StatadicFormal.Discrepancy.Energy StatadicFormal.Probability
namespace StatadicFormal.Association

variable {G I : Type*} [Fintype G] [Fintype I]

/-- Mean dissimilarity between two empirical laws on the pooled support. -/
def groupDistance (d : I → I → ℝ) (P Q : FiniteLaw I) : ℝ :=
  ∑ i, ∑ j, P.weight i * Q.weight j * d i j

/-- The group energy decomposition and its sign under finite negative type. -/
@[blueprint "STATADIC-DISCO-001"
  (statement := /-- Write $p_g$ for each empirical group law, $w_g$ for its
  share, and $A_{gh}=\sum_{ij}p_{gi}p_{hj}d_{ij}$. Suppose the symmetric
  distance array has nonpositive quadratic form on zero-sum vectors. Then
  \[
  T-W=\tfrac{n}{4}\sum_{g,h}w_gw_h(2A_{gh}-A_{gg}-A_{hh})\ge0,
  \]
  with $T=(n/2)\sum_{gh}w_gw_h A_{gh}$ and $W=(n/2)\sum_gw_g A_{gg}$. -/)
  (proof := /-- Expand the weighted pair sum. Summing out either group in the
  two diagonal terms uses $\sum_g w_g=1$ and recovers the within dispersion.
  Each remaining pair energy is minus the distance quadratic evaluated on
  $p_g-p_h$, whose mass is zero. Weight these nonnegative contributions by
  $n w_g w_h/4$. -/)]
theorem disco_between (n : ℕ) (w : FiniteLaw G) (P : G → FiniteLaw I)
    (d : I → I → ℝ) (hs : ∀ i j, d i j = d j i)
    (hc : ∀ c : I → ℝ, (∑ i, c i) = 0 → quadratic d c ≤ 0) :
    let a := fun g h => groupDistance d (P g) (P h)
    let total := n / 2 * ∑ g, ∑ h, w.weight g * w.weight h * a g h
    let within := n / 2 * ∑ g, w.weight g * a g g
    let pair := ∑ g, ∑ h, w.weight g * w.weight h * rawEnergy d (P g).weight (P h).weight
    total - within = n / 4 * pair ∧ 0 ≤ total - within := by
  dsimp only
  let a := fun g h => groupDistance d (P g) (P h)
  have he (g h : G) : rawEnergy d (P g).weight (P h).weight =
      2 * a g h - a g g - a h h := by rfl
  have hd (f : G → ℝ) : ∑ g, ∑ h, w.weight g * w.weight h * f g =
      ∑ g, w.weight g * f g := by
    apply Finset.sum_congr rfl
    intro g _
    simp_rw [mul_right_comm (w.weight g) _ (f g), ← Finset.mul_sum, w.total, mul_one]
  have hd' (f : G → ℝ) : ∑ g, ∑ h, w.weight g * w.weight h * f h =
      ∑ h, w.weight h * f h := by
    rw [Finset.sum_comm]
    simpa only [mul_comm] using hd f
  have hpair : (∑ g, ∑ h, w.weight g * w.weight h * rawEnergy d (P g).weight (P h).weight) =
      2 * (∑ g, ∑ h, w.weight g * w.weight h * a g h) -
      (∑ g, w.weight g * a g g) - ∑ g, w.weight g * a g g := by
    simp_rw [he, mul_sub, mul_left_comm (w.weight _ * w.weight _) 2, Finset.sum_sub_distrib,
      ← Finset.mul_sum]
    rw [hd, hd']
  have hnon : 0 ≤ ∑ g, ∑ h, w.weight g * w.weight h * rawEnergy d (P g).weight (P h).weight := by
    apply Finset.sum_nonneg
    intro g _
    apply Finset.sum_nonneg
    intro h _
    rw [rawEnergy_eq_neg_quadratic _ _ _ hs]
    exact mul_nonneg (mul_nonneg (w.nonneg g) (w.nonneg h))
      (neg_nonneg.mpr (hc _ (by simp [Finset.sum_sub_distrib, (P g).total, (P h).total])))
  constructor
  · rw [hpair]
    dsimp only [a]
    ring
  · rw [hpair] at hnon
    dsimp only [a] at hnon
    nlinarith [mul_nonneg (Nat.cast_nonneg (α := ℝ) n) hnon]

end StatadicFormal.Association
