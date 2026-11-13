module

public import Architect
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Directional risk of the constant-correlation shrinkage target. -/
@[expose] public section
noncomputable section
open scoped BigOperators

namespace StatadicFormal.Estimation.Covariance

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Entry of the target, including its unshrunk marginal variances. -/
def constantCorrelation (σ : ι → ℝ) (r : ℝ) (i j : ι) : ℝ :=
  σ i * σ j * (r + if i = j then 1 - r else 0)

/-- A risk projection before mixing the estimated covariance and the target. -/
def risk (S : ι → ι → ℝ) (x : ι → ℝ) : ℝ := ∑ i, ∑ j, S i j * (x i * x j)

private theorem target_risk (σ x : ι → ℝ) (r : ℝ) :
    risk (constantCorrelation σ r) x =
      (1 - r) * ∑ i, (σ i * x i) ^ 2 + r * (∑ i, σ i * x i) ^ 2 := by
  classical
  unfold risk constantCorrelation
  have h (i j : ι) :
      σ i * σ j * (r + if i = j then 1 - r else 0) * (x i * x j) =
        (if i = j then (1-r) * (σ i * x i)^2 else 0) +
          r * (σ i * x i) * (σ j * x j) := by
    split_ifs with he
    · subst j
      ring
    · ring
  simp_rw [h, Finset.sum_add_distrib]
  simp [← Finset.mul_sum, ← Finset.sum_mul, pow_two]
  exact mul_assoc _ _ _

/-- Two scalar bounds on the mean correlation suffice for a PSD shrinkage target. -/
@[blueprint "STATADIC-SHRINKAGE-001"
  (statement := /-- For $p>0$, let $F_{ii}=\sigma_i^2$ and
  $F_{ij}=r\sigma_i\sigma_j$ for $i\ne j$. Suppose
  $1-r\ge0$ and $1+(p-1)r\ge0$. For a nonnegative sample quadratic
  form $S$ and $0\le a\le1$, every contrast of $aF+(1-a)S$
  has nonnegative variance. -/)
  (proof := /-- Set $z_i=\sigma_i x_i$, $u=\sum z_i^2$ and
  $v=(\sum z_i)^2/p$. The target risk expands to
  $(1-r)(u-v)+(1+(p-1)r)v$. Cauchy--Schwarz gives $0\le v\le u$.
  Both terms and the convex combination with sample risk are nonnegative. -/)]
theorem shrinkage_risk_nonneg [Nonempty ι] (S : ι → ι → ℝ) (σ x : ι → ℝ)
    (r a : ℝ) (hS : 0 ≤ risk S x) (ha : 0 ≤ a) (ha' : a ≤ 1)
    (hr : r ≤ 1) (hr' : 0 ≤ 1 + (Fintype.card ι - 1 : ℝ) * r) :
    0 ≤ risk (fun i j => a * constantCorrelation σ r i j + (1-a) * S i j) x := by
  let p : ℝ := Fintype.card ι
  have hp : 0 < p := by
    dsimp [p]
    exact_mod_cast (Fintype.card_pos (α := ι))
  let u := ∑ i, (σ i * x i) ^ 2
  let v := (∑ i, σ i * x i) ^ 2 / p
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ : ι => (1 : ℝ))
    (fun i => σ i * x i)
  have hvu : v ≤ u := (div_le_iff₀ hp).mpr (by simpa [u, p, mul_comm] using hcs)
  have hv : 0 ≤ v := div_nonneg (sq_nonneg _) hp.le
  have hm : risk (constantCorrelation σ r) x = (1-r)*(u-v)+(1+(p-1)*r)*v := by
    rw [target_risk]
    dsimp [u, v]
    field_simp
    ring
  have htarget : 0 ≤ risk (constantCorrelation σ r) x := by
    rw [hm]
    exact add_nonneg (mul_nonneg (sub_nonneg.mpr hr) (sub_nonneg.mpr hvu)) (mul_nonneg hr' hv)
  have hmix : risk (fun i j => a * constantCorrelation σ r i j + (1-a) * S i j) x =
      a * risk (constantCorrelation σ r) x + (1-a) * risk S x := by
    simp [risk, add_mul, mul_assoc, Finset.sum_add_distrib, Finset.mul_sum]
  rw [hmix]
  exact add_nonneg (mul_nonneg ha htarget) (mul_nonneg (sub_nonneg.mpr ha') hS)

end StatadicFormal.Estimation.Covariance
