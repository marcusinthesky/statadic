module

public import Architect
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Sample volatility, constant series, and scale in the ex-post Sharpe statistic. -/
@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Estimation.Sharpe

variable {I : Type*} [Fintype I]

/-- Per-observation arithmetic mean. -/
def sampleMean (r : I → ℝ) : ℝ := (∑ i, r i) / Fintype.card I

/-- Sample standard deviation with the divisor n-1. -/
def sampleSD (r : I → ℝ) : ℝ :=
  Real.sqrt ((∑ i, (r i - sampleMean r) ^ 2) / ((Fintype.card I : ℝ) - 1))

/-- Positive rescaling preserves the sample ratio, and zero volatility
is equivalent to every row equalling the mean for at least two observations. -/
@[blueprint "STATADIC-SHARPE-001"
  (statement := /-- Let $n\ge2$, $\bar r=n^{-1}\sum_i r_i$ and
  $s^2=(n-1)^{-1}\sum_i(r_i-\bar r)^2$. Then $s=0$ exactly when
  every observation equals $\bar r$. For $c>0$,
  $s(cr)=c\,s(r)$ and the ratio $\overline{cr}/s(cr)=\bar r/s(r)$
  whenever the series has positive volatility. -/)
  (proof := /-- The divisor $n-1$ is positive. A sum of squares vanishes
  precisely when every deviation vanishes. Rescaling multiplies the
  deviations by $c$, hence the variance by $c^2$ and the standard deviation
  by $c$. Cancel the common factor in the ratio. -/)]
theorem sample_sharpe_scale (r : I → ℝ) (hn : 1 < Fintype.card I) (c : ℝ) (hc : 0 < c) :
    (sampleSD r = 0 ↔ ∀ i, r i = sampleMean r) ∧
    sampleSD (fun i => c * r i) = c * sampleSD r ∧
    sampleMean (fun i => c * r i) / sampleSD (fun i => c * r i) = sampleMean r / sampleSD r := by
  have hden : 0 < (Fintype.card I : ℝ) - 1 := by exact sub_pos.mpr (by exact_mod_cast hn)
  have hnneg : 0 ≤ ∑ i, (r i - sampleMean r) ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hm : sampleMean (fun i => c * r i) = c * sampleMean r := by
    simp only [sampleMean, ← Finset.mul_sum]; ring
  have hsd : sampleSD (fun i => c * r i) = c * sampleSD r := by
    unfold sampleSD
    rw [hm]
    simp_rw [← mul_sub, mul_pow, ← Finset.mul_sum, mul_div_assoc]
    rw [Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  refine ⟨?_, hsd, ?_⟩
  · rw [sampleSD, Real.sqrt_eq_zero (div_nonneg hnneg hden.le)]
    constructor
    · intro h i
      have hz := (div_eq_zero_iff.mp h).resolve_right hden.ne'
      have he := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg (r j - sampleMean r))).mp hz
      exact sub_eq_zero.mp (sq_eq_zero_iff.mp (he i (Finset.mem_univ _)))
    · intro h
      simp [h]
  · rw [hm, hsd, mul_div_mul_left _ _ hc.ne']

end StatadicFormal.Estimation.Sharpe
