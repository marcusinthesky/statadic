module

public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Architect
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Finite algebra behind three OLS residual diagnostics: the Durbin--Watson
range, the zero set of the Jarque--Bera statistic and deleted-row
studentization. None of these results calibrates a test or proves a reference
distribution. -/
@[expose] public section
noncomputable section
open Finset
namespace StatadicFormal.Diagnostics

/-- Successive squared differences are at most four times the residual sum of squares. -/
@[blueprint "STATADIC-DW-001"
  (statement := /-- For residuals $e_0,\ldots,e_m$ with
  $S=\sum_{i=0}^{m}e_i^2>0$, the Durbin--Watson ratio
  $\sum_{i=0}^{m-1}(e_{i+1}-e_i)^2/S$ lies in $[0,4]$. -/)
  (proof := /-- Each term satisfies $(a-b)^2\le 2a^2+2b^2$ because their
  difference is $(a+b)^2$. Summing, the shifted and unshifted squares are each
  partial sums of $S$, so the numerator is at most $4S$. -/)]
theorem durbin_watson_bounds (e : ℕ → ℝ) (m : ℕ)
    (hs : 0 < ∑ i ∈ range (m + 1), e i ^ 2) :
    0 ≤ (∑ i ∈ range m, (e (i + 1) - e i) ^ 2) / ∑ i ∈ range (m + 1), e i ^ 2 ∧
      (∑ i ∈ range m, (e (i + 1) - e i) ^ 2) / ∑ i ∈ range (m + 1), e i ^ 2 ≤ 4 := by
  have h1 : ∑ i ∈ range m, (e (i + 1) - e i) ^ 2 ≤
      ∑ i ∈ range m, (2 * e (i + 1) ^ 2 + 2 * e i ^ 2) :=
    Finset.sum_le_sum fun i _ => by nlinarith [sq_nonneg (e (i + 1) + e i)]
  have h2 : ∑ i ∈ range m, e (i + 1) ^ 2 ≤ ∑ i ∈ range (m + 1), e i ^ 2 := by
    rw [Finset.sum_range_succ']
    linarith [sq_nonneg (e 0)]
  have h3 : ∑ i ∈ range m, e i ^ 2 ≤ ∑ i ∈ range (m + 1), e i ^ 2 := by
    rw [Finset.sum_range_succ]
    linarith [sq_nonneg (e m)]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at h1
  constructor
  · exact div_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _) hs.le
  · rw [div_le_iff₀ hs]
    linarith

/-- The Jarque--Bera statistic from a sample size, skewness and kurtosis. -/
def jarqueBera (n skew kurt : ℝ) : ℝ := n / 6 * (skew ^ 2 + (kurt - 3) ^ 2 / 4)

/-- The statistic is nonnegative and vanishes only at Gaussian moments. -/
@[blueprint "STATADIC-JB-001"
  (statement := /-- For $n>0$, $JB=\tfrac n6\bigl(S^2+(K-3)^2/4\bigr)\ge0$,
  with equality exactly when the skewness $S$ is $0$ and the kurtosis $K$
  is $3$. -/)
  (proof := /-- The factor $n/6$ is positive and the bracket is a sum of
  nonnegative squares, so the product is zero exactly when both squares
  are zero. -/)]
theorem jarque_bera_zero_iff (n skew kurt : ℝ) (hn : 0 < n) :
    0 ≤ jarqueBera n skew kurt ∧ (jarqueBera n skew kurt = 0 ↔ skew = 0 ∧ kurt = 3) := by
  unfold jarqueBera
  have hpos : 0 < n / 6 := by positivity
  have hq : 0 ≤ skew ^ 2 + (kurt - 3) ^ 2 / 4 := by positivity
  refine ⟨mul_nonneg hpos.le hq, ⟨fun h => ?_, ?_⟩⟩
  · have hzero : skew ^ 2 + (kurt - 3) ^ 2 / 4 = 0 :=
      (mul_eq_zero.mp h).resolve_left hpos.ne'
    have hs : skew ^ 2 = 0 := by nlinarith [sq_nonneg skew, sq_nonneg (kurt - 3)]
    have hk : (kurt - 3) ^ 2 = 0 := by nlinarith [sq_nonneg skew, sq_nonneg (kurt - 3)]
    refine ⟨pow_eq_zero_iff two_ne_zero |>.mp hs, ?_⟩
    have := pow_eq_zero_iff two_ne_zero |>.mp hk
    linarith
  · rintro ⟨rfl, rfl⟩
    ring

/-- Deleted-row studentization is a function of internal studentization. -/
@[blueprint "STATADIC-INFLUENCE-001"
  (statement := /-- Let $m>1$ be the residual degrees of freedom, $s^2>0$ the
  residual variance, $h<1$ a leverage and $e$ the residual. Write
  $r^2=e^2/(s^2(1-h))$ and $d=(ms^2-e^2/(1-h))/(m-1)$ for the deleted-row
  variance. If $d>0$, then $e^2/(d(1-h))=r^2(m-1)/(m-r^2)$. -/)
  (proof := /-- Multiply the positivity of $d$ by $m-1$ to obtain
  $ms^2(1-h)>e^2$, hence $m>r^2$. Then $d=s^2(m-r^2)/(m-1)$, and clearing the
  positive denominators reduces both sides to the same rational expression. -/)]
theorem external_studentized_sq (m s2 h e : ℝ) (hm : 1 < m) (hs : 0 < s2) (hh : h < 1)
    (hd : 0 < (m * s2 - e ^ 2 / (1 - h)) / (m - 1)) :
    e ^ 2 / ((m * s2 - e ^ 2 / (1 - h)) / (m - 1) * (1 - h)) =
      e ^ 2 / (s2 * (1 - h)) * (m - 1) / (m - e ^ 2 / (s2 * (1 - h))) := by
  have hc : 0 < 1 - h := by linarith
  have hm1 : 0 < m - 1 := by linarith
  have hnum : 0 < m * s2 - e ^ 2 / (1 - h) := by
    have := mul_pos hd hm1
    rwa [div_mul_cancel₀ _ hm1.ne'] at this
  have hcross : e ^ 2 < m * s2 * (1 - h) := by
    rw [sub_pos, div_lt_iff₀ hc] at hnum
    linarith
  have hgap : 0 < m - e ^ 2 / (s2 * (1 - h)) := by
    rw [sub_pos, div_lt_iff₀ (mul_pos hs hc)]
    linarith
  field_simp
