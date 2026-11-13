module

public import Architect
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Ring

/-!
# One-way weighted panel absorption

The finite index ranges over the observations with one value of the selected
entity or time label. Each occupied label can have a different number of rows.
-/

@[expose] public section
noncomputable section

open scoped BigOperators

namespace StatadicFormal.Operators.Panel

variable {ι : Type*} [Fintype ι]

/-- Precision-weighted mean for one label, used when its total weight is positive. -/
def groupMean (w x : ι → ℝ) : ℝ := (∑ i, w i * x i) / ∑ i, w i

/-- Subtract the group mean from a single response or design column. -/
def within (w x : ι → ℝ) (i : ι) : ℝ := x i - groupMean w x

/-- The residual is orthogonal to the group indicator in the weighted inner product. -/
theorem sum_weighted_within (w x : ι → ℝ) (hw : 0 < ∑ i, w i) :
    ∑ i, w i * within w x i = 0 := by
  simp only [within, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, groupMean]
  field_simp [hw.ne']
  ring

/-- A second exact one-way sweep has no effect. -/
theorem within_idempotent (w x : ι → ℝ) (hw : 0 < ∑ i, w i) :
    within w (within w x) = within w x := by
  funext i
  change within w x i - (∑ j, w j * within w x j) / (∑ j, w j) = within w x i
  simp [sum_weighted_within w x hw]

/-- The weighted group mean uniquely minimizes the within-group residual sum of squares. -/
@[blueprint "STATADIC-PANEL-001"
  (statement := /-- For one occupied group let $M=\sum_i w_i>0$ and
  $m=\sum_i w_i x_i/M$. For every constant $a$,
  $\sum_i w_i(x_i-a)^2=\sum_i w_i(x_i-m)^2+M(m-a)^2$.
  Consequently the group mean is the unique least-squares intercept. -/)
  (proof := /-- First, $\sum_i w_i(x_i-m)=\sum_i w_i x_i-Mm=0$.
  Substitute $x_i-a=(x_i-m)+(m-a)$ and expand the square.
  The cross term is $2(m-a)\sum_i w_i(x_i-m)=0$.
  The remaining excess $M(m-a)^2$ vanishes only at $a=m$. -/)]
theorem within_least_squares (w x : ι → ℝ) (hw : 0 < ∑ i, w i) (a : ℝ) :
    (∑ i, w i * (x i - a) ^ 2) =
      (∑ i, w i * (within w x i) ^ 2) + (∑ i, w i) * (groupMean w x - a) ^ 2 ∧
    ((∑ i, w i * (x i - a) ^ 2) ≤ (∑ i, w i * (within w x i) ^ 2) ↔
      a = groupMean w x) := by
  have h : (∑ i, w i * (x i - a) ^ 2) =
      (∑ i, w i * (within w x i) ^ 2) + (∑ i, w i) * (groupMean w x - a) ^ 2 := by
    calc
      _ = ∑ i, (w i * (within w x i) ^ 2 + w i * within w x i * (2 * (groupMean w x - a))
          + w i * (groupMean w x - a) ^ 2) := by
        apply Finset.sum_congr rfl
        intro i _
        unfold within
        ring
      _ = _ := by
        simp only [Finset.sum_add_distrib, ← Finset.sum_mul, sum_weighted_within w x hw,
          zero_mul, add_zero]
  refine ⟨h, ?_⟩
  rw [h, add_le_iff_nonpos_right]
  constructor
  · intro hs
    by_contra hne
    exact (not_lt_of_ge hs) (mul_pos hw (sq_pos_of_ne_zero (sub_ne_zero.mpr (Ne.symm hne))))
  · rintro rfl
    simp

end StatadicFormal.Operators.Panel
