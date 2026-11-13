module

public import StatadicFormal.Estimation.Linear.OLS
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-! Capped-simplex certificates for a symmetric convex quadratic. -/
@[expose] public section
noncomputable section
open Matrix
namespace StatadicFormal.Optimization

variable {ι : Type*} [Fintype ι]

/-- Half-quadratic plus linear term used by the native affine QP. -/
def quadraticObjective (Q : Matrix ι ι ℝ) (c x : ι → ℝ) : ℝ :=
  (x ⬝ᵥ (Q *ᵥ x)) / 2 + c ⬝ᵥ x

private theorem quadratic_step (Q : Matrix ι ι ℝ) (hs : Qᵀ = Q) (c x h : ι → ℝ) :
    quadraticObjective Q c (x + h) - quadraticObjective Q c x =
      (Q *ᵥ x + c) ⬝ᵥ h + (h ⬝ᵥ (Q *ᵥ h)) / 2 := by
  have hcross := Matrix.dotProduct_transpose_mulVec Q x h
  rw [hs] at hcross
  simp only [quadraticObjective, Matrix.mulVec_add, dotProduct_add, add_dotProduct, hcross,
    dotProduct_comm (Q *ᵥ x) h]
  ring

/-- A cap on every coordinate permits a probability vector exactly when the
total available capacity reaches one. The uniform vector witnesses sufficiency. -/
@[blueprint "STATADIC-SIMPLEX-003"
  (statement := /-- For $n>0$, there exists a nonnegative vector of total mass
  one with each entry at most $u$ if and only if $nu\ge1$. -/)
  (proof := /-- Sum the $n$ upper bounds to obtain necessity.
  Conversely $1/n\le u$ makes the uniform vector feasible. -/)]
theorem capped_feasible_iff [Nonempty ι] (cap : ℝ) :
    (∃ x : ι → ℝ, (∑ i, x i) = 1 ∧ ∀ i, 0 ≤ x i ∧ x i ≤ cap) ↔
      1 ≤ Fintype.card ι * cap := by
  have hn : 0 < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr Fintype.card_pos
  constructor
  · rintro ⟨x, hx, hcap⟩
    simpa [hx, nsmul_eq_mul] using Finset.sum_le_sum (s := Finset.univ)
      (fun i _ => (hcap i).2)
  · intro hc
    refine ⟨fun _ => 1 / (Fintype.card ι : ℝ), ?_, fun _ =>
      ⟨div_nonneg zero_le_one hn.le, (div_le_iff₀ hn).mpr (by simpa [mul_comm] using hc)⟩⟩
    simp [nsmul_eq_mul, hn.ne']

/-- Exact primal, dual and complementary-slackness conditions certify a global solution. -/
@[blueprint "STATADIC-SIMPLEX-001"
  (statement := /-- Let $f(x)=x^TQx/2+c^Tx$ with $Q$ symmetric and positive semidefinite on zero-sum directions.
  Let $x$ have sum one and lie in $[0,u]^n$. Suppose lower and upper multipliers
  $l,r\ge0$ satisfy $l_ix_i=r_i(u-x_i)=0$ and
  $(Qx+c)_i-l_i+r_i=\nu$ for every coordinate. For every feasible $y$,
  $f(x)\le f(y)$. -/)
  (proof := /-- Write $h=y-x$. Its coordinates sum to zero, so the common multiplier
  drops out of $\nabla f(x)^Th$. Complementarity turns the remaining terms into
  $\sum_i l_iy_i+r_i(u-y_i)\ge0$. Adding $h^TQh/2\ge0$ gives the objective gap. -/)]
theorem capped_simplex_kkt (Q : Matrix ι ι ℝ) (hs : Qᵀ = Q)
    (hpsd : ∀ h, (∑ i, h i) = 0 → 0 ≤ h ⬝ᵥ (Q *ᵥ h)) (c x l r : ι → ℝ) (cap ν : ℝ)
    (hx : ∑ i, x i = 1) (hbox : ∀ i, 0 ≤ x i ∧ x i ≤ cap)
    (hm : ∀ i, 0 ≤ l i ∧ 0 ≤ r i)
    (hc : ∀ i, l i * x i = 0 ∧ r i * (cap - x i) = 0)
    (hs' : ∀ i, (Q *ᵥ x + c) i - l i + r i = ν) :
    (∀ i, 0 ≤ x i ∧ x i ≤ cap) ∧ ∀ y : ι → ℝ, (∑ i, y i) = 1 →
      (∀ i, 0 ≤ y i ∧ y i ≤ cap) → quadraticObjective Q c x ≤ quadraticObjective Q c y := by
  refine ⟨hbox, ?_⟩
  intro y hy hybox
  have hzero : ∑ i, ν * (y i - x i) = 0 := by
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, hy, hx, sub_self, mul_zero]
  have hlin : 0 ≤ (Q *ᵥ x + c) ⬝ᵥ (y - x) := by
    rw [← hzero]
    exact Finset.sum_le_sum fun i _ => by
      change ν * (y i - x i) ≤ (Q *ᵥ x + c) i * (y i - x i)
      rw [show (Q *ᵥ x + c) i = ν + l i - r i by linarith [hs' i]]
      nlinarith [(hc i).1, (hc i).2, mul_nonneg (hm i).1 (hybox i).1,
        mul_nonneg (hm i).2 (sub_nonneg.mpr (hybox i).2)]
  have he := quadratic_step Q hs c x (y - x)
  rw [add_sub_cancel] at he
  have ht : ∑ i, (y - x) i = 0 := by simp [Pi.sub_apply, Finset.sum_sub_distrib, hy, hx]
  linarith [hpsd (y - x) ht]

/-- A bound on the linearized subproblem gives a quantitative objective certificate. -/
@[blueprint "STATADIC-SIMPLEX-002"
  (statement := /-- Suppose $x$ is feasible for a quadratic with $Q$ symmetric and positive
  semidefinite on zero-sum directions. Let $L$ be a lower bound on $g^Ty$ over its feasible
  set, where $g=Qx+c$. For every feasible $y$, $f(x)-f(y)\le g^Tx-L$.
  This applies to the capped-simplex linear minimization gap at the returned iterate. -/)
  (proof := /-- Expand $f(y)$ about $x$ and discard the nonnegative quadratic
  remainder. Substitute $g^Ty\ge L$ into the resulting linear bound. -/)]
theorem quadratic_gap_bound (Q : Matrix ι ι ℝ) (hs : Qᵀ = Q)
    (hpsd : ∀ h, (∑ i, h i) = 0 → 0 ≤ h ⬝ᵥ (Q *ᵥ h)) (c x y : ι → ℝ) (L : ℝ)
    (hxy : (∑ i, y i) = ∑ i, x i) (hl : L ≤ (Q *ᵥ x + c) ⬝ᵥ y) :
    quadraticObjective Q c x - quadraticObjective Q c y ≤ (Q *ᵥ x + c) ⬝ᵥ x - L := by
  have he := quadratic_step Q hs c x (y - x)
  rw [add_sub_cancel, dotProduct_sub] at he
  have ht : ∑ i, (y - x) i = 0 := by simp [Pi.sub_apply, Finset.sum_sub_distrib, hxy]
  linarith [hpsd (y - x) ht]

end StatadicFormal.Optimization
