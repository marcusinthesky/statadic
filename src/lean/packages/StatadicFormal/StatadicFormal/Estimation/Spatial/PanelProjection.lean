module

public import Architect
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring

/-! When temporal pooling commutes with a period-specific spatial lag. -/
@[expose] public section
noncomputable section
open Matrix
open scoped BigOperators
namespace StatadicFormal.Estimation.Spatial

variable {τ ι : Type*} [Fintype τ] [Fintype ι]

/-- Apply the graph for each period to that period's own vector of outcomes. -/
def periodLag (W : τ → Matrix ι ι ℝ) (u : Matrix τ ι ℝ) : Matrix τ ι ℝ :=
  fun t => W t *ᵥ u t

/-- Exact condition under which time projection can be interchanged with spatial lagging. -/
@[blueprint "STATADIC-KKP-001"
  (statement := /-- Let $Q$ mix periods, $W_t$ act on units, and $L(u)_t=W_t u_t$.
  Then $Q L(u)=L(Qu)$ for every balanced panel $u$ if and only if
  $Q_{ts}\ne0$ implies $W_t=W_s$. Thus the full temporal mean commutes
  precisely for a common graph across periods. -/)
  (proof := /-- Expand both operators at period $t$ and unit $i$. Their terms
  are $Q_{ts}(W_s)_{ij}u_{sj}$ and $Q_{ts}(W_t)_{ij}u_{sj}$.
  The stated condition makes these terms equal. Conversely put a single unit
  impulse at $(s,j)$. Equality at $(t,i)$ forces equality of both matrix
  entries after cancelling the nonzero temporal coefficient. -/)]
theorem time_mixing_commutes_iff (Q : Matrix τ τ ℝ) (W : τ → Matrix ι ι ℝ) :
    (∀ u : Matrix τ ι ℝ, Q * periodLag W u = periodLag W (Q * u)) ↔
      ∀ t s, Q t s ≠ 0 → W t = W s := by
  classical
  constructor
  · intro h t s hq
    ext i j
    let u : Matrix τ ι ℝ := fun a b => if a = s then (if b = j then 1 else 0) else 0
    have hleft (a : τ) : periodLag W u a i = if a = s then W s i j else 0 := by
      by_cases ha : a = s <;> simp [periodLag, Matrix.mulVec, dotProduct, u, ha, mul_ite]
    have hright (b : ι) : (Q * u) t b = if b = j then Q t s else 0 := by
      change (∑ a, Q t a * u a b) = _
      by_cases hb : b = j <;> simp [u, hb, mul_ite]
    have he := congrFun (congrFun (h u) t) i
    change (∑ a, Q t a * periodLag W u a i) = ∑ b, W t i b * (Q * u) t b at he
    simp only [hleft, hright, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ,
      ite_true] at he
    exact mul_left_cancel₀ hq (he.trans (mul_comm _ _)).symm
  · intro h u
    ext t i
    simp only [Matrix.mul_apply, periodLag, Matrix.mulVec, dotProduct, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro s _
    by_cases hq : Q t s = 0
    · simp [hq]
    · rw [h t s hq]
      ring

end StatadicFormal.Estimation.Spatial
