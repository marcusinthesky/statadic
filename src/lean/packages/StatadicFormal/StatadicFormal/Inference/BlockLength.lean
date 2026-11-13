module

public import Architect
public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Interior continuous block length for a leading mean-squared-error criterion. -/
@[expose] public section
noncomputable section
namespace StatadicFormal.Inference

/-- Leading squared bias and variance term in the block-length criterion. -/
def blockRisk (G D n b : ℝ) : ℝ := G^2 / b^2 + D * b / n

/-- Global optimality and uniqueness from the cube-root balance. -/
@[blueprint "STATADIC-BLOCKLENGTH-001"
  (statement := /-- Let $n,D,b_0>0$ and $D b_0^3=2nG^2$. Then
  $b_0$ uniquely minimizes $R(b)=G^2/b^2+Db/n$ over $b>0$. -/)
  (proof := /-- Substitute the balance condition and factor the risk gap as
  $R(b)-R(b_0)=(D/n)(b-b_0)^2(2b+b_0)/(2b^2)$.
  The factor multiplying the square is strictly positive on the stated domain. -/)]
theorem block_risk_minimum (G D n b₀ b : ℝ) (hD : 0 < D) (hn : 0 < n)
    (h₀ : 0 < b₀) (hb : 0 < b) (hbal : D * b₀^3 = 2 * n * G^2) :
    blockRisk G D n b₀ ≤ blockRisk G D n b ∧
      (blockRisk G D n b = blockRisk G D n b₀ ↔ b = b₀) := by
  have hG : G^2 = D * b₀^3 / (2*n) := (eq_div_iff (by positivity)).mpr (by linarith)
  have hgap : blockRisk G D n b - blockRisk G D n b₀ =
      (D * (2*b+b₀) / (2*n*b^2)) * (b-b₀)^2 := by
    unfold blockRisk
    rw [hG]
    field_simp
    ring
  have hfactor : 0 < D * (2*b+b₀) / (2*n*b^2) := by positivity
  refine ⟨sub_nonneg.mp (hgap ▸ mul_nonneg hfactor.le (sq_nonneg _)), ?_⟩
  constructor
  · intro he
    rw [he, sub_self, eq_comm, mul_eq_zero] at hgap
    rcases hgap with h | h
    · exact (hfactor.ne' h).elim
    · exact sub_eq_zero.mp (sq_eq_zero_iff.mp h)
  · rintro rfl
    rfl

end StatadicFormal.Inference
