module

public import Architect
public import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Tactic.FieldSimp

/-! Domain and separation of the Euclidean distance pulled back along normalization. -/

@[expose] public section
namespace StatadicFormal.Geometry
variable {H : Type*} [NormedAddCommGroup H] [NormedSpace ℝ H]

/-- Unit direction of a nonzero vector. The geometric contract excludes the zero input. -/
noncomputable def direction (x : H) : H := ‖x‖⁻¹ • x

/-- The normalized chord inherits triangle inequality and separates precisely the positive rays. -/
@[blueprint "STATADIC-GROUND-001"
  (statement := /-- On nonzero vectors put $u(x)=x/\|x\|$ and
  $d(x,y)=\|u(x)-u(y)\|$. Then $d(x,z)\le d(x,y)+d(y,z)$, and
  $d(x,y)=0$ exactly when $y=cx$ for some $c>0$. -/)
  (proof := /-- Apply the ambient triangle inequality to the three directions.
  Zero distance gives equality of directions. Multiply by $\|y\|$ to obtain
  $y=(\|y\|/\|x\|)x$. Conversely, for $c>0$ the factor $c$ cancels
  against the same factor in $\|cx\|=c\|x\|$. -/)]
theorem normalized_chord_laws (x y z : H) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    ‖direction x - direction z‖ ≤ ‖direction x - direction y‖ + ‖direction y - direction z‖ ∧
      (‖direction x - direction y‖ = 0 ↔ ∃ c : ℝ, 0 < c ∧ y = c • x) ∧ ‖direction z‖ = 1 := by
  refine ⟨norm_sub_le_norm_sub_add_norm_sub _ _ _, ⟨?_, ?_⟩, ?_⟩
  · intro h
    have he := sub_eq_zero.mp (norm_eq_zero.mp h)
    refine ⟨‖y‖ / ‖x‖, div_pos (norm_pos_iff.mpr hy) (norm_pos_iff.mpr hx), ?_⟩
    have hh := congrArg (fun v : H ↦ ‖y‖ • v) he.symm
    simpa [direction, smul_smul, div_eq_mul_inv, norm_ne_zero_iff.mpr hy] using hh
  · rintro ⟨c, hc, rfl⟩
    simp [direction, norm_smul, Real.norm_of_nonneg hc.le, smul_smul, hc.ne']
  · simp [direction, norm_smul, norm_ne_zero_iff.mpr hz]

end StatadicFormal.Geometry
