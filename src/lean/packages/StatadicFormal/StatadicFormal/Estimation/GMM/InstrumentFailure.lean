module

public import StatadicFormal.Estimation.GMM.AffineLimit
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! Population failures that sample rank and overidentification diagnostics cannot exclude. -/

@[expose] public section
noncomputable section
open Matrix Filter
open scoped Topology

namespace StatadicFormal.Estimation.GMM

/-- A misspecification in the derivative column space is invisible to fitted moment residuals. -/
@[blueprint "STATADIC-IV-002"
  (statement := /-- For $g(b)=a+Db$ with $D^TWD$ invertible, suppose at a
  structural value $b_0$ the moment violation equals $D h$ for some $h\ne0$.
  The weighted coefficient equals $b_0-h\ne b_0$, and every population
  moment vanishes there, even when there are more moments than coefficients. -/)
  (proof := /-- Substitute $g(b_0)=Dh$ into $\hat b-b_0=-Lg(b_0)$.
  Since $LD=I$, the error is $-h$. Subtract $Dh$ from $g(b_0)$ to see
  that the misspecified coefficient fits all moment equations. -/)]
theorem invisible_moment_violation {m p : Type*} [Fintype m] [Fintype p] [DecidableEq p]
    (D : Matrix m p ℝ) (W : Matrix m m ℝ) (a : m → ℝ) (b₀ h : p → ℝ)
    (hr : (Dᵀ * W * D).det ≠ 0) (hh : h ≠ 0) (hv : a + D *ᵥ b₀ = D *ᵥ h) :
    coefficient a D W = b₀ - h ∧ coefficient a D W ≠ b₀ ∧
      a + D *ᵥ coefficient a D W = 0 := by
  have he := (affine_identification D W a b₀ hr).1
  rw [hv, Matrix.mulVec_mulVec, response_mul_derivative _ _ hr, Matrix.one_mulVec] at he
  have hb : coefficient a D W = b₀ - h := by
    rw [sub_eq_iff_eq_add] at he
    simpa [sub_eq_add_neg, add_comm] using he
  refine ⟨hb, ?_, ?_⟩
  · simpa [hb] using hh
  · rw [hb, Matrix.mulVec_sub, ← add_sub_assoc, hv, sub_self]

/-- Nonzero scalar rank alone cannot bound coefficient sensitivity near weak identification. -/
@[blueprint "STATADIC-IV-003"
  (statement := /-- Let $g(b)=a+db$ with $d\ne0$. Perturbing $a$ by $e$
  changes its zero by $-e/d$. In particular, for any tolerance $t>0$ and
  requested coefficient change $M>0$, one can choose a nonzero $d$ and a
  perturbation $|e|\le t$ for which the change has magnitude at least $M$. -/)
  (proof := /-- Subtract the two scalar solutions. Take $e=t$ and $d=t/M$,
  which meets the rank condition and attains magnitude $M$. -/)]
theorem weak_rank_sensitivity (a d e : ℝ) :
    (-(a + e) / d) - (-a / d) = -e / d ∧
      ∀ t M : ℝ, 0 < t → 0 < M → ∃ d' e' : ℝ, d' ≠ 0 ∧ |e'| ≤ t ∧
        M ≤ |(-(a + e') / d') - (-a / d')| := by
  constructor
  · ring
  · intro t M ht hM
    refine ⟨t / M, t, div_ne_zero ht.ne' hM.ne', (abs_of_pos ht).le, ?_⟩
    have he : -(a + t) / (t / M) - -a / (t / M) = -M := by field_simp; ring
    rw [he, abs_neg, abs_of_pos hM]

end StatadicFormal.Estimation.GMM
