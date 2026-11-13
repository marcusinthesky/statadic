module

public import StatadicFormal.Estimation.Linear.OLS
public import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Residual certificates for least squares and identification of a partial slope. -/
@[expose] public section
noncomputable section
open Matrix StatadicFormal.Estimation.Linear
namespace StatadicFormal.Optimization

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Orthogonality is a global least-squares certificate, including singular designs. -/
@[blueprint "STATADIC-LSTSQ-001"
  (statement := /-- For $r=y-Xb$, suppose $X^Tr=0$. For every alternative
  $b+h$, the squared residual increases by exactly $\|Xh\|^2$.
  The coefficient is unique precisely when $Xh=0$ implies $h=0$. -/)
  (proof := /-- Expand the squared norm of $r-Xh$. Its mixed term vanishes
  because $r^TXh=h^TX^Tr=0$. A competing point attains equality precisely
  when its fitted-vector increment has zero squared norm. -/)]
theorem least_squares_gap (X : Matrix ι κ ℝ) (y : ι → ℝ) (b : κ → ℝ)
    (hn : Xᵀ *ᵥ (y - X *ᵥ b) = 0) :
    let loss := fun u => (y - X *ᵥ u) ⬝ᵥ (y - X *ᵥ u)
    (∀ h, loss (b + h) = loss b + (X *ᵥ h) ⬝ᵥ (X *ᵥ h)) ∧
      ((∀ h, loss (b + h) = loss b → h = 0) ↔ ∀ h, X *ᵥ h = 0 → h = 0) := by
  let r := y - X *ᵥ b
  have ident (h : κ → ℝ) :
      (y - X *ᵥ (b + h)) ⬝ᵥ (y - X *ᵥ (b + h)) = r ⬝ᵥ r + (X *ᵥ h) ⬝ᵥ (X *ᵥ h) := by
    have orth : r ⬝ᵥ (X *ᵥ h) = 0 := by
      rw [← Matrix.dotProduct_transpose_mulVec, show Xᵀ *ᵥ r = 0 from hn, dotProduct_zero]
    rw [Matrix.mulVec_add, ← sub_sub]
    change (r - X *ᵥ h) ⬝ᵥ (r - X *ᵥ h) = _
    simp [sub_dotProduct, dotProduct_sub, dotProduct_comm (X *ᵥ h) r, orth]
  refine ⟨ident, ?_⟩
  change (∀ h, _ = r ⬝ᵥ r → h = 0) ↔ _
  simp only [ident, add_eq_left, dotProduct_self_eq_zero]

/-- Frisch--Waugh--Lovell with explicit residual variation as the denominator. -/
@[blueprint "STATADIC-FWL-001"
  (statement := /-- Let $Z$ be a nuisance design with invertible Gram matrix,
  $z=x-Z\hat\gamma$ and $t=y-Z\hat\delta$ its least-squares residuals.
  If $e=y-Za-bx$ is orthogonal to $Z$ and $x$, and $z^Tz>0$, then
  $b=z^Tt/(z^Tz)$. -/)
  (proof := /-- The first residualization gives $Z^Tz=0$, hence
  $z^Tt=z^Ty$ and $z^Tz=z^Tx$. Orthogonality of $e$ to $x$ and to
  every nuisance column gives $z^Te=0$. Substitute $y=Za+bx+e$ and divide. -/)]
theorem partial_slope [DecidableEq κ] (Z : Matrix ι κ ℝ) (x y : ι → ℝ) (a : κ → ℝ) (b : ℝ)
    (hgram : IsUnit (Zᵀ * Z).det) :
    let z := x - Z *ᵥ olsCoefficients Z x
    let t := y - Z *ᵥ olsCoefficients Z y
    let e := y - (Z *ᵥ a + b • x)
    Zᵀ *ᵥ e = 0 → x ⬝ᵥ e = 0 → 0 < z ⬝ᵥ z → b = (z ⬝ᵥ t) / (z ⬝ᵥ z) := by
  let z := x - Z *ᵥ olsCoefficients Z x
  let e := y - (Z *ᵥ a + b • x)
  dsimp only
  intro he hx hden
  have hz : Zᵀ *ᵥ z = 0 := by
    simp [z, Matrix.mulVec_sub, Matrix.mulVec_mulVec, ols_normal_equations Z x hgram]
  have hn (v : κ → ℝ) : z ⬝ᵥ (Z *ᵥ v) = 0 := by
    rw [← Matrix.dotProduct_transpose_mulVec, hz, dotProduct_zero]
  have hze : z ⬝ᵥ e = 0 := by
    change (x - Z *ᵥ olsCoefficients Z x) ⬝ᵥ e = 0
    rw [sub_dotProduct, dotProduct_comm (Z *ᵥ _), ← Matrix.dotProduct_transpose_mulVec,
      show Zᵀ *ᵥ e = 0 from he, dotProduct_zero, hx, sub_zero]
  have hh : z ⬝ᵥ y = b * (z ⬝ᵥ x) := by
    change z ⬝ᵥ (y - (Z *ᵥ a + b • x)) = 0 at hze
    rw [dotProduct_sub, dotProduct_add, hn, zero_add, dotProduct_smul, smul_eq_mul] at hze
    exact sub_eq_zero.mp hze
  apply (eq_div_iff hden.ne').mpr
  change b * (z ⬝ᵥ (x - Z *ᵥ _)) = z ⬝ᵥ (y - Z *ᵥ _)
  simp [dotProduct_sub, hn, hh]

end StatadicFormal.Optimization
