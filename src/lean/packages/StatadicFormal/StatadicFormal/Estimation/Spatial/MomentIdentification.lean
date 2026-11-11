module

public import StatadicFormal.Estimation.Spatial.ErrorMoments
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Population identification for the spatial-error quadratic moments and a SARAR alias. -/
@[expose] public section
noncomputable section
open Matrix
namespace StatadicFormal.Estimation.Spatial

/-- A component of the quadratic error moment after subtracting its value at the true parameter. -/
def momentChange (a b u v : ℝ) : ℝ := a * (u - v) + b * (u ^ 2 - v ^ 2)

/-- The exact secant criterion distinguishes global identification from local curvature. -/
@[blueprint "STATADIC-SPATIAL-004"
  (statement := /-- Suppose the population error moments at $\lambda_0$ vanish.
  Write their change at $r$ as $a_j(r-\lambda_0)+b_j(r^2-\lambda_0^2)$,
  absorbing the sign convention into the coefficients. On an admissible
  set $S$ containing $\lambda_0$, the parameter is uniquely identified exactly
  when every other $r\in S$ has some $a_j+b_j(r+\lambda_0)\ne0$. -/)
  (proof := /-- Factor each moment change into
  $(r-\lambda_0)[a_j+b_j(r+\lambda_0)]$. At a distinct candidate the
  first factor is nonzero, so all moments vanish precisely when every second
  factor vanishes. -/)]
theorem spatial_error_identified_iff {ι : Type*} (a b : ι → ℝ) (v : ℝ) (S : Set ℝ)
    (hv : v ∈ S) :
    (∃! r : ℝ, r ∈ S ∧ ∀ j, momentChange (a j) (b j) r v = 0) ↔
      ∀ r ∈ S, r ≠ v → ∃ j, a j + b j * (r + v) ≠ 0 := by
  classical
  have hfac (j r) : momentChange (a j) (b j) r v = (r-v) * (a j + b j * (r+v)) := by
    unfold momentChange
    ring
  have htrue : v ∈ S ∧ ∀ j, momentChange (a j) (b j) v v = 0 := by simp [hv, momentChange]
  constructor
  · intro hu r hr hne
    by_contra! h
    have hz : ∀ j, momentChange (a j) (b j) r v = 0 := fun j => by rw [hfac, h j, mul_zero]
    exact hne (hu.unique ⟨hr,hz⟩ htrue)
  · intro h
    refine ⟨v,htrue,?_⟩
    rintro r ⟨hr, hz⟩
    by_contra hne
    obtain ⟨j, hj⟩ := h r hr hne
    exact (mul_ne_zero (sub_ne_zero.mpr hne) hj) ((hfac j r).symm.trans (hz j))

/-- A nonzero derivative at the fitted root can coexist with another interior root. -/
@[blueprint "STATADIC-SPATIAL-005"
  (statement := /-- On $(-1,1)$ the quadratic moment $m(r)=r^2-1/4$
  has two roots, $1/2$ and $-1/2$, although its derivative at $1/2$ is nonzero. -/)
  (proof := /-- Substitute the two candidates and differentiate $r^2-1/4$.
  Both roots are strictly interior, and the derivative at the positive one is one. -/)]
theorem two_admissible_error_roots :
    let v : ℝ := 1 / 2
    let u : ℝ := -1 / 2
    v ∈ Set.Ioo (-1 : ℝ) 1 ∧ u ∈ Set.Ioo (-1 : ℝ) 1 ∧ u ≠ v ∧
      momentChange 0 1 u v = 0 ∧ 0 + 2 * (1 : ℝ) * v = 1 := by
  norm_num [momentChange, Set.mem_Ioo]

/-- A shared graph can make lag and error filters observationally interchangeable. -/
@[blueprint "STATADIC-SPATIAL-006"
  (statement := /-- With the same spatial matrix $W$, let $A(r)=I-rW$.
  Then $A(r)A(s)=A(s)A(r)$. In a zero-mean SARAR system with the same spherical
  innovation law, swapping lag and error coefficients leaves the composed
  innovation filter and hence the implied covariance unchanged whenever the filters are invertible. -/)
  (proof := /-- Multiply out the filters. The coefficients of $W$ and $W^2$
  are $-(r+s)$ and $rs$, invariant to the swap. Inverting the identical
  composed filter gives the same covariance factor. -/)]
theorem common_graph_filter_alias {ι : Type*} [Fintype ι] [DecidableEq ι]
    (W : Matrix ι ι ℝ) (r s : ℝ) :
    let F := fun u : ℝ => (1 : Matrix ι ι ℝ) - u • W
    F r * F s = F s * F r ∧
      (F r * F s)⁻¹ * ((F r * F s)⁻¹)ᵀ = (F s * F r)⁻¹ * ((F s * F r)⁻¹)ᵀ := by
  dsimp only
  have h : (1 - r • W) * (1 - s • W) = (1 - s • W) * (1 - r • W) := by
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one, Matrix.smul_mul,
      Matrix.mul_smul, smul_sub, smul_smul, mul_comm s r]
    abel
  exact ⟨h, by rw [h]⟩

end StatadicFormal.Estimation.Spatial
