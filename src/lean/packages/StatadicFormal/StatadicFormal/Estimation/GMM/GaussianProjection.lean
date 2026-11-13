module

public import Mathlib.Probability.Distributions.Gaussian.Multivariate
public import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional
public import Architect

/-! The efficient overidentification reference as the squared length of a standard Gaussian
projected onto the unestimated directions. -/
@[expose] public section
noncomputable section
open MeasureTheory ProbabilityTheory Module
open scoped InnerProductSpace
namespace StatadicFormal.Estimation.GMM

/-- Chi-square reference defined by a sum of independent squared standard normals. -/
def chiSquareLaw (q : ℕ) : Measure ℝ :=
  (Measure.pi fun _ : Fin q => gaussianReal 0 1).map (fun z => ∑ j, (z j)^2)

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

private theorem project_standard_gaussian (K : Submodule ℝ E) :
    (stdGaussian E).map K.orthogonalProjectionOnto = stdGaussian K := by
  apply Measure.ext_of_charFun (E := K)
  ext t
  calc
    _ = charFun (stdGaussian E) (t : E) := by
      rw [charFun_apply, integral_map (by fun_prop) (by fun_prop), charFun_apply]
      simp only [Submodule.inner_orthogonalProjectionOnto_eq_of_mem_right]
    _ = _ := by rw [charFun_stdGaussian, charFun_stdGaussian]; rfl

/-- The residual degrees of freedom are the codimension of the estimated tangent space. -/
@[blueprint "STATADIC-JTEST-002"
  (statement := /-- In a finite-dimensional real inner product space $E$, let
  $K$ be the fitted tangent subspace. For a standard Gaussian $G$, the law of
  $\|\operatorname{proj}_{K^\perp}G\|^2$ is the law of a sum of
  $\dim E-\dim K$ independent squared standard normals. This is the
  chi-square reference for an efficiently whitened residual projection. -/)
  (proof := /-- The Gaussian characteristic function at a vector in
  $K^\perp$ is unchanged by orthogonal projection, so the projected law is
  standard Gaussian on that subspace. Express it in any orthonormal basis:
  the coordinates are independent standard normals and the squared norm is
  their sum of squares. Orthogonal dimensions add to the ambient dimension. -/)]
theorem overid_gaussian_reference (K : Submodule ℝ E) :
    (stdGaussian E).map (fun z => ‖Kᗮ.orthogonalProjectionOnto z‖ ^ 2) =
      chiSquareLaw (finrank ℝ E - finrank ℝ K) := by
  let H := Kᗮ
  have hd : finrank ℝ E - finrank ℝ K = finrank ℝ H := by
    have := K.finrank_add_finrank_orthogonal
    dsimp [H]
    omega
  rw [hd]
  let U := (stdOrthonormalBasis ℝ H).repr
  have hU := stdGaussian_map U
  calc
    _ = ((stdGaussian E).map H.orthogonalProjectionOnto).map (fun v : H => ‖v‖ ^ 2) := by
      rw [Measure.map_map (by fun_prop) (by fun_prop)]
      rfl
    _ = (stdGaussian H).map (fun v : H => ‖v‖ ^ 2) := by rw [project_standard_gaussian H]
    _ = ((stdGaussian H).map U).map (fun y => ‖y‖ ^ 2) := by
      rw [Measure.map_map (by fun_prop) (by fun_prop)]
      simp [Function.comp_def]
    _ = _ := by
      rw [hU, ← map_pi_eq_stdGaussian, Measure.map_map (by fun_prop) (by fun_prop)]
      congr 1
      funext z
      simp [EuclideanSpace.real_norm_sq_eq]

end StatadicFormal.Estimation.GMM
