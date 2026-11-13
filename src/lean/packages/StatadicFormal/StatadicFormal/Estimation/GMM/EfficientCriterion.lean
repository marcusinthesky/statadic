module

public import StatadicFormal.Estimation.GMM.GaussianProjection
public import StatadicFormal.Estimation.GMM.Overidentification

/-! Whiten the actual GMM criterion to identify its reference law. -/
@[expose] public section
noncomputable section
open Matrix MeasureTheory ProbabilityTheory WithLp Module
open scoped InnerProductSpace
namespace StatadicFormal.Estimation.GMM

variable {m p : Type*} [Fintype m] [Fintype p] [DecidableEq m] [DecidableEq p]

private theorem whitened_projection (D : Matrix m p ℝ) (C : Matrix m m ℝ) (z : m → ℝ)
    (hr : (Dᵀ * (Cᵀ * C) * D).det ≠ 0) :
    let K := LinearMap.range (C * D).toEuclideanLin
    Kᗮ.starProjection (toLp 2 (C *ᵥ z)) =
      toLp 2 (C *ᵥ (residualMap D (Cᵀ * C) *ᵥ z)) := by
  let W := Cᵀ * C
  let B := C * D
  let P := residualMap D W
  let K := LinearMap.range B.toEuclideanLin
  let r := C *ᵥ (P *ᵥ z)
  let u := C *ᵥ z
  change Kᗮ.starProjection (toLp 2 u) = toLp 2 r
  have hBP : Bᵀ * C * P = 0 := by
    have hrel : Bᵀ * C = Dᵀ * W := by rw [show B = C * D from rfl, Matrix.transpose_mul,
      Matrix.mul_assoc]
    rw [hrel]
    change (Dᵀ * W) * (1 - D * ((Dᵀ * W * D)⁻¹ * (Dᵀ * W))) = 0
    rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc (Dᵀ * W) D,
      ← Matrix.mul_assoc (Dᵀ * W * D) _ _, Matrix.mul_nonsing_inv _ (isUnit_iff_ne_zero.mpr hr), Matrix.one_mul, sub_self]
  have hperp : toLp 2 r ∈ Kᗮ := by
    apply (K.mem_orthogonal _).2
    rintro _ ⟨v, rfl⟩
    change inner ℝ (toLp 2 (B *ᵥ ofLp v)) (toLp 2 r) = 0
    rw [EuclideanSpace.inner_toLp_toLp, star_trivial, ← Matrix.dotProduct_transpose_mulVec]
    change ofLp v ⬝ᵥ Bᵀ *ᵥ (C *ᵥ (P *ᵥ z)) = 0
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, hBP, Matrix.zero_mulVec, dotProduct_zero]
  apply Submodule.eq_starProjection_of_mem_orthogonal hperp
  apply K.le_orthogonal_orthogonal
  refine ⟨toLp 2 (response D W *ᵥ z), ?_⟩
  apply (WithLp.equiv 2 _).injective
  change (C * D) *ᵥ (response D W *ᵥ z) = u - r
  dsimp only [u, r, P, residualMap]
  rw [Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.mulVec_sub, sub_sub_cancel,
    ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]

/-- Efficient whitening turns the actual limit quadratic into a chi-square
with moment count minus parameter count degrees of freedom. -/
@[blueprint "STATADIC-JTEST-003"
  (statement := /-- Suppose a matrix $C$ whitens the limiting score law
  ($CZ$ is standard Gaussian), the actual limiting weight is $W=C^T C$,
  and $D^T W D$ is nonsingular. For $P=I-DL$, the law of
  $Z^T P^T W P Z$ is chi-square with $m-p$ degrees of freedom. -/)
  (proof := /-- The whitened residual $CPz$ is orthogonal to the columns of
  $CD$ by the weighted normal equations. The fitted component $Cz-CPz$
  lies in their span, so $CPz$ is the orthogonal residual projection of
  $Cz$. The weighted Gram gives a left inverse for $CD$, hence this span
  has dimension $p$. Apply the standard Gaussian projection law. -/)]
theorem efficient_criterion_reference (D : Matrix m p ℝ) (C : Matrix m m ℝ)
    (ν : Measure (m → ℝ)) (hr : (Dᵀ * (Cᵀ * C) * D).det ≠ 0)
    (hwhite : ν.map (fun z => toLp 2 (C *ᵥ z)) = stdGaussian (EuclideanSpace ℝ m)) :
    ν.map (fun z => z ⬝ᵥ (residualWeight D (Cᵀ * C) *ᵥ z)) =
      chiSquareLaw (Fintype.card m - Fintype.card p) := by
  let W := Cᵀ * C
  let B := C * D
  let K := LinearMap.range B.toEuclideanLin
  let P := residualMap D W
  let A := (Dᵀ * W * D)⁻¹ * Dᵀ * Cᵀ
  have hAB : A * B = 1 := by
    have h := response_mul_derivative D W hr
    simpa only [A, B, W, response, Matrix.mul_assoc] using h
  have hinj : Function.Injective B.toEuclideanLin := by
    intro x y h
    have h' := congrArg (fun v : EuclideanSpace ℝ m => A *ᵥ ofLp v) h
    change A *ᵥ (B *ᵥ ofLp x) = A *ᵥ (B *ᵥ ofLp y) at h'
    apply (WithLp.equiv 2 _).injective
    change ofLp x = ofLp y
    simpa only [Matrix.mulVec_mulVec, hAB, Matrix.one_mulVec] using h'
  have hdim : finrank ℝ K = Fintype.card p := by
    simpa only [finrank_euclideanSpace] using LinearMap.finrank_range_of_inj hinj
  let proj := fun u : EuclideanSpace ℝ m => ‖Kᗮ.orthogonalProjectionOnto u‖ ^ 2
  have eqn (z : m → ℝ) : proj (toLp 2 (C *ᵥ z)) = z ⬝ᵥ (residualWeight D W *ᵥ z) := by
    change ‖Kᗮ.starProjection (toLp 2 (C *ᵥ z))‖ ^ 2 = _
    rw [whitened_projection D C z hr, EuclideanSpace.real_norm_sq_eq]
    change (∑ i, ((C *ᵥ (P *ᵥ z)) i) ^ 2) = z ⬝ᵥ ((Pᵀ * W * P) *ᵥ z)
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Matrix.dotProduct_transpose_mulVec,
      dotProduct_comm (W *ᵥ _)]
    change _ = (P *ᵥ z) ⬝ᵥ ((Cᵀ * C) *ᵥ (P *ᵥ z))
    rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_transpose_mulVec]
    simp only [dotProduct, sq]
  calc
    _ = (ν.map (fun z => toLp 2 (C *ᵥ z))).map proj := by
      rw [Measure.map_map (by dsimp [proj]; fun_prop) (by fun_prop)]
      congr 1
      funext z
      exact (eqn z).symm
    _ = _ := by
      rw [hwhite, show (stdGaussian (EuclideanSpace ℝ m)).map proj = _ from overid_gaussian_reference K,
        hdim, finrank_euclideanSpace]

end StatadicFormal.Estimation.GMM
