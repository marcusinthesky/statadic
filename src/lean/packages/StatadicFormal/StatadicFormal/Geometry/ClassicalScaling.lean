module

public import Architect
public import Mathlib.Analysis.Matrix.Order
import Mathlib.Tactic.Ring

/-! Distance reconstruction by classical scaling without truncating eigenvalues. -/

@[expose] public section
noncomputable section
open Matrix
open scoped BigOperators MatrixOrder
namespace StatadicFormal.Geometry
variable {ι : Type*} [Fintype ι]

/-- Entrywise expansion of minus one half times the double-centred symmetric squared distances. -/
def centeredDistanceGram (D : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  let row := fun i => (∑ j, D i j) / Fintype.card ι
  let grand := (∑ i, row i) / Fintype.card ι
  fun i j => -(D i j - row i - row j + grand) / 2

/-- The full PSD Gram has real coordinates and reproduces the supplied hollow dissimilarities. -/
@[blueprint "STATADIC-EMBEDDING-001"
  (statement := /-- Let $D$ be symmetric, hollow squared dissimilarities on a nonempty
  finite roster. Form $B_{ij}=-(D_{ij}-\bar D_i-\bar D_j+\bar D)/2$.
  If $B$ is positive semidefinite there is a real matrix $X$ with $B=XX^T$ and
  $D_{ij}=\sum_l (X_{il}-X_{jl})^2$ for every pair. Conversely, any such full
  factor of $B$ certifies positive semidefiniteness. -/)
  (proof := /-- Factor the positive semidefinite matrix as $XX^T$.
  The row means and grand mean cancel from $B_{ii}+B_{jj}-2B_{ij}$.
  Since $D_{ii}=D_{jj}=0$, the result is $D_{ij}$. Expand the three row
  products and collect squares coordinate by coordinate. A real Gram factor has
  nonnegative quadratic forms. -/)]
theorem pcoa_gram_certificate [DecidableEq ι] (D : Matrix ι ι ℝ) (hd : ∀ i, D i i = 0) :
    (centeredDistanceGram D).PosSemidef ↔ ∃ X : Matrix ι ι ℝ,
      centeredDistanceGram D = X * Xᵀ ∧ ∀ i j, D i j = ∑ l, (X i l - X j l) ^ 2 := by
  constructor
  · intro hb
    obtain ⟨X, hx⟩ := CStarAlgebra.nonneg_iff_eq_mul_star_self.mp hb.nonneg
    have hf : centeredDistanceGram D = X * Xᵀ := by simpa [Matrix.star_eq_conjTranspose] using hx
    refine ⟨X, hf, fun i j => ?_⟩
    have recover : D i j = centeredDistanceGram D i i + centeredDistanceGram D j j -
        2 * centeredDistanceGram D i j := by
      simp only [centeredDistanceGram, hd]
      ring
    rw [recover, hf]
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Finset.mul_sum, ← Finset.sum_add_distrib,
      ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro l _
    ring
  · rintro ⟨X, hf, _⟩
    rw [hf]
    simpa using Matrix.posSemidef_self_mul_conjTranspose X

private def avg (f : ι → ℝ) : ℝ := (∑ i, f i) / Fintype.card ι

private theorem avg_add (f g : ι → ℝ) : avg (fun i => f i + g i) = avg f + avg g := by
  simp [avg, Finset.sum_add_distrib, add_div]

private theorem avg_sub (f g : ι → ℝ) : avg (fun i => f i - g i) = avg f - avg g := by
  simp [avg, Finset.sum_sub_distrib, sub_div]

private theorem avg_mul (a : ℝ) (f : ι → ℝ) : avg (fun i => a * f i) = a * avg f := by
  simp [avg, ← Finset.mul_sum, mul_div_assoc]

private theorem avg_const [Nonempty ι] (a : ℝ) : avg (fun _ : ι => a) = a := by
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp [avg, nsmul_eq_mul, hn]

private theorem avg_sum {κ : Type*} [Fintype κ] (f : ι → κ → ℝ) :
    avg (fun i => ∑ l, f i l) = ∑ l, avg (fun i => f i l) := by
  simp only [avg, ← Finset.sum_div]
  rw [Finset.sum_comm]

/-- The entry formula equals the matrix multiplication performed by the PCoA kernel. -/
theorem centeredDistanceGram_eq_double_center [DecidableEq ι] (D : Matrix ι ι ℝ) (hs : ∀ i j, D i j = D j i) :
    let J : Matrix ι ι ℝ := fun i j => (if i = j then 1 else 0) - (Fintype.card ι : ℝ)⁻¹
    centeredDistanceGram D = (-1 / 2 : ℝ) • (J * D * J) := by
  classical
  let J : Matrix ι ι ℝ := fun i j => (if i = j then 1 else 0) - (Fintype.card ι : ℝ)⁻¹
  have hleft (i j) : (J * D) i j = D i j - avg (D j) := by
    change (∑ k, ((if i = k then 1 else 0) - (Fintype.card ι : ℝ)⁻¹) * D k j) = _
    simp [sub_mul, ite_mul, Finset.sum_sub_distrib, ← Finset.mul_sum,
      avg, div_eq_inv_mul, hs j]
  have hright (A : Matrix ι ι ℝ) (i j) : (A * J) i j = A i j - avg (A i) := by
    change (∑ k, A i k * ((if k = j then 1 else 0) - (Fintype.card ι : ℝ)⁻¹)) = _
    simp [mul_sub, mul_ite, Finset.sum_sub_distrib, ← Finset.sum_mul,
      avg, div_eq_mul_inv]
  ext i j
  change -(D i j - avg (D i) - avg (D j) + avg (fun a => avg (D a))) / 2 =
      (-1 / 2 : ℝ) * ((J * D) * J) i j
  rw [hright, hleft, funext (hleft i), avg_sub]
  ring

private theorem avg_sq_dist [Nonempty ι] (x : ι → ℝ) (a : ℝ) :
    avg (fun i => (a - x i) ^ 2) = a ^ 2 - (2 * a) * avg x + avg (fun i => x i ^ 2) := by
  simp only [show ∀ i, (a - x i) ^ 2 = a ^ 2 - (2 * a) * x i + x i ^ 2 by intro i; ring,
    avg_add, avg_sub, avg_const, avg_mul]

/-- A real Euclidean realization yields the Gram of its column-centered coordinates. -/
private theorem gram_of_realization {κ : Type*} [Fintype κ] [Nonempty ι]
    (x : Matrix ι κ ℝ) (D : Matrix ι ι ℝ) (hd : ∀ i j, D i j = ∑ l, (x i l - x j l) ^ 2) :
    let Y : Matrix ι κ ℝ := fun i l => x i l - avg (fun j => x j l)
    centeredDistanceGram D = Y * Yᵀ := by
  let m := fun l => avg (fun j => x j l)
  let v := fun l => avg (fun j => (x j l) ^ 2)
  have row (i) : avg (D i) = ∑ l, (x i l ^ 2 - (2 * x i l) * m l + v l) := by
    simp only [funext (hd i), avg_sum, avg_sq_dist, m, v]
  have grand : avg (fun j => avg (D j)) = ∑ l, (v l - (2 * m l) * m l + v l) := by
    simp_rw [row, show ∀ i l, (2 * x i l) * m l = (2 * m l) * x i l by intro i l; ring]
    simp only [avg_sum, avg_add, avg_sub, avg_mul, avg_const, v, m]
  ext i j
  change -(D i j - avg (D i) - avg (D j) + avg (fun a => avg (D a))) / 2 =
    ∑ l, (x i l - m l) * (x j l - m l)
  rw [hd, row i, row j, grand]
  simp only [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib, ← Finset.sum_neg_distrib,
    Finset.sum_div]
  apply Finset.sum_congr rfl
  intro l _
  ring

/-- A hollow dissimilarity is squared Euclidean exactly when its double-centered Gram is PSD. -/
@[blueprint "STATADIC-EMBEDDING-002"
  (statement := /-- A symmetric hollow squared-dissimilarity matrix on a nonempty finite
  roster is realized by Euclidean coordinates exactly when its double-centered Gram is
  positive semidefinite. All axes are retained in this exact criterion. -/)
  (proof := /-- For the forward direction factor the PSD Gram and recover squared distances.
  For the converse, let $m_l$ be the mean of coordinate $l$. Expanding the squares in
  each row mean and the grand mean identifies the centered Gram with $YY^T$, where
  $Y_{il}=X_{il}-m_l$. Its quadratic form is nonnegative. -/)]
theorem pcoa_psd_iff_euclidean [DecidableEq ι] [Nonempty ι] (D : Matrix ι ι ℝ) (hd : ∀ i, D i i = 0) :
    (centeredDistanceGram D).PosSemidef ↔
      ∃ X : Matrix ι ι ℝ, ∀ i j, D i j = ∑ l, (X i l - X j l) ^ 2 := by
  constructor
  · intro h
    obtain ⟨X, _, hX⟩ := (pcoa_gram_certificate D hd).mp h
    exact ⟨X, hX⟩
  · rintro ⟨X, hX⟩
    rw [gram_of_realization X D hX]
    let Y : Matrix ι ι ℝ := fun i l => X i l - avg (fun j => X j l)
    change (Y * Yᵀ).PosSemidef
    simpa using Matrix.posSemidef_self_mul_conjTranspose Y

/-- Truncating the spectrum at a common tolerance bounds each squared-distance error. -/
@[blueprint "STATADIC-EMBEDDING-003"
  (statement := /-- Let $U$ have orthonormal rows and write a reconstructed squared
  distance as $D_{ij}=\sum_l \lambda_l(U_{il}-U_{jl})^2$. Assume all eigenvalues
  are at least $-t$, where $t\ge0$. Retain only eigenvalues above $t$. Then for
  distinct rows the resulting squared distance differs from $D_{ij}$ by at most $2t$. -/)
  (proof := /-- On a discarded axis $|\lambda_l|\le t$. Thus the absolute
  omitted sum is at most $t\sum_l(U_{il}-U_{jl})^2$. Distinct orthonormal rows
  give this last sum as $1+1-2(0)=2$. -/)]
theorem pcoa_truncation_error {κ : Type*} [Fintype κ] (u v eigen : κ → ℝ) (t : ℝ)
    (ht : 0 ≤ t) (hvalid : ∀ l, -t ≤ eigen l)
    (hu : ∑ l, u l ^ 2 = 1) (hv : ∑ l, v l ^ 2 = 1) (huv : ∑ l, u l * v l = 0) :
    |(∑ l, eigen l * (u l - v l) ^ 2) -
      ∑ l, (if t < eigen l then eigen l else 0) * (u l - v l) ^ 2| ≤ 2 * t := by
  have hs : ∑ l, (u l - v l) ^ 2 = 2 := by
    simp only [show ∀ l, (u l - v l) ^ 2 = u l ^ 2 + v l ^ 2 - 2 * (u l * v l)
      by intro l; ring, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hu, hv, huv]
    norm_num
  rw [← Finset.sum_sub_distrib]
  calc
    _ ≤ _ := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ l, t * (u l - v l) ^ 2 := by
      apply Finset.sum_le_sum
      intro l _
      split_ifs with hl
      · simp [mul_nonneg ht (sq_nonneg _)]
      · simp only [zero_mul, sub_zero, abs_mul, abs_of_nonneg (sq_nonneg (u l - v l))]
        exact mul_le_mul_of_nonneg_right (abs_le.mpr ⟨hvalid l, le_of_not_gt hl⟩) (sq_nonneg _)
    _ = 2 * t := by rw [← Finset.mul_sum, hs, mul_comm]

end StatadicFormal.Geometry
