module

public import StatadicFormal.Estimation.Linear.FiniteLaw
public import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic.Ring

/-! Quadratic moment conditions for a spatial autoregressive error filter. -/
@[expose] public section
noncomputable section
open Matrix StatadicFormal.Probability
open scoped BigOperators

namespace StatadicFormal.Estimation.Spatial

variable {ι Ω : Type*} [Fintype ι]

/-- The sample equation used to fit the spatial error coefficient. -/
@[blueprint "STATADIC-GMM-001"
  (statement := /-- Put $v=Wu$ and $e=u-\lambda v$. For any symmetric
  moment matrix $A$ on $n>0$ units,
  \[
    e^TAe/n=u^TAu/n-\lambda(2v^TAu/n)+\lambda^2 v^TAv/n .
  \] -/)
  (proof := /-- Expand both occurrences of $e$. The mixed products
  $u^TAv$ and $v^TAu$ agree by symmetry. Collect them and divide by $n$.
  This gives the two coefficient signs used in the sample moment system. -/)]
theorem filtered_quadratic_moment (W A : Matrix ι ι ℝ)
    (hA : Aᵀ = A) (u : ι → ℝ) (r : ℝ) :
    let v := W *ᵥ u
    let e := u - r • v
    let n : ℝ := Fintype.card ι
    (e ⬝ᵥ (A *ᵥ e)) / n =
      (u ⬝ᵥ (A *ᵥ u)) / n - r * (2 * (v ⬝ᵥ (A *ᵥ u)) / n) +
        r ^ 2 * ((v ⬝ᵥ (A *ᵥ v)) / n) := by
  dsimp only
  have hcross := Matrix.dotProduct_transpose_mulVec A u (W *ᵥ u)
  rw [hA] at hcross
  simp only [Matrix.mulVec_sub, Matrix.mulVec_smul, sub_dotProduct, dotProduct_sub,
    dotProduct_smul, smul_dotProduct, smul_eq_mul, hcross]
  ring

/-- Derivation of the population zero moment without assuming equal variances. -/
@[blueprint "STATADIC-GMM-002"
  (statement := /-- Let $\varepsilon$ be a real vector on a finite probability
  space. Assume $E[\varepsilon_i\varepsilon_j]=0$ whenever $i\ne j$.
  If $A_{ii}=0$ for every $i$, then
  $E[\sum_{i,j}A_{ij}\varepsilon_i\varepsilon_j]=0$. -/)
  (proof := /-- Exchange the finite expectation with both index sums.
  Each diagonal summand vanishes by the hypothesis on $A$. Every off-diagonal
  summand vanishes by the corresponding raw second moment. -/)]
theorem zero_diagonal_expected_moment [Fintype Ω]
    (P : FiniteLaw Ω) (A : Matrix ι ι ℝ) (ε : ι → Ω → ℝ)
    (hdiag : ∀ i, A i i = 0)
    (hpair : ∀ i j, i ≠ j → P.mean (fun ω => ε i ω * ε j ω) = 0) :
    P.mean (fun ω => ∑ i, ∑ j, A i j * (ε i ω * ε j ω)) = 0 := by
  simp_rw [P.mean_sum, P.mean_smul]
  apply Finset.sum_eq_zero
  intro i _
  apply Finset.sum_eq_zero
  intro j _
  by_cases h : i = j
  · subst j
    rw [hdiag, zero_mul]
  · rw [hpair i j h, mul_zero]

end StatadicFormal.Estimation.Spatial
