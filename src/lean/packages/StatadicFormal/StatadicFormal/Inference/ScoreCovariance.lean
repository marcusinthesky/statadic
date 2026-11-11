module

public import StatadicFormal.Probability.FiniteExtensions
public import Architect
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Matrix.Mul
public import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Tactic.Ring

/-! Exact covariance diagnostics for independent score rows and two-way panel clusters. -/
@[expose] public section
noncomputable section
open Matrix
open scoped BigOperators

namespace StatadicFormal.Inference

/-- HC0 and HC1 can have a null contrast even when coordinate variances are positive. -/
@[blueprint "STATADIC-GLMCOV-001"
  (statement := /-- Let $U$ be the matrix of fitted score rows, $B$ the
  coefficient response map, and $c>0$ the common scale, including the sample
  normalization. For $V=c BU^TUB^T$, a contrast $v$ has
  $v^TVv=c\|UB^Tv\|^2$. In particular, its variance vanishes exactly when
  every fitted score row annihilates $B^Tv$. -/)
  (proof := /-- Move both transposes across the scalar product to write the
  quadratic form as a sum of squared projected scores. A positive common
  multiplier preserves its zeros. A sum of real squares vanishes precisely
  when every term vanishes. -/)]
theorem score_sandwich_contrast {ι p : Type*} [Fintype ι] [Fintype p]
    (U : Matrix ι p ℝ) (B : Matrix p p ℝ) (c : ℝ) (hc : 0 < c) (v : p → ℝ) :
    let V := c • (B * Uᵀ * U * Bᵀ)
    let z := U *ᵥ (Bᵀ *ᵥ v)
    v ⬝ᵥ (V *ᵥ v) = c * (z ⬝ᵥ z) ∧ (v ⬝ᵥ (V *ᵥ v) = 0 ↔ z = 0) := by
  dsimp only
  have hb (y : p → ℝ) : v ⬝ᵥ (B *ᵥ y) = y ⬝ᵥ (Bᵀ *ᵥ v) :=
    (Matrix.dotProduct_transpose_mulVec B y v).symm
  have hid : v ⬝ᵥ ((c • (B * Uᵀ * U * Bᵀ)) *ᵥ v) =
      c * ((U *ᵥ (Bᵀ *ᵥ v)) ⬝ᵥ (U *ᵥ (Bᵀ *ᵥ v))) := by
    simp only [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul, ← Matrix.mulVec_mulVec, hb]
    rw [dotProduct_comm (Uᵀ *ᵥ _), Matrix.dotProduct_transpose_mulVec]
  rw [hid]
  simp [hc.ne', dotProduct_self_eq_zero]

/-- Cluster membership matrix, one nonzero per observation column. -/
def clusterMembership {ι g : Type*} [DecidableEq g] (group : ι → g) : Matrix g ι ℝ :=
  fun k i => if group i = k then 1 else 0

private theorem membership_gram {ι g : Type*} [Fintype g] [DecidableEq g] (group : ι → g)
    (i j : ι) :
    ((clusterMembership group)ᵀ * clusterMembership group) i j =
      if group i = group j then 1 else 0 := by
  classical
  simp [Matrix.mul_apply, clusterMembership, eq_comm]

/-- For unique entity-time keys the intersection correction is the row-level Gram. -/
@[blueprint "STATADIC-PANELCOV-001"
  (statement := /-- For influence rows $U$ with unique entity--time keys,
  form the entity and time membership matrices $E,T$. The unadjusted
  two-way covariance $(EU)^T(EU)+(TU)^T(TU)-U^TU$ equals $U^TKU$,
  where $K_{ij}$ is one when rows share an entity or a time, and zero otherwise. -/)
  (proof := /-- The two membership Gram matrices pick the pairs sharing each
  index. Uniqueness makes their intersection the diagonal. Inclusion--exclusion
  therefore gives $E^TE+T^TT-I=K$. Multiply on each side by the influences. -/)]
theorem cluster_union_covariance {ι g t p : Type*} [Fintype ι] [DecidableEq ι]
    [Fintype g] [DecidableEq g] [Fintype t] [DecidableEq t]
    (entity : ι → g) (time : ι → t)
    (hkey : Function.Injective (fun i => (entity i, time i))) (U : Matrix ι p ℝ) :
    let E := clusterMembership entity
    let T := clusterMembership time
    let K : Matrix ι ι ℝ := fun i j => if entity i = entity j ∨ time i = time j then 1 else 0
    (E * U)ᵀ * (E * U) + (T * U)ᵀ * (T * U) - Uᵀ * U = Uᵀ * K * U := by
  let E := clusterMembership entity
  let T := clusterMembership time
  let K : Matrix ι ι ℝ := fun i j => if entity i = entity j ∨ time i = time j then 1 else 0
  have hK : K = Eᵀ * E + Tᵀ * T - 1 := by
    ext i j
    simp only [Matrix.sub_apply, Matrix.add_apply, E, T, membership_gram, Matrix.one_apply, K]
    by_cases he : entity i = entity j <;> by_cases ht : time i = time j
    · obtain rfl := hkey (Prod.ext he ht)
      simp
    · have hn : i ≠ j := fun h => ht (congrArg time h)
      simp [he, ht, hn]
    · have hn : i ≠ j := fun h => he (congrArg entity h)
      simp [he, ht, hn]
    · have hn : i ≠ j := fun h => he (congrArg entity h)
      simp [he, ht, hn]
  change (E * U)ᵀ * (E * U) + (T * U)ᵀ * (T * U) - Uᵀ * U = Uᵀ * K * U
  rw [hK]
  simp [Matrix.transpose_mul, Matrix.mul_sub, Matrix.mul_add, Matrix.sub_mul, Matrix.add_mul,
    Matrix.mul_assoc]

/-- The two-way mask captures the variance of the total when every excluded pair
has zero cross moment. This models actual centered influences before residual fitting. -/
@[blueprint "STATADIC-PANELCOV-002"
  (statement := /-- Let $z_i$ be centered, scalar influence contributions under
  a finite law. Suppose $E[z_i z_j]=0$ whenever rows differ in both entity and
  time. Then $\operatorname{Var}(\sum_i z_i)=
  E[\sum_{i,j}1\{e_i=e_j\ \mathrm{or}\ t_i=t_j\}\,z_i z_j]$. Shared pairs
  may have arbitrary cross moments. -/)
  (proof := /-- The total has mean zero. Expand its second moment as the
  sum of all pair expectations. The dependence assumption removes every
  pair outside the cluster union. Exchange the finite sums and expectation
  to obtain the claimed masked expression. -/)]
theorem cluster_union_variance {ι g t Ω : Type*} [Fintype ι] [Fintype Ω]
    [DecidableEq g] [DecidableEq t] (P : Probability.FiniteLaw Ω)
    (entity : ι → g) (time : ι → t) (z : ι → Ω → ℝ) (hmean : ∀ i, P.mean (z i) = 0)
    (hcross : ∀ i j, entity i ≠ entity j → time i ≠ time j →
      P.mean (fun ω => z i ω * z j ω) = 0) :
    P.variance (fun ω => ∑ i, z i ω) =
      P.mean (fun ω => ∑ i, ∑ j, if entity i = entity j ∨ time i = time j
        then z i ω * z j ω else 0) := by
  have htotal : P.mean (fun ω => ∑ i, z i ω) = 0 := by rw [P.mean_sum]; simp [hmean]
  rw [P.research_variance_of_centered _ htotal]
  simp_rw [pow_two, Finset.sum_mul, Finset.mul_sum, P.mean_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  by_cases h : entity i = entity j ∨ time i = time j
  · simp [h]
  · simp [h, hcross i j (fun he => h (.inl he)) (fun ht => h (.inr ht))]

end StatadicFormal.Inference
