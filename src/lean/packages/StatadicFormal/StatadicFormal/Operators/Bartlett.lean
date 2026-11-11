module

public import StatadicFormal.Probability.FiniteExtensions
public import Architect
public import Mathlib.Algebra.Order.Star.Real
public import Mathlib.Basic.Real.Basic
public import Mathlib.LinearAlgebra.Matrix.PosDef
public import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring

/-! The finite Bartlett lag kernel as a Gram matrix of overlapping windows. -/
@[expose] public section
noncomputable section
open Matrix
open scoped BigOperators

namespace StatadicFormal.Operators

/-- Number of width-(b+1) moving windows containing both time indices. -/
def bartlettCount (b t s : ℕ) : ℕ := (b+1) - (max t s - min t s)

/-- The unnormalized rectangular window membership matrix for a finite record. -/
def windowMembership (n b : ℕ) : Matrix (Fin (n+b)) (Fin n) ℝ :=
  fun k i => if i.val ≤ k.val ∧ k.val < i.val+b+1 then 1 else 0

private theorem window_overlap (n b : ℕ) (t s : Fin n) :
    (((windowMembership n b)ᵀ * windowMembership n b) t s) = bartlettCount b t s := by
  let L := Finset.Ico t.val (t.val+b+1)
  let R := Finset.Ico s.val (s.val+b+1)
  have hind (k : ℕ) :
      (if t.val ≤ k ∧ k < t.val+b+1 then (1 : ℝ) else 0) *
        (if s.val ≤ k ∧ k < s.val+b+1 then (1 : ℝ) else 0) =
        if k ∈ L ∩ R then 1 else 0 := by
    simp only [L, R, Finset.mem_inter, Finset.mem_Ico]
    split_ifs <;> simp_all
  have hsub : L ∩ R ⊆ Finset.range (n+b) := by
    intro k hk
    have h := (Finset.mem_Ico.mp (Finset.mem_inter.mp hk).1).2
    exact Finset.mem_range.mpr (by omega)
  change ∑ k : Fin (n+b), _ = _
  simp only [windowMembership, Matrix.transpose_apply, hind]
  rw [Fin.sum_univ_eq_sum_range (fun k => if k ∈ L ∩ R then (1 : ℝ) else 0)]
  rw [Finset.sum_boole]
  have heq : (Finset.range (n+b)).filter (fun k => k ∈ L ∩ R) = L ∩ R := by
    ext k
    simp only [Finset.mem_filter, and_iff_right_iff_imp]
    exact fun hk => hsub hk
  rw [heq, Finset.Ico_inter_Ico, Nat.card_Ico]
  congr 1
  unfold bartlettCount
  omega

/-- The Bartlett time-pair form stays PSD for an arbitrary finite score matrix. -/
@[blueprint "STATADIC-HAC-001"
  (statement := /-- Let $U$ contain $n>0$ ordered score rows and let
  $b\ge0$ be an integer. Define $K_{ts}=(b+1-|t-s|)_+/(b+1)$ on the
  observed time positions. Then the Bartlett covariance $U^TKU/n$ is
  positive semidefinite. Scores outside the observation range contribute zero. -/)
  (proof := /-- Introduce the matrix $A_{kt}=1$ for $t\le k<t+b+1$
  and zero otherwise. Counting the integer overlap of two windows gives
  $(A^TA)_{ts}=(b+1-|t-s|)_+$. Thus the covariance is the Gram matrix
  of $AU$ divided by $n(b+1)>0$. -/)]
theorem bartlett_covariance_psd {p : Type*} [Fintype p] (n b : ℕ) (hn : 0 < n)
    (U : Matrix (Fin n) p ℝ) :
    let K : Matrix (Fin n) (Fin n) ℝ := fun t s => (bartlettCount b t s : ℝ) / (b+1)
    ((n : ℝ)⁻¹ • (Uᵀ * K * U)).PosSemidef := by
  let A := windowMembership n b
  let K : Matrix (Fin n) (Fin n) ℝ := fun t s => (bartlettCount b t s : ℝ) / (b+1)
  have hK : K = ((b : ℝ)+1)⁻¹ • (Aᵀ * A) := by
    ext t s
    simp [K, A, window_overlap, div_eq_inv_mul]
  change ((n : ℝ)⁻¹ • (Uᵀ * K * U)).PosSemidef
  rw [hK, Matrix.mul_smul, Matrix.smul_mul]
  have hgram : (Uᵀ * (Aᵀ * A) * U).PosSemidef := by
    simpa [Matrix.conjTranspose_eq_transpose_of_trivial, Matrix.transpose_mul, Matrix.mul_assoc]
      using Matrix.posSemidef_conjTranspose_mul_self (A * U)
  exact (hgram.smul (by positivity : 0 ≤ ((b : ℝ)+1)⁻¹)).smul (by positivity)

/-- The common denominator, even under stationary second moments, attenuates
positive lags in a finite sample. -/
@[blueprint "STATADIC-HAC-002"
  (statement := /-- Let a real time series under a finite law have the same
  raw second moment $\gamma_h=E[z_t z_{t+h}]$ at every observed pair of lag
  $h$. For $n>0$ the expectation of the uncentered lag sum with denominator
  $n$ is $((n-h)_+/n)\gamma_h$. This equals a lag covariance interpretation
  when the process has mean zero. -/)
  (proof := /-- Move expectation inside the sum. Exactly $(n-h)_+$ indices
  have both endpoints in the record. Replace each pair expectation by the
  common lag moment and divide by $n$. -/)]
theorem lag_cross_moment {Ω : Type*} [Fintype Ω] (P : Probability.FiniteLaw Ω)
    (z : ℕ → Ω → ℝ) (n h : ℕ) (γ : ℝ)
    (hpair : ∀ t < n-h, P.mean (fun ω => z t ω * z (t+h) ω) = γ) :
    P.mean (fun ω => (∑ t : Fin (n-h), z t.val ω * z (t.val+h) ω) / (n : ℝ)) =
      ((n-h : ℕ) : ℝ) / n * γ := by
  simp only [div_eq_inv_mul, P.mean_smul, P.mean_sum, hpair _ (Fin.isLt _),
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

end StatadicFormal.Operators
