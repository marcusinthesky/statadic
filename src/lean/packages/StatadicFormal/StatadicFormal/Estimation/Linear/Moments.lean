module

public import StatadicFormal.Estimation.Linear.FiniteLaw
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

@[expose] public section

noncomputable section

/-!
# Linear estimator moments on finite probability spaces

The design and coefficients are deterministic. Unbiasedness requires both a
correct mean model and the reproduction equation. Spherical second moments
are used only for the variance theorem, not for unbiasedness.
Reference: searle_1971; casella_berger_2002.
-/

open scoped BigOperators
namespace StatadicFormal.Estimation.Linear
open StatadicFormal.Probability

variable {ι κ Ω : Type*} [Fintype ι] [Fintype κ] [Fintype Ω]

/-- A scalar linear estimate, including a coefficient contrast. -/
def estimate (a : ι → ℝ) (Y : ι → Ω → ℝ) : Ω → ℝ :=
  fun ω => ∑ i, a i * Y i ω

/-- Mean of a linear estimator; errors need not be independent. -/
theorem mean_estimate (P : FiniteLaw Ω) (a : ι → ℝ) (Y : ι → Ω → ℝ) :
    P.mean (estimate a Y) = ∑ i, a i * P.mean (Y i) := by
  unfold estimate
  rw [P.mean_sum]
  apply Finset.sum_congr rfl
  intro i _
  exact P.mean_smul (a i) (Y i)

/-- Exact bias formula for a linear estimate of a fixed-design contrast.
The mean error μ can be nonzero; no exogeneity claim is assumed by the code. -/
theorem bias_formula (P : FiniteLaw Ω)
    (a : ι → ℝ) (X : ι → κ → ℝ) (β t : κ → ℝ)
    (ε : ι → Ω → ℝ) (μ : ι → ℝ)
    (hμ : ∀ i, P.mean (ε i) = μ i)
    (hreproduce : ∀ j, ∑ i, a i * X i j = t j) :
    P.bias (estimate a (fun i ω => (∑ j, X i j * β j) + ε i ω))
      (∑ j, t j * β j) = ∑ i, a i * μ i := by
  classical
  have hsignal : (∑ i, a i * (∑ j, X i j * β j)) = ∑ j, t j * β j := by
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    calc
      (∑ i, a i * (X i j * β j)) = (∑ i, a i * X i j) * β j := by
        rw [Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = t j * β j := by rw [hreproduce j]
  unfold FiniteLaw.bias
  rw [mean_estimate]
  simp_rw [P.mean_add, P.mean_const, hμ, mul_add]
  rw [Finset.sum_add_distrib, hsignal]
  ring

/-- Finite-sample unbiasedness for any linear estimator reproducing the contrast. -/
theorem unbiased (P : FiniteLaw Ω)
    (a : ι → ℝ) (X : ι → κ → ℝ) (β t : κ → ℝ)
    (ε : ι → Ω → ℝ)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hreproduce : ∀ j, ∑ i, a i * X i j = t j) :
    P.mean (estimate a (fun i ω => (∑ j, X i j * β j) + ε i ω)) =
      ∑ j, t j * β j := by
  have hb := bias_formula P a X β t ε (fun _ => 0) hzero hreproduce
  simp only [mul_zero, Finset.sum_const_zero] at hb
  unfold FiniteLaw.bias at hb
  linarith

/-- Variance of a zero-mean linear noise estimate with diagonal second moments.
The pair-moment assumption is weaker than independence, but is explicit. -/
theorem variance_diagonal [DecidableEq ι]
    (P : FiniteLaw Ω) (a : ι → ℝ) (ε : ι → Ω → ℝ) (σ2 : ι → ℝ)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 i else 0) :
    P.variance (estimate a ε) = ∑ i, σ2 i * (a i) ^ 2 := by
  have hm : P.mean (estimate a ε) = 0 := by
    rw [mean_estimate]
    simp [hzero]
  have hexpand : (fun ω => (estimate a ε ω) ^ 2) =
      (fun ω => ∑ i, ∑ j, (a i * a j) * (ε i ω * ε j ω)) := by
    funext ω
    unfold estimate
    rw [pow_two, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  unfold FiniteLaw.variance
  rw [hm]
  simp only [sub_zero]
  rw [hexpand, P.mean_sum]
  simp_rw [P.mean_sum, P.mean_smul, hpair]
  apply Finset.sum_congr rfl
  intro i _
  simp [mul_ite, pow_two, mul_comm]

/-- Variance of a zero-mean linear noise estimate with spherical second moments. -/
theorem variance_spherical [DecidableEq ι]
    (P : FiniteLaw Ω) (a : ι → ℝ) (ε : ι → Ω → ℝ) (σ2 : ℝ)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0) :
    P.variance (estimate a ε) = σ2 * ∑ i, (a i) ^ 2 := by
  rw [Finset.mul_sum]
  exact variance_diagonal P a ε (fun _ => σ2) hzero hpair

/-- A deterministic signal changes the mean but not the variance of an estimate. -/
theorem variance_shifted_estimate
    (P : FiniteLaw Ω) (a s : ι → ℝ) (ε : ι → Ω → ℝ) :
    P.variance (estimate a (fun i ω => s i + ε i ω)) =
      P.variance (estimate a ε) := by
  have h : estimate a (fun i ω => s i + ε i ω) =
      (fun ω => estimate a ε ω + ∑ i, a i * s i) := by
    funext ω
    simp only [estimate, mul_add, Finset.sum_add_distrib]
    ring
  rw [h]
  exact P.variance_add_const (estimate a ε) (∑ i, a i * s i)

end StatadicFormal.Estimation.Linear
