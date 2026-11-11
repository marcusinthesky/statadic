module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Data.Fintype.BigOperators
public import Architect
import Mathlib.Tactic.Ring

/-!
# Finite bias–variance decomposition

Finite specialization of Joram Soch, *The Book of Statistical Proofs*, proof P5,
“Partition of the mean squared error into bias and variance” (2019-11-27),
https://statproofbook.github.io/P/mse-bnv.html, theorem and proof equations
`MSE-ref1`–`MSE-ref2`. The argument inserts the mean, expands the square, and
uses the vanishing expected centered term. This models neither infinite sample
spaces nor floating-point computations.
-/

public section

open scoped BigOperators

noncomputable section

namespace StatadicFormal.Probability

/-- A finite probability law, represented by normalized nonnegative weights. -/
structure FiniteLaw (Ω : Type*) [Fintype Ω] where
  /-- Probability mass assigned to each outcome. -/
  weight : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ weight ω
  total : ∑ ω, weight ω = 1

namespace FiniteLaw

variable {Ω : Type*} [Fintype Ω] (P : FiniteLaw Ω)

/-- Expectation under a finite law. -/
@[expose] def mean (f : Ω → ℝ) : ℝ := ∑ ω, P.weight ω * f ω

/-- Expected squared deviation from the mean. -/
@[expose] def variance (f : Ω → ℝ) : ℝ := P.mean (fun ω ↦ (f ω - P.mean f) ^ 2)

/-- Difference between expected estimate and target. -/
@[expose] def bias (f : Ω → ℝ) (θ : ℝ) : ℝ := P.mean f - θ

/-- Expected squared error around the target. -/
@[expose] def mse (f : Ω → ℝ) (θ : ℝ) : ℝ := P.mean (fun ω ↦ (f ω - θ) ^ 2)

private theorem mean_const (c : ℝ) : P.mean (fun _ ↦ c) = c := by
  unfold mean
  rw [← Finset.sum_mul, P.total, one_mul]

private theorem mean_add (f g : Ω → ℝ) :
    P.mean (fun ω ↦ f ω + g ω) = P.mean f + P.mean g := by
  simp [mean, mul_add, Finset.sum_add_distrib]

private theorem mean_sub (f g : Ω → ℝ) :
    P.mean (fun ω ↦ f ω - g ω) = P.mean f - P.mean g := by
  simp [mean, mul_sub, Finset.sum_sub_distrib]

private theorem mean_smul (c : ℝ) (f : Ω → ℝ) :
    P.mean (fun ω ↦ c * f ω) = c * P.mean f := by
  unfold mean
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro ω _
  ring

private theorem mean_centered (f : Ω → ℝ) :
    P.mean (fun ω ↦ f ω - P.mean f) = 0 := by
  rw [mean_sub, mean_const]
  ring

/-- Finite-space specialization of Soch's bias–variance decomposition (proof P5). -/
@[blueprint "STATADIC-MOMENTS-001"
  (statement := /-- Let $\Omega$ be finite and let $P$ have nonnegative weights
  $p_\omega$ summing to one. For a real estimator $f:\Omega\to\mathbb R$ and
  target $\theta\in\mathbb R$, write
  \[
  \mathbb E_P[g]=\sum_{\omega\in\Omega}p_\omega g(\omega),\quad
  m=\mathbb E_P[f],\quad \operatorname{Bias}=m-\theta,\quad
  \operatorname{Var}=\mathbb E_P[(f-m)^2].
  \]
  Then $\operatorname{MSE}=\mathbb E_P[(f-\theta)^2]$ satisfies
  \[
  \operatorname{MSE}=\operatorname{Var}+\operatorname{Bias}^2.
  \] -/)
  (proof := /-- Following proof P5, equations MSE-ref1 and MSE-ref2, insert and
  subtract $m$ and expand:
  \[
  (f-\theta)^2=(f-m)^2+2(m-\theta)(f-m)+(m-\theta)^2.
  \]
  Finite expectation is linear, $\mathbb E_P[f-m]=m-m=0$, and normalized
  weights leave constants unchanged. Taking expectations therefore eliminates
  the cross term and yields $\operatorname{MSE}=\operatorname{Var}+
  \operatorname{Bias}^2$. -/)]
theorem mse_decomposition (f : Ω → ℝ) (θ : ℝ) :
    P.mse f θ = P.variance f + (P.bias f θ) ^ 2 := by
  have h : (fun ω ↦ (f ω - θ) ^ 2) =
      (fun ω ↦ (f ω - P.mean f) ^ 2 +
        (2 * (P.mean f - θ)) * (f ω - P.mean f) + (P.mean f - θ) ^ 2) := by
    funext ω
    ring
  unfold mse
  rw [h, mean_add, mean_add, mean_smul, mean_centered, mean_const]
  simp [variance, bias]

end FiniteLaw
end StatadicFormal.Probability
