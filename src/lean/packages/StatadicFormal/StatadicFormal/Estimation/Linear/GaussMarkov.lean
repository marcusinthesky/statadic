module

public import StatadicFormal.Estimation.Linear.Moments
public import Architect
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

@[expose] public section

noncomputable section

/-!
# Gauss–Markov for scalar contrasts

The candidate weights b lie in the design column space. Competitor weights a
reproduce the same contrast. We DERIVE orthogonality and the sum-of-squares gap;
we do not assume the covariance gap is PSD. This characterizes OLS contrast
weights once their normal equations and reproduction conditions are established.
The file does not prove that STATADIC's QR/SVD solver realizes those exact equations.
Reference: searle_1971. Probability scope: finite sample spaces only.
-/

open scoped BigOperators
namespace StatadicFormal.Estimation.Linear
open StatadicFormal.Probability

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Row-space normality and equal design action force orthogonality. -/
theorem normal_orthogonality
    (X : ι → κ → ℝ) (a b : ι → ℝ) (c : κ → ℝ)
    (hnormal : ∀ i, b i = ∑ j, c j * X i j)
    (hsame : ∀ j, ∑ i, a i * X i j = ∑ i, b i * X i j) :
    ∑ i, b i * (a i - b i) = 0 := by
  classical
  calc
    (∑ i, b i * (a i - b i)) =
        ∑ i, (∑ j, c j * X i j) * (a i - b i) := by
      apply Finset.sum_congr rfl
      intro i _
      exact congrArg (fun z => z * (a i - b i)) (hnormal i)
    _ = ∑ j, c j * (∑ i, X i j * (a i - b i)) := by
      simp_rw [Finset.sum_mul]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro j _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = 0 := by
      apply Finset.sum_eq_zero
      intro j _
      have hcol : (∑ i, X i j * (a i - b i)) = 0 := by
        calc
          (∑ i, X i j * (a i - b i)) =
              (∑ i, a i * X i j) - ∑ i, b i * X i j := by
            rw [← Finset.sum_sub_distrib]
            apply Finset.sum_congr rfl
            intro i _
            ring
          _ = 0 := by rw [hsame j]; ring
      rw [hcol, mul_zero]

/-- Pythagorean variance-coefficient decomposition. -/
theorem squares_decomposition (a b : ι → ℝ)
    (horth : ∑ i, b i * (a i - b i) = 0) :
    (∑ i, (a i) ^ 2) = (∑ i, (b i) ^ 2) + ∑ i, (a i - b i) ^ 2 := by
  have h : (∑ i, (a i) ^ 2) =
      (∑ i, (b i) ^ 2) + (∑ i, (a i - b i) ^ 2) +
        2 * (∑ i, b i * (a i - b i)) := by
    calc
      (∑ i, (a i) ^ 2) =
          ∑ i, ((b i) ^ 2 + (a i - b i) ^ 2 + 2 * (b i * (a i - b i))) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by simp [Finset.sum_add_distrib, Finset.mul_sum]
  rw [horth] at h
  simpa using h

/-- Exact variance gap for an OLS-normal contrast and an unbiased competitor. -/
theorem gauss_markov_gap {Ω : Type*} [Fintype Ω] [DecidableEq ι]
    (P : FiniteLaw Ω) (X : ι → κ → ℝ) (a b : ι → ℝ) (c : κ → ℝ)
    (ε : ι → Ω → ℝ) (σ2 : ℝ)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0)
    (hnormal : ∀ i, b i = ∑ j, c j * X i j)
    (hsame : ∀ j, ∑ i, a i * X i j = ∑ i, b i * X i j) :
    P.variance (estimate a ε) - P.variance (estimate b ε) =
      σ2 * ∑ i, (a i - b i) ^ 2 := by
  rw [variance_spherical P a ε σ2 hzero hpair,
      variance_spherical P b ε σ2 hzero hpair]
  rw [squares_decomposition a b (normal_orthogonality X a b c hnormal hsame)]
  ring

/-- Gauss–Markov variance ordering in the stated linear, unbiased contrast class. -/
theorem gauss_markov_efficiency {Ω : Type*} [Fintype Ω] [DecidableEq ι]
    (P : FiniteLaw Ω) (X : ι → κ → ℝ) (a b : ι → ℝ) (c : κ → ℝ)
    (ε : ι → Ω → ℝ) (σ2 : ℝ) (hσ : 0 ≤ σ2)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0)
    (hnormal : ∀ i, b i = ∑ j, c j * X i j)
    (hsame : ∀ j, ∑ i, a i * X i j = ∑ i, b i * X i j) :
    P.variance (estimate b ε) ≤ P.variance (estimate a ε) := by
  have hgap := gauss_markov_gap P X a b c ε σ2 hzero hpair hnormal hsame
  have hnon : 0 ≤ σ2 * ∑ i, (a i - b i) ^ 2 :=
    mul_nonneg hσ (Finset.sum_nonneg (fun i _ => sq_nonneg (a i - b i)))
  linarith

/-- The same ordering for full observations with any fixed signal, not just noise. -/
theorem gauss_markov_observation_efficiency {Ω : Type*} [Fintype Ω] [DecidableEq ι]
    (P : FiniteLaw Ω) (X : ι → κ → ℝ) (a b s : ι → ℝ) (c : κ → ℝ)
    (ε : ι → Ω → ℝ) (σ2 : ℝ) (hσ : 0 ≤ σ2)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0)
    (hnormal : ∀ i, b i = ∑ j, c j * X i j)
    (hsame : ∀ j, ∑ i, a i * X i j = ∑ i, b i * X i j) :
    P.variance (estimate b (fun i ω => s i + ε i ω)) ≤
      P.variance (estimate a (fun i ω => s i + ε i ω)) := by
  rw [variance_shifted_estimate, variance_shifted_estimate]
  exact gauss_markov_efficiency P X a b c ε σ2 hσ hzero hpair hnormal hsame


/-- Finite-law scalar specialization of Stępniak (2010), Theorem 4.1,
pp. 4–5: a reproduced contrast with weights in the design column space is
unbiased and has no greater variance than any other reproduced contrast.
The existence and construction of such weights are separate obligations. -/
@[blueprint "STATADIC-LINEAR-004"
  (statement := /-- Let $X\in\mathbb R^{n\times p}$, $t,\beta\in\mathbb R^p$,
  and $Y=X\beta+\varepsilon$ on a finite probability space. Assume
  $\mathbb E\varepsilon=0$, $\mathbb E(\varepsilon\varepsilon^T)=\sigma^2I_n$
  with $\sigma^2>0$, and choose $a,b\in\mathbb R^n$ with
  \[ X^Ta=X^Tb=t,\qquad b\in\operatorname{col}(X). \]
  Then the supplied contrast weights $b$ satisfy
  \[ \mathbb E(b^TY)=t^T\beta,\qquad
     \operatorname{Var}(b^TY)\leq\operatorname{Var}(a^TY). \]
  -/)
  (proof := /-- Stępniak (2010), Theorem 4.1, pp. 4--5, decomposes a
  competitor into the least-squares component and an orthogonal difference.
  Here $X^T(a-b)=0$ and $b\in\operatorname{col}(X)$ give $b^T(a-b)=0$;
  spherical moments therefore yield the variance gap
  \[ \operatorname{Var}(a^TY)-\operatorname{Var}(b^TY)
     =\sigma^2\sum_i(a_i-b_i)^2\geq0. \]
  This conditional statement assumes the displayed weights; their existence
  and least-squares construction are separate results. -/)]
theorem finite_gauss_markov_contrast {Ω : Type*} [Fintype Ω] [DecidableEq ι]
    (P : FiniteLaw Ω) (X : ι → κ → ℝ) (a b : ι → ℝ) (c t β : κ → ℝ)
    (ε : ι → Ω → ℝ) (σ2 : ℝ) (hσ : 0 < σ2)
    (hzero : ∀ i, P.mean (ε i) = 0)
    (hpair : ∀ i j, P.mean (fun ω => ε i ω * ε j ω) =
      if i = j then σ2 else 0)
    (hnormal : ∀ i, b i = ∑ j, c j * X i j)
    (hrep : ∀ j, ∑ i, b i * X i j = t j)
    (hrepA : ∀ j, ∑ i, a i * X i j = t j) :
    P.mean (estimate b (fun i ω => (∑ j, X i j * β j) + ε i ω)) =
        ∑ j, t j * β j ∧
    P.variance (estimate b (fun i ω => (∑ j, X i j * β j) + ε i ω)) ≤
      P.variance (estimate a (fun i ω => (∑ j, X i j * β j) + ε i ω)) := by
  constructor
  · exact unbiased P b X β t ε hzero hrep
  · apply gauss_markov_observation_efficiency P X a b (fun i => ∑ j, X i j * β j)
        c ε σ2 (le_of_lt hσ) hzero hpair hnormal
    intro j
    rw [hrepA j, hrep j]

end StatadicFormal.Estimation.Linear
