module

public import StatadicFormal.Probability.FiniteExtensions
public import Architect
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum

/-!
# Finite-law U/V pair reductions and MMD-squared bias

Source: Gretton et al. (2012), JMLR 13, 723–773, Lemma 6 / eq. (3), p. 728;
the empirical-estimator argument in its proof, p. 729, and eq. (5), p. 729.
Unequal sample sizes and all cross-sample terms are retained. No rooting of a
signed U-statistic is allowed or justified. Pair-moment hypotheses are explicit:
this file does not derive them from iid sampling, construct RKHS embeddings,
prove characteristic-kernel separation, or verify STATADIC floating-point code.
The V-bias identity is a derived diagonal-counting calculation, not a separate
numbered theorem of the paper. Certification status comes from the native audit.
-/
@[expose] public section
noncomputable section
open scoped BigOperators
namespace StatadicFormal.Discrepancy.MMD
open StatadicFormal.Probability
variable {Ω ι κ : Type*} [Fintype Ω] [Fintype ι] [Fintype κ]

/-- V average includes the within-sample diagonal. -/
def pairV (K : ι → ι → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ i, ∑ j, K i j ω) / (Fintype.card ι : ℝ) ^ 2

/-- U average removes the diagonal. Requires sample size greater than one. -/
def pairU (K : ι → ι → Ω → ℝ) (ω : Ω) : ℝ :=
  ((∑ i, ∑ j, K i j ω) - ∑ i, K i i ω) /
    ((Fintype.card ι : ℝ) * ((Fintype.card ι : ℝ) - 1))

/-- Cross average uses all observations in both groups. -/
def crossAverage (K : ι → κ → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ i, ∑ j, K i j ω) /
    ((Fintype.card ι : ℝ) * (Fintype.card κ : ℝ))

private theorem mean_div (P : FiniteLaw Ω) (f : Ω → ℝ) (c : ℝ) :
    P.mean (fun ω => f ω / c) = P.mean f / c := by
  simpa [div_eq_mul_inv, mul_comm] using P.mean_smul c⁻¹ f

/-- Expected within-diagonal sum. -/
theorem research_diagonal_mean (P : FiniteLaw Ω) (K : ι → ι → Ω → ℝ)
    (δ : ℝ) (h : ∀ i, P.mean (K i i) = δ) :
    P.mean (fun ω => ∑ i, K i i ω) = (Fintype.card ι : ℝ) * δ := by
  rw [P.mean_sum]
  simp [h]

/-- Expected full pair sum from separate diagonal and off-diagonal moments. -/
theorem research_pair_sum_mean [DecidableEq ι] (P : FiniteLaw Ω)
    (K : ι → ι → Ω → ℝ) (δ μ : ℝ)
    (h : ∀ i j, P.mean (K i j) = if i = j then δ else μ) :
    P.mean (fun ω => ∑ i, ∑ j, K i j ω) =
      (Fintype.card ι : ℝ) * ((Fintype.card ι : ℝ) * μ + (δ - μ)) := by
  rw [P.mean_sum]
  simp_rw [P.mean_sum, h]
  have heq : ∀ i j : ι, (if i = j then δ else μ) =
      μ + if i = j then δ - μ else 0 := by
    intro i j
    split_ifs <;> ring
  simp_rw [heq, Finset.sum_add_distrib]
  simp
  ring

/-- Unbiased U pair average from supplied common diagonal and off-diagonal moments. -/
theorem research_pairU_unbiased [DecidableEq ι] (P : FiniteLaw Ω)
    (K : ι → ι → Ω → ℝ) (δ μ : ℝ)
    (h : ∀ i j, P.mean (K i j) = if i = j then δ else μ)
    (hn : (Fintype.card ι : ℝ) ≠ 0)
    (hn1 : (Fintype.card ι : ℝ) - 1 ≠ 0) :
    P.mean (pairU K) = μ := by
  unfold pairU
  rw [mean_div, P.mean_sub, research_pair_sum_mean P K δ μ h,
    research_diagonal_mean P K δ (fun i => by simpa using h i i)]
  field_simp [hn, hn1]
  ring

/-- Exact V pair bias is the diagonal minus off-diagonal moment divided by n. -/
theorem research_pairV_mean [DecidableEq ι] (P : FiniteLaw Ω)
    (K : ι → ι → Ω → ℝ) (δ μ : ℝ)
    (h : ∀ i j, P.mean (K i j) = if i = j then δ else μ)
    (hn : (Fintype.card ι : ℝ) ≠ 0) :
    P.mean (pairV K) = μ + (δ - μ) / (Fintype.card ι : ℝ) := by
  unfold pairV
  rw [mean_div, research_pair_sum_mean P K δ μ h]
  field_simp [hn]

/-- Cross average expectation from common cross moments. -/
theorem research_cross_mean (P : FiniteLaw Ω) (K : ι → κ → Ω → ℝ)
    (μ : ℝ) (h : ∀ i j, P.mean (K i j) = μ)
    (hn : (Fintype.card ι : ℝ) ≠ 0) (hm : (Fintype.card κ : ℝ) ≠ 0) :
    P.mean (crossAverage K) = μ := by
  unfold crossAverage
  rw [mean_div, P.mean_sum]
  simp_rw [P.mean_sum, h]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp [hn, hm]

/-- The unequal-sample MMD² U estimator. It can take negative sample values. -/
def mmdU (XX : ι → ι → Ω → ℝ) (YY : κ → κ → Ω → ℝ)
    (XY : ι → κ → Ω → ℝ) (ω : Ω) : ℝ :=
  pairU XX ω + pairU YY ω - 2 * crossAverage XY ω

/-- The MMD² V estimator. A kernel PSD assumption is needed for nonnegativity. -/
def mmdV (XX : ι → ι → Ω → ℝ) (YY : κ → κ → Ω → ℝ)
    (XY : ι → κ → Ω → ℝ) (ω : Ω) : ℝ :=
  pairV XX ω + pairV YY ω - 2 * crossAverage XY ω

@[blueprint "STATADIC-MMD-001"
  (statement := /-- On a finite probability law, let real pair arrays represent
  groups of sizes $m,n\geq2$. Assume common within-diagonal means
  $\delta_X,\delta_Y$, within off-diagonal means
  $\mu_X,\mu_Y$, and cross mean $\mu_{XY}$. The off-diagonal U estimator with
  all cross terms has expectation $\mu_X+\mu_Y-2\mu_{XY}$. -/)
  (proof := /-- Linearity of finite expectation and the $m(m-1)$ and $n(n-1)$
  off-diagonal counts give the result; see Gretton et al., Lemma 6, eq. (3),
  p. 728, and its empirical-estimator proof step, p. 729.
  The sampling assumptions that imply these moments are not derived here. -/)]
theorem research_mmdU_unbiased [DecidableEq ι] [DecidableEq κ]
    (P : FiniteLaw Ω) (XX : ι → ι → Ω → ℝ) (YY : κ → κ → Ω → ℝ)
    (XY : ι → κ → Ω → ℝ) (δX δY μX μY μXY : ℝ)
    (hx : ∀ i j, P.mean (XX i j) = if i = j then δX else μX)
    (hy : ∀ i j, P.mean (YY i j) = if i = j then δY else μY)
    (hxy : ∀ i j, P.mean (XY i j) = μXY)
    (hm : (Fintype.card ι : ℝ) ≠ 0) (hm1 : (Fintype.card ι : ℝ)-1 ≠ 0)
    (hn : (Fintype.card κ : ℝ) ≠ 0) (hn1 : (Fintype.card κ : ℝ)-1 ≠ 0) :
    P.mean (mmdU XX YY XY) = μX + μY - 2 * μXY := by
  unfold mmdU
  rw [P.mean_sub, P.mean_add, P.mean_smul,
    research_pairU_unbiased P XX δX μX hx hm hm1,
    research_pairU_unbiased P YY δY μY hy hn hn1,
    research_cross_mean P XY μXY hxy hm hn]

/-- Supporting diagonal-counting identity for the V reduction in equation (5).
This does not establish the sign of the bias, PSD, or distribution separation. -/
theorem research_mmdV_bias [DecidableEq ι] [DecidableEq κ]
    (P : FiniteLaw Ω) (XX : ι → ι → Ω → ℝ) (YY : κ → κ → Ω → ℝ)
    (XY : ι → κ → Ω → ℝ) (δX δY μX μY μXY : ℝ)
    (hx : ∀ i j, P.mean (XX i j) = if i = j then δX else μX)
    (hy : ∀ i j, P.mean (YY i j) = if i = j then δY else μY)
    (hxy : ∀ i j, P.mean (XY i j) = μXY)
    (hm : (Fintype.card ι : ℝ) ≠ 0) (hn : (Fintype.card κ : ℝ) ≠ 0) :
    P.bias (mmdV XX YY XY) (μX + μY - 2 * μXY) =
      (δX - μX) / (Fintype.card ι : ℝ) +
      (δY - μY) / (Fintype.card κ : ℝ) := by
  unfold FiniteLaw.bias mmdV
  rw [P.mean_sub, P.mean_add, P.mean_smul,
    research_pairV_mean P XX δX μX hx hm,
    research_pairV_mean P YY δY μY hy hn,
    research_cross_mean P XY μXY hxy hm hn]
  ring

end StatadicFormal.Discrepancy.MMD
