module

public import Mathlib.Probability.Moments.Variance
public import Architect
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! A probability-scale accuracy bound for randomized integration replicates. -/
@[expose] public section
noncomputable section
open MeasureTheory ProbabilityTheory
namespace StatadicFormal.Optimization

/-- Replicate mean-square accuracy and tail probability under explicit sampling hypotheses. -/
@[blueprint "STATADIC-INTEGRATION-002"
  (statement := /-- Suppose $R>0$ probability-scale replicate estimates have common
  expectation $p$, common finite variance $v$, and zero covariance across
  distinct replicates. Their average $A$ has mean $p$, variance $v/R$, and
  $\Pr(|A-p|\ge t)\le v/(Rt^2)$ for $t>0$. -/)
  (proof := /-- Linearity preserves the common expectation. Expand the variance
  of the sum as a double sum of covariances. Only the $R$ diagonal terms survive.
  Divide by $R^2$ and apply Chebyshev's inequality to the resulting average. -/)]
theorem replicate_mean_accuracy {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι] [Nonempty ι]
    [DecidableEq ι] (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ι → Ω → ℝ) (p v t : ℝ)
    (ht : 0 < t) (hs : ∀ i, MemLp (X i) 2 μ) (hm : ∀ i, ∫ ω, X i ω ∂μ = p)
    (hc : ∀ i j, covariance (X i) (X j) μ = if i = j then v else 0) :
    let n : ℝ := Fintype.card ι
    let A := fun ω => n⁻¹ * ∑ i, X i ω
    (∫ ω, A ω ∂μ) = p ∧ variance A μ = v / n ∧
      μ {ω | t ≤ |A ω - p|} ≤ ENNReal.ofReal ((v / n) / t ^ 2) := by
  let n : ℝ := Fintype.card ι
  let A := fun ω => n⁻¹ * ∑ i, X i ω
  have hn : n ≠ 0 := (Nat.cast_pos.mpr Fintype.card_pos : (0 : ℝ) < Fintype.card ι).ne'
  have hmean : (∫ ω, A ω ∂μ) = p := by
    simp only [A, integral_const_mul, integral_finsetSum _ (fun i _ => (hs i).integrable
      (by norm_num)), hm, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    change n⁻¹ * (n * p) = p
    field_simp
  have hvar : variance A μ = v / n := by
    dsimp only [A]
    rw [variance_const_mul, variance_fun_sum hs]
    simp only [hc, Finset.sum_ite_eq, Finset.mem_univ, ite_true, Finset.sum_const,
      Finset.card_univ, nsmul_eq_mul]
    change n⁻¹ ^ 2 * (n * v) = v / n
    field_simp
  refine ⟨hmean, hvar, ?_⟩
  have ha : MemLp A 2 μ :=
    (memLp_finsetSum Finset.univ (fun i _ => hs i)).const_mul _
  simpa only [hmean, hvar] using meas_ge_le_variance_div_sq ha ht

end StatadicFormal.Optimization
