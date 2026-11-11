module

public import StatadicFormal.Probability.FiniteExtensions
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! Exact moments of the sample mean under independent empirical resampling. -/
@[expose] public section
noncomputable section
open StatadicFormal.Probability
namespace StatadicFormal.Inference

variable {ι : Type*} [Fintype ι]

/-- Expectation of a function of the accumulated sum after independent draws
from the same finite law. A row may be drawn again at every step. -/
def iidSumExpectation (P : FiniteLaw ι) (z : ι → ℝ) : ℕ → ℝ → (ℝ → ℝ) → ℝ
  | 0, s, f => f s
  | n + 1, s, f => P.mean (fun i => iidSumExpectation P z n (s + z i) f)

private theorem centered_sum_moments (P : FiniteLaw ι) (z : ι → ℝ) (hz : P.mean z = 0)
    (n : ℕ) (s : ℝ) :
    iidSumExpectation P z n s id = s ∧
    iidSumExpectation P z n s (fun v => v ^ 2) =
      s ^ 2 + n * P.mean (fun i => z i ^ 2) := by
  induction n generalizing s with
  | zero => simp [iidSumExpectation]
  | succ n ih =>
    constructor
    · simp only [iidSumExpectation, (ih _).1]
      rw [P.mean_add, P.mean_const, hz, add_zero]
    · simp only [iidSumExpectation, (ih _).2]
      have expand : (fun i => (s + z i) ^ 2) =
          (fun i => s ^ 2 + (2 * s) * z i + z i ^ 2) := by funext i; ring
      rw [P.mean_add, expand, P.mean_add, P.mean_add, P.mean_smul, hz]
      simp only [P.mean_const, Nat.cast_succ]
      ring

/-- Conditional mean and variance of the resampled mean. Uniform row masses
give Efron's empirical bootstrap, including repeated observed values. -/
@[blueprint "STATADIC-BOOTSTRAP-001"
  (statement := /-- Fix observations $x_i$ with normalized row masses $p_i$.
  Draw $m>0$ rows independently with replacement. Set
  $\mu=\sum p_ix_i$ and $v=\sum p_i(x_i-\mu)^2$. Then
  $E_*(\bar X^*-\mu)=0$ and
  $E_*[(\bar X^*-\mu)^2]=v/m$. Uniform masses give the empirical law. -/)
  (proof := /-- Each draw contributes $z_i=(x_i-\mu)/m$ to the centered mean.
  Its expectation is zero and its second moment is $v/m^2$.
  Conditional on an accumulated sum $s$, expand $(s+z_i)^2$.
  Averaging removes the mixed term. Induction over the independent draws
  gives $s^2+m v/m^2$ at $s=0$. -/)]
theorem empirical_bootstrap_mean_moments (P : FiniteLaw ι) (x : ι → ℝ)
    (m : ℕ) (hm : 0 < m) :
    let z := fun i => (x i - P.mean x) / (m : ℝ)
    iidSumExpectation P z m 0 id = 0 ∧
      iidSumExpectation P z m 0 (fun s => s ^ 2) = P.variance x / m := by
  let z := fun i => (x i - P.mean x) / (m : ℝ)
  have hz : P.mean z = 0 := by
    simp only [z, div_eq_inv_mul, P.mean_smul, P.research_mean_center, mul_zero]
  have hv : P.mean (fun i => z i ^ 2) = ((m : ℝ)⁻¹) ^ 2 * P.variance x := by
    simp only [z, div_eq_inv_mul, mul_pow, P.mean_smul, FiniteLaw.variance]
  have h := centered_sum_moments P z hz m 0
  refine ⟨h.1, h.2.trans ?_⟩
  have hne : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  rw [hv]
  field_simp
  ring

/-- The unconditional bias of the empirical variance of the mean can occur
already for two iid observations drawn from a fair Bernoulli population. -/
@[blueprint "STATADIC-BOOTSTRAP-002"
  (statement := /-- For two independent fair Bernoulli observations, the
  expected ideal empirical bootstrap variance of their mean is $1/16$.
  The actual sampling variance of the mean is $1/8$. -/)
  (proof := /-- The four observed pairs have equal probabilities. Their
  conditional bootstrap variances are $0,1/8,1/8,0$ since two empirical
  draws are made. Their sample means are $0,1/2,1/2,1$, with mean $1/2$
  and variance $1/8$. -/)]
theorem two_row_bootstrap_bias :
    let bit : FiniteLaw Bool := ⟨fun _ => 1 / 2, fun _ => by norm_num, by norm_num⟩
    let x := fun i : Bool => if i then (1 : ℝ) else 0
    let data := fun i j (k : Bool) => if k then x i else x j
    let v := fun i j => bit.variance (data i j) / 2
    bit.mean (fun i => bit.mean (v i)) = 1 / 16 ∧
      bit.mean (fun i => bit.mean (fun j => ((x i + x j) / 2 - 1 / 2) ^ 2)) = 1 / 8 := by
  norm_num [FiniteLaw.mean, FiniteLaw.variance, Fintype.sum_bool]

end StatadicFormal.Inference
