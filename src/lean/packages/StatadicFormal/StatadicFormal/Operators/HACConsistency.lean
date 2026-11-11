module

public import Mathlib.Probability.Moments.Variance
public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
public import Mathlib.Analysis.Normed.Group.Tannery
public import Architect
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Entrywise consistency of the Bartlett long-run covariance under an explicit
fourth-moment dependence envelope. At positive lags the input is the sum of
the forward and reversed score products. Every lag uses the common row count. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory ProbabilityTheory
open scoped Topology BigOperators
namespace StatadicFormal.Operators

/-- Linear taper with width m; the last nonzero lag is m-1. -/
def lagTaper (m h : ℕ) : ℝ := if h < m then 1 - (h : ℝ) / m else 0

private lemma taper_bounds (m h : ℕ) : 0 ≤ lagTaper m h ∧ lagTaper m h ≤ 1 := by
  unfold lagTaper
  split_ifs with hl
  · have hm : 0 < (m : ℝ) := by exact_mod_cast Nat.zero_lt_of_lt hl
    constructor
    · exact sub_nonneg.mpr ((div_le_one hm).mpr (by exact_mod_cast hl.le))
    · exact sub_le_self _ (by positivity)
  · norm_num

private lemma taper_limit (s : ℕ → ℕ) (hs : Tendsto s atTop atTop) (h : ℕ) :
    Tendsto (fun n => lagTaper (s n) h) atTop (𝓝 1) := by
  have hinv := tendsto_inv_atTop_zero.comp (tendsto_natCast_atTop_atTop.comp hs :
    Tendsto (fun n => (s n : ℝ)) atTop atTop)
  have hlim : Tendsto (fun n => 1 - (h : ℝ) / s n) atTop (𝓝 1) := by
    simpa [div_eq_mul_inv] using (tendsto_const_nhds (x := (1 : ℝ))).sub
      ((tendsto_const_nhds (x := (h : ℝ))).mul hinv)
  apply hlim.congr'
  filter_upwards [hs.eventually (eventually_gt_atTop h)] with n hn
  simp [lagTaper, hn]

private lemma pairs_over_n (n h : ℕ) (hn : n ≠ 0) :
    ((n - h : ℕ) : ℝ) / n = lagTaper n h := by
  by_cases hh : h < n
  · simp only [lagTaper, hh, ite_true, Nat.cast_sub hh.le, sub_div, div_self (show (n : ℝ) ≠ 0 by exact_mod_cast hn)]
  · simp [lagTaper, hh, Nat.sub_eq_zero_of_le (le_of_not_gt hh)]

/-- The deterministic expectation kernel tends to the infinite long-run sum. -/
@[blueprint "STATADIC-HAC-003"
  (statement := /-- Write $\gamma_0$ for the contemporaneous score moment and
  $\gamma_h$ for the sum of its two orientations at positive lag. If
  $\sum_h |\gamma_h|<\infty$ and $b_n\to\infty$, then
  $\sum_h (1-h/(b_n+1))_+(1-h/(n+1))_+\gamma_h\to\sum_h\gamma_h$.
  The second factor accounts for the incomplete pairs at the sample edges. -/)
  (proof := /-- Both tapers lie in $[0,1]$ and tend to one at every fixed lag.
  Dominate the summands by the absolutely summable lag moments and pass the
  limit through the series. -/)]
theorem bartlett_population_limit (γ : ℕ → ℝ) (hγ : Summable (fun h => |γ h|))
    (b : ℕ → ℕ) (hb : Tendsto b atTop atTop) :
    Tendsto (fun n => ∑' h, lagTaper (b n + 1) h * lagTaper (n+1) h * γ h)
      atTop (𝓝 (∑' h, γ h)) := by
  have hb' : Tendsto (fun n => b n + 1) atTop atTop :=
    (tendsto_add_atTop_nat 1).comp hb
  apply tendsto_tsum_of_dominated_convergence hγ
  · intro h
    simpa using ((taper_limit _ hb' h).mul (taper_limit _ (tendsto_add_atTop_nat 1) h)).mul
      (tendsto_const_nhds (x := γ h))
  · filter_upwards with n h
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (mul_nonneg (taper_bounds _ _).1
      (taper_bounds _ _).1)]
    exact mul_le_of_le_one_left (abs_nonneg _) (((mul_le_of_le_one_left (taper_bounds _ _).1 (taper_bounds _ _).2).trans (taper_bounds _ _).2))

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

/-- Normalized lag sum for one covariance entry, on n+1 score rows with b retained lags. -/
def bartlettEntry (X : ℕ → ℕ → Ω → ℝ) (n b : ℕ) (ω : Ω) : ℝ :=
  ((n+1 : ℕ) : ℝ)⁻¹ * ∑ h ∈ Finset.range (b+1),
    lagTaper (b+1) h * ∑ t ∈ Finset.range (n+1-h), X h t ω

omit [IsProbabilityMeasure P] in
private lemma entry_memLp (X : ℕ → ℕ → Ω → ℝ) (hX : ∀ h t, MemLp (X h t) 2 P) (n b : ℕ) :
    MemLp (bartlettEntry X n b) 2 P := by
  apply MemLp.const_mul
  apply memLp_finsetSum
  intro h _
  exact (memLp_finsetSum _ (fun t _ => hX h t)).const_mul _

private lemma entry_mean (X : ℕ → ℕ → Ω → ℝ) (γ : ℕ → ℝ)
    (hX : ∀ h t, MemLp (X h t) 2 P) (hm : ∀ h t, ∫ ω, X h t ω ∂P = γ h) (n b : ℕ) :
    (∫ ω, bartlettEntry X n b ω ∂P) =
      ∑' h, lagTaper (b+1) h * lagTaper (n+1) h * γ h := by
  rw [tsum_eq_sum (s := Finset.range (b+1)) (fun h hh => by
    simp [lagTaper, Finset.mem_range.not.mp hh])]
  unfold bartlettEntry
  rw [integral_const_mul, integral_finsetSum _ (fun h _ =>
    ((memLp_finsetSum _ (fun t _ => hX h t)).integrable (by norm_num)).const_mul _), Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro h _
  rw [integral_const_mul, integral_finsetSum _ (fun t _ => (hX h t).integrable (by norm_num))]
  simp only [hm, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  rw [← pairs_over_n _ _ (Nat.succ_ne_zero n)]
  ring

/-- A primitive lag-product covariance-sum restriction gives the stochastic
error bound for the entire estimated HAC, allowing dependence across lags. -/
@[blueprint "STATADIC-HAC-004"
  (statement := /-- Write $X_{h,t}$ for one raw lag product at lag zero and
  the sum of the two oriented score products at positive lag. Let $N=n+1$.
  Assume square integrability and, for all $h,k,t,m$,
  $\sum_{s<m}|\operatorname{Cov}(X_{h,t},X_{k,s})|\le C$.
  The variance of the sample Bartlett entry with bandwidth $b$ and
  common denominator $N$ is at most $C(b+1)^2/N$. -/)
  (proof := /-- Expand the variance into covariances of time-indexed lag
  products. For every pair of lags, the sum of absolute covariances is at
  most $NC$. Multiplying by two taper weights in $[0,1]$ preserves the upper
  bound. Sum over $(b+1)^2$ lag pairs and divide by $N^2$. -/)]
theorem bartlett_entry_variance (X : ℕ → ℕ → Ω → ℝ) (C : ℝ) (hC : 0 ≤ C)
    (hX : ∀ h t, MemLp (X h t) 2 P)
    (hdep : ∀ h k t m, ∑ s ∈ Finset.range m, |covariance (X h t) (X k s) P| ≤ C)
    (n b : ℕ) :
    variance (bartlettEntry X n b) P ≤ C * (((b+1 : ℕ) : ℝ)^2 / (n+1 : ℕ)) := by
  let S := fun h ω => ∑ t ∈ Finset.range (n+1-h), X h t ω
  have hS h : MemLp (S h) 2 P := memLp_finsetSum _ (fun t _ => hX h t)
  have hcov h k : covariance (S h) (S k) P ≤ (n+1 : ℕ) * C := by
    rw [show S h = (fun ω => ∑ t ∈ Finset.range (n+1-h), X h t ω) from rfl,
      show S k = (fun ω => ∑ t ∈ Finset.range (n+1-k), X k t ω) from rfl,
      covariance_fun_sum_fun_sum' (fun i _ => hX h i) (fun j _ => hX k j)]
    calc
      _ ≤ ∑ t ∈ Finset.range (n+1-h), C :=
        Finset.sum_le_sum fun t _ => (Finset.sum_le_sum fun s _ => le_abs_self _).trans
          (hdep h k t _)
      _ ≤ _ := by
        simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.sub_le (n+1) h) hC
  let w := lagTaper (b+1)
  have hpair h k : covariance (fun ω => w h * S h ω) (fun ω => w k * S k ω) P ≤
      (n+1 : ℕ) * C := by
    rw [covariance_const_mul_left, covariance_const_mul_right]
    calc
      _ = (w h * w k) * covariance (S h) (S k) P := by ring
      _ ≤ (w h * w k) * ((n+1 : ℕ) * C) := mul_le_mul_of_nonneg_left (hcov h k)
        (mul_nonneg (taper_bounds _ _).1 (taper_bounds _ _).1)
      _ ≤ _ := mul_le_of_le_one_left (by positivity) (((mul_le_of_le_one_left (taper_bounds _ _).1 (taper_bounds _ _).2).trans (taper_bounds _ _).2))
  unfold bartlettEntry
  rw [variance_const_mul, variance_fun_sum' (fun h _ => (hS h).const_mul _)]
  calc
    _ ≤ (((n+1 : ℕ) : ℝ)⁻¹)^2 *
        ∑ h ∈ Finset.range (b+1), ∑ k ∈ Finset.range (b+1), ((n+1 : ℕ) : ℝ) * C :=
      mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun h _ =>
        Finset.sum_le_sum fun k _ => hpair h k) (sq_nonneg _)
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      have hn : ((n+1 : ℕ) : ℝ) ≠ 0 := by positivity
      field_simp

/-- From primitive moment/dependence conditions to convergence in probability. -/
@[blueprint "STATADIC-HAC-005"
  (statement := /-- With the same bounded lag-product covariance sums, suppose
  $E X_{h,t}=\gamma_h$ at every time, $\sum_h|\gamma_h|<\infty$,
  $b_n\to\infty$, and $(b_n+1)^2/(n+1)\to0$. The raw Bartlett
  estimate converges in probability to $\sum_h\gamma_h$. -/)
  (proof := /-- There are $(N-h)_+$ available time pairs. Taking expectations
  gives the Bartlett taper times the finite-record edge factor. Dominated
  convergence proves the bias vanishes. The fourth-order dependence bound
  forces the estimator variance to zero. Chebyshev's inequality at half the
  requested tolerance and the triangle inequality finish the proof. -/)]
theorem bartlett_entry_consistent (X : ℕ → ℕ → Ω → ℝ) (γ : ℕ → ℝ) (b : ℕ → ℕ)
    (C : ℝ) (hC : 0 ≤ C) (hX : ∀ h t, MemLp (X h t) 2 P)
    (hm : ∀ h t, ∫ ω, X h t ω ∂P = γ h) (hγ : Summable (fun h => |γ h|))
    (hdep : ∀ h k t m, ∑ s ∈ Finset.range m, |covariance (X h t) (X k s) P| ≤ C)
    (hb : Tendsto b atTop atTop)
    (hrate : Tendsto (fun n => ((b n+1 : ℕ) : ℝ)^2 / (n+1 : ℕ)) atTop (𝓝 0)) :
    TendstoInMeasure P (fun n => bartlettEntry X n (b n)) atTop (fun _ => ∑' h, γ h) := by
  let A := fun n => bartlettEntry X n (b n)
  have hmean : Tendsto (fun n => ∫ ω, A n ω ∂P) atTop (𝓝 (∑' h, γ h)) := by
    simpa only [A, entry_mean P X γ hX hm] using bartlett_population_limit γ hγ b hb
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  have hh : 0 < ε/2 := half_pos hε
  have hbound : Tendsto (fun n => ENNReal.ofReal (C * (((b n+1 : ℕ) : ℝ)^2 /
      (n+1 : ℕ)) / (ε/2)^2)) atTop (𝓝 0) := by
    simpa [Function.comp_def] using ENNReal.continuous_ofReal.continuousAt.tendsto.comp
      (((tendsto_const_nhds (x := C)).mul hrate).div_const ((ε/2)^2))
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hbound
    (Eventually.of_forall fun _ => bot_le)
  filter_upwards [(Metric.tendsto_nhds.mp hmean) (ε/2) hh] with n hn
  calc
    _ ≤ P {ω | ε/2 ≤ |A n ω - ∫ x, A n x ∂P|} := by
      apply measure_mono
      intro ω hω
      change ε ≤ |A n ω - ∑' h, γ h| at hω
      change ε/2 ≤ _
      have htri := abs_add_le (A n ω - ∫ x, A n x ∂P) ((∫ x, A n x ∂P) - ∑' h, γ h)
      rw [sub_add_sub_cancel] at htri
      rw [Real.dist_eq] at hn
      linarith
    _ ≤ _ := (meas_ge_le_variance_div_sq (entry_memLp P X hX _ _) hh).trans
      (ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right
        (bartlett_entry_variance P X C hC hX hdep _ _) (sq_nonneg _)))

end StatadicFormal.Operators
