module

public import StatadicFormal.Operators.HACConsistency
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-! Effect of replacing a population-centered score with its sample-centered value. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory ProbabilityTheory
open scoped Topology
namespace StatadicFormal.Operators

variable {Ω : Type*}

/-- One scalar contrast's oriented lag product, counting nonzero lags twice. -/
def orientedProduct (z : ℕ → Ω → ℝ) (h t : ℕ) (ω : Ω) : ℝ :=
  (if h = 0 then 1 else 2) * (z t ω * z (t+h) ω)

/-- The sample average has the same denominator as STATADIC's centering step. -/
def recordMean (z : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ((n+1 : ℕ) : ℝ)⁻¹ * ∑ t ∈ Finset.range (n+1), z t ω

private lemma avg_abs_bound (z : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) (K : ℝ)
    (hk : ∀ t < n+1, |z t ω| ≤ K) : |recordMean z n ω| ≤ K := by
  rw [recordMean, abs_mul, abs_of_nonneg (by positivity : 0 ≤ ((n+1 : ℕ) : ℝ)⁻¹)]
  calc
    _ ≤ ((n+1 : ℕ) : ℝ)⁻¹ * ∑ _t ∈ Finset.range (n+1), K :=
      mul_le_mul_of_nonneg_left ((Finset.abs_sum_le_sum_abs _ _).trans
        (Finset.sum_le_sum fun t ht => hk t (Finset.mem_range.mp ht))) (by positivity)
    _ = _ := by simp [nsmul_eq_mul, show (n : ℝ) + 1 ≠ 0 by positivity]

private lemma entry_diff_bound (X Y : ℕ → ℕ → Ω → ℝ) (n b : ℕ) (ω : Ω) (e : ℝ)
    (he : 0 ≤ e) (herr : ∀ h < b+1, ∀ t < n+1-h, |X h t ω - Y h t ω| ≤ e) :
    |bartlettEntry X n b ω - bartlettEntry Y n b ω| ≤ (b+1 : ℕ) * e := by
  have hw h (hh : h ∈ Finset.range (b+1)) : 0 ≤ lagTaper (b+1) h ∧ lagTaper (b+1) h ≤ 1 := by
    have hlt := Finset.mem_range.mp hh
    simp only [lagTaper, hlt, ite_true]
    exact ⟨sub_nonneg.mpr ((div_le_one (by positivity)).mpr (by exact_mod_cast hlt.le)),
      sub_le_self _ (by positivity)⟩
  unfold bartlettEntry
  rw [← mul_sub, ← Finset.sum_sub_distrib, abs_mul, abs_of_nonneg (by positivity :
    0 ≤ ((n+1 : ℕ) : ℝ)⁻¹)]
  calc
    _ ≤ ((n+1 : ℕ) : ℝ)⁻¹ * (∑ _h ∈ Finset.range (b+1), (n+1 : ℕ) * e) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
      intro h hh
      rw [← mul_sub, ← Finset.sum_sub_distrib, abs_mul, abs_of_nonneg (hw h hh).1]
      calc
        _ ≤ |∑ t ∈ Finset.range (n+1-h), (X h t ω - Y h t ω)| :=
          mul_le_of_le_one_left (abs_nonneg _) (hw h hh).2
        _ ≤ ∑ _t ∈ Finset.range (n+1-h), e := (Finset.abs_sum_le_sum_abs _ _).trans
          (Finset.sum_le_sum fun t ht => herr h (Finset.mem_range.mp hh) t (Finset.mem_range.mp ht))
        _ ≤ _ := by
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
          exact mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.sub_le (n+1) h) he
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      field_simp

/-- Finite-record bound for a scalar score contrast, using the actual sample mean. -/
@[blueprint "STATADIC-HAC-006"
  (statement := /-- Suppose every observed scalar score satisfies $|z_t|\le M$.
  Let $H$ denote the raw Bartlett estimate with bandwidth $b$ and let $H_c$
  use $z_t-\bar z$. Then $|H_c-H|\le 6M(b+1)|\bar z|$. -/)
  (proof := /-- The sample mean has absolute value at most $M$. Expanding one
  centered product changes it by at most $3M|\bar z|$, and positive lags have
  two orientations. Each normalized lag contains at most $N$ products and the
  taper never exceeds one, so summation gives the bound. -/)]
theorem centered_bartlett_error (z : ℕ → Ω → ℝ) (n b : ℕ) (ω : Ω) (K : ℝ)
    (hK : 0 ≤ K) (hz : ∀ t < n+1, |z t ω| ≤ K) :
    |bartlettEntry (orientedProduct (fun t x => z t x - recordMean z n x)) n b ω -
      bartlettEntry (orientedProduct z) n b ω| ≤ 6*K*(b+1 : ℕ)*|recordMean z n ω| := by
  let m := recordMean z n ω
  have hm : |m| ≤ K := avg_abs_bound z n ω K hz
  have hbound h (_hh : h < b+1) t (ht : t < n+1-h) :
      |orientedProduct (fun s x => z s x - recordMean z n x) h t ω -
        orientedProduct z h t ω| ≤ 6*K*|m| := by
    have hz₁ := hz t (by omega)
    have hz₂ := hz (t+h) (by omega)
    have hsum : |m - z t ω - z (t+h) ω| ≤ 3*K := by
      linarith [abs_sub (m-z t ω) (z (t+h) ω), abs_sub m (z t ω)]
    have hprod : |(z t ω-m)*(z (t+h) ω-m) - z t ω*z (t+h) ω| ≤ |m| * (3*K) := by
      rw [show (z t ω-m)*(z (t+h) ω-m) - z t ω*z (t+h) ω =
        m*(m-z t ω-z (t+h) ω) by ring, abs_mul]
      exact mul_le_mul_of_nonneg_left hsum (abs_nonneg _)
    unfold orientedProduct
    rw [← mul_sub, abs_mul]
    change |if h = 0 then (1 : ℝ) else 2| * _ ≤ _
    split_ifs <;> norm_num <;> nlinarith [hprod, mul_nonneg (abs_nonneg m) hK]
  exact (entry_diff_bound _ _ _ _ _ _ (by positivity) hbound).trans_eq (by dsimp [m]; ring)

variable [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

private theorem scaled_mean_negligible (z : ℕ → Ω → ℝ) (b : ℕ → ℕ) (K D : ℝ)
    (hD : 0 ≤ D) (hz : ∀ t, MemLp (z t) 2 P) (hmean : ∀ t, ∫ ω, z t ω ∂P = 0)
    (hcov : ∀ t m, ∑ s ∈ Finset.range m, |covariance (z t) (z s) P| ≤ D)
    (hrate : Tendsto (fun n => ((b n+1 : ℕ) : ℝ)^2 / (n+1 : ℕ)) atTop (𝓝 0)) :
    TendstoInMeasure P (fun n ω => (6*K*(b n+1 : ℕ)) * recordMean z n ω)
      atTop (fun _ => 0) := by
  have hm n : (∫ ω, recordMean z n ω ∂P) = 0 := by
    simp [recordMean, integral_const_mul, integral_finsetSum _ (fun i _ =>
      (hz i).integrable (by norm_num)), hmean]
  have hvar n : variance (recordMean z n) P ≤ D / (n+1 : ℕ) := by
    have he : recordMean z n = bartlettEntry (fun _ => z) n 0 := by
      funext ω
      simp [recordMean, bartlettEntry, lagTaper]
    rw [he]
    simpa [div_eq_mul_inv] using bartlett_entry_variance P (fun _ => z) D hD
      (fun _ t => hz t) (fun _ _ t m => hcov t m) n 0
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  have hlim : Tendsto (fun n => ENNReal.ofReal ((6*K)^2*D*
      (((b n+1 : ℕ) : ℝ)^2/(n+1 : ℕ)) / ε^2)) atTop (𝓝 0) := by
    simpa [Function.comp_def] using ENNReal.continuous_ofReal.continuousAt.tendsto.comp
      (((tendsto_const_nhds (x := (6*K)^2*D)).mul hrate).div_const (ε^2))
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hlim (fun _ => bot_le)
  intro n
  let a : ℝ := 6 * K * (b n+1 : ℕ)
  have hL : MemLp (recordMean z n) 2 P := (memLp_finsetSum _ (fun i _ => hz i)).const_mul _
  have hprob := meas_ge_le_variance_div_sq (hL.const_mul a) hε
  simp only [integral_const_mul, hm, mul_zero, sub_zero, variance_const_mul] at hprob
  change P {ω | ε ≤ ‖a * recordMean z n ω - 0‖} ≤ _
  simp only [sub_zero, Real.norm_eq_abs]
  refine hprob.trans (ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right ?_ (sq_nonneg _)))
  calc
    a^2 * variance (recordMean z n) P ≤ a^2 * (D / (n+1 : ℕ)) :=
      mul_le_mul_of_nonneg_left (hvar n) (sq_nonneg _)
    _ = _ := by dsimp [a]; ring

/-- Sample demeaning preserves a zero-mean scalar score contrast's long-run limit. -/
@[blueprint "STATADIC-HAC-007"
  (statement := /-- In addition to the raw Bartlett entry conditions, suppose
  $z_t$ is bounded by $M$, $E z_t=0$ and
  $\sum_{s<m}|\operatorname{Cov}(z_t,z_s)|\le D$ uniformly in $t,m$.
  The Bartlett estimate of $z_t-\bar z$ converges in probability to the
  same long-run variance as the raw estimate. -/)
  (proof := /-- Expanding the variance of $\bar z$ gives a bound $D/N$.
  The rate $(b+1)^2/N\to0$ therefore sends $(b+1)\bar z$ to zero in
  probability. The finite centering bound and the union bound transfer the
  raw-score limit to the sample-centered version. -/)]
theorem centered_bartlett_consistent (z : ℕ → Ω → ℝ) (γ : ℕ → ℝ) (b : ℕ → ℕ)
    (C D K : ℝ) (hC : 0 ≤ C) (hD : 0 ≤ D) (hK : 0 ≤ K)
    (hk : ∀ t ω, |z t ω| ≤ K) (hz : ∀ t, MemLp (z t) 2 P)
    (hz0 : ∀ t, ∫ ω, z t ω ∂P = 0)
    (hc : ∀ t m, ∑ s ∈ Finset.range m, |covariance (z t) (z s) P| ≤ D)
    (hp : ∀ h t, MemLp (orientedProduct z h t) 2 P)
    (hm : ∀ h t, ∫ ω, orientedProduct z h t ω ∂P = γ h)
    (hγ : Summable (fun h => |γ h|))
    (hdep : ∀ h k t m, ∑ s ∈ Finset.range m,
      |covariance (orientedProduct z h t) (orientedProduct z k s) P| ≤ C)
    (hb : Tendsto b atTop atTop)
    (hrate : Tendsto (fun n => ((b n+1 : ℕ) : ℝ)^2 / (n+1 : ℕ)) atTop (𝓝 0)) :
    TendstoInMeasure P (fun n => bartlettEntry
      (orientedProduct (fun t ω => z t ω - recordMean z n ω)) n (b n)) atTop
      (fun _ => ∑' h, γ h) := by
  let A := fun n => bartlettEntry (orientedProduct z) n (b n)
  let B := fun n ω => 6*K*(b n+1 : ℕ)*recordMean z n ω
  have ha := bartlett_entry_consistent P (orientedProduct z) γ b C hC hp hm hγ hdep hb hrate
  have hb' := scaled_mean_negligible P z b K D hD hz hz0 hc hrate
  rw [tendstoInMeasure_iff_norm] at ha hb' ⊢
  intro ε hε
  have hh : 0 < ε/2 := half_pos hε
  have hlim := (ha (ε/2) hh).add (hb' (ε/2) hh)
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le (tendsto_const_nhds (x := (0 : ENNReal)))
    (by simpa only [zero_add] using hlim) (fun _ => bot_le)
  intro n
  refine (measure_mono (t := {ω | ε/2 ≤ ‖A n ω - ∑' h, γ h‖} ∪
    {ω | ε/2 ≤ ‖B n ω - 0‖}) ?_).trans (measure_union_le _ _)
  intro ω hω
  by_contra h
  simp only [Set.mem_union, Set.mem_ofPred_eq, not_or, not_le] at h
  have hdiff := centered_bartlett_error z n (b n) ω K hK (fun t _ => hk t ω)
  have htri := abs_add_le
    (bartlettEntry (orientedProduct (fun t x => z t x-recordMean z n x)) n (b n) ω - A n ω)
    (A n ω - ∑' h, γ h)
  rw [sub_add_sub_cancel] at htri
  change ε ≤ |_ - _| at hω
  change |_ - A n ω| ≤ _ at hdiff
  simp only [B, sub_zero, Real.norm_eq_abs, abs_mul, abs_of_nonneg hK,
    abs_of_nonneg (show (0:ℝ) ≤ 6 by norm_num), abs_of_nonneg (show (0 : ℝ) ≤ (b n+1 : ℕ) by positivity)] at h
  linarith

end StatadicFormal.Operators
