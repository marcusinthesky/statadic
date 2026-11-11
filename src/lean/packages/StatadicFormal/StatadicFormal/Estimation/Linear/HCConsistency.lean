module

public import StatadicFormal.Estimation.Linear.HCLeverage
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Diffuse contrasts and vanishing leverage give a common probability limit
for HC0, the residual-degrees-of-freedom correction, HC2 and HC3. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory
open scoped Topology
namespace StatadicFormal.Estimation.Linear

/-- HC multiplier on a squared coefficient-influence contrast. -/
def hcFactor (h : ℝ) (k : ℕ) : ℝ := 1 / (1-h)^k

private lemma factor_bounds {h H : ℝ} (hh : 0 ≤ h) (hle : h ≤ H) (hH : H < 1) (k : ℕ) :
    1 ≤ hcFactor h k ∧ hcFactor h k ≤ hcFactor H k := by
  have hp : 0 < 1-H := sub_pos.mpr hH
  have hp' : 0 < 1-h := sub_pos.mpr (hle.trans_lt hH)
  constructor
  · exact (one_le_div (pow_pos hp' _)).mpr (pow_le_one₀ hp'.le (by linarith))
  · exact one_div_le_one_div_of_le (pow_pos hp _) (pow_le_pow_left₀ hp.le (by linarith) _)

/-- A common upper bound on the row leverage controls the inflation of every
nonnegative contrast contribution in the fitted covariance. -/
@[blueprint "STATADIC-HC-003"
  (statement := /-- Let $s_i\ge0$ be squared fitted influence contrasts and
  $0\le h_i\le H<1$. For $k=0,1,2$, the adjusted sum
  $A_k=\sum_i s_i/(1-h_i)^k$ lies between $A_0$ and $A_0/(1-H)^k$.
  HC2 and HC3 use $k=1,2$; HC1 uses $k=1$ with constant $h_i=p/N$. -/)
  (proof := /-- Bound each inverse leverage factor between one and
  $(1-H)^{-k}$. Multiply by the nonnegative squared contribution and sum.
  If $h_i$ approaches one the upper bound can grow without limit. -/)]
theorem hc_inflation_bounds {ι : Type*} [Fintype ι] (s h : ι → ℝ) (H : ℝ) (k : ℕ)
    (hs : ∀ i, 0 ≤ s i) (hh : ∀ i, 0 ≤ h i ∧ h i ≤ H) (hH : H < 1) :
    (∑ i, s i) ≤ ∑ i, hcFactor (h i) k * s i ∧
      (∑ i, hcFactor (h i) k * s i) ≤ hcFactor H k * ∑ i, s i := by
  refine ⟨Finset.sum_le_sum fun i _ => le_mul_of_one_le_left (hs i)
    (factor_bounds (hh i).1 (hh i).2 hH k).1, ?_⟩
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right
    (factor_bounds (hh i).1 (hh i).2 hH k).2 (hs i)

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

/-- L1 consistency of HC0/HC1/HC2/HC3 for a fixed deterministic
contrast under fourth-moment orthogonality, diffuse weights and low leverage. -/
@[blueprint "STATADIC-HC-004"
  (statement := /-- For row weights $0\le w_{ni}\le a_n\to0$ with bounded
  sum, suppose $|u_{ni}|\le K$, the squared errors have zero cross-covariance
  and bounded variance, and $E d_{ni}^2\le q_n\to0$ for the removed
  fitted errors. If $0\le h_{ni}\le H_n<1$ and $H_n\to0$, then every
  fixed $k=0,1,2$ variance estimate converges in mean absolute error to
  its sampling target $\tau_n=\sum_i w_{ni}E u_{ni}^2$.
  Hence its estimation error tends to zero in probability. -/)
  (proof := /-- Diffuse weights concentrate the oracle squared-error sum,
  and the prediction-error bound makes residual replacement negligible.
  The finite leverage inequality places the corrected sum between HC0 and
  $(1-H_n)^{-k}$ times HC0. The multiplier tends to one and $\tau_n$
  remains bounded. Markov's inequality converts the mean absolute bound
  into convergence in probability. -/)]
theorem hc_contrast_consistent (u d : (n : ℕ) → Fin (n+1) → Ω → ℝ)
    (w v h : (n : ℕ) → Fin (n+1) → ℝ) (a q H : ℕ → ℝ) (W F K : ℝ) (k : ℕ)
    (ha : ∀ n, 0 ≤ a n) (hq : ∀ n, 0 ≤ q n) (hF : 0 ≤ F) (hK : 0 ≤ K)
    (hw : ∀ n i, 0 ≤ w n i ∧ w n i ≤ a n) (hs : ∀ n, ∑ i, w n i ≤ W)
    (hH : ∀ n, H n < 1) (hh : ∀ n i, 0 ≤ h n i ∧ h n i ≤ H n)
    (hu : ∀ n i, MemLp (u n i) 2 P) (hu2 : ∀ n i, MemLp (fun ω => (u n i ω)^2) 2 P)
    (hub : ∀ n i ω, |u n i ω| ≤ K) (hd : ∀ n i, MemLp (d n i) 2 P)
    (hdq : ∀ n i, ∫ ω, (d n i ω)^2 ∂P ≤ q n)
    (hc : ∀ n i j, ProbabilityTheory.covariance (fun ω => (u n i ω)^2)
      (fun ω => (u n j ω)^2) P = if i = j then v n i else 0)
    (hv : ∀ n i, v n i ≤ F)
    (halim : Tendsto a atTop (𝓝 0)) (hqlim : Tendsto q atTop (𝓝 0))
    (hHlim : Tendsto H atTop (𝓝 0)) :
    let A := fun n ω => ∑ i, hcFactor (h n i) k * (w n i * (u n i ω-d n i ω)^2)
    let τ := fun n => ∑ i, w n i * ∫ ω, (u n i ω)^2 ∂P
    Tendsto (fun n => ∫ ω, |A n ω - τ n| ∂P) atTop (𝓝 0) ∧
      TendstoInMeasure P (fun n ω => A n ω - τ n) atTop (fun _ => 0) := by
  let s := fun n (i : Fin (n+1)) ω => w n i * (u n i ω-d n i ω)^2
  let A := fun n ω => ∑ i, hcFactor (h n i) k * s n i ω
  let B := fun n ω => ∑ i, s n i ω
  let τ := fun n => ∑ i, w n i * ∫ ω, (u n i ω)^2 ∂P
  let e := fun n => Real.sqrt (F*a n*W) + W*(2*K*Real.sqrt (q n)+q n)
  let f := fun n => hcFactor (H n) k
  have hint n i : Integrable (s n i) P :=
    (((hu n i).sub (hd n i)).integrable_sq).const_mul _
  have hbI n : Integrable (fun ω => |B n ω - τ n|) P :=
    ((integrable_finsetSum _ (fun i _ => hint n i)).sub (integrable_const _)).norm
  have haI n : Integrable (fun ω => |A n ω - τ n|) P :=
    ((integrable_finsetSum _ (fun i _ => (hint n i).const_mul _)).sub (integrable_const _)).norm
  have he n : ∫ ω, |B n ω - τ n| ∂P ≤ e n :=
    hc0_contrast_error_bound P (u n) (d n) (w n) (v n) (a n) W F K (q n)
      (ha n) hF hK (hq n) (hw n) (hs n) (hu n) (hu2 n) (hub n) (hd n) (hdq n) (hc n) (hv n)
  have ht n : τ n ≤ W*K^2 := by
    calc
      _ ≤ ∑ i, w n i * K^2 := Finset.sum_le_sum fun i _ =>
        mul_le_mul_of_nonneg_left (by
          have bound : ∀ ω, (u n i ω)^2 ≤ K^2 := fun ω => by
            nlinarith [sq_nonneg (K-|u n i ω|), mul_self_le_mul_self (abs_nonneg _) (hub n i ω),
              sq_abs (u n i ω)]
          simpa using integral_mono (hu n i).integrable_sq (integrable_const (K^2)) bound)
          (hw n i).1
      _ ≤ _ := by rw [← Finset.sum_mul]; exact mul_le_mul_of_nonneg_right (hs n) (sq_nonneg _)
  have hf n : 1 ≤ f n := (factor_bounds (hh n 0).1 (hh n 0).2 (hH n) k).1.trans
    (factor_bounds (hh n 0).1 (hh n 0).2 (hH n) k).2
  have hbound n : ∫ ω, |A n ω - τ n| ∂P ≤ f n * e n + (f n-1)*(W*K^2) := by
    calc
      _ ≤ ∫ ω, (f n * |B n ω - τ n| + (f n-1)*(W*K^2)) ∂P := by
        apply integral_mono (haI n) (((hbI n).const_mul _).add (integrable_const _))
        intro ω
        have hb := hc_inflation_bounds (fun i => s n i ω) (h n) (H n) k
          (fun i => mul_nonneg (hw n i).1 (sq_nonneg _)) (hh n) (hH n)
        change B n ω ≤ A n ω ∧ A n ω ≤ f n * B n ω at hb
        change |A n ω - τ n| ≤ f n * |B n ω - τ n| + (f n-1)*(W*K^2)
        apply abs_le.mpr
        constructor
        · nlinarith [neg_abs_le (B n ω-τ n), mul_nonneg (sub_nonneg.mpr (hf n))
            (abs_nonneg (B n ω-τ n)), mul_nonneg (sub_nonneg.mpr (hf n))
            (show 0 ≤ W*K^2 from mul_nonneg ((Finset.sum_nonneg fun i _ => (hw n i).1).trans
              (hs n)) (sq_nonneg _))]
        · nlinarith [mul_le_mul_of_nonneg_left (le_abs_self (B n ω-τ n)) ((zero_le_one).trans
            (hf n)), mul_le_mul_of_nonneg_left (ht n) (sub_nonneg.mpr (hf n))]
      _ ≤ _ := by
        rw [integral_add ((hbI n).const_mul _) (integrable_const _), integral_const_mul]
        simp only [integral_const, probReal_univ, one_smul]
        exact add_le_add_left (mul_le_mul_of_nonneg_left (he n) (zero_le_one.trans (hf n))) _
  have helim : Tendsto e atTop (𝓝 0) := by
    simpa [e] using (Real.continuous_sqrt.continuousAt.tendsto.comp
      (((tendsto_const_nhds (x := F)).mul halim).mul_const W)).add
      ((tendsto_const_nhds (x := W)).mul (((tendsto_const_nhds (x := 2*K)).mul
        (Real.continuous_sqrt.continuousAt.tendsto.comp hqlim)).add hqlim))
  have hflim : Tendsto f atTop (𝓝 1) := by
    simpa [f, hcFactor, Pi.div_def] using (tendsto_const_nhds (x := (1:ℝ))).div
      (((tendsto_const_nhds (x := (1:ℝ))).sub hHlim).pow k) (by simp)
  have main : Tendsto (fun n => ∫ ω, |A n ω - τ n| ∂P) atTop (𝓝 0) := by
    refine squeeze_zero (fun n => integral_nonneg fun _ => abs_nonneg _) hbound ?_
    simpa using (hflim.mul helim).add ((hflim.sub (tendsto_const_nhds (x := (1:ℝ)))).mul_const
      (W*K^2))
  refine ⟨main, ?_⟩
  rw [tendstoInMeasure_iff_measureReal_norm]
  intro ε hε
  apply squeeze_zero (fun _ => measureReal_nonneg) _ (by simpa using main.div_const ε)
  intro n
  simp only [sub_zero, Real.norm_eq_abs]
  change P.real {ω | ε ≤ |A n ω - τ n|} ≤ _
  exact (le_div_iff₀ hε).mpr (by simpa [mul_comm] using
    mul_meas_ge_le_integral_of_nonneg (ae_of_all _ fun ω => abs_nonneg _) (haI n) ε)

end StatadicFormal.Estimation.Linear
