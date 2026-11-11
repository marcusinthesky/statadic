module

public import StatadicFormal.Inference.BlockLength
public import StatadicFormal.Probability.ContinuousMapping
public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Tactic.FieldSimp

/-! Corrected block length, including the two lower bounds used by the implementation. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory
open scoped Topology
namespace StatadicFormal.Inference

/-- Positive real cube root in the interior Patton–Politis–White formula. -/
def blockScale (G D : ℝ) : ℝ := (2 * G^2 / D) ^ ((3 : ℝ)⁻¹)

/-- Native post-pilot expression for sample size N, variance factor c, and denominator floor ε. -/
def selectedLength (N c ε G S : ℝ) : ℝ :=
  max (|blockScale G (max (c * S^2) ε) * N ^ ((3 : ℝ)⁻¹)|) 1

private theorem scale_pos {G D : ℝ} (hG : G ≠ 0) (hD : 0 < D) : 0 < blockScale G D :=
  Real.rpow_pos_of_pos (div_pos (mul_pos (by norm_num) (sq_pos_of_ne_zero hG)) hD) _

/-- The exact cube-root expression attains the unique minimum of leading AMSE. -/
@[blueprint "STATADIC-BLOCKLENGTH-002"
  (statement := /-- For $N,D>0$ and $G\ne 0$, the displayed positive
  length $b_*=(2G^2/D)^{1/3}N^{1/3}$ uniquely minimizes
  $G^2/b^2+Db/N$ over all positive real lengths. -/)
  (proof := /-- Cube the two positive factors of $b_*$ and simplify.
  The resulting balance $Db_*^3=2NG^2$ gives uniqueness by the
  factored risk-gap identity. -/)]
theorem block_oracle (G D N : ℝ) (hG : G ≠ 0) (hD : 0 < D) (hN : 0 < N) :
    let b := blockScale G D * N ^ ((3 : ℝ)⁻¹)
    0 < b ∧ ∀ t > 0, blockRisk G D N b ≤ blockRisk G D N t ∧
      (blockRisk G D N t = blockRisk G D N b ↔ t = b) := by
  have hp := mul_pos (scale_pos hG hD) (Real.rpow_pos_of_pos hN (3 : ℝ)⁻¹)
  refine ⟨hp, fun t ht => block_risk_minimum G D N _ t hD hN hp ht ?_⟩
  have cube (x : ℝ) (hx : 0 ≤ x) : (x ^ ((3 : ℝ)⁻¹)) ^ (3 : ℕ) = x := by
    simpa only [Nat.cast_ofNat] using Real.rpow_inv_natCast_pow (n := 3) hx (by decide)
  rw [mul_pow, blockScale, cube _ (by positivity), cube _ hN.le]
  field_simp

private theorem normalized_length (N c ε G S : ℝ) (hN : 0 < N) (hε : 0 < ε) :
    selectedLength N c ε G S / N ^ ((3 : ℝ)⁻¹) =
      max (blockScale G (max (c * S^2) ε)) (N ^ ((3 : ℝ)⁻¹))⁻¹ := by
  have h : 0 < N ^ ((3 : ℝ)⁻¹) := Real.rpow_pos_of_pos hN _
  have hD := hε.trans_le (le_max_right (c * S^2) ε)
  have hb : 0 ≤ blockScale G (max (c * S^2) ε) := Real.rpow_nonneg (by positivity) _
  rw [selectedLength, ← max_div_div_right h.le, abs_of_nonneg (mul_nonneg hb h.le),
    mul_div_cancel_right₀ _ h.ne', one_div]

/-- The corrected selector converges to the oracle ratio when the pilot and floor permit it. -/
@[blueprint "STATADIC-BLOCKLENGTH-003"
  (statement := /-- Suppose the joint lag pilots $(\hat G_N,\hat S_N)$
  converge in probability to $(G,S)$ with $G\ne0$, $D=cS^2>0$.
  Use the denominator floor $\epsilon>0$ and length floor one. The ratio
  of selected length to the interior oracle converges in probability
  to $(2G^2/\max(D,\epsilon))^{1/3}/(2G^2/D)^{1/3}$. In particular it
  converges to one for $D\ge\epsilon$. -/)
  (proof := /-- Divide the selected length by $N^{1/3}$; its lower
  bound becomes $N^{-1/3}\to0$. The floored denominator is positive
  and the cube-root scale is continuous there. On almost surely
  convergent subsubsequences of the pilots apply continuity and divide
  by the positive oracle scale. -/)]
theorem block_selector_ratio {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsFiniteMeasure μ]
    (X : ℕ → Ω → ℝ × ℝ) (G S c ε : ℝ) (hG : G ≠ 0) (hD : 0 < c * S^2) (hε : 0 < ε)
    (hX : TendstoInMeasure μ X atTop (fun _ => (G,S)))
    (hm : ∀ n, AEMeasurable (X n) μ) :
    let ratio := fun (n : ℕ) ω => selectedLength (n+1) c ε (X n ω).1 (X n ω).2 /
      (blockScale G (c * S^2) * ((n : ℝ)+1) ^ ((3 : ℝ)⁻¹))
    let r := blockScale G (max (c * S^2) ε) / blockScale G (c * S^2)
    TendstoInMeasure μ ratio atTop (fun _ => r) ∧ (ε ≤ c * S^2 → r = 1) := by
  let f := fun x : ℝ × ℝ => blockScale x.1 (max (c * x.2^2) ε)
  have hf : Continuous f := by
    apply (Real.continuous_rpow_const (by norm_num : 0 ≤ (3 : ℝ)⁻¹)).comp
    exact (continuous_const.mul (continuous_fst.pow 2)).div
      ((continuous_const.mul (continuous_snd.pow 2)).max continuous_const)
      (fun x => (hε.trans_le (le_max_right _ _)).ne')
  let a := blockScale G (c * S^2)
  have ha : 0 < a := scale_pos hG hD
  let N := fun (n : ℕ) => (n : ℝ) + 1
  have hN (n : ℕ) : 0 < N n := by dsimp [N]; positivity
  have normed (n : ℕ) (x : ℝ × ℝ) : selectedLength (N n) c ε x.1 x.2 /
      (a * (N n) ^ ((3 : ℝ)⁻¹)) = max (f x) ((N n) ^ ((3 : ℝ)⁻¹))⁻¹ / a := by
    rw [mul_comm a, ← div_div, normalized_length _ _ _ _ _ (hN n) hε]
  have hinv : Tendsto (fun n => (N n ^ ((3 : ℝ)⁻¹))⁻¹) atTop (𝓝 0) :=
    ((tendsto_rpow_atTop (by norm_num : 0 < (3 : ℝ)⁻¹)).comp
      (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)).inv_tendsto_atTop
  have hpos : 0 < f (G,S) := scale_pos hG (hε.trans_le (le_max_right _ _))
  constructor
  · change TendstoInMeasure μ (fun n ω => selectedLength (N n) c ε (X n ω).1 (X n ω).2 /
      (a * (N n) ^ ((3 : ℝ)⁻¹))) atTop _
    simp_rw [normed]
    apply (exists_seq_tendstoInMeasure_atTop_iff (fun n =>
      (((hf.measurable.comp_aemeasurable (hm n)).max aemeasurable_const).div_const a).aestronglyMeasurable)).mpr
    intro ns hns
    obtain ⟨ks, hks, hl⟩ := (hX.comp hns.tendsto_atTop).exists_seq_tendsto_ae
    refine ⟨ks, hks, ?_⟩
    filter_upwards [hl] with ω hω
    have h := ((hf.tendsto _ |>.comp hω).max (hinv.comp (hns.tendsto_atTop.comp hks.tendsto_atTop))).div_const a
    rw [max_eq_left hpos.le] at h
    exact h
  · intro h
    rw [max_eq_left h]
    exact div_self ha.ne'


/-- A sufficient lag-level consistency condition for both flat-top pilots. -/
@[blueprint "STATADIC-BLOCKLENGTH-004"
  (statement := /-- Fix a finite set of retained signed lags. Suppose that
  each empirical autocorrelation converges to its population value and the
  taper stays between zero and one. On lags with nonzero population
  autocorrelation assume the taper converges to one. Both tapered pilot sums,
  with multipliers $|k|$ and $1$, converge to their untapered population
  sums. These are the full pilots when the retained lags contain the
  population support. -/)
  (proof := /-- At nonzero population lags multiply the two limits.
  At zero population lags bound the absolute tapered estimate by the
  absolute sample autocorrelation; the taper may continue to fluctuate.
  Add the limits over the finite retained lag set. -/)]
theorem finite_lag_pilots (lags : Finset ℤ) (r : ℤ → ℝ) (rh w : ℕ → ℤ → ℝ)
    (hr : ∀ k ∈ lags, Tendsto (fun n => rh n k) atTop (𝓝 (r k)))
    (hw : ∀ n k, 0 ≤ w n k ∧ w n k ≤ 1)
    (hkeep : ∀ k ∈ lags, r k ≠ 0 → Tendsto (fun n => w n k) atTop (𝓝 1)) :
    Tendsto (fun n => (∑ k ∈ lags, |(k : ℝ)| * (w n k * rh n k),
      ∑ k ∈ lags, w n k * rh n k)) atTop
      (𝓝 (∑ k ∈ lags, |(k : ℝ)| * r k, ∑ k ∈ lags, r k)) := by
  have hlag k (hk : k ∈ lags) :
      Tendsto (fun n => w n k * rh n k) atTop (𝓝 (r k)) := by
    by_cases h : r k = 0
    · rw [h]
      apply (tendsto_zero_iff_abs_tendsto_zero _).mpr
      apply squeeze_zero (fun _ => abs_nonneg _) (fun n => ?_)
        (by simpa [h] using (hr k hk).abs)
      rw [abs_mul, abs_of_nonneg (hw n k).1]
      exact mul_le_of_le_one_left (abs_nonneg _) (hw n k).2
    · simpa only [one_mul] using (hkeep k hk h).mul (hr k hk)
  exact (tendsto_finsetSum _ fun k h => (hlag k h).const_mul _).prodMk_nhds
    (tendsto_finsetSum _ hlag)

/-- The lower-bound branch is exact when the lag-weighted pilot vanishes. -/
@[blueprint "STATADIC-BLOCKLENGTH-005"
  (statement := /-- For zero lag-weighted pilot $\hat G=0$, the selector
  returns one for every sample size. In particular, a search restricted
  to lag zero can never yield an interior $N^{1/3}$ length. -/)
  (proof := /-- At lag zero the multiplier $|k|$ vanishes. The cube-root
  term is zero, and taking the maximum with one gives the stated result. -/)]
theorem block_zero_pilot (N c ε S : ℝ) :
    selectedLength N c ε 0 S = 1 ∧
      (∑ k ∈ ({0} : Finset ℤ), |(k : ℝ)| * S) = 0 := by
  norm_num [selectedLength, blockScale, Real.zero_rpow]

end StatadicFormal.Inference
