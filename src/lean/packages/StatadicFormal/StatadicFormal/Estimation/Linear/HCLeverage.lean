module

public import Mathlib.Probability.Moments.Variance
public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
public import Architect
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-! Finite moment bounds for independent-row heteroskedastic covariance and
the role of leverage for the HC1--HC3 corrections. The weights are squared
contrast coefficients on the desired inference scale. -/
@[expose] public section
noncomputable section
open Filter MeasureTheory ProbabilityTheory
open scoped Topology
namespace StatadicFormal.Estimation.Linear

variable {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι] [DecidableEq ι]
  (P : Measure Ω) [IsProbabilityMeasure P]

private theorem diagonal_variance (z : ι → Ω → ℝ) (w v : ι → ℝ)
    (hz : ∀ i, MemLp (z i) 2 P)
    (hc : ∀ i j, covariance (z i) (z j) P = if i = j then v i else 0) :
    variance (fun ω => ∑ i, w i * z i ω) P = ∑ i, w i ^ 2 * v i := by
  rw [variance_fun_sum (fun i => (hz i).const_mul (w i))]
  apply Finset.sum_congr rfl
  intro i _
  simp [covariance_const_mul_left, covariance_const_mul_right, hc, mul_ite, pow_two, mul_assoc]

/-- The leverage of a projection row bounds the mean-square change from true
innovation to fitted residual, without equal error variances. -/
@[blueprint "STATADIC-HC-001"
  (statement := /-- Assume zero-mean, square-integrable errors with zero
  pair covariances and variances $v_j\le V$. For a projection row $a$ with
  $\sum_j a_j^2=h$, the removed fitted error $d=\sum_j a_j u_j$ satisfies
  $E d^2=\sum_j a_j^2 v_j\le Vh$. For a symmetric idempotent hat matrix
  the squared row norm is its diagonal leverage. -/)
  (proof := /-- The fitted error has mean zero. Its second moment is its
  variance; expand that variance and use zero off-diagonal covariances.
  Bound each remaining variance and sum. -/)]
theorem hat_error_second_moment (u : ι → Ω → ℝ) (a v : ι → ℝ) (V h : ℝ)
    (hu : ∀ j, MemLp (u j) 2 P) (hm : ∀ j, ∫ ω, u j ω ∂P = 0)
    (hc : ∀ i j, covariance (u i) (u j) P = if i = j then v i else 0)
    (hv : ∀ i, v i ≤ V) (ha : ∑ j, a j ^ 2 = h) :
    let d := fun ω => ∑ j, a j * u j ω
    (∫ ω, (d ω)^2 ∂P) = ∑ j, a j ^ 2 * v j ∧ (∫ ω, (d ω)^2 ∂P) ≤ V*h := by
  let d := fun ω => ∑ j, a j * u j ω
  have hd : MemLp d 2 P := memLp_finsetSum _ (fun j _ => (hu j).const_mul _)
  have hmean : (∫ ω, d ω ∂P) = 0 := by
    simp [d, integral_finsetSum _ (fun j _ => ((hu j).integrable (by norm_num)).const_mul _),
      integral_const_mul, hm]
  have heq := diagonal_variance P u a v hu hc
  change variance d P = _ at heq
  rw [variance_eq_sub hd, hmean, zero_pow (by decide : 2 ≠ 0), sub_zero] at heq
  refine ⟨heq, heq.trans_le ?_⟩
  rw [mul_comm V h, ← ha, Finset.sum_mul]
  exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hv i) (sq_nonneg _)

omit [Fintype ι] [DecidableEq ι] in
private lemma integral_abs_le_sqrt_second (u : Ω → ℝ) (hu : MemLp u 2 P) :
    (∫ ω, |u ω| ∂P) ≤ Real.sqrt (∫ ω, (u ω)^2 ∂P) := by
  have hu' : MemLp (fun ω => |u ω|) 2 P := hu.norm
  have h := variance_nonneg (fun ω => |u ω|) P
  rw [variance_eq_sub hu'] at h
  apply (Real.le_sqrt (integral_nonneg (fun _ => abs_nonneg _))
    (integral_nonneg fun _ => sq_nonneg _)).mpr
  simpa [Pi.pow_apply] using (sub_nonneg.mp h)

/-- Bound the expected error of a fitted-residual HC0 contrast against the
true sampling variance at the same sample size. -/
@[blueprint "STATADIC-HC-002"
  (statement := /-- For squared influence weights $0\le w_i\le a$ of
  total mass at most $W$, assume $|u_i|\le K$, the squared errors have
  zero cross-covariances and variances at most $F$. Suppose the fitted
  adjustment $d_i$ has $E d_i^2\le q$. Then the expected absolute error
  of $\sum_i w_i(u_i-d_i)^2$ around $\sum_i w_iE u_i^2$ is at most
  $\sqrt{FaW}+W(2K\sqrt q+q)$. -/)
  (proof := /-- The variance of the oracle weighted sum of squared errors
  is at most $F\sum w_i^2\le FaW$; Cauchy--Schwarz bounds its mean absolute
  deviation. For each row, expansion changes the square by at most
  $2K|d_i|+d_i^2$. Use $E|d_i|\le\sqrt{E d_i^2}$ and sum. -/)]
theorem hc0_contrast_error_bound (u d : ι → Ω → ℝ) (w v : ι → ℝ) (a W F K q : ℝ)
    (ha : 0 ≤ a) (hF : 0 ≤ F) (hK : 0 ≤ K) (hq : 0 ≤ q)
    (hw : ∀ i, 0 ≤ w i ∧ w i ≤ a) (hsum : ∑ i, w i ≤ W)
    (hu : ∀ i, MemLp (u i) 2 P) (hu2 : ∀ i, MemLp (fun ω => (u i ω)^2) 2 P)
    (hbound : ∀ i ω, |u i ω| ≤ K) (hd : ∀ i, MemLp (d i) 2 P)
    (hdq : ∀ i, ∫ ω, (d i ω)^2 ∂P ≤ q)
    (hc : ∀ i j, covariance (fun ω => (u i ω)^2) (fun ω => (u j ω)^2) P =
      if i = j then v i else 0) (hv : ∀ i, v i ≤ F) :
    let A := fun ω => ∑ i, w i * (u i ω - d i ω)^2
    let τ := ∑ i, w i * ∫ ω, (u i ω)^2 ∂P
    (∫ ω, |A ω - τ| ∂P) ≤ Real.sqrt (F*a*W) + W*(2*K*Real.sqrt q+q) := by
  let V := fun ω => ∑ i, w i * (u i ω)^2
  let τ := ∑ i, w i * ∫ ω, (u i ω)^2 ∂P
  have hV : MemLp V 2 P := memLp_finsetSum _ (fun i _ => (hu2 i).const_mul _)
  have hτ : ∫ ω, V ω ∂P = τ := by
    simp [V, τ, integral_finsetSum _ (fun i _ => ((hu2 i).integrable (by norm_num)).const_mul _),
      integral_const_mul]
  have hsq := diagonal_variance P (fun i ω => (u i ω)^2) w v hu2 hc
  have hvar : variance V P ≤ F*a*W := by
    calc
      _ = ∑ i, w i ^ 2 * v i := hsq
      _ ≤ ∑ i, (F*a)*w i := by
        apply Finset.sum_le_sum
        intro i _
        calc
          _ ≤ w i^2 * F := mul_le_mul_of_nonneg_left (hv i) (sq_nonneg _)
          _ ≤ _ := by nlinarith [mul_nonneg (mul_nonneg hF (hw i).1) (sub_nonneg.mpr (hw i).2)]
      _ ≤ _ := by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left hsum (mul_nonneg hF ha)
  have hVt : MemLp (fun ω => V ω - τ) 2 P := hV.sub (memLp_const τ)
  have hnoise : ∫ ω, |V ω-τ| ∂P ≤ Real.sqrt (F*a*W) := by
    apply (integral_abs_le_sqrt_second P _ hVt).trans (Real.sqrt_le_sqrt ?_)
    simpa only [variance_eq_integral hV.aemeasurable, hτ] using hvar
  let A := fun ω => ∑ i, w i * (u i ω - d i ω)^2
  let B := fun i ω => w i * (2*K*|d i ω|+(d i ω)^2)
  have haI : Integrable A P := integrable_finsetSum _ fun i _ =>
    ((hu i |>.sub (hd i)).integrable_sq).const_mul _
  have hbI i : Integrable (B i) P :=
    (((hd i).norm.integrable (by norm_num)).const_mul (2*K) |>.add (hd i).integrable_sq).const_mul _
  have hDa i : Integrable (fun ω => |d i ω|) P := (hd i).norm.integrable (by norm_num)
  have hVa : Integrable (fun ω => |V ω-τ|) P := hVt.norm.integrable (by norm_num)
  have heach i : ∫ ω, B i ω ∂P ≤ w i * (2*K*Real.sqrt q+q) := by
    dsimp only [B]
    rw [integral_const_mul, integral_add ((hDa i).const_mul (2*K)) (hd i).integrable_sq,
      integral_const_mul]
    apply mul_le_mul_of_nonneg_left _ (hw i).1
    apply add_le_add _ (hdq i)
    exact mul_le_mul_of_nonneg_left ((integral_abs_le_sqrt_second P _ (hd i)).trans
      (Real.sqrt_le_sqrt (hdq i))) (by positivity)
  change (∫ ω, |A ω-τ| ∂P) ≤ _
  calc
    _ ≤ ∫ ω, (|V ω-τ| + ∑ i, B i ω) ∂P := by
      apply integral_mono ((haI.sub (integrable_const τ)).norm)
        ((hVt.norm.integrable (by norm_num)).add (integrable_finsetSum _ fun i _ => hbI i))
      intro ω
      change |A ω - τ| ≤ |V ω - τ| + ∑ i, B i ω
      have hdiff : |A ω-V ω| ≤ ∑ i, B i ω := by
        dsimp only [A,V]
        rw [← Finset.sum_sub_distrib]
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
        intro i _
        rw [← mul_sub, abs_mul, abs_of_nonneg (hw i).1]
        apply mul_le_mul_of_nonneg_left _ (hw i).1
        rw [show (u i ω-d i ω)^2-(u i ω)^2 = (d i ω)^2 - 2*u i ω*d i ω by ring]
        refine (abs_sub _ _).trans ?_
        simp only [abs_of_nonneg (sq_nonneg (d i ω)), abs_mul, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 2)]
        nlinarith [mul_le_mul_of_nonneg_right (hbound i ω) (abs_nonneg (d i ω))]
      linarith [abs_sub_le (A ω) (V ω) τ]
    _ ≤ _ := by
      rw [integral_add hVa (integrable_finsetSum Finset.univ fun i _ => hbI i),
        integral_finsetSum _ (fun i _ => hbI i)]
      apply add_le_add hnoise
      exact (Finset.sum_le_sum (fun i _ => heach i)).trans (by
        rw [← Finset.sum_mul]; exact mul_le_mul_of_nonneg_right hsum (by positivity))

end StatadicFormal.Estimation.Linear
