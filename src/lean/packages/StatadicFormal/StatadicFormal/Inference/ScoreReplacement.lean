module

public import StatadicFormal.Inference.ScoreCovariance
public import Mathlib.Probability.Moments.Variance
public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Fitted-score substitution and pooling for temporal/cluster windows. -/
@[expose] public section
noncomputable section
open Matrix Filter MeasureTheory ProbabilityTheory
open scoped Topology
namespace StatadicFormal.Inference

/-- Pool the scores by their retained label, then apply the kernel between
labels. The result is the kernel between the original row labels. -/
@[blueprint "STATADIC-PANELCOV-003"
  (statement := /-- Let $T_{ai}=1$ if the label of observed row $i$ is $a$.
  For any score matrix $U$ and label kernel $K$,
  $(TU)^TK(TU)=U^T\widetilde KU$, where
  $\widetilde K_{ij}=K_{\ell(i),\ell(j)}$.
  For calendar-time pooling this retains cross-entity pairs; using
  entity-time labels with a block-diagonal kernel suppresses them. -/)
  (proof := /-- Compute each entry of $T^TKT$. Each membership column selects
  exactly one label on either side of the kernel. Then associate the matrix
  products. Missing entity-time cells contribute no score. -/)]
theorem pooled_kernel_covariance {ι g p : Type*} [Fintype ι] [Fintype g] [DecidableEq g]
    (label : ι → g) (U : Matrix ι p ℝ) (K : Matrix g g ℝ) :
    let T := clusterMembership label
    let Q : Matrix ι ι ℝ := fun i j => K (label i) (label j)
    (T*U)ᵀ * K * (T*U) = Uᵀ * Q * U := by
  let T := clusterMembership label
  let Q : Matrix ι ι ℝ := fun i j => K (label i) (label j)
  have hp : Tᵀ*K*T = Q := by
    ext i j
    simp [Q, T, clusterMembership, Matrix.mul_apply, ite_mul, mul_ite]
  change (T*U)ᵀ * K * (T*U) = Uᵀ*Q*U
  rw [← hp, Matrix.transpose_mul]
  simp only [Matrix.mul_assoc]

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

private theorem first_abs {X : Ω → ℝ} (hX : MemLp X 2 P) {q : ℝ} (hq : ∫ ω, X ω^2 ∂P ≤ q) :
    (∫ ω, |X ω| ∂P) ≤ Real.sqrt q := by
  have h := variance_nonneg (fun ω => |X ω|) P
  rw [variance_eq_sub (show MemLp (fun ω => |X ω|) 2 P from hX.norm)] at h
  apply (Real.le_sqrt (integral_nonneg fun _ => abs_nonneg _) ((integral_nonneg
    fun _ => sq_nonneg _).trans hq)).mpr
  exact (by simpa [Pi.pow_apply] using sub_nonneg.mp h : (∫ ω, |X ω| ∂P)^2 ≤ ∫ ω, X ω^2 ∂P).trans hq

/-- Error in any bounded-weight covariance pair sum when true scores are
replaced by fitted scores with small mean-square perturbation. -/
@[blueprint "STATADIC-COVBRIDGE-001"
  (statement := /-- Let $E$ contain the retained ordered observation pairs,
  with $|k_{ij}|\le1$, and let the true score contrast satisfy $|z_i|\le K$.
  Suppose each fitted-score adjustment has $E d_i^2\le q$. Replacing $z_i$
  by $z_i-d_i$ in the normalized weighted pair sum changes it by an expected
  absolute amount at most $(|E|/N)(2K\sqrt q+q)$. -/)
  (proof := /-- Expand each pair product. The two cross terms are bounded by
  $K(|d_i|+|d_j|)$ and $|d_i d_j|\le(d_i^2+d_j^2)/2$. Take expectations
  and use Cauchy--Schwarz for each $E|d_i|$. Triangle inequalities and
  $|k_{ij}|\le1$ allow summing these bounds over retained pairs. -/)]
theorem pair_replacement_bound {n : ℕ} (E : Finset (Fin (n+1) × Fin (n+1)))
    (k : Fin (n+1) × Fin (n+1) → ℝ) (z d : Fin (n+1) → Ω → ℝ) (K q : ℝ)
    (hk : ∀ e ∈ E, |k e| ≤ 1) (hK : 0 ≤ K) (hb : ∀ i ω, |z i ω| ≤ K)
    (hz : ∀ i, MemLp (z i) 2 P) (hd : ∀ i, MemLp (d i) 2 P)
    (hq : ∀ i, ∫ ω, (d i ω)^2 ∂P ≤ q) :
    let M := fun (x : Fin (n+1) → Ω → ℝ) ω =>
      ((n+1 : ℕ) : ℝ)⁻¹ * ∑ e ∈ E, k e * (x e.1 ω * x e.2 ω)
    (∫ ω, |M (fun i a => z i a - d i a) ω - M z ω| ∂P) ≤
      ((E.card : ℝ) / (n+1 : ℕ)) * (2*K*Real.sqrt q+q) := by
  let M := fun (x : Fin (n+1) → Ω → ℝ) ω =>
    ((n+1 : ℕ) : ℝ)⁻¹ * ∑ e ∈ E, k e * (x e.1 ω * x e.2 ω)
  let B := fun (e : Fin (n+1) × Fin (n+1)) ω => K * (|d e.1 ω| + |d e.2 ω|) +
    (2:ℝ)⁻¹ * ((d e.1 ω)^2 + (d e.2 ω)^2)
  have hI (x : Fin (n+1) → Ω → ℝ) (hx : ∀ i, MemLp (x i) 2 P) : Integrable (M x) P :=
    (integrable_finsetSum E (fun e _ => ((hx e.1).integrable_mul (hx e.2)).const_mul _)).const_mul _
  have hd' i : Integrable (fun ω => |d i ω|) P := (hd i).norm.integrable (by norm_num)
  have hB e : Integrable (B e) P := (((hd' e.1).add (hd' e.2)).const_mul K).add
    (((hd e.1).integrable_sq.add (hd e.2).integrable_sq).const_mul _)
  have hEB e : ∫ ω, B e ω ∂P ≤ 2*K*Real.sqrt q+q := by
    dsimp only [B]
    have hsum : Integrable (fun ω => |d e.1 ω| + |d e.2 ω|) P :=
      (hd' e.1).add (hd' e.2)
    have hsq : Integrable (fun ω => (d e.1 ω)^2 + (d e.2 ω)^2) P :=
      (hd e.1).integrable_sq.add (hd e.2).integrable_sq
    rw [integral_add (hsum.const_mul K) (hsq.const_mul (2:ℝ)⁻¹), integral_const_mul,
      integral_const_mul, integral_add (hd' e.1) (hd' e.2), integral_add (hd e.1).integrable_sq
      (hd e.2).integrable_sq]
    linarith [mul_le_mul_of_nonneg_left (add_le_add (first_abs P (hd _) (hq e.1))
      (first_abs P (hd _) (hq e.2))) hK, hq e.1, hq e.2]
  change (∫ ω, |M (fun i a => z i a - d i a) ω - M z ω| ∂P) ≤ _
  calc
    _ ≤ ∫ ω, (((n+1 : ℕ) : ℝ)⁻¹ * ∑ e ∈ E, B e ω) ∂P := by
      apply integral_mono ((hI _ (fun i => (hz i).sub (hd i)) |>.sub (hI z hz)).norm)
        ((integrable_finsetSum E (fun e _ => hB e)).const_mul _)
      intro ω
      change |M (fun i t => z i t - d i t) ω - M z ω| ≤ _
      dsimp only [M]
      rw [← mul_sub, abs_mul, abs_of_nonneg (by positivity : 0 ≤ ((n+1 : ℕ) : ℝ)⁻¹),
        ← Finset.sum_sub_distrib]
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      apply (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
      intro e he
      rw [← mul_sub, abs_mul]
      apply (mul_le_of_le_one_left (abs_nonneg _) (hk e he)).trans
      let a := z e.1 ω
      let b := z e.2 ω
      let x := d e.1 ω
      let y := d e.2 ω
      change |(a-x)*(b-y)-a*b| ≤ K*(|x|+|y|)+2⁻¹*(x^2+y^2)
      rw [show (a-x)*(b-y)-a*b = x*y - (a*y+x*b) by ring]
      refine (abs_sub _ _).trans ((add_le_add (le_refl _) (abs_add_le _ _)).trans ?_)
      simp only [abs_mul]
      nlinarith [sq_nonneg (|x|-|y|), sq_abs x, sq_abs y,
        mul_le_mul_of_nonneg_right (show |a| ≤ K from hb _ _) (abs_nonneg y),
        mul_le_mul_of_nonneg_left (show |b| ≤ K from hb _ _) (abs_nonneg x)]
    _ ≤ _ := by
      rw [integral_const_mul, integral_finsetSum E (fun e _ => hB e)]
      calc
        _ ≤ ((n+1 : ℕ) : ℝ)⁻¹ * ∑ _e ∈ E, (2*K*Real.sqrt q+q) :=
          mul_le_mul_of_nonneg_left (Finset.sum_le_sum (fun e _ => hEB e)) (by positivity)
        _ = _ := by simp [nsmul_eq_mul]; ring

end StatadicFormal.Inference
