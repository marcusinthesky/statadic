module

public import Mathlib.Probability.Moments.Variance
public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
public import Architect
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-! One- and two-way cluster covariance on ordered pairs of score contributions. -/
@[expose] public section
noncomputable section
open MeasureTheory ProbabilityTheory Filter
open scoped Topology
namespace StatadicFormal.Inference

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

/-- The score-pair sum retained by a cluster union, scaled to one sampling row. -/
def pairMeat {n : ℕ} (E : Finset (Fin (n+1) × Fin (n+1))) (z : Fin (n+1) → Ω → ℝ) (ω : Ω) :
    ℝ := ((n+1 : ℕ) : ℝ)⁻¹ * ∑ e ∈ E, z e.1 ω * z e.2 ω

private lemma pair_sum_mean {n : ℕ} (E : Finset (Fin (n+1) × Fin (n+1)))
    (z : Fin (n+1) → Ω → ℝ) (hz : ∀ i, MemLp (z i) 2 P)
    (hm : ∀ i, ∫ ω, z i ω ∂P = 0) (hd : ∀ e ∉ E, covariance (z e.1) (z e.2) P = 0) :
    (∫ ω, pairMeat E z ω ∂P) = variance (fun ω => ∑ i, z i ω) P / (n+1 : ℕ) := by
  unfold pairMeat
  rw [integral_const_mul, integral_finsetSum _ (fun e _ => (hz e.1).integrable_mul (hz e.2)),
    variance_fun_sum hz, div_eq_inv_mul]
  congr 1
  have he (i j : Fin (n+1)) : (∫ ω, z i ω * z j ω ∂P) = covariance (z i) (z j) P := by
    rw [covariance_eq_sub (hz i) (hz j), hm, zero_mul, sub_zero]
    rfl
  simp_rw [he]
  rw [← Fintype.sum_prod_type']
  exact Finset.sum_subset (Finset.subset_univ E) fun e _ he => hd e he

omit [IsProbabilityMeasure P] in
private lemma pair_est_memLp {n : ℕ} (E : Finset (Fin (n+1) × Fin (n+1)))
    (z : Fin (n+1) → Ω → ℝ) (hz : ∀ e ∈ E, MemLp (fun ω => z e.1 ω * z e.2 ω) 2 P) :
    MemLp (pairMeat E z) 2 P := (memLp_finsetSum _ hz).const_mul _

/-- Bound the sampling fluctuations of clustered meat by the number of retained
pair products and the size of their covariance neighborhoods. -/
@[blueprint "STATADIC-CLUSTER-001"
  (statement := /-- Fix the retained ordered row pairs $E$, e.g. all pairs
  sharing an entity or a time label. Put $Y_e=z_i z_j$ for $e=(i,j)$.
  If each pair product covaries with at most $d$ retained products and all
  these covariances have absolute value at most $C$, then
  $\operatorname{Var}(N^{-1}\sum_{e\in E}Y_e)\le C|E|d/N^2$.
  For zero-mean scores with covariance zero off $E$, its expectation equals
  $\operatorname{Var}(\sum_i z_i)/N$. -/)
  (proof := /-- Off-mask covariances vanish, so the pair expectation recovers
  the full score-sum variance. Expand the sampling variance as a double
  covariance sum over pair products. Remove all off-neighborhood terms,
  count at most $d$ terms for each of $|E|$ choices, then divide by $N^2$. -/)]
theorem cluster_pair_moments {n : ℕ} (E : Finset (Fin (n+1) × Fin (n+1)))
    (z : Fin (n+1) → Ω → ℝ) (L : (Fin (n+1) × Fin (n+1)) → Finset (Fin (n+1) × Fin (n+1)))
    (d : ℕ) (C : ℝ) (hC : 0 ≤ C) (hz : ∀ i, MemLp (z i) 2 P)
    (hm : ∀ i, ∫ ω, z i ω ∂P = 0) (hd : ∀ e ∉ E, covariance (z e.1) (z e.2) P = 0)
    (hprod : ∀ e ∈ E, MemLp (fun ω => z e.1 ω * z e.2 ω) 2 P)
    (hsize : ∀ e ∈ E, (L e).card ≤ d)
    (hl : ∀ e ∈ E, ∀ f ∈ E, f ∉ L e → covariance (fun ω => z e.1 ω * z e.2 ω)
      (fun ω => z f.1 ω * z f.2 ω) P = 0)
    (hc : ∀ e ∈ E, ∀ f ∈ E, |covariance (fun ω => z e.1 ω * z e.2 ω)
      (fun ω => z f.1 ω * z f.2 ω) P| ≤ C) :
    (∫ ω, pairMeat E z ω ∂P) = variance (fun ω => ∑ i, z i ω) P / (n+1 : ℕ) ∧
      variance (pairMeat E z) P ≤ C * ((E.card : ℝ) * d / ((n+1 : ℕ) : ℝ)^2) := by
  refine ⟨pair_sum_mean P E z hz hm hd, ?_⟩
  let Y := fun (e : Fin (n+1) × Fin (n+1)) ω => z e.1 ω * z e.2 ω
  have row e (he : e ∈ E) : ∑ f ∈ E, covariance (Y e) (Y f) P ≤ d*C := by
    calc
      _ = ∑ f ∈ E ∩ L e, covariance (Y e) (Y f) P := by
        apply (Finset.sum_subset Finset.inter_subset_left ?_).symm
        intro f hf hd'
        exact hl e he f hf fun hh => hd' (Finset.mem_inter.mpr ⟨hf, hh⟩)
      _ ≤ ∑ _f ∈ E ∩ L e, C := Finset.sum_le_sum fun f hf => (le_abs_self _).trans
        (hc e he f (Finset.mem_inter.mp hf).1)
      _ ≤ _ := by
        simp only [Finset.sum_const, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast (Finset.card_le_card
          (Finset.inter_subset_right (s₁ := E) (s₂ := L e))).trans (hsize e he)) hC
  unfold pairMeat
  rw [variance_const_mul, variance_fun_sum' hprod]
  calc
    _ ≤ ((n+1 : ℕ) : ℝ)⁻¹^2 * ∑ _e ∈ E, (d:ℝ)*C :=
      mul_le_mul_of_nonneg_left (Finset.sum_le_sum row) (sq_nonneg _)
    _ = _ := by simp [nsmul_eq_mul]; ring

/-- With a sparse enough fourth-order neighborhood, the cluster estimate tracks
the actual covariance of each contrast, even when that covariance changes with N. -/
@[blueprint "STATADIC-CLUSTER-002"
  (statement := /-- Under the cluster moment conditions, if
  $|E_n|d_n/N^2\to0$ and the pair-covariance bound is uniform, the
  difference between the raw cluster meat and the true variance of the
  normalized score sum tends to zero in probability. The pair mask may
  change with $N$ and can include arbitrary within-cluster covariance. -/)
  (proof := /-- The raw cluster meat is unbiased for the finite-record
  variance target. Chebyshev's inequality bounds its error probability at
  tolerance $t>0$ by $C|E_n|d_n/(N^2t^2)$, which tends to zero. -/)]
theorem cluster_pair_consistent (E : (n : ℕ) → Finset (Fin (n+1) × Fin (n+1)))
    (z : (n : ℕ) → Fin (n+1) → Ω → ℝ)
    (L : (n : ℕ) → (Fin (n+1) × Fin (n+1)) → Finset (Fin (n+1) × Fin (n+1)))
    (d : ℕ → ℕ) (C : ℝ) (hC : 0 ≤ C) (hz : ∀ n i, MemLp (z n i) 2 P)
    (hm : ∀ n i, ∫ ω, z n i ω ∂P = 0)
    (hd : ∀ n e, e ∉ E n → covariance (z n e.1) (z n e.2) P = 0)
    (hp : ∀ n, ∀ e ∈ E n, MemLp (fun ω => z n e.1 ω * z n e.2 ω) 2 P)
    (hs : ∀ n, ∀ e ∈ E n, (L n e).card ≤ d n)
    (hl : ∀ n, ∀ e ∈ E n, ∀ f ∈ E n, f ∉ L n e →
      covariance (fun ω => z n e.1 ω * z n e.2 ω) (fun ω => z n f.1 ω * z n f.2 ω) P = 0)
    (hc : ∀ n, ∀ e ∈ E n, ∀ f ∈ E n, |covariance (fun ω => z n e.1 ω * z n e.2 ω)
      (fun ω => z n f.1 ω * z n f.2 ω) P| ≤ C)
    (hrate : Tendsto (fun n => ((E n).card : ℝ) * d n / ((n+1 : ℕ) : ℝ)^2) atTop (𝓝 0)) :
    TendstoInMeasure P (fun n ω => pairMeat (E n) (z n) ω -
      variance (fun x => ∑ i, z n i x) P / (n+1 : ℕ)) atTop (fun _ => 0) := by
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  have hlim := ENNReal.continuous_ofReal.continuousAt.tendsto.comp
    (((tendsto_const_nhds (x := C)).mul hrate).div_const (ε^2))
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le (tendsto_const_nhds (x := (0:ENNReal)))
    (by simpa [Function.comp_def] using hlim) (fun _ => bot_le)
  intro n
  obtain ⟨hmean, hvar⟩ := cluster_pair_moments P (E n) (z n) (L n) (d n) C hC
    (hz n) (hm n) (hd n) (hp n) (hs n) (hl n) (hc n)
  have hprob := meas_ge_le_variance_div_sq (pair_est_memLp P (E n) (z n) (hp n)) hε
  simp only [hmean] at hprob
  simp only [sub_zero, Real.norm_eq_abs]
  exact hprob.trans (ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right (by simpa using hvar) (sq_nonneg _)))

end StatadicFormal.Inference
