module

public import StatadicFormal.Geometry.NegativeType.Defs
public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Probability.Distributions.Gaussian.Multivariate
public import Mathlib.Probability.Distributions.Gaussian.Fernique

/-!
# Euclidean and Hilbert negative type

Ported from pricing-perspective revision `66169ad5ff07f7779fe2543e24f215454bb24704`:
`EnergyStatistics/Defs.lean:141-143` and `EnergyStatistics/NegativeType.lean:129-356`.
The Gaussian-projection lemmas also support mean embeddings without depending on
energy-statistics definitions or finite probability laws.
-/

public section

open scoped BigOperators
namespace StatadicFormal.Geometry

section OneDimensionalCND

open MeasureTheory Set

/-- The overlap of two intervals `Ioc A a`, `Ioc A b` has Lebesgue measure `min a b - A`.
This is the integral form of the *min kernel* `(a, b) ↦ min a b`, the positive-definiteness
engine behind conditional negative definiteness of `|a - b|` on `ℝ`. -/
private lemma integral_indicator_Ioc_mul {A a b : ℝ} (ha : A ≤ a) (hb : A ≤ b) :
    ∫ u, (Ioc A a).indicator (1 : ℝ → ℝ) u * (Ioc A b).indicator 1 u = min a b - A := by
  have hset : Ioc A a ∩ Ioc A b = Ioc A (min a b) := by
    ext u
    simp only [mem_inter_iff, mem_Ioc, le_min_iff]
    tauto
  have hfun : (fun u => (Ioc A a).indicator (1 : ℝ → ℝ) u * (Ioc A b).indicator 1 u) =
      (Ioc A (min a b)).indicator 1 := by
    funext u
    rw [← Set.inter_indicator_mul, hset]
    simp [Pi.one_def]
  rw [hfun, integral_indicator_one measurableSet_Ioc]
  simp [Real.volume_real_Ioc, ha, hb]

/-- **Conditional negative definiteness of `|a − b|` on `ℝ`.**

For zero-sum weights, `∑ᵢⱼ wᵢwⱼ|sᵢ−sⱼ| ≤ 0`.  Proof: `|a−b| = a + b − 2·min a b`; the linear
terms vanish by `∑w = 0`, and the min-kernel double sum is the integral of the square
`(∑ᵢ wᵢ·𝟙_{Ioc A sᵢ})²`, hence nonnegative. -/
private lemma cnd_abs_sub_real {k : ℕ} (s w : Fin k → ℝ) (hw : (∑ i, w i) = 0) :
    ∑ i, ∑ j, w i * w j * |s i - s j| ≤ 0 := by
  rcases isEmpty_or_nonempty (Fin k) with hk | hk
  · simp
  -- a uniform lower bound for the sample points
  set A : ℝ := Finset.univ.inf' Finset.univ_nonempty s with hA
  have hAle : ∀ i, A ≤ s i := fun i => Finset.inf'_le _ (Finset.mem_univ i)
  -- the indicator functions of (A, sᵢ]
  set f : Fin k → ℝ → ℝ := fun i => (Ioc A (s i)).indicator 1 with hf
  have hf_int : ∀ i, Integrable (f i) := fun i =>
    (integrable_indicator_iff measurableSet_Ioc).2 (integrableOn_const measure_Ioc_lt_top.ne)
  have hff_int : ∀ i j, Integrable (fun u => w i * w j * (f i u * f j u)) := by
    intro i j
    have heq : (fun u => f i u * f j u) = (Ioc A (s i) ∩ Ioc A (s j)).indicator 1 := by
      funext u
      rw [hf, ← Set.inter_indicator_mul]
      simp [Pi.one_def]
    refine Integrable.const_mul ?_ _
    rw [heq]
    exact (integrable_indicator_iff (measurableSet_Ioc.inter measurableSet_Ioc)).2
      (integrableOn_const ((measure_mono Set.inter_subset_left).trans_lt measure_Ioc_lt_top).ne)
  -- |sᵢ − sⱼ| = sᵢ + sⱼ − 2 min(sᵢ, sⱼ)
  have habs : ∀ i j, |s i - s j| = s i + s j - 2 * min (s i) (s j) := by
    intro i j
    rcases le_total (s i) (s j) with h | h
    · rw [abs_sub_comm, abs_of_nonneg (by linarith), min_eq_left h]; ring
    · rw [abs_of_nonneg (by linarith), min_eq_right h]; ring
  -- the linear terms vanish by zero-sum weights
  have hlin₁ : ∑ i, ∑ j, w i * w j * s i = 0 := by
    simp_rw [show ∀ i j : Fin k, w i * w j * s i = (w i * s i) * w j from fun i j => by ring,
      ← Finset.mul_sum, hw, mul_zero, Finset.sum_const_zero]
  have hlin₂ : ∑ i, ∑ j, w i * w j * s j = 0 := by
    simp_rw [show ∀ i j : Fin k, w i * w j * s j = w i * (w j * s j) from fun i j => by ring,
      ← Finset.mul_sum, ← Finset.sum_mul, hw, zero_mul]
  have hsplit : ∑ i, ∑ j, w i * w j * |s i - s j| =
      -2 * ∑ i, ∑ j, w i * w j * min (s i) (s j) := by
    simp_rw [habs]
    have expand : ∀ i j : Fin k, w i * w j * (s i + s j - 2 * min (s i) (s j)) =
        w i * w j * s i + w i * w j * s j - 2 * (w i * w j * min (s i) (s j)) :=
      fun i j => by ring
    simp_rw [expand, Finset.sum_sub_distrib, Finset.sum_add_distrib, hlin₁, hlin₂,
      ← Finset.mul_sum]
    ring
  rw [hsplit]
  -- the min-kernel double sum is ≥ 0: it equals ∫ (∑ᵢ wᵢ fᵢ)²
  have hshift : ∑ i, ∑ j, w i * w j * min (s i) (s j) =
      ∑ i, ∑ j, w i * w j * (min (s i) (s j) - A) := by
    have hAterm : ∑ i, ∑ j, w i * w j * A = 0 := by
      simp_rw [show ∀ i j : Fin k, w i * w j * A = (w i * A) * w j from fun i j => by ring,
        ← Finset.mul_sum, hw, mul_zero, Finset.sum_const_zero]
    have expand : ∀ i j : Fin k, w i * w j * (min (s i) (s j) - A) =
        w i * w j * min (s i) (s j) - w i * w j * A := fun i j => by ring
    simp_rw [expand, Finset.sum_sub_distrib, hAterm, sub_zero]
  have hkernel : ∑ i, ∑ j, w i * w j * (min (s i) (s j) - A) =
      ∫ u, (∑ i, w i * f i u) ^ 2 := by
    have hsq : ∀ u, (∑ i, w i * f i u) ^ 2 = ∑ i, ∑ j, w i * w j * (f i u * f j u) := by
      intro u
      rw [sq, Finset.sum_mul_sum]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
    simp_rw [hsq]
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hff_int i j]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [integral_finsetSum _ fun j _ => hff_int i j]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [integral_const_mul, integral_indicator_Ioc_mul (hAle i) (hAle j)]
  have hnonneg : 0 ≤ ∑ i, ∑ j, w i * w j * min (s i) (s j) := by
    rw [hshift, hkernel]
    exact integral_nonneg fun u => sq_nonneg _
  linarith

end OneDimensionalCND

section GaussianProjection

open MeasureTheory ProbabilityTheory
open scoped RealInnerProductSpace NNReal

/-- The absolute first moment of a centred real Gaussian scales as `√V`. -/
private lemma integral_abs_gaussianReal (V : ℝ≥0) :
    ∫ t, |t| ∂(gaussianReal 0 V) = Real.sqrt V * ∫ t, |t| ∂(gaussianReal 0 1) := by
  have hmap : (gaussianReal 0 1).map (Real.sqrt V * ·) = gaussianReal 0 V := by
    rw [gaussianReal_map_const_mul, mul_zero]
    congr 1
    exact NNReal.coe_injective (by simp [Real.sq_sqrt V.coe_nonneg])
  rw [← hmap, integral_map (by fun_prop) (by fun_prop)]
  simp_rw [abs_mul, abs_of_nonneg (Real.sqrt_nonneg _)]
  rw [integral_const_mul]

/-- The standard Gaussian absolute first moment is positive (we never need its value
`√(2/π)`, only positivity). -/
lemma integral_abs_gaussianReal_one_pos : 0 < ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1) := by
  have hint : Integrable (fun t : ℝ => |t|) (gaussianReal 0 1) :=
    (memLp_one_iff_integrable.mp (memLp_id_gaussianReal 1)).abs
  rw [integral_pos_iff_support_of_nonneg (fun t => abs_nonneg t) hint]
  have hsupp : Function.support (fun t : ℝ => |t|) = {(0 : ℝ)}ᶜ := by
    ext t
    simp [Function.mem_support]
  have : NullSingletonClass (gaussianReal (0 : ℝ) 1) :=
    nullSingletonClass_gaussianReal one_ne_zero
  rw [hsupp, measure_compl (measurableSet_singleton 0) (measure_ne_top _ _), measure_singleton]
  simp

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- Pushing the standard Gaussian forward along `⟪v, ·⟫` gives a centred real Gaussian with
variance `‖v‖²`. -/
private lemma map_stdGaussian_innerSL (v : E) :
    (stdGaussian E).map (innerSL ℝ v) = gaussianReal 0 (‖v‖₊ ^ 2) := by
  rw [IsGaussian.map_eq_gaussianReal (innerSL ℝ v)]
  congr 1
  · exact integral_strongDual_stdGaussian (innerSL ℝ v)
  · rw [variance_dual_stdGaussian, innerSL_apply_norm]
    refine NNReal.coe_injective ?_
    rw [Real.coe_toNNReal _ (sq_nonneg _)]
    simp

/-- Every real Gaussian projection of a finite-dimensional Hilbert space is integrable. -/
lemma integrable_abs_inner (v : E) :
    Integrable (fun g : E => |⟪v, g⟫|) (stdGaussian E) :=
  ((innerSL ℝ v).integrable_comp IsGaussian.integrable_id).abs

/-- **Gaussian projection representation of the norm**: `∫ |⟪v, g⟫| dγ(g) = ‖v‖ · c` with
`c = ∫|t| dN(0,1) > 0`.  This is what reduces Hilbert-space CND to the one-dimensional case. -/
lemma integral_abs_inner_stdGaussian (v : E) :
    ∫ g, |⟪v, g⟫| ∂(stdGaussian E) = ‖v‖ * ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1) := by
  have h1 : ∫ g, |⟪v, g⟫| ∂(stdGaussian E) =
      ∫ t, |t| ∂((stdGaussian E).map (innerSL ℝ v)) := by
    rw [integral_map (by fun_prop) (by fun_prop)]
    simp
  rw [h1, map_stdGaussian_innerSL, integral_abs_gaussianReal]
  congr 1
  push_cast
  exact Real.sqrt_sq (norm_nonneg v)

/-- CND of the distance on a finite-dimensional real inner product space, by Gaussian
projection onto the one-dimensional case. -/
private lemma cnd_dist_of_finiteDimensional {k : ℕ} (x : Fin k → E) (w : Fin k → ℝ)
    (hw : (∑ i, w i) = 0) :
    ∑ i, ∑ j, w i * w j * ‖x i - x j‖ ≤ 0 := by
  set c := ∫ t, |t| ∂(gaussianReal (0 : ℝ) 1) with hc
  have hcpos : 0 < c := integral_abs_gaussianReal_one_pos
  have hterm_int : ∀ i j : Fin k,
      Integrable (fun g : E => w i * w j * |⟪x i, g⟫ - ⟪x j, g⟫|) (stdGaussian E) := by
    intro i j
    have h := (integrable_abs_inner (x i - x j)).const_mul (w i * w j)
    simpa [inner_sub_left] using h
  have key : ∫ g, ∑ i, ∑ j, w i * w j * |⟪x i, g⟫ - ⟪x j, g⟫| ∂(stdGaussian E) =
      (∑ i, ∑ j, w i * w j * ‖x i - x j‖) * c := by
    rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hterm_int i j,
      Finset.sum_mul]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [integral_finsetSum _ fun j _ => hterm_int i j, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [integral_const_mul]
    simp_rw [← inner_sub_left]
    rw [integral_abs_inner_stdGaussian (x i - x j), ← hc]
    ring
  have hle : ∫ g, ∑ i, ∑ j, w i * w j * |⟪x i, g⟫ - ⟪x j, g⟫| ∂(stdGaussian E) ≤ 0 :=
    integral_nonpos fun g => cnd_abs_sub_real (fun i => ⟪x i, g⟫) w hw
  rw [key] at hle
  nlinarith [hle, hcpos]

end GaussianProjection

/-- **Euclidean/Hilbert spaces have negative type** (Schoenberg's theorem).

For any real inner-product space `E`, the metric `dist` is conditionally
negative definite: for every finite family of points and zero-sum weights,
the weighted double sum of pairwise distances is ≤ 0.

This gives the signed inequality needed for finite Euclidean energy.

**Proof outline** (Gaussian projection, classically equivalent to Schoenberg's
Bernstein-function argument):
1. The points lie in the finite-dimensional subspace `F = span ℝ (range x)`, so it
   suffices to prove the inequality there.
2. On a finite-dimensional space, `‖v‖ · c = ∫ |⟪v, g⟫| dγ(g)` where `γ` is the standard
   Gaussian and `c = ∫|t| dN(0,1) > 0` (`integral_abs_inner_stdGaussian`).
3. This reduces the claim pointwise in `g` to the one-dimensional inequality
   `Σᵢⱼ wᵢwⱼ|sᵢ−sⱼ| ≤ 0` for `sᵢ = ⟪xᵢ, g⟫`, proven via the min-kernel identity
   (`cnd_abs_sub_real`).

Reference: Lyons (2013), §3, equation (3.1), pp. 3291–3292. -/
theorem distNegativeType_of_innerProductSpace
    (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] :
    DistNegativeType E := by
  intro k x w hw
  simp_rw [dist_eq_norm]
  -- reduce to the span of the points, which is finite dimensional
  set F : Submodule ℝ E := Submodule.span ℝ (Set.range x) with hF
  have : FiniteDimensional ℝ F := FiniteDimensional.span_of_finite ℝ (Set.finite_range x)
  let : MeasurableSpace F := borel F
  have : BorelSpace F := ⟨rfl⟩
  set y : Fin k → F := fun i => ⟨x i, Submodule.subset_span (Set.mem_range_self i)⟩ with hy
  have hxy : ∀ i j : Fin k, ‖x i - x j‖ = ‖y i - y j‖ := by
    intro i j
    rw [← Submodule.norm_coe]
    norm_cast
  simp_rw [hxy]
  exact cnd_dist_of_finiteDimensional y w hw

end StatadicFormal.Geometry
