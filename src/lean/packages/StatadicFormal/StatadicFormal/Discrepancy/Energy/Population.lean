module

public import Architect
public import Mathlib.MeasureTheory.Function.L2Space
public import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Tactic.Ring

/-! # Population energy and Hilbert mean embeddings

Sejdinovic et al. (2013), Theorem 22. The real Hilbert feature map is supplied.
The hypotheses state Bochner integrability of the map and its squared norm under
each law. They apply to continuous as well as discrete probability measures. -/

@[expose] public section
noncomputable section

open MeasureTheory
open scoped InnerProductSpace

namespace StatadicFormal.Discrepancy.Energy

variable {α H : Type*} [MeasurableSpace α] [NormedAddCommGroup H]
  [InnerProductSpace ℝ H] [CompleteSpace H]

/-- Expectation for two independent draws, written as an iterated integral. -/
def pairMean (P Q : Measure α) (f : α → α → ℝ) : ℝ :=
  ∫ x, ∫ y, f x y ∂Q ∂P

private theorem pairMean_diagonal (P Q : Measure α) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] (d k : α → α → ℝ)
    (hp : Integrable (fun x ↦ k x x) P) (hq : Integrable (fun x ↦ k x x) Q)
    (hk : Integrable (Function.uncurry k) (P.prod Q))
    (hd : ∀ x y, d x y = k x x + k y y - 2 * k x y) :
    pairMean P Q d = (∫ x, k x x ∂P) + (∫ y, k y y ∂Q) - 2 * pairMean P Q k := by
  have hadd : Integrable (fun z : α × α ↦ k z.1 z.1 + k z.2 z.2) (P.prod Q) :=
    (hp.comp_fst Q).add (hq.comp_snd P)
  have hmul : Integrable (fun z : α × α ↦ 2 * k z.1 z.2) (P.prod Q) := hk.const_mul 2
  have hint : Integrable (Function.uncurry d) (P.prod Q) := by
    change Integrable (fun z : α × α ↦ d z.1 z.2) _
    simp_rw [hd]
    exact hadd.sub hmul
  unfold pairMean
  rw [integral_integral hint, integral_integral hk]
  simp_rw [hd]
  rw [integral_sub hadd hmul, integral_add (hp.comp_fst Q) (hq.comp_snd P), integral_const_mul]
  simp [integral_fun_fst (fun x ↦ k x x), integral_fun_snd (fun y ↦ k y y)]

/-- Algebraic part of the population identity with no assumption of positive
semidefiniteness or an available Hilbert feature representation. -/
@[blueprint "STATADIC-ENERGY-005"
  (statement := /-- Let $P,Q$ be probability measures and let a symmetric real
  array $k$ generate $d(x,y)=k(x,x)+k(y,y)-2k(x,y)$. Assume integrability of
  the diagonal under both laws and of $k$ under the three product measures
  $P\otimes P$, $P\otimes Q$ and $Q\otimes Q$. Writing $m_{AB}$ for the pair
  expectation, $2m_{PQ}d-m_{PP}d-m_{QQ}d$ equals
  $2(m_{PP}k-m_{PQ}k-m_{QP}k+m_{QQ}k)$. -/)
  (proof := /-- Integrate the generating relation with respect to each product
  measure. Both diagonal terms reduce to their marginal expectations. Symmetry
  and Fubini equate the two cross expectations. Subtracting within-law terms
  from twice the cross expectation cancels both diagonal moments. -/)]
theorem population_energy_quadratic (P Q : Measure α) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] (d k : α → α → ℝ)
    (hp : Integrable (fun x ↦ k x x) P) (hq : Integrable (fun x ↦ k x x) Q)
    (hpp : Integrable (Function.uncurry k) (P.prod P))
    (hpq : Integrable (Function.uncurry k) (P.prod Q))
    (hqq : Integrable (Function.uncurry k) (Q.prod Q))
    (hs : ∀ x y, k x y = k y x) (hd : ∀ x y, d x y = k x x + k y y - 2 * k x y) :
    2 * pairMean P Q d - pairMean P P d - pairMean Q Q d =
      2 * (pairMean P P k - pairMean P Q k - pairMean Q P k + pairMean Q Q k) := by
  have hswap : pairMean P Q k = pairMean Q P k := by
    unfold pairMean
    rw [integral_integral_swap hpq]
    exact integral_congr_ae (Filter.Eventually.of_forall fun y ↦
      integral_congr_ae (Filter.Eventually.of_forall fun x ↦ hs x y))
  rw [pairMean_diagonal P Q d k hp hq hpq hd, pairMean_diagonal P P d k hp hp hpp hd,
    pairMean_diagonal Q Q d k hq hq hqq hd, hswap]
  ring

/-- Independent pair expectations of a Hilbert feature kernel factor through its Bochner means. -/
theorem pairMean_inner (P Q : Measure α) (φ : α → H)
    (hp : Integrable φ P) (hq : Integrable φ Q) :
    pairMean P Q (fun x y ↦ ⟪φ x, φ y⟫_ℝ) =
      ⟪∫ x, φ x ∂P, ∫ y, φ y ∂Q⟫_ℝ := by
  unfold pairMean
  simp_rw [integral_inner hq, real_inner_comm (∫ y, φ y ∂Q)]
  rw [integral_inner hp, real_inner_comm]

private theorem pairMean_generated (P Q : Measure α) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] (φ : α → H) (d : α → α → ℝ)
    (hp : Integrable φ P) (hq : Integrable φ Q)
    (hp2 : Integrable (fun x ↦ ‖φ x‖ ^ 2) P)
    (hq2 : Integrable (fun x ↦ ‖φ x‖ ^ 2) Q)
    (hd : ∀ x y, d x y = ‖φ x‖ ^ 2 + ‖φ y‖ ^ 2 - 2 * ⟪φ x, φ y⟫_ℝ) :
    pairMean P Q d = (∫ x, ‖φ x‖ ^ 2 ∂P) + (∫ y, ‖φ y‖ ^ 2 ∂Q) -
      2 * ⟪∫ x, φ x ∂P, ∫ y, φ y ∂Q⟫_ℝ := by
  have row (x : α) : (∫ y, d x y ∂Q) =
      ‖φ x‖ ^ 2 + (∫ y, ‖φ y‖ ^ 2 ∂Q) - 2 * ⟪φ x, ∫ y, φ y ∂Q⟫_ℝ := by
    simp_rw [hd]
    have hadd : Integrable (fun y ↦ ‖φ x‖ ^ 2 + ‖φ y‖ ^ 2) Q :=
      (integrable_const _).add hq2
    rw [integral_sub hadd ((hq.const_inner (φ x)).const_mul 2),
      integral_add (integrable_const (‖φ x‖ ^ 2)) hq2, integral_const_mul, integral_inner hq]
    simp
  unfold pairMean
  simp_rw [row]
  have hadd : Integrable (fun x ↦ ‖φ x‖ ^ 2 + ∫ y, ‖φ y‖ ^ 2 ∂Q) P :=
    hp2.add (integrable_const _)
  rw [integral_sub hadd
      ((hp.inner_const (∫ y, φ y ∂Q)).const_mul 2),
    integral_add hp2 (integrable_const (∫ y, ‖φ y‖ ^ 2 ∂Q)), integral_const_mul]
  simp_rw [real_inner_comm (∫ y, φ y ∂Q)]
  rw [integral_inner hp, real_inner_comm]
  simp

/-- Population energy equals twice both the kernel discrepancy and the squared
norm of the difference of the Bochner means. -/
@[blueprint "STATADIC-ENERGY-004"
  (statement := /-- Suppose $P,Q$ are probability laws, $k(x,y)=\langle\phi(x),\phi(y)\rangle$
  for a real Hilbert-valued feature map, and $d(x,y)=k(x,x)+k(y,y)-2k(x,y)$.
  Assume $\phi$ and $\|\phi\|^2$ are integrable under each law. Writing $m_{AB}f$
  for the pair expectation, put $E=2m_{PQ}d-m_{PP}d-m_{QQ}d$.
  Then $E=2(m_{PP}k+m_{QQ}k-2m_{PQ}k)=2\|\mu_P-\mu_Q\|^2$,
  where $\mu_P$ and $\mu_Q$ are the Bochner feature means. Thus $E\ge0$ and
  $E=0$ exactly when the feature means agree. -/)
  (proof := /-- Integrate the generating relation for each pair of laws. Unit
  mass leaves one diagonal moment per draw. The inner-product integrals factor
  into the inner products of the two means. In the energy combination the
  diagonal moments cancel, leaving the expansion of the squared norm. -/)]
theorem population_energy_kernel_identity (P Q : Measure α) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] (φ : α → H) (d k : α → α → ℝ)
    (hp : Integrable φ P) (hq : Integrable φ Q)
    (hp2 : Integrable (fun x ↦ ‖φ x‖ ^ 2) P)
    (hq2 : Integrable (fun x ↦ ‖φ x‖ ^ 2) Q)
    (hk : ∀ x y, k x y = ⟪φ x, φ y⟫_ℝ)
    (hd : ∀ x y, d x y = k x x + k y y - 2 * k x y) :
    let energy := 2 * pairMean P Q d - pairMean P P d - pairMean Q Q d
    energy = 2 * (pairMean P P k + pairMean Q Q k - 2 * pairMean P Q k) ∧
      energy = 2 * ‖(∫ x, φ x ∂P) - (∫ x, φ x ∂Q)‖ ^ 2 ∧ 0 ≤ energy ∧
      (energy = 0 ↔ (∫ x, φ x ∂P) = ∫ y, φ y ∂Q) := by
  have hdist (x y : α) : d x y = ‖φ x‖ ^ 2 + ‖φ y‖ ^ 2 - 2 * ⟪φ x, φ y⟫_ℝ := by
    simp only [hd, hk, real_inner_self_eq_norm_sq]
  have hkernel : k = (fun x y ↦ ⟪φ x, φ y⟫_ℝ) := funext (fun x ↦ funext (hk x))
  dsimp only
  have henergy : 2 * pairMean P Q d - pairMean P P d - pairMean Q Q d =
      2 * ‖(∫ x, φ x ∂P) - (∫ y, φ y ∂Q)‖ ^ 2 := by
    rw [pairMean_generated P Q φ d hp hq hp2 hq2 hdist,
      pairMean_generated P P φ d hp hp hp2 hp2 hdist,
      pairMean_generated Q Q φ d hq hq hq2 hq2 hdist, norm_sub_sq_real]
    simp only [real_inner_self_eq_norm_sq]
    ring
  refine ⟨?_, henergy, henergy ▸ mul_nonneg (by norm_num) (sq_nonneg _), ?_⟩
  · rw [henergy, hkernel, pairMean_inner P P φ hp hp, pairMean_inner Q Q φ hq hq,
      pairMean_inner P Q φ hp hq, norm_sub_sq_real]
    simp only [real_inner_self_eq_norm_sq]
    ring
  · simp [henergy, sub_eq_zero]

end StatadicFormal.Discrepancy.Energy
