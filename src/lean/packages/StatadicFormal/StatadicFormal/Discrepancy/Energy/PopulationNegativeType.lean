module

public import StatadicFormal.Discrepancy.Energy.Population
public import StatadicFormal.Discrepancy.Energy.Finite
public import StatadicFormal.Geometry.NegativeType.Defs
public import Mathlib.Probability.HasLaw
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! From finite negative type to population energy with integrable pair costs. -/
@[expose] public section
noncomputable section
open MeasureTheory ProbabilityTheory
namespace StatadicFormal.Discrepancy.Energy

variable {α : Type*} [MeasurableSpace α]

/-- The finite zero-mass quadratic inequality for a cost; symmetry and the hollow
diagonal are separate hypotheses in the population theorem. -/
def NegativeTypeCost (d : α → α → ℝ) : Prop :=
  ∀ (n : ℕ) (x : Fin n → α) (w : Fin n → ℝ), (∑ i, w i) = 0 →
    ∑ i, ∑ j, w i * w j * d (x i) (x j) ≤ 0

omit [MeasurableSpace α] in
private theorem reindex_negative {I : Type*} [Fintype I] {d : α → α → ℝ}
    (hn : NegativeTypeCost d) (X : I → α) (w : I → ℝ) (hw : ∑ i, w i = 0) :
    ∑ i, ∑ j, w i * w j * d (X i) (X j) ≤ 0 := by
  let e := (Fintype.equivFin I).symm
  have h := hn (Fintype.card I) (X ∘ e) (w ∘ e) ((e.sum_comp w).trans hw)
  have hrow (i) : ∑ j, w (e i) * w (e j) * d (X (e i)) (X (e j)) =
      ∑ j, w (e i) * w j * d (X (e i)) (X j) :=
    e.sum_comp (fun j => w (e i) * w j * d (X (e i)) (X j))
  simp only [Function.comp_apply, hrow] at h
  rwa [e.sum_comp (fun i => ∑ j, w i * w j * d (X i) (X j))] at h

private theorem expected_finite_form {I : Type*} [Fintype I] [DecidableEq I]
    (μ : I → Measure α) [∀ i, IsProbabilityMeasure (μ i)] (d : α → α → ℝ)
    (hd : ∀ x, d x x = 0) (hn : NegativeTypeCost d)
    (hi : ∀ i j, Integrable (Function.uncurry d) ((μ i).prod (μ j)))
    (w : I → ℝ) (hw : ∑ i, w i = 0) :
    ∑ i, ∑ j, w i * w j * (if i = j then 0 else pairMean (μ i) (μ j) d) ≤ 0 := by
  let M := Measure.pi μ
  have joint (i j : I) (hij : i ≠ j) : HasLaw (fun ω : I → α => (ω i, ω j))
      ((μ i).prod (μ j)) M := by
    have ind := (iIndepFun_pi (μ := μ) (X := fun _ => id) (fun _ => aemeasurable_id)).indepFun hij
    exact ind.hasLaw_prod (measurePreserving_eval μ i).hasLaw (measurePreserving_eval μ j).hasLaw
  have integ (i j) : Integrable (fun ω : I → α => d (ω i) (ω j)) M := by
    by_cases h : i = j
    · subst j
      simp [hd]
    · exact (joint i j h).integrable_comp (hi i j)
  have mom (i j) : (∫ ω : I → α, d (ω i) (ω j) ∂M) =
      if i = j then 0 else pairMean (μ i) (μ j) d := by
    by_cases h : i = j
    · subst j
      simp [hd]
    · simp only [h, ↓reduceIte]
      exact ((joint i j h).integral_comp (hi i j).aestronglyMeasurable).trans
        (integral_prod _ (hi i j))
  have term (i j) := (integ i j).const_mul (w i * w j)
  have bound := integral_nonpos (μ := M) (fun ω : I → α => reindex_negative hn ω w hw)
  simp_rw [integral_finsetSum _ (fun i _ => integrable_finsetSum _ (fun j _ => term i j)),
    integral_finsetSum _ (fun j _ => term _ j), integral_const_mul, mom] at bound
  exact bound

/-- Population energy is nonnegative on any finite-negative-type ground with
integrable pairwise costs. No feature map or injectivity is assumed. -/
@[blueprint "STATADIC-ENERGY-006"
  (statement := /-- Suppose a symmetric hollow ground $d$ has negative type on every
  finite set. If $d$ is integrable under $P\otimes P$, $Q\otimes Q$ and both mixed
  products, write $D_{AB}=\int d(x,y)\,dA(x)\,dB(y)$. Then
  $2D_{PQ}-D_{PP}-D_{QQ}\ge0$ for arbitrary probability laws $P,Q$. -/)
  (proof := /-- Take independent samples of size $n$ from each law. Apply finite
  negative type to the pooled sample with $n$ weights $+1$ and $n$ weights $-1$.
  Integrate the finite inequality. Within each sample there are $n(n-1)$
  off-diagonal terms; diagonal costs vanish. Each mixed block contributes
  $n^2D_{PQ}$ by independence and symmetry. Divide by $n$ and let $n$ grow:
  a strictly negative population energy would contradict the resulting bound
  for a sufficiently large finite $n$. -/)]
theorem population_negative_type_energy (P Q : Measure α) [IsProbabilityMeasure P]
    [IsProbabilityMeasure Q] (d : α → α → ℝ) (hs : ∀ x y, d x y = d y x)
    (hd : ∀ x, d x x = 0) (hn : NegativeTypeCost d)
    (hpp : Integrable (Function.uncurry d) (P.prod P))
    (hpq : Integrable (Function.uncurry d) (P.prod Q))
    (hqp : Integrable (Function.uncurry d) (Q.prod P))
    (hqq : Integrable (Function.uncurry d) (Q.prod Q)) :
    0 ≤ 2 * pairMean P Q d - pairMean P P d - pairMean Q Q d := by
  have sym : pairMean Q P d = pairMean P Q d := by
    unfold pairMean
    rw [integral_integral_swap hqp]
    simp_rw [hs]
  have count (n : ℕ) (hn' : 0 < n) :
      (n : ℝ) * (pairMean P P d + pairMean Q Q d - 2 * pairMean P Q d) ≤
        pairMean P P d + pairMean Q Q d := by
    let μ : Fin n ⊕ Fin n → Measure α := Sum.elim (fun _ => P) (fun _ => Q)
    let w : Fin n ⊕ Fin n → ℝ := Sum.elim (fun _ => 1) (fun _ => -1)
    let : ∀ i, IsProbabilityMeasure (μ i) := fun i => by cases i <;> dsimp [μ] <;> infer_instance
    have h := expected_finite_form μ d hd hn
      (fun i j => by cases i <;> cases j <;> first | exact hpp | exact hpq | exact hqp | exact hqq)
      w (by simp [w, Fintype.sum_sum_type])
    have row (i : Fin n) (x : ℝ) : (∑ j : Fin n, if i = j then (0 : ℝ) else x) = (n - 1 : ℝ) * x := by
      have split (j : Fin n) : (if i = j then (0 : ℝ) else x) = x - (if i = j then x else 0) := by
        split_ifs <;> simp
      simp_rw [split, Finset.sum_sub_distrib]
      simp [nsmul_eq_mul, sub_mul]
    simp only [Fintype.sum_sum_type, μ, w, Sum.elim_inl, Sum.elim_inr, Sum.inl.injEq,
      Sum.inr.injEq, Sum.inl_ne_inr, Sum.inr_ne_inl, ite_false, one_mul, mul_one,
      neg_mul, mul_neg, neg_neg, row, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, sym] at h
    have hnR : (0 : ℝ) < n := by exact_mod_cast hn'
    apply le_of_mul_le_mul_left (a := (n : ℝ)) _ hnR
    linarith only [h]
  by_contra! hneg
  let A := pairMean P P d + pairMean Q Q d
  let Δ := A - 2 * pairMean P Q d
  have hΔ : 0 < Δ := by dsimp [Δ, A]; linarith
  obtain ⟨k, hk⟩ := exists_nat_gt (max 0 (A / Δ))
  have hkpos : 0 < k := by exact_mod_cast (lt_of_le_of_lt (le_max_left _ _) hk)
  have hstrict := (div_lt_iff₀ hΔ).mp (lt_of_le_of_lt (le_max_right _ _) hk)
  exact (not_lt_of_ge (count k hkpos)) (by simpa [A, Δ] using hstrict)

private theorem integrable_pair_dist [PseudoMetricSpace α] [BorelSpace α]
    [SecondCountableTopology α] (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (o : α) (hP : Integrable (fun x => dist x o) P) (hQ : Integrable (fun y => dist y o) Q) :
    Integrable (Function.uncurry (fun x y : α => dist x y)) (P.prod Q) := by
  refine ((hP.comp_fst Q).add (hQ.comp_snd P)).mono' ?_ (Filter.Eventually.of_forall ?_)
  · exact (continuous_fst.dist continuous_snd).aestronglyMeasurable
  · intro z
    change ‖dist z.1 z.2‖ ≤ dist z.1 o + dist z.2 o
    simpa [Real.norm_of_nonneg dist_nonneg, dist_comm o z.2] using dist_triangle z.1 o z.2

/-- In a separable Borel pseudometric space a first moment suffices for all four pair integrals. -/
@[blueprint "STATADIC-ENERGY-007"
  (statement := /-- On a second-countable Borel pseudometric space of negative type,
  probability laws $P,Q$ with integrable distances to one common origin have nonnegative
  population energy. This includes discrete, continuous and mixed laws. -/)
  (proof := /-- The triangle inequality bounds the pair distance by the sum of the
  two distances to the origin. Each is integrable on the product probability
  space, so all four pair integrals exist. Apply the finite-sample averaging
  theorem with the original finite-negative-type inequality. -/)]
theorem metric_negative_type_energy [PseudoMetricSpace α] [BorelSpace α]
    [SecondCountableTopology α] (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (o : α) (hP : Integrable (fun x => dist x o) P) (hQ : Integrable (fun y => dist y o) Q)
    (hn : StatadicFormal.Geometry.DistNegativeType α) :
    0 ≤ 2 * pairMean P Q dist - pairMean P P dist - pairMean Q Q dist :=
  population_negative_type_energy P Q dist dist_comm dist_self hn
    (integrable_pair_dist P P o hP hP) (integrable_pair_dist P Q o hP hQ)
    (integrable_pair_dist Q P o hQ hP) (integrable_pair_dist Q Q o hQ hQ)

/-- A symmetric PSD generating kernel implies population energy nonnegativity using
only its finite Gram tests; no supplied Hilbert feature representation is required. -/
@[blueprint "STATADIC-ENERGY-008"
  (statement := /-- Let $k$ be symmetric with positive-semidefinite Gram matrix
  on every finite tuple. For $d(x,y)=k(x,x)+k(y,y)-2k(x,y)$ suppose the
  four population pair costs are integrable. The energy generated by $d$ is nonnegative. -/)
  (proof := /-- For finite signed weights summing to zero the diagonal
  kernel terms cancel. The signed distance quadratic equals minus twice
  the kernel quadratic, which is nonpositive by Gram positivity.
  The generated ground is hollow and symmetric, so the population
  negative-type theorem applies. -/)]
theorem generated_psd_energy (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (k : α → α → ℝ) (hs : ∀ x y, k x y = k y x)
    (hp : ∀ n (x : Fin n → α), KernelPSD (fun i j => k (x i) (x j))) :
    let d := fun x y => k x x + k y y - 2 * k x y
    Integrable (Function.uncurry d) (P.prod P) → Integrable (Function.uncurry d) (P.prod Q) →
    Integrable (Function.uncurry d) (Q.prod P) → Integrable (Function.uncurry d) (Q.prod Q) →
    0 ≤ 2 * pairMean P Q d - pairMean P P d - pairMean Q Q d := by
  let d := fun x y => k x x + k y y - 2 * k x y
  dsimp only
  apply population_negative_type_energy
  · intro x y
    rw [hs x y, add_comm (k x x)]
  · intro x
    ring
  · intro n x w hw
    have hid := finite_energy_kernel_identity (fun i j => k (x i) (x j))
      (fun i j => d (x i) (x j)) w hw (fun _ _ => rfl)
    exact neg_nonneg.mp (hid.ge.trans' (mul_nonneg (by norm_num) (hp n x w)))

end StatadicFormal.Discrepancy.Energy
