module

public import StatadicFormal.Inference.Randomization
public import Mathlib.Data.Fin.Tuple.Basic
public import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FieldSimp

/-!
# Sampled randomization tests

Finite with-replacement specialization of Hemerik and Goeman (2018),
Definition 2 and Theorem 2 (pp. 819–820), and the inclusive rank discussion
in §3.4, equation (6) (pp. 822–823).
-/

public section

namespace StatadicFormal.Inference
noncomputable section
variable {G : Type*} [Group G] {B : ℕ}

private def ext (g : Fin B → G) : Fin (B+1) → G := Fin.cons 1 g

private def recenter (j : Fin (B+1)) (g : Fin B → G) : Fin B → G :=
  fun k => ext g (Equiv.swap 0 j k.succ) * (ext g j)⁻¹

private lemma ext_recenter (j : Fin (B+1)) (g : Fin B → G) (k : Fin (B+1)) :
    ext (recenter j g) k = ext g (Equiv.swap 0 j k) * (ext g j)⁻¹ := by
  cases k using Fin.cases with
  | zero => simp [ext]
  | succ k => rfl

private lemma recenter_involutive (j : Fin (B+1)) : Function.Involutive (recenter (G := G) j) := by
  intro g
  funext k
  change ext (recenter j g) (Equiv.swap 0 j k.succ) * (ext (recenter j g) j)⁻¹ = g k
  rw [ext_recenter, ext_recenter]
  simp [ext, mul_assoc]

open scoped BigOperators
open StatadicFormal.Inference
variable {Ω : Type*} [Fintype Ω] [Fintype G] [MulAction G Ω]

private def shift (j : Fin (B+1)) (x : Ω × (Fin B → G)) : Ω × (Fin B → G) :=
  (ext x.2 j • x.1, recenter j x.2)

omit [Fintype Ω] [Fintype G] in
private lemma shift_involutive (j : Fin (B+1)) : Function.Involutive (shift (G := G) (Ω := Ω) j) := by
  intro ⟨ω,g⟩
  change (ext (recenter j g) j • (ext g j • ω), recenter j (recenter j g)) = (ω, g)
  rw [ext_recenter]
  rw [recenter_involutive j g]
  simp [ext]

private def sampleRank (T : Ω → ℝ) (x : Ω × (Fin B → G)) : ℝ :=
  ((Finset.univ.filter (fun k : Fin (B+1) => T x.1 ≤ T (ext x.2 k • x.1))).card : ℝ)

omit [Fintype Ω] [Fintype G] in
private lemma sampleRank_shift (T : Ω → ℝ) (j : Fin (B+1)) (x : Ω × (Fin B → G)) :
    sampleRank T (shift j x) = ((Finset.univ.filter (fun k : Fin (B+1) =>
      T (ext x.2 j • x.1) ≤ T (ext x.2 k • x.1))).card : ℝ) := by
  classical
  unfold sampleRank shift
  apply congrArg (fun n : ℕ => (n : ℝ))
  apply Finset.card_bij (fun k _ => Equiv.swap 0 j k)
  · intro k hk
    simpa [ext_recenter, mul_smul] using hk
  · intro a ha b hb hab
    exact (Equiv.swap 0 j).injective hab
  · intro k hk
    refine ⟨Equiv.swap 0 j k, ?_, by simp⟩
    simpa [ext_recenter, mul_smul] using hk

omit [Fintype Ω] [Fintype G] in
private lemma count_sampleRank (T : Ω → ℝ) (x : Ω × (Fin B → G)) (α : ℝ) (hα : 0 ≤ α) :
    (∑ j : Fin (B+1), if sampleRank T (shift j x) ≤ α * (B+1 : ℝ) then (1 : ℝ) else 0) ≤ α * (B+1 : ℝ) := by
  classical
  simp_rw [sampleRank_shift]
  rw [Finset.sum_boole]
  exact upper_rank_count (fun j => T (ext x.2 j • x.1)) _ (mul_nonneg hα (by positivity))

private lemma sample_level (T : Ω → ℝ) (w : Ω → ℝ) (α : ℝ) (hα : 0 ≤ α)
    (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hinv : ∀ (g : G) ω, w (g • ω) = w ω) :
    (∑ x : Ω × (Fin B → G), (w x.1 / (Fintype.card (Fin B → G) : ℝ)) *
      if sampleRank T x ≤ α * (B+1 : ℝ) then 1 else 0) ≤ α := by
  classical
  let W : Ω × (Fin B → G) → ℝ := fun x => w x.1 / (Fintype.card (Fin B → G) : ℝ)
  have hcard : (0 : ℝ) < Fintype.card (Fin B → G) := by exact_mod_cast Fintype.card_pos
  apply finite_randomization_level W (shift (G := G) (Ω := Ω))
    (fun x => sampleRank T x ≤ α * (B+1 : ℝ)) α
  · intro x
    exact div_nonneg (hw _) (le_of_lt hcard)
  · have hg : (0 : ℝ) < Fintype.card G := by exact_mod_cast Fintype.card_pos
    simp only [W, Fintype.sum_prod_type, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Fintype.card_fun, Fintype.card_fin, Nat.cast_pow]
    calc
      (∑ x : Ω, (Fintype.card G : ℝ)^B * (w x / (Fintype.card G : ℝ)^B)) = ∑ x, w x := by
        apply Finset.sum_congr rfl
        intro x _
        field_simp
      _ = 1 := hsum
  · intro j
    let e : (Ω × (Fin B → G)) ≃ (Ω × (Fin B → G)) :=
      ⟨shift j, shift j, shift_involutive j, shift_involutive j⟩
    calc
      (∑ x, W x * if sampleRank T (shift j x) ≤ α * (B+1 : ℝ) then 1 else 0) =
          ∑ x, W (shift j x) * if sampleRank T (shift j x) ≤ α * (B+1 : ℝ) then 1 else 0 := by
            simp [W, shift, hinv]
      _ = ∑ x, W x * if sampleRank T x ≤ α * (B+1 : ℝ) then 1 else 0 :=
            by
              change (∑ x, W (e x) * if sampleRank T (e x) ≤ α * (B+1 : ℝ) then 1 else 0) = _
              exact Equiv.sum_comp e (fun x => W x * if sampleRank T x ≤ α * (B+1 : ℝ) then 1 else 0)
  · intro x
    simpa using count_sampleRank T x α hα

/-- Inclusive upper-tail add-one p-value for the supplied tuple of group transforms. -/
@[expose] def sampleP (T : Ω → ℝ) (ω : Ω) (g : Fin B → G) : ℝ :=
  (1 + ((Finset.univ.filter (fun k => T ω ≤ T (g k • ω))).card : ℝ)) / (B+1 : ℝ)

omit [Fintype Ω] [Fintype G] in
private lemma sampleRank_eq (T : Ω → ℝ) (ω : Ω) (g : Fin B → G) :
    sampleRank T (ω, g) = 1 + ((Finset.univ.filter (fun k => T ω ≤ T (g k • ω))).card : ℝ) := by
  classical
  unfold sampleRank
  rw [← Finset.sum_boole]
  rw [Fin.sum_univ_succ]
  simp [ext, Finset.sum_boole]

/-- Inclusive add-one upper-tail validity for independent uniform group draws. -/
@[blueprint "STATADIC-RANDOMIZATION-002"
  (statement := /-- Let a finite group $G$ act on a finite space $\Omega$ with
  nonnegative weights $w$ summing to one and invariant under the action. Let
  $B\in\mathbb N$, $T:\Omega\to\mathbb R$, and $\alpha\ge0$. For each tuple
  $g\in G^B$, put
  \[
  p_T(\omega,g)=\frac{1+\#\{k<B:T(\omega)\le T(g_k\omega)\}}{B+1}.
  \]
  Then, for independent uniform draws $g$,
  \[
  \sum_{\omega\in\Omega}\sum_{g\in G^B}
  \frac{w(\omega)}{|G|^B}\mathbf1\{p_T(\omega,g)\le\alpha\}\le\alpha.
  \] -/)
  (proof := /-- Include the identity at index zero. For each index $j$, swap
  zero and $j$ and right-multiply every entry by the inverse of the original
  $j$th entry, while acting by that entry on the observation. This is an
  involution preserving the product weights. It changes the observed statistic
  to the $j$th statistic and permutes the sampled values. The inclusive
  upper-rank counting bound limits the number of rejecting indices to
  $\alpha(B+1)$. Averaging over indices yields the bound. -/)]
theorem sampleP_level (T : Ω → ℝ) (w : Ω → ℝ) (α : ℝ) (hα : 0 ≤ α)
    (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hinv : ∀ (g : G) ω, w (g • ω) = w ω) :
    (∑ ω, ∑ g : Fin B → G, (w ω / (Fintype.card G : ℝ)^B) *
      if sampleP T ω g ≤ α then 1 else 0) ≤ α := by
  have hn : (0 : ℝ) < B+1 := by positivity
  have hc : (Fintype.card (Fin B → G) : ℝ) = (Fintype.card G : ℝ)^B := by
    simp
  have hh := sample_level (B := B) T w α hα hw hsum hinv
  simp only [Fintype.sum_prod_type, hc] at hh
  simpa only [sampleP, ← sampleRank_eq, div_le_iff₀ hn] using hh

theorem sampleP_level_less (T : Ω → ℝ) (w : Ω → ℝ) (α : ℝ) (hα : 0 ≤ α)
    (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hinv : ∀ (g : G) ω, w (g • ω) = w ω) :
    (∑ ω, ∑ g : Fin B → G, (w ω / (Fintype.card G : ℝ)^B) *
      if (1 + ((Finset.univ.filter (fun k => T (g k • ω) ≤ T ω)).card : ℝ)) / (B+1 : ℝ) ≤ α then 1 else 0) ≤ α := by
  simpa [sampleP] using sampleP_level (B := B) (fun ω => -T ω) w α hα hw hsum hinv

theorem sampleP_level_two_sided (T : Ω → ℝ) (w : Ω → ℝ) (α : ℝ) (hα : 0 ≤ α)
    (hw : ∀ ω, 0 ≤ w ω) (hsum : ∑ ω, w ω = 1)
    (hinv : ∀ (g : G) ω, w (g • ω) = w ω) :
    (∑ ω, ∑ g : Fin B → G, (w ω / (Fintype.card G : ℝ)^B) *
      if (1 + ((Finset.univ.filter (fun k => |T ω| ≤ |T (g k • ω)|)).card : ℝ)) / (B+1 : ℝ) ≤ α then 1 else 0) ≤ α := by
  simpa [sampleP] using sampleP_level (B := B) (fun ω => |T ω|) w α hα hw hsum hinv

end
end StatadicFormal.Inference
