module

public import StatadicFormal.Discrepancy.MMD.LinearReduction
public import StatadicFormal.Discrepancy.MMD.FiniteBias
public import StatadicFormal.Discrepancy.Energy.Finite
public import Mathlib.Data.Fintype.Sum
public import Mathlib.Tactic.NormNum

@[expose] public section
noncomputable section
open scoped BigOperators InnerProductSpace
namespace StatadicFormal.Discrepancy.MMD
open StatadicFormal.Discrepancy.Energy

/-- The linear V expression is a squared norm when masses agree. -/
@[blueprint "STATADIC-MMD-003"
  (statement := /-- For finite vector families $x_i,y_j$ in a real inner product
  space and real weights $w_i,v_j$ of equal mass, the linear V expression
  $\sum_{i,k}w_iw_k\langle x_i,x_k\rangle+\sum_{j,l}v_jv_l\langle y_j,y_l\rangle-2\sum_{i,j}w_iv_j\langle x_i,y_j\rangle$ is nonnegative. -/)
  (proof := /-- Apply the weighted mean reduction and nonnegativity of a square. -/)]
theorem linear_mmd_nonneg {ι κ E : Type*} [Fintype ι] [Fintype κ]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x : ι → E) (y : κ → E) (w : ι → ℝ) (v : κ → ℝ)
    (hmass : (∑ i, w i) = ∑ j, v j) :
    0 ≤ (∑ i, ∑ k, w i * w k * ⟪x i, x k⟫_ℝ) +
      (∑ j, ∑ l, v j * v l * ⟪y j, y l⟫_ℝ) -
      2 * (∑ i, ∑ j, w i * v j * ⟪x i, y j⟫_ℝ) := by
  rw [linear_mmd_reduction x y w v 0 hmass]
  exact sq_nonneg _

/-- A finite Gram kernel has nonnegative quadratic form. -/
theorem gram_kernel_psd {ι E : Type*} [Fintype ι]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (φ : ι → E) :
    KernelPSD (fun i j => ⟪φ i, φ j⟫_ℝ) := by
  intro c
  unfold quadratic
  have h : (∑ i, ∑ j, c i * c j * ⟪φ i, φ j⟫_ℝ) =
      ‖∑ i, c i • φ i‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq]
    simp only [sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right]
    simp_rw [mul_assoc]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    rw [real_inner_comm]
  rw [h]
  exact sq_nonneg _

/-- The symmetric pooled-kernel V expression is the signed quadratic form. -/
theorem pooled_v_eq_quadratic {ι κ : Type*} [Fintype ι] [Fintype κ]
    (k : (ι ⊕ κ) → (ι ⊕ κ) → ℝ) (w : ι → ℝ) (v : κ → ℝ)
    (hsymm : ∀ a b, k a b = k b a) :
    (∑ i, ∑ j, w i * w j * k (Sum.inl i) (Sum.inl j)) +
      (∑ i, ∑ j, v i * v j * k (Sum.inr i) (Sum.inr j)) -
      2 * (∑ i, ∑ j, w i * v j * k (Sum.inl i) (Sum.inr j)) =
      quadratic k (Sum.elim w (fun j => -v j)) := by
  have hcross : (∑ j, ∑ i, v j * w i * k (Sum.inr j) (Sum.inl i)) =
      ∑ i, ∑ j, w i * v j * k (Sum.inl i) (Sum.inr j) := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    rw [hsymm]
    ring
  unfold quadratic
  simp only [Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    neg_mul, mul_neg, Finset.sum_add_distrib, Finset.sum_neg_distrib]
  rw [hcross]
  ring

/-- PSD gives nonnegativity without requiring normalized or positive weights. -/
@[blueprint "STATADIC-MMD-004"
  (statement := /-- For a symmetric positive semidefinite real kernel $k$ on
  the finite disjoint union $\iota\sqcup\kappa$ and real weights $w_i,v_j$,
  $\sum_{i,j}w_iw_jk(i,j)+\sum_{i,j}v_iv_jk(i,j)-2\sum_{i,j}w_iv_jk(i,j)\geq0$,
  where each index is embedded in its respective component of the union. -/)
  (proof := /-- Split the quadratic form of the signed pooled weights into four
  blocks, use symmetry on the reverse cross block, and apply PSD. -/)]
theorem pooled_v_nonneg {ι κ : Type*} [Fintype ι] [Fintype κ]
    (k : (ι ⊕ κ) → (ι ⊕ κ) → ℝ) (w : ι → ℝ) (v : κ → ℝ)
    (hsymm : ∀ a b, k a b = k b a) (hpsd : KernelPSD k) :
    0 ≤ (∑ i, ∑ j, w i * w j * k (Sum.inl i) (Sum.inl j)) +
      (∑ i, ∑ j, v i * v j * k (Sum.inr i) (Sum.inr j)) -
      2 * (∑ i, ∑ j, w i * v j * k (Sum.inl i) (Sum.inr j)) := by
  rw [pooled_v_eq_quadratic k w v hsymm]
  exact hpsd _

/-- The U statistic can be negative for a linear kernel and identical samples. -/
@[blueprint "STATADIC-MMD-005"
  (statement := /-- For each of the two samples $(1,-1)$, the linear kernel
  $k(a,b)=ab$ and the unique outcome in $\mathrm{Unit}$, the finite U statistic
  $\operatorname{mmdU}$ equals $-2$. -/)
  (proof := /-- Expand the two off-diagonal sums and the cross sum. -/)]
theorem mmdU_negative_witness :
    mmdU (fun (i j : Fin 2) (_ : Unit) => (![1, -1] : Fin 2 → ℝ) i * ![1, -1] j)
      (fun (i j : Fin 2) (_ : Unit) => (![1, -1] : Fin 2 → ℝ) i * ![1, -1] j)
      (fun (i j : Fin 2) (_ : Unit) => (![1, -1] : Fin 2 → ℝ) i * ![1, -1] j) () = -2 := by
  norm_num [mmdU, pairU, crossAverage, Fin.sum_univ_succ]

/-- Padding either index set with zero coefficients preserves the V expression. -/
@[blueprint "STATADIC-MMD-006"
  (statement := /-- Given finite index sets $\iota,\kappa,\alpha,\beta$, a real
  kernel $k$ on $(\iota\sqcup\alpha)\sqcup(\kappa\sqcup\beta)$ and real weights
  $w,v$, let $\tilde w=(w,0)$ and $\tilde v=(v,0)$. Then the V expression
  $\sum_{i,j}\tilde w_i\tilde w_jk(i,j)+\sum_{i,j}\tilde v_i\tilde v_jk(i,j)-2\sum_{i,j}\tilde w_i\tilde v_jk(i,j)$
  equals the same expression restricted to $\iota$ and $\kappa$ using $w,v$. -/)
  (proof := /-- Split each finite sum over its disjoint union; padded terms vanish. -/)]
theorem weighted_v_zero_padding {ι κ α β : Type*}
    [Fintype ι] [Fintype κ] [Fintype α] [Fintype β]
    (k : ((ι ⊕ α) ⊕ (κ ⊕ β)) → ((ι ⊕ α) ⊕ (κ ⊕ β)) → ℝ)
    (w : ι → ℝ) (v : κ → ℝ) :
    (∑ i, ∑ j, Sum.elim w (fun _ : α => 0) i *
      Sum.elim w (fun _ : α => 0) j * k (Sum.inl i) (Sum.inl j)) +
    (∑ i, ∑ j, Sum.elim v (fun _ : β => 0) i *
      Sum.elim v (fun _ : β => 0) j * k (Sum.inr i) (Sum.inr j)) -
    2 * (∑ i, ∑ j, Sum.elim w (fun _ : α => 0) i *
      Sum.elim v (fun _ : β => 0) j * k (Sum.inl i) (Sum.inr j)) =
    (∑ i, ∑ j, w i * w j * k (Sum.inl (Sum.inl i)) (Sum.inl (Sum.inl j))) +
    (∑ i, ∑ j, v i * v j * k (Sum.inr (Sum.inl i)) (Sum.inr (Sum.inl j))) -
    2 * (∑ i, ∑ j, w i * v j * k (Sum.inl (Sum.inl i)) (Sum.inr (Sum.inl j))) := by
  simp [Fintype.sum_sum_type]

end StatadicFormal.Discrepancy.MMD
