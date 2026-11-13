module

public import StatadicFormal.Operators.Panel.Within
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Linarith
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-! Exact two-way within transformations for observed and rectangular panel designs. -/

@[expose] public section
open scoped BigOperators
namespace StatadicFormal.Operators.Panel

variable {I E T : Type*} [Fintype I] [Fintype E] [Fintype T]

private theorem group_orthogonal {G : Type*} [Fintype G] [DecidableEq G]
    (g : I → G) (s : I → ℝ) (f : G → ℝ)
    (h : ∀ k, (∑ i, if g i = k then s i else 0) = 0) : ∑ i, s i * f (g i) = 0 := by
  calc
    _ = ∑ i, ∑ k, (if g i = k then s i else 0) * f k := by simp [ite_mul]
    _ = ∑ k, (∑ i, if g i = k then s i else 0) * f k := by
      rw [Finset.sum_comm]
      simp only [Finset.sum_mul]
    _ = 0 := by simp [h]

/-- Weighted normal equations certify the unique fitted values even when the
entity and time dummy coefficients admit several normalizations. -/
@[blueprint "STATADIC-PANEL-002"
  (statement := /-- On a finite roster with positive row weights, suppose
  $x_i-r_i=u_{e_i}+v_{t_i}$ and the weighted residual totals vanish within
  every entity and every period. For every competing additive fit $a,b$,
  its weighted squared error exceeds that of $r$ by exactly
  $\sum_i w_i(u_{e_i}+v_{t_i}-a_{e_i}-b_{t_i})^2$. The competitor attains
  the minimum exactly when it gives the same fitted values on every row. -/)
  (proof := /-- Group the cross term first by entity, then by period. Both
  parts vanish by the weighted normal equations. The resulting excess is a
  sum of squares with positive weights, which vanishes exactly when every
  fitted row agrees. This argument permits missing cells and disconnected
  components of the entity--period graph. -/)]
theorem weighted_two_way_certificate [DecidableEq E] [DecidableEq T]
    (entity : I → E) (time : I → T) (w x r : I → ℝ) (u a : E → ℝ) (v b : T → ℝ)
    (hw : ∀ i, 0 < w i) (hf : ∀ i, x i = r i + (u (entity i) + v (time i)))
    (he : ∀ k, (∑ i, if entity i = k then w i * r i else 0) = 0)
    (ht : ∀ k, (∑ i, if time i = k then w i * r i else 0) = 0) :
    let d := fun i => (u (entity i) - a (entity i)) + (v (time i) - b (time i))
    let loss := ∑ i, w i * (x i - (a (entity i) + b (time i))) ^ 2
    loss = (∑ i, w i * r i ^ 2) + ∑ i, w i * d i ^ 2 ∧
      (loss ≤ (∑ i, w i * r i ^ 2) ↔ ∀ i, d i = 0) := by
  intro d loss
  have cross : (∑ i, w i * r i * d i) = 0 := by
    simp only [d, mul_add, Finset.sum_add_distrib]
    exact add_eq_zero_iff_eq_neg.mpr (by
      rw [group_orthogonal entity _ (fun k => u k - a k) he,
        group_orthogonal time _ (fun k => v k - b k) ht, neg_zero])
  have gap : loss = (∑ i, w i * r i ^ 2) + ∑ i, w i * d i ^ 2 := by
    calc
      _ = ∑ i, (w i * r i ^ 2 + 2 * (w i * r i * d i) + w i * d i ^ 2) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [hf i]
        dsimp [d]
        ring
      _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum, cross, mul_zero, add_zero]
  refine ⟨gap, ?_⟩
  rw [gap, add_le_iff_nonpos_right]
  constructor
  · intro hs i
    have hz := le_antisymm hs (Finset.sum_nonneg fun j _ => mul_nonneg (hw j).le (sq_nonneg _))
    have hi := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j _ => mul_nonneg (hw j).le (sq_nonneg (d j)))).mp hz i (Finset.mem_univ i)
    exact sq_eq_zero_iff.mp ((mul_eq_zero.mp hi).resolve_left (hw i).ne')
  · intro h
    simp [h]

/-- A weighted average with already normalized masses. -/
private def avg {K : Type*} [Fintype K] (w x : K → ℝ) : ℝ := ∑ i, w i * x i

private theorem avg_sub {K : Type*} [Fintype K] (w x y : K → ℝ) :
    avg w (fun i => x i - y i) = avg w x - avg w y := by
  simp [avg, mul_sub, Finset.sum_sub_distrib]

private theorem avg_const {K : Type*} [Fintype K] (w : K → ℝ) (hw : ∑ i, w i = 1) (a : ℝ) :
    avg w (fun _ => a) = a := by simp [avg, ← Finset.sum_mul, hw]

/-- With a complete roster and product weights the two weighted sweeps commute. -/
@[blueprint "STATADIC-PANEL-003"
  (statement := /-- On a rectangular panel with weights $w_{et}=p_eq_t$, where each
  marginal vector is normalized, entity and time demeaning commute. Both orders
  give $x_{et}-\bar x_{e\cdot}-\bar x_{\cdot t}+\bar x_{\cdot\cdot}$.
  Thus one sweep in each direction removes both sets of weighted group means. -/)
  (proof := /-- Linearity moves a group average through a subtraction.
  The two orders of the grand weighted average agree by interchange of finite
  sums. Normalization preserves constants, so centering a group twice does
  not change it. -/)]
theorem product_weight_centering (p : E → ℝ) (q : T → ℝ) (x : E → T → ℝ)
    (hp : ∑ e, p e = 1) (hq : ∑ t, q t = 1) :
    let row := fun e => ∑ t, q t * x e t
    let col := fun t => ∑ e, p e * x e t
    let grand := ∑ e, p e * row e
    let r := fun e t => x e t - row e - col t + grand
    (∀ e t, x e t - row e - (∑ i, p i * (x i t - row i)) = r e t) ∧
      (∀ e t, x e t - col t - (∑ j, q j * (x e j - col j)) = r e t) ∧
      (∀ e, (∑ t, q t * r e t) = 0) ∧ (∀ t, (∑ e, p e * r e t) = 0) := by
  intro row col grand r
  have swap : avg q col = grand := by
    dsimp [avg, grand, col, row]
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro e _
    apply Finset.sum_congr rfl
    intro t _
    ring
  have first (e t) : x e t - row e - avg p (fun i => x i t - row i) = r e t := by
    rw [avg_sub]
    change x e t - row e - (col t - grand) = x e t - row e - col t + grand
    ring
  have second (e t) : x e t - col t - avg q (fun j => x e j - col j) = r e t := by
    rw [avg_sub, swap]
    change x e t - col t - (row e - grand) = x e t - row e - col t + grand
    ring
  refine ⟨first, second, ?_, ?_⟩
  · intro e
    change avg q (r e) = 0
    calc
      _ = avg q (fun j => x e j - col j - avg q (fun i => x e i - col i)) :=
        congrArg (avg q) (funext fun j => (second e j).symm)
      _ = 0 := by rw [avg_sub, avg_const q hq, sub_self]
  · intro t
    change avg p (fun e => r e t) = 0
    calc
      _ = avg p (fun i => x i t - row i - avg p (fun j => x j t - row j)) :=
        congrArg (avg p) (funext fun i => (first i t).symm)
      _ = 0 := by rw [avg_sub, avg_const p hp, sub_self]

variable {G : Type*} [DecidableEq G]

/-- Weighted total over all observed rows bearing a label. -/
def groupTotal (g : I → G) (s : I → ℝ) (k : G) : ℝ :=
  ∑ i, if g i = k then s i else 0

/-- One exact group sweep on the observed roster. -/
noncomputable def centerBy (g : I → G) (w x : I → ℝ) (i : I) : ℝ :=
  x i - groupTotal g (fun j => w j * x j) (g i) / groupTotal g w (g i)

private theorem centered_total (g : I → G) (w x : I → ℝ) (k : G) (h : 0 < groupTotal g w k) :
    groupTotal g (fun i => w i * centerBy g w x i) k = 0 := by
  let s := fun i => w i * x i
  let m := groupTotal g s k / groupTotal g w k
  calc
    _ = ∑ i, ((if g i = k then s i else 0) - (if g i = k then w i else 0) * m) := by
      apply Finset.sum_congr rfl
      intro i _
      by_cases hi : g i = k <;> simp [hi, centerBy, s, m, mul_sub]
    _ = groupTotal g s k - groupTotal g w k * m := by
      rw [Finset.sum_sub_distrib, ← Finset.sum_mul]; rfl
    _ = 0 := by dsimp [m]; field_simp [h.ne']; ring

/-- At an exact fixed point of the two-step sweep, both sets of normal equations hold. -/
@[blueprint "STATADIC-PANEL-004"
  (statement := /-- For positive row weights and occupied groups, suppose a full
  entity-then-time sweep leaves an array $x$ exactly unchanged. Its entity and
  period weighted totals both vanish. Thus checking an exact fixed point, in
  addition to the additive form of the removed effects, suffices for the
  weighted two-way projection certificate. -/)
  (proof := /-- Write $y=x-A$ after the entity step and $x=y-B$ after the
  time step, so $B=-A$ rowwise. The intermediate $y$ is orthogonal to every
  entity effect and the final $x$ to every time effect. Hence the sum of
  $w_i A_i^2=w_i(x_i-y_i)A_i$ vanishes. Positivity forces $A_i=0$ in
  every row, so $y=x$ and the entity equations also hold. -/)]
theorem exact_sweep_normal_equations [DecidableEq E] [DecidableEq T]
    (e : I → E) (t : I → T) (w x : I → ℝ) (hw : ∀ i, 0 < w i)
    (he : ∀ k, 0 < groupTotal e w k) (ht : ∀ k, 0 < groupTotal t w k)
    (fix : centerBy t w (centerBy e w x) = x) :
    (∀ k, groupTotal e (fun i => w i * x i) k = 0) ∧
      (∀ k, groupTotal t (fun i => w i * x i) k = 0) := by
  let y := centerBy e w x
  let A := fun k => groupTotal e (fun i => w i * x i) k / groupTotal e w k
  let B := fun k => groupTotal t (fun i => w i * y i) k / groupTotal t w k
  have yt (k) := centered_total e w x k (he k)
  have xt (k) : groupTotal t (fun i => w i * x i) k = 0 := by
    simpa only [y, fix] using centered_total t w y k (ht k)
  have hA := group_orthogonal e (fun i => w i * y i) A yt
  have hB := group_orthogonal t (fun i => w i * x i) B xt
  have minus (i) : B (t i) = -A (e i) := by
    have h := congrFun fix i
    change x i - A (e i) - B (t i) = x i at h
    linarith
  have hz : (∑ i, w i * A (e i) ^ 2) = 0 := by
    calc
      _ = ∑ i, (-(w i * x i * B (t i)) - w i * y i * A (e i)) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [minus i]
        change _ = -(w i * x i * -A (e i)) - w i * (x i - A (e i)) * A (e i)
        ring
      _ = 0 := by rw [Finset.sum_sub_distrib, Finset.sum_neg_distrib, hB, hA]; ring
  have xy : y = x := by
    funext i
    have hi := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j _ => mul_nonneg (hw j).le (sq_nonneg (A (e j))))).mp hz i (Finset.mem_univ i)
    change x i - A (e i) = x i
    rw [sq_eq_zero_iff.mp ((mul_eq_zero.mp hi).resolve_left (hw i).ne'), sub_zero]
  exact ⟨fun k => by simpa only [← xy] using (show groupTotal e (fun i => w i * y i) k = 0 from yt k),
    xt⟩

/-- A missing cell can make a single ordered sweep differ from its reverse. -/
@[blueprint "STATADIC-PANEL-005"
  (statement := /-- With unit weights on the three observed cells
  $(0,0),(0,1),(1,0)$ and responses $(0,2,0)$, the entity-then-time
  residual in the first cell is $-1/2$, whereas time-then-entity gives zero. -/)
  (proof := /-- The entity-first intermediate is $(-1,1,0)$; the time sweep
  gives $(-1/2,0,1/2)$. In the other order the first time sweep already
  removes the entire response, which depends only on time. -/)]
theorem missing_cell_order :
    let e := fun i : Fin 3 => i == 2
    let t := fun i : Fin 3 => i == 1
    let x := fun i : Fin 3 => if i = 1 then (2 : ℝ) else 0
    let w := fun _ : Fin 3 => (1 : ℝ)
    centerBy t w (centerBy e w x) 0 = -1/2 ∧ centerBy e w (centerBy t w x) 0 = 0 := by
  have hA : (Finset.univ.filter fun i : Fin 3 => ¬ i = 2).card = 2 := by decide
  have hB : (Finset.univ.filter fun i : Fin 3 => i = 1).card = 1 := by decide
  norm_num [centerBy, groupTotal, Fin.sum_univ_succ, hA, hB]

/-- Any exact fixed point reached by iterated two-way sweeps is the global
weighted dummy projection on the original data. -/
@[blueprint "STATADIC-PANEL-006"
  (statement := /-- Starting from the observed column, repeat entity-then-time
  weighted demeaning any finite number of times. If the resulting $r$ is an
  exact fixed point, it gives the unique least-squares residual on the roster:
  for every additive fit $a_e+b_t$, its weighted loss equals the loss of $r$
  plus $\sum_i w_i[x_i-r_i-a_{e_i}-b_{t_i}]^2$. -/)
  (proof := /-- By induction, every sweep subtracts only an entity vector and
  a time vector, so $x-r$ belongs to the additive design space. The exact
  fixed-point theorem gives both sets of normal equations. The weighted
  projection certificate now gives the squared-error identity. -/)]
theorem iterated_sweep_projection [DecidableEq E] [DecidableEq T]
    (e : I → E) (t : I → T) (w x : I → ℝ) (hw : ∀ i, 0 < w i)
    (he : ∀ k, 0 < groupTotal e w k) (ht : ∀ k, 0 < groupTotal t w k) (n : ℕ) :
    let sweep := fun z => centerBy t w (centerBy e w z)
    let r := sweep^[n] x
    sweep r = r → ∀ (a : E → ℝ) (b : T → ℝ),
      (∑ i, w i * (x i - (a (e i) + b (t i))) ^ 2) =
      (∑ i, w i * r i ^ 2) + ∑ i, w i * (x i - r i - (a (e i) + b (t i))) ^ 2 := by
  intro sweep r fix a b
  have span (m : ℕ) : ∃ (u : E → ℝ) (v : T → ℝ),
      ∀ i, x i = (sweep^[m] x) i + (u (e i) + v (t i)) := by
    induction m with
    | zero => exact ⟨fun _ => 0, fun _ => 0, by simp⟩
    | succ m ih =>
      obtain ⟨u, v, huv⟩ := ih
      let prev := sweep^[m] x
      let A := fun k => groupTotal e (fun i => w i * prev i) k / groupTotal e w k
      let B := fun k => groupTotal t (fun i => w i * centerBy e w prev i) k / groupTotal t w k
      refine ⟨fun k => u k + A k, fun k => v k + B k, fun i => ?_⟩
      rw [Function.iterate_succ_apply']
      change x i = (prev i - A (e i) - B (t i)) + _
      linarith [huv i]
  obtain ⟨u, v, hf⟩ := span n
  obtain ⟨er, tr⟩ := exact_sweep_normal_equations e t w r hw he ht fix
  have gap := (weighted_two_way_certificate e t w x r u a v b hw hf er tr).1
  convert gap using 2
  apply Finset.sum_congr rfl
  intro i _
  have hi : x i - r i - (a (e i) + b (t i)) = u (e i) - a (e i) + (v (t i) - b (t i)) := by
    linarith [hf i]
  rw [hi]

end StatadicFormal.Operators.Panel
