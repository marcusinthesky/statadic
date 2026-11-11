module

public import Architect
public import Mathlib.Data.Finset.Card
public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

/-! Stability of a finite top-k neighbourhood under perturbation of its distance row. -/

public section
namespace StatadicFormal.Geometry

/-- A strict two-error boundary gap certifies top-k membership under a uniform row perturbation. -/
@[blueprint "STATADIC-NEIGHBOUR-001"
  (statement := /-- For one anchor, consider the same finite candidate roster before and
  after changing its distances from $d_i$ to $e_i$, with $|e_i-d_i|\le\epsilon$.
  If every member of $S$ is more than $2\epsilon$ closer than every nonmember,
  every perturbed top-$|S|$ set equals $S$. Ties within either side are allowed. -/)
  (proof := /-- If the equally sized sets differ, choose an original member $i$
  displaced by an outsider $j$. The gap gives
  $e_i\le d_i+\epsilon<d_j-\epsilon\le e_j$, whereas top-set membership
  requires $e_j\le e_i$. -/)]
theorem neighbour_set_of_boundary_gap {ι : Type*} (S T : Finset ι) (d e : ι → ℝ) (ε : ℝ)
    (hcard : S.card = T.card) (herr : ∀ i, |e i - d i| ≤ ε)
    (hgap : ∀ i ∈ S, ∀ j ∉ S, d i + 2 * ε < d j)
    (htop : ∀ j ∈ T, ∀ i ∉ T, e j ≤ e i) : S = T := by
  classical
  by_contra hne
  have hs : ¬ S ⊆ T := fun h ↦ hne (Finset.eq_of_subset_of_card_le h hcard.ge)
  have ht : ¬ T ⊆ S := fun h ↦ hne (Finset.eq_of_subset_of_card_le h hcard.le).symm
  obtain ⟨i, hi, hi'⟩ := Set.not_subset.mp hs
  obtain ⟨j, hj, hj'⟩ := Set.not_subset.mp ht
  linarith [(abs_le.mp (herr i)).2, (abs_le.mp (herr j)).1,
    hgap i hi j hj', htop j hj i hi']

end StatadicFormal.Geometry
