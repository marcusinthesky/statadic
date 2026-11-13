module

public import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Tactic.Ring
public import Architect
public import StatadicFormal.Probability.Finite

public section

open scoped BigOperators
namespace StatadicFormal.Discrepancy.Energy

/-- A finite signed weighted kernel quadratic form. -/
@[expose]
def quadratic {ι : Type*} [Fintype ι] (k : ι → ι → ℝ) (c : ι → ℝ) : ℝ :=
  ∑ i, ∑ j, c i * c j * k i j

/-- Finite-support algebra underlying Theorem 22 of Sejdinovic et al. (2013).
The generating kernel relation and zero total signed mass yield energy equal
 to twice the kernel quadratic form. This makes no positivity claim. -/
@[blueprint "STATADIC-ENERGY-001"
  (statement := /-- Let $c_i$ be real weights on a finite set with
  $\sum_i c_i=0$. Suppose the real arrays $d$ and $k$ satisfy
  \[
    d_{ij}=k_{ii}+k_{jj}-2k_{ij}.
  \]
  Then the signed quadratic forms satisfy
  \[
    -\sum_i\sum_j c_i c_j d_{ij}
      =2\sum_i\sum_j c_i c_j k_{ij}.
  \]
  This algebraic identity does not assume kernel positivity. -/)
  (proof := /-- Substitute the generating relation. The two diagonal terms vanish
  because $\sum_i c_i=0$, leaving
  \[
    -\sum_{i,j}c_i c_j(k_{ii}+k_{jj}-2k_{ij})
      =2\sum_{i,j}c_i c_j k_{ij}.
  \] -/)]
theorem finite_energy_kernel_identity {ι : Type*} [Fintype ι]
    (k d : ι → ι → ℝ) (c : ι → ℝ)
    (hzero : ∑ i, c i = 0)
    (hgen : ∀ i j, d i j = k i i + k j j - 2 * k i j) :
    -quadratic d c = 2 * quadratic k c := by
  unfold quadratic
  simp_rw [hgen, mul_sub, mul_add, Finset.sum_sub_distrib, Finset.sum_add_distrib]
  have hleft : (∑ i, ∑ j, c i * c j * k i i) = 0 := by
    calc
      _ = ∑ i, (c i * k i i) * (∑ j, c j) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = 0 := by simp [hzero]
  have hright : (∑ i, ∑ j, c i * c j * k j j) = 0 := by
    rw [Finset.sum_comm]
    calc
      _ = ∑ j, (c j * k j j) * (∑ i, c i) := by
        apply Finset.sum_congr rfl
        intro j _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = 0 := by simp [hzero]
  rw [hleft, hright]
  simp only [zero_add]
  have htwo : (∑ i, ∑ j, c i * c j * (2 * k i j)) =
      2 * (∑ i, ∑ j, c i * c j * k i j) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [htwo]
  ring

/-- Kernel positive semidefiniteness on this finite support. -/
@[expose]
def KernelPSD {ι : Type*} [Fintype ι] (k : ι → ι → ℝ) : Prop :=
  ∀ c : ι → ℝ, 0 ≤ quadratic k c

/-- Raw finite-support energy, with all pairs included. -/
@[expose] def rawEnergy {ι : Type*} [Fintype ι] (d : ι → ι → ℝ) (p q : ι → ℝ) : ℝ :=
  (2 * ∑ i, ∑ j, p i * q j * d i j) -
    (∑ i, ∑ j, p i * p j * d i j) - (∑ i, ∑ j, q i * q j * d i j)

/-- Symmetry turns raw energy into the negative quadratic of the weight difference. -/
theorem rawEnergy_eq_neg_quadratic {ι : Type*} [Fintype ι]
    (d : ι → ι → ℝ) (p q : ι → ℝ) (hd : ∀ i j, d i j = d j i) :
    rawEnergy d p q = -quadratic d (fun i ↦ p i - q i) := by
  have hcross : (∑ i, ∑ j, q i * p j * d i j) =
      ∑ i, ∑ j, p i * q j * d i j := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    rw [hd j i]
    ring
  have hexpand : quadratic d (fun i ↦ p i - q i) =
      (∑ i, ∑ j, p i * p j * d i j) -
      (∑ i, ∑ j, p i * q j * d i j) -
      (∑ i, ∑ j, q i * p j * d i j) +
      (∑ i, ∑ j, q i * q j * d i j) := by
    unfold quadratic
    simp only [sub_mul, mul_sub, Finset.sum_sub_distrib]
    ring
  rw [hexpand, hcross]
  unfold rawEnergy
  ring

/-- For a symmetric generating kernel, the finite empirical raw energy equals twice
its kernel quadratic form and is nonnegative when the kernel is positive semidefinite.
This finite-sum result is a specialization of Sejdinovic et al. (2013), Theorem 22;
equal mass is sufficient and no probability nonnegativity is needed for the algebra. -/
theorem finite_energy_nonneg {ι : Type*} [Fintype ι]
    (k d : ι → ι → ℝ) (p q : ι → ℝ)
    (hsymm : ∀ i j, k i j = k j i)
    (hgen : ∀ i j, d i j = k i i + k j j - 2 * k i j)
    (hpsd : KernelPSD k) (hmass : (∑ i, p i) = ∑ i, q i) :
    rawEnergy d p q = 2 * quadratic k (fun i ↦ p i - q i) ∧
      0 ≤ rawEnergy d p q := by
  have hd : ∀ i j, d i j = d j i := by
    intro i j
    rw [hgen, hgen, hsymm i j]
    ring
  have hzero : (∑ i, (p i - q i)) = 0 := by
    rw [Finset.sum_sub_distrib, hmass, sub_self]
  have hidentity := finite_energy_kernel_identity k d (fun i ↦ p i - q i) hzero hgen
  have henergy := rawEnergy_eq_neg_quadratic d p q hd
  constructor
  · rw [henergy, hidentity]
  · rw [henergy, hidentity]
    exact mul_nonneg (by norm_num) (hpsd _)

/-- Finite-probability specialization of Sejdinovic et al. (2013), Theorem 22.
The unrooted energy uses twice the cross mean minus both within-law means;
it equals twice the kernel quadratic form of the signed difference and is
nonnegative. The finite weights are normalized and nonnegative by `FiniteLaw`.
This statement does not construct an RKHS, take a square root, or verify
floating-point computation. -/
@[blueprint "STATADIC-ENERGY-002"
  (statement := /-- Let $p_i,q_i$ be nonnegative weights on a finite set, each
  summing to one. Let $k$ be symmetric and positive semidefinite, and suppose
  $d_{ij}=k_{ii}+k_{jj}-2k_{ij}$. Define the unrooted raw energy and kernel form by
  \[
    E_d(p,q)=2\sum_{i,j}p_iq_jd_{ij}
      -\sum_{i,j}p_ip_jd_{ij}-\sum_{i,j}q_iq_jd_{ij},
    \qquad Q_k(c)=\sum_{i,j}c_ic_jk_{ij}.
  \]
  Then
  \[
    E_d(p,q)=2Q_k(p-q)\geq 0.
  \]
  The equality uses the stated generating convention; no square root is taken. -/)
  (proof := /-- Set $c_i=p_i-q_i$. The probability weights have equal mass,
  so $\sum_i c_i=0$. Symmetry gives $E_d(p,q)=-\sum_{i,j}c_ic_jd_{ij}$.
  Apply the signed energy--kernel identity and positive semidefiniteness:
  \[
    E_d(p,q)=-\sum_{i,j}c_ic_jd_{ij}=2Q_k(c)\geq 0.
  \] -/)]
theorem finite_probability_energy_nonneg {ι : Type*} [Fintype ι]
    (k d : ι → ι → ℝ) (P Q : StatadicFormal.Probability.FiniteLaw ι)
    (hsymm : ∀ i j, k i j = k j i)
    (hgen : ∀ i j, d i j = k i i + k j j - 2 * k i j)
    (hpsd : KernelPSD k) :
    rawEnergy d P.weight Q.weight =
      2 * quadratic k (fun i ↦ P.weight i - Q.weight i) ∧
    0 ≤ rawEnergy d P.weight Q.weight := by
  apply finite_energy_nonneg k d P.weight Q.weight hsymm hgen hpsd
  rw [P.total, Q.total]

end StatadicFormal.Discrepancy.Energy
