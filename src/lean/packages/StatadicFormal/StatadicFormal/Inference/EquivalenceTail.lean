module

public import StatadicFormal.Discrepancy.Energy.Finite
public import StatadicFormal.Probability.FiniteExtensions
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! A finite null example for the empirical bootstrap tail at a discrepancy margin. -/
@[expose] public section
noncomputable section
open StatadicFormal.Discrepancy.Energy StatadicFormal.Probability
namespace StatadicFormal.Inference

/-- Unrooted energy from Bernoulli(p) to a unit mass at zero, with the
absolute-distance ground on the two points. -/
def bernoulliEnergy (p : ℝ) : ℝ :=
  rawEnergy (fun i j : Bool => if i = j then 0 else 1)
    (fun i => if i then p else 1 - p) (fun i => if i then 0 else 1)

/-- Energy of the empirical resample determined by two selected row indices. -/
def pairResampleEnergy (a b : Bool) (rows : Bool → Bool) : ℝ :=
  let value := fun i : Bool => if (if i then b else a) then (1 : ℝ) else 0
  bernoulliEnergy ((value (rows false) + value (rows true)) / 2)

/-- Count bootstrap replicates at or above the equivalence margin, then add one. -/
def marginTail {B : ℕ} (δ : ℝ) (draw : Fin B → ℝ) : ℝ :=
  (1 + ((Finset.univ.filter (fun k => δ ≤ draw k)).card : ℝ)) / (1 + B)

/-- Add-one smoothing alone does not give a valid one-sided null p-value. -/
@[blueprint "STATADIC-EQUIVALENCE-001"
  (statement := /-- Let $X$ be fair Bernoulli and $Y=0$, with unrooted energy
  and absolute distance. Their population energy is the null margin $\delta=1/2$.
  Observe two iid rows of $X$. For any nineteen two-row resamples of those rows,
  let $p=(1+\#\{E_b^*\ge\delta\})/20$. Under the null,
  $\Pr(p\le 1/20)\ge1/4$. -/)
  (proof := /-- The population cross mean is $1/2$ and the within-$X$
  distance mean is $1/2$, giving energy $1/2$. With probability $1/4$
  both observed rows are zero. On that event every empirical resample has
  zero energy, so the numerator is one for every row schedule. Counting this
  event alone gives the lower bound. -/)]
theorem bootstrap_margin_null_counterexample (schedule : Bool → Bool → Fin 19 → Bool → Bool) :
    let law : FiniteLaw Bool := ⟨fun _ => 1 / 2, fun _ => by norm_num, by norm_num⟩
    let p := fun a b => marginTail (1 / 2) (fun k => pairResampleEnergy a b (schedule a b k))
    bernoulliEnergy (1 / 2) = 1 / 2 ∧
      (1 / 4 : ℝ) ≤ law.mean (fun a => law.mean (fun b => if p a b ≤ 1 / 20 then 1 else 0)) := by
  have energy (p : ℝ) : bernoulliEnergy p = 2 * p ^ 2 := by
    unfold bernoulliEnergy
    rw [rawEnergy_eq_neg_quadratic _ _ _ (fun i j => by simp [eq_comm])]
    simp [quadratic]
    ring
  have hzero (rows : Bool → Bool) : pairResampleEnergy false false rows = 0 := by
    norm_num [pairResampleEnergy, energy]
  have htail : marginTail (1 / 2)
      (fun k => pairResampleEnergy false false (schedule false false k)) = 1 / 20 := by
    norm_num [marginTail, hzero]
  constructor
  · norm_num [energy]
  · dsimp only
    simp only [FiniteLaw.mean, Fintype.sum_bool, htail, le_refl, ite_true]
    split_ifs <;> norm_num

end StatadicFormal.Inference
