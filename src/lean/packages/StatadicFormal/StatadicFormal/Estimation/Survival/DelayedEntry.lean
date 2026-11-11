module

public import Architect
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic.Ring

/-! Conditioning a censored row on survival to its entry time. -/
@[expose] public section
noncomputable section
namespace StatadicFormal.Estimation.Survival

/-- The joint event/censor likelihood factorizes after delayed entry.
The second factor can be dropped for fitting failure parameters when it
contains no such parameter. -/
@[blueprint "STATADIC-SURVIVAL-001"
  (statement := /-- At exit $t$ write $f,S$ for the failure density and
  survival and $g,G$ for the censor density and survival. At entry $a<t$
  let $S_a,G_a>0$. Conditional independence gives joint contributions
  $fG/(S_aG_a)$ for an event and $Sg/(S_aG_a)$ for censoring.
  With positive evaluated factors, their log splits as
  \[
  (\log(f\text{ or }S)-\log S_a)
  +(\log(G\text{ or }g)-\log G_a).
  \]
  Survival beyond entry has conditional probability $S/S_a\in[0,1]$
  if $S\le S_a$. -/)
  (proof := /-- Select the numerator according to the observed indicator.
  Split the logarithm of the product and quotient using positivity. The
  censoring term is constant across failure parameters when its model is
  separate. Dividing $0<S\le S_a$ by $S_a>0$ bounds the conditional tail. -/)]
theorem delayed_entry_row (event : Bool) (f s g c sa ca : ℝ)
    (hf : 0 < f) (hs : 0 < s) (hg : 0 < g) (hc : 0 < c)
    (hsa : 0 < sa) (hca : 0 < ca) (horder : s ≤ sa) :
    Real.log ((if event then f * c else s * g) / (sa * ca)) =
      (Real.log (if event then f else s) - Real.log sa) +
        (Real.log (if event then c else g) - Real.log ca) ∧
      0 ≤ s / sa ∧ s / sa ≤ 1 := by
  refine ⟨?_, div_nonneg hs.le hsa.le, (div_le_one hsa).mpr horder⟩
  cases event <;>
    simp [Real.log_div, Real.log_mul, *, ne_of_gt] <;> ring

/-- Density transformation for an AFT location-scale family on log time. -/
@[blueprint "STATADIC-SURVIVAL-002"
  (statement := /-- Suppose $t>0$, $\sigma>0$, $z=(\log t-\eta)/\sigma$,
  and the standardized log-time law has survival $S_0$ with derivative
  $-p$ at $z$, where $p>0$. The AFT survival has derivative
  $-p/(t\sigma)$ at $t$, so its log event density on original duration units
  is $\log p-\log t-\log\sigma$. -/)
  (proof := /-- Differentiate the log-time standardization to obtain
  $1/(t\sigma)$. Compose with the baseline survival derivative, then use
  positivity to split the logarithm of the density into three terms. -/)]
theorem aft_density_jacobian (S : ℝ → ℝ) (t η σ p : ℝ)
    (ht : 0 < t) (hσ : 0 < σ) (hp : 0 < p)
    (hS : HasDerivAt S (-p) ((Real.log t - η) / σ)) :
    HasDerivAt (fun u => S ((Real.log u - η) / σ)) (-(p / (t * σ))) t ∧
      Real.log (p / (t * σ)) = Real.log p - Real.log t - Real.log σ := by
  constructor
  · have hz := ((Real.hasDerivAt_log ht.ne').sub_const η).div_const σ
    have hscale : (-p) * (t⁻¹ / σ) = -(p / (t * σ)) := by ring
    simpa only [Function.comp_def, hscale] using hS.comp t hz
  · rw [Real.log_div hp.ne' (mul_pos ht hσ).ne', Real.log_mul ht.ne' hσ.ne']
    ring

end StatadicFormal.Estimation.Survival
