module

public import Mathlib.Basic.Real.Basic
public import Architect
import Mathlib.Tactic.Ring

/-! A first checked statement using Mathlib's real numbers and ring tactic. -/

public section

namespace Example

/-- Expanding the square of a sum is an identity, with no extra assumptions. -/
@[blueprint "thm:square-add"
  (statement := /-- For real $x,y$, $(x+y)^2=x^2+2xy+y^2$. -/)
  (proof := /-- Expand both sides using the ring identities. -/)]
theorem square_add (x y : ℝ) : (x + y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 := by
  ring

@[blueprint "thm:square-double"
  (statement := /-- For real $x$, $(x+x)^2=x^2+2xx+x^2$. -/)
  (proof := /-- Apply \cref{thm:square-add} with $y=x$. -/)]
theorem square_double (x : ℝ) : (x + x) ^ 2 = x ^ 2 + 2 * x * x + x ^ 2 :=
  square_add x x

end Example
