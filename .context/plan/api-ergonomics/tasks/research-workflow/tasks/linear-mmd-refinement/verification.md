# Verification

The theorem compiles with `devenv shell -- just lean::build`. An independent
integrator reviewed its equal-mass reference cancellation, real inner-product
expansion and weighted source correspondence. Human semantic review is pending;
the identity does not verify floating-point or XLA execution.

The old correspondence pin for `_mmd_squared_core` predates dtype promotion
for linear kernels. The source change and `_linear_mmd_cross` were reviewed
before explicitly refreshing correspondence pins; unresolved gates and
`python_formally_verified: false` remain unchanged.
`devenv shell -- just lean::registry-test` passes all 34 fixtures and the
correspondence check after making a fixture independent of claim ordering.

The four Gram-versus-mean witness cases passed with float32 and float64 masses,
integer and floating input coordinates of order one million, nonuniform masses
and zero-mass padding. They are numerical witnesses, not error bounds.

`just lean::statement-pin` initially failed because the new claim page was
missing; the page has since been added. Statement pinning and certification
are deferred until the integrating agent reviews and refreshes the shared public
API inventory. The implementation task remains open.

## Integration review

An independent review on 2026-11-05 probed each acceptance criterion directly against
the integrated source and then passed the full `just check`. See the
[integration verification](../../verification.md).
