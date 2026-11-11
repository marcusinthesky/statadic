# Operations verification

Implementation passed independent review and final checkpoint 2 establishes
acceptance. The coordinator marked this task done from that evidence.
The publishing checks passed five tests in both default and x64-enabled modes
after structural TOML staging and independent failure reporting were corrected.
After the effective-precision correction, seven tests passed. All commands used
`devenv shell --` and locked projects.

Tests build and verify the actual example, compare repeated deterministic archive
bytes, and exercise tampered output, stale input, schema/path rejection, existing
destinations, analysis failure and unresolved citations. Failed stages retain
logs and `FAILED`, with no successful manifest.

Review corrections:

- Replaced double-quoted text substitution with structural TOML parsing/writing
  of only the staged external fields. A legal single-quoted path and an escaped
  unrelated title survive build, verification and archive.
- Verification isolates checks and reports simultaneous missing, changed and
  unavailable outputs, rather than stopping at one missing file. Declared output
  identity and media type are checked as well.
- The analysis child receives the caller's effective JAX x64 setting even when
  the inherited environment conflicts. Real subprocess probes test both modes.
- Rebuild instructions name package and engine versions and the x64 setting.
- The architecture check initially rejected an undeclared direct JAX import.
  The companion now declares the same JAX requirement as core; the refreshed
  lock and architecture check passed, including all 13 guard tests.

The TOML writer was already present in the shared lock. Evidence serialization,
inventories and deterministic archives reuse `statadic-evidence`.

The final repository check and subsequent distribution gate passed; see the
[integrated results](../../../research-release/verification.md).
