# Infrastructure adjudication

The coordinator and independent challenger read all fourteen assigned diffs;
exact paths are in [the inspection log](reviews.json). No introduced defect was
established. Retain the current direction with the execution limits below.

CI keeps x86-64 and ARM64 failures independent, fetches the LFS TeX bundle and
retains setup/check/capacity diagnostics. Hosted preflight explicitly omits long
scientific qualification; its archive is not a release-qualification claim.
Source and reference jobs have separate scopes.

The capacity script creates its log directory before pipelines use it and gates
SDK removal to disposable GitHub-hosted runners. It must not be invoked in
prepare mode locally. The TeX retry wrapper retries diagnosed HTTP 429/5xx
failures with bounded delays; other command failures preserve their exit status.

The replacement pinned Lean derivation extracts the verified archive directly
into the Nix output and retains compiler/include replacement, executable wrapper
PATH and ELF patching responsibilities. The challenger compared the complete
cached upstream toolchain implementation. Successful local devenv entry supplies
bounded environment evidence, not clean-machine or ARM64 qualification.

Generated ROADMAP/agent guidance and task views must stay owned by their explicit
creation/synchronization commands. The review adds no new CI layer or public
stability registry. Pre-release simplification should preserve diagnostic and
failure evidence; retain historical records with clear identities.

Not executed here: hosted provisioning, ARM64 jobs, LFS checkout on a new machine,
transient TeX retry faults and destructive capacity preparation. These are scoped
execution limits, not presumed passing outcomes. The final local `just check`
result is recorded independently in [verification](verification.md).
