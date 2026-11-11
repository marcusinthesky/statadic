---
description: Why the proof compiler, blueprint extraction and rendering stay together.
---

# Lean and its blueprint

The pinned Nix compiler supplies Lean and Lake for Linux x86-64 and ARM64. The
`lean-toolchain` file declares the same version for editors; a setup gate rejects
a mismatch before compiling. This avoids an imperative Elan installation that
can reference garbage-collected Nix libraries.

Mathlib and LeanArchitect follow the compiler version. The complete Lake
manifest fixes their Git revisions; checks never call `lake update`. Setup is
the network boundary, and checks verify prepared revisions before compiling.

LeanArchitect owns extraction from compiled annotations; LeanBlueprint owns
HTML rendering. Calling plasTeX directly supports this nested Lake project:
LeanBlueprint's convenience CLI assumes that the Lake and Git roots coincide.
The blueprint's Python environment is separate from application Python. Its
pygraphviz extension builds against pinned Nix Graphviz, avoiding a wheel's
bundled native libraries. Tectonic renders the same content as PDF.

Blueprint tools accompany Lean even when the separate LaTeX manuscript language
is absent. There is no deployment or manually authored proof-status database. Keep research
assumptions explicit in theorem hypotheses; mathematical explanation still
requires human review.

## Literature bridge

One StatadicFormal library groups proofs by mathematical dependency. The registry
associates stable scientific IDs with exact published statements and proof
locations; it is metadata, not a proof. Source specializations and proof steps
retain their limits. Helpers support the identified argument rather than create
new advertised statistical results.

Reusable metric geometry stays below statistical specializations.
`Geometry.NegativeType.Defs` carries only the finite metric predicate, so
consumers can state assumptions without importing Gaussian probability proofs.
`Geometry.NegativeType.Basic` owns the Hilbert proof and the Gaussian projection
operations shared with mean-embedding consumers. Energy proofs import that owner;
they do not maintain another predicate or a forwarding theorem.

The compiled census audits every first-party declaration independently of root
imports. The bridge joins this census to native LeanArchitect labels and reviewed
source mappings; neither the native ready flag nor a citation key certifies
source fidelity. Statement pins detect changes to types, assumptions and source
mappings, while source hashes reject stale build evidence. Published source and
Python correspondence still require mathematical review.

The existing governed reference records own bibliographic metadata. One generated
BibTeX database serves LaTeX and Sphinx. The docs app owns presentation and depends
on the source bridge; the statistical runtime never imports proof or site tools.

## Conventional quality

Existing pinned Batteries and Mathlib own lint rules, index generation and text fixes.
Lake owns the lint driver and native diagnostics; import-graph owns optional import
analysis. A small path adapter supplies nested source paths to Mathlib, without new
rules or dependencies. Full compilation belongs to the Lean check and an explicit
manual hook, not ordinary pre-commit. Scientific trust and provenance audits remain
independent. See the [quality guide](tools/quality/README.md).
