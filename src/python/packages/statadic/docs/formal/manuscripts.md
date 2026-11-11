# Reuse mathematical statements in a manuscript

Run `just lean::certify` in the pinned environment. Certification joins compiled
Lean declarations to native LeanArchitect annotations and writes ignored fragments
under `src/lean/build/manuscript/`. It never rewrites Python docstrings.

Copy the generated directory into your paper's retained inputs, then load the
self-contained macro registry. Your manuscript owns numbering, labels and theorem
environments:

```latex
\input{generated/statadic-mathematics.tex}
\begin{theorem}\label{thm:mmd-unbiased}
  \statadicstatement{STATADIC-MMD-001}
\end{theorem}
\begin{proof}
  \statadicproof{STATADIC-MMD-001}
\end{proof}
\paragraph{Assumptions}
\statadicassumptions{STATADIC-MMD-001}
\paragraph{Checks on applicability}
\statadicdiagnostics{STATADIC-MMD-001}
```

The assumptions and diagnostic lists come from the reviewed claim registry. They can
include Unicode mathematical notation. Use a Unicode-capable TeX engine and a
font covering the symbols when including these lists. Record the diagnostics
observed for your own data alongside the assumptions you judge credible.

The same reviewed annotation supplies the website and the statement-only TeX
fragment. The proof fragment is human-readable mathematical exposition, not an
automatic translation of a proof term. Unknown claim IDs fail explicitly.

```{raw} html
<a href="downloads/manuscript/statadic-mathematics.tex">Download the certified macro registry</a>.
```

## Review changes

A statement pin binds the elaborated type, source mapping, mathematical annotation
and the entire audited first-party Lean source graph, including the dependency
manifest and toolchain. Thus changing a definition invalidates review even when a
theorem's printed type is unchanged. This conservative policy also requests review
after unrelated first-party edits. After reviewing the changes, explicitly run
`just lean::statement-pin`; routine checks never repair pins.

Published claims require precise source/proof locations and a governed source
dossier. `original-result` and `algorithm-refinement` claims instead require an
explicit review record and must not invent a publication source. A refinement
claim must identify what algorithm and arithmetic model it covers.

## Implementation verification research

The current MMD theorem concerns a finite probability law and supplied pair
moments. It does not verify the Python/JAX implementation. An end-to-end argument
must separately connect `masked_sum`, `pair_chunks`, packed reductions, scalar
reductions, kernel semantics and compiled numerical execution.

`StatadicFormal.Discrepancy.MMD.maskedReduction_eq_leading` proves the exact rational
masking identity. The source-bound prototype records the actual Python ASTs for
`masked_sum`, `pair_chunks`, and the public/packed MMD matrix routines. Check it
with the locked docs runner and `src/lean/tools/correspondence.py check`; only
explicit `pin` accepts reviewed source changes. Conformance witnesses exercise
small leading blocks (including poisoned padding) and upper-triangle schedules.
These are tests, not proofs of Python semantics.

The remaining research gates are: upper-triangle
enumeration and padding exclusion; packed/scalar MMD equivalence; correspondence
to actual source operations; IEEE rounding/error bounds; compiler and runtime
trust. A successful conformance test supports only the tested source and inputs.
No gate is closed merely because a mathematical theorem compiled.
