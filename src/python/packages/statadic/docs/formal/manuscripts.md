---
type: explanation
description: Read and reuse mathematical statements with their hypotheses and source identities.
---

# Mathematical statements in a manuscript

## How to read this

Each claim page brings together a mathematical statement, its assumptions and the accompanying exposition, so preserve the claim identifier and source revision when citing a result. Explain which
conditions can be checked from the data and which depend on the research design.

The downloadable registry supplies LaTeX macros for statements, proofs,
assumptions and diagnostics. Use a Unicode-capable TeX engine with a font covering the symbols in the exported text, and check the hypotheses against your analysis.

```latex
\input{generated/statadic-mathematics.tex}
\begin{theorem}\label{thm:mmd-unbiased}
  \statadicstatement{STATADIC-MMD-001}
\end{theorem}
\begin{proof}
  \statadicproof{STATADIC-MMD-001}
\end{proof}
\statadicassumptions{STATADIC-MMD-001}
\statadicdiagnostics{STATADIC-MMD-001}
```

Read the hypotheses first. The proof fragment gives mathematical exposition for the statement alongside the conditions on which it depends, so those conditions remain part of the result when it appears in a manuscript. Verification of a numerical implementation
and translation of a Lean proof term are separate tasks.

```{raw} html
<a href="downloads/manuscript/statadic-mathematics.tex">Download the certified macro registry</a>.
```

For a worked analysis, see [citing a checked result](../guides/cite-a-checked-result.py).
Repository procedures for certification and source review live in the
[Lean contributor guide](https://github.com/marcusinthesky/statadic/blob/main/src/lean/README.md).
