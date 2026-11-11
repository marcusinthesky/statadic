# Documentation verification — 31 October 2026

## Delivered

Sphinx/MyST with PyData theme, a concise learning homepage, three executable
tutorials, a method chooser, glossary, troubleshooting, curated live API reference,
and six native marimo labs with static explanations, exports and downloads.
The original technical pages remain available; their question index is now
`docs/methods.md`. The full package README is retained beneath its new entry path.

The internal `statadic-docs` Python application owns documentation tools and lab
helpers. STATADIC runtime metadata is unchanged. SHA-256 checks confirmed all 421
scientific source/test files unchanged. No estimators, defaults or production
tolerances were changed.

## Executed checks

- `devenv shell -- just check`: hooks, workflow checks, STATADIC lint/types, complete
  no-Testmon fast suite and wheel/sdist build passed. The suite recorded
  **2,677 passed, 161 skipped, five warnings**, in 470.15 seconds. The command
  then exposed a docs module path error under nested workspace invocation.
- Fixed the docs recipe to resolve its own source directory. Root and workspace
  dry runs now resolve the same application. Resumed with
  `just python::docs::check`: **50 passed** with x64 disabled and **nine passed**
  in a separate x64 native process; all three tutorial scripts ran in both modes,
  all six strict marimo checks passed, and the strict Sphinx build and six
  executed code-free HTML exports passed. Generated snapshot/download links
  are checked by the build. The scientific suite was not needlessly repeated.
- `just python::statadic::test-artifacts`: both wheel routes and isolated standalone
  sdist development passed, including **45 typing tests**. This checks runtime
  closures without documentation dependencies.
- `just lean::check`, `just latex::check`, and `just agents::check` passed,
  completing the remaining root gates separately after the path repair.
- All three tutorials also passed in both precision modes in a clean installed
  environment containing only STATADIC runtime dependencies (JAX 0.11.2, NumPy
  2.5.4). This is tutorial evidence, not a qualification of those versions for
  every scientific procedure.
- Real Chrome checks exercised all three form submissions, geometry keyboard
  controls and the zero-vector guard, energy reactivity, and a covariance switch
  that retained the fitted regression slope. Search returned energy lessons;
  keyboard focus reached the skip link; the 390-pixel layout had no horizontal
  overflow. API autodoc, snapshot navigation, static content without JavaScript,
  and local generated targets were inspected. No JavaScript errors were seen.

The workspace retains its qualified JAX/JAXlib 0.11.0, NumPy 2.4.6, SciPy 1.18.0,
and pytest 8.4.2 versions. Documentation dependencies are locked separately in
ownership, within the shared workspace lock; marimo is 0.25.2.

## Limits

Slow, GPU, optional external-reference and statistical qualification suites were
not run. The 161 fast-suite skips are not passes. External network link checks,
ARM execution and usability testing with actual students were not performed.
The teaching calibration simulation does not establish statistical qualification.
No new Lean proof, public deployment, commit or push was made.
