import Lake
open Lake DSL

package proofs where
  lintDriver := "batteries/runLinter"
  leanOptions := #[⟨`warningAsError, true⟩]

require LeanArchitect from git
  "https://github.com/hanwenzhu/LeanArchitect" @ "9f3bb53916b9d6fa05b7777389e1844f82b1c9a6"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.35.0"

@[default_target]
lean_lib Example

@[default_target]
lean_lib StatadicFormal where
  srcDir := "packages/StatadicFormal"

-- Upstream Mathlib text rules; explicit paths support the nested source layout.
lean_exe qualityStyle where
  root := `tools.quality.Style
  supportInterpreter := true
