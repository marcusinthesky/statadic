set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

mod python 'src/python/justfile'
mod lean 'src/lean/justfile'
mod latex 'src/latex/justfile'
mod agents '.agents/justfile'
workflow := "env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV uv run --offline --frozen --script src/python/scripts/workflow/script.py"

# Discover this project's commands.
default:
    @just --list

# Prepare locked dependencies and Git hooks (uses the network).
setup:
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV uv run --frozen --script src/python/scripts/workflow/script.py prepare
    just python::setup
    just lean::setup
    just latex::setup
    just agents::apply
    prek install

# Check source, views and selected languages without repairing files.
check:
    bash infra/nix/check.sh
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV uv run --project src/python --offline --frozen --package statadic-docs python -m unittest discover -s infra/nix -p 'test_*.py'
    {{ workflow }} check
    # Documentation tests consume the generated Lean certification.
    just lean::check
    just python::check
    just latex::check

    just agents::check

# Run the selected language tests.
test:
    just python::test
    just lean::test
    just latex::test

# Explicitly format sources.
fmt:
    nixfmt devenv.nix infra/nix/devenv.nix infra/nix/pkgs/schematter.nix infra/nix/pkgs/lean-toolchain.nix infra/nix/pkgs/agentsync.nix
    rumdl fmt .
    just python::fmt
    just lean::fmt
    just latex::fmt

# Refresh generated indexes, child views and bibliography.
sync:
    {{ workflow }} sync

# Create a plan without overwriting existing content.
plan-new slug title:
    {{ workflow }} plan-new "$1" --title "$2"

# Create a task inside an existing plan or task.
task-new parent slug title:
    {{ workflow }} task-new "$1" "$2" --title "$3"

# Create a citation record with its directory as citation key.
reference-new key title:
    {{ workflow }} reference-new "$1" --title "$2"

# Check external links online.
links:
    git ls-files -z '*.md' | xargs -0 -r lychee --config lychee.toml --no-progress

# Qualify retained artifacts and write revision-linked release evidence.
release-check *args:
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV uv run --project src/python --offline --frozen --package statadic-docs python src/python/apps/docs/scripts/release.py "$@"

# Run bounded hosted checks; this does not establish scientific qualification.
release-preflight:
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV uv run --project src/python --offline --frozen --package statadic-docs python src/python/apps/docs/scripts/release.py --mode preflight --output build/release-preflight --archive build/release-preflight.tar

# Prepare live scientific comparators without pruning the development workspace.
setup-references:
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="{{justfile_directory()}}/.cache/statadic-reference-venv" uv sync --project src/python --locked --package statadic --group reference

# Require complete live comparisons and retain separate precision evidence.
validate-references *args:
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="{{justfile_directory()}}/.cache/statadic-reference-venv" UV_OFFLINE=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run --project src/python --offline --frozen --package statadic --group reference python src/python/packages/statadic/scripts/check_references.py --catalogue src/python/apps/studies/catalogue.json "$@"
