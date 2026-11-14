# jaxtyping shape strings are runtime contracts.
"""Public-surface guard for the dissolved ``statadic.kernels`` package.

``statadic.kernels`` was a single module, then a package, then a t46.6 deprecation
shim. **t46.10 deleted it.** Nothing named "kernel" in it was ever a Mercer/RBF
kernel, so its contents were redistributed to the stages that own them — the
energy-distance and energy-test machinery to ``statadic.discrepancy``, the
barycentre solvers to ``statadic.geometry``, and the PGD simplex solver to
``statadic.optimize``.

Nothing else pins that surface: ruff cannot see a name silently dropped from an
``__all__``, pyrefly is blind to the private symbols downstream code imports
directly, and the numerical suites only fail at collection time. These tests
assert the contract itself, in the three planes it has to hold in:

1. the ``statadic.kernels`` path is **gone**, so nothing can quietly re-acquire a
   dependency on it (this inverts the wave-B compatibility assertion, which is
   no longer expressible);
2. every name resolves on the **module that now owns it** — this is what would
   catch a re-export that silently forked;
3. every public name resolves on its **stage façade**, which is the path
   post-t46.10 code uses.

Before t46.6 the plane-2 mapping pointed at ``statadic.kernels.{_barycentre,
_common,distance,testing}``. Those submodules are gone by design, so the
pre-t46 spelling of this file was stale rather than a regression.
"""

import ast
import importlib
import pathlib

import pytest

#: Legacy paths t46.10 retired. Deleted, not deprecated.
_RETIRED_MODULE_PATHS = [
    "statadic.kernels",
    "statadic.discrepancy._kernels.simplex",
]

# Every name the `statadic.kernels` shim exported, mapped to the module that owns
# its definition after t46.6/t46.10. Private entries are here because
# downstream code imports them directly:
# `_energy_barycentre_weights_batched_core` and `_compute_energy_components` in
# the pipeline and simulation apps.

# The public half of the surface, mapped to the stage façade that exports it.
# The two names absent here are private and deliberately have no public path.

_EXPECTED_OVERLOADS = 2
_OVERLOADED = "energy_barycentre_weights"


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("dotted", _RETIRED_MODULE_PATHS)
def test_retired_shim_path_no_longer_imports(dotted: str) -> None:
    """The pre-t46.10 module paths are deleted, not merely deprecated.

    Args:
        dotted: A legacy ``statadic`` module path retired by t46.10.

    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(dotted)


@pytest.mark.integration
@pytest.mark.contract
def test_overloads_live_with_their_implementation() -> None:
    """`energy_barycentre_weights` keeps both `@overload` stubs, in one module.

    A stub separated from its implementation across a module boundary is
    silently green at runtime and invisible to ruff. The obvious guard —
    ``len(typing.get_overloads(...)) == 2`` — cannot be used here: the statadic test
    session installs the jaxtyping/beartype import hook (``conftest.py``), which
    wraps each stub so that both register under the *wrapper's*
    ``co_firstlineno`` and ``typing.get_overloads`` collapses 2 into 1. That is
    a hook/``typing`` incompatibility, not a missing stub, and it is why this
    assertion was a strict xfail until t46.6.

    Reading the source with ``ast`` pins the same contract hook-independently,
    and pins something stronger besides: that the stubs and the implementation
    are in the *same* module, which is the hazard the runtime count was only an
    indirect proxy for.
    """
    owner = importlib.import_module("statadic.geometry.barycentre.mmd.energy")
    implementation = importlib.import_module(getattr(owner, _OVERLOADED).__module__)
    source = pathlib.Path(str(implementation.__file__)).read_text(encoding="utf-8")
    defs = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == _OVERLOADED
    ]
    decorated = [
        node
        for node in defs
        if any(
            isinstance(d, ast.Name) and d.id == "overload" for d in node.decorator_list
        )
    ]
    assert len(decorated) == _EXPECTED_OVERLOADS, (
        f"{_OVERLOADED} declares {len(decorated)} @overload stubs in "
        f"{implementation.__name__}, expected {_EXPECTED_OVERLOADS}"
    )
    assert len(defs) == _EXPECTED_OVERLOADS + 1, (
        f"the implementation of {_OVERLOADED} is not in {implementation.__name__} "
        "alongside its stubs"
    )
