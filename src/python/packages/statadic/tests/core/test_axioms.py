# jaxtyping shape strings are runtime contracts.
"""``statadic.core.axioms`` — capabilities, flag algebra, and compatibility bridges.

Five planes:

* **capabilities** — read-only structural protocols compose without a nominal
  chain; the static accept/reject matrix lives in ``test_typing.py``.
* **algebra** — the Deza & Deza aliases are exactly the documented split flags,
  and strong negative type implies negative type independently.
* **bridge** — ``MARKER_AXIOMS`` connects the compile-time chain to the runtime
  flags. Without an assertion over the shipped ``CASES`` it is decorative: a
  brand could promise an axiom its function does not guarantee and nothing
  would notice.
* **mechanics** — ``requires`` is an identity function that leaves a readable
  flag behind, and survives ``conftest``'s ``install_import_hook``. If it ever
  stops being an identity it is on the hot path of a ``jit``-compiled package,
  which is the thing :func:`statadic.core.axioms.requires` exists not to be.
* **source** — "flags, never alias names" is unenforceable at runtime, because
  ``SEMIMETRIC`` *is* ``NONNEGATIVE | IDENTITY | SYMMETRY`` once evaluated. It is
  therefore checked by reading every call site with ``ast``, the same technique
  ``tests/migration/test_kernels_api.py`` uses and for the same reason: the
  property lives in the text, not in the object.

Whether each declared axiom actually *holds* is ``tests/testing/test_axioms.py``;
typed producer/consumer compatibility is checked in ``test_typing.py`` and in
the origin-aware matrix/statistic suites, rather than copied into a wiring map.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from statadic.association.decomposition import (
    disco_components,
)
from statadic.core.axioms import (
    DIVERGENCE,
    MARKER_AXIOMS,
    MARKER_PROPERTIES,
    METRIC,
    NEGATIVE_TYPE,
    PSEUDOMETRIC,
    QUASIMETRIC,
    SEMIMETRIC,
    STRONG_NEGATIVE_TYPE,
    Axioms,
    Law,
    LawProperties,
    Metric,
    NegativeType,
    NegativeTypeLaw,
    Nonnegative,
    Premetric,
    Semimetric,
    Separating,
    StrongNegativeTypeLaw,
    Symmetric,
    TriangleInequality,
    ZeroDiagonal,
    missing_axioms,
    missing_law_properties,
    required_axioms,
    requires,
)
from statadic.core.matrices import metrize_energy

from .._cases import CASES_BY_NAME, IDS
from .._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

_SRC = Path(__file__).parents[2] / "src" / "statadic"


# --- algebra ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
def test_axiom_aliases_are_the_declared_combinations() -> None:
    """The Deza & Deza aliases are exactly the documented flag sets."""
    assert DIVERGENCE == Axioms.NONNEGATIVE | (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION)
    assert SEMIMETRIC == DIVERGENCE | Axioms.SYMMETRY
    assert QUASIMETRIC == DIVERGENCE | Axioms.TRIANGLE
    assert PSEUDOMETRIC == (
        Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY | Axioms.TRIANGLE
    )
    assert METRIC == PSEUDOMETRIC | Axioms.SEPARATION
    assert METRIC == SEMIMETRIC | Axioms.TRIANGLE
    assert METRIC == QUASIMETRIC | Axioms.SYMMETRY
    # Every alias is a proper subset of METRIC except METRIC itself.
    for alias in (DIVERGENCE, SEMIMETRIC, QUASIMETRIC, PSEUDOMETRIC):
        assert alias & METRIC == alias
        assert alias != METRIC


@pytest.mark.unit
@pytest.mark.regression
def test_legacy_marker_chain_narrows_during_migration() -> None:
    """The temporary ``DMat`` marker chain remains compatible until t55.5."""
    assert issubclass(NegativeType, Metric)
    assert issubclass(Metric, Semimetric)
    assert issubclass(Semimetric, Premetric)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("capability", "attribute"),
    [
        (Law, "axioms"),
        (Law, "properties"),
        (Nonnegative, "nonnegative"),
        (ZeroDiagonal, "zero_diagonal"),
        (Separating, "separates_points"),
        (Symmetric, "symmetric"),
        (TriangleInequality, "triangle_inequality"),
        (NegativeTypeLaw, "negative_type"),
        (StrongNegativeTypeLaw, "strong_negative_type"),
    ],
)
def test_capability_metadata_is_read_only(
    capability: type[object], attribute: str
) -> None:
    """Capability fields are properties without setters, so covariance is sound."""
    descriptor = inspect.getattr_static(capability, attribute)
    assert isinstance(descriptor, property)
    assert descriptor.fset is None


# --- the MARKER_AXIOMS bridge ---------------------------------------------


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("name", IDS)
def test_marker_brand_is_implied_by_declared_axioms(name: str) -> None:
    """A ``DMat`` brand may never promise more than the function guarantees.

    ``MARKER_AXIOMS`` is the bridge from the compile-time chain to the runtime
    flags; without this assertion it is decorative. The chain is linear, so an
    off-chain axiom set — ``PSEUDOMETRIC``, or "negative type but not a metric" —
    forces the brand *down*, sometimes all the way to ``Premetric``.
    """
    case = CASES_BY_NAME[name]
    overclaimed = MARKER_AXIOMS[case.marker] & ~case.declared
    assert overclaimed == Axioms(0), (
        f"{case.name} brands {case.marker.__name__}, which promises "
        f"{overclaimed} that the function does not guarantee"
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("name", IDS)
def test_marker_property_is_backed_by_a_runtime_witness(name: str) -> None:
    """A compatibility marker may not overclaim an orthogonal property."""
    case = CASES_BY_NAME[name]
    overclaimed = MARKER_PROPERTIES[case.marker] & ~case.properties
    assert overclaimed == LawProperties(0), (
        f"{case.name} brands {case.marker.__name__}, which promises "
        f"{overclaimed} without a configured runtime witness"
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("name", IDS)
def test_marker_brand_is_the_strongest_available(name: str) -> None:
    """No case is branded weaker than the flags allow — the bridge is tight.

    ``NegativeType`` is excluded from the search on purpose: it strengthens
    ``Metric`` with a condition orthogonal to the distance flags, so
    ``MARKER_AXIOMS`` maps it to ``METRIC`` and the flags cannot tell the two
    apart. Where ``Metric`` is the flag-maximal brand, either is permitted and
    :func:`assert_negative_type` is what justifies the stronger one.
    """
    case = CASES_BY_NAME[name]
    chain = (Premetric, Semimetric, Metric)
    safe = [m for m in chain if MARKER_AXIOMS[m] & ~case.declared == Axioms(0)]
    narrowest = max(safe, key=lambda m: len(m.__mro__))
    permitted = {Metric, NegativeType} if narrowest is Metric else {narrowest}
    assert case.marker in permitted, (
        f"{case.name} brands {case.marker.__name__} but {narrowest.__name__} "
        f"is implied by {case.declared}"
    )


# --- @requires mechanics ---------------------------------------------------

#: Every decorated callable statadic ships, with the flags it must be declaring.
#: Extending ``@requires`` to a new site means extending this table — the point
#: of the decorator is that the precondition is reviewable, and a declaration
#: nobody pinned is not.
_DECLARED: list[tuple[str, object, Axioms]] = [
    # Not `SEMIMETRIC`: its parameter is `PremetricLaw`, which does not separate.
    # See the comment above the decorator in `core/matrices.py`.
    (
        "metrize_energy",
        metrize_energy,
        Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY,
    ),
    (
        "disco_components",
        disco_components,
        Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY,
    ),
]


@pytest.mark.unit
@pytest.mark.contract
def test_the_decorator_returns_the_function_unchanged() -> None:
    """``requires`` must be an identity — it sits on a jit-compiled hot path."""

    def embed(d: object) -> object:
        return d

    assert requires(Axioms.TRIANGLE)(embed) is embed


@pytest.mark.unit
@pytest.mark.contract
def test_an_undecorated_callable_declares_nothing() -> None:
    """Absence reads as ``None``, never as an empty flag."""
    assert required_axioms(len) is None


@pytest.mark.unit
@pytest.mark.contract
def test_a_non_axioms_attribute_is_not_mistaken_for_a_declaration() -> None:
    """A collision on the attribute name must not fabricate a precondition."""

    def impostor() -> None: ...

    ill_typed(impostor).__statadic_requires__ = "TRIANGLE"
    assert required_axioms(impostor) is None


# `ids` keys on the declaration's *name*, never on `str` of the whole tuple:
# stringifying a function embeds its `id()` (`<function f at 0x7f3e...>`), so
# the node ID changed every process. That is invisible serially but fatal under
# `-n auto` — xdist compares each worker's collected list and aborts the run
# with "Different tests were collected between gw0 and gw1" — and it also breaks
# `--lf`, `-k` and any cross-run history.
@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.needs_import_hook
@pytest.mark.parametrize(
    ("name", "function", "expected"),
    _DECLARED,
    ids=[name for name, _, _ in _DECLARED],
)
def test_shipped_declarations_survive_the_import_hook(
    name: str, function: object, expected: Axioms
) -> None:
    """The declaration must be readable through ``beartype``'s import hook.

    ``conftest`` installs the hook for the whole session, so simply importing
    these callables at module scope exercises it. Two earlier tests in this
    suite were invalidated by that hook (see ``conftest``), which is why this
    is pinned rather than assumed.

    Args:
        name: Callable name, for the failure message.
        function: The decorated callable.
        expected: Flags it must declare.

    """
    assert required_axioms(function) is expected, name


# --- the shortfall helper --------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("supplied", "required", "expected"),
    [
        (METRIC, SEMIMETRIC, Axioms(0)),
        (SEMIMETRIC, METRIC, Axioms.TRIANGLE),
        (PSEUDOMETRIC, METRIC, Axioms.SEPARATION),
        (
            Axioms.NONNEGATIVE | Axioms.SYMMETRY,
            METRIC,
            (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) | Axioms.TRIANGLE,
        ),
    ],
    ids=[
        "metric-satisfies-semimetric",
        "semimetric-short-triangle",
        "pseudometric-short-separation",
        "premetric-short-two",
    ],
)
def test_missing_axioms_names_the_shortfall(
    supplied: Axioms, required: Axioms, expected: Axioms
) -> None:
    """The empty flag is the pass condition; otherwise it names what is short.

    Args:
        supplied: What the producer guarantees.
        required: What the consumer declared.
        expected: The flags that must be reported missing.

    """
    assert missing_axioms(supplied, required) is expected


@pytest.mark.unit
@pytest.mark.property
def test_strong_negative_type_implies_negative_type_at_runtime() -> None:
    """The property algebra admits strong→ordinary and rejects the reverse."""
    assert not missing_law_properties(STRONG_NEGATIVE_TYPE, NEGATIVE_TYPE)
    assert (
        missing_law_properties(NEGATIVE_TYPE, STRONG_NEGATIVE_TYPE)
        is LawProperties.STRONG_NEGATIVE_TYPE
    )


# --- the source rule: flags, never alias names ----------------------------


def _requires_call_sites() -> Iterator[tuple[Path, ast.Call]]:
    """Yield every ``@requires(...)`` decorator call in the shipped package.

    Yields:
        tuple[Path, ast.Call]: ``(path, call)`` for each decorator whose callee
            is named ``requires``.

    """
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "requires"
                ):
                    yield path, decorator


def _is_flag_expression(node: ast.expr) -> bool:
    """Report whether ``node`` is built only from ``Axioms.*`` members and ``|``.

    Args:
        node: An argument expression from a ``@requires(...)`` decorator.

    Returns:
        ``True`` for ``Axioms.X`` and any ``|``-chain of them; ``False`` for a
        bare name, which is how an alias reaches this position.

    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_flag_expression(node.left) and _is_flag_expression(node.right)
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Axioms"
    )


@pytest.mark.integration
@pytest.mark.contract
def test_every_declaration_spells_flags_not_alias_names() -> None:
    """``requires=Axioms.TRIANGLE``, never ``requires=SEMIMETRIC``.

    ``statadic.core.axioms`` states the rule and gives the reason: a bare alias does
    not tell the reader which convention it follows, and "semimetric" is
    contested across the sources this package implements. Runtime cannot see the
    difference — the check therefore reads the text.
    """
    offenders = [
        f"{path.relative_to(_SRC)}:{call.lineno} -> {ast.unparse(argument)}"
        for path, call in _requires_call_sites()
        for argument in call.args
        if not _is_flag_expression(argument)
    ]
    assert not offenders, (
        f"requires() must be given Axioms.* flags, not an alias name: {offenders}"
    )


@pytest.mark.integration
@pytest.mark.contract
def test_the_shipped_declarations_are_all_pinned() -> None:
    """Every ``@requires`` site in the package appears in ``_DECLARED``.

    Without this, adding a declaration and forgetting to pin it leaves a
    precondition that no test ever reads — and, since t49 D2, one that no
    wiring edge is checked against either.
    """
    on_disk = {path.stem for path, _ in _requires_call_sites()}
    pinned = {"matrices", "decomposition"}
    assert on_disk == pinned, f"unpinned @requires modules: {on_disk ^ pinned}"


@pytest.mark.integration
@pytest.mark.contract
def test_every_dmat_taking_callable_declares_its_precondition() -> None:
    """A ``DMat[...]`` parameter without a ``@requires`` is an unstated precondition.

    t49 D3: the brand is the *static* half of the contract and ``@requires`` the
    runtime-readable half. A callable that takes ``DMat[Semimetric]`` and does
    not declare is relying on the reader to translate the marker back into
    flags, which is the translation ``statadic.core.axioms`` exists to remove.
    """
    undeclared: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        # One parse per file: `ast` node identity is per-parse, so comparing a
        # decorator from a second parse against this tree's `decorator_list`
        # never matches and every site would read as undeclared.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            takes_dmat = any(
                isinstance(arg.annotation, ast.Subscript)
                and isinstance(arg.annotation.value, ast.Name)
                and arg.annotation.value.id == "DMat"
                for arg in node.args.args
                if arg.annotation is not None
            )
            declares = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "requires"
                for decorator in node.decorator_list
            )
            if takes_dmat and not declares:
                undeclared.append(f"{path.relative_to(_SRC)}::{node.name}")
    assert not undeclared, (
        f"callables taking a DMat[...] without @requires: {undeclared}"
    )
