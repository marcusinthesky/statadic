"""Shared JAX array, shape, structural-law and static-argument vocabulary.

Shape wrappers remain runtime-resolvable for the optional annotation hook.
Experimental static dimensions erase at runtime and add no wheel dependency.
Matrix provenance belongs to core.matrices.DMat; constructors there distinguish
checked inputs, qualified producers and explicit unsafe assertions.
"""

from __future__ import annotations

import operator
from collections.abc import Callable  # noqa: TC003
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    ParamSpec,
    Protocol,
    SupportsIndex,
    TypeVar,
)

import jax
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    Key,
    Num,
    Real,
    Scalar,
    ScalarLike,
    UInt32,
)

from statadic.core.axioms import Law

if TYPE_CHECKING:
    from jax import Array as ShapedArray
    from shape_extensions import (
        Elements,
        static_jaxtyping,
    )
    from shape_extensions import (
        Int as Dimension,
    )
    from shape_extensions import (
        IntTuple as Shape,
    )
    from shape_extensions import (
        IntVar as DimensionVar,
    )
else:
    DimensionVar = int
    type Shape = tuple[int, ...]

    class Dimension:
        """A native static dimension that is an ordinary integer at runtime."""

        def __class_getitem__(cls, dimension: object) -> type[int]:
            """Erase the symbolic dimension when resolving runtime annotations."""
            del dimension
            return int

    class Elements:
        """Allow a symbolic batch shape to appear in evaluated annotations."""

        def __class_getitem__(cls, shape: object) -> tuple[()]:
            """Erase the static splice; jaxtyping checks the runtime batch axes."""
            del shape
            return ()

    class ShapedArray:
        """JAX's native static array type with runtime-safe shape subscripting.

        JAX itself does not accept shape subscripts. Wrap this type in a
        jaxtyping annotation to retain runtime shape and dtype checks.
        """

        def __class_getitem__(cls, shape: object) -> type[jax.Array]:
            """Erase static dimensions without modifying JAX's array class."""
            del shape
            return jax.Array

    def static_jaxtyping[T](dimensions: str, /) -> Callable[[T], T]:
        """Erase a static shape declaration; ``dimensions`` is read by Pyrefly."""
        del dimensions
        return lambda definition: definition


__all__ = [
    "STATIC",
    "Array",
    "ArrayLike",
    "Bool",
    "Dimension",
    "DimensionVar",
    "Elements",
    "Float",
    "Int",
    "LawT",
    "LawfulCallable",
    "Num",
    "PRNGKey",
    "Params",
    "Real",
    "ResultT",
    "Scalar",
    "ScalarLike",
    "Shape",
    "ShapedArray",
    "Static",
    "as_index",
    "static_jaxtyping",
]

#: Marker payload carried by :data:`Static`. Compared by identity, never parsed.
STATIC: Final = "statadic.static"


def as_index(value: object) -> int | None:
    """Return ``value`` as a Python ``int``, or ``None`` if it is not integral.

    Replaces the ``isinstance(value, (int, np.integer))`` test that eager doors
    across ``inference``, ``model`` and ``association`` each spelled for
    themselves before t65. Anything implementing ``__index__`` qualifies — a
    Python ``int``, a NumPy integer, a 0-d integer :class:`jax.Array` — which is
    both wider and more honest than naming one array library's scalar types.

    ``bool`` is rejected first, because it satisfies ``__index__`` and is never
    a meaningful count: ``n_folds=True`` should fail, not mean one.
    """
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        return None
    try:
        return operator.index(value)
    except TypeError:  # pragma: no cover — an object with a lying __index__
        return None


type Static[T] = Annotated[T, STATIC]
"""Mark an argument as a JIT-static (``jit(static_argnames=...)``) value.

``Static[int]`` *is* ``int`` to a type checker; the annotation records the
calling convention that ``generators/fields.py`` and the kernel modules already
rely on informally in prose.
"""

type PRNGKey = Key[Array, ""] | UInt32[Array, "2"]  # noqa: F722  # jaxtyping scalar shape
"""One typed JAX key or a legacy two-word ``uint32`` key.

Typed keys from :func:`jax.random.key` retain their PRNG implementation. The
legacy representation from :func:`jax.random.PRNGKey` uses JAX's configured
implementation. Neither representation is converted by numerical cores, and
only eager adapters turn integer seeds into keys. A batch of keys belongs at
the mapped axis of :func:`jax.vmap`, rather than at this scalar-key boundary.

Typed Threefry keys support the same JIT and per-key batching contracts as the
legacy Threefry stream. Other typed implementations follow JAX's semantics:
in particular, RBG's ``vmap`` behavior need not equal a per-key loop. Matching
seeds do not promise matching streams across implementations or future sampler
algorithms. :func:`statadic.core.random.cell_key` retains its legacy output and
existing label-folding schedule.
"""


#: Covariant structural-law parameter for typed strategies and future
#: multi-axis mathematical containers.
LawT = TypeVar("LawT", bound=Law, covariant=True)  # noqa: PLC0105

#: Exact callable parameter list propagated by :class:`LawfulCallable`.
Params = ParamSpec("Params")

#: Covariant callable result propagated by :class:`LawfulCallable`.
ResultT = TypeVar("ResultT", covariant=True)  # noqa: PLC0105


class LawfulCallable(Protocol[Params, ResultT, LawT]):
    """A callable retaining its full signature, result, and read-only law type."""

    @property
    def law(self) -> LawT:
        """Mathematical law declared for values produced by this callable."""
        ...

    def __call__(
        self,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ResultT:
        """Evaluate the callable without erasing its exact signature."""
        ...
