"""Graph-qualified schemas preserve identities and reject generated collisions."""

import pytest
from statadic.estimation.spatial._design import graph_parameter_names

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.mark.parametrize(
    ("names", "base", "columns", "spatial"),
    [
        (("rho[g]", "rho"), ("rho[g]",), (), ("rho",)),
        (("x", "W[g]:x", "W:x", "rho"), ("x", "W[g]:x"), (0,), ("rho",)),
    ],
)
def test_graph_qualified_names_reject_generated_collisions(
    names: tuple[str, ...],
    base: tuple[str, ...],
    columns: tuple[int, ...],
    spatial: tuple[str, ...],
) -> None:
    """A valid raw schema cannot become ambiguous after graph qualification."""
    with pytest.raises(ValueError, match="unique"):
        graph_parameter_names(names, base, columns, "g", spatial)
    with pytest.raises(ValueError, match="unique"):
        graph_parameter_names(None, base, columns, "g", spatial)


def test_graph_qualification_changes_only_generated_terms() -> None:
    """Caller labels stay intact while actual WX, rho and lambda terms are named."""
    names = ("intercept", "x", "W:x", "rho", "lambda")
    assert graph_parameter_names(names, names[:2], (1,), "g", ("rho", "lambda")) == (
        "intercept",
        "x",
        "W[g]:x",
        "rho[g]",
        "lambda[g]",
    )
    assert (
        graph_parameter_names(names, names[:2], (1,), None, ("rho", "lambda")) == names
    )
