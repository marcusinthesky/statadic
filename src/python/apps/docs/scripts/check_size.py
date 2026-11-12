"""Enforce the root README word cap, including authored transclusions."""

import json
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[3]
DOCS = ROOT / "src/python/packages/statadic/docs"


def words(path: Path, root: Path, ancestors: tuple[Path, ...] = ()) -> int:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or path in ancestors:
        raise ValueError(f"External or cyclic documentation include: {path}")
    text = path.read_text()
    count = len(text.split())
    pattern = (
        r"(?:`{3,}|:{3,})\{(literalinclude|include)\}\s+([^\n]+)\n((?::[^\n]*\n|\n)*)"
    )
    for match in re.finditer(pattern, text):
        source = (path.parent / match[2].strip()).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError(f"External documentation include: {source}")
        if match[1] == "include":
            count += words(source, root, (*ancestors, path))
            continue
        included = source.read_text()
        options = dict(re.findall(r"^:([\w-]+):\s*(.*)$", match[3], re.M))
        for option, after in (("start-after", True), ("end-before", False)):
            if option in options:
                marker = options[option].strip("\"'")
                before, found, remainder = included.partition(marker)
                if not found:
                    raise ValueError(f"Missing {option} marker in {source}: {marker}")
                included = remainder if after else before
        if "lines" in options:
            raise ValueError(
                "Add line-range accounting before using :lines: in onboarding"
            )
        count += len(included.split())
    return count


def budget_summary(readme_words: int, budget: dict) -> dict:
    """Check the root README against its word cap."""
    maximum = budget["readme_budget"]["maximum_words"]
    if readme_words > maximum:
        raise ValueError(f"README exceeds its budget by {readme_words - maximum} words")
    return {"readme_words": readme_words, "readme_maximum": maximum}


def main() -> None:
    budget = json.loads((APP / "word-budget.json").read_text())
    print(json.dumps(budget_summary(words(ROOT / "README.md", ROOT), budget), indent=2))


if __name__ == "__main__":
    main()
