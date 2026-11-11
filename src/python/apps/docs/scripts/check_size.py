"""Enforce the frozen first-use word budget, including authored transclusions."""

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


def main() -> None:
    budget = json.loads((APP / "word-budget.json").read_text())
    if budget["baseline_words"] != 6166 or budget["maximum_words"] != 1541:
        raise ValueError("The approved baseline and 75% cap must remain frozen")
    entrypoints = set(budget["entrypoints"])
    for directory in (DOCS / "start", DOCS / "compose"):
        entrypoints.update(
            str(path.relative_to(ROOT))
            for path in directory.rglob("*")
            if path.suffix in {".md", ".rst"}
        )
    counts = {name: words(ROOT / name, ROOT) for name in sorted(entrypoints)}
    total = sum(counts.values())
    all_docs = sum(
        len(p.read_text().split())
        for p in DOCS.rglob("*")
        if p.suffix in {".md", ".rst"}
    )
    print(
        json.dumps(
            {
                "first_use_words": total,
                "maximum": budget["maximum_words"],
                "reduction_percent": round(100 * (1 - total / 6166), 2),
                "all_docs_source_words": all_docs,
                "files": counts,
            },
            indent=2,
        )
    )
    if total > budget["maximum_words"]:
        raise SystemExit(
            f"First-use documentation exceeds its budget by {total - 1541} words"
        )


if __name__ == "__main__":
    main()
