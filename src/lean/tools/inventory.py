"""Snapshot public source exports without importing STATADIC or executing its code.

Run from any directory: ``uv run --project src/python/packages/statadic python
src/lean/tools/inventory.py --write``. Omit ``--write`` for a read-only check.
The supported boundary is every non-private module and its literal __all__.
Public-looking definitions in those modules must be explicitly exported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/python/packages/statadic"
SOURCE = PACKAGE / "src/statadic"
OUTPUT = ROOT / "src/lean/registry/public-api.json"
METHODS = ROOT / "src/lean/registry/methods.json"


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["statadic", *parts])


def literal_all(node: ast.AST, path: Path) -> list[str]:
    try:
        values = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: nonliteral __all__ needs review") from exc
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(v, str) for v in values
    ):
        raise ValueError(f"{path}: unsupported __all__")
    return list(values)


def inventory(source: Path = SOURCE, package: Path = PACKAGE) -> dict:
    modules = []
    manifest = {}
    all_bindings = {}
    all_classes = {}
    errors = []
    for path in sorted(source.rglob("*.py")):
        rel = path.relative_to(source)
        module = module_name_for(path, source)
        manifest[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = ast.parse(path.read_bytes(), filename=str(path))
        bindings = {}
        names = None
        declarations = 0
        augmented = False
        definitions = set()
        private = any(
            part.startswith("_") and part != "__init__.py" for part in rel.parts
        )
        runtime_items = [
            child
            for item in tree.body
            for child in (
                item.orelse
                if isinstance(item, ast.If)
                and isinstance(item.test, ast.Name)
                and item.test.id == "TYPE_CHECKING"
                else [item]
            )
        ]
        for item in runtime_items:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.add(item.name)
                bindings[item.name] = (
                    "class" if isinstance(item, ast.ClassDef) else "function",
                    f"{module}.{item.name}",
                )
                if isinstance(item, ast.ClassDef):
                    members = {}
                    for child in item.body:
                        if isinstance(
                            child, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and not child.name.startswith("_"):
                            decorators = child.decorator_list
                            is_property = any(
                                (
                                    isinstance(decorator, ast.Name)
                                    and decorator.id in {"property", "cached_property"}
                                )
                                or (
                                    isinstance(decorator, ast.Attribute)
                                    and decorator.attr
                                    in {
                                        "property",
                                        "cached_property",
                                        "setter",
                                        "deleter",
                                    }
                                )
                                for decorator in decorators
                            )
                            kind = "property" if is_property else "method"
                            members[child.name] = {
                                "name": child.name,
                                "kind": kind,
                                "target": f"{module}.{item.name}.{child.name}",
                            }
                    all_classes[f"{module}.{item.name}"] = list(members.values())
            elif isinstance(item, ast.TypeAlias) and isinstance(item.name, ast.Name):
                definitions.add(item.name.id)
                bindings[item.name.id] = ("type-alias", f"{module}.{item.name.id}")
            elif isinstance(item, ast.ImportFrom):
                base = (
                    module.split(".")
                    if path.name == "__init__.py"
                    else module.split(".")[:-1]
                )
                if item.level:
                    base = base[: len(base) - item.level + 1]
                    origin = ".".join([*base, *([item.module] if item.module else [])])
                else:
                    origin = item.module or ""
                for alias in item.names:
                    bindings[alias.asname or alias.name] = (
                        "import",
                        f"{origin}.{alias.name}",
                    )
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        "module",
                        alias.name,
                    )
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                targets = (
                    item.targets if isinstance(item, ast.Assign) else [item.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        if target.id == "__all__":
                            declarations += 1
                            try:
                                names = literal_all(item.value, path)
                            except ValueError:
                                if private:
                                    raise
                                names = None
                        else:
                            if (
                                isinstance(item.value, ast.Name)
                                and item.value.id in bindings
                            ):
                                bindings[target.id] = bindings[item.value.id]
                            else:
                                bindings.setdefault(
                                    target.id, ("value", f"{module}.{target.id}")
                                )
            elif (
                isinstance(item, ast.AugAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == "__all__"
            ):
                if not private:
                    augmented = True
                    names = None
                elif names is None or not isinstance(item.op, ast.Add):
                    raise ValueError(f"{path}: unsupported augmented __all__")
                else:
                    names.extend(literal_all(item.value, path))
        all_bindings[module] = bindings
        # Private source files are represented when a public module explicitly imports them.
        if private:
            continue
        if names is None or declarations != 1 or augmented:
            errors.append(f"{path}: public module needs a literal __all__")
            names = []
        for name in sorted(definitions - set(names)):
            if not name.startswith("_"):
                errors.append(
                    f"{path}: {name} is not in __all__; list it or add a leading underscore"
                )
        exports = []
        for name in sorted(set(names)):
            if name not in bindings:
                errors.append(f"{path}: __all__ exports unbound name {name}")
                continue
            kind, target = bindings[name]
            exports.append({"name": name, "kind": kind, "target": target})
        modules.append({"name": module, "source": str(rel), "exports": exports})
    if errors:
        raise ValueError("\n".join(errors))
    methods = json.loads(METHODS.read_text())["methods"]
    for module in modules:
        candidates = [
            (len(prefix), method["id"])
            for method in methods
            for prefix in method["module_prefixes"]
            if module["name"] == prefix or module["name"].startswith(prefix + ".")
        ]
        if not candidates:
            raise ValueError(f"no method family owns {module['name']}")
        max_length = max(length for length, _ in candidates)
        owners = {owner for length, owner in candidates if length == max_length}
        if len(owners) != 1:
            raise ValueError(
                f"ambiguous method ownership for {module['name']}: {owners}"
            )
        module["method_id"] = owners.pop()
        for export in module["exports"]:
            export["method_id"] = module["method_id"]
            seen = set()
            while export["kind"] == "import" and export["target"] not in seen:
                target = export["target"]
                seen.add(target)
                owner, _, name = target.rpartition(".")
                resolved = all_bindings.get(owner, {}).get(name)
                if resolved is None:
                    if target in all_bindings:
                        export["kind"] = "module"
                    elif target.startswith("statadic."):
                        raise ValueError(
                            f"unresolved STATADIC export {module['name']}.{export['name']}: {target}"
                        )
                    break
                export["kind"], export["target"] = resolved
            if export["kind"] == "class":
                export["members"] = [
                    dict(member) for member in all_classes.get(export["target"], [])
                ]
                for member in export["members"]:
                    member["method_id"] = module["method_id"]
    version = tomllib.loads((package / "pyproject.toml").read_text())["project"][
        "version"
    ]
    return {
        "schema_version": 1,
        "package_version": version,
        "source_manifest": manifest,
        "modules": modules,
    }


def module_name_for(path: Path, source: Path) -> str:
    parts = list(path.relative_to(source).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["statadic", *parts])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="explicitly regenerate the snapshot"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="exercise callable and alias detection"
    )
    args = parser.parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            source.mkdir(parents=True)
            (package / "pyproject.toml").write_text('[project]\nversion = "0.4.1"\n')
            (source / "__init__.py").write_text("__all__ = []\n")
            (source / "_private.py").write_text("def hidden(): pass\n")
            public = source / "science.py"
            public.write_text(
                "from ._private import hidden as exposed\n__all__ = ['exposed']\n"
            )
            before = inventory(source, package)
            exported = before["modules"][1]["exports"][0]
            assert (exported["name"], exported["kind"], exported["target"]) == (
                "exposed",
                "function",
                "statadic._private.hidden",
            )
            public.write_text(
                "from ._private import hidden as exposed\nclass Estimate: pass\n__all__ = ['exposed', 'Estimate']\n"
            )
            after = inventory(source, package)
            assert before != after
            assert {item["name"] for item in after["modules"][1]["exports"]} == {
                "exposed",
                "Estimate",
            }
        print("public callable and alias detection passed")
        return 0
    actual = json.dumps(inventory(), indent=2, ensure_ascii=False) + "\n"
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(actual)
        print(f"wrote {OUTPUT}")
        return 0
    if not OUTPUT.exists() or OUTPUT.read_text() != actual:
        print(
            "public API inventory is stale; run inventory.py --write", file=sys.stderr
        )
        return 1
    print("public API inventory is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
