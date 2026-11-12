#!/usr/bin/env uv run --script
# /// script
# requires-python = "~=3.13.0"
# dependencies = []
# ///
"""Check workspace directions, cross-package interfaces and hidden STATADIC imports.

Tach owns intra-package dependencies. This checker reads the same interface
rules for cross-package imports, which Tach currently treats as external.
It resolves ordinary imported aliases, not arbitrary runtime reflection.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DYNAMIC_APIS = {
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "builtins.exec",
    "builtins.eval",
    "builtins.compile",
    "builtins.globals",
    "builtins.locals",
    "builtins.vars",
    "__import__",
    "builtins.__import__",
    "importlib.import_module",
    "importlib.reload",
    "importlib.util.find_spec",
    "importlib.util.module_from_spec",
    "importlib.util.spec_from_file_location",
    "pkgutil.resolve_name",
    "pydoc.locate",
    "runpy.run_module",
    "runpy.run_path",
}
FORBIDDEN_PACKAGES = {
    "numpy",
    "scipy",
    "ctypes",
    "_ctypes",
    "cffi",
    "_cffi_backend",
    "subprocess",
    "_posixsubprocess",
    "multiprocessing",
    "posix",
    "nt",
    "_imp",
    "runpy",
    "zipimport",
    "pickle",
    "marshal",
    "cloudpickle",
    "dill",
}
EXECUTION_PREFIXES = ("os.exec", "os.spawn", "os.fork", "os.posix_spawn")
FORBIDDEN_APIS = {
    "importlib.metadata.EntryPoint",
    "importlib.metadata.EntryPoints",
    "importlib.metadata.entry_points",
    "importlib.metadata.Distribution",
    "importlib.metadata.distribution",
    "importlib.metadata.distributions",
    "os.system",
    "os.popen",
    "os.startfile",
    "jax.pure_callback",
    "jax.experimental.io_callback",
    "jax.ffi.ffi_call",
    "jax.numpy.load",
    "jax.numpy.save",
    "jax.numpy.savez",
    "jax.numpy.savez_compressed",
    "jax.numpy.frombuffer",
    "jax.numpy.fromfile",
}
REFLECTIVE_FIELDS = {
    "__builtins__",
    "__globals__",
    "__code__",
    "__closure__",
    "__subclasses__",
    "__mro__",
    "__bases__",
    "__loader__",
}
SYS_CONTAINERS = {"sys.path", "sys.meta_path", "sys.path_hooks", "sys.modules"}
CUSTOM_DERIVATIVES = {
    "statadic.optimize.psd._positive_projection",
    "statadic.optimize.psd._positive_energy",
}
CUSTOM_APIS = {"jax.custom_jvp", "jax.custom_vjp", "jax.custom_gradient"}
ENVELOPE_RULES = {"statadic.estimation.gel._dual._gel_profile_objective_kernel"}
MUTATORS = {
    "__delitem__",
    "__setitem__",
    "append",
    "clear",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    column: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code}: {self.message}"


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    if isinstance(node, ast.Subscript):
        return dotted(node.value)
    return ""


class Policy:
    def __init__(self, root: Path):
        self.root = root
        self.python = root / "src/python"
        self.tach = tomllib.loads((self.python / "tach.toml").read_text())
        project = tomllib.loads((self.python / "pyproject.toml").read_text())
        self.allowed = project["tool"]["statadic_architecture"][
            "workspace_dependencies"
        ]
        self.sources: dict[str, Path] = {}
        self.module_paths: set[str] = set()
        for relative in self.tach["source_roots"]:
            source = self.python / relative
            if not source.is_dir():
                raise ValueError(f"Missing source root: {relative}")
            for package in sorted(source.iterdir()):
                if not (package / "__init__.py").is_file():
                    continue
                if package.name in self.sources:
                    raise ValueError(f"Duplicate workspace namespace: {package.name}")
                self.sources[package.name] = package
                for path in package.rglob("*.py"):
                    self.module_paths.add(self.module(path, source))
        if self.sources.keys() != self.allowed.keys():
            raise ValueError(
                "Workspace namespaces and dependency-policy entries differ"
            )
        for package, dependencies in self.allowed.items():
            if not isinstance(dependencies, list) or any(
                target not in self.sources or target == package
                for target in dependencies
            ):
                raise ValueError(f"Invalid workspace dependencies: {package}")
        self.boundaries = sorted(
            [
                name
                for item in self.tach["modules"]
                for name in item.get("paths", [item.get("path")])
                if name
            ],
            key=len,
            reverse=True,
        )
        self.interfaces = self.tach.get("interfaces", [])
        for rule in self.interfaces:
            for pattern in rule.get("from", []) + rule["expose"]:
                re.compile(pattern)

    @staticmethod
    def module(path: Path, source: Path) -> str:
        parts = list(path.relative_to(source).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def owner(self, name: str) -> str:
        return next(
            (b for b in self.boundaries if name == b or name.startswith(b + ".")), ""
        )

    def public(self, target: str, consumer: str) -> bool:
        owner = self.owner(target)
        if not owner:
            return False
        # Importing a public module is allowed; qualified accesses are checked
        # separately. Private modules require an explicit symbol-level interface.
        if target in self.module_paths:
            return all(not part.startswith("_") for part in target.split("."))
        rules = [
            r
            for r in self.interfaces
            if any(re.fullmatch(pattern, owner) for pattern in r.get("from", [".*"]))
        ]
        if not rules:
            return False
        consumer_owner = self.owner(consumer)
        visible = [
            r
            for r in rules
            if "visibility" not in r
            or any(
                consumer == v or consumer.startswith(v + ".") or consumer_owner == v
                for v in r["visibility"]
            )
        ]
        exclusive = [r for r in visible if r.get("exclusive", False)]
        relative = target[len(owner) + 1 :]
        return any(
            re.fullmatch(pattern, relative)
            for rule in exclusive or visible
            for pattern in rule["expose"]
        )


class Visitor(ast.NodeVisitor):
    def __init__(
        self,
        policy: Policy,
        path: Path,
        package: str,
        module: str,
        *,
        test: bool = False,
    ):
        self.policy, self.path, self.package, self.module = (
            policy,
            path,
            package,
            module,
        )
        self.test = test
        self.bindings: dict[str, str] = {}
        self.findings: list[Finding] = []
        self.try_depth = 0
        self.functions: list[str] = []

    def add(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(
            Finding(
                self.path.relative_to(self.policy.root).as_posix(),
                node.lineno,
                node.col_offset + 1,
                code,
                message,
            )
        )

    def resolved(self, node: ast.AST) -> str:
        name = dotted(node)
        head, separator, tail = name.partition(".")
        return self.bindings.get(head, head) + (separator + tail if separator else "")

    def dependency(self, node: ast.AST, target: str) -> None:
        package = target.partition(".")[0]
        if self.package == "statadic" and not self.test:
            if (
                package in FORBIDDEN_PACKAGES
                or target.startswith("jax._src")
                or target in FORBIDDEN_APIS
            ):
                self.add(node, "JAX-PURITY", f"forbidden runtime dependency: {target}")
            if target.startswith("importlib") and target not in {
                "importlib.metadata.version",
                "importlib.metadata.PackageNotFoundError",
            }:
                self.add(
                    node,
                    "DYNAMIC-IMPORT",
                    f"only version metadata reads are permitted: {target}",
                )
        if package not in self.policy.sources or package == self.package:
            return
        if package not in self.policy.allowed[self.package]:
            self.add(node, "WORKSPACE", f"{self.package} cannot depend on {package}")
        elif (
            package == "statadic"
            and not self.test
            and not self.policy.public(target, self.module)
        ):
            self.add(node, "INTERFACE", f"{self.module} cannot access {target}")

    def import_under_try(self, node: ast.AST) -> None:
        if self.package == "statadic" and not self.test and self.try_depth:
            self.add(node, "TRY-IMPORT", "imports may not appear beneath try")

    def visit_Import(self, node: ast.Import) -> None:
        self.import_under_try(node)
        for alias in node.names:
            self.dependency(node, alias.name)
            self.bindings[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_under_try(node)
        base = node.module or ""
        if node.level:
            parent = (
                self.module
                if self.path.name == "__init__.py"
                else self.module.rpartition(".")[0]
            )
            parts = parent.split(".")
            base = ".".join(
                parts[: len(parts) - node.level + 1] + ([base] if base else [])
            )
        for alias in node.names:
            target = base + "." + alias.name
            self.dependency(node, target)
            self.bindings[alias.asname or alias.name] = target

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.package == "statadic" and not self.test:
            resolved = self.resolved(node)
            if (
                resolved in FORBIDDEN_APIS
                or resolved.startswith("jax._src")
                or resolved.startswith(EXECUTION_PREFIXES)
                or resolved in SYS_CONTAINERS
                or node.attr in REFLECTIVE_FIELDS
                or node.attr == "to_numpy"
            ):
                self.add(
                    node,
                    "JAX-PURITY",
                    f"forbidden execution or host escape: {resolved}",
                )
        name = dotted(node)
        head = name.partition(".")[0]
        if head in self.bindings and self.bindings[head] in self.policy.module_paths:
            target = self.resolved(node)
            # An attribute of an imported object is not another module import.
            parent = target.rpartition(".")[0]
            if parent in self.policy.module_paths:
                self.dependency(node, target)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            self.package == "statadic"
            and not self.test
            and isinstance(node.ctx, ast.Load)
        ):
            name = self.resolved(node)
            if name in DYNAMIC_APIS or name in REFLECTIVE_FIELDS:
                self.add(
                    node,
                    "DYNAMIC-IMPORT",
                    f"dynamic execution capability is forbidden: {name}",
                )

    def visit_Try(self, node: ast.Try) -> None:
        self.try_depth += 1
        self.generic_visit(node)
        self.try_depth -= 1

    visit_TryStar = visit_Try

    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolved(node.func)
        if (
            name in DYNAMIC_APIS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.dependency(node, node.args[0].value)
        if self.package == "statadic" and not self.test:
            qualified = ".".join([self.module, *self.functions])
            if name == "jax.lax.stop_gradient" and qualified not in ENVELOPE_RULES:
                self.add(
                    node,
                    "AUTODIFF",
                    "stop_gradient needs a qualified mathematical contract",
                )
            if name in CUSTOM_APIS and qualified not in CUSTOM_DERIVATIVES:
                self.add(
                    node, "AUTODIFF", "custom derivatives require an audited exception"
                )
            if name in FORBIDDEN_APIS or name.startswith(EXECUTION_PREFIXES):
                self.add(
                    node, "JAX-PURITY", f"forbidden execution or host escape: {name}"
                )
            if (
                name
                in {
                    "getattr",
                    "setattr",
                    "delattr",
                    "builtins.getattr",
                    "builtins.setattr",
                    "builtins.delattr",
                }
                and len(node.args) > 1
            ):
                attribute = node.args[1]
                if (
                    isinstance(attribute, ast.Constant)
                    and attribute.value in REFLECTIVE_FIELDS
                    or self.resolved(node.args[0]) in {"os", "sys", "builtins", "jax"}
                ):
                    self.add(
                        node, "JAX-PURITY", "reflective execution access is forbidden"
                    )
            if name in DYNAMIC_APIS:
                self.add(node, "DYNAMIC-IMPORT", f"dynamic loader is forbidden: {name}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
                self.mutation(node.func.value)
        self.generic_visit(node)

    def mutation(self, node: ast.AST) -> None:
        if self.package == "statadic" and not self.test:
            target = self.resolved(node)
            if any(
                target == name or target.startswith(name + ".")
                for name in SYS_CONTAINERS
            ):
                self.add(
                    node,
                    "SYS-MUTATION",
                    f"import-system mutation is forbidden: {target}",
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.mutation(target)
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = self.resolved(node.value)
                if (
                    name.partition(".")[0] in self.policy.sources
                    or name in SYS_CONTAINERS
                    or name in DYNAMIC_APIS
                    or name in CUSTOM_APIS | {"jax.lax.stop_gradient"}
                ):
                    self.bindings[target.id] = name
                else:
                    self.bindings.pop(target.id, None)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.mutation(node.target)
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.bindings[node.target.id] = self.resolved(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.mutation(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self.mutation(target)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        outer = self.bindings.copy()
        self.functions.append(node.name)
        if self.package == "statadic" and not self.test:
            qualified = ".".join([self.module, *self.functions])
            for decorator in node.decorator_list:
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if (
                    self.resolved(target) in CUSTOM_APIS
                    and qualified not in CUSTOM_DERIVATIVES
                ):
                    self.add(
                        node,
                        "AUTODIFF",
                        "custom derivatives require an audited exception",
                    )
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            self.bindings.pop(arg.arg, None)
        for arg in (node.args.vararg, node.args.kwarg):
            if arg:
                self.bindings.pop(arg.arg, None)
        self.generic_visit(node)
        self.bindings = outer
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def inspect(
    policy: Policy, path: Path, package: str, module: str, *, test: bool = False
) -> list[Finding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        return [
            Finding(
                path.relative_to(policy.root).as_posix(),
                getattr(error, "lineno", 1) or 1,
                getattr(error, "offset", 1) or 1,
                "PARSE",
                str(error),
            )
        ]
    visitor = Visitor(policy, path, package, module, test=test)
    visitor.visit(tree)
    return sorted(set(visitor.findings))


def check(root: Path) -> list[Finding]:
    policy = Policy(root)
    findings = []
    for package, source in policy.sources.items():
        for path in sorted(source.rglob("*.py")):
            findings.extend(
                inspect(policy, path, package, policy.module(path, source.parent))
            )
    tests = policy.sources["statadic"].parent.parent / "tests"
    for path in sorted(tests.rglob("*.py")):
        findings.extend(inspect(policy, path, "statadic", "tests", test=True))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        findings = check(args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError, re.PatternError) as error:
        print(f"architecture: CONFIG: {error}", file=sys.stderr)
        return 1
    for finding in findings:
        print(finding)
    if not findings:
        print(
            "Architecture: workspace directions, interfaces and hidden imports passed"
        )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
