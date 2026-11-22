"""Order pytest node IDs by static references to edited Python modules."""

import argparse
import ast
from importlib.util import resolve_name
import json
from pathlib import Path


TIERS = ("changed_test", "direct_reference", "one_import_edge", "native")


def _imports(nodes, package):
    """Yield (bound name, target, imported path), excluding star imports."""
    for node in nodes:
        if isinstance(node, ast.Import):
            for name in node.names:
                bound = name.asname or name.name.partition(".")[0]
                yield bound, name.name if name.asname else bound, name.name
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            if node.level:
                if not package:
                    continue
                try:
                    prefix = resolve_name(prefix, package)
                except ImportError:
                    continue
            for name in node.names:
                if name.name != "*":
                    target = f"{prefix}.{name.name}"
                    yield name.asname or name.name, target, target


def _module_prefix(name, modules):
    while name and name not in modules:
        name = name.rpartition(".")[0]
    return name


def _resolve(target, modules, exports):
    module = _module_prefix(target, modules)
    if module and target != module and module in exports:
        member = target[len(module) + 1:].partition(".")[0]
        if member in exports[module]:
            return _module_prefix(exports[module][member], modules)
    return module


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _scope_nodes(nodes):
    for node in nodes:
        yield node
        if not isinstance(node, (*_FUNCTIONS, *_COMPREHENSIONS, ast.ClassDef)):
            yield from _scope_nodes(ast.iter_child_nodes(node))


def _dependencies(node, bindings, package):
    """Resolve dotted loads against the imports visible in each lexical scope."""
    if isinstance(node, _FUNCTIONS):
        args = node.args
        parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs,
                      *([args.vararg] if args.vararg else []), *([args.kwarg] if args.kwarg else [])]
        for expr in [*getattr(node, "decorator_list", []), *args.defaults, *args.kw_defaults,
                     *(arg.annotation for arg in parameters), getattr(node, "returns", None)]:
            if expr is not None:
                yield from _dependencies(expr, bindings, package)
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        local = bindings.copy()
        for arg in parameters:
            local.pop(arg.arg, None)
        nodes = list(_scope_nodes(body))
        for item in nodes:
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                local.pop(item.id, None)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.pop(item.name, None)
        for name, target, dep in _imports(nodes, package):
            local[name] = target
            yield dep
        for stmt in body:
            yield from _dependencies(stmt, local, package)
    elif isinstance(node, ast.ClassDef):
        for expr in [*node.decorator_list, *node.bases, *node.keywords]:
            yield from _dependencies(expr, bindings, package)
        local = bindings.copy()
        for stmt in node.body:
            # A nested method captures the enclosing function, not class locals.
            scope = bindings if isinstance(stmt, (*_FUNCTIONS, ast.ClassDef)) else local
            yield from _dependencies(stmt, scope, package)
            nodes = list(_scope_nodes([stmt]))
            for item in nodes:
                if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                    local.pop(item.id, None)
            for name, target, dep in _imports(nodes, package):
                local[name] = target
                yield dep
    elif isinstance(node, _COMPREHENSIONS):
        local = bindings.copy()
        for generator in node.generators:
            yield from _dependencies(generator.iter, local, package)
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    local.pop(target.id, None)
            for expr in generator.ifs:
                yield from _dependencies(expr, local, package)
        outputs = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        for expr in outputs:
            yield from _dependencies(expr, local, package)
    else:
        parts = []
        base = node
        while isinstance(base, ast.Attribute):
            parts.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load):
            if base.id in bindings:
                yield ".".join([bindings[base.id], *reversed(parts)])
        else:
            for child in ast.iter_child_nodes(node):
                yield from _dependencies(child, bindings, package)


def _test_functions(body, prefix=()):
    for node in body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            yield "::".join((*prefix, node.name)), node
        elif isinstance(node, ast.ClassDef):
            yield from _test_functions(node.body, (*prefix, node.name))


def _inside(root, path):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"path outside checkout: {path}")
    return path


def rank_tests(root, changed_paths, roster, *, source_dir="src"):
    """Return an exact permutation with native-order ties and coverage diagnostics.

    The source tree and tested diff are available before execution. A package
    import by itself gives no credit for edits to a submodule. Module edges
    include explicit imports in conditional blocks and function bodies.
    """
    root = Path(root).resolve()
    if isinstance(changed_paths, (str, bytes, dict)) or isinstance(roster, (str, bytes, dict, set, frozenset)):
        raise ValueError("expected changed paths and an ordered roster of test IDs")
    try:
        changed_paths = list(changed_paths)
        roster = list(roster)
    except TypeError as exc:
        raise ValueError("expected changed paths and an ordered roster of test IDs") from exc
    if not all(isinstance(path, str) and path.strip() for path in changed_paths):
        raise ValueError("expected nonempty changed filenames")
    changed = set(changed_paths)
    if not all(isinstance(node, str) and node.strip() for node in roster) or len(set(roster)) != len(roster):
        raise ValueError("expected distinct pytest node IDs")
    src = _inside(root, root / source_dir)
    if not src.is_dir():
        raise ValueError(f"missing Python source directory: {src}")
    modules = {}
    packages = set()
    for path in sorted(src.rglob("*.py")):
        parts = path.relative_to(src).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if is_package else parts)
        if name:
            modules[name] = path
            if is_package:
                packages.add(name)
    edited = {name for name, path in modules.items() if path.relative_to(root).as_posix() in changed}
    paths = {node.partition("::")[0] for node in roster}
    test_tiers = {}
    unresolved = []
    parsed_tests = 0
    if edited:
        trees = {name: ast.parse(_inside(root, path).read_bytes(), str(path)) for name, path in modules.items()}
        exports = {pkg: {a: b for a, b, _ in _imports(trees[pkg].body, pkg)} for pkg in packages}
        affected = set()
        for name, tree in trees.items():
            package = name if name in packages else name.rpartition(".")[0]
            edges = {_resolve(dep, modules, exports) for _, _, dep in _imports(ast.walk(tree), package)}
            if (edges - {name}) & edited:
                affected.add(name)
        for filename in sorted(paths - changed):
            path = _inside(root, root / filename)
            if not path.is_file() or path.suffix != ".py":
                continue
            tree = ast.parse(path.read_bytes(), str(path))
            parsed_tests += 1
            package = ".".join(Path(filename).parent.parts)
            imports = {a: b for a, b, _ in _imports(tree.body, package)}
            for fn, func in _test_functions(tree.body):
                used = {_resolve(dep, modules, exports)
                        for dep in _dependencies(func, imports, package)}
                # A package points to its entire public surface; give it credit
                # only when its own initializer is among the changed files.
                precise = used - packages
                tier = 1 if used & edited else 2 if precise & affected else 3
                test_tiers[f"{filename}::{fn}"] = tier
    groups = [[] for _ in TIERS]
    for node in roster:
        filename = node.partition("::")[0]
        test = node.partition("[")[0]
        tier = 0 if filename in changed else test_tiers.get(test, 3)
        if edited and tier != 0 and test not in test_tiers:
            unresolved.append(node)
        groups[tier].append(node)
    return {"order": [node for group in groups for node in group],
            "tiers": {tier: len(group) for tier, group in zip(TIERS, groups, strict=True)},
            "unresolved_test_functions": unresolved,
            "edited_modules": sorted(edited), "source_files_parsed": len(modules) if edited else 0,
            "test_files_parsed": parsed_tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="tested Python checkout with src/ layout")
    parser.add_argument("request", type=Path, help="JSON containing changed_paths and native roster")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    print(json.dumps(rank_tests(args.root, request["changed_paths"], request["roster"])))


if __name__ == "__main__":
    main()
