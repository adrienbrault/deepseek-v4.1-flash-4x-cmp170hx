"""Source lookup for the r18 overlay audit and offline image build."""
from dataclasses import dataclass
import ast
from functools import cache
from pathlib import Path


@cache
def parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def statements(body):
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.ExceptHandler)):
            for field in ("body", "orelse", "handlers", "finalbody"):
                yield from statements(getattr(node, field, ()))


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


@dataclass(frozen=True)
class Definition:
    path: Path
    node: ast.AST

    def source(self):
        # inspect.getsource includes indentation; the runtime digest dedents it.
        start = min([self.node.lineno, *(x.lineno for x in
                    getattr(self.node, "decorator_list", ()))])
        return "".join(self.path.read_text().splitlines(keepends=True)[start-1:self.node.end_lineno])


class Sources:
    def __init__(self, *roots, mounted=None):
        self.roots = tuple(roots)
        self.mounted = mounted or {}

    @cache
    def path(self, module):
        path = module.replace(".", "/")
        for name in (path + ".py", path + "/__init__.py"):
            if name in self.mounted:
                return self.mounted[name]
            for root in self.roots:
                candidate = root / name
                if candidate.is_file():
                    return candidate
        return None

    @cache
    def lookup(self, ref):
        module, _, member = ref.partition(":")
        result = self._lookup(module, tuple(member.split(".")) if member else (), frozenset())
        return result

    def _lookup(self, module, member, seen):
        key = (module, member)
        if key in seen:
            return None
        seen = seen | {key}
        path = self.path(module)
        if path is None:
            return None
        if not member:
            return Definition(path, parse(path))
        name, *tail = member
        submodule = self._lookup(module + "." + name, tuple(tail), seen)
        if submodule:
            return submodule
        package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        tree = parse(path)
        for node in statements(tree.body):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name:
                    current = Definition(path, node)
                    for part in tail:
                        current = self._member(current, part, module, seen)
                        if current is None:
                            break
                    return current
            elif isinstance(node, ast.ImportFrom):
                parent = package[:len(package) + 1 - node.level] if node.level else []
                origin = ".".join(parent + ([node.module] if node.module else []))
                for alias in node.names:
                    if alias.name == "*":
                        result = self._lookup(origin, member, seen)
                    elif (alias.asname or alias.name) == name:
                        names = (alias.name, *tail)
                        result = (self._lookup(origin, names, seen)
                                  or self._lookup(origin + "." + alias.name, tuple(tail), seen))
                        if (result is None and not tail
                                and origin.split(".")[0] not in {"vllm", "vllm_exl3"}):
                            # External imports cannot be resolved from source roots.
                            return Definition(path, node)
                    else:
                        continue
                    if result:
                        return result
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[0]) == name:
                        dest = alias.name if alias.asname else name
                        result = self._lookup(dest, tuple(tail), seen)
                        if result:
                            return result
                        if not tail:
                            return Definition(path, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(n, ast.Name) and n.id == name for n in targets):
                    if not tail:
                        return Definition(path, node)
        return None

    def _member(self, definition, name, module, seen):
        cls = definition.node
        if not isinstance(cls, ast.ClassDef):
            return None
        for node in statements(cls.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    return Definition(definition.path, node)
            elif isinstance(node, ast.AnnAssign) and dotted(node.target) == name:
                return Definition(definition.path, node)
            elif isinstance(node, ast.Assign) and any(dotted(t) == name for t in node.targets):
                return Definition(definition.path, node)
        # Methods and fields inherited from local and reexported classes.
        for base in cls.bases:
            ref = dotted(base)
            if not ref:
                continue
            result = self._lookup(module, tuple(ref.split(".") + [name]), seen)
            if result:
                return result
        # Runtime state declared on an instance, rather than a class binding.
        for node in ast.walk(cls):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                if dotted(node) == "self." + name:
                    return Definition(definition.path, node)
        return None


def signature(node):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = node.args
    return [*[x.arg for x in args.posonlyargs], *(["/"] if args.posonlyargs else []),
            *[x.arg for x in args.args],
            *([f"*{args.vararg.arg}"] if args.vararg else ["*"] if args.kwonlyargs else []),
            *[x.arg for x in args.kwonlyargs], *([f"**{args.kwarg.arg}"] if args.kwarg else [])]
