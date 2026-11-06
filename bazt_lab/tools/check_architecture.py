"""Import gate for bazt_lab: the rules of docs/ARCHITECTURE_EXPERIMENTS.md, checked with the ast module.

    python -m bazt_lab.tools.check_architecture [--root DIR]        exit 1, one line per violation

Rule ids (printed in brackets):
  shape             bazt_lab/ holds app.py, core/, features/, tools/; core has public.py and private/; a feature has
                    __init__.py, public.py, private/, adapters/, tests/ (each a package; tests/ with a test_*.py) and a
                    non-empty CONTRACT.md; code lives in public.py, private/, adapters/ and tests/ only (a thin
                    __main__.py that imports sys and its own public and calls it is the one exception)
  core-isolated     core imports no feature
  core-public       outside core, core is imported only as bazt_lab.core.public
  feature-public    a feature imports another feature only through its public module
  app-root          nothing imports bazt_lab.app
  god-feature       only app.py imports every feature (a module importing all the other features, 3 or more)
  service-adapter   wandb, datasets, huggingface_hub, openai, sentence_transformers, torch_geometric, swarmlab, bazt and
                    llm_client are imported only in features/<f>/adapters/
  secrets           credential-shaped literals, secret environment variables and python-dotenv outside adapters,
                    load_dotenv(override=...) anywhere, secrets read from settings or present in experiments/profiles.toml
  settings-default  settings are read with core.get; `settings.get(...)` or `.get("dotted.key", ...)` hides a missing
                    key behind a default
  dynamic-import    import_module / __import__ with a name that is not a literal, outside adapters. The one exception
                    is the composition root: app.py may load f"bazt_lab.features.{name}.public"
  syntax            a file that does not parse (python files and experiments/profiles.toml)

Import forms are normalised first: relative imports are resolved against the file's package, and
`from bazt_lab.features.b import public` is read as the module bazt_lab.features.b.public, so a name that is not a
submodule (`from bazt_lab.features.b import thing`) is refused. There is no allowlist and no pragma: a violation is
fixed, not silenced.

`check_sources` takes the tree as a dict {posix path: text} and touches no disk; `check` loads a directory first.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = "bazt_lab"
APP = f"{PKG}.app"
SERVICES = frozenset({"wandb", "datasets", "huggingface_hub", "openai", "sentence_transformers", "torch_geometric",
                      "swarmlab", "bazt", "llm_client"})
SECRET_ENV = frozenset({"HF_TOKEN", "WANDB_API_KEY", "OPENROUTER_API_KEY"})
TOP_LEVEL = frozenset({"__init__.py", "app.py", "core", "features", "tools"})
SKELETON = ("__init__.py", "app.py", "core/__init__.py", "core/public.py", "core/private/__init__.py")
FEATURE_DIRS = ("private", "adapters", "tests")
SETTINGS_FILE = "experiments/profiles.toml"
TEXT_SUFFIXES = (".py", ".md", ".toml")
WIRING = re.compile(rf"{PKG}\.features\.\{{\}}\.public")      # what app.py may load dynamically
TOKEN_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{30,}|\bsk-(?:or-)?[A-Za-z0-9_-]{24,}")
SECRET_NAME = re.compile(r"(?i)(?:^|_)(?:token|api_?key|secret|password)$")
CREDENTIAL_SHAPE = re.compile(r"[A-Za-z0-9_.=-]{12,}")
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]+")
DOTTED_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


@dataclass(frozen=True)
class Where:
    """Where a file sits: the dotted package relative imports start from, its area and feature part."""
    module: str
    package: str
    area: str            # core | features | app | tools | other
    feature: str | None
    part: str            # public | private | adapters | tests | root


@dataclass(frozen=True)
class Dynamic:
    """An import whose module name is not a literal; `template` has {} where the name is computed."""
    template: str


def locate(rel: str) -> Where:
    parts = PurePosixPath(rel).with_suffix("").parts
    is_package = parts[-1] == "__init__"
    module = parts[:-1] if is_package else parts
    package = module if is_package else module[:-1]
    second = parts[1] if len(parts) > 1 else ""
    area = second if second in ("core", "features", "tools", "app") else "other"
    feature = parts[2] if area == "features" and len(parts) > 3 else None
    third = parts[3] if feature else ""
    part = third if third in (*FEATURE_DIRS, "public") and (third == "public" or len(parts) > 4) else "root"
    return Where(".".join(module), ".".join(package), area, feature, part)


# ------------------------------------------------------------------------------------------------- imports
def absolute(node: ast.ImportFrom, package: str) -> str | None:
    if node.level == 0:
        return node.module
    base = package.split(".")
    if node.level - 1 >= len(base):
        return None
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def is_namespace(module: str) -> bool:
    """A package whose imported names are read as submodules."""
    return module in (PKG, f"{PKG}.core", f"{PKG}.features") or (
        module.startswith(f"{PKG}.features.") and module.count(".") == 2)


def literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def template(node: ast.AST | None) -> str:
    """The text of a string expression, with {} for every computed part (a plain name is all computed)."""
    if isinstance(node, ast.JoinedStr):
        return "".join(literal(part) or "{}" for part in node.values)
    return literal(node) or "{}"


def call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def loader_aliases(tree: ast.AST) -> frozenset[str]:
    """Names this module binds to importlib.import_module (`from importlib import import_module as load`)."""
    return frozenset(alias.asname or alias.name for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module == "importlib"
                     for alias in node.names if alias.name == "import_module")


def name_argument(node: ast.Call) -> ast.AST | None:
    return node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "name"), None)


def targets(tree: ast.AST, package: str) -> Iterator[tuple[int, str | Dynamic]]:
    """(line, dotted module) of every import; a Dynamic marks a dynamic import whose name is not a literal."""
    aliases = loader_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = absolute(node, package)
            if module is None:
                continue
            if is_namespace(module):
                for alias in node.names:
                    yield node.lineno, f"{module}.{alias.name}"
            else:
                yield node.lineno, module
        elif isinstance(node, ast.Call) and (call_name(node) in ("import_module", "__import__")
                                             or call_name(node) in aliases):
            name = literal(name_argument(node))
            if name is not None and not name.startswith("."):
                yield node.lineno, name
            else:
                yield node.lineno, Dynamic(template(name_argument(node)))


def judge(where: Where, target: str) -> Iterator[tuple[str, str]]:
    segments = target.split(".")
    root = segments[0]
    if root in SERVICES and where.part != "adapters":
        yield "service-adapter", f"{root} is imported only in features/<feature>/adapters/"
    if root == "dotenv" and where.part != "adapters":
        yield "secrets", "python-dotenv is used only in features/<feature>/adapters/"
    if root != PKG or len(segments) < 2:
        return
    area, name = segments[1], (segments[2] if len(segments) > 2 else None)
    if area == "app" and where.module != APP:
        yield "app-root", f"{target}: nothing imports {APP}"
    elif area == "core" and where.area != "core" and segments[2:3] != ["public"]:
        yield "core-public", f"{target}: import core as `from {PKG}.core import public`"
    elif area == "features" and where.area == "core":
        yield "core-isolated", f"{target}: core imports no feature"
    elif area == "features" and name and name != where.feature and (name == "*" or segments[3:4] != ["public"]):
        yield "feature-public", f"{target}: import a feature as `from {PKG}.features.{name} import public`"


# ------------------------------------------------------------------------------------------ secrets, settings
def name_of(node: ast.AST | None) -> str | None:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else literal(node)


def looks_like_credential(text: str) -> bool:
    return bool(CREDENTIAL_SHAPE.fullmatch(text)) and not ENV_NAME.fullmatch(text)


def is_environ(node: ast.AST) -> bool:
    return ast.unparse(node).endswith("environ")


def env_key(node: ast.AST) -> str | None:
    """The literal variable name when node reads os.environ[...] / os.environ.get(...) / os.getenv(...)."""
    if isinstance(node, ast.Subscript) and is_environ(node.value):
        return literal(node.slice)
    if isinstance(node, ast.Call) and node.args and (
            (isinstance(node.func, ast.Attribute) and is_environ(node.func.value)) or call_name(node) == "getenv"):
        return literal(node.args[0])
    return None


def assignments(node: ast.AST) -> list[tuple[str | None, ast.AST | None]]:
    """(name, value) pairs that bind a value to a name: assignments, keyword arguments, dict literals."""
    if isinstance(node, ast.Assign):
        return [(name_of(t), node.value) for t in node.targets]
    if isinstance(node, ast.AnnAssign):
        return [(name_of(node.target), node.value)]
    if isinstance(node, ast.keyword):
        return [(node.arg, node.value)]
    if isinstance(node, ast.Dict):
        return [(literal(k), v) for k, v in zip(node.keys, node.values, strict=True)]
    return []


def overrides_environment(node: ast.Call) -> bool:
    """load_dotenv(..., override=X) with X anything but the literal False (override is the fourth parameter)."""
    value = next((k.value for k in node.keywords if k.arg == "override"), node.args[3] if len(node.args) > 3 else None)
    return value is not None and not (isinstance(value, ast.Constant) and value.value is False)


def reads_settings_with_get(node: ast.Call) -> bool:
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "get"):
        return False
    key = literal(node.args[0]) if node.args else None
    return (name_of(func.value) == "settings" or ast.unparse(func.value).endswith(".settings")
            or bool(key and DOTTED_KEY.fullmatch(key)))


def code_findings(tree: ast.AST, where: Where) -> Iterator[tuple[int, str, str]]:
    """(line, rule, message) for secrets in code and for settings read without core.get."""
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 1)
        text = literal(node)
        if text and TOKEN_VALUE.search(text):
            yield line, "secrets", "credential-shaped literal; secrets come from the environment"
        for name, value in assignments(node):
            text = literal(value)
            if name and text and SECRET_NAME.search(name) and looks_like_credential(text):
                yield line, "secrets", f"{name} holds a literal; secrets come from the environment"
        if where.part != "adapters" and env_key(node) in SECRET_ENV:
            yield line, "secrets", f"{env_key(node)} is read only in features/<feature>/adapters/"
        if not isinstance(node, ast.Call):
            continue
        if call_name(node) == "load_dotenv" and overrides_environment(node):
            yield line, "secrets", "load_dotenv must keep override=False: .env never replaces the environment"
        if reads_settings_with_get(node):
            yield line, "settings-default", "read settings with core.get(settings, key), not .get()"
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "get") or (
                isinstance(func, ast.Attribute) and func.attr == "get" and name_of(func.value) == "core"):
            key = literal(node.args[1]) if len(node.args) > 1 else None
            if key and SECRET_NAME.search(key.rsplit(".", 1)[-1]):
                yield line, "secrets", f"{key}: secrets are never read from settings"


def toml_secrets(tree: dict, trail: tuple = ()) -> Iterator[str]:
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from toml_secrets(value, (*trail, key))
        elif SECRET_NAME.search(key) or (isinstance(value, str) and TOKEN_VALUE.search(value)):
            yield ".".join((*trail, key))


def settings_violations(text: str) -> Iterator[Violation]:
    try:
        tree = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        yield Violation(SETTINGS_FILE, 1, "syntax", str(error))
        return
    for key in toml_secrets(tree):
        yield Violation(SETTINGS_FILE, 1, "secrets", f"{key}: settings hold no secret")


# ----------------------------------------------------------------------------------------------- the shape
def features_of(files: Mapping[str, str]) -> dict[str, set[str]]:
    """{feature: paths below it, relative to the feature} for every directory under bazt_lab/features/."""
    found: dict[str, set[str]] = {}
    for path in files:
        parts = path.split("/")
        if parts[:2] == [PKG, "features"] and len(parts) > 3 and not parts[2].startswith(("_", ".")):
            found.setdefault(parts[2], set()).add("/".join(parts[3:]))
    return found


def shape(files: Mapping[str, str]) -> Iterator[Violation]:
    for needed in SKELETON:
        if f"{PKG}/{needed}" not in files:
            yield Violation(f"{PKG}/{needed}", 1, "shape", "missing")
    inside = sorted(path for path in files if path.startswith(f"{PKG}/"))
    for name in sorted({path.split("/")[1] for path in inside} - TOP_LEVEL):
        yield Violation(f"{PKG}/{name}", 1, "shape", f"{PKG}/ holds only {', '.join(sorted(TOP_LEVEL))}")
    for path in inside:
        parts = path.split("/")
        if parts[1] == "features" and len(parts) == 3 and parts[2] != "__init__.py":
            yield Violation(path, 1, "shape", "features/ holds feature packages only")
    for name, rels in sorted(features_of(files).items()):
        yield from feature_shape(f"{PKG}/features/{name}", rels, files)


def feature_shape(base: str, rels: set[str], files: Mapping[str, str]) -> Iterator[Violation]:
    def missing(what: str) -> Violation:
        return Violation(base, 1, "shape", f"missing {what}")

    for name in ("__init__.py", "public.py"):
        if name not in rels:
            yield missing(name)
    if not files.get(f"{base}/CONTRACT.md", "").strip():
        yield missing("CONTRACT.md (non-empty)")
    for name in FEATURE_DIRS:
        if f"{name}/__init__.py" not in rels:
            yield missing(f"{name}/__init__.py")
    tests = [r for r in rels if r.startswith("tests/")]
    if tests and not any(PurePosixPath(r).match("test_*.py") for r in tests):
        yield missing("tests/test_*.py")
    feature = base.rsplit("/", 1)[1]
    for stray in sorted(r for r in rels if "/" not in r and r.endswith(".py") and r not in ("__init__.py", "public.py")):
        if stray == "__main__.py" and is_thin_main(files.get(f"{base}/{stray}", ""), feature):
            continue
        reason = ("__main__.py only imports sys and the feature's own public module and calls it"
                  if stray == "__main__.py" else "code lives in public.py, private/ or adapters/")
        yield Violation(f"{base}/{stray}", 1, "shape", reason)


PLAIN_CALL = (ast.Call, ast.Attribute, ast.Name, ast.Constant, ast.Load, ast.keyword)


def is_thin_main(source: str, feature: str) -> bool:
    """`python -m bazt_lab.features.<feature>`: only `import sys`, imports of the feature's own public module, calls of
    plain names and attributes, optionally under `if __name__ == "__main__":`."""
    own = f"{PKG}.features.{feature}"

    def thin(statements: list, top: bool) -> bool:
        for node in statements:
            if isinstance(node, ast.Import):
                ok = top and all(alias.name == "sys" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                ok = top and node.level == 0 and (node.module == f"{own}.public" or (
                    node.module == own and all(alias.name == "public" for alias in node.names)))
            elif isinstance(node, ast.Expr):
                ok = isinstance(node.value, ast.Call) and all(isinstance(n, PLAIN_CALL) for n in ast.walk(node.value))
            elif isinstance(node, ast.If):
                guard = ast.unparse(node.test).replace("'", '"')
                ok = top and guard == '__name__ == "__main__"' and not node.orelse and thin(node.body, False)
            else:
                ok = False
            if not ok:
                return False
        return True

    try:
        body = ast.parse(source).body
    except SyntaxError:
        return False
    return bool(body) and thin(body, True)


# ------------------------------------------------------------------------------------------------- the gate
def file_violations(rel: str, source: str, features: frozenset) -> list[Violation]:
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as error:
        return [Violation(rel, error.lineno or 1, "syntax", error.msg)]
    where = locate(rel)
    found: list[Violation] = []
    imported: set[str] = set()
    for line, target in targets(tree, where.package):
        if isinstance(target, Dynamic):
            wiring = where.module == APP and WIRING.fullmatch(target.template)
            if where.part != "adapters" and not wiring:
                found.append(Violation(rel, line, "dynamic-import",
                                       f"import of {target.template!r}: only a literal name is allowed here"))
            continue
        found += [Violation(rel, line, rule, message) for rule, message in judge(where, target)]
        segments = target.split(".")
        if segments[:2] == [PKG, "features"] and len(segments) > 2 and segments[2] in features - {where.feature}:
            imported.add(segments[2])
    others = features - {where.feature}
    if where.module != APP and len(others) >= 3 and others <= imported:
        found.append(Violation(rel, 1, "god-feature", f"imports all {len(others)} other features; only app.py does"))
    found += [Violation(rel, line, rule, message) for line, rule, message in code_findings(tree, where)]
    return found


def check_sources(files: Mapping[str, str]) -> list[Violation]:
    """Every violation in a tree given as {posix path from the repository root: text}, sorted, without repeats."""
    features = frozenset(features_of(files))
    found = list(shape(files))
    for rel, source in files.items():
        if rel.startswith(f"{PKG}/") and rel.endswith(".py"):
            found += file_violations(rel, source, features)
    if SETTINGS_FILE in files:
        found += settings_violations(files[SETTINGS_FILE])
    return sorted(set(found))


def load(root: Path) -> dict[str, str]:
    """The files of root/bazt_lab (text for .py, .md and .toml) and experiments/profiles.toml; no caches."""
    root = Path(root)
    files: dict[str, str] = {}
    for path in sorted((root / PKG).rglob("*")) if (root / PKG).is_dir() else []:
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            text = path.read_text(encoding="utf-8", errors="replace") if path.suffix in TEXT_SUFFIXES else ""
            files[path.relative_to(root).as_posix()] = text
    settings = root / SETTINGS_FILE
    if settings.is_file():
        files[SETTINGS_FILE] = settings.read_text(encoding="utf-8")
    return files


def check(root: Path = REPO_ROOT) -> list[Violation]:
    """Every violation under root/bazt_lab (and root/experiments/profiles.toml), sorted by path and line."""
    return check_sources(load(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bazt_lab import gate (docs/ARCHITECTURE_EXPERIMENTS.md)")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="directory that contains bazt_lab/")
    found = check(parser.parse_args(argv).root)
    for violation in found:
        print(violation)
    print(f"architecture gate: {len(found)} violation{'' if len(found) == 1 else 's'}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
