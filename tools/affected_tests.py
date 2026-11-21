"""tools/affected_tests.py -- the tests a change can break (#205, the WIP tier).

    python tools/affected_tests.py                 # list them, with why
    python tools/affected_tests.py --run           # run them (pytest -q)
    python tools/affected_tests.py --run -- -m "not slow"   # extra pytest args

A test file is affected when its DEPENDENCY CLOSURE contains a changed file.
The closure is read from the source, never hand-marked, so a new test is
classified the moment it exists:

- its Python imports, transitively, resolved inside the repo on the suite's
  own path roots (the repo root, ``src``, ``tools``, ``tests``);
- the scripts it launches in a subprocess -- an ``import x`` or ``from x
  import`` written inside a string (the CUDA checks' ``-c`` script), a
  ``"<name>.py"`` literal (``main.py`` as argv), the E2E driver, a
  ``gl_*_check`` script -- each followed in turn;
- the native module: a file that imports ``breach_physics`` depends on
  ``cpp/``, so a C++ / CUDA change reaches every test whose closure loads it.

A changed DATA file (a shader, an asset, a level, a TOML) reaches the tests
whose closure holds a Python file that names it -- its path's first component
(``shaders``, ``levels``, ``config.toml``) appears in that file's text. When
nothing names it, or the change is one the closure cannot see (``conftest.py``,
``pyproject.toml``, a deleted or renamed ``.py``), the answer is the FULL
SUITE: the tool may over-select, never under-select. Docs and pictures reach
nothing.

This is the work-in-progress tier, like ``-m "not slow"`` (it combines with
it). The full suite stays the gate before a merge.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: the suite's import roots, in its own order (tests insert these).
ROOTS = (ROOT, ROOT / "src", ROOT / "tools", ROOT / "tests")
NATIVE = "<native: cpp/>"
#: directories whose files never hold anything Python reads or runs.
_SKIP_DIRS = {".git", ".claude", "out", "scratchpad", "stubs", "__pycache__",
              "build", "build_cuda", "node_modules"}
#: changes that reach no test: the docs tree and scratch output. Everything
#: else -- a picture, a CSV, a .txt, a Markdown file -- may be read by a test
#: (a committed texture, a golden, slow_tests.txt, CLAUDE.md and the rules,
#: prototypes/ levels), so it goes through the naming rule below.
_INERT = re.compile(r"(^docs/|^scratchpad/|^out/|^\.gitignore$)")
#: changes the closure cannot see: the full suite.
_EVERYTHING = re.compile(r"(^tests/conftest\.py$|^pyproject\.toml$"
                         r"|^pytest\.ini$|^setup\.cfg$|^pyrightconfig\.json$)")

_STR_IMPORT = re.compile(r"\b(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")
_STR_PYFILE = re.compile(r"^[\w./\\-]+\.py$")
_LAUNCH_NAMES = re.compile(r"\b(e2e_drive|gl_\w+_check|_e2e_\w+_probe)\b")


def _rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


@lru_cache(maxsize=None)
def _resolve(module: str) -> str | None:
    """``a.b.c`` -> the repo file it names (a module or a package's
    ``__init__``), trying shorter prefixes for ``from a.b import name``."""
    parts = module.split(".")
    while parts:
        for root in ROOTS:
            base = root.joinpath(*parts)
            for cand in (base.with_suffix(".py"), base / "__init__.py"):
                if cand.is_file():
                    return _rel(cand)
        parts = parts[:-1]
    return None


def _resolve_pyfile(lit: str) -> str | None:
    lit = lit.replace("\\", "/")
    for root in (ROOT, ROOT / "tests", ROOT / "tools"):
        cand = root / lit
        if cand.is_file():
            return _rel(cand)
    return None


def _relative_module(rel: str, level: int, module: str | None) -> str:
    """A relative import's absolute module name, on the suite's roots."""
    parts = list(Path(rel).parent.parts)
    if parts and parts[0] in ("src", "tools", "tests"):
        parts = parts[1:]
    if level > 1:
        parts = parts[:len(parts) - (level - 1)]
    return ".".join(x for x in (".".join(parts), module or "") if x)


@lru_cache(maxsize=None)
def deps_of(rel: str) -> frozenset:
    """The repo files (and NATIVE) one Python file reaches directly."""
    text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = (_relative_module(rel, node.level, node.module)
                   if node.level else (node.module or ""))
            names.add(mod)
            names.update(f"{mod}.{a.name}" if mod else a.name
                         for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            for a, b in _STR_IMPORT.findall(s):
                names.add(a or b)
            if _STR_PYFILE.match(s.strip()):
                hit = _resolve_pyfile(s.strip())
                if hit:
                    names.add("file:" + hit)
    names.update(_LAUNCH_NAMES.findall(text))
    out: set = set()
    for n in names:
        if not n:
            continue
        if n.startswith("file:"):
            out.add(n[5:])
            continue
        if n.split(".")[0] == "breach_physics":
            out.add(NATIVE)
            continue
        hit = _resolve(n)
        if hit and hit != rel:
            out.add(hit)
    return frozenset(out)


@lru_cache(maxsize=None)
def closure(rel: str) -> frozenset:
    """Every repo file (and NATIVE) a Python file reaches, itself included."""
    seen = {rel}
    todo = [rel]
    while todo:
        cur = todo.pop()
        if cur == NATIVE:
            continue
        for d in deps_of(cur):
            if d not in seen:
                seen.add(d)
                todo.append(d)
    return frozenset(seen)


@lru_cache(maxsize=None)
def test_files() -> tuple:
    return tuple(sorted(_rel(p) for p in (ROOT / "tests").glob("test_*.py")))


@lru_cache(maxsize=None)
def python_files() -> tuple:
    out = []
    for p in ROOT.rglob("*.py"):
        if _SKIP_DIRS.intersection(p.relative_to(ROOT).parts):
            continue
        out.append(_rel(p))
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


@dataclass
class Selection:
    full: bool = False
    tests: set = field(default_factory=set)
    #: changed path -> why (how many test files it reached, or why FULL).
    why: dict = field(default_factory=dict)


def reach(path: str, candidates=None) -> tuple:
    """``(tests, why, full)`` for ONE changed repo-relative path: the test
    files among ``candidates`` (default: ``tests/test_*.py``) whose closure
    the change reaches, the reason, and whether the closure cannot see it
    (then every test is reached). THE per-path rule of :func:`select`; the
    staging gate (``tools/staging_gate.py``) asks it which change reaches a
    red test."""
    cands = test_files() if candidates is None else tuple(candidates)
    if not (ROOT / path).exists():
        return set(), "FULL: deleted or renamed file", True
    if _INERT.search(path):
        return set(), "inert (docs / scratch)", False
    if _EVERYTHING.search(path):
        return set(), "FULL: the closure cannot see this file", True
    if path.endswith(".py"):
        hit = {t for t in cands if path in closure(t)}
    elif path.startswith("cpp/"):
        hit = {t for t in cands if NATIVE in closure(t)}
    else:
        key = path.split("/")[0]
        readers = {f for f in python_files() if key in _text(f)}
        if not readers and path.endswith(".md"):
            # Prose nothing names: no test reads it (a reader of the
            # rules names .claude, of the index CLAUDE.md).
            return set(), "inert (Markdown no Python file names)", False
        if not readers:
            return set(), f"FULL: no Python file names {key!r}", True
        hit = {t for t in cands if closure(t) & readers}
    return hit, f"{len(hit)} test file(s)", False


def select(changed) -> Selection:
    """The test files ``changed`` (repo-relative posix paths) can break."""
    sel = Selection()
    for path in sorted(set(changed)):
        hit, why, full = reach(path)
        sel.full |= full
        sel.tests |= hit
        sel.why[path] = why
    return sel


_CACHED = (_resolve, deps_of, closure, test_files, python_files, _text)


@contextmanager
def rooted(root):
    """Read another checkout (a staging worktree) for the duration: the
    module's ``ROOT`` / ``ROOTS`` re-pointed and every cache cleared, both
    restored on exit."""
    global ROOT, ROOTS
    saved = ROOT, ROOTS
    for fn in _CACHED:
        fn.cache_clear()
    ROOT = Path(root).resolve()
    ROOTS = (ROOT, ROOT / "src", ROOT / "tools", ROOT / "tests")
    try:
        yield
    finally:
        ROOT, ROOTS = saved
        for fn in _CACHED:
            fn.cache_clear()


def changed_files(base: str) -> list:
    """Committed since the merge-base with ``base``, plus staged, unstaged
    and untracked changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    mb = git("merge-base", "HEAD", base).strip()
    out = set(git("diff", "--name-only", mb).split())
    out |= set(git("ls-files", "--others", "--exclude-standard").split())
    return sorted(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="affected_tests",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main",
                    help="compare against the merge-base with this ref")
    ap.add_argument("--run", action="store_true",
                    help="run the selection with pytest -q")
    args = ap.parse_args(argv)
    sel = select(changed_files(args.base))
    for path, why in sel.why.items():
        print(f"  {path}: {why}")
    if sel.full:
        cmd = ["tests"]
        print(f"FULL SUITE ({len(test_files())} test files)")
    else:
        cmd = sorted(sel.tests)
        print(f"{len(cmd)} of {len(test_files())} test files")
        for t in cmd:
            print(f"    {t}")
    if not args.run:
        return 0
    if not cmd:
        print("nothing to run")
        return 0
    return subprocess.call([sys.executable, "-m", "pytest", *cmd, "-q",
                            *extra], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
