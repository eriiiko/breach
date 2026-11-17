"""What a test does, read from its source -- the suite-timing instrument's
one classifier (#205).

Two readers: ``tools/suite_timing.py`` sorts a measured run's slowest tests
into kinds (the table in ``docs/test_suite_timing_2026-11-15.md``), and
``tests/conftest.py`` asks ``is_device`` to keep every test that drives the
GPU or opens a window on ONE xdist worker (never two CUDA contexts or two
hidden windows at once).

A test's source is its own function plus every module-level function it calls
(transitively) and every module-level fixture its signature names: the helper a
test calls is where the match is played or the subprocess launched. Patterns,
not imports, because the CUDA and GL checks run as subprocess scripts.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

# The device patterns: a CUDA check through the harness, the real main.py in a
# hidden window (the E2E driver, its probes, main.py as a subprocess's argv), a
# GL check script. A module that imports pyray opens its window IN-PROCESS, so
# every test of it counts (``is_device``).
_CUDA = r"run_cuda_script|cuda_harness\.run"
_WINDOW = r"e2e_drive|gl_\w+_check|_e2e_\w+_probe|[\"']main\.py[\"']\s*,"
DEVICE = re.compile(f"{_CUDA}|{_WINDOW}")
_PYRAY = re.compile(r"^\s*(import pyray|from pyray )", re.M)

# First match wins, in this order.
KINDS = (
    ("cuda subprocess", re.compile(_CUDA)),
    ("window / GL subprocess", re.compile(_WINDOW)),
    ("plays a match", re.compile(
        r"run_match|MatchSession|spawn_match|BatchedEnv|EnvRunner|_play\("
        r"|pinned|match_pin|play_match|bot_tournament|lockstep")),
    ("level generation", re.compile(
        r"gen_ship|ship_grammar|ship_layout|generate\(|bake_level")),
    ("source scan", re.compile(r"\.rglob\(|ast\.parse|tokenize|iter_sources")),
    ("physics scene", re.compile(
        r"Simulation\(|PhysicsEngine|run_substeps|GameMap\(|\.step\(")),
)
OTHER = "other"


@lru_cache(maxsize=None)
def _functions(path: str) -> dict[str, str]:
    """Every function in the module by name -> its source (a later def of the
    same name wins, as Python's would)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    return {n.name: ast.get_source_segment(text, n) or ""
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _closure(funcs: dict[str, str], name: str, seen: set[str]) -> str:
    if name in seen or name not in funcs:
        return ""
    seen.add(name)
    body = funcs[name]
    out = [body]
    for callee in sorted(set(re.findall(r"\b(\w+)\(", body))):
        out.append(_closure(funcs, callee, seen))
    return "\n".join(out)


@lru_cache(maxsize=None)
def source_of(path: str, name: str) -> str:
    """The test's function, its module-level callees and the fixtures named in
    its signature, as one text ('' when the module has no such function)."""
    funcs = _functions(path)
    if name not in funcs:
        return ""
    seen: set[str] = set()
    out = [_closure(funcs, name, seen)]
    signature = funcs[name].split(")", 1)[0]
    for arg in re.findall(r"\w+", signature):
        out.append(_closure(funcs, arg, seen))
    return "\n".join(out)


def kind(path: str, name: str) -> str:
    """The first KINDS row whose pattern the test's source matches."""
    src = source_of(path, name) or Path(path).read_text(
        encoding="utf-8", errors="replace")
    for label, pattern in KINDS:
        if pattern.search(src):
            return label
    return OTHER


def is_device(path: str, name: str) -> bool:
    """Does the test drive the GPU or open a window (in a subprocess, or in
    this process through pyray)?"""
    return bool(DEVICE.search(source_of(path, name))) or _imports_pyray(path)


@lru_cache(maxsize=None)
def _imports_pyray(path: str) -> bool:
    return bool(_PYRAY.search(
        Path(path).read_text(encoding="utf-8", errors="replace")))


def split_nodeid(nodeid: str) -> tuple[str, str]:
    """``tests/test_x.py::test_y[p]`` -> (``tests/test_x.py``, ``test_y``);
    a class-held test keeps its method name."""
    path, _, rest = nodeid.partition("::")
    return path, rest.split("::")[-1].split("[")[0]
