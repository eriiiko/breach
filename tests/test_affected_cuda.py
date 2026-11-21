"""tools/affected_cuda.py -- the change-based tier for the card checks.

The selector may over-select, never under-select: a card check that can run
a changed definition must be in the selection. These tests pin the rules
against oracles that are not the selector's own reading where one exists
(pytest's ``-k cuda`` filter, a plain grep for a definition), and the fixed
"touches everything" rules. The execution evidence (tripwires at real
mutation sites, a full card run) is in
docs/affected_cuda_evidence_2026-11-21.md; it needs the GPU, these do not.
"""
from __future__ import annotations

import ast
import re
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import affected_cuda as ac  # noqa: E402
import affected_tests as at  # noqa: E402


def test_the_card_set_is_every_file_cuda_green_runs_on_the_card():
    """Property: every test file holding a test that cuda_green's ``-k`` collects
    (cuda_green's ``-k`` word in its module or test name) and that launches
    the harness is a card check, and every card check is such a
    file. Breaks if a wrapper is missed (a new ``test_cuda_*.py``, a cuda
    test in another module) or a CPU-only file is counted."""
    word = ac.card_select()
    assert re.fullmatch(r"\w+", word), word
    want = set()
    for t in at.test_files():
        text = at._text(t)
        if not ac._CUDA_SRC.search(text):    # the harness launch
            continue
        if word in Path(t).stem or re.search(rf"def test\w*{word}", text):
            want.add(t)
    assert want, "no wrapper launches the harness (vacuous)"
    assert set(ac.checks()) == want


def test_every_check_script_is_reached_by_its_wrapper():
    """Property: each ``tests/cuda_*_check.py`` is in the Python closure of
    some card check, so a change to the script selects its wrapper. Breaks
    if the subprocess script stops being followed."""
    scripts = sorted(at._rel(p) for p in (ROOT / "tests").glob("cuda_*_check.py"))
    assert scripts
    reached = set().union(*(ac.py_closure(c) for c in ac.checks()))
    assert not [s for s in scripts if s not in reached]


_TU_DEF = r"^[^\n;{{}}()]*\b{name}\s*\([^;{{]*\)\s*(?:const\s*)?\{{"


def test_a_binding_reaches_the_tu_that_defines_what_it_calls():
    """Property: for every module-level binding whose body calls a
    ``breach_cuda::f(...)``, the binding's reach holds every TU in which a
    plain grep finds a definition of ``f`` with a body. Breaks if a
    TU-defined function stops linking from bindings.cpp (the linker's view)
    or a binding body is cut short (a bracket in a docstring)."""
    b = ac.bindings()
    tus = [r for r in ac.cpp_files() if r.endswith((".cu", ".cpp"))
           and r != ac.BINDINGS]
    checked = 0
    for name, rows in sorted(b.bodies.items()):
        for owner, text in rows:
            if owner:
                continue
            for fn in set(re.findall(r"breach_cuda::(\w+)\s*\(", text)):
                rx = re.compile(_TU_DEF.format(name=re.escape(fn)), re.M)
                homes = {t for t in tus if rx.search(ac._source(t))}
                if not homes:
                    continue
                missing = homes - ac.binding_reach(name)
                assert not missing, f"binding {name} -> {fn}: {missing}"
                checked += 1
    assert checked > 50, f"only {checked} binding calls checked (vacuous?)"


@pytest.mark.parametrize("path", [
    "cpp/src/bindings.cpp", "cpp/CMakeLists.txt", "cpp/build_cuda.bat",
    "cpp/gen_build_id.cmake", "cpp/src/fixed_point.h",
    "cpp/src/cuda_fixedpoint_device.cuh", "config.toml",
    "tests/conftest.py", "cpp/src/no_such_kernel_affected_cuda.cu",
    "src/simulation/no_such_module_affected_cuda.py"])
def test_what_touches_every_check_runs_the_full_card_set(path):
    """Property: the bindings, the build, the kits, config.toml, what the
    Python closure cannot see and a deleted or renamed file select EVERY
    card check. Breaks if any of them silently selects a subset."""
    sel = ac.select([path])
    assert sel.full and sel.checks == set(ac.checks()), (path, sel.why)


def test_a_shared_header_runs_the_full_card_set():
    """Property: a header more than half of the TUs include is shared and
    selects every check, whatever the reach says. Breaks if the rule is
    dropped (a layout header's change reaching only the checks that name
    one of its definitions)."""
    shared = sorted(ac.shared_headers())
    assert shared, "no shared header found (vacuous)"
    for h in shared:
        assert ac.select([h]).full, h


def test_a_unit_kernel_spares_a_check_that_never_calls_one():
    """Property: a change to a device unit kernel TU selects the checks
    whose reach holds it and spares one that never calls into the unit
    side (the hello-world check), so the tier is a real subset. Breaks if
    the selection degenerates to every check (a member name such as
    ``step`` reaching every class, a field name reaching every struct)."""
    path = "cpp/src/cuda_unit_melee.cu"
    sel = ac.select([path])
    assert not sel.full
    assert sel.checks == {c for c in ac.checks() if path in ac.cpp_reach(c)} \
        | {c for c in ac.checks() if ac.guards(c) & {"melee"}}
    assert "tests/test_cuda_unit_melee.py" in sel.checks
    assert "tests/test_cuda_s0_hello.py" not in sel.checks


def test_a_python_change_selects_through_the_closure():
    """Property: a changed Python module selects every card check whose
    affected_tests closure holds it (THE per-path rule, shared) and spares
    a check that never imports it. Breaks if Python changes go by name or
    directory instead of the closure, or select everything."""
    path = "src/simulation/conductor.py"
    sel = ac.select([path])
    assert not sel.full
    want = {c for c in ac.checks() if path in at.closure(c)}
    assert want <= sel.checks
    assert "tests/test_cuda_conductor.py" in sel.checks
    assert "tests/test_cuda_s0_hello.py" not in sel.checks


def test_a_level_reaches_the_checks_that_name_it():
    """Property: a change inside ``levels/<name>/`` selects every check
    whose closure names ``<name>`` and no check that names neither it nor a
    TOML that names it. Breaks if levels go back to the coarse rule (every
    reader of ``levels``) or are missed."""
    lvl = ROOT / "levels" / "ship_demo"
    path = next(p for p in sorted(lvl.iterdir()) if p.is_file())
    rel = at._rel(path)
    sel = ac.select([rel])
    assert not sel.full and sel.checks
    for c in ac.checks():
        named = any(re.search(r"\bship_demo\b", at._text(f))
                    for f in ac.py_closure(c))
        if named:
            assert c in sel.checks, c
    assert "tests/test_cuda_s0_hello.py" not in sel.checks


def test_a_twin_file_selects_the_checks_that_launch_its_kernels():
    """Property: a change to a twin's reference ``files`` selects every
    check whose reach launches one of that twin's kernels. Breaks if the
    twin manifest stops feeding the selection."""
    name, (files, _) = next((t, v) for t, v in sorted(ac.twins().items())
                            if t == "movement")
    py = next(f for f in files if f.endswith(".py"))
    sel = ac.select([py])
    guards = {c for c in ac.checks() if name in ac.guards(c)}
    assert guards, "no check launches the movement kernels (vacuous)"
    assert guards <= sel.checks


def test_the_run_writes_one_log_per_check_ending_in_its_exit(tmp_path):
    """Property: ``run`` runs each check as its own pytest process with
    ``-k cuda``, writes one log per check whose last line is
    ``CHECK_EXIT=<code>``, and returns 1 when any check failed. Breaks if
    logs are shared, the exit line is missing or a red check reads green."""
    good = tmp_path / "test_cuda_fake_good.py"
    bad = tmp_path / "test_cuda_fake_bad.py"
    good.write_text("def test_cuda_ok():\n    assert True\n")
    bad.write_text(textwrap.dedent("""
        def test_cuda_red():
            assert False
        def test_not_selected_by_k():
            assert False
    """))
    logs = tmp_path / "logs"
    rc = ac.run([str(good), str(bad)], logs)
    assert rc == 1
    tails = {p.stem: p.read_text().strip().splitlines()[-1]
             for p in logs.glob("*.log")}
    assert tails == {"test_cuda_fake_good": "CHECK_EXIT=0",
                     "test_cuda_fake_bad": "CHECK_EXIT=1"}


def test_member_gating_rests_on_no_virtual_dispatch():
    """Property: no C++ file under cpp/src declares a ``virtual`` member.
    The selector reaches an out-of-line ``C::m`` only while class ``C`` is
    live in a reach, which is sound only while a member runs on an object
    of its own class; a virtual call through a base would run a derived
    class's member with no name of it in the caller. Breaks the day a
    ``virtual`` lands: widen the member rule (a base's live class makes its
    derived classes live) before relying on the tier again."""
    hits = [f"{rel}" for rel in ac.cpp_files()
            if re.search(r"\bvirtual\b", ac.strip_strings(ac._source(rel)))]
    assert not hits, hits


def _literal_strings(node, tables) -> bool:
    """A tuple / list of string literals, or a module name bound to one."""
    if isinstance(node, ast.Name):
        return node.id in tables
    return isinstance(node, (ast.Tuple, ast.List)) and node.elts and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str)
        for e in node.elts)


def test_a_binding_is_never_looked_up_by_a_computed_name():
    """Property: in every Python file a card check can load, a ``getattr``
    on the module object (any name holding ``bp`` or ``breach_physics``)
    takes a string literal, an f-string, or a loop variable over a literal
    table of strings -- the forms whose names the selector reads (an
    identifier string, an f-string pattern). Breaks when a binding is looked
    up by a name the source never spells (``getattr(bp, "set_" + b)``, a
    name read from config): the selector would not see which bindings run."""
    bad = []
    files = set().union(*(ac.py_closure(c) for c in ac.checks()))
    for rel in sorted(files):
        text = at._text(rel)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        tables = {t.id for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name)
                  and _literal_strings(n.value, set())}
        loop_vars = {n.target.id for n in ast.walk(tree)
                     if isinstance(n, (ast.For, ast.comprehension))
                     and isinstance(n.target, ast.Name)
                     and _literal_strings(n.iter, tables)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "")                     == "getattr" and len(node.args) >= 2:
                obj = ast.get_source_segment(text, node.args[0]) or ""
                arg = node.args[1]
                ok = isinstance(arg, (ast.Constant, ast.JoinedStr)) or (
                    isinstance(arg, ast.Name) and arg.id in loop_vars)
                if re.search(r"bp|breach_physics", obj) and not ok:
                    bad.append(f"{rel}:{node.lineno}")
    assert not bad, bad


_BP_NAME = re.compile(r"^(self\.)?_?bp(_cuda)?$|^breach_physics$")


def test_the_module_never_reaches_a_file_that_does_not_name_it():
    """Property: the module object reaches only files whose text names it
    (``ac.MODULE_REF``), so the selector may drop module-level binding names
    from every other file (a twin's ``atan2_q16``, a UI field
    ``obs_masks``). In every Python file a card check can load: the module
    is stored only under a ``bp`` name, returned only to callers whose
    files name it, and wherever it is
    passed to a function of the caller's closure, that function's file names
    the module or its parameter is a ``bp`` name. Breaks when the module
    travels under another name (``self.mod = bp``, ``m = get_bp()`` in a
    file that never says bp, a
    ``def f(mod)`` in a file that never says bp): the selector would miss
    the bindings that file calls."""
    files = sorted(set().union(*(ac.py_closure(c) for c in ac.checks())))
    trees, defs = {}, {}
    for rel in files:
        try:
            trees[rel] = t = ast.parse(at._text(rel))
        except SyntaxError:
            continue
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = [x.arg for x in n.args.posonlyargs + n.args.args]
                if a and a[0] in ("self", "cls"):
                    a = a[1:]
                defs.setdefault(n.name, []).append((rel, a))
            elif isinstance(n, ast.ClassDef):
                for s in n.body:
                    if isinstance(s, ast.FunctionDef) and s.name == "__init__":
                        defs.setdefault(n.name, []).append(
                            (rel, [x.arg for x in s.args.args][1:]))
    bad, returners = [], set()
    for rel, t in trees.items():
        for n in ast.walk(t):
            if isinstance(n, ast.Assign) and _BP_NAME.match(ast.unparse(n.value)):
                bad += [f"{rel}:{n.lineno} stores it as {ast.unparse(x)}"
                        for x in n.targets if not _BP_NAME.match(ast.unparse(x))]
            elif isinstance(n, ast.FunctionDef) and any(
                    isinstance(r, ast.Return) and r.value is not None
                    and _BP_NAME.match(ast.unparse(r.value))
                    for r in ast.walk(n)):
                returners.add(n.name)
            elif isinstance(n, ast.Call):
                f = n.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in ("getattr", "hasattr", "isinstance", "print", "id"):
                    continue
                for i, a in enumerate(n.args):
                    if not _BP_NAME.match(ast.unparse(a)):
                        continue
                    for drel, params in defs.get(name, ()):
                        if drel not in at.closure(rel) or i >= len(params):
                            continue
                        if not _BP_NAME.match(params[i]) and \
                                not ac.MODULE_REF.search(at._text(drel)):
                            bad.append(f"{rel}:{n.lineno} {name} -> {drel}")
    # a function that RETURNS the module: every caller's file names it.
    for rel, t in trees.items():
        if ac.MODULE_REF.search(at._text(rel)):
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and (getattr(n.func, "id", None) or
                                            getattr(n.func, "attr", None)) \
                    in returners:
                bad.append(f"{rel}:{n.lineno} takes it from a call")
    assert trees and returners and not bad, bad


def test_a_name_in_a_file_that_never_names_the_module_reaches_nothing():
    """Property: an obs builder change selects the observation check and no
    check that only shares a NAME with an obs binding (``ui/draw.py``'s
    ``obs_masks`` field, reached by every unit check's closure). Breaks if
    module-level binding names count again in files that cannot hold the
    module (27 checks instead of 1, measured 2026-11-21)."""
    sel = ac.select(["cpp/src/cuda_obs_windows.cu"])
    assert "tests/test_cuda_obs.py" in sel.checks
    assert "tests/test_cuda_unit_kit.py" not in sel.checks
