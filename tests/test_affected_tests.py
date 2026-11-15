"""tools/affected_tests.py -- the WIP-tier selection (#205).

The tool may over-select, never under-select: a test whose run could load a
changed file must be in the selection. These tests pin that, against an
oracle that is not the tool (Python's own import of a test module), plus the
fixed rules the closure cannot see.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import affected_tests as at  # noqa: E402

#: one test module per area: renderer, pure sim, the match runner, a CUDA
#: wrapper. Each must exist; a renamed one is updated here, not skipped.
_SAMPLES = ("tests/test_cold_overlay.py", "tests/test_unit_schema.py",
            "tests/test_bots_v1.py", "tests/test_cuda_unit_damage.py")

_PROBE = r"""
import importlib.util, json, sys
root, rel = sys.argv[1], sys.argv[2]
for p in (root, root + "/src", root + "/cpp/build/Release", root + "/tools",
          root + "/tests"):
    sys.path.insert(0, p)
spec = importlib.util.spec_from_file_location("probe_target", root + "/" + rel)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
files = sorted({getattr(m, "__file__", None) or "" for m in list(sys.modules.values())})
print(json.dumps(files))
"""


@pytest.mark.parametrize("rel", _SAMPLES)
def test_every_module_a_test_loads_is_in_its_closure(rel):
    """Property (no under-selection): every repo .py file Python actually
    loads when it imports a test module is in that module's static closure,
    so a change to any of them selects the test. Oracle: a fresh interpreter
    importing the module with the suite's path roots. Breaks if the closure
    misses an import form (a relative import, ``from pkg import module``, an
    import inside a function the module runs at import time, a package
    ``__init__``)."""
    assert (ROOT / rel).is_file(), rel
    out = subprocess.run([sys.executable, "-c", _PROBE, str(ROOT), rel],
                         capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = set()
    for f in json.loads(out.stdout.strip().splitlines()[-1]):
        try:
            r = Path(f).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if r.endswith(".py"):
            loaded.add(r)
    assert loaded, "the probe saw no repo module (vacuous)"
    missing = loaded - at.closure(rel)
    assert not missing, f"{rel} loads files its closure lacks: {sorted(missing)}"


def test_a_subprocess_script_is_followed():
    """Property: a script a test launches in a subprocess is in its closure
    -- a CUDA wrapper reaches its ``cuda_*_check.py`` and the native module,
    an E2E test reaches ``main.py``. Breaks if string-launched scripts are
    ignored (a CUDA check change would select nothing)."""
    cuda = at.closure("tests/test_cuda_unit_damage.py")
    assert "tests/cuda_unit_damage_check.py" in cuda
    assert at.NATIVE in cuda
    assert "main.py" in at.closure("tests/test_control_direct_e2e.py")


def test_a_native_change_reaches_every_test_that_loads_the_module():
    """Property: a change under ``cpp/`` selects exactly the tests whose
    closure holds the native module, CUDA wrappers included. Breaks if C++
    changes reach only Python importers or nothing."""
    sel = at.select(["cpp/src/fixed_point.h"])
    want = {t for t in at.test_files() if at.NATIVE in at.closure(t)}
    assert not sel.full
    assert sel.tests == want
    assert "tests/test_cuda_unit_damage.py" in want


def test_a_python_change_selects_what_reaches_it_and_spares_the_rest():
    """Property: a changed module selects every test whose closure holds it
    (a direct importer included) and spares the others; a renderer module is
    the motivating case -- it spares some test. Breaks if selection goes
    by directory or by name instead of the closure, or degenerates to all."""
    changed = "renderer/cold_overlay.py"
    sel = at.select([changed])
    assert not sel.full
    assert "tests/test_cold_overlay.py" in sel.tests
    assert sel.tests == {t for t in at.test_files()
                         if changed in at.closure(t)}
    assert len(sel.tests) < len(at.test_files())


def test_what_the_closure_cannot_see_runs_everything():
    """Property: a change the closure cannot see -- conftest, pyproject, a
    deleted or renamed .py, a data file no Python file names -- selects the
    FULL suite. Breaks if any of them silently selects a subset."""
    for path in ("tests/conftest.py", "pyproject.toml",
                 "src/simulation/no_such_module_205.py",
                 "_".join(("nowhere", "dir", "205")) + "/data.bin"):
        assert at.select([path]).full, path


def test_a_data_file_reaches_the_tests_whose_closure_names_it():
    """Property: a shader change selects the tests whose closure holds a
    Python file naming ``shaders``, and not all tests. Breaks if data files
    are ignored or run everything."""
    sel = at.select(["shaders/world_resolve.fs"])
    assert not sel.full and sel.tests
    assert len(sel.tests) < len(at.test_files())


def test_docs_reach_nothing():
    """Property: a docs-only change selects no test and not the full suite.
    Breaks if inert paths fall through to FULL."""
    sel = at.select(["docs/test_suite_timing_2026-11-15.md", "README.md"])
    assert not sel.full and not sel.tests
