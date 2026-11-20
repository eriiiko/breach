"""#29 P1b -- the twin manifest and the debt ledger (residency design §6;
critique 4 B1, critique 5 M5 / M6).

Ordinary suite; nothing here needs a CUDA build and nothing skips. The
manifest (``tests/twin_manifest.toml``) covers every twin with a class, a
kind, slots, generated files and a ``level_at`` that is an ancestor of HEAD;
the generated ``files`` equal what ``tools/twin_manifest_gen.py`` derives for
the current tree (so a stale manifest fails here: regenerate and commit);
every ``src/simulation/`` module the corpus executes is in some twin's files
or classified in ``[not_twinned]``. The ledger's own logic
(``tools/twin_debt.py``) and the generator's pieces are exercised on
throwaway inputs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import twin_debt  # noqa: E402
import twin_manifest_gen as gen  # noqa: E402
from _git_fixture import commit, git, make_repo  # noqa: E402

KINDS = {"per-call", "resident launch core", "driver"}


@pytest.fixture(scope="module")
def manifest():
    return gen.load_manifest()


@pytest.fixture(scope="module")
def traced(manifest):
    """The corpus traced once for the module (~10 s: the medium match)."""
    return gen.trace_corpus(manifest)


def test_every_twin_is_fully_described(manifest):
    """Property: every twin names a class (P or T), a kind, its device
    sources, its slots, a non-empty generated file set and a ``level_at``
    that is an ancestor of HEAD; a twin marked not level names its issue.
    Breaks if a twin enters the manifest half-described, or its level_at is
    rewritten to a commit off this history (debt then cannot be computed)."""
    twins = manifest["twins"]
    assert twins
    for name, t in twins.items():
        assert t.get("class") in ("P", "T"), name
        assert t.get("kind") in KINDS, name
        assert t.get("device") and t.get("slots") and t.get("files"), name
        for d in t["device"]:
            assert (ROOT / d).is_file(), (name, d)
        assert isinstance(t.get("level"), bool), name
        if not t["level"]:
            assert t.get("issue"), name
        r = subprocess.run(["git", "-C", str(ROOT), "merge-base",
                            "--is-ancestor", t.get("level_at", ""), "HEAD"])
        assert r.returncode == 0, (name, t.get("level_at"))


def test_the_committed_files_are_the_generated_files(manifest, traced):
    """Property: each twin's committed ``files`` equal what the generator
    derives now (the corpus trace of its slots + its [cpp_stages] closure).
    Breaks when a slot starts executing another src/ module, a C++ TU gains
    an include, or a [cpp_stages] row changes without the manifest being
    regenerated -- run tools/twin_manifest_gen.py and commit."""
    files = gen.generate_files(manifest, traced)
    stale = {n: (sorted(set(fs) ^ set(manifest["twins"][n]["files"])))
             for n, fs in files.items()
             if fs != manifest["twins"][n]["files"]}
    assert not stale, ("twin manifest is stale (run tools/twin_manifest_gen"
                       f".py and commit): {stale}")


def test_every_executed_simulation_module_is_twinned_or_classified(
        manifest, traced):
    """Property: every src/simulation/ module the corpus executes is in some
    twin's files or in [not_twinned] with a reason, and [not_twinned] names
    only real modules that no twin already covers. Breaks when a new module
    joins the match path unclassified -- a twin's reference set could then
    miss it silently (critique 5 M6)."""
    files = gen.generate_files(manifest, traced)
    assert gen.unclassified(manifest, traced, files) == []
    covered = {f for fs in files.values() for f in fs}
    for mod, why in manifest.get("not_twinned", {}).items():
        assert isinstance(why, str) and why.strip(), mod
        assert (ROOT / mod).is_file(), f"[not_twinned] names {mod}: gone"
        assert mod not in covered, f"{mod} is twinned AND in [not_twinned]"


def test_every_slot_is_a_profiler_label_or_a_cpp_part(manifest):
    """Property: a twin's slots are tools/tick_stage_profile.py labels, or
    parts with a [cpp_stages] row; every [cpp_stages] key's label exists.
    Breaks if the profiler's map is renamed and a twin silently loses its
    traced files."""
    labels = set(gen.known_labels())
    cpp = manifest["cpp_stages"]
    for key in cpp:
        assert gen.traced_label(key) in labels, key
    for name, t in manifest["twins"].items():
        for slot in t["slots"]:
            assert gen.traced_label(slot) in labels, (name, slot)
            assert slot == gen.traced_label(slot) or slot in cpp, (name, slot)


# ---------------------------------------------------------------------------
# The ledger on a throwaway repo
# ---------------------------------------------------------------------------
LEDGER = """[twins.a]
class = "P"
kind = "per-call"
device = ["cpp/src/cuda_a.cu"]
slots = ["7 physics"]
level = true
level_at = "AT"
files = ["cpp/src/a.cpp", "src/simulation/a.py"]
[twins.b]
class = "T"
kind = "per-call"
device = ["cpp/src/cuda_b.cu"]
slots = ["4 shooting"]
level = LEVEL_B
issue = "#0"
level_at = "AT"
files = ["src/simulation/b.py"]
"""


def _ledger_repo(tmp_path, level_b=True):
    repo = make_repo(tmp_path / "r", {"cpp/src/a.cpp": "a\n",
                                      "src/simulation/a.py": "a\n",
                                      "src/simulation/b.py": "b\n",
                                      "docs/x.md": "x\n"})
    at = git(repo, "rev-parse", "HEAD")
    commit(repo, {"tests/twin_manifest.toml": LEDGER.replace("AT", at)
                  .replace("LEVEL_B", "true" if level_b else "false")})
    return repo


def test_debt_is_a_change_of_a_twin_file_since_level_at(tmp_path):
    """Property: a twin is in debt exactly when one of ITS files changed in
    level_at..HEAD -- an unrelated change makes no debt, another twin's file
    puts only that twin in debt -- and ``--mark-level`` clears it. Breaks if
    debt were remembered rather than computed, leaked between twins, or
    could not be cleared after a catch-up."""
    repo = _ledger_repo(tmp_path)
    assert twin_debt.debt(repo) == []
    commit(repo, {"docs/x.md": "y\n"})
    assert twin_debt.debt(repo) == []
    commit(repo, {"src/simulation/a.py": "a2\n"})
    rows = twin_debt.debt(repo)
    assert [n for n, _ in rows] == ["a"] and "a.py" in rows[0][1]
    assert twin_debt.main(["--repo", str(repo), "--mark-level", "a"]) == 0
    commit(repo, None, "level a")
    assert twin_debt.debt(repo) == []
    assert twin_debt.main(["--repo", str(repo)]) == 0


def test_a_twin_marked_not_level_is_debt_whatever_the_diff(tmp_path):
    """Property: ``level = false`` is reported with its issue even with no
    file changed, and the CLI exits 1. Breaks if a known-broken twin (#71)
    could pass the device door because nobody touched its files."""
    repo = _ledger_repo(tmp_path, level_b=False)
    rows = twin_debt.debt(repo)
    assert rows == [("b", "not level (#0)")]
    assert twin_debt.main(["--repo", str(repo)]) == 1


def test_a_level_at_off_the_history_is_an_error(tmp_path):
    """Property: a level_at that names no commit raises instead of reporting
    no debt. Breaks if a typo in the manifest silently emptied the ledger."""
    repo = _ledger_repo(tmp_path)
    p = repo / "tests" / "twin_manifest.toml"
    p.write_text(p.read_text(encoding="utf-8").replace(
        'level_at = "', 'level_at = "f00', 1), encoding="utf-8")
    with pytest.raises(twin_debt.LedgerError, match="level_at"):
        twin_debt.debt(repo)


# ---------------------------------------------------------------------------
# The generator's pieces
# ---------------------------------------------------------------------------
def test_the_cpp_closure_follows_local_includes_only(tmp_path):
    """Property: a TU's closure is itself plus every cpp/src header it
    includes with quotes, transitively; system headers and missing files
    are not followed. Breaks if a kit header (fixed_point.h) a TU pulls in
    were missing from its twin's reference set."""
    src = tmp_path / "cpp" / "src"
    src.mkdir(parents=True)
    (src / "a.cpp").write_text('#include "b.h"\n#include <vector>\n')
    (src / "b.h").write_text('#include "c.h"\n#include "gone.h"\n')
    (src / "c.h").write_text("int c;\n")
    (src / "d.h").write_text("int d;\n")
    got = gen.cpp_closure(["cpp/src/a.cpp"], tmp_path)
    assert got == {"cpp/src/a.cpp", "cpp/src/b.h", "cpp/src/c.h"}


def test_rewriting_touches_only_the_files_arrays():
    """Property: the generator replaces each twin's ``files`` array (empty,
    one-line or multi-line, comments inside) and keeps every other line;
    rewriting twice changes nothing. Breaks if regenerating drops the
    manifest's hand-kept comments, rows or another twin's array."""
    text = ('# header [x]\n[corpus]\nlevel = "l"\n'
            '[twins.a]\n# keep me\nclass = "P"\nfiles = []\n'
            '[twins.b]\nfiles = [  # c [y]\n    "old",\n]\nlevel = true\n'
            '[not_twinned]\n"m" = "r"\n')
    once = gen.rewrite_manifest_text(text, {"a": ["f1"], "b": ["f2", "f3"]})
    assert once == gen.rewrite_manifest_text(once, {"a": ["f1"],
                                                    "b": ["f2", "f3"]})
    import tomllib
    m = tomllib.loads(once)
    assert m["twins"]["a"]["files"] == ["f1"]
    assert m["twins"]["b"] == {"files": ["f2", "f3"], "level": True}
    assert "# keep me" in once and "# header [x]" in once
    assert m["not_twinned"] == {"m": "r"}


def test_the_tracer_attributes_a_frame_to_every_open_label():
    """Property: a frame lands under every slot / stage open above the phase
    (slot 7 AND its stage), a frame in ``step`` with no slot open under
    ``(rest of step)``, the controllers' frames under ``(outside step)``,
    and everything while building the match under ``(setup)`` only. Breaks
    if a stage's files were missing from its slot's set or setup leaked into
    a slot."""
    t = gen.SlotTracer(ROOT)
    f = str(ROOT / "src" / "simulation" / "gamemap.py")
    g = str(ROOT / "src" / "simulation" / "unit.py")
    h = str(ROOT / "src" / "simulation" / "combat.py")
    t.by_stack[(gen.SETUP, gen.STEP, "7 physics")].add(h)
    t.by_stack[(gen.OUTSIDE, gen.STEP, "7 physics",
                "7 combustion (host)")].add(f)
    t.by_stack[(gen.OUTSIDE, gen.STEP)].add(g)
    t.by_stack[(gen.OUTSIDE,)].add(h)
    t.by_stack[(gen.OUTSIDE, gen.STEP)].add("C:/not/the/repo.py")
    lab = t.by_label()
    assert lab["7 physics"] == lab["7 combustion (host)"] == [
        "src/simulation/gamemap.py"]
    assert lab[gen.REST_STEP] == ["src/simulation/unit.py"]
    assert lab[gen.OUTSIDE] == ["src/simulation/combat.py"]
    assert lab[gen.SETUP] == ["src/simulation/combat.py"]
