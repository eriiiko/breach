"""#29 P1b -- Class P is ENFORCED, not assumed (residency design §6; critique 5 B1).

Class-P device code is what the played ``--cuda`` / ``--resident`` game runs;
it is kept level per patch. Every CUDA gate skips in a checkout without
``cpp/build_cuda``, so this test -- ordinary suite, it NEVER skips -- asks
the branch instead: when its diff from ``git merge-base HEAD main`` touches a
Class-P path (a Class-P twin's ``files`` or ``device`` in
``tests/twin_manifest.toml``, or a ``[class_p] build`` pattern), the
untracked stamp ``.cuda_green`` must hold HEAD's SHA -- written only by a
green ``tools/cuda_green.py`` run against this checkout's own CUDA build.
On main the diff is empty and it passes. The git-workflow skill's merge step
refuses while it is red.

The verdict's machinery (``tools/twin_debt.class_p_problems``) is exercised
on throwaway repos below; the first test applies it to this checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import twin_debt  # noqa: E402
from _git_fixture import commit, git, make_repo, write  # noqa: E402


def test_a_class_p_change_merges_only_with_a_green_stamp_at_head():
    """Property: if this branch changed a Class-P path since its merge-base
    with main, every such change is committed and ``.cuda_green`` holds
    HEAD's SHA. Breaks when a Class-P change (a physics core, a device
    source, the build, the bindings) is about to merge from a checkout that
    never ran the CUDA suite on its own build at HEAD -- the #71 failure."""
    problems = twin_debt.class_p_problems(ROOT)
    assert not problems, (twin_debt.CLASS_P_MESSAGE + "\n  "
                          + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# The verdict on throwaway repos
# ---------------------------------------------------------------------------
MANIFEST = """[class_p]
build = ["cpp/CMakeLists.txt", "cpp/src/*.cu"]
[twins.p]
class = "P"
kind = "per-call"
device = ["cpp/src/cuda_p.cu"]
slots = ["7 physics"]
level = true
level_at = "x"
files = ["cpp/src/p.cpp", "src/simulation/p.py"]
[twins.t]
class = "T"
kind = "per-call"
device = ["cpp/src/cuda_t.cu"]
slots = ["4 shooting"]
level = true
level_at = "x"
files = ["src/simulation/t.py"]
"""


def _repo(tmp_path):
    repo = make_repo(tmp_path / "r", {
        "tests/twin_manifest.toml": MANIFEST,
        "cpp/src/p.cpp": "int p;\n", "src/simulation/p.py": "p = 1\n",
        "src/simulation/t.py": "t = 1\n", "docs/a.md": "a\n",
        "cpp/CMakeLists.txt": "#\n"})
    git(repo, "switch", "-q", "-c", "feature")
    return repo


def _stamp(repo, sha=None):
    write(repo, {twin_debt.STAMP: (sha or git(repo, "rev-parse", "HEAD"))
                 + "\n1 passed\n"})


def test_a_branch_without_class_p_changes_needs_no_stamp(tmp_path):
    """Property: docs and Class-T reference changes need no stamp. Breaks if
    every gameplay patch were forced through a CUDA build."""
    repo = _repo(tmp_path)
    commit(repo, {"docs/a.md": "b\n", "src/simulation/t.py": "t = 2\n"})
    assert twin_debt.class_p_problems(repo) == []


def test_a_class_p_change_needs_the_stamp_at_head(tmp_path):
    """Property: a change to a Class-P twin's reference file, its device
    source or a build pattern is refused without a stamp, refused with a
    stamp of an older commit, and passes with the stamp at HEAD. Breaks if
    any of the three Class-P path kinds slips through or a stale stamp
    counts."""
    for path in ("src/simulation/p.py", "cpp/src/cuda_p.cu",
                 "cpp/src/other.cu", "cpp/CMakeLists.txt"):
        repo = _repo(tmp_path / path.replace("/", "_"))
        old = git(repo, "rev-parse", "HEAD")
        commit(repo, {path: "changed\n"})
        assert twin_debt.class_p_problems(repo), path
        _stamp(repo, old)
        assert twin_debt.class_p_problems(repo), path
        _stamp(repo)
        assert twin_debt.class_p_problems(repo) == [], path


def test_an_uncommitted_class_p_change_is_refused_even_with_a_stamp(tmp_path):
    """Property: a Class-P edit not yet committed is refused although the
    stamp holds HEAD -- the stamp proves HEAD, not the working tree. Breaks
    if an edit made after the green run could merge under its stamp."""
    repo = _repo(tmp_path)
    commit(repo, {"cpp/src/p.cpp": "int p2;\n"})
    _stamp(repo)
    write(repo, {"cpp/src/p.cpp": "int p3;\n"})
    assert any("uncommitted" in p for p in twin_debt.class_p_problems(repo))


def test_the_integration_line_itself_needs_no_stamp(tmp_path):
    """Property: on main the diff from the merge-base is empty, so the test
    passes there with no stamp. Breaks if every run of the suite on main (or
    on a training checkout at a tag on main) went red."""
    repo = _repo(tmp_path)
    git(repo, "switch", "-q", "main")
    commit(repo, {"cpp/src/p.cpp": "int p4;\n"})
    assert twin_debt.class_p_problems(repo) == []
