"""tools/staging_gate.py -- the staging gate (one suite + one card gate for
several ready branches, Erik 2026-11-21).

Every repo here is a temporary git repo; nothing runs the real suite, a real
build or cuda_green. Each test names the property it protects and the change
that breaks it.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import staging_gate as sg  # noqa: E402


# ---------------------------------------------------------------------------
# a temporary repo
# ---------------------------------------------------------------------------
def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _write(repo: Path, files: dict) -> None:
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text), encoding="utf-8")


def _commit(repo: Path, files: dict, msg: str) -> str:
    _write(repo, files)
    _git(repo, "add", "-f", *files)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _branch(repo: Path, name: str, files: dict, start="main") -> str:
    _git(repo, "switch", "-q", "-c", name, start)
    sha = _commit(repo, files, f"{name}: work")
    _git(repo, "switch", "-q", "main")
    return sha


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "commit.gpgsign", "false")
    _commit(r, {".gitignore": ".claude/\n",
                "f.txt": "one\ntwo\nthree\n", "g.txt": "g\n"}, "base")
    return r


def _first_parent_merges(repo, commit, base):
    out = _git(repo, "rev-list", "--first-parent", "--parents",
               f"{base}..{commit}").splitlines()
    return [line.split() for line in out][::-1]


def _snapshot(repo):
    """What the caller's checkout looks like: HEAD, branch, status, refs,
    worktrees."""
    return (_git(repo, "rev-parse", "HEAD"),
            _git(repo, "branch", "--show-current"),
            _git(repo, "status", "--porcelain"),
            _git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
            _git(repo, "worktree", "list", "--porcelain"))


# ---------------------------------------------------------------------------
# (a) merge
# ---------------------------------------------------------------------------
def test_inputs_merge_no_ff_in_order_and_the_caller_is_untouched(repo):
    """Protects: every input becomes its own two-parent merge commit, in the
    order given, even when a fast-forward was possible, and the caller's
    checkout (HEAD, branch, status, refs, worktrees) is unchanged.
    Breaks when: the merge fast-forwards, reorders or drops an input, or
    merges in the caller's working tree."""
    a = _branch(repo, "a", {"a.txt": "a\n"})
    b = _branch(repo, "b", {"b.txt": "b\n"})
    base = sg.Input("main", _git(repo, "rev-parse", "main"))
    before = _snapshot(repo)
    res = sg.merge_chain(repo, base, [sg.Input("a", a), sg.Input("b", b)],
                         "t")
    assert _snapshot(repo) == before
    merges = _first_parent_merges(repo, res.commit, base.sha)
    assert [m[2] for m in merges] == [a, b]          # second parents, in order
    assert all(len(m) == 3 for m in merges)          # each a 2-parent merge
    tree = _git(repo, "ls-tree", "--name-only", res.commit).split()
    assert {"a.txt", "b.txt", "f.txt"} <= set(tree)


def test_a_conflict_stops_names_files_and_both_sides_and_resolves_nothing(
        repo):
    """Protects: a conflict stops the gate with each conflicted file and the
    two sides that touched it (an earlier input, or the base), and leaves no
    branch, worktree or merge state behind.
    Breaks when: a conflict is resolved (a strategy option, taking a side),
    the wrong side is named, or a half-staged branch is left."""
    a = _branch(repo, "a", {"f.txt": "one\nTWO-a\nthree\n"})
    b = _branch(repo, "b", {"f.txt": "one\nTWO-b\nthree\n", "x.txt": "x\n"})
    before = _snapshot(repo)
    base = sg.Input("main", _git(repo, "rev-parse", "main"))
    with pytest.raises(sg.MergeConflict) as e:
        sg.merge_chain(repo, base, [sg.Input("a", a), sg.Input("b", b)], "t")
    assert e.value.conflict.input.ref == "b"
    assert e.value.conflict.files == {"f.txt": "input a"}
    assert e.value.code == sg.EXIT_CONFLICT
    assert _snapshot(repo) == before
    # against the base: main moved the same line after c forked
    c = _branch(repo, "c", {"f.txt": "one\nTWO-c\nthree\n"})
    _commit(repo, {"f.txt": "one\nTWO-main\nthree\n"}, "main moves")
    base = sg.Input("main", _git(repo, "rev-parse", "main"))
    with pytest.raises(sg.MergeConflict) as e:
        sg.merge_chain(repo, base, [sg.Input("c", c)], "t")
    assert e.value.conflict.files == {"f.txt": "base main"}


def test_a_staged_commit_reads_back_its_base_and_inputs(repo):
    """Protects: ``--at`` on another machine recovers exactly the base and the
    inputs (refs and commits, in order) the first machine merged.
    Breaks when: the merge message's trailers and their reader drift apart."""
    a = _branch(repo, "a", {"a.txt": "a\n"})
    b = _branch(repo, "b", {"b.txt": "b\n"})
    base = sg.Input("main", _git(repo, "rev-parse", "main"))
    ins = [sg.Input("a", a), sg.Input("b", b)]
    res = sg.merge_chain(repo, base, ins, "t")
    got_base, got = sg.read_staging(repo, res.commit)
    assert (vars(got_base), [vars(i) for i in got]) == \
        (vars(base), [vars(i) for i in ins])
    with pytest.raises(sg.GateError):
        sg.read_staging(repo, base.sha)      # not a staging commit


def test_dry_run_keeps_nothing(repo, tmp_path, capsys):
    """Protects: ``--dry-run`` only merges in memory and reports: no branch,
    no worktree, no ref is created, and its summary line names the result.
    Breaks when: the dry run creates the staging branch or worktree."""
    _branch(repo, "a", {"a.txt": "a\n"})
    before = _snapshot(repo)
    code = sg.main(["main", "a", "--base", "main", "--dry-run", "--repo",
                    str(repo),
                    "--run-dir", str(tmp_path / "run")])
    assert code == 0
    assert _snapshot(repo) == before
    out = capsys.readouterr().out
    summary = json.loads(out.split("STAGING_SUMMARY ", 1)[1].splitlines()[0])
    assert summary["exit"] == 0 and summary["stages"] == "dry-run"
    assert "main: is the base" in out


def test_staging_lives_in_its_own_worktree_and_only_it_is_pushed(repo,
                                                                 tmp_path):
    """Protects (a) + (e): a real merge stage creates ``staging/<name>`` in
    its OWN worktree; ``--push-staging`` pushes that branch and nothing else
    (origin's main does not move); main never moves; ``--remove`` deletes the
    worktree and the branch, local and remote.
    Breaks when: the gate stages in the caller's checkout, merges or pushes
    main, or leaves its throwaway branch behind."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    a = _branch(repo, "a", {"a.txt": "a\n"})
    main_before = _git(repo, "rev-parse", "main")
    head_before = _git(repo, "rev-parse", "HEAD")
    run_dir = tmp_path / "run"
    code = sg.main(["a", "--base", "main", "--name", "t1", "--until",
                    "merge", "--push-staging", "--repo", str(repo),
                    "--run-dir", str(run_dir)])
    assert code == 0
    wt = repo / ".claude" / "worktrees" / "staging-t1"
    staged = _git(wt, "rev-parse", "HEAD")
    assert _git(wt, "branch", "--show-current") == "staging/t1"
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "rev-parse", "main") == main_before
    assert _git(origin, "rev-parse", "main") == main_before
    assert _git(origin, "rev-parse", "staging/t1") == staged
    assert a in _git(repo, "rev-list", "--parents", "-n", "1", staged)
    log = (run_dir / "merge.log").read_text(encoding="utf-8").splitlines()
    assert log[-1] == f"_EXIT 0 {staged}"
    assert sg.main(["--name", "t1", "--remove", "--repo", str(repo),
                    "--run-dir", str(run_dir)]) == 0
    assert not wt.exists()
    assert _git(repo, "branch", "--list", "staging/t1") == ""
    assert _git(origin, "branch", "--list", "staging/t1") == ""


def test_the_tool_never_merges_or_pushes_anything_but_staging():
    """Protects (e): the gate's only git merges are ``merge-tree`` and
    ``merge-base`` (no ``git merge``), its only push is the staging branch,
    and the merge commands it prints merge each input ``--no-ff``, in order.
    Breaks when: a ``git merge`` or a push of another ref is added."""
    src = (ROOT / "tools" / "staging_gate.py").read_text(encoding="utf-8")
    git_calls = re.findall(r'\bgit\((?:[^()]|\([^()]*\))*\)', src) + \
        re.findall(r'\["git",[^\]]*\]', src)
    assert git_calls
    assert not [c for c in git_calls if re.search(r'"merge"\s*[,)\]]', c)]
    pushes = re.findall(r'"push",[^)]*\)', src)
    assert pushes and all("self.branch" in p or "br" in p for p in pushes)
    ins = [sg.Input("x-1", "1" * 40), sg.Input("y-2", "2" * 40)]
    cmds = sg.merge_commands(ins)
    merges = [c for c in cmds if c.startswith("git merge")]
    assert merges == ["git merge --no-ff x-1", "git merge --no-ff y-2"]


# ---------------------------------------------------------------------------
# (b) builds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("line", [
    r"C:\b\cpp\src\eos.cpp(12): error C2065: 'x': undeclared identifier",
    "eos.obj : error LNK2019: unresolved external symbol foo",
    "LINK : fatal error LNK1104: cannot open file 'breach_physics.pyd'",
    r"C:\b\cpp\src\cuda_x.cu(40): error: identifier \"y\" is undefined",
    "FAILED: CMakeFiles/breach_physics.dir/src/cuda_x.cu.obj",
    "ninja: build stopped: subcommand failed.",
    "CMake Error at CMakeLists.txt:10 (find_package):",
    "BUILD_EXIT=1",
    "CONFIGURE_EXIT=2",
    "C:\\x.vcxproj : error MSB3073: The command exited with code 1.",
], ids=["msvc", "link", "link_fatal", "nvcc", "ninja_failed", "ninja_stop",
        "cmake", "build_exit", "configure_exit", "msbuild"])
def test_a_build_log_error_line_is_caught(line):
    """Protects: a build whose script exits 0 is still red when its log holds
    an MSVC, linker, nvcc, ninja, CMake or MSBuild error, or a non-zero
    ``*_EXIT=`` line of the build scripts.
    Breaks when: an error pattern is dropped or loosened."""
    assert sg.build_log_errors(f"noise\n{line}\nmore\n")


@pytest.mark.parametrize("line", [
    r"C:\b\cpp\src\eos.cpp(3): warning C4819: the file contains a character",
    "-- Build files have been written to: C:/b/cpp/build",
    "BUILD_EXIT=0",
    "    0 Error(s)",
    "[12/40] Building CUDA object CMakeFiles/cuda_unit_errors.cu.obj",
], ids=["warning", "cmake_done", "build_exit_0", "msbuild_summary",
        "progress"])
def test_a_clean_build_line_is_not_an_error(line):
    """Protects: warnings and progress lines never fail a build.
    Breaks when: an error pattern is widened to match them."""
    assert not sg.build_log_errors(line)


def test_a_built_module_must_be_a_fresh_non_empty_image(tmp_path):
    """Protects: a build is judged by its module: missing, empty, zero-filled
    (what an out-of-memory crash leaves), not a PE image, or built from
    another commit are each a red; a real image at the staging commit is not.
    Breaks when: one of those checks is dropped."""
    pyd = tmp_path / "breach_physics.cp312-win_amd64.pyd"
    log = tmp_path / "build.log"
    log.write_text("ok\n", encoding="utf-8")
    assert sg.pyd_problem(None)
    for content in (b"", b"\0" * 4096, b"ELF\x7f..."):
        pyd.write_bytes(content)
        assert sg.pyd_problem(pyd), content[:8]
    pyd.write_bytes(b"MZ" + b"\x90" * 100)
    assert sg.pyd_problem(pyd) is None
    assert sg.judge_build(log, pyd, "c" * 40, "c" * 40, 0.0) == []
    assert sg.judge_build(log, pyd, "d" * 40, "c" * 40, 0.0)


def _p(pid, ppid, name, *cmd):
    return sg.Proc(pid, ppid, name, list(cmd))


def test_the_load_counts_build_trees_and_suites_once_each():
    """Protects the machine rule (at most two builds at once, a full suite
    counts as one): a build tree counts once however many compilers it runs,
    an idle reused MSBuild node does not count, a suite counts once with its
    xdist workers, cuda_green once with its pytest, a targeted pytest not at
    all, and the gate's own tree never.
    Breaks when: each compiler counts, idle nodes block forever, or the gate
    waits on itself."""
    procs = [
        _p(1, 900, "cmake.exe", "cmake", "--build", "cpp/build_cuda"),
        _p(2, 1, "ninja.exe", "ninja"),
        _p(3, 2, "cl.exe", "cl", "/c", "a.cpp"),
        _p(4, 2, "nvcc.exe", "nvcc", "b.cu"),
        _p(5, 4, "cicc.exe", "cicc"),
        _p(10, 901, "nvcc.exe", "nvcc", "c.cu"),
        _p(20, 4242, "MSBuild.exe", "MSBuild.exe", "/nodemode:1",
           "/nodeReuse:true"),
        _p(30, 4243, "MSBuild.exe", "MSBuild.exe", "/nodemode:1"),
        _p(31, 30, "CL.exe", "cl", "/c", "d.cpp"),
        _p(40, 902, "python.exe", "python", "-m", "pytest", "tests", "-q"),
        _p(41, 40, "python.exe", "python", "-c", "import execnet"),
        _p(50, 903, "python.exe", "python", "tools/cuda_green.py"),
        _p(51, 50, "python.exe", "python", "-m", "pytest", "tests", "-q",
           "-k", "cuda"),
        _p(60, 904, "python.exe", "python", "-m", "pytest",
           "tests/test_x.py", "-q"),
        _p(61, 905, "Code.exe", "code", "tools/cuda_green.py"),
        _p(70, 906, "python.exe", "python", "tools/staging_gate.py"),
        _p(71, 70, "cmake.exe", "cmake", "--build", "cpp/build"),
        _p(72, 70, "python.exe", "python", "-m", "pytest", "tests"),
    ]
    jobs = sg.load_jobs(procs, exclude={70, 71, 72})
    assert sorted(p.pid for p in jobs["builds"]) == [1, 10, 30]
    assert sorted(p.pid for p in jobs["suites"]) == [40, 50]


# ---------------------------------------------------------------------------
# (c) the two halves of the suite + logs
# ---------------------------------------------------------------------------
def _mini_suite(root: Path, select_line='["-k", "cuda"]') -> Path:
    _write(root, {
        "tools/cuda_green.py": f"""\
            import subprocess
            def main():
                subprocess.run(["python", "-m", "pytest", "tests", "-q",
                                *{select_line}])
            """,
        "tests/test_alpha.py": """\
            def test_plain():
                pass
            def test_card_thing_cuda():
                pass
            def test_fails():
                assert False
            class TestGroup:
                def test_inner_fails(self):
                    assert 1 == 2
            """,
        "tests/test_cuda_beta.py": """\
            import pytest
            @pytest.mark.parametrize("n", [1, 2])
            def test_one(n):
                assert n == 1
            """,
        "tests/test_class_p_level.py": """\
            def test_level():
                pass
            """,
        "tests/test_broken.py": "import no_such_module_anywhere\n",
        "pytest.ini": "[pytest]\n",
    })
    return root


def test_the_two_halves_are_the_whole_suite(tmp_path):
    """Protects: collected(card half: cuda_green's own -k + the class-P file)
    + collected(cpu half) == collected(all), with no test in both, so each
    test runs on exactly one machine.
    Breaks when: a half's selection drifts from cuda_green's (a gap) or the
    class-P file runs in both (an overlap)."""
    root = _mini_suite(tmp_path / "suite")
    sel = sg.cuda_green_select(root / "tools" / "cuda_green.py")
    got = sg.partition_gaps(root, sys.executable, sel)
    assert got["gap"] == [] and got["overlap"] == [] and got["extra"] == []
    assert got["counts"]["card"] > 0 and got["counts"]["cpu"] > 0
    assert any("test_level" in n for n in got["halves"]["cuda"])


def test_drifted_halves_are_caught(tmp_path, monkeypatch):
    """Protects: the partition check sees a gap and an overlap.
    Breaks when: it stops comparing the halves with the whole."""
    root = _mini_suite(tmp_path / "suite")
    real = sg.half_args

    def gap(half, select):
        if half == "cpu":
            return ["tests", "-k", f"not ({select}) and not plain",
                    f"--ignore={sg.CLASS_P_TEST}"]
        return real(half, select)

    monkeypatch.setattr(sg, "half_args", gap)
    got = sg.partition_gaps(root, sys.executable, "cuda")
    assert got["gap"] == ["tests/test_alpha.py::test_plain"]

    def overlap(half, select):
        return ["tests"] if half == "cpu" else real(half, select)

    monkeypatch.setattr(sg, "half_args", overlap)
    got = sg.partition_gaps(root, sys.executable, "cuda")
    assert "tests/test_class_p_level.py::test_level" in got["overlap"]


def test_the_card_half_is_read_from_the_card_gates_own_source(tmp_path):
    """Protects: the card half is cuda_green's own ``-k`` expression, read
    from its source (a literal, or a module constant), never re-typed.
    Breaks when: the reader returns a fixed string or misses a constant."""
    real = sg.cuda_green_select(ROOT / "tools" / "cuda_green.py")
    src = (ROOT / "tools" / "cuda_green.py").read_text(encoding="utf-8")
    assert real and f'"{real}"' in src
    f = tmp_path / "cuda_green.py"
    f.write_text('SELECT = "gpu or card"\ncmd = ["pytest", "-k", SELECT]\n',
                 encoding="utf-8")
    assert sg.cuda_green_select(f) == "gpu or card"
    f.write_text('cmd = ["pytest", "tests"]\n', encoding="utf-8")
    with pytest.raises(sg.GateError):
        sg.cuda_green_select(f)


def test_the_junit_reds_are_the_ids_pytest_names(tmp_path):
    """Protects: the reds read from a junit file are exactly the node ids
    pytest itself reports as failed or errored (class tests, parametrized
    cases, a collection error included).
    Breaks when: the junit reader builds ids pytest would not recognise."""
    root = _mini_suite(tmp_path / "suite")
    xml = tmp_path / "r.xml"
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-rfE",
                        *sg.PYTEST_COMMON, f"--junitxml={xml}"], cwd=root,
                       capture_output=True, text=True)
    named = set()
    for ln in r.stdout.splitlines():
        m = re.match(r"^(FAILED|ERROR) (\S+)", ln)
        if m:
            named.add(m.group(2))
    assert len(named) >= 4
    assert set(sg.junit_reds(xml)) == named


def test_every_stage_log_ends_with_its_exit_and_commit(tmp_path):
    """Protects: each stage log's last line is ``_EXIT <code> <commit>``, so
    two machines compare their halves on one commit.
    Breaks when: a log loses the line, or the code or the commit in it."""
    c = "a" * 40
    log = tmp_path / "s.log"
    assert sg.run_logged([sys.executable, "-c", "print('hi')"], log,
                         tmp_path, c) == 0
    assert log.read_text(encoding="utf-8").splitlines()[-1] == f"_EXIT 0 {c}"
    rc = sg.run_logged([[sys.executable, "-c", "raise SystemExit(3)"],
                        [sys.executable, "-c", "print('never')"]], log,
                       tmp_path, c)
    text = log.read_text(encoding="utf-8")
    assert rc == 3 and "never" not in text.split("#")[-1]
    assert text.splitlines()[-1] == f"_EXIT 3 {c}"
    sg.write_log(log, ["x"], 2, c)
    assert log.read_text(encoding="utf-8").splitlines()[-1] == f"_EXIT 2 {c}"


def test_each_gate_runs_in_exactly_one_half():
    """Protects: every gate stage (cuda_green, class_p, suite, agents)
    belongs to exactly one of ``--stages cpu`` / ``cuda`` and ``all`` holds
    every stage, so the two machines together run each gate once.
    Breaks when: a gate is dropped from both halves or put in both."""
    for g in sg.GATES:
        assert (g in sg.HALVES["cpu"]) != (g in sg.HALVES["cuda"]), g
    assert sg.HALVES["all"] == set(sg.STAGES)
    assert sg.plan("cpu", "build", None)[0] == "build_cpu"
    assert sg.plan("all", None, "merge") == ["merge"]


# ---------------------------------------------------------------------------
# (d) sorting the reds
# ---------------------------------------------------------------------------
@pytest.fixture
def sorted_repo(repo):
    _commit(repo, {
        ".claude/rules/alpha.md": '---\npaths:\n  - "src/alpha*.py"\n---\n',
        ".claude/rules/beta.md": '---\npaths:\n  - "src/beta*.py"\n---\n',
        "src/alpha.py": "A = 1\n", "src/beta.py": "B = 1\n",
        "src/other.py": "O = 1\n",
        "tests/test_alpha.py": "import alpha\ndef test_t():\n    pass\n",
        "tests/test_beta.py": "import beta\ndef test_t():\n    pass\n",
        "tests/test_other.py": "import other\ndef test_t():\n    pass\n",
    }, "the modules")
    # .claude/ is ignored in the fixture repo; the rules are tracked here
    a = _branch(repo, "a", {"src/alpha.py": "A = 2\n"})
    b = _branch(repo, "b", {"src/beta.py": "B = 2\n"})
    base = sg.Input("main", _git(repo, "rev-parse", "main"))
    ins = [sg.Input("a", a), sg.Input("b", b)]
    res = sg.merge_chain(repo, base, ins, "t")
    wt = repo.parent / "wt"
    _git(repo, "worktree", "add", "-q", "--detach", str(wt), res.commit)
    return wt, base, ins


_A = "tests/test_alpha.py::test_t"
_B = "tests/test_beta.py::test_t"
_O = "tests/test_other.py::test_t"


def test_a_red_is_sorted_by_the_region_and_input_that_reach_it(sorted_repo):
    """Protects: a staging red's AREA is the rule region of the changed file
    that reaches it and its OWNER the input whose diff holds that file; a red
    no change reaches is ``(unreached)``; without a base result every red is
    "unknown base".
    Breaks when: the owner is taken from the wrong diff, the region from the
    wrong rule, or the reach is not affected_tests'."""
    wt, base, ins = sorted_repo
    s = sg.sort_reds(wt, [_A, _B, _O], ins, base.sha, [], None)
    got = {n: (area, bk) for n, area, _, bk in s.staging}
    assert got[_A] == ({"alpha.md": ["a"]}, False)
    assert got[_B] == ({"beta.md": ["b"]}, False)
    assert got[_O] == ({"(unreached)": []}, False)
    assert "(unknown base)" in "\n".join(sg.format_sorted(s))


def test_known_and_preexisting_reds_never_hide_a_new_one(sorted_repo):
    """Protects: a red in this machine's table is KNOWN and one red on the
    base result is PRE-EXISTING, neither a staging red; a known entry matches
    only its own id and parametrized cases, so every other red stays a
    staging red; a known entry that ran green is reported stale.
    Breaks when: a known entry matches by prefix or pattern, or the base
    result swallows reds not in it."""
    wt, base, ins = sorted_repo
    known = [{"test": _O, "why": "this PC"},
             {"test": "tests/test_alpha.py::test_gone", "why": "fixed"}]
    reds = [_A, _B, _O, _O + "[2]", "tests/test_other.py::test_t2"]
    s = sg.sort_reds(wt, reds, ins, base.sha, known, {_B},
                     collected={_A, _B, _O, "tests/test_alpha.py::test_gone"})
    assert sorted(n for n, _ in s.known) == [_O, _O + "[2]"]
    assert s.preexisting == [_B]
    assert sorted(n for n, *_ in s.staging) == \
        [_A, "tests/test_other.py::test_t2"]
    assert all(bk for *_, bk in s.staging)
    assert s.stale_known == ["tests/test_alpha.py::test_gone"]


def test_a_gate_that_fails_without_naming_a_test_is_a_red(sorted_repo):
    """Protects: a gate stage that exits red with no failing test in its
    junit (cuda_green refused, pytest crashed), or that never ran, is itself
    a staging red, so a missing result never reads as green.
    Breaks when: the sort trusts an empty junit file or a gate not run."""
    wt, base, ins = sorted_repo
    reds = sg.stage_reds({"cuda_green": 1, "suite": 0, "agents": None},
                         {"cuda_green": [], "suite": []})
    assert sorted(r.split("]")[0] for r in reds) == ["[agents",
                                                    "[cuda_green"]
    assert sg.stage_reds({"suite": 1}, {"suite": [_A]}) == []
    s = sg.sort_reds(wt, reds, ins, base.sha, [], None)
    assert s.staging[0][1] == {"(stage failure)": []}


def test_a_cached_base_result_reads_as_node_ids(tmp_path):
    """Protects: a base result written by a run (``reds.txt``, headed by its
    commit) or pasted from pytest's summary reads back as the same node ids.
    Breaks when: the reader keeps the ``FAILED`` prefix or the message."""
    f = tmp_path / "reds.txt"
    f.write_text(f"# commit {'b' * 40}\n{_A}\nFAILED {_B} - assert 1 == 2\n"
                 f"ERROR tests/test_x.py\n", encoding="utf-8")
    reds, commit = sg.read_result(f)
    assert reds == {_A, _B, "tests/test_x.py"} and commit == "b" * 40


# ---------------------------------------------------------------------------
# the machine table
# ---------------------------------------------------------------------------
def test_the_machine_table_names_real_builds_and_real_tests():
    """Protects: every machine row names a build that exists (a cpp/ script,
    or "cmake"), and every known red names a test function that exists, so
    the table cannot rot into hiding nothing or something else.
    Breaks when: a build script or a known test is renamed without the
    table."""
    import tomllib
    table = tomllib.loads(sg.MACHINES.read_text(encoding="utf-8"))
    assert table
    for host, row in table.items():
        for key in ("cpu_build", "cuda_build"):
            assert row[key] == "cmake" or (ROOT / row[key]).is_file(), \
                (host, key)
        for k in row.get("known_red", []):
            path, _, func = k["test"].partition("::")
            assert (ROOT / path).is_file(), k["test"]
            assert re.search(rf"def {re.escape(func)}\b", (ROOT / path)
                             .read_text(encoding="utf-8")), k["test"]
            assert k.get("why"), k["test"]
    assert sg.machine(host="no-such-host")["known_red"] == []
