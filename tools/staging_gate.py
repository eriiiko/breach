"""tools/staging_gate.py -- THE staging gate: one full suite and one
cuda_green for several ready branches at once (Erik, 2026-11-21).

A full suite costs ~1 h 40 and ``tools/cuda_green.py`` ~90 min, so ready
branches share ONE run on ONE staging commit::

    py tools/staging_gate.py 29-p89 217-eos-checkerboard          # everything
    py tools/staging_gate.py 29-p89 217-eos --dry-run             # merge only
    py tools/staging_gate.py 29-p89 217-eos --stages cuda --push-staging
    py tools/staging_gate.py --at staging/29-217-20261121-1530 --stages cpu
    py tools/staging_gate.py --name 29-217-20261121-1530 --from suite
    py tools/staging_gate.py --name 29-217-20261121-1530 --remove

(``py`` = this machine's breach env, ``docs/dev_setup.md``.)

The stages, in order (``--stages`` picks a half, ``--from`` / ``--until``
cut the list; every stage writes ``<stage>.log`` into the run folder
``.claude/staging_runs/<name>/`` of the main checkout, ending with
``_EXIT <code> <commit>``):

``merge``       the inputs, in the order given, ``--no-ff`` onto ``--base``
                (default ``origin/main``) as a throwaway branch
                ``staging/<name>`` checked out in its OWN worktree
                (``.claude/worktrees/staging-<name>``); never the caller's
                checkout. A conflict STOPS the gate, naming each file and the
                two inputs; nothing is ever resolved. With ``--at`` there is
                no merge: an existing staging commit (pushed by the other
                machine) is fetched and checked out, its inputs read back
                from the merge commits' ``Staging-*`` trailers.
``build_cpu``   this machine's CPU build (``tools/staging_machines.toml``).
``build_cuda``  this machine's CUDA build. A build script exits 0 on a
                compile failure, so each build is judged by its log (compiler,
                linker, nvcc, ninja, CMake error lines), by its ``.pyd``
                (present, non-empty, a PE image, not zero-filled) and by the
                module's ``build_id()`` == the staging commit. Before each
                build (and the suite) the gate waits while ``--max-builds``
                (2) builds or full suites already run on this PC.
``partition``   asserts the two halves are the suite: collected(card half) +
                collected(cpu half) == collected(all), no test in both.
``cuda_green``  ``tools/cuda_green.py`` in the staging worktree, as is: it
                writes the one ``.cuda_green`` stamp there on green.
``class_p``     ``tests/test_class_p_level.py`` (reads that stamp).
``suite``       ``pytest tests -q -k "not (<cuda_green's -k>)"`` minus the
                class-P file: with the two stages above, every test of
                ``tests/`` exactly once (xdist ``-n N --dist loadgroup`` when
                installed, the conftest's device group).
``agents``      ``pytest agents/tests -q``.
``sort``        sorts the reds: KNOWN on this machine (the table), present on
                the base too (``--base-result``) or "unknown base", and the
                rest by AREA (the rule region of each changed file that
                reaches the red test, ``.claude/rules/*.md`` ``paths:``) and
                OWNER (the input whose diff from its merge base holds that
                file).

``--stages``: ``all`` (default), ``cpu`` (build_cpu, suite, agents) or
``cuda`` (both builds, cuda_green, class_p). Two machines can split one
gate: one merges and ``--push-staging``; the other runs ``--at`` the pushed
branch; merge to main only when both print ``exit 0`` for the SAME commit in
their ``STAGING_SUMMARY`` line.

The gate never merges to main and never pushes anything but the staging
branch (``--push-staging``). On green it PRINTS the commands that merge each
input to main ``--no-ff``, in order.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import affected_tests  # noqa: E402

MACHINES = ROOT / "tools" / "staging_machines.toml"
CLASS_P_TEST = "tests/test_class_p_level.py"
AGENTS_TESTS = "agents/tests"
CPU_PYD_DIR = "cpp/build/Release"
CUDA_PYD_DIR = "cpp/build_cuda"

STAGES = ("merge", "build_cpu", "build_cuda", "partition", "cuda_green",
          "class_p", "suite", "agents", "sort")
HALVES = {
    "all": set(STAGES),
    "cpu": {"merge", "build_cpu", "partition", "suite", "agents", "sort"},
    "cuda": {"merge", "build_cpu", "build_cuda", "partition", "cuda_green",
             "class_p", "sort"},
}
ALIASES = {"build": "build_cpu", "gates": "partition"}
#: stages whose pytest drives the card or a window: refused while another
#: suite or cuda_green runs on this PC.
GPU_STAGES = {"cuda_green", "class_p", "suite"}
#: stages that count as one of this PC's builds (the machine rule: at most
#: two builds at once, a full suite counts as one).
BUILD_STAGES = {"build_cpu", "build_cuda", "cuda_green", "suite"}
#: the stages whose failure stops the gate (nothing after them can be
#: trusted); a red gate stage does not stop it: its reds are sorted.
STOPPERS = {"build_cpu", "build_cuda", "partition"}
#: the stages whose reds are sorted.
GATES = ("cuda_green", "class_p", "suite", "agents")
#: every pytest stage of the gate: a module that fails to import is one red,
#: never the end of a 1 h 40 run; the junit file carries each case's file.
PYTEST_COMMON = ("-p", "no:cacheprovider", "--continue-on-collection-errors",
                 "-o", "junit_family=xunit1")

EXIT_GREEN, EXIT_RED, EXIT_CONFLICT, EXIT_BUILD, EXIT_REFUSED = 0, 1, 2, 3, 4


class GateError(RuntimeError):
    """A stage cannot go on; ``code`` is the gate's exit code."""

    def __init__(self, msg: str, code: int = EXIT_REFUSED):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------
def git(repo, *args, check=True, timeout=600) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed in {repo}:\n"
                        f"{r.stderr.strip()}")
    return r.stdout.strip()


def common_root(repo) -> Path:
    """The main checkout of ``repo`` (the parent of the common git dir)."""
    d = Path(git(repo, "rev-parse", "--path-format=absolute",
                 "--git-common-dir"))
    return d.parent


def resolve(repo, ref: str) -> str:
    """``ref`` (a branch, a remote branch, a hash) -> its commit; a bare
    branch name not known locally is tried as ``origin/<ref>``."""
    for cand in (ref, f"origin/{ref}"):
        sha = git(repo, "rev-parse", "--verify", "--quiet",
                  f"{cand}^{{commit}}", check=False)
        if sha:
            return sha
    raise GateError(f"{ref!r} names no commit (fetched? `git fetch origin`)")


def is_ancestor(repo, a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "merge-base",
                           "--is-ancestor", a, b]).returncode == 0


def changed_by(repo, base_sha: str, sha: str) -> set:
    """What an input changed: ``git diff --name-only <merge base>..<input>``."""
    mb = git(repo, "merge-base", base_sha, sha)
    out = git(repo, "diff", "--name-only", f"{mb}..{sha}")
    return {p for p in out.splitlines() if p}


# ---------------------------------------------------------------------------
# merge (in memory: git merge-tree + commit-tree; no checkout is touched)
# ---------------------------------------------------------------------------
@dataclass
class Input:
    ref: str
    sha: str


@dataclass
class MergeResult:
    commit: str
    merged: list            # [Input] merged, in order
    skipped: list           # [(Input, why)] already in the chain
    log: list = field(default_factory=list)


@dataclass
class Conflict:
    input: Input
    #: path -> what the other side is ("base <ref>" or "input <ref>")
    files: dict


class MergeConflict(GateError):
    def __init__(self, conflict: Conflict, log: list):
        self.conflict = conflict
        self.log = log
        lines = [f"CONFLICT merging {conflict.input.ref} "
                 f"({conflict.input.sha[:12]}); nothing was resolved:"]
        for path, other in sorted(conflict.files.items()):
            lines.append(f"  {path}: {conflict.input.ref} vs {other}")
        super().__init__("\n".join(lines), EXIT_CONFLICT)


def trailer_message(name: str, base: Input, k: int, inp: Input) -> str:
    return (f"staging: merge {inp.ref} ({inp.sha[:12]}) into staging/{name}\n\n"
            f"Throwaway staging commit of tools/staging_gate.py; never merged "
            f"to main.\n\n"
            f"Staging-Base: {base.ref} {base.sha}\n"
            f"Staging-Input: {k} {inp.ref} {inp.sha}\n")


def merge_chain(repo, base: Input, inputs: list, name: str) -> MergeResult:
    """Merge ``inputs`` in order, ``--no-ff``, onto ``base`` as new commits
    (``git merge-tree --write-tree`` + ``git commit-tree``: the same ort merge
    ``git merge`` runs, with no working tree). A conflict raises
    :class:`MergeConflict` naming each file and both sides."""
    chain = base.sha
    merged, skipped, log = [], [], [f"base {base.ref} {base.sha}"]
    touched = []      # [(Input, set of paths)] of what is already merged
    for inp in inputs:
        if is_ancestor(repo, inp.sha, chain):
            why = "is the base" if inp.sha == base.sha else "already contained"
            skipped.append((inp, why))
            log.append(f"skip  {inp.ref} {inp.sha[:12]}: {why}")
            continue
        r = subprocess.run(["git", "-C", str(repo), "merge-tree",
                            "--write-tree", "--name-only", "--no-messages",
                            chain, inp.sha], capture_output=True, text=True)
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        if r.returncode == 1:
            files = {}
            for path in dict.fromkeys(lines[1:]):
                others = [o.ref for o, paths in touched if path in paths]
                files[path] = (" + ".join(f"input {o}" for o in others)
                               if others else f"base {base.ref}")
            log.append(f"CONFLICT {inp.ref} {inp.sha[:12]}: "
                       f"{', '.join(sorted(files))}")
            raise MergeConflict(Conflict(inp, files), log)
        if r.returncode != 0 or not lines:
            raise GateError(f"git merge-tree failed for {inp.ref}:\n"
                            f"{r.stderr.strip()}")
        tree = lines[0]
        msg = trailer_message(name, base, len(merged) + 1, inp)
        chain = subprocess.run(
            ["git", "-C", str(repo), "commit-tree", tree, "-p", chain,
             "-p", inp.sha, "-F", "-"], input=msg, capture_output=True,
            text=True, check=True).stdout.strip()
        merged.append(inp)
        touched.append((inp, changed_by(repo, base.sha, inp.sha)))
        log.append(f"merge {inp.ref} {inp.sha[:12]} -> {chain[:12]}")
    return MergeResult(chain, merged, skipped, log)


_TRAILER = re.compile(r"^Staging-(Base|Input): (.+)$", re.M)


def read_staging(repo, commit: str) -> tuple:
    """``(base, [inputs])`` of a staging commit, read back from its merge
    commits' trailers along the first-parent chain (``--at``)."""
    out = git(repo, "log", "--first-parent", "-n", "200",
              "--format=%H%x1f%B%x1e", commit)
    inputs, base = [], None
    for rec in out.split("\x1e"):
        if not rec.strip():
            continue
        _, body = rec.strip().split("\x1f", 1)
        found = dict()
        for kind, val in _TRAILER.findall(body):
            found[kind] = val.split()
        if "Input" not in found:
            break
        _, ref, sha = found["Input"]
        inputs.append(Input(ref, sha))
        base = Input(*found["Base"])
    if base is None:
        raise GateError(f"{commit[:12]} is not a staging commit (no "
                        f"Staging-Input trailer on its first-parent chain)")
    return base, inputs[::-1]


def derive_name(inputs: list, now: datetime | None = None) -> str:
    """``<issue numbers or short names>-<YYYYMMDD-HHMM>``."""
    parts = []
    for inp in inputs:
        ref = inp.ref.split("/")[-1]
        m = re.match(r"^(\d+)-", ref)
        tag = m.group(1) if m else re.sub(r"[^\w-]", "", ref)[:16]
        if tag and tag not in parts:
            parts.append(tag)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    return "-".join(parts[:4] + [stamp])


def merge_commands(inputs: list) -> list:
    """What merges the inputs to main, in order, from the main checkout --
    printed, never run."""
    return (["git switch main && git pull --ff-only"]
            + [f"git merge --no-ff {inp.ref}" for inp in inputs]
            + ["git push"])


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------
def exit_line(code: int, commit: str) -> str:
    return f"_EXIT {code} {commit}"


def write_log(path: Path, lines: list, code: int, commit: str) -> None:
    path.write_text("\n".join([*lines, exit_line(code, commit)]) + "\n",
                    encoding="utf-8")


def run_logged(cmds, log: Path, cwd, commit: str, env=None) -> int:
    """Run one command (a list) or several (a list of lists, stopping at the
    first that fails) with their output into ``log``; the log ends with
    ``_EXIT <code> <commit>``."""
    if cmds and not isinstance(cmds[0], (list, tuple)):
        cmds = [cmds]
    rc = 0
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        for cmd in cmds:
            fh.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  cwd {cwd}\n"
                     f"# {subprocess.list2cmdline([str(c) for c in cmd])}\n")
            fh.flush()
            rc = subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                                stdout=fh, stderr=subprocess.STDOUT,
                                env=env).returncode
            if rc != 0:
                break
        fh.write(f"\n{exit_line(rc, commit)}\n")
    return rc


def append_exit(log: Path, code: int, commit: str) -> None:
    """Re-close a log whose verdict changed after the command ran (a build
    script that exited 0 on a compile failure)."""
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"{exit_line(code, commit)}\n")


# ---------------------------------------------------------------------------
# builds
# ---------------------------------------------------------------------------
_BUILD_ERROR = re.compile(
    r"\berror C\d{4}\b|\berror LNK\d{4}\b|\bfatal error\b|\berror MSB\d{4}\b"
    r"|: error\b|^CMake Error|^FAILED: |ninja: build stopped"
    r"|\b(?:CONFIGURE|BUILD)_EXIT=[1-9]", re.M)


def build_log_errors(text: str) -> list:
    """The error lines of a build log (MSVC, linker, nvcc, MSBuild, ninja,
    CMake, and the build scripts' own ``*_EXIT=`` lines)."""
    return [ln.strip() for ln in text.splitlines() if _BUILD_ERROR.search(ln)]


def pyd_problem(path: Path | None) -> str | None:
    """Why a built module is unusable, or None: missing, empty, zero-filled
    (what an out-of-memory crash leaves), or not a PE image."""
    if path is None or not path.is_file():
        return "no .pyd was produced"
    data = path.read_bytes()
    if not data:
        return f"{path.name} is empty"
    if not data.strip(b"\0"):
        return f"{path.name} is zero-filled"
    if not data.startswith(b"MZ"):
        return f"{path.name} is not a PE image"
    return None


def find_pyd(d: Path) -> Path | None:
    hits = sorted(d.glob("breach_physics*.pyd")) + sorted(
        d.glob("breach_physics*.so"))
    return hits[0] if hits else None


def cmake_build(wt: Path, python: str) -> list:
    """dev_setup Step 6: configure with this env's Python and pybind11, then
    build Release (two commands)."""
    exe = Path(python).parent / "Scripts" / "cmake.exe"
    cmake = str(exe) if exe.is_file() else "cmake"
    pyb = subprocess.run([python, "-c", "import pybind11; print("
                          "pybind11.get_cmake_dir())"], capture_output=True,
                         text=True).stdout.strip()
    return [[cmake, "-S", "cpp", "-B", "cpp/build",
             f"-DPython_EXECUTABLE={python}", f"-DPYTHON_EXECUTABLE={python}",
             f"-Dpybind11_DIR={pyb}"],
            [cmake, "--build", "cpp/build", "--config", "Release"]]


def module_build_id(wt: Path, python: str, kind: str) -> str:
    """``breach_physics.build_id()`` of the worktree's CPU or CUDA build."""
    if kind == "cpu":
        script = (f"import sys; sys.path.insert(0, {str(wt / CPU_PYD_DIR)!r})\n"
                  f"import breach_physics as bp\n"
                  f"print(getattr(bp, 'build_id', lambda: 'none')())\n")
    else:   # the CUDA module only imports through the harness's DLL setup
        script = ("import sys; sys.path[:0] = ['tests', 'tools']\n"
                  "import cuda_green; print(cuda_green.module_build_id())\n")
    r = subprocess.run([python, "-c", script], cwd=str(wt),
                       capture_output=True, text=True, timeout=300)
    lines = r.stdout.strip().splitlines()
    return lines[-1] if r.returncode == 0 and lines else \
        f"<import failed: {r.stderr.strip()[-300:]}>"


def judge_build(log: Path, pyd: Path | None, build_id: str, commit: str,
                started: float) -> list:
    """Every reason a build that its script called done is not."""
    problems = build_log_errors(log.read_text(encoding="utf-8",
                                              errors="replace"))[:20]
    bad = pyd_problem(pyd)
    if bad:
        problems.append(bad)
    elif build_id != commit:
        problems.append(f"stale module: build_id {build_id!r} is not the "
                        f"staging commit {commit}")
    if pyd is not None and pyd.is_file() and pyd.stat().st_mtime < started \
            and build_id != commit:
        problems.append(f"{pyd.name} is older than this build")
    return problems


# ---------------------------------------------------------------------------
# this PC's load: other builds, suites and cuda_green runs
# ---------------------------------------------------------------------------
BUILD_TOOLS = {"cl", "nvcc", "cicc", "ptxas", "ninja", "cmake", "msbuild",
               "link", "cudafe++", "fatbinary"}


@dataclass
class Proc:
    pid: int
    ppid: int
    name: str
    cmdline: list


def _stem(name: str) -> str:
    return name.lower().removesuffix(".exe")


def is_suite_run(name: str, cmdline: list) -> bool:
    """A Python process running a full-suite-style pytest (a whole ``tests``
    directory, with or without ``-k``) or ``tools/cuda_green.py``."""
    stem = _stem(name)
    if not (stem.startswith("python") or stem in ("pytest", "py.test")):
        return False
    if any(a.replace("\\", "/").endswith("cuda_green.py") for a in cmdline):
        return True
    joined = " ".join(cmdline)
    if stem.startswith("python") and not re.search(r"-m\s+pytest\b", joined):
        return False
    return any(not a.startswith("-") and
               Path(a.replace("\\", "/").rstrip("/")).name == "tests"
               for a in cmdline[1:])


def load_jobs(procs: list, exclude: set) -> dict:
    """``{"builds": [Proc], "suites": [Proc]}``: the ROOTS of this PC's build
    trees (a build tool no ancestor of which is one; an MSBuild node kept
    alive for reuse counts only while it has a compiler under it) and its
    suite runs (no ancestor of which is one), leaving out ``exclude`` (this
    gate's own process tree)."""
    by_pid = {p.pid: p for p in procs}

    def ancestors(p):
        seen, cur = set(), by_pid.get(p.ppid)
        while cur is not None and cur.pid not in seen:
            seen.add(cur.pid)
            yield cur
            cur = by_pid.get(cur.ppid)

    def tool(p):
        return _stem(p.name) in BUILD_TOOLS

    builds, suites = [], []
    for p in procs:
        if p.pid in exclude:
            continue
        if tool(p):
            if any(tool(a) for a in ancestors(p)):
                continue
            idle_node = _stem(p.name) == "msbuild" and "nodemode" in \
                " ".join(p.cmdline).lower()
            if idle_node and not any(
                    q is not p and tool(q) and p in ancestors(q)
                    for q in procs):
                continue    # a reused MSBuild node waiting after a build
            builds.append(p)
        elif is_suite_run(p.name, p.cmdline):
            if not any(is_suite_run(a.name, a.cmdline) for a in ancestors(p)):
                suites.append(p)
    return {"builds": builds, "suites": suites}


def list_procs() -> tuple:
    """``([Proc], own pids)``: every process this user can read, and this
    gate's own tree."""
    import psutil
    procs = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        info = p.info
        procs.append(Proc(info["pid"], info["ppid"] or 0, info["name"] or "",
                          list(info["cmdline"] or [])))
    me = psutil.Process(os.getpid())
    own = {me.pid} | {c.pid for c in me.children(recursive=True)}
    return procs, own


def describe(p: Proc) -> str:
    return f"pid {p.pid} {p.name}: {' '.join(p.cmdline)[:160]}"


def wait_for_slot(max_builds: int, poll: float, force: bool, say) -> None:
    """Wait while ``max_builds`` builds or suites already run on this PC."""
    if force:
        return
    while True:
        jobs = load_jobs(*list_procs())
        running = jobs["builds"] + jobs["suites"]
        if len(running) < max_builds:
            return
        say(f"waiting: {len(running)} builds / suites running on this PC "
            f"(at most {max_builds}); next look in {poll:.0f} s")
        for p in running:
            say(f"    {describe(p)}")
        time.sleep(poll)


# ---------------------------------------------------------------------------
# the two halves of the suite
# ---------------------------------------------------------------------------
def cuda_green_select(path: Path) -> str:
    """The ``-k`` expression ``tools/cuda_green.py`` runs, read from its own
    source (the staging tree's copy), so the card half is never re-typed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value,
                                                        ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names[t.id] = node.value.value
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        elts = node.elts
        for a, b in zip(elts, elts[1:]):
            if isinstance(a, ast.Constant) and a.value == "-k":
                if isinstance(b, ast.Constant) and isinstance(b.value, str):
                    return b.value
                if isinstance(b, ast.Name) and isinstance(names.get(b.id),
                                                          str):
                    return names[b.id]
    raise GateError(f"no `-k <expr>` in {path}: the card half is unknown")


def half_args(half: str, select: str) -> list:
    """The pytest arguments of each part of the suite, as the gate runs them:
    ``card`` = cuda_green's selection, ``class_p`` = the class-P file,
    ``cpu`` = everything else in ``tests/``, ``all`` = ``tests/``."""
    return {
        "all": ["tests"],
        "card": ["tests", "-k", select],
        "class_p": [CLASS_P_TEST],
        "cpu": ["tests", "-k", f"not ({select})", f"--ignore={CLASS_P_TEST}"],
    }[half]


def collect(wt: Path, python: str, args: list) -> tuple:
    """``(node ids, output)`` pytest collects for ``args`` in ``wt``."""
    r = subprocess.run([python, "-m", "pytest", *args, "--collect-only", "-q",
                        "-p", "no:cacheprovider",
                        "--continue-on-collection-errors"],
                       cwd=str(wt), capture_output=True, text=True)
    ids = [ln.strip() for ln in r.stdout.splitlines() if "::" in ln
           and not ln.startswith(("ERROR", "FAILED", " "))]
    return ids, r.stdout[-3000:] + r.stderr[-2000:]


def partition_gaps(wt: Path, python: str, select: str) -> dict:
    """``{"gap": [...], "overlap": [...], "counts": {...}}``: the tests of
    ``tests/`` no half runs, and those two halves both run."""
    got = {h: collect(wt, python, half_args(h, select))[0]
           for h in ("all", "card", "class_p", "cpu")}
    card = set(got["card"]) | set(got["class_p"])
    cpu = set(got["cpu"])
    return {"gap": sorted(set(got["all"]) - card - cpu),
            "overlap": sorted(card & cpu),
            "extra": sorted((card | cpu) - set(got["all"])),
            "counts": {h: len(v) for h, v in got.items()},
            "halves": {"cpu": sorted(cpu), "cuda": sorted(card)}}


def xdist_args(python: str, workers) -> list:
    if workers == 0:
        return []
    has = subprocess.run([python, "-c", "import xdist"],
                         capture_output=True).returncode == 0
    if not has:
        return []
    return ["-n", str(workers or 4), "--dist", "loadgroup"]


# ---------------------------------------------------------------------------
# reds
# ---------------------------------------------------------------------------
def junit_reds(path: Path) -> list:
    """The failing and erroring node ids of a pytest junit file
    (``junit_family=xunit1``, which carries each case's file)."""
    if not path.is_file():
        return []
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        f = (case.get("file") or "").replace("\\", "/")
        cls, name = case.get("classname") or "", case.get("name") or ""
        if not cls:
            # a collection error: name is the dotted module
            out.append(f or name.replace(".", "/") + ".py")
            continue
        if not f:
            out.append(f"{cls}::{name}")
            continue
        mod = f[:-3].replace("/", ".") if f.endswith(".py") else f
        rest = cls[len(mod):].lstrip(".") if cls.startswith(mod) else ""
        parts = [f, *(rest.split(".") if rest else []), name]
        out.append("::".join(x for x in parts if x))
    return sorted(dict.fromkeys(out))


def read_result(path: Path) -> tuple:
    """``(set of red node ids, commit or None)`` from a cached result: a junit
    file, or a text file of node ids (``FAILED <id> - ...`` and ``ERROR <id>``
    lines too), optionally headed ``# commit <sha>``."""
    if path.suffix == ".xml":
        return set(junit_reds(path)), None
    reds, commit = set(), None
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        m = re.match(r"^# commit ([0-9a-f]{7,40})", ln)
        if m:
            commit = m.group(1)
        if not ln or ln.startswith("#"):
            continue
        ln = re.sub(r"^(FAILED|ERROR)\s+", "", ln).split(" - ")[0].strip()
        if "::" in ln or ln.endswith(".py"):
            reds.add(ln)
    return reds, commit


def known_match(nodeid: str, known: list) -> dict | None:
    for k in known:
        t = k["test"]
        if nodeid == t or nodeid.startswith(t + "["):
            return k
    return None


def load_rules(wt: Path) -> list:
    """``[(rule file name, [compiled globs])]`` of ``.claude/rules`` in
    ``wt``, through the path-rules hook's own parser and matcher."""
    spec = importlib.util.spec_from_file_location(
        "path_rules_hook", ROOT / ".claude" / "hooks" / "path_rules.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    out = []
    for f in sorted((wt / ".claude" / "rules").glob("*.md")):
        paths, _ = hook.parse_rule(f.read_text(encoding="utf-8"))
        if paths:
            pats = [hook.glob_to_regex(x) for p in paths
                    for x in hook.expand_braces(p)]
            out.append((f.name, pats))
    return out


def regions_of(path: str, rules: list) -> list:
    hits = [name for name, pats in rules if any(p.match(path) for p in pats)]
    return hits or ["(no region)"]


@dataclass
class Sorted:
    known: list = field(default_factory=list)       # [(id, why)]
    preexisting: list = field(default_factory=list)  # [id]
    #: [(id, {region: [owner refs]}, [culprit paths], base_known)]
    staging: list = field(default_factory=list)
    stale_known: list = field(default_factory=list)  # known, but ran green


def stage_reds(exits: dict, reds_by_stage: dict) -> list:
    """A gate stage that exited red without naming a failing test (pytest
    crashed, cuda_green REFUSED, the run was killed), or never ran (exit
    None), is itself a red, so a gate can never pass on a missing result."""
    out = []
    for st, rc in exits.items():
        if rc is None:
            out.append(f"[{st}] did not run")
        elif rc != 0 and not reds_by_stage.get(st):
            out.append(f"[{st}] exit {rc}, no failing test named "
                       f"(see {st}.log)")
    return out


def sort_reds(wt: Path, reds: list, inputs: list, base_sha: str,
              known: list, base_reds: set | None,
              collected: set | None = None) -> Sorted:
    """Sort the reds of a staging run.

    KNOWN: in this machine's table (never a staging red). PRE-EXISTING: red
    in ``base_reds`` too. Every other red is a STAGING red, with ``base_known``
    False when no base result was given ("unknown base"), and is sorted by
    the changed files that reach it (``affected_tests.reach``): each such
    file's rule regions (AREA) and the inputs whose diff holds it (OWNER). A
    red no change reaches gets the region ``(unreached)``; a stage that failed
    without naming a test, ``(stage failure)``. ``collected`` (the node ids
    the run's halves held) finds KNOWN entries that ran green."""
    out = Sorted()
    per_input = {inp.ref: changed_by(wt, base_sha, inp.sha) for inp in inputs}
    changed = sorted(set().union(*per_input.values())) if per_input else []
    rules = load_rules(wt)
    rest = []
    for nid in reds:
        k = known_match(nid, known)
        if k:
            out.known.append((nid, k.get("why", "")))
        elif base_reds is not None and nid in base_reds:
            out.preexisting.append(nid)
        else:
            rest.append(nid)
    files = sorted({n.split("::")[0] for n in rest
                    if not n.startswith("[") and (wt / n.split("::")[0])
                    .is_file()})
    reach, full = {}, []
    with affected_tests.rooted(wt):
        for path in changed:
            hit, _, is_full = affected_tests.reach(path, candidates=files)
            if is_full:
                full.append(path)
            reach[path] = hit
    for nid in rest:
        if nid.startswith("["):
            out.staging.append((nid, {"(stage failure)": []}, [],
                                base_reds is not None))
            continue
        f = nid.split("::")[0]
        culprits = [p for p in changed if f in reach[p]] or full
        area = {}
        for p in culprits:
            owners = [ref for ref, paths in per_input.items() if p in paths]
            for reg in regions_of(p, rules):
                bucket = area.setdefault(reg, [])
                bucket.extend(o for o in owners if o not in bucket)
        out.staging.append((nid, area or {"(unreached)": []}, culprits,
                            base_reds is not None))
    if collected is not None:
        out.stale_known = [
            k["test"] for k in known
            if any(known_match(c, [k]) for c in collected)
            and not any(known_match(r, [k]) for r in reds)]
    return out


def format_sorted(s: Sorted) -> list:
    lines = []
    if s.staging:
        lines.append(f"STAGING REDS ({len(s.staging)}), by region "
                     f"(reds / owners):")
        by_region = {}
        for nid, area, _, _ in s.staging:
            for reg, owners in area.items():
                n, who = by_region.setdefault(reg, [0, []])
                by_region[reg][0] = n + 1
                who.extend(o for o in owners if o not in who)
        for reg in sorted(by_region):
            n, who = by_region[reg]
            lines.append(f"  [{reg}] {n}: {', '.join(who) or 'no input'}")
        lines.append("  each red:")
        for nid, area, culprits, base_known in s.staging:
            owners = sorted({o for v in area.values() for o in v})
            tag = "" if base_known else "  (unknown base)"
            lines.append(f"    {nid}{tag}")
            lines.append(f"        owner: {', '.join(owners) or 'no input'}"
                         f"; area: {', '.join(sorted(area))}")
            if culprits:
                lines.append(f"        via {', '.join(culprits[:6])}"
                             f"{' ...' if len(culprits) > 6 else ''}")
    if s.preexisting:
        lines.append(f"PRE-EXISTING (red on the base too) "
                     f"({len(s.preexisting)}):")
        lines += [f"    {n}" for n in s.preexisting]
    if s.known:
        lines.append(f"KNOWN on this machine ({len(s.known)}):")
        lines += [f"    {n}  ({why})" for n, why in s.known]
    if s.stale_known:
        lines.append("KNOWN reds that ran green (amend "
                     "tools/staging_machines.toml):")
        lines += [f"    {n}" for n in s.stale_known]
    if not lines:
        lines.append("no reds")
    return lines


# ---------------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------------
def machine(table_path: Path = MACHINES, host: str | None = None) -> dict:
    host = host or platform.node()
    table = tomllib.loads(table_path.read_text(encoding="utf-8"))
    for key, row in table.items():
        if key.lower() == host.lower():
            return {"host": host, "known_red": [], **row}
    return {"host": host, "name": host, "cpu_build": "cmake",
            "cuda_build": "cpp/build_cuda.bat", "known_red": []}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, args, repo: Path):
        self.args = args
        self.repo = repo
        self.main = common_root(repo)
        self.python = args.python
        self.mach = machine()
        self.state = {}

    # -- paths and state ----------------------------------------------------
    def setup(self, name: str):
        self.name = name
        self.run_dir = Path(self.args.run_dir) if self.args.run_dir else \
            self.main / ".claude" / "staging_runs" / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wt = self.main / ".claude" / "worktrees" / f"staging-{name}"
        self.branch = f"staging/{name}"
        sf = self.run_dir / "state.json"
        if sf.is_file():
            self.state = json.loads(sf.read_text(encoding="utf-8"))

    def save(self):
        (self.run_dir / "state.json").write_text(
            json.dumps(self.state, indent=2), encoding="utf-8")

    def say(self, msg: str):
        print(f"staging_gate: {msg}", flush=True)

    @property
    def commit(self) -> str:
        return self.state.get("commit", "")

    @property
    def inputs(self) -> list:
        return [Input(**i) for i in self.state.get("inputs", [])]

    # -- stages ---------------------------------------------------------------
    def stage_merge(self, base: Input, inputs: list, dry: bool) -> int:
        log = self.run_dir / "merge.log"
        try:
            res = merge_chain(self.repo, base, inputs, self.name)
        except MergeConflict as e:
            write_log(log, e.log + ["", str(e)], EXIT_CONFLICT, base.sha)
            self.say(str(e))
            return EXIT_CONFLICT
        lines = res.log[:]
        for inp, why in res.skipped:
            self.say(f"{inp.ref}: {why}, not merged")
        self.state.update({
            "name": self.name, "base": vars(base),
            "inputs": [vars(i) for i in res.merged],
            "skipped": [[vars(i), w] for i, w in res.skipped],
            "commit": res.commit, "branch": self.branch, "at": False})
        if dry:
            lines.append(f"dry run: staging commit {res.commit} (not kept)")
            write_log(log, lines, 0, res.commit)
            self.say(f"dry run clean: {len(res.merged)} input(s) merge "
                     f"without conflict onto {base.ref} {base.sha[:12]}")
            return 0
        if git(self.repo, "rev-parse", "--verify", "--quiet", self.branch,
               check=False):
            raise GateError(f"{self.branch} exists: another --name, or "
                            f"--remove the old run")
        git(self.repo, "branch", self.branch, res.commit)
        git(self.repo, "worktree", "add", str(self.wt), self.branch,
            timeout=3600)
        lines.append(f"staging branch {self.branch} = {res.commit}")
        lines.append(f"worktree {self.wt}")
        if self.args.push_staging:
            git(self.repo, "push", "-u", "origin", self.branch, timeout=600)
            self.state["pushed"] = True
            lines.append(f"pushed origin/{self.branch}")
        write_log(log, lines, 0, res.commit)
        self.save()
        self.say(f"staged {self.branch} = {res.commit[:12]} in {self.wt}")
        return 0

    def stage_at(self, ref: str) -> int:
        git(self.repo, "fetch", "origin", timeout=600)
        sha = resolve(self.repo, ref)
        base, inputs = read_staging(self.repo, sha)
        if not self.wt.exists():
            git(self.repo, "worktree", "add", "--detach", str(self.wt), sha,
                timeout=3600)
        elif git(self.wt, "rev-parse", "HEAD") != sha:
            raise GateError(f"{self.wt} is at another commit; --remove it")
        self.state.update({"name": self.name, "base": vars(base),
                           "inputs": [vars(i) for i in inputs],
                           "commit": sha, "branch": None, "at": ref})
        write_log(self.run_dir / "merge.log",
                  [f"--at {ref}: no merge; checked out {sha}",
                   f"base {base.ref} {base.sha}"]
                  + [f"input {i.ref} {i.sha}" for i in inputs], 0, sha)
        self.save()
        self.say(f"at {ref} = {sha[:12]}: {len(inputs)} input(s)")
        return 0

    def stage_build(self, kind: str) -> int:
        stage = f"build_{kind}"
        wait_for_slot(self.args.max_builds, self.args.poll, self.args.force,
                      self.say)
        script = self.mach[f"{kind}_build"]
        log = self.run_dir / f"{stage}.log"
        if script == "cmake":
            cmd = cmake_build(self.wt, self.python)
        else:
            cmd = [["cmd", "/c", str(self.wt / script)] if script.endswith(
                ".bat") else ["bash", str(self.wt / script)]]
        self.say(f"{stage}: {script}")
        started = time.time()
        rc = run_logged(cmd, log, self.wt, self.commit)
        pyd_dir = self.wt / (CPU_PYD_DIR if kind == "cpu" else CUDA_PYD_DIR)
        if kind == "cpu" and find_pyd(pyd_dir) is None \
                and find_pyd(self.wt / "cpp" / "build"):
            pyd_dir.mkdir(parents=True, exist_ok=True)   # single-config gen
            shutil.copy2(find_pyd(self.wt / "cpp" / "build"), pyd_dir)
        pyd = find_pyd(pyd_dir)
        bid = module_build_id(self.wt, self.python, kind) \
            if pyd_problem(pyd) is None else "<no module>"
        problems = judge_build(log, pyd, bid, self.commit, started)
        if rc != 0:
            problems.insert(0, f"the build command exited {rc}")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"# module {pyd} build_id {bid}\n")
            for p in problems:
                fh.write(f"# PROBLEM {p}\n")
        code = EXIT_BUILD if problems else 0
        append_exit(log, code, self.commit)
        if problems:
            self.say(f"{stage} FAILED:\n  " + "\n  ".join(problems))
        else:
            self.say(f"{stage}: {pyd.name} ({pyd.stat().st_size} bytes) "
                     f"build_id = staging commit")
        return code

    def select(self) -> str:
        return cuda_green_select(self.wt / "tools" / "cuda_green.py")

    def stage_partition(self) -> int:
        sel = self.select()
        got = partition_gaps(self.wt, self.python, sel)
        lines = [f"card half: tools/cuda_green.py -k {sel!r} + {CLASS_P_TEST}",
                 f"cpu half: {' '.join(half_args('cpu', sel))}",
                 f"collected: {got['counts']}"]
        for k in ("gap", "overlap", "extra"):
            if got[k]:
                lines.append(f"{k.upper()} ({len(got[k])}):")
                lines += [f"    {n}" for n in got[k][:200]]
        code = EXIT_REFUSED if (got["gap"] or got["overlap"] or got["extra"]) \
            else 0
        write_log(self.run_dir / "partition.log", lines, code, self.commit)
        (self.run_dir / "collected.json").write_text(
            json.dumps(got["halves"]), encoding="utf-8")
        self.say(f"partition: {got['counts']} "
                 f"{'OK' if code == 0 else 'HALVES DRIFTED, see partition.log'}")
        return code

    def pytest_stage(self, stage: str, args: list, extra=()) -> int:
        xml = self.run_dir / f"{stage}.xml"
        xml.unlink(missing_ok=True)     # never sort an older attempt's reds
        cmd = [self.python, "-m", "pytest", *args, "-q", *PYTEST_COMMON,
               f"--junitxml={xml}", *extra]
        self.say(f"{stage}: {' '.join(map(str, cmd[1:]))[:160]}")
        rc = run_logged(cmd, self.run_dir / f"{stage}.log", self.wt,
                        self.commit)
        self.say(f"{stage}: exit {rc}")
        return rc

    def stage_cuda_green(self) -> int:
        wait_for_slot(self.args.max_builds, self.args.poll, self.args.force,
                      self.say)
        xml = self.run_dir / "cuda_green.xml"
        xml.unlink(missing_ok=True)     # never sort an older attempt's reds
        env = dict(os.environ)
        env["PYTEST_ADDOPTS"] = (env.get("PYTEST_ADDOPTS", "") +
                                 f" --junitxml={xml} -o junit_family=xunit1")
        rc = run_logged([self.python, "tools/cuda_green.py"],
                        self.run_dir / "cuda_green.log", self.wt, self.commit,
                        env=env)
        self.say(f"cuda_green: exit {rc}")
        return rc

    def stage_suite(self) -> int:
        wait_for_slot(self.args.max_builds, self.args.poll, self.args.force,
                      self.say)
        return self.pytest_stage("suite", half_args("cpu", self.select()),
                                 xdist_args(self.python, self.args.workers))

    def stage_agents(self) -> int:
        if not (self.wt / AGENTS_TESTS).is_dir():
            write_log(self.run_dir / "agents.log",
                      [f"no {AGENTS_TESTS} at this commit"], 0, self.commit)
            return 0
        return self.pytest_stage("agents", [AGENTS_TESTS])

    def stage_sort(self, gates: list) -> int:
        exits = self.state.get("exits", {})
        by_stage = {st: junit_reds(self.run_dir / f"{st}.xml")
                    for st in gates}
        reds = sorted({r for v in by_stage.values() for r in v})
        reds += stage_reds({st: exits.get(st) for st in gates}, by_stage)
        base_reds, notes = None, []
        if self.args.base_result:
            base_reds, at = read_result(Path(self.args.base_result))
            base_sha = self.state["base"]["sha"]
            if at and not base_sha.startswith(at):
                notes.append(f"WARNING: the base result is of {at}, the base "
                             f"is {base_sha[:12]}")
        collected = None
        cj = self.run_dir / "collected.json"
        if cj.is_file():
            halves = json.loads(cj.read_text(encoding="utf-8"))
            ran = {g for g in gates if exits.get(g) is not None}
            collected = set()
            if "suite" in ran:
                collected |= set(halves["cpu"])
            if {"cuda_green", "class_p"} <= ran:
                collected |= set(halves["cuda"])
        s = sort_reds(self.wt, reds, self.inputs, self.state["base"]["sha"],
                      self.mach.get("known_red", []), base_reds, collected)
        lines = notes + [f"gates sorted: {', '.join(gates) or 'none'} "
                         f"(exits {dict((g, exits.get(g)) for g in gates)})"]
        lines += format_sorted(s)
        (self.run_dir / "reds.txt").write_text(
            "\n".join([f"# commit {self.commit}",
                       *[r for r in reds if not r.startswith("[")]]) + "\n",
            encoding="utf-8")
        (self.run_dir / "reds.json").write_text(json.dumps({
            "commit": self.commit,
            "staging": [{"test": n, "area": a, "via": c, "base_known": b}
                        for n, a, c, b in s.staging],
            "preexisting": s.preexisting,
            "known": [{"test": n, "why": w} for n, w in s.known]},
            indent=2), encoding="utf-8")
        code = EXIT_RED if s.staging else 0
        write_log(self.run_dir / "sort.log", lines, code, self.commit)
        print("\n".join(lines))
        return code


def plan(half: str, start: str | None, until: str | None) -> list:
    chosen = [s for s in STAGES if s in HALVES[half]]
    start = ALIASES.get(start, start)
    until = ALIASES.get(until, until)
    if start:
        if start not in chosen:
            raise GateError(f"--from {start}: not a stage of --stages {half} "
                            f"({', '.join(chosen)})")
        chosen = chosen[chosen.index(start):]
    if until:
        if until not in chosen:
            raise GateError(f"--until {until}: not in {', '.join(chosen)}")
        chosen = chosen[:chosen.index(until) + 1]
    return chosen


def remove(run: Run) -> int:
    """Delete a run's worktree, its staging branch (local, and on origin when
    this gate pushed it) -- only a branch still at the commit the gate made."""
    st = run.state
    if run.wt.exists():
        git(run.repo, "worktree", "remove", "--force", str(run.wt),
            timeout=3600)
        run.say(f"removed {run.wt}")
    br = st.get("branch")
    if br and git(run.repo, "rev-parse", "--verify", "--quiet", br,
                  check=False) == st.get("commit"):
        git(run.repo, "branch", "-D", br)
        run.say(f"deleted {br}")
        if st.get("pushed"):
            git(run.repo, "push", "origin", "--delete", br, check=False)
            run.say(f"deleted origin/{br}")
    elif br:
        run.say(f"{br} kept: it is not at the gate's commit")
    git(run.repo, "worktree", "prune")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="staging_gate", description="One full suite and one cuda_green "
        "for several ready branches, on one throwaway staging commit.")
    ap.add_argument("inputs", nargs="*",
                    help="branch heads (names or hashes), merged in order")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--name", help="the run's name (default: derived)")
    ap.add_argument("--at", help="run on an existing staging commit/branch")
    ap.add_argument("--stages", choices=sorted(HALVES), default="all")
    ap.add_argument("--from", dest="start", help="resume at this stage")
    ap.add_argument("--until", help="stop after this stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="only merge (in memory) and report conflicts")
    ap.add_argument("--push-staging", action="store_true",
                    help="push the staging branch to origin after the merge")
    ap.add_argument("--base-result",
                    help="the base's reds (junit .xml or a reds.txt)")
    ap.add_argument("--workers", type=int, default=None,
                    help="xdist workers for the suite (0 = serial; default 4 "
                         "when pytest-xdist is installed)")
    ap.add_argument("--max-builds", type=int, default=2)
    ap.add_argument("--poll", type=float, default=180.0)
    ap.add_argument("--force", action="store_true",
                    help="start although the GPU or the build slots are busy")
    ap.add_argument("--remove", action="store_true",
                    help="delete the named run's worktree and staging branch")
    ap.add_argument("--run-dir")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--repo", default=str(ROOT))
    args = ap.parse_args(argv)
    repo = Path(args.repo)
    run = Run(args, repo)
    commit, code, stages = "", EXIT_REFUSED, []
    try:
        if args.remove:
            if not args.name:
                raise GateError("--remove needs --name")
            run.setup(args.name)
            return remove(run)
        resuming = bool(args.start) and ALIASES.get(args.start, args.start) \
            != "merge"
        if args.at:
            if args.dry_run or args.inputs:
                raise GateError("--at runs an existing staging commit: no "
                                "inputs and no --dry-run")
            name = args.name or re.sub(r"^(origin/)?(staging/)?", "",
                                       args.at).replace("/", "-")
            if re.fullmatch(r"[0-9a-f]{7,40}", name):
                name = name[:12]
            run.setup(name)
        elif resuming:
            if not args.name:
                raise GateError("--from needs --name (the run to resume)")
            run.setup(args.name)
            if not run.state.get("commit"):
                raise GateError(f"no staged run named {args.name} in "
                                f"{run.run_dir}")
            given = [resolve(repo, r) for r in args.inputs]
            if given and given != [i.sha for i in run.inputs]:
                raise GateError("the inputs given are not the run's inputs")
        else:
            if not args.inputs:
                raise GateError("no inputs (branch heads to stage)")
            inputs = [Input(r, resolve(repo, r)) for r in args.inputs]
            base = Input(args.base, resolve(repo, args.base))
            name = args.name or derive_name(
                [i for i in inputs if i.sha != base.sha])
            run.setup(name)
        stages = ["merge"] if args.dry_run else plan(args.stages, args.start,
                                                     args.until)
        run.say(f"{run.mach['name']} ({run.mach['host']}): stages "
                f"{', '.join(stages)}; run folder {run.run_dir}")
        if GPU_STAGES & set(stages) and not args.force:
            jobs = load_jobs(*list_procs())
            if jobs["suites"]:
                raise GateError(
                    "the GPU is busy (another suite or cuda_green runs; "
                    "--force to start anyway):\n  "
                    + "\n  ".join(describe(p) for p in jobs["suites"]))
        code, broke = 0, False
        for st in stages:
            if st == "merge":
                rc = run.stage_at(args.at) if args.at else \
                    run.stage_merge(base, inputs, args.dry_run)
                commit = run.commit
                if rc:
                    code, broke = rc, True
                    break
                continue
            commit = run.commit
            if not run.wt.exists():
                raise GateError(f"no staging worktree {run.wt}")
            if st in ("build_cpu", "build_cuda"):
                rc = run.stage_build(st.split("_")[1])
            elif st == "partition":
                rc = run.stage_partition()
            elif st == "cuda_green":
                rc = run.stage_cuda_green()
            elif st == "class_p":
                rc = run.pytest_stage("class_p", half_args("class_p", ""))
            elif st == "suite":
                rc = run.stage_suite()
            elif st == "agents":
                rc = run.stage_agents()
            else:   # sort: the gates of this half that have run, ever
                rc = run.stage_sort([g for g in GATES
                                     if g in HALVES[args.stages]])
            if st != "sort":
                run.state.setdefault("exits", {})[st] = rc
                run.save()
            if st in STOPPERS and rc:
                code, broke = rc, True
                break
            if st == "sort":
                code = rc
        if not broke and "sort" not in stages and not args.dry_run:
            exits = run.state.get("exits", {})
            code = EXIT_RED if any(exits.get(st) for st in stages) else 0
        if code == 0 and not args.dry_run and stages and stages[-1] == "sort":
            other = {"cpu": "cuda", "cuda": "cpu"}.get(args.stages)
            print("\nGREEN. To merge the inputs to main (not run by this "
                  "tool):")
            if other:
                print(f"  only once the {other} half is green at "
                      f"{run.commit} too")
            for c in merge_commands(run.inputs):
                print(f"    {c}")
    except MergeConflict as e:
        code = EXIT_CONFLICT
        print(e)
    except GateError as e:
        code = e.code
        print(f"staging_gate: REFUSED -- {e}")
    summary = {"machine": f"{run.mach['name']} ({run.mach['host']})",
               "stages": args.stages if not args.dry_run else "dry-run",
               "commit": run.commit or commit, "exit": code}
    if getattr(run, "run_dir", None):
        (run.run_dir / "summary.json").write_text(json.dumps(summary),
                                                  encoding="utf-8")
    print(f"STAGING_SUMMARY {json.dumps(summary)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
