"""tools/twin_manifest_gen.py -- GENERATE the twin manifest's file sets (#29 P1b).

Residency design §6 (re-check M6): a device twin's reference FILE SET is
generated, never hand-listed, so a transitive helper a slot runs is in the set
of every twin whose slot runs it. This tool runs THE CORPUS -- the medium
match (``MatchSession`` on ``[corpus] level``, ``[corpus] ticks`` ticks, the
CPU build ``cpp/build/Release``) -- under a ``sys.settrace`` call tracer and
records, per slot of ``Simulation.step`` and per stage of slot 7, which files
under ``src/`` executed. The slot and stage map is
``tools/tick_stage_profile.py``'s (``SLOT_WRAPS`` / ``PHYSICS_WRAPS``): the
same callables, wrapped here with a label push / pop instead of a timer. A
file executed while several labels are open (slot 7 and one of its stages) is
recorded under each of them; outside any slot it is ``(rest of step)``,
outside ``step`` ``(outside step)`` (the controllers), before the first tick
``(setup)``.

C++ cannot be traced from Python, so ``[cpp_stages]`` in the manifest maps a
stage to the C++ translation units it runs -- an explicit table this tool
keeps and the reviewer checks. A key is a traced label, or
``"<label> / <part>"`` for a part of a stage the tracer sees as one call (the
part inherits its label's Python files). Each TU brings its local
``#include "..."`` closure under ``cpp/src/`` (generated).

A twin's ``files`` = the union over its ``slots`` of the traced Python files
and the ``[cpp_stages]`` C++ closure, sorted. This tool rewrites only the
``files = [...]`` array of each ``[twins.*]`` table and leaves every other
line of the manifest (comments included) as it is.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/twin_manifest_gen.py
    ... tools/twin_manifest_gen.py --check     # exit 1 when the manifest is stale
    ... tools/twin_manifest_gen.py --labels    # print the traced map

``tests/test_twin_debt.py`` runs the same trace and fails when the committed
``files`` differ (regenerate and commit) or when a ``src/simulation/`` module
the corpus executes is in no twin's ``files`` and not in ``[not_twinned]``.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "twin_manifest.toml"
CPU_BUILD = ROOT / "cpp" / "build" / "Release"
CPP_SRC = "cpp/src"

SETUP = "(setup)"
OUTSIDE = "(outside step)"
REST_STEP = "(rest of step)"
STEP = "step"
SUBPART = " / "

_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------
def load_manifest(path=MANIFEST) -> dict:
    """The parsed manifest (``tomllib``)."""
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def label_key(label: str) -> str:
    """A ``tick_stage_profile`` label as the manifest spells it (no indent)."""
    return label.strip()


def traced_label(slot: str) -> str:
    """The traced label a manifest slot inherits its Python files from."""
    return slot.split(SUBPART, 1)[0]


def known_labels() -> list:
    """Every label the tracer can record, in slot order."""
    import tick_stage_profile as tsp
    out = [SETUP, OUTSIDE, STEP, REST_STEP]
    for _m, _q, label in tsp.SLOT_WRAPS:
        if label_key(label) not in out:
            out.append(label_key(label))
    for _m, _q, label, _d in tsp.PHYSICS_WRAPS:
        if label_key(label) not in out:
            out.append(label_key(label))
    return out


# ---------------------------------------------------------------------------
# The C++ side: explicit TUs, generated include closure
# ---------------------------------------------------------------------------
def cpp_closure(tus, repo=ROOT) -> set:
    """``tus`` (repo-relative paths) plus every local header they include,
    transitively, that exists under ``cpp/src/``."""
    repo = Path(repo)
    out, todo = set(), list(tus)
    while todo:
        rel = todo.pop()
        if rel in out:
            continue
        p = repo / rel
        if not p.is_file():
            raise FileNotFoundError(f"twin manifest names {rel}, which does "
                                    f"not exist")
        out.add(rel)
        for inc in _INCLUDE.findall(p.read_text(encoding="utf-8",
                                                errors="replace")):
            cand = f"{CPP_SRC}/{inc}"
            if (repo / cand).is_file() and cand not in out:
                todo.append(cand)
    return out


# ---------------------------------------------------------------------------
# The tracer
# ---------------------------------------------------------------------------
class SlotTracer:
    """Label push / pop wrappers on the slot and stage callables
    (``tick_stage_profile``'s map) plus a ``sys.settrace`` call hook that
    records, under every open label, the ``src/`` file of each Python frame
    entered. Nothing in the sim is read or written."""

    def __init__(self, repo=ROOT):
        self.src = (Path(repo) / "src").resolve()
        self.repo = Path(repo).resolve()
        self.stack = [SETUP]
        self.by_stack = defaultdict(set)   # tuple(stack) -> {code filename}
        self._orig = []
        self._rel = {}

    # -- wrappers ------------------------------------------------------------
    def _patch(self, cont, attr, label):
        orig = getattr(cont, attr)
        tracer = self

        def labelled(*a, **k):
            tracer.stack.append(label)
            try:
                return orig(*a, **k)
            finally:
                tracer.stack.pop()
        labelled.__name__ = getattr(orig, "__name__", attr)
        labelled.__wrapped__ = orig
        setattr(cont, attr, labelled)
        self._orig.append((cont, attr, orig))

    def install(self, bp):
        import tick_stage_profile as tsp
        from simulation import Simulation
        self._patch(Simulation, "step", STEP)
        for mod, qual, label in tsp.SLOT_WRAPS:
            try:
                cont, attr = tsp._resolve(importlib.import_module(mod), qual)
                getattr(cont, attr)
            except AttributeError:
                continue
            if not isinstance(cont, type) or attr in cont.__dict__:
                self._patch(cont, attr, label_key(label))
        for mod, qual, label, _dev in tsp.PHYSICS_WRAPS:
            owner = bp if mod == "bp" else importlib.import_module(mod)
            try:
                cont, attr = tsp._resolve(owner, qual)
                getattr(cont, attr)
            except AttributeError:
                continue
            self._patch(cont, attr, label_key(label))

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()

    # -- the call hook -------------------------------------------------------
    def _hook(self, frame, event, arg):
        if event == "call":
            self.by_stack[tuple(self.stack)].add(frame.f_code.co_filename)
        return None              # no line tracing

    def start(self):
        self._prev = sys.gettrace()
        sys.settrace(self._hook)

    def stop(self):
        sys.settrace(getattr(self, "_prev", None))

    # -- the result ----------------------------------------------------------
    def rel(self, filename: str):
        """``filename`` as a repo-relative POSIX path when under ``src/``."""
        if filename in self._rel:
            return self._rel[filename]
        out = None
        try:
            p = Path(filename).resolve()
            p.relative_to(self.src)
            out = p.relative_to(self.repo).as_posix()
        except (ValueError, OSError):
            pass
        self._rel[filename] = out
        return out

    def by_label(self) -> dict:
        """label -> sorted repo-relative ``src/`` files executed under it.
        The stack's bottom is the phase: everything while building the match
        is ``(setup)``; in the tick loop a frame with no label open is
        ``(outside step)`` (the controllers), one under ``step`` with no slot
        open ``(rest of step)``, and any other lands under every label open
        above the phase."""
        out = defaultdict(set)
        for stack, files in self.by_stack.items():
            rels = {r for r in (self.rel(f) for f in files) if r}
            if not rels:
                continue
            phase, rest = stack[0], stack[1:]
            if phase == SETUP or not rest:
                labels = {phase}
            else:
                labels = set(rest)
                if rest[-1] == STEP:
                    labels.add(REST_STEP)
            for lab in labels:
                out[lab] |= rels
        return {k: sorted(v) for k, v in out.items()}


def trace_corpus(manifest: dict, repo=ROOT) -> dict:
    """Run the manifest's ``[corpus]`` match on the CPU build under the
    tracer; returns ``label -> sorted files``. Run in a process whose
    ``breach_physics`` is (or can be imported from) the CPU build."""
    corpus = manifest["corpus"]
    for p in (ROOT, ROOT / "src", CPU_BUILD, ROOT / "tests", ROOT / "tools"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.path.remove(str(CPU_BUILD))
    sys.path.insert(0, str(CPU_BUILD))
    import breach_physics as bp
    if getattr(bp, "HAS_CUDA", False):
        raise RuntimeError(f"the corpus runs on the CPU build; this process "
                           f"imported {bp.__file__}")
    import level_loader
    from simulation.match_session import MatchSession

    tracer = SlotTracer(repo)
    tracer.install(bp)
    tracer.start()
    try:
        level = level_loader.load(corpus["level"])
        session = MatchSession(level, seed=int(corpus["seed"]),
                               breach_physics=bp)
        tracer.stack = [OUTSIDE]
        for _ in range(int(corpus["ticks"])):
            if session.is_terminal():
                break
            session.tick()
    finally:
        tracer.stop()
        tracer.uninstall()
    return tracer.by_label()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_files(manifest: dict, traced: dict, repo=ROOT) -> dict:
    """twin -> its generated ``files`` (sorted): the traced Python files of
    its slots' labels plus the C++ closure of its slots' ``[cpp_stages]``
    rows. Refuses a slot that is neither a traced label nor a
    ``[cpp_stages]`` row, by name."""
    labels = set(known_labels())
    cpp = manifest.get("cpp_stages", {})
    out = {}
    for name, twin in manifest.get("twins", {}).items():
        files = set()
        for slot in twin.get("slots", []):
            base = traced_label(slot)
            if base not in labels:
                raise ValueError(f"twin {name!r}: slot {slot!r} is not a "
                                 f"label of tools/tick_stage_profile.py")
            if slot != base and slot not in cpp:
                raise ValueError(f"twin {name!r}: slot {slot!r} names a "
                                 f"part with no [cpp_stages] row")
            files |= set(traced.get(base, ()))
            files |= cpp_closure(cpp.get(slot, ()), repo)
        out[name] = sorted(files)
    return out


def executed_modules(traced: dict, prefix="src/simulation/") -> list:
    """Every module under ``prefix`` the corpus executed, anywhere."""
    return sorted({f for files in traced.values() for f in files
                   if f.startswith(prefix)})


def unclassified(manifest: dict, traced: dict, files: dict) -> list:
    """Executed ``src/simulation/`` modules in no twin's files and not in
    ``[not_twinned]``."""
    covered = {f for fs in files.values() for f in fs}
    nt = manifest.get("not_twinned", {})
    return [m for m in executed_modules(traced)
            if m not in covered and m not in nt]


def _files_block(files) -> str:
    body = "".join(f'    "{f}",\n' for f in files)
    return ("files = [  # GENERATED by tools/twin_manifest_gen.py -- never "
            "edit by hand\n" + body + "]")


def rewrite_manifest_text(text: str, files: dict) -> str:
    """``text`` with each ``[twins.<name>]`` table's ``files`` array replaced
    by ``files[name]`` (appended at the table's end when absent). Every other
    line is kept."""
    lines = text.splitlines(keepends=True)
    header = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
    out, i, n = [], 0, len(lines)
    while i < n:
        m = header.match(lines[i])
        out.append(lines[i])
        i += 1
        if not m or not m.group(1).strip().startswith("twins."):
            continue
        name = m.group(1).strip()[len("twins."):].strip().strip('"')
        body = []
        while i < n and not header.match(lines[i]):
            body.append(lines[i])
            i += 1
        sect = "".join(body)
        block = _files_block(files.get(name, []))
        # the array: up to the first `]` outside a string or a comment
        pat = re.compile(r'^files\s*=\s*\[(?:[^\]"#]|"[^"]*"|#[^\n]*)*\]',
                         re.M)
        if pat.search(sect):
            sect = pat.sub(lambda _m: block, sect, count=1)
        else:
            stripped = sect.rstrip("\n")
            tail = sect[len(stripped):] or "\n"
            sect = stripped + "\n" + block + "\n" + tail
        out.append(sect)
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 (writing nothing) when the manifest's "
                         "files are stale or a module is unclassified")
    ap.add_argument("--labels", action="store_true",
                    help="print the traced label -> files map")
    args = ap.parse_args(argv)
    path = Path(args.manifest)
    manifest = load_manifest(path)
    traced = trace_corpus(manifest)
    if args.labels:
        for lab in known_labels():
            if lab in traced:
                print(f"[{lab}]")
                for f in traced[lab]:
                    print(f"    {f}")
    files = generate_files(manifest, traced)
    missing = unclassified(manifest, traced, files)
    for m in missing:
        print(f"UNCLASSIFIED: {m} -- executed by the corpus, in no twin's "
              f"files and not in [not_twinned]")
    stale = [n for n, fs in files.items()
             if manifest["twins"][n].get("files") != fs]
    if args.check:
        for n in stale:
            print(f"STALE: twin {n!r} files differ from the generated set "
                  f"(run tools/twin_manifest_gen.py and commit)")
        return 1 if (stale or missing) else 0
    text = path.read_text(encoding="utf-8")
    new = rewrite_manifest_text(text, files)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="\n")
        print(f"rewrote {path} ({len(stale)} twin(s) changed: "
              f"{', '.join(stale) or 'formatting only'})")
    else:
        print(f"{path} is current")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
