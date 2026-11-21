"""tests/_check_parts.py -- ONE subprocess run of a device check for many
pytest tests, each still judged by its own part (#29 DEVICE CHECK PLAYS).

A device check (``tests/cuda_obs_check.py``, ``cuda_conductor_check.py``,
``cuda_conductor_batch_check.py``) plays every match of its corpus ONCE per
subprocess (its play cache) and every part reads that one play. Before this,
each ``test_*`` wrapper started its own subprocess, so each part re-played the
corpus it shared with the others. Now the wrapper module's session fixture
starts ONE subprocess with every selected test's parts, and each test reads
its own part's verdict and output from it.

Two halves, no CUDA import in either (``cuda_harness``'s rule: the CUDA module
never enters pytest):

- check side, inside the subprocess: :func:`run_parts` runs the parts in
  order, each as a fresh process would see it -- its own ``FAILS`` list and
  the check's per-run tallies reset (``reset``), an exception caught and
  counted against THAT part only -- and prints one verdict line per part,
  ``PART_VERDICT <name>: PASS|FAIL (n) <seconds> s``;
- pytest side: :func:`selected_parts` reads which parts the selected tests of
  a wrapper module need; :func:`split` cuts the subprocess output into one
  section per part with its verdict; :func:`assert_parts` fails a test on its
  own parts alone, with its own section as the message.

The plays a check caches are shared by every part that only READS them (a
host reference, its taps, its per-decision arrays): a part never writes to a
cached play; a part that changes the match in any way (a perturbation, a
break switch, a reset, another spec, seed or level) plays its own. The check's
deliberate breaks (``OBS_BREAK`` and the others) are never part of the gate
run; by hand, ``main([...])`` of a check runs exactly as before.
"""
from __future__ import annotations

import re
import time
import traceback

VERDICT = "PART_VERDICT"
_LINE = re.compile(rf"^{VERDICT} (\S+): (PASS|FAIL)\b.*$", re.M)
_PART = re.compile(r"^PART (\S+)\s*$", re.M)


def run_parts(parts, table, fails: list, reset=None) -> int:
    """Run ``parts`` (names in ``table``) in order, each judged alone; return
    the number of failing parts. ``fails`` is the check's module-level FAILS
    list: during a part it holds only that part's failures (as in a fresh
    process); afterwards every part's failures are in it again, in order.
    ``reset()`` clears the check's per-run tallies before each part."""
    kept: list = []
    bad = 0
    for name in parts:
        if reset is not None:
            reset()
        fails.clear()
        print(f"PART {name}", flush=True)
        t0 = time.perf_counter()
        try:
            table[name]()
        except Exception as ex:            # a part's crash fails it, the rest run
            traceback.print_exc()
            fails.append(f"part {name} raised {type(ex).__name__}: {ex}")
            print("FAIL:", str(fails[-1]).encode("ascii", "replace").decode("ascii"))
        n = len(fails)
        bad += n > 0
        print(f"{VERDICT} {name}: {'PASS' if not n else f'FAIL ({n})'} "
              f"{time.perf_counter() - t0:.1f} s", flush=True)
        kept.extend(fails)
    fails[:] = kept
    return bad


def selected_parts(session, module_file: str, parts_of: dict) -> list:
    """The parts the SELECTED tests of one wrapper module need, in the order
    of ``parts_of`` (``test name -> parts``), each once. Read from
    ``session.items`` (after ``-k`` / ``-m`` deselection), so a run of one
    test plays only that test's parts."""
    from pathlib import Path
    me = Path(module_file).resolve()
    names = {item.originalname for item in session.items
             if Path(str(item.path)).resolve() == me}
    out = []
    for test, parts in parts_of.items():
        if test in names:
            out.extend(p for p in parts if p not in out)
    return out


def split(output: str) -> dict:
    """``part -> (verdict 'PASS' / 'FAIL' / None, its section of the output)``;
    None: the part started and printed no verdict (the subprocess died or
    timed out inside it), or never started."""
    starts = [(m.start(), m.group(1)) for m in _PART.finditer(output)]
    verdicts = {m.group(1): m.group(2) for m in _LINE.finditer(output)}
    out = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(output)
        out[name] = (verdicts.get(name), output[pos:end])
    return out


def assert_parts(run: dict, parts, tail: int = 6000) -> None:
    """Fail unless every part of ``parts`` printed PASS in the shared run
    ``run`` (``{"sections": split(...), "output": str, "error": str|None}``);
    the message is the failing part's own section (or the run's tail when
    the part has none)."""
    for name in parts:
        verdict, text = run["sections"].get(name, (None, ""))
        if verdict == "PASS":
            continue
        why = run.get("error") or ""
        if not text:
            text = f"part {name!r} never started in the shared run\n" + run["output"][-tail:]
        raise AssertionError(f"part {name}: {verdict or 'no verdict'} {why}\n{text[-tail:]}")


def shared_run(request, module_file: str, parts_of: dict, module: str, launch) -> dict:
    """The body of a wrapper module's session fixture: ONE run of check
    ``module`` with the parts of every selected test (``parts_of``: test ->
    (parts, the timeout its parts had alone, s)), through ``launch(body,
    timeout)`` -- the wrapper's own ``cuda_harness.run_cuda_script`` (named in
    the wrapper, so ``_suite_kinds`` classifies its tests as device tests).
    The timeout is the sum of the selected tests' own: each part keeps its
    budget; it is a hang guard, never a speed gate."""
    import subprocess
    parts = selected_parts(request.session, module_file,
                           {n: v[0] for n, v in parts_of.items()})
    timeout = sum(v[1] for v in parts_of.values() if set(v[0]) & set(parts))
    error = None
    try:
        proc = launch(f"import {module}, sys; sys.exit({module}.main({parts!r}))",
                      timeout)
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as ex:
        out = _text(ex.stdout) + "\n" + _text(ex.stderr)
        error = f"(the shared run timed out after {timeout} s)"
    return {"sections": split(out), "output": out, "error": error, "parts": parts}


def _text(x) -> str:
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
