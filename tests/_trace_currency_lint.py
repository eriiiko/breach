"""The trace-currency lints (#88 V1), as a harness module: pure functions over
the source tree that return their findings, so ``tests/test_trace_currency.py``
asserts them empty and a reviewer can run them by hand
(``python tests/_trace_currency_lint.py``) to list every hit.

Design: docs/smoke_unit_rescale_design_2026-10-20.md §10 V1, §2.7 (the fixture
inventory and its classes), critiques A2, C1, C2, C13.

Each scan reads CODE ONLY: comments and string literals (docstrings included)
are blanked by ``tokenize`` before a regex runs, so prose naming an old API is
never a finding and a real call can never hide in a comment.

The vocabulary:

* a TRACE SLICE is an index of the gas array by a trace id -- a trace name
  (``SMOKE``, ``STEAM``, ``POISON``, ``TEARGAS``, ``FUEL_GAS`` and their
  ``GAS_*`` / ``_SMOKE`` aliases, ``eng.gases.SMOKE``), a literal id 0..4, the
  ``[:N_TRACE_GASES]`` block -- or the ``.smoke`` view;
* a line is MARKED scale-free with ``# raw trace counts: scale-free (#88)``:
  a fixture whose property holds at any currency (conservation, divmod,
  splits, digests), the inventory's class R; a FLOAT plane in real units (a
  reference model, a render plane) is marked ``# real units, not counts (#88)``;
* the literal-id form is matched on the GameMap stack only (``.gas[1]``): a
  bare ``gas[0]`` is a direct-binding fixture's OWN local layout (checked by
  hand at #88 P1: cuda_bulk_flux_check, cuda_kick_check, the drag and clamp
  tests), whose trace plane, where it has one, rides along scale-free.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "renderer", "tools", "tests")
MARK = "raw trace counts: scale-free (#88)"

# Names a trace id goes by in the tree.
_TRACE_NAME = r"(?:STEAM|SMOKE|POISON|TEARGAS|FUEL_GAS|GAS_STEAM|GAS_SMOKE|GAS_POISON|GAS_TEARGAS|GAS_FUEL_GAS|_SMOKE|_STEAM)"
TRACE_SLICE = re.compile(
    r"(?:\bgas\s*\[\s*(?:[\w.]*\.)?" + _TRACE_NAME + r"\b"      # gas[SMOKE], gas[eng.gases.SMOKE]
    r"|\.gas\s*\[\s*[0-4]\s*[\],]"                             # g.gas[1], g.gas[1, y, x]
    r"|\bgas\s*\[\s*:\s*N_TRACE_GASES\b"                        # gas[:N_TRACE_GASES]
    r"|\.smoke\b(?!_))")                                        # the .smoke view

# What a write into a trace slice may be built from without a mark.
_TRACE_TOKEN = re.compile(r"(?i)\w*trace\w*|\bT_ONE\b|\bTQ\s*\(")
# Scale-free right-hand sides: a literal zero, a fresh zero plane.
_ZERO_RHS = re.compile(r"^\s*(?:-?0+(?:\.0*)?|np\.zeros(?:_like)?\s*\(.*)\s*$")
# A float plane in REAL units (a reference model, a render-side float plane).
MARK_REAL = "real units, not counts (#88)"

DELETED = ("FP_ONE", "FP_ONE_F", "SMOKE_MAX_Q", "quantize", "quantize_scalar",
           "dequantize", "dequantize_f32", "FP_SHIFT")
_DELETED_ATTR = re.compile(
    r"\b(?:gas_fixed|_gas_fx|_gas_fixed)\s*\.\s*(" + "|".join(DELETED) + r")\b(?!_)")
_DELETED_IMPORT = re.compile(
    r"from\s+simulation\.gas_fixed\s+import\s+([^\n]+)")

_BULK_CALL = re.compile(r"\b(?:quantize_bulk\w*|dequantize_bulk\w*|bulk_\w+)\s*\(")

# An int64 (or exact float64) accumulator on the same line as a sum.
_SUM = re.compile(r"\.sum\s*\(|\bnp\.sum\s*\(|\bsum\s*\(")
_WIDE = re.compile(r"int64|float64|\bint\s*\(\s*v\s*\)")


def _code_lines(text: str):
    """The source with every COMMENT and STRING token blanked to spaces (line
    structure kept), plus the raw lines (for the scale-free mark)."""
    raw = text.splitlines()
    out = [list(line) for line in raw]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return raw, raw
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        for r in range(sr, er + 1):
            if r - 1 >= len(out):
                break
            row = out[r - 1]
            c0 = sc if r == sr else 0
            c1 = ec if r == er else len(row)
            for c in range(c0, min(c1, len(row))):
                row[c] = " "
    return ["".join(r) for r in out], raw


def _git_tracked():
    """The repo's TRACKED files (absolute, resolved), or None outside a git
    checkout. The lint judges tracked code only: the working tree carries
    untracked notes, prototypes and frozen pre-#88 harnesses on purpose
    (CLAUDE.md: never `git add -A`), and a lint that read them would fail in
    the main checkout while passing in a fresh worktree."""
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {(ROOT / f).resolve() for f in out.decode("utf-8").split("\0") if f}


def iter_sources(dirs=SCAN_DIRS):
    """Every TRACKED .py under the scanned dirs (vendored trees and caches
    excluded), as (path, code_lines, raw_lines)."""
    tracked = _git_tracked()
    for d in dirs:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if tracked is not None and p.resolve() not in tracked:
                continue
            s = str(p)
            if "__pycache__" in s or "realesrgan" in s or "site-packages" in s:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            code, raw = _code_lines(text)
            yield p, code, raw


def _rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def deleted_name_hits():
    """Every reference to a DELETED ambiguous gas_fixed name (§4)."""
    hits = []
    for p, code, _raw in iter_sources():
        for i, line in enumerate(code, 1):
            m = _DELETED_ATTR.search(line)
            if m:
                hits.append(f"{_rel(p)}:{i}: gas_fixed.{m.group(1)}")
            m = _DELETED_IMPORT.search(line)
            if m:
                names = re.findall(r"\b(\w+)\b(?:\s+as\s+\w+)?", m.group(1))
                bad = [n for n in names if n in DELETED]
                if bad:
                    hits.append(f"{_rel(p)}:{i}: from gas_fixed import {bad}")
    return hits


def _split_assign(line):
    """(lhs, rhs) of a plain or augmented assignment on one code line, else
    None. Comparisons, keyword args and annotations are not assignments."""
    m = re.match(r"^(?P<lhs>[^=]*?[\]\w)])\s*(?P<op>\+=|-=|=)(?!=)\s*(?P<rhs>.*)$", line)
    if not m:
        return None
    lhs = m.group("lhs")
    if lhs.count("(") > lhs.count(")"):          # inside a call: a keyword argument
        return None
    if re.search(r"[<>!]$", lhs):
        return None
    return lhs, m.group("rhs")


def write_lint_hits():
    """Every assignment or ``+=`` / ``-=`` into a trace slice whose right-hand
    side takes no trace-currency name, is not a literal 0, and is not marked
    scale-free (§2.7, critique C1)."""
    hits = []
    for p, code, raw in iter_sources():
        for i, line in enumerate(code, 1):
            sp = _split_assign(line)
            if sp is None:
                continue
            lhs, rhs = sp
            if not TRACE_SLICE.search(lhs):
                continue
            if lhs.strip().startswith(("def ", "class ", "for ", "if ", "while ", "assert ")):
                continue
            if _TRACE_TOKEN.search(rhs) or _ZERO_RHS.match(rhs):
                continue
            if MARK in raw[i - 1] or MARK_REAL in raw[i - 1]:
                continue
            # a copy from another trace slice is in the same currency
            if TRACE_SLICE.search(rhs) and not re.search(r"[*/]|<<|>>", rhs):
                continue
            hits.append(f"{_rel(p)}:{i}: {raw[i - 1].strip()}")
    return hits


def bulk_name_on_trace_hits():
    """Every bulk_* / quantize_bulk* / dequantize_bulk* call whose argument
    list names a trace slice (critique C2)."""
    hits = []
    for p, code, raw in iter_sources():
        for i, line in enumerate(code, 1):
            for m in _BULK_CALL.finditer(line):
                depth, j = 1, m.end()
                while j < len(line) and depth:
                    depth += {"(": 1, ")": -1}.get(line[j], 0)
                    j += 1
                if TRACE_SLICE.search(line[m.end():j]):
                    hits.append(f"{_rel(p)}:{i}: {raw[i - 1].strip()}")
    return hits


def narrow_sum_hits():
    """Every sum over a trace slice on a line with no int64 (or exact float64)
    accumulator (critique A2: numpy < 2 on Windows reduces int32 in 32 bits)."""
    hits = []
    for p, code, raw in iter_sources():
        for i, line in enumerate(code, 1):
            if not TRACE_SLICE.search(line) or not _SUM.search(line):
                continue
            # only when the sum is taken OVER the slice (slice before .sum, or
            # inside sum(...)/np.sum(...))
            s_at = TRACE_SLICE.search(line).start()
            m_sum = None
            for m in _SUM.finditer(line):
                if m.group(0).startswith(".") and m.start() > s_at:
                    m_sum = m
                    break
                if not m.group(0).startswith(".") and m.start() < s_at:
                    m_sum = m
                    break
            if m_sum is None:
                continue
            if _WIDE.search(line) or MARK in raw[i - 1]:
                continue
            hits.append(f"{_rel(p)}:{i}: {raw[i - 1].strip()}")
    return hits


if __name__ == "__main__":
    for name, fn in (("deleted names", deleted_name_hits),
                     ("write lint", write_lint_hits),
                     ("bulk name on a trace slice", bulk_name_on_trace_hits),
                     ("narrow sum over a trace slice", narrow_sum_hits)):
        h = fn()
        print(f"== {name}: {len(h)}")
        for x in h:
            print("  ", x)
