"""tools/affected_cuda.py -- the card checks a change can break (the device
twin of ``tools/affected_tests.py``: the change-based tier for the GPU).

    python tools/affected_cuda.py                     # list them, with why
    python tools/affected_cuda.py --files a.cu b.py   # for these paths
    python tools/affected_cuda.py --run               # run them, a log each
    python tools/affected_cuda.py --why tests/test_cuda_obs.py   # one reach

A CARD CHECK is a test file holding a test that cuda_green's own ``-k``
collects (read from ``tools/cuda_green.py`` by the staging gate's
``cuda_green_select``, never re-typed) and whose source drives the card
through the harness (``tests/_suite_kinds.py``'s cuda pattern). Its REACH is
read from the source, never hand-marked:

- its Python closure, ``affected_tests.closure`` (the wrapper, the
  ``cuda_*_check.py`` it launches, everything they import);
- the module BINDINGS that closure can call: every attribute name, imported
  name, Name and identifier in a string constant that is a binding name of
  ``cpp/src/bindings.cpp`` (an f-string ``getattr(bp, f"get_{b}_backend")``
  matches as a pattern). A module-level binding counts only in a file whose
  text names the module object (``MODULE_REF``): a Python twin's own
  ``atan2_q16`` or a UI field ``obs_masks`` calls nothing. A class member
  counts while its class is LIVE (Python names the class, reached C++ names
  its type, or a live class derives from it);
- the C++ those bindings run, at the grain of a DEFINITION (a function, a
  class with its inline members, a namespace-scope variable, a macro, an
  enum): each binding's body names definitions, each reached definition
  names more, to a fixed point. A name reaches every definition of it that
  could be the one meant -- a header's only inside the naming file's
  ``#include`` closure, a TU's function or variable anywhere (the linker's
  view), an out-of-line member ``C::m`` only while class ``C`` is live in
  the reach (cpp/src has no virtual dispatch); a class's own members are
  never names of their own; ``A::b`` narrows ``b`` to definitions under
  ``A`` when there are any. What runs on every import is reached by every
  check: the module's init calls outside the binding bodies, each
  ``m.attr`` statement, every TU's dynamic namespace-scope initialiser;
- the twins it guards: a twin of ``tests/twin_manifest.toml`` a kernel of
  every one of whose kernel-bearing ``device`` files the reach launches.

A changed path then selects:

- ``.py``: the checks whose Python closure holds it;
- a C++ file: the checks whose reach holds one of its definitions;
- a twin's ``files`` / ``device`` path: also every check guarding the twin;
- a level ``levels/<name>/``: the checks whose closure names ``<name>`` (in
  Python, or in a repo TOML whose file name the closure mentions);
- any other data file: affected_tests' naming rule (its first path
  component named by a Python file of the closure);
- the FULL card set for what touches every check: ``bindings.cpp``, the
  build (``cpp/CMakeLists.txt``, the build scripts, the build-id
  generator), the kits (``fixed_point.h``, ``cuda_fixedpoint_device.cuh``),
  a SHARED header (included by more than half of the TUs),
  ``config.toml``, affected_tests' own "closure cannot see" files, a
  deleted or renamed file, a C++ file the index does not know or cannot
  balance, a data file nothing names.

It may over-select, never under-select. It is the work-in-progress tier for
the card and NEVER the merge gate: a Class-P merge needs ``.cuda_green`` ==
HEAD from a full ``tools/cuda_green.py`` run. Its proof is execution
(tripwires at mutation sites, one full card run, every executing check
selected), and what a static reach cannot see -- a change that alters the
build without running, dispatch the reader does not see, a red that is not
the change's -- are in ``docs/affected_cuda_evidence_2026-11-21.md``.

``--run`` runs each selected check as ``pytest <file> -q -k <cuda_green's
-k>`` (the checks drive the card through ``tests/cuda_harness.py``), one
after another, a log
per check under ``--logs`` ending in ``CHECK_EXIT=<code>``, then a summary;
it exits 1 if any check failed.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
import tomllib
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import affected_tests as at  # noqa: E402  (THE source-reading machinery)
import _suite_kinds  # noqa: E402  (THE device classifier)

CPP = ROOT / "cpp" / "src"
BINDINGS = "cpp/src/bindings.cpp"
MANIFEST = "tests/twin_manifest.toml"
_HEADER_EXT = (".h", ".cuh", ".inc", ".hpp")
_TU_EXT = (".cpp", ".cu")
#: the C++ changes that touch every check.
_CPP_EVERYTHING = re.compile(
    r"^cpp/(CMakeLists\.txt|gen_build_id\.cmake|build_\w*\.(bat|sh)"
    r"|src/bindings\.cpp|src/fixed_point\.h|src/cuda_fixedpoint_device\.cuh)$")
#: data files every check reads.
_DATA_EVERYTHING = re.compile(r"^config\.toml$")
#: a header included by more than this share of the TUs is SHARED.
SHARED_SHARE = 0.5

# ---------------------------------------------------------------------------
# C++ source reading
# ---------------------------------------------------------------------------
_KEYWORDS = frozenset("""
alignas alignof asm auto bool break case catch char class const constexpr
const_cast continue decltype default delete do double dynamic_cast else enum
explicit extern false float for friend goto if inline int long mutable
namespace new noexcept nullptr operator private protected public register
reinterpret_cast return short signed sizeof static static_assert static_cast
struct switch template this throw true try typedef typeid typename union
unsigned using virtual void volatile while override final
__global__ __device__ __host__ __forceinline__ __constant__ __shared__
__restrict__ __launch_bounds__ __align__ define include ifdef ifndef endif
pragma once elif undef defined
""".split())
#: macro-call words that may stand before a declared name's parenthesis.
_ATTR_CALLS = frozenset({"__launch_bounds__", "__align__", "alignas",
                         "decltype", "__attribute__", "__declspec",
                         "static_assert"})
_IDENT = re.compile(r"[A-Za-z_]\w*")
_CHAIN = re.compile(r"[A-Za-z_]\w*(?:\s*::\s*~?[A-Za-z_]\w*)+")
_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)"
                     r"((?:[^\n]*\\\n)*[^\n]*)", re.M)
_PREPRO = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.M)
#: a string or char literal (an escape is a backslash and any char).
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"|' r"'(?:\\.|[^'\\\n])*'")
_CALL = re.compile(r"([A-Za-z_]\w*)\s*(?:<[^<>;{}()]*>\s*)?\(")


def strip_comments(text: str) -> str:
    """C/C++ comments to spaces (newlines kept); literals kept whole."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def blank_strings(text: str) -> str:
    """Literal contents to spaces, the length kept (a bracket inside a
    docstring never unbalances a match)."""
    return _STRING.sub(lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2)
                       + m.group(0)[-1], text)


def strip_strings(text: str) -> str:
    """Literal contents to nothing (the quotes kept)."""
    return _STRING.sub('""', text)


def _match_close(text: str, i: int, open_c: str = "(", close_c: str = ")"):
    """Index just past the bracket closing the one at ``text[i]``."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_c:
            depth += 1
        elif text[j] == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(text)


def _stmt_end(text: str, i: int) -> int:
    """Index past the ``;`` ending the statement at ``i`` (depth 0)."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth <= 0:
            return j + 1
    return len(text)


def _qualified(text: str) -> tuple:
    """``(qualified uses {(A, b)} for each ``A::b`` link of a chain,
    identifiers named at least once with no ``X::`` in front)``."""
    pairs: set = set()
    for m in _CHAIN.finditer(text):
        parts = [x.lstrip("~") for x in re.split(r"\s*::\s*", m.group(0))]
        pairs.update(zip(parts, parts[1:]))
    rest = _CHAIN.sub(lambda m: re.split(r"\s*::", m.group(0))[0], text)
    return frozenset(pairs), frozenset(_IDENT.findall(rest)) - _KEYWORDS


def _calls(text: str) -> set:
    """The names ``text`` calls (an identifier before a ``(``)."""
    return set(_CALL.findall(strip_strings(text))) - _KEYWORDS


def _head_kind(head: str) -> str:
    """What a ``{`` at namespace or class scope opens, from the text
    before it: ``ns``, ``cls``, ``enum``, ``init`` or ``body``."""
    h = head.strip()
    if re.search(r"\bnamespace\b", h) or re.match(r'extern\s*"[^"]*"\s*$', h):
        return "ns"
    h2 = re.sub(r"^template\s*<.*?>\s*", "", h, flags=re.S)
    if re.match(r"^(typedef\s+)?enum\b", h2):
        return "enum"
    if re.match(r"^(typedef\s+)?(struct|class|union)\b", h2) and "(" not in h2:
        return "cls"
    if h.endswith("=") or h.endswith(",") or not h:
        return "init"
    if ")" in h or "]" in h:
        return "body"
    return "init"


def _decl_name(stmt: str):
    """``(name, kind)`` a namespace- or class-scope statement declares --
    kind ``type``, ``func`` or ``var`` -- or None."""
    s = re.sub(r"^\s*template\s*<", "<", stmt, flags=re.S)
    if s.startswith("<"):
        s = s[_match_close(s, 0, "<", ">"):]
    s = s.strip()
    if not s or re.match(r"^namespace\b", s):
        return None          # (a namespace alias declares nothing linkable)
    m = re.match(r"^using\s+(?!namespace)([A-Za-z_]\w*)\s*=", s)
    if m:
        return m.group(1), "type"
    if re.match(r"^(using|friend|public|private|protected)\b", s):
        return None
    m = re.match(r"^(?:typedef\s+)?(?:enum(?:\s+class|\s+struct)?|struct|class"
                 r"|union)\s+(?:alignas\s*\([^)]*\)\s*)?([A-Za-z_]\w*)", s)
    if m and "(" not in s.split(m.group(1), 1)[0]:
        return m.group(1), "type"
    if s.startswith("typedef"):
        ids = _IDENT.findall(re.sub(r"\(.*", "", s))
        return (ids[-1], "type") if ids else None
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            pre = s[:i].rstrip()
            m = re.search(r"(~?[A-Za-z_]\w*)\s*$", pre)
            if m and m.group(1) in _ATTR_CALLS:
                i = _match_close(s, i)
                continue
            if m and m.group(1).lstrip("~") not in _KEYWORDS:
                if "=" in s[:i]:   # a variable whose initialiser calls
                    break
                return m.group(1).lstrip("~"), "func"
            if re.search(r"operator\s*\S*\s*$", pre):
                return None
            break
        if c in "=[{;":
            break
        i += 1
    m = re.match(r"^([^=\[{;(]*)", s)
    ids = [x for x in _IDENT.findall(m.group(1) if m else s)
           if x not in _KEYWORDS]
    return (ids[-1], "var") if ids else None


@dataclass(frozen=True)
class Unit:
    """One definition of a C++ file: what it defines and what it names."""
    file: str
    #: ``(name, qualifiers, member-of class or "", links)`` it defines.
    names: tuple
    uses: frozenset
    quses: frozenset
    unqual: frozenset
    #: runs at module load (a TU's dynamic namespace-scope initialiser).
    init: frozenset = frozenset()
    #: a ``__global__`` kernel (what a twin's device side launches).
    kernel: bool = False


@lru_cache(maxsize=None)
def _source(rel: str) -> str:
    """A C++ file's text, newlines normalised, comments stripped."""
    text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    return strip_comments(text.replace("\r\n", "\n"))


@lru_cache(maxsize=None)
def includes_of(rel: str) -> tuple:
    return tuple(_INCLUDE.findall(_source(rel)))


@lru_cache(maxsize=None)
def namespaces() -> frozenset:
    """Every namespace name cpp/src declares."""
    out = set()
    for rel in cpp_files():
        out |= set(re.findall(r"\bnamespace\s+([A-Za-z_]\w*)", _source(rel)))
    return frozenset(out)


def _make_unit(rel, text, names, init=()) -> Unit:
    body = strip_strings(text)
    q, unq = _qualified(body)
    head = body.split("{", 1)[0]
    return Unit(rel, tuple(names), frozenset(_IDENT.findall(body)) - _KEYWORDS,
                q, unq, frozenset(init), "__global__" in head)


def _links(is_tu, kind, stmt, at_ns, quals) -> bool:
    """LINKAGE (reachable from another TU): a TU's function with a body, or
    its namespace-scope variable that is neither static, const nor
    constexpr; never in an anonymous namespace."""
    if not is_tu or "" in quals:
        return False
    if re.match(r"^\s*(?:template\s*<.*?>\s*)?(?:inline\s+)?static\b", stmt,
                re.S):
        return False
    if kind == "func":
        return True
    const = (re.search(r"\b(constexpr|const)\b", stmt.split("=")[0])
             and not re.search(r"\bextern\b", stmt))
    return kind == "var" and at_ns and not const


@lru_cache(maxsize=None)
def units_of(rel: str) -> tuple:
    """The definitions of one C++ file, each a Unit; ONE unit holding the
    whole file when its braces do not balance (an ``#if`` that opens a
    block twice)."""
    raw = _source(rel)
    is_tu = rel.endswith(_TU_EXT)
    units = []
    for m in _DEFINE.finditer(raw):
        units.append(_make_unit(rel, m.group(2), [(m.group(1), (), "", False)]))
    text = blank_strings(_PREPRO.sub(lambda m: re.sub(r"[^\n]", " ",
                                                      m.group(0)), raw))
    if text.count("{") != text.count("}"):
        return (_make_unit(rel, raw, []),) + tuple(units)
    stack: list = []          # namespace names (and "" for anonymous)
    start, i, n = 0, 0, len(text)
    while i < n:
        c = text[i]
        if c == "}":
            if stack:
                stack.pop()
            start = i + 1
        elif c in ";{":
            head = text[start:i]
            kind = _head_kind(head) if c == "{" else "decl"
            if kind == "ns":
                mm = re.search(r"namespace\s+([A-Za-z_]\w*)", head)
                stack.append(mm.group(1) if mm else
                             ("" if "namespace" in head else "extern"))
                start = i + 1
                i += 1
                continue
            end = i + 1
            if c == "{":
                close = _match_close(text, i, "{", "}")
                end = close if kind == "body" else _stmt_end(text, close)
            chunk_raw = raw[start:end]
            names, init = _names_of(rel, text[start:end], head, kind, stack,
                                    is_tu)
            if names or init or chunk_raw.strip(" \t\n;"):
                units.append(_make_unit(rel, chunk_raw, names, init))
            start = i = end
            continue
        i += 1
    return tuple(units)


def _names_of(rel, chunk, head, kind, stack, is_tu):
    """``(names, init calls)`` one namespace-scope chunk defines."""
    quals = tuple(stack)
    at_ns = True
    got = _decl_name(head if kind != "decl" else chunk.rstrip(";"))
    names, init = [], set()
    if not got and kind != "enum":
        return names, init
    name, dkind = got or ("", "type")
    if kind == "enum":
        mm = re.search(r"\benum\s+(?:class\s+|struct\s+)?([A-Za-z_]\w*)", head)
        if mm:
            names.append((mm.group(1), quals, "", False))
        body = chunk[chunk.find("{") + 1:chunk.rfind("}")]
        for part in body.split(","):
            mm = re.match(r"\s*([A-Za-z_]\w*)", part)
            if mm:
                names.append((mm.group(1), quals, "", False))
        return names, init
    if kind == "cls" or (dkind == "type" and kind == "decl"):
        # a class is reached through its NAME; its members (fields, inline
        # methods) come with it and are never names of their own -- a bare
        # local ``h`` must not reach every struct with a field ``h``.
        names.append((name, quals, "", False))
        return names, init
    if dkind == "func":
        if kind != "body" and is_tu:
            return names, init      # a TU's prototype defines nothing
        member = ""
        q = quals
        mq = re.search(r"([A-Za-z_]\w*)\s*::\s*~?([A-Za-z_]\w*)\s*$",
                       head.split("(")[0] if kind == "body" else
                       chunk.split("(")[0])
        if mq and mq.group(2) == name:
            q = quals + (mq.group(1),)
            if mq.group(1) not in namespaces():
                member = mq.group(1)
        links = kind == "body" and _links(is_tu, "func", head, at_ns, quals)
        names.append((name, q, member, links))
        return names, init
    names.append((name, quals, "",
                  _links(is_tu, dkind, chunk, at_ns, quals)))
    if is_tu and dkind == "var" and not re.search(r"\bconstexpr\b", chunk):
        init = _calls(chunk.split("=", 1)[-1] if "=" in chunk else chunk)
    return names, init


@lru_cache(maxsize=None)
def cpp_files() -> tuple:
    exts = _HEADER_EXT + _TU_EXT
    return tuple(sorted(at._rel(p) for p in CPP.iterdir()
                        if p.is_file() and p.suffix in exts))


@lru_cache(maxsize=None)
def include_closure(rel: str) -> frozenset:
    seen, todo = {rel}, [rel]
    while todo:
        for inc in includes_of(todo.pop()):
            cand = CPP / inc
            if cand.is_file():
                hit = at._rel(cand)
                if hit not in seen:
                    seen.add(hit)
                    todo.append(hit)
    return frozenset(seen)


@lru_cache(maxsize=None)
def all_units() -> tuple:
    """Every unit of cpp/src but bindings.cpp's module body."""
    out = []
    for rel in cpp_files():
        for u in units_of(rel):
            if rel == BINDINGS and any(nm == "PYBIND11_MODULE"
                                       for nm, *_ in u.names):
                continue
            out.append(u)
    return tuple(out)


@lru_cache(maxsize=None)
def _def_index() -> dict:
    """name -> ((unit index, qualifiers, member-of, links), ...)."""
    idx: dict = {}
    for k, u in enumerate(all_units()):
        for name, quals, member, links in u.names:
            idx.setdefault(name, []).append((k, quals, member, links))
    return idx


@lru_cache(maxsize=None)
def _defined_names() -> tuple:
    """Per unit, the names it defines (a class is LIVE in a reach when a
    reached unit defines its name)."""
    return tuple(frozenset(nm for nm, *_ in u.names) for u in all_units())


def _targets(rel: str, uses, quses, unqual) -> tuple:
    """``(plain, members)`` the names ``uses`` (written in file ``rel``) can
    mean. PLAIN unit indices: a header's definition inside ``rel``'s
    include closure, a TU's linkage function or variable anywhere. MEMBERS
    ``(index, class)``: an out-of-line member function ``C::m``, reached
    only while class ``C`` is live in the reach (cpp/src has no virtual
    dispatch: a member runs only on an object of its class, whose type the
    reached code must name). A qualified use ``A::b`` narrows ``b`` to
    definitions under ``A`` when ``b`` is never named unqualified and such
    definitions exist."""
    inc = include_closure(rel)
    units = all_units()
    idx = _def_index()
    qmap: dict = {}
    for a, b in quses:
        qmap.setdefault(b, set()).add(a)
    plain, members = set(), set()
    for u in uses:
        cands = [(k, quals, member) for k, quals, member, links
                 in idx.get(u, ()) if units[k].file in inc or links]
        if u in qmap and u not in unqual:
            narrowed = [c for c in cands if set(c[1]) & qmap[u]]
            if narrowed:
                cands = narrowed
        for k, _, member in cands:
            if member:
                members.add((k, member))
            else:
                plain.add(k)
    return frozenset(plain), frozenset(members)


@lru_cache(maxsize=None)
def _unit_edges(k: int) -> tuple:
    u = all_units()[k]
    plain, members = _targets(u.file, u.uses, u.quses, u.unqual)
    return plain - {k}, members


def _close_units(seeds) -> frozenset:
    """The units ``seeds`` (``(plain, members)``) reach, to a fixed point:
    a member waits until its class is live."""
    plain, members = seeds
    names = _defined_names()
    seen: set = set()
    live: set = set()
    waiting: dict = {}
    todo = deque()

    def take(k):
        if k in seen:
            return
        seen.add(k)
        todo.append(k)
        for nm in names[k] - live:
            live.add(nm)
            for w in waiting.pop(nm, ()):
                take(w)

    def offer(k, cls):
        if cls in live:
            take(k)
        else:
            waiting.setdefault(cls, []).append(k)

    for k in plain:
        take(k)
    for k, cls in members:
        offer(k, cls)
    while todo:
        p, m = _unit_edges(todo.popleft())
        for g in p:
            take(g)
        for g, cls in m:
            offer(g, cls)
    return frozenset(seen)


@lru_cache(maxsize=None)
def shared_headers() -> frozenset:
    """Headers included (transitively) by more than SHARED_SHARE of the
    TUs: a change to one touches every check."""
    tus = [r for r in cpp_files() if r.endswith(_TU_EXT)]
    count: dict = {}
    for t in tus:
        for h in include_closure(t) - {t}:
            count[h] = count.get(h, 0) + 1
    return frozenset(h for h, k in count.items() if k > SHARED_SHARE * len(tus))


# ---------------------------------------------------------------------------
# The bindings
# ---------------------------------------------------------------------------
_DEF_CALL = re.compile(
    r"\.def(?:_static|_readonly|_readwrite|_property|_property_readonly"
    r"|_readonly_static|_readwrite_static|_property_static"
    r"|_property_readonly_static)?\s*\(\s*(?:\"(\w+)\"|(py::init))")
_ATTR = re.compile(r"\bm\.attr\(\s*\"(\w+)\"\s*\)\s*=")
_CLASS = re.compile(r"py::(?:class_|enum_)\s*<")
_VALUE = re.compile(r"\.value\(\s*\"(\w+)\"")


@dataclass
class Bindings:
    #: Python name -> ``((owner, text), ...)``: the C++ text each binding of
    #: that name runs, ``owner`` the Python class it is a member of ("" for
    #: a module-level def or attr; a class's constructor is the class name).
    bodies: dict = field(default_factory=dict)
    #: bound class (Python name) -> (its C++ type, its bound C++ bases).
    classes: dict = field(default_factory=dict)
    #: the module body outside every binding and local helper.
    residue: str = ""
    #: a module block's local struct or lambda: name -> text.
    helpers: dict = field(default_factory=dict)
    #: every ``m.attr(...) = ...`` statement (each runs on import).
    attrs: list = field(default_factory=list)


@lru_cache(maxsize=None)
def bindings() -> Bindings:
    raw = _source(BINDINGS)
    b = Bindings()
    mod = re.search(r"PYBIND11_MODULE\s*\(\s*\w+\s*,\s*m\s*\)\s*\{", raw)
    if not mod:
        raise RuntimeError("bindings.cpp has no PYBIND11_MODULE")
    body = raw[mod.end():_match_close(blank_strings(raw), mod.end() - 1,
                                      "{", "}") - 1]
    # brackets are matched on ``bl`` (literals blanked, same length), names
    # read from ``body``.
    bl = blank_strings(body)
    spans: list = []
    class_of: list = []   # (start, stmt_end, header_text, pyname)
    for m in _CLASS.finditer(bl):
        targ_end = _match_close(bl, m.end() - 1, "<", ">")
        paren = bl.find("(", targ_end)
        call_end = _match_close(bl, paren)
        header = bl[m.start():call_end]
        pm = re.search(r"\"(\w+)\"", body[paren:call_end])
        pyname = pm.group(1) if pm else ""
        targs = [re.split(r"::", a.strip())[-1].strip()
                 for a in bl[m.end():targ_end - 1].split(",")]
        b.classes[pyname] = (targs[0], tuple(targs[1:]))
        class_of.append((m.start(), _stmt_end(bl, call_end), header, pyname))
        spans.append((m.start(), call_end))
        b.bodies.setdefault(pyname, []).append(("", header))

    def owner(pos):
        cls = [c for c in class_of if c[0] <= pos < c[1]]
        return cls[-1] if cls else None

    for m in _DEF_CALL.finditer(body):
        close = _match_close(bl, body.find("(", m.start()))
        cls = owner(m.start())
        name = m.group(1) or (cls[3] if cls else "")
        own = "" if not cls or m.group(2) else cls[3]
        b.bodies.setdefault(name, []).append(
            (own, (cls[2] if cls else "") + "\n" + bl[m.start():close]))
        spans.append((m.start(), close))
    for m in _VALUE.finditer(body):
        close = _match_close(bl, body.find("(", m.start()))
        cls = owner(m.start())
        b.bodies.setdefault(m.group(1), []).append(
            (cls[3] if cls else "",
             (cls[2] if cls else "") + "\n" + bl[m.start():close]))
        spans.append((m.start(), close))
    for m in _ATTR.finditer(body):
        close = _stmt_end(bl, m.end())
        b.bodies.setdefault(m.group(1), []).append(("", bl[m.start():close]))
        b.attrs.append(bl[m.start():close])
        spans.append((m.start(), close))
    spans.sort()
    keep, pos = [], 0
    for s, e in spans:
        if s > pos:
            keep.append(bl[pos:s])
        pos = max(pos, e)
    keep.append(bl[pos:])
    residue = "".join(keep)
    for rx in (r"\bstruct\s+([A-Za-z_]\w*)[^;{]*\{",
               r"\bauto\s+([A-Za-z_]\w*)\s*=\s*\[[^\]]*\]\s*\("):
        for m in list(re.finditer(rx, residue)):
            at_end = residue[m.end() - 1]
            brace = residue.find("{", m.end() - 1 if at_end == "{" else
                                 _match_close(residue, m.end() - 1))
            stop = _stmt_end(residue, _match_close(residue, brace, "{", "}"))
            b.helpers[m.group(1)] = (b.helpers.get(m.group(1), "")
                                     + residue[m.start():stop])
            residue = (residue[:m.start()] + " " * (stop - m.start())
                       + residue[stop:])
    b.residue = residue
    b.bodies = {k: tuple(v) for k, v in b.bodies.items() if k}
    return b


def _expand_helpers(names: set) -> set:
    """Identifiers plus those of every module-local helper they name."""
    hb = bindings().helpers
    out, todo = set(names), list(names)
    while todo:
        n = todo.pop()
        if n in hb:
            for x in set(_IDENT.findall(strip_strings(hb[n]))) - _KEYWORDS:
                if x not in out:
                    out.add(x)
                    todo.append(x)
    return out


@lru_cache(maxsize=None)
def _seeds_of_text(text: str) -> tuple:
    """``(plain, members)`` C++ ``text`` written in bindings.cpp names
    (through a module-local helper it names, too)."""
    body = strip_strings(text)
    ids = _expand_helpers(set(_IDENT.findall(body)) - _KEYWORDS)
    q, unq = _qualified(body)
    unq = unq | (ids - set(_IDENT.findall(body)))
    return _targets(BINDINGS, ids, q, unq)


@lru_cache(maxsize=None)
def _units_of_texts(texts: frozenset) -> frozenset:
    """The units a set of bindings.cpp texts reaches TOGETHER (a class one
    makes live serves another's member calls)."""
    plain, members = set(), set()
    for t in texts:
        p, m = _seeds_of_text(t)
        plain |= p
        members |= m
    return _close_units((frozenset(plain), frozenset(members)))


def _files(units) -> frozenset:
    return frozenset({BINDINGS} | {all_units()[k].file for k in units})


def _reach_of_text(text: str) -> frozenset:
    """The C++ files C++ ``text`` written in bindings.cpp reaches (with what
    every import runs)."""
    return _files(_units_of_texts(frozenset({text, _global_text()})))


@lru_cache(maxsize=None)
def _global_text() -> str:
    """What every import of the module RUNS, as C++ text to reach from: the
    calls of the module's init code outside every binding and helper, of
    every ``m.attr`` statement, and of every TU's dynamic namespace-scope
    initialiser. (A name only mentioned -- a constant, a type -- runs
    nothing.)"""
    b = bindings()
    calls = _calls(b.residue)
    for a in b.attrs:
        calls |= _calls(a)
    for u in all_units():
        calls |= u.init
    return " ".join(sorted(f"{c}()" for c in calls))


@lru_cache(maxsize=None)
def binding_reach(name: str) -> frozenset:
    """The C++ files every binding of one name runs."""
    return _reach_of_text("\n".join(t for _, t in
                                    bindings().bodies.get(name, ())))


@lru_cache(maxsize=None)
def _unit_names(units: frozenset) -> frozenset:
    out: set = set()
    for k in units:
        out |= all_units()[k].uses
    return frozenset(out)


@lru_cache(maxsize=None)
def reach_of_hits(hits: frozenset) -> tuple:
    """``(units, live classes)`` a set of hit binding names runs. A
    module-level binding runs when hit; a class member only while its class
    is LIVE: Python names the class, or code already reached names its C++
    type (an instance can come back from it: a returned object, a member
    exposed by ``def_readonly``), or a live class derives from it. Followed
    to a fixed point, starting from what every import runs."""
    b = bindings()
    by_type: dict = {}
    for py, (ctype, _) in b.classes.items():
        by_type.setdefault(ctype, set()).add(py)
    texts = {_global_text()}
    live = {py for py in b.classes if py in hits}
    while True:
        units = _units_of_texts(frozenset(texts))
        names = _unit_names(units)
        for ctype, pys in by_type.items():
            if ctype in names:
                live |= pys
        for py in list(live):
            for base in b.classes[py][1]:
                live |= by_type.get(base, set())
        new = {t for n in hits for own, t in b.bodies.get(n, ())
               if not own or own in live} - texts
        if not new:
            return units, frozenset(live)
        texts |= new


# ---------------------------------------------------------------------------
# What a Python file can call on the module
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def py_names(rel: str) -> tuple:
    """``(names, patterns)``: every attribute name, imported name, Name and
    identifier in a string constant of a Python file; f-strings as regexes
    (their formatted parts any identifier text)."""
    text = at._text(rel)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset(_IDENT.findall(text)), ()
    names: set = set()
    pats: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.update(node.name.split("."))
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(_IDENT.findall(node.value))
        elif isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(re.escape(v.value))
                else:
                    parts.append(r"\w*")
            pat = "".join(parts)
            if re.search(r"[A-Za-z_]", pat.replace(r"\w*", "")):
                pats.add("^" + pat + "$")
    return frozenset(names), tuple(sorted(pats))


@lru_cache(maxsize=None)
def _file_hits(rel: str) -> frozenset:
    """The binding names one Python file can call."""
    names, pats = py_names(rel)
    known = bindings().bodies
    hit = {n for n in names if n in known}
    for p in pats:
        rx = re.compile(p)
        hit |= {k for k in known if rx.search(k)}
    return frozenset(hit)


# ---------------------------------------------------------------------------
# The card checks
# ---------------------------------------------------------------------------
_CUDA_SRC = re.compile(_suite_kinds._CUDA)


@lru_cache(maxsize=None)
def card_select() -> str:
    """The ``-k`` expression ``tools/cuda_green.py`` runs, read from its own
    source through the staging gate's reader (never re-typed)."""
    import staging_gate
    return staging_gate.cuda_green_select(ROOT / "tools" / "cuda_green.py")


@lru_cache(maxsize=None)
def card_tests(rel: str) -> tuple:
    """The tests of one file that cuda_green's ``-k`` collects and that
    drive the card. A ``-k`` that is one word matches as pytest's does (the
    word in the module, class or test name, any case); any other expression
    counts every harness-launching test (over-selects, never misses)."""
    try:
        tree = ast.parse(at._text(rel))
    except SyntaxError:
        return ()
    word = card_select().lower()
    plain = re.fullmatch(r"\w+", word) is not None
    in_name = not plain or word in Path(rel).stem.lower()
    out = []

    def consider(fn, cls_named=False):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and fn.name.startswith("test") \
                and (in_name or cls_named or word in fn.name.lower()) \
                and _CUDA_SRC.search(_suite_kinds.source_of(
                    str(ROOT / rel), fn.name)):
            out.append(fn.name)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                consider(sub, word in node.name.lower())
        else:
            consider(node)
    return tuple(out)


@lru_cache(maxsize=None)
def checks() -> tuple:
    """Every card check (a test file), sorted."""
    return tuple(t for t in at.test_files() if card_tests(t))


@lru_cache(maxsize=None)
def py_closure(check: str) -> frozenset:
    return frozenset(x for x in at.closure(check) if x != at.NATIVE)


#: a Python file that can hold the module object names it so (the guard
#: test_the_module_never_reaches_a_file_that_does_not_name_it pins it).
MODULE_REF = re.compile(r"\bbp\b|\b_bp\w*|breach_physics")


@lru_cache(maxsize=None)
def _module_only() -> frozenset:
    """Bindings bound only at module level (a def, an attr, a class's
    constructor): callable only through the module object."""
    return frozenset(n for n, rows in bindings().bodies.items()
                     if all(not own for own, _ in rows))


@lru_cache(maxsize=None)
def bindings_of(check: str) -> frozenset:
    """The module bindings a check's closure can call. A module-level
    binding counts only in a file that names the module object: a Python
    twin's own ``atan2_q16`` or a UI field ``obs_masks`` calls nothing. A
    member counts anywhere (an object of its class can be handed to any
    file); its class's liveness gates it."""
    hit: set = set()
    for rel in py_closure(check):
        names = _file_hits(rel)
        if not MODULE_REF.search(at._text(rel)):
            names = names - _module_only()
        hit |= names
    return frozenset(hit)


@lru_cache(maxsize=None)
def cpp_reach(check: str) -> frozenset:
    """Every C++ file one of whose definitions a check can run."""
    return _files(reach_of_hits(bindings_of(check))[0])


# ---------------------------------------------------------------------------
# The twins
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def twins() -> dict:
    """twin -> (files, device) of tests/twin_manifest.toml."""
    m = tomllib.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    return {k: (tuple(v.get("files", ())), tuple(v.get("device", ())))
            for k, v in m.get("twins", {}).items()}


@lru_cache(maxsize=None)
def guards(check: str) -> frozenset:
    """The twins a check guards: its reach launches a kernel (a
    ``__global__`` definition) in EVERY one of the twin's ``device`` files
    that holds one. Any one file says too little: a twin lists the shared
    store and event TUs beside its own kernel, and every check runs the
    observation's layout fingerprint at import."""
    units = reach_of_hits(bindings_of(check))[0]
    launched = {all_units()[k].file for k in units if all_units()[k].kernel}
    out = set()
    for t, (_, dev) in twins().items():
        own = {d for d in dev if d in _kernel_files()}
        if own and own <= launched:
            out.add(t)
    return frozenset(out)


@lru_cache(maxsize=None)
def _kernel_files() -> frozenset:
    return frozenset(u.file for u in all_units() if u.kernel)


# ---------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------
@dataclass
class Selection:
    full: bool = False
    checks: set = field(default_factory=set)
    #: changed path -> why (how many checks it reached, or why FULL).
    why: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def _toml_texts() -> tuple:
    out = []
    for p in ROOT.rglob("*.toml"):
        parts = p.relative_to(ROOT).parts
        if at._SKIP_DIRS.intersection(parts) or parts[0] == "levels":
            continue
        out.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    return tuple(out)


def _names_level(check: str, name: str) -> bool:
    word = re.compile(r"\b" + re.escape(name) + r"\b")
    texts = [at._text(f) for f in py_closure(check)]
    if any(word.search(t) for t in texts):
        return True
    blob = "\n".join(texts)
    return any(word.search(t) and fname in blob for fname, t in _toml_texts())


def reach(path: str) -> tuple:
    """``(checks, why, full)`` for ONE changed path: the card checks it
    reaches, the reason, and whether it touches every check. A C++ path,
    ``config.toml`` and a level are this module's; every other path is
    ``affected_tests.reach`` over the card checks (THE per-path rule for
    Python and data, never a copy)."""
    every = checks()
    if (ROOT / path).exists() and (_DATA_EVERYTHING.search(path)
                                   or _CPP_EVERYTHING.search(path)):
        return set(), "FULL: touches every check", True
    if path.startswith("cpp/") and (ROOT / path).exists():
        if path not in cpp_files():
            return set(), "FULL: a cpp/ file the index does not know", True
        if path in shared_headers():
            return set(), "FULL: a shared header", True
        if _unbalanced(path):
            return set(), "FULL: its braces do not balance", True
        hit = {c for c in every if path in cpp_reach(c)}
    elif path.startswith("levels/") and path.count("/") >= 2 \
            and (ROOT / path).exists():
        name = path.split("/")[1]
        hit = {c for c in every if _names_level(c, name)}
    else:
        hit, why, full = at.reach(path, every)
        if full or not hit and why.startswith("inert"):
            return set(hit), why, full
        hit = set(hit)
    for t, (files, dev) in twins().items():
        if path in files or path in dev:
            hit |= {c for c in every if t in guards(c)}
    return hit, f"{len(hit)} card check(s)", False


def select(changed) -> Selection:
    """The card checks ``changed`` (repo-relative posix paths) can break."""
    sel = Selection()
    for path in sorted(set(changed)):
        hit, why, full = reach(path)
        sel.full |= full
        sel.checks |= hit
        sel.why[path] = why
    if sel.full:
        sel.checks = set(checks())
    return sel


def _unbalanced(rel: str) -> bool:
    text = blank_strings(_PREPRO.sub("", _source(rel)))
    return text.count("{") != text.count("}")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run(selected, logs: Path, extra=()) -> int:
    """Each check as ``pytest <file> -q -k <cuda_green's -k>``, one after another, a log
    each ending in ``CHECK_EXIT=<code>``; 1 if any failed."""
    logs.mkdir(parents=True, exist_ok=True)
    bad = []
    t_all = time.perf_counter()
    for i, check in enumerate(sorted(selected), 1):
        log = logs / (Path(check).stem + ".log")
        t0 = time.perf_counter()
        print(f"[{i}/{len(selected)}] {check} ...", end=" ", flush=True)
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", check, "-q", "-k", card_select(),
                 "-p", "no:cacheprovider", *extra],
                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
            dt = time.perf_counter() - t0
            fh.write(f"\nCHECK_SECONDS={dt:.1f}\nCHECK_EXIT={proc.returncode}\n")
        print(f"exit {proc.returncode} ({dt:.0f} s)", flush=True)
        if proc.returncode != 0:
            bad.append(check)
    print(f"{len(selected)} check(s) in {time.perf_counter() - t_all:.0f} s; "
          f"{len(bad)} failed{': ' + ', '.join(bad) if bad else ''}")
    print(f"logs: {logs}")
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="affected_cuda",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main",
                    help="compare against the merge-base with this ref")
    ap.add_argument("--files", nargs="+",
                    help="these paths instead of the diff against --base")
    ap.add_argument("--run", action="store_true",
                    help="run the selection, a log per check")
    ap.add_argument("--logs", default=str(ROOT / "out" / "affected_cuda"),
                    help="where --run writes its logs")
    ap.add_argument("--why", metavar="CHECK",
                    help="print one check's reach and exit")
    args = ap.parse_args(argv)
    if args.why:
        c = args.why.replace("\\", "/")
        print(f"{c}: {len(py_closure(c))} Python files, "
              f"{len(bindings_of(c))} bindings, "
              f"{len(cpp_reach(c))} C++ files; guards {sorted(guards(c))}")
        for f in sorted(cpp_reach(c)):
            print(f"    {f}")
        return 0
    changed = args.files if args.files else at.changed_files(args.base)
    sel = select([p.replace("\\", "/") for p in changed])
    for path, why in sel.why.items():
        print(f"  {path}: {why}")
    chosen = sorted(sel.checks)
    print(f"{'FULL CARD SET: ' if sel.full else ''}{len(chosen)} of "
          f"{len(checks())} card checks")
    for c in chosen:
        print(f"    {c}")
    if not args.run:
        return 0
    if not chosen:
        print("nothing to run")
        return 0
    return run(chosen, Path(args.logs), extra)


if __name__ == "__main__":
    raise SystemExit(main())
