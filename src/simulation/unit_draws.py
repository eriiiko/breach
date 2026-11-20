"""THE DEVICE DRAWS' Python face (#29 P7.0a; residency design 4.4c decision 5).

The device forms of :meth:`simulation.draws.MatchDraws.at` / ``.next``,
:func:`simulation.draws.cone_offset_q16` and :func:`~simulation.draws.roll_q16`
live in ``cpp/src/cuda_unit_draws.cuh`` over the vendored Philox
(``cpp/src/philox32.h``). This module is what they need from the Python side:

* THE SALTS reach C++ as GENERATED constants (``cpp/src/unit_draws_layout.h``,
  :func:`header_text`, written by ``tools/gen_unit_layout.py --write``,
  fingerprint-checked against the compiled module by :func:`check_compiled`):
  a ``Salt`` member is never typed by hand in a ``.cu``.
* THE KEY AND THE TICK reach a kernel as ONE per-env row ``(seed_lo, seed_hi,
  tick_word)`` int64 (:func:`draw_key` / :func:`draw_keys`): ``sim.draws.key``
  and ``sim.draws.tick & 0xFFFFFFFF`` -- the tick a :class:`~simulation.draws.
  Draw` puts in its counter's first word (the legacy WEGO ``tick_base``
  included), read at the slot's entry. A batch stacks the rows ``(N, 3)``;
  thread (env e, row r) reads row e.
* ``.next``'s device form is a COUNTER LOCAL TO THE ROW'S THREAD (one per
  site): exact while every ``.next`` user of one salt sits in ONE kernel per
  tick. :data:`NEXT_SITES` records the ONE call site in ``src/`` of each salt
  drawn with ``.next`` (part of the header and its fingerprint);
  :func:`next_sites` scans ``src/`` for every ``.next(Salt.X, ...)`` call and
  :func:`next_site_problems` names a second site (``tests/test_unit_draws.py``
  goes red) -- a second site would restart the device count silently.

NOT PART OF THE PLAYED TICK: Class T (training only). Import-light.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulation.rng_salts import Salt

#: The generated header (C++'s copy of the salts and the ``.next`` sites).
HEADER_REL = "cpp/src/unit_draws_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"

#: A draw key row: (seed_lo, seed_hi, tick_word).
KEY_FIELDS = ("seed_lo", "seed_hi", "tick")
KEY_W = len(KEY_FIELDS)
_M32 = (1 << 32) - 1

#: #29 P7e.1: THE TICK ROW, one per env on the device, written ONCE per tick
#: (the host latch while host gameplay is in the loop) and read by every unit
#: kernel in place of per-launch tick / key arguments: ``tick`` (``sim.tick``
#: at the moment the slot runs: the tick-end slots read it after the
#: advance), the draw key (:data:`KEY_FIELDS`' three columns, in that order,
#: from :data:`TICK_ROW_KEY`), and ``tiles_changed``, the device form of
#: ``GameMap._tile_serial`` (seeded from the host, bumped ON THE DEVICE by
#: the topology seam's ``on_tile_changed``; the horde cache's refresh rule
#: reads it).
TICK_ROW_FIELDS = ("tick", "seed_lo", "seed_hi", "draw_tick", "tiles_changed")
TICK_ROW_W = len(TICK_ROW_FIELDS)
TICK_ROW_KEY = TICK_ROW_FIELDS.index("seed_lo")
TICK_ROW_TILES = TICK_ROW_FIELDS.index("tiles_changed")


@dataclass(frozen=True)
class NextSite:
    """The ONE ``src/`` call of ``.next`` for ``salt``: in ``module`` (a path
    relative to ``src/``), inside the function ``function``."""
    salt: str
    module: str
    function: str


#: Every salt drawn with ``MatchDraws.next``, with its ONE call site (design
#: 4.4c decision 5: SHOT_CONE and BEAM_CONE are one shooter's draws in slot 4;
#: MELEE_CRIT is the order path's melee, which the device door refuses).
NEXT_SITES: tuple = (
    NextSite("SHOT_CONE", "simulation/combat.py", "fire_burst"),
    NextSite("BEAM_CONE", "simulation/combat.py", "fire_beam"),
    NextSite("MELEE_CRIT", "simulation/combat.py", "melee_blow"),
)


# ---------------------------------------------------------------------------
# The key and the tick
# ---------------------------------------------------------------------------
def draw_key(sim) -> np.ndarray:
    """One env's draw key row ``(seed_lo, seed_hi, tick_word)`` (``(3,)``
    int64): what every ``sim.draws`` draw of this tick puts in its key and
    its counter's first word."""
    d = sim.draws
    lo, hi = d.key
    return np.array([int(lo), int(hi), int(d.tick) & _M32], dtype=np.int64)


def draw_keys(sims) -> np.ndarray:
    """``(N, 3)``: env e's row is :func:`draw_key` of ``sims[e]``."""
    sims = list(sims)
    out = np.zeros((len(sims), KEY_W), dtype=np.int64)
    for e, s in enumerate(sims):
        out[e] = draw_key(s)
    return out


# ---------------------------------------------------------------------------
# #29 P7e.1: the tick row
# ---------------------------------------------------------------------------
def tick_row(sim) -> np.ndarray:
    """One env's host TICK ROW ``(TICK_ROW_W,)`` int64 at this moment: the
    latch the host writes once per tick (``sim.tick``, :func:`draw_key`,
    ``GameMap._tile_serial``)."""
    row = np.zeros(TICK_ROW_W, dtype=np.int64)
    row[0] = int(sim.tick)
    row[TICK_ROW_KEY:TICK_ROW_KEY + KEY_W] = draw_key(sim)
    row[TICK_ROW_TILES] = int(getattr(sim.gmap, "_tile_serial", 0))
    return row


def tick_rows(ticks, keys=None, tiles=None) -> np.ndarray:
    """``(N, TICK_ROW_W)`` host rows from per-env columns: ``ticks`` ``(N,)``,
    ``keys`` ``(N, 3)`` draw keys (default zeros), ``tiles`` ``(N,)`` the
    tile-change counts (default zeros) -- a gate's synthetic latch."""
    t = np.asarray(ticks, dtype=np.int64).reshape(-1)
    out = np.zeros((t.shape[0], TICK_ROW_W), dtype=np.int64)
    out[:, 0] = t
    if keys is not None:
        out[:, TICK_ROW_KEY:TICK_ROW_KEY + KEY_W] = np.asarray(
            keys, dtype=np.int64).reshape(-1, KEY_W)
    if tiles is not None:
        out[:, TICK_ROW_TILES] = np.asarray(tiles, dtype=np.int64).reshape(-1)
    return out


class TickRow:
    """THE TICK ROW on the device: ``(N, TICK_ROW_W)`` int64 (a CuPy array),
    written once per tick by :meth:`write` (the host latch) and read in place
    by every unit kernel (its address :attr:`ptr`); the ``tiles_changed``
    column moves on the device (the topology seam bumps it). :meth:`env` is
    env e's row alone, a view of the same memory (an N = 1 run)."""

    def __init__(self, n_env: int | None = None, _buf=None):
        import cupy as cp
        if _buf is not None:
            self.buf = _buf
        else:
            if n_env is None or int(n_env) < 1:
                raise ValueError("tick row: n_env must be >= 1")
            self.buf = cp.zeros((int(n_env), TICK_ROW_W), dtype=np.int64)
        self.n_env = int(self.buf.shape[0])

    @classmethod
    def of(cls, rows) -> "TickRow":
        """A row holding host ``rows`` ``(N, TICK_ROW_W)`` (:func:`tick_rows`)."""
        r = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, TICK_ROW_W)
        t = cls(r.shape[0])
        t.write(r)
        return t

    @classmethod
    def of_sims(cls, sims) -> "TickRow":
        """A row holding each sim's :func:`tick_row`."""
        return cls.of(np.stack([tick_row(s) for s in sims]))

    def write(self, rows) -> None:
        """THE LATCH: every env's row from host ``rows`` (one H2D, in place)."""
        r = np.ascontiguousarray(rows, dtype=np.int64).reshape(self.n_env, TICK_ROW_W)
        self.buf.set(r)

    @property
    def ptr(self) -> int:
        return int(self.buf.data.ptr)

    def rows(self) -> np.ndarray:
        """The rows ``(N, TICK_ROW_W)`` on the host (a download)."""
        return self.buf.get()

    def tiles(self) -> np.ndarray:
        """``(N,)`` the tile-change counts (a download)."""
        return self.rows()[:, TICK_ROW_TILES].copy()

    def env(self, e: int) -> "TickRow":
        """Env ``e``'s row alone: a view of the same device memory."""
        return TickRow(_buf=self.buf[int(e):int(e) + 1])


def row_ptr(trow, n_env: int, who: str) -> int:
    """A kernel face's tick-row argument checked: a :class:`TickRow` for
    ``n_env`` envs (never per-launch host ticks, #29 P7e.1); its address."""
    if not isinstance(trow, TickRow):
        raise TypeError(f"{who}: the tick row must be a unit_draws.TickRow (the "
                        f"device row every kernel reads), not {type(trow).__name__}")
    if trow.n_env != int(n_env):
        raise ValueError(f"{who}: the tick row holds {trow.n_env} envs, the store "
                         f"{int(n_env)}")
    return trow.ptr


# ---------------------------------------------------------------------------
# The one site per .next salt
# ---------------------------------------------------------------------------
def _salt_of(node):
    """The ``Salt`` member a call argument names (``Salt.X``, ``_Salt.X``,
    ``rng_salts.Salt.X``), or None."""
    if isinstance(node, ast.Attribute) and node.attr in Salt.__members__:
        return node.attr
    return None


def next_sites(root=None) -> list:
    """Every ``<obj>.next(Salt.X, ...)`` call under ``src/`` (``root``:
    the repo; default this checkout) as ``(salt, module, function, line)``,
    ``module`` relative to ``src/`` with forward slashes, ``function`` the
    innermost enclosing ``def`` (``<module>`` at top level)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    src = root / "src"
    out = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        out.extend(sites_in_source(path.read_text(encoding="utf-8"), rel))
    return out


def sites_in_source(text: str, module: str) -> list:
    """:func:`next_sites` over one module's source text."""
    tree = ast.parse(text)
    out = []

    def walk(node, func):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef,
                                                 ast.AsyncFunctionDef)) else func
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "next" and child.args):
                s = _salt_of(child.args[0])
                if s is not None:
                    out.append((s, module, func, child.lineno))
            walk(child, f)

    walk(tree, "<module>")
    return out


def next_site_problems(sites) -> list:
    """The ways ``sites`` (:func:`next_sites`) break :data:`NEXT_SITES`, one
    string each: a salt drawn with ``.next`` at a second site, a salt drawn
    with ``.next`` that the table does not name, a recorded site that is gone
    or moved to another function."""
    table = {s.salt: s for s in NEXT_SITES}
    by_salt: dict = {}
    for salt, module, func, line in sites:
        by_salt.setdefault(salt, []).append((module, func, line))
    out = []
    for salt, where in sorted(by_salt.items()):
        rec = table.get(salt)
        if rec is None:
            out.append(f"Salt.{salt} is drawn with .next at {where} but "
                       f"unit_draws.NEXT_SITES does not record it")
            continue
        if len(where) > 1:
            out.append(f"Salt.{salt} is drawn with .next at {len(where)} sites "
                       f"{where}: the device counter is local to ONE kernel's "
                       f"thread, so a second site restarts its count")
        for module, func, line in where:
            if (module, func) != (rec.module, rec.function):
                out.append(f"Salt.{salt}'s .next at {module}:{line} ({func}) is "
                           f"not its recorded site {rec.module} ({rec.function})")
    for salt, rec in table.items():
        if salt not in by_salt:
            out.append(f"Salt.{salt}'s recorded .next site {rec.module} "
                       f"({rec.function}) is gone")
    return out


# ---------------------------------------------------------------------------
# The generated header
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    salts: tuple              # (name, value) in Salt order
    next_sites: tuple
    fingerprint: str

    @property
    def width(self) -> int:
        return KEY_W


def layout() -> Layout:
    salts = tuple((m.name, int(m.value)) for m in Salt)
    text = "".join(f"salt:{n}:{v}\n" for n, v in salts)
    text += "".join(f"next:{s.salt}:{s.module}:{s.function}\n" for s in NEXT_SITES)
    text += "".join(f"key:{i}:{n}\n" for i, n in enumerate(KEY_FIELDS))
    text += "".join(f"trow:{i}:{n}\n" for i, n in enumerate(TICK_ROW_FIELDS))
    return Layout(salts=salts, next_sites=NEXT_SITES,
                  fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/unit_draws_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_draws.py",
        "// (#29 P7.0a, residency design 4.4c decision 5) -- NEVER EDIT BY HAND.",
        "// The device draws' vocabulary: every simulation.rng_salts.Salt member,",
        "// the draw key row a kernel reads per env, and the ONE src/ call site of",
        "// each salt drawn with MatchDraws.next (its device counter is local to",
        "// one kernel's thread). A new salt or .next site: run",
        f"// `{GENERATOR}` and commit; tests/test_unit_draws.py",
        "// refuses a stale copy and a second .next site.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace unit_draws {",
        "",
        f'constexpr const char* UNIT_DRAWS_FINGERPRINT = "{lay.fingerprint}";',
        "",
        "// The salts (simulation.rng_salts.Salt), each its four letters as one",
        "// big-endian word.",
    ]
    for n, v in lay.salts:
        lines.append(f"constexpr uint32_t SALT_{n} = 0x{v:08X}u;")
    lines += ["", "// The draw key row (draw_key: one per env)."]
    for i, n in enumerate(KEY_FIELDS):
        lines.append(f"constexpr int DRAW_KEY_{n.upper()} = {i};")
    lines.append(f"constexpr int DRAW_KEY_W = {KEY_W};")
    lines += ["", "// #29 P7e.1: THE TICK ROW (unit_draws.TickRow: one per env, written once",
              "// per tick, read by every unit kernel); the draw key's three columns from",
              "// TICK_ROW_KEY, in DRAW_KEY_* order."]
    for i, n in enumerate(TICK_ROW_FIELDS):
        lines.append(f"constexpr int TICK_ROW_{n.upper()} = {i};")
    lines.append(f"constexpr int TICK_ROW_KEY = {TICK_ROW_KEY};")
    lines.append(f"constexpr int TICK_ROW_W = {TICK_ROW_W};")
    lines += ["", "// The salts drawn with .next, each at ONE src/ site (module, function):"]
    for s in lay.next_sites:
        lines.append(f"//   {s.salt}: {s.module} ({s.function})")
        lines.append(f"constexpr bool NEXT_SITE_{s.salt} = true;")
    lines.append(f"constexpr int N_NEXT_SITES = {len(lay.next_sites)};")
    lines += ["", "}  // namespace unit_draws", ""]
    return "\n".join(lines)


def check_compiled(bp) -> None:
    """Refuse a compiled module whose draws header is not this table's."""
    got = bp.unit_draws_layout()["fingerprint"]
    want = layout().fingerprint
    if got != want:
        raise ValueError(f"unit draws: the compiled draws layout ({got[:12]}) is "
                         f"not simulation.unit_draws' ({want[:12]}): run "
                         f"`{GENERATOR}` and rebuild cpp/build_cuda")


__all__ = ["HEADER_REL", "GENERATOR", "KEY_FIELDS", "KEY_W", "NextSite",
           "TICK_ROW_FIELDS", "TICK_ROW_W", "TICK_ROW_KEY", "TICK_ROW_TILES",
           "tick_row", "tick_rows", "TickRow", "row_ptr",
           "NEXT_SITES", "draw_key", "draw_keys", "next_sites", "sites_in_source",
           "next_site_problems", "Layout", "layout", "header_text",
           "check_compiled"]
