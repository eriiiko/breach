"""THE SWARM KERNEL TABLES (arc #63 P3, docs/swarm_P3_impl.md v4.3).

Every value a swarm kernel reads is ONE ROW here (the Device tables
discipline of ``simulation.unit_tables``, adopted; its module and generator
are not shared): a name, the function reading it from the SAME validated
object the CPU reference reads -- the species row in ``gmap.swarm_species``
(``src.row``), the launch context ``swarm.launch_context(gmap, sim_time)``
(``src.ctx``: the tick actually stepped, the map's tile size), or
``rng_salts.Salt`` -- and the range the kernel's integer arithmetic was proved
for (refused by name outside it, before anything is launched).

One env's rows are one int64 vector of width ``width(sp)``; a batch is
``(N, width)``, one row per env (``step_q`` depends on the map's tile size),
packed per launch by :func:`values_batch`.

C++ reads offsets ONLY from the generated, checked-in
``cpp/src/swarm_tables_layout.h`` (``tools/gen_swarm_layout.py --write``),
whose ``SWARM_TABLES_FINGERPRINT`` (a hash of the names, order and ranges)
the compiled module exports; ``swarm.engine_args`` refuses a module whose
fingerprint is not :func:`fingerprint`, and ``tests/test_swarm_tables.py``
refuses a stale header. A new value a swarm kernel reads is: one row,
``--write``, commit, rebuild both modules.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

from simulation import swarm, swarm_fixed, swarm_larva
from simulation.rng_salts import Salt

HEADER_REL = "cpp/src/swarm_tables_layout.h"
GENERATOR = "tools/gen_swarm_layout.py --write"

_I32_MIN, _I32_MAX = -(1 << 31), (1 << 31) - 1
_U32_MAX = (1 << 32) - 1
_ONE = swarm_fixed.FP_ONE
_PI = swarm_fixed.PI_Q16
# The sweep angle's ceiling is the field's own maximum, quantize(pi / 2) =
# 102944 (PI_Q16 // 2 = 102943 would refuse the field's maximum). With a
# canonical heading |h| <= PI_Q16 the sweep angle stays < 1.5 * PI_Q16 + 1,
# deep inside the trig kit's pinned +-4 pi band (fixed_point.h sin_q16).
_HALF_PI_Q = swarm_fixed.quantize_scalar(3.141592653589793 / 2)


class TableSource(NamedTuple):
    """What a row reads from: the species row in force and the launch context."""
    row: object
    ctx: swarm.SpeciesContext


@dataclass(frozen=True)
class TableRow:
    """One table value a swarm kernel reads: ``read(src)`` from the
    reference's own object; ``lo`` .. ``hi`` the range the law's integer
    arithmetic was proved for."""
    name: str
    read: Callable
    lo: int
    hi: int


def _attr(name):
    return lambda src: int(getattr(src.row, name))


def _step_q(src):
    return swarm.movement_step_q(src.row, src.ctx)[1]


def _p_base(src):
    return swarm_larva.tumble_probabilities(src.row, src.ctx)[0]


def _p_alarm(src):
    return swarm_larva.tumble_probabilities(src.row, src.ctx)[1]


# The larva's rows, in THIS order (the generated header's offsets). The four
# temperatures are full int32 (the comfort |T - T_prefer| is formed in int64);
# step_q < 1 tile is the no-tunnelling proof (§3.3); the probabilities feed a
# Q16 compare (L5); the salts are counter words.
ROWS: dict = {
    "larva": (
        TableRow("T_prefer_q", _attr("T_prefer_q"), _I32_MIN, _I32_MAX),
        TableRow("T_lethal_hot_q", _attr("T_lethal_hot_q"), _I32_MIN, _I32_MAX),
        TableRow("T_ctmin_q", _attr("T_ctmin_q"), _I32_MIN, _I32_MAX),
        TableRow("T_coma_exit_q", _attr("T_coma_exit_q"), _I32_MIN, _I32_MAX),
        TableRow("turn_bias_q", _attr("turn_bias_q"), 0, _ONE),
        TableRow("turn_min_q", _attr("turn_min_q"), 0, _PI),
        TableRow("turn_max_q", _attr("turn_max_q"), 0, _PI),
        TableRow("sweep_angle_q", _attr("sweep_angle_q"), 1, _HALF_PI_Q),
        TableRow("step_q", _step_q, 1, _ONE - 1),
        TableRow("p_base_q", _p_base, 0, _ONE),
        TableRow("p_alarm_q", _p_alarm, 0, _ONE),
        TableRow("salt_tumble", lambda src: int(Salt.SWARM_TUMBLE), 0, _U32_MAX),
        TableRow("salt_wall_turn", lambda src: int(Salt.SWARM_WALL_TURN), 0, _U32_MAX),
    ),
}

if set(ROWS) != set(swarm.SWARM_SPECIES):
    raise ValueError("swarm_tables.ROWS keys must equal swarm.SWARM_SPECIES")
for _sp, _rows in ROWS.items():
    if len({r.name for r in _rows}) != len(_rows):
        raise ValueError(f"swarm_tables.ROWS[{_sp!r}]: row names must be unique")


def width(sp: str) -> int:
    return len(ROWS[sp])


def offset(sp: str, name: str) -> int:
    for i, r in enumerate(ROWS[sp]):
        if r.name == name:
            return i
    raise KeyError(f"swarm tables: species {sp!r} has no row {name!r}")


def _layout_text() -> str:
    lines = []
    for sp in swarm.SWARM_SPECIES:
        for i, r in enumerate(ROWS[sp]):
            lines.append(f"{sp}:{r.name}:{i}:{r.lo}:{r.hi}\n")
        lines.append(f"{sp}:W:{width(sp)}\n")
    return "".join(lines)


_FINGERPRINT = None


def fingerprint() -> str:
    """SHA-256 over every species' row names, order and ranges (cached: the
    rows are module constants)."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        _FINGERPRINT = hashlib.sha256(_layout_text().encode("ascii")).hexdigest()
    return _FINGERPRINT


def header_text() -> str:
    """The generated C++ header's full text (``cpp/src/swarm_tables_layout.h``)."""
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_swarm_layout.py from src/simulation/swarm_tables.py",
        "// (arc #63 P3, docs/swarm_P3_impl.md v4.3) -- NEVER EDIT BY HAND.",
        "// The swarm kernels' table rows: one int64 vector of SWARM_<SP>_WIDTH per",
        "// env, each row read from the validated object the CPU reference reads",
        "// (the species row, the launch context, rng_salts.Salt). A new value a",
        f"// swarm kernel reads is a row of swarm_tables.ROWS, then `{GENERATOR}`,",
        "// commit, rebuild both modules; tests/test_swarm_tables.py refuses a",
        "// stale copy and swarm.engine_args a module whose fingerprint differs.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace swarm_tables {",
        "",
        f'constexpr const char* SWARM_TABLES_FINGERPRINT = "{fingerprint()}";',
    ]
    for sp in swarm.SWARM_SPECIES:
        p = sp.upper()
        lines += ["", f"// species {sp!r}: <row> is its int64's index in an env's vector."]
        for i, r in enumerate(ROWS[sp]):
            lines.append(f"constexpr int SWARM_{p}_{r.name.upper()} = {i};")
        lines.append(f"constexpr int SWARM_{p}_WIDTH = {width(sp)};")
    lines += ["", "}  // namespace swarm_tables", ""]
    return "\n".join(lines)


class Layout(NamedTuple):
    width: int
    fingerprint: str


def layout() -> Layout:
    """For the generator's report (the unit_tables idiom)."""
    return Layout(width=sum(width(sp) for sp in swarm.SWARM_SPECIES),
                  fingerprint=fingerprint())


def values(sp: str, gmap, sim_time) -> np.ndarray:
    """One env's ``(width(sp),)`` int64 rows of species ``sp``, read from the
    map's species row and the launch context. Refused by name: a value that
    is not an integer, a value outside its row's proved range."""
    src = TableSource(row=gmap.swarm_species.row(sp),
                      ctx=swarm.launch_context(gmap, sim_time))
    out = np.zeros(width(sp), dtype=np.int64)
    for i, r in enumerate(ROWS[sp]):
        v = r.read(src)
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(f"swarm tables: [swarm.{sp}] row {r.name!r} = {v!r} "
                             f"is not an integer")
        v = int(v)
        if not r.lo <= v <= r.hi:
            raise ValueError(
                f"swarm tables: [swarm.{sp}] row {r.name!r} = {v} is outside the "
                f"range the {sp} kernel was proved for [{r.lo}, {r.hi}]")
        out[i] = v
    return out


def values_batch(sp: str, gmaps, sim_time) -> np.ndarray:
    """``(N, width(sp))``: env e's rows are :func:`values` of ``gmaps[e]``."""
    gmaps = list(gmaps)
    out = np.zeros((len(gmaps), width(sp)), dtype=np.int64)
    for e, g in enumerate(gmaps):
        out[e] = values(sp, g, sim_time)
    return out


def read(sp: str, vec, name: str) -> int:
    """A row's value from one env's vector."""
    return int(np.asarray(vec)[offset(sp, name)])


def check_compiled(bp) -> None:
    """Refuse a compiled module whose tables fingerprint is not this one's
    (by name; a stale header or an unrebuilt module)."""
    got = getattr(bp, "SWARM_TABLES_FINGERPRINT", None)
    if got != fingerprint():
        raise RuntimeError(
            f"swarm tables: the compiled module's SWARM_TABLES_FINGERPRINT "
            f"{got!r} is not swarm_tables.fingerprint() {fingerprint()!r} -- run "
            f"`{GENERATOR}`, commit, and rebuild both modules")


__all__ = ["HEADER_REL", "GENERATOR", "TableSource", "TableRow", "ROWS", "width",
           "offset", "fingerprint", "header_text", "layout", "values",
           "values_batch", "read", "check_compiled"]
