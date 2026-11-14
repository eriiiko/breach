"""THE device tables of the unit kernels (#29 P6.2; residency design §4.11 and
§4.4b, "Re-check at P6.2's start", item 1).

Every TABLE value a device unit kernel reads -- a dial of the unit impulse
law, the turning law's class dials, a stability, a status kind's flags, the
exposure decay, a class's full health, the zombie team, a gas plane's index --
is ONE ROW of :data:`ROWS`: its name, its width (an int, or a callable reading
the config table it spans, as the store's block widths do), the function that
reads its value(s) from the SAME validated object the reference law reads
(``sim.impulse_table``, ``Simulation._marine_turn_rate_q16``, the status
registry, ``exposure.get_settings()``, the species and env-profile tables,
``unit_traits.class_hp_q`` ...) -- never re-derived from config -- and the
range the kernels' integer arithmetic was proved for (``lo`` .. ``hi``,
refused by name at :func:`values` when a value leaves it).

The rows of one env are one int64 vector of width ``T`` (:func:`layout`); a
batch is ``(N, T)``, one row per env, because some tables are quantized PER
MAP (the impulse table: its tile size). C++ reads the offsets ONLY from the
GENERATED, checked-in ``cpp/src/unit_tables_layout.h`` (:func:`header_text`,
written by ``tools/gen_unit_layout.py --write`` beside the store's layout;
``tests/test_unit_tables.py`` refuses a stale copy, and ``DeviceStore``
refuses a compiled module whose tables fingerprint is not this schema's). The
same header carries the per-unit constants a kernel needs by name: each hook's
index in the ``traits`` block (``UNIT_HOOK_<NAME>``) and each status kind's
id (``UNIT_STATUS_<NAME>``) -- they are part of the fingerprint.

A NEW TABLE VALUE A KERNEL READS IS: one row here (from the reference's own
object, with its proved range), ``tools/gen_unit_layout.py --write``, commit
the header, rebuild ``cpp/build_cuda``. A group-D unit value (a stability, a
class hp) is reached on the device through the group-S column that selects
its row (``env_profile_name``, ``species_id``, ``is_zombie``), never stored
per unit.

NOT PART OF THE PLAYED TICK: read by the unit store's driver
(``unit_store.DeviceStore``) for the Class-T kernels only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from simulation import unit_fixed

ONE = unit_fixed.FP_ONE
I64_MIN = -(2 ** 63)
I64_MAX = 2 ** 63 - 1
#: The generated header (C++'s copy of the table layout).
HEADER_REL = "cpp/src/unit_tables_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"


# ---------------------------------------------------------------------------
# The sources (each the object a reference law reads)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSource:
    """What one env's rows are read from: the sim's own per-map objects and
    the process-wide validated tables."""
    sim: Any

    @property
    def impulse(self):
        return self.sim.impulse_table


def _profiles():
    from simulation.loadouts import get_tables
    return get_tables().env_profiles.by_name


def _species():
    from simulation.species import SPECIES_REGISTRY
    return SPECIES_REGISTRY


def _status_registry():
    from simulation.status import STATUS_REGISTRY
    return STATUS_REGISTRY


def _q(v) -> int:
    return unit_fixed.quantize_scalar(float(v))


# The status flag bits (``status_flags``): what a kind TAKES AWAY (an AND
# composition on the device is an OR of these) and whether it lays the unit
# prone.
FLAG_NO_MOVE, FLAG_NO_ACT, FLAG_NO_AIM, FLAG_PRONE = 1, 2, 4, 8
#: ``status_stacking`` codes.
STACKING_CODES = {"refresh": 0, "stack": 1, "max": 2}


def _status_flags(src):
    out = []
    for r in _status_registry():
        f = 0
        f |= 0 if r.can_move else FLAG_NO_MOVE
        f |= 0 if r.can_act else FLAG_NO_ACT
        f |= 0 if r.can_aim else FLAG_NO_AIM
        f |= FLAG_PRONE if r.is_prone else 0
        out.append(f)
    return tuple(out)


def _status_stacking(src):
    return tuple(STACKING_CODES[r.stacking] for r in _status_registry())


def _stability_profiles(src):
    return tuple(_q(getattr(p, "stability", 1.0))
                 for p in _profiles().values())


def _stability_species(src):
    return tuple(_q(getattr(s.environment, "stability", 1.0))
                 for s in _species().values())


def _stability_zombie(src):
    from simulation.species import ZOMBIE_STABILITY
    return _q(ZOMBIE_STABILITY)


def _hp_class(zombie):
    def f(src):
        from simulation.unit_traits import class_hp_q
        return class_hp_q(zombie)
    return f


def _zombie_team(src):
    from simulation.factions import ZOMBIE_TEAM
    return int(ZOMBIE_TEAM)


def _gas_index(name):
    def f(src):
        from simulation import gases
        return int(getattr(gases, name))
    return f


def _exposure_decay(src):
    from simulation import exposure
    return int(exposure.get_settings().decay_q)


def _stack_cap(src):
    from simulation.unit_schema import STATUS_STACK_CAP
    return int(STATUS_STACK_CAP)


def _n_kinds():
    return len(_status_registry())


def _n_profiles():
    return len(_profiles())


def _n_species():
    return len(_species())


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableRow:
    """One table value (or a fixed run of them) a kernel reads: ``value``
    reads it from the reference's own object (``src``: a
    :class:`TableSource`); ``lo`` .. ``hi`` is the range the kernels'
    arithmetic was proved for (refused by name outside it)."""
    name: str
    width: int | Callable[[], int]
    value: Callable[[TableSource], Any]
    lo: int
    hi: int
    note: str = ""

    def w(self) -> int:
        return self.width() if callable(self.width) else int(self.width)


_V = 1 << 22          # a velocity bound: every square of a sum of two < 2^47
_IMP = lambda attr: (lambda src: getattr(src.impulse, attr))  # noqa: E731

ROWS: tuple[TableRow, ...] = (
    # -- the unit impulse law (unit_impulse.ImpulseTable, per map) ----------
    TableRow("impulse_cap_q", 1, _IMP("cap_q"), 1, _V,
             "one tick of standing hold, Q16 tiles/tick"),
    TableRow("impulse_cap_prone_q", 1, _IMP("cap_prone_q"), 1, _V,
             "one tick of a prone body's friction"),
    TableRow("impulse_zombie_hold_q16", 1, _IMP("zombie_hold_q16"), 0,
             1 << 20, "the zombie state's hold multiplier, Q16"),
    TableRow("impulse_v_inplace_q", 3, _IMP("v_inplace_q"), 1, _V,
             "(fwd, side, back) in-place capacity"),
    TableRow("impulse_v_fall_q", 3, _IMP("v_fall_q"), 1, _V,
             "(fwd, side, back) fall limit at stability 1"),
    TableRow("impulse_react_ticks", 1, _IMP("react_ticks"), 0, 1 << 16,
             "the reaction delay after an overflow"),
    TableRow("impulse_n_rec", 1, _IMP("n_rec"), 1, 256,
             "the recovery horizon (n_rec * margin squared fits int64)"),
    TableRow("impulse_inertia_g", 1, _IMP("inertia_g"), 1, 1 << 40,
             "yaw inertia, gram-tile^2"),
    TableRow("impulse_getup_ticks", 1, _IMP("getup_ticks"), 1, 1 << 20,
             "the knockdown's duration"),
    TableRow("impulse_wind_k_q", 1, _IMP("wind_k_q"), 0, 1 << 40,
             "the wind's impulse per unit of load, standing (Q16)"),
    TableRow("impulse_wind_k_prone_q", 1, _IMP("wind_k_prone_q"), 0,
             1 << 40, "... prone"),
    # -- the turning law's class dials (the TURN hook's class values) -------
    TableRow("turn_rate_class_q16", 1,
             lambda src: src.sim._marine_turn_rate_q16, 1, 1 << 18,
             "[marine] turn_rate_deg_per_s per tick, Q16 radians"),
    TableRow("turn_accel_class_q16", 1,
             lambda src: src.sim._marine_turn_accel_q16, 1, 1 << 18,
             "[marine] turn_accel_rad_per_s2 per tick^2, Q16 radians"),
    # -- stability (the HOLD hook's class value: exchange._stability_for) ---
    TableRow("stability_zombie_q16", 1, _stability_zombie, 0, 1 << 20,
             "species.ZOMBIE_STABILITY, quantized"),
    TableRow("stability_profile_q16", _n_profiles, _stability_profiles, 0,
             1 << 20, "per [env_profiles.*] row (env_profile_name's index)"),
    TableRow("stability_species_q16", _n_species, _stability_species, 0,
             1 << 20, "per species (species_id's index): a unit with no "
             "env profile reads its species' environment"),
    # -- the status registry ---------------------------------------------------
    TableRow("status_flags", _n_kinds, _status_flags, 0, 15,
             "per kind: FLAG_NO_MOVE | FLAG_NO_ACT | FLAG_NO_AIM | FLAG_PRONE"),
    TableRow("status_stacking", _n_kinds, _status_stacking, 0, 2,
             "per kind: 0 refresh, 1 stack, 2 max"),
    TableRow("status_stack_cap", 1, _stack_cap, 1, 1 << 10,
             "unit_schema.STATUS_STACK_CAP"),
    # -- exposure ---------------------------------------------------------------
    TableRow("exposure_decay_q", 1, _exposure_decay, 0, ONE,
             "[damage.exposure]'s per-tick decay, Q16"),
    # -- the class's full health (the HP hook's class values) ----------------
    TableRow("hp_class_marine_q", 1, _hp_class(False), 1, 1 << 40,
             "unit_traits.class_hp_q(False)"),
    TableRow("hp_class_zombie_q", 1, _hp_class(True), 1, 1 << 40,
             "unit_traits.class_hp_q(True)"),
    # -- identities -------------------------------------------------------------
    TableRow("zombie_team", 1, _zombie_team, 0, I64_MAX,
             "factions.ZOMBIE_TEAM"),
    TableRow("gas_o2", 1, _gas_index("O2"), 0, 255,
             "the O2 bulk plane's index in GameMap.gas"),
    TableRow("gas_n2", 1, _gas_index("INERT_N2"), 0, 255,
             "the inert N2 bulk plane's index in GameMap.gas"),
)

ROW_BY_NAME = {r.name: r for r in ROWS}
assert len(ROW_BY_NAME) == len(ROWS), "a table row is declared twice"


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableColumn:
    name: str
    offset: int
    width: int


@dataclass(frozen=True)
class Layout:
    columns: tuple
    width: int
    hooks: tuple              # hook names, traits-block order
    status_kinds: tuple       # status kind names, registry order
    fingerprint: str

    def column(self, name: str) -> TableColumn:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"unit tables: no row {name!r}")


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    """THE table layout from :data:`ROWS` (cached on the widths, which read
    config tables)."""
    from simulation.unit_traits import HOOKS
    widths = tuple(r.w() for r in ROWS)
    hooks = tuple(h.name for h in HOOKS)
    kinds = tuple(r.name for r in _status_registry())
    key = (widths, hooks, kinds)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    cols, off = [], 0
    for r, w in zip(ROWS, widths):
        if w < 1:
            raise ValueError(f"unit tables: row {r.name!r} has width {w}")
        cols.append(TableColumn(r.name, off, w))
        off += w
    text = "".join(f"{c.name}:{c.offset}:{c.width}\n" for c in cols)
    text += f"T:{off}\n" + "".join(f"hook:{i}:{n}\n"
                                   for i, n in enumerate(hooks))
    text += "".join(f"status:{i}:{n}\n" for i, n in enumerate(kinds))
    lay = Layout(columns=tuple(cols), width=off, hooks=hooks,
                 status_kinds=kinds,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/unit_tables_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_tables.py",
        "// (#29 P6.2, residency design 4.11 / 4.4b) -- NEVER EDIT BY HAND.",
        "// The device tables of the unit kernels: one int64 vector of",
        "// UNIT_TABLES_W per env, each row read from the reference's own",
        "// validated object (the impulse table, the turning dials, the status",
        "// registry, ...). A new table value is a row of unit_tables.ROWS, then",
        f"// `{GENERATOR}` and commit; tests/test_unit_tables.py refuses",
        "// a stale copy. Also: each hook's index in the `traits` block and each",
        "// status kind's id (part of the fingerprint).",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace unit_tables {",
        "",
        f'constexpr const char* UNIT_TABLES_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int UNIT_TABLES_W = {lay.width};",
        "",
        "// <row>: its first int64 in an env's vector; <row>_W: its width.",
    ]
    for c in lay.columns:
        n = c.name.upper()
        lines.append(f"constexpr int UNIT_TAB_{n} = {c.offset};")
        lines.append(f"constexpr int UNIT_TAB_{n}_W = {c.width};")
    lines += ["", "// The hooks' indices in the unit's `traits` block "
              "(unit_traits.HOOKS order)."]
    for i, n in enumerate(lay.hooks):
        lines.append(f"constexpr int UNIT_HOOK_{n.upper()} = {i};")
    lines.append(f"constexpr int UNIT_N_HOOKS = {len(lay.hooks)};")
    lines += ["", "// The status kinds' ids (status.STATUS_REGISTRY order)."]
    for i, n in enumerate(lay.status_kinds):
        lines.append(f"constexpr int UNIT_STATUS_{n.upper()} = {i};")
    lines.append(f"constexpr int UNIT_N_STATUS_KINDS = {len(lay.status_kinds)};")
    lines += ["", "// status_flags bits; status_stacking codes.",
              f"constexpr int64_t UNIT_FLAG_NO_MOVE = {FLAG_NO_MOVE};",
              f"constexpr int64_t UNIT_FLAG_NO_ACT = {FLAG_NO_ACT};",
              f"constexpr int64_t UNIT_FLAG_NO_AIM = {FLAG_NO_AIM};",
              f"constexpr int64_t UNIT_FLAG_PRONE = {FLAG_PRONE};"]
    for k, v in STACKING_CODES.items():
        lines.append(f"constexpr int64_t UNIT_STACK_{k.upper()} = {v};")
    lines += ["", "}  // namespace unit_tables", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The values
# ---------------------------------------------------------------------------
def values(sim) -> np.ndarray:
    """One env's table vector (``(T,)`` int64) read from ``sim``'s objects.
    Refused by name: a value that is not an int, a row whose value count is
    not its width, a value outside its row's proved range."""
    lay = layout()
    src = TableSource(sim)
    out = np.zeros(lay.width, dtype=np.int64)
    for r, c in zip(ROWS, lay.columns):
        v = r.value(src)
        vals = tuple(v) if isinstance(v, (tuple, list)) else (v,)
        if len(vals) != c.width:
            raise ValueError(f"unit tables: row {r.name!r} has {len(vals)} "
                             f"values, its width is {c.width}")
        for i, x in enumerate(vals):
            if isinstance(x, bool) or int(x) != x:
                raise ValueError(f"unit tables: row {r.name!r}[{i}] = {x!r} "
                                 f"is not an integer")
            x = int(x)
            if not r.lo <= x <= r.hi:
                raise ValueError(
                    f"unit tables: row {r.name!r}[{i}] = {x} is outside the "
                    f"range the unit kernels were proved for [{r.lo}, "
                    f"{r.hi}]")
            out[c.offset + i] = x
    return out


def values_batch(sims) -> np.ndarray:
    """``(N, T)``: env e's row is :func:`values` of ``sims[e]``."""
    sims = list(sims)
    out = np.zeros((len(sims), layout().width), dtype=np.int64)
    for e, s in enumerate(sims):
        out[e] = values(s)
    return out


def read(vec, name: str):
    """A row's value(s) from one env's vector (an int, or a tuple for a row
    wider than one)."""
    c = layout().column(name)
    v = [int(x) for x in np.asarray(vec)[c.offset:c.offset + c.width]]
    return v[0] if c.width == 1 else tuple(v)


__all__ = ["HEADER_REL", "GENERATOR", "TableSource", "TableRow", "ROWS",
           "ROW_BY_NAME", "FLAG_NO_MOVE", "FLAG_NO_ACT", "FLAG_NO_AIM",
           "FLAG_PRONE", "STACKING_CODES", "Layout", "layout", "header_text",
           "values", "values_batch", "read"]
