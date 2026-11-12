"""The unit schema -- THE one table of what a :class:`~simulation.unit.Unit`
holds (#29 P3a; ``docs/residency_batching_design_2026-11-07.md`` §4.4 and its
re-plan §4.4a).

Every attribute a unit holds is in exactly ONE of three groups, and
``Unit.__slots__`` is generated from them, so an undeclared attribute raises
at the line that sets it:

- **Group S, the synced columns** (:data:`COLUMNS`): the unit's state, one
  row each -- its integer KIND, its DEFAULT (applied at construction by
  :func:`apply_defaults`, the one place a column's default lives), its DIGEST
  rule (*always*, or only while its encoding differs from the default's --
  absence-transparent, the ``DIGEST_SECTIONS`` precedent) and its INTEGER
  ENCODING (``encode`` / ``decode``) for every column whose host value is not
  a plain int: an intent object, a ``None``, a kit string (a row index of a
  named table), a fixed block. These are the device store's columns (P6).
- **Group D, the derived constants** (:data:`DERIVED`): never written after
  spawn, rebuilt from group S's kit columns by the derivation each row names
  -- not columns, not hashed. The four per-unit optics overrides
  (:data:`OPTICS_OVERRIDES`) are declared slots left UNSET: the stamp reads
  them by ``getattr(u, name, map_default)`` (no game path sets them; tests do).
- **Group R** (:data:`REFERENCE_ONLY`): host-only state never read on the
  continuous match path (the WEGO / onephase state), or a diagnostic, each
  with its reason.

THE RULE: a new unit attribute is a row here -- a column of group S, or a
group-D derivation, or a ``REFERENCE_ONLY`` entry with its reason -- in the
same patch that adds it.

The encoding reuses the entity system's field kinds (``int`` / ``q16`` /
``bool`` / ``enum``, :mod:`simulation.entities.schema`) and
:mod:`simulation.entities.serialize`'s ``name|int64`` row bytes
(:func:`record_bytes`, under the ``UNIT_SECT_V2`` preamble); the unit-only
kinds are ``table`` (a kit string as its row index in a named config table,
``-1`` = none), ``block`` (a fixed-width list of ints), ``optional`` (an
intent object or ``None``, encoded as a presence flag + its integer fields).
The spray's aim is a Q16 angle and its target a Q16 point since P3e, encoded
inside its ``optional`` columns as those ints; the hp is the Q16 int ``hp_q``
since P3e (``current_hp`` its float view). The positions are Q16 integers since P3c
(``x_q`` / ``y_q``; ``Unit.x`` / ``Unit.y`` are their read-only float views,
written only through ``Unit.set_pos_q`` / ``Unit.set_pos``).

The capacities live here too: the status block's width
(:data:`STATUS_CAPACITY`, with the ``STACK_STACK`` cap
:data:`STATUS_STACK_CAP` enforced at ``status.apply_status``'s door) and the
planted-charge table's (:data:`CHARGE_CAPACITY`, enforced at
``charges.place``), each overflow refused and counted. And the planted
charges' own table, :data:`CHARGE_SCHEMA`.

Import-light: nothing here imports :mod:`simulation.unit` at module level
(``unit.py`` builds its ``__slots__`` from this module); the config tables
are read lazily, at encode time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from simulation.entities.schema import KIND_BOOL, KIND_ENUM, KIND_INT, KIND_Q16
from simulation.entities.serialize import _pack_i64

# ---------------------------------------------------------------------------
# Sentinels (the integer stand-ins for a host ``None``)
# ---------------------------------------------------------------------------
#: ``Unit.current_mag`` of a magazine never bound (the lazy first-trigger
#: bind, ``combat.mag_bind``; was ``None`` before P3a). A bound magazine holds
#: 0 .. ``mag_size`` rounds.
MAG_UNBOUND = -1
#: ``Unit.speed_ticks_per_tile`` when the zombie walk takes its cadence from
#: the ``[speeds]`` table (was ``None`` before P3a); an override is >= 1.
SPEED_FROM_TABLE = -1
#: ``Unit.throw_reach_q`` when the throw named no reach (the v0.1 fixed one).
THROW_NO_REACH = -(2 ** 31)
#: The encoded ``Unit.pending_plant`` with no plant latched (host ``None``).
NO_PLANT = -1
#: A table column's "none" row index (``""`` / ``None``).
NO_ROW = -1
#: A planted charge's ``det_tick`` encoded when remote (host ``None``).
NO_TIMER = -1
#: A planted charge's approach tile encoded when it has none (host ``None``).
NO_TILE = -1

# ---------------------------------------------------------------------------
# Capacities
# ---------------------------------------------------------------------------
#: Instances of ONE ``STACK_STACK`` status kind a unit may hold at once; one
#: more is REFUSED at ``status.apply_status``'s door and counted
#: (``status.OVERFLOW_REFUSED``). No shipped row applies a stacking kind
#: (POISONED is the only one; no weapon or coupling row applies it yet), and
#: no test stacks more than a handful: 8 is never reached (asserted by
#: ``tests/test_unit_schema.py``).
STATUS_STACK_CAP = 8
#: The status block's width (rows of ``[kind, magnitude_q16,
#: remaining_ticks, source_id]``): every non-stacking kind holds at most ONE
#: instance and every stacking kind at most :data:`STATUS_STACK_CAP`, so the
#: block can never overflow once the stack cap holds -- the width is that
#: bound, computed from the status registry (16 today: 8 + 1 x 8).
STATUS_CAPACITY = None          # bound lazily by status_capacity()
#: Live planted charges a sim holds at once (``Simulation.planted_charges``);
#: a plant into a full table is REFUSED at ``charges.place`` and counted
#: (``Simulation.charges_refused``). A match plants a handful at most (one C4
#: per rifleman); 32 is never reached by any test or pinned match (asserted).
CHARGE_CAPACITY = 32

#: The section preamble of a unit record's ``name|int64`` bytes.
UNIT_SECT_PREAMBLE = b"UNIT_SECT_V2\n"
CHARGE_SECT_PREAMBLE = b"CHARGE_SECT_V1\n"

# Unit-only kinds (beside the entity system's int / q16 / bool / enum).
KIND_TABLE = "table"          # a config row name as its row index (-1 none)
KIND_BLOCK = "block"          # a fixed-width list of ints
KIND_OPTIONAL = "optional"    # an intent object or None: presence + ints

DIGEST_ALWAYS = "always"
DIGEST_NONDEFAULT = "nondefault"


def status_capacity() -> int:
    """The status block's width (see :data:`STATUS_CAPACITY`)."""
    global STATUS_CAPACITY
    if STATUS_CAPACITY is None:
        from simulation.status import STACK_STACK, STATUS_REGISTRY
        STATUS_CAPACITY = sum(STATUS_STACK_CAP if r.stacking == STACK_STACK
                              else 1 for r in STATUS_REGISTRY)
    return STATUS_CAPACITY


# ---------------------------------------------------------------------------
# Named tables (a kit string's row index)
# ---------------------------------------------------------------------------
def _species_names():
    from simulation.species import SPECIES_REGISTRY
    return SPECIES_REGISTRY


def _loadout_names():
    from simulation.loadouts import get_tables
    return get_tables().loadouts.by_name


def _profile_names():
    from simulation.loadouts import get_tables
    return get_tables().env_profiles.by_name


def _weapon_names():
    from simulation.weapons import get_tables
    return get_tables().weapons.by_name


def _ammo_names():
    from simulation.weapons import get_tables
    return get_tables().ammo.by_name


#: The named tables a ``table`` column indexes, each a callable returning the
#: live ``{name: row}`` dict; the row index is the name's position in it
#: (config order).
TABLES: dict[str, Callable[[], dict]] = {
    "species": _species_names,
    "loadouts": _loadout_names,
    "env_profiles": _profile_names,
    "weapons": _weapon_names,
    "ammo": _ammo_names,
}
_INDEX_CACHE: dict = {}


def _table_index(table: str) -> tuple[tuple, dict]:
    rows = TABLES[table]()
    key = (table, id(rows), len(rows))
    hit = _INDEX_CACHE.get(key)
    if hit is None:
        names = tuple(rows)
        hit = (names, {n: i for i, n in enumerate(names)})
        _INDEX_CACHE[key] = hit
    return hit


def table_row(table: str, name) -> int:
    """``name``'s row index in ``table``; refuses an unknown name."""
    names, index = _table_index(table)
    try:
        return index[name]
    except KeyError:
        raise ValueError(f"unit schema: {name!r} names no row of the "
                         f"{table} table (known: {list(names)})") from None


def table_name(table: str, row: int):
    names, _ = _table_index(table)
    return names[row]


# ---------------------------------------------------------------------------
# The column row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Column:
    """One synced column. ``attr`` is the name read from a unit (a property
    for the facing and the hp), ``store`` the slot it lives in and the one
    :func:`apply_defaults` and ``decode`` write. ``default`` is the HOST
    default (a callable for a fresh list); ``encode`` / ``decode`` map the
    host value to the integer encoding (an int, or a tuple of ints of
    ``width``) and back. ``key`` is the column's name in the digest record;
    ``digest_of`` (optional) hashes a value derived from the unit instead of
    the encoding (the turning law's target is hashed as the turn still to
    go). ``until`` names the patch that makes a float column integer."""
    name: str
    kind: str
    block: str
    default: Any
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    digest: str = DIGEST_NONDEFAULT
    width: int | Callable[[], int] = 1
    attr: str = ""
    store: str = ""
    key: str = ""
    digest_of: Callable[[Any], Any] | None = None
    until: str = ""
    note: str = ""

    def __post_init__(self):
        for f in ("attr", "store", "key"):
            if not getattr(self, f):
                object.__setattr__(self, f, self.name)

    def host_default(self):
        return self.default() if callable(self.default) else self.default

    def encoded(self, u):
        """The encoding of ``u``'s value (``getattr``: a plain stand-in or a
        unit built before a column existed reads its default)."""
        v = getattr(u, self.attr, _MISSING)
        if v is _MISSING:
            v = self.host_default()
        return self.encode(v)

    def digest_value(self, u):
        if self.digest_of is not None:
            return self.digest_of(u)
        return self.encoded(u)

    def digest_default(self):
        if self.digest_of is not None:
            return 0
        return self.encode(self.host_default())


_MISSING = object()


# --- encoders --------------------------------------------------------------
def _enc_int(v):
    if isinstance(v, bool) or int(v) != v:
        raise ValueError(f"unit schema: an int column holds {v!r}")
    return int(v)


def _enc_bool(v):
    return 1 if v else 0


def _dec_bool(e):
    return bool(e)


def _table_codec(table: str, none):
    def enc(v):
        return NO_ROW if v == none or v is None else table_row(table, v)

    def dec(e):
        return none if e == NO_ROW else table_name(table, e)
    return enc, dec


_LIFE_CHOICES = ("alive", "dead")


def _enc_life(v):
    return _LIFE_CHOICES.index(getattr(v, "value", v))


def _life_default():
    from simulation.unit import LifeState
    return LifeState.ALIVE


def _dec_life(e):
    from simulation.unit import LifeState
    return LifeState(_LIFE_CHOICES[e])


def _enc_block(v):
    return tuple(int(x) for x in v)


def _dec_block(e):
    return [int(x) for x in e]


def _enc_sentinel(sentinel):
    def enc(v):
        return sentinel if v is None else _enc_int(v)
    return enc


def _dec_identity(e):
    return e


# --- the optional (intent) codecs -------------------------------------------
def _enc_move(v):
    if v is None:
        return (0, 0, 0, 0)
    return (1, int(v.dx_q), int(v.dy_q), int(v.throttle_q))


def _dec_move(e):
    if not e[0]:
        return None
    from simulation.intents import MoveDirIntent
    return MoveDirIntent(e[1], e[2], e[3])


def _enc_aim(v):
    if v is None:
        return (0, 0, 0)
    return (1, int(v.dx_q), int(v.dy_q))


def _dec_aim(e):
    if not e[0]:
        return None
    from simulation.intents import AimIntent
    return AimIntent(e[1], e[2])


def _enc_throw(v):
    if v is None:
        return (0, 0, 0, 0, 0, 0)
    return (1, int(v.dx_q), int(v.dy_q), int(v.fuse_ticks),
            THROW_NO_REACH if v.reach_q is None else int(v.reach_q),
            int(v.kind))


def _dec_throw(e):
    if not e[0]:
        return None
    from simulation.intents import ThrowIntent
    return ThrowIntent(e[1], e[2], e[3],
                       None if e[4] == THROW_NO_REACH else e[4], e[5])


def _enc_point_q16(v):
    """The spray's nominal target, a Q16 point (#29 P3e), or None."""
    if v is None:
        return (0, 0, 0)
    return (1, _enc_int(v[0]), _enc_int(v[1]))


def _dec_point_q16(e):
    return None if not e[0] else (int(e[1]), int(e[2]))


def _enc_angle_q16(v):
    """The spray's bearing, a Q16 angle (#29 P3e), or None."""
    return (0, 0) if v is None else (1, _enc_int(v))


def _dec_angle_q16(e):
    return None if not e[0] else int(e[1])


# --- the status block --------------------------------------------------------
_EMPTY_STATUS_ROW = (-1, 0, 0, -1)


def _enc_statuses(v):
    """The fixed status block: :func:`status_capacity` rows of ``(kind,
    magnitude_q16, remaining_ticks, source_id)`` in LIST order (the P0 sync
    order), a ``None`` source as -1, padded with ``(-1, 0, 0, -1)``; flat."""
    w = status_capacity()
    rows = [(int(st.kind), int(st.magnitude_q16), int(st.remaining_ticks),
             -1 if st.source_id is None else int(st.source_id)) for st in v]
    if len(rows) > w:
        raise ValueError(f"unit schema: {len(rows)} statuses exceed the "
                         f"block's capacity {w}")
    rows += [_EMPTY_STATUS_ROW] * (w - len(rows))
    return tuple(x for r in rows for x in r)


def _dec_statuses(e):
    from simulation.status import StatusEffect
    out = []
    for i in range(0, len(e), 4):
        kind, mag, ticks, src = e[i:i + 4]
        if kind < 0:
            break
        out.append(StatusEffect(kind=kind, magnitude_q16=mag,
                                remaining_ticks=ticks,
                                source_id=None if src < 0 else src))
    return out


# --- defaults that read a table ------------------------------------------------
def _exposure_default():
    from simulation.exposure import new_block
    return new_block()


def _exposure_width():
    from simulation.damage import N_DAMAGE_TYPES
    return N_DAMAGE_TYPES


def _grenade_width():
    from simulation.weapons import get_tables
    return len(get_tables().hand_grenade_rows)


def _grenade_default():
    return [0] * _grenade_width()


def _turn_of(u):
    """The turning law's target hashed as the turn still to go,
    ``wrap(target - facing)`` (the target itself is never zero)."""
    from simulation.turning import wrap_q16
    t = getattr(u, "facing_target_q", None)
    if t is None:
        return 0
    return int(wrap_q16(int(t) - int(u.facing_q)))


# ---------------------------------------------------------------------------
# GROUP S -- the synced columns (§4.4a's table, in its order)
# ---------------------------------------------------------------------------
def _c(name, kind, block, default, enc=_enc_int, dec=_dec_identity, **kw):
    return Column(name=name, kind=kind, block=block, default=default,
                  encode=enc, decode=dec, **kw)


_TBL = _table_codec
_SP_E, _SP_D = _TBL("species", None)
_LO_E, _LO_D = _TBL("loadouts", None)
_EP_E, _EP_D = _TBL("env_profiles", None)
_WP_E, _WP_D = _TBL("weapons", "")
_AM_E, _AM_D = _TBL("ammo", "")

COLUMNS: tuple[Column, ...] = (
    # identity
    _c("id", KIND_INT, "identity", -1, digest=DIGEST_ALWAYS),
    _c("slot", KIND_INT, "identity", -1),
    _c("team", KIND_INT, "identity", 0),
    _c("is_zombie", KIND_BOOL, "identity", False, _enc_bool, _dec_bool),
    _c("life_state", KIND_ENUM, "identity", _life_default, _enc_life, _dec_life,
       digest=DIGEST_ALWAYS, note="alive / dead; `alive` is its property"),
    # kit (row indices on the device; strings on the host)
    _c("species_id", KIND_TABLE, "kit", "human", _SP_E, _SP_D),
    _c("loadout_name", KIND_TABLE, "kit", None, _LO_E, _LO_D),
    _c("env_profile_name", KIND_TABLE, "kit", None, _EP_E, _EP_D),
    _c("weapon_id", KIND_TABLE, "kit", "", _WP_E, _WP_D),
    _c("melee_id", KIND_TABLE, "kit", "", _WP_E, _WP_D),
    _c("charge_ammo", KIND_TABLE, "kit", "", _AM_E, _AM_D),
    _c("magazines_weapon", KIND_TABLE, "kit", "", _WP_E, _WP_D),
    _c("magazines", KIND_INT, "kit", 0),
    _c("footprint", KIND_INT, "kit", 3),
    _c("mass_g", KIND_INT, "kit", 0, note="body + kit, integer grams"),
    # position
    # Q16 tiles (#29 P3c): `x` / `y` are their read-only float views;
    # written only through Unit.set_pos_q (Unit.set_pos the float door)
    _c("x_q", KIND_Q16, "position", 0, digest=DIGEST_ALWAYS, store="_x_q",
       note="Q16 tiles; `x` = x_q / 65536 (read-only), P3c"),
    _c("y_q", KIND_Q16, "position", 0, digest=DIGEST_ALWAYS, store="_y_q",
       note="Q16 tiles; `y` = y_q / 65536 (read-only), P3c"),
    _c("facing_q", KIND_Q16, "position", 0, digest=DIGEST_ALWAYS,
       store="_facing_q", key="facing"),
    _c("facing_target_q", KIND_Q16, "position", 0, key="turn",
       digest_of=_turn_of),
    _c("omega_q", KIND_Q16, "position", 0, key="omega"),
    # motion
    _c("speed_ticks_per_tile", KIND_INT, "motion", SPEED_FROM_TABLE,
       _enc_sentinel(SPEED_FROM_TABLE)),
    _c("zombie_move_accumulator", KIND_INT, "motion", 0),
    # health
    _c("current_hp", KIND_Q16, "health", 0, attr="hp_q",
       digest=DIGEST_ALWAYS, store="_hp_q",
       note="the Q16 int hp_q since #29 P3e (current_hp its float view); "
            "hashed under current_hp as before"),
    _c("statuses", KIND_BLOCK, "health", list, _enc_statuses, _dec_statuses,
       width=lambda: 4 * status_capacity()),
    _c("exposure", KIND_BLOCK, "health", _exposure_default, _enc_block,
       _dec_block, width=_exposure_width),
    # impulse
    *(_c(n, KIND_INT, "impulse", 0) for n in (
        "imp_x", "imp_y", "imp_l", "stand_x", "stand_y", "swx", "swy",
        "pvx", "pvy", "effort", "react_ticks")),
    # magazine
    _c("current_mag", KIND_INT, "magazine", MAG_UNBOUND,
       _enc_sentinel(MAG_UNBOUND)),
    _c("spare_mags", KIND_INT, "magazine", -1),
    _c("reload_done_tick", KIND_INT, "magazine", -1),
    # grenades and charges
    _c("grenade_counts", KIND_BLOCK, "grenades", _grenade_default,
       _enc_block, _dec_block, width=_grenade_width),
    _c("has_explosive", KIND_INT, "grenades", 0),
    _c("throw_release_tick", KIND_INT, "grenades", -1),
    _c("throw_kind", KIND_INT, "grenades", 0),
    _c("throw_reach_q", KIND_Q16, "grenades", THROW_NO_REACH),
    _c("throw_fuse_ticks", KIND_INT, "grenades", 0),
    _c("plant_done_tick", KIND_INT, "grenades", -1),
    _c("plant_fuse_ticks", KIND_INT, "grenades", 0),
    # cool-downs
    _c("last_fire_tick", KIND_INT, "cooldowns", -999),
    _c("last_blow_tick", KIND_INT, "cooldowns", -1),
    _c("last_shove_tick", KIND_INT, "cooldowns", -1),
    _c("last_melee_tick", KIND_INT, "cooldowns", -999),
    # spray
    _c("spray_ticks_left", KIND_INT, "spray", 0),
    _c("spray_target", KIND_OPTIONAL, "spray", None, _enc_point_q16,
       _dec_point_q16, width=3),
    _c("spray_aim_angle", KIND_OPTIONAL, "spray", None, _enc_angle_q16,
       _dec_angle_q16, width=2),
    # zombie
    _c("zombie_activated", KIND_BOOL, "zombie", False, _enc_bool, _dec_bool),
    _c("killed_by_zombie", KIND_BOOL, "zombie", False, _enc_bool, _dec_bool),
    _c("zombie_noise_y", KIND_INT, "zombie", -1),
    _c("zombie_noise_x", KIND_INT, "zombie", -1),
    _c("zombie_noise_ticks", KIND_INT, "zombie", 0),
    _c("zombie_anchor_y", KIND_INT, "zombie", -1),
    _c("zombie_anchor_x", KIND_INT, "zombie", -1),
    # flags
    _c("flashlight_on", KIND_BOOL, "flags", True, _enc_bool, _dec_bool),
    # order (#195 1f): the unit's order (simulation.unit_orders, the ONE
    # writer; information only until rung 4), hashed only while held
    _c("order_what", KIND_INT, "order", 0,
       note="unit_orders.ORDER_WHAT code; 0 = none"),
    _c("order_where", KIND_INT, "order", -1,
       note="a room id, or ORDER_OPENING_BASE + an opening id; -1 = none"),
    _c("order_until", KIND_INT, "order", -1,
       note="the first tick it no longer holds; -1 = until told otherwise"),
    _c("order_target", KIND_INT, "order", -1,
       note="the priority target's unit id; -1 = none"),
    # intent state: the held verbs and the latches
    _c("live_move_dir", KIND_OPTIONAL, "intent", None, _enc_move, _dec_move,
       width=4),
    _c("live_aim", KIND_OPTIONAL, "intent", None, _enc_aim, _dec_aim,
       width=3),
    _c("live_trigger", KIND_BOOL, "intent", False, _enc_bool, _dec_bool),
    _c("pending_throw", KIND_OPTIONAL, "intent", None, _enc_throw,
       _dec_throw, width=6),
    _c("pending_reload", KIND_BOOL, "intent", False, _enc_bool, _dec_bool),
    _c("pending_shove", KIND_BOOL, "intent", False, _enc_bool, _dec_bool),
    _c("pending_melee", KIND_BOOL, "intent", False, _enc_bool, _dec_bool),
    _c("pending_plant", KIND_INT, "intent", None, _enc_sentinel(NO_PLANT),
       lambda e: None if e == NO_PLANT else e),
    _c("pending_use", KIND_BOOL, "intent", False, _enc_bool, _dec_bool),
)

COLUMN_BY_NAME = {c.name: c for c in COLUMNS}
assert len(COLUMN_BY_NAME) == len(COLUMNS), "a column is declared twice"
assert len({c.key for c in COLUMNS}) == len(COLUMNS), "a digest key repeats"


def column_width(c: Column) -> int:
    return c.width() if callable(c.width) else c.width


# ---------------------------------------------------------------------------
# GROUP D -- derived constants (never written after spawn)
# ---------------------------------------------------------------------------
def _species(u):
    from simulation.species import get_species
    return get_species(u.species_id)


def _d_attrs(u):
    from simulation.generation import predefined_unit_attributes
    return predefined_unit_attributes(_species(u))


def _d_offsets(u):
    if u.footprint == 3:
        return list(_species(u).default_offsets)
    return [(dx, dy) for dy in range(u.footprint)
            for dx in range(u.footprint)]


def _d_environment(u):
    if u.env_profile_name is None:
        return _species(u).environment
    from simulation.loadouts import get_tables
    return get_tables().env_profile(u.env_profile_name)


def _d_inventory(u):
    from simulation.inventory import Inventory
    return Inventory()


def _d_frag_slot(u):
    from simulation.weapons import FRAG_ROW, get_tables
    rows = get_tables().hand_grenade_rows
    return rows.index(FRAG_ROW) if FRAG_ROW in rows else -1


#: name -> its derivation from group S (the species, the footprint, the
#: env profile, the weapon table): what a device rebuilds instead of storing.
DERIVED: dict[str, Callable[[Any], Any]] = {
    "base_stats": lambda u: _d_attrs(u)[0],
    "mass": lambda u: _d_attrs(u)[1],
    "base_speed": lambda u: _d_attrs(u)[2],
    "offsets": _d_offsets,
    "environment": _d_environment,
    "mitigation": lambda u: _species(u).mitigation,
    "inventory": _d_inventory,
    "_frag_slot": _d_frag_slot,
}
#: The four per-unit optics overrides the stamp reads by ``getattr`` with a
#: map default (``GameMap.stamp_units``): declared slots, left UNSET -- no
#: game path sets them (tests do), so the map default is their derivation.
OPTICS_OVERRIDES = ("permeability", "wave_absorb", "heat_atten", "light_atten")

# ---------------------------------------------------------------------------
# GROUP R -- REFERENCE_ONLY (host only; never read on the continuous match
# path, or a diagnostic)
# ---------------------------------------------------------------------------
_WEGO = "WEGO / onephase state: inert under ContinuousRealtime"
REFERENCE_ONLY: dict[str, str] = {
    "orders": _WEGO + " (the order queue; ALWAYS EMPTY on the continuous "
                      "path, where slot 4's process_shooting still looks "
                      "for a fire order in it)",
    "current_order_type": _WEGO,
    "ap": _WEGO + " (action points; left the digest at P3a)",
    "plan": _WEGO + " (the compiled timeline Plan)",
    "plan_round": _WEGO,
    "busy_until_tick": _WEGO,
    "gcd_until_tick": _WEGO,
    "swap_cd_until_tick": _WEGO,
    "action_cd": _WEGO,
    "loadout": _WEGO + " (primary + secondary slots; weapon_id mirrors it)",
    "belt": _WEGO + " (the quick-item belt)",
    "active_slot": _WEGO,
    "overwatch_facing": _WEGO,
    "overwatch_half_deg": _WEGO,
    "move_path": _WEGO + " (the precomputed path; ALWAYS EMPTY on the "
                         "continuous path, whose slot 3 still reads its "
                         "length -- it walks nothing)",
    "path_tick_offset": _WEGO + " (the path's pause offset; slot 3 bumps it "
                                "for a downed unit on every path, but only "
                                "the empty path reads it)",
    "spray_order": _WEGO + " (the fire order a spray burst rides)",
    "fire_target": _WEGO + " (never read)",
    "recent_attackers": "the WEGO idle stance's 'who shot me this round'; "
                        "written by damage.apply_packet on every path, read "
                        "only by WEGO / onephase (engagement, ruleset)",
    "impulse_books": "the impulse law's momentum books: a diagnostic "
                     "(unit_impulse.books_identity), written, never read "
                     "by the law",
    "name": "a label (conversion prefixes it 'Z-'); never read by a rule",
    "awakened": "a hidden Hartmann flag, written only at construction, read "
                "only by stats.is_stat_player_visible (the UI policy)",
}

#: The backing slots of the properties (``facing_q`` / ``facing``,
#: ``current_hp``): a column's ``store`` when it differs from its ``attr``.
_PRIVATE_STORES = tuple(c.store for c in COLUMNS if c.store != c.attr)
#: The property names a column is read through when they differ from its
#: name (``hp_q`` for ``current_hp``, #29 P3e).
_COLUMN_ATTRS = tuple(c.attr for c in COLUMNS if c.attr != c.name)


def slot_names() -> tuple[str, ...]:
    """``Unit.__slots__``: group S (by store), group D, the optics
    overrides, group R -- each name exactly once."""
    names = ([c.store for c in COLUMNS] + list(DERIVED)
             + list(OPTICS_OVERRIDES) + list(REFERENCE_ONLY))
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise AssertionError(f"unit schema: {dup} declared in two groups")
    return tuple(names)


def group_of(name: str) -> str:
    """'S', 'D' or 'R' for a unit attribute (the property names of a column
    -- ``facing_q``, ``current_hp`` -- are S); KeyError when undeclared."""
    if (name in COLUMN_BY_NAME or name in _PRIVATE_STORES
            or name in _COLUMN_ATTRS):
        return "S"
    if name in DERIVED or name in OPTICS_OVERRIDES:
        return "D"
    if name in REFERENCE_ONLY:
        return "R"
    raise KeyError(name)


def apply_defaults(u) -> None:
    """Every group-S column of ``u`` at its schema default -- the one place
    a column's default lives (``Unit.__init__`` calls it first, then sets
    the kit and the spawn values). Writes the store, never a property."""
    for c in COLUMNS:
        object.__setattr__(u, c.store, c.host_default())


# ---------------------------------------------------------------------------
# The record (the digest reads it) and its bytes
# ---------------------------------------------------------------------------
def encode_unit(u) -> dict:
    """Every column's encoding (the device row, as a dict)."""
    return {c.name: c.encoded(u) for c in COLUMNS}


def decode_into(u, row: dict) -> None:
    """Write a :func:`encode_unit` row back onto ``u`` (the bridge's
    direction store -> unit; P6's)."""
    for c in COLUMNS:
        object.__setattr__(u, c.store, c.decode(row[c.name]))


def digest_record(u) -> dict:
    """THE unit record the synced digest hashes (#29 P3a): every group-S
    column under its key, an *always* column always, every other only while
    its encoding differs from its default's (absence-transparent), in schema
    order. Works by ``getattr`` on any object (a unit of an older tree, a
    stand-in), a missing attribute reading its default."""
    rec = {}
    for c in COLUMNS:
        v = c.digest_value(u)
        if c.digest == DIGEST_ALWAYS or v != c.digest_default():
            rec[c.key] = v
    return rec


def _row_bytes(key: str, value) -> bytes:
    if isinstance(value, tuple):
        body = _pack_i64(f"unit '{key}' width", len(value)) + b"".join(
            _pack_i64(f"unit '{key}'", x) for x in value)
    else:
        body = _pack_i64(f"unit '{key}'", value)
    return key.encode("ascii") + b"|" + body + b"\n"


def record_bytes(rec: dict) -> bytes:
    """A :func:`digest_record` as ``name|int64`` rows
    (:mod:`simulation.entities.serialize`'s encoding; a block is its width
    then its values), terminated by an empty line."""
    return b"".join(_row_bytes(k, v) for k, v in rec.items()) + b"\n"


def unit_section_bytes(records) -> bytes:
    """The ``UNIT_SECT_V2`` bytes of a tick's records (id order)."""
    return UNIT_SECT_PREAMBLE + b"".join(record_bytes(r) for r in records)


# ---------------------------------------------------------------------------
# The planted charges' table (#193; ``charges.PlantedCharge``)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChargeColumn:
    name: str
    encode: Callable[[Any], Any]
    width: int = 1


def _enc_approach(v):
    return (NO_TILE, NO_TILE) if v is None else (int(v[0]), int(v[1]))


CHARGE_SCHEMA: tuple[ChargeColumn, ...] = (
    ChargeColumn("x", _enc_int),
    ChargeColumn("y", _enc_int),
    ChargeColumn("det_tick", _enc_sentinel(NO_TIMER)),
    ChargeColumn("ammo_name", _AM_E),
    ChargeColumn("owner_id", _enc_int),
    ChargeColumn("approach", _enc_approach, width=2),
    ChargeColumn("live", _enc_bool),
)


def charge_record(c) -> tuple:
    """One planted charge as its integer row (flat, :data:`CHARGE_SCHEMA`
    order)."""
    out = []
    for col in CHARGE_SCHEMA:
        v = col.encode(getattr(c, col.name))
        out.extend(v if isinstance(v, tuple) else (v,))
    return tuple(out)


def charge_section_bytes(rows) -> bytes:
    return CHARGE_SECT_PREAMBLE + b"".join(
        b"".join(_pack_i64("charge", x) for x in r) + b"\n" for r in rows)


__all__ = [
    "MAG_UNBOUND", "SPEED_FROM_TABLE", "THROW_NO_REACH", "NO_PLANT", "NO_ROW",
    "NO_TIMER", "NO_TILE", "STATUS_STACK_CAP", "CHARGE_CAPACITY",
    "status_capacity", "Column", "COLUMNS", "COLUMN_BY_NAME", "DERIVED",
    "OPTICS_OVERRIDES", "REFERENCE_ONLY", "slot_names", "group_of",
    "apply_defaults", "encode_unit", "decode_into", "digest_record",
    "record_bytes", "unit_section_bytes", "CHARGE_SCHEMA", "charge_record",
    "charge_section_bytes", "TABLES", "table_row", "column_width",
]
