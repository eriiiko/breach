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


def _traits_default():
    from simulation.unit_traits import neutral_block
    return neutral_block()


def _traits_width():
    from simulation.unit_traits import N_HOOKS
    return N_HOOKS


def _grenade_width():
    from simulation.weapons import get_tables
    return len(get_tables().hand_grenade_rows)


def _grenade_default():
    return [0] * _grenade_width()


def _poison_cause_default():
    """``poison_gas``'s index in ``events.CAUSE_NAMES`` (arc #63 P5a)."""
    from simulation.events import CAUSE_NAMES, Cause
    return CAUSE_NAMES.index(Cause.POISON_GAS.value)


def _body_temp_default():
    from simulation.body_temp import ambient_q
    return ambient_q()


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
    # traits (#29 U1): the hook block, Q16 multipliers on class values
    # (simulation.unit_traits.HOOKS order), all ONE = neutral = absent from
    # the digest; written only by unit_traits.compile_unit at spawn, kept
    # by a conversion
    _c("traits", KIND_BLOCK, "traits", _traits_default, _enc_block,
       _dec_block, width=_traits_width,
       note="Q16 hooks, unit_traits.HOOKS order; ONE = neutral"),
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
    # poison (#29 PX; simulation.poison, the ONE writer): the depot and the
    # blood (Q16 units) and the unit credited with the latest dose
    _c("poison_x0", KIND_Q16, "health", 0,
       note="the poison depot, Q16 units (simulation.poison)"),
    _c("poison_x1", KIND_Q16, "health", 0,
       note="the poison in the blood, Q16 units (simulation.poison)"),
    _c("poison_src", KIND_INT, "health", -1,
       note="the unit credited with the latest dose; -1 = the environment "
            "or none"),
    # the two slow states (#215): the body temperature (simulation.body_temp,
    # written by the heat coupling row alone; hashed while it differs from
    # ambient) and the stress level (simulation.stress, fed through its one
    # producer door, cleared by its own tick; hashed while non-zero)
    _c("body_temp_q", KIND_Q16, "health", _body_temp_default,
       note="Q16 degrees C; starts at [combat] heat_ambient_ref "
            "(simulation.body_temp)"),
    _c("stress_q", KIND_INT, "health", 0,
       note="stress points x 2^24 (simulation.stress.LEVEL_ONE)"),
    # arc #63 P5a: the CAUSE of the latest dose (its index in
    # events.CAUSE_NAMES), which the law's packets carry; the default is the
    # gas's, so an unpoisoned or gas-poisoned unit hashes as before
    _c("poison_cause", KIND_INT, "health", _poison_cause_default,
       note="the latest dose's cause, an events.CAUSE_NAMES index; "
            "poison_gas's by default (simulation.poison)"),
    # arc #63 P5b: the corpse (simulation.corpse, the ONE writer): the
    # infestation level the swarm's count plane raises on a dead unit,
    # hashed only while non-default, so a match without larvae hashes
    # exactly as before (P5c: the body is the hp below zero, no column)
    _c("infest_q", KIND_Q16, "health", 0,
       note="the corpse's infestation level, Q16 (1.0 = ONE); 0 while alive"),
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
    "hit_shape": "a creature's hit set (#106 P6d3, creature_system.HitShape): "
                 "DERIVED every tick at slot 5b from the creature body's "
                 "integer chain, which the __creature__ digest section "
                 "carries; None for every other unit, and never read for "
                 "one (Unit._creature_hit_shape asks the species first). "
                 "Read for a creature only, by the bullet march and the "
                 "laser skewer through Unit.in_hit_tiles; host-side on "
                 "every path (the device conductor refuses a creature)",
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
class TableColumn:
    """One column of an OBJECT TABLE (#29 P7.0b, design 4.4c decision 3: a
    host list of world objects -- the planted charges today -- as rows of
    int64): ``name`` the object's attribute, ``encode`` its value to an int
    (or a tuple of ``width`` ints), ``decode`` back (the inverse:
    ``simulation.object_tables.unpack`` rebuilds the objects from it)."""
    name: str
    encode: Callable[[Any], Any]
    width: int = 1
    decode: Callable[[Any], Any] = None
    #: #29 P7a.2: how the value is READ from a host object (default: the
    #: attribute ``name``) -- a log's tuples have no attributes, and a list
    #: whose every member is live by construction (the rounds in flight)
    #: has no live attribute to read.
    get: Callable[[Any], Any] = None

    def __post_init__(self):
        if self.decode is None:
            object.__setattr__(self, "decode", _dec_identity)

    def value_of(self, obj):
        return self.get(obj) if self.get is not None else getattr(obj, self.name)


#: The planted charges' columns were declared under this name first (#193).
ChargeColumn = TableColumn


@dataclass(frozen=True)
class ObjectTable:
    """An object table's schema (#29 P7.0b): EVERY fact of the table lives
    here, none in ``simulation.object_tables`` (the one module over them).
    ``name`` (the generated header's ``OBJ_<NAME>_*`` names), ``columns``
    (row order), ``capacity`` (rows per env: past it nothing is written,
    the overflow is counted and refused by name), ``live`` (the column whose
    0 marks a row DEAD: the stable compaction drops it), ``build`` (the
    decoded attributes as a dict -> a fresh host object: unpack's
    constructor)."""
    name: str
    columns: tuple
    capacity: int
    live: str
    build: Callable[[dict], Any]
    #: #29 P6b.1: the variable-length members, each a :class:`SideSpan` (a
    #: span of rows into a SIDE table, itself an ObjectTable of
    #: :data:`OBJECT_TABLES`); ``build`` receives each member's decoded side
    #: objects under the span's name.
    spans: tuple = ()


@dataclass(frozen=True)
class SideSpan:
    """#29 P6b.1 (design 4.4c decision 3: "a variable-length member ... is a
    span into a side table"): an object's member ``name`` -- ``items(obj)``,
    a list of side objects in the member's order -- stored as rows of the
    side table ``side`` (another :class:`ObjectTable`), the object's own row
    holding the first side row in column ``start`` and the row count in
    column ``count``. The side rows of a list are its objects' members
    concatenated in list order, so ``start`` is the running sum of the counts
    before it: ``simulation.object_tables`` fills both columns, never the
    object. ``max_count`` bounds one object's member (refused by name past
    it)."""
    name: str
    side: str
    start: str
    count: str
    items: Callable[[Any], list]
    max_count: int


def _enc_approach(v):
    return (NO_TILE, NO_TILE) if v is None else (int(v[0]), int(v[1]))


def _dec_approach(e):
    return None if tuple(e) == (NO_TILE, NO_TILE) else (int(e[0]), int(e[1]))


CHARGE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("x", _enc_int),
    TableColumn("y", _enc_int),
    TableColumn("det_tick", _enc_sentinel(NO_TIMER),
                decode=lambda e: None if e == NO_TIMER else e),
    TableColumn("ammo_name", _AM_E, decode=_AM_D),
    TableColumn("owner_id", _enc_int),
    TableColumn("approach", _enc_approach, width=2, decode=_dec_approach),
    TableColumn("live", _enc_bool, decode=_dec_bool),
)


def _build_charge(attrs: dict):
    from simulation.charges import PlantedCharge
    c = PlantedCharge(attrs["x"], attrs["y"], attrs["ammo_name"],
                      attrs["owner_id"], det_tick=attrs["det_tick"],
                      approach=attrs["approach"])
    c.live = attrs["live"]
    return c


#: THE planted-charge table (``Simulation.planted_charges``), the object
#: tables' pilot: its capacity is :data:`CHARGE_CAPACITY` (the same bound
#: ``charges.place`` refuses at, there counted over LIVE charges).
CHARGE_TABLE = ObjectTable(name="charge", columns=CHARGE_SCHEMA,
                           capacity=CHARGE_CAPACITY, live="live",
                           build=_build_charge)
# ---------------------------------------------------------------------------
# #29 P7b: the grenades in flight (``Simulation.projectiles``,
# ``combat.Projectile``)
# ---------------------------------------------------------------------------
#: Grenades in flight a sim's device table holds at once: STORAGE (the
#: reference list has no bound). A unit releases at most one grenade a tick
#: and the hand grenade's fuse is at most a few seconds, so a match holds a
#: handful; the P7b gate scenes reach 13. Past it the device append is
#: refused by name (``object_tables.ObjectTableOverflow``).
GRENADE_CAPACITY = 64


def _enc_fuse_ticks(v):
    """``fuse_ticks``: the integer fuse, ``NO_TIMER`` for ``None`` (the WEGO
    order path, whose fuse is ``fuse_seconds``); a fuse equal to the
    sentinel itself is refused (it would decode as ``None``)."""
    if v is None:
        return NO_TIMER
    v = _enc_int(v)
    if v == NO_TIMER:
        raise ValueError(f"unit schema: a grenade fuse of {v} ticks is the "
                         f"NO_TIMER sentinel")
    return v


def _enc_q16_exact(v):
    """A float attribute that must be an exact Q16 value (``fuse_seconds``:
    0.0 on the continuous throw), as its Q16 integer; refused otherwise."""
    from simulation import unit_fixed
    q = unit_fixed.quantize_scalar(float(v))          # p3e-door: exact or refused
    if unit_fixed.dequantize_scalar(q) != float(v):
        raise ValueError(f"unit schema: {v!r} is not an exact Q16 value")
    return q


def _dec_q16(e):
    from simulation import unit_fixed
    return unit_fixed.dequantize_scalar(int(e))


GRENADE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("proj_type", _enc_int),
    TableColumn("owner_id", _enc_int),
    TableColumn("start_x_q", _enc_int),
    TableColumn("start_y_q", _enc_int),
    TableColumn("target_x_q", _enc_int),
    TableColumn("target_y_q", _enc_int),
    TableColumn("pos_x_q", _enc_int),
    TableColumn("pos_y_q", _enc_int),
    TableColumn("fuse_seconds", _enc_q16_exact, decode=_dec_q16),
    TableColumn("fuse_ticks", _enc_fuse_ticks,
                decode=lambda e: None if e == NO_TIMER else e),
    TableColumn("thrown_tick", _enc_int),
    TableColumn("ammo_name", _AM_E, decode=_AM_D),
    TableColumn("travel_speed_q", _enc_int),
    # #215: the flight is a chain of legs -- start_* / target_* are the
    # CURRENT leg (the throw's until the first bounce), leg_tick its first
    # tick, bounces the legs a wall has started so far.
    TableColumn("leg_tick", _enc_int),
    TableColumn("bounces", _enc_int),
    # THE LIVE COLUMN, encoded: 1 while the grenade flies, 0 once slot 2
    # detonated it (the host's ``detonated``); the slot-10 prune is the
    # table's stable compaction over it.
    TableColumn("detonated", lambda v: 0 if v else 1,
                decode=lambda e: not e),
)


def _build_grenade(attrs: dict):
    from simulation import unit_fixed
    from simulation.combat import Projectile
    dq = unit_fixed.dequantize_scalar
    p = Projectile(attrs["proj_type"], dq(attrs["start_x_q"]),
                   dq(attrs["start_y_q"]), dq(attrs["target_x_q"]),
                   dq(attrs["target_y_q"]), attrs["fuse_seconds"],
                   attrs["thrown_tick"], ammo_name=attrs["ammo_name"],
                   owner_id=attrs["owner_id"], fuse_ticks=attrs["fuse_ticks"])
    if p.travel_speed_q != attrs["travel_speed_q"]:
        raise ValueError(f"unit schema: a grenade row's travel_speed_q "
                         f"{attrs['travel_speed_q']} is not its round's "
                         f"({p.travel_speed_q}, {attrs['ammo_name']!r})")
    p.pos_x_q, p.pos_y_q = attrs["pos_x_q"], attrs["pos_y_q"]
    # #215: a bounced leg's start / target are no float views of the throw
    # (exact Q16 either way, but set from the row's own integers)
    p.start_x_q, p.start_y_q = attrs["start_x_q"], attrs["start_y_q"]
    p.target_x_q, p.target_y_q = attrs["target_x_q"], attrs["target_y_q"]
    p.leg_tick, p.bounces = attrs["leg_tick"], attrs["bounces"]
    p.detonated = attrs["detonated"]
    return p


#: THE grenade table (``Simulation.projectiles``, #29 P7b): row i the i-th
#: grenade of the list; slot 1b's throws append, slot 2 flies and fuses them
#: (a fuse-out a PAYLOAD record), slot 10's prune compacts.
GRENADE_TABLE = ObjectTable(name="grenade", columns=GRENADE_SCHEMA,
                            capacity=GRENADE_CAPACITY, live="detonated",
                            build=_build_grenade)

#: Every object table, in the generated header's order. A new table (the
#: rounds, the blast sources, an entity class) is a schema here and a row of
#: this tuple: ``tools/gen_unit_layout.py --write``, commit, rebuild
#: ``cpp/build_cuda``.
OBJECT_TABLES: tuple[ObjectTable, ...] = (CHARGE_TABLE, GRENADE_TABLE)


# ---------------------------------------------------------------------------
# #29 P7a.2: the ROUNDS IN FLIGHT (``Simulation.bullets``, ``combat.
# BulletInFlight``) and the NOISE LOG (``GameMap.noise_log``, ``noise.emit``)
# ---------------------------------------------------------------------------
#: Rows per env of the round table: storage, sized above the most rounds the
#: gate scenes and ship_demo's bots match hold at once (a K5 at 1/3 m tiles
#: lives up to three ticks; a shotgun's eight pellets per trigger).
ROUND_CAPACITY = 128
#: Rows per env of the noise list: it is drained at slot 5's head; between two
#: drains it holds a shot per shooter, a detonation per payload, a bash per
#: zombie (the 9e / bash noises cross into the next tick).
NOISE_CAPACITY = 64


def _payload_names():
    from simulation.weapons import get_tables
    return tuple(get_tables().payloads.by_name)


def _enc_weapon_row(v):
    return table_row("weapons", v.name)


def _dec_weapon_row(e):
    from simulation.weapons import get_tables
    return get_tables().weapons.by_name[table_name("weapons", e)]


def _enc_ammo_row(v):
    return table_row("ammo", v.name)


def _dec_ammo_row(e):
    from simulation.weapons import get_tables
    return get_tables().ammo.by_name[table_name("ammo", e)]


def _enc_payload_row(v):
    if v is None:
        return NO_ROW
    names = _payload_names()
    if v.name not in names:
        raise ValueError(f"unit schema: payload {v.name!r} is no row of the "
                         f"payload table")
    return names.index(v.name)


def _dec_payload_row(e):
    if e == NO_ROW:
        return None
    from simulation.weapons import get_tables
    return get_tables().payloads.by_name[_payload_names()[e]]


def _enc_spawn_tick(v):
    if v is None:
        raise ValueError("unit schema: a round in flight with no spawn tick "
                         "(never fired through combat.fire_burst) has no row")
    return _enc_int(v)


def _enc_no_cover(v):
    # physical cover (OnePhaseWEGO's march_cover) is WEGO-only: the device
    # march is the continuous path's and refuses a round carrying it
    if v is not None:
        raise ValueError("unit schema: a round carrying physical cover "
                         "(march_cover, OnePhaseWEGO) has no device row")
    return 0


def _enc_no_facing_defense(v):
    if v:
        raise ValueError("unit schema: a round under the facing defense "
                         "(march_facing_defense, OnePhaseWEGO) has no device row")
    return 0


def _always_live(_obj):
    return True


def _build_round(attrs: dict):
    from simulation.combat import BulletInFlight
    b = BulletInFlight(None, attrs["shooter_id"], attrs["weapon"], attrs["ammo"],
                       attrs["ox_q"], attrs["oy_q"], attrs["angle_q"],
                       attrs["sx_q"], attrs["sy_q"])
    for k in ("rx_q", "ry_q", "prev_ix", "prev_iy", "remaining_steps",
              "speed_q16", "budget_q16", "spawn_tick", "ordinal"):
        setattr(b, k, attrs[k])
    b.payload = attrs["payload"]
    return b


#: THE round table (``Simulation.bullets``): every value a round carries
#: across ticks, its object references as rows (the shooter by its id -- a
#: converted shooter keeps it --, the weapon / ammo by their schema rows, the
#: payload by its ``[payloads.*]`` row). Every member of the host list flies
#: (the list is replaced by the survivors each tick), so ``live`` reads True;
#: the device marks a finished round 0 and compacts it away. ``unpack``
#: leaves ``shooter`` None: ``unit_kernels.rounds_to_host`` binds it by id.
ROUND_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("shooter_id", _enc_int),
    TableColumn("weapon", _enc_weapon_row, decode=_dec_weapon_row),
    TableColumn("ammo", _enc_ammo_row, decode=_dec_ammo_row),
    TableColumn("payload", _enc_payload_row, decode=_dec_payload_row),
    TableColumn("angle_q", _enc_int),
    TableColumn("rx_q", _enc_int),
    TableColumn("ry_q", _enc_int),
    TableColumn("prev_ix", _enc_int),
    TableColumn("prev_iy", _enc_int),
    TableColumn("remaining_steps", _enc_int),
    TableColumn("speed_q16", _enc_int),
    TableColumn("budget_q16", _enc_int),
    TableColumn("ox_q", _enc_int),
    TableColumn("oy_q", _enc_int),
    TableColumn("sx_q", _enc_int),
    TableColumn("sy_q", _enc_int),
    TableColumn("spawn_tick", _enc_spawn_tick),
    TableColumn("ordinal", _enc_int),
    TableColumn("march_cover", _enc_no_cover, decode=lambda e: None),
    TableColumn("march_facing_defense", _enc_no_facing_defense,
                decode=lambda e: False),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
ROUND_TABLE = ObjectTable(name="round", columns=ROUND_SCHEMA,
                          capacity=ROUND_CAPACITY, live="live",
                          build=_build_round)


def _build_noise(attrs: dict):
    return (attrs["fy"], attrs["fx"], attrs["radius"])


#: THE noise list (``GameMap.noise_log``): ``noise.emit``'s ``(fy, fx,
#: radius)`` tuples in emission order (hearing ties to the earlier noise).
#: Every logged noise is live; it persists across the tick boundary and is
#: drained at slot 5's head (design 4.4c decision 4).
NOISE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("fy", _enc_int, get=lambda n: n[0]),
    TableColumn("fx", _enc_int, get=lambda n: n[1]),
    TableColumn("radius", _enc_int, get=lambda n: n[2]),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
NOISE_TABLE = ObjectTable(name="noise", columns=NOISE_SCHEMA,
                          capacity=NOISE_CAPACITY, live="live",
                          build=_build_noise)

OBJECT_TABLES = OBJECT_TABLES + (ROUND_TABLE, NOISE_TABLE)   # #29 P7a.2

#: #29 P7c.1: rows per env of the blast log: storage, above the most discs one
#: tick detonates in the gate scenes (it lives one tick, emptied at slot 9e).
BLAST_LOG_CAPACITY = 64


def _build_blast(attrs: dict):
    return (attrs["fy"], attrs["fx"], attrs["radius"], attrs["wall_q"])


#: #29 P7c.1: THE blast log (``GameMap.blast_log``): every ``radius > 0``
#: detonation's ``(fy, fx, radius, wall_damage_q)``, in call order, appended
#: only by ``execute_payload`` and drained at slot 9e (the heat sources read
#: it, P6b.4). Every logged disc is live; the list lives one tick.
BLAST_LOG_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("fy", _enc_int, get=lambda b: b[0]),
    TableColumn("fx", _enc_int, get=lambda b: b[1]),
    TableColumn("radius", _enc_int, get=lambda b: b[2]),
    TableColumn("wall_q", _enc_int, get=lambda b: b[3]),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
BLAST_LOG_TABLE = ObjectTable(name="blast", columns=BLAST_LOG_SCHEMA,
                              capacity=BLAST_LOG_CAPACITY, live="live",
                              build=_build_blast)

OBJECT_TABLES = OBJECT_TABLES + (BLAST_LOG_TABLE,)          # #29 P7c.1


def table_record(table: ObjectTable, obj, given=None) -> tuple:
    """One host object as its integer row (flat, ``table.columns`` order).
    ``given`` (#29 P6b.1): column name -> value, read instead of the object
    (a side span's ``start`` / ``count``, which the list position decides)."""
    out = []
    for col in table.columns:
        raw = given[col.name] if given and col.name in given else col.value_of(obj)
        v = col.encode(raw)
        out.extend(v if isinstance(v, tuple) else (v,))
    return tuple(out)


# ---------------------------------------------------------------------------
# #29 P6b.1: THE DOORS (``Simulation._doors``, ``door_system.DoorRuntime``),
# the first ENTITY table, and its side table of span tiles
# ---------------------------------------------------------------------------
#: Doors a sim's device table holds: storage (nothing spawns a door at run
#: time). #215: every doorway is a door entity, so ship_demo holds 60 (its
#: 240 tiles) and a recipe ship more; was 32 (playground's 6).
DOOR_CAPACITY = 128
#: Span tiles over all of a sim's doors (the door tile side table): storage
#: (ship_demo's 60 doors hold 240). It also bounds one door's span (a span is
#: the side table's rows). Was 256.
DOOR_TILE_CAPACITY = 1024


def _door_tiles(d) -> list:
    """A door's span tiles with their hp (the S4 pin: ``hp[i]`` belongs to
    ``span[i]``), row-major: ``[(ty, tx, hp), ...]``."""
    span, hp = list(d.span), list(d.hp)
    if len(span) != len(hp):
        raise ValueError(f"unit schema: a door's span ({len(span)} tiles) and "
                         f"hp ({len(hp)} rows) differ in length")
    return [(int(t[0]), int(t[1]), int(h)) for t, h in zip(span, hp)]


class DoorRow:
    """What the door table holds of one door, rebuilt by ``unpack``: the
    runtime state ``DoorRuntime`` carries across ticks (``alive``, ``state``,
    ``want_open``, ``span``, ``hp``) and its ``ordinal``. The device form;
    ``unit_kernels.doors_to_host`` lands it on the sim's own runtimes."""
    __slots__ = ("ordinal", "alive", "state", "want_open", "span", "hp")

    def __init__(self, ordinal, alive, state, want_open, span, hp):
        self.ordinal, self.alive, self.state = ordinal, alive, state
        self.want_open, self.span, self.hp = want_open, list(span), list(hp)


def _build_door_tile(attrs: dict):
    return (attrs["ty"], attrs["tx"], attrs["hp"])


def _build_door(attrs: dict):
    tiles = attrs["tiles"]
    return DoorRow(attrs["ordinal"], attrs["alive"], attrs["state"],
                   attrs["want_open"], [(t[0], t[1]) for t in tiles],
                   [t[2] for t in tiles])


#: THE door tile side table: one row per span tile, the doors' spans
#: concatenated in door order, each row-major (``ty``, ``tx``) with the tile's
#: ``hp`` (Q16: the panel's own hp, copied at the open, restamped at the
#: close). Every row is live (doors are never removed: no compaction).
DOOR_TILE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ty", _enc_int, get=lambda t: t[0]),
    TableColumn("tx", _enc_int, get=lambda t: t[1]),
    TableColumn("hp", _enc_int, get=lambda t: t[2]),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
DOOR_TILE_TABLE = ObjectTable(name="door_tile", columns=DOOR_TILE_SCHEMA,
                              capacity=DOOR_TILE_CAPACITY, live="live",
                              build=_build_door_tile)

#: THE door table (``Simulation._doors``, ordinal order: the order the 9e(d)
#: sweep walks and a USE request's ``door`` index): ``ordinal`` (a load
#: constant, the entity digest's key), ``alive``, ``state`` (0 closed / 1 open
#: / 2 destroyed), ``want_open`` (the synced latch), the span into the door
#: tile table (``tile_start``, ``tile_count``) and ``live`` (always 1: a
#: destroyed door stays a row, dead by ``alive``).
DOOR_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("state", _enc_int),
    TableColumn("want_open", _enc_bool, decode=_dec_bool),
    TableColumn("tile_start", _enc_int),
    TableColumn("tile_count", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
DOOR_TABLE = ObjectTable(
    name="door", columns=DOOR_SCHEMA, capacity=DOOR_CAPACITY, live="live",
    build=_build_door,
    spans=(SideSpan("tiles", "door_tile", "tile_start", "tile_count",
                    _door_tiles, DOOR_TILE_CAPACITY),))

OBJECT_TABLES = OBJECT_TABLES + (DOOR_TILE_TABLE, DOOR_TABLE)   # #29 P6b.1


# ---------------------------------------------------------------------------
# #29 P6b.2: THE DUCTS AND THE VENTS (``Simulation._ducts`` / ``_vents``,
# ``vent_system.DuctRuntime`` / ``VentRuntime``), the second and third ENTITY
# tables: every attribute the 9e(d) circulation sweep reads or writes, ordinal
# order each (the order the sweep walks)
# ---------------------------------------------------------------------------
#: Ducts a sim's device table holds: storage (nothing spawns a duct at run
#: time; the shipped maximum is playground's 2, the gate scenes reach 5).
DUCT_CAPACITY = 16
#: Vents a sim's device table holds: storage (playground's 4; the gate scenes
#: reach 12). It also bounds one duct's supply vents (the sweep's scratch).
VENT_CAPACITY = 64
#: The trace slots of a duct's plenum (``vent_system.N_TRACE``: steam, smoke,
#: poison, teargas, fuel gas -- the filter columns' order).
VENT_N_TRACE = 5
#: A vent's role codes (``vent_system``: ``v.role == "return"`` extracts,
#: anything else supplies).
VENT_ROLES = ("supply", "return")


def filter_names() -> tuple:
    """The ``[filters.*]`` rows of config.toml, SORTED by name: a duct row's
    ``filter`` is its name's index here, and the device tables' filter rows
    are in this order (``unit_tables``' ``vent_filter_eff_q``)."""
    from config import CFG
    f = getattr(CFG, "filters", None)
    return tuple(sorted(vars(f).keys())) if f is not None else ()


def _enc_filter(name):
    names = filter_names()
    if name not in names:
        raise ValueError(f"unit schema: a duct's filter {name!r} is not a "
                         f"[filters.*] row ({names})")
    return names.index(name)


def _dec_filter(e):
    return filter_names()[int(e)]


def _duct_filter(d):
    # a runtime's authored field, or a decoded row's name
    return d.inst.fields["filter"] if hasattr(d, "inst") else d.filter


def _enc_vec(width):
    def enc(v):
        v = list(v)
        if len(v) != width:
            raise ValueError(f"unit schema: a {width}-wide column holds {len(v)} values")
        return tuple(_enc_int(x) for x in v)
    return enc


def _dec_vec(e):
    return [int(x) for x in e]


def _enc_role(v):
    # vent_system._duct_sweep's test: "return" extracts, any other role supplies
    return 1 if v == "return" else 0


def _dec_role(e):
    return VENT_ROLES[int(e)]


def _vent_duct_ordinal(v):
    # the vent's duct as that duct's ENTITY ORDINAL (-1: unwired / dangling);
    # a decoded row carries it as is
    if hasattr(v, "duct_ordinal"):
        return v.duct_ordinal
    return -1 if v.duct is None else int(v.duct.ordinal)


class DuctRow:
    """What the duct table holds of one duct, rebuilt by ``unpack``: the
    plenum ledger ``DuctRuntime`` carries across ticks, its ``ordinal``, its
    filter's name and ``alive``. ``unit_kernels.ducts_to_host`` lands it on
    the sim's own runtimes."""
    __slots__ = ("ordinal", "filter", "o2_raw", "n2_raw", "trace_raw", "e_plenum",
                 "sink", "e_wipe", "rail_lo_hits", "rail_hi_hits", "alive")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


class VentRow:
    """What the vent table holds of one vent (``VentRuntime``'s state and its
    load constants), rebuilt by ``unpack``; ``unit_kernels.vents_to_host``
    lands it on the sim's own runtimes."""
    __slots__ = ("ordinal", "duct_ordinal", "role", "aperture_y", "aperture_x",
                 "rate_raw_per_s", "tps", "accum", "guard_skips", "alive")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


def _build_duct(attrs: dict):
    return DuctRow(**{k: attrs[k] for k in DuctRow.__slots__})


def _build_vent(attrs: dict):
    return VentRow(**{k: attrs[k] for k in VentRow.__slots__})


#: THE duct table (``Simulation._ducts``, ordinal order: the order the sweep
#: walks): ``ordinal`` (a load constant, the entity digest's key), ``filter``
#: (the ``[filters.*]`` row's index in :func:`filter_names`), the plenum
#: ledger (``o2_raw``, ``n2_raw``, ``trace_raw[5]``, ``e_plenum``, ``sink[5]``,
#: ``e_wipe``, the rail hit counts: int64 here, unbounded ints in Python, the
#: device's proved range refused by name), ``alive`` and ``live`` (always 1).
DUCT_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("filter", _enc_filter, decode=_dec_filter, get=_duct_filter),
    TableColumn("o2_raw", _enc_int),
    TableColumn("n2_raw", _enc_int),
    TableColumn("trace_raw", _enc_vec(VENT_N_TRACE), width=VENT_N_TRACE, decode=_dec_vec),
    TableColumn("e_plenum", _enc_int),
    TableColumn("sink", _enc_vec(VENT_N_TRACE), width=VENT_N_TRACE, decode=_dec_vec),
    TableColumn("e_wipe", _enc_int),
    TableColumn("rail_lo_hits", _enc_int),
    TableColumn("rail_hi_hits", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
DUCT_TABLE = ObjectTable(name="duct", columns=DUCT_SCHEMA, capacity=DUCT_CAPACITY,
                         live="live", build=_build_duct)

#: THE vent table (``Simulation._vents``, ordinal order: within a duct the
#: sweep walks its returns, then its supplies, each in this order):
#: ``ordinal``, ``duct_ordinal`` (its duct's entity ordinal, -1 unwired),
#: ``role`` (0 supply / 1 return), the aperture tile, the load constants
#: ``rate_raw_per_s`` and ``tps``, the Bresenham ``accum``, the diagnostic
#: ``guard_skips``, ``alive`` and ``live`` (always 1).
VENT_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("duct_ordinal", _enc_int, get=_vent_duct_ordinal),
    TableColumn("role", _enc_role, decode=_dec_role),
    TableColumn("aperture_y", _enc_int),
    TableColumn("aperture_x", _enc_int),
    TableColumn("rate_raw_per_s", _enc_int),
    TableColumn("tps", _enc_int),
    TableColumn("accum", _enc_int),
    TableColumn("guard_skips", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
VENT_TABLE = ObjectTable(name="vent", columns=VENT_SCHEMA, capacity=VENT_CAPACITY,
                         live="live", build=_build_vent)

OBJECT_TABLES = OBJECT_TABLES + (DUCT_TABLE, VENT_TABLE)   # #29 P6b.2


# ---------------------------------------------------------------------------
# #29 P7c.2: THE BLAST SOURCES (``GameMap.expansions``: a comparison row's
# multi-tick divergence source, ``blast_expansion.start_expansion`` /
# ``take_source_plane``) and their side table of walked tiles
# ---------------------------------------------------------------------------
#: Sources alive at once per env: storage (a comparison row lives at most
#: ``expansion_ticks - 1`` ticks; the gate scenes keep under a dozen alive).
SOURCE_CAPACITY = 32
#: Walked tiles over all of an env's live sources: storage (a radius-5 walk on
#: open floor holds 69 tiles); it also bounds one source's walk.
SOURCE_TILE_CAPACITY = 2048


def _source_tiles(e) -> list:
    """A source's walked tiles, row-major (``reach_tiles``' order)."""
    return [(int(t[0]), int(t[1])) for t in e[0]]


def _build_source_tile(attrs: dict):
    return (attrs["y"], attrs["x"])


def _build_source(attrs: dict):
    return [list(attrs["tiles"]), attrs["s_q"], attrs["ticks"]]


#: THE source tile side table: the sources' walked tiles concatenated in list
#: order, each walk row-major. Every row is live (a dead source's tiles leave
#: with it at the span compaction).
SOURCE_TILE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("y", _enc_int, get=lambda t: t[0]),
    TableColumn("x", _enc_int, get=lambda t: t[1]),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
SOURCE_TILE_TABLE = ObjectTable(name="source_tile", columns=SOURCE_TILE_SCHEMA,
                                capacity=SOURCE_TILE_CAPACITY, live="live",
                                build=_build_source_tile)

#: THE blast-source table (``GameMap.expansions``, list order: the order
#: ``take_source_plane`` adds them in): the per-tile source ``s_q`` (Q16 1/s),
#: the ``ticks`` still to deliver, the span into the source tile table and
#: ``live`` (``ticks > 0``: every host entry is live, the take drops the rest).
SOURCE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("s_q", _enc_int, get=lambda e: e[1]),
    TableColumn("ticks", _enc_int, get=lambda e: e[2]),
    TableColumn("tile_start", _enc_int),
    TableColumn("tile_count", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=lambda e: int(e[2]) > 0),
)
SOURCE_TABLE = ObjectTable(
    name="source", columns=SOURCE_SCHEMA, capacity=SOURCE_CAPACITY, live="live",
    build=_build_source,
    spans=(SideSpan("tiles", "source_tile", "tile_start", "tile_count",
                    _source_tiles, SOURCE_TILE_CAPACITY),))

OBJECT_TABLES = OBJECT_TABLES + (SOURCE_TILE_TABLE, SOURCE_TABLE)   # #29 P7c.2


# ---------------------------------------------------------------------------
# #29 P6b.3: THE SIGNALS -- the signal bus (``Simulation._signal_bus``), the
# sensors (``_sensors``), the logic nodes (``_logic_nodes``), the pumps
# (``_pumps``) and the doors' wire drives (``_door_drives``) as entity tables:
# every attribute slot 9e's (a), (b), (c), (d) pumps and (e) read or write,
# in the reference's order (the bus in slot order, the rest in ordinal order)
# ---------------------------------------------------------------------------
#: Bus slots per env: storage (airlock_demo holds 17; the gate scene ~70).
SIGNAL_CAPACITY = 256
#: Sensors per env (airlock_demo 2; the gate scene 12).
SENSOR_CAPACITY = 32
#: Area-disc tiles over all of an env's sensors (a 2 m disc at 1/3 m tiles
#: holds 113); it also bounds one sensor's disc.
SENSOR_TILE_CAPACITY = 1024
#: Logic nodes per env (airlock_demo 3; the gate scene ~20).
NODE_CAPACITY = 64
#: Driving wires over all of an env's nodes' ``in`` inputs.
NODE_IN_CAPACITY = 256
#: Pumps per env (airlock_demo 1).
PUMP_CAPACITY = 16
#: Driving wires over all of an env's pumps' inject / extract inputs.
PUMP_IN_CAPACITY = 64
#: Door drive wires per env (airlock_demo 4).
DOOR_DRIVE_CAPACITY = 64

#: A bus slot's 9e(a) EMIT kind (``Simulation._signal_emit``): none (a sensor
#: value, a node output, a pump's at_target, a dangling emitter, any other
#: name), a door's ``is_open`` / ``alive`` (from the door table's row
#: ``emit_ref``), or a CONSTANT ``alive`` (an entity whose alive never changes
#: in the tick: ``emit_ref`` holds it).
SIGNAL_EMITS = ("none", "door_open", "door_alive", "constant")
#: A sensor's kind (``sensor_system``'s three runtimes).
SENSOR_KINDS = ("field", "clock", "motion")
#: The logic-node classes (``logic_nodes.build_logic_nodes``' branches).
NODE_KINDS = ("decider", "gate_and", "gate_or", "gate_not", "filter",
              "airlock_controller")
#: The airlock controller's wired inputs and command outputs, in its row's
#: ``a_in`` / ``a_out`` column order (``AirlockControllerRuntime``'s slots).
AIRLOCK_INPUTS = ("presence", "inner_open", "outer_open", "inner_alive",
                  "outer_alive", "at_far", "at_near", "chamber_p",
                  "inner_side_p", "outer_side_p")
AIRLOCK_OUTPUTS = ("inner_close", "inner_open_cmd", "outer_close",
                   "outer_open_cmd", "pump_inject", "pump_extract", "busy",
                   "near_pump_inject", "near_pump_extract")
_AIRLOCK_IN_ATTR = ("i_presence", "i_inner_open", "i_outer_open", "i_inner_alive",
                    "i_outer_alive", "i_at_far", "i_at_near", "i_chamber_p",
                    "i_inner_side_p", "i_outer_side_p")
_AIRLOCK_OUT_ATTR = ("o_inner_close", "o_inner_open", "o_outer_close", "o_outer_open",
                     "o_inject", "o_extract", "o_busy", "o_near_inject", "o_near_extract")
#: No slot (an unwired input, a node with no ``out``, a pump with no
#: ``at_target`` slot, a decider with no alive source).
NO_SLOT = -1


def sensor_channels() -> tuple:
    """The accessor's FROZEN channel order (``sensor_accessor.CHANNEL_ORDER``):
    a sensor row's ``channel`` is its index here."""
    from simulation.sensor_accessor import CHANNEL_ORDER
    return tuple(c.value for c in CHANNEL_ORDER)


def comparators() -> tuple:
    """The decider's comparator tokens: a node row's ``cmp`` is its index."""
    from simulation.entities.nodes import COMPARATORS
    return tuple(COMPARATORS)


def signal_names() -> tuple:
    """Every signal name a bus slot can hold (every registered class's
    SIGNALS, the free ``alive``, a door's ``is_open``), SORTED: a bus row's
    ``name`` is its index here."""
    from simulation.entities import REGISTRY
    names = {"alive", "is_open"}
    for cls in REGISTRY.values():
        names.update(str(s.name) for s in getattr(cls, "SIGNALS", ()))
    return tuple(sorted(names))


def _enc_code(choices):
    def enc(v):
        c = choices() if callable(choices) else choices
        if v not in c:
            raise ValueError(f"unit schema: {v!r} is not one of {c}")
        return c.index(v)
    return enc


def _dec_code(choices):
    def dec(e):
        c = choices() if callable(choices) else choices
        return c[int(e)]
    return dec


def _enc_slot(v):
    return NO_SLOT if v is None else int(v)


def _dec_slot(e):
    return None if int(e) == NO_SLOT else int(e)


def _enc_slots(width):
    def enc(v):
        v = list(v)
        if len(v) != width:
            raise ValueError(f"unit schema: a {width}-slot column holds {len(v)}")
        return tuple(_enc_slot(x) for x in v)
    return enc


def _dec_slots(e):
    return [_dec_slot(x) for x in e]


class _Row:
    """A table's row object (``unpack``'s product and the record functions'
    input): plain attributes, ``__slots__`` the columns."""
    __slots__ = ()

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def as_tuple(self):
        return tuple(getattr(self, k) for k in self.__slots__)

    def __eq__(self, other):
        return type(self) is type(other) and self.as_tuple() == other.as_tuple()

    def __repr__(self):
        return f"{type(self).__name__}{self.as_tuple()!r}"


class SignalRow(_Row):
    """One bus slot: its ``(ordinal, name)``, ``pub`` / ``stg``, whether it
    is a NODE slot (swapped at 9e(e)), and its 9e(a) emit."""
    __slots__ = ("ordinal", "name", "pub", "stg", "node", "emit", "emit_ref")


class SensorRow(_Row):
    """One sensor (``sensor_system``'s runtimes): its kind and value slot, a
    field sensor's channel, gas slice, sample tile and area disc, a clock's
    period, a motion plate's tile, r2, line-of-sight flag and filters."""
    __slots__ = ("ordinal", "alive", "kind", "slot", "channel", "gas", "sy", "sx",
                 "has_disc", "disc", "period", "r2", "needs_los", "faction",
                 "min_footprint")


class NodeRow(_Row):
    """One logic node (``logic_nodes``' evaluators): its class, its ``out``
    slot, the driving slots of its ``in`` input, a decider's predicate CODE and
    operands, a filter's ``k`` and ``ema``, an airlock controller's input and
    output slots, constants and state machine (``state``, ``dwell``)."""
    __slots__ = ("ordinal", "kind", "out", "ins", "cmp", "threshold",
                 "require_alive", "alive_slot", "k", "ema", "a_in", "a_out",
                 "far_is_outer", "equalize_cmd", "interior", "band", "state",
                 "dwell")


class PumpRow(_Row):
    """One pump (``PumpRuntime``): its latch, slot, port, source, target,
    bands, quantum and the driving slots of inject (0) / extract (1)."""
    __slots__ = ("ordinal", "alive", "at_target", "at_slot", "port_y", "port_x",
                 "has_source", "src_y", "src_x", "target", "band", "release",
                 "delta_n_q", "ins")


class DriveRow(_Row):
    """One wire into a door's ``open`` / ``close`` (``Simulation._door_drives``):
    the door table's row, close (1) or open (0), the driving slot."""
    __slots__ = ("door", "close", "slot")


def _build_row(cls):
    def build(attrs: dict):
        return cls(**{k: attrs[k] for k in cls.__slots__})
    return build


def _build_tile(attrs: dict):
    return (attrs["ty"], attrs["tx"])


def _build_in_slot(attrs: dict):
    return attrs["slot"]


def _build_pump_in(attrs: dict):
    return (attrs["slot"], attrs["extract"])


#: THE bus table (``SignalBus.slots`` order, (ordinal, name) sorted): one row
#: per slot.
SIGNAL_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("name", _enc_code(signal_names), decode=_dec_code(signal_names)),
    TableColumn("pub", _enc_int),
    TableColumn("stg", _enc_int),
    TableColumn("node", _enc_bool, decode=_dec_bool),
    TableColumn("emit", _enc_code(SIGNAL_EMITS), decode=_dec_code(SIGNAL_EMITS)),
    TableColumn("emit_ref", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
SIGNAL_TABLE = ObjectTable(name="signal", columns=SIGNAL_SCHEMA,
                           capacity=SIGNAL_CAPACITY, live="live",
                           build=_build_row(SignalRow))

#: THE sensor disc side table: each sensor's area disc tiles in the
#: runtime's order (dy then dx, clipped), concatenated in sensor order.
SENSOR_TILE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ty", _enc_int, get=lambda t: t[0]),
    TableColumn("tx", _enc_int, get=lambda t: t[1]),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
SENSOR_TILE_TABLE = ObjectTable(name="sensor_tile", columns=SENSOR_TILE_SCHEMA,
                                capacity=SENSOR_TILE_CAPACITY, live="live",
                                build=_build_tile)

#: THE sensor table (``Simulation._sensors``, ordinal order).
SENSOR_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("kind", _enc_code(SENSOR_KINDS), decode=_dec_code(SENSOR_KINDS)),
    TableColumn("slot", _enc_int),
    TableColumn("channel", _enc_int),          # sensor_channels() index, -1 none
    TableColumn("gas", _enc_int),              # the channel's gas slice, -1 none
    TableColumn("sy", _enc_int),
    TableColumn("sx", _enc_int),
    TableColumn("has_disc", _enc_bool, decode=_dec_bool),
    TableColumn("tile_start", _enc_int),
    TableColumn("tile_count", _enc_int),
    TableColumn("period", _enc_int),
    TableColumn("r2", _enc_int),
    TableColumn("needs_los", _enc_bool, decode=_dec_bool),
    TableColumn("faction", _enc_int),
    TableColumn("min_footprint", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
SENSOR_TABLE = ObjectTable(
    name="sensor", columns=SENSOR_SCHEMA, capacity=SENSOR_CAPACITY, live="live",
    build=_build_row(SensorRow),
    spans=(SideSpan("disc", "sensor_tile", "tile_start", "tile_count",
                    lambda s: list(s.disc), SENSOR_TILE_CAPACITY),))

#: THE node input side table: each node's driving slots of ``in`` (pinned
#: wire order), concatenated in node order.
NODE_IN_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("slot", _enc_int, get=lambda s: s),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
NODE_IN_TABLE = ObjectTable(name="node_in", columns=NODE_IN_SCHEMA,
                            capacity=NODE_IN_CAPACITY, live="live",
                            build=_build_in_slot)

#: THE node table (``Simulation._logic_nodes``, ordinal order).
NODE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("kind", _enc_code(NODE_KINDS), decode=_dec_code(NODE_KINDS)),
    TableColumn("out", _enc_slot, decode=_dec_slot),
    TableColumn("in_start", _enc_int),
    TableColumn("in_count", _enc_int),
    TableColumn("cmp", _enc_int),              # comparators() index
    TableColumn("threshold", _enc_int),
    TableColumn("require_alive", _enc_bool, decode=_dec_bool),
    TableColumn("alive_slot", _enc_slot, decode=_dec_slot),
    TableColumn("k", _enc_int),
    TableColumn("ema", _enc_int),
    TableColumn("a_in", _enc_slots(len(AIRLOCK_INPUTS)), width=len(AIRLOCK_INPUTS),
                decode=_dec_slots),
    TableColumn("a_out", _enc_slots(len(AIRLOCK_OUTPUTS)), width=len(AIRLOCK_OUTPUTS),
                decode=_dec_slots),
    TableColumn("far_is_outer", _enc_bool, decode=_dec_bool),
    TableColumn("equalize_cmd", _enc_int),
    TableColumn("interior", _enc_bool, decode=_dec_bool),
    TableColumn("band", _enc_int),
    TableColumn("state", _enc_int),
    TableColumn("dwell", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
NODE_TABLE = ObjectTable(
    name="node", columns=NODE_SCHEMA, capacity=NODE_CAPACITY, live="live",
    build=_build_row(NodeRow),
    spans=(SideSpan("ins", "node_in", "in_start", "in_count",
                    lambda n: list(n.ins), NODE_IN_CAPACITY),))

#: THE pump input side table: each pump's inject slots (extract 0), then its
#: extract slots (extract 1), concatenated in pump order.
PUMP_IN_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("slot", _enc_int, get=lambda s: s[0]),
    TableColumn("extract", _enc_bool, decode=_dec_bool, get=lambda s: s[1]),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
PUMP_IN_TABLE = ObjectTable(name="pump_in", columns=PUMP_IN_SCHEMA,
                            capacity=PUMP_IN_CAPACITY, live="live",
                            build=_build_pump_in)

#: THE pump table (``Simulation._pumps``, ordinal order).
PUMP_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("at_target", _enc_int),
    TableColumn("at_slot", _enc_slot, decode=_dec_slot),
    TableColumn("port_y", _enc_int),
    TableColumn("port_x", _enc_int),
    TableColumn("has_source", _enc_bool, decode=_dec_bool),
    TableColumn("src_y", _enc_int),
    TableColumn("src_x", _enc_int),
    TableColumn("target", _enc_int),
    TableColumn("band", _enc_int),
    TableColumn("release", _enc_int),
    TableColumn("delta_n_q", _enc_int),
    TableColumn("in_start", _enc_int),
    TableColumn("in_count", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
PUMP_TABLE = ObjectTable(
    name="pump", columns=PUMP_SCHEMA, capacity=PUMP_CAPACITY, live="live",
    build=_build_row(PumpRow),
    spans=(SideSpan("ins", "pump_in", "in_start", "in_count",
                    lambda p: list(p.ins), PUMP_IN_CAPACITY),))

#: THE door drive table: every wire into a door's open / close, in door
#: (= door table row) order, each door's open wires then its close wires.
DOOR_DRIVE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("door", _enc_int),
    TableColumn("close", _enc_bool, decode=_dec_bool),
    TableColumn("slot", _enc_int),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
DOOR_DRIVE_TABLE = ObjectTable(name="door_drive", columns=DOOR_DRIVE_SCHEMA,
                               capacity=DOOR_DRIVE_CAPACITY, live="live",
                               build=_build_row(DriveRow))

OBJECT_TABLES = OBJECT_TABLES + (SIGNAL_TABLE, SENSOR_TILE_TABLE, SENSOR_TABLE,
                                 NODE_IN_TABLE, NODE_TABLE, PUMP_IN_TABLE, PUMP_TABLE,
                                 DOOR_DRIVE_TABLE)   # #29 P6b.3


# ---------------------------------------------------------------------------
# #29 P6b.4: THE TIMED CHARGES and THE HEAT SOURCES (``Simulation.
# _timed_charges`` / ``_heat_sources``, ``timed_charge_system.
# TimedChargeRuntime`` / ``heat_source_system.HeatSourceRuntime``), the two
# entity classes that detonate: every attribute the 9e(d') / 9e(d'') sweeps
# read or write, ordinal order each (the order the sweeps walk)
# ---------------------------------------------------------------------------
#: Timed charges a sim's device table holds: storage (nothing spawns one at
#: run time; the shipped maximum is blast_quad's 9, the gate scenes reach 12).
TIMED_CHARGE_CAPACITY = 32
#: Heat sources a sim's device table holds: storage (no shipped level holds
#: one; a generated ship one per tagged room; the gate scenes reach 12).
HEAT_SOURCE_CAPACITY = 32


class TimedRow(_Row):
    """What the timed-charge table holds of one charge (``TimedChargeRuntime``:
    its countdown and heat latch, its load constants), rebuilt by ``unpack``;
    ``unit_kernels.timed_charges_to_host`` lands it on the sim's own runtimes."""
    __slots__ = ("ordinal", "fy", "fx", "payload", "period_ticks", "ticks_to_fire",
                 "heat", "trigger_temp_q16", "fuse_ticks", "enabled", "fired", "alive")


class HeatRow(_Row):
    """What the heat-source table holds of one source (``HeatSourceRuntime``:
    its hp, alive and enabled, its load constants), rebuilt by ``unpack``;
    ``unit_kernels.heat_sources_to_host`` lands it on the sim's own runtimes."""
    __slots__ = ("ordinal", "fy", "fx", "heat_q16", "heat_radius", "enabled", "hp_q16",
                 "death_payload", "max_temp_q16", "alive")


#: THE timed-charge table (``Simulation._timed_charges``, ordinal order):
#: ``ordinal``, the tile, ``payload`` (the ``[payloads.*]`` row's index),
#: ``period_ticks`` (0 = once), the countdown ``ticks_to_fire`` (-1 spent /
#: disabled / waiting), the heat trigger (``heat``, ``trigger_temp_q16``,
#: ``fuse_ticks``, the latch ``fired``), ``enabled``, ``alive`` (always 1:
#: nothing kills a charge) and ``live`` (1).
TIMED_CHARGE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("fy", _enc_int),
    TableColumn("fx", _enc_int),
    TableColumn("payload", _enc_payload_row, decode=_dec_payload_row),
    TableColumn("period_ticks", _enc_int),
    TableColumn("ticks_to_fire", _enc_int),
    TableColumn("heat", _enc_bool, decode=_dec_bool),
    TableColumn("trigger_temp_q16", _enc_int),
    TableColumn("fuse_ticks", _enc_int),
    TableColumn("enabled", _enc_bool, decode=_dec_bool),
    TableColumn("fired", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
TIMED_CHARGE_TABLE = ObjectTable(name="timed_charge", columns=TIMED_CHARGE_SCHEMA,
                                 capacity=TIMED_CHARGE_CAPACITY, live="live",
                                 build=_build_row(TimedRow))

#: THE heat-source table (``Simulation._heat_sources``, ordinal order):
#: ``ordinal``, the tile, the per-tick centre deposit ``heat_q16`` and its
#: strict DISC radius ``heat_radius`` (radius + 1), ``enabled``, ``hp_q16``
#: (the wall_hp currency), ``death_payload`` (the row's index, -1 none),
#: ``max_temp_q16`` (0 = no cap), ``alive`` and ``live`` (1).
HEAT_SOURCE_SCHEMA: tuple[TableColumn, ...] = (
    TableColumn("ordinal", _enc_int),
    TableColumn("fy", _enc_int),
    TableColumn("fx", _enc_int),
    TableColumn("heat_q16", _enc_int),
    TableColumn("heat_radius", _enc_int),
    TableColumn("enabled", _enc_bool, decode=_dec_bool),
    TableColumn("hp_q16", _enc_int),
    TableColumn("death_payload", _enc_payload_row, decode=_dec_payload_row),
    TableColumn("max_temp_q16", _enc_int),
    TableColumn("alive", _enc_bool, decode=_dec_bool),
    TableColumn("live", _enc_bool, decode=_dec_bool, get=_always_live),
)
HEAT_SOURCE_TABLE = ObjectTable(name="heat_source", columns=HEAT_SOURCE_SCHEMA,
                                capacity=HEAT_SOURCE_CAPACITY, live="live",
                                build=_build_row(HeatRow))

OBJECT_TABLES = OBJECT_TABLES + (TIMED_CHARGE_TABLE, HEAT_SOURCE_TABLE)   # #29 P6b.4


def charge_record(c) -> tuple:
    """One planted charge as its integer row (flat, :data:`CHARGE_SCHEMA`
    order)."""
    return table_record(CHARGE_TABLE, c)


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
    "TableColumn", "ChargeColumn", "ObjectTable", "CHARGE_TABLE",
    "OBJECT_TABLES", "table_record",
    "ROUND_CAPACITY", "NOISE_CAPACITY", "ROUND_SCHEMA", "ROUND_TABLE",
    "NOISE_SCHEMA", "NOISE_TABLE", "GRENADE_CAPACITY", "GRENADE_SCHEMA",
    "GRENADE_TABLE", "BLAST_LOG_CAPACITY", "BLAST_LOG_SCHEMA", "BLAST_LOG_TABLE",
]
__all__ += ["SideSpan", "DOOR_CAPACITY", "DOOR_TILE_CAPACITY", "DoorRow",
            "DOOR_TILE_SCHEMA", "DOOR_TILE_TABLE", "DOOR_SCHEMA", "DOOR_TABLE"]
__all__ += ["DUCT_CAPACITY", "VENT_CAPACITY", "VENT_N_TRACE", "VENT_ROLES",
            "filter_names", "DuctRow", "VentRow", "DUCT_SCHEMA", "DUCT_TABLE",
            "VENT_SCHEMA", "VENT_TABLE"]
__all__ += ["SOURCE_CAPACITY", "SOURCE_TILE_CAPACITY", "SOURCE_TILE_SCHEMA",
            "SOURCE_TILE_TABLE", "SOURCE_SCHEMA", "SOURCE_TABLE"]
__all__ += ["SIGNAL_CAPACITY", "SENSOR_CAPACITY", "SENSOR_TILE_CAPACITY",
            "NODE_CAPACITY", "NODE_IN_CAPACITY", "PUMP_CAPACITY", "PUMP_IN_CAPACITY",
            "DOOR_DRIVE_CAPACITY", "SIGNAL_EMITS", "SENSOR_KINDS", "NODE_KINDS",
            "AIRLOCK_INPUTS", "AIRLOCK_OUTPUTS", "NO_SLOT", "sensor_channels",
            "comparators", "signal_names", "SignalRow", "SensorRow", "NodeRow",
            "PumpRow", "DriveRow", "SIGNAL_TABLE", "SENSOR_TILE_TABLE", "SENSOR_TABLE",
            "NODE_IN_TABLE", "NODE_TABLE", "PUMP_IN_TABLE", "PUMP_TABLE",
            "DOOR_DRIVE_TABLE"]
__all__ += ["TIMED_CHARGE_CAPACITY", "HEAT_SOURCE_CAPACITY", "TimedRow", "HeatRow",
            "TIMED_CHARGE_SCHEMA", "TIMED_CHARGE_TABLE", "HEAT_SOURCE_SCHEMA",
            "HEAT_SOURCE_TABLE"]
