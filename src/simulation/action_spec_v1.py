"""simulation/action_spec_v1.py -- action spec ``v1`` as code (#195 1h).

THE contract is ``docs/rl_agent_rung1_note_1h_2026-11-12.md`` (sections 1,
3, 8; design ``docs/rl_agent_design_2026-11-11.md`` 2.1-2.8, 8.2). A trainer
sends FIVE int32 per row, ``(feet, aim, trigger, hand, arg)``, each an index
into its head's option table:

- ``feet`` (49): 0 = stand; ``1 + 3 d + s`` = compass direction ``d``
  (0..15, :data:`COMPASS_Q16`) at speed ``s`` (0 walk, 1 half, 2 full);
- ``aim`` (``18 + E``): 0..15 a compass direction, 16 "where I walk", 17
  "keep the throw's aim", ``18 + i`` list entry ``i`` of the observation the
  row was chosen on (the pointer is turned into a game id by the decoder,
  ``control_policy.AgentPolicy.set_decisions``, and never kept as a
  position);
- ``trigger`` (2): release / hold;
- ``hand`` (``7 + K``): :data:`FIXED_HANDS` then one throw per hand-grenade
  row in config order (``WeaponsTables.hand_grenade_rows``) -- a new grenade
  row APPENDS an entry and moves no index; the flashlight entry is
  RESERVED, always masked and decoded as nothing (design 2.6);
- ``arg`` (5): a throw's reach (0..3 = a quarter .. all of the hand
  grenade's ``max_throw_range``; 4 = onto the aimed unit), a plant's timer
  (0..3, :func:`plant_timers`); any other verb: 0.

THE NUMBERS come from their table rows, quantized once per match by
:func:`action_tables` -- never a literal. THE MASKS (:func:`build_masks`)
are built at the start of the decision tick, ONE call of the 1e gate per
hand bit (``simulation.verb_gates``) -- never a copy of a decision. A head
left out of learning (design 8.2) is a head with exactly ONE legal option
(:func:`heads_used`); no head is ever all-masked. The argument's mask is a
pure function of the drawn hand and aim (:func:`arg_mask`).

The inverse helpers (:func:`feet_index`, :func:`aim_index_for_vector`,
:func:`aim_index_for_unit`, :func:`throw_choice`, :func:`plant_choice`) turn
a bot's decision into the nearest choice of THESE tables (1j).

Pure host Python; imports the gates, the weapons tables, the integer kit and
the one control-seam quantizer -- no controller, nothing from ``renderer/``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control_source import quantize_stick_direction
from simulation import verb_gates
from simulation.movement import THROTTLE_ONE
from simulation.turning import PI_Q16
from simulation.unit_fixed import atan2_q16, cos_q16, quantize_scalar, sin_q16
from simulation.weapons import PLANT_WEAPON, fuse_ticks_from_seconds

ACTION_SPEC_NAME = "v1"
#: The run file's ``[action] spec`` values: ``"raw"`` = the generated
#: columns of ``control_policy.action_schema`` (the default), ``"v1"`` this.
ACTION_SPECS = ("raw", "v1")
HEADS = ("feet", "aim", "trigger", "hand", "arg")

# -- feet ---------------------------------------------------------------------
N_DIRS = 16
N_SPEEDS = 3                         # walk, half, full
N_FEET = 1 + N_DIRS * N_SPEEDS       # 49
FEET_STAND = 0

#: The 16 compass directions as Q16.16 unit vectors, ``d`` at ``d * 22.5``
#: degrees counter-clockwise from east in the facing convention (``(cos,
#: -sin)``, world y down) -- a CHECKED-IN literal table (no ``math.cos`` at
#: load); ``tests/test_action_spec_v1.py`` holds it to the stick quantizer
#: and to ``atan2_q16`` landing in each bin. Feet and aim share it.
COMPASS_Q16 = (
    (65536, 0), (60547, -25080), (46341, -46341), (25080, -60547),
    (0, -65536), (-25080, -60547), (-46341, -46341), (-60547, -25080),
    (-65536, 0), (-60547, 25080), (-46341, 46341), (-25080, 60547),
    (0, 65536), (25080, 60547), (46341, 46341), (60547, 25080),
)

# -- aim ----------------------------------------------------------------------
AIM_WALK = 16                        # "where I walk": set_aim(0, 0)
AIM_KEEP = 17                        # "keep the throw's aim"
AIM_TOKENS = 18                      # entry i is AIM_TOKENS + i
#: The list entry types an aim may point at (an enemy seen now, a
#: last-seen record: a unit in the team picture). v1 masks the others
#: (mates, objects, doors) -- they open by a mask change (design 2.8).
AIMABLE_TYPES = (1, 2)

# -- trigger ------------------------------------------------------------------
N_TRIGGER = 2

# -- hand ---------------------------------------------------------------------
#: The fixed hand verbs, in index order; the throws follow, one per
#: hand-grenade row.
FIXED_HANDS = ("nothing", "reload", "melee", "shove", "use", "plant",
               "flashlight")
HAND_NOTHING, HAND_RELOAD, HAND_MELEE, HAND_SHOVE, HAND_USE, HAND_PLANT, \
    HAND_FLASHLIGHT = range(len(FIXED_HANDS))
HAND_THROW0 = len(FIXED_HANDS)       # 7: throw of hand-grenade row k = 7 + k

# -- arg ----------------------------------------------------------------------
N_ARG = 5
ARG_ONTO = 4                         # a throw onto the aimed unit
N_FRACTIONS = 4                      # the four reaches / the four timers

#: The masks a ``"v1"`` action run serves in ``obs`` (each with a ``final_``
#: twin), in head order; the argument's mask is :func:`arg_mask`'s.
MASK_KEYS = ("mask_feet", "mask_aim", "mask_trigger", "mask_hand")


def head_sizes(n_entries: int, n_kinds: int) -> tuple:
    """The five heads' sizes for a list of ``n_entries`` entries and
    ``n_kinds`` hand-grenade rows: ``(49, 18 + E, 2, 7 + K, 5)``."""
    return (N_FEET, AIM_TOKENS + int(n_entries), N_TRIGGER,
            HAND_THROW0 + int(n_kinds), N_ARG)


def hand_names(hand_grenade_rows) -> tuple:
    """Each hand entry's stable name (design 2.5): the fixed verbs, then
    ``"throw:<row>"`` per hand-grenade row in config order."""
    return FIXED_HANDS + tuple(f"throw:{r}" for r in hand_grenade_rows)


def mask_shapes(n_envs: int, n_rows: int, n_entries: int,
                n_kinds: int) -> dict:
    """The mask keys' shapes (``(N, U, size)`` uint8 each)."""
    s = head_sizes(n_entries, n_kinds)
    n, u = int(n_envs), int(n_rows)
    return {k: ((n, u, s[h]), np.uint8) for h, k in enumerate(MASK_KEYS)}


# ---------------------------------------------------------------------------
# The numbers inside the tables (quantized once per match)
# ---------------------------------------------------------------------------
def icbrt_round(n: int) -> int:
    """The cube root of the non-negative int ``n`` rounded to nearest, in
    exact integers: ``t`` the floor root, ``+1`` when ``8 n >= (2t + 1)^3``
    (the half-way point (t + 1/2)^3 cubed out)."""
    n = int(n)
    if n < 0:
        raise ValueError(f"icbrt_round: {n} < 0")
    if n < 2:
        return n
    t = 1 << ((n.bit_length() + 2) // 3)     # >= the root
    while True:
        s = (2 * t + n // (t * t)) // 3
        if s >= t:
            break
        t = s
    while t * t * t > n:
        t -= 1
    while (t + 1) ** 3 <= n:
        t += 1
    if 8 * n >= (2 * t + 1) ** 3:
        t += 1
    return t


def plant_timers(lo: int, hi: int) -> tuple:
    """The four plant timers from the row's range ``[lo, hi]`` ticks: ``t_j``
    = the integer cube root, rounded to nearest, of ``lo^(3-j) hi^j`` -- the
    ends ``lo`` and ``hi``, spaced by one factor, no float power."""
    return tuple(icbrt_round(lo ** (3 - j) * hi ** j) for j in range(4))


@dataclass(frozen=True)
class ActionTables:
    """The spec's numbers for one match (:func:`action_tables`)."""
    n_slots: int
    n_entries: int
    hand_grenade_rows: tuple
    throttles: tuple             # walk, half, full (Q16 throttle)
    fuse_ticks: int              # the one hand-grenade fuse
    reaches_q: tuple             # the four reaches, Q16 tiles
    plant_timers: tuple          # the four plant timers, ticks
    aim_lo: int                  # list positions [aim_lo, aim_hi) hold the
    aim_hi: int                  # aimable blocks (seen enemies, records)

    @property
    def n_kinds(self) -> int:
        return len(self.hand_grenade_rows)

    @property
    def sizes(self) -> tuple:
        return head_sizes(self.n_entries, self.n_kinds)

    @property
    def hand_names(self) -> tuple:
        return hand_names(self.hand_grenade_rows)


def action_tables(sim, n_slots: int) -> ActionTables:
    """The tables of ``sim``'s match for a layout of ``n_slots`` roster
    slots, from their rows: the Throttle row's walk / half / full, the hand
    grenade's ``fuse_default_seconds`` and ``max_throw_range``, the plant
    row's fuse range -- each quantized once."""
    from simulation.obs_spec_v1 import CAPS_FIXED, ENTRIES_BEYOND_SLOTS
    tables = sim.weapons_tables
    tps = sim._tps
    hand = tables.weapons.by_name["hand_grenade"]
    c4 = tables.weapons.by_name[PLANT_WEAPON]
    rng_q = int(hand.max_throw_range) << 16
    n_slots = int(n_slots)
    aim_lo = n_slots - 1
    return ActionTables(
        n_slots=n_slots, n_entries=n_slots + ENTRIES_BEYOND_SLOTS,
        hand_grenade_rows=tuple(tables.hand_grenade_rows),
        throttles=(0, THROTTLE_ONE // 2, THROTTLE_ONE),
        fuse_ticks=fuse_ticks_from_seconds(hand.fuse_default_seconds, tps),
        reaches_q=tuple((j + 1) * rng_q // N_FRACTIONS
                        for j in range(N_FRACTIONS)),
        plant_timers=plant_timers(
            fuse_ticks_from_seconds(c4.fuse_min_seconds, tps),
            fuse_ticks_from_seconds(c4.fuse_max_seconds, tps)),
        aim_lo=aim_lo, aim_hi=aim_lo + CAPS_FIXED[0] + CAPS_FIXED[1])


# ---------------------------------------------------------------------------
# Decoding helpers (the decoder's and the observation's one geometry)
# ---------------------------------------------------------------------------
def feet_option(index: int) -> tuple:
    """``(direction, speed)`` of feet ``index`` (``None`` for stand)."""
    index = int(index)
    if index == FEET_STAND:
        return None
    k = index - 1
    return (k // N_SPEEDS, k % N_SPEEDS)


def aim_vector(me, centre):
    """The Q16.16 aim vector from ``me``'s centre tile to the tile
    ``centre`` = ``(x, y)``, through the one control-seam quantizer (the
    bots' ``_direction_q``); ``None`` at a zero offset. Called by the
    decoder and by the observation's ``on_next_tick`` column."""
    dx = int(centre[0]) - int(me.center_tile_x())
    dy = int(centre[1]) - int(me.center_tile_y())
    qx, qy, mag = quantize_stick_direction(float(dx), float(dy), 0.0)
    return None if mag == 0 else (qx, qy)


def onto_reach_q(me, centre) -> int:
    """The lob distance onto ``centre`` (the bots' ``_lob``): the centre-tile
    distance as Q16 tiles, ``quantize_scalar(sqrt(d2))``."""
    dx = int(centre[0]) - int(me.center_tile_x())
    dy = int(centre[1]) - int(me.center_tile_y())
    return quantize_scalar(float(dx * dx + dy * dy) ** 0.5)


def facing_vector(u) -> tuple:
    """``u``'s facing as a Q16 vector ``(cos, -sin)`` (world y down): the
    direction a throw is issued with (a wind-up releases along the facing
    the body has then)."""
    f = int(u.facing_q)
    return (cos_q16(f), -sin_q16(f))


# ---------------------------------------------------------------------------
# The masks (note section 3)
# ---------------------------------------------------------------------------
def default_masks(out: dict, i: int) -> None:
    """Row ``i`` of ``out``'s masks as an INVALID row's: exactly one legal
    option per head (stand, "where I walk", release, nothing)."""
    for k in MASK_KEYS:
        out[k][i] = 0
    out["mask_feet"][i, FEET_STAND] = 1
    out["mask_aim"][i, AIM_WALK] = 1
    out["mask_trigger"][i, 0] = 1
    out["mask_hand"][i, HAND_NOTHING] = 1


def build_masks(sim, units, list_type, list_mask, list_uid,
                tables: ActionTables, decimation: int, out: dict) -> None:
    """The four masks for every row, at the START of the decision tick
    ``T = sim.tick``: ``units`` one living unit or ``None`` per row (the env's
    ``row_units``), ``list_type`` / ``list_mask`` / ``list_uid`` the
    observation's ``(U, E)`` columns just built, ``out`` the ``(U, size)``
    views of :data:`MASK_KEYS`. Each hand bit is ONE call of its 1e gate.
    ``list_uid`` is not read: an entry's legality is its type and its mask
    (a record of an enemy that died unseen stays legal, design 2.4)."""
    del list_uid
    T = int(sim.tick)
    last = T + int(decimation) - 1
    wt = sim.weapons_tables
    aimable = (np.asarray(list_mask) != 0) & np.isin(
        np.asarray(list_type).astype(np.int64), AIMABLE_TYPES)
    n_kinds = tables.n_kinds
    for i, u in enumerate(units):
        if u is None:
            default_masks(out, i)
            continue
        out["mask_feet"][i] = 1
        out["mask_trigger"][i] = 1
        aim = out["mask_aim"][i]
        if int(u.throw_release_tick) >= last:
            # the wind-up holds the aim on every tick of the coming decision
            aim[...] = 0
            aim[AIM_KEEP] = 1
        else:
            aim[:AIM_KEEP] = 1
            aim[AIM_KEEP] = 0
            aim[AIM_TOKENS:] = aimable[i]
        hand = out["mask_hand"][i]
        hand[HAND_NOTHING] = 1
        hand[HAND_RELOAD] = verb_gates.can_reload(sim, u)
        hand[HAND_MELEE] = verb_gates.can_melee(u, T, wt, at_tick_start=True)
        hand[HAND_SHOVE] = verb_gates.can_shove(sim, u)
        hand[HAND_USE] = verb_gates.can_use(sim, u)
        hand[HAND_PLANT] = verb_gates.can_plant(sim, u)
        hand[HAND_FLASHLIGHT] = 0                       # reserved (design 2.6)
        for k in range(n_kinds):
            hand[HAND_THROW0 + k] = verb_gates.can_throw(sim, u, k)


def arg_mask(hand, aim):
    """The argument's mask, a pure function of the drawn ``hand`` and
    ``aim`` (any equal shapes, int): ``(..., 5)`` uint8 -- a throw: 0..3,
    and 4 iff the aim is a list entry; a plant: 0..3; any other hand: 0
    alone."""
    hand = np.asarray(hand, dtype=np.int64)
    aim = np.asarray(aim, dtype=np.int64)
    out = np.zeros(hand.shape + (N_ARG,), dtype=np.uint8)
    throw = hand >= HAND_THROW0
    four = throw | (hand == HAND_PLANT)
    out[..., :N_FRACTIONS] = four[..., None]
    out[..., ARG_ONTO] = throw & (aim >= AIM_TOKENS)
    out[..., 0] |= ~four
    return out


def heads_used(masks: dict, argm) -> dict:
    """Per head, whether it enters the learning signal: more than one legal
    option (design 8.2: a head with ONE legal option has probability 1,
    log-probability 0, no gradient, no entropy). ``masks`` holds
    :data:`MASK_KEYS`; ``argm`` is :func:`arg_mask`'s result."""
    out = {h: np.asarray(masks[k]).sum(axis=-1) > 1
           for h, k in zip(HEADS, MASK_KEYS)}
    out["arg"] = np.asarray(argm).sum(axis=-1) > 1
    return out


def check_actions(actions, sizes) -> None:
    """Refuse any index outside ``[0, size)`` of its head, naming the env,
    the row and the head (``actions`` ``(N, U, 5)``): an out-of-range
    integer is a trainer bug, never decoded."""
    a = np.asarray(actions)
    for h, n in enumerate(sizes):
        bad = np.argwhere((a[..., h] < 0) | (a[..., h] >= n))
        if len(bad):
            e, r = (int(v) for v in bad[0][:2])
            raise ValueError(
                f"action out of range: env {e}, row {r}, head {HEADS[h]!r} "
                f"sent {int(a[e, r, h])} (size {n})")


def masked_choices(actions, masks: dict):
    """Per row, how many heads' sent options were masked (``(N, U)``
    uint8): the four served masks and the realised :func:`arg_mask`."""
    a = np.asarray(actions, dtype=np.int64)
    count = np.zeros(a.shape[:-1], dtype=np.uint8)
    for h, k in enumerate(MASK_KEYS):
        m = np.asarray(masks[k])
        got = np.take_along_axis(m, a[..., h:h + 1], axis=-1)[..., 0]
        count += (got == 0)
    am = arg_mask(a[..., 3], a[..., 1])
    got = np.take_along_axis(am, a[..., 4:5], axis=-1)[..., 0]
    count += (got == 0)
    return count


# ---------------------------------------------------------------------------
# The inverse helpers (1j: a bot decision -> the nearest choice)
# ---------------------------------------------------------------------------
def _nearest_dir(dx_q: int, dy_q: int, stride: int = 1) -> int:
    """The compass bin nearest the angle of ``(dx_q, dy_q)`` (``atan2_q16``,
    the facing convention) among every ``stride``-th bin (1j: 2 = the eight
    even bins), ties to the bin counter-clockwise; at ``stride = 1`` the 1h
    formula, bit for bit."""
    a = atan2_q16(-int(dy_q), int(dx_q))
    s = int(stride)
    k = (a * 8 + s * (PI_Q16 >> 1)) // (s * PI_Q16)
    return (k * s) % N_DIRS


def _nearest(values, v) -> int:
    best = None
    for j, x in enumerate(values):
        d = abs(int(x) - int(v))
        if best is None or d < best[0]:
            best = (d, j)
    return best[1]


def feet_index(dx_q, dy_q, throttle_q, tables: ActionTables,
               stride: int = 1) -> int:
    """The feet option nearest a move ``(dx_q, dy_q)`` at ``throttle_q``:
    the nearest compass bin by angle (among every ``stride``-th) and the
    nearest of the three throttles; a cleared move (``None``) or a zero
    vector -> stand."""
    if dx_q is None or (int(dx_q) == 0 and int(dy_q) == 0):
        return FEET_STAND
    return (1 + N_SPEEDS * _nearest_dir(dx_q, dy_q, stride)
            + _nearest(tables.throttles, throttle_q))


def aim_index_for_vector(dx_q, dy_q, stride: int = 1) -> int:
    """The nearest compass token of an aim vector (among every
    ``stride``-th); a zero vector (or ``None``) -> "where I walk"."""
    if dx_q is None or (int(dx_q) == 0 and int(dy_q) == 0):
        return AIM_WALK
    return _nearest_dir(dx_q, dy_q, stride)


def aim_index_for_unit(uid, list_uid_row, list_type_row):
    """The aim index of the list entry holding unit ``uid`` among the
    aimable types, or ``None``."""
    uids = np.asarray(list_uid_row)
    types = np.asarray(list_type_row).astype(np.int64)
    hit = np.flatnonzero((uids == int(uid)) & np.isin(types, AIMABLE_TYPES))
    return None if len(hit) == 0 else AIM_TOKENS + int(hit[0])


def throw_choice(kind: int, reach_q: int, aimed: bool,
                 tables: ActionTables) -> tuple:
    """``(hand, arg)`` of a throw of hand-grenade row ``kind``: onto the
    aimed entry (``aimed``) or the nearest of the four reaches."""
    if not 0 <= int(kind) < tables.n_kinds:
        raise ValueError(f"throw_choice: kind {kind} is not a hand-grenade "
                         f"row ({tables.hand_grenade_rows})")
    arg = ARG_ONTO if aimed else _nearest(tables.reaches_q, reach_q)
    return (HAND_THROW0 + int(kind), arg)


def plant_choice(fuse_ticks: int, tables: ActionTables) -> tuple:
    """``(hand, arg)`` of a plant: the nearest of the four timers (ticks)."""
    return (HAND_PLANT, _nearest(tables.plant_timers, fuse_ticks))


__all__ = [
    "ACTION_SPEC_NAME", "ACTION_SPECS", "HEADS", "N_DIRS", "N_SPEEDS",
    "N_FEET", "FEET_STAND", "COMPASS_Q16", "AIM_WALK", "AIM_KEEP",
    "AIM_TOKENS", "AIMABLE_TYPES", "N_TRIGGER", "FIXED_HANDS", "HAND_NOTHING",
    "HAND_RELOAD", "HAND_MELEE", "HAND_SHOVE", "HAND_USE", "HAND_PLANT",
    "HAND_FLASHLIGHT", "HAND_THROW0", "N_ARG", "ARG_ONTO", "N_FRACTIONS",
    "MASK_KEYS", "head_sizes", "hand_names", "mask_shapes", "icbrt_round",
    "plant_timers", "ActionTables", "action_tables", "feet_option",
    "aim_vector", "onto_reach_q", "facing_vector", "default_masks",
    "build_masks", "arg_mask", "heads_used", "check_actions",
    "masked_choices", "feet_index", "aim_index_for_vector",
    "aim_index_for_unit", "throw_choice", "plant_choice",
]
