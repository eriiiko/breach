"""Unit traits -- how ONE unit differs from its class (#29 U1;
``docs/unit_design_2026-11-14.md`` v2, sections 2, 3 and 11(b)).

A HOOK is this individual's factor on a CLASS value that a table holds (a
``[speeds]`` row, a weapon row's ``reload_seconds``, ``[marine] hp`` ...): a
Q16 multiplier in the unit's ``traits`` block (a group-S unit-schema column,
default all :data:`ONE`, so absent from every digest while neutral). The
table row stays the class value; the hook says how this person differs.

- :func:`compile_unit` is the ONLY writer of ``traits``. It runs at SPAWN
  (``Simulation.add_unit``) and never inside a tick; a conversion keeps the
  block (the row is the same unit -- a strong crewman is a strong zombie):
  what differs for a zombie is the CLASS value, selected on ``is_zombie`` at
  the law, as before. Today every sheet compiles to ``ONE`` (the stats are
  the species means and no formula maps them yet: #199); ``hooks=`` is the
  sheet override a test (and later the drawn sheet) hands in, refused by
  :func:`check_traits` outside its door.
- Each hooked value has ONE accessor and EVERY reader calls it -- the law,
  the bots' predictors, the observation, the masks, the panel -- never the
  raw row (``tests/test_unit_traits.py`` scans ``src/``). The accessor is
  ``round(class value x hook)``: ONE more integer rounding on the cached
  integer class value (:func:`scaled`, ``unit_fixed.mul_q16_round``), the
  identity at ``ONE`` (and skipped there, so a neutral unit is today's unit
  bit for bit). Never a float.
- What an accessor returns is fixed for a tick (D5): today it reads only the
  block and a table, neither of which a tick writes.

THE HOOK TABLE (:data:`HOOKS`, the order of the block) and where each one's
accessor lives:

=========  ===============================================  ==================
hook       scales                                           accessor
=========  ===============================================  ==================
HP         ``[marine] hp`` / ``[zombie] hp``                :func:`max_hp_q`
WALK       the walk cost (a zombie's shamble)               ``FootprintSamples
                                                            .speed_hook_q16``
                                                            in ``movement.
                                                            default_speed``
SPRINT     the full-speed cost (a zombie's sprint)          the same member
TURN       the top turning speed and the acceleration       ``Simulation.unit_
                                                            turn_rate_q16`` /
                                                            ``_accel_q16``
HOLD       the knockdown stability                          ``unit_impulse.
                                                            stability_q16``
STRIKE     a melee row's damage and the zombie bite's       :func:`strike_
                                                            damage_q`
PUSH       a melee row's impulse and the shove's            :func:`push_
                                                            impulse_q`
SPREAD     the shot cone (aimed, snap, beam, onephase's)    :func:`cone_q16`
                                                            (+ :func:`shot_
                                                            cone_q16`)
HANDLING   a reload's, a throw wind-up's, a plant's ticks   :func:`reload_
                                                            ticks`, :func:`
                                                            throw_windup_
                                                            ticks`, :func:`
                                                            plant_ticks`
=========  ===============================================  ==================

A new hook is a row of :data:`HOOKS` (APPENDED: the block's order is the
device store's), ONE accessor, and every reader through it; its device twin
follows the twin ledger (Class T at catch-up).

Import-light: nothing here imports :mod:`simulation.unit` (the unit schema
reads :data:`N_HOOKS` for the block's width and ``unit.py`` builds its
``__slots__`` from the schema).
"""
from __future__ import annotations

from dataclasses import dataclass

from config import CFG
from simulation import unit_fixed

#: The Q16 one: a neutral hook.
ONE = unit_fixed.FP_ONE
#: The door's bounds on a hook (the integer headroom of every accessor; each
#: carries its own int64 range argument below).
HOOK_MIN = ONE // 8
HOOK_MAX = 8 * ONE


@dataclass(frozen=True)
class Hook:
    """One row of the hook table: its name, the class value it scales and
    the accessor every reader calls."""
    name: str
    scales: str
    accessor: str


#: THE hook table, in the block's order (append only).
HOOKS: tuple[Hook, ...] = (
    Hook("HP", "the class's full health: [marine] hp, [zombie] hp for a "
         "zombie", "unit_traits.max_hp_q"),
    Hook("WALK", "the walk speed (a zombie's shamble)",
         "movement.default_speed via FootprintSamples.speed_hook_q16"),
    Hook("SPRINT", "the full speed (a zombie's sprint)",
         "movement.default_speed via FootprintSamples.speed_hook_q16"),
    Hook("TURN", "the top turning speed and the angular acceleration",
         "Simulation.unit_turn_rate_q16 / unit_turn_accel_q16"),
    Hook("HOLD", "the knockdown stability (the fall limit, the effort "
         "knockdown)", "unit_impulse.stability_q16"),
    Hook("STRIKE", "the damage of a melee blow and of a bite",
         "unit_traits.strike_damage_q"),
    Hook("PUSH", "the impulse of a melee blow and of a shove",
         "unit_traits.push_impulse_q"),
    Hook("SPREAD", "the shot cone's half-angle: aimed, snap, beam",
         "unit_traits.cone_q16"),
    Hook("HANDLING", "the ticks of a reload, a throw's wind-up and a plant",
         "unit_traits.reload_ticks / throw_windup_ticks / plant_ticks"),
)
N_HOOKS = len(HOOKS)
HOOK_INDEX = {h.name: i for i, h in enumerate(HOOKS)}
(HP, WALK, SPRINT, TURN, HOLD, STRIKE, PUSH, SPREAD,
 HANDLING) = range(N_HOOKS)
assert tuple(HOOK_INDEX) == ("HP", "WALK", "SPRINT", "TURN", "HOLD",
                             "STRIKE", "PUSH", "SPREAD", "HANDLING")


def neutral_block() -> list:
    """A fresh all-``ONE`` block (the schema column's default)."""
    return [ONE] * N_HOOKS


# ---------------------------------------------------------------------------
# The door and the compile step (the only writer)
# ---------------------------------------------------------------------------
def check_traits(sim, traits) -> None:
    """THE door of a hook block: ``N_HOOKS`` Q16 ints, each in
    ``[HOOK_MIN, HOOK_MAX]``; ``TURN`` also may not take the turning law past
    its trial bound at ``sim``'s class dials (``turning.TURN_TRIALS_MAX``,
    the device loop's fixed length) nor round a dial to 0. Refused by hook
    name (ValueError)."""
    from simulation import turning
    if len(traits) != N_HOOKS:
        raise ValueError(f"unit traits: a block holds {N_HOOKS} hooks, got "
                         f"{len(traits)}")
    for h, v in zip(HOOKS, traits):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"unit traits: hook {h.name} = {v!r} is not a "
                             f"Q16 integer")
        if not HOOK_MIN <= v <= HOOK_MAX:
            raise ValueError(f"unit traits: hook {h.name} = {v} is outside "
                             f"[{HOOK_MIN}, {HOOK_MAX}] (ONE/8 .. 8 ONE)")
    if sim is not None:
        h = traits[TURN]
        w = scaled_by(h, sim._marine_turn_rate_q16)
        a = scaled_by(h, sim._marine_turn_accel_q16)
        if w < 1 or a < 1:
            raise ValueError(f"unit traits: hook TURN = {h} rounds a turning "
                             f"dial to 0 (w {w}, a {a})")
        trials = -(-min(w, turning.PI_Q16) // a)
        if trials > turning.TURN_TRIALS_MAX:
            raise ValueError(f"unit traits: hook TURN = {h} takes the turning "
                             f"law to {trials} trials a tick (at most "
                             f"{turning.TURN_TRIALS_MAX})")


def compile_unit(sim, u, hooks=None) -> None:
    """THE compile step (design 2.2): the unit's hooks and its compiled
    quantities from its sheet, at spawn only. Writes ``u.traits`` (every
    hook ``ONE`` -- no formula maps a stat yet, #199 -- bar the sheet
    overrides ``hooks``, a ``{hook name: Q16 int}`` mapping) through
    :func:`check_traits`, and ``u.mass_g`` through ``unit_impulse.bind_mass``
    (body + kit). Never called inside a tick; a conversion keeps the block."""
    from simulation import unit_impulse
    block = neutral_block()
    for name, v in (hooks or {}).items():
        if name not in HOOK_INDEX:
            raise ValueError(f"unit traits: {name!r} names no hook (known: "
                             f"{list(HOOK_INDEX)})")
        block[HOOK_INDEX[name]] = v
    check_traits(sim, block)
    u.traits = block
    unit_impulse.bind_mass(u)


# ---------------------------------------------------------------------------
# The one rounding
# ---------------------------------------------------------------------------
def hook(u, h: int) -> int:
    """``u``'s hook ``h`` (a Q16 int); a stand-in without a block reads
    ``ONE``."""
    t = getattr(u, "traits", None)
    return ONE if t is None else t[h]


def scaled_by(hq: int, class_value: int) -> int:
    """``class_value`` (an integer in its own currency) times the Q16 hook
    ``hq``, rounded ONCE half away from zero (``unit_fixed.mul_q16_round``);
    ``class_value`` itself at ``ONE``. int64: ``|class_value| * HOOK_MAX``
    -- each accessor below states its bound."""
    if hq == ONE:
        return class_value
    return unit_fixed.mul_q16_round(class_value, hq)


def scaled(u, h: int, class_value: int) -> int:
    """:func:`scaled_by` with ``u``'s hook ``h``."""
    return scaled_by(hook(u, h), class_value)


# ---------------------------------------------------------------------------
# HP
# ---------------------------------------------------------------------------
def class_hp_q(is_zombie: bool) -> int:
    """The class's full health in Q16: ``[zombie] hp`` for a zombie, else
    ``[marine] hp`` -- the config constant quantized once at this door (read
    per call, so an F5 reload is seen)."""
    v = CFG.zombie.hp if is_zombie else CFG.marine.hp
    q = unit_fixed.quantize_scalar(float(v))                  # p3e-door
    if q <= 0:
        raise ValueError(f"unit traits: the class hp {v!r} must be > 0")
    return q


def max_hp_q(u) -> int:
    """THE full health of ``u`` (Q16): its class's (:func:`class_hp_q`, on
    ``is_zombie``) times its ``HP`` hook. The spawn, the conversion, the
    panel and ``bots_v1``'s fall-back read it. int64: hp <= 2^24 Q16 (256
    points) x 2^19 = 2^43."""
    return scaled(u, HP, class_hp_q(bool(getattr(u, "is_zombie", False))))


# ---------------------------------------------------------------------------
# STRIKE, PUSH
# ---------------------------------------------------------------------------
def strike_damage_q(u, weapon=None, *, crit_mult=None) -> int:
    """THE damage (Q16, pre-mitigation) of ``u``'s strike: a melee row's
    ``melee_damage`` (``weapon``; a crit's ``crit_mult`` applied first, on
    whole points, as the order path always did), or -- ``weapon`` None --
    the zombie bite's ``[zombie] melee_damage``; times ``u``'s ``STRIKE``
    hook. int64: damage <= 2^24 Q16 x 2^19 = 2^43."""
    if weapon is None:
        base = unit_fixed.quantize_scalar(
            float(CFG.zombie.melee_damage))                    # p3e-door
    else:
        amount = int(weapon.melee_damage)
        if crit_mult is not None:
            from simulation.attack_resolver import scale_half_away
            amount = scale_half_away(amount, crit_mult)
        base = amount << unit_fixed.FP_SHIFT
    return scaled(u, STRIKE, base)


def push_impulse_q(table, u, weapon=None) -> int:
    """THE impulse (Q16 gram-tiles per tick) ``u`` puts into a blow of its
    melee row ``weapon`` (``table.melee_push_q``, 0 for a row without one)
    or -- ``weapon`` None -- into its shove (``table.shove_q``); times its
    ``PUSH`` hook. int64: a push <= 2^34 (a 1000 N s blow on 1/3 m tiles)
    x 2^19 = 2^53."""
    if weapon is None:
        base = int(table.shove_q)
    else:
        base = int(table.melee_push_q.get(weapon.name, 0))
    return scaled(u, PUSH, base)


# ---------------------------------------------------------------------------
# SPREAD
# ---------------------------------------------------------------------------
def cone_deg_of(weapon, can_aim: bool = True) -> float:
    """The CLASS shot cone a weapon row selects, in degrees: ``spread_deg``
    when the shooter can aim, ``spread_snap_deg`` when its statuses forbid
    it (mechanics/06's can_aim) -- the ONE selection (combat's two trigger
    paths and onephase's situational cone start from it)."""
    return weapon.spread_deg if can_aim else weapon.spread_snap_deg


def cone_q16(u, cone_deg) -> int:
    """THE shot cone of ``u`` (Q16.16 radians) for a class half-angle in
    degrees (:func:`cone_deg_of`, or onephase's situational cone built on
    it): the cone door ``combat.cone_q16_of_deg`` times ``u``'s ``SPREAD``
    hook. Read by the draw sites (``combat.fire_burst`` / ``fire_beam``,
    every path) and, through :func:`shot_cone_q16`, by every predictor.
    int64: a cone <= PI_Q16 < 2^18 x 2^19 = 2^37."""
    from simulation.combat import cone_q16_of_deg
    return scaled(u, SPREAD, cone_q16_of_deg(cone_deg))


def shot_cone_q16(u, weapon, can_aim: bool = True) -> int:
    """``u``'s shot cone with ``weapon`` (Q16 radians): :func:`cone_q16` of
    the row's selection -- what the bots' trigger discipline and the
    observation's aim judgements read."""
    return cone_q16(u, cone_deg_of(weapon, can_aim))


# ---------------------------------------------------------------------------
# HANDLING
# ---------------------------------------------------------------------------
def _handling(u, base_ticks: int) -> int:
    """A duration's class ticks times ``u``'s ``HANDLING`` hook, never below
    one tick when the class value is (a 0-tick wind-up or plant is another
    code path, and stays the class's). int64: ticks <= 2^16 x 2^19."""
    base = int(base_ticks)
    if base <= 0:
        return base
    t = scaled(u, HANDLING, base)
    return t if t >= 1 else 1


def reload_ticks(u, weapon) -> int:
    """THE reload stall of ``u`` with ``weapon`` (ticks): the row's
    ``reload_ticks`` times ``HANDLING``."""
    return _handling(u, weapon.reload_ticks)


def throw_windup_ticks(u, rows) -> int:
    """THE throw wind-up of ``u`` (ticks): ``[weapons.hand_grenade]
    throw_windup_seconds`` (the row's ``throw_windup_ticks``) times
    ``HANDLING``. ``rows`` is the weapon table's ``{name: row}``
    (``WeaponsTables.weapons.by_name``)."""
    return _handling(u, rows["hand_grenade"].throw_windup_ticks)


def plant_ticks(u, rows) -> int:
    """THE plant of ``u`` (ticks): the plant row's (``weapons.PLANT_WEAPON``)
    ``plant_ticks`` times ``HANDLING``; ``rows`` as above."""
    from simulation.weapons import PLANT_WEAPON
    return _handling(u, rows[PLANT_WEAPON].plant_ticks)


__all__ = [
    "ONE", "HOOK_MIN", "HOOK_MAX", "Hook", "HOOKS", "N_HOOKS", "HOOK_INDEX",
    "HP", "WALK", "SPRINT", "TURN", "HOLD", "STRIKE", "PUSH", "SPREAD",
    "HANDLING", "neutral_block", "check_traits", "compile_unit", "hook",
    "scaled_by", "scaled", "class_hp_q", "max_hp_q", "strike_damage_q",
    "push_impulse_q", "cone_deg_of", "cone_q16", "shot_cone_q16",
    "reload_ticks", "throw_windup_ticks", "plant_ticks",
]
