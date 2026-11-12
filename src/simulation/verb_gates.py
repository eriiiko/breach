"""The hand verbs' GATES (#195 patch 1e; ``docs/rl_agent_rung1_note_1e_2026-
11-12.md``, design ``docs/rl_agent_design_2026-11-11.md`` 2.4).

WHETHER A HAND VERB WOULD ACT IF ISSUED NOW is ONE pure read per verb, here
and nowhere else. The verb calls its gate at its consumption (slot 1b for
throw, plant, use, reload, shove; slot 4a for melee) and the policy's masks
(1h) call the same gate at the START of the tick, so the mask and the effect
cannot drift apart.

Every function is a PURE READ: no write, no settle, no event, no draw, no
noise, no cool-down. It answers ``bool`` (the masks are legal / not legal; no
reason code). The alive / not-zombie terms repeat the 1b envelope
(``Simulation._consume_direct_intents``: it acts only for a living non-zombie
unit) so a mask gets the whole answer; at a verb's own call site they are
always true. Possession is not a term: issuing a hand verb grants it, and
melee needs none.

WHEN THE GATE IS READ. The mask is made from the state at the START of tick T
(after tick T-1's step, before slot 1a); a verb decides at its consumption
point. The gates are written so that one function answers both, exactly,
except for events in between that can only FORBID a verb (a slot-1a
knockdown, a slot-2 blast, a slot-2b damage-over-time death): gate False at
tick start => the verb issued at T does nothing (the design's one-sided
promise). What stays one step early, by design: a gate True at tick start may
still do nothing, and for shove and melee the TARGET is never a term -- a
miss costs nothing, so whether anyone is in reach is no gate's business.

* Slot-1b verbs: the state read at tick start equals the state read at 1b,
  with one formulation: ``can_plant`` is decided after ``_advance_throw`` has
  released a wind-up whose release tick is T, so "busy" is
  ``throw_release_tick > tick`` (a wind-up releasing at T still reads T at
  tick start, and the plant acts on that very tick). ``can_throw`` is decided
  BEFORE any advance, so its busy terms are the plain ``>= 0``.
* Melee (4a) reads the state after 1b and the 2b sweep: ``can_melee`` takes
  the moment as a required keyword. ``at_tick_start=False`` (the verb): the
  literal terms. ``at_tick_start=True`` (the mask): the statuses that the
  coming sweep releases are already gone (``composed_flags(after_sweep=
  True)``), and a wind-up or plant is busy only if ``_advance_throw`` /
  ``_advance_plant`` will keep it at 1b (not past, and the unit able to act).

THE TWO OVERSIGHTS ARE KEPT EXACTLY (design 11, Erik's to rule on): a stunned
(knocked-down, paralysed) unit can reload and can use a door; reload also
ignores a wind-up and a plant. ``can_reload`` and ``can_use`` carry no
``can_act`` term and ``tests/test_verb_gates.py`` pins that, so a later ruling
is a deliberate test change.

A new hand verb brings its gate here in the same patch -- never a mask's copy
of the decision.
"""
from __future__ import annotations

from simulation import combat, unit_impulse
from simulation.status import composed_flags


def _acts(u) -> bool:
    """The 1b envelope's unit terms: living and not a zombie."""
    return u.alive and not u.is_zombie


def can_throw(sim, u, kind: int) -> bool:
    """THROW of hand-grenade kind ``kind``: not already winding up, not
    planting, at least one of that kind carried, and able to act. An unknown
    kind is refused by name, as the verb refuses it."""
    sim.weapons_tables.hand_grenade_kind(kind)           # refuses by name
    return (_acts(u)
            and u.throw_release_tick < 0
            and u.plant_done_tick < 0
            and u.grenade_counts[kind] > 0
            and composed_flags(u).can_act)


def can_plant(sim, u) -> bool:
    """PLANT: not planting, no wind-up that will still be running at the
    plant's consumption (``throw_release_tick > tick``; a wind-up releasing
    on this very tick has let go of the hand by then), a charge carried, and
    able to act."""
    return (_acts(u)
            and u.plant_done_tick < 0
            and u.throw_release_tick <= sim.tick
            and u.has_explosive > 0 and bool(u.charge_ammo)
            and composed_flags(u).can_act)


def use_door(sim, u):
    """The door USE would flip: the first live door whose span touches the
    unit's footprint or its one-tile ring (``Simulation.door_at``, row-major
    over the ring), or ``None``. THE scan of the verb, moved here."""
    x0, y0 = u.tile_x, u.tile_y
    fp = u.footprint
    # Footprint tiles plus a one-tile ring so a marine standing beside a
    # door can open it without overlapping it.
    for ty in range(y0 - 1, y0 + fp + 1):
        for tx in range(x0 - 1, x0 + fp + 1):
            door = sim.door_at(ty, tx)
            if door is not None and door.alive:
                return door
    return None


def can_use(sim, u) -> bool:
    """USE: a live door within reach (:func:`use_door`). No ``can_act`` term
    (the oversight, kept)."""
    return _acts(u) and use_door(sim, u) is not None


def can_reload(sim, u) -> bool:
    """RELOAD: the weapon tracks a magazine, no stall is running, the
    magazine is not full, and a spare exists (counted, or untracked: a
    loadout without ``magazines`` reloads without spending one). Read through
    ``combat.mag_status`` -- the magazine as ``mag_settle`` would leave it, a
    never-bound unit showing a full one -- so the gate writes nothing. No
    ``can_act``, wind-up or plant term (the oversight, kept)."""
    if not _acts(u):
        return False
    w = sim.weapons_tables.weapons.by_name.get(getattr(u, "weapon_id", ""))
    if w is None or w.mag_size <= 0:
        return False
    rounds, spares, reloading = combat.mag_status(u, w, sim.tick)
    return (not reloading and rounds < w.mag_size and spares != 0)


def can_shove(sim, u) -> bool:
    """SHOVE: able to act and off cool-down. Whether anyone stands in reach
    is not a term (a miss spends no cool-down)."""
    return (_acts(u)
            and composed_flags(u).can_act
            and unit_impulse.shove_cooled(sim.impulse_table, u, sim.tick))


def can_melee(u, tick: int, tables, *, at_tick_start: bool) -> bool:
    """MELEE: a melee row, able to act, no wind-up, no plant, the row's
    cadence run. ``tables`` is the weapons tables (``Simulation.
    weapons_tables``); the moment is a required keyword (the module
    docstring): ``at_tick_start=False`` is slot 4a's own read (the verb),
    ``True`` the same decision read at the start of the tick (the mask).
    Whether anyone is in the arc is not a term."""
    if not _acts(u) or not u.melee_id:
        return False
    if at_tick_start:
        if not composed_flags(u, after_sweep=True).can_act:
            return False
        # What 1b does to a wind-up / plant first: kept only while not past
        # and the unit can still act (``_advance_throw`` / ``_advance_plant``).
        holds = composed_flags(u).can_act
        if holds and (u.throw_release_tick > tick
                      or u.plant_done_tick > tick):
            return False
    else:
        if (not composed_flags(u).can_act or u.throw_release_tick >= 0
                or u.plant_done_tick >= 0):
            return False
    weapon = tables.weapons.by_name[u.melee_id]
    return (u.last_blow_tick < 0
            or tick - u.last_blow_tick >= weapon.rof_interval_ticks)


__all__ = ["can_throw", "can_plant", "use_door", "can_use", "can_reload",
           "can_shove", "can_melee"]
