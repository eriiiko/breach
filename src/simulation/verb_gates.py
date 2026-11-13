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

THE TRIGGER (183-no-fire-windup) has one gate too, ``can_fire``: the hands
are not winding up a throw. It is read by the shooting slot (4) and the
sprays, by the trigger mask and by the fire lanes; the flashlight has none.

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


def _reach_key(u, door):
    """The row-major index, over the unit's footprint plus its one-tile ring
    (the verb's own geometry), of the first ring tile ``door``'s span holds;
    ``None`` when the span touches none. THE one statement of "in reach":
    :func:`door_in_reach` reads it, :func:`use_door` orders by it."""
    x0, y0 = u.tile_x, u.tile_y
    fp = u.footprint
    side = fp + 2
    best = None
    for ty, tx in door.span:
        ry, rx = int(ty) - (y0 - 1), int(tx) - (x0 - 1)
        if 0 <= ry < side and 0 <= rx < side:
            k = ry * side + rx
            if best is None or k < best:
                best = k
    return best


def door_in_reach(u, door) -> bool:
    """Whether ``door``'s span touches ``u``'s footprint or its one-tile ring
    -- the use verb's own geometry per door (#195 1g: the door entries'
    ``in_reach``). Liveness is not a term. A pure read."""
    return _reach_key(u, door) is not None


def use_door(sim, u):
    """The door USE would flip: the first live door whose span touches the
    unit's footprint or its one-tile ring -- the live door holding the
    row-major FIRST ring tile any live door holds (what the ring scan
    through ``Simulation.door_at`` returned; spans are disjoint, so the tile
    names one door), or ``None``. THE scan of the verb, on
    :func:`door_in_reach`'s geometry."""
    best = None
    for door in sim._doors:
        if not door.alive:
            continue
        k = _reach_key(u, door)
        if k is not None and (best is None or k < best[0]):
            best = (k, door)
    return None if best is None else best[1]


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


def can_fire(u, tick: int, *, at_tick_start: bool) -> bool:
    """FIRE (the trigger, 183-no-fire-windup; Erik's ruling 2026-11-13: "a
    marine should not be able to fire when throwing a grenade"): the hands
    are not winding up a throw. ONLY that term (and the 1b envelope): the
    cadence, the magazine, ``can_act`` and the archetype stay the shooting
    slot's own reads, so a True here never promises a round -- a False
    promises none (the one-sided promise).

    THE TICK BOUNDARY. The shooting slot (4) runs after 1b, where
    ``_advance_throw`` releases a wind-up whose release tick has come, so on
    the RELEASE tick the hand is free and a held trigger fires again; on the
    tick a wind-up STARTS (1b, before slot 4) the trigger already does
    nothing (a 0-tick wind-up releases at once and blocks nothing).
    ``at_tick_start=False`` is slot 4's own read (the literal ``< 0``).
    ``at_tick_start=True`` asks, from the state at the START of a tick T,
    whether the hand will be free at slot 4 of tick ``tick`` (>= T): busy
    only if the wind-up is still running then (``throw_release_tick >
    tick``) and 1b will keep it (the unit able to act -- ``_advance_throw``
    drops it otherwise). The v1 masks ask it for a decision's LAST tick (a
    hold that fires on any tick of the decision is legal); the fire lanes
    for the coming tick."""
    if not _acts(u):
        return False
    if at_tick_start:
        return not (composed_flags(u).can_act
                    and u.throw_release_tick > tick)
    return u.throw_release_tick < 0


__all__ = ["can_throw", "can_plant", "door_in_reach", "use_door", "can_use",
           "can_reload", "can_shove", "can_melee", "can_fire"]
