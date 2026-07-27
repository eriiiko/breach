"""THE TIMELINE — OnePhaseWEGO's plan compiler and executor (design §3/§6/§14).

Design §3: "a unit's plan is a **timeline** — (move to A, arrive 1.4 s) ->
(GCD to 1.9 s) -> (fire to 3.0 s)". This module is that sentence, made real.
It is the ONE place that knows how long things take, and everything else reads
its answer:

- the **planning UI** (§16) shows the arrival timestamps that ARE the compiled
  schedule — the teal endpoint label, the shoot hologram's moment, and (when it
  ships) the scrub preview are all queries against a :class:`Plan`;
- the **executor** below walks the same structure at 24 Hz;
- and because time is the only currency (§3), the compiler is also the whole
  cost model — there is nothing else charging anything anywhere.

Two invariants make this tractable:

**1. The schedule is authoritative for TIME; execution may under-deliver in
SPACE.** A compiled move step owns ticks ``[start, end)`` no matter what
happens in the world. If the path is blocked mid-round (§14: halt in place,
continue remaining non-move orders, no auto-repath, replan next round) or the
unit is knocked down, it simply stops covering ground — it never borrows ticks
from the next step, and the steps after it keep the times the player was shown.
This is what makes the displayed plan honest: the times cannot silently drift.

**2. Every tick in a plan is an ABSOLUTE tick.** The ruleset's clock is
free-running (see :class:`~simulation.ruleset.OnePhaseWEGO`), so a schedule
crossing a round boundary needs no fixups — which is exactly why §13's
"invisible seams" cost nothing to implement, and why a sustained order keeps
running into the next round until the player replaces it.

**The GCD is charged from a step's START, not its end** (§3: "the GCD gates
*changing* action"). For an instantaneous action that is the obvious reading;
for a sustained one it makes the GCD the minimum time before you may flip to a
different action, which is the design's intent stated as arithmetic. A weapon's
own salvo never re-triggers it (``gcd_exempt_within_salvo``) because the whole
salvo is ONE step.

Determinism (IRON RULE): integer tick arithmetic throughout; positions ride the
same plain-IEEE interpolation the shipped WEGO path uses; the one angular test
(forward vs reversing, §6) is a DOT PRODUCT against a threshold computed once
through the deterministic integer trig kit (``unit_fixed.cos_rad``) — no libm
on the sim path, no per-tick transcendental. No RNG is drawn here; the shooting
it triggers draws from ``sim.rng`` exactly as the shipped path does.
"""
from __future__ import annotations

import math

from config import CFG
from simulation.action_registry import ActionDef
from simulation.combat import mag_gate, mag_spend, _dispatch_trigger
from simulation.movement import FootprintSamples, default_speed
from simulation.orders import (
    ORDER_AMBUSH, ORDER_DETONATE, ORDER_HOLD, ORDER_MARK, ORDER_MOVE,
    ORDER_MOVE_SHOOT, ORDER_OVERWATCH, ORDER_SHOOT, ORDER_SWAP,
    ONEPHASE_MOVE_ORDER_TYPES,
)
from simulation.status import composed_flags
from simulation import unit_fixed

try:
    from pathfinding import astar
    HAS_PATHFINDING = True
except ImportError:                                   # pragma: no cover
    HAS_PATHFINDING = False


#: A sustained tail step has no end — it runs until the player replaces it
#: (§9: "persists across rounds until replaced").
INDEFINITE = None


# ---------------------------------------------------------------------------
# Aim-relative speed + situational spread (§6, §7)
# ---------------------------------------------------------------------------
def _reverse_cos_threshold() -> float:
    """``cos(reverse_angle_deg)`` through the deterministic integer trig kit.

    The forward/reversing test (§6) is "is the aim direction more than
    ``reverse_angle_deg`` away from the direction of travel?", which is a dot
    product against this constant — no per-step ``atan2``, and no libm anywhere
    (``math.radians`` is a plain IEEE multiply; ``cos_rad`` is the kit).
    """
    return unit_fixed.cos_rad(math.radians(CFG.onephase.reverse_angle_deg))


def is_reversing(travel_dx, travel_dy, aim_dx, aim_dy) -> bool:
    """True when the unit is backing away from where it is aiming (§6).

    Both vectors are normalized here (``math.sqrt`` is IEEE correctly-rounded —
    a basic operation, not a transcendental, the same call the control seam
    already relies on). A zero-length vector means "no opinion" -> not
    reversing.
    """
    tl = math.sqrt(travel_dx * travel_dx + travel_dy * travel_dy)
    al = math.sqrt(aim_dx * aim_dx + aim_dy * aim_dy)
    if tl <= 0.0 or al <= 0.0:
        return False
    dot = (travel_dx * aim_dx + travel_dy * aim_dy) / (tl * al)
    return dot <= _reverse_cos_threshold()


def speed_pct(order_type: int, reversing: bool) -> float:
    """The §6 aim-relative speed table, as a fraction of full move speed.

    | Move, no engagement                | 100 %  (this IS the old Sprint) |
    | Move & shoot, target ahead         | 60-75 % (dial)                  |
    | Move & shoot, reversing            | ~25 %   (dial)                  |
    """
    if order_type != ORDER_MOVE_SHOOT:
        return float(CFG.onephase.move_speed_pct)
    return float(CFG.onephase.move_shoot_reverse_speed_pct if reversing
                 else CFG.onephase.move_shoot_speed_pct)


def tile_cadence(gmap, unit, fy, fx, order_type, reversing) -> int:
    """Ticks this unit spends entering the tile-block at (fy, fx).

    Two independent multipliers compose onto the full-speed base, in the order
    the engine already establishes: TERRAIN first (``default_speed`` — the
    area-averaged mobility under the footprint, so furniture is 2.5x slower),
    then the §6 aim-relative fraction. Full speed is the marine SPRINT cadence
    because v1's Move folds Sprint in (§5).

    Integer in, integer out; the one divide is IEEE and round-half-up, matching
    the weapon table's meter->tile derivation.
    """
    base = int(CFG.movement.marine_sprint_ticks_per_tile)
    samples = FootprintSamples(
        mobility=gmap.footprint_mobility(fy, fx, unit.footprint))
    terrain = default_speed(samples, base)
    pct = speed_pct(order_type, reversing)
    if pct >= 1.0:
        return terrain
    cost = int(terrain / pct + 0.5)
    return cost if cost >= 1 else 1


def spread_deg_for(weapon, order_type, reversing=False, overwatch=False,
                   can_aim=True) -> float:
    """Accuracy IS spread angle (§7) — no to-hit roll, no modifier stack.

    One dial per situation, multiplying the weapon row's own aimed cone:
    firing on the move opens it, backpedaling opens it further, a narrowed
    overwatch cone tightens it slightly. The ``can_aim`` suppression
    (mechanics/06 BLINDED / STUNNED / PARALYZED) still swaps in the weapon's
    snap cone first — a blinded marine does not get to aim, whatever the
    situation multiplier says.
    """
    base = float(weapon.spread_deg if can_aim else weapon.spread_snap_deg)
    if order_type == ORDER_MOVE_SHOOT:
        base *= float(CFG.onephase.spread_reverse_mult if reversing
                      else CFG.onephase.spread_move_shoot_mult)
    if overwatch:
        base *= float(CFG.onephase.spread_overwatch_mult)
    return base


# ---------------------------------------------------------------------------
# The compiled plan
# ---------------------------------------------------------------------------
class PlanStep:
    """One scheduled action: an order, its registry row, and its tick window.

    ``end_tick`` is ``INDEFINITE`` for a sustained tail. ``path`` is the
    per-tick position list for a movement step (``path[i]`` is where the unit
    should be at ``start_tick + i``) — the compiler's prediction, which
    execution may fall short of but never exceeds.
    """

    __slots__ = ("order", "action", "start_tick", "end_tick", "path",
                 "fired_ticks", "retired", "blocked", "started")

    def __init__(self, order, action: ActionDef, start_tick: int,
                 end_tick, path=None):
        self.order = order
        self.action = action
        self.start_tick = int(start_tick)
        self.end_tick = end_tick
        self.path = path or []
        self.fired_ticks = 0     # rounds sent downrange inside this step
        self.retired = False     # completed and removed from the order queue
        self.blocked = False     # §14: the path died under it; it halted
        # Its one-shot opening effect has run. Tracked as a FLAG rather than
        # inferred from ``tick == start_tick`` because instantaneous steps
        # (mark, swap, detonate — duration 0) occupy no window at all: several
        # can legitimately fall on the same tick, and each must still open
        # exactly once, in queue order.
        self.started = False

    @property
    def order_type(self) -> int:
        return self.order.order_type

    def contains(self, tick: int) -> bool:
        if tick < self.start_tick:
            return False
        return self.end_tick is INDEFINITE or tick < self.end_tick

    def duration_ticks(self):
        if self.end_tick is INDEFINITE:
            return INDEFINITE
        return self.end_tick - self.start_tick

    def __repr__(self):
        end = "..." if self.end_tick is INDEFINITE else self.end_tick
        return (f"PlanStep({self.action.name}, {self.start_tick}->{end}, "
                f"path={len(self.path)})")


class Plan:
    """A unit's compiled timeline (design §3).

    Deliberately a plain, re-derivable structure: it is recompiled from the
    order queue whenever the queue changes, and never patched in place — so
    what the player sees and what the executor runs cannot diverge.
    """

    __slots__ = ("unit_id", "compiled_at_tick", "steps")

    def __init__(self, unit_id: int, compiled_at_tick: int, steps=()):
        self.unit_id = int(unit_id)
        self.compiled_at_tick = int(compiled_at_tick)
        self.steps = list(steps)

    def step_at(self, tick: int):
        """The step owning ``tick``, or ``None``. Steps are non-overlapping and
        in ascending order by construction, so the first match is THE match."""
        for s in self.steps:
            if not s.retired and s.contains(tick):
                return s
        return None

    def pending(self, tick: int) -> list:
        return [s for s in self.steps
                if not s.retired and (s.end_tick is INDEFINITE
                                      or s.end_tick > tick)]

    def end_tick(self):
        """When the whole plan finishes, or ``INDEFINITE`` if it never does."""
        if not self.steps:
            return None
        last = self.steps[-1]
        return last.end_tick

    # -- the UI's queries (§16) ----------------------------------------
    def arrival_tick(self, step: PlanStep):
        """Absolute tick the unit finishes ``step``."""
        return step.end_tick

    def seconds_into_round(self, tick: int, round_start_tick: int) -> float:
        """The number the teal endpoint label shows: "2.3" = arrives 2.3 s into
        the round (§16)."""
        return (int(tick) - int(round_start_tick)) / float(
            CFG.clock.ticks_per_second)

    def __repr__(self):
        return f"Plan(unit={self.unit_id}, steps={self.steps!r})"


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------
def compile_plan(sim, unit) -> Plan:
    """Turn ``unit.orders`` into a :class:`Plan` anchored at ``sim.tick``.

    The cursor walk, in the design's own order of concerns:

    1. an order may not start before the unit is free (``busy_until_tick``),
       before its own ``start_condition`` allows (Hold's / Detonate's absolute
       tick), before the GCD lapses if the action triggers it, or before that
       action's own cooldown lapses (the weapon swap's 0.75 s);
    2. its length is its path (movement), its registry duration (channeled and
       instantaneous actions), or open-ended (sustained);
    3. it charges the GCD from its START and its own cooldown from its END.

    Sustained steps are closed by whatever follows them; a sustained tail runs
    ``INDEFINITE``, which is how a standing shoot order survives the seam.
    """
    table = sim.actions_table
    tps = int(CFG.clock.ticks_per_second)
    gcd_ticks = int(CFG.onephase.gcd_ticks)

    now = int(sim.tick)
    cursor = max(now, int(getattr(unit, "busy_until_tick", 0)))
    gcd_ready = int(getattr(unit, "gcd_until_tick", 0))
    cd_ready = dict(getattr(unit, "action_cd", None) or {})
    # A live weapon-swap cooldown is the one carried timer with its own row.
    swap_until = int(getattr(unit, "swap_cd_until_tick", 0))
    if swap_until:
        cd_ready["swap_weapon"] = max(cd_ready.get("swap_weapon", 0),
                                      swap_until)

    # The unit's projected position as the plan unfolds — movement steps
    # compile from where the PREVIOUS step leaves it, not from where the unit
    # stands now (the shipped _compute_player_paths does the same).
    px, py = float(unit.x), float(unit.y)

    steps: list[PlanStep] = []
    for order in unit.orders:
        action = _action_for(table, order)
        if action is None:
            continue                     # a legacy order type: not ours to run

        start = cursor
        if action.triggers_gcd:
            start = max(start, gcd_ready)
        cd = cd_ready.get(action.name, 0)
        if cd:
            start = max(start, cd)
        if action.start_condition == "at_time":
            # Hold-until-t and scheduled detonation both name an ABSOLUTE tick
            # (§5, §12). A moment already past does not rewind the cursor.
            at = order.det_tick if order.order_type == ORDER_DETONATE \
                else order.start_tick
            if at is not None:
                start = max(start, int(at))

        path = None
        if order.order_type in ONEPHASE_MOVE_ORDER_TYPES:
            path, (px, py) = _compile_move(sim, unit, order, px, py)
            duration = len(path)
        elif order.order_type == ORDER_HOLD:
            # "Hold (until t): wait at the current position until a chosen
            # time" — the wait IS the step, so its end is that moment.
            until = order.start_tick
            duration = max(0, int(until) - start) if until is not None else 0
        elif action.sustained:
            duration = INDEFINITE
        else:
            duration = int(action.duration_ticks)

        end = INDEFINITE if duration is INDEFINITE else start + duration
        steps.append(PlanStep(order, action, start, end, path))

        if action.triggers_gcd:
            # Charged from the START (see the module docstring).
            gcd_ready = start + gcd_ticks
        if action.cooldown_ticks:
            base = start if end is INDEFINITE else end
            cd_ready[action.name] = base + int(action.cooldown_ticks)
        cursor = max(start, gcd_ready if action.triggers_gcd else start) \
            if end is INDEFINITE else max(end, start)

    _close_sustained_steps(steps)
    del tps                                # (kept above for readability only)
    return Plan(unit.id, now, steps)


def _action_for(table, order):
    """The registry row an order executes through — its explicit
    ``action_name`` if the UI named one, else the reverse lookup. A legacy
    order type (TwoPhaseWEGO's) has no row and returns ``None``."""
    name = getattr(order, "action_name", None)
    if name:
        return table.get(name)
    try:
        return table.for_order_type(order.order_type)
    except KeyError:
        return None


def _close_sustained_steps(steps) -> None:
    """A sustained step ends where the next step begins; a sustained TAIL runs
    indefinitely. Done in a second pass because a step cannot know its
    successor while the cursor is still walking forward."""
    for i, s in enumerate(steps):
        if s.end_tick is not INDEFINITE:
            continue
        if i + 1 < len(steps):
            s.end_tick = max(s.start_tick, steps[i + 1].start_tick)


def _compile_move(sim, unit, order, from_x, from_y):
    """Per-tick positions for one movement order, plus the end position.

    Tile-grid A* (§4: pathfinding stays tile A*) with continuous per-tick
    interpolation, and the §6 aim-relative cadence applied PER TILE — the
    forward/reversing test is re-evaluated at each tile step, because a path
    that curves around a corner can start forward and end backpedaling.
    """
    gmap = sim.gmap
    h, w = gmap.material.shape
    fp = unit.footprint
    tx, ty = int(order.target_fx), int(order.target_fy)

    def blocked(x, y, _g=gmap, _fp=fp):
        return not _g.is_passable_block(y, x, _fp)

    cx, cy = int(from_x), int(from_y)
    if HAS_PATHFINDING:
        segment = astar(cx, cy, tx, ty, blocked, w, h)
        tiles = segment[1:] if segment and len(segment) > 1 else []
    else:                                              # pragma: no cover
        tiles = [(tx, ty)]

    # Where the gun points while moving (§6's two-part gesture): the aim
    # anchor if the order carries one, else the shot target's tile, else no
    # opinion (plain Move — always 100 %).
    aim = _aim_point(sim, order)

    positions = []
    prev_x, prev_y = float(from_x), float(from_y)
    for (tile_x, tile_y) in tiles:
        dx, dy = tile_x - prev_x, tile_y - prev_y
        reversing = False
        if aim is not None:
            reversing = is_reversing(dx, dy, aim[0] - prev_x, aim[1] - prev_y)
        step_ticks = tile_cadence(gmap, unit, tile_y, tile_x,
                                  order.order_type, reversing)
        for st in range(step_ticks):
            frac = (st + 1) / step_ticks
            positions.append((prev_x + (tile_x - prev_x) * frac,
                              prev_y + (tile_y - prev_y) * frac))
        prev_x, prev_y = float(tile_x), float(tile_y)
    return positions, (prev_x, prev_y)


def _aim_point(sim, order):
    """The (x, y) a moving unit keeps its gun on, or ``None``."""
    anchor = getattr(order, "aim_anchor", None)
    if anchor is not None:
        return (float(anchor[0]), float(anchor[1]))
    tid = getattr(order, "target_unit_id", None)
    if tid is not None:
        target = sim.get_unit(tid)
        if target is not None:
            return (float(target.center_tile_x()), float(target.center_tile_y()))
    if order.order_type == ORDER_MOVE_SHOOT:
        return (float(order.target_fx), float(order.target_fy))
    return None


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------
def drive_units(sim) -> None:
    """One tick of OnePhaseWEGO unit simulation (the ruleset's slot 3+4).

    Two passes over the roster, in the shipped order of concerns — all
    movement, then all shooting — so that a shot resolves against the
    positions every unit reached this tick, exactly as the legacy
    ``_update_player_movement`` / ``process_shooting`` pair does.
    """
    for u in sim.units:
        if u.team == 0 and u.alive:
            _advance_movement(sim, u)
    for u in sim.units:
        if u.team == 0 and u.alive:
            _resolve_action(sim, u)


def _plan_for(sim, unit):
    plan = getattr(unit, "plan", None)
    if plan is None:
        plan = compile_plan(sim, unit)
        unit.plan = plan
    return plan


def _advance_movement(sim, unit) -> None:
    """Walk one tick of the unit's compiled path.

    Three ways to cover no ground, all of which cost the tick rather than
    deferring it (invariant 1 — the schedule is authoritative for time):

    - the unit's composed ``can_move`` is suppressed (knocked down, …);
    - §14 plan invalidation: the next position is no longer passable (a wall
      dropped, a door shut, rubble landed). v1 behavior is HALT IN PLACE —
      the move step is marked blocked and covers nothing further, the
      remaining non-move orders continue on their scheduled ticks, and no
      auto-repath happens. Erik has explicitly NOT settled this (§14/§20.1);
      it is the one deliberately open semantic in the design;
    - the step has run past the end of its path (arrived early).
    """
    step = _plan_for(sim, unit).step_at(sim.tick)
    if step is None or not step.path or step.blocked:
        return
    if not composed_flags(unit).can_move:
        return
    idx = sim.tick - step.start_tick
    if not (0 <= idx < len(step.path)):
        return
    nx, ny = step.path[idx]
    if not sim.gmap.is_passable_block(int(ny), int(nx), unit.footprint):
        step.blocked = True                       # §14: halt in place
        return
    unit.face_towards(nx, ny)
    unit.x, unit.y = nx, ny


def _resolve_action(sim, unit) -> None:
    """Run this tick's non-movement effects, then retire finished steps.

    Ordering matters three ways:

    - **open before retire**, so an instantaneous step (duration 0) still
      happens on the tick it is due rather than being swept away unrun;
    - **open every due step, in queue order**, so a chain of instantaneous
      steps (mark, then swap, then detonate) all fire on the same tick — which
      is exactly what a plan that schedules them together promised;
    - **retire last**, so a step's window is only closed once its effects for
      this tick are done.
    """
    plan = _plan_for(sim, unit)
    if not composed_flags(unit).can_act:
        # Suppression delays, never cancels (mechanics/06 §4) — and it does
        # not shift the schedule either. Nothing opens and nothing retires.
        return

    for step in plan.steps:
        if step.retired or step.started:
            continue
        if sim.tick < step.start_tick:
            break                       # steps are in ascending start order
        step.started = True
        _begin_step(sim, unit, step)

    step = plan.step_at(sim.tick)
    if step is not None and step.order_type in (ORDER_SHOOT, ORDER_MOVE_SHOOT):
        _shoot_tick(sim, unit, step)

    _retire_finished(sim, unit, plan)


def _begin_step(sim, unit, step) -> None:
    """Side effects that happen ONCE, on the tick a step opens."""
    action = step.action
    if action.triggers_gcd:
        unit.gcd_until_tick = step.start_tick + int(CFG.onephase.gcd_ticks)
    if action.cooldown_ticks:
        base = (step.start_tick if step.end_tick is INDEFINITE
                else step.end_tick)
        cds = getattr(unit, "action_cd", None)
        if cds is None:
            cds = unit.action_cd = {}
        cds[action.name] = base + int(action.cooldown_ticks)
    if step.end_tick is not INDEFINITE:
        unit.busy_until_tick = max(int(getattr(unit, "busy_until_tick", 0)),
                                   step.end_tick)

    ot = step.order_type
    if ot == ORDER_SWAP:
        swap_weapon(sim, unit, step)
    elif ot == ORDER_OVERWATCH:
        set_overwatch(sim, unit, step)
    elif ot == ORDER_MARK:
        mark_target(sim, unit, step)


def _retire_finished(sim, unit, plan) -> None:
    """Drop steps whose window has closed, and the orders behind them.

    Retiring the ORDER (not just the step) is what makes a recompile safe: a
    plan is always rebuilt from the pending queue, so a completed action can
    never be re-scheduled and re-run by the next recompile.
    """
    for step in plan.steps:
        if step.retired or not step.started:
            continue
        if step.end_tick is INDEFINITE or sim.tick < step.end_tick:
            continue
        step.retired = True
        if step.order in unit.orders:
            unit.orders.remove(step.order)


# ---------------------------------------------------------------------------
# Action resolutions
# ---------------------------------------------------------------------------
def _shoot_tick(sim, unit, step) -> None:
    """One tick of a Shoot / Move & Shoot step (§5, §7).

    Aim TRACKS the target during execution (§5) — the bearing is recomputed
    from the target's live centre every tick, which is what makes shooting a
    moving zombie work without any lead-prediction machinery. Cadence, magazine
    and archetype dispatch are the shipped combat path verbatim; the only
    OnePhaseWEGO-specific input is the situational spread (§7).
    """
    target = _shot_target(sim, step)
    if target is None:
        return
    weapon = _active_weapon(sim, unit)
    if weapon is None:
        return
    if sim.tick - unit.last_fire_tick < weapon.rof_interval_ticks:
        return
    if not mag_gate(unit, weapon, sim.tick):
        return

    ux, uy = unit.center_tile_x(), unit.center_tile_y()
    tx, ty = target.center_tile_x(), target.center_tile_y()
    dist = math.sqrt((ux - tx) ** 2 + (uy - ty) ** 2)
    if dist > weapon.range_tiles:
        return
    # Walls block the ORDER; cover does not (§7 — cover eats the rays
    # physically, in the march, so shooting at someone behind a crate is a
    # legal order that mostly feeds the crate).
    if not sim.gmap.has_los(uy, ux, ty, tx):
        return

    reversing = _step_is_reversing(sim, unit, step)
    spread = spread_deg_for(weapon, step.order_type, reversing=reversing,
                            can_aim=composed_flags(unit).can_aim)
    _dispatch_trigger(sim.gmap, sim.units, unit, ux, uy, tx, ty, sim.tick,
                      sim.shots, sim.real_time, sim.rng, sim.tick_events,
                      sim.bullets, weapon, spread, sim.edit_queue)
    mag_spend(unit, weapon, sim.tick)
    unit.last_fire_tick = sim.tick
    step.fired_ticks += 1


def _shot_target(sim, step):
    tid = getattr(step.order, "target_unit_id", None)
    if tid is None:
        return None
    target = sim.get_unit(tid)
    if target is None or not target.alive or target.team == 0:
        return None
    return target


def _step_is_reversing(sim, unit, step) -> bool:
    """Recompute the §6 forward/reversing test from the unit's CURRENT travel
    direction — the compiled path's tangent at this tick — so the spread the
    player feels matches the movement they can see."""
    if step.order_type != ORDER_MOVE_SHOOT or not step.path:
        return False
    idx = sim.tick - step.start_tick
    if not (1 <= idx < len(step.path)):
        return False
    x0, y0 = step.path[idx - 1]
    x1, y1 = step.path[idx]
    aim = _aim_point(sim, step.order)
    if aim is None:
        return False
    return is_reversing(x1 - x0, y1 - y0, aim[0] - x1, aim[1] - y1)


def _active_weapon(sim, unit):
    wid = getattr(unit, "weapon_id", "")
    if not wid:
        return None
    return sim.weapons_tables.weapons.by_name.get(wid)


def swap_weapon(sim, unit, step) -> None:
    """Primary <-> secondary (§15): free except the 0.75 s swap cooldown.

    A swapped weapon arrives with a fresh magazine and no half-finished burst —
    the same coupled state reset ``Simulation.debug_cycle_weapon`` performs,
    for the same reason (cadence/mag/burst state belongs to the weapon that was
    firing, not to the unit).
    """
    loadout = getattr(unit, "loadout", None)
    if not loadout or len(loadout) < 2:
        return
    unit.active_slot = 1 - int(getattr(unit, "active_slot", 0))
    unit.weapon_id = loadout[unit.active_slot]
    unit.current_mag = None
    unit.reload_done_tick = -1
    unit.spray_ticks_left = 0
    unit.spray_order = None
    unit.spray_target = None
    unit.swap_cd_until_tick = step.start_tick + int(
        CFG.onephase.weapon_swap_ticks)


def set_overwatch(sim, unit, step) -> None:
    """Establish the persistent overwatch state (§9). The engagement logic
    itself is P6; this is the state the plan writes."""
    order = step.order
    ux, uy = unit.center_tile_x(), unit.center_tile_y()
    dx = float(order.target_fx) - ux
    dy = float(order.target_fy) - uy
    if dx == 0.0 and dy == 0.0:
        facing = float(unit.facing)
    else:
        facing = unit_fixed.atan2_rad(-dy, dx)
    half = order.cone_half_deg
    if half is None:
        half = float(CFG.onephase.overwatch_cone_half_deg)
    half = max(float(CFG.onephase.overwatch_cone_min_half_deg),
               min(float(CFG.onephase.overwatch_cone_max_half_deg),
                   float(half)))
    unit.overwatch_facing = facing
    unit.overwatch_half_deg = half
    unit.facing = facing


def mark_target(sim, unit, step) -> None:
    """Mark an enemy for the whole team (§11). The per-team table lives on the
    sim because it is a TEAM fact, not a unit's; every targeting function reads
    it. Marks persist until the target dies or is unmarked."""
    tid = getattr(step.order, "target_unit_id", None)
    if tid is None:
        return
    marks = getattr(sim, "marks", None)
    if marks is None:
        marks = sim.marks = {}
    marks.setdefault(int(unit.team), set()).add(int(tid))


__all__ = [
    "INDEFINITE", "Plan", "PlanStep", "compile_plan", "drive_units",
    "is_reversing", "mark_target", "set_overwatch", "speed_pct",
    "spread_deg_for", "swap_weapon", "tile_cadence",
]
