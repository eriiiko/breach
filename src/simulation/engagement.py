"""Standing engagements — Overwatch, Ambush, Idle stance (design §9/§10/§13).

The three ways a unit shoots WITHOUT a shoot order being the thing it is doing
this instant. They share one shape: a target-selection rule, a release
condition, and then the ordinary weapon path.

**Overwatch (§9)** — a persistent STATE, not an order: aim a cone in a chosen
direction and engage whatever enters it, continuously (normal weapon behaviour
while targets are in the cone, not one reaction shot), across round boundaries,
until replaced. The cone's width is player-set and its PRIMARY purpose is
target control — targets outside it are ignored, so narrowing is indirect
target selection. The cost is side blindness, priced by
:func:`simulation.vision.defense_multiplier`.

**Ambush (§10)** — the synced attack, and the whole reason cross-round
choreography works. Any number of units queue ``Ambush(target)`` on the same
enemy. Readiness is *just* "the unit has reached the Ambush order in its
queue" — no LOS condition, no other semantics (Erik's simplification), which is
what makes timing composable: the breacher's queue [move -> detonate -> Ambush]
means everyone else, already waiting on the barrier, fires the moment the
charge blows.

**Idle stance (§13)** — a unit with nothing queued returns fire at its
attackers, preferring marked ones, and does NOT free-fire at everything it
sees. "You're meant to give orders to your whole (small) squad; return fire is
a floor, not an AI."

Determinism: candidate scans iterate ``sim.units`` in id order; every ranking
is a total order on integer/float keys with the unit id as the final
tie-break, so no set or dict iteration order can ever decide who gets shot.
No RNG is drawn here — the weapon path draws its own cone, as always.
"""
from __future__ import annotations

import math

from config import CFG
from simulation import vision
from simulation.orders import ORDER_AMBUSH


# ---------------------------------------------------------------------------
# Shared target ranking
# ---------------------------------------------------------------------------
def _marked(sim, team) -> set:
    return sim.marks.get(int(team), set())


def _cone_alignment(unit, target) -> float:
    """How close a target sits to the cone's centre line, as a COSINE.

    Returns 1.0 dead ahead, falling to -1.0 directly behind. The design speaks
    of "closest to cone center" as an angle, but ranking by angle and ranking
    by ``-cos(angle)`` are the same total order on [0, pi] — so the cosine IS
    the ranking key, and no ``acos`` is needed. That matters: an inverse
    trigonometric call is libm, banned on the sim path by the number-ingress
    rule, and it would buy nothing here but a ULP of drift.
    """
    facing = unit.overwatch_facing
    if facing is None:
        facing = unit.facing
    fx, fy = vision.facing_vector(facing)
    dx = target.center_tile_x() - unit.center_tile_x()
    dy = target.center_tile_y() - unit.center_tile_y()
    d = math.sqrt(dx * dx + dy * dy)
    if d <= 0.0:
        return 1.0
    return (dx * fx + dy * fy) / d


def rank_targets(sim, unit, candidates) -> list:
    """§9's target priority, as a total order:

    1. **marked targets** (§11) — the whole point of marking;
    2. **easiest to hit** = the largest exposed profile, which physical cover
       (§7) makes computable;
    3. **closest to cone centre**;
    4. unit id, so the order is total and reproducible.

    Returned best-first.
    """
    marks = _marked(sim, unit.team)
    scored = []
    for t in candidates:
        scored.append((
            0 if int(t.id) in marks else 1,          # marked first
            -vision.exposed_profile(sim, unit, t),   # most exposed first
            -_cone_alignment(unit, t),               # nearest the cone centre
            int(t.id),
        ))
        scored[-1] = (scored[-1], t)
    scored.sort(key=lambda pair: pair[0])
    return [t for _key, t in scored]


def _enemies_visible_to(sim, unit) -> list:
    return [u for u in sim.units
            if u.team != unit.team and u.alive and vision.can_see(sim, unit, u)]


# ---------------------------------------------------------------------------
# Overwatch (§9)
# ---------------------------------------------------------------------------
def on_overwatch(unit) -> bool:
    return getattr(unit, "overwatch_facing", None) is not None


def overwatch_candidates(sim, unit) -> list:
    """Enemies inside the overwatch cone that the unit can actually see.

    The cone is the TARGET CONTROL device (§9): anything outside it is ignored,
    which is how narrowing the cone selects targets without a target list.
    """
    half = unit.overwatch_half_deg
    if half is None:
        half = float(CFG.onephase.overwatch_cone_half_deg)
    thr = vision.cos_threshold(half)
    out = []
    for t in _enemies_visible_to(sim, unit):
        dx = t.center_tile_x() - unit.center_tile_x()
        dy = t.center_tile_y() - unit.center_tile_y()
        if vision.within_cone(unit.overwatch_facing, half, dx, dy,
                              threshold=thr):
            out.append(t)
    return out


def overwatch_target(sim, unit):
    ranked = rank_targets(sim, unit, overwatch_candidates(sim, unit))
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Ambush (§10)
# ---------------------------------------------------------------------------
def ambush_groups(sim) -> dict:
    """``{target_unit_id: [unit, ...]}`` — every unit with a PENDING Ambush
    order on that target, in unit-id order.

    Membership is having the order queued; readiness is having reached it
    (:func:`is_ready`). The distinction is the entire mechanism: a breacher
    still walking to the door is a member holding the group, not a shirker.
    """
    groups: dict = {}
    for u in sim.units:
        if u.team != 0 or not u.alive:
            continue
        plan = getattr(u, "plan", None)
        if plan is None:
            continue
        for step in plan.steps:
            if step.retired or step.order_type != ORDER_AMBUSH:
                continue
            tid = getattr(step.order, "target_unit_id", None)
            if tid is not None:
                groups.setdefault(int(tid), []).append(u)
            break
    return groups


def is_ready(sim, unit) -> bool:
    """Has ``unit`` reached its Ambush order? (§10 — the whole readiness rule.)

    No LOS condition, no other semantics. Erik's simplification, and what makes
    the barrier composable with Hold-until-t and scheduled detonations.
    """
    plan = getattr(unit, "plan", None)
    if plan is None:
        return False
    step = plan.step_at(sim.tick)
    return step is not None and step.order_type == ORDER_AMBUSH


def update_ambush(sim) -> None:
    """Evaluate every ambush group's release condition, once per tick.

    - **Fire condition:** the instant ALL living members are ready, the group
      releases and everyone fires together.
    - **Sprung:** if ANY member is fired upon, all *ready* members open fire
      immediately; not-yet-ready members keep walking their queues and join on
      arrival (they see an already-released group).
    - **Dead members drop out** of the count, so a casualty cannot deadlock a
      group into holding forever.
    """
    released = sim.ambush_released
    groups = ambush_groups(sim)
    for tid in sorted(groups):
        members = groups[tid]
        if tid in released:
            continue
        if all(is_ready(sim, u) for u in members):
            released[tid] = sim.tick
            continue
        # Sprung: somebody in the group is being shot at.
        if any(u.recent_attackers for u in members):
            released[tid] = sim.tick
    # Forget groups that no longer exist, so a later ambush on the same target
    # starts from a clean barrier rather than inheriting an old release.
    for tid in list(released):
        if tid not in groups:
            del released[tid]


def ambush_may_fire(sim, unit, step) -> bool:
    """Is this unit's ambush step released — and past its stagger offset?

    ``ambush_stagger_ticks`` (§20 item 6) is 0 by default: a perfectly
    simultaneous volley. Raising it walks the group's fire out over a few
    ticks, in unit-id order, for a ragged-volley feel.
    """
    tid = getattr(step.order, "target_unit_id", None)
    if tid is None:
        return False
    release = sim.ambush_released.get(int(tid))
    if release is None:
        return False
    stagger = int(CFG.onephase.ambush_stagger_ticks)
    if stagger <= 0:
        return True
    members = ambush_groups(sim).get(int(tid), [])
    order = [int(u.id) for u in members]
    slot = order.index(int(unit.id)) if int(unit.id) in order else 0
    return sim.tick >= release + slot * stagger


def drop_unready_ambushes(sim) -> None:
    """The §10 timeout backstop: at the round boundary, a group that never
    became ready reverts to idle stance — no infinite holds.

    Only UNRELEASED groups are dropped; a released ambush has already become
    ordinary shooting and is left alone.
    """
    groups = ambush_groups(sim)
    for tid in sorted(groups):
        if tid in sim.ambush_released:
            continue
        for u in groups[tid]:
            plan = getattr(u, "plan", None)
            if plan is None:
                continue
            for step in plan.steps:
                if step.retired or step.order_type != ORDER_AMBUSH:
                    continue
                step.retired = True
                if step.order in u.orders:
                    u.orders.remove(step.order)
                break


# ---------------------------------------------------------------------------
# Idle stance (§13)
# ---------------------------------------------------------------------------
def idle_target(sim, unit):
    """Whom an order-less unit returns fire at (§13).

    Candidates are its ATTACKERS only — never everything it can see; that
    restraint is the design's point ("return fire is a floor, not an AI").
    Among them the §9 ranking applies, so a marked attacker is preferred.
    """
    if not CFG.onephase.idle_return_fire:
        return None
    attackers = getattr(unit, "recent_attackers", None)
    if not attackers:
        return None
    candidates = []
    for u in sim.units:                      # id order, not dict order
        if int(u.id) not in attackers or not u.alive or u.team == unit.team:
            continue
        if vision.can_see(sim, unit, u):
            candidates.append(u)
    ranked = rank_targets(sim, unit, candidates)
    return ranked[0] if ranked else None


__all__ = [
    "ambush_groups", "ambush_may_fire", "drop_unready_ambushes", "idle_target",
    "is_ready", "on_overwatch", "overwatch_candidates", "overwatch_target",
    "rank_targets", "update_ambush",
]
