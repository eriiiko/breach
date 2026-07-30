"""VISION v1 — the first-class visibility model (onephase_wego design §8).

Before this, LOS was only half-real: raycasts existed for lighting and
auto-fire, but nothing could be *asked* what a unit could see. Overwatch
acquisition (§9), Ambush (§10), marking (§11), flanking and enemy visibility
all lean on that question, so vision becomes a system.

The model, in the design's words:

- each unit has a **facing vision cone** (half-angle dial) with **UNLIMITED
  range** — walls are the only limit (Erik's ruling: a max vision range is
  unrealistic and disliked). Cost is bounded by map geometry, not by a range
  constant, because a LOS test is a ray-march that terminates at the first
  wall;
- plus a short **360 deg awareness radius** (hearing / peripheral);
- **team vision is the union** of its members' cones;
- **fog of war is visibility GATING only** — an enemy your team cannot see
  simply is not drawn. No fog layer, no last-known-position ghosts in v1
  (unseen = gone).

Two predicates fall out of the same cones: **discovered** (you have entered an
enemy's vision) and **flanked** (you were attacked from outside your own arc).
§9 generalizes the second into defense: facing determines vulnerability, and
overwatch carries a harsher rear penalty for the side-blindness it buys.

Determinism (IRON RULE). Every angular test here is a **dot product against a
cosine threshold** taken once through the deterministic integer trig kit
(``unit_fixed.cos_rad`` / ``sin_rad``) — there is no ``atan2`` per pair, no
libm on the sim path, and no accumulated angle arithmetic to drift. Occlusion
is the shipped integer Bresenham march (``GameMap.has_los``). Results are
returned as sorted tuples, and every consumer that makes a CHOICE iterates
``sim.units`` in id order, so no set-iteration order can ever reach gameplay.

Honesty note carried from the design: in deterministic lockstep multiplayer
the full state lives on both machines, so fog is honor-system client-side.
This is designed as if enforced; enforcement is a later server problem.
"""
from __future__ import annotations

import math

from config import CFG
from simulation import unit_fixed


# ---------------------------------------------------------------------------
# Angular primitives — dot products, never atan2
# ---------------------------------------------------------------------------
def cos_threshold(half_angle_deg: float) -> float:
    """``cos(half_angle)`` through the integer trig kit.

    A point lies inside a cone iff the cosine of its bearing offset is at or
    above this — one multiply-add per test, and no transcendental anywhere
    near the sim path.
    """
    return unit_fixed.cos_rad(math.radians(float(half_angle_deg)))


def facing_vector(facing_rad: float):
    """Unit vector of a facing, in WORLD coordinates (y down).

    ``Unit.facing`` is math-convention (0 = East, CCW positive, y up); the grid
    is y-down. The single negation here is that conversion, kept in one place
    so no caller has to remember it.
    """
    return (unit_fixed.cos_rad(facing_rad), -unit_fixed.sin_rad(facing_rad))


def within_cone(facing_rad: float, half_deg: float, dx: float, dy: float,
                threshold=None) -> bool:
    """Is the world-space offset ``(dx, dy)`` inside the cone?

    A zero offset (the observer's own position) counts as inside — you can
    always see where you are standing. ``threshold`` may be passed in when a
    caller is testing many points against ONE cone, so the cosine is taken
    once rather than per point.
    """
    d2 = dx * dx + dy * dy
    if d2 <= 0.0:
        return True
    fx, fy = facing_vector(facing_rad)
    dot = (dx * fx + dy * fy) / math.sqrt(d2)
    thr = cos_threshold(half_deg) if threshold is None else threshold
    return dot >= thr


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------
def ray_clear(sim, y0, x0, y1, x1) -> bool:
    """Is the segment between two tile points unobstructed?

    THE single occlusion query for vision and for the exposed-profile metric.
    Walls block (the shipped integer Bresenham march). Cover joins here in P5:
    a cover entity that declares ``blocks_los`` occludes, while an ordinary
    crate does NOT — you can see over a crate, you just cannot shoot through
    it, which is why §7's physical cover lives in the bullet march and not in
    this function.
    """
    if not sim.gmap.has_los(int(y0), int(x0), int(y1), int(x1)):
        return False
    for cov in getattr(sim, "cover", ()) or ():
        if cov.blocks_los and cov.alive and cov.blocks_segment(x0, y0, x1, y1):
            return False
    return True


# ---------------------------------------------------------------------------
# The per-tick vision state
# ---------------------------------------------------------------------------
class VisionState:
    """One tick's answer to "who can see whom".

    Rebuilt at most once per tick and cached on the sim: overwatch, ambush,
    idle return-fire and the renderer's fog gate all ask the same question in
    the same tick, and they must all get the same answer.
    """

    __slots__ = ("tick", "seen_by", "visible_by_team")

    def __init__(self, tick: int):
        self.tick = int(tick)
        #: ``{target_unit_id: (observer_id, ...)}`` — sorted, deterministic.
        self.seen_by: dict[int, tuple] = {}
        #: ``{team: (enemy_unit_id, ...)}`` — the union of the team's cones.
        self.visible_by_team: dict[int, tuple] = {}

    def is_visible_to_team(self, team: int, unit_id: int) -> bool:
        return int(unit_id) in self.visible_by_team.get(int(team), ())

    def observers_of(self, unit_id: int) -> tuple:
        return self.seen_by.get(int(unit_id), ())


def can_see(sim, observer, target) -> bool:
    """Can ``observer`` see ``target`` right now? (§8)

    Two independent ways, both requiring an unobstructed line — walls are the
    only limit, and they limit BOTH:

    1. the facing cone, at unlimited range;
    2. the short 360 deg awareness radius (hearing / peripheral) — which is
       why a marine is not blind to something standing right behind him.

    A dead observer sees nothing; a unit always "sees" itself.
    """
    if observer is target:
        return True
    if not observer.alive or not target.alive:
        return False

    ox, oy = observer.center_tile_x(), observer.center_tile_y()
    tx, ty = target.center_tile_x(), target.center_tile_y()
    dx, dy = tx - ox, ty - oy

    radius = float(CFG.onephase.awareness_radius_tiles)
    in_awareness = (dx * dx + dy * dy) <= radius * radius
    if not in_awareness:
        if not within_cone(observer.facing,
                           CFG.onephase.vision_cone_half_deg, dx, dy):
            return False
    return ray_clear(sim, oy, ox, ty, tx)


def compute(sim) -> VisionState:
    """Build this tick's :class:`VisionState`.

    Iterates ``sim.units`` in list (id) order for both loops, so the tuples it
    stores are in a fixed order regardless of how any caller later iterates
    them. O(units^2) LOS tests — with squad-scale rosters that is nothing, and
    each test is a march bounded by map geometry (§8's cost note).
    """
    state = VisionState(sim.tick)
    by_team: dict[int, list] = {}
    for target in sim.units:
        if not target.alive:
            continue
        observers = []
        for observer in sim.units:
            if observer.team == target.team or not observer.alive:
                continue
            if can_see(sim, observer, target):
                observers.append(int(observer.id))
                by_team.setdefault(int(observer.team), []).append(
                    int(target.id))
        if observers:
            state.seen_by[int(target.id)] = tuple(sorted(observers))
    state.visible_by_team = {t: tuple(sorted(set(ids)))
                             for t, ids in by_team.items()}
    return state


def state_for(sim) -> VisionState:
    """The cached :class:`VisionState` for the current tick, computing it on
    first ask. Cheap to call repeatedly, which is the point — every consumer
    in a tick must see one consistent answer."""
    cached = getattr(sim, "_vision_cache", None)
    if cached is not None and cached.tick == sim.tick:
        return cached
    cached = compute(sim)
    sim._vision_cache = cached
    return cached


def visible_enemy_ids(sim, team: int) -> tuple:
    """The enemies ``team`` can currently see — TEAM VISION, the union of its
    members' cones (§8). This is the renderer's fog gate: an id absent here is
    simply not drawn."""
    return state_for(sim).visible_by_team.get(int(team), ())


# ---------------------------------------------------------------------------
# Predicates off the same cones (§8)
# ---------------------------------------------------------------------------
def is_discovered(sim, unit) -> bool:
    """Has ``unit`` entered an enemy's vision?"""
    return bool(state_for(sim).observers_of(unit.id))


def is_flanked(unit, from_x: float, from_y: float) -> bool:
    """Was ``unit`` attacked from outside its own facing arc? (§8/§9)

    The arc is the defensive front — wider than the vision cone by default,
    because "I am covering that way" is coarser than "I can see that".
    """
    dx = float(from_x) - unit.center_tile_x()
    dy = float(from_y) - unit.center_tile_y()
    return not within_cone(unit.facing, CFG.onephase.facing_arc_half_deg,
                           dx, dy)


def defense_multiplier(unit, from_x: float, from_y: float) -> float:
    """Damage multiplier for an attack arriving from ``(from_x, from_y)``.

    §9: "attacked from outside your facing arc = increased vulnerability
    (tunable modifier), for all units, with overwatch simply having a wider
    rear penalty. Facing determines defense." A unit standing on overwatch has
    bought target control with side blindness, and this is that cost priced in.
    """
    if not is_flanked(unit, from_x, from_y):
        return 1.0
    if getattr(unit, "overwatch_facing", None) is not None:
        return float(CFG.onephase.overwatch_rear_damage_mult)
    return float(CFG.onephase.flank_damage_mult)


# ---------------------------------------------------------------------------
# "Easiest to hit" — the largest exposed profile (§7/§9)
# ---------------------------------------------------------------------------
def exposed_profile(sim, shooter, target) -> float:
    """Fraction of ``target``'s silhouette that ``shooter`` has a clear line to.

    §7: physical cover makes "easiest to hit" *computable* — "the target with
    the largest exposed profile" — instead of a cover-bonus stat. The
    silhouette is sampled on an N x N grid across the target's footprint
    (``vision_profile_samples``), and each sample is one occlusion query, so a
    marine hugging a crate is protected exactly as much as the geometry says.

    Returns 0.0 (fully covered) .. 1.0 (wide open). Deterministic: a fixed
    sample lattice in a fixed order, integer marches, no RNG.
    """
    n = max(1, int(CFG.onephase.vision_profile_samples))
    fp = int(target.footprint)
    sx, sy = shooter.center_tile_x(), shooter.center_tile_y()
    clear = 0
    for iy in range(n):
        for ix in range(n):
            # Sample cell centres across the footprint: (i + 0.5) / n of the
            # way across, so the lattice is symmetric and never lands exactly
            # on a footprint edge.
            px = target.x + (ix + 0.5) * fp / n
            py = target.y + (iy + 0.5) * fp / n
            if ray_clear(sim, sy, sx, py, px):
                clear += 1
    return clear / float(n * n)


__all__ = [
    "VisionState", "can_see", "compute", "cos_threshold",
    "defense_multiplier", "exposed_profile", "facing_vector", "is_discovered",
    "is_flanked", "ray_clear", "state_for", "visible_enemy_ids",
    "within_cone",
]
