"""UI view-models — pure, headless, testable (onephase_wego design §16).

Every decision the OnePhaseWEGO interface makes lives here as data. No raylib,
no window, no sim mutation: given a :class:`~simulation.simulation.Simulation`
and a little UI state, each function returns plain structures that
:mod:`ui.draw` renders without thinking.

The design's UI section is short but specific, and this module is its
one-to-one implementation:

- **Teal path viz** — ordering a move draws the path line; the endpoint
  footprint is highlighted and labelled with the ARRIVAL TIME ("2.3" = arrives
  2.3 s into the round). Shift-click waypoint strings show a footprint marker
  at each clicked intermediate point.
- **Shoot hologram** — a shoot order from a future position shows a teal ghost
  of the marine at the firing tile, indicating its target.
- **The timeline is the concept.** v1 ships the timestamps; the scrubbable
  preview ("drag a time slider, see every marine's ghost at time t") is the
  natural extension, and determinism makes that preview EXACT — it is a dry
  run of your own plan. :func:`position_at` is that primitive, shipped now
  because the holograms already needed it; the slider is then a few lines of
  draw code whenever Erik wants it.
- **Hotbar** — renders the ACTION REGISTRY (§5); dragging an item from the
  inventory to a slot binds slot -> registry row.
- **Planning clock** — submit-within-N, simultaneous reveal; 0 = untimed.
- **Fog of war** — visibility gating only: an enemy the team cannot see is not
  in the drawable list at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import CFG
from simulation import vision
from simulation.orders import ONEPHASE_MOVE_ORDER_TYPES
from simulation.timeline import INDEFINITE


# ---------------------------------------------------------------------------
# Hotbar (§16) — the registry, rendered
# ---------------------------------------------------------------------------
#: The default slot -> action binding for the shipped keymap (Erik's ruling 1:
#: numbers 1..0 are hotbar slots; right-click is always Move, so slot 1 is a
#: convenience rather than the only way to move). Slots 9 and 0 are left free
#: for belt items, which is what makes "drag an item onto the hotbar" mean
#: something out of the box.
DEFAULT_HOTBAR = ("move", "shoot", "move_shoot", "overwatch", "ambush",
                  "hold", "use_hand_grenade", "plant_charge", None, None)


@dataclass
class HotbarSlot:
    """One rendered hotbar cell."""
    index: int                  # 0-based; the key shown is (index + 1) % 10
    action_name: str = ""
    label: str = ""
    icon: str = ""
    enabled: bool = True
    #: Seconds until this action becomes available again (cooldown or GCD).
    cooldown_remaining: float = 0.0
    #: Why it is greyed out, for the tooltip. "" when enabled.
    reason: str = ""
    #: Item stock for item-backed rows; None for standing verbs.
    count: int = None

    @property
    def key_label(self) -> str:
        return str((self.index + 1) % 10)

    @property
    def bound(self) -> bool:
        return bool(self.action_name)


def default_bindings(unit=None) -> list:
    """A fresh binding list. A unit's BELT fills the free tail slots (§15 —
    belt slots ARE hotbar slots), so a marine's grenades and charges are on
    the bar without anybody dragging anything."""
    slots = list(DEFAULT_HOTBAR)
    if unit is not None:
        free = [i for i, s in enumerate(slots) if s is None]
        for item, i in zip(getattr(unit, "belt", ()) or (), free):
            slots[i] = f"use_{item}"
    return slots


def bind_slot(bindings, index: int, action_name) -> list:
    """Bind a hotbar slot to a registry row — the drop half of "drag an item
    from the inventory to the hotbar" (§16). Returns a NEW list; the caller
    owns its own UI state, and nothing here mutates in place."""
    out = list(bindings)
    if 0 <= index < len(out):
        out[index] = action_name
    return out


def hotbar(sim, unit, bindings=None) -> list:
    """Build the hotbar's cells for ``unit``.

    Greying-out is computed from the same timers the compiler reads, so what
    the bar shows and what an order would actually do cannot disagree: an
    action is unavailable while its own cooldown runs, while the GCD runs (if
    it triggers the GCD), or when its item is spent.
    """
    table = sim.actions_table
    tps = float(CFG.clock.ticks_per_second)
    slots = list(bindings) if bindings is not None else default_bindings(unit)
    now = int(sim.tick)
    gcd_until = int(getattr(unit, "gcd_until_tick", 0)) if unit else 0
    action_cd = (getattr(unit, "action_cd", None) or {}) if unit else {}
    swap_until = int(getattr(unit, "swap_cd_until_tick", 0)) if unit else 0

    out = []
    for i, name in enumerate(slots):
        if not name or name not in table.by_name:
            out.append(HotbarSlot(index=i))
            continue
        action = table.get(name)
        count = None
        reason = ""
        remaining = 0
        if action.item:
            count = _item_count(sim, unit, action.item)
            if count is not None and count <= 0:
                reason = "none left"
        own_cd = int(action_cd.get(name, 0))
        if name == "swap_weapon":
            own_cd = max(own_cd, swap_until)
        if own_cd > now:
            remaining = max(remaining, own_cd - now)
            reason = reason or "cooling down"
        if action.triggers_gcd and gcd_until > now:
            remaining = max(remaining, gcd_until - now)
            reason = reason or "global cooldown"
        out.append(HotbarSlot(
            index=i, action_name=name, label=action.label, icon=action.icon,
            enabled=not reason, cooldown_remaining=remaining / tps,
            reason=reason, count=count))
    return out


def _item_count(sim, unit, item):
    attr = sim._ITEM_COUNTERS.get(item) if unit is not None else None
    return None if attr is None else int(getattr(unit, attr, 0))


# ---------------------------------------------------------------------------
# The timeline made visible (§16)
# ---------------------------------------------------------------------------
@dataclass
class PathViz:
    """The teal path line for one movement step."""
    points: list = field(default_factory=list)   # per-tick (x, y), tile coords
    endpoint: tuple = (0.0, 0.0)
    footprint: int = 3
    arrival_seconds: float = 0.0
    blocked: bool = False


@dataclass
class WaypointMarker:
    """A footprint marker at a shift-clicked intermediate point (§16)."""
    x: float
    y: float
    footprint: int = 3
    arrival_seconds: float = 0.0


@dataclass
class Hologram:
    """A teal ghost of the marine at a future firing position (§16)."""
    x: float
    y: float
    footprint: int = 3
    at_seconds: float = 0.0
    action_name: str = ""
    target: tuple = None        # (x, y) tile coords, or None


@dataclass
class TargetMarker:
    """A teal marker on an ENEMY's footprint (Erik, first play session).

    The design's §16 covers where a marine will BE and when; it never says
    what an order is aimed AT, so an ordered shot was invisible until the
    round ran. This closes that: every enemy the selected marine's plan
    points at gets its footprint marked, labelled with the action and the
    moment it happens, and the enemy under the cursor is marked too while a
    unit-targeting action is armed — so you can see what you are about to
    pick before you commit to it.
    """
    unit_id: int
    x: float
    y: float
    footprint: int = 3
    action_name: str = ""
    at_seconds: float = 0.0
    hovered: bool = False       # under the cursor right now, not yet ordered


@dataclass
class PlanOverlay:
    paths: list = field(default_factory=list)
    waypoints: list = field(default_factory=list)
    holograms: list = field(default_factory=list)
    targets: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.paths or self.waypoints or self.holograms
                    or self.targets)


def position_at(unit, tick: int):
    """Where ``unit`` is scheduled to be at an absolute ``tick``.

    THE scrub-preview primitive (§16). Determinism is what makes it worth
    having: the compiled plan is not an estimate, so this is an exact dry run
    of the player's own orders — the same number the executor will produce
    when that tick arrives, assuming the world does not intervene.

    Walks the plan's movement steps in order, carrying the last known
    position, so a query between two moves answers "standing where the
    previous move left me", which is what a ghost should show.

    Tick convention: the answer is the position OBSERVABLE when
    ``sim.tick == tick`` — i.e. after tick ``tick - 1`` has executed, which is
    what reading ``unit.x`` at that moment gives you. Hence the ``- 1``: a
    step's first path entry is reached at the END of its start tick, so
    ``position_at(unit, step.start_tick)`` is where the unit stands as the
    step BEGINS.
    """
    plan = getattr(unit, "plan", None)
    pos = (float(unit.x), float(unit.y))
    if plan is None:
        return pos
    for step in plan.steps:
        if step.order_type not in ONEPHASE_MOVE_ORDER_TYPES or not step.path:
            continue
        if int(tick) <= step.start_tick:
            break
        idx = int(tick) - step.start_tick - 1
        if idx >= len(step.path):
            pos = step.path[-1]
        else:
            pos = step.path[idx]
            break
    return (float(pos[0]), float(pos[1]))


def enemy_at(sim, tile, team: int = 0):
    """The living enemy whose footprint covers ``tile``, or ``None``."""
    if tile is None:
        return None
    fx, fy = int(tile[0]), int(tile[1])
    for u in sim.units:
        if (u.alive and u.team != team
                and u.tile_x <= fx < u.tile_x + u.footprint
                and u.tile_y <= fy < u.tile_y + u.footprint):
            return u
    return None


def plan_overlay(sim, unit, hover_tile=None, armed_action=None) -> PlanOverlay:
    """The whole planning visualization for one unit (§16).

    Arrival times are read straight off the compiled schedule — they ARE the
    schedule, not a separate estimate — which is the property that makes the
    numbers on screen trustworthy.
    """
    overlay = PlanOverlay()
    _add_target_markers(sim, unit, overlay, hover_tile, armed_action)
    plan = getattr(unit, "plan", None)
    if plan is None:
        return overlay
    tps = float(CFG.clock.ticks_per_second)
    round_start = sim.round_start_tick()

    def secs(tick):
        return (int(tick) - round_start) / tps

    move_steps = [s for s in plan.steps
                  if s.order_type in ONEPHASE_MOVE_ORDER_TYPES and s.path
                  and not s.retired]
    for i, step in enumerate(move_steps):
        end = step.path[-1]
        overlay.paths.append(PathViz(
            points=list(step.path), endpoint=(float(end[0]), float(end[1])),
            footprint=int(unit.footprint),
            arrival_seconds=secs(step.end_tick), blocked=step.blocked))
        # Every move but the LAST is an intermediate waypoint of the string —
        # the last one's footprint is the highlighted destination instead.
        if i < len(move_steps) - 1:
            overlay.waypoints.append(WaypointMarker(
                x=float(end[0]), y=float(end[1]),
                footprint=int(unit.footprint),
                arrival_seconds=secs(step.end_tick)))

    for step in plan.steps:
        if step.retired or step.order_type in ONEPHASE_MOVE_ORDER_TYPES:
            continue
        if step.action.targeting not in ("unit", "tile", "direction"):
            continue
        # A hologram is only interesting when the action happens somewhere the
        # marine is not standing yet — that is the whole point of showing it.
        gx, gy = position_at(unit, step.start_tick)
        if (abs(gx - unit.x) < 1e-9) and (abs(gy - unit.y) < 1e-9):
            continue
        overlay.holograms.append(Hologram(
            x=gx, y=gy, footprint=int(unit.footprint),
            at_seconds=secs(step.start_tick), action_name=step.action.name,
            target=_step_target_xy(sim, step)))
    return overlay


def _add_target_markers(sim, unit, overlay, hover_tile, armed_action) -> None:
    """Mark every enemy this marine's plan points at, plus the hover pick.

    Ordered targets are collected first and in plan order, so the marker a
    player is most likely to be reading (the next thing that happens) comes
    first; the hover marker is only added when it is not already an ordered
    target, so hovering something you already told the marine to shoot does
    not stack two rings on it.
    """
    tps = float(CFG.clock.ticks_per_second)
    round_start = sim.round_start_tick()
    ordered = set()
    plan = getattr(unit, "plan", None)
    if plan is not None:
        for step in plan.steps:
            if step.retired:
                continue
            tid = getattr(step.order, "target_unit_id", None)
            if tid is None:
                continue
            target = sim.get_unit(tid)
            if target is None or not target.alive:
                continue
            ordered.add(int(tid))
            overlay.targets.append(TargetMarker(
                unit_id=int(tid), x=float(target.x), y=float(target.y),
                footprint=int(target.footprint),
                action_name=step.action.name,
                at_seconds=(step.start_tick - round_start) / tps))

    if not armed_action:
        return
    action = sim.actions_table.by_name.get(armed_action)
    if action is None or action.targeting != "unit":
        return
    hovered = enemy_at(sim, hover_tile, team=unit.team)
    if hovered is None or int(hovered.id) in ordered:
        return
    overlay.targets.append(TargetMarker(
        unit_id=int(hovered.id), x=float(hovered.x), y=float(hovered.y),
        footprint=int(hovered.footprint), action_name=action.name,
        hovered=True))


def _step_target_xy(sim, step):
    tid = getattr(step.order, "target_unit_id", None)
    if tid is not None:
        target = sim.get_unit(tid)
        if target is not None:
            return (float(target.center_tile_x()),
                    float(target.center_tile_y()))
    return (float(step.order.target_fx), float(step.order.target_fy))


# ---------------------------------------------------------------------------
# Planning clock (§16)
# ---------------------------------------------------------------------------
@dataclass
class PlanningClock:
    enabled: bool
    remaining_seconds: float = 0.0
    total_seconds: float = 0.0
    expired: bool = False

    @property
    def fraction(self) -> float:
        if not self.enabled or self.total_seconds <= 0:
            return 1.0
        return max(0.0, min(1.0, self.remaining_seconds / self.total_seconds))


def planning_clock(sim, elapsed_seconds: float) -> PlanningClock:
    """The multiplayer submit timer (§16).

    ``planning_clock_seconds = 0`` means untimed, which is the single-player
    default — so this returns a disabled clock and the caller draws nothing.
    On timeout the caller simply resumes; units with no orders idle per §13
    (return fire only), which is why running out of time is a soft failure
    rather than a lost turn.
    """
    total = float(CFG.onephase.planning_clock_seconds)
    if total <= 0:
        return PlanningClock(enabled=False)
    remaining = max(0.0, total - float(elapsed_seconds))
    return PlanningClock(enabled=True, remaining_seconds=remaining,
                         total_seconds=total, expired=remaining <= 0.0)


# ---------------------------------------------------------------------------
# Fog of war (§8) — gating, nothing more
# ---------------------------------------------------------------------------
def drawable_enemies(sim, team: int = 0) -> list:
    """Enemies the renderer may draw.

    Fog in v1 is visibility gating ONLY: no fog layer, no last-known-position
    ghosts — unseen is simply gone. A ruleset without a vision model
    (``fog_of_war`` False) returns everything, exactly as it always drew.
    """
    enemies = [u for u in sim.units if u.team != team and u.alive]
    if not getattr(sim.ruleset, "fog_of_war", False):
        return enemies
    visible = set(sim.visible_enemy_ids(team))
    return [u for u in enemies if int(u.id) in visible]


# ---------------------------------------------------------------------------
# Flashlights (§8) — render-only, both variants
# ---------------------------------------------------------------------------
@dataclass
class FlashlightCone:
    unit_id: int
    x: float
    y: float
    facing: float               # radians, unit convention (y-up, 0 = East)
    half_deg: float
    range_tiles: float


def flashlight_cones(sim, team: int = 0, mode: str = "team",
                     selected_unit_id=None, cursor_tile=None,
                     paused: bool = False) -> list:
    """Marine-carried flashlights, as the expression of the facing cone (§8).

    Two variants exist deliberately — "build BOTH variants (selected unit only
    / whole team) behind a toggle and feel it out" (§20 item 3) — selected by
    ``mode``. During PLANNING the flashlight aims toward the cursor, which is
    what makes it a planning affordance rather than decoration.

    **Render-only in v1.** These cones must not reach the sim: the moment
    lights affect gameplay (zombies drawn to light, vision limited to lit
    areas) they cross into a stealth system, which is a deliberate decision
    for another session and not drift to be smuggled in here.
    """
    half = float(CFG.onephase.vision_cone_half_deg)
    out = []
    for u in sim.units:
        if u.team != team or not u.alive:
            continue
        # `selected_unit_id or -1` would be a bug here: unit id 0 is falsy.
        if mode == "selected" and (selected_unit_id is None
                                   or int(u.id) != int(selected_unit_id)):
            continue
        facing = float(u.facing)
        if paused and cursor_tile is not None:
            dx = float(cursor_tile[0]) - u.center_tile_x()
            dy = float(cursor_tile[1]) - u.center_tile_y()
            if dx or dy:
                # World y is down, facing is y-up: the one negation, matching
                # Unit.face_towards. math.atan2 is fine HERE and only here —
                # this value is render-only and never re-enters the sim.
                facing = math.atan2(-dy, dx)
        out.append(FlashlightCone(
            unit_id=int(u.id), x=float(u.center_tile_x()),
            y=float(u.center_tile_y()), facing=facing, half_deg=half,
            range_tiles=25.0))
    return out


# ---------------------------------------------------------------------------
# The DS3 menu (§15)
# ---------------------------------------------------------------------------
#: Erik's spec: "Start/menu button -> [Inventory, Equipment, Character,
#: Options…, Quit]". Works beautifully on controller; in WEGO the planning
#: pause hosts it naturally; in ContinuousRealtime it overlays without pausing,
#: exactly like DS.
DS3_PAGES = ("Inventory", "Equipment", "Character", "Options", "Quit")


@dataclass
class MenuRow:
    label: str
    value: str = ""
    action_name: str = ""       # set when the row is draggable to the hotbar
    slot: int = None            # loadout slot index, for Equipment rows


@dataclass
class MenuModel:
    pages: tuple = DS3_PAGES
    page_index: int = 0
    rows: list = field(default_factory=list)

    @property
    def page(self) -> str:
        return self.pages[self.page_index % len(self.pages)]


def ds3_menu(sim, unit, page_index: int = 0) -> MenuModel:
    """Build one page of the Dark Souls 3-pattern menu (§15).

    Inventory rows carry their ``action_name``, which is exactly what makes
    them draggable onto a hotbar slot: the drop calls :func:`bind_slot` with
    that name, and slot and belt end up being the same system wearing two
    skins.
    """
    model = MenuModel(page_index=page_index % len(DS3_PAGES))
    if unit is None:
        return model
    page = model.page

    if page == "Inventory":
        for item in (getattr(unit, "belt", ()) or ()):
            name = f"use_{item}"
            if name not in sim.actions_table.by_name:
                continue
            count = _item_count(sim, unit, item)
            model.rows.append(MenuRow(
                label=sim.actions_table.get(name).label,
                value="" if count is None else str(count),
                action_name=name))
    elif page == "Equipment":
        names = ("Primary", "Secondary")
        for i, weapon_id in enumerate(getattr(unit, "loadout", ()) or ()):
            marker = " (active)" if i == int(
                getattr(unit, "active_slot", 0)) else ""
            model.rows.append(MenuRow(
                label=names[i] if i < len(names) else f"Slot {i + 1}",
                value=f"{weapon_id}{marker}", slot=i))
    elif page == "Character":
        stats = getattr(unit, "base_stats", None)
        model.rows.append(MenuRow(label="Name", value=str(unit.name)))
        model.rows.append(MenuRow(label="HP",
                                  value=f"{unit.current_hp:.0f}"))
        if stats is not None:
            for attr in ("vitality", "strength", "agility"):
                if hasattr(stats, attr):
                    model.rows.append(MenuRow(
                        label=attr.title(),
                        value=f"{getattr(stats, attr):.0f}"))
    elif page == "Options":
        model.rows.append(MenuRow(label="Round length",
                                  value=f"{CFG.clock.round_duration_seconds} s"))
        model.rows.append(MenuRow(label="Global cooldown",
                                  value=f"{CFG.onephase.gcd_seconds} s"))
        model.rows.append(MenuRow(label="Weapon swap",
                                  value=f"{CFG.onephase.weapon_swap_seconds} s"))
    elif page == "Quit":
        model.rows.append(MenuRow(label="Quit to desktop"))
    return model


__all__ = [
    "DEFAULT_HOTBAR", "DS3_PAGES", "FlashlightCone", "HotbarSlot", "Hologram",
    "MenuModel", "MenuRow", "PathViz", "PlanOverlay", "PlanningClock",
    "TargetMarker", "WaypointMarker", "bind_slot", "default_bindings",
    "drawable_enemies", "ds3_menu", "enemy_at", "flashlight_cones", "hotbar",
    "plan_overlay", "planning_clock", "position_at",
]
