"""THE tracers -- a round in flight, drawn as a glowing streak (#141).

WHAT IT IS. Every ``"tracer"`` effect of the renderer's queue (one per
``ShotFiredEvent``: the path a round covered in one sim tick) becomes a
:class:`Tracer`: a white-hot STREAK that flies the path at the round's own
speed (``speed_scale`` 1 = honest: a rifle round crosses a room inside one
or two frames -- Erik's choice, 2026-11-12; lower slows it for the eye),
smeared over ``shutter_s`` of flight like a camera's exposure, with a soft
HALO round it, a dim AFTERGLOW on the stretch of path behind it (what it
passed within two e-folds of ``glow_s``; fading once it has arrived), and, where the round stopped on a wall, a pinpoint IMPACT FLASH
and a burst of sparks thrown back off the wall (the particle primitive's
``MuzzleSparkSystem`` with its own ``[render.tracers.sparks]`` dials,
queued by the renderer from :attr:`Tracers.impacts_last_frame`).

HOW IT IS DRAWN. No shader of its own: a tracer is at most four SPRITES of
THE particle primitive (``particles.ParticleDraw``, ``particle_sprite.fs``:
a soft capsule along a segment, additive linear HDR, coloured only through
the blackbody GPU LUT), packed by :meth:`Tracers.pack` in that draw's own
row layout. They are drawn after the units with the muzzle flame (a round
flies at gun height), under the ceiling soot, with the same compensation.

WHERE A ROUND IS. Pure kinematics, no integration: ``head = v (t - t_first
+ shutter)``, ``tail = v (t - t_first)``, clipped to the path, on the SIM
clock from the first frame the tracer is drawn (paused holds it). The path
is the sim's own (the effect's ``from`` -> ``to``); a round's first segment
starts at the shooter's muzzle instead of its centre.

RENDER-ONLY. Reads the effect queue, a unit's weapon id and muzzle, and the
``solid`` mask (what a round stopped on); writes no synced state; float.
Pure numpy and testable headless (the GL face is the particle draw's).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .muzzle_flame import _as_dict, typed_row

GLOW_FLOOR = 0.02         # an afterglow below this share of itself is gone
HALO_RADIUS = 4.0         # the halo's radius over the streak's
IMPACT_BACK_TILES = 0.45  # sparks are born this far back off the wall


@dataclass(frozen=True)
class TracerKind:
    """One ``[render.tracers.kinds.<name>]`` row."""
    speed_scale: float       # x the round's own speed (1 = honest)
    shutter_s: float         # the streak is the path flown in this long
    width_m: float           # the streak's radius
    temp_k: float            # its colour on the blackbody ramp
    brightness: float        # its radiance, x the ramp's emission
    halo_gain: float         # the halo's radiance over the streak's
    glow_s: float            # the afterglow's e-fold, seconds
    glow_gain: float         # the afterglow's radiance over the streak's
    glow_temp_k: float
    impact_sparks: int       # mean sparks off a wall (0 = none)
    impact_flash_s: float    # the impact flash's e-fold
    impact_flash_m: float    # its radius


_KIND_TYPES = {f.name: f.type for f in fields(TracerKind)}
_TOP_KEYS = ("enabled", "default_kind", "weapons", "seed", "cap",
             "min_screen_px", "kinds", "sparks")


@dataclass(frozen=True)
class TracerSettings:
    enabled: bool
    default_kind: str
    weapons: Dict[str, str]
    seed: int
    cap: int
    min_screen_px: float
    kinds: Dict[str, TracerKind]

    def kind_for(self, weapon_id: str) -> Optional[TracerKind]:
        name = self.weapons.get(weapon_id or "", self.default_kind)
        return self.kinds[name] if name else None


def _kind(name: str, row) -> TracerKind:
    w = f"[render.tracers.kinds.{name}]"
    k = TracerKind(**typed_row(row, _KIND_TYPES, w))
    for f in ("speed_scale", "shutter_s", "width_m", "temp_k", "glow_s",
              "glow_temp_k", "impact_flash_s", "impact_flash_m"):
        if not getattr(k, f) > 0.0:
            raise ValueError(f"{w} {f} must be > 0")
    for f in ("brightness", "halo_gain", "glow_gain"):
        if getattr(k, f) < 0.0:
            raise ValueError(f"{w} {f} must be >= 0")
    if k.impact_sparks < 0:
        raise ValueError(f"{w} impact_sparks must be >= 0")
    return k


def settings_from_config(cfg, weapon_ids: Optional[Sequence[str]] = None
                         ) -> TracerSettings:
    """``[render.tracers]`` of a loaded config (``CFG``) or of a plain dict
    holding ``render.tracers``, read every frame (F5 retunes live). A
    REFUSING door: a missing or unknown key, a mistyped value, a ``weapons``
    key that is no ``[weapons.*]`` id, a kind name that is no ``kinds`` row
    -- each raises ValueError naming it. The ``sparks`` table is the
    particle module's (``particles.muzzle_spark_settings(cfg, "tracers")``)
    and only required to exist here."""
    if isinstance(cfg, Mapping):
        sec = cfg.get("render", {}).get("tracers")
        weapons_cfg = cfg.get("weapons")
    else:
        sec = getattr(getattr(cfg, "render", None), "tracers", None)
        weapons_cfg = getattr(cfg, "weapons", None)
    if sec is None:
        raise ValueError("config.toml has no [render.tracers] section")
    d = _as_dict(sec)
    w = "[render.tracers]"
    if not isinstance(d, dict):
        raise ValueError(f"{w} must be a table")
    for key in _TOP_KEYS:
        if key not in d:
            raise ValueError(f"{w} is missing {key!r}")
    bad = sorted(set(d) - set(_TOP_KEYS))
    if bad:
        raise ValueError(f"{w} has unknown keys {bad}")
    if not isinstance(d["enabled"], bool):
        raise ValueError(f"{w} enabled must be true or false")
    for key in ("seed", "cap"):
        if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 0:
            raise ValueError(f"{w} {key} must be an integer >= 0")
    msp = d["min_screen_px"]
    if not isinstance(msp, (int, float)) or isinstance(msp, bool) or msp < 0:
        raise ValueError(f"{w} min_screen_px must be a number >= 0")
    if not isinstance(d["kinds"], dict) or not d["kinds"]:
        raise ValueError(f"{w} kinds must hold at least one row")
    kinds = {name: _kind(name, row) for name, row in d["kinds"].items()}
    if weapon_ids is None:
        weapon_ids = sorted(_as_dict(weapons_cfg) or {}) \
            if weapons_cfg is not None else []
    if not isinstance(d["weapons"], dict):
        raise ValueError(f"{w} weapons must be a table of weapon id = kind")
    for wid, name in d["weapons"].items():
        if wid not in weapon_ids:
            raise ValueError(f"{w} weapons.{wid}: not a [weapons.*] id")
        if not isinstance(name, str) or (name and name not in kinds):
            raise ValueError(f"{w} weapons.{wid} = {name!r}: not a kinds row "
                             f"{sorted(kinds)} (\"\" = no tracer)")
    dk = d["default_kind"]
    if not isinstance(dk, str) or (dk and dk not in kinds):
        raise ValueError(f"{w} default_kind = {dk!r}: not a kinds row "
                         f"{sorted(kinds)} (\"\" = no tracer)")
    return TracerSettings(enabled=d["enabled"], default_kind=dk,
                          weapons=dict(d["weapons"]), seed=int(d["seed"]),
                          cap=int(d["cap"]), min_screen_px=float(msp),
                          kinds=kinds)


def round_speeds(cfg) -> Dict[str, float]:
    """weapon id -> its round's muzzle speed in METRES per second: the
    weapon row's ``ammo_family`` -> the first ``[ammo.*]`` row of that
    family with a ``muzzle_mps`` > 0 (config order). A weapon with none (a
    beam, a plasma bolt) is absent: its tracer covers each tick's segment in
    one tick, which is what a round still in flight honestly does."""
    if isinstance(cfg, Mapping):
        weapons, ammo = cfg.get("weapons") or {}, cfg.get("ammo") or {}
    else:
        weapons = _as_dict(getattr(cfg, "weapons", None)) or {}
        ammo = _as_dict(getattr(cfg, "ammo", None)) or {}
    by_family: Dict[str, float] = {}
    for row in ammo.values():
        if not isinstance(row, dict):
            continue
        v = row.get("muzzle_mps")
        fam = row.get("family")
        if (isinstance(fam, str) and isinstance(v, (int, float))
                and not isinstance(v, bool) and v > 0 and fam not in by_family):
            by_family[fam] = float(v)
    out = {}
    for wid, row in weapons.items():
        fam = row.get("ammo_family") if isinstance(row, dict) else None
        if fam in by_family:
            out[wid] = by_family[fam]
    return out


def stopped_on_wall(solid: Optional[np.ndarray], a, b) -> bool:
    """Did the round whose segment ran ``a`` -> ``b`` (tiles) stop on a
    wall? The march stops ON the solid tile it entered (or the last open
    position before a map edge), so: ``b``'s own tile, or the tile half a
    step beyond it along the flight, is solid."""
    if solid is None:
        return False
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return False
    h, w = solid.shape
    for k in (0.0, 0.75):
        ix = int(math.floor(float(b[0]) + dx / n * k))
        iy = int(math.floor(float(b[1]) + dy / n * k))
        if 0 <= iy < h and 0 <= ix < w and solid[iy, ix]:
            return True
    return False


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------

@dataclass
class Tracer:
    unit: int
    tick0: int
    kind: TracerKind
    a: Tuple[float, float]              # path start, tiles
    b: Tuple[float, float]              # path end, tiles
    length: float                       # tiles
    speed: float                        # tiles per sim second (scaled)
    wall: bool                          # the round stopped on a wall
    t_first: Optional[float] = None
    age: float = 0.0                    # sim seconds since t_first
    impacted: bool = False
    t_impact: Optional[float] = None    # age at which the head arrived

    @property
    def direction(self) -> Tuple[float, float]:
        n = max(self.length, 1e-9)
        return ((self.b[0] - self.a[0]) / n, (self.b[1] - self.a[1]) / n)

    def trail(self) -> Tuple[float, float, float]:
        """(from, to, fade) of the afterglow along the path: the stretch the
        round passed within the last two e-folds, at full strength while it
        flies and fading from its arrival on."""
        k = self.kind
        t_end = self.length / max(self.speed, 1e-9)
        if self.age < t_end:
            to, fade = min(self.speed * self.age, self.length), 1.0
        else:
            to, fade = self.length, math.exp(-(self.age - t_end) / k.glow_s)
        return max(0.0, to - self.speed * 2.0 * k.glow_s), to, fade

    def span(self) -> Tuple[float, float]:
        """(tail, head) of the streak along the path, tiles, at this age."""
        tail = min(self.speed * self.age, self.length)
        head = min(self.speed * (self.age + self.kind.shutter_s), self.length)
        return tail, head


@dataclass(frozen=True)
class Impact:
    """A round's streak just arrived at a wall: where (tiles, backed off
    the wall), the direction the sparks leave in (back along the flight)."""
    x: float
    y: float
    dir_x: float
    dir_y: float
    sparks: int
    unit: int
    tick0: int


class Tracers:
    """The live tracers. :meth:`consume` turns every new ``"tracer"`` effect
    into one (marking it ``"streak_seen"``; a re-delivered copy of the same
    segment is not taken twice); :meth:`update` ages them on the sim clock
    from the first frame each is drawn and reports the streaks that just
    reached a wall (:attr:`impacts_last_frame`); :meth:`pack` hands the
    particle draw its sprites. Never more than ``cap`` alive."""

    def __init__(self, settings: TracerSettings):
        self.settings = settings
        self.live: List[Tracer] = []
        self._seen: Dict[tuple, bool] = {}
        self.enabled_override: Optional[bool] = None
        self.impacts_last_frame: List[Impact] = []

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def refresh_settings(self, settings: TracerSettings) -> None:
        self.settings = settings
        if not self.enabled:
            self.live = []

    def consume(self, effects, *, weapon_of: Callable[[int], Optional[str]],
                speed_of: Callable[[str], Optional[float]],
                start_of: Callable[[int], Optional[Tuple[float, float]]],
                solid: Optional[np.ndarray], tile_size_m: float, tps: float,
                sim_tick: int) -> None:
        """``weapon_of(unit)`` -> weapon id (None: the shooter is not drawn,
        no tracer); ``speed_of(weapon)`` -> m/s or None (one tick per
        segment); ``start_of(unit)`` -> the muzzle (tiles) for a round's
        first segment, or None (the segment's own start)."""
        s = self.settings
        for fx in effects:
            if fx.get("kind") != "tracer" or fx.get("streak_seen"):
                continue
            fx["streak_seen"] = True
            if not self.enabled:
                continue
            unit = fx.get("unit")
            a, b = fx.get("from"), fx.get("to")
            if unit is None or a is None or b is None:
                continue
            a = (float(a[0]), float(a[1])); b = (float(b[0]), float(b[1]))
            tick0 = int(fx.get("tick0", sim_tick))
            key = (int(unit), tick0, a, b)
            if key in self._seen:
                continue
            self._seen[key] = True
            wid = weapon_of(int(unit))
            kind = None if wid is None else s.kind_for(str(wid))
            if kind is None:
                continue
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg < 1e-6:
                continue
            wall = stopped_on_wall(solid, a, b)
            if fx.get("launch", True):
                m = start_of(int(unit))
                # the muzzle, unless the round ended before reaching it
                if m is not None and ((b[0] - m[0]) * (b[0] - a[0])
                                      + (b[1] - m[1]) * (b[1] - a[1])) > 0.0:
                    a = (float(m[0]), float(m[1]))
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length < 1e-6:
                continue
            mps = speed_of(str(wid))
            v = (mps / float(tile_size_m) if mps else seg * float(tps))
            self.live.append(Tracer(
                unit=int(unit), tick0=tick0, kind=kind, a=a, b=b,
                length=length, speed=v * kind.speed_scale, wall=wall))
        if len(self._seen) > 8192:
            self._seen = {k: True for k in self._seen
                          if k[1] >= sim_tick - 8}
        if len(self.live) > s.cap:
            self.live = self.live[len(self.live) - s.cap:]

    def update(self, t_sim: float) -> None:
        self.impacts_last_frame = []
        alive = []
        for tr in self.live:
            if tr.t_first is None:
                tr.t_first = float(t_sim)
            tr.age = max(float(t_sim) - tr.t_first, 0.0)
            k = tr.kind
            arrived = tr.speed * (tr.age + k.shutter_s) >= tr.length
            if arrived and not tr.impacted:
                tr.impacted = True
                tr.t_impact = tr.age
                if tr.wall:
                    dx, dy = tr.direction
                    back = min(IMPACT_BACK_TILES, 0.5 * tr.length)
                    self.impacts_last_frame.append(Impact(
                        x=tr.b[0] - dx * back, y=tr.b[1] - dy * back,
                        dir_x=-dx, dir_y=-dy, sparks=k.impact_sparks,
                        unit=tr.unit, tick0=tr.tick0))
            flying = tr.speed * tr.age < tr.length
            glowing = tr.trail()[2] > GLOW_FLOOR
            flashing = (tr.wall and tr.t_impact is not None and math.exp(
                -(tr.age - tr.t_impact) / k.impact_flash_s) > GLOW_FLOOR)
            if flying or glowing or flashing:
                alive.append(tr)
        self.live = alive

    def pack(self, view, px_per_tile: float, tile_size_m: float,
             screen_px_per_world_px: float = 1.0) -> np.ndarray:
        """The live tracers' sprites, ``(2, k, 4)`` float32 in the particle
        draw's layout: row 0 = (head x, head y, vx, vy) world px and px/s,
        row 1 = (T K, weight, radius px, streak s) -- a segment ``p0 -> p1``
        is a sprite at ``p1`` with velocity ``p1 - p0`` and streak 1 s. The
        weight undoes the sprite's motion-blur dimming (``r / (r + seg)``),
        so a streak's core radiance is its kind's, whatever its length."""
        ppt = float(px_per_tile)
        ppm = ppt / float(tile_size_m)
        r_floor = self.settings.min_screen_px / max(
            float(screen_px_per_world_px), 1e-6)
        rows = []

        def sprite(p0, p1, temp, radiance, radius_px):
            r = float(radius_px)
            gain = 1.0
            if r_floor > r > 0.0:          # zoomed out: wider, same light
                gain, r = r / r_floor, r_floor
            vx, vy = (p1[0] - p0[0]) * ppt, (p1[1] - p0[1]) * ppt
            seg = 0.5 * math.hypot(vx, vy)
            rows.append(((p1[0] * ppt, p1[1] * ppt, vx, vy),
                         (temp, radiance * gain * (r + seg) / r, r, 1.0)))

        for tr in self.live:
            if tr.t_first is None:
                continue
            if view is not None:
                lo_x, hi_x = sorted((tr.a[0], tr.b[0]))
                lo_y, hi_y = sorted((tr.a[1], tr.b[1]))
                if (hi_x < view[0] - 1 or lo_x > view[2] + 1
                        or hi_y < view[1] - 1 or lo_y > view[3] + 1):
                    continue
            k = tr.kind
            dx, dy = tr.direction

            def at(s):
                return (tr.a[0] + dx * s, tr.a[1] + dy * s)

            r_px = k.width_m * ppm
            tail, head = tr.span()
            if head > tail:
                sprite(at(tail), at(head), k.temp_k, k.brightness, r_px)
                if k.halo_gain > 0.0:
                    sprite(at(tail), at(head), k.temp_k,
                           k.brightness * k.halo_gain, r_px * HALO_RADIUS)
            g0, g1, glow = tr.trail()
            if k.glow_gain > 0.0 and glow > GLOW_FLOOR and g1 > g0:
                sprite(at(g0), at(g1), k.glow_temp_k,
                       k.brightness * k.glow_gain * glow, r_px)
            if tr.wall and tr.t_impact is not None:
                f = math.exp(-(tr.age - tr.t_impact) / k.impact_flash_s)
                if f > GLOW_FLOOR:
                    p = at(tr.length)
                    sprite(p, p, k.temp_k, k.brightness * f,
                           k.impact_flash_m * ppm)
        out = np.empty((2, len(rows), 4), np.float32)
        for i, (r0, r1) in enumerate(rows):
            out[0, i] = r0
            out[1, i] = r1
        return out


__all__ = ["TracerKind", "TracerSettings", "settings_from_config",
           "round_speeds", "stopped_on_wall", "Tracer", "Impact", "Tracers",
           "GLOW_FLOOR", "HALO_RADIUS", "IMPACT_BACK_TILES"]
