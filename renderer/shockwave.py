"""The shock front -- the second term of the RESOLVE'S DISTORTION BLOCK
(graphics arc G4 "The explosion look", patch P2;
docs/explosion_look_design_2026-11-07.md §3.3 as amended by its §6 critique
table: B2, S1, S4, N1, N2, N4, N5, S8).

WHAT IT IS. A detonation sends a refractive ring across its room: a thin band
where the floor seen through the air is pushed outward, travelling at
``speed_tiles_s`` and gone after ``life_s``. It is a UV offset inside the one
resolve (``shaders/world_resolve.fs``, ``shock_term`` -- one line in
``distortion_px``, beside the heat shimmer), never a second pass; the HUD is
never distorted and the tone-map stays the one tone-map.

WHY SCRIPTED. The pressure planes cannot give it a shape: the projection
equalises a room in one tick (design §2.2, measured 2026-11-07), so there is
no travelling front to refract. The ring is a pure function of the
detonation event (its ``pos``, ``strength``, ``approach``, ``tick0``) and the
level's open tiles.

THE WALK (B2). The ring never shows through a wall: at detonation it takes
the canonical Blast reach walk, ``blast_expansion.reach_depth(open_mask(gmap),
fy, fx, r2, approach)`` with ``r2`` the ring's whole reach (``speed * life`` +
1 tiles, squared), and builds ONE per-tile distance plane from it
(:func:`distance_plane`). Tiles the walk does not reach hold
:data:`UNREACHED` (a large value, never -1), so the bilinear read at a wall
lands far outside the band and the front stops there instead of inventing a
second one. Up to four rings' planes are the RGBA channels of ONE float
texture, re-uploaded only when the ring set changes.

THE DISTANCE. The walk's hop count is an L1 (taxicab) distance: in an open
room its equal-depth lines are diamonds, so a ring on raw hops would be a
square turned 45 degrees. The plane is therefore the straight line from the
centre PLUS the walk's DETOUR -- ``hypot(dy, dx) + (depth - L1 to the nearest
seed)`` -- which is exactly the straight line wherever the walk went straight
(the detour is 0: the ring is round in an open room) and longer by the way
round wherever the walk had to go round a wall (a tile behind a wall reached
through a gap is passed by the ring LATER than its straight-line distance).
CLAUDE'S CALL (an implementation of "the ring rides the walk"); Erik judges
the look at the HUMAN-TEST.

THE OFFSET (its spec is :func:`ring_offset_px`; the shader mirrors it), per
pixel at tile position p (world px / px_per_tile, y down):

    offset = clamp( Σ_i g(d_i(p) - r_i) * unit(p - c_i) * A_i ,  A_max )

  * ``d_i`` = ring i's plane read bilinearly at p (CLAMP), ``c_i`` = the
    blast's tile centre (``pos`` + 0.5, N2), ``r_i = speed * age_i``.
  * ``g(x) = exp(-(x / w)^2)`` on ``|x| < 2 w``, exactly 0 outside (N4),
    ``w = width_tiles``: sharper than a tile, evaluated per pixel.
  * ``A_i = amp_tiles * sqrt(strength_i) * px_per_tile * zoom_fade``;
    ``A_max`` the same with the strongest live ring's strength (N4).
  * A pixel outside the live rings' band box (+1 tile) is exactly zero, and
    with no live ring ``u_shock_n == 0`` and the shader returns before any
    fetch: the resolve is its plain point fetch, bit for bit.

THE CLOCK (S1). A ring ages on the SIM clock, ``age = (tick - tick0 +
alpha) / tps`` from ``upload_state(alpha=, paused=)``: paused, it holds; a
look shot reproduces.

RENDER-ONLY. Reads the event and the open mask; writes no synced state; pyray
only in the GL class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .heat_shimmer import bilinear_tiles, zoom_fade

UNREACHED = 1.0e4          # a tile the walk did not reach (never -1: B2)
MAX_SLOTS = 4              # the RGBA channels of the one plane texture
_UNIT_PLANE = 7            # the shimmer holds 5 and 6
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_2D = 0x0DE1


@dataclass(frozen=True)
class ShockwaveSettings:
    """Every ``[render.shockwave]`` dial (the config is the live value)."""
    enabled: bool
    speed_tiles_s: float     # the front's speed, tiles per second
    width_tiles: float       # the band's e-fold half-width w (support 2 w)
    amp_tiles: float         # the peak offset at strength 1, tiles
    life_s: float            # a ring dies at this age
    max_rings: int           # live rings (1..4), the oldest dropped
    zoom_zero: float         # screen px per tile at and below which it is off
    zoom_full: float         # ... and from which it is at full strength

    @property
    def reach_tiles(self) -> float:
        """How far a ring ever gets: its walk's radius (+1 tile)."""
        return self.speed_tiles_s * self.life_s + 1.0


_KINDS = {f.name: f.type for f in fields(ShockwaveSettings)}


def config_keys() -> Tuple[str, ...]:
    return tuple(_KINDS)


def settings_from_config(cfg) -> ShockwaveSettings:
    """The ``[render.shockwave]`` settings of a loaded config (``CFG``), read
    every frame (F5 retunes live). A REFUSING door: a missing section, a
    missing key or a mistyped value raises ValueError naming it (a dial that
    fell back to a default would hide a typo while tuning)."""
    sec = getattr(getattr(cfg, "render", None), "shockwave", None)
    if sec is None:
        raise ValueError("config.toml has no [render.shockwave] section")
    vals = {}
    for name, kind in _KINDS.items():
        if not hasattr(sec, name):
            raise ValueError(f"[render.shockwave] is missing {name!r}")
        v = getattr(sec, name)
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"[render.shockwave] {name} has the wrong type: "
                             f"{v!r}")
        vals[name] = v
    s = ShockwaveSettings(**vals)
    if not 1 <= s.max_rings <= MAX_SLOTS:
        raise ValueError(f"[render.shockwave] max_rings must be 1..{MAX_SLOTS}"
                         f" (one RGBA texture): {s.max_rings}")
    for name in ("speed_tiles_s", "width_tiles", "life_s"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[render.shockwave] {name} must be > 0")
    if s.amp_tiles < 0.0:
        raise ValueError("[render.shockwave] amp_tiles must be >= 0")
    if s.zoom_full <= s.zoom_zero:
        raise ValueError("[render.shockwave] zoom_full must exceed zoom_zero")
    return s


# ---------------------------------------------------------------------------
# The law (pure)
# ---------------------------------------------------------------------------

def distance_plane(mask: np.ndarray, fy: int, fx: int, reach_tiles: float,
                   approach=None) -> np.ndarray:
    """Ring distance per tile (float32, (h, w)), from the canonical walk
    ``blast_expansion.reach_depth`` from (fy, fx) under ``r2 =
    ceil(reach_tiles)^2``: the straight line from the centre plus the walk's
    detour (its depth minus the L1 distance to the nearest seed; module
    docstring). Unreached tiles hold :data:`UNREACHED`."""
    from simulation.blast_expansion import reach_depth, walk_seeds
    h, w = mask.shape
    plane = np.full((h, w), UNREACHED, np.float32)
    r = int(math.ceil(float(reach_tiles)))
    depth = reach_depth(mask, int(fy), int(fx), r * r, approach)
    if not depth:
        return plane
    seeds = walk_seeds(mask, int(fy), int(fx), approach)
    tiles = np.array(list(depth.keys()), np.int64)
    hops = np.array(list(depth.values()), np.float64)
    ys, xs = tiles[:, 0], tiles[:, 1]
    l1 = np.min(np.stack([np.abs(ys - sy) + np.abs(xs - sx)
                          for sy, sx in seeds]), axis=0).astype(np.float64)
    straight = np.hypot(ys - float(fy), xs - float(fx))
    plane[ys, xs] = (straight + np.maximum(hops - l1, 0.0)).astype(np.float32)
    return plane


def band(x, w: float):
    """``g(x) = exp(-(x / w)^2)`` on ``|x| < 2 w``, exactly 0 outside."""
    x = np.asarray(x, np.float64)
    return np.where(np.abs(x) < 2.0 * w, np.exp(-(x / w) ** 2), 0.0)


def band_box(planes: Sequence[np.ndarray], radii: Sequence[float],
             w: float) -> Optional[Tuple[float, float, float, float]]:
    """The tile box (x0, y0, x1, y1) outside which every ring's offset is
    exactly zero, or None when none can be non-zero. A bilinear read is in a
    ring's band only if all four corner tiles were reached and one of them is
    within ``2 w + 3`` of ``r`` (neighbouring reached tiles differ by at most
    3: a hop, its L1 and its straight line each move by <= 1), so the box is
    those tiles dilated by one tile (the bilinear footprint)."""
    hit = None
    for p, r in zip(planes, radii):
        m = np.abs(p.astype(np.float64) - float(r)) < 2.0 * w + 3.0
        hit = m if hit is None else (hit | m)
    if hit is None or not hit.any():
        return None
    h, wd = hit.shape
    rows = np.flatnonzero(hit.any(axis=1))
    cols = np.flatnonzero(hit.any(axis=0))
    return (float(max(cols[0] - 1, 0)), float(max(rows[0] - 1, 0)),
            float(min(cols[-1] + 2, wd)), float(min(rows[-1] + 2, h)))


def ring_offset_px(x_tiles, y_tiles, rings, w: float,
                   box=None) -> np.ndarray:
    """THE SPEC of the shader's ``shock_term``: the summed offset in world px
    (x east, y south) at tile positions (``x_tiles``, ``y_tiles``; any equal
    shapes). ``rings`` is a sequence of ``(plane, cx, cy, r, A)`` -- the
    ring's distance plane, its centre (tiles, the tile centre), its radius
    (tiles) and its amplitude (px, zoom-faded). The sum is clamped to
    ``A_max`` = the largest ring amplitude. ``box`` (x0, y0, x1, y1) in tiles
    zeroes everything outside it (the shader's early-out; None = no box)."""
    x = np.asarray(x_tiles, np.float64)
    y = np.asarray(y_tiles, np.float64)
    out = np.zeros(x.shape + (2,), np.float64)
    if not rings:
        return out
    a_max = max(float(rg[4]) for rg in rings)
    for plane, cx, cy, r, amp in rings:
        d = bilinear_tiles(np.asarray(plane, np.float64), x, y)
        k = band(d - float(r), w) * float(amp)
        dx, dy = x - float(cx), y - float(cy)
        ln = np.hypot(dx, dy)
        safe = np.maximum(ln, 1e-6)
        ux = np.where(ln > 1e-6, dx / safe, 0.0)
        uy = np.where(ln > 1e-6, dy / safe, 0.0)
        out[..., 0] += k * ux
        out[..., 1] += k * uy
    ln = np.linalg.norm(out, axis=-1, keepdims=True)
    out = np.where(ln > a_max, out * (a_max / np.maximum(ln, 1e-12)), out)
    if box is not None:
        x0, y0, x1, y1 = box
        inside = (x >= x0) & (y >= y0) & (x < x1) & (y < y1)
        out = np.where(inside[..., None], out, 0.0)
    return out


@dataclass
class Ring:
    """One live ring: its centre (tiles), strength, birth tick, plane slot."""
    cx: float
    cy: float
    strength: float
    tick0: int
    slot: int
    plane: np.ndarray


class RingSet:
    """The live rings and their clock -- the pure half of :class:`Shockwave`
    (no GL), so the twin's tests drive exactly what the GL face uploads."""

    def __init__(self, settings: ShockwaveSettings):
        self.settings = settings
        self.rings: list = []
        self.t_sim = 0.0             # (tick + alpha) / tps, held while paused
        self._tick = None
        self._alpha = 0.0
        self._tps = 24.0
        self.dirty = True            # the slot planes changed

    def start(self, pos, strength: float, tick0: int, mask: np.ndarray,
              approach=None) -> Ring:
        """A detonation at tile ``pos`` = (fx, fy) (the event's order): a new
        ring, the oldest dropped beyond ``max_rings``."""
        fx, fy = int(pos[0]), int(pos[1])
        plane = distance_plane(mask, fy, fx, self.settings.reach_tiles,
                               approach)
        cap = int(self.settings.max_rings)
        while len(self.rings) >= cap:
            self.rings.pop(0)
        used = {rg.slot for rg in self.rings}
        slot = min(s for s in range(MAX_SLOTS) if s not in used)
        rg = Ring(cx=fx + 0.5, cy=fy + 0.5, strength=max(float(strength), 0.0),
                  tick0=int(tick0), slot=slot, plane=plane)
        self.rings.append(rg)
        self.dirty = True
        return rg

    def set_clock(self, sim_tick: int, alpha: Optional[float], paused: bool,
                  tps: float) -> None:
        """The sim clock: ``(tick + alpha) / tps``; paused on the same tick it
        holds (so the ring stands still)."""
        tick = int(sim_tick)
        self._tps = float(tps)
        if paused and self._tick == tick:
            return
        a = 0.0 if (alpha is None or paused) else min(max(float(alpha), 0.0),
                                                      0.999999)
        self._tick, self._alpha = tick, a
        self.t_sim = (tick + a) / self._tps

    def age(self, rg: Ring) -> float:
        """``(tick - tick0 + alpha) / tps`` (exact in the tick difference)."""
        if self._tick is None:
            return 0.0
        return max((self._tick - rg.tick0 + self._alpha) / self._tps, 0.0)

    def expire(self) -> None:
        """Drop the rings at or past ``life_s`` (and beyond ``max_rings`` when
        a retune lowered it)."""
        life = float(self.settings.life_s)
        keep = [rg for rg in self.rings if self.age(rg) < life]
        keep = keep[-int(self.settings.max_rings):]
        if len(keep) != len(self.rings):
            self.dirty = True
        self.rings = keep

    def packed_planes(self, h: int, w: int) -> np.ndarray:
        """The (h, w, 4) float32 upload: ring slot k in channel k, empty
        channels :data:`UNREACHED`."""
        pack = np.full((h, w, 4), UNREACHED, np.float32)
        for rg in self.rings:
            pack[..., rg.slot] = rg.plane
        return pack

    def twin_rings(self, px_per_tile: float, zoom_px_per_tile: float):
        """The live rings as :func:`ring_offset_px` takes them."""
        s = self.settings
        fade = zoom_fade(zoom_px_per_tile, s.zoom_zero, s.zoom_full)
        out = []
        for rg in self.rings:
            amp = s.amp_tiles * math.sqrt(rg.strength) * px_per_tile * fade
            out.append((rg.plane, rg.cx, rg.cy, s.speed_tiles_s * self.age(rg),
                        amp))
        return out


# ---------------------------------------------------------------------------
# The GL face (the resolve binds it)
# ---------------------------------------------------------------------------

class Shockwave:
    """The ring set's GPU inputs: the packed distance planes (one RGBA32F
    texture, bilinear, clamped) and the per-ring uniforms; the resolve reads
    them through :meth:`uniforms` / :meth:`bind` (``WorldComposite.resolve``).
    """

    UNIFORMS = (("u_shock_n", "u_shock_plane", "u_shock_tiles",
                 "u_shock_par", "u_shock_box")
                + tuple(f"u_shock_ring[{k}]" for k in range(MAX_SLOTS)))

    def __init__(self, grid_h: int, grid_w: int, settings: ShockwaveSettings):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        self.h, self.w = int(grid_h), int(grid_w)
        self.set = RingSet(settings)
        # a tool's switch (a look shot's `hide = ["shock"]`); no key: the
        # game's switch is [render.shockwave] enabled
        self.enabled_override: Optional[bool] = None
        self.zoom_px_per_tile = 0.0
        self.plane_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        rl.set_texture_filter(self.plane_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self.last_update_ms = 0.0

    @property
    def settings(self) -> ShockwaveSettings:
        return self.set.settings

    @property
    def enabled(self) -> bool:
        if getattr(self, "enabled_override", None) is not None:
            return bool(self.enabled_override)
        return bool(self.set.settings.enabled)

    def refresh_settings(self, settings: ShockwaveSettings) -> None:
        self.set.settings = settings

    def update(self, *, effects: Sequence[dict],
               mask_fn: Callable[[], np.ndarray], sim_tick: int,
               alpha: Optional[float], paused: bool, tps: float,
               zoom_px_per_tile: float) -> None:
        """Per frame: start a ring for every ``"blast"`` effect not yet seen
        (marked ``"shock_seen"``, so a frame with several uploads never starts
        one twice), keep the sim clock and the zoom, drop dead rings, and
        re-upload the planes when the set changed. ``mask_fn`` is the
        canonical ``blast_expansion.open_mask`` of the map, asked only when a
        ring starts."""
        import time
        t0 = time.perf_counter()
        self.zoom_px_per_tile = float(zoom_px_per_tile)
        self.set.set_clock(sim_tick, alpha, paused, tps)
        mask = None
        for fx in effects:
            if fx.get("kind") != "blast" or fx.get("shock_seen"):
                continue
            fx["shock_seen"] = True
            if not self.enabled:
                continue
            if mask is None:
                mask = np.asarray(mask_fn(), bool)
            self.set.start(fx["pos"], fx.get("strength", 0.0),
                           fx.get("tick0", sim_tick), mask,
                           approach=fx.get("approach"))
        if not self.enabled and self.set.rings:
            self.set.rings = []
            self.set.dirty = True
        self.set.expire()
        if self.set.dirty:
            self._core.update_rgba32f_texture(
                self.plane_tex, self.set.packed_planes(self.h, self.w))
            self.set.dirty = False
        self.last_update_ms = (time.perf_counter() - t0) * 1000.0

    def _frame(self, px_per_tile: float):
        """``([(slot, twin ring)], box)`` of the live rings with a non-zero
        amplitude this frame."""
        twins = self.set.twin_rings(float(px_per_tile), self.zoom_px_per_tile)
        live = [(rg.slot, tw) for rg, tw in zip(self.set.rings, twins)
                if tw[4] > 0.0]
        box = None
        if live:
            box = band_box([tw[0] for _, tw in live],
                           [tw[3] for _, tw in live],
                           self.settings.width_tiles)
        return live, box

    def active(self, px_per_tile: float) -> bool:
        """Whether the resolve's shock term runs this frame (on, a live ring
        with a non-zero amplitude whose band is somewhere on the map)."""
        if not self.enabled or not self.set.rings:
            return False
        live, box = self._frame(px_per_tile)
        return bool(live) and box is not None

    def uniforms(self, px_per_tile: float):
        """``{name: values}`` of the resolve's shock uniforms this frame
        (sampler excluded): per slot (cx, cy, r, A); A = 0 marks an empty
        slot."""
        live, box = self._frame(px_per_tile)
        on = bool(live) and box is not None
        a_max = max((tw[4] for _, tw in live), default=0.0)
        slots = [(0.0, 0.0, 0.0, 0.0)] * MAX_SLOTS
        for slot, (_, cx, cy, r, amp) in live:
            slots[slot] = (float(cx), float(cy), float(r), float(amp))
        out = {"u_shock_n": (int(len(live)) if on else 0,),
               "u_shock_tiles": (float(self.w), float(self.h)),
               "u_shock_par": (float(self.settings.width_tiles), float(a_max),
                               float(px_per_tile), 0.0),
               "u_shock_box": tuple(float(b) for b in (box or (0, 0, 0, 0)))}
        for k in range(MAX_SLOTS):
            out[f"u_shock_ring[{k}]"] = slots[k]
        return out

    def bind(self) -> None:
        gl = self._gl
        gl.ActiveTexture(_GL_TEXTURE0 + _UNIT_PLANE)
        gl.BindTexture(_GL_TEXTURE_2D, int(self.plane_tex.id))
        gl.ActiveTexture(_GL_TEXTURE0)

    def unbind(self) -> None:
        gl = self._gl
        gl.ActiveTexture(_GL_TEXTURE0 + _UNIT_PLANE)
        gl.BindTexture(_GL_TEXTURE_2D, 0)
        gl.ActiveTexture(_GL_TEXTURE0)

    def unload(self) -> None:
        self._rl.unload_texture(self.plane_tex)


SAMPLER_UNITS = {"u_shock_plane": _UNIT_PLANE}

__all__ = [
    "ShockwaveSettings", "settings_from_config", "config_keys",
    "distance_plane", "band", "band_box", "ring_offset_px", "Ring", "RingSet",
    "Shockwave", "SAMPLER_UNITS", "UNREACHED", "MAX_SLOTS",
]
