"""THE particle primitive -- graphics arc G3 "Fire", patch F4a (embers).

Design: ``docs/g3_fire_design_2026-10-23.md`` §6.1, §8 ("Particles"), §10(b)
("Particle primitive"), §13.3 (the VFX clock).

WHAT IT IS. The one particle system of the renderer (embers now; sparks,
debris later -- consumers EXTEND it, never build a second one). Three parts:

  * :class:`Particles` -- the state: structure-of-arrays numpy, CAPPED
    (``cap``), every particle with an age, a lifetime, a birth temperature
    and a current temperature. Nothing here ever outlives its lifetime or
    exceeds the cap (``tests/test_particles.py``).
  * the motion law (:func:`integrate`): drag toward the AIR velocity (the
    frame's ONE tamed wind x tps, tiles/s, plus the twin's curl noise at the
    particle, ``fire_vfx_ref.curl_noise_at``), Newton cooling toward ambient
    in closed form (so the temperature is monotone in age by construction),
    a fade over the last part of life; killed past its lifetime, below the
    ramp's glow floor, off the map, or entering a solid / vacuum tile.
  * :class:`ParticleDraw` -- the GL face: motion-stretched, ADDITIVE,
    LINEAR-HDR sprites drawn into the scene RT (G0's one tone-map does the
    rest), coloured ONLY through the blackbody GPU LUT
    (``FireVFXPass.lut_tex``, the one ramp's GPU face). One direct-GL draw
    per frame: the sprites' data goes up as one small RGBA32F texture and
    ``shaders/particle_sprite.vs`` builds a quad per particle from
    ``gl_VertexID`` (the ``core.DirectPasses`` idiom -- no VBO, no per-
    particle Python call). Destination alpha is never written.

THE EMBERS (the first consumer, :class:`EmberSystem`). Spawned from the CPU
per-tick source list of the flame VFX (``FireVFXPass``'s pack: THE source
law, ``fire_vfx_ref.flame_sources``) -- never from a GPU readback (§6.1, M11):
a source tile spawns at ``rate * I`` per second (Poisson, seeded RNG), born at
``T <= T_src`` of the tile that feeds it, inside the area its flame stands on
(a permeable fuel's own tile, a wood fire's strip of open air), with an
outward kick. They run on the VFX CLOCK (``SimClock.t_noise``: sim time +
alpha, plus the idle counter while paused when ``idle_when_paused``), so they
move exactly when the flame moves, and a look shot (one tick per frame, seed
fixed) is reproducible. ``[render.embers]`` holds every dial (Ctrl+R live).

NOT HERE: the renderer's ``_effects`` queue (tracers, explosion rings, hit
splats) is LEGACY DISPLAY-SPACE EFFECTS, drawn after the resolve -- it is not
a second particle system and is not migrated in G3 (design §6.1, minor 7).

RENDER-ONLY. Reads read-only views of sim fields (through the flame VFX's
pack and ``writeable=False`` masks), writes no synced state, never imports a
sim writer; the GL class imports pyray lazily, everything above it is pure
numpy and testable headless. Float math is fine (render layer).

Credits: William T. Reeves, "Particle Systems -- a Technique for Modeling a
Class of Fuzzy Objects", SIGGRAPH 1983 (stochastic emission, lifetimes,
per-particle attributes); Robert Bridson, Jim Hourihan, Marcus Nordenstam,
"Curl-Noise for Procedural Fluid Flow", SIGGRAPH 2007 (the divergence-free
drift, the twin's modes); Isaac Newton's law of cooling (the closed-form
temperature). Archived in ``docs/papers/fire_vfx_references.md``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from . import fire_vfx_ref as ref

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

# Sampler units of the sprite draw (above raylib's batch units 0..3).
_UNIT_DATA, _UNIT_LUT = 5, 6
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_2D = 0x0DE1
_GL_TRIANGLES = 0x0004

FADE_FROM = 0.6       # the fade-out starts at this share of a particle's life
MAX_STEP_S = 0.25     # one integration step never spans more VFX time


# ---------------------------------------------------------------------------
# Settings -- [render.embers]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmberSettings:
    """Every ``[render.embers]`` dial (the config is the live value; these
    defaults are only the schema and a missing-key fallback)."""
    enabled: bool = True
    rate: float = 4.0               # embers per second per source tile at I = 1
    cap: int = 256                  # never more alive at once
    lifetime_s: float = 3.0         # the longest an ember lives (each draws 60-100 %)
    cool_s: float = 5.0             # e-fold of (T - T_amb), Newton cooling
    t_birth_jitter: float = 0.08    # born at T_src * (1 - U(0, jitter)): never hotter
    kick_speed: float = 3.5         # tiles/s: the speed it leaves the fuel with
    drag: float = 1.0               # 1/s: relaxation of its velocity toward the air
    curl: float = 1.4               # tiles/s of curl-noise drift at the particle
    curl_wavelength_tiles: float = 2.0
    curl_rate_hz: float = 0.7
    curl_modes: int = 8
    size_tiles: float = 0.05        # sprite half-width (the glow's radius)
    min_screen_px: float = 1.0      # zoomed out, the radius never drops below this
    streak_s: float = 0.06          # motion stretch: the trail is |v| * streak_s long
    brightness: float = 10.0        # x the ramp's Le(T): a spark outshines its surface
    seed: int = 4321

    def curl_dials(self) -> "ref.FireVFXDials":
        """The twin's dial set that :func:`fire_vfx_ref.curl_noise_modes`
        reads, with the embers' own noise scale -- THE curl noise, a
        different instance."""
        return ref.FireVFXDials(seed=int(self.seed),
                                curl_modes=int(self.curl_modes),
                                curl_wavelength_tiles=float(
                                    self.curl_wavelength_tiles),
                                curl_rate_hz=float(self.curl_rate_hz))


def config_keys() -> Tuple[str, ...]:
    """Every key ``[render.embers]`` is expected to carry."""
    return tuple(f.name for f in fields(EmberSettings))


def settings_from_config(cfg) -> EmberSettings:
    """``[render.embers]`` -> settings (schema types kept; a missing key
    falls back to the schema default)."""
    sec = getattr(getattr(cfg, "render", None), "embers", None)
    base = EmberSettings()
    kw = {}
    for f in fields(EmberSettings):
        d = getattr(base, f.name)
        kw[f.name] = (d if sec is None or not hasattr(sec, f.name)
                      else type(d)(getattr(sec, f.name)))
    kw["cap"] = max(0, int(kw["cap"]))
    return EmberSettings(**kw)


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------

_FIELDS = ("x", "y", "vx", "vy", "age", "life", "t0", "temp")
_COL = {f: i for i, f in enumerate(_FIELDS)}
X, Y, VX, VY, AGE, LIFE, T0, TEMP = range(8)


def _column(i: int):
    def get(self):
        return self.d[:, i]

    def put(self, v):
        self.d[:, i] = v
    return property(get, put)


class Particles:
    """The particle state, at most ``cap`` alive: ONE ``(n, 8)`` float64
    array ``d`` (columns ``x y vx vy age life t0 temp``; one fancy index per
    kill, one concatenate per spawn -- the CPU budget), with a property per
    column.

    Positions in TILES (x east, y south; tile (i, j) spans [j, j+1) x [i,
    i+1)), velocities in tiles/s, ages and lifetimes in seconds of the VFX
    clock, temperatures in Kelvin (``t0`` at birth, ``temp`` now)."""

    x, y, vx, vy = _column(X), _column(Y), _column(VX), _column(VY)
    age, life, t0, temp = _column(AGE), _column(LIFE), _column(T0), _column(TEMP)

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.d = np.zeros((0, len(_FIELDS)))

    @property
    def n(self) -> int:
        return int(self.d.shape[0])

    def add(self, **arrs) -> int:
        """Append particles (every field of ``_FIELDS`` but ``age`` / ``temp``
        given; age starts at 0, temp at t0) -- only as many as the cap leaves
        room for, the first ones offered (the caller shuffles if it cares).
        Returns how many were added."""
        room = max(0, self.cap - self.n)
        k = min(room, int(np.asarray(arrs["x"]).shape[0]))
        if k == 0:
            return 0
        new = np.zeros((k, len(_FIELDS)))
        for f in ("x", "y", "vx", "vy", "life", "t0"):
            new[:, _COL[f]] = np.asarray(arrs[f], dtype=np.float64)[:k]
        new[:, TEMP] = new[:, T0]
        self.d = np.concatenate([self.d, new])
        return k

    def keep(self, mask: np.ndarray) -> None:
        self.d = self.d[mask]

    def set_cap(self, cap: int) -> None:
        """A live cap change: a lower cap drops the OLDEST beyond it."""
        self.cap = int(cap)
        if self.n > self.cap:
            order = np.argsort(self.age, kind="stable")[:self.cap]
            self.keep(np.sort(order))

    def clear(self) -> None:
        self.d = self.d[:0]


# ---------------------------------------------------------------------------
# The ember law (pure)
# ---------------------------------------------------------------------------

def cooled_temperature(t0, age, t_amb_k: float, cool_s: float) -> np.ndarray:
    """Newton cooling in closed form: ``T_amb + (T0 - T_amb) exp(-age /
    cool_s)`` -- monotone non-increasing in age for ``T0 >= T_amb``, and
    never above the birth temperature (it never re-heats)."""
    t0 = np.asarray(t0, dtype=np.float64)
    return t_amb_k + (t0 - t_amb_k) * np.exp(np.asarray(age, np.float64)
                                             * (-1.0 / max(float(cool_s),
                                                           1e-6)))


def fade(age, life) -> np.ndarray:
    """The draw weight's fade over the last ``1 - FADE_FROM`` of a life:
    1 until then, smoothly to 0 at ``life``; monotone non-increasing (no
    fade-in: a spark is born at its brightest)."""
    u = (np.asarray(age, np.float64) / np.maximum(life, 1e-9)
         - FADE_FROM) * (1.0 / (1.0 - FADE_FROM))
    u = np.minimum(np.maximum(u, 0.0), 1.0)
    return 1.0 - u * u * (3.0 - 2.0 * u)


def ember_radiance(temp, weight, ramp) -> np.ndarray:
    """(N, 3) linear radiance of an ember's core: ``Le(T) * weight`` with
    ``Le = chroma * intensity`` of THE ramp -- the spec of
    ``shaders/particle_sprite.fs`` (which reads the same through the
    blackbody GPU LUT)."""
    chroma, inten = ramp.emission_at_kelvin(np.asarray(temp, np.float64))
    return chroma * (inten * np.asarray(weight, np.float64))[:, None]


class SourceTiles(NamedTuple):
    """One tick's source tiles, flattened once per pack (the spawn's
    per-frame cost is then one Poisson draw when nothing is born):
    window tile (``ty``, ``tx``), ``cover`` code, feeder intensity
    ``i_feed = s / s0`` and ``t_src`` (K)."""
    ty: np.ndarray
    tx: np.ndarray
    cover: np.ndarray
    i_feed: np.ndarray
    t_src: np.ndarray


def source_tiles(sources, s0: float) -> Optional[SourceTiles]:
    """``FlameSources`` -> its source tiles (``cover`` != 0); None when
    nothing burns."""
    if sources is None:
        return None
    cover = np.asarray(sources.cover)
    ty, tx = np.nonzero(cover)
    if ty.size == 0:
        return None
    return SourceTiles(ty=ty, tx=tx, cover=cover[ty, tx].astype(np.int64),
                       i_feed=np.asarray(sources.s)[ty, tx]
                       / max(float(s0), 1e-12),
                       t_src=np.asarray(sources.t_src_k)[ty, tx])


_SPAWN_KEYS = ("x", "y", "vx", "vy", "t0", "life")


def ember_spawns(tiles: Optional[SourceTiles], origin: Tuple[int, int],
                 dt: float, settings: EmberSettings, strip_frac: float,
                 rng: np.random.Generator) -> dict:
    """The embers born in ``dt`` seconds from one tick's source tiles
    (:func:`source_tiles` of the flame VFX's pack: THE source law on a
    window whose north-west tile is ``origin = (y0, x0)``).

    A source tile spawns at ``rate * I`` per second, ``I`` its feeder's
    intensity -- so no fire, no ember. (Drawn as ONE Poisson count over all
    tiles, split among them in proportion to their rates: the same process
    as a Poisson per tile.) Each is born INSIDE the area its flame stands on
    (``COVER_FULL``: anywhere in the tile; a face strip: within
    ``strip_frac`` of that face, the open air beside burning wood), at ``T =
    T_src * (1 - U(0, t_birth_jitter))`` (never hotter than its feeder),
    leaving the fuel at up to ``kick_speed``: away from the wood across a
    strip, in any direction over permeable fuel. Returns arrays x, y, vx,
    vy, t0, life (tiles, tiles/s, K, s)."""
    if tiles is None or settings.rate <= 0.0 or dt <= 0.0:
        return {k: np.zeros(0) for k in _SPAWN_KEYS}
    rates = np.maximum(settings.rate * tiles.i_feed, 0.0)
    total = float(rates.sum())
    n = int(rng.poisson(total * float(dt))) if total > 0.0 else 0
    if n == 0:
        return {k: np.zeros(0) for k in _SPAWN_KEYS}
    # every uniform this spawn needs, in ONE draw (rows: tile, x, y, face,
    # angle, speed, temperature, life) -- the CPU budget, and the order
    # the seeded stream is consumed in
    u = rng.random((8, n))
    idx = (np.searchsorted(np.cumsum(rates) / total, u[0], side="right")
           if rates.size > 1 else np.zeros(n, dtype=np.int64))
    idx = np.minimum(idx, rates.size - 1)
    cv = tiles.cover[idx]
    full = (cv & ref.COVER_FULL) != 0
    fx = u[1].copy()
    fy = u[2].copy()
    ang = 2.0 * np.pi * u[4]
    dirx, diry = np.cos(ang), np.sin(ang)
    strip = ~full
    if strip.any():
        # a face strip (chosen uniformly among the faces this tile's cover
        # names): within strip_frac of the face, pushed AWAY from the
        # burning wood behind it, +-60 degrees
        faces = np.array([ref.FACE_N, ref.FACE_E, ref.FACE_S, ref.FACE_W])
        has = (cv[:, None] & faces[None, :]) != 0               # (n, 4)
        nf = np.maximum(has.sum(axis=1), 1)
        pick = np.minimum((u[3] * nf).astype(np.int64), nf - 1)
        which = np.argmax(np.cumsum(has, axis=1) > pick[:, None], axis=1)
        sf = float(strip_frac)
        a, b = u[1], u[2] * sf
        for face, (px, py, ox, oy) in (
                (0, (a, b, 0.0, 1.0)),               # N: y in [0, sf), push south
                (1, (1.0 - b, a, -1.0, 0.0)),        # E: x in (1-sf, 1], push west
                (2, (a, 1.0 - b, 0.0, -1.0)),        # S: y in (1-sf, 1], push north
                (3, (b, a, 1.0, 0.0))):              # W: x in [0, sf), push east
            sel = strip & (which == face)
            if sel.any():
                sp = (2.0 * u[4][sel] - 1.0) * (np.pi / 3.0)
                c, s_ = np.cos(sp), np.sin(sp)
                fx[sel] = px[sel] if np.ndim(px) else px
                fy[sel] = py[sel] if np.ndim(py) else py
                dirx[sel] = ox * c - oy * s_
                diry[sel] = ox * s_ + oy * c
    speed = settings.kick_speed * (0.4 + 0.6 * u[5])
    jit = 1.0 - float(settings.t_birth_jitter) * u[6]
    life = settings.lifetime_s * (0.6 + 0.4 * u[7])
    return dict(x=tiles.tx[idx] + float(origin[1]) + fx,
                y=tiles.ty[idx] + float(origin[0]) + fy,
                vx=dirx * speed, vy=diry * speed,
                t0=tiles.t_src[idx] * jit, life=life)


def _tiles_of(xy: np.ndarray, shape) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """(iy, ix) of each particle's tile, clamped to the map, and whether it
    is on the map at all."""
    h, w = shape
    x, y = xy[:, 0], xy[:, 1]
    inside = (x >= 0.0) & (x < w) & (y >= 0.0) & (y < h)
    ix = np.minimum(np.maximum(x, 0.0), w - 1).astype(np.int64)
    iy = np.minimum(np.maximum(y, 0.0), h - 1).astype(np.int64)
    return iy, ix, inside


def integrate(p: Particles, dt: float, t: float, *, air_fn: Callable,
              blocked: Optional[np.ndarray], settings: EmberSettings,
              t_amb_k: float, glow_min_k: float, modes) -> None:
    """Advance every particle by ``dt`` s of the VFX clock (ending at noise
    time ``t``): drag toward the air (``air_fn(iy, ix)`` -> (N, 2) tiles/s,
    the tamed wind x tps at each particle's tile) plus the curl noise, move,
    age, cool; then kill what is past its lifetime, cooled below the glow
    floor (it shows nothing), off the map, or inside a ``blocked`` (solid
    or vacuum) tile."""
    if p.n == 0 or dt <= 0.0:
        return
    d = p.d
    if blocked is not None:
        iy, ix, _ = _tiles_of(d[:, X:Y + 1], blocked.shape)
        air = np.asarray(air_fn(iy, ix), dtype=np.float64)
    else:
        air = np.zeros((p.n, 2))
    if settings.curl != 0.0:
        air = air + settings.curl * ref.curl_noise_at(d[:, X], d[:, Y], t,
                                                      modes)
    a = 1.0 - math.exp(-max(float(settings.drag), 0.0) * dt)
    d[:, VX:VY + 1] += (air - d[:, VX:VY + 1]) * a
    d[:, X:Y + 1] += d[:, VX:VY + 1] * dt
    d[:, AGE] += dt
    d[:, TEMP] = cooled_temperature(d[:, T0], d[:, AGE], t_amb_k,
                                    settings.cool_s)
    alive = (d[:, AGE] < d[:, LIFE]) & (d[:, TEMP] > glow_min_k)
    if blocked is not None:
        iy, ix, inside = _tiles_of(d[:, X:Y + 1], blocked.shape)
        alive &= inside & ~blocked[iy, ix]
    if not alive.all():
        p.keep(alive)


def pack_sprites(p: Particles, view, px_per_tile: float,
                 settings: EmberSettings,
                 screen_px_per_world_px: float = 1.0) -> np.ndarray:
    """The sprites in ``view`` ((x0, y0, x1, y1) tiles, None = all), one
    tile of margin -> ``(2, k, 4)`` float32, the data texture's two rows:
    row 0 = (x, y, vx, vy) in world px and px/s, row 1 = (T K, weight =
    brightness * fade, radius px, streak s). k = 0 draws nothing.

    Zoomed out (``screen_px_per_world_px`` = the camera's zoom / the world
    RT's px per tile), a spark's radius is floored at ``min_screen_px``
    SCREEN pixels and its weight scaled by ``(r / r_floor)^2`` -- the same
    light on a footprint the downsampling blit cannot skip, so a moving
    spark does not pop in and out."""
    d = p.d
    if view is not None and p.n:
        vx0, vy0, vx1, vy1 = view
        x, y = d[:, X], d[:, Y]
        d = d[(x >= vx0 - 1.0) & (x <= vx1 + 1.0)
              & (y >= vy0 - 1.0) & (y <= vy1 + 1.0)]
    k = int(d.shape[0])
    out = np.empty((2, k, 4), np.float32)
    if k == 0:
        return out
    ppt = float(px_per_tile)
    out[0] = d[:, X:VY + 1] * ppt
    r = settings.size_tiles * ppt
    r_floor = settings.min_screen_px / max(float(screen_px_per_world_px),
                                           1e-6)
    gain = 1.0
    if r_floor > r > 0.0:
        gain, r = (r / r_floor) ** 2, r_floor
    out[1, :, 0] = d[:, TEMP]
    out[1, :, 1] = settings.brightness * gain * fade(d[:, AGE], d[:, LIFE])
    out[1, :, 2] = r
    out[1, :, 3] = settings.streak_s
    return out


# ---------------------------------------------------------------------------
# The embers (the first consumer)
# ---------------------------------------------------------------------------

class EmberSystem:
    """The embers' CPU side: per frame, :meth:`update` steps the particles
    on the VFX clock and spawns from the flame VFX's current sources.
    Pure numpy (no GL): the renderer owns a :class:`ParticleDraw` for the
    picture."""

    def __init__(self, settings: EmberSettings, *, t_amb_k: float,
                 glow_min_k: float):
        self.settings = settings
        self.t_amb_k = float(t_amb_k)
        self.glow_min_k = float(glow_min_k)
        self.rng = np.random.default_rng(int(settings.seed))
        self.p = Particles(settings.cap)
        self.enabled_override: Optional[bool] = None
        self._t_prev: Optional[float] = None
        self._modes_key = None
        self._modes = None
        self._src_obj = None
        self._src_s0 = None
        self._src_tiles: Optional[SourceTiles] = None
        self.spawned_last_frame = 0
        self.last_update_ms = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def toggle(self) -> bool:
        """The live on/off key (G3 F4a, HUMAN-TEST 2)."""
        self.enabled_override = not self.enabled
        if not self.enabled:
            self.p.clear()
        return self.enabled

    def refresh_settings(self, settings: EmberSettings) -> None:
        if settings.cap != self.p.cap:
            self.p.set_cap(settings.cap)
        self.settings = settings
        if not self.enabled:
            self.p.clear()

    def modes(self):
        s = self.settings
        key = (s.seed, s.curl_modes, s.curl_wavelength_tiles, s.curl_rate_hz)
        if key != self._modes_key:
            self._modes = ref.curl_noise_modes(s.curl_dials())
            self._modes_key = key
        return self._modes

    def update(self, *, t: float, sources, origin: Tuple[int, int],
               s0: float, strip_frac: float, wind_fn: Callable,
               tps: float, blocked: Optional[np.ndarray]) -> None:
        """One frame at VFX noise time ``t`` (monotone): integrate the alive
        embers over the elapsed VFX time, then spawn the new ones of that
        span from ``sources`` (None = nothing burns). ``wind_fn`` returns
        the frame's ONE tamed wind (tiles per tick) and is called only when
        a particle exists to ride it; ``blocked`` = solid | vacuum (full
        map, read only)."""
        t0w = time.perf_counter()
        self.spawned_last_frame = 0
        if not self.enabled:
            self._t_prev = float(t)
            self.last_update_ms = (time.perf_counter() - t0w) * 1000.0
            return
        dt = 0.0 if self._t_prev is None else max(float(t) - self._t_prev, 0.0)
        self._t_prev = float(t)
        dt = min(dt, MAX_STEP_S)
        s = self.settings
        if dt > 0.0:
            if self.p.n:
                def air(iy, ix):
                    # index first, convert the few rows (never the map)
                    return np.asarray(wind_fn())[iy, ix].astype(
                        np.float64) * float(tps)

                integrate(self.p, dt, t, air_fn=air, blocked=blocked,
                          settings=s, t_amb_k=self.t_amb_k,
                          glow_min_k=self.glow_min_k, modes=self.modes())
            if sources is not self._src_obj or s0 != self._src_s0:
                self._src_obj, self._src_s0 = sources, s0
                self._src_tiles = source_tiles(sources, s0)
            new = ember_spawns(self._src_tiles, origin, dt, s, strip_frac,
                               self.rng)
            if new["x"].size:
                room = self.p.cap - self.p.n
                if new["x"].size > room:          # a fair subset, not the first
                    keep = np.sort(self.rng.permutation(new["x"].size)[:room])
                    new = {k: v[keep] for k, v in new.items()}
                self.spawned_last_frame = self.p.add(**new)
        self.last_update_ms = (time.perf_counter() - t0w) * 1000.0


# ---------------------------------------------------------------------------
# The GL face
# ---------------------------------------------------------------------------

_SMOKE_MARKER = "// <SMOKE_SHADOW>"
_UNIT_SMOKE = 7

# The sprite's stages (``shaders/particle_sprite.fs`` u_stage), in draw order:
# in the scene under the soot, then the compensation after it (§13.7).
SPRITE_STAGES = {"scene": 0, "compensation": 1}


def sprite_shader_sources() -> Tuple[str, str]:
    """(vertex, fragment) source of the sprite draw, the fragment COMPOSED
    with the smoke cast product's shared GLSL (``lit3d._SMOKE_SHADOW_GLSL``:
    ``smokeCoverAt``) at its marker -- the one copy of that convention (also
    what ``tests/test_one_tonemap`` scans)."""
    from .lit3d import _SMOKE_SHADOW_GLSL
    fs = (SHADERS_DIR / "particle_sprite.fs").read_text(encoding="utf-8")
    if _SMOKE_MARKER not in fs:
        raise ValueError(f"particle_sprite.fs lacks '{_SMOKE_MARKER}'")
    return ((SHADERS_DIR / "particle_sprite.vs").read_text(encoding="utf-8"),
            fs.replace(_SMOKE_MARKER, _SMOKE_SHADOW_GLSL))


class ParticleDraw:
    """Draws packed sprites (:func:`pack_sprites`) into the BOUND target --
    the scene RT, linear HDR -- additively, keeping the destination alpha,
    in ONE direct-GL draw call."""

    _UNIFORMS = ("u_parts", "u_lut", "u_world_px", "u_lut_k", "u_lut_n",
                 "u_stage", "u_glow_through_ceil", "u_smoke_cover")

    def __init__(self, cap: int):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        vs, fs = sprite_shader_sources()
        sh = rl.load_shader_from_memory(vs, fs)
        if sh.id == 0 or sh.id == rl.rl_get_shader_id_default():
            raise RuntimeError("particle sprite shader failed to compile/link")
        self.shader = sh
        self._loc = {u: rl.get_shader_location(sh, u) for u in self._UNIFORMS}
        self._cache = {}
        self.tex = None
        self._alloc(cap)
        self.draws_last_frame = 0
        self.sprites_last_frame = 0

    def _alloc(self, cap: int) -> None:
        if self.tex is not None:
            self._rl.unload_texture(self.tex)
        self.width = max(1, int(cap))
        self.tex = self._core.create_dynamic_rgba32f_texture(self.width, 2)

    def draw(self, sprites: np.ndarray, *, world_px_w: float,
             world_px_h: float, lut_tex, lut_k: Tuple[float, float],
             lut_n: int, stage: str = "scene", smoke_cover=None,
             glow_through_ceil: float = 1.0) -> int:
        """Draw ``sprites`` ((2, k, 4) float32) at ``stage`` ("scene": under
        the soot; "compensation": after it, weighted by the soot's alpha x
        ``1 - glow_through_ceil``, read from ``smoke_cover`` = the relief's
        cast product texture -- skipped without one). The data texture is
        uploaded by the "scene" stage; the compensation reuses it. Returns
        the draw calls issued (0 when there is nothing to draw: no
        particle, no draw)."""
        k = int(sprites.shape[1])
        comp = stage == "compensation"
        if not comp:
            self.draws_last_frame = 0
            self.sprites_last_frame = k
        if k == 0 or (comp and (smoke_cover is None
                                or glow_through_ceil >= 1.0)):
            return 0
        rl, gl = self._rl, self._gl
        if not comp:
            if k > self.width:
                self._alloc(k)
            a = np.ascontiguousarray(sprites, dtype=np.float32)
            rl.update_texture_rec(self.tex,
                                  rl.Rectangle(0.0, 0.0, float(k), 2.0),
                                  rl.ffi.cast("void *",
                                              rl.ffi.from_buffer("float[]",
                                                                 a)))
        # additive RGB (ONE, ONE), destination alpha kept (ZERO, ONE);
        # begin_blend_mode flushes raylib's batch (the custom factors mark
        # the mode modified) and sets the GL blend state the draw uses
        rl.rl_set_blend_factors_separate(rl.RL_ONE, rl.RL_ONE, rl.RL_ZERO,
                                         rl.RL_ONE, rl.RL_FUNC_ADD,
                                         rl.RL_FUNC_ADD)
        rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM_SEPARATE)
        gl.UseProgram(self.shader.id)
        self._set_i("u_parts", _UNIT_DATA)
        self._set_i("u_lut", _UNIT_LUT)
        self._set_f("u_world_px", float(world_px_w), float(world_px_h))
        self._set_f("u_lut_k", float(lut_k[0]), float(lut_k[1]))
        self._set_f("u_lut_n", float(lut_n))
        self._set_i("u_stage", SPRITE_STAGES[stage])
        self._set_i("u_smoke_cover", _UNIT_SMOKE)
        self._set_f("u_glow_through_ceil", float(glow_through_ceil))
        gl.BindVertexArray(gl.vao)
        units = [(_UNIT_DATA, self.tex.id), (_UNIT_LUT, lut_tex.id)]
        if comp:
            units.append((_UNIT_SMOKE, smoke_cover.id))
        for unit, tid in units:
            gl.ActiveTexture(_GL_TEXTURE0 + unit)
            gl.BindTexture(_GL_TEXTURE_2D, int(tid))
        gl.DrawArrays(_GL_TRIANGLES, 0, 6 * k)
        for unit, _ in units:
            gl.ActiveTexture(_GL_TEXTURE0 + unit)
            gl.BindTexture(_GL_TEXTURE_2D, 0)
        gl.ActiveTexture(_GL_TEXTURE0)
        gl.BindVertexArray(0)
        gl.UseProgram(0)
        rl.end_blend_mode()
        self.draws_last_frame += 1
        return 1

    def _set_i(self, name: str, v: int) -> None:
        loc = self._loc.get(name, -1)
        if loc >= 0 and self._cache.get(name) != (v,):
            self._cache[name] = (v,)
            self._gl.Uniform1i(loc, int(v))

    def _set_f(self, name: str, *v: float) -> None:
        loc = self._loc.get(name, -1)
        if loc < 0 or self._cache.get(name) == v:
            return
        self._cache[name] = v
        if len(v) == 1:
            self._gl.Uniform1f(loc, v[0])
        else:
            self._gl.Uniform2f(loc, v[0], v[1])

    def unload(self) -> None:
        self._rl.unload_shader(self.shader)
        self._rl.unload_texture(self.tex)


__all__ = [
    "EmberSettings", "settings_from_config", "config_keys", "Particles",
    "cooled_temperature", "fade", "ember_radiance", "SourceTiles",
    "source_tiles", "ember_spawns",
    "integrate", "pack_sprites", "EmberSystem", "ParticleDraw",
    "sprite_shader_sources", "FADE_FROM", "SPRITE_STAGES",
]
