"""THE particle primitive -- graphics arc G3 "Fire", patch F4a (embers).

Design: ``docs/g3_fire_design_2026-10-23.md`` §6.1, §8 ("Particles"), §10(b)
("Particle primitive"), §13.3 (the VFX clock).

WHAT IT IS. The one particle system of the renderer (embers, sparks, debris
-- consumers EXTEND it, never build a second one). Three parts:

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

THE SPARKS (the second consumer, :class:`SparkSystem`; graphics G4 P3,
``docs/explosion_look_design_2026-11-07.md`` §3.4). Thrown by a detonation:
each new ``"blast"`` effect of the renderer's queue is consumed once
(``"sparks_seen"``) and spawns ``rate * sqrt(strength)`` sparks on the CPU at
the centres of the blast walk's seeds (``blast_expansion.walk_seeds``, handed
in by the renderer: the centre when open, else beside it), uniform in
direction, radially fast, at ``t_born_k``, from a generator seeded by the
blast's ``(tick0, pos)``. From then on they are particles like the embers:
the same :func:`integrate` (drag toward the frame's tamed wind x tps plus the
same curl-noise field; Newton cooling; dead in a wall), the same
:func:`pack_sprites` and the same :class:`ParticleDraw` -- the two systems'
sprites concatenated into ONE draw per stage. ``[render.sparks]`` holds
every dial, behind a refusing door.

THE DEBRIS (the third consumer, :class:`DebrisSystem`; graphics G4 P4,
design §3.4, §6 S5 / N6, §8 P4). Each new ``"blast"`` effect is consumed once
(``"debris_seen"``) and throws ``rate * strength`` dark CHIPS from the same
walk seeds, radially, a hashed rotation and size each (THE fire's hash,
``fire_vfx_ref._hash01``). They are things on the FLOOR, so they have their
own twin stage, :func:`integrate_debris` (Erik's ruling at the screen,
2026-11-07): drag toward the frame's tamed wind x tps SCALED BY THE TILE'S
GAS DENSITY (bulk N over the ambient, the prop sway's read: in vacuum the air
term is zero), then a density-independent GROUND FRICTION (a constant
deceleration to rest: a draught never moves a lying chip, a blast's outflow
carries it); a chip that would enter a wall STOPS at its last open position
and lives on; it fades over its last ``fade_s`` and dies at its life. A dark
chip cannot be drawn additively, so :class:`ParticleDraw` has a second
stage, ``"debris"``: premultiplied alpha, lit by the light field through the
lit-3D block (``lit3d._FIELD_SAMPLE_GLSL``), under the flame and the units.
The ember / spark :func:`integrate` and :func:`pack_sprites` are untouched.

THE MUZZLE SPARKS (the fourth consumer, :class:`MuzzleSparkSystem`; #87,
``docs/muzzle_flame_design_2026-11-12.md`` §4). Thrown by a shot: the muzzle
flame (``renderer/muzzle_flame.py``) queues each new flash's sparks at the
barrel's tip, in a forward cone; from then on they are blast sparks in every
respect (the same :func:`integrate`, :func:`pack_sprites` and draw).
``[render.muzzle_flame.sparks]`` holds their dials, behind a refusing door.
A second instance throws the BULLET IMPACTS' sparks back off a wall (#141,
``renderer/tracers.py``; ``[render.tracers.sparks]``, the same dials). The
tracers themselves are sprites of :class:`ParticleDraw` (their own instance,
drawn after the units), packed by ``tracers.Tracers.pack``.

THE SPRAY (the fifth consumer, :class:`SpraySystem`; water look arc #97
W3, ``docs/water_look_design_2026-11-19.md``). Each new ``"blast"`` effect
is consumed once (``"spray_seen"``) and throws ``rate x strength x`` the WET
share of its walk's seeds of water droplets from those wet seeds -- a blast
in a dry room throws none. A droplet's flight is ANALYTIC (critique minor:
no integration): a straight ground track at its launch speed, clipped at the
last open point before a wall, and a rise and fall ``z = g age (life - age)
/ 2`` over ``life = 2 up / g`` (:func:`spray_height`), on the SIM clock; when
it lands it is gone and its landing point is handed to the water's foam as
ONE dab (``WaterPass.add_dabs``, a per-frame source list -- the flame's
pack pattern, never a GPU readback). A round that stops at a wall beside the
water throws a few back off it (:meth:`SpraySystem.splash`, at the tracers'
impact). Drawn in the premultiplied lit ``"debris"`` stage
(:func:`pack_spray`, their own albedo). ``[render.spray]`` holds every dial.

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
        # a chip (DebrisSystem) hands its own `temp` column (its hashed
        # size); every glowing particle starts at its birth temperature
        new[:, TEMP] = (np.asarray(arrs["temp"], dtype=np.float64)[:k]
                        if "temp" in arrs else new[:, T0])
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
# The sparks (the second consumer) -- [render.sparks]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SparkSettings:
    """Every ``[render.sparks]`` dial (graphics G4 P3; design
    ``docs/explosion_look_design_2026-11-07.md`` §3.4 + §6). The config is
    the live value and every key is required
    (:func:`spark_settings_from_config` refuses). ``drag``, ``curl``,
    ``cool_s``, ``size_tiles``, ``min_screen_px``, ``streak_s`` and
    ``brightness`` mean what the embers' do: they are the names
    :func:`integrate` and :func:`pack_sprites` read."""
    enabled: bool
    rate: float              # sparks per detonation at strength 1 (x sqrt(strength))
    v0_tiles_s: float        # radial birth speed, tiles/s
    v_spread: float          # the speed is v0 * (1 +- v_spread * U(-1, 1))
    t_born_k: float          # birth temperature, K (they only cool)
    cool_s: float            # e-fold of (T - T_amb), Newton
    life_s: float            # mean lifetime, s (VFX clock)
    life_jitter: float       # each lives life_s * (1 +- life_jitter * U(-1, 1))
    drag: float              # 1/s: how fast the air catches them
    curl: float              # tiles/s of curl-noise drift (the embers' field)
    size_tiles: float        # sprite radius, tiles
    min_screen_px: float     # zoomed out, the radius floor in SCREEN px
    streak_s: float          # motion stretch: the trail is |v| x this long
    brightness: float        # x the ramp's Le(T)
    cap: int                 # never more alive at once
    seed: int                # mixed with each blast's (tick0, pos)


_SPARK_KINDS = {f.name: f.type for f in fields(SparkSettings)}


def spark_config_keys() -> Tuple[str, ...]:
    """Every key ``[render.sparks]`` must carry."""
    return tuple(_SPARK_KINDS)


def spark_settings_from_config(cfg) -> SparkSettings:
    """``[render.sparks]`` -> settings, read every frame. A REFUSING door
    (the shock front's): a missing section, a missing key or a mistyped
    value raises ValueError naming it -- a dial that fell back to a default
    would hide a typo while tuning."""
    sec = getattr(getattr(cfg, "render", None), "sparks", None)
    if sec is None:
        raise ValueError("config.toml has no [render.sparks] section")
    vals = {}
    for name, kind in _SPARK_KINDS.items():
        if not hasattr(sec, name):
            raise ValueError(f"[render.sparks] is missing {name!r}")
        v = getattr(sec, name)
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"[render.sparks] {name} has the wrong type: "
                             f"{v!r}")
        vals[name] = v
    s = SparkSettings(**vals)
    for name in ("life_s", "cool_s", "size_tiles"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[render.sparks] {name} must be > 0")
    for name in ("rate", "v0_tiles_s", "drag", "brightness", "streak_s",
                 "min_screen_px", "curl"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[render.sparks] {name} must be >= 0")
    for name in ("v_spread", "life_jitter"):
        if not 0.0 <= getattr(s, name) < 1.0:
            raise ValueError(f"[render.sparks] {name} must be in [0, 1)")
    if s.cap < 0:
        raise ValueError("[render.sparks] cap must be >= 0")
    return s


def spark_count(strength: float, rate: float) -> int:
    """How many sparks a detonation of row ``strength`` throws:
    ``round(rate * sqrt(strength))`` (design §3.4) -- 0 for a row without a
    strength."""
    return int(round(max(float(rate), 0.0)
                     * math.sqrt(max(float(strength), 0.0))))


def spark_rng(seed: int, tick0: int, pos) -> np.random.Generator:
    """THE spark spawn's RNG: seeded from the dial's ``seed``, the blast's
    ``tick0`` and its tile -- so the same effect spawns the same sparks
    (a look shot reproduces) and two blasts never share a stream."""
    key = [int(seed) & 0xFFFFFFFF, int(tick0) & 0xFFFFFFFF,
           int(pos[0]) & 0xFFFFFFFF, int(pos[1]) & 0xFFFFFFFF]
    return np.random.default_rng(key)


def spark_spawns(n: int, seeds, settings: SparkSettings,
                 rng: np.random.Generator) -> dict:
    """``n`` sparks of one detonation: born at the CENTRES (+0.5, N2) of the
    blast walk's ``seeds`` ((y, x) tiles, ``blast_expansion.walk_seeds``:
    the centre when it is open, else beside it -- never inside a wall), one
    seed each uniformly, a uniform direction, radial speed ``v0 * (1 +-
    v_spread * u)``, at ``t_born_k``, living ``life_s * (1 +- life_jitter *
    u)``. Returns arrays x, y, vx, vy, t0, life (tiles, tiles/s, K, s)."""
    seeds = list(seeds or ())
    if n <= 0 or not seeds:
        return {k: np.zeros(0) for k in _SPAWN_KEYS}
    # every uniform in ONE draw (rows: seed, angle, speed, life) -- the
    # order the seeded stream is consumed in
    u = rng.random((4, int(n)))
    sy = np.array([s_[0] for s_ in seeds], np.float64)
    sx = np.array([s_[1] for s_ in seeds], np.float64)
    idx = np.minimum((u[0] * len(seeds)).astype(np.int64), len(seeds) - 1)
    ang = 2.0 * np.pi * u[1]
    speed = settings.v0_tiles_s * (1.0 + settings.v_spread
                                   * (2.0 * u[2] - 1.0))
    life = settings.life_s * (1.0 + settings.life_jitter
                              * (2.0 * u[3] - 1.0))
    return dict(x=sx[idx] + 0.5, y=sy[idx] + 0.5,
                vx=np.cos(ang) * speed, vy=np.sin(ang) * speed,
                t0=np.full(int(n), float(settings.t_born_k)), life=life)


SPARK_MAX_HOP = 0.5   # tiles: one integration substep never moves a spark further


class SparkSystem:
    """The sparks' CPU side (graphics G4 P3, the particle primitive's
    SECOND consumer): per frame, :meth:`update` consumes every new
    ``"blast"`` effect of the renderer's queue ONCE (marking it
    ``"sparks_seen"``; never another consumer's flag), spawns its
    ``rate * sqrt(strength)`` sparks at the blast walk's seeds, and steps
    the alive ones through THE motion law (:func:`integrate`: drag toward
    the frame's one tamed wind x tps plus the curl noise; Newton cooling in
    closed form; dead past their lifetime, below the glow floor, off the map
    or inside a solid / vacuum tile) on the VFX clock. Pure numpy (no GL, no
    sim import: the walk's seeds come in through ``seeds_fn``)."""

    def __init__(self, settings: SparkSettings, *, t_amb_k: float,
                 glow_min_k: float):
        self.settings = settings
        self.t_amb_k = float(t_amb_k)
        self.glow_min_k = float(glow_min_k)
        self.p = Particles(settings.cap)
        # a look shot's `hide = ["sparks"]` (tools; never a key): None = the
        # config's `enabled`
        self.enabled_override: Optional[bool] = None
        self._t_prev: Optional[float] = None
        self.spawned_last_frame = 0
        self.last_update_ms = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def refresh_settings(self, settings: SparkSettings) -> None:
        if settings.cap != self.p.cap:
            self.p.set_cap(settings.cap)
        self.settings = settings
        if not self.enabled:
            self.p.clear()

    def consume(self, effects, seeds_fn: Callable, sim_tick: int) -> dict:
        """Mark every new ``"blast"`` effect ``"sparks_seen"`` and return the
        sparks it spawns (concatenated; none while disabled). ``seeds_fn(fy,
        fx, approach)`` is the blast walk's seed rule over the map's open
        mask, asked only for a blast that throws a spark."""
        s = self.settings
        out = []
        for fx in effects:
            if fx.get("kind") != "blast" or fx.get("sparks_seen"):
                continue
            fx["sparks_seen"] = True
            if not self.enabled:
                continue
            n = spark_count(fx.get("strength", 0.0), s.rate)
            if n == 0:
                continue
            pos = fx["pos"]
            seeds = seeds_fn(int(pos[1]), int(pos[0]), fx.get("approach"))
            rng = spark_rng(s.seed, fx.get("tick0", sim_tick), pos)
            out.append(spark_spawns(n, seeds, s, rng))
        if not out:
            return {k: np.zeros(0) for k in _SPAWN_KEYS}
        return {k: np.concatenate([o[k] for o in out]) for k in _SPAWN_KEYS}

    def update(self, *, t: float, effects, seeds_fn: Callable,
               sim_tick: int, wind_fn: Callable, tps: float,
               blocked: Optional[np.ndarray], modes) -> None:
        """One frame at VFX noise time ``t`` (monotone): integrate the alive
        sparks over the elapsed VFX time (in substeps of at most
        :data:`SPARK_MAX_HOP` tiles, so a fast spark never steps over a
        wall), then add the new blasts' sparks, born at ``t`` with age 0.
        ``modes`` is the embers' curl-noise field (one air, one drift)."""
        t0w = time.perf_counter()
        self.spawned_last_frame = 0
        dt = 0.0 if self._t_prev is None else max(float(t) - self._t_prev, 0.0)
        self._t_prev = float(t)
        dt = min(dt, MAX_STEP_S)
        s = self.settings
        if not self.enabled:
            self.consume(effects, seeds_fn, sim_tick)     # seen, never thrown
            self.last_update_ms = (time.perf_counter() - t0w) * 1000.0
            return
        if dt > 0.0 and self.p.n:
            def air(iy, ix):
                # index first, convert the few rows (never the map)
                return np.asarray(wind_fn())[iy, ix].astype(
                    np.float64) * float(tps)

            v = float(np.max(np.hypot(self.p.vx, self.p.vy)))
            k = max(1, int(math.ceil(v * dt / SPARK_MAX_HOP)))
            h = dt / k
            for i in range(k):
                if self.p.n == 0:
                    break
                integrate(self.p, h, float(t) - dt + (i + 1) * h, air_fn=air,
                          blocked=blocked, settings=s, t_amb_k=self.t_amb_k,
                          glow_min_k=self.glow_min_k, modes=modes)
        new = self.consume(effects, seeds_fn, sim_tick)
        if new["x"].size:
            room = max(self.p.cap - self.p.n, 0)
            if new["x"].size > room:              # a fair subset, not the first
                rng = np.random.default_rng([int(s.seed) & 0xFFFFFFFF,
                                             int(sim_tick) & 0xFFFFFFFF])
                keep = np.sort(rng.permutation(new["x"].size)[:room])
                new = {k_: v_[keep] for k_, v_ in new.items()}
            self.spawned_last_frame = self.p.add(**new)
        self.last_update_ms = (time.perf_counter() - t0w) * 1000.0


# ---------------------------------------------------------------------------
# The muzzle sparks (the fourth consumer) -- [render.muzzle_flame.sparks]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MuzzleSparkSettings:
    """Every ``[render.muzzle_flame.sparks]`` dial (#87,
    ``docs/muzzle_flame_design_2026-11-12.md`` §4). ``drag``, ``curl``,
    ``cool_s``, ``size_tiles``, ``min_screen_px``, ``streak_s`` and
    ``brightness`` mean what the embers' and the blast sparks' do: they are
    the names :func:`integrate` and :func:`pack_sprites` read. How MANY a
    shot throws is its flame kind's ``sparks``."""
    enabled: bool
    cone_deg: float          # the cone's half-angle round the barrel (a normal's sigma)
    v0_mps: float            # birth speed, METRES per second
    v_spread: float          # the speed is v0 * (1 +- v_spread * U(-1, 1))
    t_born_k: float
    cool_s: float
    life_s: float
    life_jitter: float
    drag: float
    curl: float
    size_tiles: float
    min_screen_px: float
    streak_s: float
    brightness: float
    cap: int
    seed: int                # the cap's fair-subset draw (SparkSystem.update)


_MUZZLE_SPARK_KINDS = {f.name: f.type for f in fields(MuzzleSparkSettings)}


def muzzle_spark_settings(cfg, table: str = "muzzle_flame"
                          ) -> MuzzleSparkSettings:
    """``[render.<table>.sparks]`` -> settings, read every frame: the muzzle
    sparks' (``table`` "muzzle_flame") or the bullet impacts' ("tracers",
    #141 -- the same dials, thrown back off a wall). A REFUSING door
    (``muzzle_flame.typed_row``: a missing, unknown or mistyped key raises
    ValueError naming it)."""
    from collections.abc import Mapping
    from .muzzle_flame import _as_dict, typed_row
    if isinstance(cfg, Mapping):
        sec = cfg.get("render", {}).get(table, {}).get("sparks")
    else:
        sec = getattr(getattr(getattr(cfg, "render", None), table,
                              None), "sparks", None)
    w = f"[render.{table}.sparks]"
    if sec is None:
        raise ValueError(f"config.toml has no {w} section")
    s = MuzzleSparkSettings(**typed_row(_as_dict(sec), _MUZZLE_SPARK_KINDS, w))
    for name in ("life_s", "cool_s", "size_tiles"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"{w} {name} must be > 0")
    for name in ("cone_deg", "v0_mps", "drag", "brightness", "streak_s",
                 "min_screen_px", "curl"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"{w} {name} must be >= 0")
    for name in ("v_spread", "life_jitter"):
        if not 0.0 <= getattr(s, name) < 1.0:
            raise ValueError(f"{w} {name} must be in [0, 1)")
    if s.cap < 0:
        raise ValueError(f"{w} cap must be >= 0")
    return s


def muzzle_spark_spawns(n: int, pose, tile_size_m: float,
                        settings: MuzzleSparkSettings,
                        rng: np.random.Generator) -> dict:
    """``n`` sparks of one shot: born AT the muzzle (``pose`` = (x, y,
    dir x, dir y), tiles), thrown round the barrel's direction by a normal
    angle of sigma ``cone_deg``, at ``v0_mps`` (converted once to tiles/s)
    ``* (1 +- v_spread * u)``, at ``t_born_k``, living ``life_s * (1 +-
    life_jitter * u)``. Returns the spawn arrays (tiles, tiles/s, K, s)."""
    if n <= 0:
        return {k: np.zeros(0) for k in _SPAWN_KEYS}
    x, y, dx, dy = (float(v) for v in pose)
    ang = (math.atan2(dy, dx)
           + rng.normal(0.0, math.radians(settings.cone_deg), int(n)))
    u = rng.random((2, int(n)))
    speed = (settings.v0_mps / max(float(tile_size_m), 1e-6)
             * (1.0 + settings.v_spread * (2.0 * u[0] - 1.0)))
    life = settings.life_s * (1.0 + settings.life_jitter * (2.0 * u[1] - 1.0))
    return dict(x=np.full(int(n), x), y=np.full(int(n), y),
                vx=np.cos(ang) * speed, vy=np.sin(ang) * speed,
                t0=np.full(int(n), float(settings.t_born_k)), life=life)


class MuzzleSparkSystem(SparkSystem):
    """The muzzle sparks' CPU side (#87, the particle primitive's FOURTH
    consumer): a :class:`SparkSystem` whose spawns are QUEUED by the muzzle
    flame (:meth:`throw`, one call per new flash) instead of read from a
    ``"blast"`` effect. Everything else is the blast sparks': the same
    :meth:`update` (THE motion law in substeps, the cap's fair subset), the
    same :func:`pack_sprites`, the same draw."""

    def __init__(self, settings: MuzzleSparkSettings, *, t_amb_k: float,
                 glow_min_k: float):
        super().__init__(settings, t_amb_k=t_amb_k, glow_min_k=glow_min_k)
        self._queue = []

    def throw(self, n: int, pose, tile_size_m: float,
              rng: np.random.Generator) -> None:
        """Queue ``n`` sparks from the muzzle at ``pose``; they are born at
        the next :meth:`update`. Nothing while disabled."""
        if self.enabled and n > 0:
            self._queue.append(muzzle_spark_spawns(n, pose, tile_size_m,
                                                   self.settings, rng))

    def consume(self, effects, seeds_fn, sim_tick: int) -> dict:
        out, self._queue = self._queue, []
        if not out or not self.enabled:
            return {k: np.zeros(0) for k in _SPAWN_KEYS}
        return {k: np.concatenate([o[k] for o in out]) for k in _SPAWN_KEYS}


# ---------------------------------------------------------------------------
# The debris (the third consumer) -- [render.debris]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DebrisSettings:
    """Every ``[render.debris]`` dial (graphics G4 P4; design
    ``docs/explosion_look_design_2026-11-07.md`` §3.4, §6 S5 / N6, §8 P4).
    The config is the live value and every key is required
    (:func:`debris_settings_from_config` refuses)."""
    enabled: bool
    rate: float              # chips per detonation at strength 1 (x strength)
    v0_tiles_s: float        # radial birth speed, tiles/s
    v_spread: float          # the speed is v0 * (1 +- v_spread * U(-1, 1))
    drag: float              # 1/s at 1 atm: the air term's rate (x density)
    friction_tiles_s2: float  # ground friction: a constant deceleration to rest
    life_s: float            # mean lifetime, s (VFX clock)
    life_jitter: float       # each lives life_s * (1 +- life_jitter * U(-1, 1))
    size_tiles: float        # a chip's half-width, tiles
    size_jitter: float       # each is size_tiles * (1 +- size_jitter * hash)
    albedo: tuple            # linear RGB, the chip's colour under white light
    alpha: float             # its opacity (premultiplied in the draw)
    fade_s: float            # the opacity fades to 0 over the last fade_s
    seed: int                # mixed with each blast's (tick0, pos)
    cap: int                 # never more alive at once


_DEBRIS_KINDS = {f.name: f.type for f in fields(DebrisSettings)}

# The density factor's ceiling: an overpressured tile drags a chip at most
# this many times as hard as one at 1 atm (the prop sway's shipped
# ``density_max``; Erik's momentum-flux read, static_props.apply_gas_density).
DEBRIS_DENSITY_MAX = 2.0
DEBRIS_MAX_HOP = 0.5   # tiles: one integration substep never moves a chip further


def debris_config_keys() -> Tuple[str, ...]:
    """Every key ``[render.debris]`` must carry."""
    return tuple(_DEBRIS_KINDS)


def debris_settings_from_config(cfg) -> DebrisSettings:
    """``[render.debris]`` -> settings, read every frame. A REFUSING door
    (the sparks'): a missing section, a missing key or a mistyped value
    raises ValueError naming it."""
    sec = getattr(getattr(cfg, "render", None), "debris", None)
    if sec is None:
        raise ValueError("config.toml has no [render.debris] section")
    vals = {}
    for name, kind in _DEBRIS_KINDS.items():
        if not hasattr(sec, name):
            raise ValueError(f"[render.debris] is missing {name!r}")
        v = getattr(sec, name)
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif kind in ("tuple", tuple):
            ok = (isinstance(v, (list, tuple)) and len(v) == 3
                  and all(isinstance(c, (int, float))
                          and not isinstance(c, bool) for c in v))
            v = tuple(float(c) for c in v) if ok else v
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"[render.debris] {name} has the wrong type: "
                             f"{v!r}")
        vals[name] = v
    s = DebrisSettings(**vals)
    for name in ("life_s", "size_tiles", "fade_s"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[render.debris] {name} must be > 0")
    for name in ("rate", "v0_tiles_s", "drag", "friction_tiles_s2"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[render.debris] {name} must be >= 0")
    for name in ("v_spread", "life_jitter", "size_jitter"):
        if not 0.0 <= getattr(s, name) < 1.0:
            raise ValueError(f"[render.debris] {name} must be in [0, 1)")
    if not 0.0 <= s.alpha <= 1.0:
        raise ValueError("[render.debris] alpha must be in [0, 1]")
    if not all(0.0 <= c <= 1.0 for c in s.albedo):
        raise ValueError("[render.debris] albedo must be in [0, 1]")
    if s.cap < 0:
        raise ValueError("[render.debris] cap must be >= 0")
    return s


def debris_count(strength: float, rate: float) -> int:
    """How many chips a detonation of row ``strength`` throws:
    ``round(rate * strength)`` (design §3.4: linear in strength) -- 0 for a
    row without a strength."""
    return int(round(max(float(rate), 0.0) * max(float(strength), 0.0)))


_M64 = (1 << 64) - 1


def debris_keys(n: int, tick0: int, pos) -> np.ndarray:
    """The per-chip keys of THE fire's hash (``fire_vfx_ref._hash01``, the
    vectorised twin of ``fire_vfx._hash01``): one blast's ``(tick0, x, y)``
    folded into a base, plus the chip's index -- so a chip's rotation and
    size are fixed for life and reproduce."""
    base = (((int(tick0) & 0xFFFFFF) << 40) ^ ((int(pos[1]) & 0xFFFFF) << 20)
            ^ (int(pos[0]) & 0xFFFFF))
    with np.errstate(over="ignore"):
        return (np.uint64((base * 4099) & _M64)
                + np.arange(int(n), dtype=np.uint64))


def debris_spawns(n: int, seeds, settings: DebrisSettings,
                  rng: np.random.Generator, keys: np.ndarray) -> dict:
    """``n`` chips of one detonation: born at the CENTRES (+0.5) of the blast
    walk's ``seeds`` (never inside a wall), one seed each uniformly, a
    uniform direction, radial speed ``v0 * (1 +- v_spread * u)``, living
    ``life_s * (1 +- life_jitter * u)``. A chip carries no temperature: its
    ``t0`` column holds its ROTATION (rad) and its ``temp`` column its SIZE
    factor, both from THE fire's hash of ``keys`` (never the RNG, never a
    second hash). Returns arrays x, y, vx, vy, t0, temp, life."""
    seeds = list(seeds or ())
    if n <= 0 or not seeds:
        return {k: np.zeros(0) for k in _SPAWN_KEYS + ("temp",)}
    # every uniform in ONE draw (rows: seed, angle, speed, life)
    u = rng.random((4, int(n)))
    sy = np.array([s_[0] for s_ in seeds], np.float64)
    sx = np.array([s_[1] for s_ in seeds], np.float64)
    idx = np.minimum((u[0] * len(seeds)).astype(np.int64), len(seeds) - 1)
    ang = 2.0 * np.pi * u[1]
    speed = settings.v0_tiles_s * (1.0 + settings.v_spread
                                   * (2.0 * u[2] - 1.0))
    life = settings.life_s * (1.0 + settings.life_jitter
                              * (2.0 * u[3] - 1.0))
    salt = int(settings.seed) * 2
    rot = 2.0 * np.pi * ref._hash01(keys, salt + 1)
    size = 1.0 + settings.size_jitter * (2.0 * ref._hash01(keys, salt + 2)
                                         - 1.0)
    return dict(x=sx[idx] + 0.5, y=sy[idx] + 0.5,
                vx=np.cos(ang) * speed, vy=np.sin(ang) * speed,
                t0=rot, temp=size, life=life)


def integrate_debris(p: Particles, dt: float, *, air_fn: Callable,
                     density_fn: Callable, blocked: np.ndarray,
                     settings: DebrisSettings) -> None:
    """THE chips' motion law (Erik's ruling at the screen, 2026-11-07), the
    twin stage beside :func:`integrate`: advance every chip by ``dt`` s of
    the VFX clock.

    1. Drag toward the local air's motion (``air_fn(iy, ix)`` -> (N, 2)
       tiles/s: the frame's ONE tamed wind x tps), at the rate ``drag x
       rho``, ``rho`` = ``density_fn(iy, ix)`` (the tile's bulk gas N over
       the ambient, the prop sway's read) clamped to [0,
       :data:`DEBRIS_DENSITY_MAX`]: in vacuum the air term is ZERO.
    2. Ground friction, independent of the air: the speed drops by
       ``friction_tiles_s2 x dt`` toward zero (never reverses) -- so a
       draught whose pull ``drag x rho x |air|`` is below the friction never
       moves a lying chip, and a blast's outflow does.
    3. Move; a chip that would enter a ``blocked`` tile (or leave the map)
       STOPS at its last open position, velocity zero, alive (a chip is a
       thing on the floor). A chip whose own tile has become blocked (a
       sealed wall) dies.
    4. Age; dead at its lifetime."""
    if p.n == 0 or dt <= 0.0:
        return
    iy, ix, inside = _tiles_of(p.d[:, X:Y + 1], blocked.shape)
    ok = inside & ~blocked[iy, ix]
    if not ok.all():
        p.keep(ok)
        if p.n == 0:
            return
        iy, ix = iy[ok], ix[ok]
    d = p.d
    air = np.asarray(air_fn(iy, ix), dtype=np.float64)
    rho = np.minimum(np.maximum(np.asarray(density_fn(iy, ix),
                                           dtype=np.float64), 0.0),
                     DEBRIS_DENSITY_MAX)
    a = 1.0 - np.exp(-max(float(settings.drag), 0.0) * rho * dt)
    v = d[:, VX:VY + 1]
    v += (air - v) * a[:, None]
    sp = np.hypot(v[:, 0], v[:, 1])
    f = max(float(settings.friction_tiles_s2), 0.0) * dt
    v *= np.where(sp > f, (sp - f) / np.maximum(sp, 1e-30), 0.0)[:, None]
    old = d[:, X:Y + 1].copy()
    d[:, X:Y + 1] += v * dt
    d[:, AGE] += dt
    iy2, ix2, inside2 = _tiles_of(d[:, X:Y + 1], blocked.shape)
    hit = ~inside2 | blocked[iy2, ix2]
    if hit.any():
        d[hit, X:Y + 1] = old[hit]
        d[hit, VX:VY + 1] = 0.0
    alive = d[:, AGE] < d[:, LIFE]
    if not alive.all():
        p.keep(alive)


def debris_weight(age, life, alpha: float, fade_s: float) -> np.ndarray:
    """A chip's opacity: ``alpha`` until its last ``fade_s`` seconds, then
    linearly to 0 at its life -- monotone non-increasing in age."""
    rest = np.asarray(life, np.float64) - np.asarray(age, np.float64)
    return float(alpha) * np.minimum(np.maximum(rest / max(float(fade_s),
                                                           1e-9), 0.0), 1.0)


def pack_debris(p: Particles, view, px_per_tile: float,
                settings: DebrisSettings) -> np.ndarray:
    """The chips in ``view`` ((x0, y0, x1, y1) tiles, None = all), one tile
    of margin -> ``(2, k, 4)`` float32, the debris stage's data texture:
    row 0 = (x, y world px, rotation rad, half-width px), row 1 = (opacity,
    0, 0, 0). k = 0 draws nothing. The ember / spark :func:`pack_sprites` is
    a separate function and untouched."""
    d = p.d
    if view is not None and p.n:
        vx0, vy0, vx1, vy1 = view
        x, y = d[:, X], d[:, Y]
        d = d[(x >= vx0 - 1.0) & (x <= vx1 + 1.0)
              & (y >= vy0 - 1.0) & (y <= vy1 + 1.0)]
    k = int(d.shape[0])
    out = np.zeros((2, k, 4), np.float32)
    if k == 0:
        return out
    ppt = float(px_per_tile)
    out[0, :, 0] = d[:, X] * ppt
    out[0, :, 1] = d[:, Y] * ppt
    out[0, :, 2] = d[:, T0]                                  # its rotation
    out[0, :, 3] = settings.size_tiles * d[:, TEMP] * ppt    # x its size
    out[1, :, 0] = debris_weight(d[:, AGE], d[:, LIFE], settings.alpha,
                                 settings.fade_s)
    return out


class DebrisSystem:
    """The chips' CPU side (graphics G4 P4, the particle primitive's THIRD
    consumer): per frame, :meth:`update` consumes every new ``"blast"``
    effect ONCE (``"debris_seen"``; never another consumer's flag), throws
    its ``rate * strength`` chips at the blast walk's seeds, and steps the
    alive ones through :func:`integrate_debris` on the VFX clock. Pure numpy
    (no GL, no sim import: the seeds, the wind and the density come in as
    callables)."""

    def __init__(self, settings: DebrisSettings):
        self.settings = settings
        self.p = Particles(settings.cap)
        # a look shot's `hide = ["debris"]` (tools; never a key): None = the
        # config's `enabled`
        self.enabled_override: Optional[bool] = None
        self._t_prev: Optional[float] = None
        self.spawned_last_frame = 0
        self.last_update_ms = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def refresh_settings(self, settings: DebrisSettings) -> None:
        if settings.cap != self.p.cap:
            self.p.set_cap(settings.cap)
        self.settings = settings
        if not self.enabled:
            self.p.clear()

    def consume(self, effects, seeds_fn: Callable, sim_tick: int) -> dict:
        """Mark every new ``"blast"`` effect ``"debris_seen"`` and return the
        chips it throws (concatenated; none while disabled). ``seeds_fn(fy,
        fx, approach)`` is the blast walk's seed rule over the open mask."""
        s = self.settings
        keys_all = _SPAWN_KEYS + ("temp",)
        out = []
        for fx in effects:
            if fx.get("kind") != "blast" or fx.get("debris_seen"):
                continue
            fx["debris_seen"] = True
            if not self.enabled:
                continue
            n = debris_count(fx.get("strength", 0.0), s.rate)
            if n == 0:
                continue
            pos = fx["pos"]
            tick0 = fx.get("tick0", sim_tick)
            seeds = seeds_fn(int(pos[1]), int(pos[0]), fx.get("approach"))
            # THE blast RNG key (seed, tick0, x, y), the sparks' rule, with
            # the debris' own seed: two consumers never share a stream
            rng = spark_rng(s.seed, tick0, pos)
            out.append(debris_spawns(n, seeds, s, rng,
                                     debris_keys(n, tick0, pos)))
        if not out:
            return {k: np.zeros(0) for k in keys_all}
        return {k: np.concatenate([o[k] for o in out]) for k in keys_all}

    def update(self, *, t: float, effects, seeds_fn: Callable,
               sim_tick: int, wind_fn: Callable, tps: float,
               density_fn: Callable, blocked: np.ndarray) -> None:
        """One frame at VFX noise time ``t`` (monotone): step the alive
        chips over the elapsed VFX time (substeps of at most
        :data:`DEBRIS_MAX_HOP` tiles, so no chip steps over a wall), then
        add the new blasts' chips, born at ``t`` with age 0. ``wind_fn``
        returns the frame's ONE tamed wind (tiles per tick); ``density_fn(iy,
        ix)`` the tiles' bulk N over the ambient."""
        t0w = time.perf_counter()
        self.spawned_last_frame = 0
        dt = 0.0 if self._t_prev is None else max(float(t) - self._t_prev, 0.0)
        self._t_prev = float(t)
        dt = min(dt, MAX_STEP_S)
        s = self.settings
        if not self.enabled:
            self.consume(effects, seeds_fn, sim_tick)     # seen, never thrown
            self.last_update_ms = (time.perf_counter() - t0w) * 1000.0
            return
        if dt > 0.0 and self.p.n:
            def air(iy, ix):
                # index first, convert the few rows (never the map)
                return np.asarray(wind_fn())[iy, ix].astype(
                    np.float64) * float(tps)

            iy, ix, _ = _tiles_of(self.p.d[:, X:Y + 1], blocked.shape)
            a0 = air(iy, ix)
            v = max(float(np.max(np.hypot(self.p.vx, self.p.vy))),
                    float(np.max(np.hypot(a0[:, 0], a0[:, 1]))))
            k = max(1, int(math.ceil(v * dt / DEBRIS_MAX_HOP)))
            h = dt / k
            for _ in range(k):
                if self.p.n == 0:
                    break
                integrate_debris(self.p, h, air_fn=air, density_fn=density_fn,
                                 blocked=blocked, settings=s)
        new = self.consume(effects, seeds_fn, sim_tick)
        if new["x"].size:
            room = max(self.p.cap - self.p.n, 0)
            if new["x"].size > room:              # a fair subset, not the first
                rng = np.random.default_rng([int(s.seed) & 0xFFFFFFFF,
                                             int(sim_tick) & 0xFFFFFFFF])
                keep = np.sort(rng.permutation(new["x"].size)[:room])
                new = {k_: v_[keep] for k_, v_ in new.items()}
            self.spawned_last_frame = self.p.add(**new)
        self.last_update_ms = (time.perf_counter() - t0w) * 1000.0


# ---------------------------------------------------------------------------
# The spray (the fifth consumer) -- [render.spray]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpraySettings:
    """Every ``[render.spray]`` dial (water look arc #97 W3; design
    ``docs/water_look_design_2026-11-19.md`` W3 + critique minors). The
    config is the live value and every key is required
    (:func:`spray_settings_from_config` refuses)."""
    enabled: bool
    rate: float              # droplets per detonation at strength 1, x the wet share of its seeds
    speed_mps: float         # horizontal launch speed, m/s
    up_mps: float            # vertical launch speed, m/s (the flight is 2 up / g)
    spread: float            # each speed x (1 +- spread x U(-1, 1))
    size_tiles: float        # a droplet's half-width, tiles
    albedo: tuple            # linear RGB under white light
    alpha: float             # its opacity (premultiplied in the draw)
    seed: int                # mixed with each blast's (tick0, pos)
    cap: int                 # never more in the air at once
    splash_drops: int        # droplets a round stopping at a wet wall throws
    splash_speed_mps: float  # their horizontal speed, m/s
    splash_up_mps: float     # their vertical speed, m/s


_SPRAY_KINDS = {f.name: f.type for f in fields(SpraySettings)}
G_MPS2 = 9.81            # the droplets' gravity
SPRAY_PATH_SAMPLES = 16  # the wall test along a droplet's ground track
SPLASH_HALF_ANGLE = math.radians(60.0)   # a splash leaves within this of the
                                         # flight reversed


def spray_config_keys() -> Tuple[str, ...]:
    """Every key ``[render.spray]`` must carry."""
    return tuple(_SPRAY_KINDS)


def spray_settings_from_config(cfg) -> SpraySettings:
    """``[render.spray]`` -> settings, read every frame. A REFUSING door
    (the debris'): a missing section, a missing key or a mistyped value
    raises ValueError naming it."""
    sec = getattr(getattr(cfg, "render", None), "spray", None)
    if sec is None:
        raise ValueError("config.toml has no [render.spray] section")
    vals = {}
    for name, kind in _SPRAY_KINDS.items():
        if not hasattr(sec, name):
            raise ValueError(f"[render.spray] is missing {name!r}")
        v = getattr(sec, name)
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif kind in ("tuple", tuple):
            ok = (isinstance(v, (list, tuple)) and len(v) == 3
                  and all(isinstance(c, (int, float))
                          and not isinstance(c, bool) for c in v))
            v = tuple(float(c) for c in v) if ok else v
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"[render.spray] {name} has the wrong type: "
                             f"{v!r}")
        vals[name] = v
    s = SpraySettings(**vals)
    for name in ("size_tiles", "up_mps", "splash_up_mps"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[render.spray] {name} must be > 0")
    for name in ("rate", "speed_mps", "splash_speed_mps"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[render.spray] {name} must be >= 0")
    if not 0.0 <= s.spread < 1.0:
        raise ValueError("[render.spray] spread must be in [0, 1)")
    if not 0.0 <= s.alpha <= 1.0:
        raise ValueError("[render.spray] alpha must be in [0, 1]")
    if not all(0.0 <= c <= 1.0 for c in s.albedo):
        raise ValueError("[render.spray] albedo must be in [0, 1]")
    if s.cap < 0 or s.splash_drops < 0:
        raise ValueError("[render.spray] cap and splash_drops must be >= 0")
    return s


def spray_count(strength: float, rate: float, wet_share: float) -> int:
    """How many droplets a detonation throws: ``round(rate x strength x
    wet_share)`` -- the share of its walk's seeds that stand in water, so
    a blast in a dry room throws none."""
    return int(round(max(float(rate), 0.0) * max(float(strength), 0.0)
                     * min(max(float(wet_share), 0.0), 1.0)))


def spray_reach_tiles(speed_mps: float, up_mps: float, spread: float,
                      tile_m: float) -> float:
    """The farthest a droplet of these dials lands from where it left (the
    ground track of the fastest, highest one), tiles: ``v_max x 2 up_max /
    g`` over the tile."""
    v = float(speed_mps) * (1.0 + float(spread))
    up = float(up_mps) * (1.0 + float(spread))
    return v * 2.0 * up / G_MPS2 / float(tile_m)


def spray_height(age, life, g_tiles: float) -> np.ndarray:
    """THE droplet's height (tiles) over the water at ``age`` of a flight of
    ``life`` s: the analytic rise and fall ``z = g age (life - age) / 2``,
    0 at launch and at landing, highest at mid-flight."""
    a = np.asarray(age, np.float64)
    return 0.5 * float(g_tiles) * a * (np.asarray(life, np.float64) - a)


def _clip_to_open(x0, y0, vx, vy, life, blocked: Optional[np.ndarray]):
    """Shorten each droplet's ground track so it lands at its last OPEN
    sample before a wall (it falls down the wall face): the horizontal
    velocity is scaled, the flight time kept."""
    if blocked is None or x0.size == 0:
        return vx, vy
    h, w = blocked.shape
    f = np.ones_like(x0)
    for k in range(1, SPRAY_PATH_SAMPLES + 1):
        u = k / SPRAY_PATH_SAMPLES
        px = x0 + vx * life * u
        py = y0 + vy * life * u
        ix = np.floor(px).astype(np.int64)
        iy = np.floor(py).astype(np.int64)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        hit = ~inside | blocked[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
        f = np.where(hit & (f >= 1.0), (k - 1) / SPRAY_PATH_SAMPLES, f)
    return vx * f, vy * f


def spray_spawns(n: int, points, settings: SpraySettings,
                 rng: np.random.Generator, tile_m: float,
                 blocked: Optional[np.ndarray] = None, *,
                 speed_mps: Optional[float] = None,
                 up_mps: Optional[float] = None,
                 heading: Optional[Tuple[float, float]] = None,
                 half_angle: float = math.pi) -> dict:
    """``n`` droplets leaving ``points`` ((k, 2) x, y tiles; one each,
    uniformly), at ``speed_mps`` / ``up_mps`` (default the dials) x (1 +-
    spread x U), in a direction uniform within ``half_angle`` of
    ``heading`` (default: every direction). The flight is ANALYTIC: a
    droplet's ``life`` is ``2 up / g``, its ground track straight at
    ``(vx, vy)`` tiles/s and clipped to the last open point before a wall
    (:func:`_clip_to_open`). Its launch point is kept in the ``t0`` (x) and
    ``temp`` (y) columns. Returns arrays x, y, vx, vy, t0, temp, life."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if n <= 0 or pts.shape[0] == 0:
        return {k: np.zeros(0) for k in _SPAWN_KEYS + ("temp",)}
    sp = settings.speed_mps if speed_mps is None else float(speed_mps)
    up = settings.up_mps if up_mps is None else float(up_mps)
    u = rng.random((4, int(n)))                  # point, angle, speed, up
    idx = np.minimum((u[0] * pts.shape[0]).astype(np.int64),
                     pts.shape[0] - 1)
    base = 0.0 if heading is None else math.atan2(heading[1], heading[0])
    ang = base + (2.0 * u[1] - 1.0) * float(half_angle)
    v = sp * (1.0 + settings.spread * (2.0 * u[2] - 1.0)) / float(tile_m)
    vz = up * (1.0 + settings.spread * (2.0 * u[3] - 1.0))
    life = 2.0 * vz / G_MPS2
    x0, y0 = pts[idx, 0], pts[idx, 1]
    vx, vy = _clip_to_open(x0, y0, np.cos(ang) * v, np.sin(ang) * v, life,
                           blocked)
    return dict(x=x0.copy(), y=y0.copy(), vx=vx, vy=vy, t0=x0, temp=y0,
                life=life)


def step_spray(p: Particles, dt: float) -> np.ndarray:
    """Advance every droplet ``dt`` s along its analytic flight (``x = x0
    + vx age``); a droplet whose age reaches its life LANDS: it is removed
    and its landing point -- ``x0 + vx life`` exactly -- returned ((k, 2)
    tiles), one row per droplet. Nothing outlives its life."""
    if p.n == 0 or dt <= 0.0:
        return np.zeros((0, 2))
    d = p.d
    d[:, AGE] += dt
    land = d[:, AGE] >= d[:, LIFE]
    a = np.minimum(d[:, AGE], d[:, LIFE])
    d[:, X] = d[:, T0] + d[:, VX] * a
    d[:, Y] = d[:, TEMP] + d[:, VY] * a
    out = d[land][:, X:Y + 1].copy()
    if land.any():
        p.keep(~land)
    return out


def pack_spray(p: Particles, view, px_per_tile: float,
               settings: SpraySettings) -> np.ndarray:
    """The droplets in ``view`` ((x0, y0, x1, y1) tiles, None = all) ->
    ``(2, k, 4)`` float32 in the debris stage's layout (row 0 = x, y world
    px, rotation 0, half-width px; row 1 = opacity)."""
    d = p.d
    if view is not None and p.n:
        vx0, vy0, vx1, vy1 = view
        x, y = d[:, X], d[:, Y]
        d = d[(x >= vx0 - 1.0) & (x <= vx1 + 1.0)
              & (y >= vy0 - 1.0) & (y <= vy1 + 1.0)]
    k = int(d.shape[0])
    out = np.zeros((2, k, 4), np.float32)
    if k == 0:
        return out
    ppt = float(px_per_tile)
    out[0, :, 0] = d[:, X] * ppt
    out[0, :, 1] = d[:, Y] * ppt
    out[0, :, 3] = settings.size_tiles * ppt
    out[1, :, 0] = settings.alpha
    return out


class SpraySystem:
    """The droplets' CPU side (#97 W3, the particle primitive's FIFTH
    consumer): per frame :meth:`update` steps the droplets in the air on the
    SIM clock and collects the ones that LAND (``landings_last_frame``: one
    row per droplet, the foam's dabs), then consumes every new ``"blast"``
    effect ONCE (``"spray_seen"``; never another consumer's flag) and throws
    ``rate x strength x wet share`` droplets from its walk's WET seeds;
    :meth:`splash` throws a few back off a wall a round stopped at beside
    the water. Pure numpy (no GL, no sim import: the seeds, the depth and
    the walls come in as arguments)."""

    def __init__(self, settings: SpraySettings):
        self.settings = settings
        self.p = Particles(settings.cap)
        self.enabled_override: Optional[bool] = None
        self._t_prev: Optional[float] = None
        self.spawned_last_frame = 0
        self.landings_last_frame = np.zeros((0, 2))
        self.last_update_ms = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def refresh_settings(self, settings: SpraySettings) -> None:
        if settings.cap != self.p.cap:
            self.p.set_cap(settings.cap)
        self.settings = settings
        if not self.enabled:
            self.p.clear()

    def _add(self, new: dict, salt: int) -> int:
        if new["x"].size == 0:
            return 0
        room = max(self.p.cap - self.p.n, 0)
        if new["x"].size > room:                  # a fair subset, not the first
            rng = np.random.default_rng([int(self.settings.seed) & 0xFFFFFFFF,
                                         int(salt) & 0xFFFFFFFF])
            keep = np.sort(rng.permutation(new["x"].size)[:room])
            new = {k_: v_[keep] for k_, v_ in new.items()}
        return self.p.add(**new)

    def consume(self, effects, seeds_fn: Callable, sim_tick: int,
                depth: Optional[np.ndarray], tile_m: float,
                blocked: Optional[np.ndarray] = None) -> dict:
        """Mark every new ``"blast"`` effect ``"spray_seen"`` and return the
        droplets it throws (none while disabled or on a dry map).
        ``seeds_fn(fy, fx, approach)`` is the blast walk's seed rule;
        ``depth`` the packed per-tile water depth (m; None = dry)."""
        s = self.settings
        keys_all = _SPAWN_KEYS + ("temp",)
        out = []
        for fx in effects:
            if fx.get("kind") != "blast" or fx.get("spray_seen"):
                continue
            fx["spray_seen"] = True
            if not self.enabled or depth is None:
                continue
            pos = fx["pos"]
            seeds = list(seeds_fn(int(pos[1]), int(pos[0]),
                                  fx.get("approach")) or ())
            if not seeds:
                continue
            h, w = depth.shape
            wet = [(int(sy), int(sx)) for sy, sx in seeds
                   if 0 <= sy < h and 0 <= sx < w and depth[sy, sx] > 0.0]
            n = spray_count(fx.get("strength", 0.0), s.rate,
                            len(wet) / len(seeds))
            if n == 0:
                continue
            tick0 = fx.get("tick0", sim_tick)
            rng = spark_rng(s.seed, tick0, pos)
            pts = np.array([(sx + 0.5, sy + 0.5) for sy, sx in wet])
            out.append(spray_spawns(n, pts, s, rng, tile_m, blocked))
        if not out:
            return {k: np.zeros(0) for k in keys_all}
        return {k: np.concatenate([o[k] for o in out]) for k in keys_all}

    def splash(self, x: float, y: float, dir_x: float, dir_y: float,
               tile_m: float, rng: np.random.Generator,
               blocked: Optional[np.ndarray] = None) -> int:
        """A round stopped at a wall beside the water (``tracers``' impact:
        where, the direction back along its flight): throw
        ``splash_drops`` droplets back off the wall within
        :data:`SPLASH_HALF_ANGLE`. Returns how many were added."""
        s = self.settings
        if not self.enabled or s.splash_drops <= 0:
            return 0
        new = spray_spawns(s.splash_drops, [(x, y)], s, rng, tile_m, blocked,
                           speed_mps=s.splash_speed_mps,
                           up_mps=s.splash_up_mps, heading=(dir_x, dir_y),
                           half_angle=SPLASH_HALF_ANGLE)
        k = self._add(new, int(x * 977) ^ int(y * 131))
        self.spawned_last_frame += k
        return k

    def update(self, *, t: float, effects, seeds_fn: Callable,
               sim_tick: int, depth: Optional[np.ndarray], tile_m: float,
               blocked: Optional[np.ndarray] = None) -> np.ndarray:
        """One frame at SIM time ``t`` (monotone; holds while paused): step
        the droplets in the air, collect the LANDINGS (returned and kept in
        :attr:`landings_last_frame`: one row per droplet that came down this
        frame, splashes' included), then add the new blasts' droplets, born
        at ``t``."""
        t0w = time.perf_counter()
        self.spawned_last_frame = 0
        dt = 0.0 if self._t_prev is None else max(float(t) - self._t_prev, 0.0)
        self._t_prev = float(t)
        if not self.enabled:
            self.p.clear()
        land = step_spray(self.p, dt)
        self.landings_last_frame = land
        new = self.consume(effects, seeds_fn, sim_tick, depth, tile_m,
                           blocked)
        self.spawned_last_frame += self._add(new, sim_tick)
        self.last_update_ms = (time.perf_counter() - t0w) * 1000.0
        return land


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


# The DEBRIS stage (graphics G4 P4): THE way a DARK particle is drawn --
# premultiplied alpha, never additive, never a third blend path. Its own
# program (shaders/particle_debris.vs/.fs) and data texture, so it draws
# before the flame without touching the sprites' upload.
DEBRIS_STAGE = "debris"
_COMMON_MARKER = "// <COMMON>"
_FIELD_MARKER = "// <FIELD_SAMPLE>"
_UNIT_LIGHT_A, _UNIT_LIGHT_B = 8, 9


def debris_shader_sources() -> Tuple[str, str]:
    """(vertex, fragment) source of the debris stage, the fragment COMPOSED
    with the lit-3D block (``lit3d._COMMON_GLSL`` at ``// <COMMON>``,
    ``lit3d._FIELD_SAMPLE_GLSL`` at ``// <FIELD_SAMPLE>``): a chip reads the
    light field exactly as a prop does -- the one copy of that read."""
    from .lit3d import _COMMON_GLSL, _FIELD_SAMPLE_GLSL
    fs = (SHADERS_DIR / "particle_debris.fs").read_text(encoding="utf-8")
    for marker in (_COMMON_MARKER, _FIELD_MARKER):
        if marker not in fs:
            raise ValueError(f"particle_debris.fs lacks '{marker}'")
    fs = fs.replace(_COMMON_MARKER, _COMMON_GLSL).replace(
        _FIELD_MARKER, _FIELD_SAMPLE_GLSL)
    return ((SHADERS_DIR / "particle_debris.vs").read_text(encoding="utf-8"),
            fs)


class ParticleDraw:
    """Draws packed sprites (:func:`pack_sprites`) into the BOUND target --
    the scene RT, linear HDR -- additively, keeping the destination alpha,
    in ONE direct-GL draw call. Its second stage, ``"debris"``
    (:meth:`draw_debris`, graphics G4 P4), draws packed chips
    (:func:`pack_debris`) premultiplied, lit by the light field."""

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
        # the debris stage: compiled on its first chip
        self._debris_shader = None
        self._debris_loc = {}
        self._debris_tex = None
        self._debris_width = 0
        self.debris_draws_last_frame = 0
        self.debris_sprites_last_frame = 0

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

    _DEBRIS_UNIFORMS = ("u_parts", "u_world_px", "texture1", "texture2",
                        "u_light_z", "u_ambient", "u_light_gain", "u_albedo",
                        "u_smoke_cover")

    def draw_debris(self, sprites: np.ndarray, *, light,
                    albedo: Tuple[float, float, float]) -> int:
        """The ``"debris"`` stage: draw packed chips (:func:`pack_debris`,
        (2, k, 4) float32) into the bound scene RT, PREMULTIPLIED (ONE,
        ONE_MINUS_SRC_ALPHA on RGB and alpha: the scene RT's premultiplied
        convention, a chip of opacity a darkens what is under it by 1 - a),
        coloured ``albedo x (ambient + exposure x light)`` with the light
        read from ``light`` (a ``lit3d.LightFieldCtx``: ``light_tex_a/b``,
        the smoke's cast shadow, the light curve -- the props' read). One
        direct-GL draw; 0 when there is no chip."""
        k = int(sprites.shape[1])
        self.debris_sprites_last_frame = k
        self.debris_draws_last_frame = 0
        if k == 0:
            return 0
        rl, gl = self._rl, self._gl
        from . import lit3d
        if self._debris_shader is None:
            vs, fs = debris_shader_sources()
            sh = rl.load_shader_from_memory(vs, fs)
            if sh.id == 0 or sh.id == rl.rl_get_shader_id_default():
                raise RuntimeError("debris shader failed to compile/link")
            self._debris_shader = sh
            self._debris_loc = {u: rl.get_shader_location(sh, u)
                                for u in self._DEBRIS_UNIFORMS}
        sh = self._debris_shader
        if k > self._debris_width:
            if self._debris_tex is not None:
                rl.unload_texture(self._debris_tex)
            self._debris_width = max(k, self.width)
            self._debris_tex = self._core.create_dynamic_rgba32f_texture(
                self._debris_width, 2)
        a = np.ascontiguousarray(sprites, dtype=np.float32)
        rl.update_texture_rec(self._debris_tex,
                              rl.Rectangle(0.0, 0.0, float(k), 2.0),
                              rl.ffi.cast("void *",
                                          rl.ffi.from_buffer("float[]", a)))
        # premultiplied alpha (ONE, ONE_MINUS_SRC_ALPHA) on RGB AND alpha --
        # the scene's premultiplied convention (the floor smoke, the water);
        # never additive: a dark chip must darken the floor
        rl.rl_set_blend_factors_separate(rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,
                                         rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,
                                         rl.RL_FUNC_ADD, rl.RL_FUNC_ADD)
        rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM_SEPARATE)
        U = rl.ShaderUniformDataType
        loc = self._debris_loc

        def put(name, vals, kind):
            if loc.get(name, -1) >= 0:
                rl.set_shader_value(sh, loc[name],
                                    rl.ffi.new(f"float[{len(vals)}]",
                                               [float(v) for v in vals]),
                                    kind)

        def put_i(name, v):
            if loc.get(name, -1) >= 0:
                rl.set_shader_value(sh, loc[name], rl.ffi.new("int *", int(v)),
                                    U.SHADER_UNIFORM_INT)

        put_i("u_parts", _UNIT_DATA)
        put_i("texture1", _UNIT_LIGHT_A)
        put_i("texture2", _UNIT_LIGHT_B)
        put_i("u_smoke_cover", _UNIT_SMOKE)
        put("u_world_px", (light.world_px_w, light.world_px_h),
            U.SHADER_UNIFORM_VEC2)
        put("u_light_z", (1.0,), U.SHADER_UNIFORM_FLOAT)
        put("u_ambient", tuple(light.ambient)[:3], U.SHADER_UNIFORM_VEC3)
        put("u_light_gain", (light.light_gain,), U.SHADER_UNIFORM_FLOAT)
        put("u_albedo", tuple(albedo), U.SHADER_UNIFORM_VEC3)
        cast = (light.smoke_cast if light.smoke_cover is not None
                else (0.0, 0.0, 0.0, 0.0))
        lit3d.set_smoke_shadow_uniforms(sh, light.smoke_grid, cast,
                                        light.smoke_cast2)
        lit3d.set_light_curve_uniforms(sh, light.curve)
        gl.UseProgram(sh.id)
        gl.BindVertexArray(gl.vao)
        units = [(_UNIT_DATA, self._debris_tex.id),
                 (_UNIT_LIGHT_A, light.tex_a.id),
                 (_UNIT_LIGHT_B, light.tex_b.id)]
        if light.smoke_cover is not None:
            units.append((_UNIT_SMOKE, light.smoke_cover.id))
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
        self.debris_draws_last_frame = 1
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
        if self._debris_shader is not None:
            self._rl.unload_shader(self._debris_shader)
        if self._debris_tex is not None:
            self._rl.unload_texture(self._debris_tex)


__all__ = [
    "EmberSettings", "settings_from_config", "config_keys", "Particles",
    "cooled_temperature", "fade", "ember_radiance", "SourceTiles",
    "source_tiles", "ember_spawns",
    "integrate", "pack_sprites", "EmberSystem", "ParticleDraw",
    "sprite_shader_sources", "FADE_FROM", "SPRITE_STAGES",
    "SparkSettings", "spark_settings_from_config", "spark_config_keys",
    "spark_count", "spark_rng", "spark_spawns", "SparkSystem",
    "SPARK_MAX_HOP",
    "MuzzleSparkSettings", "muzzle_spark_settings", "muzzle_spark_spawns",
    "MuzzleSparkSystem",
    "DebrisSettings", "debris_settings_from_config", "debris_config_keys",
    "debris_count", "debris_keys", "debris_spawns", "integrate_debris",
    "debris_weight", "pack_debris", "DebrisSystem", "DEBRIS_MAX_HOP",
    "DEBRIS_DENSITY_MAX", "DEBRIS_STAGE", "debris_shader_sources",
    "SpraySettings", "spray_settings_from_config", "spray_config_keys",
    "spray_count", "spray_reach_tiles", "spray_height", "spray_spawns",
    "step_spray", "pack_spray", "SpraySystem", "G_MPS2",
]
