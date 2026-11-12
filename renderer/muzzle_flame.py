"""THE muzzle flame -- the fire at a gun's barrel when it fires (#87).

Design: ``docs/muzzle_flame_design_2026-11-12.md``.

WHAT IT IS. A procedural flame, new for every shot, drawn additively in
linear HDR into the scene RT (the one tone-map does the rest), coloured only
through the blackbody GPU LUT. Four parts:

  * the settings door (:func:`settings_from_config`, ``[render.muzzle_flame]``:
    which KIND a weapon's flash is, and each kind's dials -- refusing);
  * the LAW, pure numpy: :func:`shot_rows` draws one shot's shape on the CPU
    (its jets, whether it is a fireball) from a generator seeded by the shot,
    and :func:`flame_radiance` is the per-pixel radiance of that shape at an
    age -- THE SPEC of ``shaders/muzzle_flame.fs`` (a change lands here
    first, then in the shader; ``tests/gl_muzzle_flame_check.py`` holds them
    together);
  * :class:`MuzzleFlames`, the CPU state: one flash per (unit, tick) of a
    launched ``"tracer"`` effect, aged on the SIM clock from the first frame
    it is drawn, placed every frame on its shooter's muzzle;
  * :class:`MuzzleFlameDraw`, the GL face: one direct-GL draw of a quad per
    flash from a small RGBA32F data texture (the particle sprite's idiom).

The room LIGHT of a shot is not here: it is the frame-light assembly's
(``frame_lights.muzzle_flash_lights``, ``[render.muzzle_flash]``).

RENDER-ONLY. Reads a unit's id, weapon id, place and facing; writes no
synced state; float math (render layer). pyray only in the GL class.

Credits: the tapered-tongue + eroding-noise construction follows the usual
real-time flame-card practice (Reeves 1983 for per-particle stochastic
attributes; Perlin 1985 "An Image Synthesizer" for the turbulence that tears
the edge); the colour is the project's one blackbody ramp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

MAX_JETS = 5              # the forward tongue + up to four side jets
ROWS = 4 + MAX_JETS       # the data texture's rows per flash
NOISE_SIZE = 256          # the fBm bake's side (4 octaves: 8 texels a finest cell)
_NOISE_CELLS = 4.0        # advected_noise._FBM_BASE_PERIOD: base cells per texture

# The law's fixed constants (twin and shader; a look change is a dial, a
# change HERE is a change of the law and of the shader with it).
GROW_FROM, GROW_BY, GROW_AGE = 0.9, 0.3, 1.0 / 3.0   # lengths leap out early
WOBBLE_AT_BIRTH, WOBBLE_WITH_AGE = 0.7, 1.6          # x the jet's width
TEAR_AT_BIRTH, TEAR_WITH_AGE = 0.30, 1.3
NOISE_CONTRAST = 1.7
HALO_K = 1700.0           # the halo's colour temperature
EDGE_FROM = 0.85          # the quad's window: 1 inside this share, 0 at the edge

_SMOKE_MARKER = "// <SMOKE_SHADOW>"
_UNIT_DATA, _UNIT_LUT, _UNIT_NOISE, _UNIT_SMOKE = 5, 6, 8, 7
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_2D = 0x0DE1
_GL_TRIANGLES = 0x0004
STAGES = {"scene": 0, "compensation": 1}


# ---------------------------------------------------------------------------
# Settings -- [render.muzzle_flame]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlameKind:
    """One ``[render.muzzle_flame.kinds.<name>]`` row. Lengths in METRES
    before ``size``; temperatures in Kelvin on the blackbody ramp."""
    size: float              # THE size dial: scales every length below
    length_m: float          # the forward tongue
    width_m: float
    side_jets: int           # 0..4 side jets (pairs mirror; an odd one strays)
    side_len_m: float
    side_width_m: float
    side_angle_deg: float    # a side jet's angle off the barrel (+- its draw)
    core_m: float            # the white ball at the muzzle
    halo_m: float            # the soft glow's radius
    halo_gain: float
    t_core_k: float
    t_edge_k: float
    gain: float              # x the ramp's Le(T)
    tear: float              # how hard the noise eats the edge, [0, 1]
    fireball_chance: float   # share of shots that are a short fat ball
    duration_s: float        # sim seconds from the first drawn frame to dark
    sparks: int              # mean sparks a shot throws (0 = none)


_KIND_TYPES = {f.name: f.type for f in fields(FlameKind)}
_TOP_KEYS = ("enabled", "default_kind", "weapons", "seed", "cap", "kinds",
             "sparks")


@dataclass(frozen=True)
class FlameSettings:
    enabled: bool
    default_kind: str                 # any firing weapon not in `weapons`; "" = none
    weapons: Dict[str, str]           # [weapons.*] id -> kind name ("" = none)
    seed: int
    cap: int                          # live flashes, the oldest dropped
    kinds: Dict[str, FlameKind]

    def kind_for(self, weapon_id: str) -> Optional[FlameKind]:
        name = self.weapons.get(weapon_id or "", self.default_kind)
        return self.kinds[name] if name else None


def _as_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _as_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(
            obj, (str, bytes, int, float, list)):
        return {k: _as_dict(v) for k, v in vars(obj).items()}
    return obj


def typed_row(row, types: Mapping[str, object], where: str) -> dict:
    """``row`` (a dict) checked against ``types`` (name -> bool / int /
    float, or their names): every key present, none unknown, none mistyped
    -- refused by name."""
    if not isinstance(row, dict):
        raise ValueError(f"{where} must be a table")
    bad = sorted(set(row) - set(types))
    if bad:
        raise ValueError(f"{where} has unknown keys {bad}")
    out = {}
    for name, kind in types.items():
        if name not in row:
            raise ValueError(f"{where} is missing {name!r}")
        v = row[name]
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"{where} {name} has the wrong type: {v!r}")
        out[name] = v
    return out


def _kind(name: str, row) -> FlameKind:
    w = f"[render.muzzle_flame.kinds.{name}]"
    k = FlameKind(**typed_row(row, _KIND_TYPES, w))
    for f in ("size", "length_m", "width_m", "core_m", "halo_m", "duration_s",
              "gain"):
        if not getattr(k, f) > 0.0:
            raise ValueError(f"{w} {f} must be > 0")
    for f in ("side_len_m", "side_width_m", "halo_gain"):
        if getattr(k, f) < 0.0:
            raise ValueError(f"{w} {f} must be >= 0")
    if not 0 <= k.side_jets <= MAX_JETS - 1:
        raise ValueError(f"{w} side_jets must be 0..{MAX_JETS - 1}")
    if k.side_jets and not (k.side_len_m > 0.0 and k.side_width_m > 0.0):
        raise ValueError(f"{w} side jets need side_len_m and side_width_m > 0")
    if not 0.0 < k.side_angle_deg < 180.0:
        raise ValueError(f"{w} side_angle_deg must be in (0, 180)")
    for f in ("tear", "fireball_chance"):
        if not 0.0 <= getattr(k, f) <= 1.0:
            raise ValueError(f"{w} {f} must be in [0, 1]")
    if not k.t_core_k >= k.t_edge_k > 0.0:
        raise ValueError(f"{w} needs t_core_k >= t_edge_k > 0")
    if k.sparks < 0:
        raise ValueError(f"{w} sparks must be >= 0")
    return k


def settings_from_config(cfg, weapon_ids: Optional[Sequence[str]] = None
                         ) -> FlameSettings:
    """``[render.muzzle_flame]`` of a loaded config (``CFG``) or of a plain
    dict holding ``render.muzzle_flame``, read every frame (F5 retunes live).
    A REFUSING door: a missing or unknown key, a mistyped value, a
    ``weapons`` key that is no ``[weapons.*]`` id (``weapon_ids``; None = the
    config's own table), a kind name that is no ``kinds`` row -- each raises
    ValueError naming it. The ``sparks`` table is the particle module's
    (``particles.muzzle_spark_settings``) and only required to exist here."""
    if isinstance(cfg, Mapping):
        sec = cfg.get("render", {}).get("muzzle_flame")
        weapons_cfg = cfg.get("weapons")
    else:
        sec = getattr(getattr(cfg, "render", None), "muzzle_flame", None)
        weapons_cfg = getattr(cfg, "weapons", None)
    if sec is None:
        raise ValueError("config.toml has no [render.muzzle_flame] section")
    d = _as_dict(sec)
    w = "[render.muzzle_flame]"
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
                             f"{sorted(kinds)} (\"\" = no flame)")
    dk = d["default_kind"]
    if not isinstance(dk, str) or (dk and dk not in kinds):
        raise ValueError(f"{w} default_kind = {dk!r}: not a kinds row "
                         f"{sorted(kinds)} (\"\" = no flame)")
    return FlameSettings(enabled=d["enabled"], default_kind=dk,
                         weapons=dict(d["weapons"]), seed=int(d["seed"]),
                         cap=int(d["cap"]), kinds=kinds)


# ---------------------------------------------------------------------------
# The law (pure): one shot's shape, and its radiance per pixel
# ---------------------------------------------------------------------------

def shot_rng(seed: int, unit_id: int, tick0: int) -> np.random.Generator:
    """THE generator of one shot: the dial's seed, the shooter and the tick
    -- the same shot draws the same flame (a look shot reproduces), two
    shots never share a stream."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF,
                                  int(unit_id) & 0xFFFFFFFF,
                                  int(tick0) & 0xFFFFFFFF])


@dataclass(frozen=True)
class Shot:
    """One shot's drawn shape. ``rows`` is ``(2 + MAX_JETS, 4)`` float32, the
    data texture's rows 2..: ``(core m, halo m, halo gain, gain)``,
    ``(T core, T edge, tear, half m)``, then per jet ``(angle rad, length m,
    width m, gain)`` (length 0 = no jet). ``ofs`` is its noise offset (uv),
    ``half_m`` the half side of the square it is dark outside of."""
    rows: np.ndarray
    ofs: Tuple[float, float]
    half_m: float
    fireball: bool
    sparks: int


def shot_rows(kind: FlameKind, rng: np.random.Generator) -> Shot:
    """Draw one shot of ``kind``: the forward tongue (a small angle error, a
    length and width draw), its side jets (mirrored pairs, each side with
    its own angle, length, width and gain; an odd one strays), and with
    ``fireball_chance`` a FIREBALL: a shorter, fatter tongue and a bigger
    core. ``size`` scales every length."""
    k = float(kind.size)
    fireball = bool(rng.random() < kind.fireball_chance)
    var = rng.uniform(0.8, 1.25)
    length = kind.length_m * k * var * (0.7 if fireball else 1.0)
    width = kind.width_m * k * rng.uniform(0.85, 1.2) * (2.1 if fireball else 1.0)
    jets = [(rng.normal(0.0, 0.035), length, width, 1.0)]
    for j in range(kind.side_jets // 2):
        for sgn in (-1.0, 1.0):
            a = math.radians(kind.side_angle_deg * rng.uniform(0.65, 1.35)
                             * (1.0 - 0.4 * j))
            jets.append((sgn * a,
                         kind.side_len_m * k * rng.uniform(0.5, 1.25),
                         kind.side_width_m * k * rng.uniform(0.7, 1.5),
                         rng.uniform(0.75, 1.0)))
    if kind.side_jets % 2:
        jets.append((rng.uniform(-1.2, 1.2),
                     kind.side_len_m * k * rng.uniform(0.4, 0.9),
                     kind.side_width_m * k, 0.7))
    ofs = (float(rng.random()), float(rng.random()))
    n_sparks = 0
    if kind.sparks > 0:
        n_sparks = int(rng.integers(max(kind.sparks // 2, 0),
                                    kind.sparks + (kind.sparks + 1) // 2 + 1))
    core = kind.core_m * k * (1.9 if fireball else 1.0)
    halo = kind.halo_m * k
    grow = GROW_FROM + GROW_BY
    reach = max(j[1] * grow + 3.0 * j[2] for j in jets)
    half = max(reach, 0.25 * length * grow + 1.5 * 2.6 * halo,
               2.6 * core) / EDGE_FROM
    rows = np.zeros((2 + MAX_JETS, 4), np.float32)
    rows[0] = (core, halo, kind.halo_gain, kind.gain)
    rows[1] = (kind.t_core_k, kind.t_edge_k, kind.tear, half)
    for i, jet in enumerate(jets[:MAX_JETS]):
        rows[2 + i] = jet
    return Shot(rows=rows, ofs=ofs, half_m=float(half), fireball=fireball,
                sparks=n_sparks)


def noise_field() -> np.ndarray:
    """THE flame's noise: the one tiling fBm bake (``advected_noise
    .bake_fbm_rgba``), (NOISE_SIZE, NOISE_SIZE, 4) uint8 -- R and B tear the
    body, G wobbles the centre line. The GL class uploads exactly this."""
    from .advected_noise import bake_fbm_rgba
    return bake_fbm_rgba(NOISE_SIZE, octaves=4)


def sample_noise(noise: np.ndarray, u, v, channel: int) -> np.ndarray:
    """``noise`` read as the GPU reads it: REPEAT wrap, bilinear, texel
    centres at ``(i + 0.5) / n``, 0..1."""
    n = noise.shape[0]
    x = np.asarray(u, np.float64) * n - 0.5
    y = np.asarray(v, np.float64) * n - 0.5
    x0 = np.floor(x); y0 = np.floor(y)
    fx = x - x0; fy = y - y0
    ix0 = np.mod(x0, n).astype(np.int64); iy0 = np.mod(y0, n).astype(np.int64)
    ix1 = (ix0 + 1) % n; iy1 = (iy0 + 1) % n
    c = noise[..., channel].astype(np.float64) * (1.0 / 255.0)
    return ((c[iy0, ix0] * (1 - fx) + c[iy0, ix1] * fx) * (1 - fy)
            + (c[iy1, ix0] * (1 - fx) + c[iy1, ix1] * fx) * fy)


def grow_factor(age: float) -> float:
    """A jet's length over its drawn length at ``age``: it leaps out in the
    first third of the flash's life."""
    return GROW_FROM + GROW_BY * min(max(age, 0.0) / GROW_AGE, 1.0)


def flame_density(x, y, shot: Shot, age: float, noise: np.ndarray
                  ) -> np.ndarray:
    """The flame's density in [0, 1] at local metres ``(x, y)`` (x along the
    barrel) for ``shot`` at ``age`` in [0, 1): the max of the torn jets and
    the core ball, without the halo."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    rows = shot.rows.astype(np.float64)
    ou, ov = shot.ofs
    grow = grow_factor(age)
    dens = np.zeros(np.broadcast(x, y).shape)
    nz = np.zeros_like(dens); wsum = np.zeros_like(dens)
    for i in range(MAX_JETS):
        ang, length, width, g = rows[2 + i]
        if length <= 0.0:
            continue
        length = length * grow
        ca, sa = math.cos(ang), math.sin(ang)
        s = x * ca + y * sa
        t = -x * sa + y * ca
        u = s / length
        uc = np.clip(u, 0.0, 1.0)
        wob = sample_noise(noise, s / (length * 0.45 * _NOISE_CELLS)
                           + 0.37 * i + ou,
                           np.full_like(s, age * 1.5 / _NOISE_CELLS + ov),
                           1) - 0.5
        t = t + wob * width * (WOBBLE_AT_BIRTH + WOBBLE_WITH_AGE * age) * uc
        inside = (u > 0.0) & (u < 1.0)
        prof = np.where(inside, np.power(uc, 0.45)
                        * np.power(1.0 - uc, 0.8) * 1.9, 0.0)
        wj = width * np.maximum(prof, 1e-4)
        body = np.exp(-np.square(t / wj)) * inside
        along = np.power(np.clip(1.15 - u, 0.0, 1.0), 0.6)
        n1 = sample_noise(noise,
                          (s - age * length * 0.6)
                          / (width * 5.0 * _NOISE_CELLS) + ou + 0.21 * i,
                          t / (width * 0.9 * _NOISE_CELLS) + ov, 0)
        n2 = sample_noise(noise,
                          s / (width * 1.6 * _NOISE_CELLS) + ov + 0.13 * i,
                          t / (width * 0.5 * _NOISE_CELLS) + ou, 2)
        n = np.clip((0.6 * n1 + 0.4 * n2 - 0.5) * NOISE_CONTRAST + 0.5,
                    0.0, 1.0)
        d = body * along
        dens = np.maximum(dens, d * g)
        nz += n * d; wsum += d
    nz = np.where(wsum > 1e-6, nz / np.maximum(wsum, 1e-6), 0.5)
    tear = rows[1, 2] * (TEAR_AT_BIRTH + TEAR_WITH_AGE * age)
    body = np.clip((dens - tear * nz) / max(1.0 - 0.45 * tear, 1e-3), 0.0, 1.0)
    body = body * body * (3.0 - 2.0 * body)
    r = np.hypot(x, y)
    core = np.exp(-np.square(r / rows[0, 0])) * (1.0 - age) ** 1.5
    return np.clip(np.maximum(body, core), 0.0, 1.0)


def flame_radiance(x, y, shot: Shot, age: float, noise: np.ndarray, ramp
                   ) -> np.ndarray:
    """THE LAW: linear HDR rgb ``(..., 3)`` of ``shot`` at local metres
    ``(x, y)`` and ``age`` in [0, 1) -- the spec of ``muzzle_flame.fs``.
    Exactly 0 at and beyond the shot's square (``half_m``) and at age >= 1.
    ``ramp`` is the one ``BlackbodyRamp`` (its ``emission_at_kelvin``)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    out_shape = np.broadcast(x, y).shape + (3,)
    if not 0.0 <= age < 1.0:
        return np.zeros(out_shape)
    rows = shot.rows.astype(np.float64)
    core_m, halo_m, halo_gain, gain = rows[0]
    t_core, t_edge, _tear, half = rows[1]
    d = flame_density(x, y, shot, age, noise)
    cool = (1.0 - age) ** 1.4
    fade = (1.0 - age) ** 0.8
    t_c = t_edge + (t_core - t_edge) * (0.25 + 0.75 * cool)
    kelvin = t_edge * (0.85 + 0.15 * cool) + (t_c - t_edge) * np.power(d, 2.6)
    chroma, inten = ramp.emission_at_kelvin(kelvin)
    rgb = chroma * (inten * np.sqrt(d) * gain * fade)[..., None]
    hc, hi = ramp.emission_at_kelvin(np.array([HALO_K]))
    cx = 0.25 * rows[2, 1] * grow_factor(age)
    hr = np.hypot((x - cx) / 1.5, y)
    halo = (np.exp(-np.square(hr / halo_m))
            + 0.6 * np.exp(-np.square(hr / (0.35 * halo_m))))
    rgb = rgb + hc[0] * (halo * halo_gain * hi[0] * cool * fade)[..., None]
    edge = np.maximum(np.abs(x), np.abs(y)) / half
    win = np.clip((1.0 - edge) / (1.0 - EDGE_FROM), 0.0, 1.0)
    win = win * win * (3.0 - 2.0 * win)
    return rgb * win[..., None]


# ---------------------------------------------------------------------------
# Where the muzzle is
# ---------------------------------------------------------------------------

def muzzle_pose_from_matrix(prop_world: np.ndarray,
                            muzzle: Sequence[float], px_per_tile: float
                            ) -> Optional[Tuple[float, float, float, float]]:
    """The muzzle's place and the barrel's direction ON THE FLOOR PLANE, in
    tiles: ``(x, y, dir_x, dir_y)`` from the held prop's world matrix (world
    px; the 3D pass's x = world x, z = world y) and its ``muzzle`` point in
    file units (the barrel runs along the file's +x). None when the barrel
    points straight up or down (no direction on the floor)."""
    m = np.asarray(prop_world, np.float64)
    p = np.einsum("ij,j->i", m[:3, :3],
                  np.asarray(muzzle, np.float64)) + m[:3, 3]
    ax = m[:3, 0]
    n = math.hypot(ax[0], ax[2])
    if n < 1e-9:
        return None
    ppt = float(px_per_tile)
    return (float(p[0]) / ppt, float(p[2]) / ppt,
            float(ax[0]) / n, float(ax[2]) / n)


def fallback_pose(x: float, y: float, footprint: float, facing: float
                  ) -> Tuple[float, float, float, float]:
    """A unit drawn with no prop: the flame starts at its footprint's edge
    along its facing (the light's rule). Tiles; facing in radians, y down."""
    fp = float(footprint)
    dx, dy = math.cos(facing), math.sin(facing)
    return (x + fp / 2.0 + dx * fp / 2.0, y + fp / 2.0 + dy * fp / 2.0,
            dx, dy)


# ---------------------------------------------------------------------------
# The CPU state
# ---------------------------------------------------------------------------

@dataclass
class Flash:
    unit: int
    tick0: int
    kind: FlameKind
    shot: Shot
    aim: Optional[Tuple[float, float]] = None   # the round's own direction
    pose: Optional[Tuple[float, float, float, float]] = None
    t_first: Optional[float] = None     # sim seconds at its first drawn frame
    age: float = 0.0                    # normalised, [0, 1)


class MuzzleFlames:
    """The live flashes. Per frame: :meth:`consume` turns every new launched
    ``"tracer"`` effect into ONE flash per (unit, tick) (marking the effect
    ``"flame_seen"``; never another consumer's flag); :meth:`update` ages
    them on the sim clock from the first frame each is drawn (so a paused
    game holds a flash and every shot shows its age-0 frame), puts each on
    its shooter's muzzle (``pose_fn(unit_id)`` -> (x, y, dir_x, dir_y) tiles
    or None = keep the last) -- pointing along the ROUND's own direction
    (the tracer's ``from`` -> ``to``) when the effect carries one, so the
    flame and the tracer agree whatever the gun's drawn pose, else along
    the barrel -- and drops the spent ones; :meth:`pack` hands
    the draw its data rows. Never more than ``cap`` alive (the oldest go)."""

    def __init__(self, settings: FlameSettings):
        self.settings = settings
        self.flashes: List[Flash] = []
        self._seen: Dict[Tuple[int, int], bool] = {}
        self.enabled_override: Optional[bool] = None
        self.born_last_frame: List[Flash] = []

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def refresh_settings(self, settings: FlameSettings) -> None:
        self.settings = settings
        if not self.enabled:
            self.flashes = []

    def consume(self, effects, weapon_of: Callable[[int], Optional[str]],
                sim_tick: int) -> None:
        """Every new ``"tracer"`` effect that left the barrel (``launch``)
        -> a flash of its shooter's weapon's kind; a burst's rounds of one
        tick are one flash. ``weapon_of(unit_id)`` -> the weapon id, or None
        for a unit that is gone (no flash)."""
        s = self.settings
        for fx in effects:
            if fx.get("kind") != "tracer" or fx.get("flame_seen"):
                continue
            fx["flame_seen"] = True
            if not self.enabled or not fx.get("launch", True):
                continue
            unit = fx.get("unit")
            if unit is None:
                continue
            tick0 = int(fx.get("tick0", sim_tick))
            key = (int(unit), tick0)
            if key in self._seen:
                continue
            self._seen[key] = True
            wid = weapon_of(int(unit))
            if wid is None:
                continue
            kind = s.kind_for(str(wid))
            if kind is None:
                continue
            aim = None
            a, b = fx.get("from"), fx.get("to")
            if a is not None and b is not None:
                ax, ay = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
                n = math.hypot(ax, ay)
                if n > 1e-6:
                    aim = (ax / n, ay / n)
            self.flashes.append(Flash(
                unit=int(unit), tick0=tick0, kind=kind, aim=aim,
                shot=shot_rows(kind, shot_rng(s.seed, int(unit), tick0))))
        if len(self._seen) > 4096:          # keys older than any live effect
            keep = {(f.unit, f.tick0) for f in self.flashes}
            self._seen = {k: True for k in self._seen
                          if k in keep or k[1] >= sim_tick - 8}
        if len(self.flashes) > s.cap:
            self.flashes = self.flashes[len(self.flashes) - s.cap:]

    def update(self, t_sim: float,
               pose_fn: Callable[[int], Optional[Tuple[float, float, float,
                                                       float]]]) -> None:
        """One frame at sim time ``t_sim`` (seconds, (tick + alpha) / tps)."""
        self.born_last_frame = []
        alive = []
        for f in self.flashes:
            pose = pose_fn(f.unit)
            if pose is not None:
                f.pose = (pose if f.aim is None
                          else (pose[0], pose[1], f.aim[0], f.aim[1]))
            if f.pose is None:
                continue                      # nowhere to draw it, ever
            if f.t_first is None:
                f.t_first = float(t_sim)
                self.born_last_frame.append(f)
            f.age = max(float(t_sim) - f.t_first, 0.0) / f.kind.duration_s
            if f.age < 1.0:
                alive.append(f)
        self.flashes = alive

    def pack(self, view, px_per_tile: float, tile_size_m: float) -> np.ndarray:
        """The flashes in ``view`` ((x0, y0, x1, y1) tiles, None = all) ->
        ``(ROWS, k, 4)`` float32: row 0 ``(x px, y px, dir x, dir y)``, row 1
        ``(age, noise u, noise v, px per metre)``, rows 2.. the shot's."""
        ppm = float(px_per_tile) / float(tile_size_m)
        rows = []
        for f in self.flashes:
            if f.pose is None or f.t_first is None:
                continue
            x, y, dx, dy = f.pose
            if view is not None:
                m = f.shot.half_m / float(tile_size_m)
                if not (view[0] - m <= x <= view[2] + m
                        and view[1] - m <= y <= view[3] + m):
                    continue
            a = np.empty((ROWS, 4), np.float32)
            a[0] = (x * px_per_tile, y * px_per_tile, dx, dy)
            a[1] = (f.age, f.shot.ofs[0], f.shot.ofs[1], ppm)
            a[2:] = f.shot.rows
            rows.append(a)
        if not rows:
            return np.empty((ROWS, 0, 4), np.float32)
        return np.ascontiguousarray(np.stack(rows, axis=1))


def rasterise(packed: np.ndarray, width: int, height: int, noise: np.ndarray,
              ramp) -> np.ndarray:
    """The twin's picture of a pack: ``(height, width, 3)`` linear HDR, world
    px (y down), every flash added -- what :class:`MuzzleFlameDraw` adds to
    a black target (the GL check's reference; tools' contact sheets)."""
    img = np.zeros((height, width, 3))
    ys, xs = np.mgrid[0:height, 0:width]
    px = xs + 0.5; py = ys + 0.5
    for i in range(packed.shape[1]):
        a = packed[:, i].astype(np.float64)
        x0, y0, dx, dy = a[0]
        age, ou, ov, ppm = a[1]
        shot = Shot(rows=packed[2:, i], ofs=(float(ou), float(ov)),
                    half_m=float(a[3, 3]), fireball=False, sparks=0)
        lx = ((px - x0) * dx + (py - y0) * dy) / ppm
        ly = (-(px - x0) * dy + (py - y0) * dx) / ppm
        img += flame_radiance(lx, ly, shot, float(age), noise, ramp)
    return img


# ---------------------------------------------------------------------------
# The GL face
# ---------------------------------------------------------------------------

def shader_sources() -> Tuple[str, str]:
    """(vertex, fragment) source, the fragment COMPOSED with the smoke cast
    product's shared GLSL (``lit3d._SMOKE_SHADOW_GLSL``: ``smokeCoverAt``) at
    its marker -- the one copy of that read."""
    from .lit3d import _SMOKE_SHADOW_GLSL
    fs = (SHADERS_DIR / "muzzle_flame.fs").read_text(encoding="utf-8")
    if _SMOKE_MARKER not in fs:
        raise ValueError(f"muzzle_flame.fs lacks '{_SMOKE_MARKER}'")
    return ((SHADERS_DIR / "muzzle_flame.vs").read_text(encoding="utf-8"),
            fs.replace(_SMOKE_MARKER, _SMOKE_SHADOW_GLSL))


class MuzzleFlameDraw:
    """Draws packed flashes (:meth:`MuzzleFlames.pack`) into the BOUND
    target -- the scene RT, linear HDR -- additively, keeping the
    destination alpha, in ONE direct-GL draw call (the particle sprite's
    idiom: a quad per flash from ``gl_VertexID`` over a data texture)."""

    _F1 = ("u_lut_n", "u_glow_through_ceil")
    _UNIFORMS = ("u_data", "u_lut", "u_noise", "u_smoke_cover", "u_world_px",
                 "u_lut_k", "u_stage") + _F1

    def __init__(self, cap: int = 16):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        vs, fs = shader_sources()
        sh = rl.load_shader_from_memory(vs, fs)
        if sh.id == 0 or sh.id == rl.rl_get_shader_id_default():
            raise RuntimeError("muzzle flame shader failed to compile/link")
        self.shader = sh
        self._loc = {u: rl.get_shader_location(sh, u) for u in self._UNIFORMS}
        self.noise_tex = core.create_dynamic_rgba_texture(NOISE_SIZE,
                                                          NOISE_SIZE)
        core.update_rgba_texture(self.noise_tex, noise_field())
        rl.set_texture_filter(self.noise_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_wrap(self.noise_tex, rl.TextureWrap.TEXTURE_WRAP_REPEAT)
        self.tex = None
        self.width = 0
        self._alloc(cap)
        self.draws_last_frame = 0
        self.flashes_last_frame = 0

    def _alloc(self, cap: int) -> None:
        if self.tex is not None:
            self._rl.unload_texture(self.tex)
        self.width = max(1, int(cap))
        self.tex = self._core.create_dynamic_rgba32f_texture(self.width, ROWS)

    def draw(self, packed: np.ndarray, *, world_px_w: float,
             world_px_h: float, lut_tex, lut_k: Tuple[float, float],
             lut_n: int, stage: str = "scene", smoke_cover=None,
             glow_through_ceil: float = 1.0) -> int:
        """Draw ``packed`` ((ROWS, k, 4) float32) at ``stage`` ("scene":
        under the ceiling soot; "compensation": after it, weighted by the
        soot's alpha x ``1 - glow_through_ceil`` -- skipped without a
        cover). The "scene" stage uploads the data; the compensation reuses
        it. Returns the draw calls issued (no flash, no draw)."""
        k = int(packed.shape[1])
        comp = stage == "compensation"
        if not comp:
            self.draws_last_frame = 0
            self.flashes_last_frame = k
        if k == 0 or (comp and (smoke_cover is None
                                or glow_through_ceil >= 1.0)):
            return 0
        rl, gl = self._rl, self._gl
        if not comp:
            if k > self.width:
                self._alloc(k)
            a = np.ascontiguousarray(packed, dtype=np.float32)
            rl.update_texture_rec(
                self.tex, rl.Rectangle(0.0, 0.0, float(k), float(ROWS)),
                rl.ffi.cast("void *", rl.ffi.from_buffer("float[]", a)))
        rl.rl_set_blend_factors_separate(rl.RL_ONE, rl.RL_ONE, rl.RL_ZERO,
                                         rl.RL_ONE, rl.RL_FUNC_ADD,
                                         rl.RL_FUNC_ADD)
        rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM_SEPARATE)
        gl.UseProgram(self.shader.id)
        loc = self._loc
        for name, unit in (("u_data", _UNIT_DATA), ("u_lut", _UNIT_LUT),
                           ("u_noise", _UNIT_NOISE),
                           ("u_smoke_cover", _UNIT_SMOKE)):
            if loc[name] >= 0:
                gl.Uniform1i(loc[name], unit)
        if loc["u_stage"] >= 0:
            gl.Uniform1i(loc["u_stage"], STAGES[stage])
        if loc["u_world_px"] >= 0:
            gl.Uniform2f(loc["u_world_px"], float(world_px_w),
                         float(world_px_h))
        if loc["u_lut_k"] >= 0:
            gl.Uniform2f(loc["u_lut_k"], float(lut_k[0]), float(lut_k[1]))
        if loc["u_lut_n"] >= 0:
            gl.Uniform1f(loc["u_lut_n"], float(lut_n))
        if loc["u_glow_through_ceil"] >= 0:
            gl.Uniform1f(loc["u_glow_through_ceil"], float(glow_through_ceil))
        gl.BindVertexArray(gl.vao)
        units = [(_UNIT_DATA, self.tex.id), (_UNIT_LUT, lut_tex.id),
                 (_UNIT_NOISE, self.noise_tex.id)]
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

    def unload(self) -> None:
        self._rl.unload_shader(self.shader)
        self._rl.unload_texture(self.tex)
        self._rl.unload_texture(self.noise_tex)


__all__ = [
    "FlameKind", "FlameSettings", "settings_from_config", "typed_row",
    "shot_rng", "Shot", "shot_rows", "noise_field", "sample_noise",
    "grow_factor", "flame_density", "flame_radiance",
    "muzzle_pose_from_matrix", "fallback_pose", "Flash", "MuzzleFlames",
    "rasterise", "shader_sources", "MuzzleFlameDraw", "MAX_JETS", "ROWS",
    "NOISE_SIZE", "STAGES",
]
