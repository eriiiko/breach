"""The smoke RELIEF pass (graphics arc G1, docs/graphics_direction_
2026-10-18.md §2.1, §2.2, §2.4 top-down integrator, §8.2) -- smoke seen from
above as a lit height field, drawn in the gas medium's slot.

Thesis (§2): three-dimensional smoke seen from above is a RELIEF -- a height
field with a surface. The eye reads (a) its silhouette, (b) the shading of
its surface by the lamps, (c) the shadows it throws, (d) what glows through
it. This pass builds that surface per pixel of the world RT from the sim's
own fields and lights it with the sweep's own light:

  * COVERAGE ``C`` -- a narrow smoothstep of the column's extinction depth,
    pushed in and out by the baked billow field ``B`` (renderer/noise_bake.py)
    that rides the real wind on advected_noise.py's two-phase clock: a sharp,
    billowed edge.
  * HEIGHT ``h`` -- thick columns stand tall, ``B`` bulges their tops.
  * LIGHT -- ``N`` from the height field, ``L`` from the light field's own
    direction lifted by a lamp height, the tile's incoming RGB; a horizon
    march along ``L`` self-shadows the crevices; powder, AO and a rim term.
  * The sweep's in-scatter (god rays) is kept and laid where the smoke is.

Output is premultiplied LINEAR HDR (inscatter, 1 - T) into the scene RT, the
same compositing contract as the B2 medium (``renderer/gas_medium.py``); the
ONE tone-map is the world resolve, never here.

SPECIES LAYERS (Erik's ruling 2 of 2026-10-18): hot soot is a CEILING layer
drawn ABOVE the units; steam, teargas, poison (and fuel vapour) are FLOOR
layers drawn below them, so marines wade in haze and stand under smoke. The
grouping is data, ``[render.gas_relief] ceiling_gases``; each group is one
draw of the same shader with its own textures.

RENDER-ONLY, determinism-exempt: reads dequantized copies of ``gmap.gas``,
``gmap.temperature`` and the wind planes, and the light accessor's field
through LightingPass's textures; writes only its own GPU textures. The
packing below is pyray-free so it is testable headless; the class owns the
GL side.

Dials: ``[render.gas_relief]`` in config.toml, read EVERY FRAME through
:func:`relief_settings` (Ctrl+R retunes live), which refuses a missing or
mistyped key. The noise clock and its advection gain are the B2 detail
pass's (``[render.gas_detail] cycle_seconds`` / ``adv_gain``): one noise
clock.

Credit (repo rule -- cite what a file implements; the shader header carries
the per-term citations):
  - Andrew Schneider & Nathan Vos, "The Real-Time Volumetric Cloudscapes of
    Horizon Zero Dawn", SIGGRAPH 2015 -- coverage remap, Perlin-Worley
    billows, the powder term.
  - Sebastien Hillaire, "Physically Based and Unified Volumetric Rendering
    in Frostbite", SIGGRAPH 2015 -- scatter/transmittance factoring.
  - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 -- the two-phase
    flow-map crossfade (via renderer/advected_noise.py).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .gas_medium import (TraceGasOptics, gas_optical_depth, tau_curve)

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

GROUPS = ("floor", "ceiling")

# The billow texture's coarsest Worley lattice spans the texture this many
# times (renderer/noise_bake.BASE_CELLS): the shader's texture period in
# tiles is BASE_CELLS x billow_cell_tiles.
_BILLOW_SIZE = 256
# The pass binds its samplers on texture units from here up: raylib's batch
# owns units 0..4 (texture0 + RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS = 4).
_FIRST_UNIT = 5
_TOP_LIGHT_SMOOTH_PASSES = 3
# The crossfade's phase-jitter field: one period of the billow bake's smooth
# fBm channel spans this many tiles (coarse -> neighbours share a phase).
_JITTER_WL_TILES = 24.0
# The CAST PRODUCT's resolution (graphics G2): texels per tile. The shadow it
# casts is soft and the billows are >= 1.5 tiles, so 8 carries their shape
# at ~1/36 of the world RT's pixels (48 px per tile).
PRODUCT_PX_PER_TILE = 8


# ---------------------------------------------------------------------------
# Settings ([render.gas_relief]), read every frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReliefSettings:
    """The ``[render.gas_relief]`` dials. Units in the field comments."""
    enabled: bool
    ceiling_gases: tuple          # gas names drawn above the units
    billow_cell_tiles: float      # coarsest billow cell, tiles
    billow_octaves: int           # bake-time octave count (1..3)
    warp_tiles: float             # domain warp of the billow lookup, tiles
    boil_t0: float                # game temperature where boiling starts
    boil_t1: float                # ... and is full
    boil_speedup: float           # hot clock rate / base clock rate
    drift_cap_tiles: float        # most a noise layer drifts in one cycle
    cov_tau_ceiling: float        # extinction depth at the coverage edge,
    cov_tau_floor: float          #   per group
    cov_edge_tiles: float         # edge width across a plume's edge, tiles
    cov_soft_ceiling: float       # least edge width, e-folds of depth (flat
    cov_soft_floor: float         #   haze stays soft), per group
    cov_push_tiles: float         # how far the billows move the edge, tiles
    haze_ceiling: float           # share of the depth kept outside the edge,
    haze_floor: float             #   [0, 1], per group
    height_tiles: float           # tallest column, tiles
    height_tau: float             # extinction depth of a full-height column
    billow_amp: float             # share of the height the billows carve [0,1]
    lamp_z_ceiling: float         # lamp elevation (z of unnormalized L) for the
    lamp_z_floor: float           #   ceiling / floor group
    top_light_tiles: int          # reach of the light over a column's top, tiles
    top_light_falloff: float      # its weight per tile of distance, (0, 1]
    top_light_ceiling: float      # how much of it lights the top, [0, 1],
    top_light_floor: float        #   per group
    wrap: float                   # wrap lighting: N.L floor for thin smoke
    shadow: float                 # self-shadow strength [0,1]
    shadow_len_tiles: float       # horizon march length, tiles
    shadow_soft_tiles: float      # occluder height for full shadow, tiles
    powder: float                 # multi-scatter brightening of thin parts
    ao: float                     # crevice darkening from the height Laplacian
    rim: float                    # silver-lining strength
    rim_g: float                  # Henyey-Greenstein forward anisotropy (0,1)
    glow_texture: float           # billow texture on the in-scatter, [0, 1]
    cast_shadow_ceiling: float    # G2: how much the ceiling layer's cast
    cast_shadow_floor: float      #   shadow takes from the light, [0, 1]
    cast_parallax: float          # tiles of shadow offset per tile of height
    cast_lift_ceiling: float      # the ceiling layer's underside, tiles above
                                  #   the floor
    debug_view: int               # 0 = off; 1 coverage, 2 height, 3 N.L,
                                  #   4 self-shadow, 5 AO, 6 boil, 7 the
                                  #   cast shadow on the floor (G2)


_KINDS = {f.name: f.type for f in fields(ReliefSettings)}


def relief_settings(cfg) -> ReliefSettings:
    """The ``[render.gas_relief]`` settings of a loaded config (``CFG``).
    Raises ValueError naming the key when the section or a key is missing
    or mistyped -- a dial that silently fell back to a default would hide a
    typo while tuning (the ``unit_looks.unit_shading`` precedent)."""
    section = getattr(getattr(cfg, "render", None), "gas_relief", None)
    if section is None:
        raise ValueError("config.toml has no [render.gas_relief] section")
    values = {}
    for name, kind in _KINDS.items():
        if not hasattr(section, name):
            raise ValueError(f"[render.gas_relief] is missing {name!r}")
        v = getattr(section, name)
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
            v = bool(v) if ok else v
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif kind in ("float", float):
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        else:                                   # tuple of gas names
            ok = (isinstance(v, (list, tuple))
                  and all(isinstance(x, str) for x in v))
            v = tuple(v) if ok else v
        if not ok:
            raise ValueError(f"[render.gas_relief] {name} has the wrong type: "
                             f"{v!r}")
        values[name] = v
    s = ReliefSettings(**values)
    if not 1 <= s.billow_octaves <= 3:
        raise ValueError("[render.gas_relief] billow_octaves must be 1..3")
    positive = ("billow_cell_tiles", "drift_cap_tiles", "cov_tau_ceiling",
                "cov_tau_floor", "cov_edge_tiles", "cov_soft_ceiling",
                "cov_soft_floor",
                "height_tau", "boil_speedup", "top_light_falloff")
    bad = [k for k in positive if getattr(s, k) <= 0]
    if bad or s.boil_t1 <= s.boil_t0:
        raise ValueError(f"[render.gas_relief] needs {', '.join(positive)} > 0 "
                         f"and boil_t1 > boil_t0 (bad: {bad})")
    if not 0.0 <= s.rim_g < 1.0:
        raise ValueError("[render.gas_relief] rim_g must be in [0, 1)")
    unit = [k for k in ("cast_shadow_ceiling", "cast_shadow_floor")
            if not 0.0 <= getattr(s, k) <= 1.0]
    if unit or s.cast_parallax < 0.0 or s.cast_lift_ceiling < 0.0:
        raise ValueError("[render.gas_relief] cast_shadow_* must be in [0, 1] "
                         "and cast_parallax, cast_lift_ceiling >= 0 "
                         f"(bad: {unit})")
    return s


# ---------------------------------------------------------------------------
# Pure packing (pyray-free)
# ---------------------------------------------------------------------------

def group_members(names: Sequence[str], ceiling_gases: Sequence[str]
                  ) -> Dict[str, np.ndarray]:
    """``{"ceiling": mask, "floor": mask}`` over the trace gases: a gas named
    in ``ceiling_gases`` is drawn above the units, every other trace gas is
    a floor layer. Every gas lands in exactly one group. An unknown name is
    refused (a typo would silently move soot under the marines)."""
    names = [str(n) for n in names]
    unknown = sorted(set(ceiling_gases) - set(names))
    if unknown:
        raise ValueError(f"[render.gas_relief] ceiling_gases names unknown "
                         f"gas(es) {unknown} (trace gases: {names})")
    ceiling = np.array([n in set(ceiling_gases) for n in names], dtype=bool)
    return {"ceiling": ceiling, "floor": ~ceiling}


# Largest finite float16 (an upload past it would read +inf).
_F16_MAX = float(np.finfo(np.float16).max)


def bulk_reflectance(single_scatter_albedo: np.ndarray) -> np.ndarray:
    """Diffuse reflectance of an optically THICK, isotropically scattering
    medium from its single-scatter albedo ``w``:
    ``R = (1 - s) / (1 + s)``, ``s = sqrt(1 - w)`` (the two-stream /
    diffusion result; Hapke's isotropic-scatterer reflectance). A smoke
    deck is many scatterings deep, so its surface returns far less than one
    scatter does when ``w`` is small: soot (0.22) reads 0.06 -- charcoal --
    while steam (0.92) reads 0.56. Monotone, R(0) = 0, R(1) = 1.

    Credit: Bruce Hapke, "Bidirectional reflectance spectroscopy 1.
    Theory", J. Geophys. Res. 86 (1981) -- the isotropic-scatterer
    reflectance; Kubelka & Munk (1931) for the two-stream form."""
    w = np.clip(np.asarray(single_scatter_albedo, dtype=np.float64), 0.0, 1.0)
    s = np.sqrt(1.0 - w)
    return (1.0 - s) / (1.0 + s)


def relief_group_layers(gas_density: np.ndarray, optics: TraceGasOptics,
                        glow: np.ndarray, member: np.ndarray, *,
                        tau_curve_a: float, tau_curve_b: float,
                        glow_gain: float, effect_gas_floor: float = 0.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """One species group's per-tile inputs -> ``(medium, cover)``, each
    (H, W, 4) float16, for the relief shader.

    medium.rgb  the group's SHARE of the sweep's in-scattered glow, linear:
                ``glow * S_group / S_all`` per channel, where ``S`` is the
                in-scatter coefficient sum ``Σ k_scat[s] ρ_s`` -- the glow is
                linear in it, so the split is exact and the groups' shares
                sum to the whole glow. Times ``glow_gain``, plus the
                legibility floor for the group's effect gases (as in the B2
                medium).
    medium.a    the group's OCCLUSION depth ``tau'``: the B2 medium's own
                ``gas_optical_depth`` over the group's gases, through the
                shared ``tau_curve`` -- so ``1 - exp(-tau')`` is exactly the
                B2 layer's alpha for that group.
    cover.rgb   the group's surface REFLECTANCE: the table's single-scatter
                ``scatter_albedo``, weighted by each gas's share of the
                extinction, turned into what a THICK cloud of it reflects
                (:func:`bulk_reflectance`: soot 0.22 -> 0.06, steam 0.92 ->
                0.56) -- the surface is lit as a body, not as one scatter.
    cover.a     the EXTINCTION depth ``Σ (k_s + mean k_scat[s]) ρ_s``: how
                much stuff is in the column -- absorbers AND scatterers, so a
                white steam cloud has a body even though it barely absorbs.
                It drives coverage and height; no curve (the coverage dials
                shape it).
    """
    rho = np.maximum(np.asarray(gas_density, dtype=np.float64), 0.0)
    m = np.asarray(member, dtype=bool)
    k_abs = np.where(m, optics.k_s, 0.0).astype(np.float64)
    k_scat = np.asarray(optics.k_scat, dtype=np.float64)          # (S, 3)
    tau = gas_optical_depth(rho, k_abs, base_absorb_scale=1.0,
                            plume_k_scale=1.0)
    tau_p = tau_curve(tau, tau_curve_a, tau_curve_b)

    s_all = np.tensordot(k_scat.T, rho, axes=([1], [0]))          # (3, H, W)
    s_grp = np.tensordot((k_scat * m[:, None]).T, rho, axes=([1], [0]))
    share = np.where(s_all > 0.0, s_grp / np.maximum(s_all, 1e-30), 0.0)
    g = np.asarray(glow, dtype=np.float64) * float(glow_gain)
    glow_grp = g * np.moveaxis(share, 0, -1)
    if effect_gas_floor > 0.0:
        alb = np.asarray(optics.scatter_albedo, dtype=np.float64)
        for s in np.flatnonzero(m & optics.effect_mask):
            d = np.clip(rho[s], 0.0, 1.0)
            glow_grp = glow_grp + effect_gas_floor * d[..., None] * alb[s]

    k_ext = (np.asarray(optics.k_s, dtype=np.float64)
             + k_scat.mean(axis=1)) * m                            # (S,)
    ext_s = k_ext[:, None, None] * rho                             # (S, H, W)
    tau_ext = ext_s.sum(axis=0)
    alb_w = np.tensordot(np.asarray(optics.scatter_albedo,
                                    dtype=np.float64).T, ext_s,
                         axes=([1], [0]))                          # (3, H, W)
    albedo = np.where(tau_ext > 0.0, alb_w / np.maximum(tau_ext, 1e-30), 0.0)
    albedo = bulk_reflectance(albedo)

    h, w = tau.shape
    medium = np.empty((h, w, 4), dtype=np.float16)
    medium[..., :3] = np.clip(glow_grp, 0.0, _F16_MAX)
    medium[..., 3] = np.clip(tau_p, 0.0, _F16_MAX)
    cover = np.empty((h, w, 4), dtype=np.float16)
    cover[..., :3] = np.clip(np.moveaxis(albedo, 0, -1), 0.0, 1.0)
    cover[..., 3] = np.clip(tau_ext, 0.0, _F16_MAX)
    return medium, cover


def _window_max(a: np.ndarray, r: int, axis: int,
                weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Max over a (2r+1)-wide window along ``axis`` (0 or 1, edge-padded),
    each tap times ``weights[k]`` (2r+1,) if given."""
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    out = None
    for k in range(2 * r + 1):
        tap = p[k:k + n] if axis == 0 else p[:, k:k + n]
        if weights is not None:
            tap = tap * weights[k]
        out = tap.copy() if out is None else np.maximum(out, tap, out=out)
    return out


def _box3(a: np.ndarray) -> np.ndarray:
    """One separable 3x3 box pass over axes 0, 1 (edge-padded)."""
    p = np.pad(a, ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    hz = p[:, :-2] + p[:, 1:-1] + p[:, 2:]
    return (hz[:-2] + hz[1:-1] + hz[2:]) * (1.0 / 9.0)


def top_light_field(light_rgb: np.ndarray, radius_tiles: int,
                    falloff: float = 1.0) -> np.ndarray:
    """The light over a smoke column's TOP, (H, W, 3) float32, per channel:
    the brightest incoming light within ``radius_tiles``, each neighbour
    weighted ``falloff ** (|dx| + |dy|)`` (tiles) -- an attenuated max
    filter, separable because the weight is a product -- then 3x3 box
    passes to soften its plateaus (one pass left the filter's tile-square
    steps visible as quadrants on a lit puff). Without the falloff a fire
    tile's own (enormous) light was carried two tiles out undiminished and
    lit the soot over a burning crate like a lamp would.

    Why: the sweep is planar, so inside a thick soot plume the tile's own
    light is what crossed the plume sideways -- almost nothing (pre-#88 soot
    absorbs ~138 per unit density per tile). Seen from above, a ceiling
    layer's top is lit by lamps whose light passes OVER the plume, i.e. by
    roughly the light the room has around it. The max filter is that
    estimate: the light at the plume's lamp-side edge carried
    ``radius_tiles`` inward. It is a filtered READ of LightingPass's field
    (the one light), never a second light path. Known limit: it reaches
    across a wall thinner than the radius."""
    out = np.asarray(light_rgb, dtype=np.float32)
    r = max(int(radius_tiles), 0)
    if r > 0:
        w = float(falloff) ** np.abs(np.arange(-r, r + 1, dtype=np.float32))
        out = _window_max(out, r, 1, w)
        out = _window_max(out, r, 0, w)
        for _ in range(_TOP_LIGHT_SMOOTH_PASSES):
            out = _box3(out)
    return np.ascontiguousarray(out, dtype=np.float32)


def boil_field(temperature_game: np.ndarray, t0: float, t1: float
               ) -> np.ndarray:
    """Per-tile BOIL weight in [0, 1] from the game temperature (ΔT above
    ambient): 0 at or below ``t0``, 1 at or above ``t1``, smooth between.
    Hot smoke churns faster -- the shader blends the billow field from the
    base clock toward a faster clock by this weight (§2.1)."""
    x = np.clip((np.asarray(temperature_game, dtype=np.float64) - t0)
                / max(t1 - t0, 1e-9), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _crc(a: np.ndarray) -> int:
    """CRC32 of an array's bytes (a cheap content key for repacking)."""
    import zlib
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8))


def clip_tile_rect(tile_rect, grid_w: int, grid_h: int
                   ) -> Tuple[int, int, int, int]:
    """``tile_rect`` (x0, y0, x1, y1) clamped to the grid, as ints (floor
    of the start, ceiling of the end); None -> the whole grid."""
    if tile_rect is None:
        return 0, 0, int(grid_w), int(grid_h)
    x0, y0, x1, y1 = tile_rect
    return (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
            min(int(grid_w), int(np.ceil(x1))),
            min(int(grid_h), int(np.ceil(y1))))


def presence_mask(tau_ext: np.ndarray, reach_tiles: int = 2) -> np.ndarray:
    """(H, W) float32: 1 where the group has ANY extinction within
    ``reach_tiles`` tiles (a square dilation), else 0. The shader discards a
    pixel whose (bilinear) mask is exactly 0 before any other read, so the
    reach must cover the bicubic reconstruction's support (2 texels) -- a
    pixel it discards can see no smoke."""
    m = (np.asarray(tau_ext) > 0.0).astype(np.float32)
    r = int(reach_tiles)
    if r > 0:
        m = _window_max(_window_max(m, r, 1), r, 0)
    return m


def pack_relief_dynamics(tamed_wind: np.ndarray, boil: np.ndarray,
                         floor_mask: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """(H, W, 4) float16: RG = the tamed wind in tiles/tick (from
    ``gas_detail.tame_wind``, THE wind seam -- never re-derived here),
    B = boil weight, A = the floor group's presence mask (0 if None)."""
    h, w = boil.shape
    out = np.zeros((h, w, 4), dtype=np.float16)
    out[..., 0] = tamed_wind[..., 0]
    out[..., 1] = tamed_wind[..., 1]
    out[..., 2] = boil
    if floor_mask is not None:
        out[..., 3] = floor_mask
    return out


def cast_uniforms(s: ReliefSettings, active: bool = True
                  ) -> Tuple[Tuple[float, float, float, float],
                             Tuple[float, float, float, float]]:
    """The two vec4 uniforms the cast-shadow RECEIVERS read (graphics G2;
    ``lit3d._SMOKE_SHADOW_GLSL``): ``u_smoke_cast = (ceiling strength,
    floor strength, parallax, ceiling lift)`` and ``u_smoke_cast2 =
    (tallest column, debug flag, 0, 0)``, tiles; the flag (``debug_view``
    7) makes the floor show the shadow factor itself. ``active`` False (the relief off,
    smoke hidden) zeroes both strengths, which the GLSL reads as "no shadow"
    before it samples anything."""
    on = 1.0 if (active and s.enabled) else 0.0
    return ((s.cast_shadow_ceiling * on, s.cast_shadow_floor * on,
             s.cast_parallax, s.cast_lift_ceiling),
            (s.height_tiles, 1.0 if s.debug_view == 7 else 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# The GL pass
# ---------------------------------------------------------------------------

class GasReliefPass:
    """Owns ``shaders/gas_relief.fs``, the baked billow + jitter textures,
    and per frame the two groups' (medium, cover) textures and one dynamics
    texture. ``draw(group, ...)`` draws one species layer premultiplied."""

    _UNIFORMS = (
        "u_cover", "u_dyn", "u_light_a", "u_light_b", "u_billow",
        "u_grid", "u_phase", "u_tau_ticks", "u_phase_hot", "u_tau_ticks_hot",
        "u_adv_gain", "u_drift_cap", "u_billow_tiles", "u_billow_mean", "u_warp_tiles",
        "u_jitter_wl", "u_cov_tau", "u_cov_edge", "u_cov_soft", "u_cov_push", "u_haze", "u_height",
        "u_height_tau", "u_billow_amp", "u_lamp_z", "u_wrap", "u_light_gain",
        "u_ambient", "u_shadow", "u_shadow_len", "u_shadow_soft", "u_powder",
        "u_ao", "u_rim", "u_rim_g", "u_debug", "u_toplight", "u_top_light",
        "u_glow_texture", "u_group", "u_mode",
    )

    def __init__(self, grid_h: int, grid_w: int, settings: ReliefSettings):
        import pyray as rl
        from . import core
        self._rl = rl
        self._core = core
        self.h, self.w = grid_h, grid_w
        self.settings = settings
        self.enabled = settings.enabled
        self.shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"), str(SHADERS_DIR / "gas_relief.fs"))
        self._loc = {}
        for name in self._UNIFORMS:
            loc = rl.get_shader_location(self.shader, name)
            if loc == -1:
                print(f"[gas_relief] WARN: shader uniform '{name}' not found")
            self._loc[name] = loc
        self.medium_tex = {g: core.create_dynamic_rgba16f_texture(grid_w, grid_h)
                           for g in GROUPS}
        self.cover_tex = {g: core.create_dynamic_rgba16f_texture(grid_w, grid_h)
                          for g in GROUPS}
        self.dyn_tex = core.create_dynamic_rgba16f_texture(grid_w, grid_h)
        self.top_tex = core.create_dynamic_rgba16f_texture(grid_w, grid_h)
        self._top_packed = np.zeros((grid_h, grid_w, 4), dtype=np.float16)
        self._has = {g: False for g in GROUPS}
        self._masks = {g: np.zeros((grid_h, grid_w), np.float32)
                       for g in GROUPS}
        self._pack_key = None
        self._baked_octaves = None
        self.billow_tex = None
        self._billow_mean = 0.5
        self._bake_billow(settings.billow_octaves)
        self._phase = (0.0, 0.0)
        self._tau_ticks = (0.0, 0.0)
        self._adv_gain = 1.0
        self._optics_id = None
        self._optics: Optional[TraceGasOptics] = None
        self._members = None
        self.last_pack_ms = 0.0
        # The CAST PRODUCT (G2): the relief's own occlusion + height per
        # PRODUCT_PX_PER_TILE texels, rendered once per frame before the
        # scene (render_product) and read by the shadow receivers and the
        # heat layer's glow-through. ONE product, never a second smoke model.
        self.product_rt = core.load_render_texture_rgba16f(
            grid_w * PRODUCT_PX_PER_TILE, grid_h * PRODUCT_PX_PER_TILE)
        rl.set_texture_filter(self.product_rt.texture,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_wrap(self.product_rt.texture,
                            rl.TextureWrap.TEXTURE_WRAP_CLAMP)
        self.product_live = False

    @classmethod
    def from_config(cls, grid_h: int, grid_w: int, cfg) -> "GasReliefPass":
        return cls(grid_h, grid_w, relief_settings(cfg))

    # ---- textures --------------------------------------------------------

    def _static_rgba(self, img: np.ndarray, *, point: bool):
        rl, core = self._rl, self._core
        tex = core.create_dynamic_rgba_texture(img.shape[1], img.shape[0])
        core.update_rgba_texture(tex, np.ascontiguousarray(img, dtype=np.uint8))
        rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_POINT if point
                              else rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_REPEAT)
        return tex

    def _bake_billow(self, octaves: int) -> None:
        from .noise_bake import bake_billow_rgba
        img = bake_billow_rgba(_BILLOW_SIZE, octaves=int(octaves))
        if self.billow_tex is not None:
            self._rl.unload_texture(self.billow_tex)
        self.billow_tex = self._static_rgba(img, point=False)
        self._billow_mean = float(img[..., 0].mean()) / 255.0
        self._baked_octaves = int(octaves)

    # ---- per frame ---------------------------------------------------------

    def refresh_settings(self, cfg) -> None:
        """Re-read ``[render.gas_relief]`` (every frame: Ctrl+R is live)."""
        self.settings = relief_settings(cfg)
        self.enabled = self.settings.enabled
        if self.settings.billow_octaves != self._baked_octaves:
            self._bake_billow(self.settings.billow_octaves)

    def update(self, gas_density_trace: np.ndarray, glow: np.ndarray,
               gas_table, tamed_wind: np.ndarray,
               temperature_game: np.ndarray, light_rgb: np.ndarray, *,
               sim_tick: int,
               ticks_per_second: float, cycle_seconds: float,
               adv_gain: float, tau_curve_a: float, tau_curve_b: float,
               glow_gain: float, effect_gas_floor: float) -> None:
        """Pack and upload both groups' textures and the dynamics, and
        advance the two noise clocks on the SIM tick (never wall time)."""
        import time
        from .advected_noise import advection_phase
        from .gas_medium import trace_gas_optics
        t0 = time.perf_counter()
        s = self.settings
        if self._optics_id != id(gas_table):
            self._optics = trace_gas_optics(gas_table)
            self._optics_id = id(gas_table)
            self._members = None
        if self._members is None or self._members[0] != s.ceiling_gases:
            self._members = (s.ceiling_gases,
                             group_members(self._optics.names, s.ceiling_gases))
        members = self._members[1]
        # The tile inputs change once per sim tick (or on a relight, a tool
        # write, a dial): repack only when their CONTENT changed -- a CRC of
        # the arrays plus the dials is ~0.05 ms against ~1 ms of packing,
        # and a paused or 60-fps-over-24-tps frame repacks nothing.
        key = (_crc(gas_density_trace), _crc(glow), _crc(light_rgb), s,
               float(tau_curve_a), float(tau_curve_b), float(glow_gain),
               float(effect_gas_floor), id(self._optics))
        if key != self._pack_key:
            self._pack_key = key
            for g in GROUPS:
                medium, cover = relief_group_layers(
                    gas_density_trace, self._optics, glow, members[g],
                    tau_curve_a=tau_curve_a, tau_curve_b=tau_curve_b,
                    glow_gain=glow_gain, effect_gas_floor=effect_gas_floor)
                self._has[g] = bool((cover[..., 3] > 0).any())
                self._masks[g] = presence_mask(cover[..., 3].astype(np.float32))
                if self._has[g]:
                    self._core.update_rgba16f_texture(self.medium_tex[g], medium)
                    self._core.update_rgba16f_texture(self.cover_tex[g], cover)
            self._top_packed[..., :3] = np.clip(
                top_light_field(light_rgb, s.top_light_tiles,
                                s.top_light_falloff), 0.0, _F16_MAX)
            self._top_packed[..., 3] = self._masks["ceiling"]
            self._core.update_rgba16f_texture(self.top_tex, self._top_packed)
        masks = self._masks
        boil = boil_field(temperature_game, s.boil_t0, s.boil_t1)
        self._core.update_rgba16f_texture(
            self.dyn_tex, pack_relief_dynamics(tamed_wind, boil,
                                               masks["floor"]))
        base = advection_phase(int(sim_tick), cycle_seconds, ticks_per_second)
        hot = advection_phase(int(sim_tick), cycle_seconds / s.boil_speedup,
                              ticks_per_second)
        self._phase = (base.phase, hot.phase)
        self._tau_ticks = (base.tau_ticks, hot.tau_ticks)
        self._adv_gain = float(adv_gain)
        self.last_pack_ms = (time.perf_counter() - t0) * 1000.0

    def has_layer(self, group: str) -> bool:
        return self._has[group]

    # ---- draw ----------------------------------------------------------------

    def _f(self, name: str, v: float) -> None:
        rl = self._rl
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new("float[1]", [float(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def _v(self, name: str, *v: float) -> None:
        rl = self._rl
        kind = {2: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2,
                3: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3}[len(v)]
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new(f"float[{len(v)}]", [float(x) for x in v]),
                            kind)

    def _i(self, name: str, v: int) -> None:
        rl = self._rl
        rl.set_shader_value(self.shader, self._loc[name],
                            rl.ffi.new("int[1]", [int(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def draw(self, group: str, lighting, dst_w: int, dst_h: int,
             tile_rect: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Draw one species layer over the world RT, premultiplied
        (``out = src.rgb + dst * (1 - src.a)``, the volume operator).

        ``tile_rect = (x0, y0, x1, y1)`` (tiles, end-exclusive) limits the
        quad to that part of the world -- the camera's view: the shader costs
        per world-RT pixel, and a big level's RT is several times a 1080p
        screen. None draws the whole world."""
        if not self._has[group]:
            return
        self._rl.begin_blend_mode(self._rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        self._draw_group(group, lighting, dst_w, dst_h, tile_rect, mode=0)
        self._rl.end_blend_mode()

    def render_product(self, lighting, active: bool,
                       tile_rect: Optional[Tuple[int, int, int, int]] = None
                       ) -> None:
        """Render the CAST PRODUCT (graphics G2) into ``product_rt``: per
        texel the ceiling layer's (alpha, height) in RG and the floor
        layer's in BA -- the same coverage, height and occlusion the layer
        draws composite, from the same shader (``u_mode`` 1). Cleared to 0
        (no smoke) first; ``active`` False leaves it cleared. Must run
        OUTSIDE any other texture mode (it binds its own target): the
        renderer calls it before the scene RT is begun."""
        rl = self._rl
        rl.begin_texture_mode(self.product_rt)
        rl.clear_background(rl.Color(0, 0, 0, 0))
        self.product_live = bool(active and self.enabled
                                 and (self._has["ceiling"]
                                      or self._has["floor"]))
        if self.product_live:
            pw = self.w * PRODUCT_PX_PER_TILE
            ph = self.h * PRODUCT_PX_PER_TILE
            # (r, g, b, a) sums: the groups write disjoint channels.
            rl.begin_blend_mode(rl.BlendMode.BLEND_ADD_COLORS)
            for g in GROUPS:
                if self._has[g]:
                    self._draw_group(g, lighting, pw, ph, tile_rect, mode=1)
            rl.end_blend_mode()
        rl.end_texture_mode()

    @property
    def product_texture(self):
        """The cast product's texture (a raylib RenderTexture's colour
        attachment: rows run bottom-up, so a reader samples
        ``(u, 1 - v)``; ``lit3d._SMOKE_SHADOW_GLSL`` does)."""
        return self.product_rt.texture

    def _draw_group(self, group: str, lighting, dst_w: int, dst_h: int,
                    tile_rect, mode: int) -> None:
        """Set the uniforms, bind the samplers and draw one group's quad in
        ``mode`` (0 = the layer, 1 = the cast product). The caller owns the
        blend mode and the target."""
        rl, s = self._rl, self.settings
        self._v("u_grid", self.w, self.h)
        self._f("u_phase", self._phase[0])
        self._f("u_phase_hot", self._phase[1])
        self._f("u_tau_ticks", self._tau_ticks[0])
        self._f("u_tau_ticks_hot", self._tau_ticks[1])
        self._f("u_adv_gain", self._adv_gain)
        from .noise_bake import BASE_CELLS
        self._f("u_billow_tiles", BASE_CELLS * s.billow_cell_tiles)
        self._f("u_billow_mean", self._billow_mean)
        self._f("u_warp_tiles", s.warp_tiles)
        self._f("u_jitter_wl", _JITTER_WL_TILES)
        ceiling = group == "ceiling"
        self._f("u_drift_cap", s.drift_cap_tiles)
        self._f("u_cov_tau", s.cov_tau_ceiling if ceiling else s.cov_tau_floor)
        self._f("u_cov_edge", s.cov_edge_tiles)
        self._f("u_cov_soft", s.cov_soft_ceiling if ceiling
                else s.cov_soft_floor)
        self._f("u_cov_push", s.cov_push_tiles)
        self._f("u_haze", s.haze_ceiling if ceiling else s.haze_floor)
        self._f("u_height", s.height_tiles)
        self._f("u_height_tau", s.height_tau)
        self._f("u_billow_amp", s.billow_amp)
        self._f("u_lamp_z", s.lamp_z_ceiling if ceiling else s.lamp_z_floor)
        self._f("u_top_light", s.top_light_ceiling if ceiling
                else s.top_light_floor)
        self._f("u_wrap", s.wrap)
        self._f("u_light_gain", lighting.light_gain)
        self._v("u_ambient", *lighting.ambient)
        self._f("u_shadow", s.shadow)
        self._f("u_shadow_len", s.shadow_len_tiles)
        self._f("u_shadow_soft", s.shadow_soft_tiles)
        self._f("u_powder", s.powder)
        self._f("u_ao", s.ao)
        self._f("u_rim", s.rim)
        self._f("u_rim_g", s.rim_g)
        self._f("u_glow_texture", s.glow_texture)
        self._i("u_debug", s.debug_view if mode == 0 else 0)
        self._i("u_group", 1 if ceiling else 0)
        self._i("u_mode", int(mode))

        rl.begin_shader_mode(self.shader)
        # SIX samplers besides texture0, and raylib's batch binds at most four
        # (RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS): set_shader_value_texture
        # silently drops the fifth on, which read as an all-zero billow field.
        # So the pass binds its own units above raylib's (0..4) through rlgl,
        # and puts the active unit back to 0 before raylib draws anything.
        extra = (("u_cover", self.cover_tex[group]), ("u_dyn", self.dyn_tex),
                 ("u_toplight", self.top_tex),
                 ("u_light_a", lighting.light_tex_a),
                 ("u_light_b", lighting.light_tex_b),
                 ("u_billow", self.billow_tex))
        for i, (name, tex) in enumerate(extra):
            unit = _FIRST_UNIT + i
            rl.rl_active_texture_slot(unit)
            rl.rl_enable_texture(tex.id)
            self._i(name, unit)
        rl.rl_active_texture_slot(0)
        x0, y0, x1, y1 = clip_tile_rect(tile_rect, self.w, self.h)
        if x1 > x0 and y1 > y0:
            ppx, ppy = float(dst_w) / self.w, float(dst_h) / self.h
            src = rl.Rectangle(float(x0), float(y0), float(x1 - x0),
                               float(y1 - y0))
            dst = rl.Rectangle(x0 * ppx, y0 * ppy, (x1 - x0) * ppx,
                               (y1 - y0) * ppy)
            # texture0 = the group's medium texture (glow share + occlusion)
            rl.draw_texture_pro(self.medium_tex[group], src, dst,
                                rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_shader_mode()                 # flushes the batch: draws now
        for i in range(len(extra)):
            rl.rl_active_texture_slot(_FIRST_UNIT + i)
            rl.rl_disable_texture()
        rl.rl_active_texture_slot(0)

    def unload(self) -> None:
        rl = self._rl
        rl.unload_shader(self.shader)
        for t in list(self.medium_tex.values()) + list(self.cover_tex.values()):
            rl.unload_texture(t)
        rl.unload_texture(self.dyn_tex)
        rl.unload_texture(self.top_tex)
        rl.unload_texture(self.billow_tex)
        rl.unload_render_texture(self.product_rt)


__all__ = [
    "GROUPS", "ReliefSettings", "relief_settings", "group_members",
    "bulk_reflectance", "relief_group_layers", "top_light_field",
    "boil_field", "presence_mask", "clip_tile_rect", "pack_relief_dynamics",
    "cast_uniforms", "PRODUCT_PX_PER_TILE", "GasReliefPass",
]
