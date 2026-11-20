"""Water surface optics pass (graphics/water_rendering.md §7 steps 1+2; the
water look arc #97, docs/water_look_design_2026-11-19.md).

A SEPARATE GLSL fragment pass (not an extension of lighting.fs, §5) that turns
the sim's water fields into convincing water: perturbed normal -> a see-
through refraction + Beer-Lambert depth-tint BASE (with a small Fresnel
ambient sheen) + an ADDITIVE GGX glint (reusing the light buffer) that rides on
top as HDR specular light. The glint is added, not Fresnel-blended at ~2% (which
crushed it top-down) — see shaders/water.fs.

W1 (#97): THE FINE SURFACE THAT FLOWS. The normal carries, besides the
simulated tile-scale ripple, a fine surface of wave slopes in METRES carried by
the simulated flow (Vlachos's two-phase flow map through
``renderer/advected_noise.py``, every lookup hex-tiled after Mikkelsen 2022).
Its law, its settings door and its packing are ``renderer/water_surface.py``
(the numpy twin the shader is GL-checked against); this module is the GL face.

It draws on the floor, after the lit ship and before the floor smoke layer and
the units (G2). DRY COSTS NOTHING (critique M4): :meth:`WaterPass.update` asks
the INT depth plane ``.any()`` before anything is dequantized, packs and
uploads only when the sim state changed (once per sim tick; CRC-keyed while
paused, so a paused pour shows), and :meth:`WaterPass.draw` draws nothing on a
dry map and otherwise only over the view ∩ the wet tiles' box.

Plumbing mirrors LightingPass. Two per-tile RGBA16F textures, bilinear:
    u_water  R = ripple (m)   G = ripple_v (m/s)
             B = water_depth (m)   A = the relative agitation (m/s:
             ``water_surface.relative_agitation``, the fine surface's
             strength and the foam both read it)
    u_flow   R, G = flow (m/s, y south)   B = solid   A = previous depth (m)
plus the static ripple bake (``noise_bake.bake_ripple_rgba``, RGBA32F,
REPEAT, mipmapped) and its curvature companion (W4,
``noise_bake.bake_ripple_curvature_rgba``, the same state), the foam image (W2, bilinear for the draw) and the
static bubble bake (``noise_bake.bake_foam_rgba``, REPEAT, mipmapped). The light textures and the level diffuse need no new
plumbing; the smoke's cast product and the light curve come from the lit-3D
seam (``lit3d._SMOKE_SHADOW_GLSL`` + ``_LIGHT_CURVE_GLSL``, spliced at the
shader's ``// <SMOKE_SHADOW>`` marker -- the lighting.py precedent).

THE CLOCK is the SIM clock ``(tick + alpha) / tps`` through the pass's own
``fire_vfx.SimClock`` (idle when paused, re-anchored when the tick goes back),
never wall time (critique B2). Its fixed step is the foam's (W2,
``foam_step_s``): each frame owes the foam the steps due on that clock.

W4 (#97): SHARP CAUSTICS. The floor's source light is multiplied by the
fine surface's focusing, ``min(1/|det J|, caustic_max)`` of the refraction
map (Wallace's area method as an analytic Jacobian; the twin's
``water_surface.caustic_term``), read from the ripple's curvature companion
through the same hex-tiled, flow-mapped lookup. It replaced the Laplacian
caustic, its procedural detail and the last ``u_time``.

W2 (#97): FOAM WITH A LIFE. A foam amount per cell (``foam_cells_per_tile``
per tile) in its own :class:`core.PingPong` image, allocated LAZILY on the
first wet frame (a dry map never owns one), stepped by
:meth:`WaterPass.simulate` (``shaders/water_foam.fs``, direct GL, over the
wet box in cells only) -- made by sources, carried by the flow (MacCormack +
clamp, substeps of at most one cell), lost by decay; reset when the sim tick
goes back (a replay seek) and when the map dries out; a level load builds a
new pass. Its law is ``renderer/water_foam.py`` (the twin). The draw reads it
bilinear and thresholds a bubble raft (``noise_bake.bake_foam_rgba``) into
lace. Picture state only: never synced, never read by the simulation.

Every dial is ``[graphics.water]``, read every frame through
``water_surface.water_settings`` (the one refusing door) and pushed to the
shader by :meth:`WaterPass.refresh_settings` when it changed.
"""
from __future__ import annotations

import math
import time
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyray as rl

from . import core
from . import water_foam
from .lighting import art_src_and_uv_rect
from .water_surface import (CAUSTIC_K, FlowFrame, WaterSettings, flow_frame,
                            pack_flow, relative_agitation, wet_box)


SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

# Sampler units beyond raylib's batch (which binds texture0 + 4): the smoke's
# cast product, the flow texture, the ripple bake, the foam image, the bubble
# bake (render-effects rule: samplers from unit 5 up). The foam step's own
# direct-GL block binds 8..10 (the water, the flow, the foam front).
_UNIT_SMOKE, _UNIT_FLOW, _UNIT_RIPPLE = 5, 6, 7
_UNIT_FOAM, _UNIT_BUBBLES = 8, 9
_UNIT_CURV = 10                     # W4: the ripple's curvature companion
_STEP_UNITS = {"u_water": 8, "u_flow": 9, "u_foam": 10}
RIPPLE_SIZE = 256
FOAM_TEX_SIZE = 256
_UNSET = object()


def load_water_fs() -> str:
    """``shaders/water.fs`` with lit3d's smoke cast shadow + light curve GLSL
    spliced in at its ``// <SMOKE_SHADOW>`` marker and the advected-noise
    crossfade (``advected_noise.ADVECTION_GLSL``) at ``// <ADVECTION>``.
    Pure text, no GL."""
    from .advected_noise import ADVECTION_GLSL
    from .lit3d import _LIGHT_CURVE_GLSL, _SMOKE_SHADOW_GLSL
    src = (SHADERS_DIR / "water.fs").read_text(encoding="utf-8")
    for marker in ("// <SMOKE_SHADOW>", "// <ADVECTION>"):
        if src.count(marker) != 1:
            raise RuntimeError(f"shaders/water.fs lost its {marker} marker")
    return (src.replace("// <SMOKE_SHADOW>",
                        _SMOKE_SHADOW_GLSL + _LIGHT_CURVE_GLSL)
               .replace("// <ADVECTION>", ADVECTION_GLSL))


def _ripple_texture() -> rl.Texture:
    """The ripple bake as an RGBA32F texture: REPEAT, mipmapped, trilinear
    (critique M8: minified, the slopes average toward flat)."""
    from .noise_bake import bake_ripple_rgba
    return _static_texture(bake_ripple_rgba(RIPPLE_SIZE))


def _ripple_curvature_texture() -> rl.Texture:
    """The ripple's curvature companion (W4's caustic,
    ``noise_bake.bake_ripple_curvature_rgba``): the same size and sampler
    state as the ripple bake, so the two are read at the same texels."""
    from .noise_bake import bake_ripple_curvature_rgba
    return _static_texture(bake_ripple_curvature_rgba(RIPPLE_SIZE))


def _bubble_texture() -> rl.Texture:
    """The foam's bubble bake (W2), RGBA32F, REPEAT, mipmapped, trilinear."""
    from .noise_bake import bake_foam_rgba
    return _static_texture(bake_foam_rgba(FOAM_TEX_SIZE))


def load_foam_fs() -> str:
    """``shaders/water_foam.fs``, the foam step's pass program."""
    return (SHADERS_DIR / "water_foam.fs").read_text(encoding="utf-8")


def _static_texture(img: np.ndarray) -> rl.Texture:
    n = img.shape[0]
    tex = core.create_dynamic_rgba32f_texture(n, n)
    core.update_rgba32f_texture(tex, img)
    ptr = rl.ffi.new("Texture *", tex)
    rl.gen_texture_mipmaps(ptr)
    tex = ptr[0]
    rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_REPEAT)
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_TRILINEAR)
    return tex


class WaterPass:
    """Owns the water shader, the two per-tile textures, the ripple bake and
    the pass's sim clock."""

    def __init__(self, grid_h: int, grid_w: int,
                 settings: Optional[WaterSettings] = None,
                 tile_size_m: float = 1.0 / 3.0):
        from .fire_vfx import SimClock
        self.h = int(grid_h)
        self.w = int(grid_w)
        self.tile_size_m = float(tile_size_m)

        # u_water: R = ripple, G = ripple_v, B = water_depth, A = 0
        self.packed = np.zeros((self.h, self.w, 4), dtype=np.float16)
        self.water_tex = core.create_dynamic_rgba16f_texture(self.w, self.h)
        # u_flow: (vx, vy, solid, previous depth) -- critique M11
        self.flow_packed = np.zeros((self.h, self.w, 4), dtype=np.float16)
        self.flow_tex = core.create_dynamic_rgba16f_texture(self.w, self.h)
        for t in (self.water_tex, self.flow_tex):
            rl.set_texture_filter(t, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self.ripple_tex = _ripple_texture()
        self.ripple_curv_tex = _ripple_curvature_texture()
        self.bubble_tex = _bubble_texture()
        self._prev_depth = np.zeros((self.h, self.w), np.float32)
        self._last_depth = np.zeros((self.h, self.w), np.float32)
        self._have_prev = False

        # Picture state: whether anything is wet, the wet box (tiles, end-
        # exclusive, dilated one tile), the last packed state's key.
        self.wet = False
        self.box: Optional[tuple] = None
        self._key = None
        # the sim clock, stepped in the foam's fixed steps (W2)
        self.clock = SimClock(1.0 / 48.0, cap=water_foam.FOAM_STEP_CAP)
        self._wall_prev: Optional[float] = None
        self.frame = FlowFrame(0.0, 1.0, 0.0, 0.0, 1.0)
        self.last_update_ms = 0.0

        # W2: the foam. Its PingPong is allocated on the first wet frame
        # (simulate); the steps owed, the per-tick fastest flow, the box the
        # passes ran over (cells) and the reset flag are plain values set by
        # update and spent by simulate.
        self.foam: Optional[core.PingPong] = None
        self.foam_m = 0
        self._foam_steps = 0
        self._v_max = 0.0
        self._foam_box: Optional[tuple] = None
        self._foam_reset = False
        self._last_tick: Optional[int] = None
        self.foam_steps_last_frame = 0
        self.foam_passes_last_frame = 0
        self.last_simulate_ms = 0.0
        self._step_shader = None
        self._step_locs: dict = {}
        self._step_ucache: dict = {}

        # [art.align] transform (level format v2 §1.3) — same convention as
        # LightingPass.draw_lit_world. None = legacy full-stretch draw.
        self.art_align = None

        self.shader = core.load_shader_from_source(
            str(SHADERS_DIR / "lighting.vs"), load_water_fs())
        self._locs: dict = {}
        self._curve_sent = _UNSET

        self._has_height = False
        self._height_tex = None
        self._set_i("u_has_height", 0)
        self._set_vec("u_texel", (1.0 / self.w, 1.0 / self.h))
        self._set_vec("u_art_uv_rect", (0.0, 0.0, 1.0, 1.0))
        self.set_srgb_decode(True)
        self.set_light_z(0.5)
        self.set_ambient((0.18, 0.18, 0.22))
        self._set_f("u_light_gain", 1.0)
        self.settings: Optional[WaterSettings] = None
        if settings is None:
            from config import CFG
            from .water_surface import water_settings
            settings = water_settings(CFG)
        self.refresh_settings(settings)

    # ---- uniform helpers -----------------------------------------------

    def _loc(self, name: str) -> int:
        loc = self._locs.get(name)
        if loc is None:
            loc = rl.get_shader_location(self.shader, name)
            if loc == -1:
                print(f"[water] WARN: shader uniform '{name}' not found "
                      "(loc=-1)")
            self._locs[name] = loc
        return loc

    def _set_f(self, name: str, v: float) -> None:
        rl.set_shader_value(self.shader, self._loc(name),
                            rl.ffi.new("float[1]", [float(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def _set_i(self, name: str, v: int) -> None:
        rl.set_shader_value(self.shader, self._loc(name),
                            rl.ffi.new("int[1]", [int(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def _set_vec(self, name: str, v) -> None:
        n = len(v)
        kind = {2: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2,
                3: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3,
                4: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4}[n]
        rl.set_shader_value(self.shader, self._loc(name),
                            rl.ffi.new(f"float[{n}]", [float(x) for x in v]),
                            kind)

    def set_srgb_decode(self, on: bool) -> None:
        self._set_i("u_srgb_decode", 1 if on else 0)

    def set_light_z(self, z: float) -> None:
        """Keep the water pass's L-reconstruction z in step with LightingPass."""
        self._set_f("u_light_z", max(0.0, min(1.5, float(z))))

    def set_ambient(self, rgb) -> None:
        """Push the global ambient (mirror LightingPass.set_ambient): the
        refracted floor and the faint sheen are lit by ``ambient + sources``
        like the dry floor. game_renderer keeps it in step every frame."""
        self._set_vec("u_ambient", tuple(float(c) for c in rgb[:3]))

    def set_art_align(self, offset_px, px_per_tile) -> None:
        """Bind the level's [art.align] transform (mirrors LightingPass)."""
        if isinstance(px_per_tile, (list, tuple)):
            ppt_x, ppt_y = float(px_per_tile[0]), float(px_per_tile[1])
        else:
            ppt_x = ppt_y = float(px_per_tile)
        self.art_align = (float(offset_px[0]), float(offset_px[1]),
                          ppt_x, ppt_y)

    def set_height_texture(self, tex: Optional[rl.Texture]) -> None:
        """Bind the level's static floor heightmap (greyscale relief) once at
        load; None disables the attenuation (the dormant default)."""
        self._height_tex = tex
        self._has_height = tex is not None
        self._set_i("u_has_height", 1 if self._has_height else 0)

    # ---- settings (the one door's values, pushed when they change) ------

    def refresh_settings(self, s: WaterSettings) -> None:
        """Push ``[graphics.water]`` (``water_surface.water_settings``) to the
        shader; a frame with unchanged settings costs one compare."""
        if s == self.settings:
            return
        self.settings = s
        for name in ("roughness_base", "roughness_agitation", "fog_density",
                     "refract_strength", "r0", "ripple_scale",
                     "glint_strength", "alpha_scale", "alpha_min",
                     "alpha_max", "ca_amount",
                     "height_scale", "height_edge", "height_floor"):
            uname = {"roughness_base": "u_roughness_base",
                     "roughness_agitation": "u_roughness_agitation",
                     "fog_density": "u_fog_density",
                     "refract_strength": "u_refract_strength",
                     "glint_strength": "u_glint_strength",
                     "ca_amount": "u_ca_amount"}.get(name, "u_" + name)
            self._set_f(uname, getattr(s, name))
        self._set_vec("u_water_color", s.water_color)
        # W4: the caustic (strength, max, the refraction law's 1 - 1/n)
        self._set_vec("u_caustic", (s.caustic_strength, s.caustic_max,
                                    CAUSTIC_K, 0.0))
        self._set_f("u_agit_ref", s.agitation_ref)
        self._set_i("u_debug_view", s.debug_view)
        from .noise_bake import FOAM_CELLS
        self._set_vec("u_foam_a", (s.foam_bubble_m[0] * FOAM_CELLS,
                                   s.foam_bubble_m[1] * FOAM_CELLS,
                                   s.foam_cover, s.foam_shadow))
        self._set_vec("u_foam_b", (water_foam.FOAM_EDGE,
                                   water_foam.FOAM_ALBEDO,
                                   water_foam.FOAM_LARGE_OFFSET[0],
                                   water_foam.FOAM_LARGE_OFFSET[1]))
        self.clock.set_dt(s.foam_step_s)
        self._push_frame()

    def _push_frame(self) -> None:
        s, f = self.settings, self.frame
        self._set_vec("u_grid_m", (self.w * self.tile_size_m,
                                   self.h * self.tile_size_m))
        self._set_vec("u_detail", (s.period_m, s.detail_strength_still,
                                   s.detail_strength_flow, f.fade))
        self._set_vec("u_flow_ph", (f.phase, f.cycle_s, f.drift_u, f.drift_v))
        self._set_vec("u_flow_lim", (s.flow_max_offset_m, s.flow_edge_depth_m,
                                     s.flow_phase_noise_m, s.flow_speed_ref))

    # ---- per-frame update ----------------------------------------------

    def update(self, *, depth_q: np.ndarray, flow_vx_q: np.ndarray,
               flow_vy_q: np.ndarray, solid: np.ndarray,
               ripple: Optional[np.ndarray], ripple_v: Optional[np.ndarray],
               sim_tick: int, alpha: Optional[float] = None,
               paused: bool = False, tps: float = 24.0,
               tile_size_m: Optional[float] = None,
               zoom_px_per_tile: float = 32.0) -> None:
        """Per frame: advance the sim clock; on a DRY map stop after one
        ``.any()`` on the int depth plane (no dequantize, no pack, no
        upload); otherwise re-pack the two grid textures only when the sim
        state changed -- once per tick while running, CRC-keyed on the int
        planes while paused -- and set the frame's flow uniforms. Every field
        is read, never written."""
        t0 = time.perf_counter()
        if tile_size_m is not None:
            self.tile_size_m = float(tile_size_m)
        wall_dt = 0.0 if self._wall_prev is None else t0 - self._wall_prev
        self._wall_prev = t0
        if self._last_tick is not None and int(sim_tick) < self._last_tick:
            self._foam_reset = True          # a seek / a new sim: start over
            self._have_prev = False
        self._last_tick = int(sim_tick)
        n_steps = self.clock.advance(int(sim_tick), alpha, float(tps),
                                     paused=paused, idle_when_paused=True,
                                     wall_dt=wall_dt)
        if not np.asarray(depth_q).any():
            if self.wet:
                self._foam_reset = True      # dried out: nothing survives
            self.wet = False
            self.box = None
            self._key = None
            self._foam_steps = 0
            self.last_update_ms = (time.perf_counter() - t0) * 1000.0
            return
        if not paused:
            key = ("tick", int(sim_tick))
        else:
            key = ("crc",) + tuple(
                zlib.crc32(np.ascontiguousarray(a).view(np.uint8))
                for a in (depth_q, flow_vx_q, flow_vy_q, solid))
        if key != self._key or not self.wet:
            self._key = key
            self._pack(depth_q, flow_vx_q, flow_vy_q, solid, ripple, ripple_v)
        self.wet = True
        self._foam_steps = min(self._foam_steps + n_steps,
                               water_foam.FOAM_STEP_CAP)
        self.frame = flow_frame(self.clock.t_noise, float(tps),
                                float(zoom_px_per_tile), self.settings)
        self._push_frame()
        self.last_update_ms = (time.perf_counter() - t0) * 1000.0

    def _pack(self, depth_q, flow_vx_q, flow_vy_q, solid, ripple,
              ripple_v) -> None:
        from simulation import water_fixed
        depth = water_fixed.dequantize_f32(depth_q)
        self.packed[..., 0] = ripple if ripple is not None else 0.0
        self.packed[..., 1] = ripple_v if ripple_v is not None else 0.0
        self.packed[..., 2] = depth
        # the agitation (W1's strength, W2's foam): ripple_v against its
        # wet neighbours' -- a room heaving as one is still
        self.packed[..., 3] = (
            0.0 if ripple_v is None else relative_agitation(
                ripple_v, (depth > 0) & ~np.asarray(solid, bool)))
        core.update_rgba16f_texture(self.water_tex, self.packed)
        # the previous TICK's depth (a paused repack keeps it); the first
        # packed tick has none, so it is its own (no advancing edge at load:
        # a standing pool must not foam over on its first frame)
        if not self._have_prev:
            self._prev_depth = self._last_depth = depth
            self._have_prev = True
        elif self._key[0] == "tick":
            self._prev_depth, self._last_depth = self._last_depth, depth
        else:
            self._last_depth = depth
        pack_flow(water_fixed.dequantize_f32(flow_vx_q),
                  water_fixed.dequantize_f32(flow_vy_q), solid,
                  self._prev_depth, self.flow_packed)
        core.update_rgba16f_texture(self.flow_tex, self.flow_packed)
        self.box = wet_box(depth_q)
        self._v_max = water_foam.flow_max(self.flow_packed)

    # ---- the foam step (W2; outside any texture mode) --------------------

    def reset_foam(self) -> None:
        """Forget every foam cell at the next :meth:`simulate` (a replay seek
        does this through a backward tick; a caller may ask directly)."""
        self._foam_reset = True

    def _step_set(self, gp, name: str, *v) -> None:
        """A uniform of the step program, sent only when it changed."""
        loc = self._step_locs.get(name)
        if loc is None:
            loc = self._step_locs[name] = rl.get_shader_location(
                self._step_shader, name)
        if loc < 0 or self._step_ucache.get(name) == v:
            return
        self._step_ucache[name] = v
        gl = gp.gl
        if len(v) == 1 and isinstance(v[0], int):
            gl.Uniform1i(loc, v[0])
        elif len(v) == 1:
            gl.Uniform1f(loc, float(v[0]))
        elif len(v) == 2:
            gl.Uniform2f(loc, float(v[0]), float(v[1]))
        elif len(v) == 4:
            gl.Uniform4fv(loc, 1, gl.floats(v))
        else:
            raise ValueError(name)

    def _ensure_foam(self, m: int) -> None:
        if self.foam is not None and self.foam_m == m:
            return
        if self.foam is not None:              # a new cell size: new image
            self.foam.unload()
        self.foam = core.PingPong(self.w * m, self.h * m, fp32=True)
        self.foam.set_filter(True)             # the draw reads it bilinear
        self.foam_m = m
        self._foam_box = None
        self._foam_reset = False               # born zeroed
        if self._step_shader is None:
            self._step_shader = core.load_pass_shader(load_foam_fs())

    def simulate(self) -> int:
        """Spend the foam steps this frame owes (``update`` counted them on
        the sim clock): each step ``water_foam.substeps`` passes of
        ``shaders/water_foam.fs`` over the wet box in cells, in ONE direct-GL
        block. Must run OUTSIDE any texture mode (the renderer calls it
        before the scene RT is begun, beside the flame's). A dry map, or a
        frame that owes nothing, issues nothing; the foam image is allocated
        on the first wet frame. Returns the pass count."""
        t0 = time.perf_counter()
        self.foam_steps_last_frame = 0
        self.foam_passes_last_frame = 0
        if self.foam is not None and self._foam_reset:
            self.foam.clear()
            self._foam_box = None
            self._foam_reset = False
        n_steps, self._foam_steps = self._foam_steps, 0
        if not self.wet or self.box is None or n_steps <= 0:
            self.last_simulate_ms = (time.perf_counter() - t0) * 1000.0
            return 0
        s = self.settings
        m = int(s.foam_cells_per_tile)
        self._ensure_foam(m)
        box = water_foam.cell_box(self.box, m)
        leaving = water_foam.leaving_rects(self._foam_box, box)
        if leaving:
            self.foam.clear_rects(leaving)
        self._foam_box = box
        n_sub = water_foam.substeps(self._v_max, s.foam_step_s, m,
                                    self.tile_size_m)
        dt = s.foam_step_s / n_sub
        x0, y0, x1, y1 = box
        foam = self.foam
        with core.DirectPasses() as gp:
            gp.use(self._step_shader)
            self._step_set(gp, "u_boxes", float(x0), float(y0), float(x1),
                           float(y1))
            self._step_set(gp, "u_target", float(foam.width),
                           float(foam.height))
            self._step_set(gp, "u_grid", float(self.w), float(self.h))
            self._step_set(gp, "u_m", float(m))
            self._step_set(gp, "u_dt", dt)
            self._step_set(gp, "u_cell_k", dt * m / self.tile_size_m)
            self._step_set(gp, "u_decay_f", math.exp(-dt / s.foam_decay_s))
            self._step_set(gp, "u_src_a", s.foam_gain_froude, s.foam_froude,
                           s.foam_gain_steep, s.foam_gain_edge)
            self._step_set(gp, "u_src_b", s.foam_gain_wall, s.flow_speed_ref,
                           s.agitation_ref, s.flow_edge_depth_m)
            for name, unit in _STEP_UNITS.items():
                self._step_set(gp, name, unit)
            gp.bind(_STEP_UNITS["u_water"], self.water_tex.id)
            gp.bind(_STEP_UNITS["u_flow"], self.flow_tex.id)
            for _ in range(n_steps * n_sub):
                gp.bind(_STEP_UNITS["u_foam"], foam.front_id)
                foam.direct_pass(gp, 1)
        self.foam_steps_last_frame = n_steps
        self.foam_passes_last_frame = n_steps * n_sub
        self.last_simulate_ms = (time.perf_counter() - t0) * 1000.0
        return self.foam_passes_last_frame

    # ---- draw ----------------------------------------------------------

    def draw_rect(self, view_tiles: Optional[Sequence[float]]):
        """The tile rectangle this frame draws: the wet box ∩ the view
        (``(x0, y0, x1, y1)`` tiles, floats allowed; None = no clip), or
        None when there is nothing to draw."""
        if not self.wet or self.box is None:
            return None
        x0, y0, x1, y1 = self.box
        if view_tiles is not None:
            vx0, vy0, vx1, vy1 = view_tiles
            x0 = max(x0, int(np.floor(vx0)))
            y0 = max(y0, int(np.floor(vy0)))
            x1 = min(x1, int(np.ceil(vx1)))
            y1 = min(y1, int(np.ceil(vy1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def draw(self, diffuse: rl.Texture, light_tex_a: rl.Texture,
             light_tex_b: rl.Texture, world_px_w: int, world_px_h: int, *,
             view_tiles: Optional[Sequence[float]] = None,
             smoke_cast=None, light_curve=None,
             light_gain: float = 1.0) -> bool:
        """Draw the water over the wet box ∩ the view of the world RT;
        returns whether anything was drawn (nothing on a dry map).

        Caller must already be inside BeginTextureMode(world_rt), after the
        lit ship and before the floor smoke layer. Premultiplied blend
        (``rgb = base*alpha + glint``, ``a = alpha``): the base composites
        over the floor, the GGX glint rides on top as additive HDR light.
        ``smoke_cast`` is ``GameRenderer._smoke_cast`` (the cast product and
        its uniforms; None = no shadow), ``light_curve`` the lighting pass's
        ``curve_uniforms`` (None = off), ``light_gain`` its exposure.

        Samplers beyond raylib's four are bound on their own units (5, 6, 7)
        through rlgl, inside BeginShaderMode (raylib gotcha: a
        SetShaderValueTexture before BeginShaderMode can target the wrong
        shader -- LightingPass.draw_lit_world)."""
        rect = self.draw_rect(view_tiles)
        if rect is None:
            return False
        from .lit3d import set_light_curve_uniforms, set_smoke_shadow_uniforms
        cover = None
        if smoke_cast is not None:
            cover, grid, cast, cast2 = smoke_cast
            if cover is None:
                cast = (0.0, 0.0, 0.0, 0.0)
            set_smoke_shadow_uniforms(self.shader, grid, cast, cast2)
        else:
            set_smoke_shadow_uniforms(self.shader, (float(self.w),
                                                    float(self.h)),
                                      (0.0, 0.0, 0.0, 0.0),
                                      (1.0, 0.0, 0.0, 0.0))
        # the curve's 256-float LUT is re-sent only when the lighting pass
        # re-baked it (a new tuple), not every frame
        if light_curve is not self._curve_sent:
            set_light_curve_uniforms(self.shader, light_curve)
            self._curve_sent = light_curve
        self._set_f("u_light_gain", light_gain)

        sx, sy, sw, sh = art_src_and_uv_rect(
            self.art_align, self.w, self.h,
            float(diffuse.width), float(diffuse.height))[0]
        uv_rect = art_src_and_uv_rect(
            self.art_align, self.w, self.h,
            float(diffuse.width), float(diffuse.height))[1]
        x0, y0, x1, y1 = rect
        src = rl.Rectangle(sx + sw * x0 / self.w, sy + sh * y0 / self.h,
                           sw * (x1 - x0) / self.w, sh * (y1 - y0) / self.h)
        px_x = float(world_px_w) / self.w
        px_y = float(world_px_h) / self.h
        dst = rl.Rectangle(x0 * px_x, y0 * px_y, (x1 - x0) * px_x,
                           (y1 - y0) * px_y)

        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.begin_shader_mode(self.shader)
        rl.set_shader_value_texture(self.shader, self._loc("u_diffuse"),
                                    diffuse)
        rl.set_shader_value_texture(self.shader, self._loc("u_light_a"),
                                    light_tex_a)
        rl.set_shader_value_texture(self.shader, self._loc("u_light_b"),
                                    light_tex_b)
        rl.set_shader_value_texture(self.shader, self._loc("u_water"),
                                    self.water_tex)
        # The height sampler is always backed (the diffuse as an inert
        # stand-in when the level has no heightmap: u_has_height == 0 never
        # reads it).
        rl.set_shader_value_texture(
            self.shader, self._loc("u_height"),
            self._height_tex if self._height_tex is not None else diffuse)
        self._set_vec("u_art_uv_rect", uv_rect)
        # W2: the foam image (none yet: the flow texture stands in and
        # u_foam_on = 0 never reads it) and the bubble bake
        self._set_i("u_foam_on", 1 if self.foam is not None else 0)
        units = ((_UNIT_SMOKE, "u_smoke_cover", cover),
                 (_UNIT_FLOW, "u_flow", self.flow_tex),
                 (_UNIT_RIPPLE, "u_ripple", self.ripple_tex),
                 (_UNIT_CURV, "u_ripple_curv", self.ripple_curv_tex),
                 (_UNIT_FOAM, "u_foam",
                  self.foam.front if self.foam is not None else None),
                 (_UNIT_BUBBLES, "u_bubbles", self.bubble_tex))
        for unit, name, tex in units:
            rl.rl_active_texture_slot(unit)
            # no cast product / no foam yet: the flow texture stands in
            # (strengths are 0 / u_foam_on is 0, so nothing reads it)
            rl.rl_enable_texture((tex if tex is not None
                                  else self.flow_tex).id)
            self._set_i(name, unit)
        rl.rl_active_texture_slot(0)
        rl.draw_texture_pro(diffuse, src, dst, rl.Vector2(0, 0), 0.0,
                            rl.WHITE)
        rl.end_shader_mode()
        rl.end_blend_mode()
        for unit, _name, _tex in units:
            rl.rl_active_texture_slot(unit)
            rl.rl_disable_texture()
        rl.rl_active_texture_slot(0)
        return True

    def unload(self) -> None:
        rl.unload_shader(self.shader)
        rl.unload_texture(self.water_tex)
        rl.unload_texture(self.flow_tex)
        rl.unload_texture(self.ripple_tex)
        rl.unload_texture(self.ripple_curv_tex)
        rl.unload_texture(self.bubble_tex)
        if self.foam is not None:
            self.foam.unload()
            self.foam = None
        if self._step_shader is not None:
            rl.unload_shader(self._step_shader)
            self._step_shader = None


__all__ = ["WaterPass", "load_water_fs", "load_foam_fs"]
