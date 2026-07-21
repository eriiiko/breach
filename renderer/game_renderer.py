"""GameRenderer: thin orchestrator over core/lighting/overlays/world_composite.

Public API (used by main.py):
    renderer.upload_state(gmap, light_sources)
    renderer.begin_frame()
    renderer.compose_world(...)        # everything in world space, inside RT
    renderer.blit_world_to_screen()    # camera blit from RT to map area
    renderer.draw_panel(state)         # right-side UI on top of screen
    renderer.end_frame()

State flow:
    Game logic ---> upload_state ---> textures (light/smoke/fire)
    Render: compose_world -> world RT -> blit_to_screen -> screen
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pyray as rl

from config import CFG
# The composed behavior flags (mechanics/06 §4) — pure read of unit statuses;
# drives the minimal prone visual (P4: knocked-down units render sideways).
from simulation.status import composed_flags

from . import core
from .camera import Camera2D
from .lighting import LightingPass
from .overlays import (
    FieldOverlay, FireOverlay, GlowOverlay, HeatFieldOverlay,
    WaterFieldOverlay,
    draw_unit, draw_waypoint_line, draw_grid, draw_text, draw_panel_background,
)
from .pressure_overlay import PressureOverlay
from .sprites import UnitSprites
from .unit_model_renderer import UnitModelRenderer
from .water import WaterPass
from .world_composite import WorldComposite


@dataclass
class RenderConfig:
    """Static configuration set at construction time. Camera state lives on
    the Camera2D object, not here — keep RenderConfig immutable after init.

    `world_px_per_tile` is the only place this value is configured; the
    WorldComposite reads it via cfg.world_px_per_tile at construction. Do
    not duplicate this value elsewhere — single source of truth.
    """
    map_px_w: int          # pixels for the map area (window minus panel)
    map_px_h: int
    panel_px_w: int        # right-side panel width
    grid_w: int            # physics grid width in tiles
    grid_h: int            # physics grid height in tiles
    world_px_per_tile: float = 24.0   # world RT resolution (independent of zoom)
    # Phase-0 3D marines (docs/anim_phase0_impl_2026-07-20.md): render-only,
    # default OFF. When True, GameRenderer loads a rigged glTF and draws units
    # as animated 3D bodies at the unit-draw slot instead of the 2D sprites;
    # when False the sprite path is byte-for-byte unchanged. Feel-gated. The J
    # key flips it live (poll_toggles) once the model is loaded.
    use_3d_units: bool = False


class GameRenderer:
    """Encapsulates pyray drawing. One per game session."""

    def __init__(self, level_data, breach_physics, cfg: RenderConfig,
                 initial_camera: Optional[Camera2D] = None,
                 borderless: bool = False):
        self.level = level_data
        self.bp = breach_physics
        self.cfg = cfg

        total_w = cfg.map_px_w + cfg.panel_px_w
        total_h = cfg.map_px_h
        core.init_window(total_w, total_h,
                         title=f"Breach — {level_data.name}",
                         borderless=borderless)

        # Camera (default: top-left of world, zoom set to fit width or fixed)
        if initial_camera is not None:
            self.camera = initial_camera
        else:
            # Pick a zoom such that the world width fits in the viewport
            default_zoom = cfg.map_px_w / cfg.grid_w
            self.camera = Camera2D(
                pos_tile_x=0.0, pos_tile_y=0.0,
                zoom_px_per_tile=default_zoom,
                viewport_px_w=cfg.map_px_w, viewport_px_h=cfg.map_px_h,
                world_size_tile_w=cfg.grid_w, world_size_tile_h=cfg.grid_h,
            )

        # World RT — all world-space draws go into this.
        self.world = WorldComposite(
            world_tile_w=cfg.grid_w, world_tile_h=cfg.grid_h,
            world_px_per_tile=cfg.world_px_per_tile,
        )

        # Level textures + lighting + overlays
        self.textures = core.load_level_textures(level_data)
        self.raycaster = breach_physics.Raycaster()
        # Smoke optics: decoupled per-channel Beer-Lambert absorption + a
        # SEPARATE additive scatter/glow budget (ch.05 §6.1 §6). Bound from
        # the [smoke] config section so the look is tunable live (F5 reload).
        #   transmission:  trans_c = exp(-absorption[c] * density * absorb_scale)
        #   scatter/glow:  smoke_glow[c] += local_light[c] * scatter_albedo[c] * density
        # Dialing smoke_absorb_scale DOWN gives the long-beam "flashlight travels
        # far through smoke and still glows" look (the beam survives deep smoke
        # because exp(-tau) never hits 0). Defaults approximate the shipped look.
        smoke_cfg = getattr(CFG, "smoke", None)
        self.raycaster.smoke_absorption_rgb = tuple(
            getattr(smoke_cfg, "smoke_absorption", (1.0, 1.0, 1.0)))
        self.raycaster.smoke_scatter_albedo = tuple(
            getattr(smoke_cfg, "smoke_scatter_albedo", (1.0, 1.0, 1.0)))
        self.raycaster.smoke_absorb_scale = float(
            getattr(smoke_cfg, "smoke_absorb_scale", 1.4))
        self.lighting = LightingPass(self.raycaster, cfg.grid_h, cfg.grid_w)
        # [art.align] (level format v2 §1.3): bind the level's explicit art
        # alignment so draw_lit_world samples the art through the transform
        # (art pixel offset_px lands on grid (0,0); px_per_tile art pixels
        # span one tile). Levels without an explicit [art.align] — every v1
        # level — keep LightingPass's legacy stretch-art-to-grid draw, which
        # is bit-identical to the pre-F2 output. getattr guards stub levels.
        if (getattr(level_data, "art_align_explicit", False)
                and level_data.art_px_per_tile):
            self.lighting.set_art_align(level_data.art_offset_px,
                                        level_data.art_px_per_tile)
        # Upload the level's vacuum mask once — the shader uses it to discard
        # vacuum pixels so the screen-fixed background shows through.
        from level_loader import materials_from_tilemap
        _mat, vacuum_mask = materials_from_tilemap(level_data.tilemap,
                                                   level_data.version)
        self.lighting.set_vacuum_mask(vacuum_mask)
        # BC render tint (boundary_conditions_spec_2026-07-19 B5) — RENDER-ONLY,
        # determinism-EXEMPT: it never reads or writes any synced sim field, so
        # it cannot perturb a golden/lockstep. On a planetside `boundary ==
        # "ambient"` map the SPACE-code ring tiles are the open SKY, not vacuum;
        # the vacuum shader still discards them (materials_from_tilemap marks the
        # SPACE code as the vacuum mask regardless of boundary), so we paint a
        # soft daylight gradient BEHIND the map instead of the starfield/black
        # backdrop and the sky shows through the ring. Space maps: flag False ->
        # the exact prior background path (byte-identical render).
        self._ambient_sky_map = (
            getattr(level_data, "boundary", "space") == "ambient")
        # smoke^gamma render-contrast knob (ch.05 §6.1 step 5): a power curve on
        # the RENDERED smoke opacity (FieldOverlay.update), not the sim field.
        # gamma > 1 crushes thin smoke toward transparent and sharpens wispy
        # edges (filmic), killing the flat-fog look; 1.0 = off. Bound from
        # [smoke] smoke_render_gamma (default 1.5). Like the other [smoke] optics
        # above it is read at init; Ctrl+R config reload does not re-push it
        # (consistent with the renderer's other smoke params — see ch.12). The
        # tuning tool (tools/lighting_demo.py) re-pushes it live via a slider.
        smoke_render_gamma = float(
            getattr(smoke_cfg, "smoke_render_gamma", 1.5))
        self.smoke_overlay = FieldOverlay(cfg.grid_h, cfg.grid_w,
                                          tint=(190, 195, 210), max_alpha=180,
                                          gamma=smoke_render_gamma)
        self.fire_overlay = FireOverlay(cfg.grid_h, cfg.grid_w)
        # God-ray / lit-smoke glow (ch.05): additive shaft from the ray march's
        # smoke_glow output. Supersedes the retired light_modulation surface-tint.
        self.glow_overlay = GlowOverlay(cfg.grid_h, cfg.grid_w)
        self.pressure_overlay = PressureOverlay(cfg.grid_h, cfg.grid_w)
        # Debug temperature overlay (engine/06): black-body ramp over
        # gmap.temperature. temp_display_max = the ΔT that maps to white-hot;
        # default ~300 == the wood ignition_temp so an igniting tile reads at
        # the top of the ramp. Tunable via [display] temp_display_max. Off by
        # default; toggled with T. RENDER-ONLY — never mutates the field.
        temp_display_max = float(
            getattr(getattr(CFG, "display", None), "temp_display_max", 300.0))
        self.temperature_overlay = HeatFieldOverlay(
            cfg.grid_h, cfg.grid_w, temp_display_max=temp_display_max)
        # Water overlay v2 (water W6b; canon engine/07 §6 placeholder): depth-
        # blue tint + ripple shading + foam + ambient sines over the sim's
        # water fields. All four knobs bind from [display] with getattr
        # defaults (the W2b water_display_max precedent) and are RENDER-ONLY +
        # RESTART-BOUND: read once here; Ctrl+R re-reads config.toml but never
        # re-binds the renderer's overlays. Off by default; toggled with V
        # (moved from O in A6 — O is the dev door-toggle key).
        # RENDER-ONLY — never mutates any field.
        disp = getattr(CFG, "display", None)
        self.water_overlay = WaterFieldOverlay(
            cfg.grid_h, cfg.grid_w,
            depth_display_max=float(getattr(disp, "water_display_max", 1.0)),
            ripple_shade=float(getattr(disp, "water_ripple_shade", 0.35)),
            foam_thresh=float(getattr(disp, "water_foam_thresh", 0.03)),
            ambient_base=float(getattr(disp, "water_ambient_base", 0.06)))
        # Water OPTICS pass (graphics/water_rendering.md §7 steps 1+2): the
        # shipped GLSL Fresnel/GGX/refraction look that REPLACES the CPU
        # WaterFieldOverlay placeholder in the same compose slot. Binds the
        # [graphics.water] uniforms once; reuses the lighting pass's light
        # textures + the level diffuse (no new plumbing). DORMANT-SAFE: the
        # shader emits alpha 0 on dry tiles, so a ship with no standing water
        # renders identically (the safety property the gate checks). The
        # WaterFieldOverlay above is kept constructed (so nothing else breaks)
        # but is no longer drawn — the GLSL pass is the live water render.
        self.water_pass = WaterPass(cfg.grid_h, cfg.grid_w)
        if (getattr(level_data, "art_align_explicit", False)
                and level_data.art_px_per_tile):
            self.water_pass.set_art_align(level_data.art_offset_px,
                                          level_data.art_px_per_tile)
        # Heightmap attenuation (graphics/water_rendering.md §2 §8): bind the
        # level's static floor relief so the water pass fades its alpha where
        # raised features poke above the surface (alpha-only — not depth/tint).
        # Levels with no [art.bare] height load textures.height == None, which
        # leaves u_has_height = 0 and the pass behaves exactly as before (no
        # attenuation — the dormant default the default level relies on).
        self.water_pass.set_height_texture(getattr(self.textures, "height", None))

        # Toggles
        self.show_grid = False
        self.show_smoke = True
        self.show_fire = True
        self.show_lighting = True
        self.show_normal_map = True
        self.normal_y_flipped = False
        self.srgb_decode = True
        self.show_debug_coords = False
        # Pressure colormap defaults ON in the main game — explosions
        # look dramatic by default. Toggle with F7.
        self.show_pressure = True
        # Debug temperature overlay (engine/06) — OFF by default; toggle with T.
        self.show_temperature = False
        # Water OPTICS pass (graphics/water_rendering.md) — ON by default; the
        # GLSL pass is dormant-safe (alpha 0 on dry tiles), so a dry ship looks
        # identical whether it's on or off. Toggle with O to disable the water
        # render entirely (e.g. to A/B against the bare floor).
        self.show_water = True

        # Frame timing
        self.last_frame_ms = 0.0
        self.last_raycast_ms = 0.0
        # Render-animation epoch: the water overlay's ambient sines take a
        # seconds clock; epoch-relative keeps the float32 sine phases small.
        # Wall-clock (animates through pause) — render-only, determinism-
        # irrelevant by the locked canon §6 visual-only rule.
        self._anim_t0 = time.perf_counter()

        # Visual effects list (short-lived). Each entry is a dict with
        # "kind", lifecycle ("t" seconds elapsed, "life" total) and
        # kind-specific payload (e.g. "from"/"to" for tracers). Populated
        # by consume_events, advanced by _advance_effects, drawn by
        # _draw_effects_world.
        self._effects: list = []

        # S2b render FLOAT BRIDGE scratch: gmap.gas is int32 Q16.16, but the C++
        # raycaster gas optics are float — dequantize the (N,h,w) planes into this
        # reused float32 buffer each frame the cast runs (allocated lazily on first
        # use; resized if the grid shape changes).
        self._gas_render_f = None

        self.lighting.set_ambient((0.18, 0.18, 0.22))

        # Unit sprites — loaded once, unloaded in shutdown().
        self.sprites = UnitSprites()
        self.sprites.load()

        # Phase-0 3D marines (render-only, toggle-gated). The 2D sprite path
        # above is always loaded (the toggle can flip live, and it's the
        # fallback if the model fails to load). The model + its clips are loaded
        # ONLY when use_3d_units, so the sprite-only path pays no load cost. The
        # top-down Camera3D is framed to the world RT once. Per-unit animation
        # state lives inside UnitModelRenderer (keyed by unit.id) — never on Unit.
        self.unit_models = UnitModelRenderer()
        self._unit_cam3d = None
        if self.cfg.use_3d_units:
            self.unit_models.load()
            self._unit_cam3d = UnitModelRenderer.make_camera(
                self.world.world_px_w, self.world.world_px_h)

    # ---- per-frame physics->GPU upload ---------------------------------

    def upload_state(self, gmap, light_sources: Optional[List] = None) -> None:
        t_start = time.perf_counter()

        # Light field. Occlusion is the per-channel DYNAMIC attenuation field
        # (ch.03 §units, ch.02 §static×dynamic): pass `gmap.dyn_light_atten`
        # (h, w, 3) = static material attenuation MAX'd with stamped-unit
        # opacity, rebuilt each tick in `stamp_units`. Opaque walls/units
        # ([1,1,1]) block exactly like the old wall hard-stop; glass transmits.
        # Away from units it equals the static field, so behaviour matches S2;
        # over a unit footprint it restores the pre-S2 unit shadow.
        if self.show_lighting and light_sources:
            t_ray = time.perf_counter()
            # Pass gmap.heat (Q16.16 deposit) and gmap.smoke_glow (god-ray
            # glow) so the march writes both Slice-4 outputs in-place. The cast
            # still lives here in the renderer; it moves into the sim in S5.
            # Multi-gas coloured optics (engine/05 §6.2): pass the full (N,h,w)
            # gas array + the per-gas absorption/scatter tables so the march sums
            # all gases density-weighted per channel (poison greens the beam,
            # black_smoke dims it, mixing falls out of the sum). `gmap.smoke` is
            # just the black_smoke slice of `gmap.gas`.
            # S2b: gmap.gas is int32 Q16.16 — DEQUANTIZE the (N,h,w) array to
            # float32 density for the render cast (the C++ raycaster's gas optics
            # are float; render-only FLOAT BRIDGE, one source of truth = the int
            # field). Reused scratch to avoid a per-frame alloc.
            from simulation import gas_fixed
            if (self._gas_render_f is None
                    or self._gas_render_f.shape != gmap.gas.shape):
                self._gas_render_f = np.empty(gmap.gas.shape, dtype=np.float32)
            np.multiply(gmap.gas, 1.0 / gas_fixed.FP_ONE_F,
                        out=self._gas_render_f, casting="unsafe")
            self.lighting.compute_light_field(
                light_sources, self._gas_render_f,
                gmap.gases.absorption, gmap.gases.scatter_albedo,
                gmap.dyn_light_atten,
                heat=gmap.heat, smoke_glow=gmap.smoke_glow,
                heat_atten=gmap.heat_atten,
            )
            self.last_raycast_ms = (time.perf_counter() - t_ray) * 1000
        else:
            self.lighting.light_rgb.fill(0)
            self.lighting.light_map.fill(0)
            self.lighting.light_dx.fill(0)
            self.lighting.light_dy.fill(0)
            self.lighting.packed_a.fill(0)
            self.lighting.packed_b.fill(0)
            # No cast this frame -> no deposits. Clear the glow so a stale shaft
            # doesn't linger (heat is sim-owned; left to the sim's cleanup).
            gmap.smoke_glow.fill(0)
            core.update_rgba16f_texture(self.lighting.light_tex_a,
                                        self.lighting.packed_a)
            core.update_rgba16f_texture(self.lighting.light_tex_b,
                                        self.lighting.packed_b)
            self.last_raycast_ms = 0.0

        # Smoke is drawn as a flat grey density medium — alpha is density-driven
        # only (always there regardless of light). The lit-smoke colour now
        # comes from the additive god-ray glow overlay (gmap.smoke_glow, the
        # energy the smoke scattered), NOT a surface-tint multiply — the old
        # light_modulation path is retired (ch.03 C16, ch.05 §God-rays).
        if self.show_smoke:
            # S2b: gmap.smoke is int32 Q16.16 (the BLACK_SMOKE view) — DEQUANTIZE
            # to float32 density for the overlay (render-only FLOAT BRIDGE). The
            # _gas_render_f scratch already holds the dequantized planes when the
            # cast ran this frame; recompute the smoke slice cheaply regardless so
            # the overlay is correct even when show_lighting is off.
            from simulation import gas_fixed
            self.smoke_overlay.update(gas_fixed.dequantize_f32(gmap.smoke))
            self.glow_overlay.update(gmap.smoke_glow)
        # Fire still gets the vacuum mask: combustion requires oxygen, so
        # fire physically cannot exist in vacuum. Keep this until the fire
        # sim is taught to extinguish at vacuum tiles directly.
        if self.show_fire:
            # S3a: gmap.fire is int32 Q16.16 — dequantize to real [0,1] for the
            # overlay (render-only boundary).
            from simulation import fire_fixed
            fire_real = fire_fixed.dequantize_f32(gmap.fire)
            fire_view = np.where(gmap.is_vacuum, 0.0, fire_real)
            self.fire_overlay.update(fire_view)
        # Pressure colormap: refresh the per-tile texture from the current
        # atmosphere + wave_p fields. Skipped when toggled off to save the
        # numpy work.
        if self.show_pressure:
            self.pressure_overlay.update(gmap)
        # Debug temperature overlay: refresh the black-body texture from the
        # Q16.16 temperature field. Skipped when toggled off to save the work.
        # Render-only — gmap.temperature is read, never written.
        if self.show_temperature:
            self.temperature_overlay.update(gmap.temperature)
        # Water overlay v2 (W6b): depth tint + ripple shading + foam +
        # ambient sines. Skipped when toggled off; the overlay itself also
        # early-outs when the ship is dry (zero-water fast path). Render-only
        # — every field is read, never written. `t_start` is this frame's
        # perf_counter sample from the top of upload_state (no extra clock
        # call); epoch-relative so the sine phases stay float32-small.
        # Water OPTICS pass (graphics/water_rendering.md): pack ripple +
        # water_depth into the RGBA16F water texture every frame. The pass is
        # the LIVE water render now (not a debug toggle), so it uploads
        # unconditionally — but it is DORMANT-SAFE (the shader emits alpha 0 on
        # dry tiles) and has its own dry-ship fast path (skips the upload when
        # there is no standing water and the texture is already blank), so a dry
        # ship costs one .any(). RENDER-ONLY — every field is read, never
        # written. Keep the light_z in step with the lighting pass so the
        # glints' L matches the ship's lighting.
        self.water_pass.set_light_z(self.lighting.light_z)
        # Keep the water's global ambient in step with the lit ship's ambient
        # (lighting.fs lights by `ambient + sources`; the water body must too,
        # else the refracted floor goes black outside any source and shallow
        # water reads as a void only visible in the flashlight). self.lighting
        # .ambient is the single source of truth, fed by the demo's ambient
        # sliders + config — so those sliders drive the water ambient as well.
        self.water_pass.set_ambient(self.lighting.ambient)
        # S1: water_depth is int32 Q16.16 metres — DEQUANTIZE to float32 metres
        # for the render pass (ripple/ripple_v stay float, render-only). This is
        # the renderer-boundary dequantize (one source of truth: the int field).
        from simulation import water_fixed
        self.water_pass.update(
            water_fixed.dequantize_f32(gmap.water_depth),
            ripple=gmap.ripple, ripple_v=gmap.ripple_v)

        self.lighting.set_use_normal(self.show_normal_map)
        self.last_frame_ms = (time.perf_counter() - t_start) * 1000

    # ---- frame lifecycle ------------------------------------------------

    def begin_frame(self) -> None:
        core.begin_frame(clear_color=(0, 0, 0, 255))

    def end_frame(self) -> None:
        core.end_frame()

    def should_close(self) -> bool:
        return core.should_close()

    # ---- world-space compose phase --------------------------------------

    def compose_world(self, units_marines: Sequence = (),
                      units_zombies: Sequence = (),
                      projectiles: Sequence = (),
                      orders_phase1: Optional[dict] = None,
                      orders_phase2: Optional[dict] = None,
                      current_phase: int = 0,
                      doors: Sequence = ()) -> None:
        """Draw every world-space layer into the world RT.

        Order: lit ship (diffuse + normal + light), smoke, fire, units,
        projectiles, waypoints, visual effects, grid. Each is drawn at
        world-pixel coordinates inside the RT — no camera math; the RT
        IS the world.

        Clear color is fully transparent so vacuum/breach areas (where the
        shader discards) show through to the screen-fixed background.

        ``projectiles`` is a sequence of objects with ``.fx``, ``.fy``,
        and ``.proj_type`` (the simulation's Projectile dataclass works).
        ``orders_phase1`` / ``orders_phase2`` are ``{unit_id: [(x, y), ...]}``
        waypoint polylines per phase. ``current_phase`` (0 or 1) controls
        which is drawn brighter; the other is dimmed.
        """
        self.world.begin(clear_color=(0, 0, 0, 0))

        # 1. Lit ship — covers the entire world RT
        if self.textures.diffuse:
            self.lighting.draw_lit_world(
                self.textures.diffuse,
                self.textures.normal,
                world_px_w=self.world.world_px_w,
                world_px_h=self.world.world_px_h,
            )

        # 2. Smoke + fire overlays — stretched to world RT bounds.
        # Smoke is packed with PREMULTIPLIED alpha (see FieldOverlay.update),
        # so we draw it with BLEND_ALPHA_PREMULTIPLY for correct Porter-Duff
        # compositing — preserves the destination alpha (ship stays opaque)
        # instead of Raylib's default BLEND_ALPHA which reduces dest alpha
        # when src alpha < 1.
        if self.show_smoke:
            rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
            self._draw_overlay_to_world(self.smoke_overlay.tex)
            rl.end_blend_mode()
            # God-ray glow: additive shaft composited WITH the smoke, before
            # units (ch.05 §God-rays). GlowOverlay.draw sets its own RGB-only
            # additive blend — additive passes must NOT write dest alpha, or
            # the world RT's alpha saturates and vacuum tiles render opaque
            # black under the premultiplied blit (overlays.
            # _begin_additive_rgb_only_blend). Not premultiplied. Supersedes
            # the retired light_modulation surface-tint.
            self.glow_overlay.draw(
                0, 0, self.world.world_px_w, self.world.world_px_h)
        if self.show_fire:
            rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
            self._draw_overlay_to_world(self.fire_overlay.tex)
            rl.end_blend_mode()
        # Pressure colormap (white shockwave + coloured radiating bands at
        # explosions). Off-by-toggle F7. The overlay was updated in
        # upload_state alongside smoke/fire to keep all per-frame work in
        # one place; here we just draw the precomputed texture.
        if self.show_pressure:
            self.pressure_overlay.draw_into_world_rt(
                self.world.world_px_w, self.world.world_px_h
            )
        # Debug temperature overlay (engine/06): additive black-body ramp over
        # gmap.temperature. Drawn AFTER the field overlays but BEFORE units so
        # units stay readable on top of the heat glow. Off-by-toggle (T).
        if self.show_temperature:
            self.temperature_overlay.draw(
                0, 0, self.world.world_px_w, self.world.world_px_h)
        # Water OPTICS pass (graphics/water_rendering.md §5): the GLSL
        # Fresnel/GGX/refraction pass, in the slot the CPU WaterFieldOverlay
        # used to draw (after the field overlays, before units — the squad
        # reads over the water; the lit ship beneath is the floor sample). It
        # binds the lighting pass's light textures + the level diffuse and
        # draws premultiplied; the shader emits alpha 0 on dry tiles so this is
        # a no-op with no standing water (dormant-safe). Off-by-toggle (V).
        if self.show_water and self.textures.diffuse:
            self.water_pass.draw(
                self.textures.diffuse,
                self.lighting.light_tex_a, self.lighting.light_tex_b,
                world_px_w=self.world.world_px_w,
                world_px_h=self.world.world_px_h,
                anim_t=time.perf_counter() - self._anim_t0)

        # 2b. A6 dev door draw (a6 doors design §14, N4 honest-minimal):
        # closed door tiles get a flat amber fill, open spans a thin
        # outline (something to aim the O key at), destroyed doors
        # nothing. Render-read only — `doors` is the sim's DoorRuntime
        # list (span/state/alive); no overlay system, no field writes.
        if doors:
            self._draw_doors_world(doors)

        # 3. Units, waypoints, projectiles, effects, grid — drawn in world-pixel space
        if orders_phase1 or orders_phase2:
            self._draw_orders_world(orders_phase1, orders_phase2, current_phase)
        self._draw_units_world(units_marines, units_zombies)
        if projectiles:
            self._draw_projectiles_world(projectiles)
        # Visual effects (tracers, explosions, hit splats) — driven by
        # tick events the renderer pulls from the sim via consume_events.
        self._draw_effects_world()
        if self.show_grid:
            self._draw_grid_world()

        self.world.end()

    # A6 dev door colors (render-only; the tileset's amber door reads).
    _DOOR_CLOSED_FILL = (208, 168, 62, 210)     # amber panel
    _DOOR_OPEN_OUTLINE = (208, 168, 62, 160)    # thin frame on the air gap

    def _draw_doors_world(self, doors: Sequence) -> None:
        """Flat per-tile fill (closed) / outline (open) for entity doors —
        the a6 design §14 honest-minimal draw. ``doors`` items expose
        ``span`` ((fy, fx) tiles), ``state`` (0 closed / 1 open / 2
        destroyed) and ``alive``; destroyed doors draw nothing (the wreck
        is just breached air)."""
        wpt = self.world.world_px_per_tile
        for d in doors:
            if not getattr(d, "alive", True):
                continue
            closed = int(getattr(d, "state", 0)) == 0
            for (fy, fx) in d.span:
                x = int(fx * wpt)
                y = int(fy * wpt)
                side = int(wpt) if wpt >= 1 else 1
                if closed:
                    rl.draw_rectangle(x, y, side, side,
                                      rl.Color(*self._DOOR_CLOSED_FILL))
                else:
                    rl.draw_rectangle_lines(
                        x, y, side, side,
                        rl.Color(*self._DOOR_OPEN_OUTLINE))

    def _draw_overlay_to_world(self, field_tex: rl.Texture) -> None:
        """Stretch a physics-resolution texture across the full world RT."""
        src = rl.Rectangle(0, 0, float(field_tex.width), float(field_tex.height))
        dst = rl.Rectangle(0, 0, float(self.world.world_px_w),
                           float(self.world.world_px_h))
        rl.draw_texture_pro(field_tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

    def _draw_units_world(self, marines: Sequence, zombies: Sequence) -> None:
        wpt = self.world.world_px_per_tile
        lmap = self.lighting.light_map
        H, W = lmap.shape
        # Same ambient floor the lit-ship shader uses (lighting.fs does
        # diffuse * (ambient + incoming_rgb * ndotl)). light_map is the
        # render-side scalar brightness (max over the RGB channels). Without
        # this, units in unlit rooms are pitch-black while the ship around
        # them still shows a faint ambient baseline — visually mismatched.
        amb = self.lighting.ambient
        amb_floor = (amb[0] + amb[1] + amb[2]) / 3.0
        def light_at(u):
            # Sample at the unit's center tile (footprint // 2 offset).
            fp = int(getattr(u, "footprint", 3))
            cx = int(u.x) + fp // 2
            cy = int(u.y) + fp // 2
            base = float(lmap[cy, cx]) if (0 <= cx < W and 0 <= cy < H) else 0.0
            return amb_floor + base

        # Patch 0 — per-channel RGB provider for the 3D marines. Sample the SAME
        # baked RGB light field the ship shader reads (self.lighting.light_rgb,
        # (H, W, 3) f32) at the unit's foot tile, add the vec3 ambient floor and
        # the ship's exposure gain, and return a per-channel multiplier in
        # [0, 1]. This carries light COLOUR (a red lamp reddens the marine) and
        # OCCLUSION (an unlit/shadowed room darkens it) — the parity the old
        # max-collapsed grey scalar (light_at) threw away. It is the flat
        # fallback; the P1 lit shader samples the field per-fragment instead.
        lrgb = self.lighting.light_rgb
        gain = self.lighting.light_gain
        def light_rgb_at(u):
            fp = int(getattr(u, "footprint", 3))
            cx = int(u.x) + fp // 2
            cy = int(u.y) + fp // 2
            if 0 <= cx < W and 0 <= cy < H:
                inc = lrgb[cy, cx]
                return (min(1.0, amb[0] + float(inc[0]) * gain),
                        min(1.0, amb[1] + float(inc[1]) * gain),
                        min(1.0, amb[2] + float(inc[2]) * gain))
            return (min(1.0, amb[0]), min(1.0, amb[1]), min(1.0, amb[2]))

        # Phase-0 3D marines: when toggled on AND the model loaded, draw units as
        # animated 3D bodies (nested begin_mode_3d inside the already-open world
        # RT) and skip the sprite path entirely. Marines green, zombies red — the
        # same read as the sprite tints. light_at keeps the local-light dimming.
        # If the model failed to load, draw_units no-ops and we fall through to
        # the unchanged sprite path below. This branch is the ONLY change to the
        # unit-draw slot; with use_3d_units False it is never entered.
        if self.cfg.use_3d_units and self.unit_models.ready \
                and self._unit_cam3d is not None:
            clock = time.perf_counter() - self._anim_t0
            # P1: hand the marine shader the SAME baked light field the ship is
            # lit by (light_tex_a/b) + world dims + the ship's ambient/gain/
            # normal-y-sign (single source of truth = self.lighting), so the
            # marine samples the field per-fragment and matches the ship beside
            # it. light_rgb_fn stays as the flat CPU fallback (shader off / not
            # compiled). normal_y_sign follows the ship's H-toggle.
            from renderer.unit_model_renderer import LightFieldCtx
            light_ctx = LightFieldCtx(
                tex_a=self.lighting.light_tex_a,
                tex_b=self.lighting.light_tex_b,
                world_px_w=float(self.world.world_px_w),
                world_px_h=float(self.world.world_px_h),
                ambient=self.lighting.ambient,
                light_gain=self.lighting.light_gain,
                normal_y_sign=(-1.0 if self.normal_y_flipped else 1.0),
            )
            self.unit_models.draw_units(
                marines, wpt, clock, self._unit_cam3d,
                base_tint=(90, 200, 90, 255), light_rgb_fn=light_rgb_at,
                light_ctx=light_ctx)
            self.unit_models.draw_units(
                zombies, wpt, clock, self._unit_cam3d,
                base_tint=(210, 70, 70, 255), light_rgb_fn=light_rgb_at,
                light_ctx=light_ctx)
            return

        for m in marines:
            if not getattr(m, "alive", True):
                continue
            # facing is now float radians; facing_compass() → "N"/"NE"/...
            compass = (m.facing_compass() if callable(getattr(m, "facing_compass", None))
                       else getattr(m, "facing", "N"))
            sprite = self.sprites.get_marine(compass)
            draw_unit(m.x, m.y, wpt, (60, 180, 60, 255),
                      label=getattr(m, "name", ""),
                      footprint_tiles=getattr(m, "footprint", 3),
                      sprite=sprite,
                      light_intensity=light_at(m),
                      is_prone=composed_flags(m).is_prone)
        for z in zombies:
            if not getattr(z, "alive", True):
                continue
            sprite = self.sprites.get_zombie(z)
            draw_unit(z.x, z.y, wpt, (200, 50, 50, 255),
                      footprint_tiles=getattr(z, "footprint", 3),
                      sprite=sprite,
                      light_intensity=light_at(z),
                      is_prone=composed_flags(z).is_prone)

    # Single hue (cyan) for both phases; the currently-planning phase is
    # drawn bright, the other phase dimmer. Same colour communicates
    # "they're both your plan"; brightness disambiguates which one is
    # active for editing.
    _PHASE_BRIGHT = (60, 200, 255, 230)
    _PHASE_DIM    = (60, 200, 255, 90)

    def _draw_orders_world(self, orders_p1: Optional[dict],
                            orders_p2: Optional[dict],
                            current_phase: int) -> None:
        wpt = self.world.world_px_per_tile

        def draw_lines(orders, color):
            if not orders:
                return
            for waypoints in orders.values():
                if len(waypoints) < 2:
                    continue
                for a, b in zip(waypoints, waypoints[1:]):
                    draw_waypoint_line(a, b, wpt, color=color)

        # Draw the non-current phase first so the current phase is on top.
        if current_phase == 0:
            draw_lines(orders_p2, self._PHASE_DIM)
            draw_lines(orders_p1, self._PHASE_BRIGHT)
        else:
            draw_lines(orders_p1, self._PHASE_DIM)
            draw_lines(orders_p2, self._PHASE_BRIGHT)

    def _draw_projectiles_world(self, projectiles: Sequence) -> None:
        """Draw each in-flight projectile as a small marker.

        Grenades = orange circle; future kinds (plasma, rockets) get their
        own simple shapes here. Detonated projectiles are skipped.
        """
        wpt = self.world.world_px_per_tile
        from renderer.coords import tile_to_world_px
        from simulation.orders import ORDER_GRENADE
        for proj in projectiles:
            if getattr(proj, "detonated", False):
                continue
            cx = tile_to_world_px(proj.fx + 0.5, wpt)
            cy = tile_to_world_px(proj.fy + 0.5, wpt)
            kind = getattr(proj, "proj_type", -1)
            if kind == ORDER_GRENADE:
                # Grenade: red-orange filled circle + a thin dark outline.
                r = max(2.0, 0.4 * wpt)
                rl.draw_circle(int(cx), int(cy), r, rl.Color(255, 120, 40, 255))
                rl.draw_circle_lines(int(cx), int(cy), r, rl.Color(40, 20, 0, 255))
            else:
                # Generic fallback: small yellow dot.
                rl.draw_circle(int(cx), int(cy), 3.0, rl.Color(255, 255, 100, 255))

    def _draw_effects_world(self) -> None:
        """Draw the renderer's short-lived visual effects (tracers, blasts).

        Effects are pushed by :meth:`consume_events` and tick down each
        frame via :meth:`_advance_effects`. Drawn into the world RT so
        the camera transforms them like everything else.
        """
        wpt = self.world.world_px_per_tile
        from renderer.coords import tile_to_world_px
        for fx in self._effects:
            t = fx["t"]
            life = fx["life"]
            alpha_norm = max(0.0, 1.0 - t / max(life, 1e-6))
            kind = fx["kind"]
            if kind == "tracer":
                a, b = fx["from"], fx["to"]
                x1 = tile_to_world_px(a[0], wpt)
                y1 = tile_to_world_px(a[1], wpt)
                x2 = tile_to_world_px(b[0], wpt)
                y2 = tile_to_world_px(b[1], wpt)
                col = rl.Color(255, 240, 160, int(220 * alpha_norm))
                rl.draw_line_ex(rl.Vector2(x1, y1), rl.Vector2(x2, y2),
                                1.5, col)
            elif kind == "beam":
                # W2 laser (LaserFiredEvent): a distinct beam — crimson core
                # with a thin white-hot centreline, thicker than a tracer.
                # Glow-as-light-source is deferred to the explosion-light
                # pass (mechanics/03 §8); this is just the line.
                a, b = fx["from"], fx["to"]
                x1 = tile_to_world_px(a[0], wpt)
                y1 = tile_to_world_px(a[1], wpt)
                x2 = tile_to_world_px(b[0], wpt)
                y2 = tile_to_world_px(b[1], wpt)
                outer = rl.Color(255, 40, 40, int(200 * alpha_norm))
                core = rl.Color(255, 220, 220, int(240 * alpha_norm))
                rl.draw_line_ex(rl.Vector2(x1, y1), rl.Vector2(x2, y2),
                                3.0, outer)
                rl.draw_line_ex(rl.Vector2(x1, y1), rl.Vector2(x2, y2),
                                1.0, core)
            elif kind == "explosion":
                pos = fx["pos"]
                radius = fx["radius"]
                cx = tile_to_world_px(pos[0] + 0.5, wpt)
                cy = tile_to_world_px(pos[1] + 0.5, wpt)
                # Expanding ring then fading flash.
                grow = 1.0 - alpha_norm    # 0 -> 1 as effect ages
                r_wpx = (0.3 + grow * radius) * wpt
                ring = rl.Color(255, 200, 100, int(220 * alpha_norm))
                rl.draw_circle_lines(int(cx), int(cy), r_wpx, ring)
                # Inner flash on first frames.
                if alpha_norm > 0.6:
                    flash = rl.Color(255, 255, 220, int(180 * (alpha_norm - 0.6) / 0.4))
                    rl.draw_circle(int(cx), int(cy), 0.5 * radius * wpt, flash)
            elif kind == "hit":
                pos = fx["pos"]
                cx = tile_to_world_px(pos[0] + 0.5, wpt)
                cy = tile_to_world_px(pos[1] + 0.5, wpt)
                col = rl.Color(255, 60, 60, int(220 * alpha_norm))
                rl.draw_circle(int(cx), int(cy), 4.0, col)
            elif kind == "jet":
                # W6 spray jet (SprayJetEvent): a translucent cone fan from
                # the sprayer toward the burst target — re-emitted every
                # deposit tick, so it reads as a continuous hose while the
                # burst lives and dies ~0.13 s after it ends. Flame = warm
                # two-layer fan; miasma = a single fainter sickly-green fan.
                # Render-side flicker only (time-based) — the sim is done.
                import math as _m
                ax = tile_to_world_px(fx["apex"][0], wpt)
                ay = tile_to_world_px(fx["apex"][1], wpt)
                dx = fx["target"][0] - fx["apex"][0]
                dy = fx["target"][1] - fx["apex"][1]
                d = _m.hypot(dx, dy)
                if d > 1e-6:
                    ang = _m.atan2(dy, dx)
                    half = _m.radians(fx["half_deg"])
                    length = fx["range"] * wpt
                    flicker = 0.85 + 0.15 * _m.sin(
                        37.0 * (time.perf_counter() - self._anim_t0)
                        + 0.7 * ax)
                    def _fan(frac_len, frac_half, color):
                        L = length * frac_len * flicker
                        h = half * frac_half
                        p0 = rl.Vector2(ax, ay)
                        p1 = rl.Vector2(ax + L * _m.cos(ang - h),
                                        ay + L * _m.sin(ang - h))
                        p2 = rl.Vector2(ax + L * _m.cos(ang + h),
                                        ay + L * _m.sin(ang + h))
                        # Raylib winding: draw both orders so the fan can't
                        # vanish on a back-facing vertex order.
                        rl.draw_triangle(p0, p1, p2, color)
                        rl.draw_triangle(p0, p2, p1, color)
                    if fx["flavor"] == "flame":
                        _fan(1.0, 1.0, rl.Color(255, 110, 30,
                                                int(60 * alpha_norm)))
                        _fan(0.72, 0.62, rl.Color(255, 190, 70,
                                                  int(80 * alpha_norm)))
                        _fan(0.38, 0.35, rl.Color(255, 240, 170,
                                                  int(110 * alpha_norm)))
                    else:   # "miasma" — fainter, sickly, no hot core
                        _fan(1.0, 1.0, rl.Color(120, 190, 70,
                                                int(38 * alpha_norm)))
                        _fan(0.6, 0.6, rl.Color(160, 210, 90,
                                                int(50 * alpha_norm)))
            elif kind == "glow":
                # W6 plasma bolt (ProjectileGlowEvent): a glowing orb at the
                # bolt's last reported position — re-emitted per tick, so a
                # slow bolt reads as a flying ember with a short trail.
                pos = fx["pos"]
                cx = tile_to_world_px(pos[0], wpt)
                cy = tile_to_world_px(pos[1], wpt)
                halo = rl.Color(255, 140, 50, int(90 * alpha_norm))
                core = rl.Color(255, 235, 190, int(230 * alpha_norm))
                rl.draw_circle(int(cx), int(cy), max(3.0, 0.55 * wpt), halo)
                rl.draw_circle(int(cx), int(cy), max(1.5, 0.22 * wpt), core)

    def transient_light_specs(self) -> list:
        """Light sources implied by the LIVE jet/glow effects (W6) — the
        'transient light emitter' half of the spray/plasma visuals.

        Returns plain dicts (x, y, max_range, intensity, color) in TILE
        coordinates; main.py converts them to ``bp.LightSource`` and appends
        them to the frame's source list (the renderer stays free of the
        physics module, and the sim is never touched — lights are a pure
        function of the renderer's own effect queue). A flame jet lights
        from ~1/3 down the cone axis in warm orange; a miasma jet barely
        glows (faint sickly green); a plasma bolt carries a small warm
        light with it. Intensities fade with the effect's remaining life.
        """
        import math as _m
        specs = []
        for fx in self._effects:
            alpha = max(0.0, 1.0 - fx["t"] / max(fx["life"], 1e-6))
            if fx["kind"] == "jet":
                dx = fx["target"][0] - fx["apex"][0]
                dy = fx["target"][1] - fx["apex"][1]
                d = _m.hypot(dx, dy)
                if d <= 1e-6:
                    continue
                ux, uy = dx / d, dy / d
                reach = float(fx["range"])
                lx = fx["apex"][0] + ux * reach * 0.35
                ly = fx["apex"][1] + uy * reach * 0.35
                if fx["flavor"] == "flame":
                    specs.append({
                        "x": lx, "y": ly,
                        "max_range": int(min(20, reach + 6)),
                        "intensity": 1.8 * alpha,
                        "color": (1.0, 0.55, 0.22),
                    })
                else:   # miasma
                    specs.append({
                        "x": lx, "y": ly,
                        "max_range": 8,
                        "intensity": 0.35 * alpha,
                        "color": (0.45, 0.8, 0.35),
                    })
            elif fx["kind"] == "glow":
                specs.append({
                    "x": float(fx["pos"][0]), "y": float(fx["pos"][1]),
                    "max_range": 10,
                    "intensity": 1.2 * alpha,
                    "color": (1.0, 0.65, 0.35),
                })
        return specs

    def consume_events(self, events: Sequence) -> None:
        """Read simulation tick events, spawn matching visual effects.

        Called once per frame after compose_world (or before — order
        doesn't matter since this only queues, not draws). The
        renderer maintains its own short-lived effect list; the sim
        does not track decay or fade.

        Recognised event types: :class:`simulation.events.ShotFiredEvent`,
        :class:`ExplosionEvent`, :class:`UnitHitEvent`. Unknown event
        types are ignored — additive design lets the sim emit new
        events without breaking older renderers.
        """
        # Lazy import — keeps renderer importable without the simulation pkg.
        from simulation.events import (
            ShotFiredEvent, LaserFiredEvent, ExplosionEvent, UnitHitEvent,
            SprayJetEvent, ProjectileGlowEvent,
        )
        for ev in events:
            if isinstance(ev, ShotFiredEvent):
                self._effects.append({
                    "kind": "tracer",
                    "from": ev.from_tile,
                    "to": ev.to_tile,
                    "t": 0.0,
                    "life": 0.18,    # ~5 frames @ 30 FPS
                })
            elif isinstance(ev, LaserFiredEvent):
                self._effects.append({
                    "kind": "beam",
                    "from": ev.from_tile,
                    "to": ev.to_tile,
                    "t": 0.0,
                    "life": 0.22,    # a touch longer than a tracer
                })
            elif isinstance(ev, SprayJetEvent):
                # W6 flame-jet / miasma-jet: short-lived on purpose — the
                # sim re-emits every deposit tick (24 Hz), so the fan reads
                # continuous during a burst and vanishes right after it.
                self._effects.append({
                    "kind": "jet",
                    "apex": ev.from_tile,
                    "target": ev.to_tile,
                    "range": ev.range_tiles,
                    "half_deg": ev.cone_half_angle_degrees,
                    "flavor": ev.kind,
                    "t": 0.0,
                    "life": 0.13,
                })
            elif isinstance(ev, ProjectileGlowEvent):
                self._effects.append({
                    "kind": "glow",
                    "pos": ev.pos,
                    "flavor": ev.kind,
                    "t": 0.0,
                    "life": 0.12,
                })
            elif isinstance(ev, ExplosionEvent):
                self._effects.append({
                    "kind": "explosion",
                    "pos": ev.pos,
                    "radius": ev.radius,
                    "t": 0.0,
                    "life": 0.6,
                })
            elif isinstance(ev, UnitHitEvent):
                # We don't know the unit's position from the event alone —
                # main.py looks up the unit and passes its pos when
                # converting; for now skip drawing UnitHit unless we
                # extend the event. Kept here to acknowledge the contract.
                pass

    def _advance_effects(self, dt: float) -> None:
        """Tick effect lifetimes; drop expired entries."""
        for fx in self._effects:
            fx["t"] += dt
        self._effects = [fx for fx in self._effects if fx["t"] < fx["life"]]

    def _draw_grid_world(self) -> None:
        wpt = self.world.world_px_per_tile
        draw_grid(self.cfg.grid_w, self.cfg.grid_h, wpt, step=3)

    # ---- final blit -----------------------------------------------------

    def draw_background_to_screen(self) -> None:
        """Draw the level's screen-fixed background behind the map area.
        Stretched to fill the map viewport. Camera-independent."""
        # BC (B5): on an AMBIENT map with no authored backdrop, paint the open
        # SKY behind the map so the ring tiles read as planet atmosphere, not
        # starfield. RENDER-ONLY (no sim field touched); a level that authors its
        # own background keeps it. A soft vertical daylight gradient (zenith blue
        # -> paler horizon haze); Erik tunes the look at B5.
        if self._ambient_sky_map and self.textures.background is None:
            x = int(self.camera.viewport_screen_x)
            y = int(self.camera.viewport_screen_y)
            w = int(self.cfg.map_px_w)
            h = int(self.cfg.map_px_h)
            rl.draw_rectangle_gradient_v(
                x, y, w, h,
                rl.Color(96, 152, 214, 255),    # zenith daylight blue
                rl.Color(176, 206, 230, 255))   # paler horizon haze
            return
        bg = self.textures.background
        if bg is None:
            return
        src = rl.Rectangle(0, 0, float(bg.width), float(bg.height))
        dst = rl.Rectangle(
            float(self.camera.viewport_screen_x),
            float(self.camera.viewport_screen_y),
            float(self.cfg.map_px_w), float(self.cfg.map_px_h),
        )
        rl.draw_texture_pro(bg, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

    def blit_world_to_screen(self) -> None:
        """Single DrawTexturePro from world RT to the map area of the screen.
        Camera transform happens here and nowhere else.

        No scissor — the destination rectangle == the map area, so there is
        nothing to clip. Scissor would only be needed if we drew into the
        panel by accident (which we don't).
        """
        self.world.blit_to_screen(
            self.camera,
            self.camera.viewport_screen_x, self.camera.viewport_screen_y,
            self.cfg.map_px_w, self.cfg.map_px_h,
        )

    # ---- panel ----------------------------------------------------------

    def draw_panel(self, sim=None, selected_unit=None,
                   planning_phase: int = 0,
                   current_mode: Optional[int] = None) -> None:
        """Right-side info panel. ``sim`` may be ``None`` for the demo;
        when provided we show turn/phase/tick state and (optionally) the
        selected unit's HP / inventory / orders summary."""
        cfg = self.cfg
        panel_x = cfg.map_px_w
        draw_panel_background(panel_x, 0, cfg.panel_px_w, cfg.map_px_h)
        x = panel_x + 12
        y = 12
        draw_text(self.level.name, x, y, 20)
        y += 28
        draw_text(f"{cfg.grid_w} x {cfg.grid_h} tiles", x, y, 14)
        y += 22

        # ----- Game state (if a Simulation was passed) -----
        if sim is not None:
            paused = sim.is_paused()
            state_label = "PAUSED — planning" if paused else "EXECUTING"
            state_color = (255, 220, 120, 255) if paused else (120, 255, 120, 255)
            draw_text(state_label, x, y, 16, color=state_color)
            y += 22
            draw_text(f"Round {getattr(sim, 'turn_number', 1)}   "
                      f"Phase {sim.get_phase() + 1}/2", x, y, 14)
            y += 18
            draw_text(f"Tick {sim.get_tick()} / "
                      f"{sim._ticks_per_round}", x, y, 13,
                      color=(180, 180, 200, 255))
            y += 18
            n_events = len(getattr(sim, "tick_events", []))
            draw_text(f"Tick events: {n_events}", x, y, 12,
                      color=(150, 150, 180, 255))
            y += 18
            if paused:
                draw_text(f"Planning phase {planning_phase + 1}",
                          x, y, 12, color=(200, 200, 120, 255))
                y += 16
            y += 6

            # Selected unit summary.
            if selected_unit is not None and getattr(selected_unit, "alive", False):
                draw_text(f"Selected: {selected_unit.name}", x, y, 14,
                          color=(120, 220, 255, 255))
                y += 18
                # W6: the equipped weapon row — the weapon-cycle key's (N)
                # on-screen feedback during Erik's tuning session.
                weapon_id = getattr(selected_unit, "weapon_id", "")
                if weapon_id:
                    draw_text(f"Weapon: {weapon_id}  (N cycles)", x, y, 13,
                              color=(255, 200, 120, 255))
                    y += 16
                from simulation.stats import effective_vitality
                draw_text(f"HP: {int(selected_unit.current_hp)}/{int(effective_vitality(selected_unit))}",
                          x, y, 13)
                y += 16
                draw_text(f"AP: {selected_unit.ap[0]}, {selected_unit.ap[1]}",
                          x, y, 13)
                y += 16
                draw_text(f"Grenades: {selected_unit.has_grenade}   "
                          f"Charges: {selected_unit.has_explosive}",
                          x, y, 12)
                y += 18
                n_orders = len(selected_unit.orders)
                draw_text(f"Queued orders: {n_orders}",
                          x, y, 12, color=(180, 180, 180, 255))
                y += 18
            if current_mode is not None:
                from simulation.orders import ORDER_NAMES
                mode_name = ORDER_NAMES.get(current_mode, str(current_mode))
                draw_text(f"Mode: {mode_name}", x, y, 13,
                          color=(255, 200, 120, 255))
                y += 22

        draw_text(f"FPS: {rl.get_fps()}", x, y, 14)
        y += 18
        draw_text(f"Raycast: {self.last_raycast_ms:.1f} ms", x, y, 14)
        y += 18
        draw_text(f"Frame:   {self.last_frame_ms:.1f} ms", x, y, 14)
        y += 28
        draw_text("Toggles:", x, y, 14, color=(180, 200, 255, 255))
        y += 20
        for label, on in [
            ("F1 grid",        self.show_grid),
            ("F2 smoke",       self.show_smoke),
            ("F3 fire",        self.show_fire),
            ("F4 light",       self.show_lighting),
            ("F5 normal map",  self.show_normal_map),
            ("F6 coords",      self.show_debug_coords),
            ("F7 pressure",    self.show_pressure),
            ("T  temperature", self.show_temperature),
            ("O  water optics", self.show_water),
            ("M  3D units",    self.cfg.use_3d_units),
            ("B  bilinear",    self.lighting.bilinear),
            ("G  sRGB",        self.srgb_decode),
            ("H  flip-Y norm", self.normal_y_flipped),
        ]:
            color = (180, 255, 180, 255) if on else (140, 140, 140, 255)
            draw_text(label, x, y, 13, color=color)
            y += 16
        y += 6
        draw_text(f"[/] Light Z: {self.lighting.light_z:.2f}", x, y, 13,
                  color=(220, 220, 180, 255))
        y += 16
        # Order-control cheat sheet.
        y += 10
        draw_text("Orders:", x, y, 14, color=(180, 200, 255, 255))
        y += 20
        for label in [
            "1  Move & Attack",
            "2  Move w/ Cover",
            "3  Sprint",
            "F  Fire (target tile)",
            "G  Grenade  (wheel = fuse)",
            "B  Breach   (wheel = slot)",
            "Tab  toggle unit's phase",
            "Bksp  undo last order",
            "Esc  clear selection",
            "Space  resume / pause",
        ]:
            draw_text(label, x, y, 12, color=(170, 170, 190, 255))
            y += 15
        y += 6
        draw_text("Ctrl+R reload config", x, y, 11,
                  color=(140, 140, 160, 255))
        y += 14

    # ---- input ----------------------------------------------------------

    def poll_toggles(self) -> None:
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F1):
            self.show_grid = not self.show_grid
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F2):
            self.show_smoke = not self.show_smoke
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F3):
            self.show_fire = not self.show_fire
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F4):
            self.show_lighting = not self.show_lighting
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F5):
            self.show_normal_map = not self.show_normal_map
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F6):
            self.show_debug_coords = not self.show_debug_coords
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F7):
            self.show_pressure = not self.show_pressure
        # T: debug temperature overlay (black-body ramp over gmap.temperature).
        if rl.is_key_pressed(rl.KeyboardKey.KEY_T):
            self.show_temperature = not self.show_temperature
        # V: toggle the water optics pass (GLSL Fresnel/GGX/refraction). The
        # pass is dormant-safe (alpha 0 on dry tiles); toggling only matters
        # once water is on the floor — disable to A/B against the bare floor.
        # (Moved O -> V in A6: O is the dev door-toggle key, ruling 5 —
        # a dual binding would flip the water rendering on every door
        # toggle. See input_handler.py.)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_V):
            self.show_water = not self.show_water
        # J: flip the Phase-0 3D marines live (render-only, feel gate). Lazy-load
        # the rigged model + build the top-down camera the first time it turns
        # on, so a sprite-only session never pays the load cost. If the model
        # fails to load, draw_units no-ops and the sprite path stays in effect.
        if rl.is_key_pressed(rl.KeyboardKey.KEY_M):
            self.cfg.use_3d_units = not self.cfg.use_3d_units
            if self.cfg.use_3d_units and not self.unit_models.ready:
                self.unit_models.load()
                self._unit_cam3d = UnitModelRenderer.make_camera(
                    self.world.world_px_w, self.world.world_px_h)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_B):
            self.lighting.toggle_bilinear()
        if rl.is_key_pressed(rl.KeyboardKey.KEY_H):
            self.normal_y_flipped = not self.normal_y_flipped
            self.lighting.set_normal_y_sign(-1.0 if self.normal_y_flipped else 1.0)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_G):
            self.srgb_decode = not self.srgb_decode
            self.lighting.set_srgb_decode(self.srgb_decode)
        # Adjust light Z (vertical incidence) live: [ = lower (grazing),
        # ] = higher (overhead). 0..1.5 range. Hold Shift for fine steps.
        shift_held = (rl.is_key_down(rl.KeyboardKey.KEY_LEFT_SHIFT) or
                      rl.is_key_down(rl.KeyboardKey.KEY_RIGHT_SHIFT))
        step = 0.02 if shift_held else 0.1
        if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT_BRACKET):
            self.lighting.set_light_z(self.lighting.light_z - step)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT_BRACKET):
            self.lighting.set_light_z(self.lighting.light_z + step)
        # Zoom: Q = out, E = in. Also bound to - / + for US keyboards.
        # Held auto-repeats. Mouse wheel works too.
        zoom_step = 1.0 if shift_held else 4.0
        K = rl.KeyboardKey
        zoom_out = (rl.is_key_down(K.KEY_Q) or rl.is_key_down(K.KEY_MINUS) or
                    rl.is_key_down(K.KEY_KP_SUBTRACT))
        zoom_in  = (rl.is_key_down(K.KEY_E) or rl.is_key_down(K.KEY_EQUAL) or
                    rl.is_key_down(K.KEY_KP_ADD))
        if zoom_out:
            self.camera.set_zoom(self.camera.zoom_px_per_tile - zoom_step * 0.4)
        if zoom_in:
            self.camera.set_zoom(self.camera.zoom_px_per_tile + zoom_step * 0.4)
        # Mouse wheel zoom
        wheel = rl.get_mouse_wheel_move()
        if wheel != 0:
            self.camera.set_zoom(self.camera.zoom_px_per_tile + wheel * zoom_step)

    def update_camera(self, dt: float, pan_speed_tiles_per_s: float = 30.0) -> None:
        K = rl.KeyboardKey
        dx = dy = 0.0
        if rl.is_key_down(K.KEY_A) or rl.is_key_down(K.KEY_LEFT):  dx -= 1
        if rl.is_key_down(K.KEY_D) or rl.is_key_down(K.KEY_RIGHT): dx += 1
        if rl.is_key_down(K.KEY_W) or rl.is_key_down(K.KEY_UP):    dy -= 1
        if rl.is_key_down(K.KEY_S) or rl.is_key_down(K.KEY_DOWN):  dy += 1
        if dx == 0 and dy == 0:
            return
        speed = pan_speed_tiles_per_s
        if rl.is_key_down(K.KEY_LEFT_SHIFT) or rl.is_key_down(K.KEY_RIGHT_SHIFT):
            speed *= 3
        self.camera.pan(dx * speed * dt, dy * speed * dt)

    # ---- coordinate conversions -----------------------------------------

    def mouse_to_tile(self) -> Optional[tuple]:
        """Mouse screen position -> integer world tile (for clicks /
        snapped placement). None if outside the camera's viewport."""
        coord = self.mouse_to_tile_float()
        if coord is None:
            return None
        return int(coord[0]), int(coord[1])

    def mouse_to_tile_float(self) -> Optional[tuple]:
        """Mouse screen position -> fractional world tile (for things like
        flashlight position where sub-tile precision matters)."""
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        if not self.camera.contains_screen_px(mx, my):
            return None
        return self.camera.screen_px_to_world_tile(mx, my)

    # ---- debug HUD ------------------------------------------------------

    def draw_debug_hud(self, gmap) -> None:
        """Cursor tile (x, y) + material readout. F6 toggles. Useful for
        scouting spawn coordinates and verifying level geometry."""
        if not self.show_debug_coords:
            return
        mouse_f = self.mouse_to_tile_float()
        if mouse_f is None:
            text = "tile (-, -) - cursor outside map"
        else:
            cx, cy = int(mouse_f[0]), int(mouse_f[1])
            H, W = gmap.solid.shape
            if 0 <= cx < W and 0 <= cy < H:
                # A6 (design §14.1): names come from the canon table, so
                # door_closed (and every material) reads correctly — the
                # old hardcoded 3-entry dict mislabeled all solids "hull".
                from simulation.materials import MATERIAL_NAMES
                mat_val = int(gmap.material[cy, cx])
                if gmap.is_vacuum[cy, cx]:
                    mat = "vacuum"
                else:
                    mat = MATERIAL_NAMES.get(mat_val, f"mat{mat_val}")
                blocked = bool(gmap.solid[cy, cx] or gmap.is_vacuum[cy, cx])
                tag = "BLOCKED" if blocked else "walkable"
                text = f"tile ({cx}, {cy}) | {mat} | {tag}"
            else:
                text = f"tile ({cx}, {cy}) - out of bounds"
        pad, font_size = 6, 16
        x0, y0 = 12, 40
        tw = rl.measure_text(text, font_size)
        rl.draw_rectangle(x0, y0, tw + 2 * pad, font_size + 2 * pad,
                          rl.Color(0, 0, 0, 180))
        rl.draw_text(text, x0 + pad, y0 + pad, font_size,
                     rl.Color(255, 230, 120, 255))

    # ---- shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        self.sprites.unload()
        self.unit_models.unload()
        self.textures.unload_all()
        rl.unload_shader(self.lighting.shader)
        rl.unload_texture(self.lighting.light_tex_a)
        rl.unload_texture(self.lighting.light_tex_b)
        rl.unload_texture(self.lighting.vacuum_tex)
        rl.unload_texture(self.smoke_overlay.tex)
        rl.unload_texture(self.fire_overlay.tex)
        rl.unload_texture(self.glow_overlay.tex)
        rl.unload_texture(self.temperature_overlay.tex)
        rl.unload_texture(self.water_overlay.tex)
        self.water_pass.unload()
        self.pressure_overlay.unload()
        self.world.unload()
        core.shutdown()


__all__ = ["GameRenderer", "RenderConfig"]
