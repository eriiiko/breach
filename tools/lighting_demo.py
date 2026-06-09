"""tools/lighting_demo.py — Lighting parameter tuning tool for Breach.

Standalone script: loads UNHCR Vessel, runs a live physics sim, and lets
Erik dial in visual parameters (ambient, lighting, smoke tint, grenade
blast) via raygui sliders while seeing the result in real time.

Run:
    C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py

Controls:
    WASD / arrows  — pan camera
    Q / E          — zoom out / in
    Mouse wheel    — zoom
    Space          — pause / resume sim
    T              — toggle grenade-spawn mode (click to detonate)
    Left click     — spawn grenade (when in spawn mode)
    N / Shift+N    — explosion smoke noise down / up (live cloud-texture dial)
    G              — toggle sRGB decode (renderer toggle from game)
    F1             — toggle grid overlay
    F2             — toggle smoke
    F4             — toggle lighting

The right panel exposes all visual sliders. Save/Load presets to/from
tools/lighting_presets.toml.
"""
from __future__ import annotations

import sys
import time
import tomllib
from pathlib import Path
from typing import Optional

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

import breach_physics as bp
from config import CFG
from level_loader import load as load_level
from renderer import GameRenderer
from renderer import core as rcore
from renderer.camera import Camera2D
from renderer.game_renderer import RenderConfig
from simulation import Simulation
from simulation.unit import Unit
from simulation.physics import apply_explosion, add_explosion_smoke

# ---------------------------------------------------------------------------
# Preset file location
# ---------------------------------------------------------------------------
PRESETS_PATH = Path(__file__).resolve().parent / "lighting_presets.toml"

# ---------------------------------------------------------------------------
# Defaults — §4 of the patch plan
# ---------------------------------------------------------------------------
DEFAULTS = {
    "ambient_r": 0.10,
    "ambient_g": 0.10,
    "ambient_b": 0.13,
    "light_z": 0.5,
    "normal_strength": 1.0,
    "use_normal": True,
    "srgb_decode": True,
    "flash_max_range": 25.0,
    "flash_intensity": 2.5,
    "flash_angle_spread": 6.283,
    "smoke_tint_r": 190.0,
    "smoke_tint_g": 195.0,
    "smoke_tint_b": 210.0,
    "smoke_max_alpha": 180.0,
    # smoke^gamma render contrast (ch.05 §6.1 step 5): power curve on the
    # rendered smoke opacity, render-only. 1.0 = off; ~1.2 dense -> 2.5 thin.
    "smoke_render_gamma": 1.5,
    "show_pressure": True,
    "pressure_scale": 2.0,
    "blast_radius": 6.0,
    "blast_pressure": 10.0,
    "wall_damage": 200.0,
    "unit_damage": 60.0,
    "fuse_seconds": 0.0,
    "smoke_amount": 0.3,  # default tuning baseline; 1.0 fills 2 rooms per nade
    # Per-tile contrast of the explosion smoke deposit (ch.05 §4). Drawn uniform
    # in [1 - noise, 1.0]: 0 = flat blob, 0.6 = old look, 0.85 = ragged holes
    # (config default), 1.0 = maximal contrast. Live dial Erik can nudge by eye
    # (slider in the panel, plus N / Shift+N keys).
    "explosion_smoke_noise": 0.85,
}

# Pressure colormap was previously implemented here; lifted into
# renderer/pressure_overlay.py so the main game can use it too. The demo
# now drives renderer.show_pressure + renderer.pressure_overlay.pressure_scale
# from the panel state.

# ---------------------------------------------------------------------------
# TOML save/load (hand-written — no tomli-w dependency)
# ---------------------------------------------------------------------------

def _state_to_toml_section(name: str, s: dict) -> str:
    """Serialise the demo state dict to a TOML section string."""
    lines = [f"[{name}]"]
    lines.append(f"ambient = [{s['ambient_r']:.4f}, {s['ambient_g']:.4f}, {s['ambient_b']:.4f}]")
    lines.append(f"light_z = {s['light_z']:.4f}")
    lines.append(f"normal_strength = {s['normal_strength']:.4f}")
    lines.append(f"use_normal = {'true' if s['use_normal'] else 'false'}")
    lines.append(f"srgb_decode = {'true' if s['srgb_decode'] else 'false'}")
    lines.append(
        f"flashlight = {{ max_range = {s['flash_max_range']:.2f}, "
        f"intensity = {s['flash_intensity']:.4f}, "
        f"angle_spread = {s['flash_angle_spread']:.4f} }}"
    )
    lines.append(
        f"smoke_tint = [{int(s['smoke_tint_r'])}, "
        f"{int(s['smoke_tint_g'])}, {int(s['smoke_tint_b'])}]"
    )
    lines.append(f"smoke_max_alpha = {int(s['smoke_max_alpha'])}")
    lines.append(f"pressure_scale = {s['pressure_scale']:.4f}")
    lines.append(
        f"grenade = {{ blast_radius = {int(s['blast_radius'])}, "
        f"pressure = {s['blast_pressure']:.2f}, "
        f"wall_damage = {int(s['wall_damage'])}, "
        f"unit_damage = {int(s['unit_damage'])}, "
        f"fuse_seconds = {s['fuse_seconds']:.2f}, "
        f"smoke_amount = {s['smoke_amount']:.4f} }}"
    )
    return "\n".join(lines)


def save_preset(name: str, state: dict) -> str:
    """Write or update a named preset in tools/lighting_presets.toml.

    Returns a status message suitable for the HUD.
    """
    # Sanitise name — TOML section headers cannot be empty or contain ].
    name = name.strip().replace("]", "_") or "default"

    # Read existing presets, replace/add the named section.
    existing: dict = {}
    if PRESETS_PATH.exists():
        try:
            with open(PRESETS_PATH, "rb") as f:
                existing = tomllib.load(f)
        except Exception:
            existing = {}  # corrupt file — start fresh

    # Serialise the new section first, then any others.
    existing[name] = {}   # mark the key so it appears in iteration

    lines = [
        "# Auto-managed by tools/lighting_demo.py — do not hand-edit while the demo is open.",
        "",
        _state_to_toml_section(name, state),
        "",
    ]
    for key, val in existing.items():
        if key == name:
            continue
        sec_state = _toml_dict_to_state(val)
        lines.append(_state_to_toml_section(key, sec_state))
        lines.append("")

    PRESETS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return f"Saved '{name}' → {PRESETS_PATH.name}"


def _toml_dict_to_state(d: dict) -> dict:
    """Convert a parsed TOML preset dict back into the flat state dict."""
    s = dict(DEFAULTS)
    if "ambient" in d:
        s["ambient_r"], s["ambient_g"], s["ambient_b"] = d["ambient"]
    for k in ("light_z", "normal_strength", "use_normal", "srgb_decode", "pressure_scale"):
        if k in d:
            s[k] = d[k]
    if "flashlight" in d:
        fl = d["flashlight"]
        s["flash_max_range"] = float(fl.get("max_range", s["flash_max_range"]))
        s["flash_intensity"] = float(fl.get("intensity", s["flash_intensity"]))
        s["flash_angle_spread"] = float(fl.get("angle_spread", s["flash_angle_spread"]))
    if "smoke_tint" in d:
        s["smoke_tint_r"], s["smoke_tint_g"], s["smoke_tint_b"] = [
            float(v) for v in d["smoke_tint"]
        ]
    if "smoke_max_alpha" in d:
        s["smoke_max_alpha"] = float(d["smoke_max_alpha"])
    if "grenade" in d:
        g = d["grenade"]
        s["blast_radius"] = float(g.get("blast_radius", s["blast_radius"]))
        s["blast_pressure"] = float(g.get("pressure", s["blast_pressure"]))
        s["wall_damage"] = float(g.get("wall_damage", s["wall_damage"]))
        s["unit_damage"] = float(g.get("unit_damage", s["unit_damage"]))
        s["fuse_seconds"] = float(g.get("fuse_seconds", s["fuse_seconds"]))
        s["smoke_amount"] = float(g.get("smoke_amount", s["smoke_amount"]))
    return s


def load_preset(name: str) -> Optional[dict]:
    """Load a named preset from tools/lighting_presets.toml. None if missing."""
    if not PRESETS_PATH.exists():
        return None
    with open(PRESETS_PATH, "rb") as f:
        all_presets = tomllib.load(f)
    if name not in all_presets:
        return None
    return _toml_dict_to_state(all_presets[name])


def list_presets() -> list[str]:
    """Return names of all presets in the presets file."""
    if not PRESETS_PATH.exists():
        return []
    with open(PRESETS_PATH, "rb") as f:
        return list(tomllib.load(f).keys())


# ---------------------------------------------------------------------------
# GUI panel state — mutable floats for raygui sliders
# ---------------------------------------------------------------------------

class PanelState:
    """Holds all tunable parameters as plain Python floats / bools.

    raygui reads/writes through ffi float* / int* pointers. We keep the
    Python-side values here and sync them into ffi buffers each frame.
    """

    def __init__(self):
        self._v = dict(DEFAULTS)

        # Grenade spawn mode
        self.spawn_mode = False
        self.paused = False

        # Preset name for save/load (16 char buffer)
        self.preset_name = "default"

        # Preset dropdown state
        self.dropdown_open = False
        self.dropdown_active = 0

        # Status message (shown briefly after Save/Load)
        self.status_msg = ""
        self.status_until = 0.0

    def get(self, key: str):
        return self._v[key]

    def set(self, key: str, val) -> None:
        self._v[key] = val

    def as_dict(self) -> dict:
        return dict(self._v)

    def apply_dict(self, d: dict) -> None:
        for k in DEFAULTS:
            if k in d:
                self._v[k] = d[k]

    def reset_defaults(self) -> None:
        self._v = dict(DEFAULTS)


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

PANEL_W = 340
SLIDER_H = 16
SLIDER_W = 200
LABEL_W = 110
ROW_GAP = 20   # tight packing to fit all sections in 720px height


def _slider(state: PanelState, key: str,
            label: str, lo: float, hi: float,
            x: int, y: int) -> int:
    """Draw a labelled slider, update state if changed. Returns new y."""
    rl.gui_label(rl.Rectangle(x, y, LABEL_W, SLIDER_H), label)
    val_ptr = rl.ffi.new("float *", state.get(key))
    rl.gui_slider_bar(
        rl.Rectangle(x + LABEL_W, y, SLIDER_W, SLIDER_H),
        "", f"{state.get(key):.2f}",
        val_ptr, lo, hi,
    )
    state.set(key, val_ptr[0])
    return y + ROW_GAP


def _checkbox(state: PanelState, key: str,
              label: str, x: int, y: int) -> int:
    """Draw a checkbox, update state if toggled. Returns new y."""
    checked_ptr = rl.ffi.new("bool *", state.get(key))
    rl.gui_check_box(rl.Rectangle(x, y, SLIDER_H, SLIDER_H), label, checked_ptr)
    state.set(key, bool(checked_ptr[0]))
    return y + ROW_GAP


def _section_header(label: str, x: int, y: int) -> int:
    """Draw a section divider label. Returns new y."""
    rl.gui_label(rl.Rectangle(x, y, PANEL_W - 20, 14),
                 f"-- {label} --")
    return y + 18


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. Load level + build sim ----
    level_name = getattr(CFG.display, "level", "unhcr_vessel")
    level = load_level(level_name)
    print(f"[lighting_demo] Level: {level.name}  {level.width}x{level.height} tiles")

    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    # Clean ship — NO demo hazards. Grenades provide the only smoke/pressure.
    sim.set_paused(False)   # run from the start in the demo

    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    print(f"[lighting_demo] Spawned {len(level.spawns)} units")

    # ---- 2. Window ----
    screen_w, screen_h = 1280, 720   # windowed by default
    panel_px_w = PANEL_W
    map_px_w = screen_w - panel_px_w
    map_px_h = screen_h
    cfg = RenderConfig(
        map_px_w=map_px_w, map_px_h=map_px_h,
        panel_px_w=panel_px_w,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=24.0,
    )

    fit_zoom = map_px_w / max(level.width, 1)
    initial_zoom = max(12.0, min(40.0, fit_zoom))
    initial_camera = Camera2D(
        pos_tile_x=0.0, pos_tile_y=0.0,
        zoom_px_per_tile=initial_zoom,
        viewport_px_w=map_px_w, viewport_px_h=map_px_h,
        world_size_tile_w=level.width, world_size_tile_h=level.height,
    )

    # GameRenderer creates the window internally.
    renderer = GameRenderer(level, bp, cfg,
                            initial_camera=initial_camera,
                            borderless=False)

    # Pressure overlay is now built into the renderer (renderer/pressure_overlay.py),
    # shared with the main game. No demo-local allocations needed.

    # ---- 4. Panel state ----
    state = PanelState()

    # Apply default preset from file if it exists
    saved = load_preset("default")
    if saved:
        state.apply_dict(saved)
        print("[lighting_demo] Loaded preset 'default' from lighting_presets.toml")

    # Apply initial ambient to the renderer
    renderer.lighting.set_ambient((state.get("ambient_r"),
                                   state.get("ambient_g"),
                                   state.get("ambient_b")))
    renderer.lighting.set_light_z(state.get("light_z"))
    renderer.lighting.set_normal_strength(state.get("normal_strength"))
    renderer.lighting.set_use_normal(state.get("use_normal"))
    renderer.lighting.set_srgb_decode(state.get("srgb_decode"))
    renderer.smoke_overlay.tint_r = int(state.get("smoke_tint_r"))
    renderer.smoke_overlay.tint_g = int(state.get("smoke_tint_g"))
    renderer.smoke_overlay.tint_b = int(state.get("smoke_tint_b"))
    renderer.smoke_overlay.max_alpha = int(state.get("smoke_max_alpha"))
    renderer.smoke_overlay.gamma = float(state.get("smoke_render_gamma"))

    # ---- 5. Sim timing ----
    last_time = time.perf_counter()
    sim_dt = 1.0 / float(CFG.clock.ticks_per_second)
    tick_accum = 0.0
    max_catch_up = 5

    # For click spawn — debounce so one press = one grenade
    last_click_handled = False

    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # ---- Input: toggles ----
            renderer.poll_toggles()
            renderer.update_camera(dt)

            K = rl.KeyboardKey
            if rl.is_key_pressed(K.KEY_SPACE):
                state.paused = not state.paused
            # T = toggle grenade-spawn mode. (G is taken by sRGB decode in poll_toggles.)
            if rl.is_key_pressed(K.KEY_T):
                state.spawn_mode = not state.spawn_mode
            # N / Shift+N = nudge explosion smoke noise down / up live, so the
            # cloud's initial texture can be dialled in by eye between throws.
            # (Mirrors the panel slider; clamped to [0, 1].)
            if rl.is_key_pressed(K.KEY_N):
                shift = (rl.is_key_down(K.KEY_LEFT_SHIFT) or
                         rl.is_key_down(K.KEY_RIGHT_SHIFT))
                cur = state.get("explosion_smoke_noise")
                cur = cur + 0.05 if shift else cur - 0.05
                state.set("explosion_smoke_noise", min(1.0, max(0.0, cur)))

            # ---- Sim tick ----
            if not state.paused:
                # The sim auto-pauses at end of round. In the demo we want
                # continuous physics — re-enable after each auto-pause so
                # smoke/pressure keeps evolving even when no grenades land.
                if sim.is_paused():
                    sim.set_paused(False)

                tick_accum += dt
                steps = 0
                while tick_accum >= sim_dt and steps < max_catch_up:
                    sim.step()
                    tick_accum -= sim_dt
                    steps += 1

            # ---- Grenade spawn (left click when in spawn mode) ----
            left_down = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
            if state.spawn_mode and left_down and not last_click_handled:
                tile = renderer.mouse_to_tile()
                if tile is not None:
                    tx, ty = tile
                    H, W = sim.gmap.material.shape
                    if 0 <= tx < W and 0 <= ty < H:
                        r = max(1, int(state.get("blast_radius")))
                        apply_explosion(
                            sim.gmap, sim.edit_queue, ty, tx, r,
                            state.get("blast_pressure"),
                            state.get("wall_damage"),
                        )
                        # smoke_amount scales the deposit
                        smoke_mult = state.get("smoke_amount")
                        if smoke_mult > 0:
                            _tmp_rng = np.random.default_rng(None)
                            # Deposit smoke: call add_explosion_smoke then scale
                            # We don't have a scale param — deposit first, measure
                            # delta, scale the delta. Simpler: just call N times or
                            # clamp. Spec says "multiplier on add_explosion_smoke
                            # deposit" — we interpret that as calling once and
                            # scaling the newly deposited values.
                            before = sim.gmap.smoke.copy()
                            add_explosion_smoke(
                                sim.gmap, sim.edit_queue, ty, tx, r,
                                noise=state.get("explosion_smoke_noise"))
                            delta = sim.gmap.smoke - before
                            # Rescale: new = before + delta * mult (clamped to 1.0)
                            # IN-PLACE write: the C++ atmosphere / raycaster
                            # solvers hold a pointer to this specific numpy
                            # array. Replacing the array via `gmap.smoke = ...`
                            # leaves C++ pointing at the old memory and the
                            # raycaster ends up sampling garbage on subsequent
                            # frames — which manifests visually as the ship
                            # going transparent after the first grenade.
                            np.copyto(sim.gmap.smoke,
                                      np.clip(before + delta * smoke_mult,
                                              0.0, 1.0))
                        # Queue a visual flash so the renderer draws the ring.
                        renderer._effects.append({
                            "kind": "explosion",
                            "pos": (tx, ty),
                            "radius": r,
                            "t": 0.0,
                            "life": 0.6,
                        })
                last_click_handled = True
            elif not left_down:
                last_click_handled = False

            # ---- Sync smoke overlay params ----
            renderer.smoke_overlay.tint_r = int(state.get("smoke_tint_r"))
            renderer.smoke_overlay.tint_g = int(state.get("smoke_tint_g"))
            renderer.smoke_overlay.tint_b = int(state.get("smoke_tint_b"))
            renderer.smoke_overlay.max_alpha = int(state.get("smoke_max_alpha"))
            # smoke^gamma render contrast — live re-push so the slider takes
            # effect immediately (render-only; never touches the sim field).
            renderer.smoke_overlay.gamma = float(state.get("smoke_render_gamma"))

            # ---- Lighting setters ----
            renderer.lighting.set_ambient((state.get("ambient_r"),
                                           state.get("ambient_g"),
                                           state.get("ambient_b")))
            renderer.lighting.set_light_z(state.get("light_z"))
            renderer.lighting.set_normal_strength(state.get("normal_strength"))
            renderer.lighting.set_use_normal(state.get("use_normal"))
            renderer.lighting.set_srgb_decode(state.get("srgb_decode"))

            # ---- Build mouse flashlight ----
            sources = []
            mouse_f = renderer.mouse_to_tile_float()
            if mouse_f is not None:
                src = bp.LightSource()
                src.x = float(mouse_f[0])
                src.y = float(mouse_f[1])
                src.max_range = int(max(1, state.get("flash_max_range")))
                src.intensity = state.get("flash_intensity")
                src.angle_spread = state.get("flash_angle_spread")
                # Flashlight — cool white (profile: flashlight).
                src.color = (1.0, 1.0, 0.95)
                src.jitter = 0.0
                sources.append(src)

            # ---- Upload physics state ----
            renderer.upload_state(sim.gmap, light_sources=sources)
            renderer.consume_events(sim.tick_events)
            renderer._advance_effects(dt)

            # Pressure overlay is now owned by the renderer (shared with
            # main game). Sync the slider state to the renderer's overlay
            # and the toggle to renderer.show_pressure — the actual update
            # + draw happen inside upload_state / compose_world.
            renderer.show_pressure = bool(state.get("show_pressure"))
            renderer.pressure_overlay.pressure_scale = float(
                state.get("pressure_scale")
            )

            # ---- Draw ----
            renderer.begin_frame()
            renderer.compose_world(
                units_marines=sim.marines(),
                units_zombies=sim.zombies(),
                projectiles=sim.projectiles,
            )

            renderer.draw_background_to_screen()
            renderer.blit_world_to_screen()

            # ---- HUD ----
            _draw_hud(renderer, sim.gmap, state, now)

            # ---- raygui panel ----
            _draw_panel(state, renderer, now)

            renderer.end_frame()

    finally:
        renderer.shutdown()


# ---------------------------------------------------------------------------
# HUD (§6)
# ---------------------------------------------------------------------------

def _draw_hud(renderer: GameRenderer, gmap, state: PanelState,
              now: float) -> None:
    """Cursor tile coords + pressure + smoke, always visible top-left."""
    mouse_f = renderer.mouse_to_tile_float()
    H, W = gmap.solid.shape

    spawn_tag = " [SPAWN MODE — click to detonate]" if state.spawn_mode else ""
    pause_tag = " [PAUSED]" if state.paused else ""
    header = f"BREACH Lighting Demo{spawn_tag}{pause_tag}"

    if mouse_f is None:
        tile_line = "tile (-, -) — outside map"
        pressure_line = "pressure: —"
        smoke_line = "smoke: —"
    else:
        cx, cy = int(mouse_f[0]), int(mouse_f[1])
        if 0 <= cx < W and 0 <= cy < H:
            if gmap.is_vacuum[cy, cx]:
                mat_name = "vacuum"
            elif gmap.solid[cy, cx]:
                mat_name = "hull"
            else:
                mat_val = int(gmap.material[cy, cx])
                mat_name = {0: "air", 1: "hull", 3: "door"}.get(mat_val, f"mat{mat_val}")
            tile_line = f"tile ({cx}, {cy}) — {mat_name}"
            total_p = float(gmap.atmosphere[cy, cx] + gmap.wave_p[cy, cx])
            pressure_line = f"pressure: {total_p:.3f}"
            smoke_line = f"smoke: {float(gmap.smoke[cy, cx]):.3f}"
        else:
            tile_line = f"tile ({cx}, {cy}) — out of bounds"
            pressure_line = "pressure: —"
            smoke_line = "smoke: —"

    lines = [header, tile_line, pressure_line, smoke_line]
    font_size = 15
    pad = 6
    max_w = max(rl.measure_text(line, font_size) for line in lines)
    box_w = max_w + 2 * pad
    box_h = len(lines) * (font_size + 4) + 2 * pad
    x0, y0 = 12, 12
    rl.draw_rectangle(x0, y0, box_w, box_h, rl.Color(0, 0, 0, 180))
    for i, line in enumerate(lines):
        color = rl.Color(255, 230, 80, 255) if i == 0 else rl.Color(200, 230, 255, 255)
        rl.draw_text(line, x0 + pad, y0 + pad + i * (font_size + 4), font_size, color)

    # Status message (Save / Load feedback) or pressure hint
    if state.status_msg and now < state.status_until:
        rl.draw_text(state.status_msg, x0, y0 + box_h + 8, 14,
                     rl.Color(120, 255, 120, 255))
    elif state.get("show_pressure"):
        rl.draw_text("[P overlay ON — spawn a grenade to see shockwave]",
                     x0, y0 + box_h + 8, 11, rl.Color(180, 180, 100, 200))


# ---------------------------------------------------------------------------
# raygui panel (§4)
# ---------------------------------------------------------------------------

def _draw_panel(state: PanelState, renderer: GameRenderer,
                now: float) -> None:
    """Draw the full raygui slider panel on the right side of the window."""
    cfg = renderer.cfg
    px = cfg.map_px_w        # panel left edge
    pw = cfg.panel_px_w

    # Panel background
    rl.draw_rectangle(px, 0, pw, cfg.map_px_h, rl.Color(20, 20, 28, 240))
    rl.draw_line_ex(rl.Vector2(px, 0), rl.Vector2(px, cfg.map_px_h),
                    2.0, rl.Color(120, 120, 140, 255))

    x = px + 10
    y = 10

    # Title + perf on same line
    rl.draw_text("Lighting Demo", x, y, 16, rl.Color(200, 220, 255, 255))
    fps_str = f"  FPS:{rl.get_fps()} RT:{renderer.last_raycast_ms:.0f}ms"
    rl.draw_text(fps_str, x + 100, y + 2, 11, rl.Color(140, 140, 160, 255))
    y += 22

    # -- §4.1 Ambient --
    y = _section_header("Ambient", x, y)
    y = _slider(state, "ambient_r", "Amb R", 0.0, 1.0, x, y)
    y = _slider(state, "ambient_g", "Amb G", 0.0, 1.0, x, y)
    y = _slider(state, "ambient_b", "Amb B", 0.0, 1.0, x, y)

    # -- §4.2 Lighting --
    y = _section_header("Lighting", x, y)
    y = _slider(state, "light_z", "Light Z", 0.0, 1.5, x, y)
    y = _slider(state, "normal_strength", "Norm strength", 0.0, 2.0, x, y)
    y = _checkbox(state, "use_normal", "Use normal map", x, y)
    y = _checkbox(state, "srgb_decode", "sRGB decode", x, y)

    # -- §4.3 Mouse flashlight --
    y = _section_header("Flashlight", x, y)
    y = _slider(state, "flash_max_range", "Max range", 5.0, 40.0, x, y)
    y = _slider(state, "flash_intensity", "Intensity", 0.0, 5.0, x, y)
    y = _slider(state, "flash_angle_spread", "Spread", 0.0, 6.283, x, y)

    # -- §4.4 Smoke overlay --
    y = _section_header("Smoke overlay", x, y)
    y = _slider(state, "smoke_tint_r", "Tint R", 0.0, 255.0, x, y)
    y = _slider(state, "smoke_tint_g", "Tint G", 0.0, 255.0, x, y)
    y = _slider(state, "smoke_tint_b", "Tint B", 0.0, 255.0, x, y)
    y = _slider(state, "smoke_max_alpha", "Max alpha", 0.0, 255.0, x, y)
    # smoke^gamma render contrast (render-only). 1.0 = off; >1 = wispier/filmic.
    y = _slider(state, "smoke_render_gamma", "Gamma", 1.0, 3.0, x, y)

    # -- §4.5 Pressure overlay --
    y = _section_header("Pressure overlay", x, y)
    y = _checkbox(state, "show_pressure", "Show pressure", x, y)
    y = _slider(state, "pressure_scale", "P scale", 0.5, 10.0, x, y)

    # -- §4.6 Grenade tuning --
    y = _section_header("Grenade [T=spawn mode]", x, y)
    y = _slider(state, "blast_radius", "Radius", 1.0, 15.0, x, y)
    y = _slider(state, "blast_pressure", "Pressure", 1.0, 30.0, x, y)
    y = _slider(state, "wall_damage", "Wall dmg", 0.0, 1000.0, x, y)
    # unit_damage stored but not applied in direct-spawn (no apply_blast_damage call)
    y = _slider(state, "fuse_seconds", "Fuse (s)", 0.0, 5.0, x, y)
    y = _slider(state, "smoke_amount", "Smoke mult", 0.0, 2.0, x, y)
    # Per-tile contrast of the deposited cloud (ch.05 §4). 0 = flat blob,
    # 0.85 = ragged holes (default), 1.0 = max. Also N / Shift+N keys.
    y = _slider(state, "explosion_smoke_noise", "Noise", 0.0, 1.0, x, y)

    # -- §4.7 Save / Load --
    y = _section_header("Presets", x, y)

    # Preset name text box
    rl.gui_label(rl.Rectangle(x, y, LABEL_W, SLIDER_H), "Name:")
    _draw_preset_textbox(state, x + LABEL_W, y, SLIDER_W, SLIDER_H)
    y += ROW_GAP

    # Save button
    # Save + Reset on same row
    if rl.gui_button(rl.Rectangle(x, y, 70, 20), "Save"):
        name = state.preset_name.strip() or "default"
        state.status_msg = save_preset(name, state.as_dict())
        state.status_until = now + 3.0
    if rl.gui_button(rl.Rectangle(x + 80, y, 110, 20), "Reset defaults"):
        state.reset_defaults()
    y += 24

    # Load button + preset dropdown
    presets = list_presets()
    if presets:
        if state.dropdown_active >= len(presets):
            state.dropdown_active = 0
        active_ptr = rl.ffi.new("int *", state.dropdown_active)
        # gui_dropdown_box: editMode=True = open. Returns 1 when user picks.
        clicked = rl.gui_dropdown_box(
            rl.Rectangle(x, y, 150, 20),
            ";".join(presets),
            active_ptr,
            state.dropdown_open,
        )
        state.dropdown_active = active_ptr[0]
        if clicked:
            state.dropdown_open = not state.dropdown_open

        if rl.gui_button(rl.Rectangle(x + 158, y, 60, 20), "Load"):
            chosen = presets[state.dropdown_active]
            loaded = load_preset(chosen)
            if loaded:
                state.apply_dict(loaded)
                state.status_msg = f"Loaded '{chosen}'"
                state.status_until = now + 3.0
            state.dropdown_open = False
        y += 26
    else:
        rl.gui_label(rl.Rectangle(x, y, PANEL_W - 20, 14), "(save a preset first)")
        y += 18

    # Keybind reminder
    rl.draw_text("Space=pause  T=spawn  WASD=pan  Q/E=zoom  G=sRGB",
                 x, y, 10, rl.Color(120, 120, 140, 255))


# ---------------------------------------------------------------------------
# Preset name text input — simple custom impl via gui_text_box
# ---------------------------------------------------------------------------

_textbox_edit = False   # module-level since there's only one text box


MAX_PRESET_NAME = 32   # maximum preset name length


def _draw_preset_textbox(state: PanelState,
                          x: int, y: int, w: int, h: int) -> None:
    """Draw an editable text box for the preset name.

    gui_text_box takes a cffi char* — we build it from state.preset_name
    each frame and write back after raygui edits it.
    """
    global _textbox_edit
    # Build a null-padded byte buffer of exactly MAX_PRESET_NAME and hand it
    # to ffi.new as the char[] initializer. Per-cell assignment doesn't work
    # because Python 3's bytes-iteration yields ints, not single-byte values.
    s_bytes = state.preset_name.encode("ascii", "replace")[:MAX_PRESET_NAME - 1]
    padded = s_bytes + b"\x00" * (MAX_PRESET_NAME - len(s_bytes))
    cstr = rl.ffi.new(f"char[{MAX_PRESET_NAME}]", padded)

    clicked = rl.gui_text_box(rl.Rectangle(x, y, w, h), cstr, MAX_PRESET_NAME,
                               _textbox_edit)
    if clicked:
        _textbox_edit = not _textbox_edit

    state.preset_name = rl.ffi.string(cstr).decode("ascii", "replace")


if __name__ == "__main__":
    main()
