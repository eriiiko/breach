"""tools/lighting_demo.py — Lighting parameter tuning tool for Breach.

Standalone script: loads UNHCR Vessel, runs a live physics sim, and lets
Erik dial in visual parameters (ambient, lighting, smoke tint, grenade
blast) via raygui sliders while seeing the result in real time.

Run:
    C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py
    C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py --level fire_studio
    (--flood pre-fills the ship interior with standing water at startup, for
     tuning the water look without pouring by hand; --windowed for fixed 1280x720)

Controls (keymap audited — O is intentionally unused):
    WASD / arrows  — pan camera
    Q / E / wheel  — zoom
    Space          — pause / resume sim
    T              — toggle grenade-spawn mode (click to detonate)
    Left click     — spawn grenade (when in spawn mode)
    N / Shift+N    — explosion smoke noise down / up (live cloud-texture dial)
    U              — pour water under cursor (0.2 m per press)
    F / Shift+F    — flood the whole interior to ~0.6 m / drain all water
    P / Shift+P    — tilt ship +2° / −2° (the Titanic dial, clamped ±20°)
    G              — toggle sRGB decode (renderer toggle from game)
    V              — toggle water optics pass (renderer toggle; O moved -> V)
    L              — toggle fire lights   F1/F2/F4 — grid / smoke / lighting
    --- B2 studio (tool-side sim writes at the cursor) ---
    I              — ignite the tile under the cursor (3x3, force-flammable)
    J              — puff SMOKE (soot) under the cursor
    K              — puff STEAM under the cursor
    C              — toggle the door under the cursor (DoorRuntime, not paint)
    1              — toggle the static lamp group (demo-side)
    2              — toggle the rotating beacon on / off (demo-side)

The right panel exposes all visual sliders (incl. the B2 gas-medium dials).
The top-left readout is the hover-tile "microscope" (T + pseudo-Kelvin, fire,
material, the five trace gases + O2). Save/Load presets to/from
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
from simulation import gas_fixed
from simulation import atmosphere_fixed
from simulation.unit import Unit
from simulation.physics import apply_explosion, add_explosion_smoke
from simulation.gases import STEAM, SMOKE
from level_lights import monotonic_total_tick
from renderer.fire_lights import FireLightSelector
from renderer.frame_lights import (build_frame_light_sources,
                                    build_static_light_sources)
from renderer.hover_readout import pack_hover_readout

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
    # Render exposure (engine/08 §Falloff is density). The pure-density raycaster
    # makes light_rgb a PHYSICAL 1/r field where `intensity` = total emitted power
    # (N-independent) — ~ray_count× dimmer than the legacy per-ray-dist_atten
    # field. This master dial maps physical power -> display brightness. Drag it
    # FIRST after the redesign lands: physics tunes power, exposure tunes look.
    "light_gain": 10.0,
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
    # (Fire & Heat Beauty B2 P2: smoke_render_gamma is DELETED — the gas-medium
    # tau-curve below subsumes it; the legacy A/B smoke path bakes the old 1.5
    # as a frozen constant in the renderer, so it needs no dial here.)
    # --- B2 gas-medium dials (§6). The sliders drive renderer.gas_medium live
    # (P2 consumers). Defaults = honest/identity (config [render.*]).
    # soot_yield/smoke_emission mirror the SIM config (display-only here — Erik
    # tunes them in the P5 feel session; this patch never changes their values).
    # Re-init from CFG at startup. --------------------------------------------
    "legacy_smoke_on": False,
    "gm_plume_k_scale": 1.0,
    "gm_tau_curve_a": 1.0,
    "gm_tau_curve_b": 1.0,
    "gm_glow_gain": 1.0,
    "gm_effect_gas_floor": 0.0,
    "gm_fuel_haze_on": False,
    "gd_enabled": True,
    "gd_noise_octaves": 4.0,
    "gd_noise_wavelength_tiles": 3.0,
    "gd_adv_gain": 1.0,
    "gd_cycle_seconds": 2.5,
    "gd_erode_strength": 0.6,
    "gd_warp_px": 3.0,
    "gd_dither_on": True,
    "speckle_mode": 2.0,     # stepped slider: 0=off 1=noise 2=soot
    "speckle_amp": 0.25,
    "soot_yield": 0.3,       # mirror of [physics.combustion] (display-only, P1)
    "smoke_emission": 0.8,   # mirror of [physics.fire] (display-only, P1)
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
    # --- Water surface optics ([graphics.water]; live in the demo only) -----
    # Overwritten below from config.toml at startup so the panel opens where the
    # config sits; dragging a slider re-pushes the matching WaterPass setter
    # every frame. Pour water with U to see them bite. (Defaults mirror config.)
    "water_glint_strength": 2.0,
    "water_roughness_base": 0.08,
    "water_roughness_agitation": 0.6,
    "water_fog_density": 3.0,
    "water_refract_strength": 0.02,
    "water_r0": 0.02,
    "water_ripple_scale": 8.0,
    "water_alpha_scale": 6.0,
    "water_alpha_min": 0.15,
    "water_alpha_max": 0.95,
    # Phase 2 (mood pass): caustics / foam / chromatic aberration / wave size.
    "water_caustic_strength": 2.5,
    "water_caustic_scale": 6.0,
    "water_foam_threshold": 0.02,
    "water_foam_intensity": 0.6,
    "water_ca_amount": 0.012,
    "water_wave_scale": 2.0,
    "water_ambient_amp": 0.06,
    # Heightmap attenuation (alpha-only): only bites on a level WITH a heightmap
    # (unhcr_vessel_2). On the demo's default level the water pass has no height
    # texture, so these sliders are inert (u_has_height = 0).
    "water_height_scale": 0.4,
    "water_height_edge": 0.1,
    # height_floor: the heightmap value treated as floor level (the "level 0"
    # baseline) — subtracted from relief before scaling so standing water clears
    # the floor immediately instead of having to over-fill (the art heightmap is
    # a RELATIVE depth map with the floor at a nonzero relief). Inert on the
    # demo's default level (no heightmap).
    "water_height_floor": 0.3,
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
    lines.append(f"light_gain = {s['light_gain']:.4f}")
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
    lines.append(
        f"water = {{ glint_strength = {s['water_glint_strength']:.4f}, "
        f"roughness_base = {s['water_roughness_base']:.4f}, "
        f"roughness_agitation = {s['water_roughness_agitation']:.4f}, "
        f"fog_density = {s['water_fog_density']:.4f}, "
        f"refract_strength = {s['water_refract_strength']:.4f}, "
        f"r0 = {s['water_r0']:.4f}, "
        f"ripple_scale = {s['water_ripple_scale']:.4f}, "
        f"alpha_scale = {s['water_alpha_scale']:.4f}, "
        f"alpha_min = {s['water_alpha_min']:.4f}, "
        f"alpha_max = {s['water_alpha_max']:.4f}, "
        f"caustic_strength = {s['water_caustic_strength']:.4f}, "
        f"caustic_scale = {s['water_caustic_scale']:.4f}, "
        f"foam_threshold = {s['water_foam_threshold']:.4f}, "
        f"foam_intensity = {s['water_foam_intensity']:.4f}, "
        f"ca_amount = {s['water_ca_amount']:.4f}, "
        f"wave_scale = {s['water_wave_scale']:.4f}, "
        f"ambient_amp = {s['water_ambient_amp']:.4f}, "
        f"height_scale = {s['water_height_scale']:.4f}, "
        f"height_edge = {s['water_height_edge']:.4f}, "
        f"height_floor = {s['water_height_floor']:.4f} }}"
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
    for k in ("light_z", "light_gain", "normal_strength", "use_normal", "srgb_decode", "pressure_scale"):
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
    # Water surface optics. The inline `water` table maps short keys to the
    # flat water_* state; .get falls back to the DEFAULTS value already in `s`,
    # so OLD presets without a water table load cleanly (no KeyError).
    if "water" in d:
        w = d["water"]
        s["water_glint_strength"] = float(w.get("glint_strength", s["water_glint_strength"]))
        s["water_roughness_base"] = float(w.get("roughness_base", s["water_roughness_base"]))
        s["water_roughness_agitation"] = float(w.get("roughness_agitation", s["water_roughness_agitation"]))
        s["water_fog_density"] = float(w.get("fog_density", s["water_fog_density"]))
        s["water_refract_strength"] = float(w.get("refract_strength", s["water_refract_strength"]))
        s["water_r0"] = float(w.get("r0", s["water_r0"]))
        s["water_ripple_scale"] = float(w.get("ripple_scale", s["water_ripple_scale"]))
        s["water_alpha_scale"] = float(w.get("alpha_scale", s["water_alpha_scale"]))
        s["water_alpha_min"] = float(w.get("alpha_min", s["water_alpha_min"]))
        s["water_alpha_max"] = float(w.get("alpha_max", s["water_alpha_max"]))
        s["water_caustic_strength"] = float(w.get("caustic_strength", s["water_caustic_strength"]))
        s["water_caustic_scale"] = float(w.get("caustic_scale", s["water_caustic_scale"]))
        s["water_foam_threshold"] = float(w.get("foam_threshold", s["water_foam_threshold"]))
        s["water_foam_intensity"] = float(w.get("foam_intensity", s["water_foam_intensity"]))
        s["water_ca_amount"] = float(w.get("ca_amount", s["water_ca_amount"]))
        s["water_wave_scale"] = float(w.get("wave_scale", s["water_wave_scale"]))
        s["water_ambient_amp"] = float(w.get("ambient_amp", s["water_ambient_amp"]))
        s["water_height_scale"] = float(w.get("height_scale", s["water_height_scale"]))
        s["water_height_edge"] = float(w.get("height_edge", s["water_height_edge"]))
        s["water_height_floor"] = float(w.get("height_floor", s["water_height_floor"]))
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

        # B2 studio: DEMO-SIDE light grouping (NOT an entity system — the
        # harness owns the lamp list and the helper rebuilds sources; passing
        # [] drops a group this frame). 1 toggles the static lamps, 2 the beacon.
        self.lamps_on = True
        self.beacon_on = True

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
# Flood / drain — pre-fill the ship interior with standing water for tuning
# ---------------------------------------------------------------------------

# Default standing-water level for --flood / the F key (metres). ~0.6 m clears
# the floor and pokes furniture/crates through (a ~0.3-0.5 m relief), which is
# exactly the case the height_floor / heightmap dials are tuned against.
FLOOD_LEVEL_M = 0.6


def _parse_level_arg() -> Optional[str]:
    """Read an optional ``--level NAME`` override (B2 P1; the studio session
    runs ``--level fire_studio``). Default stays ``CFG.display.level``."""
    if "--level" not in sys.argv:
        return None
    i = sys.argv.index("--level")
    try:
        name = sys.argv[i + 1]
    except IndexError:
        raise SystemExit("--level requires a level folder name, e.g. "
                         "--level fire_studio")
    if name.startswith("--"):
        raise SystemExit(f"--level requires a level folder name, got {name!r}")
    return name


# ---------------------------------------------------------------------------
# B2 studio injection + door toggle (TOOL-side direct sim writes — the debug
# pattern of src/input_handler.py:255-301). TOOLS may write sim fields; the
# RENDERER never does. All land immediately, even while paused.
# ---------------------------------------------------------------------------

def _inject_fire(sim, tile) -> None:
    """Ignite a 3x3 patch at the cursor tile (force-flammable + full seed)."""
    if tile is None:
        return
    fx, fy = tile
    gmap = sim.gmap
    h, w = gmap.fire.shape
    if not (0 <= fy < h and 0 <= fx < w):
        return
    y0, y1 = max(0, fy - 1), min(h, fy + 2)
    x0, x1 = max(0, fx - 1), min(w, fx + 2)
    gmap.flammable[y0:y1, x0:x1] = True
    from simulation import fire_fixed
    gmap.fire[y0:y1, x0:x1] = fire_fixed.quantize_scalar(1.0)


def _inject_gas(sim, tile, species: int) -> None:
    """Puff a 3x3 blob of ``species`` at full density at the cursor tile."""
    if tile is None:
        return
    fx, fy = tile
    gmap = sim.gmap
    _, h, w = gmap.gas.shape
    if not (0 <= fy < h and 0 <= fx < w):
        return
    y0, y1 = max(0, fy - 1), min(h, fy + 2)
    x0, x1 = max(0, fx - 1), min(w, fx + 2)
    from simulation import gas_fixed
    gmap.gas[species, y0:y1, x0:x1] = gas_fixed.SMOKE_MAX_Q


def _toggle_door(sim, tile) -> None:
    """Flip the want_open latch of the door under the cursor (DoorRuntime — the
    9e sweep applies/retries it next unpaused tick), NOT tile paint."""
    if tile is None:
        return
    fx, fy = tile
    door = sim.door_at(fy, fx)          # door_at takes (fy, fx)
    if door is None:
        print(f"[demo] no door at tile ({fx}, {fy})")
        return
    if not door.alive:
        print(f"[demo] door '{door.id}' at ({fx}, {fy}) is destroyed")
        return
    door.want_open = not door.want_open
    print(f"[demo] door '{door.id}' want_open -> {int(door.want_open)} "
          f"(state={door.state}; applies next unpaused tick)")


def flood_interior(gmap, level_m: float = FLOOD_LEVEL_M) -> int:
    """Pre-fill every INTERIOR tile with standing water (direct write).

    Interior = a tile that is NON-solid AND NON-vacuum — i.e. there is actual
    floor there (not a wall, not open space outside the hull). This is the same
    mask the atmosphere/water solvers treat as "inside the ship", so we only pour
    where water could physically stand. Writes gmap.water_depth in place (like the
    U pour) so it lands immediately. Returns the number of tiles filled.
    """
    interior = (~gmap.solid) & (~gmap.is_vacuum)
    gmap.water_depth[interior] = float(level_m)
    return int(interior.sum())


def drain_all(gmap) -> int:
    """Zero out all standing water (the Shift+F drain). Returns tiles cleared."""
    n = int((gmap.water_depth > 0.0).sum())
    gmap.water_depth[:] = 0.0
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. Load level + build sim ----
    # --level overrides config for one launch (the studio runs
    # --level fire_studio); default stays [display] level.
    level_name = _parse_level_arg() or getattr(CFG.display, "level",
                                               "unhcr_vessel")
    level = load_level(level_name)
    print(f"[lighting_demo] Level: {level.name}  {level.width}x{level.height} tiles")

    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    # Clean ship — NO demo hazards. Grenades provide the only smoke/pressure.
    sim.set_paused(False)   # run from the start in the demo

    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    print(f"[lighting_demo] Spawned {len(level.spawns)} units")

    # --flood: pre-fill the ship interior with standing water at startup so the
    # water look/heightmap dials can be tuned without pouring tile-by-tile (U).
    # Direct in-place write to gmap.water_depth (like the U pour / the --auto
    # seed below), so it lands on the first frame. Works headlessly (--auto).
    if "--flood" in sys.argv:
        n = flood_interior(sim.gmap)
        print(f"[lighting_demo] --flood: filled {n} interior tiles to "
              f"{FLOOD_LEVEL_M} m standing water")

    # ---- 2. Window ----
    # Borderless at the real monitor resolution by default (same pattern as
    # main.py): open the window first, then query the actual size and lay the
    # panel/map out to fit. `--windowed` falls back to the old fixed 1280x720.
    BORDERLESS = "--windowed" not in sys.argv
    if BORDERLESS:
        rcore.init_window(0, 0, title="Breach — Lighting Demo",
                          borderless=True)
        screen_w, screen_h = rcore.get_monitor_size()
    else:
        screen_w, screen_h = 1280, 720
    panel_px_w = PANEL_W
    map_px_w = screen_w - panel_px_w
    map_px_h = screen_h
    cfg = RenderConfig(
        map_px_w=map_px_w, map_px_h=map_px_h,
        panel_px_w=panel_px_w,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=float(getattr(CFG.rendering, "world_px_per_tile",
                                        24.0)),
    )

    fit_zoom = map_px_w / max(level.width, 1)
    initial_zoom = max(12.0, min(40.0, fit_zoom))
    initial_camera = Camera2D(
        pos_tile_x=0.0, pos_tile_y=0.0,
        zoom_px_per_tile=initial_zoom,
        viewport_px_w=map_px_w, viewport_px_h=map_px_h,
        world_size_tile_w=level.width, world_size_tile_h=level.height,
    )

    # GameRenderer attaches to the already-open window in borderless mode
    # (or creates the fixed one in --windowed).
    renderer = GameRenderer(level, bp, cfg,
                            initial_camera=initial_camera,
                            borderless=BORDERLESS)

    # Pressure overlay is now built into the renderer (renderer/pressure_overlay.py),
    # shared with the main game. No demo-local allocations needed.

    # ---- B2: shared frame-light assembly (statics + beacon + fire) ----
    # The SAME helper main.py uses (renderer/frame_lights.py) so the studio gets
    # the level's lamps + rotating beacon + B1 fire lights with no drift. Lamps
    # and beacon are DEMO-SIDE toggles (state.lamps_on / state.beacon_on) — NOT
    # an entity system; passing [] to the helper drops a group this frame.
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    ticks_per_round = int(CFG.clock.ticks_per_round)
    static_lights, beacon_lights, lights_off = build_static_light_sources(
        bp, level.lights, level.width, level.height, sim_time_per_tick)
    fire_selector = FireLightSelector.from_config(CFG)
    if level.lights:
        _off = f"  ({len(lights_off)} off-grid skipped)" if lights_off else ""
        print(f"[lighting_demo] Lights: {len(static_lights)} static + "
              f"{len(beacon_lights)} beacon from level.toml{_off}")

    # FINAL keybindings (also on the panel; printed here since the tall panel
    # can scroll the on-screen help off the bottom). Keymap AUDITED — O is
    # avoided (documented water-overlay key); the studio keys land on free keys.
    print("[lighting_demo] KEYS  I=ignite  J=smoke  K=steam  C=door  "
          "1=lamps  2=beacon")
    print("[lighting_demo]       Space=pause T=spawn U=water F/Shift+F=flood "
          "P=tilt N=noise  |  L=firelights V=water G=sRGB")
    print("[lighting_demo]       WASD/arrows=pan  Q/E/wheel=zoom  "
          "F1/F2/F4=overlays  F9=legacy smoke A/B  (O intentionally unused)")

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
    renderer.lighting.set_light_gain(state.get("light_gain"))
    renderer.lighting.set_normal_strength(state.get("normal_strength"))
    renderer.lighting.set_use_normal(state.get("use_normal"))
    renderer.lighting.set_srgb_decode(state.get("srgb_decode"))
    renderer.smoke_overlay.tint_r = int(state.get("smoke_tint_r"))
    renderer.smoke_overlay.tint_g = int(state.get("smoke_tint_g"))
    renderer.smoke_overlay.tint_b = int(state.get("smoke_tint_b"))
    renderer.smoke_overlay.max_alpha = int(state.get("smoke_max_alpha"))

    # Apply the initial water-optics params from config so the panel opens
    # where [graphics.water] sits (the main game restart-binds these; here they
    # are live). getattr-defaults keep the demo alive if the block is absent.
    _wc = getattr(getattr(CFG, "graphics", None), "water", None)
    if _wc is not None:
        state.set("water_glint_strength",
                  float(getattr(_wc, "glint_strength", 2.0)))
        state.set("water_roughness_base",
                  float(getattr(_wc, "roughness_base", 0.08)))
        state.set("water_roughness_agitation",
                  float(getattr(_wc, "roughness_agitation", 0.6)))
        state.set("water_fog_density",
                  float(getattr(_wc, "fog_density", 3.0)))
        state.set("water_refract_strength",
                  float(getattr(_wc, "refract_strength", 0.02)))
        state.set("water_r0", float(getattr(_wc, "r0", 0.02)))
        state.set("water_ripple_scale",
                  float(getattr(_wc, "ripple_scale", 8.0)))
        state.set("water_alpha_scale",
                  float(getattr(_wc, "alpha_scale", 6.0)))
        state.set("water_alpha_min", float(getattr(_wc, "alpha_min", 0.15)))
        state.set("water_alpha_max", float(getattr(_wc, "alpha_max", 0.95)))
        # Phase 2 (mood pass).
        state.set("water_caustic_strength",
                  float(getattr(_wc, "caustic_strength", 2.5)))
        state.set("water_caustic_scale",
                  float(getattr(_wc, "caustic_scale", 6.0)))
        state.set("water_foam_threshold",
                  float(getattr(_wc, "foam_threshold", 0.02)))
        state.set("water_foam_intensity",
                  float(getattr(_wc, "foam_intensity", 0.6)))
        state.set("water_ca_amount", float(getattr(_wc, "ca_amount", 0.012)))
        state.set("water_wave_scale", float(getattr(_wc, "wave_scale", 2.0)))
        state.set("water_ambient_amp",
                  float(getattr(_wc, "ambient_amp", 0.06)))
        state.set("water_height_scale",
                  float(getattr(_wc, "height_scale", 0.4)))
        state.set("water_height_edge",
                  float(getattr(_wc, "height_edge", 0.1)))
        state.set("water_height_floor",
                  float(getattr(_wc, "height_floor", 0.3)))

    # B2 dials: open the sliders where config.toml sits (getattr-defaults keep
    # the demo alive if a block is absent). soot_yield / smoke_emission MIRROR
    # the sim config for display — this patch never writes them back to the sim.
    _rend = getattr(CFG, "render", None)
    _gm = getattr(_rend, "gas_medium", None)
    _gd = getattr(_rend, "gas_detail", None)
    _sp = getattr(_rend, "speckle", None)
    if _rend is not None:
        state.set("legacy_smoke_on",
                  bool(getattr(_rend, "legacy_smoke_on", False)))
    if _gm is not None:
        state.set("gm_plume_k_scale", float(getattr(_gm, "plume_k_scale", 1.0)))
        state.set("gm_tau_curve_a", float(getattr(_gm, "tau_curve_a", 1.0)))
        state.set("gm_tau_curve_b", float(getattr(_gm, "tau_curve_b", 1.0)))
        state.set("gm_glow_gain", float(getattr(_gm, "glow_gain", 1.0)))
        state.set("gm_effect_gas_floor",
                  float(getattr(_gm, "effect_gas_floor", 0.0)))
        state.set("gm_fuel_haze_on", bool(getattr(_gm, "fuel_haze_on", False)))
    if _gd is not None:
        state.set("gd_enabled", bool(getattr(_gd, "enabled", True)))
        state.set("gd_noise_octaves", float(getattr(_gd, "noise_octaves", 4)))
        state.set("gd_noise_wavelength_tiles",
                  float(getattr(_gd, "noise_wavelength_tiles", 3.0)))
        state.set("gd_adv_gain", float(getattr(_gd, "adv_gain", 1.0)))
        state.set("gd_cycle_seconds", float(getattr(_gd, "cycle_seconds", 2.5)))
        state.set("gd_erode_strength",
                  float(getattr(_gd, "erode_strength", 0.6)))
        state.set("gd_warp_px", float(getattr(_gd, "warp_px", 3.0)))
        state.set("gd_dither_on", bool(getattr(_gd, "dither_on", True)))
    if _sp is not None:
        _mode_idx = {"off": 0.0, "noise": 1.0, "soot": 2.0}.get(
            str(getattr(_sp, "mode", "soot")), 2.0)
        state.set("speckle_mode", _mode_idx)
        state.set("speckle_amp", float(getattr(_sp, "amp", 0.25)))
    _comb = getattr(getattr(CFG, "physics", None), "combustion", None)
    _fire = getattr(getattr(CFG, "physics", None), "fire", None)
    if _comb is not None:
        state.set("soot_yield", float(getattr(_comb, "soot_yield", 0.3)))
    if _fire is not None:
        state.set("smoke_emission", float(getattr(_fire, "smoke_emission", 0.8)))

    # ---- 5. Sim timing ----
    last_time = time.perf_counter()
    sim_dt = 1.0 / float(CFG.clock.ticks_per_second)
    tick_accum = 0.0
    max_catch_up = 5

    # For click spawn — debounce so one press = one grenade
    last_click_handled = False

    # --auto: render a fixed number of frames then exit 0 (smoke test; no input
    # injection). Mirrors test_main_smoke / align_level_art's --auto tails.
    auto = "--auto" in sys.argv
    AUTO_FRAMES = 120
    frames = 0

    # Under --auto there is no cursor/keyboard to pour water, so seed a small
    # puddle directly: this drives the water pass OFF its dormant early-out so
    # the new additive-glint + alpha-ramp branch is actually exercised (and the
    # per-frame WaterPass setters run against a live pass). Direct in-place
    # write — same pattern as the U-key pour below.
    if auto:
        H, W = sim.gmap.material.shape
        cy, cx = H // 2, W // 2
        for ty in range(max(0, cy - 2), min(H, cy + 3)):
            for tx in range(max(0, cx - 2), min(W, cx + 3)):
                if not sim.gmap.solid[ty, tx]:
                    sim.gmap.water_depth[ty, tx] = 0.4

    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # ---- Input: toggles ----
            renderer.poll_toggles()
            renderer.update_camera(dt)
            # Legacy smoke A/B (B2 P2): reflect an F9 key flip (poll_toggles)
            # into the panel checkbox so the key and the checkbox agree; the
            # checkbox value is pushed back to the renderer right after the panel
            # draw (see the post-_draw_panel push below).
            state.set("legacy_smoke_on", renderer.legacy_smoke_on)

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
            # U = pour 0.2 m of water under the cursor (clamped at 2.5 m
            # total) — the main game's debug-pour, demo-local copy. Direct
            # write, so it lands even while paused.
            if rl.is_key_pressed(K.KEY_U):
                tile = renderer.mouse_to_tile()
                if tile is not None:
                    tx, ty = tile
                    H, W = sim.gmap.material.shape
                    if (0 <= tx < W and 0 <= ty < H
                            and not sim.gmap.solid[ty, tx]):
                        sim.gmap.water_depth[ty, tx] = min(
                            sim.gmap.water_depth[ty, tx] + 0.2, 2.5)
            # F = flood the whole interior to the default level; Shift+F = drain
            # all water. Fast fill/drain for tuning the water look (and the
            # heightmap / height_floor dials) without pouring tile-by-tile with U.
            # Direct write, so it lands even while paused.
            if rl.is_key_pressed(K.KEY_F):
                shift = (rl.is_key_down(K.KEY_LEFT_SHIFT) or
                         rl.is_key_down(K.KEY_RIGHT_SHIFT))
                if shift:
                    n = drain_all(sim.gmap)
                    print(f"[demo] drained all water ({n} tiles)")
                else:
                    n = flood_interior(sim.gmap)
                    print(f"[demo] flooded interior -> {FLOOD_LEVEL_M} m "
                          f"({n} tiles)")
            # P / Shift+P = ship tilt_x +2 / −2 degrees (clamped ±20) — the
            # Titanic dial; standing water slides to the low side.
            if rl.is_key_pressed(K.KEY_P):
                shift = (rl.is_key_down(K.KEY_LEFT_SHIFT) or
                         rl.is_key_down(K.KEY_RIGHT_SHIFT))
                step_r = float(np.radians(-2.0 if shift else 2.0))
                lim = float(np.radians(20.0))
                sim.gmap.tilt_x = max(-lim, min(lim,
                                                sim.gmap.tilt_x + step_r))
                print(f"[demo] ship tilt_x -> "
                      f"{np.degrees(sim.gmap.tilt_x):+.1f} deg")

            # ---- B2 studio keys: injection at cursor + door + light toggles --
            # Keymap AUDITED: renderer.poll_toggles owns F1-F7 + F9 (F9 = legacy
            # smoke A/B, B2 P2 — F8 is the input_handler recorder dump in-game) /
            # T / L / V / M / B / H / G / [ ] / Q / E; the demo owns
            # Space / T / N / U / F / P.
            # These land on FREE keys (I/J/K/C/1/2). O is AVOIDED per the audit
            # directive (the demo help documents O as the water-depth overlay).
            _tile = renderer.mouse_to_tile()
            if rl.is_key_pressed(K.KEY_I):        # ignite tile at cursor
                _inject_fire(sim, _tile)
            if rl.is_key_pressed(K.KEY_J):        # puff SMOKE (soot) at cursor
                _inject_gas(sim, _tile, SMOKE)
            if rl.is_key_pressed(K.KEY_K):        # puff STEAM at cursor
                _inject_gas(sim, _tile, STEAM)
            if rl.is_key_pressed(K.KEY_C):        # toggle door under cursor
                _toggle_door(sim, _tile)
            if rl.is_key_pressed(K.KEY_ONE):      # toggle the static lamp group
                state.lamps_on = not state.lamps_on
                print(f"[demo] lamps {'ON' if state.lamps_on else 'OFF'}")
            if rl.is_key_pressed(K.KEY_TWO):      # toggle the rotating beacon
                state.beacon_on = not state.beacon_on
                print(f"[demo] beacon {'ON' if state.beacon_on else 'OFF'}")

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
                            # Rescale: new = before + delta * mult (clamped to 1.0).
                            # smoke is int32 Q16.16 (S2b) — rescale in REAL units
                            # then requantize (round-half-away-from-zero, matching
                            # the C++ boundary), so the float `smoke_mult` scaling
                            # stays correct.
                            # IN-PLACE write: the C++ atmosphere / raycaster
                            # solvers hold a pointer to this specific numpy
                            # array. Replacing the array via `gmap.smoke = ...`
                            # leaves C++ pointing at the old memory and the
                            # raycaster ends up sampling garbage on subsequent
                            # frames — which manifests visually as the ship
                            # going transparent after the first grenade.
                            new_real = np.clip(
                                gas_fixed.dequantize(before)
                                + gas_fixed.dequantize(delta) * smoke_mult,
                                0.0, 1.0)
                            np.copyto(sim.gmap.smoke,
                                      gas_fixed.quantize(new_real))
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

            # ---- Sync smoke overlay params (legacy A/B path) ----
            renderer.smoke_overlay.tint_r = int(state.get("smoke_tint_r"))
            renderer.smoke_overlay.tint_g = int(state.get("smoke_tint_g"))
            renderer.smoke_overlay.tint_b = int(state.get("smoke_tint_b"))
            renderer.smoke_overlay.max_alpha = int(state.get("smoke_max_alpha"))

            # ---- Sync the B2 gas-medium dials LIVE (P2) ----
            # The tau-curve / plume-k / glow-gain / gas-floor sliders drive the
            # premultiplied gas-medium layer immediately (render-only). The
            # legacy_smoke_on A/B is reconciled with the F9 key toggle around the
            # panel draw (poll_toggles reflect + the post-panel push).
            gm = renderer.gas_medium
            gm.plume_k_scale = float(state.get("gm_plume_k_scale"))
            gm.tau_curve_a = float(state.get("gm_tau_curve_a"))
            gm.tau_curve_b = float(state.get("gm_tau_curve_b"))
            gm.glow_gain = float(state.get("gm_glow_gain"))
            gm.effect_gas_floor = float(state.get("gm_effect_gas_floor"))

            # ---- Sync the B2 gas-DETAIL dials LIVE (P3) ----
            # The advected-noise shader dials drive shaders/gas_medium.fs
            # immediately (render-only). noise_octaves is a BAKE-time param —
            # gas_detail lazily rebakes the fBm when it changes (a few ms, only
            # on change). enabled toggles the whole pass (else the plain P2 layer
            # draws — byte-for-byte the P2 look).
            gd = renderer.gas_detail
            gd.enabled = bool(state.get("gd_enabled"))
            gd.noise_octaves = int(round(state.get("gd_noise_octaves")))
            gd.noise_wavelength_tiles = float(
                state.get("gd_noise_wavelength_tiles"))
            gd.adv_gain = float(state.get("gd_adv_gain"))
            gd.cycle_seconds = float(state.get("gd_cycle_seconds"))
            gd.erode_strength = float(state.get("gd_erode_strength"))
            gd.warp_px = float(state.get("gd_warp_px"))
            gd.dither_on = bool(state.get("gd_dither_on"))

            # ---- Lighting setters ----
            renderer.lighting.set_ambient((state.get("ambient_r"),
                                           state.get("ambient_g"),
                                           state.get("ambient_b")))
            renderer.lighting.set_light_z(state.get("light_z"))
            renderer.lighting.set_light_gain(state.get("light_gain"))
            renderer.lighting.set_normal_strength(state.get("normal_strength"))
            renderer.lighting.set_use_normal(state.get("use_normal"))
            renderer.lighting.set_srgb_decode(state.get("srgb_decode"))

            # ---- Water optics setters (live; the main game restart-binds) ----
            wp = renderer.water_pass
            wp.set_glint_strength(state.get("water_glint_strength"))
            wp.set_roughness_base(state.get("water_roughness_base"))
            wp.set_roughness_agitation(state.get("water_roughness_agitation"))
            wp.set_fog_density(state.get("water_fog_density"))
            wp.set_refract_strength(state.get("water_refract_strength"))
            wp.set_r0(state.get("water_r0"))
            wp.set_ripple_scale(state.get("water_ripple_scale"))
            wp.set_alpha_scale(state.get("water_alpha_scale"))
            wp.set_alpha_min(state.get("water_alpha_min"))
            wp.set_alpha_max(state.get("water_alpha_max"))
            # Phase 2 (mood pass) — caustics / foam / CA / wave size.
            wp.set_caustic_strength(state.get("water_caustic_strength"))
            wp.set_caustic_scale(state.get("water_caustic_scale"))
            wp.set_foam_threshold(state.get("water_foam_threshold"))
            wp.set_foam_intensity(state.get("water_foam_intensity"))
            wp.set_ca_amount(state.get("water_ca_amount"))
            wp.set_wave_scale(state.get("water_wave_scale"))
            wp.set_ambient_amp(state.get("water_ambient_amp"))
            # Heightmap attenuation (alpha-only). Only bites on a level WITH a
            # height texture bound; inert on the demo's default level.
            wp.set_height_scale(state.get("water_height_scale"))
            wp.set_height_edge(state.get("water_height_edge"))
            wp.set_height_floor(state.get("water_height_floor"))

            # ---- Lights: level statics + beacon + fire (shared assembly) ----
            # The SAME helper main.py uses (B2 P1). Lamps/beacon are DEMO-SIDE
            # toggles: pass [] to drop a group this frame. total_tick = the
            # MONOTONIC sim tick on the SIM clock so the beacon freezes on pause
            # + replays exactly (never wall dt).
            total_tick = monotonic_total_tick(
                sim.turn_number, ticks_per_round, sim.tick)
            frame = build_frame_light_sources(
                bp,
                static_lights if state.lamps_on else [],
                beacon_lights if state.beacon_on else [],
                total_tick=total_tick, sim_time_per_tick=sim_time_per_tick,
                fire_selector=fire_selector,
                temperature_field=sim.gmap.temperature,
                blackbody_ramp=renderer.blackbody_ramp,
                show_fire_lights=renderer.show_fire_lights)
            sources = frame.sources
            renderer.set_fire_light_stats(frame.fire_count, frame.fire_peaks,
                                          fire_selector.max_lights)

            # ---- Mouse flashlight (caller-side, slider-driven) ----
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
            # total_tick (the MONOTONIC sim clock above) drives the P3 gas-detail
            # crossfade too — sim tick, never wall time (replay-identical smoke).
            renderer.upload_state(sim.gmap, light_sources=sources,
                                  sim_tick=total_tick)
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
            # Legacy smoke A/B: push the checkbox back to the renderer (a click
            # this frame takes effect next frame; the F9 key path took effect at
            # the top of the frame). One source of truth = renderer.legacy_smoke_on.
            renderer.legacy_smoke_on = bool(state.get("legacy_smoke_on"))

            renderer.end_frame()

            frames += 1
            if auto and frames >= AUTO_FRAMES:
                # Pour a little water under the auto path so the water pass +
                # its live setters are actually exercised (not just dormant).
                break
    finally:
        renderer.shutdown()

    if auto:
        print(f"OK — lighting_demo rendered {frames} frames (--auto)")


# ---------------------------------------------------------------------------
# HUD (§6)
# ---------------------------------------------------------------------------

def _draw_hud(renderer: GameRenderer, gmap, state: PanelState,
              now: float) -> None:
    """Header + the hover-tile "microscope" readout (B2 P1), top-left.

    The readout — T in game units AND pseudo-Kelvin, fire intensity, material
    name, the five trace gases (steam/smoke/poison/teargas/fuel_gas) + O2 — is
    packed by renderer.hover_readout (pyray-free, unit-tested headless). READ-
    ONLY gmap reads. The T->Kelvin conversion is REUSED from the black-body
    ramp (kelvin = kelvin_ambient + k_temp_to_kelvin * T_game), so the readout
    and the emissive overlay agree.
    """
    spawn_tag = " [SPAWN — click to detonate]" if state.spawn_mode else ""
    pause_tag = " [PAUSED]" if state.paused else ""
    lamp_tag = "" if state.lamps_on else "  lamps:OFF"
    beacon_tag = "" if state.beacon_on else "  beacon:OFF"
    header = f"BREACH Lighting Demo{spawn_tag}{pause_tag}{lamp_tag}{beacon_tag}"

    mouse_f = renderer.mouse_to_tile_float()
    if mouse_f is None:
        lines = [header, "tile (-, -) — outside map"]
    else:
        cx, cy = int(mouse_f[0]), int(mouse_f[1])
        # Reuse the black-body ramp's own game-ΔT -> pseudo-Kelvin conversion.
        readout = pack_hover_readout(
            gmap, cx, cy, renderer.blackbody_ramp._kelvin_from_tgame)
        if readout is None:
            lines = [header, f"tile ({cx}, {cy}) — out of bounds"]
        else:
            # atmosphere + wave_p are int32 Q16.16 (S2a/S2c) — dequantize.
            total_p = atmosphere_fixed.dequantize(
                gmap.atmosphere[cy, cx] + gmap.wave_p[cy, cx])
            lines = [header] + readout.lines + [f"pressure: {total_p:8.3f} atm"]

    font_size = 15
    pad = 6
    max_w = max(rl.measure_text(line, font_size) for line in lines)
    box_w = max_w + 2 * pad
    box_h = len(lines) * (font_size + 4) + 2 * pad
    x0, y0 = 12, 12
    rl.draw_rectangle(x0, y0, box_w, box_h, rl.Color(0, 0, 0, 190))
    for i, line in enumerate(lines):
        color = (rl.Color(255, 230, 80, 255) if i == 0
                 else rl.Color(200, 230, 255, 255))
        rl.draw_text(line, x0 + pad, y0 + pad + i * (font_size + 4),
                     font_size, color)

    # Status message (Save / Load feedback).
    if state.status_msg and now < state.status_until:
        rl.draw_text(state.status_msg, x0, y0 + box_h + 8, 14,
                     rl.Color(120, 255, 120, 255))


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
    y = _slider(state, "light_gain", "Light gain", 0.0, 60.0, x, y)
    y = _slider(state, "light_z", "Light Z", 0.0, 1.5, x, y)
    y = _slider(state, "normal_strength", "Norm strength", 0.0, 2.0, x, y)
    y = _checkbox(state, "use_normal", "Use normal map", x, y)
    y = _checkbox(state, "srgb_decode", "sRGB decode", x, y)

    # -- §4.3 Mouse flashlight --
    y = _section_header("Flashlight", x, y)
    y = _slider(state, "flash_max_range", "Max range", 5.0, 40.0, x, y)
    y = _slider(state, "flash_intensity", "Intensity", 0.0, 5.0, x, y)
    y = _slider(state, "flash_angle_spread", "Spread", 0.0, 6.283, x, y)

    # -- §4.4 Smoke overlay (LEGACY A/B path — F9 or the checkbox below) --
    y = _section_header("Smoke overlay (legacy)", x, y)
    y = _slider(state, "smoke_tint_r", "Tint R", 0.0, 255.0, x, y)
    y = _slider(state, "smoke_tint_g", "Tint G", 0.0, 255.0, x, y)
    y = _slider(state, "smoke_tint_b", "Tint B", 0.0, 255.0, x, y)
    y = _slider(state, "smoke_max_alpha", "Max alpha", 0.0, 255.0, x, y)
    # (smoke_render_gamma slider removed in B2 P2 — the gas-medium tau-curve
    #  below subsumes it; the legacy path bakes the old 1.5 as a constant.)

    # -- B2 gas medium (P2 LIVE: these sliders drive renderer.gas_medium now;
    #    the detail/speckle dials below are still P3/P4 plumbing) --
    y = _section_header("Gas medium (B2)", x, y)
    y = _checkbox(state, "legacy_smoke_on", "Legacy A/B (F9)", x, y)
    y = _slider(state, "gm_plume_k_scale", "Plume k", 0.0, 4.0, x, y)
    y = _slider(state, "gm_tau_curve_a", "Tau a", 0.0, 3.0, x, y)
    y = _slider(state, "gm_tau_curve_b", "Tau b", 0.2, 4.0, x, y)
    y = _slider(state, "gm_glow_gain", "Glow gain", 0.0, 4.0, x, y)
    y = _slider(state, "gm_effect_gas_floor", "Gas floor", 0.0, 1.0, x, y)
    y = _checkbox(state, "gm_fuel_haze_on", "Fuel haze", x, y)
    # The soot handover pair (existing SIM config; DISPLAY-only in P1 — Erik
    # tunes it by feel in the P5 session, never this patch).
    y = _slider(state, "soot_yield", "Soot yield", 0.0, 1.0, x, y)
    y = _slider(state, "smoke_emission", "Smoke emit", 0.0, 3.0, x, y)
    # Detail shader (P3) + speckle (P4) dials.
    y = _checkbox(state, "gd_enabled", "Detail on", x, y)
    y = _slider(state, "gd_noise_octaves", "Octaves", 1.0, 6.0, x, y)
    y = _slider(state, "gd_noise_wavelength_tiles", "Noise wl", 1.0, 6.0, x, y)
    y = _slider(state, "gd_adv_gain", "Adv gain", 0.0, 4.0, x, y)
    y = _slider(state, "gd_cycle_seconds", "Cycle s", 0.5, 6.0, x, y)
    y = _slider(state, "gd_erode_strength", "Erode", 0.0, 1.0, x, y)
    y = _slider(state, "gd_warp_px", "Warp px", 0.0, 8.0, x, y)
    y = _checkbox(state, "gd_dither_on", "Dither", x, y)
    y = _slider(state, "speckle_mode", "Speckle012", 0.0, 2.0, x, y)
    y = _slider(state, "speckle_amp", "Speckle amp", 0.0, 1.0, x, y)

    # -- §4.4b Water optics (live; pour with U to see them bite) --
    y = _section_header("Water [U=pour]", x, y)
    y = _slider(state, "water_glint_strength", "Glint", 0.0, 8.0, x, y)
    y = _slider(state, "water_roughness_base", "Roughness", 0.02, 0.5, x, y)
    y = _slider(state, "water_roughness_agitation", "Rough agit", 0.0, 2.0, x, y)
    y = _slider(state, "water_fog_density", "Fog dens", 0.2, 12.0, x, y)
    y = _slider(state, "water_refract_strength", "Refract", 0.0, 0.08, x, y)
    y = _slider(state, "water_r0", "R0 Fresnel", 0.0, 0.2, x, y)
    y = _slider(state, "water_ripple_scale", "Ripple scl", 0.0, 24.0, x, y)
    y = _slider(state, "water_alpha_scale", "Alpha scl", 0.0, 20.0, x, y)
    y = _slider(state, "water_alpha_min", "Alpha min", 0.0, 0.6, x, y)
    y = _slider(state, "water_alpha_max", "Alpha max", 0.4, 1.0, x, y)
    # Phase 2 (mood pass): caustics / foam / CA / wave size.
    y = _slider(state, "water_caustic_strength", "Caustic", 0.0, 8.0, x, y)
    y = _slider(state, "water_caustic_scale", "Caust scl", 0.0, 24.0, x, y)
    y = _slider(state, "water_foam_threshold", "Foam thr", 0.001, 0.1, x, y)
    y = _slider(state, "water_foam_intensity", "Foam int", 0.0, 2.0, x, y)
    y = _slider(state, "water_ca_amount", "Chrom ab", 0.0, 0.06, x, y)
    y = _slider(state, "water_wave_scale", "Wave scl", 0.2, 6.0, x, y)
    y = _slider(state, "water_ambient_amp", "Amb amp", 0.0, 0.3, x, y)
    # Heightmap attenuation (alpha-only; needs a level WITH a heightmap to bite).
    y = _slider(state, "water_height_scale", "Height scl", 0.0, 2.0, x, y)
    y = _slider(state, "water_height_edge", "Height edge", 0.01, 0.5, x, y)
    # height_floor = the "level 0" baseline relief (subtracted before scaling so
    # water clears the floor immediately). Inert on a level with no heightmap.
    y = _slider(state, "water_height_floor", "Height floor", 0.0, 1.0, x, y)

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

    # Keybind reminder — the FINAL studio bindings (keymap audited; O avoided).
    rl.draw_text("I=ignite  J=smoke  K=steam  C=door  1=lamps  2=beacon",
                 x, y, 10, rl.Color(150, 205, 150, 255))
    y += 12
    rl.draw_text("Space=pause T=spawn U=water F=flood P=tilt N=nz L=firelights",
                 x, y, 10, rl.Color(120, 120, 140, 255))
    y += 12
    rl.draw_text("WASD=pan Q/E=zoom G=sRGB V=water F1/F2/F4 (O avoided)",
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
