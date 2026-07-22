"""Harness studio wiring (B2 P1): preset tolerance + cursor injection + door.

Imports tools/lighting_demo.py as a module (no window — main() is __main__-
guarded) and exercises the parts that are NOT the GL loop:

  - preset load TOLERATES old/renamed/missing keys (the gate item): an empty or
    partial preset dict packs into a full state without KeyError, and the new B2
    dials fall back to their defaults;
  - the tool-side cursor writes (_inject_fire / _inject_gas / _toggle_door) bite
    a real fire_studio sim (TOOLS may write sim fields);
  - --level parsing.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_lighting_demo_studio.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from level_loader import load  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.gases import SMOKE, STEAM  # noqa: E402
from simulation.materials import MAT_DOOR_CLOSED  # noqa: E402
from simulation.unit import Unit  # noqa: E402

_DEMO_PATH = ROOT / "tools" / "lighting_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("_lighting_demo", _DEMO_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


demo = _load_demo()


def _studio_sim():
    lvl = load("fire_studio")
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    for s in lvl.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    sim.set_paused(False)
    return sim


# ---------------------------------------------------------------------------
# Preset tolerance (gate item) + the B2 dials in DEFAULTS
# ---------------------------------------------------------------------------
def test_defaults_carry_b2_dials():
    d = demo.DEFAULTS
    # B2 P2 DELETED smoke_render_gamma (the gas-medium tau-curve subsumes it).
    assert "smoke_render_gamma" not in d
    for k in ("legacy_smoke_on", "gm_plume_k_scale", "gm_tau_curve_a",
              "gm_tau_curve_b", "gm_glow_gain", "gm_effect_gas_floor",
              "gm_fuel_haze_on", "gd_enabled", "gd_cycle_seconds",
              "speckle_mode", "speckle_amp", "soot_yield", "smoke_emission"):
        assert k in d, f"missing B2 dial {k!r} in DEFAULTS"


def test_empty_preset_tolerated():
    """An empty/old preset (no water table, no B2 keys) packs cleanly."""
    s = demo._toml_dict_to_state({})
    # Full state, all defaults — no KeyError, new dials at their defaults.
    assert s["gm_plume_k_scale"] == demo.DEFAULTS["gm_plume_k_scale"]
    assert s["gm_tau_curve_b"] == demo.DEFAULTS["gm_tau_curve_b"]
    assert set(s) >= set(demo.DEFAULTS)


def test_partial_old_preset_tolerated():
    """A preset carrying only legacy keys loads; B2 dials stay default."""
    old = {"light_gain": 12.0, "smoke_tint": [1, 2, 3], "smoke_max_alpha": 100}
    s = demo._toml_dict_to_state(old)
    assert s["light_gain"] == 12.0
    assert s["gm_glow_gain"] == demo.DEFAULTS["gm_glow_gain"]     # untouched
    assert s["gd_enabled"] == demo.DEFAULTS["gd_enabled"]


def test_load_missing_preset_returns_none():
    assert demo.load_preset("~~definitely-not-a-preset~~") is None


# ---------------------------------------------------------------------------
# Tool-side cursor injection + door toggle bite a real sim
# ---------------------------------------------------------------------------
def test_inject_fire_lights_the_cursor_patch():
    sim = _studio_sim()
    assert int(sim.gmap.fire.max()) == 0
    demo._inject_fire(sim, (12, 12))        # tile = (fx, fy)
    assert int(sim.gmap.fire.max()) > 0, "ignite must seed the fire field"
    # 3x3 patch around (12,12).
    assert int(sim.gmap.fire[12, 12]) > 0


def test_inject_gas_puffs_smoke_and_steam():
    sim = _studio_sim()
    demo._inject_gas(sim, (14, 12), SMOKE)
    demo._inject_gas(sim, (20, 12), STEAM)
    assert int(sim.gmap.gas[SMOKE, 12, 14]) > 0
    assert int(sim.gmap.gas[STEAM, 12, 20]) > 0
    # Cross-check the species isolation: smoke tile carries no steam.
    assert int(sim.gmap.gas[STEAM, 12, 14]) == 0


def test_toggle_door_flips_latch_and_opens():
    sim = _studio_sim()
    door = sim.door_at(5, 37)
    assert door is not None and door.want_open is False
    demo._toggle_door(sim, (37, 5))         # tile = (fx, fy)
    assert door.want_open is True
    for _ in range(3):
        sim.step()
    assert int(sim.gmap.material[5, 37]) != MAT_DOOR_CLOSED, "door opened"


def test_inject_none_tile_is_noop():
    sim = _studio_sim()
    demo._inject_fire(sim, None)
    demo._inject_gas(sim, None, SMOKE)
    demo._toggle_door(sim, None)
    assert int(sim.gmap.fire.max()) == 0


# ---------------------------------------------------------------------------
# --level parsing
# ---------------------------------------------------------------------------
def test_parse_level_arg(monkeypatch):
    monkeypatch.setattr(demo.sys, "argv",
                        ["lighting_demo.py", "--level", "fire_studio"])
    assert demo._parse_level_arg() == "fire_studio"
    monkeypatch.setattr(demo.sys, "argv", ["lighting_demo.py"])
    assert demo._parse_level_arg() is None


if __name__ == "__main__":
    test_defaults_carry_b2_dials()
    test_empty_preset_tolerated()
    test_partial_old_preset_tolerated()
    test_load_missing_preset_returns_none()
    test_inject_fire_lights_the_cursor_patch()
    test_inject_gas_puffs_smoke_and_steam()
    test_toggle_door_flips_latch_and_opens()
    test_inject_none_tile_is_noop()
    print("OK — lighting_demo studio wiring: presets tolerant, injection + door")
