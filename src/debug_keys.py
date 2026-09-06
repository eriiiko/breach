"""The DEV keys — ignite / gas / water / door / tilt / weapon-cycle / dump.

Extracted verbatim from :mod:`input_handler` so two control schemes can share
one implementation. Behaviour is UNCHANGED for the shipped WEGO input, which
still calls :func:`handle_debug_keys` every frame.

Why it moved (onephase_wego design §17): "In game mode, the control scheme
owns every binding. All diagnostic/graphical toggles scattered today … are
removed from game-mode keys and retreat behind a debug mode." Erik's kickoff
ruling picked the hatch: ``main.py --control onephase --debug`` re-arms exactly
this set on exactly the keys it already uses, and without the flag OnePhaseWEGO
game mode has none of them. Designing the eventual debug game mode properly is
its own future evening (design §19 OUT); this module is the minimum eviction
that makes the clean keymap possible now.

Every function here writes world state DIRECTLY rather than through the field-
edit queue, on purpose: the queue only flushes in an unpaused step, and a
tuning aid has to land while you are staring at a paused frame. No gameplay
path reaches any of it.
"""
from __future__ import annotations

import math

import numpy as np
import pyray as rl

from config import CFG
from simulation.gases import GAS_NAMES, N_GASES


# How far above a material's own ignition point the `I` key heats its patch
# (game units == kelvin since G12). A DECISIVE igniter: measured survival is
# marginal within ~+25 of ignition and reliable above it (fire session #12
# lab sweeps, 2026-09-06) — a graze is what the future exposure-integral
# ignition (G8) will model, not this dev key.
DEBUG_IGNITE_MARGIN = 55.0


class DebugKeyState:
    """The tiny bit of state the dev keys carry (which gas J drops)."""

    def __init__(self):
        self.selected_gas = 0        # STEAM (== GAS id 0)


def handle_debug_keys(sim, renderer, state, selected_unit_id=None) -> None:
    """Poll and apply every dev key for this frame.

    Bindings (unchanged from the shipped input handler):

    - ``Ctrl+R`` reload config.toml · ``F8`` manual recorder dump
    - ``I`` ignite under the cursor · ``J`` spawn the selected gas ·
      ``K`` cycle that gas · ``U`` pour 0.2 m of water
    - ``O`` toggle the door under the cursor (A6 doors v0, the synced latch)
    - ``P`` / ``Shift+P`` tilt the ship +/- 2 degrees
    - ``N`` cycle the selected unit's weapon through the armory (W6)
    """
    K = rl.KeyboardKey

    ctrl_held = (rl.is_key_down(K.KEY_LEFT_CONTROL) or
                 rl.is_key_down(K.KEY_RIGHT_CONTROL))
    if ctrl_held and rl.is_key_pressed(K.KEY_R):
        CFG.reload()

    if rl.is_key_pressed(K.KEY_F8) and sim.recorder is not None:
        sim.recorder.dump("manual")

    if rl.is_key_pressed(K.KEY_I):
        debug_ignite(sim, renderer)

    if rl.is_key_pressed(K.KEY_K):
        state.selected_gas = (state.selected_gas + 1) % N_GASES
        print(f"[debug] selected gas -> {state.selected_gas} "
              f"({GAS_NAMES[state.selected_gas]})")

    if rl.is_key_pressed(K.KEY_J):
        debug_spawn_gas(sim, renderer, state.selected_gas)

    if rl.is_key_pressed(K.KEY_U):
        debug_pour_water(sim, renderer)

    if rl.is_key_pressed(K.KEY_N) and selected_unit_id is not None:
        new_weapon = sim.debug_cycle_weapon(selected_unit_id)
        if new_weapon is not None:
            u = sim.get_unit(selected_unit_id)
            print(f"[debug] {getattr(u, 'name', 'unit')} weapon -> "
                  f"{new_weapon}")

    if rl.is_key_pressed(K.KEY_O):
        debug_toggle_door(sim, renderer)

    if rl.is_key_pressed(K.KEY_P):
        shift_held = (rl.is_key_down(K.KEY_LEFT_SHIFT) or
                      rl.is_key_down(K.KEY_RIGHT_SHIFT))
        step = math.radians(-2.0 if shift_held else 2.0)
        lim = math.radians(20.0)
        gmap = sim.gmap
        gmap.tilt_x = max(-lim, min(lim, gmap.tilt_x + step))
        print(f"[debug] ship tilt_x -> {math.degrees(gmap.tilt_x):+.1f} deg")


def debug_ignite(sim, renderer) -> None:
    """Ignite a small patch at the tile under the cursor — as an IGNITER does.

    Forces the patch flammable so a fire starts anywhere (no need for a wood
    wall), and lands immediately even while paused.

    DELIVERS HEAT, not just a fire value (fire session #12, 2026-09-06): since
    R3 the O2 demand rides ``hotf = (T - fire_T_ext)/fire_T_span``, so a tile
    seeded at AMBIENT draws no oxygen, deposits no heat, and fades without ever
    burning its fuel (measured: I -> 0 in 60 s, wall_hp untouched). That is the
    physics — combustion needs the fuel bed at pyrolysis temperature — and the
    in-engine ignition path already respects it (it only ever fires on a tile
    whose temperature crossed its own ``ignition_temp``). This key predates R3
    and was the last cold-start bootstrap left; it now heats the patch the way
    the match/flamethrower that "pressed" it would, and seeds at
    ``ignition_seed`` so the in-game curve is the one the tuning lab plots
    (tools/fire_tuning_lab.py).

    The temperature write is SOLID-TILES-ONLY: a gas cell's ``temperature`` is
    the energy field's mirror and is never written directly (CLAUDE.md).
    """
    tile = renderer.mouse_to_tile()
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
    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.12))
    gmap.fire[y0:y1, x0:x1] = fire_fixed.quantize_scalar(seed_i)

    # Heat the SOLID tiles of the patch to their own ignition point plus a
    # margin that reliably takes (a decisive igniter, not a graze). Each
    # material's ignition point is `fire_T_ext_plane + ignition_to_ext_delta`
    # — the plane the solver itself subtracts, so this is per-material exact.
    delta = float(getattr(CFG.physics.fire, "ignition_to_ext_delta", 200.0))
    # `thermal_solid`, not `solid`: furniture is climbable/permeable (solid ==
    # False) yet is a thermal solid whose temperature the solver owns — and
    # thermal_solid is exactly the "not a gas cell" test the mirror rule wants.
    solid = gmap.thermal_solid[y0:y1, x0:x1]
    patch = gmap.temperature[y0:y1, x0:x1]
    t_ext = fire_fixed.dequantize(gmap.fire_T_ext_plane[y0:y1, x0:x1])
    hot_q = fire_fixed.quantize(t_ext + delta + DEBUG_IGNITE_MARGIN)
    gmap.temperature[y0:y1, x0:x1] = np.where(solid & (hot_q > patch),
                                              hot_q, patch).astype(patch.dtype)


def debug_spawn_gas(sim, renderer, gas_id: int) -> None:
    """Spawn a blob of ``gas_id`` under the cursor (engine/05 §6.2, M2)."""
    tile = renderer.mouse_to_tile()
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
    gmap.gas[gas_id, y0:y1, x0:x1] = gas_fixed.SMOKE_MAX_Q


def debug_pour_water(sim, renderer) -> None:
    """Pour 0.2 m of water on the tile under the cursor (water W2b)."""
    tile = renderer.mouse_to_tile()
    if tile is None:
        return
    fx, fy = tile
    gmap = sim.gmap
    h, w = gmap.water_depth.shape
    if not (0 <= fy < h and 0 <= fx < w):
        return
    if gmap.solid[fy, fx]:
        return
    from simulation import water_fixed
    cur_m = float(gmap.water_depth[fy, fx]) / water_fixed.FP_ONE_F
    gmap.water_depth[fy, fx] = water_fixed.quantize_scalar(min(cur_m + 0.2, 2.5))


def debug_toggle_door(sim, renderer) -> None:
    """Flip the ``want_open`` latch of the door under the cursor.

    ``mouse_to_tile()`` returns (x, y); ``door_at`` takes (fy, fx) — the same
    (col,row)->(row,col) flip every debug key applies (the N9 pin).
    """
    tile = renderer.mouse_to_tile()
    if tile is None:
        return
    fx, fy = tile
    door = sim.door_at(fy, fx)
    if door is None:
        print(f"[debug] no door at tile ({fx}, {fy})")
        return
    if not door.alive:
        print(f"[debug] door '{door.id}' at ({fx}, {fy}) is destroyed "
              f"— inputs are dead")
        return
    door.want_open = not door.want_open
    print(f"[debug] door '{door.id}' want_open -> {int(door.want_open)} "
          f"(state={door.state}; the 9e sweep applies/retries next unpaused "
          f"tick)")


__all__ = [
    "DebugKeyState", "debug_ignite", "debug_pour_water", "debug_spawn_gas",
    "debug_toggle_door", "handle_debug_keys",
]
