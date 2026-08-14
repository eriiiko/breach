"""Game input — translates pyray polling into Simulation actions.

Single class :class:`WEGOPlanningInput` (P2, control_modularity design
§3b: renamed/wrapped from the original ``InputHandler`` — behavior
UNCHANGED, ``InputHandler`` stays as a compatibility alias) that owns the
small bit of presentation-layer state that the input system needs (which
unit is selected, which order-placement mode the player is in, the
current grenade fuse setting, the current detonation slot, which phase is
being planned). The simulation is **not** told about these — they are
pure UI state. The handler reads pyray events and pushes the matching
orders into ``sim.apply_action``.

This is one of possibly several :class:`~control_source.ControlSource`
implementations (the WEGO planning/keyboard/mouse one); ``main.py``
selects which to instantiate via ``--control`` (default ``wego``, this
class). See ``control_source.py`` for the seam and the other planned
sources (``GamepadDirect``, ``AgentPolicy`` — both P3, not built yet).

Why not put this on the renderer? Because the renderer is supposed to
draw, not interpret intent. Keeping input here means the renderer stays
swappable.

Key bindings (decisions locked this morning):

- Click a marine → select it.
- Right-click a destination, or left-click after selection → place an
  order of the current mode.
- 1 / 2 / 3 → Move&Attack / Move w/Cover / Sprint
- F → Fire mode (then click target)
- G → Grenade mode (then click target, scroll wheel sets fuse seconds)
- B → Door explosive mode (scroll wheel cycles detonation slot)
- Tab → switch planning phase (Phase 1 ↔ Phase 2)
- Spacebar → toggle pause (resume execution / pause)
- Backspace → undo last order
- Ctrl+R → reload config.toml from disk
- Esc → clear selection / cancel mode
- F8 → manual physics recorder dump
- I → DEBUG ignite the tile under the cursor
- J → DEBUG spawn the selected gas under the cursor
- K → DEBUG cycle the selected gas (white→black→poison→teargas→fuel)
- U → DEBUG pour 0.2 m of water on the tile under the cursor
- N → DEBUG cycle the selected unit's weapon through the armory (W6)
- O → DEBUG toggle the door entity under the cursor (A6 doors v0 — flips
  the synced want_open latch; the KEY is dev-only, ruling 5. The
  renderer's water-optics toggle moved O → V to free this key.)
- P / Shift+P → DEBUG tilt the ship: tilt_x +2/-2 degrees (clamped to +/-20)

F5 is NOT remapped here — it's still the renderer's normal-map toggle
(see ``GameRenderer.poll_toggles``). The patch plan moved config reload
to Ctrl+R for this reason.
"""
from __future__ import annotations

import pyray as rl

from config import CFG
from control_source import ControlSource
from debug_keys import DebugKeyState, handle_debug_keys
from simulation.weapons import get_tables as weapon_tables
from simulation.orders import (
    DET_START_PHASE1,
    ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT,
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE,
    Order,
)


class WEGOPlanningInput(ControlSource):
    """Bundles input-state + pyray polling glue. One per main loop."""

    def __init__(self):
        self.selected_unit_id = None
        self.current_mode = ORDER_MOVE_ATTACK
        # Planning phase is remembered per-unit: first selection defaults
        # to Phase 1; Tab toggles only the selected unit's entry. So
        # switching between marines preserves where you left off.
        self.per_unit_phase: dict[int, int] = {}
        # Fuse knobs come off the hand_grenade weapon row (mechanics/03 W1
        # re-home — same numbers the old CFG.weapons.grenade.* keys held).
        self.grenade_fuse = weapon_tables().weapons.by_name[
            "hand_grenade"].fuse_default_seconds
        self.det_slot = DET_START_PHASE1
        # DEBUG key state (which gas the J key drops). Lives in `debug_keys`
        # now so both control schemes share one implementation; the
        # ``selected_gas`` property below keeps the old attribute working.
        self._debug_state = DebugKeyState()

    @property
    def selected_gas(self) -> int:
        return self._debug_state.selected_gas

    @selected_gas.setter
    def selected_gas(self, value: int) -> None:
        self._debug_state.selected_gas = int(value)

    @property
    def planning_phase(self) -> int:
        """Current planning phase for the selected unit. 0 if no selection."""
        if self.selected_unit_id is None:
            return 0
        return self.per_unit_phase.get(self.selected_unit_id, 0)

    # ------------------------------------------------------------------
    # Per-frame poll. Returns nothing; side effects on ``sim``.
    # ------------------------------------------------------------------
    def handle_frame(self, sim, renderer):
        """Run all input polling for the current frame.

        ``sim`` is the :class:`simulation.Simulation`. ``renderer`` is
        the :class:`renderer.GameRenderer` (used only for the
        ``mouse_to_tile()`` conversion + selecting which key bindings
        the renderer still owns, e.g. F1-F5 stays renderer-side).
        """
        # ---- Global keys ----
        K = rl.KeyboardKey

        # The DEV keys (Ctrl+R reload, F8 dump, I/J/K/U/N/O/P) moved verbatim
        # into `debug_keys` so the OnePhaseWEGO scheme can gate the SAME set
        # behind --debug (onephase_wego design §17). Behaviour here is
        # unchanged: this scheme still polls them every frame, as it always has.
        handle_debug_keys(sim, renderer, self._debug_state,
                          self.selected_unit_id)

        # Spacebar: pause toggle. If we're resuming AND there are queued
        # grenade orders, materialise them before time starts flowing.
        if rl.is_key_pressed(K.KEY_SPACE):
            was_paused = sim.is_paused()
            sim.set_paused(not was_paused)
            if was_paused and not sim.is_paused():
                # Starting execution — spawn pending grenade projectiles
                # at the unit's planned end position. Mirrors the legacy
                # _start_execution path. Only run when transitioning
                # paused -> running on the FIRST tick of the round, so
                # we don't double-spawn mid-round.
                if sim.tick == 0:
                    sim.spawn_projectiles_from_grenade_orders()

        # Backspace: undo last order on the selected unit.
        if rl.is_key_pressed(K.KEY_BACKSPACE) and self.selected_unit_id is not None:
            sim.undo_last_order(self.selected_unit_id)

        # Tab: switch the SELECTED unit's planning phase. No-op if no
        # selection (planning a phase without a unit makes no sense).
        if rl.is_key_pressed(K.KEY_TAB) and self.selected_unit_id is not None:
            cur = self.per_unit_phase.get(self.selected_unit_id, 0)
            self.per_unit_phase[self.selected_unit_id] = 1 - cur

        # Esc: clear selection / reset mode.
        if rl.is_key_pressed(K.KEY_ESCAPE):
            if self.selected_unit_id is not None:
                self.selected_unit_id = None
            else:
                self.current_mode = ORDER_MOVE_ATTACK

        # Mode hotkeys (planning-only).
        if sim.is_paused():
            if rl.is_key_pressed(K.KEY_ONE):
                self.current_mode = ORDER_MOVE_ATTACK
            elif rl.is_key_pressed(K.KEY_TWO):
                self.current_mode = ORDER_MOVE_COVER
            elif rl.is_key_pressed(K.KEY_THREE):
                self.current_mode = ORDER_SPRINT
            elif rl.is_key_pressed(K.KEY_F):
                self.current_mode = ORDER_FIRE
            elif rl.is_key_pressed(K.KEY_G):
                self.current_mode = ORDER_GRENADE
            elif rl.is_key_pressed(K.KEY_B):
                # B is "explosive" only while planning — outside planning it
                # remains the renderer's bilinear toggle (mode-keys are
                # ignored when sim is running).
                self.current_mode = ORDER_EXPLOSIVE

            # Mouse wheel adjusts grenade fuse or detonation slot in
            # planning. (Renderer also reads the wheel for zoom — both
            # fire; we accept the dual binding for v1.)
            wheel = rl.get_mouse_wheel_move()
            if wheel != 0:
                if self.current_mode == ORDER_GRENADE:
                    hand_grenade = weapon_tables().weapons.by_name[
                        "hand_grenade"]
                    self.grenade_fuse = max(
                        hand_grenade.fuse_min_seconds,
                        min(hand_grenade.fuse_max_seconds,
                            self.grenade_fuse + wheel * 0.5))
                elif self.current_mode == ORDER_EXPLOSIVE:
                    self.det_slot = int((self.det_slot
                                          + (1 if wheel > 0 else -1)) % 3)

        # ---- Mouse clicks (planning-only) ----
        if sim.is_paused() and rl.is_mouse_button_pressed(
                rl.MouseButton.MOUSE_BUTTON_LEFT):
            self._handle_left_click(sim, renderer)
        elif sim.is_paused() and rl.is_mouse_button_pressed(
                rl.MouseButton.MOUSE_BUTTON_RIGHT):
            self._handle_right_click(sim, renderer)

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------
    def _handle_left_click(self, sim, renderer):
        tile = renderer.mouse_to_tile()
        if tile is None:
            return
        fx, fy = tile

        # Try to select a marine first.
        for u in sim.units:
            if (u.alive and u.team == 0 and
                    u.tile_x <= fx < u.tile_x + u.footprint
                    and u.tile_y <= fy < u.tile_y + u.footprint):
                self.selected_unit_id = u.id
                # First-time selection defaults to Phase 1.
                self.per_unit_phase.setdefault(u.id, 0)
                return
        # Otherwise place an order with the current mode.
        if self.selected_unit_id is not None:
            self._place_order(sim, fx, fy)

    def _handle_right_click(self, sim, renderer):
        if self.selected_unit_id is None:
            return
        tile = renderer.mouse_to_tile()
        if tile is None:
            return
        fx, fy = tile
        self._place_order(sim, fx, fy)

    def _place_order(self, sim, fx, fy):
        mode = self.current_mode
        phase = self.planning_phase

        if mode in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
            # Move target snapped so unit center is under cursor; clamp
            # to map bounds. Use selected unit's footprint if available.
            u = sim.get_unit(self.selected_unit_id)
            fp = u.footprint if u is not None else 3
            tx = fx - fp // 2
            ty = fy - fp // 2
            h, w = sim.gmap.material.shape
            tx = max(0, min(w - fp, tx))
            ty = max(0, min(h - fp, ty))
            order = Order(mode, tx, ty, phase)
            sim.apply_action(self.selected_unit_id, order)
        elif mode == ORDER_GRENADE:
            order = Order(ORDER_GRENADE, fx, fy, phase,
                          grenade_fuse=self.grenade_fuse)
            sim.apply_action(self.selected_unit_id, order)
        elif mode == ORDER_EXPLOSIVE:
            order = Order(ORDER_EXPLOSIVE, fx, fy, phase,
                          det_slot=self.det_slot)
            sim.apply_action(self.selected_unit_id, order)
        elif mode == ORDER_FIRE:
            order = Order(ORDER_FIRE, fx, fy, phase)
            sim.apply_action(self.selected_unit_id, order)


# Compatibility alias (P2): pre-rename call sites/imports that still say
# ``InputHandler`` keep working unchanged — same class, same behavior.
InputHandler = WEGOPlanningInput

__all__ = ["WEGOPlanningInput", "InputHandler"]
