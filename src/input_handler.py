"""Game input — translates pyray polling into Simulation actions.

Single class :class:`InputHandler` that owns the small bit of
presentation-layer state that the input system needs (which unit is
selected, which order-placement mode the player is in, the current
grenade fuse setting, the current detonation slot, which phase is being
planned). The simulation is **not** told about these — they are pure UI
state. The handler reads pyray events and pushes the matching orders
into ``sim.apply_action``.

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

F5 is NOT remapped here — it's still the renderer's normal-map toggle
(see ``GameRenderer.poll_toggles``). The patch plan moved config reload
to Ctrl+R for this reason.
"""
from __future__ import annotations

import pyray as rl

from config import CFG
from simulation.gases import GAS_NAMES, N_GASES
from simulation.orders import (
    DET_START_PHASE1,
    ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT,
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE,
    Order,
)


class InputHandler:
    """Bundles input-state + pyray polling glue. One per main loop."""

    def __init__(self):
        self.selected_unit_id = None
        self.current_mode = ORDER_MOVE_ATTACK
        # Planning phase is remembered per-unit: first selection defaults
        # to Phase 1; Tab toggles only the selected unit's entry. So
        # switching between marines preserves where you left off.
        self.per_unit_phase: dict[int, int] = {}
        self.grenade_fuse = CFG.weapons.grenade.fuse_default_seconds
        self.det_slot = DET_START_PHASE1
        # DEBUG multi-gas spawn (engine/05 §6.2, M2): which gas the J key drops
        # under the cursor. K cycles white_smoke -> black_smoke -> poison ->
        # teargas -> fuel_gas. Pure UI state — the sim never sees it.
        self.selected_gas = 0   # WHITE_SMOKE (== GAS id 0)

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

        # Ctrl+R: hot-reload config.
        ctrl_held = (rl.is_key_down(K.KEY_LEFT_CONTROL) or
                     rl.is_key_down(K.KEY_RIGHT_CONTROL))
        if ctrl_held and rl.is_key_pressed(K.KEY_R):
            CFG.reload()

        # F8: manual recorder dump.
        if rl.is_key_pressed(K.KEY_F8) and sim.recorder is not None:
            sim.recorder.dump("manual")

        # I: DEBUG ignite the tile under the cursor (engine/06 fire tuning).
        # Drops fire anywhere — forces the tile flammable so Erik can light up
        # any spot without hunting for a wood wall. Writes gmap.fire directly
        # (not via sim.edit) so it lands immediately even while paused. Debug
        # only; guarded behind a single key, no gameplay path touches it.
        if rl.is_key_pressed(K.KEY_I):
            self._debug_ignite(sim, renderer)

        # K: DEBUG cycle which gas the spawn key drops (engine/05 §6.2, M2).
        # Prints the new selection so Erik can see what J will spawn.
        if rl.is_key_pressed(K.KEY_K):
            self.selected_gas = (self.selected_gas + 1) % N_GASES
            print(f"[debug] selected gas -> {self.selected_gas} "
                  f"({GAS_NAMES[self.selected_gas]})")

        # J: DEBUG spawn a blob of the currently-selected gas under the cursor
        # (engine/05 §6.2, M2). Writes gmap.gas[sel] directly so it lands
        # immediately even while paused — the colour render then sums it with any
        # other gases present. Debug-only; mirrors the I-ignite path.
        if rl.is_key_pressed(K.KEY_J):
            self._debug_spawn_gas(sim, renderer)

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
                    self.grenade_fuse = max(
                        CFG.weapons.grenade.fuse_min_seconds,
                        min(CFG.weapons.grenade.fuse_max_seconds,
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
    # Debug actions
    # ------------------------------------------------------------------
    def _debug_ignite(self, sim, renderer):
        """DEBUG: ignite a small patch at the tile under the mouse cursor.

        Forces the patch flammable and sets ``fire`` directly so a fire starts
        anywhere (no need for a wood wall) and lands immediately even while
        paused. A render/tuning aid only — no gameplay code reaches this path.
        """
        tile = renderer.mouse_to_tile()
        if tile is None:
            return
        fx, fy = tile
        gmap = sim.gmap
        h, w = gmap.fire.shape
        if not (0 <= fy < h and 0 <= fx < w):
            return
        # Light a 3x3 patch so it reads at typical zoom; clamp to bounds.
        y0, y1 = max(0, fy - 1), min(h, fy + 2)
        x0, x1 = max(0, fx - 1), min(w, fx + 2)
        gmap.flammable[y0:y1, x0:x1] = True   # force-flammable for debug
        gmap.fire[y0:y1, x0:x1] = 1.0          # full-intensity seed

    def _debug_spawn_gas(self, sim, renderer):
        """DEBUG: spawn a blob of the selected gas at the tile under the cursor.

        Writes the selected slice of ``gmap.gas`` directly (a small clamped
        patch at full density) so the coloured ray march picks it up immediately,
        even while paused. Mirrors :meth:`_debug_ignite`. ``gmap.smoke`` is a view
        of the black_smoke slice, so spawning black_smoke also fills ``smoke``.
        A render/tuning aid only — no gameplay code reaches this path.
        """
        tile = renderer.mouse_to_tile()
        if tile is None:
            return
        fx, fy = tile
        gmap = sim.gmap
        _, h, w = gmap.gas.shape
        if not (0 <= fy < h and 0 <= fx < w):
            return
        # A 3x3 patch at full density so it reads at typical zoom; clamp to bounds.
        y0, y1 = max(0, fy - 1), min(h, fy + 2)
        x0, x1 = max(0, fx - 1), min(w, fx + 2)
        gmap.gas[self.selected_gas, y0:y1, x0:x1] = 1.0

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


__all__ = ["InputHandler"]
