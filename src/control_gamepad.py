"""``GamepadDirect`` — direct action control for one possessed marine (P3).

Design: ``docs/control_modularity_design_2026-07-22.md`` §3b (gamepad FIRST —
Erik is traveling with a controller and no mouse). One :class:`ControlSource`
among several; ``main.py`` selects it with ``--control gamepad``, which also
picks the :class:`~simulation.ruleset.ContinuousRealtime` ruleset (no phases,
no pause) via :meth:`initial_ruleset`.

Each FRAME (60 Hz) this samples the raylib gamepad through pyray and pushes
per-tick intents at the possessed unit through the ``Simulation`` intent facade
(``set_move_dir`` / ``set_aim`` / ``set_trigger`` / ``throw_grenade_intent`` /
``use_intent``). The sim samples the current held state at each TICK boundary
(24 Hz, §5): aim is frame-rate-continuous in the control layer; physics is
tick-rate. Stick floats are quantized to Q16.16 at the seam
(``control_source.quantize_stick_direction``) so only fixed-point intents ever
enter the sim (the determinism iron rule).

Button / stick map (Xbox-style names; the raylib enum is layout-agnostic):

    Left stick     -> MOVE_DIR (walk; hold LB for SPRINT)
    Right stick    -> AIM (facing; holds last facing when centered)
    Right trigger  -> TRIGGER held/released (auto-fire at weapon cadence)
    A (south)      -> THROW grenade along aim/facing
    X (west)       -> USE (toggle an adjacent door)

The actual controller FEEL is Erik's HUMAN-TEST — this module is not driven in
the headless test suite (it needs a live pad). The one piece the tests DO pin
is the poll->intent quantization, which is the pure
:func:`control_source.quantize_stick_direction` (synthetic floats in,
fixed-point out); this class only wires that pure function to the pad and the
facade.
"""
from __future__ import annotations

import pyray as rl

from control_source import ControlSource, quantize_stick_direction
from simulation.orders import ORDER_MOVE_ATTACK, ORDER_SPRINT
from simulation.ruleset import ContinuousRealtime
from simulation.weapons import get_tables as weapon_tables

# Which physical gamepad (raylib supports several; player 1 is index 0).
_PAD = 0
# Deadzones: sticks rest slightly off-center on real hardware.
_MOVE_DEADZONE = 0.20
_AIM_DEADZONE = 0.25
# Trigger axis rests at -1.0 (released) and travels to +1.0 (fully pulled) on
# raylib's normalization; fire once past the midpoint.
_TRIGGER_ON = 0.0


class GamepadDirect(ControlSource):
    """Possess the first team-0 marine and drive it with a controller (§3b)."""

    def __init__(self):
        # The possessed unit id — bound lazily on the first frame a unit
        # exists (the editor's first SPAWN T / first team-0 spawn, §8).
        self.possessed_id = None
        # UI-state attributes main.py reads off every ControlSource for the
        # HUD/panel (see control_source.py). Direct control has no planning
        # phase and one fixed "mode".
        self.current_mode = ORDER_MOVE_ATTACK
        self.planning_phase = 0
        # Edge-detection state for the single-shot buttons (throw / use).
        self._prev_throw = False
        self._prev_use = False
        # Grenade fuse comes off the hand_grenade weapon row (same source the
        # WEGO input uses), read once at construction.
        self.grenade_fuse = weapon_tables().weapons.by_name[
            "hand_grenade"].fuse_default_seconds

    # ControlSource sim-construction hooks (§3a/§3b): gamepad => continuous,
    # unpaused.
    def initial_ruleset(self):
        return ContinuousRealtime()

    def starts_paused(self) -> bool:
        return False

    @property
    def selected_unit_id(self):
        """main.py reads this for the panel/selection highlight — the possessed
        unit IS the selection under direct control."""
        return self.possessed_id

    # ------------------------------------------------------------------
    def _bind_possessed(self, sim) -> None:
        """Lazily pick the possessed unit: the first living team-0 unit
        (default per §3b — the editor's first SPAWN T). Re-binds if the current
        possessed unit is gone (dead/removed)."""
        cur = sim.get_unit(self.possessed_id) if self.possessed_id is not None \
            else None
        if cur is not None and cur.alive:
            return
        for u in sim.units:
            if u.team == 0 and u.alive:
                self.possessed_id = u.id
                return
        self.possessed_id = None

    def _axis(self, axis) -> float:
        """Read one gamepad axis, defaulting to 0.0 if the enum is missing on
        this pyray build (belt-and-suspenders — the standard enums exist)."""
        try:
            return float(rl.get_gamepad_axis_movement(_PAD, axis))
        except Exception:
            return 0.0

    def handle_frame(self, sim, renderer) -> None:
        """Sample the pad and emit this frame's intents for the possessed unit.

        No-op when no controller is connected (``is_gamepad_available`` False)
        — the headless test path and a keyboard-only launch simply produce no
        intents, and the sim ticks on with the unit idle.
        """
        if not rl.is_gamepad_available(_PAD):
            return
        self._bind_possessed(sim)
        uid = self.possessed_id
        if uid is None:
            return

        GA = rl.GamepadAxis
        GB = rl.GamepadButton

        # ---- Left stick -> MOVE_DIR (walk; LB = sprint) ----
        lx = self._axis(GA.GAMEPAD_AXIS_LEFT_X)
        ly = self._axis(GA.GAMEPAD_AXIS_LEFT_Y)
        mdx, mdy, mmag = quantize_stick_direction(lx, ly, _MOVE_DEADZONE)
        if mmag == 0:
            sim.clear_move_dir(uid)
        else:
            sprint = rl.is_gamepad_button_down(
                _PAD, GB.GAMEPAD_BUTTON_LEFT_TRIGGER_1)  # LB
            speed_mode = ORDER_SPRINT if sprint else ORDER_MOVE_ATTACK
            sim.set_move_dir(uid, mdx, mdy, speed_mode)

        # ---- Right stick -> AIM (facing) ----
        rx = self._axis(GA.GAMEPAD_AXIS_RIGHT_X)
        ry = self._axis(GA.GAMEPAD_AXIS_RIGHT_Y)
        adx, ady, amag = quantize_stick_direction(rx, ry, _AIM_DEADZONE)
        if amag != 0:
            sim.set_aim(uid, adx, ady)

        # ---- Right trigger -> TRIGGER held/released (auto-fire) ----
        rt = self._axis(GA.GAMEPAD_AXIS_RIGHT_TRIGGER)
        sim.set_trigger(uid, rt > _TRIGGER_ON)

        # ---- A (south) -> THROW grenade (edge) along aim, else facing ----
        throw_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_FACE_DOWN)
        if throw_down and not self._prev_throw:
            # Throw along the current aim stick if aiming, else along facing.
            if amag != 0:
                tdx, tdy = adx, ady
            else:
                tdx, tdy = self._facing_dir(sim, uid)
            sim.throw_grenade_intent(uid, tdx, tdy, self.grenade_fuse)
        self._prev_throw = throw_down

        # ---- X (west) -> USE (edge) ----
        use_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_FACE_LEFT)
        if use_down and not self._prev_use:
            sim.use_intent(uid)
        self._prev_use = use_down

    def _facing_dir(self, sim, uid):
        """Q16.16 unit vector along the possessed unit's current facing — the
        THROW fallback when the aim stick is centered. Uses the deterministic
        integer trig kit (no libm on the sim path)."""
        from simulation import unit_fixed
        from simulation.intents import FP_ONE
        u = sim.get_unit(uid)
        if u is None:
            return (FP_ONE, 0)   # +X default
        # facing is math-style (Y-up); world Y is down -> negate the y term.
        dx = unit_fixed.cos_rad(u.facing)
        dy = -unit_fixed.sin_rad(u.facing)
        return (unit_fixed.quantize_scalar(dx), unit_fixed.quantize_scalar(dy))


__all__ = ["GamepadDirect"]
