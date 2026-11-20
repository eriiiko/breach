"""``GamepadHalf`` — the PAD half of direct control (P3; gameplay pack G1p).

Design: ``docs/control_modularity_design_2026-07-22.md`` §3b (gamepad first)
and ``docs/gameplay_pack_v1_design_2026-11-10.md`` §3 G1p. Since G1p there is
ONE direct-control scheme, :class:`control_direct.DirectControl`
(``--control direct``; ``--control gamepad`` is its alias): it reads the pad
AND the keyboard + mouse every frame and uses whichever device was touched
last. This module is the pad's half of it: it turns one frame of the raylib
gamepad into a :class:`control_direct.DeviceFrame` and never calls the sim's
intent facade itself (the scheme emits every device's frame through one
path, ``DirectControl._emit``).

Stick floats are quantized to Q16.16 at the seam
(``control_source.quantize_stick_direction``, and the stick's deflection to
the move THROTTLE through ``control_source.stick_throttle_q``) so only
fixed-point intents ever enter the sim (the determinism iron rule).

Button / stick map (Xbox-style names; the raylib enum is layout-agnostic):

    Left stick     -> MOVE_DIR; its deflection is the throttle (the edge of
                      the deadzone walks, full deflection sprints); LB held
                      = full throttle
    Right stick    -> AIM (facing; holds last facing when centered)
    Right trigger  -> TRIGGER held/released (auto-fire at weapon cadence)
    A (south)      -> THROW grenade along aim/facing
    X (west)       -> USE (toggle an adjacent door)
    Y (north)      -> RELOAD (discard the magazine, run the reload stall)
    B (east)       -> CYCLE the selected grenade kind (G3p; A throws it)
    RB             -> SHOVE the adjacent unit nearest the facing (G5p-a2)
    R3 (right stick click), held -> MELEE: strike the adjacent enemy nearest
                      the facing at the weapon's cadence (G6p)
    D-pad up       -> toggle the FLASHLIGHT (G7p)
    D-pad down     -> PLANT the charge at the feet (#193)
    D-pad left     -> step the plant TIMER (#193)

The actual controller FEEL is Erik's HUMAN-TEST — this module is not driven in
the headless test suite (it needs a live pad). What the tests pin is the pure
seam it wires to the pad (``quantize_stick_direction``, ``stick_throttle_q``).
"""
from __future__ import annotations

import pyray as rl

from control_source import quantize_stick_direction, stick_throttle_q

# Which physical gamepad (raylib supports several; player 1 is index 0).
_PAD = 0
# Deadzones: sticks rest slightly off-center on real hardware.
_MOVE_DEADZONE = 0.20
_AIM_DEADZONE = 0.25
# Trigger axis rests at -1.0 (released) and travels to +1.0 (fully pulled) on
# raylib's normalization; fire once past the midpoint.
_TRIGGER_ON = 0.0
#: Q16 full throttle (LB held).
_THROTTLE_ONE = 1 << 16


class GamepadHalf:
    """The pad's half of :class:`control_direct.DirectControl`: one frame of
    the controller as a ``DeviceFrame`` (or nothing when no pad is
    connected)."""

    def __init__(self):
        # Edge-detection state for the single-shot buttons (throw / use).
        self._prev_throw = False
        self._prev_use = False
        self._prev_reload = False
        self._prev_cycle = False
        self._prev_shove = False
        self._prev_flashlight = False
        self._prev_plant = False
        self._prev_timer = False

    @staticmethod
    def available() -> bool:
        return bool(rl.is_gamepad_available(_PAD))

    def _axis(self, axis) -> float:
        """Read one gamepad axis, defaulting to 0.0 if the enum is missing on
        this pyray build (belt-and-suspenders — the standard enums exist)."""
        try:
            return float(rl.get_gamepad_axis_movement(_PAD, axis))
        except Exception:
            return 0.0

    def touched(self) -> bool:
        """Was the pad used this frame: a stick past its deadzone, the
        trigger pulled, or any button down. False with no pad."""
        if not self.available():
            return False
        GA = rl.GamepadAxis
        sticks = ((GA.GAMEPAD_AXIS_LEFT_X, GA.GAMEPAD_AXIS_LEFT_Y,
                   _MOVE_DEADZONE),
                  (GA.GAMEPAD_AXIS_RIGHT_X, GA.GAMEPAD_AXIS_RIGHT_Y,
                   _AIM_DEADZONE))
        for ax, ay, dz in sticks:
            if quantize_stick_direction(self._axis(ax), self._axis(ay),
                                        dz)[2]:
                return True
        if self._axis(GA.GAMEPAD_AXIS_RIGHT_TRIGGER) > _TRIGGER_ON:
            return True
        return any(rl.is_gamepad_button_down(_PAD, b) for b in range(1, 18))

    def read(self, facing_dir):
        """This frame's pad as a ``DeviceFrame``; ``facing_dir()`` is the
        throw's fallback direction (the unit's facing) when the aim stick is
        centred. Updates the edge state, so call it once per frame."""
        from control_direct import DeviceFrame
        # DEBUG (human-test): log which raylib gamepad button index fires, so
        # the physical->enum mapping can be pinned (grenade/use came out on the
        # wrong physical button in the first human-test). One line per press.
        for _b in range(1, 18):  # raylib GamepadButton indices 1..17
            if rl.is_gamepad_button_pressed(_PAD, _b):
                print(f"[gamepad] raylib button index {_b} pressed")

        GA = rl.GamepadAxis
        GB = rl.GamepadButton

        # ---- Left stick -> MOVE_DIR; deflection = throttle (LB = full) ----
        mdx, mdy, mmag = quantize_stick_direction(
            self._axis(GA.GAMEPAD_AXIS_LEFT_X),
            self._axis(GA.GAMEPAD_AXIS_LEFT_Y), _MOVE_DEADZONE)
        move = None
        if mmag != 0:
            full = rl.is_gamepad_button_down(
                _PAD, GB.GAMEPAD_BUTTON_LEFT_TRIGGER_1)  # LB
            throttle = (_THROTTLE_ONE if full
                        else stick_throttle_q(mmag, _MOVE_DEADZONE))
            move = (mdx, mdy, throttle)

        # ---- Right stick -> AIM (facing) ----
        adx, ady, amag = quantize_stick_direction(
            self._axis(GA.GAMEPAD_AXIS_RIGHT_X),
            self._axis(GA.GAMEPAD_AXIS_RIGHT_Y), _AIM_DEADZONE)
        aim = (adx, ady) if amag != 0 else None

        # ---- Right trigger -> TRIGGER held/released (auto-fire) ----
        trigger = self._axis(GA.GAMEPAD_AXIS_RIGHT_TRIGGER) > _TRIGGER_ON

        # ---- A (south) -> THROW grenade (edge) along aim, else facing ----
        throw = None
        throw_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_FACE_DOWN)
        if throw_down and not self._prev_throw:
            tdx, tdy = aim if aim is not None else facing_dir()
            throw = (tdx, tdy, None)        # the fixed v0.1 reach
        self._prev_throw = throw_down

        # ---- X (west) -> USE (edge) ----
        use_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_FACE_LEFT)
        use = use_down and not self._prev_use
        self._prev_use = use_down

        # ---- Y (north) -> RELOAD (edge) ----
        reload_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_FACE_UP)
        reload = reload_down and not self._prev_reload
        self._prev_reload = reload_down

        # ---- B (east) -> CYCLE the grenade kind (edge) ----
        cycle_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_FACE_RIGHT)
        cycle = cycle_down and not self._prev_cycle
        self._prev_cycle = cycle_down

        # ---- RB -> SHOVE (edge; G5p-a2) ----
        shove_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_TRIGGER_1)
        shove = shove_down and not self._prev_shove
        self._prev_shove = shove_down

        # ---- R3 (right stick click), held -> MELEE (G6p) ----
        melee = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_RIGHT_THUMB)

        # ---- D-pad up -> toggle the FLASHLIGHT (edge; G7p) ----
        flash_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_LEFT_FACE_UP)
        flashlight = flash_down and not self._prev_flashlight
        self._prev_flashlight = flash_down

        # ---- D-pad down -> PLANT the charge (edge; #193) ----
        plant_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_LEFT_FACE_DOWN)
        plant = plant_down and not self._prev_plant
        self._prev_plant = plant_down

        # ---- D-pad left -> step the plant TIMER (edge; #193) ----
        timer_down = rl.is_gamepad_button_down(
            _PAD, GB.GAMEPAD_BUTTON_LEFT_FACE_LEFT)
        timer = timer_down and not self._prev_timer
        self._prev_timer = timer_down

        return DeviceFrame(move=move, aim=aim, trigger=trigger, throw=throw,
                           use=use, reload=reload, cycle_grenade=cycle,
                           shove=shove, melee=melee,
                           flashlight_toggle=flashlight,
                           plant_charge=plant, cycle_charge_timer=timer)


__all__ = ["GamepadHalf"]
