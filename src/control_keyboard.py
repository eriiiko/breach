"""``KeyboardDirect`` -- direct action control for one unit by keyboard + mouse
(#106 P6f), the keyboard twin of :class:`control_gamepad.GamepadDirect`.

``main.py --control keyboard`` selects it; like the gamepad it runs under
:class:`~simulation.ruleset.ContinuousRealtime` (no phases, no pause) and
drives ONE unit -- the first living unit of ``sim.player_team`` (the squad's
first marine; under ``--play creature`` the creature) -- through the per-tick
intent facade (``set_move_dir`` / ``set_aim`` / ``set_trigger`` /
``use_intent``). Built so a human can drive the octopus without a pad
(Erik, 2026-10-31: "can I make it move?").

Key map:

    W A S D       -> MOVE_DIR (8 directions) at FULL speed (``speed_q`` ONE)
    + Left Shift  -> ... at ``[control.keyboard] slow_speed`` (the octopus:
                     its full crawl)
    + Left Ctrl   -> ... at ``[control.keyboard] mid_speed`` (the octopus:
                     its umbrella cruise)
    RMB held      -> AIM toward the mouse cursor (a creature reaches; released,
                     the aim is cleared)
    LMB held      -> TRIGGER (a marine fires; a creature is unarmed)
    F             -> USE (a creature grabs its prop; a marine toggles a door)

WASD steer the unit here, so the renderer does not pan the camera with them
(``pans_camera_with_wasd`` False); the ARROWS still pan. The marines' speed
mode is ``ORDER_MOVE_ATTACK`` (Shift / Ctrl only set the analog speed, which
the marines' law never reads).

The poll is injectable (``key_down`` / ``key_pressed`` / ``mouse_down``
callables, raylib's by default) so the scheme runs headless in the tests with
``renderer=None``; the pure half is :func:`keyboard_move`.
"""
from __future__ import annotations

import math

from control_source import (ControlSource, first_controlled_unit,
                            quantize_stick_direction)
from simulation.orders import ORDER_MOVE_ATTACK
from simulation.ruleset import ContinuousRealtime

ONE = 1 << 16


def keyboard_settings(cfg=None) -> tuple[float, float]:
    """``[control.keyboard]`` (``slow_speed``, ``mid_speed``): fractions of the
    unit's maximum speed, refused when missing, mistyped or outside [0, 1]."""
    if cfg is None:
        from config import CFG
        cfg = CFG
    try:
        tab = cfg.control.keyboard
    except AttributeError:
        raise ValueError("config.toml [control.keyboard] is missing") from None
    out = []
    for key in ("slow_speed", "mid_speed"):
        v = getattr(tab, key, None)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"[control.keyboard] {key} must be a number (got {v!r})")
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"[control.keyboard] {key} must be in [0, 1] (got {v})")
        out.append(float(v))
    return out[0], out[1]


def keyboard_move(up: bool, down: bool, left: bool, right: bool, slow: bool, mid: bool,
                  slow_speed: float, mid_speed: float):
    """The pure key -> MOVE_DIR map: ``(dx_q, dy_q, speed_q)`` -- a Q16 unit
    vector (screen axes: up = -y) and the Q16 analog speed (ONE; Shift ->
    ``slow_speed``, else Ctrl -> ``mid_speed``) -- or None when no direction
    is held (opposite keys cancel)."""
    dx = (1 if right else 0) - (1 if left else 0)
    dy = (1 if down else 0) - (1 if up else 0)
    if dx == 0 and dy == 0:
        return None
    n = math.sqrt(dx * dx + dy * dy)
    qx, qy, _ = quantize_stick_direction(dx / n, dy / n, deadzone=0.0)
    frac = slow_speed if slow else (mid_speed if mid else 1.0)
    from simulation.unit_fixed import quantize_scalar
    return qx, qy, max(0, min(ONE, quantize_scalar(frac)))


def _rl_key(name):
    import pyray as rl
    return getattr(rl.KeyboardKey, f"KEY_{name}")


class KeyboardDirect(ControlSource):
    """Drive the first living unit of ``sim.player_team`` with WASD (+ Shift /
    Ctrl speeds), the mouse (aim, fire) and F (use)."""

    drives_one_unit = True
    pans_camera_with_wasd = False

    def __init__(self, key_down=None, key_pressed=None, mouse_down=None, cfg=None):
        self.slow_speed, self.mid_speed = keyboard_settings(cfg)
        if key_down is None or key_pressed is None or mouse_down is None:
            import pyray as rl
            key_down = key_down or (lambda name: rl.is_key_down(_rl_key(name)))
            key_pressed = key_pressed or (lambda name: rl.is_key_pressed(_rl_key(name)))
            mouse_down = mouse_down or (lambda b: rl.is_mouse_button_down(
                getattr(rl.MouseButton, f"MOUSE_BUTTON_{b}")))
        self._down, self._pressed, self._mouse = key_down, key_pressed, mouse_down
        self.possessed_id = None
        # UI state main.py reads off every ControlSource
        self.current_mode = ORDER_MOVE_ATTACK
        self.planning_phase = 0

    def initial_ruleset(self):
        return ContinuousRealtime()

    def starts_paused(self) -> bool:
        return False

    @property
    def selected_unit_id(self):
        """The driven unit IS the selection (main.py's panel / highlight)."""
        return self.possessed_id

    def handle_frame(self, sim, renderer) -> None:
        self.possessed_id = first_controlled_unit(sim, self.possessed_id)
        uid = self.possessed_id
        if uid is None:
            return
        down = self._down
        mv = keyboard_move(down("W"), down("S"), down("A"), down("D"),
                           down("LEFT_SHIFT"), down("LEFT_CONTROL"),
                           self.slow_speed, self.mid_speed)
        if mv is None:
            sim.clear_move_dir(uid)
        else:
            sim.set_move_dir(uid, mv[0], mv[1], ORDER_MOVE_ATTACK, speed_q=mv[2])
        # RMB -> AIM toward the cursor (needs the renderer's tile mapping);
        # released -> no aim (a creature's reach ends)
        if not self._mouse("RIGHT"):
            sim.clear_aim(uid)
        elif renderer is not None:
            u = sim.get_unit(uid)
            at = renderer.mouse_to_tile_float()
            if u is not None and at is not None:
                ax = float(at[0]) - (u.x + u.footprint / 2.0)
                ay = float(at[1]) - (u.y + u.footprint / 2.0)
                n = math.sqrt(ax * ax + ay * ay)
                if n > 0.0:
                    qx, qy, _ = quantize_stick_direction(ax / n, ay / n, deadzone=0.0)
                    sim.set_aim(uid, qx, qy)
        sim.set_trigger(uid, bool(self._mouse("LEFT")))
        if self._pressed("F"):
            sim.use_intent(uid)


__all__ = ["KeyboardDirect", "keyboard_move", "keyboard_settings"]
