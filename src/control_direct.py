"""``DirectControl`` — THE direct-control scheme (gameplay pack v1, G1p).

Design: ``docs/gameplay_pack_v1_design_2026-11-10.md`` §3 G1p (on
``docs/control_modularity_design_2026-07-22.md`` §3b). ``main.py --control
direct`` (``--control gamepad`` is an alias) possesses the first living
marine of ``sim.player_team`` and drives it through the sim's per-tick intent
facade (``set_move_dir`` / ``set_aim`` / ``set_trigger`` /
``throw_grenade_intent`` / ``use_intent`` / ``reload_intent`` /
``shove_intent`` / ``melee_intent`` / ``set_flashlight``) under
``ContinuousRealtime``.

Every frame it reads BOTH devices and uses whichever was touched last (Erik:
"the player plays with whatever they have connected"). A device is a HALF:
it turns one frame of its hardware into a :class:`DeviceFrame` and never
calls the facade; :meth:`DirectControl._emit` is the one place a frame
becomes intents. A new device is a half, never a second scheme.

- the PAD half: ``control_gamepad.GamepadHalf``;
- the KEYBOARD + MOUSE half: :class:`KeyboardMouseHalf` here. Keys are DATA,
  the ``config.toml [controls.keys]`` table (action -> key name), read
  through the refusing door :func:`key_table`. Movement is screen-relative
  (the ``"centred"`` camera is north-up): forward is up, strafes are left /
  right; the ``full_speed`` key holds the throttle at ONE (the sprint),
  released it is 0 (the walk). The AIM is the mouse: the vector from the
  marine's centre to the cursor's world position through
  ``control_source.quantize_point_aim`` (the stick's own seam); the throw
  goes to the cursor, its reach the distance, clamped by the sim as every
  reach is. The mouse is ignored while ``GameRenderer.pointer_captured()``
  (the light-curve panel owns it).

Every action of the table is live now (:data:`INERT_ACTIONS` is empty).
``melee`` (G6p; the right mouse button, pad R3) is ``melee_intent``: the
button HELD sends it every frame and the sim's own cadence (the weapon row's)
decides when a blow lands, so holding the button keeps striking the adjacent
enemy nearest the facing. ``flashlight`` (G7p; Q, pad D-pad up) TOGGLES the
possessed unit's light: the control keeps the on/off state
(:attr:`DirectControl.flashlight_off`, seeded from the unit's own bit when a
unit is possessed) and holds it with ``set_flashlight`` every frame, as the
trigger is held. ``shove`` (G5p-a2;
Space, pad RB) is ``shove_intent``: pressed once it pushes the adjacent unit
nearest the facing through the unit impulse law.
``reload`` (G2p) is ``reload_intent``: pressed once it discards the magazine
in the gun and runs the reload stall. ``cycle_grenade`` (G3p; key 4, pad B)
steps the SELECTED grenade kind through the kinds the unit still carries and
``throw`` throws the selected kind (``throw_grenade_intent(..., kind=)``):
the selection is control-layer state (:attr:`DirectControl.grenade_kind`,
never a sim field), moves on by itself to the next carried kind when the
selected one runs out, and the HUD reads it (``ui.model.panel_state_lines``,
``grenade_kind=``). The camera (``[display] camera_mode``, the
:attr:`camera_mode` main.py reads) is the follow camera's, never camera code
here.

Pyray is imported lazily by the halves' readers, so the key-table door is a
pure function (``tests/test_control_direct.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

from control_source import (ControlSource, quantize_point_aim,
                            quantize_stick_direction)

#: Every action the key table must bind, in table order.
KEY_ACTIONS = ("move_forward", "move_back", "strafe_left", "strafe_right",
               "trigger", "melee", "shove", "reload", "throw",
               "cycle_grenade", "flashlight", "use", "full_speed")
#: Bound but with no verb behind them yet: none since G7p (melee G6p,
#: flashlight G7p, shove G5p-a2 are live). A new bound-but-inert action is
#: listed here until its verb lands.
INERT_ACTIONS = ()

#: Q16 one -- the throttle of a held ``full_speed``.
_THROTTLE_ONE = 1 << 16
#: A cursor within this many tiles of the marine's centre names no aim.
MOUSE_DEADZONE_TILES = 0.25


def _key_names() -> dict:
    """The key-name vocabulary -> ``("key" | "mouse", raylib code)``. The
    codes are raylib's (``KeyboardKey`` / ``MouseButton``; letters and digits
    are their ASCII codes), written here so the door needs no pyray;
    ``tests/test_control_direct.py`` holds them equal to pyray's enums."""
    names = {}
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        names[c] = ("key", ord(c))
    names.update({
        "SPACE": ("key", 32), "ESCAPE": ("key", 256), "ENTER": ("key", 257),
        "TAB": ("key", 258), "BACKSPACE": ("key", 259),
        "RIGHT": ("key", 262), "LEFT": ("key", 263), "DOWN": ("key", 264),
        "UP": ("key", 265),
        "LEFT_SHIFT": ("key", 340), "LEFT_CONTROL": ("key", 341),
        "LEFT_ALT": ("key", 342), "RIGHT_SHIFT": ("key", 344),
        "RIGHT_CONTROL": ("key", 345), "RIGHT_ALT": ("key", 346),
        "MOUSE_LEFT": ("mouse", 0), "MOUSE_RIGHT": ("mouse", 1),
        "MOUSE_MIDDLE": ("mouse", 2),
    })
    for i in range(1, 13):
        names[f"F{i}"] = ("key", 289 + i)
    return names


KEY_NAMES = _key_names()


@dataclass(frozen=True)
class Binding:
    """One bound action: its key name and the raylib (device, code)."""
    name: str
    device: str        # "key" | "mouse"
    code: int


@dataclass(frozen=True)
class KeyTable:
    """The parsed ``[controls.keys]`` table: action -> :class:`Binding`."""
    bindings: dict

    def __getitem__(self, action: str) -> Binding:
        return self.bindings[action]


def key_table(cfg) -> KeyTable:
    """THE refusing door of ``[controls.keys]`` (a loaded config, ``CFG``):
    every action of :data:`KEY_ACTIONS` bound to one known key name. Raises
    ValueError naming the action or the key when the table is missing, an
    action is missing or unknown, a key name is unknown (or not a string),
    or one key is bound to two actions -- a silent default or a doubled key
    would hide a typo."""
    controls = getattr(cfg, "controls", None)
    section = getattr(controls, "keys", None) if controls is not None else None
    if section is None:
        raise ValueError("config.toml has no [controls.keys] table")
    given = {k: v for k, v in vars(section).items() if not k.startswith("_")}
    unknown = sorted(set(given) - set(KEY_ACTIONS))
    if unknown:
        raise ValueError(f"[controls.keys] names unknown action(s) {unknown} "
                         f"(known: {list(KEY_ACTIONS)})")
    bindings = {}
    used = {}
    for action in KEY_ACTIONS:
        if action not in given:
            raise ValueError(f"[controls.keys] is missing {action!r}")
        name = given[action]
        if not isinstance(name, str) or name.upper() not in KEY_NAMES:
            raise ValueError(f"[controls.keys] {action} = {name!r} is not a "
                             f"known key name")
        name = name.upper()
        if name in used:
            raise ValueError(f"[controls.keys] key {name!r} is bound to both "
                             f"{used[name]!r} and {action!r}")
        used[name] = action
        device, code = KEY_NAMES[name]
        bindings[action] = Binding(name, device, code)
    return KeyTable(bindings)


@dataclass
class DeviceFrame:
    """One frame of one device, in the intent facade's currency (Q16.16).

    ``move`` ``(dx_q, dy_q, throttle_q)`` or None (stop: ``clear_move_dir``);
    ``aim`` ``(dx_q, dy_q)`` or None (keep the facing); ``trigger`` held;
    ``throw`` ``(dx_q, dy_q, reach_q | None)`` on the frame it was pressed;
    ``use``, ``reload``, ``cycle_grenade``, ``shove`` and
    ``flashlight_toggle`` True on the frame they were pressed; ``melee`` True
    on every frame its button is held (the sim's cadence spaces the blows)."""
    move: tuple | None
    aim: tuple | None
    trigger: bool
    throw: tuple | None
    use: bool
    reload: bool = False
    cycle_grenade: bool = False
    shove: bool = False
    melee: bool = False
    flashlight_toggle: bool = False


def next_carried_kind(counts, current):
    """The grenade kind to select next (pure): the first kind AFTER
    ``current`` (wrapping round) of which the unit still carries one
    (``counts[k] > 0``); ``current`` itself when it is the only kind carried;
    ``None`` when the unit carries none. ``current`` may be None (nothing
    selected yet: the search starts at kind 0)."""
    n = len(counts)
    start = -1 if current is None else int(current)
    for step in range(1, n + 1):
        k = (start + step) % n
        if counts[k] > 0:
            return k
    return None


def keyboard_move(forward: bool, back: bool, left: bool, right: bool,
                  full_speed: bool):
    """The keyboard's move as a ``DeviceFrame.move`` (pure): screen-relative
    (north-up: forward is -y), opposite keys cancel, a diagonal is a unit
    vector through the stick seam, throttle ONE with ``full_speed`` held,
    else 0. None when no direction is held."""
    sx = (1.0 if right else 0.0) - (1.0 if left else 0.0)
    sy = (1.0 if back else 0.0) - (1.0 if forward else 0.0)
    dx_q, dy_q, mag = quantize_stick_direction(sx, sy, 0.0)
    if mag == 0:
        return None
    return (dx_q, dy_q, _THROTTLE_ONE if full_speed else 0)


class KeyboardMouseHalf:
    """The keyboard + mouse half of :class:`DirectControl`, bound by a
    :class:`KeyTable`."""

    def __init__(self, keys: KeyTable):
        self.keys = keys
        self._last_mouse = None

    @staticmethod
    def _rl():
        import pyray as rl
        return rl

    def _down(self, action: str) -> bool:
        rl = self._rl()
        b = self.keys[action]
        if b.device == "mouse":
            return bool(rl.is_mouse_button_down(b.code))
        return bool(rl.is_key_down(b.code))

    def _pressed(self, action: str) -> bool:
        rl = self._rl()
        b = self.keys[action]
        if b.device == "mouse":
            return bool(rl.is_mouse_button_pressed(b.code))
        return bool(rl.is_key_pressed(b.code))

    def touched(self) -> bool:
        """Was the keyboard or mouse used this frame: any bound key or
        button down, or the mouse moved."""
        rl = self._rl()
        mouse = (rl.get_mouse_x(), rl.get_mouse_y())
        moved = self._last_mouse is not None and mouse != self._last_mouse
        self._last_mouse = mouse
        return moved or any(self._down(a) for a in KEY_ACTIONS)

    def read(self, unit, renderer, facing_dir):
        """This frame's keyboard + mouse as a ``DeviceFrame`` for ``unit``
        (the possessed marine). The mouse names the aim only over the map
        and while the renderer does not capture the pointer; the trigger and
        the throw need it then too (a click on the panel is the panel's)."""
        move = keyboard_move(self._down("move_forward"),
                             self._down("move_back"),
                             self._down("strafe_left"),
                             self._down("strafe_right"),
                             self._down("full_speed"))
        point = None
        captured = renderer is None or renderer.pointer_captured()
        if not captured:
            point = renderer.mouse_to_tile_float()
        aim = None
        reach_q = None
        if point is not None:
            half = 0.5 * float(getattr(unit, "footprint", 1))
            dx_q, dy_q, reach_q = quantize_point_aim(
                float(unit.x) + half, float(unit.y) + half,
                point[0], point[1], MOUSE_DEADZONE_TILES)
            if dx_q or dy_q:
                aim = (dx_q, dy_q)
            else:
                reach_q = None
        trigger = (not captured) and self._down("trigger")
        throw = None
        if (not captured) and self._pressed("throw"):
            throw = ((aim[0], aim[1], reach_q) if aim is not None
                     else (*facing_dir(), None))
        use = self._pressed("use")
        reload = self._pressed("reload")
        cycle = self._pressed("cycle_grenade")
        shove = self._pressed("shove")
        melee = (not captured) and (self._down("melee")
                                    or self._pressed("melee"))
        flashlight = self._pressed("flashlight")
        return DeviceFrame(move=move, aim=aim, trigger=trigger, throw=throw,
                           use=use, reload=reload, cycle_grenade=cycle,
                           shove=shove, melee=melee,
                           flashlight_toggle=flashlight)


class DirectControl(ControlSource):
    """Possess the first living marine of ``sim.player_team`` and drive it
    with whichever device (pad, keyboard + mouse) was touched last."""

    def __init__(self, keys: KeyTable | None = None):
        from config import CFG
        from renderer.follow_camera import camera_mode
        from simulation.orders import ORDER_MOVE_ATTACK
        from simulation.weapons import get_tables as weapon_tables
        self.keys = key_table(CFG) if keys is None else keys
        #: ``[display] camera_mode`` (the follow camera's door): how main.py
        #: frames the possessed marine.
        self.camera_mode = camera_mode(CFG)
        # The possessed unit id — bound lazily on the first frame a unit
        # exists (the editor's first SPAWN T / first team-0 spawn, §8).
        self.possessed_id = None
        # UI-state attributes main.py reads off every ControlSource for the
        # HUD/panel (see control_source.py). Direct control has no planning
        # phase and one fixed "mode".
        self.current_mode = ORDER_MOVE_ATTACK
        self.planning_phase = 0
        # Grenade fuse comes off the hand_grenade weapon row (same source the
        # WEGO input uses), read once at construction and converted ONCE to
        # the integer ticks the throw verb takes (G3p).
        from simulation.weapons import fuse_ticks_from_seconds
        self.grenade_fuse = fuse_ticks_from_seconds(
            weapon_tables().weapons.by_name["hand_grenade"]
            .fuse_default_seconds, CFG.clock.ticks_per_second)
        #: The SELECTED grenade kind (G3p): an index of the hand-grenade rows
        #: (``WeaponsTables.hand_grenade_rows``), None until a unit carrying
        #: one is possessed. Control-layer state, never on the sim.
        self.grenade_kind = None
        #: The flashlight's OFF state the control holds (G7p): control-layer
        #: state seeded from the possessed unit's own bit (so a unit darkened
        #: by anything else is not lit by the next frame) and sent every
        #: frame through ``set_flashlight``. The toggle flips it.
        self.flashlight_off = False
        self._flash_unit_id = None
        self._kbm = KeyboardMouseHalf(self.keys)
        self._pad = None          # built on the first frame (pyray import)
        #: The device in use: "keyboard" until the pad is touched.
        self.active = "keyboard"

    # ControlSource sim-construction hooks (§3a/§3b): direct => continuous,
    # unpaused.
    def initial_ruleset(self):
        from simulation.ruleset import ContinuousRealtime
        return ContinuousRealtime()

    def starts_paused(self) -> bool:
        return False

    @property
    def owns_camera_keys(self) -> bool:
        """The movement keys are this scheme's: main.py does not pan the view
        with WASD / arrows, nor zoom with Q / E (the wheel still zooms)."""
        return True

    @property
    def selected_unit_id(self):
        """main.py reads this for the panel/selection highlight — the possessed
        unit IS the selection under direct control (and what the camera
        follows)."""
        return self.possessed_id

    # ------------------------------------------------------------------
    def _bind_possessed(self, sim) -> None:
        """Lazily pick the possessed unit: the first living unit of
        ``sim.player_team`` (default per §3b — the editor's first SPAWN T).
        Re-binds if the current possessed unit is gone (dead/removed) or no
        longer the player's (turned zombie: the sim ignores a zombie's
        intents, so the camera and the keys move on to a living marine)."""
        cur = sim.get_unit(self.possessed_id) if self.possessed_id is not None \
            else None
        if (cur is not None and cur.alive and not cur.is_zombie
                and cur.team == sim.player_team):
            return
        for u in sim.units:
            if u.team == sim.player_team and u.alive:
                self.possessed_id = u.id
                return
        self.possessed_id = None

    def handle_frame(self, sim, renderer) -> None:
        """Read both devices, pick the one touched last, and emit its frame's
        intents for the possessed unit."""
        import pyray as rl
        if self._pad is None:
            from control_gamepad import GamepadHalf
            self._pad = GamepadHalf()
        self._bind_possessed(sim)
        uid = self.possessed_id
        if uid is None:
            return
        u = sim.get_unit(uid)

        # DEBUG (human-test): keyboard N cycles the possessed unit's weapon
        # through the armory — the same `sim.debug_cycle_weapon` the WEGO input
        # uses on N (input_handler.py), so Erik can feel each archetype
        # (projectile / hitscan / spray) under direct control.
        if rl.is_key_pressed(rl.KeyboardKey.KEY_N):
            sim.debug_cycle_weapon(uid)

        if self._kbm.touched():
            self.active = "keyboard"
        elif self._pad.touched():
            self.active = "pad"
        self._sync_grenade_kind(u)
        self._sync_flashlight(u)
        facing = lambda: self._facing_dir(u)          # noqa: E731
        if self.active == "pad":
            if not self._pad.available():
                self.active = "keyboard"
                return
            frame = self._pad.read(facing)
        else:
            frame = self._kbm.read(u, renderer, facing)
        self._emit(sim, uid, frame)

    def _emit(self, sim, uid, frame: DeviceFrame) -> None:
        """THE one place a device's frame becomes intents."""
        if frame.move is None:
            sim.clear_move_dir(uid)
        else:
            sim.set_move_dir(uid, *frame.move)
        if frame.aim is not None:
            sim.set_aim(uid, *frame.aim)
        sim.set_trigger(uid, frame.trigger)
        if frame.cycle_grenade:
            u = sim.get_unit(uid)
            if u is not None:
                k = next_carried_kind(u.grenade_counts, self.grenade_kind)
                if k is not None:
                    self.grenade_kind = k
        if frame.throw is not None and self.grenade_kind is not None:
            tdx, tdy, reach_q = frame.throw
            sim.throw_grenade_intent(uid, tdx, tdy, self.grenade_fuse,
                                     reach_q=reach_q, kind=self.grenade_kind)
        if frame.use:
            sim.use_intent(uid)
        if frame.reload:
            sim.reload_intent(uid)
        if frame.shove:
            sim.shove_intent(uid)
        if frame.melee:
            sim.melee_intent(uid)
        if frame.flashlight_toggle:
            self.flashlight_off = not self.flashlight_off
        sim.set_flashlight(uid, self.flashlight_off)

    def _sync_flashlight(self, u) -> None:
        """Seed the held flashlight state from a newly possessed unit's own
        bit (the control follows the unit, not the other way round)."""
        if u is None or u.id == self._flash_unit_id:
            return
        self._flash_unit_id = u.id
        self.flashlight_off = not getattr(u, "flashlight_on", True)

    def _sync_grenade_kind(self, u) -> None:
        """Keep the selection on a kind the unit still carries: when the
        selected kind has run out (or none is selected yet) it moves to the
        next carried kind; a unit carrying none leaves it where it is."""
        if u is None:
            return
        counts = u.grenade_counts
        sel = self.grenade_kind
        if sel is None or not 0 <= sel < len(counts) or counts[sel] <= 0:
            k = next_carried_kind(counts, sel)
            if k is not None:
                self.grenade_kind = k

    @property
    def selected_grenade_kind(self):
        """The selected grenade kind for the HUD (main.py hands it to the
        panel); None when nothing is selected."""
        return self.grenade_kind

    @staticmethod
    def _facing_dir(u):
        """Q16.16 unit vector along ``u``'s current facing — the THROW
        fallback when no aim is named. Uses the deterministic integer trig
        kit (no libm on the sim path)."""
        from simulation import unit_fixed
        from simulation.intents import FP_ONE
        if u is None:
            return (FP_ONE, 0)   # +X default
        # facing is math-style (Y-up); world Y is down -> negate the y term.
        dx = unit_fixed.cos_rad(u.facing)
        dy = -unit_fixed.sin_rad(u.facing)
        return (unit_fixed.quantize_scalar(dx), unit_fixed.quantize_scalar(dy))


__all__ = ["DirectControl", "DeviceFrame", "next_carried_kind", "KeyboardMouseHalf", "KeyTable",
           "Binding", "KEY_ACTIONS", "INERT_ACTIONS", "KEY_NAMES",
           "key_table", "keyboard_move", "MOUSE_DEADZONE_TILES"]
