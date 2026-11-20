"""G1p (gameplay pack v1, #183) -- THE direct-control scheme (``control_direct.py``).

Pure parts, no window: the ``[controls.keys]`` refusing door, the key-name
vocabulary against pyray's enums, the keyboard's move, the ``[display]
camera_mode`` door and the locked follow camera, the launch-flag alias, and
the one frame -> intents path every device half goes through. The devices
themselves are Erik's HUMAN-TEST; the window path is
``tests/test_control_direct_e2e.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG, Namespace  # noqa: E402
import control_direct as cd  # noqa: E402
from control_direct import (DeviceFrame, DirectControl, KEY_ACTIONS,  # noqa: E402
                            INERT_ACTIONS, key_table, keyboard_move)
from renderer.camera import Camera2D  # noqa: E402
from renderer.follow_camera import (CAMERA_MODES, FollowCamera,  # noqa: E402
                                    camera_mode, follow_settings)
from simulation.intents import FP_ONE  # noqa: E402

SHIPPED = dict(move_forward="R", move_back="F", strafe_left="D",
               strafe_right="G", trigger="MOUSE_LEFT", melee="MOUSE_RIGHT",
               shove="SPACE", reload="Z", throw="W", cycle_grenade="4",
               flashlight="Q", use="E", full_speed="A",
               # #193 (Erik blessed the defaults, 2026-11-11)
               plant_charge="X", cycle_charge_timer="5")


def _cfg(keys):
    return Namespace({"controls": {"keys": dict(keys)}})


# ---------------------------------------------------------------------------
# (a) the key table door
# ---------------------------------------------------------------------------
def test_the_shipped_table_is_erik_s_ruling_and_binds_every_action():
    """Property: config.toml's [controls.keys] passes the door, binds every
    action the scheme reads, and is the table Erik ruled (2026-11-10).
    Breaks if the shipped table loses an action or drifts from the ruling
    without a new one."""
    t = key_table(CFG)
    assert set(t.bindings) == set(KEY_ACTIONS)
    assert {a: b.name for a, b in t.bindings.items()} == SHIPPED
    assert set(INERT_ACTIONS) <= set(KEY_ACTIONS)


@pytest.mark.parametrize("action", ["move_forward", "trigger", "full_speed"])
def test_the_door_refuses_a_missing_action_by_name(action):
    """Property: a table without an action is refused naming that action.
    Breaks if a missing binding falls back to a default key silently."""
    keys = dict(SHIPPED)
    del keys[action]
    with pytest.raises(ValueError, match=f"missing '{action}'"):
        key_table(_cfg(keys))


@pytest.mark.parametrize("bad", ["KEY_R", "R2", "mouse_4", "", 7])
def test_the_door_refuses_an_unknown_key_by_name(bad):
    """Property: a key name outside the vocabulary is refused naming the
    action and the key. Breaks if a typo binds nothing (a dead key)."""
    keys = dict(SHIPPED, reload=bad)
    with pytest.raises(ValueError, match=r"reload = .* is not a known key"):
        key_table(_cfg(keys))


def test_the_door_refuses_an_unknown_action_and_a_doubled_key():
    """Property: an action the scheme does not know is refused by name, and
    so is one key bound to two actions. Breaks if a misspelt action is
    ignored (its real action left on the default) or a key fires two verbs."""
    with pytest.raises(ValueError, match="unknown action.*'crouch'"):
        key_table(_cfg(dict(SHIPPED, crouch="C")))
    with pytest.raises(ValueError, match="'R' is bound to both"):
        key_table(_cfg(dict(SHIPPED, use="R")))
    with pytest.raises(ValueError, match=r"\[controls.keys\]"):
        key_table(Namespace({}))


def test_the_key_names_are_raylib_s_codes():
    """Property: every name of the door's vocabulary maps to pyray's own enum
    value (KeyboardKey.KEY_<name>, MouseButton.MOUSE_BUTTON_<name>). Breaks
    if a hand-written code drifts from raylib (a bound key reads another)."""
    import pyray as rl
    digits = dict(zip("0123456789", ["ZERO", "ONE", "TWO", "THREE", "FOUR",
                                     "FIVE", "SIX", "SEVEN", "EIGHT",
                                     "NINE"]))
    for name, (device, code) in cd.KEY_NAMES.items():
        if device == "mouse":
            enum = getattr(rl.MouseButton,
                           "MOUSE_BUTTON_" + name.split("_", 1)[1])
        else:
            enum = getattr(rl.KeyboardKey, "KEY_" + digits.get(name, name))
        assert int(enum) == code, name


# ---------------------------------------------------------------------------
# (b) the keyboard's move
# ---------------------------------------------------------------------------
def test_the_keyboard_move_is_screen_relative_with_a_two_state_throttle():
    """Property: forward is screen-up (-y, the north-up camera), back +y,
    strafes -x / +x; a diagonal is a unit vector; opposite keys cancel;
    the throttle is 0 (walk) or ONE (full speed held). Breaks if a key moves
    the wrong way, a diagonal is faster, or the keyboard sends a speed-mode
    id instead of a throttle."""
    assert keyboard_move(True, False, False, False, False) == (0, -FP_ONE, 0)
    assert keyboard_move(False, True, False, False, True) == (0, FP_ONE,
                                                              FP_ONE)
    assert keyboard_move(False, False, True, False, False) == (-FP_ONE, 0, 0)
    assert keyboard_move(False, False, False, True, False) == (FP_ONE, 0, 0)
    dx, dy, t = keyboard_move(True, False, False, True, False)
    assert dx == -dy == 46341 and t == 0
    assert keyboard_move(True, True, False, False, True) is None
    assert keyboard_move(False, False, False, False, True) is None


# ---------------------------------------------------------------------------
# (c) the camera mode
# ---------------------------------------------------------------------------
def test_the_camera_mode_door_refuses_an_unbuilt_mode():
    """Property: [display] camera_mode is one of CAMERA_MODES ("centred"
    shipped); "player_up" (the G1p-b spike) and a missing key are refused by
    name. Breaks if an unbuilt framing silently falls back."""
    assert camera_mode(CFG) in CAMERA_MODES
    for bad in ("player_up", "Centred", 3):
        with pytest.raises(ValueError, match="camera_mode"):
            camera_mode(Namespace({"display": {"camera_mode": bad}}))
    with pytest.raises(ValueError, match="camera_mode"):
        camera_mode(Namespace({"display": {}}))


def test_the_centred_camera_holds_the_possessed_unit_in_the_middle():
    """Property: a follow camera LOCKED on a unit puts the view centre on
    the unit's centre every frame (no easing lag), unless the world clamp
    stops it at an edge; it never zooms. Breaks if the lock eases like the
    match camera or drifts off the unit."""
    cam = Camera2D(pos_tile_x=0.0, pos_tile_y=0.0, zoom_px_per_tile=20.0,
                   viewport_px_w=400, viewport_px_h=300,
                   world_size_tile_w=200, world_size_tile_h=200)
    u = SimpleNamespace(id=5, x=80.0, y=60.0, footprint=3, alive=True,
                        team=0, slot=0, name="M")
    fc = FollowCamera(follow_settings(CFG))
    for x in (80.0, 81.25, 95.5):
        u.x = x
        fc.lock(u.id)
        fc.apply(cam, [u], 1 / 60)
        w, h = cam.visible_tiles()
        assert (cam.pos_tile_x + 0.5 * w, cam.pos_tile_y + 0.5 * h) == \
            (x + 1.5, 61.5)
        assert cam.zoom_px_per_tile == 20.0
    u.x = 1.0                                 # at the world edge: clamped
    fc.lock(u.id)
    fc.apply(cam, [u], 1 / 60)
    assert cam.pos_tile_x == 0.0


# ---------------------------------------------------------------------------
# (d) the scheme: the alias, possession, the one emit path
# ---------------------------------------------------------------------------
def test_gamepad_is_an_alias_of_direct():
    """Property: --control gamepad and --control direct build the same one
    scheme. Breaks if a second direct-control class comes back."""
    import control_source
    a = control_source.create_control_source("direct")
    b = control_source.create_control_source("gamepad")
    assert type(a) is type(b) is DirectControl
    assert a.owns_camera_keys and a.camera_mode in CAMERA_MODES
    assert not a.starts_paused()


class _FakeSim:
    """Records every facade call in order."""

    def __init__(self, units, player_team=0):
        self.units = units
        self.player_team = player_team
        self.calls = []

    def get_unit(self, uid):
        return next((u for u in self.units if u.id == uid), None)

    def __getattr__(self, name):
        if name in ("set_move_dir", "clear_move_dir", "set_aim", "set_trigger",
                    "throw_grenade_intent", "use_intent", "melee_intent",
                    "set_flashlight", "shove_intent", "reload_intent"):
            return lambda *a, **k: self.calls.append((name, a, k))
        raise AttributeError(name)


def _unit(uid, team=0, alive=True, zombie=False):
    return SimpleNamespace(id=uid, team=team, alive=alive, is_zombie=zombie)


def test_a_frame_becomes_intents_through_the_one_emit_path():
    """Property: a device frame is issued as exactly the facade calls it
    names -- a move with its throttle (or a clear), the aim only when named,
    the trigger always, a throw toward its point with its reach and the
    hand grenade's fuse, a use on its edge. Breaks if a device half reaches
    the facade by another path or an unnamed aim overwrites the facing."""
    dc = DirectControl(keys=key_table(CFG))
    dc.grenade_kind = 1                     # the selected kind (G3p)
    sim = _FakeSim([_unit(1)])
    dc._emit(sim, 1, DeviceFrame(move=(FP_ONE, 0, FP_ONE), aim=(0, -FP_ONE),
                                 trigger=True, throw=(0, -FP_ONE, 5 * FP_ONE),
                                 use=True))
    assert sim.calls == [
        ("set_move_dir", (1, FP_ONE, 0, FP_ONE), {}),
        ("set_aim", (1, 0, -FP_ONE), {}),
        ("set_trigger", (1, True), {}),
        ("throw_grenade_intent", (1, 0, -FP_ONE, dc.grenade_fuse),
         {"reach_q": 5 * FP_ONE, "kind": 1}),
        ("use_intent", (1,), {}),
        ("set_flashlight", (1, False), {}),   # G7p: the held light, every frame
    ]
    sim.calls.clear()
    dc._emit(sim, 1, DeviceFrame(move=None, aim=None, trigger=False,
                                 throw=None, use=False))
    assert sim.calls == [("clear_move_dir", (1,), {}),
                         ("set_trigger", (1, False), {}),
                         ("set_flashlight", (1, False), {})]


def test_possession_moves_on_from_a_dead_or_turned_marine():
    """Property: the scheme possesses the first living marine of the
    player's team, and moves on when it dies or turns zombie (a zombie's
    intents are ignored by the sim, so the keys and the camera would hold a
    puppet). Breaks if a converted marine stays possessed."""
    dc = DirectControl(keys=key_table(CFG))
    a, b = _unit(1), _unit(2)
    sim = _FakeSim([a, b])
    dc._bind_possessed(sim)
    assert dc.possessed_id == 1
    a.is_zombie, a.team = True, 255
    dc._bind_possessed(sim)
    assert dc.possessed_id == 2
    b.alive = False
    dc._bind_possessed(sim)
    assert dc.possessed_id is None
