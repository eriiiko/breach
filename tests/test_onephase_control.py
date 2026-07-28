"""P9 — the OnePhaseWEGO control scheme and the §17 keymap eviction.

``control_onephase`` imports pyray, so these tests drive it through a fake
raylib backend: the module's ``rl`` binding is swapped for a stub that reports
exactly the keys and clicks a test presses. That keeps the KEYMAP itself —
which is a design decision Erik made, not an implementation detail — under
test, without a window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("pyray", reason="control layer needs the raylib binding")

from level_loader import LevelData  # noqa: E402
from simulation import orders as O  # noqa: E402
from simulation.ruleset import OnePhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402
import control_onephase  # noqa: E402
from control_onephase import OnePhaseWEGOInput  # noqa: E402


# ---------------------------------------------------------------------------
# A fake raylib: only what the control scheme actually polls.
# ---------------------------------------------------------------------------
class _Keys:
    """Key ids as plain strings, so a test can press "SPACE" by name."""

    def __getattr__(self, name):
        if name.startswith("KEY_"):
            return name[4:]
        raise AttributeError(name)


class _MouseButton:
    MOUSE_BUTTON_LEFT = "LMB"
    MOUSE_BUTTON_RIGHT = "RMB"


class FakeRl:
    KeyboardKey = _Keys()
    MouseButton = _MouseButton

    def __init__(self):
        self.pressed = set()
        self.down = set()
        self.mouse_pressed = set()

    def is_key_pressed(self, key):
        return key in self.pressed

    def is_key_down(self, key):
        return key in self.down or key in self.pressed

    def is_mouse_button_pressed(self, btn):
        return btn in self.mouse_pressed

    def get_frame_time(self):
        return 1.0 / 60.0

    def clear(self):
        self.pressed.clear()
        self.down.clear()
        self.mouse_pressed.clear()


class FakeRenderer:
    def __init__(self, tile=(0, 0)):
        self.tile = tile

    def mouse_to_tile(self):
        return self.tile


@pytest.fixture
def fake_rl(monkeypatch):
    fake = FakeRl()
    monkeypatch.setattr(control_onephase, "rl", fake)
    return fake


def _level(h=40, w=40):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    return LevelData(name="onephase_control", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim():
    return Simulation(_level(), seed=17, breach_physics=None,
                      enable_recorder=False, ruleset=OnePhaseWEGO())


def _marine(sim, x=8.0, y=8.0, name="m"):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _zombie(sim, x=20.0, y=8.0, name="z"):
    u = Unit(name, x=x, y=y, team=1)
    sim.add_unit(u)
    return u


def _frame(inp, sim, fake, renderer=None, keys=(), down=(), mouse=None):
    fake.clear()
    fake.pressed.update(keys)
    fake.down.update(down)
    if mouse:
        fake.mouse_pressed.add(mouse)
    inp.handle_frame(sim, renderer or FakeRenderer())


# ---------------------------------------------------------------------------
# Construction: the two halves of the loadable game chosen together
# ---------------------------------------------------------------------------
def test_the_scheme_picks_its_ruleset_and_starts_paused():
    inp = OnePhaseWEGOInput()
    assert isinstance(inp.initial_ruleset(), OnePhaseWEGO)
    assert inp.starts_paused() is True


def test_the_factory_knows_the_name():
    from control_source import create_control_source
    assert isinstance(create_control_source("onephase"), OnePhaseWEGOInput)
    assert create_control_source("onephase", debug=True).debug is True


def test_phases_are_gone_from_the_hud_contract():
    """§2: no phases. The property survives only because the shipped panel
    signature takes it."""
    assert OnePhaseWEGOInput().planning_phase == 0


# ---------------------------------------------------------------------------
# §17 — the diagnostic eviction (Erik's ruling 2)
# ---------------------------------------------------------------------------
def test_game_mode_polls_no_diagnostic_key(fake_rl, monkeypatch):
    called = []
    monkeypatch.setattr(control_onephase, "handle_debug_keys",
                        lambda *a, **k: called.append(1))
    inp = OnePhaseWEGOInput(debug=False)
    sim = _sim()
    _marine(sim)
    _frame(inp, sim, fake_rl, keys={"I", "J", "K", "U", "N", "O", "P"})
    assert called == [], "a diagnostic key was polled in game mode"


def test_debug_flag_rearms_them(fake_rl, monkeypatch):
    called = []
    monkeypatch.setattr(control_onephase, "handle_debug_keys",
                        lambda *a, **k: called.append(1))
    inp = OnePhaseWEGOInput(debug=True)
    sim = _sim()
    _frame(inp, sim, fake_rl)
    assert called == [1]


def test_renderer_toggles_follow_the_same_rule():
    assert OnePhaseWEGOInput(debug=False).wants_renderer_toggles is False
    assert OnePhaseWEGOInput(debug=True).wants_renderer_toggles is True


def test_the_shipped_schemes_keep_their_keys():
    from control_source import ControlSource
    assert ControlSource().wants_renderer_toggles is True


# ---------------------------------------------------------------------------
# The keymap (Erik's ruling 1)
# ---------------------------------------------------------------------------
def test_lmb_selects_a_marine(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    _frame(inp, sim, fake_rl, FakeRenderer((9, 9)), mouse="LMB")
    assert inp.selected_unit_id == u.id


def test_rmb_is_always_move_no_mode_needed(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, FakeRenderer((8, 20)), mouse="RMB")
    assert len(u.orders) == 1
    assert u.orders[0].order_type == O.ORDER_MOVE
    assert inp.armed_action is None, "Move must need no armed mode"


def test_a_plain_rmb_replaces_the_plan_and_shift_appends(fake_rl):
    """§13 + §16: new orders interrupt by default; shift-click builds a
    waypoint string instead."""
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, FakeRenderer((8, 20)), mouse="RMB")
    _frame(inp, sim, fake_rl, FakeRenderer((20, 20)), mouse="RMB",
           down={"LEFT_SHIFT"})
    assert len(u.orders) == 2, "shift-RMB should have appended"
    _frame(inp, sim, fake_rl, FakeRenderer((8, 12)), mouse="RMB")
    assert len(u.orders) == 1, "a plain RMB should have replaced the plan"


def test_space_submits_the_round(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    sim.set_paused(True)
    _frame(inp, sim, fake_rl, keys={"SPACE"})
    assert sim.is_paused() is False


def test_tab_cycles_marines(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    a = _marine(sim, 8.0, 8.0, name="a")
    b = _marine(sim, 12.0, 8.0, name="b")
    _frame(inp, sim, fake_rl, keys={"TAB"})
    assert inp.selected_unit_id == a.id
    _frame(inp, sim, fake_rl, keys={"TAB"})
    assert inp.selected_unit_id == b.id
    _frame(inp, sim, fake_rl, keys={"TAB"})
    assert inp.selected_unit_id == a.id, "cycling must wrap"


def test_q_swaps_the_weapon(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"Q"})
    assert any(o.order_type == O.ORDER_SWAP for o in u.orders)


def test_backspace_undoes(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, FakeRenderer((8, 20)), mouse="RMB")
    assert len(u.orders) == 1
    _frame(inp, sim, fake_rl, keys={"BACKSPACE"})
    assert u.orders == []


def test_escape_cancels_the_armed_action_then_the_selection(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim)
    inp.selected_unit_id = u.id
    inp.armed_action = "shoot"
    _frame(inp, sim, fake_rl, keys={"ESCAPE"})
    assert inp.armed_action is None and inp.selected_unit_id == u.id
    _frame(inp, sim, fake_rl, keys={"ESCAPE"})
    assert inp.selected_unit_id is None


def test_l_toggles_the_flashlight_variant(fake_rl):
    """§20 item 3: build both, let feel decide."""
    inp = OnePhaseWEGOInput()
    sim = _sim()
    assert inp.flashlight_mode == "team"
    _frame(inp, sim, fake_rl, keys={"L"})
    assert inp.flashlight_mode == "selected"
    _frame(inp, sim, fake_rl, keys={"L"})
    assert inp.flashlight_mode == "team"


# ---------------------------------------------------------------------------
# The hotbar (§16)
# ---------------------------------------------------------------------------
def test_a_number_key_arms_its_slot(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"TWO"})
    assert inp.armed_action == "shoot"


def test_an_armed_unit_targeted_action_applies_on_lmb(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    z = _zombie(sim, 20.0, 8.0)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"TWO"})
    _frame(inp, sim, fake_rl, FakeRenderer((21, 9)), mouse="LMB")
    assert any(o.order_type == O.ORDER_SHOOT and o.target_unit_id == z.id
               for o in u.orders)
    assert inp.armed_action is None, "the armed slot should disarm after use"


def test_a_unit_targeted_action_needs_a_unit(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"TWO"})
    _frame(inp, sim, fake_rl, FakeRenderer((25, 25)), mouse="LMB")
    assert u.orders == [], "a shoot order landed on empty floor"


def test_a_targetless_action_fires_immediately(fake_rl):
    """Making the player click an empty tile to swap weapons would be a mode
    for nothing."""
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"SEVEN"})    # a targeted item slot...
    assert inp.armed_action is not None
    inp.armed_action = None
    inp.bindings = list(inp.bindings)
    inp.bindings[0] = "swap_weapon"
    _frame(inp, sim, fake_rl, keys={"ONE"})
    assert inp.armed_action is None
    assert any(o.order_type == O.ORDER_SWAP for o in u.orders)


def test_x_arms_marking_and_marking_stays_armed(fake_rl):
    """§11: marking is a command, not a commitment — mark three targets in a
    row without re-pressing X."""
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    a = _zombie(sim, 20.0, 8.0, name="a")
    b = _zombie(sim, 24.0, 8.0, name="b")
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"X"})
    assert inp.armed_action == "mark"
    _frame(inp, sim, fake_rl, FakeRenderer((21, 9)), mouse="LMB")
    assert inp.armed_action == "mark", "marking disarmed itself"
    _frame(inp, sim, fake_rl, FakeRenderer((25, 9)), mouse="LMB")
    marked = {o.target_unit_id for o in u.orders
              if o.order_type == O.ORDER_MARK}
    assert {a.id, b.id} <= marked


def test_clicking_your_own_marine_selects_rather_than_spending_an_item(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0, name="a")
    mate = _marine(sim, 16.0, 8.0, name="b")
    inp.selected_unit_id = u.id
    u.has_grenade = 2
    _frame(inp, sim, fake_rl, keys={"SEVEN"})     # arm the grenade
    _frame(inp, sim, fake_rl, FakeRenderer((17, 9)), mouse="LMB")
    assert inp.selected_unit_id == mate.id
    assert u.has_grenade == 2, "a grenade was spent on a misclick"


# ---------------------------------------------------------------------------
# The DS3 menu (§15)
# ---------------------------------------------------------------------------
def test_i_opens_the_menu_and_it_swallows_input(fake_rl):
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim)
    inp.selected_unit_id = u.id
    _frame(inp, sim, fake_rl, keys={"I"})
    assert inp.menu_open is True
    _frame(inp, sim, fake_rl, FakeRenderer((8, 20)), mouse="RMB")
    assert u.orders == [], "the world took input while the menu was open"


def test_menu_pages_cycle_and_escape_closes(fake_rl):
    import ui
    inp = OnePhaseWEGOInput()
    sim = _sim()
    _frame(inp, sim, fake_rl, keys={"I"})
    _frame(inp, sim, fake_rl, keys={"RIGHT"})
    assert inp.menu_page == 1
    _frame(inp, sim, fake_rl, keys={"LEFT"})
    assert inp.menu_page == 0
    _frame(inp, sim, fake_rl, keys={"LEFT"})
    assert inp.menu_page == len(ui.DS3_PAGES) - 1, "pages must wrap"
    _frame(inp, sim, fake_rl, keys={"ESCAPE"})
    assert inp.menu_open is False


# ---------------------------------------------------------------------------
# Scheduled detonation defaults (§12)
# ---------------------------------------------------------------------------
def test_a_charge_defaults_to_blowing_at_the_top_of_the_next_round(fake_rl):
    """§12's breach opening: door blows at 0.0, grenades and fire follow."""
    inp = OnePhaseWEGOInput()
    sim = _sim()
    u = _marine(sim, 8.0, 8.0)
    u.has_explosive = 1
    inp.selected_unit_id = u.id
    inp.bindings = list(inp.bindings)
    inp.bindings[0] = "plant_charge"
    _frame(inp, sim, fake_rl, keys={"ONE"})
    _frame(inp, sim, fake_rl, FakeRenderer((14, 8)), mouse="LMB")
    charge = [o for o in u.orders if o.order_type == O.ORDER_EXPLOSIVE]
    assert charge
    assert charge[0].det_tick == sim.round_start_tick() + sim.ticks_per_round


# ---------------------------------------------------------------------------
# The seam (§16): the control source writes ONLY through the facade
# ---------------------------------------------------------------------------
def test_the_control_source_never_mutates_units_directly():
    src = Path(control_onephase.__file__).read_text(encoding="utf-8")
    for forbidden in ("u.x =", "u.y =", "unit.x =", ".current_hp",
                      ".plan =", ".orders.append", ".orders ="):
        assert forbidden not in src, (
            f"control_onephase writes {forbidden!r} directly; orders must go "
            f"through the Simulation facade (design §16)")
