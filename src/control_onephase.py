"""``OnePhaseWEGOInput`` — the game-mode control scheme (design §16/§17).

§17: "**In game mode, the control scheme owns every binding.** The
OnePhaseWEGO keymap is designed from scratch for play only." Erik designed it
at kickoff (ruling 1 — hotbar + RMB-move):

======================  =====================================================
MOUSE
----------------------  -----------------------------------------------------
LMB                     select a marine; with a slot armed, apply that action
                        at the cursor
RMB                     MOVE here — the primary order, no mode needed
Shift+RMB               queue a waypoint (append, don't replace)
Wheel                   zoom
======================  =====================================================

======================  =====================================================
KEYS
----------------------  -----------------------------------------------------
1..0                    hotbar slots (arm the bound registry row)
Q                       weapon swap (primary <-> secondary)
X                       mark target
Space                   SUBMIT — execute the round
Backspace               undo last order
Esc                     cancel the armed action / clear the selection
Tab                     cycle selected marine
I                       inventory (the DS3 menu)
L                       flashlight variant toggle (§20 item 3)
WASD / arrows           pan
======================  =====================================================

Everything diagnostic lives behind ``--debug`` (Erik's ruling 2) — see
:mod:`debug_keys`. Without that flag this scheme polls no diagnostic key at
all, which is the whole point of §17.

The seam (§16): this class WRITES only through ``sim.apply_action`` /
``sim.undo_last_order``, and READS sim state + the action registry for its own
UI state. It never mutates a unit or the world directly. Its UI state
(selection, armed slot, hotbar bindings, menu page, flashlight mode) is
consumed by ``main.py`` and :mod:`ui.draw`; the simulation is never told about
any of it.
"""
from __future__ import annotations

import pyray as rl

from config import CFG
from control_source import ControlSource
from debug_keys import DebugKeyState, handle_debug_keys
from simulation.orders import (
    ORDER_AMBUSH, ORDER_DETONATE, ORDER_EXPLOSIVE, ORDER_GRENADE, ORDER_HOLD,
    ORDER_MARK, ORDER_MOVE, ORDER_MOVE_SHOOT, ORDER_OVERWATCH, ORDER_SHOOT,
    ORDER_SWAP, Order,
)
from simulation.ruleset import OnePhaseWEGO
import ui


class OnePhaseWEGOInput(ControlSource):
    """Keyboard + mouse planning input for :class:`OnePhaseWEGO`."""

    def __init__(self, debug: bool = False):
        self.debug = bool(debug)
        self.selected_unit_id = None
        #: The hotbar row the next LMB will apply, or None (RMB is always Move,
        #: so the player spends most of the round with nothing armed).
        self.armed_action = None
        self.bindings = list(ui.DEFAULT_HOTBAR)
        self.menu_open = False
        self.menu_page = 0
        #: "team" | "selected" — §20 item 3 says build both and let feel decide.
        self.flashlight_mode = "team"
        self._debug_state = DebugKeyState()
        # Wheel-set schedule dials, in SECONDS from the start of the current
        # round (§5 Hold-until-t, §12 "detonate anywhere in the round, 0-4 s").
        # Erik's design note: a hold only ever needs a number in [0, round], so
        # the wheel is the whole interface — no timeline scrubber required.
        # Both may run one round-length PAST the end, which is how "at the top
        # of the next round" is expressible: the clock is monotonic, so that is
        # just a larger number.
        self.hold_seconds = float(CFG.clock.round_duration_seconds)
        self.det_seconds = float(CFG.clock.round_duration_seconds)
        #: Wall-clock seconds spent in the current planning pause, for the
        #: multiplayer submit timer (§16). Reset on every resume.
        self.planning_elapsed = 0.0
        self._was_paused = True

    # -- ControlSource construction hooks ------------------------------
    def initial_ruleset(self):
        return OnePhaseWEGO()

    def starts_paused(self) -> bool:
        return True

    @property
    def wants_renderer_toggles(self) -> bool:
        """§17: the renderer's diagnostic keys are game-mode keys too, and are
        evicted with the rest unless ``--debug`` is on."""
        return self.debug

    # -- HUD state main.py reads ---------------------------------------
    @property
    def planning_phase(self) -> int:
        """Phases are gone (§2). Reported as 0 so the shipped panel/renderer
        signature keeps working without learning about rulesets."""
        return 0

    @property
    def current_mode(self) -> int:
        """The armed action, expressed as the legacy order-type int the panel
        renders. Move when nothing is armed — which is the truth, since RMB is
        always Move."""
        if not self.armed_action:
            return ORDER_MOVE
        return self._table_order_type(self.armed_action)

    def _table_order_type(self, name):
        return ORDER_MOVE if name is None else _ORDER_BY_ACTION.get(name,
                                                                    ORDER_MOVE)

    # ------------------------------------------------------------------
    # Per-frame poll
    # ------------------------------------------------------------------
    def handle_frame(self, sim, renderer) -> None:
        K = rl.KeyboardKey
        paused = sim.is_paused()

        # Track how long this planning pause has lasted, for the §16 clock.
        if paused and self._was_paused:
            self.planning_elapsed += rl.get_frame_time()
        elif paused and not self._was_paused:
            self.planning_elapsed = 0.0
        self._was_paused = paused

        if self.debug:
            handle_debug_keys(sim, renderer, self._debug_state,
                              self.selected_unit_id)

        # ---- the DS3 menu owns input while it is open (§15) ----
        if rl.is_key_pressed(K.KEY_I):
            self.menu_open = not self.menu_open
        if self.menu_open:
            if rl.is_key_pressed(K.KEY_RIGHT) or rl.is_key_pressed(K.KEY_E):
                self.menu_page = (self.menu_page + 1) % len(ui.DS3_PAGES)
            if rl.is_key_pressed(K.KEY_LEFT) or rl.is_key_pressed(K.KEY_Q):
                self.menu_page = (self.menu_page - 1) % len(ui.DS3_PAGES)
            if rl.is_key_pressed(K.KEY_ESCAPE):
                self.menu_open = False
            return

        # ---- global verbs ----
        if rl.is_key_pressed(K.KEY_SPACE):
            # SUBMIT: the planning pause ends and the round runs. There is no
            # grenade pre-spawn to do — under this ruleset a throw is an
            # ordinary scheduled action (§12), not something materialized up
            # front from a planned end position.
            sim.set_paused(not paused)

        if rl.is_key_pressed(K.KEY_TAB):
            self._cycle_selection(sim)

        if rl.is_key_pressed(K.KEY_L):
            self.flashlight_mode = ("selected" if self.flashlight_mode == "team"
                                    else "team")

        if rl.is_key_pressed(K.KEY_ESCAPE):
            if self.armed_action:
                self.armed_action = None
            else:
                self.selected_unit_id = None

        if rl.is_key_pressed(K.KEY_BACKSPACE) and self.selected_unit_id is not None:
            sim.undo_last_order(self.selected_unit_id)

        # ---- hotbar (1..0) ----
        for i, key in enumerate((K.KEY_ONE, K.KEY_TWO, K.KEY_THREE, K.KEY_FOUR,
                                 K.KEY_FIVE, K.KEY_SIX, K.KEY_SEVEN,
                                 K.KEY_EIGHT, K.KEY_NINE, K.KEY_ZERO)):
            if rl.is_key_pressed(key):
                self._arm_slot(sim, i)

        # ---- direct verbs ----
        if rl.is_key_pressed(K.KEY_Q) and self.selected_unit_id is not None:
            self._issue(sim, Order(ORDER_SWAP, 0, 0, 0,
                                   action_name="swap_weapon"), replace=False)
        if rl.is_key_pressed(K.KEY_X):
            self.armed_action = "mark"

        # ---- the wheel sets the SCHEDULE dial for a time-carrying action ----
        # Only while such an action is armed, so the wheel keeps meaning zoom
        # the rest of the time (the shipped binding, and what a player expects
        # by default).
        dial = self._armed_dial(sim)
        if dial is not None:
            wheel = rl.get_mouse_wheel_move()
            if wheel:
                self._nudge_dial(dial, 0.25 if wheel > 0 else -0.25)

        # ---- mouse ----
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
            self._left_click(sim, renderer)
        elif rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_RIGHT):
            shift = (rl.is_key_down(K.KEY_LEFT_SHIFT) or
                     rl.is_key_down(K.KEY_RIGHT_SHIFT))
            self._right_click(sim, renderer, append=shift)

    # ------------------------------------------------------------------
    # Selection + arming
    # ------------------------------------------------------------------
    def _cycle_selection(self, sim) -> None:
        marines = [u for u in sim.units if u.team == 0 and u.alive]
        if not marines:
            self.selected_unit_id = None
            return
        ids = [u.id for u in marines]
        if self.selected_unit_id in ids:
            nxt = ids[(ids.index(self.selected_unit_id) + 1) % len(ids)]
        else:
            nxt = ids[0]
        self.selected_unit_id = nxt
        self.armed_action = None

    def _arm_slot(self, sim, index: int) -> None:
        """Arm a hotbar slot. A slot whose action needs no target fires
        immediately — making the player click an empty tile to swap weapons
        would be a mode for nothing."""
        bindings = self._live_bindings(sim)
        if not (0 <= index < len(bindings)):
            return
        name = bindings[index]
        if not name or name not in sim.actions_table.by_name:
            return
        action = sim.actions_table.get(name)
        if action.targeting == "none":
            if self.selected_unit_id is not None:
                self._issue(sim, Order(action.order_type, 0, 0, 0,
                                       action_name=name), replace=False)
            return
        self.armed_action = name

    # ------------------------------------------------------------------
    # Wheel-set schedule dials (§5 Hold-until-t, §12 detonation time)
    # ------------------------------------------------------------------
    def armed_dial_seconds(self, sim):
        """The wheel-set moment for the armed action, or ``None`` if it does
        not carry one. Read by the HUD so the dial is visible while turning."""
        dial = self._armed_dial(sim)
        return None if dial is None else float(getattr(self, dial))

    def _armed_dial(self, sim):
        """Which schedule dial the wheel is currently adjusting, or None."""
        if not self.armed_action:
            return None
        action = sim.actions_table.by_name.get(self.armed_action)
        if action is None:
            return None
        if action.order_type == ORDER_HOLD:
            return "hold_seconds"
        if action.order_type in (ORDER_EXPLOSIVE, ORDER_DETONATE):
            return "det_seconds"
        return None

    def _nudge_dial(self, dial: str, delta: float) -> None:
        """Clamp to ``[0, 2 x round]`` — 0 s means "immediately", one round
        length is the end of THIS round, and beyond that reaches into the next
        one (which the monotonic clock makes an ordinary number, §12)."""
        span = 2.0 * float(CFG.clock.round_duration_seconds)
        setattr(self, dial, max(0.0, min(span, getattr(self, dial) + delta)))

    def _tick_for(self, sim, seconds: float) -> int:
        """Seconds-into-this-round -> the absolute tick the sim schedules on."""
        return sim.round_start_tick() + int(
            round(float(seconds) * CFG.clock.ticks_per_second))

    def _live_bindings(self, sim) -> list:
        """Bindings for the selected unit — the belt fills the free tail slots
        (§15), so a marine's own items are on the bar."""
        unit = sim.get_unit(self.selected_unit_id) \
            if self.selected_unit_id is not None else None
        base = ui.default_bindings(unit)
        return [self.bindings[i] if self.bindings[i] != ui.DEFAULT_HOTBAR[i]
                else base[i] for i in range(len(base))]

    # ------------------------------------------------------------------
    # Clicks
    # ------------------------------------------------------------------
    def _left_click(self, sim, renderer) -> None:
        tile = renderer.mouse_to_tile()
        if tile is None:
            return
        fx, fy = tile
        # Selecting a marine always wins over applying an action to the tile
        # it is standing on — misclicking your own squadmate should not spend
        # a grenade.
        for u in sim.units:
            if (u.alive and u.team == 0
                    and u.tile_x <= fx < u.tile_x + u.footprint
                    and u.tile_y <= fy < u.tile_y + u.footprint):
                self.selected_unit_id = u.id
                self.armed_action = None
                return
        if self.selected_unit_id is None or not self.armed_action:
            return
        self._apply_armed(sim, fx, fy)

    def _right_click(self, sim, renderer, append: bool) -> None:
        """RMB is always Move (§ Erik's ruling 1) — the primary order needs no
        mode. Shift appends a waypoint instead of replacing the plan."""
        if self.selected_unit_id is None:
            return
        tile = renderer.mouse_to_tile()
        if tile is None:
            return
        u = sim.get_unit(self.selected_unit_id)
        if u is None:
            return
        tx, ty = _footprint_anchor(sim, u, tile)
        self._issue(sim, Order(ORDER_MOVE, tx, ty, 0, action_name="move"),
                    replace=not append)

    def _apply_armed(self, sim, fx, fy) -> None:
        name = self.armed_action
        action = sim.actions_table.get(name)
        u = sim.get_unit(self.selected_unit_id)
        if u is None:
            return

        target_unit_id = None
        if action.targeting == "unit":
            enemy = _enemy_at(sim, fx, fy)
            if enemy is None:
                return              # a unit-targeted action needs a unit
            target_unit_id = enemy.id

        tx, ty = fx, fy
        if action.order_type in (ORDER_MOVE, ORDER_MOVE_SHOOT):
            tx, ty = _footprint_anchor(sim, u, (fx, fy))

        order = Order(action.order_type, tx, ty, 0, action_name=name,
                      target_unit_id=target_unit_id)

        if action.order_type == ORDER_HOLD:
            # Hold UNTIL t (§5), where t is the wheel-set moment inside this
            # round — Erik's point: the scrubber was never needed for this,
            # because a hold only ever needs "a number in [0, round]".
            order.start_tick = self._tick_for(sim, self.hold_seconds)
        elif action.order_type in (ORDER_EXPLOSIVE, ORDER_DETONATE):
            # §12: a detonation is a MOMENT, "schedulable anywhere in the
            # round, 0-4 s". The wheel sets it; the default is the top of the
            # NEXT round — the breach opening the design names (door blows at
            # 0.0, grenades and fire follow), which is why the dial runs one
            # round-length past the end.
            order.det_tick = self._tick_for(sim, self.det_seconds)
        elif action.order_type == ORDER_GRENADE:
            order.grenade_fuse = 2.0
        elif action.order_type == ORDER_OVERWATCH:
            order.cone_half_deg = None      # the dialled default (§9)

        if self._issue(sim, order, replace=False):
            # Marking is a command, not a commitment — stay armed so a player
            # can mark three targets in a row without re-pressing X.
            if action.order_type != ORDER_MARK:
                self.armed_action = None

    def _issue(self, sim, order, replace: bool) -> bool:
        """Place an order. ``replace`` starts a fresh plan for the unit — which
        is what a plain (unshifted) click means (§13: new orders interrupt by
        default; shift-click builds a string instead)."""
        if self.selected_unit_id is None:
            return False
        if replace:
            sim.begin_new_plan(self.selected_unit_id)
        return bool(sim.apply_action(self.selected_unit_id, order))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _footprint_anchor(sim, unit, tile):
    """Snap a click so the unit's CENTRE lands under the cursor, clamped to the
    map — the shipped move-order convention."""
    fx, fy = tile
    fp = unit.footprint
    h, w = sim.gmap.material.shape
    tx = max(0, min(w - fp, fx - fp // 2))
    ty = max(0, min(h - fp, fy - fp // 2))
    return tx, ty


def _enemy_at(sim, fx, fy):
    for u in sim.units:
        if (u.alive and u.team != 0
                and u.tile_x <= fx < u.tile_x + u.footprint
                and u.tile_y <= fy < u.tile_y + u.footprint):
            return u
    return None


#: action name -> legacy order-type int, for the panel's ``current_mode``.
_ORDER_BY_ACTION = {
    "move": ORDER_MOVE, "shoot": ORDER_SHOOT, "move_shoot": ORDER_MOVE_SHOOT,
    "overwatch": ORDER_OVERWATCH, "ambush": ORDER_AMBUSH, "hold": ORDER_HOLD,
    "swap_weapon": ORDER_SWAP, "mark": ORDER_MARK,
    "plant_charge": ORDER_EXPLOSIVE, "detonate": ORDER_DETONATE,
    "use_hand_grenade": ORDER_GRENADE, "use_breach_charge": ORDER_EXPLOSIVE,
}


__all__ = ["OnePhaseWEGOInput"]
