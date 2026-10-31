"""``CreatureControl`` -- creatures as a match controller, ``"creature"`` (#106 P6d2).

In a match every unit has ONE driver and it is a controller (the match
controller registry, ``control_source.register_controller``). This one drives
every living creature unit of its team (``CREATURE_TEAM`` by default) through
the per-tick intent facade, once per tick, with no renderer
(``handle_frame(sim, None)``): it never touches a body, a buffer or a
primitive -- the creature's hub turns the intents into its goal at slot 5b
(``creature_primitives``' goal table).

v1 behaviour -- a simple, honest hazard (the brain is #108):

* a living NON-creature unit the creature can SEE (``vision.can_see``, the one
  sight oracle: walls block, the facing cone at any range plus the
  awareness radius around it) -> MOVE_DIR toward the NEAREST such unit (ties
  by unit id): the creature crawls at it;
* nobody seen -> no intent: the creature idles (its hub's default).

It never swims, reaches or grabs (those verbs exist -- ORDER_MOVE_COVER /
ORDER_SPRINT, AIM, USE -- for a later brain). It draws no randomness and
reads only synced state (unit positions, the map), so a match plays the same
bit for bit on every run. ``creature_controller = "idle"`` in match.toml
leaves the creatures to their own idle (``IdleControl`` emits nothing).
"""
from __future__ import annotations

import math

from control_source import ControlSource, quantize_stick_direction
from simulation.factions import CREATURE_TEAM
from simulation.orders import ORDER_MOVE_ATTACK


class CreatureControl(ControlSource):
    """Drives every living creature unit on ``team`` toward the nearest unit it
    can see, once per tick."""

    def __init__(self, team: int = CREATURE_TEAM):
        self.team = int(team)

    def handle_frame(self, sim, renderer=None) -> None:
        from simulation import vision
        from simulation.creature_system import is_creature
        for u in sim.units:
            if not (u.alive and u.team == self.team and is_creature(u)):
                continue
            target = self._nearest_seen(sim, u, vision, is_creature)
            if target is None:
                sim.clear_move_dir(u.id)
                continue
            dx = target.center_tile_x() - u.center_tile_x()
            dy = target.center_tile_y() - u.center_tile_y()
            n = math.sqrt(dx * dx + dy * dy)
            if n <= 0.0:
                sim.clear_move_dir(u.id)
                continue
            qx, qy, _ = quantize_stick_direction(dx / n, dy / n, deadzone=0.0)
            sim.set_move_dir(u.id, qx, qy, ORDER_MOVE_ATTACK)

    @staticmethod
    def _nearest_seen(sim, u, vision, is_creature):
        best, best_d2 = None, None
        for t in sim.units:
            if t is u or not t.alive or is_creature(t):
                continue
            if not vision.can_see(sim, u, t):
                continue
            dx = t.center_tile_x() - u.center_tile_x()
            dy = t.center_tile_y() - u.center_tile_y()
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2 or (d2 == best_d2 and t.id < best.id):
                best, best_d2 = t, d2
        return best


__all__ = ["CreatureControl"]
