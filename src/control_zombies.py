"""``ZombieControl`` — the horde as a match controller, ``"zombies"`` (#122).

rl-ready scope §3 item 1 / §5 item 4: in a match EVERY controller is a
:class:`~control_source.ControlSource` — the bots, the horde, later any
policy. The zombie AI is WRAPPED here, not rewritten: the law stays
``simulation.ai_zombie.update_zombies_tick`` (trigger radius + LOS, chain
propagation, nearest prey, melee, A* movement), run at the tick conductor's
zombie slot exactly where it always ran. What changes in a match is WHO asks
for it: the match runners hand the horde to its controller
(``Simulation.hand_zombies_to_controller``), after which the conductor runs
the law only for the zombies a controller names through the ZOMBIE intent
(``Simulation.drive_zombies``) — one driver per unit, never two.

The controller owns every living zombie, the spawned horde and every marine
converted since (conversion — ``convert_marines_to_zombies`` — stays where
MatchRules runs it, and moves the unit to ``ZOMBIE_TEAM``; its roster slot
stays its old team's). It emits one intent per tick naming them all, so the
law sees the same unit list, in the same order, as the conductor's own call:
a match plays bit for bit as it did before #122. It draws no randomness and
reads only synced state. Works with ``renderer=None``.

``zombie_controller = "idle"`` in match.toml gives the horde to
:class:`~control_source.IdleControl` instead, which emits nothing: the horde
stands where it spawned (a passive horde for tests).
"""
from __future__ import annotations

from control_source import ControlSource
from simulation.factions import ZOMBIE_TEAM


class ZombieControl(ControlSource):
    """Drives every living zombie through the ZOMBIE intent, once per tick."""

    def __init__(self, team: int = ZOMBIE_TEAM):
        if team != ZOMBIE_TEAM:
            raise ValueError(f"the zombie controller drives ZOMBIE_TEAM "
                             f"({ZOMBIE_TEAM}), not team {team}")
        self.team = team

    def handle_frame(self, sim, renderer=None) -> None:
        ids = [u.id for u in sim.units if u.alive and u.is_zombie]
        if ids:
            sim.drive_zombies(ids)


__all__ = ["ZombieControl"]
