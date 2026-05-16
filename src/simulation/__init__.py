"""Breach simulation package.

Headless, deterministic gameplay logic — no rendering or input dependencies.

Importable as ``simulation.*`` (e.g. ``from simulation import Simulation``) when
``<repo>/src`` is on ``sys.path``. ``main.py`` adds it via
``sys.path.insert(0, str(ROOT / "src"))``.

Phase 1 lifted the gameplay code out of the legacy ``game.py`` into individual
modules. Phase 2 introduces :class:`Simulation`, the central facade that owns
the GameMap, unit list, order queue, physics runner, and recorder and exposes
the ``apply_action / step / get_state`` triad described in
``docs/architecture.md`` Section 2. Both ``main.py`` (human play) and a future
``train.py`` (AI rollouts) talk to ``Simulation`` the same way.
"""

from __future__ import annotations

from simulation.simulation import Simulation, SimState

__all__ = ["Simulation", "SimState"]
