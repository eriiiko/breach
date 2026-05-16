"""Breach simulation package.

Headless, deterministic gameplay logic — no rendering or input dependencies.

Importable as ``simulation.*`` (e.g. ``from simulation import GameMap``) when
``<repo>/src`` is on ``sys.path``. ``main.py`` adds it via
``sys.path.insert(0, str(ROOT / "src"))``.

Phase 1 lifts the gameplay code out of the legacy ``game.py`` into individual
modules here. The future ``simulation.Simulation`` facade (Phase 2) will own
the GameMap, unit list, order queue, physics runner, and recorder and expose
the ``apply_action / step / get_state`` triad described in
``docs/architecture.md`` Section 2.
"""

from __future__ import annotations
