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

The ``Simulation``/``SimState`` re-export is LAZY (PEP 562): the editor and
other schema-only tooling import leaf modules (``simulation.materials``,
``simulation.entities``) directly, and the import-light rule (entity design
§3b, CI-tested) requires that doing so never drags in the sim loop. An eager
``from simulation.simulation import ...`` here would pull the whole physics
stack on ANY ``simulation.*`` import; the module-level ``__getattr__`` defers
it to first attribute access, so every existing ``from simulation import
Simulation`` call site is unchanged.
"""

from __future__ import annotations

__all__ = ["Simulation", "SimState"]


def __getattr__(name):
    if name in __all__:
        from simulation import simulation as _simulation
        return getattr(_simulation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
