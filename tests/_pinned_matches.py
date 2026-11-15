"""tests/_pinned_matches.py -- the two pinned ship_demo matches, played ONCE
per pytest process and shared by every test that only WATCHES them (#205).

Before this, six slow tests each re-played one of the same two matches just
to hang a different read-only ``on_tick`` observer on it (ship_demo seed 1
under ``bots``, 2721 ticks, ~50 s; and its ``bots_v1`` twin, 264 ticks,
~90 s) -- about 500 s of a serial suite run.

:func:`pinned_play` plays a match through THE runner (``run_match``) and keeps,
for every tick, a :class:`TickView`: a deep copy of what the observers read
off the sim right after that tick's ``step()`` (the units, the tick's events,
the planted charges, ``charges_refused``, the tick number). :func:`watch`
feeds an observer those views in tick order and returns the match's
``MatchOutcome``. An observer written for ``run_match(..., on_tick=f)`` runs
unchanged: it reads the same attributes, with the same values, at the same
point of the tick.

Rules:
- Only a test that changes NOTHING about the match (no monkeypatch, no other
  spec, seed or level, no mutated unit) may use it. A test that changes the
  match plays its own -- that change is what it tests.
- An observer must not write to a view (``run_match``'s own rule for
  ``on_tick``): the views are shared by every later test in the process.
- The shared play is self-checking: ``test_bots_v1_play_the_pre_g8p_pinned_match``
  and ``test_the_controller_plays_the_pinned_ship_demo_match`` compare its
  outcome AND its per-tick unit stream with their pins, so a recording that
  drifted from a fresh play fails there.
"""
from __future__ import annotations

import copy
import functools
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: the two pinned matches: ``level.match`` as shipped, and every team on
#: ``bots_v1`` (``test_bots_v1.v1_spec``).
WHICH = ("bots", "bots_v1")


@dataclass(frozen=True)
class TickView:
    """What an ``on_tick`` observer reads off the sim after one tick."""
    tick: int
    units: list
    tick_events: list
    planted_charges: list
    charges_refused: int


@dataclass(frozen=True)
class PinnedPlay:
    outcome: object                 # run_match.MatchOutcome
    ticks: tuple                    # TickView per tick, in order
    #: how much ``status.OVERFLOW_REFUSED`` grew during the play.
    overflow_refused: int


def ship_demo():
    import level_loader
    return level_loader.load("ship_demo", str(ROOT / "levels"))


def spec_of(level, which):
    if which == "bots":
        return None
    if which == "bots_v1":
        return replace(level.match, teams=tuple(
            replace(t, controller="bots_v1") for t in level.match.teams))
    raise ValueError(f"no pinned match {which!r}; known: {WHICH}")


@functools.lru_cache(maxsize=None)
def pinned_play(which: str) -> PinnedPlay:
    """Play pinned match ``which`` once per process; keep every tick's view."""
    from run_match import run_match
    from simulation import status
    level = ship_demo()
    spec = spec_of(level, which)
    views = []

    def record(sim):
        # ONE deepcopy call, so a charge and its owner stay the same object.
        units, events, charges = copy.deepcopy((
            list(sim.units), list(sim.tick_events),
            list(getattr(sim, "planted_charges", ()) or ())))
        views.append(TickView(tick=sim.tick, units=units, tick_events=events,
                              planted_charges=charges,
                              charges_refused=sim.charges_refused))
    before = sum(status.OVERFLOW_REFUSED.values())
    outcome = run_match(level, spec, on_tick=record)
    return PinnedPlay(outcome=outcome, ticks=tuple(views),
                      overflow_refused=sum(status.OVERFLOW_REFUSED.values())
                      - before)


def watch(which: str, on_tick):
    """Run ``on_tick`` over every tick of pinned match ``which``, in order,
    as ``run_match(level, spec, on_tick=on_tick)`` would; return its
    ``MatchOutcome``."""
    play = pinned_play(which)
    for view in play.ticks:
        on_tick(view)
    return play.outcome
