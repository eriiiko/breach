"""G1p-c (gameplay pack v1, #183) -- the render-only ``ReloadEvent``.

``combat.reload_begin`` appends ``ReloadEvent(unit_id, ticks)`` to the tick's
events on the tick a reload STARTS -- the manual one (``reload_intent``) and
the automatic one (the magazine empties) alike -- with ``ticks`` the stall's
length. Like ``ShotFiredEvent`` it is outside the synced event digest, so it
moves no digest. Its consumer is the animation session (#184).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_match import run_match  # noqa: E402
from simulation import combat as combat_mod  # noqa: E402
from simulation import simulation as sim_mod  # noqa: E402
from simulation.events import ReloadEvent  # noqa: E402
from test_magazines import (MAG, RELOAD_TICKS, _Range,  # noqa: E402
                            _reload_match)


class _Watch(_Range):
    """The magazines' range, also collecting ``(tick, ReloadEvent)``."""

    def __init__(self, magazines=3):
        super().__init__(magazines=magazines)
        self.reloads: list = []

    def step(self) -> int:
        t = self.sim.tick
        n = super().step()
        self.reloads += [(t, e) for e in self.sim.tick_events
                         if isinstance(e, ReloadEvent)]
        return n


def test_a_manual_reload_emits_exactly_one_event_with_the_stall_ticks():
    """Property: consuming a manual reload emits exactly ONE ReloadEvent on
    that tick, naming the unit and the stall's ticks, and nothing on the
    ticks before or after it (the stall itself emits nothing). Breaks if the
    event is emitted from the intent instead of the reload's start, once per
    stall tick, or with another length."""
    r = _Watch()
    r.hold()
    r.run_until_shots(5)
    r.hold(False)
    r.run(3)
    assert r.reloads == []
    r.sim.reload_intent(r.uid)
    t0 = r.sim.tick
    r.run(RELOAD_TICKS + 10)
    assert len(r.reloads) == 1
    t, ev = r.reloads[0]
    assert t == t0
    assert (ev.unit_id, ev.ticks) == (r.uid, RELOAD_TICKS)


def test_the_automatic_reload_emits_one_event_per_emptied_magazine():
    """Property: each magazine fired empty starts the automatic reload and
    emits exactly one ReloadEvent, on the tick of the magazine's last round,
    with the stall's ticks -- two magazines, two events; the unit going dry
    (no spare) emits none. Breaks if the automatic path does not go through
    the one emission site, or a dry unit claims a reload."""
    r = _Watch(magazines=3)
    r.hold()
    r.run_until_shots(3 * MAG)
    r.run(RELOAD_TICKS + 50)                  # dry now: nothing more starts
    assert [e.ticks for _, e in r.reloads] == [RELOAD_TICKS, RELOAD_TICKS]
    assert all(e.unit_id == r.uid for _, e in r.reloads)
    last_rounds = [r.shot_ticks[MAG - 1], r.shot_ticks[2 * MAG - 1]]
    assert [t for t, _ in r.reloads] == last_rounds


def test_a_dry_reload_intent_emits_nothing():
    """Property: the verb with no spare magazine starts no reload and emits
    no event. Breaks if the event is emitted before the spare check."""
    r = _Watch(magazines=1)
    r.hold()
    r.run_until_shots(3)
    r.hold(False)
    r.sim.reload_intent(r.uid)
    r.run(RELOAD_TICKS + 5)
    assert r.reloads == []


def test_the_event_moves_no_digest(tmp_path, monkeypatch):
    """Property: the event is render-only -- a match whose units reload (by
    choice and by emptying) gives the same result, field digest and tick
    digest with the emission switched off (``reload_begin`` handed no event
    list) as with it on, and it did emit with it on. Breaks if the event
    enters the synced digest or the emission changes any sim state."""
    lvl = _reload_match(tmp_path, monkeypatch)
    seen = []
    real_begin = combat_mod.reload_begin

    def counting(u, weapon, tick, events=None):
        started = real_begin(u, weapon, tick, events=events)
        if started and events is not None:
            seen.append(events[-1])
        return started

    monkeypatch.setattr(combat_mod, "reload_begin", counting)
    monkeypatch.setattr(sim_mod, "reload_begin", counting)
    on = run_match(lvl)
    assert seen and all(isinstance(e, ReloadEvent) for e in seen)

    def silent(u, weapon, tick, events=None):
        return real_begin(u, weapon, tick, events=None)

    monkeypatch.setattr(combat_mod, "reload_begin", silent)
    monkeypatch.setattr(sim_mod, "reload_begin", silent)
    off = run_match(lvl)
    assert (on.result, on.field_digest, on.tick_digest) == \
        (off.result, off.field_digest, off.tick_digest)
