"""The explosion look's render cue (G4 #75, docs/explosion_look_design_2026-11-07.md
P1): an ExplosionEvent becomes ONE ``"blast"`` effect (the legacy ``"explosion"``
ring is deleted), and the game loops hand the renderer every sim step's events.
No window, no GL: ``consume_events`` only queues. Each test names the property it
protects and the change that must break it.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("pyray")

from renderer.game_renderer import GameRenderer  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402


class _NoPose:
    def queue_events(self, events):
        pass


def _renderer():
    r = GameRenderer.__new__(GameRenderer)   # no window: consume_events only queues
    r._effects = []
    r._last_sim_tick = 0
    r.unit_models = _NoPose()
    return r


def test_an_explosion_event_becomes_one_blast_effect_and_never_the_legacy_ring():
    """Property: each ExplosionEvent queues exactly one "blast" effect carrying the
    event's strength, approach, flash columns and the tick it arrived on, born
    `fresh` (not aged on its first frame, as tracers), and no "explosion" effect
    exists. BREAKS IF: the legacy ring is re-added, the effect drops a field the
    look reads, or tick0 is taken from anything but the caller's tick."""
    r = _renderer()
    ev = ExplosionEvent(pos=(5, 6), radius=2, kind="grenade", strength=1.5,
                        approach=(6, 4))
    r.consume_events([ev], sim_tick=48)
    blasts = [fx for fx in r._effects if fx["kind"] == "blast"]
    assert len(blasts) == 1
    assert not any(fx["kind"] == "explosion" for fx in r._effects)
    b = blasts[0]
    assert b["strength"] == 1.5 and b["approach"] == (6, 4)
    assert b["pos"] == (5, 6) and b["radius"] == 2 and b["kind_name"] == "grenade"
    assert b["flash"] == ev.flash and b["flash_decay_s"] == ev.flash_decay_s
    assert b["flash_color"] == ev.flash_color
    assert b["tick0"] == 48 and b["fresh"] is True and b["t"] == 0.0
    r._advance_effects(10.0)         # a slow first frame does not age it
    assert r._effects and r._effects[0]["t"] == 0.0
    # without an explicit tick, the last upload_state tick stamps it
    r2 = _renderer()
    r2._last_sim_tick = 77
    r2.consume_events([ev])
    assert r2._effects[0]["tick0"] == 77


def _blocks(tree):
    """Every statement list in the module (bodies of loops, ifs, functions)."""
    for node in ast.walk(tree):
        for field in ("body", "orelse"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield stmts


def _is_call(stmt, owner, name):
    for n in ast.walk(stmt):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == name and isinstance(n.func.value, ast.Name)
                and n.func.value.id == owner):
            return True
    return False


@pytest.mark.parametrize("path", ["main.py", "tools/lighting_demo.py"])
def test_the_game_loop_consumes_tick_events_after_every_step(path):
    """Property (design §6 S3): `Simulation.step` clears `tick_events`, so every
    `renderer.consume_events(sim.tick_events ...)` sits in the same statement
    block as, and after, a `sim.step()` -- never once per frame after the step
    loop, where all but the last step's detonations were lost and a paused frame
    re-consumed the same events. The main loop is not runnable headless here, so
    this reads its source. BREAKS IF: the consume moves back out of the step loop.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    found = 0
    for stmts in _blocks(tree):
        for i, st in enumerate(stmts):
            if not (isinstance(st, ast.Expr) and _is_call(st, "renderer",
                                                          "consume_events")):
                continue
            src = ast.unparse(st)
            if "tick_events" not in src:
                continue          # a local event list (lighting_demo's detonate)
            found += 1
            assert any(isinstance(p, ast.Expr) and _is_call(p, "sim", "step")
                       for p in stmts[:i]), \
                f"{path}: consume_events(sim.tick_events) not after a sim.step() " \
                f"in its block (line {st.lineno})"
    assert found >= 1
