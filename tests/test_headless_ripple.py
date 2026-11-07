"""#29 P4a — HEADLESS: the render-only W6a ripple is skipped, and nothing
synced notices.

``PhysicsRunner.headless`` (-> ``PhysicsEngine.headless``) is set by the
headless runners (``MatchSession``, so ``tools/run_match.py`` and
``rl_env``), never by ``main.py``. With it on, ``step_tail`` does not call
``step_ripple`` (and the resident tick neither runs it nor downloads its
inputs for it). The ripple is render-only float state, outside the field
digest (tests/field_digest_spec.toml's excluded_float_fields), so the flag
must change NO digest.

The scene is test_water_ripple's rippling room: a sealed room with a painted
pool and a pressure bump over it (the splash source |P - P_prev|), so the
windowed engine really ripples (the non-vacuity guard).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT / "tests"))

import breach_physics as bp  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import field_digest, tick_digest  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from water_q16 import q  # noqa: E402

SEED = 42
N_TICKS = 60


def _rippling_room(headless: bool):
    tm = np.ones((12, 12), dtype=np.int32)
    tm[1:11, 1:11] = 4
    level = LevelData(name="headless_ripple_room", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=0.333, diffuse_path=Path("."))
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    g.water_depth[interior] = q(0.3)        # a painted pool
    g.atmosphere[5, 4:8] += q(0.8)          # a pressure bump over it (the splash)
    sim.physics_runner.headless = headless
    sim.set_paused(False)
    return sim


def _run(headless: bool):
    sim = _rippling_room(headless)
    rippled = False
    digests = []
    for _ in range(N_TICKS):
        sim.step()
        rippled = rippled or bool(sim.gmap.ripple.any())
        snap = capture_snapshot(sim)
        digests.append((field_digest(snap), tick_digest(snap)))
    return rippled, digests


def test_headless_skips_the_ripple_and_moves_no_digest():
    """Property: on a rippling scene, a headless engine leaves ripple /
    ripple_v exactly zero while the windowed engine ripples, and the per-tick
    field digest and tick digest are identical with the flag on and off for
    60 ticks. Breaks if `headless` stops skipping step_ripple, if the skip
    moves a synced field (the ripple feeding anything back), or if the flag
    starts gating anything synced."""
    windowed_rippled, windowed = _run(headless=False)
    headless_rippled, headless = _run(headless=True)
    assert windowed_rippled, "the windowed run never rippled -- vacuous scene"
    assert not headless_rippled, "the headless run rippled -- step_ripple was not skipped"
    for t, (a, b) in enumerate(zip(windowed, headless)):
        assert a == b, f"tick {t}: a digest moved with the headless flag ({a} vs {b})"


def test_a_match_session_runs_headless_and_a_simulation_does_not():
    """Property: MatchSession (THE headless match loop: run_match, the farm)
    sets its runner headless; a plain Simulation (main.py's) does not. Breaks
    if the headless runners stop setting the flag (the ripple and the render
    planes would cost every training tick) or if a windowed game gets it (the
    water would stop rippling on screen)."""
    from level_loader import load as load_level
    from simulation.match_session import MatchSession
    ses = MatchSession(load_level("ship_demo"), seed=1, breach_physics=bp)
    assert ses.sim.physics_runner.headless is True
    assert bool(ses.sim.physics_runner.engine.headless) is True
    sim = _rippling_room(headless=False)
    plain = Simulation(sim.level, seed=SEED, breach_physics=bp, enable_recorder=False)
    assert plain.physics_runner.headless is False
