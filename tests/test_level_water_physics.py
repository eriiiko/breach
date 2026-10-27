"""[water] initial state — physics-bound half of P5's test spine
(engine/15 §2.3, docs/patch_levels_p5_water.md §2.5).

Runs the committed ``levels/aquarium_demo`` through the REAL loader path
into a full ``Simulation`` and pins the two seed properties the design gate
proved on paper:

  1. Σ-conservation: 100 runner ticks leave the total depth conserved
     (template: test_water_integration.test_runner_conserves_painted_water);
  2. at-rest BYTE-IDENTITY: the glass-bounded FLAT seed is bit-exactly
     unchanged after 100 ticks (physics critique: uniform surface + mirror
     BC at solid faces -> zero gradient -> zero velocity -> zero flux; the
     W3 ratio is identically 1.0; inductive over ticks). This property is
     proven for seeds whose wetted boundary is ENTIRELY solid — NEVER
     assert UNCHANGED for open-edge pools (they spread by design); the
     headless suite (test_level_water.py) pins the demo tank's
     solid-bounded precondition.

Also pins the no-tick-1-spike contract: the runner's ``_water_depth_before``
lazy seed counts level-painted water as pre-existing, so the sealed room's
atmosphere sees no W3 compression from the seed itself.

This module needs the compiled ``breach_physics`` extension and therefore
uses the module-level skip pattern of tests/test_bedrock_cliff_counts.py —
the ONLY correct one (an unguarded import hard-fails at collection). On a
worktree without cpp/build it SKIPS whole; the orchestrator re-runs it on
the full checkout before merge.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_level_water_physics.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import breach_physics as bp
except ImportError as exc:  # pragma: no cover - build missing
    pytest.skip(f"breach_physics extension not built: {exc}",
                allow_module_level=True)

from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from water_q16 import deq  # noqa: E402

SEED = 42
TICKS = 100          # < ticks_per_round (240): no end-of-round auto-pause


def _aquarium_sim():
    """The REAL loader path (design §2.5): the committed demo level, no
    synthetic LevelData — exactly what main.py would build."""
    level = load_level("aquarium_demo")
    sim = Simulation(level, seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    return sim


def test_aquarium_seed_conserves_total_over_100_ticks():
    """Σ depth conserved through 100 full runner ticks (the
    test_runner_conserves_painted_water template's bound)."""
    sim = _aquarium_sim()
    g = sim.gmap
    total0 = float(deq(g.water_depth).sum(dtype=np.float64))
    assert total0 > 0.0, "aquarium seed did not land"
    sim.set_paused(False)

    for _ in range(TICKS):
        sim.step()

    assert np.isfinite(deq(g.water_depth)).all()
    total1 = float(deq(g.water_depth).sum(dtype=np.float64))
    assert abs(total1 - total0) / max(total0, 1e-12) < 1e-4, (
        f"aquarium leaked water: {total0} -> {total1}")


def test_aquarium_flat_glass_bounded_seed_byte_unchanged():
    """The at-rest pin (design §1, scoped): the glass-bounded FLAT 1.2 m
    tank is BIT-EXACTLY unchanged after 100 ticks — depth field byte-equal
    to the tick-0 seed, flow velocities never kicked. Solid-bounded seeds
    ONLY (the headless suite pins the demo's boundary); an open-edge pool
    would spread by design and must never get this assertion."""
    sim = _aquarium_sim()
    g = sim.gmap
    seed0 = g.water_depth.copy()
    assert seed0.any()
    sim.set_paused(False)

    for _ in range(TICKS):
        sim.step()

    assert np.array_equal(g.water_depth, seed0), (
        "glass-bounded flat seed moved — the at-rest bit-exactness "
        "(mirror BC, zero gradient) is broken")
    assert not g.flow_vx.any() and not g.flow_vy.any(), (
        "at-rest tank acquired flow velocity")


def test_aquarium_seed_counts_as_preexisting_no_tick1_spike():
    """The _water_depth_before lazy seed (physics_runner.py) copies the
    CURRENT depth on the first call, so level-painted water is
    'pre-existing': the W3 displacement accounting must not compress the
    room's atmosphere on tick 1 (no spike). One tick, then the runner's
    snapshot equals the seed and the atmosphere stays finite and sane."""
    sim = _aquarium_sim()
    g = sim.gmap
    seed0 = g.water_depth.copy()
    atm0 = g.atmosphere.copy()
    sim.set_paused(False)

    sim.step()

    before = sim.physics_runner._water_depth_before
    assert before is not None
    assert np.array_equal(before, seed0), (
        "_water_depth_before did not arm with the level seed")
    # No W3 compression pulse from the seed itself: the at-rest tank
    # displaces nothing NEW, so the interior atmosphere should only move by
    # the ordinary EOS equilibrium settle, never a water-shaped pulse.
    #
    # G12 NOTE (2026-08-31, issue #12, docs/fire_g12_one_map_patch_2026-08-31.md
    # §6 point 1): pre-G12 this was bit-identical (no move at all) because the
    # old EOS pressure calibration (C = 1/290, T_amb_abs = 290) is an exact
    # reciprocal pair in the Q16.16 chain, so N = quantize(1.0) solved
    # straight back to itself. Under G12 (C = 1/293) that quantized exactness
    # is lost by a few LSB, so every interior cell settles from 65536 to
    # 65542 raw (~0.01%) on tick 1 — confirmed UNIFORM (every moved cell
    # lands on the same value, not a spatial pulse shape) and confirmed NOT
    # water-shaped (same calibration-only mechanism as
    # test_water_displacement.py's wet-static gate, which measures a
    # different uniform value — 65635 — in its own differently-sized sealed
    # room; the two need not agree, only each be internally uniform).
    moved = g.atmosphere != atm0
    assert not moved.any() or np.all(g.atmosphere[moved] == 65542), (
        "tick-1 atmosphere moved off the measured G12 uniform settle value "
        "(65542 raw) -- either a real water-shaped compression spike, or "
        "this pin needs re-measuring")


def test_reset_reapplies_the_seed_without_spike():
    """Scout fact m4: reset() builds a fresh GameMap AND a fresh runner —
    the seed reapplies and _water_depth_before re-arms (drain the tank,
    reset, the water is back; no spike machinery needed)."""
    sim = _aquarium_sim()
    seed0 = sim.gmap.water_depth.copy()
    sim.gmap.water_depth[:] = 0                   # "the player drained it"
    sim.reset(seed=SEED)
    assert np.array_equal(sim.gmap.water_depth, seed0), (
        "reset() did not reapply the [water] seed")
    sim.set_paused(False)
    sim.step()                                    # arms the fresh snapshot
    assert np.array_equal(sim.physics_runner._water_depth_before, seed0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
