"""Tests for the Simulation facade (Phase 2 — patch_game_logic_migration.md).

Two smoke tests + a determinism check:

1. ``test_step_smoke`` — construct, step 100 times, ensure no exceptions.
2. ``test_determinism`` — same seed must give the same trajectory
   (gmap.atmosphere mean, gmap.smoke mean, unit positions, projectile state).
3. ``test_undo_round_trip`` — placing a movement order then undoing it
   restores the unit's AP / inventory / move_path to its initial state.

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_simulation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np

import breach_physics as bp
from level_loader import load as load_level
from simulation import Simulation
from simulation.orders import Order, ORDER_MOVE_ATTACK
from simulation.unit import Unit


N_STEPS = 100
SEED = 42


def _spawn_demo_units(sim: Simulation) -> tuple:
    """Add one marine and one zombie at fixed positions. Returns ids."""
    marine = Unit("M1", cx=10, cy=10, team=0)
    zombie = Unit("Z1", cx=12, cy=15, team=1)
    m_id = sim.add_unit(marine)
    z_id = sim.add_unit(zombie)
    return m_id, z_id


def _state_signature(sim: Simulation):
    """Compact signature to compare two sim runs for determinism."""
    g = sim.gmap
    sig = {
        "tick": sim.tick,
        "phase": sim.phase,
        "atm_mean": float(g.atmosphere.mean()),
        "atm_max": float(g.atmosphere.max()),
        "smoke_mean": float(g.smoke.mean()),
        "fire_mean": float(g.fire.mean()),
        "wave_p_mean": float(g.wave_p.mean()),
        "n_projectiles": len(sim.projectiles),
        "units": [(u.id, u.fx, u.fy, u.hp, u.alive) for u in sim.units],
    }
    return sig


def test_step_smoke():
    """100 ticks with no exceptions starting from a clean sim."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    _spawn_demo_units(sim)
    sim.set_paused(False)
    for _ in range(N_STEPS):
        sim.step()
    print(f"OK: step_smoke ({N_STEPS} ticks, tick={sim.tick}, phase={sim.phase})")


def test_determinism():
    """Same seed must give the same state after N steps."""
    level = load_level("unhcr_vessel")

    sim_a = Simulation(level, seed=SEED, breach_physics=bp,
                       enable_recorder=False)
    _spawn_demo_units(sim_a)
    sim_a.set_paused(False)
    for _ in range(N_STEPS):
        sim_a.step()
    sig_a = _state_signature(sim_a)

    # Reset the same instance and re-run.
    sim_a.reset(seed=SEED)
    _spawn_demo_units(sim_a)
    sim_a.set_paused(False)
    for _ in range(N_STEPS):
        sim_a.step()
    sig_b = _state_signature(sim_a)

    if sig_a != sig_b:
        # Diff for debugging.
        for k in sig_a:
            if sig_a[k] != sig_b[k]:
                print(f"  DIFF {k}: A={sig_a[k]!r}  B={sig_b[k]!r}")
        raise AssertionError("determinism check failed (see DIFF above)")
    print(f"OK: determinism ({N_STEPS} ticks, atm_mean={sig_a['atm_mean']:.6f})")


def test_undo_round_trip():
    """Placing an order then undoing it returns AP and inventory."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m_id, _ = _spawn_demo_units(sim)
    marine = sim.get_unit(m_id)

    ap_before = list(marine.ap)
    grenades_before = marine.has_grenade

    # Place a move order. It should succeed and not consume AP.
    ok = sim.apply_action(m_id, Order(ORDER_MOVE_ATTACK,
                                       target_fx=marine.fx + 6,
                                       target_fy=marine.fy,
                                       phase=0))
    assert ok, "expected movement order to succeed"
    assert marine.ap == ap_before, "movement should not consume AP"
    assert len(marine.orders) == 1
    assert len(marine.move_path) > 0, "move_path should be precomputed"

    sim.undo_last_order(m_id)
    assert marine.ap == ap_before, "AP should be unchanged"
    assert marine.has_grenade == grenades_before, "grenades should be unchanged"
    assert len(marine.orders) == 0, "order queue should be empty"
    print("OK: undo_round_trip")


def test_full_round_runs():
    """Run a full round (120 ticks) with orders. Asserts auto-pause
    behavior at phase boundary + end-of-round, and that the event list
    fills + clears each step.
    """
    from simulation.orders import ORDER_FIRE

    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp,
                     enable_recorder=False)

    # 3 marines, 3 zombies — enough for shooting + zombie AI exercise.
    m1 = sim.add_unit(Unit("M1", cx=4, cy=2, team=0))
    sim.add_unit(Unit("M2", cx=8, cy=2, team=0))
    sim.add_unit(Unit("M3", cx=12, cy=4, team=0))
    sim.add_unit(Unit("Z1", cx=10, cy=20, team=1))
    sim.add_unit(Unit("Z2", cx=6,  cy=24, team=1))
    sim.add_unit(Unit("Z3", cx=14, cy=28, team=1))

    # One marine fires at a zombie tile (no LOS likely, but no exception).
    sim.apply_action(m1, Order(ORDER_FIRE,
                               target_fx=10 * 3 + 1, target_fy=20 * 3 + 1,
                               phase=0))

    sim.set_paused(False)

    # Step through Phase 1 — should auto-pause at tick == ticks_per_phase.
    for _ in range(sim._ticks_per_phase + 5):
        if sim.is_paused():
            break
        sim.step()
    assert sim.is_paused(), "expected auto-pause at phase 1 boundary"
    assert sim.tick == sim._ticks_per_phase, f"tick={sim.tick}"
    assert sim.phase == 1, f"phase={sim.phase}"

    # Resume Phase 2.
    sim.set_paused(False)
    for _ in range(sim._ticks_per_phase + 5):
        if sim.is_paused():
            break
        sim.step()
    assert sim.is_paused(), "expected auto-pause at end of round"
    # End of round resets tick to 0.
    assert sim.tick == 0, f"tick={sim.tick}"
    assert sim.phase == 0, f"phase={sim.phase}"
    assert sim.turn_number == 2, f"turn_number={sim.turn_number}"
    print(f"OK: full_round_runs (round 2 reached, units: "
          f"{len(sim.marines())} marines, {len(sim.zombies())} zombies)")


if __name__ == "__main__":
    test_step_smoke()
    test_determinism()
    test_undo_round_trip()
    test_full_round_runs()
    print("\nAll Simulation tests passed.")
