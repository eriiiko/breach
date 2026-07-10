"""The W3 gas-weapons gate: gas grenades end-to-end + the two owed coupling
rows (mechanics/05 §1 — gas[teargas] -> BLINDED, gas[poison] -> POISON DoT).

What is locked here:

  - GAS GRENADES E2E: a smoke/tear/poison grenade order (ammo_name on the
    shipped ORDER_GRENADE flow) detonates into the RIGHT gas slice and the
    C++ transport carries the cloud on subsequent ticks (all slices are
    stepped by the per-gas loop — the transport verification of record);
  - TEARGAS -> BLINDED -> SNAP CONE (the owed P3 can_aim consumer): the
    coupling row applies refresh-stacked BLINDED above the density
    threshold; a blinded unit's AIMED fire order draws from the SNAP cone
    (predicted draw-for-draw on a parallel generator);
  - POISON DoT: exact per-tick amounts (the heat row's idiom — dps x
    density / tps, quantized round-half-away at the HP boundary), the
    threshold edge (one count below = nothing), and ZOMBIE IMMUNITY
    (resist 0 -> no packet, no event, 0 damage — lazy emission);
  - DORMANCY: gas-free planes cost one .any() and change nothing; the rows
    draw no RNG ever.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_gas_weapons.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import gas_fixed, unit_fixed  # noqa: E402
from simulation.combat import process_shooting  # noqa: E402
from simulation.events import ShotFiredEvent, UnitHitEvent  # noqa: E402
from simulation.exchange import apply_poison_dose, apply_teargas_blind  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import (  # noqa: E402
    N_TRACE_GASES, POISON as GAS_POISON, TEARGAS as GAS_TEARGAS,
    WHITE_SMOKE,
)
from simulation.orders import ORDER_FIRE, ORDER_GRENADE, Order  # noqa: E402
from simulation.status import (  # noqa: E402
    BLINDED, STATUS_REGISTRY, composed_flags,
)
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import AmmoDef, WeaponDef  # noqa: E402
from simulation import weapons as weapons_mod  # noqa: E402

SEED = 20260705


def _level(h=24, w=24, edits=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w3_gas", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _fill_footprint(slice_, unit, density):
    q = gas_fixed.quantize_scalar(density)
    for (tx, ty) in unit.occupied_tiles():
        slice_[ty, tx] = q
    return q


# ---------------------------------------------------------------------------
# Gas grenades end-to-end (+ the transport verification of record)
# ---------------------------------------------------------------------------
def test_gas_grenade_e2e_deposits_and_transports():
    """A smoke grenade through the shipped LOBBED flow: has_grenade
    decrements, the white_smoke slice fills at the target, every other
    slice stays empty — and the cloud SURVIVES subsequent ticks (the C++
    per-gas transport loop steps every non-empty slice: verified in code,
    physics_engine.cpp run_substeps' gi-loop; this is the empirical twin)."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=3, y=3, team=0)
    mid = sim.add_unit(m)
    grenades_before = m.has_grenade
    assert sim.apply_action(mid, Order(
        ORDER_GRENADE, target_fx=12, target_fy=10, phase=0,
        grenade_fuse=0.25, ammo_name="grenade_smoke"))
    assert m.has_grenade == grenades_before - 1     # the single count pool
    sim.spawn_projectiles_from_grenade_orders()

    # Fuse 0.25 s -> detonation at tick 6; the deposit flushes NEXT step
    # (detonation enqueues during slot 2; the flush is 6b of that same step
    # -> visible after the detonation step returns).
    for _ in range(8):
        sim.set_paused(False)
        sim.step()
    ws_total = int(sim.gmap.gas[WHITE_SMOKE].astype(np.int64).sum())
    assert ws_total > 0
    # Trace slices only (0..N_TRACE_GASES-1) — the bulk O2/inert_N2 pair
    # (EOS refactor P1) always carries ambient air, unrelated to a smoke
    # grenade deposit, and moves under its own conservative transport.
    for g in range(N_TRACE_GASES):
        if g != WHITE_SMOKE:
            assert not sim.gmap.gas[g].any(), f"slice {g} moved on a smoke grenade"

    # Transport: the cloud persists (advects/diffuses, integer-SL gentle
    # decay — not wiped) across further ticks.
    for _ in range(6):
        sim.set_paused(False)
        sim.step()
    ws_later = int(sim.gmap.gas[WHITE_SMOKE].astype(np.int64).sum())
    assert ws_later > 0
    # Diffusion spread it: more nonzero cells than the initial disc alone
    # would explain shrinking — assert the cloud still has real mass.
    assert ws_later > ws_total // 4


def test_tear_and_poison_grenades_fill_their_slices():
    for ammo_name, slice_id in [("grenade_tear", GAS_TEARGAS),
                                ("grenade_poison", GAS_POISON)]:
        sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                         enable_recorder=False)
        m = Unit("M", x=3, y=3, team=0)
        mid = sim.add_unit(m)
        assert sim.apply_action(mid, Order(
            ORDER_GRENADE, target_fx=12, target_fy=10, phase=0,
            grenade_fuse=0.25, ammo_name=ammo_name))
        sim.spawn_projectiles_from_grenade_orders()
        for _ in range(8):
            sim.set_paused(False)
            sim.step()
        assert sim.gmap.gas[slice_id].any()
        # Trace slices only — see the smoke-grenade test above.
        for g in range(N_TRACE_GASES):
            if g != slice_id:
                assert not sim.gmap.gas[g].any()


# ---------------------------------------------------------------------------
# Teargas -> BLINDED -> snap cone (the owed can_aim consumer)
# ---------------------------------------------------------------------------
def test_teargas_row_applies_refresh_stacked_blinded():
    gmap = _room()
    u = Unit("M", x=8, y=8, team=0)
    u.id = 1
    # Below threshold: nothing (threshold edge — one count under).
    thresh_q = gas_fixed.quantize_scalar(float(CFG.exchange.teargas_blind_density))
    for (tx, ty) in u.occupied_tiles():
        gmap.gas[GAS_TEARGAS][ty, tx] = thresh_q - 1
    apply_teargas_blind([u], gmap)
    assert not getattr(u, "statuses", [])

    # At threshold: BLINDED lands with the config duration; composed
    # can_aim goes False; everything else stays permitted.
    for (tx, ty) in u.occupied_tiles():
        gmap.gas[GAS_TEARGAS][ty, tx] = thresh_q
    apply_teargas_blind([u], gmap)
    assert len(u.statuses) == 1
    st = u.statuses[0]
    assert st.kind == BLINDED
    assert st.remaining_ticks == int(CFG.exchange.teargas_blind_ticks)
    flags = composed_flags(u)
    assert flags.can_aim is False
    assert flags.can_move is True and flags.can_act is True
    assert flags.is_prone is False
    assert STATUS_REGISTRY[BLINDED].dtype is None       # pure CC, no packets

    # REFRESH stacking: tick it down, re-qualify -> the SAME instance re-ups.
    st.remaining_ticks = 3
    apply_teargas_blind([u], gmap)
    assert len(u.statuses) == 1
    assert u.statuses[0] is st
    assert st.remaining_ticks == int(CFG.exchange.teargas_blind_ticks)


def test_blinded_unit_fires_snap_cone_on_an_aimed_order():
    """The consumer end-to-end: teargas -> the coupling row -> BLINDED ->
    process_shooting selects spread_snap_deg for an EXPLICIT fire order.
    Predicted draw-for-draw: the parallel generator scales the same uniform
    by the SNAP cone; the un-gassed control uses the AIMED cone (0 deg —
    exact axis)."""
    w = WeaponDef("w3_tear_test", "projectile", ammo_family="w3_tear_fam",
                  spread_deg=0.0, spread_snap_deg=25.0, range_tiles=30,
                  shots_per_trigger=1)
    a = AmmoDef("w3_tear_std", "w3_tear_fam", "kinetic", damage=10,
                speed_tiles_per_tick=96.0)
    a.speed_q16 = unit_fixed.quantize_scalar(96.0)

    def scene(gassed):
        gmap = _room()
        shooter = Unit("S", x=2, y=8, team=0)
        shooter.id = 1
        shooter.weapon_id = w.name
        shooter.orders = [Order(ORDER_FIRE, target_fx=15, target_fy=9, phase=0)]
        if gassed:
            _fill_footprint(gmap.gas[GAS_TEARGAS], shooter, 0.5)
            apply_teargas_blind([shooter], gmap)      # the coupling row
            assert composed_flags(shooter).can_aim is False
        return gmap, shooter

    tables = weapons_mod.get_tables()
    tables.weapons.by_name[w.name] = w
    tables.ammo.by_name[a.name] = a
    try:
        # (a) BLINDED: the aimed order draws from the SNAP cone (25 deg).
        gmap, shooter = scene(gassed=True)
        rng = np.random.default_rng(SEED)
        parallel = np.random.default_rng(SEED)
        snap_draw = float(parallel.uniform(-math.radians(25.0),
                                           math.radians(25.0)))
        assert abs(snap_draw) > 1e-3                  # comfortably off-axis
        shots, events = [], []
        process_shooting(gmap, [shooter], 0, shots, 0.0, rng, events=events)
        tracers = [e for e in events if isinstance(e, ShotFiredEvent)]
        assert len(tracers) == 1
        assert tracers[0].to_tile[1] != 9.0           # the snap cone moved it
        assert rng.bit_generator.state == parallel.bit_generator.state

        # (b) Control (no gas): the aimed cone (0 deg) flies the exact axis.
        gmap2, shooter2 = scene(gassed=False)
        rng2 = np.random.default_rng(SEED)
        shots2, events2 = [], []
        process_shooting(gmap2, [shooter2], 0, shots2, 0.0, rng2,
                         events=events2)
        tracers2 = [e for e in events2 if isinstance(e, ShotFiredEvent)]
        assert len(tracers2) == 1
        assert tracers2[0].to_tile[1] == 9.0          # exact axis: aimed
    finally:
        weapons_mod.rebuild_tables()


def test_teargas_blinds_through_the_simulation_tick():
    """Full-conductor wiring (step 9c3): a marine standing in teargas is
    BLINDED after one stepped tick; the status suppresses from the next."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=8, y=8, team=0)
    sim.add_unit(m)
    _fill_footprint(sim.gmap.gas[GAS_TEARGAS], m, 0.6)
    sim.set_paused(False)
    sim.step()
    kinds = [st.kind for st in getattr(m, "statuses", [])]
    assert BLINDED in kinds
    assert composed_flags(m).can_aim is False


# ---------------------------------------------------------------------------
# Poison DoT — exact per-tick amounts, threshold edge, zombie immunity
# ---------------------------------------------------------------------------
def test_poison_dose_exact_per_tick_amounts():
    """The heat row's idiom, pinned: applied = quantize_hp_delta(
    poison_dps x (density_q / 65536) x (1 / tps)) — round-half-away onto
    the Q16.16 grid. Full density at the standard values (dps 6, 24 tps)
    is EXACTLY 0.25 HP/tick (16384 counts): 6 HP/s."""
    tps = int(CFG.clock.ticks_per_second)
    dps = float(CFG.exchange.poison_dps)

    for density in (1.0, 0.5):
        gmap = _room()
        u = Unit("M", x=8, y=8, team=0)
        u.id = 1
        u.current_hp = 1000.0
        q = _fill_footprint(gmap.gas[GAS_POISON], u, density)
        events = []
        apply_poison_dose([u], gmap, tps, events=events)
        expected = unit_fixed.quantize_hp_delta(
            dps * (q / 65536.0) * (1.0 / tps))
        assert u.current_hp == 1000.0 - expected
        hits = [e for e in events if isinstance(e, UnitHitEvent)]
        assert len(hits) == 1
        assert hits[0].damage == expected
        assert hits[0].source == "poison_gas"

    # The standard-value anchor: full density = exactly 0.25 HP per tick.
    assert unit_fixed.quantize_hp_delta(dps * 1.0 * (1.0 / tps)) == 0.25

    # Three ticks accumulate exactly 3x (integer-grid deltas sum exactly).
    gmap = _room()
    u = Unit("M", x=8, y=8, team=0)
    u.current_hp = 1000.0
    _fill_footprint(gmap.gas[GAS_POISON], u, 1.0)
    for _ in range(3):
        apply_poison_dose([u], gmap, tps, events=None)
    assert u.current_hp == 1000.0 - 3 * 0.25


def test_poison_threshold_edge():
    """One count below quantize(poison_min_density): NOTHING (no packet, no
    event). At the threshold exactly: the packet lands."""
    tps = int(CFG.clock.ticks_per_second)
    thresh_q = gas_fixed.quantize_scalar(float(CFG.exchange.poison_min_density))

    gmap = _room()
    u = Unit("M", x=8, y=8, team=0)
    u.current_hp = 1000.0
    for (tx, ty) in u.occupied_tiles():
        gmap.gas[GAS_POISON][ty, tx] = thresh_q - 1
    events = []
    apply_poison_dose([u], gmap, tps, events=events)
    assert u.current_hp == 1000.0 and events == []

    for (tx, ty) in u.occupied_tiles():
        gmap.gas[GAS_POISON][ty, tx] = thresh_q
    apply_poison_dose([u], gmap, tps, events=events)
    expected = unit_fixed.quantize_hp_delta(
        float(CFG.exchange.poison_dps) * (thresh_q / 65536.0) * (1.0 / tps))
    assert expected > 0
    assert u.current_hp == 1000.0 - expected
    assert len(events) == 1


def test_zombie_poison_immunity_zero_damage_no_events():
    """resist_mult[POISON] = 0 on the zombie overlay: a zombie parked in
    FULL-density poison takes 0 damage and draws NO packet at all (lazy
    emission — no 0-damage event spam on a horde standing in gas)."""
    tps = int(CFG.clock.ticks_per_second)
    gmap = _room()
    z = Unit("Z", x=8, y=8, team=1)
    z.id = 7
    z.current_hp = 400.0
    _fill_footprint(gmap.gas[GAS_POISON], z, 1.0)
    events = []
    for _ in range(10):
        apply_poison_dose([z], gmap, tps, events=events)
    assert z.current_hp == 400.0
    assert events == []
    # The marine control in the SAME cloud takes the exact dose.
    m = Unit("M", x=8, y=8, team=0)
    m.current_hp = 1000.0
    apply_poison_dose([m], gmap, tps, events=events)
    assert m.current_hp == 1000.0 - 0.25


def test_poison_drains_a_marine_through_the_simulation():
    """Conductor wiring (9c3) e2e: a marine in a poison cloud loses hp tick
    over tick through the pipeline; a zombie twin is untouched."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=8, y=8, team=0)
    z = Unit("Z", x=14, y=14, team=1)
    sim.add_unit(m)
    sim.add_unit(z)
    hp_m, hp_z = m.current_hp, z.current_hp
    _fill_footprint(sim.gmap.gas[GAS_POISON], m, 1.0)
    _fill_footprint(sim.gmap.gas[GAS_POISON], z, 1.0)
    for _ in range(5):
        sim.set_paused(False)
        sim.step()
    assert m.current_hp < hp_m                      # the dose lands
    assert z.current_hp == hp_z                     # they don't breathe
    poison_hits = [e for e in sim.tick_events if isinstance(e, UnitHitEvent)
                   and e.source == "poison_gas"]
    assert poison_hits and all(h.unit_id == m.id for h in poison_hits)


def test_gas_rows_are_dormant_and_draw_no_rng():
    """Zero gas anywhere: the rows early-out on one integer .any() — no
    status, no packet, no hp movement. And structurally: neither row even
    TAKES a generator (no RNG by construction)."""
    import inspect
    assert "rng" not in inspect.signature(apply_teargas_blind).parameters
    assert "rng" not in inspect.signature(apply_poison_dose).parameters
    gmap = _room()
    u = Unit("M", x=8, y=8, team=0)
    u.current_hp = 123.0
    apply_teargas_blind([u], gmap)
    apply_poison_dose([u], gmap, 24, events=None)
    assert u.current_hp == 123.0
    assert not getattr(u, "statuses", [])


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
