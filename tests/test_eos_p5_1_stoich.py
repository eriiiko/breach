"""EOS refactor P5.1 — stoichiometric fuel consumption (docs/
eos_refactor_design.md §5 v2.5 amendment, decisions log #17).

CombustionSolver now takes `wall_hp` MUTABLE: per neighbour burn the SOURCE
tile pays `fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, burn))` — the
same unbiased round-to-nearest sink idiom as fire_simulation.cpp's
wall_damage depletion — FLOORED at FUEL_FLOOR == 1 Q16.16 LSB after each of
the up-to-4 neighbour subtractions (N,S,W,E order). The no-fuel gate moved
from `wall_hp <= 0` to `wall_hp <= 1 LSB`: a fully-charred tile's ember is
OUT (no O2 draw, no heat deposit). THE 1-LSB RULE (Erik, 2026-07-11): this
pass NEVER destroys a tile — structural destruction stays exclusively
FireSimulation's I>0 wall_damage path.

Gate tiers (the v2.5 block's gate list):

1. Unit tests, on CombustionSolver.step directly: fuel decrement EXACT
   (integer-replicated in Python, including the N,S,W,E fold order and the
   per-subtraction floor) and deterministic (bit-identical across two
   runs); the 1-LSB floor never crossed; no destruction originates from
   combustion; a charred tile (hp == 1) burns nothing and deposits nothing.
2. The LIFECYCLE E2E — Erik's original fire-lifecycle vision end to end:
   ignite -> O2-starve (flame I dies) -> ember persists (T >= ignition,
   I = 0, wall_hp draining) -> O2 inflow re-ignites a proper flame (I > 0,
   FLAME-scale wall_damage resumes) -> seal again -> char-out at the floor,
   ember extinguishes (no further O2 draw), wall stands at exactly 1 LSB —
   and one hit destroys it.

The lifecycle E2E runs the GAME loop faithfully: PhysicsRunner.step +
apply_temperature_ignition + the per-tick heat clear, in Simulation.step's
order (simulation.py: physics -> ignition -> heat.fill(0)). The v2.4 P4
E2Es predate that and deliberately omit the ignition call; the ember story
NEEDS it (re-ignition IS the ignition path).

Staging notes (all deterministic, no RNG, established test idioms):
- The EMBER WINDOW is real engine physics worth naming: combustion burns a
  neighbour SITE when its O2 > o2_thresh_burn (0.03), while ignition seeds
  a flame from the MEAN over the 4 open neighbours (o2_threshold 0.01). One
  site at 0.035 with the other three at 0 gives mean 0.00875 < 0.01: the
  ember draws O2 with NO flame — the exact (I=0, T>=ign, hp draining)
  emergent state decisions #17 describes. The test drips O2 through that
  window on purpose (and re-zeroes the other three sites each tick so the
  mean stays pinned below the ignition gate — deterministic staging, not a
  physics change).
- O2 boundary conditions (the starve, the inflow, the seal) are direct
  gas-plane edits — the e2e_3 tank-rupture / e2e_4 flood idiom from
  test_eos_p4_combustion.py. Waiting for a sealed room to spend its own O2
  takes thousands of ticks (the designed sealed-smolder regime).
- The char-out phase cranks `fuel_per_o2` (0.7 -> 45.0). That dial is THE
  ember-lifetime dial by design ("larger -> they char out fast", design §5
  v2.5 / §9); the crank compresses a multi-thousand-tick ember into test
  time without touching any other physics. It is re-quantized per step in
  C++, so a mid-run change is well-defined and deterministic.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_eos_p5_1_stoich.py -q
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                     # noqa: E402
from simulation.combat import apply_temperature_ignition, chew_wall  # noqa: E402
from simulation.gases import O2, INERT_N2, BLACK_SMOKE           # noqa: E402
from simulation.materials import MAT_AIR, MAT_WOOD, MaterialTable  # noqa: E402
from simulation import fire_fixed, gas_fixed                     # noqa: E402

# Reuse the P4 file's scenario builders / runner / ignite helpers (same
# conventions, one source of truth for the sealed-room fixture).
sys.path.insert(0, str(ROOT / "tests"))
from test_eos_p4_combustion import (                             # noqa: E402
    SEED_TICK_DT, _sealed_room, _runner, _ignite, IGN_WOOD_Q16)

FP_ONE = 65536
FUEL_FLOOR = 1          # combustion.h FUEL_FLOOR — one Q16.16 LSB
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))   # combustion.cpp D4 (N, S, W, E)


def _q(v):
    """Python twin of fixed_point.h quantize(): round-half-away-from-zero."""
    s = float(v) * FP_ONE
    return int(s + 0.5) if s >= 0.0 else int(s - 0.5)


def _narrow_round(wide):
    """Python twin of fixed_point.h narrow_round(): (wide + 0.5ulp) >> 16."""
    return (wide + (1 << 15)) >> 16


# ---------------------------------------------------------------------------
# Direct-solver fixture (the tier-1 idiom from test_eos_p4_combustion.py)
# ---------------------------------------------------------------------------
def _solver_scene(h=7, w=7, hp=60.0, n_o2=(0.21, 0.21, 0.21, 0.21)):
    """One flammable solid tile at the centre, hot, 4 open neighbours with
    per-site O2 (N, S, W, E order), ambient N2 everywhere open."""
    gas = np.zeros((7, h, w), dtype=np.int32)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    ign = np.zeros((h, w), dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)

    cy, cx = h // 2, w // 2
    solid[cy, cx] = True
    flammable[cy, cx] = True
    wall_hp[cy, cx] = _q(hp) if hp >= 1.0 / FP_ONE else int(round(hp * FP_ONE))
    ign[cy, cx] = IGN_WOOD_Q16
    temperature[cy, cx] = IGN_WOOD_Q16 * 2
    gas[INERT_N2][~solid] = _q(0.79)
    for (dy, dx), o2v in zip(N4, n_o2):
        gas[O2][cy + dy, cx + dx] = _q(o2v)
    return gas, solid, is_vacuum, flammable, wall_hp, fire, ign, temperature, (cy, cx)


def _step(comb, scene, dt=0.25):
    gas, solid, is_vacuum, flammable, wall_hp, fire, ign, temperature, _ = scene
    comb.step(gas, O2, INERT_N2, BLACK_SMOKE, temperature, wall_hp, fire,
              flammable, solid, is_vacuum, ign, dt, 1.0, 0.05)


# ---------------------------------------------------------------------------
# TIER 1 — unit gates on the new consumption
# ---------------------------------------------------------------------------
def test_fuel_decrement_exact_and_deterministic():
    """The decrement is the EXACT integer transaction the v2.5 spec wrote:
    per open neighbour j (N,S,W,E order), burn = min(burn_rate*dt, O2[j])
    and the source tile pays narrow_round(fuel_per_o2_q * burn) — replicated
    here in pure Python ints, with DISTINCT per-site O2 so the fold order is
    pinned too. Then: two scratch-built runs of a 20-step evolving scenario
    hash bit-identically (the determinism half of the gate)."""
    comb = bp.CombustionSolver()
    comb.burn_rate = 1.0
    comb.o2_thresh_burn = 0.03
    comb.fuel_per_o2 = 0.7
    dt = 0.25

    n_o2 = (0.21, 0.05, 0.035, 0.02)   # N, S, W burn; E is below the gate
    scene = _solver_scene(n_o2=n_o2)
    wall_hp = scene[4]
    cy, cx = scene[8]
    hp0 = int(wall_hp[cy, cx])

    # Python-int replication of the C++ arithmetic (fixed_point.h twins).
    cap_q = _q(1.0 * dt)
    thresh_q = _q(0.03)
    fuel_q = _q(0.7)
    expected = hp0
    for o2v in n_o2:
        o2q = _q(o2v)
        if o2q <= thresh_q:
            continue
        burn = min(cap_q, o2q)
        expected -= _narrow_round(fuel_q * burn)
        expected = max(expected, FUEL_FLOOR)
    _step(comb, scene, dt)
    assert int(wall_hp[cy, cx]) == expected, (
        f"fuel decrement not exact: {int(wall_hp[cy, cx])} != {expected} "
        f"(start {hp0})")
    assert expected < hp0, "vacuous: nothing burned"

    # Determinism: two scratch builds of a 20-step evolving run, bit-equal.
    def _digest():
        c = bp.CombustionSolver()
        c.burn_rate = 2.0
        c.fuel_per_o2 = 0.7
        sc = _solver_scene(n_o2=(0.21, 0.18, 0.09, 0.04))
        h = hashlib.sha256()
        for _ in range(20):
            _step(c, sc, dt=1.0 / 24.0)
            for arr in (sc[0], sc[4], sc[7]):   # gas, wall_hp, temperature
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    assert _digest() == _digest(), "two identical runs diverged"


def test_one_lsb_floor_never_crossed():
    """The floor is applied after EACH neighbour subtraction: with fuel to
    cover only part of the first burn, wall_hp lands exactly at FUEL_FLOOR
    and never below — while the REMAINING neighbours still burn their O2
    this tick (the gate is per-tile at loop entry, per the v2.5 spec: the
    floor caps the fuel store, it does not truncate the tick)."""
    comb = bp.CombustionSolver()
    comb.burn_rate = 4.0
    comb.fuel_per_o2 = 50.0    # one burn costs far more than the fuel left
    scene = _solver_scene(hp=0.5, n_o2=(0.21, 0.21, 0.21, 0.21))
    gas, wall_hp = scene[0], scene[4]
    cy, cx = scene[8]
    o2_before = [int(gas[O2][cy + dy, cx + dx]) for dy, dx in N4]

    _step(comb, scene, dt=0.25)
    assert int(wall_hp[cy, cx]) == FUEL_FLOOR, (
        f"floor missed: wall_hp == {int(wall_hp[cy, cx])}, expected exactly "
        f"{FUEL_FLOOR}")
    o2_after = [int(gas[O2][cy + dy, cx + dx]) for dy, dx in N4]
    assert all(a < b for a, b in zip(o2_after, o2_before)), (
        "a neighbour burn was skipped mid-tick — the v2.5 floor semantics "
        "(per-tile gate at loop entry, floor after each subtraction) changed")

    # And repeated ticks can never take it below the floor (it is now
    # charred: the hp <= FUEL_FLOOR gate skips it entirely).
    for _ in range(50):
        _step(comb, scene, dt=0.25)
    assert int(wall_hp[cy, cx]) == FUEL_FLOOR


def test_no_destruction_originates_from_combustion():
    """Aggressive dials, hundreds of ticks, O2 refilled every tick: the wall
    chars to the floor and STAYS a wall — hp never <= 0 (the destruction
    predicate FireSimulation and the structural paths share), masks
    untouched. Combustion has no destroyed-tile channel at all (its binding
    returns None); this pins the invariant the design names: structural
    destruction remains exclusively FireSimulation's I>0 path."""
    comb = bp.CombustionSolver()
    comb.burn_rate = 8.0
    comb.fuel_per_o2 = 25.0
    scene = _solver_scene(hp=60.0)
    gas, solid, flammable, wall_hp = scene[0], scene[1], scene[3], scene[4]
    cy, cx = scene[8]

    for _ in range(400):
        for dy, dx in N4:                       # a fresh oxygen bath each tick
            gas[O2][cy + dy, cx + dx] = _q(0.21)
        _step(comb, scene, dt=0.25)
        assert int(wall_hp[cy, cx]) >= FUEL_FLOOR, (
            f"combustion took wall_hp to {int(wall_hp[cy, cx])} — below the "
            "1-LSB floor (destruction territory)")
    assert int(wall_hp[cy, cx]) == FUEL_FLOOR, "vacuous: never charred out"
    assert bool(solid[cy, cx]) and bool(flammable[cy, cx])


def test_charred_tile_burns_nothing_deposits_nothing():
    """hp == 1 LSB is the no-fuel gate now (was hp <= 0): a fully-charred
    tile's ember is OUT — zero O2 draw, zero soot/N2 product, zero heat
    deposit, even under a rich oxygen bath (otherwise the perpetual-ember
    hole just re-opens one LSB lower — design §5 v2.5)."""
    comb = bp.CombustionSolver()
    comb.burn_rate = 8.0
    scene = _solver_scene(hp=1.0 / FP_ONE)      # exactly 1 raw count
    gas, temperature, wall_hp = scene[0], scene[7], scene[4]
    cy, cx = scene[8]
    assert int(wall_hp[cy, cx]) == FUEL_FLOOR   # fixture sanity

    gas_before = gas.copy()
    temp_before = temperature.copy()
    for _ in range(30):
        _step(comb, scene, dt=0.25)
    assert np.array_equal(gas, gas_before), "a charred tile drew/produced gas"
    assert np.array_equal(temperature, temp_before), (
        "a charred tile deposited heat")
    assert int(wall_hp[cy, cx]) == FUEL_FLOOR


def test_fuel_per_o2_config_plumbing():
    """config.toml [physics.combustion].fuel_per_o2 reaches the solver via
    PhysicsRunner the same way the other combustion dials do."""
    pr = _runner()
    # (float32 member on the C++ struct — compare at float32 precision)
    assert abs(pr.combustion.fuel_per_o2 - 0.7) < 1e-6, (
        f"fuel_per_o2 not plumbed from config (got {pr.combustion.fuel_per_o2})")


# ---------------------------------------------------------------------------
# TIER 2 — the lifecycle E2E (Erik's fire-lifecycle vision, decisions #17)
# ---------------------------------------------------------------------------
def test_lifecycle_ember_reignite_charout():
    """ignite -> O2-starve (flame dies) -> ember persists (T >= ignition,
    I = 0, wall_hp draining) -> O2 inflow re-ignites a proper flame ->
    seal -> char-out at the 1-LSB floor, ember extinguishes (no further O2
    draw), the wall STANDS — and one hit destroys it. Game-faithful loop
    (physics -> temperature-ignition -> heat clear, Simulation.step's
    order); staging idioms per the module docstring."""
    C = 4
    gmap = _sealed_room(hh=9, wood_at=(C, C))
    pr = _runner()
    _ignite(gmap, (C, C), intensity=0.6, temp_mult=1.5)
    open_cells = (~gmap.solid) & (~gmap.is_vacuum)

    destroyed = []

    def tick():
        destroyed.extend(pr.step(gmap, SEED_TICK_DT) or [])
        # The game loop's ignition step (simulation.py: physics -> ignition
        # -> heat clear), with config.toml [physics.fire]'s shipped values.
        apply_temperature_ignition(gmap, o2_threshold=0.01, ignition_seed=0.1)
        gmap.heat.fill(0)

    def drip(o2v=0.035):
        """Feed the EMBER WINDOW: one site above o2_thresh_burn (0.03),
        sisters re-zeroed so the ignition MEAN stays pinned < 0.01."""
        gmap.gas[O2][C - 1, C] = gas_fixed.quantize_scalar(o2v)
        for dy, dx in ((1, 0), (0, -1), (0, 1)):
            gmap.gas[O2][C + dy, C + dx] = 0

    # ---- Phase A: a real flame on ambient O2 ------------------------------
    for _ in range(30):
        tick()
    assert int(gmap.fire[C, C]) > 0, "phase A: the seeded flame died too soon"
    hp_a = int(gmap.wall_hp[C, C])

    # ---- O2-starve: the room's oxygen is spent/vented (boundary edit) -----
    gmap.gas[O2][open_cells] = 0
    for k in range(80):
        tick()
        if int(gmap.fire[C, C]) == 0:
            break
    assert int(gmap.fire[C, C]) == 0, "phase A: flame never starved"
    assert int(gmap.temperature[C, C]) >= IGN_WOOD_Q16, (
        "phase A: tile cooled below ignition at flame death — no ember to test")
    assert int(gmap.wall_hp[C, C]) > FUEL_FLOOR, "phase A: no fuel left (bad scenario)"

    # ---- Phase B: the EMBER — I stays 0, T >= ignition, fuel draining -----
    hp_b0 = int(gmap.wall_hp[C, C])
    for _ in range(40):
        drip()
        tick()
        assert int(gmap.fire[C, C]) == 0, (
            "phase B: the ember window leaked a flame (mean-O2 gate crossed)")
        assert int(gmap.temperature[C, C]) >= IGN_WOOD_Q16, (
            "phase B: ember went cold mid-phase")
    hp_b1 = int(gmap.wall_hp[C, C])
    assert hp_b1 < hp_b0, (
        "phase B: ember drew O2 without consuming fuel — the v2.5 "
        "stoichiometric drain is not engaging at ember scale")

    # ---- Phase C: O2 inflow -> a PROPER flame (ignition path) -------------
    for dy, dx in N4:
        gmap.gas[O2][C + dy, C + dx] = gas_fixed.quantize_scalar(0.3)
    hp_c0 = int(gmap.wall_hp[C, C])
    i_peak = 0
    for _ in range(25):
        tick()
        i_peak = max(i_peak, int(gmap.fire[C, C]))
    seed_q = fire_fixed.quantize_scalar(0.1)
    assert i_peak >= seed_q, "phase C: O2 inflow did not re-ignite the ember"
    assert i_peak > seed_q, (
        "phase C: the flame never GREW past its ignition seed — not a proper "
        "flame (the logistic should feed on 0.3 O2)")
    assert int(gmap.wall_hp[C, C]) < hp_c0, (
        "phase C: no consumption during the re-lit flame (wall_damage + "
        "combustion both draw here)")

    # ---- Seal again: inflow over; the flame starves back to zero ----------
    gmap.gas[O2][open_cells] = 0
    for k in range(80):
        tick()
        if int(gmap.fire[C, C]) == 0:
            break
    assert int(gmap.fire[C, C]) == 0, "seal: flame 2 never starved"

    # ---- Phase D: char-out at the floor (ember-lifetime dial cranked) -----
    pr.combustion.fuel_per_o2 = 45.0   # THE ember-lifetime dial (design §9)
    for k in range(300):
        if int(gmap.wall_hp[C, C]) <= FUEL_FLOOR:
            break
        drip()
        tick()
        assert int(gmap.fire[C, C]) == 0, "phase D: char-out leaked a flame"
        assert int(gmap.wall_hp[C, C]) >= FUEL_FLOOR, (
            "phase D: combustion crossed the 1-LSB floor")
    assert int(gmap.wall_hp[C, C]) == FUEL_FLOOR, (
        f"phase D: never charred out (hp={int(gmap.wall_hp[C, C])})")
    assert destroyed == [], (
        f"destruction originated inside the burn phases: {destroyed}")

    # ---- Phase E: ember OUT — charred tile draws nothing ------------------
    # Bulk O2 transport is exactly conservative (P1 gate) and combustion is
    # its only sink, so with the boundary edits stopped the room's O2 total
    # is CONSTANT unless something burns. Keep the drip site fed: a charred
    # tile must not touch it.
    gmap.gas[O2][open_cells] = 0
    drip()
    o2_total0 = int(gmap.gas[O2].astype(np.int64).sum())
    for _ in range(40):
        tick()
        assert int(gmap.wall_hp[C, C]) == FUEL_FLOOR, "phase E: hp moved at the floor"
        assert int(gmap.fire[C, C]) == 0, "phase E: a flame lit on a charred tile"
    o2_total1 = int(gmap.gas[O2].astype(np.int64).sum())
    assert o2_total1 == o2_total0, (
        f"phase E: O2 was drawn after char-out ({o2_total0} -> {o2_total1}) — "
        "the hp <= 1 LSB gate is not extinguishing the ember")
    assert destroyed == []
    assert bool(gmap.solid[C, C]) and bool(gmap.flammable[C, C]), (
        "the charred wall should still STAND (1-LSB rule)")
    assert gmap.material[C, C] == MAT_WOOD

    # ---- Epilogue: charred tissue paper — one hit destroys it -------------
    chew_wall(gmap, C, C, 1)   # any structural damage source; 1 dmg >> 1 LSB
    assert gmap.material[C, C] == MAT_AIR and not bool(gmap.solid[C, C]), (
        "a 1-LSB charred wall should fall to a single hit (destroy_wall path)")


if __name__ == "__main__":
    test_fuel_decrement_exact_and_deterministic()
    test_one_lsb_floor_never_crossed()
    test_no_destruction_originates_from_combustion()
    test_charred_tile_burns_nothing_deposits_nothing()
    test_fuel_per_o2_config_plumbing()
    test_lifecycle_ember_reignite_charout()
    print("OK: EOS P5.1 stoichiometric-fuel tests passed")
