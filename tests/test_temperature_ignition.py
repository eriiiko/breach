"""Ignition from temperature (engine/06 §4 "Ignition", proposal §6 step 4b).

The READ side of the temperature substrate. The C++ ``TemperatureSolver``
(convert -> conduction -> cooling) fills the Q16.16 ``temperature`` field each
tick; this consumer ignites a FLAMMABLE tile when its temperature crosses the
per-material (Q16.16-quantized) ``ignition_temp`` AND oxygen is present (the
same air-side-neighbour O2 check the existing fire uses). EDGE-TRIGGERED (Fable
ruling 2026-07-24): an ARMED tile seeds ``fire = ignition_seed`` ONCE, disarms,
and re-arms only when it cools below ``ignition_temp`` — the ``max``-floor is
gone, so a lit tile (``I > 0``) is never touched (the zombie-smolder fix).

This module exercises the consumer directly (temperature / atmosphere injected
on a small synthetic grid — no renderer, no ray pass) and asserts that a normal
``Simulation.step()`` leaves ``temperature`` ~0 / ``fire`` unchanged (the path is
DORMANT in-game until a heat source is wired, the intended safe seam).

Verifies:
  - a flammable tile at/over ``ignition_temp`` WITH O2 ignites (fire >= I_seed);
  - the SAME tile WITHOUT O2 (vacuum / low-atmosphere neighbours) does NOT;
  - a NON-flammable tile never ignites, even when red-hot;
  - ignition never modifies a lit tile (``I > 0``) — neither lowering a bigger
    fire nor re-flooring a smaller/declining one (the edge-trigger, no max-floor);
  - the edge-trigger arm/disarm/re-arm cycle (armed -> seed -> disarm -> cool ->
    re-arm -> re-seed);
  - below-threshold temperature does nothing;
  - determinism: same inputs -> bit-identical ``fire``;
  - dormancy: a full ``Simulation.step()`` on a normal level leaves
    ``temperature`` ~0 and ``fire`` unchanged (no heat sources -> no ignition).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_temperature_ignition.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.combat import apply_temperature_ignition  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_HULL, MAT_WOOD,
    MaterialTable,
)
from simulation import fire_fixed  # noqa: E402  S3a: gmap.fire is int32 Q16.16
from simulation.unit import Unit  # noqa: E402

# Q16.16 scale — must match the `temperature` field and the material table's
# load-time ignition_temp quantization (TEMP_SCALE == HEAT_SCALE).
TEMP_SCALE = 65536

_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])   # 300 * 65536
IGN_SEED = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
O2_THRESHOLD = float(getattr(CFG.physics.fire, "o2_threshold", 0.60))
# S3a: gmap.fire is int32 Q16.16. The seed intensity quantizes once so the
# `fire >= I_seed` / `fire == I_seed` assertions compare integer counts.
IGN_SEED_Q = fire_fixed.quantize_scalar(IGN_SEED)


class _GasTableStub:
    """Minimal stand-in for GameMap's real GasTable — the continuous-O2 law
    (docs/continuous_o2_law_design_2026-07-24.md §2.4) reads BOTH `o2` and
    `inert_n2` (the mole-fraction denominator N_total = O2 + inert_N2)."""

    def __init__(self):
        from simulation.gases import O2, INERT_N2
        self.name_to_id = {"o2": O2, "inert_n2": INERT_N2}


class _GMapStub:
    """Minimal gmap carrying exactly the fields ``apply_temperature_ignition``
    reads: a 3x3 grid with a chosen material in the CENTRE tile and AIR around
    it (the air ring is the O2 source for a flammable centre). The air ring's
    REAL O2 (EOS refactor P4 — was ``atmosphere``) defaults to 1.0 (ample O2);
    pass ``atm`` to override (e.g. 0 for a vacuum-surrounded tile). Keeps the
    consumer tests fast and geometry-free, mirroring the _HeatStub pattern in
    test_unit_heat_damage.py."""

    def __init__(self, centre_mat=MAT_WOOD, atm=1.0):
        from simulation.gases import N_GASES, O2, INERT_N2
        self.materials = _TBL
        self.gases = _GasTableStub()
        m = np.full((3, 3), MAT_AIR, dtype=np.int8)
        m[1, 1] = centre_mat
        self.material = m
        self.flammable = _TBL.flammable[m]
        self.solid = (_TBL.permeability[m] <= 0.0)
        # Air ring carries a REAL O2/inert-N2 composition; solid tiles hold no gas
        # (== GameMap). The continuous-O2 law gates on the MOLE FRACTION X =
        # O2/(O2+inert_N2), so `atm` names the ring's O2 fraction: O2 = atm, N2 =
        # 1-atm at unit density -> X = atm exactly. (`atm` kept as the param name
        # for minimal diff against the callers below, all of which pass an
        # O2_THRESHOLD-relative fraction.)
        from simulation import atmosphere_fixed as _fx
        o2 = np.where(self.solid, 0, _fx.quantize_scalar(float(atm))).astype(np.int32)
        n2 = np.where(self.solid, 0,
                      _fx.quantize_scalar(max(0.0, 1.0 - float(atm)))).astype(np.int32)
        self.gas = np.zeros((N_GASES, 3, 3), dtype=np.int32)
        self.gas[O2] = o2
        self.gas[INERT_N2] = n2
        # S3a: the O2 mask excludes vacuum neighbours (matching the C++ fire P
        # gate). Here the "no O2" case is modelled as low-O2 air (atm=0), not
        # flagged vacuum — so is_vacuum is all-False and the air ring still
        # counts (mean -> 0 -> below threshold), preserving the prior behaviour.
        self.is_vacuum = np.zeros((3, 3), dtype=bool)
        self.temperature = np.zeros((3, 3), dtype=np.int32)
        # S3a: gmap.fire is int32 Q16.16.
        self.fire = np.zeros((3, 3), dtype=np.int32)
        # o2-law: ignition needs fuel (wall_hp > 0). Q16.16 HP, 100 everywhere —
        # only the flammable centre's value is consulted (air isn't flammable).
        self.wall_hp = np.full((3, 3), 100 * TEMP_SCALE, dtype=np.int32)
        # Edge-trigger arm (Fable 2026-07-24): start ARMED so the unlit centre
        # seeds on its first hot+O2 tick; disarms once lit, re-arms once cooled.
        self.ignition_armed = np.ones((3, 3), dtype=bool)


def _ignite(gmap):
    apply_temperature_ignition(gmap, O2_THRESHOLD, IGN_SEED)


# ---------------------------------------------------------------------------
# Consumer tests (temperature / atmosphere injected directly)
# ---------------------------------------------------------------------------
def test_quantized_threshold_matches_field_domain():
    # Guard the load-time quantization the consumer compares against: wood's
    # ignition_temp (300) quantized into the Q16.16 `temperature` domain.
    assert IGN_WOOD_Q16 == 300 * TEMP_SCALE
    assert int(_TBL.ignition_temp_q16[MAT_HULL]) == 0  # non-flammable -> 0


def test_flammable_hot_with_oxygen_ignites():
    # A flammable (wood) tile at exactly its threshold, with O2 -> fire >= I_seed.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16
    assert g.fire[1, 1] == 0
    _ignite(g)
    assert g.fire[1, 1] >= IGN_SEED_Q
    assert g.fire[1, 1] == IGN_SEED_Q


def test_above_threshold_also_ignites():
    # Well over the threshold ignites just the same (the test is `>=`).
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16 * 4
    _ignite(g)
    assert g.fire[1, 1] >= IGN_SEED_Q


def test_hot_without_oxygen_does_not_ignite():
    # The SAME hot flammable tile, but its air neighbours are vacuum (atmosphere
    # below o2_threshold) -> the O2 check fails -> no ignition.
    g = _GMapStub(MAT_WOOD, atm=0.0)   # air ring at 0 atmosphere (vacuum)
    g.temperature[1, 1] = IGN_WOOD_Q16 * 4
    _ignite(g)
    assert g.fire[1, 1] == 0.0, "ignited in vacuum (O2 check failed to gate)"


def test_low_atmosphere_below_threshold_does_not_ignite():
    # Just below the O2 threshold -> still no ignition (boundary of the predicate).
    g = _GMapStub(MAT_WOOD, atm=O2_THRESHOLD - 0.05)
    g.temperature[1, 1] = IGN_WOOD_Q16 * 4
    _ignite(g)
    assert g.fire[1, 1] == 0.0


def test_nonflammable_hot_never_ignites():
    # A non-flammable (hull) tile, red-hot, with plenty of O2 -> never ignites.
    # Hull's ignition_temp_q16 is 0, so the threshold compare alone would pass;
    # the flammable gate is what (correctly) stops it.
    g = _GMapStub(MAT_HULL, atm=1.0)
    g.temperature[1, 1] = 1 << 28        # extremely hot
    _ignite(g)
    assert g.fire[1, 1] == 0.0, "non-flammable tile ignited"


def test_below_threshold_does_nothing():
    # Hot but one count BELOW the quantized threshold -> no ignition.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16 - 1
    _ignite(g)
    assert g.fire[1, 1] == 0.0


def test_zero_temperature_does_nothing():
    # The dormant case in miniature: temperature 0 everywhere -> no ignition.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    _ignite(g)
    assert float(g.fire.max()) == 0.0


def test_max_does_not_lower_existing_fire():
    # An existing fire larger than I_seed (e.g. set by an explosion) is preserved:
    # ignition uses max(fire, I_seed), never assign.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16
    big_q = fire_fixed.quantize_scalar(0.8)   # S3a: Q16.16
    assert big_q > IGN_SEED_Q
    g.fire[1, 1] = big_q
    _ignite(g)
    assert g.fire[1, 1] == big_q, "max() lowered a bigger existing fire"


def test_declining_fire_below_seed_is_not_refloored():
    # EDGE-TRIGGER (Fable ruling 2026-07-24): a flickering fire BELOW I_seed is
    # NOT raised to I_seed. Temperature ignition never modifies a tile with I > 0
    # (the `fire == 0` gate) — this REMOVES the old max-floor that re-seeded a
    # dying fire every hot tick (the eternal 0.1 zombie smolder). The fire logistic
    # alone now decides a lit tile's fate.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16
    small_q = fire_fixed.quantize_scalar(IGN_SEED / 2.0)   # S3a: Q16.16
    g.fire[1, 1] = small_q
    _ignite(g)
    assert g.fire[1, 1] == small_q, "temperature ignition re-floored a declining fire"


# ---------------------------------------------------------------------------
# Edge-trigger arm / disarm / re-arm (Fable ruling 2026-07-24 — zombie-smolder fix)
# ---------------------------------------------------------------------------
def test_armed_hot_tile_seeds_then_disarms():
    # (1) armed -> seed -> disarm: an ARMED, hot, oxygenated, fuelled, UNLIT tile
    # ignites ONCE to exactly I_seed and then DISARMS (so it can't re-seed).
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16
    assert bool(g.ignition_armed[1, 1])          # armed at the start
    assert g.fire[1, 1] == 0
    _ignite(g)
    assert g.fire[1, 1] == IGN_SEED_Q            # seeded to exactly I_seed
    assert not bool(g.ignition_armed[1, 1])      # and DISARMED


def test_disarmed_declining_fire_not_refloored():
    # (2) a declining fire is NOT re-floored: a DISARMED tile whose fire is below
    # I_seed is left untouched (no max-floor), and stays disarmed while hot.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16
    g.ignition_armed[1, 1] = False               # already spent its arm this episode
    declining = fire_fixed.quantize_scalar(IGN_SEED * 0.4)
    g.fire[1, 1] = declining
    _ignite(g)
    assert g.fire[1, 1] == declining, "re-floored a declining fire"
    assert not bool(g.ignition_armed[1, 1]), "re-armed while still hot"


def test_rearm_only_after_cooling():
    # (3) re-arm on cool: a disarmed tile re-arms the moment it drops below
    # ignition_temp — and NOT while it is still hot.
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.ignition_armed[1, 1] = False
    g.temperature[1, 1] = IGN_WOOD_Q16           # still hot
    _ignite(g)
    assert not bool(g.ignition_armed[1, 1]), "re-armed while still hot"
    g.temperature[1, 1] = IGN_WOOD_Q16 - 1       # cooled just below threshold
    _ignite(g)
    assert bool(g.ignition_armed[1, 1]), "did not re-arm after cooling"


def test_full_ignite_die_cool_reignite_cycle():
    # (4) the whole hysteresis loop at the unit level (the physics is emulated by
    # direct fire/temperature writes between calls, since apply_temperature_
    # ignition is the unit under test): seed -> grow/decline (never re-floored) ->
    # snap to 0 while hot (NOT re-seeded — the smolder fix) -> cool (re-arm) ->
    # re-heat (re-seed a fresh fire).
    g = _GMapStub(MAT_WOOD, atm=1.0)
    g.temperature[1, 1] = IGN_WOOD_Q16

    # arm -> seed -> disarm
    _ignite(g)
    assert g.fire[1, 1] == IGN_SEED_Q and not bool(g.ignition_armed[1, 1])

    # peak then decline: a lit tile (I>0) is never touched while disarmed
    g.fire[1, 1] = fire_fixed.quantize_scalar(0.5)
    _ignite(g)
    assert g.fire[1, 1] == fire_fixed.quantize_scalar(0.5)
    g.fire[1, 1] = fire_fixed.quantize_scalar(IGN_SEED * 0.3)
    _ignite(g)
    assert g.fire[1, 1] == fire_fixed.quantize_scalar(IGN_SEED * 0.3)

    # the logistic snaps I to 0 while the tile is STILL HOT -> NOT re-seeded
    g.fire[1, 1] = 0
    _ignite(g)
    assert g.fire[1, 1] == 0, "re-seeded a dead-but-still-hot tile (the smolder bug)"
    assert not bool(g.ignition_armed[1, 1])

    # the tile finally cools -> re-arms; still 0 (cold => no seed)
    g.temperature[1, 1] = IGN_WOOD_Q16 - 1
    _ignite(g)
    assert bool(g.ignition_armed[1, 1]) and g.fire[1, 1] == 0

    # a neighbour re-heats it above threshold -> a fresh fire seeds, disarms again
    g.temperature[1, 1] = IGN_WOOD_Q16
    _ignite(g)
    assert g.fire[1, 1] == IGN_SEED_Q and not bool(g.ignition_armed[1, 1])


def test_deterministic_same_inputs_bit_identical():
    # Same inputs -> bit-identical `fire` across two independent runs.
    rng = np.random.default_rng(7)

    def run():
        g = _GMapStub(MAT_WOOD, atm=1.0)
        # A spread of temperatures around the threshold on the (single) flammable
        # tile across repeated calls would only ever touch [1,1]; instead drive a
        # range of seeds deterministically through the single tile per run.
        g.temperature[1, 1] = IGN_WOOD_Q16 + int(rng.integers(0, 10))
        _ignite(g)
        return g.fire.copy()

    rng = np.random.default_rng(7)
    a = run()
    rng = np.random.default_rng(7)
    b = run()
    assert np.array_equal(a, b), "ignition is not deterministic"


# ---------------------------------------------------------------------------
# Dormancy: a normal Simulation.step() must not ignite (no heat sources yet)
# ---------------------------------------------------------------------------
def test_full_step_is_dormant_temperature_and_fire_unchanged():
    """On a normal level with no heat source, a full Simulation.step() leaves
    `temperature` ~0 and `fire` unchanged — the ignition path is present but
    DORMANT (no behaviour change), the intended safe seam."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    m = Unit("M1", x=14, y=50, team=0)
    sim.add_unit(m)
    sim.set_paused(False)

    fire_before = sim.gmap.fire.copy()

    # Run several ticks; with no heat deposited, `temperature` must stay 0 and no
    # tile may ignite from temperature.
    for _ in range(5):
        sim.step()

    assert int(sim.gmap.temperature.max()) == 0, (
        "temperature became non-zero with no heat source (substrate not dormant)"
    )
    assert np.array_equal(sim.gmap.fire, fire_before), (
        "fire changed on a normal tick (ignition path is not dormant)"
    )
