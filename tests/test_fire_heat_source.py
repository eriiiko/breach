"""K2 — fire as a sim-side heat ray source (proposal §1, fire_design_notes).

This is the keystone that ACTIVATES the whole temperature pipeline: each burning
tile (``fire > 0``) becomes a short-range heat :class:`LightSource` cast IN THE
SIM at the START of the physics step, BEFORE the :class:`TemperatureSolver`, so
this tick's fire heat converts to temperature this same tick. The cast deposits
ONLY into ``gmap.heat`` (Q16.16, saturating-add, occluded per-tile by
``heat_atten`` per K1); the render-side glow is a separate later step.

The downstream consumers (heat -> temperature convert -> conduction -> cooling ->
ignition + unit heat damage) were already wired but DORMANT (no sim heat source);
K2 is what lights them. This module exercises the FULL chain headless:

  (a) ``heat`` / ``temperature`` rise on nearby SOLID tiles (not on air);
  (b) heat is attenuated by ``heat_atten`` — a wall blocks the fire's heat
      beyond it; a clear path heats further (occlusion via the K1 channel);
  (c) a flammable wood wall held near a burning tile crosses ``ignition_temp``
      and IGNITES through a full ``Simulation.step()`` (Step E via fire heat);
  (d) a unit next to the fire loses HP to heat damage (Step D), and a zombie
      loses exactly ``zombie.fire_damage_multiplier`` (4x) a marine;
  (e) determinism — same seed/scene -> bit-identical ``temperature`` after N
      ticks (fixed ray count/angles, fixed source order, integer add, no RNG);
  (f) with the conservative ``k_fire_heat``, a lone fire does NOT instantly
      firestorm the map in a couple of ticks.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fire_heat_source.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402
from simulation.materials import (  # noqa: E402
    MaterialTable, MAT_AIR, MAT_WOOD, MAT_HULL,
)
from simulation.gases import GasTable, N_GASES  # noqa: E402
from simulation import fire_fixed  # noqa: E402  S3a: gmap.fire is int32 Q16.16
from simulation.unit import Unit  # noqa: E402

# S3a: gmap.fire is int32 Q16.16. Helpers for the real-Simulation tests that
# seed the field at real intensity.
FIRE_Q = fire_fixed.quantize_scalar
FIRE_001_Q = fire_fixed.quantize_scalar(0.01)

HEAT_SCALE = 65536          # Q16.16 (== TEMP_SCALE), shared heat/temperature domain
_TBL = MaterialTable.from_config()
# Per-gas optics table for the multi-gas march (engine/05 §6.2). Gases never
# attenuate the heat channel, so an empty gas field leaves the heat cast
# bit-identical to the pre-multigas single-smoke path — these tests stay valid.
_GASES = GasTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])   # 300 * 65536


# ---------------------------------------------------------------------------
# A bare scene carrying only the fields cast_fire_heat reads, for the
# deposit/occlusion/footprint probes (no level geometry, no renderer). The
# full-chain tests below use a real Simulation.
# ---------------------------------------------------------------------------
class _FireScene:
    """Synthetic (h, w) grid: choose materials, light one tile, cast heat."""

    def __init__(self, h, w):
        self.material = np.full((h, w), MAT_AIR, dtype=np.int8)
        # S3a: gmap.fire is int32 Q16.16 (cast_fire_heat dequantizes on read).
        self.fire = np.zeros((h, w), dtype=np.int32)
        self.heat = np.zeros((h, w), dtype=np.int32)
        # P-R4 (ruling A1 + amendment 5): the cast's three read planes and its
        # two outputs. `temperature` is now an INPUT — the exchange's magnitude
        # comes from the emitters' own temperature, not from a payload dial.
        self.temperature = np.zeros((h, w), dtype=np.int32)
        self.rad_net = np.zeros((h, w), dtype=np.int32)    # SIGNED energy ledger
        # P-F1a (v6.1 rule 4): the per-tile SKY ledger — the ONLY entry that
        # leaves the tile books. `rad_net.sum() + rad_amb.sum() == 0` exactly.
        self.rad_amb = np.zeros((h, w), dtype=np.int32)
        self.rad_flux = np.zeros((h, w), dtype=np.int32)   # D3 damage sensor
        self._h, self._w = h, w
        # Multi-gas march inputs (engine/05 §6.2): an empty gas field + the canon
        # per-gas tables. Gases do not attenuate heat, so this is inert here.
        self.gas = np.zeros((N_GASES, h, w), dtype=np.float32)
        self.gases = _GASES

    def set_wood(self, y, x):
        self.material[y, x] = MAT_WOOD

    def set_hull(self, y, x):
        self.material[y, x] = MAT_HULL

    def light(self, y, x, intensity, T_game=443.0):
        self.material[y, x] = MAT_WOOD       # fire only ever lives on flammable
        self.fire[y, x] = fire_fixed.quantize_scalar(float(intensity))  # S3a: Q16.16
        # P-R4: an emitter radiates against its OWN temperature. The retired
        # painter took a `k_fire_heat * I` payload instead and did not care what
        # the tile's temperature was; the net-T^4 exchange does. 443 game is the
        # blessed plateau the P-R5 tune targets.
        self.temperature[y, x] = fire_fixed.quantize_scalar(float(T_game))

    def _rebuild(self):
        m = self.material
        self.heat_atten = np.ascontiguousarray(_TBL.heat_atten[m], dtype=np.float32)
        self.dyn_light_atten = np.ascontiguousarray(
            _TBL.light_atten[m], dtype=np.float32)
        self.smoke = np.zeros((self._h, self._w), dtype=np.float32)
        # P-R4 per-tile thermal columns the exchange reads (limiter budget +
        # the warm-emitter mask), projected from the material table like the
        # optics columns above.
        self.heat_inv_shift = np.ascontiguousarray(
            _TBL.heat_inv_shift[m], dtype=np.int32)
        self.thermal_solid = np.ascontiguousarray(_TBL.thermal_solid[m], dtype=bool)

    def cast(self, runner, tick=0):
        """Run the tick's fire cast and return BOTH P-R4 outputs, in energy
        units: (rad_net, rad_flux).

        P-R4 (ruling A1): there is no `heat` output any more. The fire no longer
        PAINTS one-way energy into every cell its rays cross — it runs an
        antisymmetric net-T^4 EXCHANGE into `rad_net` (signed; solids only,
        because air has heat_atten == 0 and by Kirchhoff neither absorbs nor
        emits), plus D3's positive-only `rad_flux` SENSOR at air cells, which is
        what unit damage reads and is deliberately outside the energy ledger."""
        self._rebuild()
        self.heat[:] = 0
        self.rad_net[:] = 0
        self.rad_amb[:] = 0
        self.rad_flux[:] = 0
        runner.cast_fire_heat(self, tick=tick)
        return (self.rad_net.astype(np.float64) / HEAT_SCALE,
                self.rad_flux.astype(np.float64) / HEAT_SCALE)


def _runner(k=None):
    """P-R4: `k_fire_heat` is GONE (the painter is retired). The old `k`
    argument is accepted and IGNORED so the call sites below still read as the
    scenarios they were; the exchange's magnitude now comes from `rad_scale`
    (the E° bake) and the emitters' own temperatures."""
    return PhysicsRunner(bp)


# ---------------------------------------------------------------------------
# P-R4 re-anchor helper (ruling amendment 5 D2).
#
# These tests hold a burner lit by writing `gmap.fire` directly. Under the
# retired PAINTER that was enough: the deposit was `k_fire_heat * I`, a payload
# dial that did not care what the tile's temperature was. Under the net-T^4
# EXCHANGE an emitter radiates against its OWN temperature — a synthetically
# lit but ICE-COLD tile emits E[0], four orders below a flame, and correctly
# heats almost nothing. In play a burning tile is HOT (combustion's H_bed holds
# it at the ~440 game plateau, P-R4 gate f), so holding BOTH fields is what
# actually reproduces the scenario these tests describe.
# ---------------------------------------------------------------------------
FLAME_T_GAME = 443.0            # the blessed plateau (ruling A3 / P-R4 gate f)


def _hold_burner(g, y, x, intensity=0.8, T_game=FLAME_T_GAME):
    """Hold a tile burning AND at flame temperature (see the note above)."""
    g.fire[y, x] = FIRE_Q(float(intensity))
    g.temperature[y, x] = FIRE_Q(float(T_game))


# ---------------------------------------------------------------------------
# (a) the exchange lands on SOLIDS, the sensor lands on AIR
#
# RE-ANCHORED AT P-R4 (ruling amendment 5 D2). This test asserted the painter's
# signature behaviour: "the burning tile gets the full self-deposit (all 8 rays
# land at distance 0)" and "heat radiates OUTWARD across the surrounding AIR".
# Both are now WRONG BY CONSTRUCTION and their inversion is the whole point of
# the patch:
#   * a tile does NOT heat itself — the source is its own first marched cell, so
#     E[T_s] - E[T_s] == 0 exactly (ruling A1.2, and Erik's "I do not think our
#     radiation will heat its own tile");
#   * AIR takes no ENERGY — heat_atten == 0, so by Kirchhoff it neither absorbs
#     nor emits. The painter's air-heating died with the painter.
# The INTENT survives intact: a fire must move heat into the things around it,
# and must register flux where a unit would stand. That is what is asserted now.
# ---------------------------------------------------------------------------
def test_fire_deposits_heat_on_source_and_radiates():
    """RE-ANCHORED AGAIN AT P-F1a (rules 3 and 4).

    Two further inversions on top of P-R4's, both deliberate:

      * THE NEIGHBOUR MOVED OFF THE CONTACT FACE. Under v7 rule 3 a ray stepping
        from a solid into a FACE-ADJACENT solid terminates with no deposit and
        no charge -- conduction owns contact (Erik ruling 3). A neighbour at
        (5, 6) therefore receives NOTHING radiatively, by design, and asserting
        otherwise would be asserting against the law. The absorber is now
        air-separated, as a wall across an open room is.
      * CONSERVATION IS THE LEDGER IDENTITY. The emitter's rays now reach the
        grid edge and are charged there (rule 4), so `rad_net.sum()` is
        NEGATIVE by exactly what the sky ledger received.
    """
    r = _runner()
    sc = _FireScene(11, 11)
    sc.light(5, 5, 0.8)
    # A wood COLUMN one air tile away: air-separated (rule 3 never fires between
    # it and the emitter) and wide enough that the 8-ray fan cannot miss it on
    # any tick of the D4 rotation.
    for y in range(11):
        sc.set_wood(y, 7)
    rad, flux = sc.cast(r)
    assert rad[5, 5] < 0, "the burning tile did not LOSE heat by radiating"
    assert rad[:, 7].sum() > 0, "the air-separated solid wall gained no heat"
    # Rule 4's ledger identity, on the raw integer planes (the returned arrays
    # are scaled floats; the books close to the COUNT, so check the integers).
    assert int(sc.rad_net.sum()) + int(sc.rad_amb.sum()) == 0, (
        "the exchange did not conserve: rad_net + rad_amb != 0")
    assert int(sc.rad_amb.sum()) > 0, "no ray escaped an 11x11 open grid"
    # The surrounding AIR registers incident flux (D3) but takes no energy.
    ring_air = flux[4:7, 4:7].copy()
    ring_air[1, 1] = 0.0                   # exclude the source itself
    assert ring_air.max() > 0, "no radiant flux registered around the fire"
    air = _TBL.heat_atten[sc.material] <= 0.0
    assert abs(rad[air]).max() == 0.0, "AIR absorbed energy - Kirchhoff violated"


def test_contact_faces_are_radiation_inert():
    """v7 rule 3, stated positively: a FACE-ADJACENT solid receives NOTHING.

    New at P-F1a. This is the semantics that took the first-ring absorber out of
    the test above, so it is worth pinning directly rather than leaving it as an
    inference: contact is conduction's domain, and the radiative books simply do
    not enumerate those directions. The same tile, moved one cell further out so
    that air separates it, DOES receive -- which is what makes this a statement
    about contact rather than about reach.
    """
    r = _runner()
    # Swept over a FULL D4 rotation. A single tile on the pure (+2, 0) axis is
    # only swept by the fan on SOME ticks, so a one-tick probe would be
    # measuring the rotation, not the law.
    n_ticks = int(r.fire_ray_count)
    touch_total = 0
    sep_total = 0
    for tick in range(n_ticks):
        touching = _FireScene(11, 11)
        touching.light(5, 5, 0.8)
        touching.set_wood(5, 6)            # FACE-ADJACENT to the emitter
        rad_touch, _ = touching.cast(r, tick=tick)
        touch_total += int(touching.rad_net[5, 6])
        assert int(touching.rad_net.sum()) + int(touching.rad_amb.sum()) == 0

        separated = _FireScene(11, 11)
        separated.light(5, 5, 0.8)
        separated.set_wood(5, 7)           # one air tile away
        separated.cast(r, tick=tick)
        sep_total += int(separated.rad_net[5, 7])
        assert int(separated.rad_net.sum()) + int(separated.rad_amb.sum()) == 0

    print(f"\nP-F1a rule 3 - over {n_ticks} ticks: face-adjacent solid received "
          f"{touch_total} counts, air-separated solid received {sep_total}")
    assert touch_total == 0, (
        "a face-adjacent solid absorbed radiation - rule 3 (contact faces are "
        "radiation-inert) is not being applied")
    assert sep_total > 0, (
        "the air-separated solid received nothing either - the scene is "
        "vacuous, so the contact assertion above proves nothing")


def test_no_fire_no_heat():
    # No emitter -> the pass is a no-op on BOTH outputs (mirrors the C++
    # early-exit, which P-R4 widened to `burning OR warm-solid`).
    r = _runner()
    sc = _FireScene(7, 7)              # nothing lit, everything at ambient
    rad, flux = sc.cast(r)
    assert abs(rad).max() == 0.0
    assert flux.max() == 0.0


def test_hotter_fire_reaches_farther():
    # max_range = range_base + range_per_intensity * I -> a full blaze reaches a
    # strictly larger footprint than a guttering flame. RE-ANCHORED onto the D3
    # flux sensor: the range model is unchanged by P-R4, but the observable that
    # covers open air is now `rad_flux` (the exchange itself only lands on
    # solids, and an empty room has none).
    r = _runner()
    lo = _FireScene(15, 15); lo.light(7, 7, 0.1)
    hi = _FireScene(15, 15); hi.light(7, 7, 1.0)
    n_lo = int((lo.cast(r)[1] > 0).sum())
    n_hi = int((hi.cast(r)[1] > 0).sum())
    assert n_hi > n_lo, f"hotter fire should reach more tiles ({n_hi} vs {n_lo})"


# ---------------------------------------------------------------------------
# (b) occlusion: a wall blocks the fire's heat beyond it (heat_atten / K1)
#
# RE-ANCHORED: same scenario, same geometry, same intent — the observable moves
# from the painter's `heat` to P-R4's flux sensor, which carries the identical
# `heat_survival` occlusion (that is exactly why D3 exists: the damage sampler's
# "already correctly occluded incident flux" contract had to survive).
# ---------------------------------------------------------------------------
def test_wall_blocks_fire_heat_clear_path_heats_further():
    r = _runner(k=200.0)
    # Burner at (6,6) on a 13x13 grid. The fixed-angle fan sends a ray out along
    # ROW 5 to the LEFT: (6,6)->(5,5)->(5,4)->(5,3)->(5,2) (verified geometry).
    # A hull wall on that ray path must zero every tile beyond it.
    clear = _FireScene(13, 13); clear.light(6, 6, 0.8)
    hc = clear.cast(r)[1]
    # The clear leftward ray lights tiles 3-4 out along row 5.
    assert hc[5, 3] > 0 and hc[5, 2] > 0, (
        f"clear leftward ray should light tiles 3-4 out: {hc[5, 2:6]}")

    blocked = _FireScene(13, 13); blocked.light(6, 6, 0.8)
    blocked.set_hull(5, 4)                 # wall ON the ray path
    hb = blocked.cast(r)[1]
    # Flux still reaches up to the wall, then is killed beyond it (heat_atten
    # 1.0 drives heat_survival to 0 AFTER that cell's deposit).
    assert hb[5, 3] == 0.0 and hb[5, 2] == 0.0, (
        f"hull must block fire heat beyond it: got {hb[5, 3]}, {hb[5, 2]}")
    # And the clear path genuinely reached FURTHER than the blocked one.
    assert hc[5, 2] > hb[5, 2]


def test_heat_lands_on_solid_not_lost_in_air_conversion():
    # (a) cross-check via temperature: after one convert pass, a SOLID tile that
    # received heat has temperature > 0, while an AIR tile (kappa 0) stays 0 even
    # though heat was deposited on it (temperature lives on solids only).
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=3, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 14] = MAT_WOOD          # burner
    g.material[50, 15] = MAT_WOOD          # adjacent solid
    g._update_caches()
    sim.set_paused(False)
    g.fire[50, 14] = FIRE_Q(0.8)
    sim.step()
    # Solid neighbour heated. EOS P2 (design §4): the air tile next to the fire
    # now ALSO holds gas temperature (radiation deposit via dT = dE/(N*c_v)) —
    # the old "air must stay 0" doctrine was retired by locked decision 7.
    assert int(g.temperature[50, 15]) > 0, "adjacent solid did not heat"
    assert int(g.temperature[49, 14]) > 0, "air received no gas temperature (P2 deposit missing)"


# ---------------------------------------------------------------------------
# (c) full chain: heat -> temperature -> ignition through Simulation.step()
# ---------------------------------------------------------------------------
def _chain_sim():
    """The full-chain scenario: a burner and an AIR-SEPARATED wood target.

    P-F1a re-anchor (v7 rule 3): the target sits OFF THE CONTACT FACE. A
    face-adjacent plank is conduction's business now, not radiation's — a ray
    stepping solid-into-face-adjacent-solid terminates with no deposit and no
    charge, so an adjacent target measures the conduction path while claiming to
    measure the radiative one. Air-separated is the geometry the chain's own
    canon describes: "a fire radiates heat across an open room; distant wood
    catches".
    """
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 14] = MAT_WOOD          # burner
    g.material[50, 15] = MAT_AIR           # the open tile it radiates ACROSS
    g.material[50, 16] = MAT_WOOD          # target (air-separated)
    g._update_caches()
    # P-F1b RE-ANCHOR: THE PIN IS GONE. P-R4 pinned cool_shift = 9 here because
    # the shipped 5 was a painter-era leftover and the arc had not yet tuned the
    # dial — the test had to pick the value "the rest of the arc measures at".
    # P-F1b IS that tune: [materials.wood] now ships cool_shift = 13 (the
    # cellulosic re-tune; with T_emit_gate above every ignition_temp, cool_shift
    # is what stands between an incoming radiative flux and the neighbour's
    # ignition). So this scenario now runs the SHIPPED material table, and the
    # chain is measured on the configuration the game actually loads.
    sim.set_paused(False)
    return sim, g


def test_full_chain_radiation_heats_air_separated_wood():
    """THE CHAIN: radiation -> temperature, across an open tile, end-to-end.

    RE-ANCHORED AT P-F1a. This test used to assert IGNITION. At P-F1a's frozen
    dials it cannot reach it, and that is the patch's NAMED, EXPECTED outcome
    (the P-R2-form acceptance) rather than a defect — so the test is SPLIT:
    this half owns the chain, which is intact and strictly checkable, and the
    ignition MAGNITUDE moves to the strict-xfail below that P-F1b will trip.

    What is strictly gated here: with the cellular spread deleted, radiation is
    the ONLY path by which the target can warm at all. It starts at ambient 0
    and its temperature must climb monotonically-in-the-large to a real
    plateau — heat crossed an air gap, landed on a solid, and converted. Every
    link (emitter mask -> emission ray -> rule-1 pair -> rad_net -> the Pass-1
    fold) has to work for that number to move at all.
    """
    sim, g = _chain_sim()
    assert int(g.temperature[50, 16]) == 0, "target did not start at ambient"
    peak = 0
    for _ in range(1, 200):
        _hold_burner(g, 50, 14)            # hold the burner lit AND hot
        sim.step()
        peak = max(peak, int(g.temperature[50, 16]))
    peak_game = peak / 65536.0
    print(f"\nP-F1a chain - air-separated wood reached {peak_game:.1f} game "
          f"(ignition_temp {IGN_WOOD_Q16 / 65536.0:.0f})")
    assert peak > 0, (
        "the air-separated target never warmed at all - the radiation chain "
        "(emitter mask -> emission ray -> pair -> rad_net -> Pass-1 fold) is "
        "broken somewhere, not merely under-calibrated")
    # A real plateau well above the noise floor, not a single LSB of drift.
    # P-F1b: at the recalibrated dials this reaches wood's ignition_temp (the
    # sibling test below owns that magnitude); the frozen-dial figure this
    # bound was written against was ~182.
    assert peak_game > 100.0, (
        f"the target only reached {peak_game:.1f} game - the chain is moving "
        f"far less than the recalibrated dials predict")
    # The rail that guards the signed fold must never engage in a gate scenario.
    assert int(sim.physics_runner.temperature.t_low_rail_hits) == 0, (
        "the Pass-1 LOW rail engaged - the budget argument in v7.2 is wrong")


def test_full_chain_heat_ignites_air_separated_wood():
    """The ignition MAGNITUDE — RESTORED AT P-F1b (the designed handoff).

    This test carried a strict xfail through P-F1a. Its reason, kept here as the
    tombstone: "At P-F1a's FROZEN dials the v7 books cannot carry an
    air-separated plank to wood's 300-game ignition_temp: measured, it plateaus
    at ~182 game... the receiver STALLS AT THE GATE: T_emit_gate is 180, so as
    the target crosses ~180 it becomes an emitter itself and begins paying its
    OWN sky in the ~7 directions that leave the world while still receiving on
    the ~1 direction that sees the burner... strict=True ON PURPOSE: when P-F1b
    restores ignition this test FAILS as an unexpected PASS, which is the
    handoff signal." It did, at the first P-F1b bench run, and this is the
    conversion to a plain assertion.

    WHAT MOVED (docs/fire_recalibration_2026-08-02.md): `T_emit_gate` 180 -> 310,
    which is above every flammable ignition_temp, so the receiver no longer
    starts paying sky before it can light; and [materials.wood] `cool_shift`
    5 -> 13, which is what sets how high a one-ray-per-tick incident flux can
    carry a non-casting solid. The two together are the gate-wall fix.

    THE BAND. The receiver is crossed by ~1 of the burner's 8 fan rays, so its
    ceiling is where a_s*a_r*w*(E[T_s]-E[T_r]) == T/2^cool_shift. Against a
    burner pinned at FLAME_T_GAME that ceiling is a few tens of game units above
    wood's 300, i.e. the pass has real but not lavish margin — the band asserted
    below is "crosses 300, in the tens-of-seconds regime the chain is supposed
    to feel like", not "blows past it".
    """
    sim, g = _chain_sim()
    ignited_tick = None
    for t in range(1, 400):
        _hold_burner(g, 50, 14)
        sim.step()
        if g.fire[50, 16] > 0:
            ignited_tick = t
            break
    peak_game = int(g.temperature[50, 16]) / 65536.0
    print(f"\nP-F1b chain - air-separated wood ignited at tick {ignited_tick} "
          f"({(ignited_tick or 0) / sim._tps:.1f} s), T = {peak_game:.1f} game")
    assert ignited_tick is not None, (
        "air-separated wood never ignited from fire heat")
    assert int(g.temperature[50, 16]) >= IGN_WOOD_Q16, (
        "target ignited without its temperature crossing ignition_temp")
    assert ignited_tick > sim._tps // 4, (
        f"ignited too fast ({ignited_tick} ticks) - heat path is not gentle")
    # The recalibrated band: a radiative one-gap chain is a TENS-OF-SECONDS
    # event, not an instant one and not a never (P-F1b gate (d); Erik's spread
    # ruling puts crate-to-crate at 30-60 s and first ignition from cold at
    # minutes). This burner is pinned at flame temperature, i.e. the fastest
    # case the chain ever sees.
    assert ignited_tick <= 12 * sim._tps, (
        f"ignited after {ignited_tick / sim._tps:.1f} s - the recalibration is "
        f"under-delivering on the radiative chain")


# ---------------------------------------------------------------------------
# (d) unit heat damage + zombie 4x, through a full Simulation.step()
# ---------------------------------------------------------------------------
def test_unit_next_to_fire_loses_hp_and_zombie_takes_4x():
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    fy, fx = 50, 25                        # burner near the horizontal centre
    g.material[fy, fx] = MAT_WOOD
    g._update_caches()

    # Place a marine and a zombie MIRROR-symmetric about the fire column so they
    # feel an identical incident flux -> the only damage difference is the zombie
    # fire multiplier. Footprints: marine [20..22], zombie [28..30] (both 5 tiles
    # from the fire column on opposite sides), shared row band.
    marine = Unit("M1", x=fx - 5, y=fy - 1, team=0)
    zombie = Unit("Z1", x=fx + 3, y=fy - 1, team=1)
    sim.add_unit(marine)
    sim.add_unit(zombie)
    assert not marine.is_zombie and zombie.is_zombie
    sim.set_paused(False)

    sim.step()                             # first tick stamps the units
    marine.current_hp = zombie.current_hp = 1000.0
    hp0 = 1000.0
    _hold_burner(g, fy, fx)                # P-R4: lit AND at flame temperature
    sim.step()

    dmg_marine = hp0 - marine.current_hp
    dmg_zombie = hp0 - zombie.current_hp
    assert dmg_marine > 0.0, "marine took no heat damage next to the fire"
    assert dmg_zombie > 0.0, "zombie took no heat damage next to the fire"
    ratio = dmg_zombie / dmg_marine
    assert abs(ratio - float(CFG.zombie.fire_damage_multiplier)) < 1e-3, (
        f"zombie should take {CFG.zombie.fire_damage_multiplier}x a marine, "
        f"got {ratio:.4f}")
    print(f"\n[unit heat dmg] marine={dmg_marine:.4f} zombie={dmg_zombie:.4f} "
          f"ratio={ratio:.3f}")


def test_unit_away_from_fire_unharmed():
    # A unit far from any fire takes no heat damage (the deposit is short-range +
    # occluded, so a distant tile reads 0 flux).
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 25] = MAT_WOOD
    g._update_caches()
    far = Unit("M1", x=10, y=100, team=0)   # nowhere near the fire
    sim.add_unit(far)
    sim.set_paused(False)
    sim.step()
    far.current_hp = 1000.0
    g.fire[50, 25] = FIRE_Q(0.8)
    sim.step()
    assert far.current_hp == 1000.0, "a distant unit took heat damage"


# ---------------------------------------------------------------------------
# (e) determinism — same scene/seed -> bit-identical temperature after N ticks
# ---------------------------------------------------------------------------
def test_determinism_bit_identical_temperature():
    def run():
        level = load_level("unhcr_vessel")
        sim = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False)
        g = sim.gmap
        for (yy, xx) in [(50, 14), (50, 15), (50, 16)]:
            g.material[yy, xx] = MAT_WOOD
        g._update_caches()
        sim.set_paused(False)
        for _ in range(8):
            g.fire[50, 14] = FIRE_Q(0.8)   # hold a steady source
            sim.step()
        return g.temperature.copy()

    a = run()
    b = run()
    assert np.array_equal(a, b), "fire-heat -> temperature is not bit-identical"
    assert int(a.max()) > 0, "scene produced no temperature (nothing to compare)"


def test_cast_fire_heat_does_not_touch_rng():
    # The heat path must NOT consume sim.rng (it is sim-affecting and must be
    # bit-identical regardless of RNG state). Casting fire heat leaves the
    # generator's bit-state untouched.
    r = _runner(k=200.0)
    sc = _FireScene(11, 11)
    sc.light(5, 5, 0.8)
    rng = np.random.default_rng(123)
    before = rng.bit_generator.state
    sc.cast(r)                              # cast_fire_heat takes no rng arg
    after = rng.bit_generator.state
    assert before == after, "the fire-heat pass perturbed an RNG (it must not)"
    # P-R4 D4: the fan's phase now advances with the TICK — still no RNG. The
    # rotation is a pure function of (x, y, tick), so the same tick reproduces
    # the same plane exactly and no generator is touched.
    a = sc.cast(r, tick=3)[0].copy()
    b = sc.cast(r, tick=3)[0].copy()
    assert np.array_equal(a, b), "the per-tick fan rotation is not deterministic"
    assert rng.bit_generator.state == before, "the rotation consumed RNG state"


# ---------------------------------------------------------------------------
# (f) conservative k_fire_heat -> no instant firestorm
# ---------------------------------------------------------------------------
def test_lone_fire_does_not_firestorm_in_a_couple_ticks():
    # Build a hollow wood ROOM (28 wall tiles around an air interior), ignite ONE
    # wall tile, and confirm the fire does NOT engulf the whole structure in a
    # couple of ticks. With the cellular spread DELETED, spread is now purely
    # radiation -> heat -> temperature -> ignition, which is gentle: a neighbour
    # needs a few seconds of conducted heat to cross ignition_temp, so a lone
    # fire cannot firestorm the structure in a handful of ticks.
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    y0, x0 = 40, 10
    for d in range(8):
        g.material[y0, x0 + d] = MAT_WOOD
        g.material[y0 + 7, x0 + d] = MAT_WOOD
        g.material[y0 + d, x0] = MAT_WOOD
        g.material[y0 + d, x0 + 7] = MAT_WOOD
    g._update_caches()
    n_wood = int((g.material == MAT_WOOD).sum())
    assert n_wood >= 20
    sim.set_paused(False)
    g.fire[y0, x0] = FIRE_Q(0.8)
    counts = []
    for _ in range(3):
        g.fire[y0, x0] = max(int(g.fire[y0, x0]), FIRE_Q(0.8))
        sim.step()
        counts.append(int((g.fire > FIRE_001_Q).sum()))
    # After a couple of ticks only a tiny fraction of the wall is alight — NOT a
    # map-wide firestorm. (Far below half the structure.)
    assert max(counts) < n_wood // 2, (
        f"a lone fire firestormed the structure too fast: {counts} of {n_wood}")


# ---------------------------------------------------------------------------
# Placement: the fire heat pass runs INSIDE the tick (heat is wiped end-of-tick
# but its effect on temperature persists) — a guard that K2 is actually wired
# into Simulation.step (not just callable in isolation).
# ---------------------------------------------------------------------------
def test_fire_heat_is_wired_into_simulation_step():
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=5, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 14] = MAT_WOOD
    g._update_caches()
    sim.set_paused(False)
    assert int(g.temperature.max()) == 0
    g.fire[50, 14] = FIRE_Q(0.8)
    sim.step()
    # A single plain step() raised temperature on the burning tile -> the fire
    # heat pass ran inside the tick, before the TemperatureSolver. (heat itself
    # is cleared at end of tick; temperature persists.)
    assert int(g.temperature[50, 14]) > 0, (
        "Simulation.step did not run the fire heat pass (temperature stayed 0)")
    assert int(g.heat.max()) == 0, "heat was not cleared at end of tick"
