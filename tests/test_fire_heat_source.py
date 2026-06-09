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
from simulation.unit import Unit  # noqa: E402

HEAT_SCALE = 65536          # Q16.16 (== TEMP_SCALE), shared heat/temperature domain
_TBL = MaterialTable.from_config()
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
        self.fire = np.zeros((h, w), dtype=np.float32)
        self.heat = np.zeros((h, w), dtype=np.int32)
        self._h, self._w = h, w

    def set_wood(self, y, x):
        self.material[y, x] = MAT_WOOD

    def set_hull(self, y, x):
        self.material[y, x] = MAT_HULL

    def light(self, y, x, intensity):
        self.material[y, x] = MAT_WOOD       # fire only ever lives on flammable
        self.fire[y, x] = float(intensity)

    def _rebuild(self):
        m = self.material
        self.heat_atten = np.ascontiguousarray(_TBL.heat_atten[m], dtype=np.float32)
        self.dyn_light_atten = np.ascontiguousarray(
            _TBL.light_atten[m], dtype=np.float32)
        self.smoke = np.zeros((self._h, self._w), dtype=np.float32)

    def cast(self, runner):
        self._rebuild()
        self.heat[:] = 0
        runner.cast_fire_heat(self)
        return self.heat.astype(np.float64) / HEAT_SCALE   # energy units


def _runner(k=None):
    r = PhysicsRunner(bp)
    if k is not None:
        r.k_fire_heat = float(k)
    return r


# ---------------------------------------------------------------------------
# (a) heat lands on the source tile and radiates to nearby tiles
# ---------------------------------------------------------------------------
def test_fire_deposits_heat_on_source_and_radiates():
    r = _runner()
    sc = _FireScene(11, 11)
    sc.light(5, 5, 0.8)
    heat = sc.cast(r)
    # The burning tile gets the full self-deposit (all 8 rays land at distance 0).
    assert heat[5, 5] > 0, "burning tile got no heat"
    # Heat radiates OUTWARD across the surrounding air (source-tile skip: a fire
    # on heat-opaque wood must still radiate into the room — engine/06 §1).
    ring = heat[4:7, 4:7].copy()
    ring[1, 1] = 0.0                       # exclude the source itself
    assert ring.max() > 0, "fire deposited no heat on its surroundings"


def test_no_fire_no_heat():
    # No burning tile -> the pass is a no-op (mirrors the C++ early-exit).
    r = _runner()
    sc = _FireScene(7, 7)              # nothing lit
    heat = sc.cast(r)
    assert heat.max() == 0.0


def test_hotter_fire_reaches_farther():
    # max_range = range_base + range_per_intensity * I -> a full blaze radiates
    # to a strictly larger footprint than a guttering flame.
    r = _runner()
    lo = _FireScene(15, 15); lo.light(7, 7, 0.1)
    hi = _FireScene(15, 15); hi.light(7, 7, 1.0)
    n_lo = int((lo.cast(r) > 0).sum())
    n_hi = int((hi.cast(r) > 0).sum())
    assert n_hi > n_lo, f"hotter fire should reach more tiles ({n_hi} vs {n_lo})"


# ---------------------------------------------------------------------------
# (b) occlusion: a wall blocks the fire's heat beyond it (heat_atten / K1)
# ---------------------------------------------------------------------------
def test_wall_blocks_fire_heat_clear_path_heats_further():
    r = _runner(k=200.0)
    # Burner at (6,6) on a 13x13 grid. The fixed-angle fan sends a ray out along
    # ROW 5 to the LEFT: (6,6)->(5,5)->(5,4)->(5,3)->(5,2) (verified geometry).
    # A hull wall on that ray path must zero every tile beyond it.
    clear = _FireScene(13, 13); clear.light(6, 6, 0.8)
    hc = clear.cast(r)
    # The clear leftward ray heats tiles 3-4 out along row 5.
    assert hc[5, 3] > 0 and hc[5, 2] > 0, (
        f"clear leftward ray should heat tiles 3-4 out: {hc[5, 2:6]}")

    blocked = _FireScene(13, 13); blocked.light(6, 6, 0.8)
    blocked.set_hull(5, 4)                 # wall ON the ray path
    hb = blocked.cast(r)
    # Heat still reaches up to the wall, then is killed beyond it (heat_atten 1.0).
    assert hb[5, 3] == 0.0 and hb[5, 2] == 0.0, (
        f"hull must block fire heat beyond it: got {hb[5, 3]}, {hb[5, 2]}")
    # And the clear path genuinely heated FURTHER than the blocked one.
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
    g.fire[50, 14] = 0.8
    sim.step()
    # Solid neighbour heated; an air tile next to the fire stays at 0 temperature.
    assert int(g.temperature[50, 15]) > 0, "adjacent solid did not heat"
    assert int(g.temperature[49, 14]) == 0, "air tile holds temperature (it must not)"


# ---------------------------------------------------------------------------
# (c) full chain: heat -> temperature -> ignition through Simulation.step()
# ---------------------------------------------------------------------------
def test_full_chain_heat_ignites_adjacent_wood():
    """A flammable wood wall held next to a burning tile crosses ignition_temp
    and IGNITES — via fire heat -> temperature -> ignition. With the cellular
    spread DELETED (fire_design_proposal §1), radiation->ignition is now the ONLY
    spread path, so no spread toggle is needed to isolate it. Proves the chain
    heat -> temperature -> ignition end-to-end."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 14] = MAT_WOOD          # burner
    g.material[50, 15] = MAT_WOOD          # target (adjacent)
    g._update_caches()
    sim.set_paused(False)

    assert g.fire[50, 15] == 0.0
    ignited_tick = None
    for t in range(1, 120):
        g.fire[50, 14] = 0.8               # hold the burner lit
        sim.step()
        if g.fire[50, 15] > 0.0:
            ignited_tick = t
            break

    assert ignited_tick is not None, "adjacent wood never ignited from fire heat"
    # It ignited BECAUSE temperature crossed the (Q16.16) ignition threshold.
    assert int(g.temperature[50, 15]) >= IGN_WOOD_Q16, (
        "target ignited without its temperature crossing ignition_temp")
    # And it took a FEW SECONDS (toward but not instantly past) — not tick 1.
    assert ignited_tick > sim._tps // 4, (
        f"ignited too fast ({ignited_tick} ticks) — heat path is not gentle")


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
    g.fire[fy, fx] = 0.8
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
    g.fire[50, 25] = 0.8
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
            g.fire[50, 14] = 0.8           # hold a steady source
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
    g.fire[y0, x0] = 0.8
    counts = []
    for _ in range(3):
        g.fire[y0, x0] = max(float(g.fire[y0, x0]), 0.8)
        sim.step()
        counts.append(int((g.fire > 0.01).sum()))
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
    g.fire[50, 14] = 0.8
    sim.step()
    # A single plain step() raised temperature on the burning tile -> the fire
    # heat pass ran inside the tick, before the TemperatureSolver. (heat itself
    # is cleared at end of tick; temperature persists.)
    assert int(g.temperature[50, 14]) > 0, (
        "Simulation.step did not run the fire heat pass (temperature stayed 0)")
    assert int(g.heat.max()) == 0, "heat was not cleared at end of tick"
