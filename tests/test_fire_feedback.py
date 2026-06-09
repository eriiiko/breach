"""Fire intensity FEEDBACK model (fire_design_proposal §2/§3/§5).

The cellular spread is DELETED; spread is now radiation -> heat -> temperature ->
ignition (covered by test_fire_heat_source.py / test_temperature_ignition.py).
This file pins the new per-tile life/death of an already-lit fire:

  T     = temperature[i] / TEMP_SCALE            (the conduction-pass field)
  F     = clamp01(wall_hp[i] / fuel_ref)         (fuel from remaining wall HP)
  P     = mean atmosphere over open (non-solid,non-vacuum) 4-neighbours  (O2 proxy)
  W     = |wind| from the shared wind field
  hot   = clamp01((T - fire_T_ext) / fire_T_span)
  o2    = smoothstep(P_min, P_full, P)           (pressure IS oxygen)
  avail = F * o2
  grow  = k_grow * avail * hot * I*(1-I) * (1 + k_wind_fan*W)
  die   = k_die*(1 - avail*hot)*I + k_wind_strip*W*(1-I)*I
  I    += dt*(grow - die);  clamp01;  if I < I_min -> 0
  atmosphere[i] += max(fire_pressure_gain*I*(1 - atmosphere[i]/p_expand_ref)*dt, 0)

Cases: a fed fire GROWS; a starved one (cold OR no fuel OR low pressure) DECAYS to
0; a vented room DIES (decompression); a fire whose wall_hp runs out DIES
(burnout); wind FANS a big fire and BLOWS OUT a small one (crossover); the plume
deposit raises the fire's own atmosphere so wind points OUTWARD; spread happens
via radiation->ignition with the cellular stencil gone; determinism (bit-identical
fire field); and a conservative-default fire does NOT firestorm a wood room.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fire_feedback.py -q
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

TEMP_SCALE = 65536          # Q16.16 (== HEAT_SCALE), shared heat/temperature domain
_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])   # 300 * 65536
DT = 0.1                    # a generous game-tick-sized step for the feedback


def _params_runner():
    """A FireSimulation with the config [physics.fire] defaults bound (exactly
    what the game runs), plus those defaults read back for assertions."""
    r = PhysicsRunner(bp)
    return r.fire


# ---------------------------------------------------------------------------
# Synthetic single-tile feedback harness. A 3x3 grid with a flammable WOOD tile
# in the centre and AIR around it; the air ring supplies the pressure (O2 proxy)
# read. We drive temperature / wall_hp / atmosphere / wind directly and step the
# C++ FireSimulation in isolation (no level, no renderer) so each feedback term
# is exercised cleanly and deterministically.
# ---------------------------------------------------------------------------
class _FeedbackScene:
    def __init__(self, *, I, T=0.0, wall_hp=60.0, atm=1.0, wind=0.0):
        m = np.full((3, 3), MAT_AIR, dtype=np.int8)
        m[1, 1] = MAT_WOOD
        self.material = m
        self.flammable = np.ascontiguousarray(_TBL.flammable[m])
        self.solid = np.ascontiguousarray(_TBL.permeability[m] <= 0.0)
        self.is_vacuum = np.zeros((3, 3), dtype=bool)
        # Air ring carries the chosen pressure; the solid centre holds none.
        self.atmosphere = np.where(
            self.solid, 0.0, float(atm)).astype(np.float32)
        self.smoke = np.zeros((3, 3), dtype=np.float32)
        self.wall_hp = np.zeros((3, 3), dtype=np.float32)
        self.wall_hp[1, 1] = float(wall_hp)
        self.temperature = np.zeros((3, 3), dtype=np.int32)
        self.temperature[1, 1] = int(round(float(T) * TEMP_SCALE))
        self.fire = np.zeros((3, 3), dtype=np.float32)
        self.fire[1, 1] = float(I)
        # Uniform wind in +x (magnitude `wind`). The feedback only reads |wind|.
        self.wind_x = np.full((3, 3), float(wind), dtype=np.float32)
        self.wind_y = np.zeros((3, 3), dtype=np.float32)

    def step(self, fire_sim, dt=DT):
        fire_sim.step(
            self.fire, self.atmosphere, self.smoke, self.wall_hp,
            self.temperature, self.wind_x, self.wind_y,
            self.solid, self.is_vacuum, self.flammable,
            dt,
        )
        return float(self.fire[1, 1])


# ---------------------------------------------------------------------------
# Fed fire GROWS; starved fires DECAY to 0
# ---------------------------------------------------------------------------
def test_fed_fire_grows():
    # Hot (T=500 >> fire_T_ext 350), fuelled (wall_hp 60 -> F=1), pressured
    # (atm 1.0 -> o2=1), no wind. grow > die -> intensity climbs.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.3, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    after = sc.step(fs)
    assert after > 0.3, f"a fed fire should grow (0.3 -> {after})"


def test_cold_fire_decays_to_zero():
    # No heat (T=0 -> hot=0): grow=0, die = k_die*(1-0)*I. Fire dies out.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.5, T=0.0, wall_hp=60.0, atm=1.0, wind=0.0)
    last = 0.5
    for _ in range(200):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a cold fire must decay to 0, stuck at {last}"


def test_no_fuel_fire_decays_to_zero():
    # Hot + pressured but NO fuel (wall_hp 0 -> F=0 -> avail=0): fire starves.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.5, T=500.0, wall_hp=0.0, atm=1.0, wind=0.0)
    last = 0.5
    for _ in range(200):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a fire with no fuel must die, stuck at {last}"


def test_low_pressure_fire_decays_to_zero():
    # Hot + fuelled but LOW pressure (atm 0.3 < P_min 0.6 -> o2=0 -> avail=0):
    # decompression extinguishes the fire.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.5, T=500.0, wall_hp=60.0, atm=0.3, wind=0.0)
    last = 0.5
    for _ in range(200):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a low-pressure fire must die, stuck at {last}"


def test_vented_room_extinguishes():
    # A fire fully surrounded by VACUUM (a hull breach drained the room) reads
    # P=0 (no open, non-vacuum neighbour) -> o2=0 -> dies. The
    # decompression-extinguishes-fire loop via a READ, not a kill-threshold.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.6, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    sc.is_vacuum[:] = True
    sc.is_vacuum[1, 1] = False            # the burning solid itself is not vacuum
    last = 0.6
    for _ in range(200):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a vacuum-vented fire must die, stuck at {last}"


# ---------------------------------------------------------------------------
# Burnout: wall_hp runs out -> the fire starves (the fuel brake)
# ---------------------------------------------------------------------------
def test_burnout_when_wall_hp_runs_out():
    # Hot + pressured + a SMALL fuel store. wall_damage drains wall_hp every step;
    # as wall_hp -> 0, F -> 0, the fire starves and dies. (It may also breach the
    # wall at hp<=0 and zero the fire that way — either is a valid burnout.)
    fs = _params_runner()
    sc = _FeedbackScene(I=0.6, T=500.0, wall_hp=3.0, atm=1.0, wind=0.0)
    # Make it a real wall so burn-through can fire if hp hits 0.
    sc.solid[1, 1] = True
    hp0 = float(sc.wall_hp[1, 1])
    last = 0.6
    for _ in range(500):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"fire should burn out as fuel runs out, stuck at {last}"
    assert float(sc.wall_hp[1, 1]) < hp0, "wall_hp did not deplete (no burn)"


# ---------------------------------------------------------------------------
# Wind: FANS a big fire, BLOWS OUT a small one (the crossover)
# ---------------------------------------------------------------------------
def test_wind_fans_a_big_fire():
    # Same well-fed big fire, with vs without wind. The (1 + k_wind_fan*W) factor
    # means the windy one grows MORE per step (a firestorm forming).
    fs = _params_runner()
    calm = _FeedbackScene(I=0.6, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    windy = _FeedbackScene(I=0.6, T=500.0, wall_hp=60.0, atm=1.0, wind=1.0)
    for _ in range(5):
        calm.step(fs)
        windy.step(fs)
    assert windy.fire[1, 1] > calm.fire[1, 1], (
        f"wind should fan a big fire hotter "
        f"(windy {windy.fire[1, 1]:.4f} vs calm {calm.fire[1, 1]:.4f})")


def _ticks_to_die(scene, fire_sim, max_ticks=400):
    for t in range(1, max_ticks + 1):
        if scene.step(fire_sim) == 0.0:
            return t
    return None


def test_wind_blows_out_a_small_fire():
    # The realistic crossover (fire_design_proposal §5): the SAME gust that fans a
    # big blaze SNUFFS a small/marginal one. A small, marginally-fed fire (low
    # drive: T just above extinction -> `hot` small) is in the band where the
    # k_wind_strip*W*(1-I)*I blow-out dominates the k_wind_fan growth, so a strong
    # wind extinguishes it STRICTLY FASTER than calm air does. (A fully-fed big
    # fire under the same wind GROWS — see test_wind_fans_a_big_fire — that is the
    # crossover: big fanned, small snuffed.)
    fs = _params_runner()
    # Marginal drive: T=360 is barely above fire_T_ext (350); hot ~= 0.067, so the
    # fire is dying anyway — wind makes it die SOONER (the blow-out).
    calm = _FeedbackScene(I=0.15, T=360.0, wall_hp=60.0, atm=1.0, wind=0.0)
    windy = _FeedbackScene(I=0.15, T=360.0, wall_hp=60.0, atm=1.0, wind=5.0)
    t_calm = _ticks_to_die(calm, fs)
    t_windy = _ticks_to_die(windy, fs)
    assert t_calm is not None and t_windy is not None, (
        f"both marginal fires should die (calm={t_calm}, windy={t_windy})")
    assert t_windy < t_calm, (
        f"a strong wind should blow out a small fire SOONER "
        f"(windy {t_windy} ticks vs calm {t_calm} ticks)")


# ---------------------------------------------------------------------------
# Plume: the own-tile pressure deposit raises the fire's atmosphere so wind
# (= -grad p) points OUTWARD (smoke pushed away, not pulled in)
# ---------------------------------------------------------------------------
def test_plume_raises_own_atmosphere_wind_points_outward():
    fs = _params_runner()
    # 5x5 all-air room with one burning WOOD tile in the centre; uniform 1.0 atm.
    h = w = 5
    material = np.full((h, w), MAT_AIR, dtype=np.int8)
    material[2, 2] = MAT_WOOD
    flammable = np.ascontiguousarray(_TBL.flammable[material])
    solid = np.ascontiguousarray(_TBL.permeability[material] <= 0.0)
    is_vacuum = np.zeros((h, w), dtype=bool)
    atmosphere = np.full((h, w), 1.0, dtype=np.float32)
    smoke = np.zeros((h, w), dtype=np.float32)
    wall_hp = np.zeros((h, w), dtype=np.float32)
    wall_hp[2, 2] = 60.0
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[2, 2] = int(round(500.0 * TEMP_SCALE))
    fire = np.zeros((h, w), dtype=np.float32)
    fire[2, 2] = 0.8
    wind_x = np.zeros((h, w), dtype=np.float32)
    wind_y = np.zeros((h, w), dtype=np.float32)

    atm_before = float(atmosphere[2, 2])
    for _ in range(10):
        fs.step(fire, atmosphere, smoke, wall_hp, temperature,
                wind_x, wind_y, solid, is_vacuum, flammable, DT)
    atm_after = float(atmosphere[2, 2])
    assert atm_after > atm_before, (
        f"plume must RAISE the fire's own atmosphere "
        f"({atm_before} -> {atm_after}), so smoke is pushed out")
    # wind = -grad(p): on the air tile just to the +x side of the fire, the
    # pressure DECREASES outward (centre is the local max), so -d(atm)/dx > 0 ->
    # wind points AWAY from the fire (outward). Sample the right neighbour.
    grad_x = atmosphere[2, 3] - atmosphere[2, 2]    # < 0 (centre is the peak)
    wind_x_right = -grad_x                            # > 0 -> points away (+x)
    assert wind_x_right > 0.0, (
        "wind next to the fire must point OUTWARD (smoke pushed away, not pulled in)")


def test_plume_does_not_subtract_atmosphere_near_fire():
    # Guard against the DELETED backwards O2-consumption: the fire must NOT lower
    # the atmosphere of any neighbour (that sucked smoke IN). After stepping, no
    # open neighbour's atmosphere is below its start value.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.8, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    before = sc.atmosphere.copy()
    for _ in range(5):
        sc.step(fs)
    # Every air tile is >= its starting atmosphere (no suction toward the fire).
    air = ~sc.solid
    assert np.all(sc.atmosphere[air] >= before[air] - 1e-6), (
        "fire must not subtract atmosphere from its neighbours (the old O2 sink)")


# ---------------------------------------------------------------------------
# Spread is now radiation -> ignition (the cellular stencil is GONE)
# ---------------------------------------------------------------------------
def test_spread_is_radiation_only_no_cellular_stencil():
    # A burning wood tile lights an ADJACENT wood tile via heat -> temperature ->
    # ignition (the radiation path), while a far flammable tile with no heat path
    # does NOT light — i.e. there is no cellular stencil reaching across gaps.
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=11, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 14] = MAT_WOOD          # burner
    g.material[50, 15] = MAT_WOOD          # adjacent target (radiation reaches it)
    g.material[50, 40] = MAT_WOOD          # far target (no heat path)
    g._update_caches()
    sim.set_paused(False)

    adj_lit = False
    for _ in range(120):
        g.fire[50, 14] = 0.8               # hold the burner
        sim.step()
        if g.fire[50, 15] > 0.0:
            adj_lit = True
            break
    assert adj_lit, "adjacent wood never ignited via radiation"
    # The far tile, with no heat reaching it, stayed cold and unlit -> the old
    # gap-leaping cellular stencil is gone.
    assert g.fire[50, 40] == 0.0, "a far flammable tile lit with no heat path"
    assert int(g.temperature[50, 40]) == 0, "far tile heated with no heat path"


# ---------------------------------------------------------------------------
# Determinism: same scene -> bit-identical fire field
# ---------------------------------------------------------------------------
def test_feedback_determinism_bit_identical():
    def run():
        fs = _params_runner()
        sc = _FeedbackScene(I=0.4, T=500.0, wall_hp=60.0, atm=1.0, wind=0.7)
        for _ in range(30):
            sc.step(fs)
        return sc.fire.copy()

    a = run()
    b = run()
    assert np.array_equal(a, b), "fire feedback is not bit-identical across runs"
    assert float(a[1, 1]) > 0.0, "scene produced no fire (nothing to compare)"


# ---------------------------------------------------------------------------
# Conservative defaults do NOT firestorm a wood-filled room
# ---------------------------------------------------------------------------
def test_conservative_default_does_not_firestorm_wood_room():
    # A hollow WOOD room, ignite ONE tile, run a few ticks with the SHIPPED config
    # [physics.fire] defaults. With cellular spread deleted and the heat path
    # gentle, the structure must NOT be engulfed in a handful of ticks.
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=2, breach_physics=bp, enable_recorder=False)
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
    counts = []
    for _ in range(5):
        g.fire[y0, x0] = max(float(g.fire[y0, x0]), 0.8)   # hold the seed lit
        sim.step()
        counts.append(int((g.fire > 0.01).sum()))
    assert max(counts) < n_wood // 2, (
        f"a lone fire firestormed the wood room too fast: {counts} of {n_wood}")
