"""The W4 SPRAY gate: Dragon-7 flamethrower + Miasma Vent (mechanics/03 §5).

What is locked here:

  - CONE MEMBERSHIP EXACTNESS: the integer-safe predicate (squared Q16.16
    dot compare, kit trig only) against hand-computed boundary tiles —
    inside/outside/behind/perpendicular/range-edge, apex never a member;
  - OCCLUSION: flames land ON a wall face but never pour through it
    (gmap.has_los gating);
  - FALLOFF PINNED: exact Q16.16 heat/gas deposits under the documented
    1/isqrt(dist^2) integer falloff, the nozzle rule (no deposit on the
    shooter's own footprint), heat lands on solids / gas does not;
  - IGNITION E2E (CPU backend): a Dragon-7 burst ignites a wood wall within
    the derived tick count — heat -> TemperatureSolver convert ->
    temperature >= ignition_temp -> fire, the whole engine/06 path;
  - TWO-TERMINALS: no W4 code touches unit HP (structural + runtime proof)
    — a marine standing in flames loses HP via the EXISTING heat|max
    exchange row only, and the spray machinery draws NO RNG ever;
  - MIASMA: sustained poison accumulation in the cone; a marine drains via
    the W3 gas[poison] row; a zombie in the same cloud takes 0 (immunity);
  - CAN_ACT INTERRUPTION: a stun stops the burst that tick, consumes the
    fire order, no resume after the status clears;
  - STATIONARY-ONLY + AUTO-FIRE SKIP: a move order in the same phase
    blocks the burst; Move & Attack auto-fire ignores spray weapons;
  - BURST/MAG/RELOAD CADENCE: 4 back-to-back bursts (mag counts BURSTS),
    the 4 s reload stall, refill on the next trigger — all exact ticks;
  - DORMANCY REPLICA: a spray-free scripted firefight is bit-identical
    (fields + hp + events + RNG end-state) to a twin with the W4 deposit
    pass no-opped — the pre-W4 trajectory, untouched.

Run:
    C:/Users/steen/miniconda3/python.exe -m pytest tests/test_spray_weapons.py -q
"""
from __future__ import annotations

import inspect
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
from simulation import gas_fixed  # noqa: E402
from simulation.combat import (  # noqa: E402
    deposit_spray_cone, process_shooting, process_sprays, spray_cone_tiles,
)
from simulation.events import UnitHitEvent  # noqa: E402
from simulation.field_edit import EditQueue, heat_quantize  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import FUEL_GAS, N_TRACE_GASES, POISON as GAS_POISON  # noqa: E402
from simulation.orders import (  # noqa: E402
    ORDER_FIRE, ORDER_MOVE_ATTACK, Order,
)
from simulation.status import STUNNED, apply_status  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

SEED = 20260707


# ---------------------------------------------------------------------------
# Scaffolding (the test_payloads shape)
# ---------------------------------------------------------------------------
def _level(h=24, w=24, edits=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w4_spray", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _dragon():
    t = get_tables()
    w = t.weapons.by_name["dragon_7"]
    return w, t.ammo_for_weapon(w)


def _miasma():
    t = get_tables()
    w = t.weapons.by_name["miasma_vent"]
    return w, t.ammo_for_weapon(w)


def _shooter(weapon_name, x=5, y=9):
    """A marine at anchor (x, y) — centre tile (x+1, y+1) — armed with
    ``weapon_name``."""
    u = Unit("S", x=x, y=y, team=0)
    u.id = 1
    u.weapon_id = weapon_name
    return u


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


# ---------------------------------------------------------------------------
# 1. Cone membership exactness — hand-computed boundary tiles
# ---------------------------------------------------------------------------
def test_cone_membership_hand_computed_boundary_tiles():
    """Dragon-7 half-angle 15 deg, aimed due EAST from apex (10, 10), range
    8, open room. tan(15 deg) = 0.26795, so the hand-computed boundary:

        (dx=4, dy=1)  -> 14.04 deg  IN   (comfortably inside vs kit ULP)
        (dx=3, dy=1)  -> 18.43 deg  OUT
        (dx=7, dy=2)  -> 15.95 deg  OUT
        (dx=8, dy=0)  -> on-axis, dist^2 = 64 = range^2   IN (range edge)
        (dx=8, dy=2)  -> 14.04 deg but dist^2 = 68 > 64   OUT (outranged)
        apex (0, 0)   -> never a member
        (dx=-1, 0)    -> behind, OUT;  (0, dy=-1) -> perpendicular, OUT
    """
    gmap = _room()
    members = {(y, x) for (y, x, _d) in spray_cone_tiles(
        gmap, 10, 10, 18.0, 10.0, 8, 15.0)}

    assert (11, 14) in members          # dx 4, dy 1 — inside the edge
    assert (9, 14) in members           # mirrored below-axis twin
    assert (11, 13) not in members      # dx 3, dy 1 — outside
    assert (9, 13) not in members
    assert (12, 17) not in members      # dx 7, dy 2 — 15.95 deg, outside
    assert (10, 18) in members          # on-axis range edge (dist == 8)
    assert (12, 18) not in members      # in-angle but out of range
    assert (10, 10) not in members      # the apex itself
    assert (10, 9) not in members       # behind
    assert (9, 10) not in members       # perpendicular (90 deg off-axis)
    assert (10, 11) in members          # dist 1 on-axis

    # Cross-check EVERY member against plain float geometry (test-side libm
    # is fine): angle-off-axis <= 15 deg + margin, dist <= 8. The integer
    # predicate and the float cone agree away from the kit's ~1e-5 ULP band.
    for (y, x) in members:
        dx, dy = x - 10, y - 10
        ang = abs(math.degrees(math.atan2(dy, dx)))
        assert ang <= 15.0 + 1e-3, f"({y},{x}) at {ang:.3f} deg leaked in"
        assert dx * dx + dy * dy <= 64


def test_cone_follows_aim_direction():
    """The same predicate aimed NORTH: the cone flips axes exactly."""
    gmap = _room()
    members = {(y, x) for (y, x, _d) in spray_cone_tiles(
        gmap, 12, 10, 10.0, 4.0, 8, 15.0)}
    assert (11, 10) in members          # 1 north of apex
    assert (8, 11) in members           # dy -4, dx 1 (14.04 deg)   IN
    assert (9, 11) not in members       # dy -3, dx 1 (18.43 deg)   OUT
    assert (13, 10) not in members      # behind (south)


# ---------------------------------------------------------------------------
# 2. Occlusion — flames land on the wall face, never beyond it
# ---------------------------------------------------------------------------
def test_occlusion_wall_stops_the_pour_but_takes_the_flame():
    """A hull wall 3 east of the apex: the wall tile itself IS a member
    (the flame lands ON the face — how wood catches), every on-axis tile
    beyond it is occluded out."""
    gmap = _room(edits=[(10, 13, 1)])   # hull at dx 3 on the aim axis
    members = {(y, x) for (y, x, _d) in spray_cone_tiles(
        gmap, 10, 10, 18.0, 10.0, 8, 15.0)}
    assert (10, 11) in members and (10, 12) in members
    assert (10, 13) in members          # the wall face receives the flame
    for x in range(14, 19):
        assert (10, x) not in members, f"(10,{x}) got flame through a wall"


# ---------------------------------------------------------------------------
# 3. Falloff pinned — exact Q16.16 deposits, nozzle rule, solid handling
# ---------------------------------------------------------------------------
def test_falloff_pinned_exact_deposits_and_nozzle_rule():
    """Dragon-7 (heat 2400 — the W6 rescale riding the 10 m range;
    fuel_gas 0.15) fired due east by a marine whose centre is (10, 10):
    per member tile the deposit is column / max(1, isqrt(dist^2)),
    quantized ONCE at the FieldEdit combine. Pinned:

        (10, 12)  dist 2 -> heat 2400/2 = 1200 -> 1200 * 65536 = 78643200
        (10, 13)  dist 3 -> heat 2400/3 = 800  -> 800 * 65536  = 52428800
        (10, 14)  dist 4 -> heat 600           -> 39321600
        (11, 14)  isqrt(17) = 4 -> heat 600    -> 39321600
        gas (10, 12): quantize(0.15/2 = 0.075) -> 4915

    The shooter's own 3x3 footprint (nozzle rule) takes NOTHING — the
    dist-1 on-axis tile (10, 11) is inside it. No RNG is drawn."""
    gmap = _room()
    queue = EditQueue()
    u = _shooter("dragon_7", x=9, y=9)          # centre tile (10, 10)
    weapon, ammo = _dragon()
    rng = np.random.default_rng(SEED)
    state0 = rng.bit_generator.state

    deposit_spray_cone(gmap, queue, u, weapon, ammo, 18.0, 10.0)
    queue.flush(gmap, rng)

    assert int(gmap.heat[10, 12]) == 78643200
    assert int(gmap.heat[10, 13]) == 52428800
    assert int(gmap.heat[10, 14]) == 39321600
    assert int(gmap.heat[11, 14]) == 39321600
    assert int(gmap.gas[FUEL_GAS][10, 12]) == gas_fixed.quantize_scalar(0.075)
    # Nozzle rule: the shooter's own footprint tiles take no deposit.
    for (tx, ty) in u.occupied_tiles():
        assert int(gmap.heat[ty, tx]) == 0
        assert int(gmap.gas[FUEL_GAS][ty, tx]) == 0
    # No RNG, ever (spray edits carry noise = 0).
    assert rng.bit_generator.state == state0
    # Pin the quantize-once arithmetic itself (2400/3 = 800 exactly).
    assert heat_quantize(2400.0 / 3.0) == 52428800


def test_heat_lands_on_solid_wood_but_gas_does_not():
    """The heat field has no skip-mask (that is how walls catch fire); the
    gas policy skips solids (gas does not enter walls)."""
    gmap = _room(edits=[(10, 13, 2)])   # wood at dx 3
    queue = EditQueue()
    u = _shooter("dragon_7", x=9, y=9)
    weapon, ammo = _dragon()
    deposit_spray_cone(gmap, queue, u, weapon, ammo, 18.0, 10.0)
    queue.flush(gmap, np.random.default_rng(SEED))
    assert int(gmap.heat[10, 13]) == 52428800         # flame ON the face
    assert int(gmap.gas[FUEL_GAS][10, 13]) == 0       # solid skip-mask
    assert int(gmap.gas[FUEL_GAS][10, 12]) > 0        # open tile in front


# ---------------------------------------------------------------------------
# 4. Ignition end-to-end (CPU backend): heat -> temperature -> fire
# ---------------------------------------------------------------------------
def test_dragon_ignites_wood_within_the_derived_tick_count():
    """The config derivation of record (ammo.fuel_standard, the W6 2400
    rescale): at dist 2 the wood tile crosses ignition_temp 300 at ~2
    ticks (T_inf 4650), dist 3 at ~3 (T_inf 3100), dist 8 — near the old
    full range — at ~9 (T_inf 1162): the whole near cone catches
    near-instantly and the reach tracks the new 10 m range. Whole-engine
    path: FieldEdit heat -> C++ TemperatureSolver convert ->
    apply_temperature_ignition.

    EOS refactor P4 (merged from main, design §6 item 3): the ignition O2
    gate now reads the REAL local N_O2 mean instead of the atmosphere/P
    proxy — the spray's own heat expands the local air (p* rises ->
    outward wind), transiently thinning REAL O2 before donor-cell flux
    resupplies it, so ignition can land a few ticks later than the pure
    temperature crossing (main widened its pre-W6 dist-3 window for the
    same reason). Upper bounds here carry the same slack over the W6
    temperature-crossing estimates; measured deterministic across
    reruns."""
    for dist, tick_lo, tick_hi in ((2, 0, 16), (3, 1, 24), (8, 3, 44)):
        wood_x = 6 + dist
        sim = Simulation(_level(edits=[(10, wood_x, 2)]), seed=SEED,
                         breach_physics=bp, enable_recorder=False)
        s = _shooter("dragon_7")                      # centre (6, 10)
        sid = sim.add_unit(s)
        assert sim.apply_action(sid, Order(
            ORDER_FIRE, target_fx=wood_x, target_fy=10, phase=0))
        ignite_tick = None
        for i in range(48):
            _step(sim)
            if sim.gmap.fire[10, wood_x] > 0:
                ignite_tick = i
                break
        assert ignite_tick is not None, f"dist {dist}: wood never ignited"
        assert tick_lo <= ignite_tick <= tick_hi, (
            f"dist {dist}: ignited at tick {ignite_tick}, expected "
            f"[{tick_lo}, {tick_hi}]")


# ---------------------------------------------------------------------------
# 5. Two-terminals: no W4 code touches unit HP; the heat row does the work
# ---------------------------------------------------------------------------
def test_spray_code_never_touches_unit_hp_structural_and_runtime():
    """Structural: the spray machinery takes no generator — it cannot draw.
    (W6 amendment of the W4 pin: process_sprays DOES now take an ``events``
    list, but only to append the RENDER-ONLY SprayJetEvent — the flame-jet
    visual; the runtime proof below shows a full burst still emits no
    packet and moves no HP, and deposit_spray_cone itself still takes
    neither rng nor events.) Runtime: a full Dragon-7 burst over a victim
    with NO physics attached (the exchange rows never run) leaves every HP
    bit-identical, even though the cone deposited heat all over the
    victim's tiles."""
    assert "rng" not in inspect.signature(process_sprays).parameters
    assert "rng" not in inspect.signature(deposit_spray_cone).parameters
    assert "events" not in inspect.signature(deposit_spray_cone).parameters

    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    s = _shooter("dragon_7")                          # centre (6, 10)
    victim = Unit("V", x=8, y=9, team=0)              # centre (9, 10), dist 3
    sid = sim.add_unit(s)
    sim.add_unit(victim)
    hp_s, hp_v = s.current_hp, victim.current_hp
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    seen_kinds = set()
    for _ in range(40):                               # a full burst and more
        _step(sim)
        for e in sim.tick_events:
            seen_kinds.add(type(e).__name__)
    # Deposits DID land on the victim's tiles (no physics -> no heat clear).
    assert any(int(sim.gmap.heat[ty, tx]) > 0
               for (tx, ty) in victim.occupied_tiles())
    # ... and nobody's HP moved: the spray wrote fields, nothing else.
    assert s.current_hp == hp_s
    assert victim.current_hp == hp_v
    # The only event a burst emits is the render-only jet — no UnitHit,
    # no UnitKilled (the two-terminals invariant, W6 form).
    assert "SprayJetEvent" in seen_kinds
    assert "UnitHitEvent" not in seen_kinds
    assert "UnitKilledEvent" not in seen_kinds


def test_marine_in_flames_loses_hp_via_the_existing_heat_row():
    """The same scene WITH the engine attached: the victim drains through
    apply_environmental_damage (source 'heat') — the EXISTING coupling row,
    zero new damage code. The shooter (nozzle rule) is untouched."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    s = _shooter("dragon_7")
    victim = Unit("V", x=8, y=9, team=0)
    sid = sim.add_unit(s)
    sim.add_unit(victim)
    hp_s, hp_v = s.current_hp, victim.current_hp
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    heat_hits = []
    for _ in range(12):
        _step(sim)
        heat_hits += [e for e in sim.tick_events
                      if isinstance(e, UnitHitEvent) and e.source == "heat"]
    assert victim.current_hp < hp_v                   # the flames cook
    assert s.current_hp == hp_s                       # the sprayer does not
    assert heat_hits and all(h.unit_id == victim.id for h in heat_hits)


# ---------------------------------------------------------------------------
# 6. Miasma Vent: sustained poison accumulation; zombie immunity
# ---------------------------------------------------------------------------
def test_miasma_sustained_poison_drains_a_marine():
    """The vent paints the near cone past poison_min_density within a few
    ticks and HOLDS it (sustained emission vs the grenade's one-shot);
    damage rides the W3 gas[poison] row (source 'poison_gas'). Only the
    poison slice moves — a vent is not a flamethrower (no heat, no fire,
    no blindness: poison is not teargas)."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    s = _shooter("miasma_vent")
    victim = Unit("V", x=8, y=9, team=0)
    sid = sim.add_unit(s)
    sim.add_unit(victim)
    hp_v = victim.current_hp
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    poison_hits = []
    for _ in range(16):
        _step(sim)
        poison_hits += [e for e in sim.tick_events
                        if isinstance(e, UnitHitEvent)
                        and e.source == "poison_gas"]
    assert sim.gmap.gas[GAS_POISON].any()
    assert victim.current_hp < hp_v
    assert poison_hits and all(h.unit_id == victim.id for h in poison_hits)
    # Poison ONLY: no heat deposit, no fire, no other TRACE gas slice touched
    # (the bulk O2/inert_N2 pair, EOS refactor P1, always carries ambient air).
    assert not sim.gmap.fire.any()
    for g in range(N_TRACE_GASES):
        if g != GAS_POISON:
            assert not sim.gmap.gas[g].any(), f"slice {g} moved on a vent"
    # No blindness (no status at all) — poison is not teargas.
    assert not getattr(victim, "statuses", [])


def test_zombie_in_the_miasma_cloud_takes_zero():
    """Zombies don't breathe (resist_mult[POISON] = 0, W3 lazy emission):
    a zombie hosed point-blank by the Miasma Vent for a full burst takes 0
    damage and draws no packet — poison is not the anti-horde answer."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    s = _shooter("miasma_vent")
    z = Unit("Z", x=10, y=9, team=1)                  # centre (11, 10), dist 5
    sid = sim.add_unit(s)
    sim.add_unit(z)
    hp_z = z.current_hp
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=11, target_fy=10, phase=0))
    for _ in range(24):
        _step(sim)
        assert not any(isinstance(e, UnitHitEvent) and e.unit_id == z.id
                       for e in sim.tick_events)
    assert sim.gmap.gas[GAS_POISON].any()             # the cloud is real
    assert z.current_hp == hp_z                       # ... and irrelevant


# ---------------------------------------------------------------------------
# 7. can_act interruption: burst stops, order consumed, no resume
# ---------------------------------------------------------------------------
def test_stun_interrupts_the_burst_and_consumes_the_order():
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    s = _shooter("dragon_7")
    sid = sim.add_unit(s)
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    _step(sim, 3)
    assert s.spray_ticks_left == 36 - 3               # mid-burst
    assert s.get_fire_order_in_phase(0) is not None

    apply_status(s, STUNNED, magnitude=0, duration_ticks=8)
    _step(sim)                                        # the interruption tick
    assert s.spray_ticks_left == 0                    # stopped THAT tick
    assert s.get_fire_order_in_phase(0) is None       # order CONSUMED
    assert s.spray_order is None and s.spray_target is None

    # No resume: the stun expires, the order is gone, nothing re-arms.
    _step(sim, 20)
    assert s.spray_ticks_left == 0
    assert not any(o.order_type == ORDER_FIRE for o in s.orders)


# ---------------------------------------------------------------------------
# 8. Stationary-only rule + auto-fire skip
# ---------------------------------------------------------------------------
def test_stationary_rule_move_order_blocks_the_burst():
    """A movement order in the SAME phase blocks the spray trigger — the
    sprayer stands still (v1 rule of record). No deposits, no mag touch."""
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    s = _shooter("dragon_7")
    sid = sim.add_unit(s)
    assert sim.apply_action(sid, Order(
        ORDER_MOVE_ATTACK, target_fx=8, target_fy=9, phase=0))
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    _step(sim, 30)
    assert s.spray_ticks_left == 0
    assert not sim.gmap.heat.any()                    # nothing ever deposited
    assert s.current_mag is None                      # mag state untouched
    assert s.last_fire_tick == -999


def test_auto_fire_skips_spray_weapons_and_draws_nothing():
    """Move & Attack with a Dragon-7 and a visible enemy in range: no burst,
    no deposit, no RNG draw, no cadence/mag state touched."""
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    s = _shooter("dragon_7")
    z = Unit("Z", x=9, y=9, team=1)                   # centre dist 4, in range
    sid = sim.add_unit(s)
    sim.add_unit(z)
    assert sim.apply_action(sid, Order(
        ORDER_MOVE_ATTACK, target_fx=6, target_fy=10, phase=0))
    state0 = sim.rng.bit_generator.state
    _step(sim, 10)
    assert s.spray_ticks_left == 0
    assert not sim.gmap.heat.any()
    assert not sim.gmap.gas[FUEL_GAS].any()
    assert s.last_fire_tick == -999 and s.current_mag is None
    assert sim.rng.bit_generator.state == state0


# ---------------------------------------------------------------------------
# 9. Burst / mag / reload cadence — exact ticks (mag counts BURSTS)
# ---------------------------------------------------------------------------
def test_burst_mag_reload_cadence_exact_ticks():
    """Standing fire orders + Dragon-7 (mag 4 BURSTS, burst 36 ticks,
    reload 96 ticks): bursts chain back-to-back for 4 x 36 = 144 deposit
    ticks, the tank swap stalls the next 60 (the reload window opened at
    the 4th TRIGGER, tick 108 -> done 204), and the trigger at 204 refills
    and fires burst 5. Driven straight through the combat layer; the RNG
    is never consumed."""
    gmap = _room()
    queue = EditQueue()
    u = _shooter("dragon_7")                          # centre (6, 10)
    u.orders = [Order(ORDER_FIRE, target_fx=12, target_fy=10, phase=0),
                Order(ORDER_FIRE, target_fx=12, target_fy=10, phase=1)]
    rng = np.random.default_rng(SEED)
    state0 = rng.bit_generator.state
    shots = []

    deposit_ticks = []
    mag_at = {}
    for t in range(210):
        process_shooting(gmap, [u], t, shots, 0.0, rng, queue=queue)
        process_sprays(gmap, [u], queue)
        if len(queue):
            deposit_ticks.append(t)
        queue.clear()                                 # never flushed — cadence only
        if t in (0, 108, 143, 150, 204):
            mag_at[t] = u.current_mag

    # 4 back-to-back bursts: every tick 0..143 deposits; then the stall.
    assert deposit_ticks[:144] == list(range(144))
    assert all(not (144 <= t < 204) for t in deposit_ticks)
    assert 204 in deposit_ticks                       # burst 5 after reload
    # Mag counts BURSTS: 4 -> 3 at trigger 0; 0 after trigger 4 (tick 108);
    # reload window 108 + 96 = 204; refilled-and-spent -> 3 at tick 204.
    assert mag_at[0] == 3
    assert mag_at[108] == 0 and mag_at[143] == 0 and mag_at[150] == 0
    assert u.reload_done_tick == 108 + 96
    assert mag_at[204] == 3
    assert rng.bit_generator.state == state0          # NO RNG anywhere in W4


# ---------------------------------------------------------------------------
# 10. Dormancy replica — no spray equipped == the pre-W4 trajectory
# ---------------------------------------------------------------------------
_FIELDS = ("gas", "atmosphere", "wave_source", "wave_p", "fire", "wall_hp",
           "material", "temperature", "water_depth", "is_vacuum")


def test_dormancy_no_spray_weapon_is_bit_identical_to_prew4():
    """A scripted k5 firefight (the shipped loadout — no spray weapon
    anywhere) on the LIVE W4 code vs a twin whose spray deposit pass is
    no-opped (== the pre-W4 conductor): every field, every hp, the event
    stream, and the RNG end-state are bit-identical, tick for tick. The
    W4 seams cost a spray-free trajectory nothing."""
    import simulation.simulation as sim_mod

    def build():
        sim = Simulation(_level(edits=[(10, 14, 2)]), seed=SEED,
                         breach_physics=bp, enable_recorder=False)
        m = Unit("M", x=3, y=9, team=0)               # k5 from config default
        z = Unit("Z", x=16, y=9, team=1)
        mid = sim.add_unit(m)
        sim.add_unit(z)
        assert m.weapon_id == "k5_carbine"            # the [marine] weapon key
        assert sim.apply_action(mid, Order(
            ORDER_FIRE, target_fx=17, target_fy=10, phase=0))
        return sim

    def run(sim, n=20):
        traj = []
        for _ in range(n):
            _step(sim)
            snap = {f: np.copy(getattr(sim.gmap, f)) for f in _FIELDS}
            snap["__heat__"] = np.copy(sim.gmap.heat)
            snap["__events__"] = repr(sim.tick_events)
            snap["__hp__"] = tuple(u.current_hp for u in sim.units)
            snap["__pos__"] = tuple((u.x, u.y) for u in sim.units)
            traj.append(snap)
        return traj, sim.rng.bit_generator.state

    traj_new, rng_new = run(build())

    saved = sim_mod.process_sprays
    sim_mod.process_sprays = lambda *a, **k: None     # the pre-W4 conductor
    try:
        traj_old, rng_old = run(build())
    finally:
        sim_mod.process_sprays = saved

    for t, (sn, so) in enumerate(zip(traj_new, traj_old)):
        for f in _FIELDS:
            assert np.array_equal(sn[f], so[f]), \
                f"tick {t}: field '{f}' diverged from the pre-W4 replica"
        assert np.array_equal(sn["__heat__"], so["__heat__"]), \
            f"tick {t}: heat diverged"
        assert sn["__events__"] == so["__events__"], f"tick {t}: events"
        assert sn["__hp__"] == so["__hp__"], f"tick {t}: hp"
        assert sn["__pos__"] == so["__pos__"], f"tick {t}: positions"
    assert rng_new == rng_old, "RNG stream moved vs the pre-W4 replica"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
