"""The W2 unified march (mechanics/03 §2) + the accuracy trinity wired live
(§3; mechanics/06 §5) — the weapons-W2 gate.

What is locked here:

  - SPEED AS DATA: a round whose speed_tiles_per_tick covers its range
    resolves in the firing tick (the k5 path — 96 t/t >= range 90); a slower
    round PERSISTS in flight (>= 3 ticks exercised) and advances by an exact
    integer Q16.16 step budget (2.5 t/t marches 2, 3, 2, 3 tiles);
  - the AIM/SNAP MODE RULE: an explicit stationary fire order uses
    spread_deg; Move & Attack auto-fire uses spread_snap_deg;
  - EXPOSURE VS COVER: entering a footprint through a cover_exposure < 1.0
    tile draws exactly one exposure uniform; a failed roll ABSORBS the shot
    into the cover tile (wall-damage chew, tracer ends there, no packet);
    a flanking approach (no cover on the entry tile) draws NOTHING — the
    RNG stream position is asserted bit-identical to cone-draws-only;
  - LAZY-DRAW STREAM STABILITY: a scripted k5 firefight (crit 0, no cover)
    is bit-identical — endpoints, hp, events, and the generator state — to
    a replica of the PRE-W2 march algorithm run on a parallel generator;
  - CRIT VS FACING: crit% = crit_chance x arc multiplier (front/flank/
    behind), predicted draw-for-draw against a parallel generator; the
    amount scales by crit_mult in exact ints; facing is deterministic
    (same seed => same facing, same events);
  - WALL CHEW: missed shots deposit ammo wall_damage on the stopping solid
    via the apply_explosion wall-HP path; chew can destroy furniture so it
    stops being cover (the widened destroy_wall gate).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_weapons_march.py -q
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
from simulation import unit_fixed, wall_fixed  # noqa: E402
from simulation import weapons as weapons_mod  # noqa: E402
from simulation.combat import BulletInFlight, chew_wall, fire_burst  # noqa: E402
from simulation.events import ShotFiredEvent, UnitHitEvent  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_FURNITURE, MAT_HULL,
)
from simulation.orders import (  # noqa: E402
    ORDER_FIRE, ORDER_MOVE_ATTACK, Order,
)
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import AmmoDef, WeaponDef  # noqa: E402

SEED = 20260705


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------
def _level(h=24, w=24, edits=()):
    """A hull-walled room in level-format v2 (codes ARE material ids):
    border hull(1), interior air(0), plus explicit (y, x, code) edits."""
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w2_march", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _test_weapon(name="w2test", speed=96.0, range_tiles=30, damage=10,
                 wall_damage=2, spread=0.0, snap=0.0, shots=1,
                 crit=0.0, crit_mult=2.0):
    """A direct-construction weapon/ammo pair (the dict-table test path).
    speed_q16 is normally derived by AmmoTable — set it here the same way."""
    w = WeaponDef(name, "projectile", ammo_family=name + "_fam",
                  spread_deg=spread, spread_snap_deg=snap,
                  range_tiles=range_tiles, shots_per_trigger=shots,
                  crit_chance=crit, crit_mult=crit_mult)
    a = AmmoDef(name + "_std", name + "_fam", "kinetic", damage=damage,
                wall_damage=wall_damage, speed_tiles_per_tick=speed)
    a.speed_q16 = unit_fixed.quantize_scalar(float(speed))
    return w, a


def _marine(x, y, name="M", team=0):
    u = Unit(name, x=x, y=y, team=team)
    u.current_hp = 1e9        # keep targets alive across bursts
    return u


def _hits(events):
    return [e for e in events if isinstance(e, UnitHitEvent)]


def _tracers(events):
    return [e for e in events if isinstance(e, ShotFiredEvent)]


# ---------------------------------------------------------------------------
# A. Speed as data — same-tick resolution + in-flight persistence
# ---------------------------------------------------------------------------
def test_k5_resolves_same_tick_and_bullets_list_stays_empty():
    """The shipped small-arm path: speed (96) >= range (90) => the burst
    fully resolves inside fire_burst; nothing persists."""
    gmap = _room()
    shooter = _marine(2, 8, "S")
    target = _marine(14, 8, "T")
    shooter.id, target.id = 1, 2
    rng = np.random.default_rng(SEED)
    shots, events, bullets = [], [], []
    fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               bullets=bullets)
    assert bullets == []                      # resolved in the firing tick
    assert len(_hits(events)) == 5            # k5 burst, 12 tiles, 3 deg cone
    assert len(_tracers(events)) == 5


def test_slow_round_persists_in_flight_across_ticks():
    """A 2 t/t round with 11 tiles to cross persists >= 3 ticks and advances
    exactly its integer budget each tick."""
    gmap = _room()
    shooter = _marine(2, 8, "S")
    target = _marine(14, 8, "T")     # footprint x in [14,16] — 11 steps away
    shooter.id, target.id = 1, 2
    w, a = _test_weapon(speed=2.0, range_tiles=20, damage=10)
    rng = np.random.default_rng(SEED)
    shots, events, bullets = [], [], []
    fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               weapon=w, ammo=a, spread_deg=0.0, bullets=bullets)

    assert len(bullets) == 1                  # still flying after the fire tick
    b = bullets[0]
    assert b.rx == 5.0 and b.ry == 9.0        # 2 exact unit steps east
    assert not _hits(events)

    # Ticks 2..5: 2 tiles each (integer budget, no fraction at speed 2.0).
    expected_rx = [7.0, 9.0, 11.0, 13.0]
    flew = 1
    for exp_rx in expected_rx:
        alive = b.advance(gmap, [shooter, target], shots, 0.0, rng,
                          events=events)
        assert alive
        assert b.rx == exp_rx and b.ry == 9.0
        flew += 1
    assert flew >= 3                          # the gate: >= 3 ticks in flight

    # Tick 6: steps onto x=14 -> the footprint -> connecting hit, flight ends.
    alive = b.advance(gmap, [shooter, target], shots, 0.0, rng, events=events)
    assert not alive
    hits = _hits(events)
    assert len(hits) == 1 and hits[0].unit_id == 2
    assert hits[0].damage == 10.0


def test_fractional_speed_budget_carries():
    """2.5 t/t marches 2, 3, 2, 3 — the Q16.16 fraction carries exactly."""
    gmap = _room()
    shooter = _marine(2, 8, "S")
    shooter.id = 1
    w, a = _test_weapon(speed=2.5, range_tiles=18, damage=10)
    rng = np.random.default_rng(SEED)
    shots, events, bullets = [], [], []
    fire_burst(gmap, [shooter], shooter, 3, 9, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               weapon=w, ammo=a, spread_deg=0.0, bullets=bullets)
    b = bullets[0]
    assert b.rx == 5.0                        # tick 1: floor(2.5) = 2 steps
    seq = []
    for _ in range(3):
        b.advance(gmap, [shooter], shots, 0.0, rng, events=events)
        seq.append(b.rx)
    # budget: .5 -> 3 steps; .0 -> 2 steps; .5 -> 3 steps
    assert seq == [8.0, 10.0, 13.0]


def test_in_flight_through_simulation_tick_slot():
    """End-to-end: a slow round fired via ORDER_FIRE persists on
    Simulation.bullets across >= 3 steps and connects in slot 2."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    shooter = Unit("S", x=2, y=8, team=0)
    target = Unit("T", x=14, y=8, team=0)
    s_id = sim.add_unit(shooter)
    sim.add_unit(target)
    # Inject a slow test weapon into the (shared) tables built at sim
    # construction; one burst only: the cadence gate (500 ticks) lets the
    # first trigger through (tick 0 - last_fire_tick -999 = 999 >= 500) and
    # blocks every re-fire inside the test window.
    w, a = _test_weapon(name="w2_slug", speed=2.0, range_tiles=20, damage=10)
    w.rof_interval_ticks = 500
    sim.weapons_tables.weapons.by_name[w.name] = w
    sim.weapons_tables.ammo.by_name[a.name] = a
    shooter.weapon_id = w.name
    try:
        assert sim.apply_action(s_id, Order(ORDER_FIRE, target_fx=15,
                                            target_fy=9, phase=0))
        sim.set_paused(False)
        sim.step()                            # tick 0: fires, 2 tiles flown
        assert len(sim.bullets) == 1
        in_flight_steps = 1
        hit_events = []
        for _ in range(8):
            sim.set_paused(False)
            sim.step()
            hit_events += _hits(sim.tick_events)
            if not sim.bullets:
                break
            in_flight_steps += 1
        assert in_flight_steps >= 3           # persisted across >= 3 ticks
        assert len(hit_events) == 1
        assert hit_events[0].damage == 10.0   # marine target: no site rule
    finally:
        weapons_mod.rebuild_tables()          # drop the injected rows


# ---------------------------------------------------------------------------
# B. The aim/snap mode rule
# ---------------------------------------------------------------------------
def test_fire_order_aims_and_move_attack_snaps():
    """spread_deg=0 / spread_snap_deg=25: the explicit fire order flies the
    exact axis (every tracer y == 9.0 bitwise); Move & Attack auto-fire draws
    from the snap cone (first tracer y != 9.0, draw predicted non-zero)."""
    w, a = _test_weapon(name="w2_aim", speed=96.0, range_tiles=30,
                        damage=10, spread=0.0, snap=25.0, shots=3)

    # (a) Explicit stationary fire order -> AIMED cone (0 deg).
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    shooter = Unit("S", x=2, y=8, team=0)
    target = Unit("T", x=14, y=8, team=0)
    s_id = sim.add_unit(shooter)
    sim.add_unit(target)
    sim.weapons_tables.weapons.by_name[w.name] = w
    sim.weapons_tables.ammo.by_name[a.name] = a
    shooter.weapon_id = w.name
    try:
        assert sim.apply_action(s_id, Order(ORDER_FIRE, target_fx=15,
                                            target_fy=9, phase=0))
        sim.set_paused(False)
        sim.step()
        aimed = _tracers(sim.tick_events)
        assert len(aimed) == 3
        for t in aimed:
            assert t.to_tile[1] == 9.0        # exact axis: cone scaled to 0.0
    finally:
        weapons_mod.rebuild_tables()

    # (b) Move & Attack -> SNAP cone (25 deg). Same seed, same geometry
    # (a zombie enemy so auto-fire acquires it; it acts after shooting).
    sim2 = Simulation(_level(), seed=SEED, breach_physics=bp,
                      enable_recorder=False)
    shooter2 = Unit("S", x=2, y=8, team=0)
    zombie = Unit("Z", x=14, y=8, team=1)
    s2_id = sim2.add_unit(shooter2)
    sim2.add_unit(zombie)
    sim2.weapons_tables.weapons.by_name[w.name] = w
    sim2.weapons_tables.ammo.by_name[a.name] = a
    shooter2.weapon_id = w.name
    try:
        # Zero-length Move & Attack (terrain-passable — own tile).
        assert sim2.apply_action(s2_id, Order(ORDER_MOVE_ATTACK, target_fx=2,
                                              target_fy=8, phase=0))
        # Predict the first snap draw on a parallel generator: the sim's rng
        # is fresh-seeded; the burst is its first consumer this tick.
        parallel = np.random.default_rng(SEED)
        cone = math.radians(25.0)
        first_draw = float(parallel.uniform(-cone, cone))
        assert abs(first_draw) > 1e-3         # comfortably off-axis
        sim2.set_paused(False)
        sim2.step()
        snapped = _tracers(sim2.tick_events)
        assert len(snapped) == 3
        assert snapped[0].to_tile[1] != 9.0   # the snap cone moved the round
    finally:
        weapons_mod.rebuild_tables()


# ---------------------------------------------------------------------------
# C. Exposure vs cover
# ---------------------------------------------------------------------------
def _cover_scene():
    """Furniture at (y=9, x=13), directly on the approach to a target whose
    footprint starts at x=14 — the entry tile for a west->east shot."""
    gmap = _room(edits=[(9, 13, MAT_FURNITURE)])
    shooter = _marine(2, 8, "S")
    target = _marine(14, 8, "T")
    shooter.id, target.id = 1, 2
    return gmap, shooter, target


def test_exposure_roll_pass_and_fail_with_absorption_chew():
    """Predicted draw-for-draw: pass => packet lands, crate untouched;
    fail => NO packet, the crate eats the round's wall_damage, the tracer
    ends on the cover tile."""
    w, a = _test_weapon(speed=96.0, range_tiles=30, damage=10, wall_damage=2)
    exposure = None
    saw_pass = saw_fail = False
    for seed in range(40):
        gmap, shooter, target = _cover_scene()
        if exposure is None:
            from simulation.attack_resolver import cover_exposure_at
            exposure = cover_exposure_at(gmap, 9, 13)
            assert exposure == float(np.float32(0.55))
        hp_before = int(gmap.wall_hp[9, 13])
        rng = np.random.default_rng(seed)
        parallel = np.random.default_rng(seed)
        cone = math.radians(0.0)
        float(parallel.uniform(-cone, cone))          # the cone draw
        connects = float(parallel.uniform(0.0, 1.0)) < exposure
        shots, events = [], []
        fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
                   tick=0, shots=shots, real_time=0.0, rng=rng,
                   events=events, weapon=w, ammo=a, spread_deg=0.0)
        # The march consumed exactly cone + exposure draws — no more.
        assert rng.bit_generator.state == parallel.bit_generator.state
        hits = _hits(events)
        tracer = _tracers(events)[0]
        if connects:
            saw_pass = True
            assert len(hits) == 1 and hits[0].unit_id == 2
            assert int(gmap.wall_hp[9, 13]) == hp_before      # crate untouched
            assert tracer.hit_target_id == 2
        else:
            saw_fail = True
            assert hits == []                                  # absorbed
            assert int(gmap.wall_hp[9, 13]) == \
                hp_before - wall_fixed.quantize_scalar(2.0)    # the chew
            assert int(tracer.to_tile[0]) == 13                # ends on cover
            assert tracer.hit_target_id is None
    assert saw_pass and saw_fail   # 0.55 must show both branches over 40 seeds


def test_directional_bypass_flank_draws_nothing():
    """Approach from the south — the footprint entry tile is open air, so the
    crate at (9,13) never matters: ZERO exposure draws (stream position
    bit-identical to cone-draws-only), and the hit lands at full exposure."""
    gmap = _room(edits=[(9, 13, MAT_FURNITURE)])
    shooter = _marine(14, 18, "S")      # center (15, 19), due south of target
    target = _marine(14, 8, "T")        # footprint y in [8,10]
    shooter.id, target.id = 1, 2
    w, a = _test_weapon(speed=96.0, range_tiles=30, damage=10, wall_damage=2)
    rng = np.random.default_rng(SEED)
    parallel = np.random.default_rng(SEED)
    cone = math.radians(0.0)
    float(parallel.uniform(-cone, cone))              # cone draw ONLY
    shots, events = [], []
    fire_burst(gmap, [shooter, target], shooter, 15, 19, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               weapon=w, ammo=a, spread_deg=0.0)
    assert rng.bit_generator.state == parallel.bit_generator.state
    hits = _hits(events)
    assert len(hits) == 1 and hits[0].unit_id == 2    # full-exposure connect
    assert int(gmap.wall_hp[9, 13]) == wall_fixed.quantize_scalar(30.0)


# ---------------------------------------------------------------------------
# Lazy-draw stream stability: bit-identical to the PRE-W2 march
# ---------------------------------------------------------------------------
def _prew2_burst_replica(gmap, units, shooter, fx1, fy1, fx2, fy2, rng):
    """The shipped (pre-W2) fire_burst algorithm, verbatim: k5 row numbers,
    one cone draw per bullet, kit-trig steps, the same stop rules — computing
    endpoints + hit targets WITHOUT mutating anything. The W2 march must
    reproduce this bit-for-bit when no cover intervenes and crit_chance == 0
    (the lazy-roll rule)."""
    tables = weapons_mod.get_tables()
    k5 = tables.weapons.by_name["k5_carbine"]
    cone = math.radians(k5.spread_deg)
    base_angle = unit_fixed.atan2_rad(fy2 - fy1, fx2 - fx1)
    h, w = gmap.material.shape
    out = []
    for _ in range(k5.shots_per_trigger):
        angle = base_angle + float(rng.uniform(-cone, cone))
        step_x = unit_fixed.cos_rad(angle)
        step_y = unit_fixed.sin_rad(angle)
        rx, ry = float(fx1), float(fy1)
        hit_unit = None
        for _step in range(int(k5.range_tiles)):
            rx += step_x
            ry += step_y
            ix, iy = int(rx), int(ry)
            if 0 <= iy < h and 0 <= ix < w:
                if gmap.solid[iy, ix]:
                    break
            else:
                break
            for e in units:
                if e is shooter or not e.alive:
                    continue
                if (e.tile_x <= ix < e.tile_x + e.footprint
                        and e.tile_y <= iy < e.tile_y + e.footprint):
                    hit_unit = e
                    break
            if hit_unit:
                break
        out.append((rx, ry, hit_unit))
    return out


def test_lazy_draws_k5_bit_identical_to_prew2_firefight():
    """A scripted k5 burst on a zombie (crit 0, no cover): endpoints, hp
    deltas, event stream, and the generator END STATE all equal the pre-W2
    replica on a parallel generator — the ONLY consumers are the 5 cone
    draws (the lazy-roll rule made structural)."""
    gmap = _room()
    shooter = _marine(2, 8, "S")
    zombie = Unit("Z", x=14, y=8, team=1)
    zombie.current_hp = 1e9
    shooter.id, zombie.id = 1, 2
    units = [shooter, zombie]

    rng = np.random.default_rng(SEED)
    parallel = np.random.default_rng(SEED)
    replica = _prew2_burst_replica(gmap, units, shooter, 3, 9, 15, 9, parallel)

    shots, events = [], []
    fire_burst(gmap, units, shooter, 3, 9, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events)

    # Stream: exactly the 5 cone draws, nothing else (crit 0, no cover).
    assert rng.bit_generator.state == parallel.bit_generator.state
    # Endpoints bitwise equal, hit-for-hit.
    tracers = _tracers(events)
    assert len(tracers) == len(replica) == 5
    n_hits_replica = 0
    for t, (rx, ry, hit) in zip(tracers, replica):
        assert t.to_tile == (rx, ry)
        if hit is not None:
            n_hits_replica += 1
            assert t.hit_target_id == hit.id
    # HP: the shipped zombie site rule, per connecting bullet.
    per_bullet = unit_fixed.quantize_hp_delta(
        int(10 * CFG.zombie.bullet_damage_multiplier))
    expected_hp = 1e9
    for _ in range(n_hits_replica):
        expected_hp -= per_bullet
    assert zombie.current_hp == expected_hp
    # Firing set the shooter's facing to the aim bearing (kit, y-up).
    assert shooter.facing == unit_fixed.atan2_rad(-(9 - 9), 15 - 3) == 0.0


# ---------------------------------------------------------------------------
# D. Crit vs facing
# ---------------------------------------------------------------------------
def test_crit_arc_multipliers_predicted_draw_for_draw():
    """crit_chance 0.15: front x1 (p=.15), flank x2 (p=.3), behind x4 (p=.6)
    — each seed's crit predicted on a parallel generator; a crit doubles the
    packet amount (scale_half_away(10, 2.0) == 20) before mitigation."""
    w, a = _test_weapon(speed=96.0, range_tiles=30, damage=10,
                        crit=0.15, crit_mult=2.0)
    arcs = {
        math.pi:      1.0,   # target faces WEST, toward the shooter: front
        math.pi / 2:  2.0,   # faces NORTH: flank
        0.0:          4.0,   # faces EAST, away: behind
    }
    for facing, mult in arcs.items():
        saw_crit = saw_normal = False
        for seed in range(30):
            gmap = _room()
            shooter = _marine(2, 8, "S")
            target = _marine(14, 8, "T")
            shooter.id, target.id = 1, 2
            target.facing = facing
            rng = np.random.default_rng(seed)
            parallel = np.random.default_rng(seed)
            cone = math.radians(0.0)
            float(parallel.uniform(-cone, cone))            # cone draw
            crit = float(parallel.uniform(0.0, 1.0)) < 0.15 * mult
            shots, events = [], []
            fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
                       tick=0, shots=shots, real_time=0.0, rng=rng,
                       events=events, weapon=w, ammo=a, spread_deg=0.0)
            assert rng.bit_generator.state == parallel.bit_generator.state
            hits = _hits(events)
            assert len(hits) == 1
            expected = 20.0 if crit else 10.0
            assert hits[0].damage == expected
            saw_crit |= crit
            saw_normal |= not crit
        assert saw_crit and saw_normal, \
            f"facing {facing}: both outcomes must appear over 30 seeds"


def test_crit_chance_zero_draws_nothing():
    """The lazy-roll rule at the crit site: crit_chance == 0 (every shipped
    weapon) never touches the stream beyond the cone draw."""
    gmap = _room()
    shooter = _marine(2, 8, "S")
    target = _marine(14, 8, "T")
    shooter.id, target.id = 1, 2
    w, a = _test_weapon(speed=96.0, range_tiles=30, damage=10, crit=0.0)
    rng = np.random.default_rng(SEED)
    parallel = np.random.default_rng(SEED)
    float(parallel.uniform(-math.radians(0.0), math.radians(0.0)))
    shots, events = [], []
    fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               weapon=w, ammo=a, spread_deg=0.0)
    assert rng.bit_generator.state == parallel.bit_generator.state
    assert _hits(events)[0].damage == 10.0


def test_facing_determinism_same_seed_same_firefight():
    """Same seed twice: identical facings, identical event streams."""
    w, a = _test_weapon(speed=96.0, range_tiles=30, damage=10,
                        crit=0.15, crit_mult=2.0)

    def run():
        gmap = _room()
        shooter = _marine(2, 8, "S")
        target = _marine(14, 8, "T")
        shooter.id, target.id = 1, 2
        target.facing = 0.0                    # behind: p = .6
        rng = np.random.default_rng(7)
        shots, events = [], []
        fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
                   tick=0, shots=shots, real_time=0.0, rng=rng,
                   events=events, weapon=w, ammo=a, spread_deg=0.0)
        return (shooter.facing, target.facing, target.current_hp,
                [(type(e).__name__, getattr(e, "damage", None)) for e in events])

    assert run() == run()


# ---------------------------------------------------------------------------
# Wall chew (missed shots are not deleted)
# ---------------------------------------------------------------------------
def test_missed_shots_chew_the_stopping_wall():
    """5 zero-spread rounds into the east wall: the tile eats exactly
    5 x quantize(wall_damage) and survives (hull 300 HP)."""
    gmap = _room()
    shooter = _marine(2, 8, "S")
    shooter.id = 1
    w, a = _test_weapon(speed=96.0, range_tiles=40, damage=10,
                        wall_damage=2, shots=5)
    rng = np.random.default_rng(SEED)
    shots, events = [], []
    fire_burst(gmap, [shooter], shooter, 3, 9, 22, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               weapon=w, ammo=a, spread_deg=0.0)
    q300 = wall_fixed.quantize_scalar(300.0)
    q2 = wall_fixed.quantize_scalar(2.0)
    assert int(gmap.wall_hp[9, 23]) == q300 - 5 * q2
    assert int(gmap.material[9, 23]) == MAT_HULL          # still standing


def test_chew_destroys_furniture_so_it_stops_being_cover():
    """The widened destroy_wall gate: a chew that zeroes a (non-solid)
    furniture tile converts it to air — the crate stops being cover."""
    from simulation.attack_resolver import cover_exposure_at
    gmap = _room(edits=[(9, 13, MAT_FURNITURE)])
    assert cover_exposure_at(gmap, 9, 13) < 1.0
    chew_wall(gmap, 9, 13, 40)              # > the crate's 30 HP
    assert int(gmap.material[9, 13]) == MAT_AIR
    assert not bool(gmap.solid[9, 13])
    assert not bool(gmap.is_vacuum[9, 13])
    assert cover_exposure_at(gmap, 9, 13) == 1.0          # concealment gone


def test_marching_round_requires_positive_speed():
    """Misauthored projectile ammo (speed 0) fails LOUDLY at fire time."""
    import pytest
    w, a = _test_weapon(speed=0.0)
    a.speed_q16 = 0
    with pytest.raises(ValueError, match="speed_tiles_per_tick"):
        BulletInFlight(None, -1, w, a, 3, 9, 0.0, 1.0, 0.0)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
