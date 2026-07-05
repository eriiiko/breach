"""The W3 ammo-economy + GL-6 launcher gate (mechanics/03 §4).

What is locked here:

  - GL-6 DETONATE-AT-STOP, all three stop classes: first SOLID (the blast
    centres on the wall tile, like a charge on a door), first UNIT FOOTPRINT
    (the entry tile — and NO direct-hit packet: the ONLY unit damage is the
    blast), and MAX RANGE (mid-air airburst at the march's final tile);
  - MAG / RELOAD CADENCE: 6 triggers on a mag, emptying starts the
    auto-reload stall (reload_seconds exactly), the gate blocks mid-reload,
    the first trigger past the stall refills and fires — proven at the gate
    level AND through process_shooting AND through a full Simulation run;
  - DORMANCY: mag_size == 0 (the k5 and every pre-W3 weapon) never touches
    the new unit state — current_mag stays None, reload_done_tick stays -1;
  - the round boundary tops the magazine off (the v1 rule; also the
    tick-rewind correctness twin of last_fire_tick = -999).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_ammo_economy.py -q
"""
from __future__ import annotations

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
from simulation import wall_fixed  # noqa: E402
from simulation.combat import fire_burst, mag_gate, mag_spend, process_shooting  # noqa: E402
from simulation.events import ExplosionEvent, ShotFiredEvent, UnitHitEvent  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_HULL  # noqa: E402
from simulation.orders import ORDER_FIRE, Order  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402
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
    return LevelData(name="w3_econ", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _fire_gl6(gmap, units, shooter, fx2, fy2, *, queue, rng, events,
              bullets, shots):
    """One zero-spread GL-6 trigger from the shooter's centre, using the
    REAL config rows (gl6_revolver -> 40mm_frag -> frag_standard)."""
    t = get_tables()
    gl6 = t.weapons.by_name["gl6_revolver"]
    ammo = t.ammo_for_weapon(gl6)
    assert ammo.name == "40mm_frag" and ammo.damage == 0
    fire_burst(gmap, units, shooter, shooter.center_tile_x(),
               shooter.center_tile_y(), fx2, fy2, tick=0, shots=shots,
               real_time=0.0, rng=rng, events=events, weapon=gl6, ammo=ammo,
               spread_deg=0.0, bullets=bullets, queue=queue)


def _advance_to_detonation(gmap, units, bullets, queue, rng, events, shots,
                           max_ticks=64):
    """March the in-flight round to its stop (the sim slot-2 loop shape)."""
    for _ in range(max_ticks):
        if not bullets:
            return
        survivors = []
        for b in bullets:
            if b.advance(gmap, units, shots, 0.0, rng, events=events,
                         queue=queue):
                survivors.append(b)
        bullets[:] = survivors


def _explosions(events):
    return [e for e in events if isinstance(e, ExplosionEvent)]


# ---------------------------------------------------------------------------
# GL-6 detonate-at-stop — the three stop classes
# ---------------------------------------------------------------------------
def test_gl6_detonates_on_the_stopping_wall():
    """The round stops ON the east hull tile and the frag_standard blast
    centres there: kind='shell', the wall eats the payload's 200 x falloff
    (NOT bullet chew — 40 mm wall_damage is 0), inner-radius neighbours chew
    with exact Q16.16 falloff."""
    gmap = _room()
    shooter = Unit("S", x=2, y=8, team=0)
    shooter.id = 1
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    events, bullets, shots = [], [], []
    _fire_gl6(gmap, [shooter], shooter, 23, 9, queue=queue, rng=rng,
              events=events, bullets=bullets, shots=shots)
    assert len(bullets) == 1                       # 1.25 t/t: it flies
    _advance_to_detonation(gmap, [shooter], bullets, queue, rng, events, shots)
    assert bullets == []

    ex = _explosions(events)
    assert len(ex) == 1
    assert ex[0].kind == "shell" and ex[0].radius == 5
    assert ex[0].pos == (23, 9)                    # the wall tile itself
    queue.flush(gmap, rng)
    # The stopped-on hull tile: falloff 1.0 -> exactly quantize(200) off 300.
    q300 = wall_fixed.quantize_scalar(300.0)
    assert int(gmap.wall_hp[9, 23]) == q300 - wall_fixed.quantize_scalar(200.0)
    assert int(gmap.material[9, 23]) == MAT_HULL   # 100 HP left — standing
    # A border tile 3 up the wall: falloff 1 - 3/5 = 0.4 -> quantize(80).
    assert int(gmap.wall_hp[12, 23]) == q300 - wall_fixed.quantize_scalar(80.0)
    # The blast deposited a wave (the disc skips solids; interior cells got it).
    assert gmap.wave_source.any()


def test_gl6_detonates_on_unit_footprint_no_direct_hit_packet():
    """The round detonates at the FOOTPRINT ENTRY tile: the target takes
    ONLY blast damage (source 'explosion' — never a 0-damage direct-hit
    packet), computed by the exact apply_blast_damage falloff."""
    gmap = _room()
    shooter = Unit("S", x=2, y=8, team=0)
    target = Unit("T", x=14, y=8, team=0)
    shooter.id, target.id = 1, 2
    target.current_hp = 1e9
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    events, bullets, shots = [], [], []
    _fire_gl6(gmap, [shooter, target], shooter, 15, 9, queue=queue, rng=rng,
              events=events, bullets=bullets, shots=shots)
    _advance_to_detonation(gmap, [shooter, target], bullets, queue, rng,
                           events, shots)

    ex = _explosions(events)
    assert len(ex) == 1 and ex[0].kind == "shell"
    det_x, det_y = ex[0].pos
    assert det_x == 14                              # the entry tile (x in [14..16])
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    assert hits, "the blast must reach the target"
    assert all(h.source == "explosion" for h in hits)   # NO 'bullet' packet
    # Exact blast amount at the target's centre (the apply_blast_damage form).
    frag = get_tables().payloads.by_name["frag_standard"]
    dist = float(np.sqrt((target.center_tile_x() - det_x) ** 2
                         + (target.center_tile_y() - det_y) ** 2))
    expected = int(frag.unit_damage * (1.0 - dist / frag.radius))
    assert expected >= CFG.combat.blast_damage_threshold
    target_hits = [h for h in hits if h.unit_id == 2]
    assert len(target_hits) == 1 and target_hits[0].damage == float(expected)
    assert target.current_hp == 1e9 - expected


def test_gl6_airbursts_at_max_range():
    """No wall, no unit for 40+ tiles: the round expires at its range cap
    and the payload executes mid-air at the march's final tile (origin x 3
    + 40 exact 1.0-steps east = x 43)."""
    gmap = _room(h=20, w=60)
    shooter = Unit("S", x=2, y=8, team=0)
    shooter.id = 1
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    events, bullets, shots = [], [], []
    _fire_gl6(gmap, [shooter], shooter, 55, 9, queue=queue, rng=rng,
              events=events, bullets=bullets, shots=shots)
    _advance_to_detonation(gmap, [shooter], bullets, queue, rng, events, shots)
    ex = _explosions(events)
    assert len(ex) == 1 and ex[0].kind == "shell"
    assert ex[0].pos == (43, 9)                     # 3 + range 40, exact steps
    queue.flush(gmap, rng)
    assert gmap.wave_source.any()                   # the airburst deposited


# ---------------------------------------------------------------------------
# Mag / reload cadence
# ---------------------------------------------------------------------------
def test_mag_gate_and_spend_cadence_exact_ticks():
    """The GL-6 numbers at the gate level: 6 triggers empty the cylinder;
    emptying starts the 72-tick (3.0 s @ 24 tps) stall; the gate blocks
    inside it and refills at the first attempt past it. The stall between
    shot 6 and shot 7 is exactly reload_seconds."""
    gl6 = get_tables().weapons.by_name["gl6_revolver"]
    assert gl6.rof_interval_ticks == 29 and gl6.reload_ticks == 72
    u = Unit("S", x=2, y=8, team=0)
    assert u.current_mag is None and u.reload_done_tick == -1

    fire_ticks = []
    for i in range(6):
        t = i * 29
        assert mag_gate(u, gl6, t)
        mag_spend(u, gl6, t)
        fire_ticks.append(t)
    assert u.current_mag == 0
    empty_tick = fire_ticks[-1]                     # 145
    assert u.reload_done_tick == empty_tick + 72    # the stall starts NOW

    # Mid-reload attempts (the rof gate would allow them): blocked.
    assert not mag_gate(u, gl6, empty_tick + 29)
    assert not mag_gate(u, gl6, empty_tick + 71)
    # First attempt past the stall: refill + fire — exactly 3.0 s after
    # the emptying trigger.
    resume = empty_tick + 72
    assert mag_gate(u, gl6, resume)
    assert u.current_mag == 6
    assert (resume - empty_tick) == int(round(
        gl6.reload_seconds * CFG.clock.ticks_per_second))
    mag_spend(u, gl6, resume)
    assert u.current_mag == 5


def test_mag_cadence_through_process_shooting():
    """process_shooting honours the gate: an emptied GL-6 does NOT fire
    mid-reload and fires again once the stall passes (fire orders in both
    phases keep the trigger held across the phase split)."""
    t = get_tables()
    gl6 = t.weapons.by_name["gl6_revolver"]
    gmap = _room()
    u = Unit("S", x=2, y=8, team=0)
    u.id = 1
    u.weapon_id = "gl6_revolver"
    u.orders = [Order(ORDER_FIRE, target_fx=15, target_fy=9, phase=0),
                Order(ORDER_FIRE, target_fx=15, target_fy=9, phase=1)]
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    shots, events, bullets = [], [], []

    fired_at = []
    for tick in range(0, 226):
        n_before = len([e for e in events if isinstance(e, ShotFiredEvent)])
        process_shooting(gmap, [u], tick, shots, 0.0, rng, events=events,
                         bullets=bullets, queue=queue)
        # Drain in-flight rounds so their detonations don't clutter events;
        # count only the FIRING tick's tracer.
        n_after = len([e for e in events if isinstance(e, ShotFiredEvent)])
        if n_after > n_before:
            fired_at.append(tick)
        _advance_to_detonation(gmap, [u], bullets, queue, rng, events, shots)

    # 6 shots at the rof cadence, then the 72-tick reload stall, then #7:
    # 145 + 72 = 217 (rof would have allowed 174 — the stall dominates).
    assert fired_at == [0, 29, 58, 87, 116, 145, 217]
    assert u.current_mag == 5                        # shot 7 came off a fresh mag


def test_k5_mag_size_zero_never_touches_the_state():
    """DORMANCY: the shipped k5 (mag_size 0) fires bursts forever without
    binding a magazine — the W3 fields stay at their virgin values, so no
    shipped scenario can move (the golden-safety argument in unit form)."""
    gmap = _room()
    u = Unit("S", x=2, y=8, team=0)
    u.id = 1
    u.orders = [Order(ORDER_FIRE, target_fx=15, target_fy=9, phase=0)]
    rng = np.random.default_rng(SEED)
    shots, events = [], []
    for tick in range(0, 40):
        process_shooting(gmap, [u], tick, shots, 0.0, rng, events=events)
    assert [e for e in events if isinstance(e, ShotFiredEvent)]  # it fired
    assert u.current_mag is None
    assert u.reload_done_tick == -1


def test_round_boundary_resets_mag_state():
    """_end_round tops the magazine off and clears the reload stall (the
    tick counter rewinds, so a carried reload_done_tick would wrongly stall
    into the next round — the last_fire_tick = -999 twin)."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=3, y=3, team=0)
    sim.add_unit(m)
    m.current_mag = 2
    m.reload_done_tick = 500
    sim._end_round()
    assert m.current_mag is None
    assert m.reload_done_tick == -1
    assert m.last_fire_tick == -999


def test_gl6_full_chain_through_the_simulation():
    """E2E: a GL-6 marine's fire order sends a 40 mm round downrange in slot
    2 across ~10 ticks and the shell detonates (kind='shell') — order ->
    mag spend -> march -> stop -> executor, all inside Simulation.step."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=2, y=8, team=0)
    mid = sim.add_unit(m)
    m.weapon_id = "gl6_revolver"
    assert sim.apply_action(mid, Order(ORDER_FIRE, target_fx=15, target_fy=9,
                                       phase=0))
    detonated = []
    sim.set_paused(False)
    for _ in range(24):
        sim.set_paused(False)
        sim.step()
        detonated += [e for e in sim.tick_events
                      if isinstance(e, ExplosionEvent) and e.kind == "shell"]
        if detonated:
            break
    assert detonated, "the 40 mm shell must detonate within a second"
    assert m.current_mag == 5                       # one round spent
    assert weapons_mod.get_tables().weapons.by_name["gl6_revolver"].mag_size == 6


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
