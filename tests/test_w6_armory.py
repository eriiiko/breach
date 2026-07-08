"""The W6 gate: the full armory as data + meter-based ranges (mechanics/03).

What is locked here:

  - METER -> TILE CONVERSION pinned: range_m / tile_size_m, round-half-up,
    max(1, ...), quantized ONCE at table build (engine/14 door 2) — proven at
    TWO tile sizes (the 1.0 m/tile pinned-test convention and the 0.333
    m/tile playground), through the dict-table path, the real config, AND
    the Simulation facade binding (gmap.tile_size_m); authoring range_m and
    range_tiles together is loud;
  - THE ARMORY loads + validates: every mechanics/03 §6 row present with its
    quoted numbers (spread / trigger / damage / speed / loudness), the
    default_ammo selection seam resolves (P12 vs MP-11 sharing 9mm, Lance-5
    drawing the heavy cell while the Lance-3 keeps its standard), and the
    W3 row-identity rule holds for the incendiary pair;
  - JACKHAMMER-8 pellets: shots_per_trigger == 8 == tracers per trigger ==
    cone draws per trigger (door-4 stream pinned draw-for-draw);
  - PLASMA: direct-hit HEAT packet (damage 40, the W6 both-halves rule) AND
    detonate-at-stop through the payload executor — splash payload row
    object identity, the one-shot heat splash deposit, the render-only
    ProjectileGlowEvent stream while in flight, event kind "shell";
  - SPRAY JET EVENT: emitted once per DEPOSITING tick (kind "flame" /
    "miasma"), absent on spray-free ticks, and NOT part of the synced event
    digest surface;
  - WEAPON-CYCLE through the facade: sim.debug_cycle_weapon walks every
    triggerable row in config order (LOBBED/PLACED skipped), resets
    mag/burst state, refuses zombies — and a cycled weapon actually FIRES;
  - DRAGON-7 ignition-reach derivation pinned at the new scale (2400 ->
    reach 31 tiles >= the playground's 30; the Dragon-9's 4800 -> 62 >= 60);
  - DORMANCY REPLICA: the canonical A/B scenario's 30-tick aggregate digest
    still equals the pinned golden 07c3f370... AND the sim RNG end-state is
    the untouched fresh-seed state (the scenario draws ZERO randomness —
    any W6 code sneaking a draw or a field write trips this).

Run:
    C:/Users/steen/miniconda3/python.exe -m pytest tests/test_w6_armory.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG, ticks_from_seconds  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.combat import fire_burst  # noqa: E402
from simulation.events import (  # noqa: E402
    ExplosionEvent, ProjectileGlowEvent, ShotFiredEvent, SprayJetEvent,
    UnitHitEvent,
)
from simulation.gamemap import GameMap  # noqa: E402
from simulation.orders import ORDER_FIRE, Order  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import (  # noqa: E402
    FIRE_ORDER_ARCHETYPES, WeaponsTables, get_tables,
)

SEED = 20260707

# The §6 armory as data — every row W6 owes, present and triggerable.
W6_WEAPON_ROWS = {
    "p12_whisper", "mp11_pdw", "lr50", "jackhammer_8", "lance_5",
    "sunspot", "helios", "dragon_9_heavy",
}


# ---------------------------------------------------------------------------
# Scaffolding (the test_spray_weapons shape)
# ---------------------------------------------------------------------------
def _level(h=24, w=24, edits=(), tile_size_m=1.0):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w6_armory", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=tile_size_m, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _marine(name, x, y, weapon=None, team=0, uid=1):
    u = Unit(name, x=x, y=y, team=team)
    u.id = uid
    if weapon is not None:
        u.weapon_id = weapon
    return u


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _dict_tables(tile_size_m, weapon_extra=None):
    """Minimal dict-table bundle (the GasTable test-config path) with one
    ranged weapon authored in METERS."""
    weapon = {"archetype": "projectile", "ammo_family": "w6f",
              "range_m": 10.0}
    if weapon_extra:
        weapon.update(weapon_extra)
    return WeaponsTables(
        {"w6gun": weapon},
        {"w6round": {"family": "w6f", "dtype": "kinetic", "damage": 1,
                     "speed_tiles_per_tick": 96.0}},
        {},
        CFG.clock.ticks_per_second,
        tile_size_m=tile_size_m,
    )


# ---------------------------------------------------------------------------
# 1. Meter -> tile conversion: pinned at two tile sizes, quantize-once
# ---------------------------------------------------------------------------
def test_meter_conversion_pinned_at_two_tile_sizes():
    """range_tiles = max(1, int(range_m / tile_size_m + 0.5)) — one IEEE
    divide + round-half-up, derived at table build. 10 m is 10 tiles in
    the 1.0 m/tile test worlds and 30 tiles on the 0.333 m/tile
    playground grid (10 / 0.333 = 30.03 -> 30)."""
    w1 = _dict_tables(1.0).weapons.by_name["w6gun"]
    assert w1.range_tiles == 10 and isinstance(w1.range_tiles, int)
    assert w1.range_m == 10.0

    w3 = _dict_tables(0.333).weapons.by_name["w6gun"]
    assert w3.range_tiles == 30 and isinstance(w3.range_tiles, int)


def test_meter_conversion_rounding_and_floor():
    """Round-half-up (3 m at 2 m/tile = 1.5 -> 2 tiles) and the >= 1 floor
    (0.4 m at 1 m/tile -> 1 tile, never 0: an authored weapon always
    reaches at least the adjacent tile)."""
    t = _dict_tables(2.0, weapon_extra={"range_m": 3.0})
    assert t.weapons.by_name["w6gun"].range_tiles == 2
    t = _dict_tables(1.0, weapon_extra={"range_m": 0.4})
    assert t.weapons.by_name["w6gun"].range_tiles == 1


def test_meter_conversion_is_quantized_once_at_build():
    """The quantize-once rule (door 2): range_tiles is a plain stored int,
    derived at build — mutating the table's tile size afterwards changes
    NOTHING (no consumer re-derives per tick)."""
    t = _dict_tables(0.333)
    w = t.weapons.by_name["w6gun"]
    before = w.range_tiles
    t.tile_size_m = 999.0
    t.weapons.tile_size_m = 999.0
    assert w.range_tiles == before == 30


def test_authoring_both_range_columns_is_loud():
    with pytest.raises(ValueError, match="ambiguous"):
        _dict_tables(1.0, weapon_extra={"range_tiles": 12})


def test_real_config_ranges_pin_the_compatibility_rule():
    """The pinned scenarios' worlds are 1.0 m/tile: every pre-W6 weapon's
    effective tile range there is BIT-IDENTICAL to its old range_tiles
    (k5 90, lance_3 60, gl6 40, miasma 7). The playground (0.333) derives
    3x. The ONE deliberate exception: dragon_7 = 10 m (Erik's rescale;
    was 8 tiles)."""
    t1 = WeaponsTables.from_config()                      # 1.0 default
    tp = WeaponsTables.from_config(tile_size_m=0.333)     # the playground
    for name, old_tiles in (("k5_carbine", 90), ("lance_3", 60),
                            ("gl6_revolver", 40), ("miasma_vent", 7)):
        assert t1.weapons.by_name[name].range_tiles == old_tiles, name
    assert t1.weapons.by_name["dragon_7"].range_tiles == 10   # the exception
    assert tp.weapons.by_name["k5_carbine"].range_tiles == 270
    assert tp.weapons.by_name["dragon_7"].range_tiles == 30
    assert tp.weapons.by_name["dragon_9_heavy"].range_tiles == 60


def test_simulation_binds_the_level_tile_size():
    """The facade path: Simulation derives the ranges with ITS level's
    tile_size_m (0.5 here -> k5 = 180 tiles), and the shared module tables
    rebuild along with it."""
    sim = Simulation(_level(tile_size_m=0.5), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    assert sim.gmap.tile_size_m == 0.5
    assert sim.weapons_tables.weapons.by_name["k5_carbine"].range_tiles == 180
    assert get_tables() is sim.weapons_tables
    # Restore the module-global convention for the tests that follow.
    Simulation(_level(tile_size_m=1.0), seed=SEED, breach_physics=None,
               enable_recorder=False)


# ---------------------------------------------------------------------------
# 2. The armory loads + validates (the §6 table pinned)
# ---------------------------------------------------------------------------
def test_every_w6_armory_row_loads_and_is_triggerable():
    t = get_tables()
    assert W6_WEAPON_ROWS <= set(t.weapons.by_name)
    for name in W6_WEAPON_ROWS:
        w = t.weapons.by_name[name]
        assert w.archetype in FIRE_ORDER_ARCHETYPES, name
        assert w.range_tiles >= 1, name
        a = t.ammo_for_weapon(w)          # resolves (default_ammo or family)
        assert a.family == w.ammo_family, name


def test_armory_numbers_pin_the_section6_table():
    """Spot-pin the §6 armory quotes onto the loaded rows."""
    t = get_tables()
    tps = CFG.clock.ticks_per_second

    p12 = t.weapons.by_name["p12_whisper"]
    assert (p12.spread_deg, p12.spread_snap_deg) == (2.5, 5.0)
    assert p12.loudness == 0.15                     # the row's identity
    assert p12.rof_interval_ticks == ticks_from_seconds(0.25, tps)
    assert t.ammo_for_weapon(p12).damage == 12      # 9mm_subsonic

    mp11 = t.weapons.by_name["mp11_pdw"]
    assert mp11.shots_per_trigger == 4
    assert t.ammo_for_weapon(mp11).damage == 7      # 9mm_fmj — SAME family
    assert t.ammo_for_weapon(mp11).family == "9mm" == p12.ammo_family

    lr50 = t.weapons.by_name["lr50"]
    assert (lr50.spread_deg, lr50.spread_snap_deg) == (0.25, 2.0)
    r50 = t.ammo_for_weapon(lr50)
    assert (r50.damage, r50.ap, r50.speed_tiles_per_tick) == (90, 10, 128.0)

    lance5 = t.weapons.by_name["lance_5"]
    assert lance5.archetype == "hitscan" and lance5.spread_deg == 0.05
    assert t.ammo_for_weapon(lance5).damage == 55           # the heavy cell
    assert t.ammo_for_weapon("lance_3").damage == 25        # untouched (W2)

    sunspot = t.weapons.by_name["sunspot"]
    helios = t.weapons.by_name["helios"]
    a_s, a_h = t.ammo_for_weapon(sunspot), t.ammo_for_weapon(helios)
    assert (a_s.damage, a_s.dtype, a_s.speed_tiles_per_tick) == (40, "heat", 1.5)
    assert (a_h.damage, a_h.dtype, a_h.speed_tiles_per_tick) == (70, "heat", 1.25)
    assert a_s.glow == a_h.glow == "plasma"
    assert a_s.payload == "plasma_splash_small"
    assert a_h.payload == "plasma_splash_large"

    d9 = t.weapons.by_name["dragon_9_heavy"]
    assert d9.archetype == "spray"
    assert d9.burst_ticks == ticks_from_seconds(2.0, tps) == 48
    assert d9.mag_size == 3
    a9 = t.ammo_for_weapon(d9)
    assert a9.name == "fuel_heavy" and a9.heat_deposit == 4800.0
    # The Dragon-7 keeps first-family-match onto its own round.
    assert t.ammo_for_weapon("dragon_7").name == "fuel_standard"


def test_incendiary_pair_shares_one_payload_row_object():
    """The W3 row-identity rule for the W6 incendiaries: hand grenade and
    40 mm point at THE SAME PayloadDef object (one definition of
    'incendiary', two deliveries)."""
    t = get_tables()
    row = t.payloads.by_name["incendiary_splash"]
    assert t.payload_for_ammo("grenade_incendiary") is row
    assert t.payload_for_ammo("40mm_incendiary") is row
    assert row.ignite_radius > 0


# ---------------------------------------------------------------------------
# 3. Jackhammer-8: pellets = shots_per_trigger = tracers = cone draws
# ---------------------------------------------------------------------------
def test_jackhammer_pellet_count_and_draw_stream():
    t = get_tables()
    jack = t.weapons.by_name["jackhammer_8"]
    assert jack.shots_per_trigger == 8

    gmap = _room()
    shooter = _marine("S", 2, 8, uid=1)
    target = _marine("T", 14, 8, uid=2)
    target.current_hp = 1e9
    rng = np.random.default_rng(SEED)
    shots, events = [], []
    fire_burst(gmap, [shooter, target], shooter, 3, 9, 15, 9,
               tick=0, shots=shots, real_time=0.0, rng=rng, events=events,
               weapon=jack, spread_deg=jack.spread_deg)
    tracers = [e for e in events if isinstance(e, ShotFiredEvent)]
    assert len(tracers) == 8                      # 8 pellets, 8 tracers

    # Draw-for-draw: exactly 8 door-4 cone uniforms, nothing else (no
    # cover on the approach, crit_chance 0 — the lazy-roll rule).
    parallel = np.random.default_rng(SEED)
    cone = math.radians(jack.spread_deg)
    for _ in range(8):
        parallel.uniform(-cone, cone)
    assert rng.bit_generator.state == parallel.bit_generator.state


# ---------------------------------------------------------------------------
# 4. Plasma: direct hit + detonate-at-stop + splash row identity + glow
# ---------------------------------------------------------------------------
def test_plasma_direct_hit_packet_and_detonation_at_stop():
    """A Sunspot bolt fired at a marine victim: ProjectileGlowEvents while
    in flight (1.5 t/t — several ticks), then ON THE STOP TICK a direct-hit
    HEAT packet of exactly 40 (marine mitigation is a no-op) AND the splash
    payload at the entry tile — the one-shot heat disc lands (physics
    detached: the deposit stays visible), event kind 'shell'."""
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    s = _marine("S", 3, 9, weapon="sunspot")
    victim = Unit("V", x=12, y=9, team=0)         # centre (13, 10), dist 9
    sid = sim.add_unit(s)
    sim.add_unit(victim)
    hp_v = victim.current_hp

    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=13, target_fy=10, phase=0))

    glow_ticks = 0
    hits, booms = [], []
    for _ in range(12):
        _step(sim)
        if any(isinstance(e, ProjectileGlowEvent) for e in sim.tick_events):
            glow_ticks += 1
        hits += [e for e in sim.tick_events if isinstance(e, UnitHitEvent)]
        booms += [e for e in sim.tick_events if isinstance(e, ExplosionEvent)]
        if booms:
            break

    assert glow_ticks >= 3                        # a visibly slow bolt
    assert booms and booms[0].kind == "shell"     # detonate-at-stop (W3 rule)
    assert hits and hits[0].unit_id == victim.id
    assert victim.current_hp == hp_v - 40         # the §6 "40 HEAT" direct hit
    # The splash heat disc landed around the stop tile (heat_amount 3200,
    # LINEAR falloff over heat_radius 2) — physics detached, so it persists.
    stop_region = sim.gmap.heat[8:13, 10:16]
    assert int(stop_region.max()) > 0
    # ...and the bolt in flight deposited NO heat (glow is render-only).
    assert int(sim.gmap.heat[:, :9].max()) == 0


def test_plasma_splash_payload_identity_with_its_row():
    """The BulletInFlight-resolved payload IS the table row object (the W3
    row-identity gate applied to plasma), and the row carries the W6 heat
    splash columns."""
    from simulation.combat import BulletInFlight
    t = get_tables()
    sunspot = t.weapons.by_name["sunspot"]
    ammo = t.ammo_for_weapon(sunspot)
    b = BulletInFlight(None, 1, sunspot, ammo, 5.0, 5.0, 0.0, 1.0, 0.0)
    row = t.payloads.by_name["plasma_splash_small"]
    assert b.payload is row
    assert row.heat_amount == 3200.0 and row.heat_radius == 2.0
    assert row.ignite_radius == 1.5
    assert row.unit_damage == 0        # the DIRECT HIT carries the unit damage


def test_plasma_ignites_wood_at_the_splash_end_to_end():
    """With the engine attached: a Sunspot bolt into a wood wall segment —
    the splash's one-shot heat converts to temperature the same tick and
    the wall face ignites (centre jump 3200/8 = 400 > wood's 300 after the
    same-tick cool), plus the ignite ring seeds fire directly. A 3-tile
    wood column, because the bolt's aimed 1.5° cone draw (door 4) wanders
    ~±0.2 tiles over the 8-tile flight — the SEGMENT is spread-proof."""
    wood = [(9, 12, 2), (10, 12, 2), (11, 12, 2)]
    sim = Simulation(_level(edits=wood), seed=SEED,
                     breach_physics=bp, enable_recorder=False)
    s = _marine("S", 3, 9, weapon="sunspot")      # centre (4, 10)
    sid = sim.add_unit(s)
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    lit = False
    for _ in range(16):
        _step(sim)
        if int(sim.gmap.fire[9:12, 12].max()) > 0:
            lit = True
            break
    assert lit, "the plasma splash never ignited the wood face"


# ---------------------------------------------------------------------------
# 5. The spray jet event: present while depositing, absent otherwise
# ---------------------------------------------------------------------------
def test_spray_jet_event_emitted_during_burst_and_absent_otherwise():
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    s = _marine("S", 5, 9, weapon="dragon_7")     # centre (6, 10)
    sid = sim.add_unit(s)

    # Before any fire order: no jets, ever.
    _step(sim, 3)
    assert not any(isinstance(e, SprayJetEvent) for e in sim.tick_events)

    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=12, target_fy=10, phase=0))
    jets = []
    for _ in range(36):
        _step(sim)
        jets += [e for e in sim.tick_events if isinstance(e, SprayJetEvent)]
    assert jets, "no SprayJetEvent during a live burst"
    j = jets[0]
    assert j.kind == "flame"
    assert j.unit_id == s.id
    assert j.from_tile == (s.center_tile_x(), s.center_tile_y())
    assert j.to_tile == (12.0, 10.0)
    assert j.range_tiles == 10 and j.cone_half_angle_degrees == 15.0
    # One event per DEPOSITING tick, none after the mag's last burst stalls.
    assert len(jets) <= 36 * 4


def test_miasma_jet_event_kind_is_miasma():
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    s = _marine("S", 5, 9, weapon="miasma_vent")
    sid = sim.add_unit(s)
    assert sim.apply_action(sid, Order(
        ORDER_FIRE, target_fx=11, target_fy=10, phase=0))
    _step(sim, 2)
    jets = [e for e in sim.tick_events if isinstance(e, SprayJetEvent)]
    assert jets and jets[0].kind == "miasma"


def test_render_events_are_outside_the_synced_digest_surface():
    """The W6 events must never move the determinism digest: the harness
    hashes ONLY UnitHit/UnitKilled (verified, not assumed — the launch
    note's 'the digest doesn't hash events' check)."""
    from field_ab_harness import _SYNCED_EVENT_TYPES
    assert set(_SYNCED_EVENT_TYPES) == {"UnitHitEvent", "UnitKilledEvent"}
    assert "SprayJetEvent" not in _SYNCED_EVENT_TYPES
    assert "ProjectileGlowEvent" not in _SYNCED_EVENT_TYPES


# ---------------------------------------------------------------------------
# 6. The weapon-cycle debug action through the facade
# ---------------------------------------------------------------------------
def test_weapon_cycle_walks_every_triggerable_row_through_the_facade():
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    m = _marine("M", 5, 9)                        # k5_carbine (config default)
    mid = sim.add_unit(m)
    assert m.weapon_id == "k5_carbine"

    expected = [name for name, w in
                sim.weapons_tables.weapons.by_name.items()
                if w.archetype in FIRE_ORDER_ARCHETYPES]
    assert "hand_grenade" not in expected         # LOBBED never cycles in
    assert "breach_charge" not in expected        # PLACED never cycles in
    assert W6_WEAPON_ROWS <= set(expected)

    # A full lap starting after the k5 returns to the k5 (wrap-around).
    start = expected.index("k5_carbine")
    seen = [sim.debug_cycle_weapon(mid) for _ in range(len(expected))]
    assert seen == expected[start + 1:] + expected[:start + 1]
    assert m.weapon_id == "k5_carbine"

    # The swap resets the coupled state (fresh mag, no ghost burst).
    m.current_mag = 1
    m.spray_ticks_left = 7
    sim.debug_cycle_weapon(mid)
    assert m.current_mag is None and m.reload_done_tick == -1
    assert m.spray_ticks_left == 0

    # Zombies refuse (no weapon rows — the ai_zombie path).
    z = Unit("Z", x=15, y=15, team=1)
    zid = sim.add_unit(z)
    assert sim.debug_cycle_weapon(zid) is None


def test_cycled_weapon_actually_fires_through_the_order_path():
    """Cycle a marine onto the Jackhammer via the facade, place an ordinary
    fire order, step: 8 pellet tracers — the debug key's whole point."""
    sim = Simulation(_level(), seed=SEED, breach_physics=None,
                     enable_recorder=False)
    m = _marine("M", 5, 9)
    mid = sim.add_unit(m)
    while m.weapon_id != "jackhammer_8":
        assert sim.debug_cycle_weapon(mid) is not None
    assert sim.apply_action(mid, Order(
        ORDER_FIRE, target_fx=18, target_fy=10, phase=0))
    _step(sim)
    tracers = [e for e in sim.tick_events if isinstance(e, ShotFiredEvent)]
    assert len(tracers) == 8


# ---------------------------------------------------------------------------
# 7. Dragon ignition-reach derivations pinned at the new scale
# ---------------------------------------------------------------------------
def test_dragon_heat_deposits_track_their_ranges():
    """The config derivation of record: sustained-jet ignition needs
    T_inf(d) = (31/8) * D / d >= 300 (wood), so the reach in tiles is
    floor(31 * D / (8 * 300)). Dragon-7 (D 2400) -> 31 tiles >= its 30-tile
    playground range; Dragon-9 (D 4800) -> 62 >= 60. The heat scale rides
    the range dial — one discipline, two rows."""
    t = get_tables()
    tp = WeaponsTables.from_config(tile_size_m=0.333)   # playground binding
    for weapon, deposit in (("dragon_7", 2400.0), ("dragon_9_heavy", 4800.0)):
        a = t.ammo_for_weapon(weapon)
        assert a.heat_deposit == deposit, weapon
        reach_tiles = (31 * int(a.heat_deposit)) // (8 * 300)
        range_tiles = tp.weapons.by_name[weapon].range_tiles
        assert reach_tiles >= range_tiles, (
            f"{weapon}: ignition reach {reach_tiles} < range {range_tiles}")


# ---------------------------------------------------------------------------
# 8. Dormancy replica: the canonical scenario, bit-identical incl. RNG
# ---------------------------------------------------------------------------
GOLDEN_AGGREGATE = "07c3f37043c62cb47ec1abfef1a59d47c5f7a9c313490b38ecd2ddc543d1833d"


def test_canonical_scenario_golden_and_untouched_rng():
    """The strongest replica there is: the canonical A/B scenario's 30-tick
    aggregate digest (every field, every cell, every tick + the synced unit
    state) equals the golden pinned since W1 — W6 moved NOTHING. And the
    sim RNG end-state equals a fresh generator's: the scenario consumes
    ZERO randomness, so any W6 draw sneaking onto the stream trips here
    even if no hashed field moved yet."""
    from field_ab_harness import (
        SEED as AB_SEED, SIM_FIELDS, UNIT_DIGEST_KEY,
        _capture_unit_state, _snapshot, default_scenario_sim,
    )
    from field_digest import trajectory_digest

    # capture_trajectory owns its sim; replicate its body verbatim so the
    # generator stays inspectable afterwards.
    sim = default_scenario_sim()
    traj = []
    for _ in range(30):
        sim.set_paused(False)
        sim.step()
        snap = _snapshot(sim.gmap, SIM_FIELDS)
        snap[UNIT_DIGEST_KEY] = _capture_unit_state(sim)
        traj.append(snap)

    assert trajectory_digest(traj) == GOLDEN_AGGREGATE, (
        "the canonical scenario's aggregate digest moved — W6 changed a "
        "dormant trajectory; this is a bug, never a re-baseline")
    fresh = np.random.default_rng(AB_SEED)
    assert sim.rng.bit_generator.state == fresh.bit_generator.state, (
        "the canonical scenario drew RNG — some W6 path consumed the stream")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
