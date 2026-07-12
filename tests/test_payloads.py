"""The W3 payload EXECUTOR gate (mechanics/03 §4 — simulation.payloads).

What is locked here:

  - BYTE-IDENTITY REPLICA (the load-bearing behaviour-preservation proof):
    executing ``frag_standard`` / ``breach_focus`` through
    :func:`execute_payload` reproduces the pre-W3 inline detonation triple
    (apply_explosion -> apply_blast_damage -> add_explosion_smoke ->
    ExplosionEvent) BIT-FOR-BIT — every field array, every unit hp, the
    event stream, and the RNG end-state;
  - FULL-SIM DORMANCY (the W2 replica-test pattern): a scripted round with
    ONLY shipped weapons (a frag grenade + a door charge — no gas, no
    launcher, no C4) on the W3 code is bit-identical, tick for tick, to a
    twin sim whose executor is rebound to the verbatim pre-W3 site body —
    fields, unit hp, synced events, and the generator end-state;
  - PAYLOAD ROW SHARING: hand-grenade rounds and 40 mm rounds reference THE
    SAME payload row objects (one definition of frag/smoke/tear/poison, two
    deliveries);
  - GAS DEPOSIT EXACTNESS: hand-computed Q16.16 expectations for the radial
    linear falloff, the [0,1] saturation clamp, the solid skip, slice
    targeting per species, additivity, and the NO-RNG guarantee;
  - INCENDIARY IGNITE RING: fire = max(fire, intensity x falloff) — never
    lowers an existing fire, skips non-flammable tiles;
  - C4 DET-SLOT: the demolition_c4 numbers through the shipped
    ORDER_EXPLOSIVE flow with ammo_name = "demo_c4".

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_payloads.py -q
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
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import gas_fixed, fire_fixed, wall_fixed  # noqa: E402
from simulation.events import ExplosionEvent, UnitHitEvent  # noqa: E402
from simulation.exchange import apply_blast_damage  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import (  # noqa: E402
    BLACK_SMOKE, N_TRACE_GASES, POISON as GAS_POISON, TEARGAS as GAS_TEARGAS,
    WHITE_SMOKE,
)
from simulation.orders import (  # noqa: E402
    DET_START_PHASE1, ORDER_EXPLOSIVE, ORDER_GRENADE, Order,
)
from simulation.payloads import emit_gas, execute_payload, ignite_ring  # noqa: E402
from simulation.physics import add_explosion_smoke, apply_explosion  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

SEED = 20260705


# ---------------------------------------------------------------------------
# Scaffolding (the test_weapons_march shape)
# ---------------------------------------------------------------------------
def _level(h=24, w=24, edits=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w3_payloads", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _hits(events):
    return [e for e in events if isinstance(e, UnitHitEvent)]


def _explosions(events):
    return [e for e in events if isinstance(e, ExplosionEvent)]


# ---------------------------------------------------------------------------
# The pre-W3 detonation site, VERBATIM (simulation.py / combat.py @ ebbc16b):
# the inline triple + event. This is the replica both byte-identity proofs
# compare against.
# ---------------------------------------------------------------------------
def _prew3_site_replica(gmap, queue, units, fy, fx, payload, rng, events=None,
                        kind="explosion"):
    radius = payload.radius
    apply_explosion(gmap, queue, fy, fx, radius, payload.pressure,
                    payload.wall_damage)
    apply_blast_damage(units, fx, fy, radius, payload.unit_damage,
                       events=events)
    add_explosion_smoke(gmap, queue, fy, fx, radius)
    if events is not None:
        events.append(ExplosionEvent(pos=(fx, fy), radius=radius, kind=kind))


_FIELDS = ("gas", "atmosphere", "wave_source", "wave_p", "fire", "wall_hp",
           "material", "water_depth", "is_vacuum")


def _twin(edits=()):
    """One deterministic world: map + queue + rng + two units in blast range
    (a marine and a zombie — the zombie exercises the blast path's neutral
    BLAST mitigation on the state overlay)."""
    gmap = _room(edits=edits)
    queue = EditQueue()
    m = Unit("M", x=8, y=8, team=0)
    z = Unit("Z", x=13, y=11, team=1)
    m.id, z.id = 1, 2
    m.current_hp = z.current_hp = 1e9
    rng = np.random.default_rng(SEED)
    return gmap, queue, [m, z], rng, []


def test_frag_and_breach_byte_identical_to_prew3_triple():
    """THE REPLICA GATE: for both shipped payload rows the executor's world
    is indistinguishable from the pre-W3 inline site — fields, hp, events,
    and the generator end-state (the smoke noise draws at flush)."""
    tables = get_tables()
    cases = [
        ("frag_standard", "grenade", 10, 11, ()),          # open-floor frag
        ("breach_focus", "door_explosive", 9, 12,
         ((9, 12, 2),)),                                    # charge ON a wood wall
    ]
    for payload_name, kind, fy, fx, edits in cases:
        payload = tables.payloads.by_name[payload_name]

        g_a, q_a, units_a, rng_a, ev_a = _twin(edits)
        _prew3_site_replica(g_a, q_a, units_a, fy, fx, payload, rng_a,
                            events=ev_a, kind=kind)
        q_a.flush(g_a, rng_a)

        g_b, q_b, units_b, rng_b, ev_b = _twin(edits)
        execute_payload(g_b, q_b, units_b, fy, fx, payload, rng_b,
                        events=ev_b, kind=kind)
        q_b.flush(g_b, rng_b)

        for f in _FIELDS:
            assert np.array_equal(getattr(g_a, f), getattr(g_b, f)), \
                f"{payload_name}: field '{f}' diverged from the pre-W3 site"
        for ua, ub in zip(units_a, units_b):
            assert ua.current_hp == ub.current_hp
            assert ua.alive == ub.alive
        assert repr(ev_a) == repr(ev_b)
        assert rng_a.bit_generator.state == rng_b.bit_generator.state


def test_shipped_weapons_full_run_bit_identical_to_prew3():
    """FULL-SIM DORMANCY (the W2 replica pattern): a scripted round throwing
    a frag grenade + firing a door charge — and NOTHING W3-new — is
    bit-identical, tick for tick, between the live W3 code and a twin whose
    executor is rebound to the verbatim pre-W3 site body. Fields, unit hp,
    synced events, RNG end-state."""
    import simulation.combat as combat_mod
    import simulation.simulation as sim_mod

    def build():
        sim = Simulation(_level(edits=((9, 12, 2),)), seed=SEED,
                         breach_physics=bp, enable_recorder=False)
        m = Unit("M", x=3, y=3, team=0)
        z = Unit("Z", x=16, y=16, team=1)
        mid = sim.add_unit(m)
        sim.add_unit(z)
        assert sim.apply_action(mid, Order(
            ORDER_GRENADE, target_fx=11, target_fy=10, phase=0,
            grenade_fuse=0.5))
        assert sim.apply_action(mid, Order(
            ORDER_EXPLOSIVE, target_fx=12, target_fy=9, phase=0,
            det_slot=DET_START_PHASE1))
        sim.spawn_projectiles_from_grenade_orders()
        sim.set_paused(False)
        return sim

    def run(sim, n=20):
        traj = []
        for _ in range(n):
            sim.set_paused(False)
            sim.step()
            snap = {f: np.copy(getattr(sim.gmap, f)) for f in _FIELDS}
            snap["__events__"] = repr(sim.tick_events)
            snap["__hp__"] = tuple(u.current_hp for u in sim.units)
            snap["__alive__"] = tuple(u.alive for u in sim.units)
            traj.append(snap)
        return traj, sim.rng.bit_generator.state

    # The W3 path (the live executor).
    traj_new, rng_new_state = run(build())

    # The pre-W3 twin: rebind BOTH detonation sites' executor binding to the
    # verbatim inline-site replica (the bare-name import contract).
    saved_sim, saved_combat = sim_mod.execute_payload, combat_mod.execute_payload
    sim_mod.execute_payload = _prew3_site_replica
    combat_mod.execute_payload = _prew3_site_replica
    try:
        traj_old, rng_old_state = run(build())
    finally:
        sim_mod.execute_payload = saved_sim
        combat_mod.execute_payload = saved_combat

    assert len(traj_new) == len(traj_old)
    for t, (sn, so) in enumerate(zip(traj_new, traj_old)):
        for f in _FIELDS:
            assert np.array_equal(sn[f], so[f]), \
                f"tick {t}: field '{f}' diverged from the pre-W3 replica"
        assert sn["__events__"] == so["__events__"], f"tick {t}: events diverged"
        assert sn["__hp__"] == so["__hp__"], f"tick {t}: hp diverged"
        assert sn["__alive__"] == so["__alive__"], f"tick {t}: life diverged"
    assert rng_new_state == rng_old_state, "RNG stream moved vs pre-W3"


# ---------------------------------------------------------------------------
# Payload row sharing — one definition, two deliveries
# ---------------------------------------------------------------------------
def test_hand_and_40mm_rounds_share_the_same_payload_row_objects():
    t = get_tables()
    pairs = [
        ("grenade_frag", "40mm_frag", "frag_standard"),
        ("grenade_smoke", "40mm_smoke", "smoke_screen"),
        ("grenade_tear", "40mm_tear", "tear_burst"),
        ("grenade_poison", "40mm_poison", "poison_cloud"),
    ]
    for hand, tube, row_name in pairs:
        row = t.payloads.by_name[row_name]
        assert t.payload_for_ammo(hand) is row       # object identity —
        assert t.payload_for_ammo(tube) is row       # ONE definition


def test_c4_and_gl6_rows_load_with_their_standard_values():
    t = get_tables()
    c4 = t.payload_for_ammo("demo_c4")
    assert (c4.radius, c4.pressure, c4.wall_damage, c4.unit_damage) == \
        (8, 25.0, 800, 150)
    assert c4.clear_smoke is True and c4.emit_blast_smoke is True
    gl6 = t.weapons.by_name["gl6_revolver"]
    assert gl6.archetype == "projectile" and gl6.ammo_family == "40mm"
    assert (gl6.mag_size, gl6.reload_seconds) == (6, 3.0)
    assert (gl6.spread_deg, gl6.spread_snap_deg, gl6.range_tiles) == (3.0, 5.0, 40)
    assert t.ammo_for_weapon(gl6).name == "40mm_frag"   # first-family-match
    c4w = t.weapons.by_name["c4_satchel"]
    assert c4w.archetype == "placed" and c4w.ammo_family == "demo_charge"
    # The breach charge keeps demo_breach as its family's first (default) row.
    assert t.ammo_for_weapon("breach_charge").name == "demo_breach"


# ---------------------------------------------------------------------------
# Gas deposit exactness (Q16.16, hand-computed) + slice targeting + no RNG
# ---------------------------------------------------------------------------
def _disc_weight(dy, dx, radius):
    """The FieldEdit DISC LINEAR falloff arithmetic, replicated: strict
    dist < radius membership, weight = 1 - dist/radius (float chain)."""
    dist = float(np.sqrt(float(dy * dy + dx * dx)))
    if not (dist < radius):
        return None
    return 1.0 - dist / radius


def test_gas_deposit_exact_q16_falloff_clamp_skip_and_slice():
    """Hand-computed per-tile expectations for smoke_screen (amount 1.5,
    radius 4): centre saturates to FP_ONE (the [0,1] clamp), every in-disc
    tile holds quantize(min(1, 1.5 x (1 - d/4))), solid tiles are skipped,
    the ring d >= 4 is untouched, and ONLY the white_smoke slice moves."""
    gmap = _room(edits=[(10, 13, 1)])     # a hull tile INSIDE the disc
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    state_before = rng.bit_generator.state

    emit_gas(gmap, queue, 10, 11, "white_smoke", 1.5, 4)
    queue.flush(gmap, rng)

    # NO RNG: the deposit is deliberately noise-free (unlike the blast cloud).
    assert rng.bit_generator.state == state_before

    ws = gmap.gas[WHITE_SMOKE]
    # Centre: weight 1.0 -> 1.5 clamps to 1.0 -> FP_ONE counts exactly.
    assert int(ws[10, 11]) == gas_fixed.FP_ONE
    # On-axis d=2: weight 0.5 -> 0.75 -> 49152 counts exactly (hand: .75*2^16).
    assert int(ws[10, 9]) == 49152
    # Diagonal d=sqrt(5) (dy=1,dx=2): the float falloff chain, quantized once.
    w = _disc_weight(1, 2, 4.0)
    assert int(ws[11, 13]) == gas_fixed.quantize_scalar(1.5 * w)
    # d=3 on-axis: weight 0.25 -> 0.375 -> 24576 counts.
    assert int(ws[13, 11]) == 24576
    # The solid tile inside the disc: SKIPPED (gas does not enter walls).
    assert int(ws[10, 13]) == 0
    # Outside the strict disc (d=4 exactly on-axis): untouched.
    assert int(ws[10, 15]) == 0 and int(ws[14, 11]) == 0
    # Slice targeting: every OTHER TRACE gas plane is untouched (the bulk
    # O2/inert_N2 pair, EOS refactor P1, always carries ambient air).
    for g in range(N_TRACE_GASES):
        if g != WHITE_SMOKE:
            assert not gmap.gas[g].any(), f"slice {g} moved on a white_smoke deposit"


def test_gas_deposit_is_additive_with_saturation_guard():
    """Two identical deposits: unsaturated tiles double exactly; the core
    stays clamped at FP_ONE (the saturation guard)."""
    gmap = _room()
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    for _ in range(2):
        emit_gas(gmap, queue, 10, 11, "teargas", 0.3, 3)
        queue.flush(gmap, rng)
    tg = gmap.gas[GAS_TEARGAS]
    # Centre: 2 x 0.3 = 0.6 (below clamp) -> quantize(0.3)+combine(0.3) —
    # the combine dequantizes exactly (n/65536), so twice = quantize(0.6).
    assert int(tg[10, 11]) == gas_fixed.quantize_scalar(0.6)
    # Saturation: a 1.5-amount deposit twice stays clamped at FP_ONE.
    gmap2 = _room()
    for _ in range(2):
        emit_gas(gmap2, queue, 10, 11, "poison", 1.5, 3)
        queue.flush(gmap2, rng)
    assert int(gmap2.gas[GAS_POISON][10, 11]) == gas_fixed.FP_ONE


def test_gas_payload_species_route_to_their_slices():
    """tear_burst -> the TEARGAS slice; poison_cloud -> the POISON slice;
    smoke_screen -> WHITE_SMOKE; unknown species fail loudly."""
    t = get_tables()
    for row_name, slice_id in [("tear_burst", GAS_TEARGAS),
                               ("poison_cloud", GAS_POISON),
                               ("smoke_screen", WHITE_SMOKE)]:
        gmap = _room()
        queue = EditQueue()
        rng = np.random.default_rng(SEED)
        payload = t.payloads.by_name[row_name]
        execute_payload(gmap, queue, [], 10, 11, payload, rng, events=None,
                        kind="grenade")
        queue.flush(gmap, rng)
        assert gmap.gas[slice_id].any()
        for g in range(N_TRACE_GASES):
            if g != slice_id:
                assert not gmap.gas[g].any()
        # A pure-gas payload deposits NO explosion side effects.
        assert not gmap.wave_source.any()
        assert not gmap.fire.any()
        assert not gmap.gas[BLACK_SMOKE].any()
    # Loud on a typo'd species (belt-and-suspenders at the emit site).
    import pytest
    with pytest.raises(KeyError):
        emit_gas(_room(), EditQueue(), 10, 11, "nerve_gas", 1.0, 3)


def test_pure_gas_payload_still_emits_the_detonation_event():
    """The detonation happened whatever the payload mix: a smoke grenade
    emits ExplosionEvent (radius 0, kind carried through) — the renderer
    ignores unknown kinds by design."""
    t = get_tables()
    gmap = _room()
    queue = EditQueue()
    rng = np.random.default_rng(SEED)
    events = []
    execute_payload(gmap, queue, [], 10, 11, t.payloads.by_name["smoke_screen"],
                    rng, events=events, kind="grenade")
    ex = _explosions(events)
    assert len(ex) == 1
    assert ex[0].pos == (11, 10) and ex[0].radius == 0
    assert ex[0].kind == "grenade"
    assert not _hits(events)                       # no unit damage from gas


# ---------------------------------------------------------------------------
# Incendiary ignite ring — max-write, never lowers, flammable-only
# ---------------------------------------------------------------------------
def test_incendiary_ignite_ring_max_write_never_lowers():
    """fire = max(fire, 0.5 x (1 - d/2.5)) over flammable tiles: a cold wood
    tile ignites to the exact falloff value; a hotter existing fire is NEVER
    lowered; a non-flammable tile stays cold."""
    # Wood (code 2, flammable) at d=1 and d=2 on-axis; furniture (6) at d=1
    # north; the centre tile is air (non-flammable -> no deposit).
    gmap = _room(edits=[(10, 12, 2), (10, 13, 2), (9, 11, 6)])
    queue = EditQueue()
    rng = np.random.default_rng(SEED)

    # Pre-existing BIG fire on the furniture tile — must never lower.
    big_q = fire_fixed.quantize_scalar(0.9)
    gmap.fire[9, 11] = big_q

    t = get_tables()
    payload = t.payloads.by_name["incendiary_splash"]
    assert payload.ignite_radius == 2.5 and payload.ignite_intensity == 0.5
    execute_payload(gmap, queue, [], 10, 11, payload, rng, events=None,
                    kind="grenade")
    queue.flush(gmap, rng)

    # d=1 wood: weight = 1 - 1/2.5 = 0.6 -> fire = quantize(0.3).
    assert int(gmap.fire[10, 12]) == fire_fixed.quantize_scalar(0.5 * 0.6)
    # d=2 wood: weight = 1 - 2/2.5 = 0.2 -> fire = quantize(0.1).
    assert int(gmap.fire[10, 13]) == fire_fixed.quantize_scalar(0.5 * 0.2)
    # The burning furniture tile: 0.9 > 0.3 -> UNCHANGED (max, never lowers).
    assert int(gmap.fire[9, 11]) == big_q
    # The air centre tile: non-flammable -> skipped by the fire policy.
    assert int(gmap.fire[10, 11]) == 0


# ---------------------------------------------------------------------------
# C4 through the shipped det-slot flow
# ---------------------------------------------------------------------------
def test_c4_det_slot_detonation_with_demolition_numbers():
    """ORDER_EXPLOSIVE + ammo_name='demo_c4' rides the breach-charge flow:
    the det slot fires the demolition_c4 payload — hull wall destroyed at
    the target, exact Q16.16 chew at range, blast damage with the 150/8
    falloff, kind='door_explosive' event."""
    # Target the west border hull at (10, 0); a marine 6 tiles east of it.
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=5, y=9, team=0)      # centre (6, 10) -> dist 6 from (0,10)
    mid = sim.add_unit(m)
    m.current_hp = 1e9
    assert sim.apply_action(mid, Order(
        ORDER_EXPLOSIVE, target_fx=0, target_fy=10, phase=0,
        det_slot=DET_START_PHASE1, ammo_name="demo_c4"))
    assert m.has_explosive == 1          # the shipped count pool decremented

    hp_before = m.current_hp
    sim.set_paused(False)
    sim.step()

    # The targeted hull tile: 800 >= 300 HP -> destroyed (breached to vacuum).
    assert int(sim.gmap.material[10, 0]) == 0 or bool(sim.gmap.is_vacuum[10, 0])
    # A border hull tile 6 up the wall: dist 6, falloff 1 - 6/8 = 0.25 ->
    # exact Q16.16 chew of 800 x 0.25 = 200 off the 300-HP hull.
    expected = wall_fixed.quantize_scalar(300.0) - \
        wall_fixed.quantize_scalar(800.0 * 0.25)
    assert int(sim.gmap.wall_hp[16, 0]) == expected
    # Blast damage: the marine at dist 6.0 -> int(150 x 0.25) = 37 (>= thresh).
    hits = [e for e in sim.tick_events if isinstance(e, UnitHitEvent)
            and e.source == "explosion"]
    assert len(hits) == 1 and hits[0].damage == 37.0
    assert m.current_hp == hp_before - 37.0
    ex = _explosions(sim.tick_events)
    assert len(ex) == 1 and ex[0].kind == "door_explosive" and ex[0].radius == 8


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
