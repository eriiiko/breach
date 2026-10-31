"""The HEAT-SOURCE entity (levelgen vocabulary E.2 row 3,
docs/levelgen_zone_vocabulary_2026-10-29.md; Erik's ruling 2026-10-31).

A ``heat_source`` (engine, galley stove, furnace) deposits ``heat_per_s``
every tick through THE heat deposit path (``payloads.deposit_heat``: a
``heat`` FieldEdit the temperature solver converts and books), withholds it
while its own tile reads at or above ``max_temp``, and dies when an
explosion disc (``gmap.blast_log``, appended by ``execute_payload``) takes its
``hp`` to 0 -- executing its ``death_payload`` once, which delivers the heat
that starts the fire (CLAUDE.md "Starting a fire").

Each test names the property it protects and the change that must break it.
Fires are only ever started here by delivered heat -- never a ``fire`` write.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_heat_source.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
from config import CFG  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.serialize import entity_records  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402
from simulation.gas_fixed import quantize_q16_scalar  # noqa: E402
from simulation.heat_source_system import (EVENT_KIND,  # noqa: E402
                                           HeatSourceError)
from simulation.materials import (MAT_AIR, MAT_FURNITURE, MAT_HULL,  # noqa: E402
                                  MAT_WOOD)
from simulation.payloads import execute_payload  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
SRC_AT = (7, 7)             # (y, x)
FURN_AT = ((7, 9), (8, 9))  # two furniture tiles 2 tiles from the source
HEAT = 5.0                  # heat/s: a gas tile runs hot (~1500 game deg)
CAP = 60.0                  # game degrees above ambient


def _room(entities=(), hh=16, furniture=True):
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    if furniture:
        for yx in FURN_AT:
            tm[yx] = MAT_FURNITURE
    return LevelData(name="heat_source", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[])


def _source(eid="engine", ordinal=0, **over):
    fields = {f.name: f.default for f in REGISTRY["heat_source"].FIELDS}
    fields.update(x=SRC_AT[1], y=SRC_AT[0], heat_per_s=HEAT)
    fields.update(over)
    return EntityInstance(id=eid, class_name="heat_source", ordinal=ordinal,
                          tags=(), fields=fields)


def _sim(entities, seed=3, furniture=True, physics=bp):
    return Simulation(_room(entities, furniture=furniture), seed=seed,
                      breach_physics=physics, enable_recorder=False)


def _step(sim):
    sim.set_paused(False)            # the ruleset auto-pauses at seams
    sim.step()


def _t(sim, yx=SRC_AT):
    return int(sim.gmap.temperature[yx])


def _blast(sim, row="breach_focus", at=SRC_AT):
    """A HEAT-FREE blast (breach_focus carries no heat) through THE
    executor, so any fire that follows is the death payload's."""
    execute_payload(sim.gmap, sim.edit_queue, sim.units, at[0], at[1],
                    sim.weapons_tables.payloads.by_name[row], sim.rng,
                    events=sim.tick_events)


# ---------------------------------------------------------------------------
# Deposit
# ---------------------------------------------------------------------------

def test_a_source_heats_its_tile_and_a_disabled_one_does_not():
    """Property: an enabled source raises its own tile's temperature above
    ambient within a few ticks and keeps it well above ambient for a second
    of sim time (the deposit lands every tick); an identical source with
    enabled = false leaves the tile at ambient. Breaks if the sweep is not
    called, deposits nothing (a per-tick quantum quantized to 0, a radius-0
    disc that the strict DISC iterator empties), or ignores ``enabled``."""
    on, off = _sim([_source()]), _sim([_source(enabled=False)])
    t_on, t_off = [], []
    for _ in range(TPS):
        _step(on)
        _step(off)
        t_on.append(_t(on))
        t_off.append(_t(off))
    assert t_on[3] > 0, "the source had not heated its tile after 4 ticks"
    assert min(t_on[TPS // 4:]) > quantize_q16_scalar(100.0), t_on[-5:]
    assert max(t_off) == 0, "a disabled source heated its tile"


def test_the_deposit_rises_monotonically_until_conduction_matters():
    """Property: over the first ticks the source tile's temperature never
    falls -- each tick adds one deposit. Breaks if the deposit is
    intermittent (enqueued on alternate ticks, dropped at the flush) or
    signed wrong."""
    sim = _sim([_source()])
    ts = []
    for _ in range(8):
        _step(sim)
        ts.append(_t(sim))
    assert all(b >= a for a, b in zip(ts, ts[1:])), ts
    assert ts[-1] > ts[0]


def test_max_temp_holds_the_tile_near_the_cap_while_the_uncapped_twin_runs_away():
    """Property: with ``max_temp`` set, the source tile never exceeds the cap
    by more than one tick's rise over a long run (the source withholds its
    deposit at or above the cap; one deposit already enqueued can still
    land), while the uncapped twin ends far above the cap. Breaks if the cap
    is not read, read in the wrong currency (degrees vs Q16), compared the
    wrong way, or if the source writes ``temperature`` itself."""
    capped, free = _sim([_source(max_temp=CAP)]), _sim([_source()])
    cap_q = quantize_q16_scalar(CAP)
    ts = []
    for _ in range(10 * TPS):
        _step(capped)
        _step(free)
        ts.append(_t(capped))
    one_tick_rise = max(b - a for a, b in zip(ts, ts[1:]))
    assert max(ts) >= cap_q, "vacuous: the capped source never reached its cap"
    assert max(ts) <= cap_q + one_tick_rise, (max(ts), cap_q, one_tick_rise)
    assert one_tick_rise < cap_q, "one deposit is larger than the cap itself"
    assert _t(free) > 3 * cap_q, "vacuous: the uncapped twin did not run away"


# ---------------------------------------------------------------------------
# The energy books
# ---------------------------------------------------------------------------

def _gas_books(g):
    return int(g.gas_energy[g._gas_energy_accountable()].astype(object).sum())


def _terms(g, eos, tsolver, comb, engine):
    """The full ledger's right-hand side, test_thermostat_books' shape (the
    copy in test_bulk_gas_payload): EOS (absolute per step), thermal solver
    gas side, combustion, the seams, water evac (absolute), the solid side."""
    return (
        int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
        - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
        + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
        + int(eos.e_rail_sum),
        int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
        + int(tsolver.e_gas_rail_sum),
        -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
        + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
        int(g.gas_energy_seam_net()),
        -int(engine.e_water_evac_export_sum),
        int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
        + int(comb.e_comb_solid_heat_sum)
        + int(tsolver.e_solid_low_rail_sum) + int(tsolver.e_solid_seam_sum),
    )


def _closure_bad(sim, n, before_step=None):
    """Step ``n`` times; the ticks whose total ledger did not close exactly."""
    g = sim.gmap
    rp = sim.physics_runner
    eos, tsolver, comb, engine = (rp.eos, rp.engine.temperature,
                                  rp.combustion, rp.engine)
    prev_total = _gas_books(g) + int(tsolver.solid_energy_books_sum)
    prev = _terms(g, eos, tsolver, comb, engine)
    bad = []
    for i in range(n):
        if before_step is not None:
            before_step(i, sim)
        _step(sim)
        total = _gas_books(g) + int(tsolver.solid_energy_books_sum)
        t = _terms(g, eos, tsolver, comb, engine)
        expected = (t[0] + (t[1] - prev[1]) + (t[2] - prev[2])
                    + (t[3] - prev[3]) + t[4] + (t[5] - prev[5]))
        resid = (total - prev_total) - expected
        if resid:
            bad.append((i, resid))
        prev_total, prev = total, t
    return bad


def test_closure_identity_holds_with_a_source_heating_gas_and_fuel():
    """Property: with a radius-2 source depositing every tick on gas AND on
    the furniture beside it (thermal solids) until the fuel catches, the
    across-tick closure identity ``Δ(Σ gas_energy + solid books) == EOS +
    thermal + combustion + seams + water`` is EXACT in int64 on every tick,
    and the source's heat shows up in the thermal solver's gas and solid
    deposit counters. Breaks if the source lands heat outside the solver's
    counted deposit (a ``temperature`` or ``gas_energy`` write, a new
    uncounted channel)."""
    sim = _sim([_source(radius=2)])
    ts = sim.physics_runner.engine.temperature
    gas0, sol0 = int(ts.e_gas_deposit_sum), int(ts.e_solid_deposit_sum)
    bad = _closure_bad(sim, 4 * TPS)
    assert int(ts.e_gas_deposit_sum) > gas0, "vacuous: no gas deposit booked"
    assert int(ts.e_solid_deposit_sum) > sol0,         "vacuous: the disc never reached the furniture"
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


def test_closure_identity_holds_across_a_source_dying():
    """Property: a running source blown up mid-run (the heat-free blast,
    then its ``engine_death`` blast and heat executed from slot 9e) keeps
    the closure identity exact on every tick. FUEL-FREE ROOM on purpose:
    a blast with ``pressure > 0`` next to a thermal solid breaks the
    identity by a constant on its own, with no heat source anywhere
    (measured 2026-10-31 on the base commit 8da03d6: breach_focus beside two
    furniture tiles, residual 2859024384 on the blast tick) -- a
    pre-existing explosion gap recorded for its own patch, not this one's.
    Breaks if the death path writes energy outside the counted channels."""
    sim = _sim([_source(death_payload="engine_death")], furniture=False)

    def blast(i, s):
        if i == TPS:
            _blast(s)

    bad = _closure_bad(sim, 3 * TPS, before_step=blast)
    assert not sim._heat_sources[0].alive, "vacuous: the source never died"
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


# ---------------------------------------------------------------------------
# Death
# ---------------------------------------------------------------------------

def _engine_room(death_payload):
    """A capped, gently running engine next to furniture (the source never
    lights it on its own), blown up by a heat-free blast after 5 s; return
    (sim, first step with fire after the blast or None)."""
    sim = _sim([_source(heat_per_s=1.0, max_temp=30.0,
                        death_payload=death_payload)])
    for _ in range(5 * TPS):
        _step(sim)
    assert int(sim.gmap.fire.astype(np.int64).sum()) == 0, \
        "the running engine lit the room by itself (the scenario is moot)"
    _blast(sim)
    first = None
    for i in range(3 * TPS):
        _step(sim)
        if first is None and int(sim.gmap.fire.astype(np.int64).sum()) > 0:
            first = i
    return sim, first


def test_a_destroyed_engine_starts_the_fire_its_death_payload_carries():
    """Property: a heat-free blast on an engine (hp 200 < breach_focus's 500
    wall damage) kills it -- ``alive`` 0, ``hp`` 0, no more deposits -- and
    its ``engine_death`` row executes once (one ExplosionEvent of the
    heat_source kind), whose heat lights the furniture beside it through
    ``apply_temperature_ignition`` within a second; the twin without a death
    payload dies too but no fire follows. Breaks if the blast log is not
    appended or not drained, if hp is not read in the wall law, if the death
    payload is skipped or executed twice, or if engine_death stops carrying
    enough heat to light fuel 2 tiles away."""
    sim, first = _engine_room("engine_death")
    rt = sim._heat_sources[0]
    assert not rt.alive and rt.hp_q16 == 0 and not rt.enabled
    assert first is not None and first < TPS, \
        f"no fire within a second of the engine's death (first={first})"
    assert any(int(sim.gmap.fire[yx]) > 0 for yx in FURN_AT)
    quiet, quiet_first = _engine_room("")
    assert not quiet._heat_sources[0].alive
    assert quiet_first is None, "a death without a payload lit a fire"


def test_the_death_payload_executes_exactly_once():
    """Property: a dead source never re-fires its death payload, whatever
    further blasts land on it. Breaks if death does not latch ``alive``."""
    sim = _sim([_source(death_payload="engine_death")], furniture=False)
    kinds = []
    for i in range(3 * TPS):
        if i in (5, 30, 60):
            _blast(sim)
        _step(sim)
        kinds += [e for e in sim.tick_events
                  if isinstance(e, ExplosionEvent) and e.kind == EVENT_KIND]
    assert len(kinds) == 1, kinds


def test_a_blast_beyond_reach_or_too_weak_leaves_the_source_alive():
    """Property: damage follows the wall law -- a disc that does not reach
    the source does nothing, and a weaker one only lowers hp. Breaks if the
    hook ignores the disc's radius, or kills on any hit."""
    sim = _sim([_source(hp=400)], furniture=False)
    rt = sim._heat_sources[0]
    hp0 = rt.hp_q16
    _blast(sim, at=(SRC_AT[0], SRC_AT[1] + 5))     # breach_focus radius 3
    _step(sim)
    assert rt.alive and rt.hp_q16 == hp0
    _blast(sim, row="frag_standard", at=(SRC_AT[0], SRC_AT[1] + 1))
    _step(sim)
    assert rt.alive and 0 < rt.hp_q16 < hp0


def test_chained_engines_go_up_in_one_tick():
    """Property: one engine's death blast takes the next engine within its
    disc in the SAME sweep (the death payload's disc joins the discs being
    read), even when the second engine precedes it in ordinal order.
    Engine ``a`` (ordinal 0) sits 4 tiles from a heat-free blast (out of its
    radius 3) and 2 from engine ``b`` (ordinal 1), which the blast kills;
    ``b``'s engine_death disc (radius 3, wall_damage 300: 100 at 2 tiles)
    then kills ``a`` (hp 100). Breaks if a death payload's blast is only read
    next tick, or lost."""
    a = _source("e_a", 0, x=SRC_AT[1] + 2, hp=100,
                death_payload="engine_death")
    b = _source("e_b", 1, hp=100, death_payload="engine_death")
    sim = _sim([a, b], furniture=False)
    _blast(sim, at=(SRC_AT[0], SRC_AT[1] - 2))
    _step(sim)
    assert [s.alive for s in sim._heat_sources] == [False, False]
    assert sum(isinstance(e, ExplosionEvent) and e.kind == EVENT_KIND
               for e in sim.tick_events) == 2


# ---------------------------------------------------------------------------
# Digest + dormancy
# ---------------------------------------------------------------------------

def test_enabled_and_hp_are_digested():
    """Property: two sims differing only in a source's ``enabled`` (or only
    in its live hp) differ in their ENTITY_SECT record and tick_digest.
    Breaks if ``enabled`` stops being a synced kind or the ``hp`` runtime
    row is dropped."""
    a = _sim([_source()], furniture=False, physics=None)
    b = _sim([_source(enabled=False)], furniture=False, physics=None)
    c = _sim([_source()], furniture=False, physics=None)
    c._heat_sources[0].hp_q16 -= 1
    (ra,), (rb,), (rc,) = (entity_records(s.entities) for s in (a, b, c))
    assert b"hp|" in ra and ra != rb and ra != rc
    da, db, dc = (tick_digest(capture_snapshot(s, capture_units=False))
                  for s in (a, b, c))
    assert da != db and da != dc


def test_a_source_free_level_is_untouched_by_the_blast_log():
    """Property: on a level with no heat source, a blast's log entry is
    cleared at 9e and nothing else changes -- the per-tick digest equals
    the same run with the log entry removed by hand before the step.
    Breaks if the log leaks into any synced state or grows unbounded."""
    a, b = _sim([], physics=bp), _sim([], physics=bp)
    for s in (a, b):
        _blast(s)
    b.gmap.blast_log.clear()
    for _ in range(5):
        _step(a)
        _step(b)
        assert a.gmap.blast_log == []
    assert (tick_digest(capture_snapshot(a, capture_units=False))
            == tick_digest(capture_snapshot(b, capture_units=False)))


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("over, words", [
    ({"death_payload": "no_such_row"}, "death_payload"),
    ({"heat_per_s": -1.0}, "heat_per_s"),
    ({"heat_per_s": 0.0}, "heat_per_s"),
    ({"heat_per_s": 1e-9}, "below one Q16"),
    ({"radius": 99}, "radius"),
    ({"hp": 0}, "hp"),
    ({"max_temp": -5.0}, "max_temp"),
    ({"x": 99}, "out of the"),
])
def test_the_door_refuses_a_bad_source_by_name(over, words):
    """Property: an unknown death payload, a non-positive or sub-LSB heat, a
    radius out of bounds, a non-positive hp, a negative cap and an off-grid
    tile are load errors naming the field. Breaks if any is silently
    accepted (a cold source, a typo'd payload that never fires)."""
    with pytest.raises(HeatSourceError, match=words):
        _sim([_source(**over)], furniture=False, physics=None)


def test_level_lib_round_trips_a_source(tmp_path):
    """Property: the one writer writes every heat_source field and the
    loader reads them back equal through the registry's door, re-formatting
    byte-identically. Breaks if a field's kind cannot be written or parsed."""
    lvl = tmp_path / "lvl"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="rt", tile_size_m=1.0)
    tm = np.full((8, 8), MAT_HULL, dtype=np.int32)
    tm[1:7, 1:7] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (8, 8)).save(lvl / "diffuse.png")
    keys = ("x", "y", "heat_per_s", "radius", "enabled", "hp",
            "death_payload", "max_temp")
    inst = EntityInstance(id="engine", class_name="heat_source", ordinal=0,
                          fields={"x": 3, "y": 4, "heat_per_s": 2.5,
                                  "radius": 1, "enabled": True, "hp": 250,
                                  "death_payload": "engine_death",
                                  "max_temp": 80.0},
                          authored_keys=keys)
    level_lib.write_managed_blocks(
        lvl / "level.toml",
        {"entity": lambda nl: level_lib.format_entity_lines([inst], nl)})
    data = level_loader.load("lvl", str(tmp_path))
    (back,) = data.entities
    assert {k: back.fields[k] for k in keys} == {k: inst.fields[k] for k in keys}
    assert (level_lib.format_entity_lines([back])
            == level_lib.format_entity_lines([inst]))
    sim = Simulation(data, seed=1, breach_physics=None, enable_recorder=False)
    assert len(sim._heat_sources) == 1


def test_the_engine_death_row_reaches_ignite_heat_two_tiles_out():
    """Property: the placeholder ``engine_death`` row carries a blast and at
    least IGNITE_HEAT on every tile within 2 of the engine (the deposit's
    LINEAR falloff), so the death satisfies the Starting-a-fire rule. Breaks
    if a retune drops the heat below the igniter's floor."""
    from simulation.payloads import IGNITE_HEAT
    from simulation.weapons import rebuild_tables
    p = rebuild_tables(tile_size_m=1.0).payloads.by_name["engine_death"]
    assert p.radius > 0 and p.wall_damage > 0
    assert p.heat_amount * (1.0 - 2.0 / p.heat_radius) >= IGNITE_HEAT


# ---------------------------------------------------------------------------
# gen_ship's heat_source tag
# ---------------------------------------------------------------------------

def _graph(tags):
    import gen_ship as G
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 10, 10, "section", tags),
             G.Room(2, 26, 4, 10, 10, "spawn", {"spawn_team": 1})]
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4)]
    return G.ShipGraph(width=40, height=18, hull=(3, 3, 36, 14), rooms=rooms,
                       edges=edges, mirror_x=False, max_per_team=4,
                       zombies=0)


HS_TAG = {"heat_per_s": 2.0, "radius": 1, "max_temp": 120.0,
          "death_payload": "engine_death"}


def test_a_heat_source_tagged_room_gets_one_source_and_loads(tmp_path):
    """Property: a gen_ship room tagged ``heat_source`` (beside a hazard and
    fuel) is written as exactly one heat_source entity carrying the tag's
    fields, on a floor tile of that room with no fuel in its 3x3 and not in
    the hazard charge's 3x3, and the level loads and steps headless with the
    source heating its tile. Breaks if stage 2 drops the tag, places the
    source on a wall, fuel or another entity, or writes a field the loader's
    door refuses."""
    import gen_ship as G
    hz = {"payload": "magazine_cookoff", "trigger_temp": 150.0,
          "fuse_s": 1.0}
    g = _graph({"fuel": 2, "hazard": hz, "heat_source": HS_TAG})
    emb = G.embed(g, np.random.Generator(np.random.PCG64(0)))
    assert G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                      graph=g) == []
    plan = G.ShipPlan(seed=0, size="fixture", attempt=0, graph=g,
                      embedding=emb)
    G.write(plan, tmp_path / "hs")
    lv = level_loader.load("hs", levels_dir=str(tmp_path))
    srcs = [e for e in lv.entities if e.class_name == "heat_source"]
    (chg,) = [e for e in lv.entities if e.class_name == "timed_charge"]
    assert len(srcs) == 1
    f = srcs[0].fields
    assert {k: f[k] for k in HS_TAG} == HS_TAG
    x, y = f["x"], f["y"]
    x0, y0, x1, y1 = g.rooms[1].box
    assert x0 <= x <= x1 and y0 <= y <= y1
    tm = plan.tilemap
    assert tm[y, x] == MAT_AIR
    assert not np.isin(tm[y - 1:y + 2, x - 1:x + 2],
                       (MAT_FURNITURE, MAT_WOOD)).any()
    assert max(abs(chg.fields["x"] - x), abs(chg.fields["y"] - y)) > 1
    sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
    (rt,) = sim._heat_sources
    for _ in range(5):
        _step(sim)
    assert rt.alive and int(sim.gmap.temperature[rt.fy, rt.fx]) > 0


def test_a_malformed_heat_source_tag_is_refused():
    """Property: the heat_source tag is schema-checked like every room tag.
    Breaks if a source without positive heat passes stage 1."""
    import gen_ship as G
    with pytest.raises(G.GraphSchemaError, match="heat_source"):
        _graph({"heat_source": dict(HS_TAG, heat_per_s=0)})
