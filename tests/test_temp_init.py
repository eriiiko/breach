"""The INITIAL-TEMPERATURE level field, ``temp_init.npy`` (levelgen vocabulary
E.2 row 4, docs/levelgen_zone_vocabulary_2026-10-29.md; B.2's hot
engineering).

An optional int32 Q16.16 grid beside level.toml, in GAME DEGREES ABOVE
AMBIENT (0 == ambient, so an absent file and an all-zero file are the same
level), refused at the loader door by name (dtype, shape, sub-ambient -- a
parked physics ruling --, above T_MAX_PHYS), written only by
``level_lib.write_temp_init_npy``, and seeded once by ``GameMap.__init__``
after the air seed: gas cells through ``seed_gas_temperature`` (mirror and
energy together), thermal solids on their own ``temperature`` truth,
anything else refused.

Each test names the property it protects and the change that must break it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_temp_init.py -q
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
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.gas_fixed import quantize_q16_scalar  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_FURNITURE,  # noqa: E402
                                  MAT_HULL, MAT_WOOD)
from test_heat_source import _gas_books, _terms  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
HOT = 80.0                       # game degrees: hot, below any ignition
HOT_Q = quantize_q16_scalar(HOT)
HOT_BOX = (slice(2, 6), slice(2, 6))     # (y, x) a hot corner of the room
FURN_AT = (10, 10)


def _tilemap(hh=16):
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    tm[FURN_AT] = MAT_FURNITURE
    return tm


def _level(temp_q=None, tm=None):
    tm = _tilemap() if tm is None else tm
    return LevelData(name="temp_init", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=[], wires=[], temp_init_q=temp_q)


def _seed(gas=True, solid=True):
    t = np.zeros(_tilemap().shape, dtype=np.int32)
    if gas:
        t[HOT_BOX] = HOT_Q
    if solid:
        t[FURN_AT] = HOT_Q
    return t


def _sim(level, seed=3):
    return Simulation(level, seed=seed, breach_physics=bp,
                      enable_recorder=False)


def _step(sim):
    sim.set_paused(False)
    sim.step()


def _solid_books_now(g):
    """Σ cap_real·T over thermal solids, priced by the engine's own kit
    (GameMap._solid_books_cell) -- the value the solver's snapshot will
    hold, available BEFORE the first step (the snapshot is 0 until then)."""
    ys, xs = np.nonzero(g.thermal_solid)
    return sum(g._solid_books_cell(int(y), int(x)) for y, x in zip(ys, xs))


# ---------------------------------------------------------------------------
# The seed at load
# ---------------------------------------------------------------------------

def test_seeded_gas_cells_load_at_the_seed_with_energy_consistent():
    """Property: after load, every seeded gas cell's temperature MIRROR
    equals the seed, its ``gas_energy`` is exactly ``N·(T + T_amb)``, and
    the mirror re-derived from that energy (``E // N − T_amb``) is the seed
    to the count; unseeded gas cells keep the ambient energy. Breaks if the
    seed writes the mirror without the energy (a bare ``temperature[...] =``
    after ``refresh_gas_energy``), runs before the air seed, or lands in the
    wrong currency."""
    sim = _sim(_level(_seed(solid=False)))
    g = sim.gmap
    t_amb = g._gas_energy_t_amb_raw()
    n = g._gas_bulk_n_raw()
    e = g.gas_energy.astype(np.int64)
    hot = np.zeros(g.temperature.shape, dtype=bool)
    hot[HOT_BOX] = True
    assert g._gas_energy_accountable()[hot].all()
    assert (g.temperature[hot] == HOT_Q).all()
    assert (e[hot] == n[hot] * (HOT_Q + t_amb)).all()
    assert ((e[hot] // n[hot]) - t_amb == HOT_Q).all()
    cold = g._gas_energy_accountable() & ~hot
    assert (g.temperature[cold] == 0).all()
    assert (e[cold] == n[cold] * t_amb).all()


def test_a_seeded_thermal_solid_holds_its_seed_at_load():
    """Property: a thermal solid seeded hot carries the seed on its own
    ``temperature`` at load and no ``gas_energy`` (it is not gas). Breaks if
    the seed skips solids or routes them through the gas seam."""
    g = _sim(_level(_seed(gas=False))).gmap
    assert g.thermal_solid[FURN_AT]
    assert int(g.temperature[FURN_AT]) == HOT_Q
    assert int(g.gas_energy[FURN_AT]) == 0
    assert _solid_books_now(g) > 0


def test_the_closure_identity_holds_from_tick_one_with_hot_gas_and_solid():
    """Property: with hot gas AND a hot thermal solid seeded at load, the
    across-tick closure identity ``Δ(Σ gas_energy + solid books) == EOS +
    thermal + combustion + seams + water`` is exact in int64 on EVERY tick,
    tick 1 included, measured from the load state (the solid books priced
    at load by the engine's own capacity kit). Breaks if the seed lands
    after the books open (a mint on tick 1), leaves the energy out of
    agreement with the mirror (the recovery would move E), or the solid
    seed is booked somewhere it should not be."""
    sim = _sim(_level(_seed()))
    g = sim.gmap
    rp = sim.physics_runner
    eos, ts, comb, engine = (rp.eos, rp.engine.temperature, rp.combustion,
                             rp.engine)
    prev_total = _gas_books(g) + _solid_books_now(g)
    prev = _terms(g, eos, ts, comb, engine)
    bad = []
    for i in range(2 * TPS):
        _step(sim)
        total = _gas_books(g) + int(ts.solid_energy_books_sum)
        t = _terms(g, eos, ts, comb, engine)
        expected = (t[0] + (t[1] - prev[1]) + (t[2] - prev[2])
                    + (t[3] - prev[3]) + t[4] + (t[5] - prev[5]))
        resid = (total - prev_total) - expected
        if resid:
            bad.append((i, resid))
        prev_total, prev = total, t
    assert int(ts.e_gas_cond_sum) != 0 or int(ts.e_solid_cond_sum) != 0, \
        "vacuous: no heat moved"
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


def test_a_seeded_room_evolves_under_the_solver_without_a_rail():
    """Property: a seeded HOT ROOM (all its gas and the thermal solid inside
    it at the seed, the shape gen_ship's ``temp`` tag and B.2's hot
    engineering author) starts at its seed and only the solver moves it:
    after one tick it is still hot (not reset to ambient), over two seconds
    it never exceeds the seed (nothing adds heat) and stays well above
    ambient, and no rail engages -- the EOS rail books nothing, the thermal
    solver's gas rail and solid low rail never hit, T_MAX_PHYS never binds.
    Breaks if the seed is overwritten at bind time, a seed above the
    physical maximum slips through, or mirror and energy disagree (the
    recovery's rails would write back).

    NOT covered, on purpose: a hot thermal solid seeded in AMBIENT air. Its
    step edge drives the adjacent gas a few degrees sub-ambient on tick 1
    and the solids that air touches then hit the low rail (measured
    2026-10-31: one furniture tile at 80 deg, 1094 low-rail hits in 2 s; a
    hot room around it, 0) -- a pre-existing response to any step-heated
    solid, reported with this patch, not fixed by it."""
    t = np.zeros(_tilemap().shape, dtype=np.int32)
    t[1:15, 1:15] = HOT_Q
    sim = _sim(_level(t))
    g = sim.gmap
    rp = sim.physics_runner
    eos, ts = rp.eos, rp.engine.temperature
    room = g._gas_energy_accountable() | g.thermal_solid
    room[0, :] = room[-1, :] = room[:, 0] = room[:, -1] = False
    _step(sim)
    assert int(g.temperature[room].min()) > HOT_Q // 2, "seed lost on tick 1"
    peak = []
    for _ in range(2 * TPS):
        _step(sim)
        peak.append(float(g.temperature[room].mean()))
    assert max(peak) <= HOT_Q
    assert int(g.temperature[room].min()) > HOT_Q // 2
    assert int(eos.e_rail_sum) == 0
    assert int(eos.t_max_phys_hits) == 0
    assert int(ts.e_gas_rail_sum) == 0
    assert int(ts.t_low_rail_hits) == 0
    assert int(ts.e_solid_low_rail_sum) == 0


def test_no_seed_and_an_all_zero_seed_are_the_same_level():
    """Property: ``temp_init_q`` None and an all-zero grid give bit-identical
    sims after a second of ticks (``tick_digest``). Breaks if the seed path
    does anything on a zero grid (a re-derivation of the energy, a write
    that perturbs the mirror)."""
    a = _sim(_level(None))
    b = _sim(_level(np.zeros(_tilemap().shape, dtype=np.int32)))
    for _ in range(TPS):
        _step(a)
        _step(b)
    assert tick_digest(capture_snapshot(a)) == tick_digest(capture_snapshot(b))


def test_a_seed_on_the_vacuum_ring_is_refused():
    """Property: a non-zero seed on the vacuum ring is refused by name at
    load -- only gas cells and thermal solids hold a temperature; the ring
    is a boundary condition, not state. (A wall without thermal mass is
    refused by the same mask; no shipped material is one today.) Breaks if
    the seed silently writes a hot vacuum."""
    tm = np.full((18, 18), level_loader.SPACE_CODE, dtype=np.int32)
    tm[1:17, 1:17] = MAT_HULL
    tm[2:16, 2:16] = MAT_AIR
    t = np.zeros(tm.shape, dtype=np.int32)
    t[0, 0] = HOT_Q
    lv = LevelData(name="bad", version="3", path=Path("."), tilemap=tm,
                   tile_size_m=1.0, diffuse_path=Path("."), entities=[],
                   wires=[], temp_init_q=t)
    with pytest.raises(ValueError, match="temp_init.npy.*without thermal "
                                         "mass, vacuum or the ambient ring"):
        _sim(lv)


# ---------------------------------------------------------------------------
# The file: loader door + the one writer
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


def _ship(tmp_path, tags, name="ship"):
    import gen_ship as G
    g = _graph(tags)
    emb = G.embed(g, np.random.Generator(np.random.PCG64(0)))
    assert G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                      graph=g) == []
    plan = G.ShipPlan(seed=0, size="fixture", attempt=0, graph=g,
                      embedding=emb)
    G.write(plan, tmp_path / name)
    return g, plan, tmp_path / name


@pytest.mark.parametrize("grid, words", [
    (lambda s: np.full(s, 1.0, dtype=np.float32), "dtype int32"),
    (lambda s: np.zeros((s[0] + 1, s[1]), dtype=np.int32), "shape"),
    (lambda s: np.full(s, -1, dtype=np.int32), "sub-ambient"),
    (lambda s: np.full(s, quantize_q16_scalar(16000.0) + 1, dtype=np.int32),
     "T_MAX_PHYS"),
])
def test_the_loader_door_refuses_a_bad_file_by_name(tmp_path, grid, words):
    """Property: a temp_init.npy of the wrong dtype or shape, with a
    sub-ambient value (a parked physics ruling) or a value above the solvers'
    physical maximum, is refused at load with a message naming the file and
    the fault. Breaks if any of the four checks is dropped. (The file is
    hand-written here on purpose: the one writer refuses to write it.)"""
    _g, plan, d = _ship(tmp_path, {})
    np.save(d / "temp_init.npy", grid(plan.tilemap.shape))
    with pytest.raises(ValueError, match=f"temp_init.npy.*{words}|"
                                         f"{words}.*temp_init.npy"):
        level_loader.load(d.name, levels_dir=str(tmp_path))


def test_the_sub_ambient_refusal_says_why():
    """Property: the sub-ambient refusal names the parked ruling, so an
    author knows it is a physics decision, not a typo. Breaks if the
    message loses the reason."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        tm = _tilemap()
        np.save(base / "temp_init.npy", np.full(tm.shape, -5, dtype=np.int32))
        with pytest.raises(ValueError, match="Part F item 3"):
            level_loader._parse_temp_init_grid(base, tm)


def test_level_lib_round_trips_and_deletes_a_dormant_grid(tmp_path):
    """Property: ``write_temp_init_npy`` writes an int32 file the loader
    reads back equal; None and an all-zero grid delete it (0 is ambient, so
    a dormant file never exists); it refuses what the door refuses
    (sub-ambient, above T_MAX_PHYS, non-integer). Breaks if the writer and
    the door drift apart or a dormant file is left behind."""
    tm = _tilemap()
    t = _seed()
    _, has = level_lib.write_temp_init_npy(tmp_path, t.astype(np.int64))
    assert has
    back = level_loader._parse_temp_init_grid(tmp_path, tm)
    assert back.dtype == np.int32 and np.array_equal(back, t)
    _, has = level_lib.write_temp_init_npy(tmp_path, np.zeros_like(t))
    assert not has and not (tmp_path / "temp_init.npy").exists()
    level_lib.write_temp_init_npy(tmp_path, t)
    level_lib.write_temp_init_npy(tmp_path, None)
    assert not (tmp_path / "temp_init.npy").exists()
    for bad in (-t, t * 0 + quantize_q16_scalar(16000.0) + 1,
                t.astype(np.float32)):
        with pytest.raises(ValueError):
            level_lib.write_temp_init_npy(tmp_path, bad)
    assert not (tmp_path / "temp_init.npy").exists()


# ---------------------------------------------------------------------------
# gen_ship's temp tag
# ---------------------------------------------------------------------------

def test_a_temp_tagged_room_loads_hot_and_its_neighbours_ambient(tmp_path):
    """Property: a gen_ship room tagged ``temp`` writes temp_init.npy with
    that room's floor AIR at the quantized tag (fuel clusters left ambient),
    and the level loads with those gas cells' mirror at the seed while the
    untagged rooms are ambient. Breaks if stage 2 drops the tag, seeds walls
    or fuel, or writes a file the loader refuses."""
    g, plan, d = _ship(tmp_path, {"temp": HOT, "fuel": 2})
    assert (d / "temp_init.npy").is_file()
    lv = level_loader.load(d.name, levels_dir=str(tmp_path))
    sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
    tm = plan.tilemap
    x0, y0, x1, y1 = g.rooms[1].box
    box = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    air = tm[box] == MAT_AIR
    fuel = np.isin(tm[box], (MAT_FURNITURE, MAT_WOOD))
    assert fuel.any(), "vacuous: no fuel in the tagged room"
    t = sim.gmap.temperature[box]
    assert (t[air] == HOT_Q).all()
    assert (t[fuel] == 0).all()
    for other in (0, 2):
        x0, y0, x1, y1 = g.rooms[other].box
        assert (sim.gmap.temperature[y0:y1 + 1, x0:x1 + 1] == 0).all()


def test_an_untagged_ship_writes_no_temp_file(tmp_path):
    """Property: with no room tagged ``temp`` stage 2 writes NO temp_init.npy
    (dormancy: generated levels without hot rooms are byte-identical to
    before the field), and a regenerate removes a stale one. Breaks if the
    writer emits an all-zero file or write() forgets the file in its
    regenerated set."""
    _g, _plan, d = _ship(tmp_path, {})
    assert not (d / "temp_init.npy").exists()
    np.save(d / "temp_init.npy", np.zeros((18, 40), dtype=np.int32))
    _ship(tmp_path, {})
    assert not (d / "temp_init.npy").exists()


@pytest.mark.parametrize("bad", [0, -5.0, "hot", True])
def test_a_malformed_temp_tag_is_refused(bad):
    """Property: the temp tag is schema-checked like every room tag (a
    positive number). Breaks if a zero, negative or non-numeric temp passes
    stage 1."""
    import gen_ship as G
    with pytest.raises(G.GraphSchemaError, match="temp"):
        _graph({"temp": bad})
