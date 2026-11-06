"""#134 -- the GRILLE material: blocks walking, passes air.

``[materials.grille]`` (MAT_GRILLE) is a steel grating: mobility 0, but
permeability 1.0 and the GAS thermal regime, so to the gas it is exactly an
air tile (docs/levelgen_zone_vocabulary_2026-10-29.md §0.4, the 2026-11-06
addition). These tests run the real engine through the Simulation facade on
two sealed rooms joined only by a two-tile opening in the wall between them,
the opening made of grille, of steel (no opening) or of the legacy door
(walkable, flow-solid). The levels are written through ``level_lib`` and
loaded through the loader's door; pressure is seeded through
``air_init.npy`` (the sanctioned per-tile atmosphere seed), smoke through a
FieldEdit ADD.

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_lib  # noqa: E402
import level_loader  # noqa: E402
from config import CFG  # noqa: E402
from pathfinding import astar  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.atmosphere_fixed import FP_ONE  # noqa: E402
from simulation.field_edit import EditMode, Falloff, FieldEdit, Region  # noqa: E402
from simulation.gas_fixed import quantize_q16_scalar  # noqa: E402
from simulation.gases import SMOKE  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_DOOR, MAT_GRILLE, MAT_HULL, MAT_STEEL, MATERIAL_NAMES,
    MaterialTable,
)
from test_heat_source import _closure_bad  # noqa: E402  (THE closure helper)

H, W, WALL_X = 12, 27, 13          # room A: x 1..12, room B: x 14..25
OPENING = (5, 6)                   # the two wall rows the opening takes
A_BOX = (1, 1, 12, H - 2)          # (x0, y0, x1, y1) inclusive
B_BOX = (WALL_X + 1, 1, W - 2, H - 2)
P_HIGH = FP_ONE * 3 // 2           # room A at 1.5 atm, room B at 1.0
T_HOT = 100.0                      # room A's gas, game degrees above ambient
TICKS = 72                         # 3 s at 24 tps
GAS_PLANES = ("atmosphere", "gas", "gas_energy", "temperature", "wind_x",
              "wind_y")


def _tilemap(opening: int) -> np.ndarray:
    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = MAT_AIR
    tm[1:H - 1, WALL_X] = MAT_STEEL
    for y in OPENING:
        tm[y, WALL_X] = opening
    return tm


def _level(tmp_path, opening: int, *, pressurise: bool):
    name = f"two_rooms_{MATERIAL_NAMES[opening]}_{int(pressurise)}"
    lvl = tmp_path / name
    lvl.mkdir(parents=True)
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = _tilemap(opening)
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    if pressurise:                 # room A: 1.5 atm AND 100 degrees hotter
        x0, y0, x1, y1 = A_BOX
        air = np.full((H, W), FP_ONE, dtype=np.int32)
        air[y0:y1 + 1, x0:x1 + 1] = P_HIGH
        level_lib.write_air_init_npy(lvl, air, npy_bak=False)
        hot = np.zeros((H, W), dtype=np.int32)
        hot[y0:y1 + 1, x0:x1 + 1] = quantize_q16_scalar(T_HOT)
        level_lib.write_temp_init_npy(lvl, hot, npy_bak=False)
    from PIL import Image
    Image.new("RGB", (W, H)).save(lvl / "diffuse.png")
    return level_loader.load(name, str(tmp_path))


def _sim(tmp_path, opening, *, pressurise=True):
    return Simulation(_level(tmp_path, opening, pressurise=pressurise),
                      seed=1, breach_physics=bp, enable_recorder=False)


def _mean_p(g, box) -> float:
    x0, y0, x1, y1 = box
    return float(g.atmosphere[y0:y1 + 1, x0:x1 + 1].astype(np.int64).mean()
                 / FP_ONE)


def _box_sum(field, box) -> int:
    x0, y0, x1, y1 = box
    return int(field[y0:y1 + 1, x0:x1 + 1].astype(np.int64).sum())


def _step(sim):
    sim.set_paused(False)
    sim.step()


# ---------------------------------------------------------------------------
# The material door
# ---------------------------------------------------------------------------

def test_the_grille_row_is_air_to_the_gas_and_a_wall_to_walking():
    """PROPERTY: the shipped grille row passes the material table's doors
    and reads, column by column, as the brief says: not walkable (mobility
    0), fully permeable (not solid), in the GAS thermal regime (so it is on
    the gas-energy accountable set like air, with heat_atten 0 as that
    regime requires), its conduction faces equal to air's, light-passing
    (0 <= light_atten < 1), not flammable, never bursting, with hp so a
    blast can open it. And the rows that existed before it keep their ids
    (compared by NAME). BREAKS IF the row is authored as a thermal solid or
    a partial permeability (gas would not cross it as air), made walkable,
    inserted before an existing row (every level re-materializes), or a
    face to air differs from air's own."""
    tbl = MaterialTable.from_config(CFG)
    g = MAT_GRILLE
    assert tbl.names[g] == "grille"
    assert int(tbl.mobility[g]) == 0
    assert float(tbl.permeability[g]) == 1.0
    assert not bool(tbl.thermal_solid[g])
    assert float(tbl.heat_atten[g]) == 0.0
    assert 0.0 <= float(tbl.light_atten[g].max()) < 1.0
    assert not bool(tbl.flammable[g])
    assert float(tbl.burst_threshold[g]) <= 0.0
    assert float(tbl.hp[g]) > 0.0
    air = MAT_AIR
    for other in range(tbl.n):
        assert tbl.face_shift_table[g, other] == tbl.face_shift_table[air,
                                                                      other]
    from simulation import physics
    assert g in physics._BLAST_WALL_MATERIALS
    # Every row that existed before #134 keeps its id: by name, the ids of
    # the first rows are the table's positions, and the grille is the last.
    before = ("air", "hull", "wood", "door", "steel", "glass", "furniture",
              "door_closed", "kindling", "foliage")
    for name in before:
        assert MATERIAL_NAMES[tbl.names.index(name)] == name
        assert tbl.names.index(name) < g
    assert g == max(MATERIAL_NAMES)


# ---------------------------------------------------------------------------
# Air: pressure, smoke, energy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each opening's run (3 s; room A high and hot), the energy ledger
    checked every tick, the gas planes kept at the end. The AIR opening is
    the reference the grille must match."""
    out = {}
    for mat in (MAT_GRILLE, MAT_AIR, MAT_STEEL, MAT_DOOR):
        sim = _sim(tmp_path_factory.mktemp(MATERIAL_NAMES[mat]), mat)
        g = sim.gmap
        d0 = _mean_p(g, A_BOX) - _mean_p(g, B_BOX)
        e_b0 = _box_sum(g.gas_energy, B_BOX)
        bad = _closure_bad(sim, TICKS)
        out[mat] = dict(d0=d0, d=_mean_p(g, A_BOX) - _mean_p(g, B_BOX),
                        bad=bad, e_b=_box_sum(g.gas_energy, B_BOX) - e_b0,
                        planes={k: np.array(getattr(g, k)).copy()
                                for k in GAS_PLANES})
    return out


def test_two_rooms_joined_by_a_grille_equalise_and_steel_or_a_door_do_not(
        runs):
    """PROPERTY: two sealed rooms at 1.5 and 1.0 atm joined ONLY by a
    two-tile grille equalise -- the pressure difference decays to a small
    fraction of its start within 3 s -- while the same rooms joined by
    steel, or by the legacy door (walkable but flow-solid), keep almost all
    of it. BREAKS IF the grille row stops being permeable (it decays like
    steel), or the legacy door starts passing gas (it decays like the
    grille)."""
    for mat in runs:
        assert runs[mat]["d0"] > 0.45, (MATERIAL_NAMES[mat], runs[mat])
    assert abs(runs[MAT_GRILLE]["d"]) < 0.1 * runs[MAT_GRILLE]["d0"], runs
    for mat in (MAT_STEEL, MAT_DOOR):
        assert runs[mat]["d"] > 0.9 * runs[mat]["d0"], (MATERIAL_NAMES[mat],
                                                        runs[mat])


def test_the_energy_closure_identity_holds_exactly_through_a_grille(runs):
    """PROPERTY (arc #54's whole-tick closure identity, the full ledger of
    test_heat_source's helper): while hot, dense gas and its energy rush
    through the grille, every tick closes exactly in int64, and room B's gas
    energy really grows (non-vacuous). BREAKS IF a writer moves gas energy
    on or through a grille cell outside the four counted groups."""
    for mat in runs:
        assert runs[mat]["bad"] == [], (MATERIAL_NAMES[mat],
                                        runs[mat]["bad"][:5])
    assert runs[MAT_GRILLE]["e_b"] > 0


def test_to_the_gas_a_grille_is_exactly_an_air_tile(runs):
    """PROPERTY (the row's design): the same hot, pressurised two-room run
    with the opening made of GRILLE and made of AIR ends bit-identical in
    every gas plane -- atmosphere, the gas stack (bulk and trace), gas
    energy, the temperature mirror, the wind. Gas, smoke, pressure and gas
    energy cross a grille exactly as they cross air; only walking differs.
    BREAKS IF the grille row is made a permeable THERMAL SOLID (its cells
    leave the accountable set and carry no energy face), given a partial
    permeability, wave damping, conduction or heat extinction unlike air's
    -- or if any gas-side system starts reading mobility."""
    for k in GAS_PLANES:
        assert np.array_equal(runs[MAT_GRILLE]["planes"][k],
                              runs[MAT_AIR]["planes"][k]), k


def test_smoke_released_in_one_room_reaches_the_other_through_a_grille(
        tmp_path):
    """PROPERTY: smoke (a trace gas) released in room A at equal pressure
    reaches room B through the grille within 5 s, and never through steel.
    BREAKS IF trace transport (the face flux and diffusion) treats a grille
    tile as a wall, or a steel wall starts leaking trace gas."""
    got = {}
    for mat in (MAT_GRILLE, MAT_STEEL):
        lvl_dir = tmp_path / f"smoke_{MATERIAL_NAMES[mat]}"
        lvl_dir.mkdir()
        sim = _sim(lvl_dir, mat, pressurise=False)
        sim.edit_queue.enqueue(FieldEdit(
            field="smoke", region=Region.DISC, coords=(5.5, WALL_X - 3, 2.0),
            amount=4.0, mode=EditMode.ADD, falloff=Falloff.FLAT,
            source_id=991))
        for _ in range(5 * int(CFG.clock.ticks_per_second)):
            _step(sim)
        smoke = sim.gmap.gas[SMOKE]
        assert _box_sum(smoke, A_BOX) > 0, "no smoke in room A: vacuous"
        got[mat] = _box_sum(smoke, B_BOX)
    assert got[MAT_GRILLE] > 0
    assert got[MAT_STEEL] == 0


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------

def test_a_unit_can_neither_stand_on_nor_path_through_a_grille(tmp_path):
    """PROPERTY: no 3x3 footprint that covers a grille tile is enterable,
    and A* (the one pathfinder, on GameMap.is_passable_block -- the
    predicate movement and the bots use) finds no route between two rooms
    joined only by a grille, while the same rooms joined by a door-wide run
    of legacy door tiles flanked by grilles (a vented door) have one.
    BREAKS IF the grille row gains mobility, or the walk predicate starts
    reading permeability instead of mobility."""
    g = _sim(tmp_path / "g", MAT_GRILLE, pressurise=False).gmap
    y = OPENING[0]
    assert not g.is_passable(y, WALL_X)
    for oy in range(y - 2, y + 1):
        for ox in range(WALL_X - 2, WALL_X + 1):
            assert not g.is_passable_block(oy, ox, 3)
    ay, ax, by, bx = 4, 4, 4, W - 6

    def route(gmap):
        return astar(ax, ay, bx, by,
                     lambda x, yy: not gmap.is_passable_block(yy, x, 3),
                     W, H)

    assert not route(g)
    # The vented door: four door tiles in the wall, a grille either side.
    tm = _tilemap(MAT_STEEL)
    tm[3:7, WALL_X] = MAT_DOOR
    tm[2, WALL_X] = tm[7, WALL_X] = MAT_GRILLE
    lvl = tmp_path / "vented"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="vented", tile_size_m=1.0)
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (W, H)).save(lvl / "diffuse.png")
    gv = Simulation(level_loader.load("vented", str(tmp_path)), seed=1,
                    breach_physics=bp, enable_recorder=False).gmap
    assert route(gv)
    assert not gv.solid[2, WALL_X] and gv.solid[4, WALL_X]


# ---------------------------------------------------------------------------
# Sight (Erik's ruling 2026-11-06: the grille is NOT see-through)
# ---------------------------------------------------------------------------

def test_a_grille_stops_a_line_of_sight_but_not_what_travels_with_the_air(
        tmp_path):
    """PROPERTY: across a grille opening no line of SIGHT is clear (the map's
    ``has_los`` and the one sight oracle ``vision.ray_clear`` both say no),
    exactly as across steel, while the PHYSICAL line (``physical=True``, what
    a flame jet's cone asks) is clear, exactly as across an air opening --
    and a spray cone aimed through the grille reaches tiles of the far room.
    BREAKS IF the grille row loses ``blocks_sight``, if ``has_los`` goes back
    to reading ``solid`` for sight, or if the jet starts reading the sight
    mask (flames would stop at a grating that passes air)."""
    from simulation import vision
    from simulation.combat import spray_cone_tiles
    y = OPENING[0]
    ax, bx = WALL_X - 4, WALL_X + 4
    sims = {m: _sim(tmp_path / MATERIAL_NAMES[m], m, pressurise=False)
            for m in (MAT_GRILLE, MAT_STEEL, MAT_AIR)}
    sight = {m: s.gmap.has_los(y, ax, y, bx) for m, s in sims.items()}
    oracle = {m: vision.ray_clear(s, y, ax, y, bx) for m, s in sims.items()}
    phys = {m: s.gmap.has_los(y, ax, y, bx, physical=True)
            for m, s in sims.items()}
    assert sight == {MAT_GRILLE: False, MAT_STEEL: False, MAT_AIR: True}
    assert oracle == sight
    assert phys == {MAT_GRILLE: True, MAT_STEEL: False, MAT_AIR: True}

    def far_side(m):
        return any(x > WALL_X for (_y, x, _d) in spray_cone_tiles(
            sims[m].gmap, y, ax, bx, y, 10, 20.0))

    assert far_side(MAT_GRILLE) and far_side(MAT_AIR)
    assert not far_side(MAT_STEEL)


def test_sight_is_sealed_to_gas_unless_a_row_says_otherwise(tmp_path):
    """PROPERTY: a material row that does not declare ``blocks_sight`` blocks
    sight iff it is sealed to gas (permeability == 0) -- the rule line of
    sight had before the column existed, so no such row's behaviour moved --
    and on a loaded map the sight mask is the solid mask plus exactly the
    tiles of declaring rows, and follows a tile when it changes.
    BREAKS IF the derived default changes (glass, doors or crates would
    change who sees whom), or the sight mask is not patched with its tile."""
    tbl = MaterialTable.from_config()
    declared = {name for name in tbl.names
                if getattr(getattr(CFG.materials, name), "blocks_sight",
                           None) is not None}
    assert "grille" in declared
    for i, name in enumerate(tbl.names):
        if name not in declared:
            assert bool(tbl.blocks_sight[i]) == bool(tbl.permeability[i] <= 0)
    assert tbl.blocks_sight[MAT_GRILLE] and tbl.permeability[MAT_GRILLE] > 0

    g = _sim(tmp_path, MAT_GRILLE, pressurise=False).gmap
    is_declared = np.isin(g.material, [i for i, n in enumerate(tbl.names)
                                       if n in declared])
    assert np.array_equal(g.sight_block & ~is_declared, g.solid & ~is_declared)
    y = OPENING[0]
    assert g.sight_block[y, WALL_X] and not g.solid[y, WALL_X]
    g.destroy_wall(y, WALL_X)               # the grating is blown out
    assert not g.sight_block[y, WALL_X]
    assert g.has_los(y, WALL_X - 4, y, WALL_X + 4)
    # ... and the other way: a tile that BECOMES a grille (through the one
    # tile-change seam) blocks sight from that moment, still open to the gas.
    g.material[y, WALL_X] = MAT_GRILLE
    g.on_tile_changed(y, WALL_X)
    assert g.sight_block[y, WALL_X] and not g.solid[y, WALL_X]
    assert not g.has_los(y, WALL_X - 4, y, WALL_X + 4)
