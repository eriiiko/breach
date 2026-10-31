"""#106 P7 -- levels/octopus_tank (tools/gen_octopus_tank.py), the level the
octopus render is played in.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_octopus_tank_level.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_octopus_tank as GEN  # noqa: E402
import level_loader  # noqa: E402

LEVEL_DIR = ROOT / "levels" / "octopus_tank"
_OUTPUTS = ("level.toml", "tilemap.csv", "diffuse.png", "water_init.npy", "match.toml")


def test_generator_is_byte_deterministic(tmp_path):
    """PROPERTY: two generator runs are byte-identical and the committed
    folder is the tool's output (bytes for the text and the .npy, PIXELS for
    the PNG -- encoders differ across machines).

    BREAKS IF: the level is hand-edited instead of regenerated, or the
    generator draws anything unseeded."""
    a, b = tmp_path / "a", tmp_path / "b"
    GEN.main(a)
    GEN.main(b)
    for name in _OUTPUTS:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    for name in ("level.toml", "tilemap.csv", "water_init.npy", "match.toml"):
        assert (a / name).read_bytes() == (LEVEL_DIR / name).read_bytes(), name
    from PIL import Image
    assert np.array_equal(np.asarray(Image.open(a / "diffuse.png").convert("RGBA")),
                          np.asarray(Image.open(LEVEL_DIR / "diffuse.png").convert("RGBA")))


def test_the_tank_is_flooded_and_its_octopus_and_squad_stand_on_open_floor():
    """PROPERTY: the level loads through the loader's doors; every floor tile
    carries the flood depth (the chain's water-everywhere assumption); the
    octopus's rest pose and the squad are on open floor; match.toml drives
    the octopus with the creature controller against two teams.

    BREAKS IF: the room shrinks under the animal's 3 m reach, the flood
    misses a tile, or the match loses its creature."""
    from simulation import water_fixed
    from simulation.creature_system import check_creature_spawn
    lvl = level_loader.load("octopus_tank", str(ROOT / "levels"))
    depth = np.load(LEVEL_DIR / "water_init.npy")
    floor = depth > 0
    assert floor.sum() == (GEN.W - 2) * (GEN.H - 2)
    assert (depth[floor] == water_fixed.quantize_scalar(GEN.FLOOD_M)).all()
    assert lvl.creatures == [GEN.OCTOPUS]
    check_creature_spawn(lvl.creatures[0], floor, GEN.TILE_SIZE_M, "tank")
    assert len(lvl.spawns) == 3
    for s in lvl.spawns:
        assert floor[int(s.y):int(s.y) + 3, int(s.x):int(s.x) + 3].all()
    assert lvl.match is not None and lvl.match.creatures == (GEN.OCTOPUS,)
    assert lvl.match.creature_controller == "creature"
    assert len(lvl.match.teams) == 2
