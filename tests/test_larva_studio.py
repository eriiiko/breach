"""The larva studio (arc #63 P4): tools/gen_larva_studio.py -> levels/larva_studio.

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_larva_studio as gen  # noqa: E402

FILES = ("level.toml", "tilemap.csv", "diffuse.png", "zones.npy", "temp_init.npy")


def test_the_studio_generates_deterministically_and_is_the_checked_in_one(tmp_path):
    """Property: two runs of the generator write byte-identical folders, equal
    to the checked-in levels/larva_studio. Breaks if the generator draws, or
    the level is edited without re-running it."""
    gen.main(tmp_path / "a")
    gen.main(tmp_path / "b")
    for name in FILES:
        a = (tmp_path / "a" / name).read_bytes()
        assert a == (tmp_path / "b" / name).read_bytes(), name
        assert a == (ROOT / "levels" / gen.LEVEL_NAME / name).read_bytes(), name


def test_the_studio_passes_the_lint(tmp_path):
    """Property: the studio as the generator writes it is airtight and its
    broods can spawn. Breaks if the hull leaks or a brood loses its free
    tiles."""
    import level_airtight
    gen.main(tmp_path / gen.LEVEL_NAME)
    assert level_airtight.check(gen.LEVEL_NAME, str(tmp_path)) is True


def test_larvae_reach_the_warm_rooms_comfort_band():
    """Property: over a headless minute some larvae are alive in the warm
    room within 5 K of their preferred temperature (generous: the probe saw
    ~30 inside at 60 s). Breaks if the warm room stops being warm, the
    doorway closes, or the broods cannot leave the nursery."""
    bp = pytest.importorskip("breach_physics")
    import level_loader
    from simulation import swarm
    from simulation.ruleset import ContinuousRealtime
    from simulation.simulation import Simulation
    lv = level_loader.load(gen.LEVEL_NAME, str(ROOT / "levels"))
    sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    assert sim.swarm.live_count("larva") == sum(b[3] for b in gen.BROODS)
    for _ in range(1440):
        sim.set_paused(False)
        sim.step()
    g = sim.gmap
    v = swarm.swarm_column(g, "larva", "valid") != 0
    x = swarm.swarm_column(g, "larva", "pos_x")[v] >> 16
    y = swarm.swarm_column(g, "larva", "pos_y")[v] >> 16
    t_pref = g.swarm_species.row("larva").T_prefer_q
    near = np.abs(g.temperature[y, x].astype(np.int64) - t_pref) <= 5 * (1 << 16)
    x0, y0, x1, y1 = gen.WARM
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    assert int((inside & near).sum()) >= 3
