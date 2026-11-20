"""The swarm dot view (arc #63 P4): its pure vertex builders and that it is
render-only. The GL face is exercised by the e2e still (docs/swarm_P4_impl.md).

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import swarm_dots as sd  # noqa: E402
from simulation import swarm, swarm_fixed  # noqa: E402

ONE = swarm_fixed.FP_ONE
PPT = 48.0


def _store(states, valid=None):
    n = len(states)
    x = (np.arange(n, dtype=np.int64) * ONE + ONE // 2).astype(np.int32)
    y = np.full(n, 3 * ONE, dtype=np.int32)
    h = np.zeros(n, dtype=np.int32)
    v = np.ones(n, dtype=np.uint8) if valid is None else np.asarray(valid, np.uint8)
    return x, y, h, np.asarray(states, dtype=np.int32), v


def test_one_sprite_per_live_unit_coloured_by_state():
    """Property: pack_dots gives one sprite per VALID slot at its position;
    a crawling unit (RUN, EAT) is bone white with a heading tick of 0.28
    tile, a COMA unit blue-grey with no tick; the coma code is the store's.
    Breaks if an empty slot is drawn, a state is coloured wrongly or the
    coma tick is drawn."""
    assert sd.AI_COMA == swarm.AI_COMA
    st = [swarm.AI_RUN, swarm.AI_EAT, swarm.AI_COMA, swarm.AI_RUN]
    x, y, h, s, v = _store(st, valid=[1, 1, 1, 0])
    out = sd.pack_dots(x, y, h, s, v, PPT)
    assert out.shape == (2, 3, 4) and out.dtype == np.float32
    assert np.allclose(out[0, :, 0], (np.arange(3) + 0.5) * PPT)
    assert np.allclose(out[1, 0, :3], sd.CRAWL_LIN) and np.allclose(out[1, 1, :3], sd.CRAWL_LIN)
    assert np.allclose(out[1, 2, :3], sd.COMA_LIN)
    assert np.allclose(out[0, 0, 2:], (sd.TICK_LENGTH_TILES * PPT, 0.0), atol=1e-3)
    assert np.allclose(out[0, 2, 2:], 0.0)
    assert (out[1, :, 3] == 1.0).all()


def test_the_tick_follows_the_heading():
    """Property: the tick points along (cos h, sin h), the law's step
    direction (y down). Breaks if the heading is read in another frame."""
    x, y, _h, s, v = _store([swarm.AI_RUN])
    h = np.array([swarm_fixed.quantize_scalar(np.pi / 2)], dtype=np.int32)
    out = sd.pack_dots(x, y, h, s, v, PPT)
    assert np.allclose(out[0, 0, 2:], (0.0, sd.TICK_LENGTH_TILES * PPT), atol=1e-3)


def test_flecks_fade_over_their_life_and_keep_the_newest():
    """Property: a fleck is opaque at its death, half at 0.75 s, gone at 1.5
    s (pruned), dark red-brown; past the cap the oldest go first. Breaks if
    the fade, its length, the colour or the cap's order changes."""
    assert sd.fleck_alpha([0.0, 0.75, 1.5, 3.0]).tolist() == [1.0, 0.5, 0.0, 0.0]
    f = sd.DeathFlecks(cap=4)
    f.add([1.0, 2.0], [1.0, 1.0], 10.0)
    out = f.pack(10.75, PPT)
    assert out.shape[1] == 2 and np.allclose(out[1, :, 3], 0.5)
    assert np.allclose(out[1, :, :3], sd.FLECK_LIN) and np.allclose(out[0, :, 2:], 0.0)
    assert f.pack(11.5, PPT).shape[1] == 0 and len(f) == 0
    f.add(np.arange(6.0), np.zeros(6), 20.0)
    assert len(f) == 4 and f.x.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_the_builders_never_write_the_store():
    """Property: pack_dots reads read-only views of the store without error
    (it writes nothing). Breaks if the builder writes its inputs."""
    arrs = [a.copy() for a in _store([swarm.AI_RUN, swarm.AI_COMA])]
    for a in arrs:
        a.setflags(write=False)
    sd.pack_dots(*arrs, PPT)


def test_nothing_on_the_sim_side_reads_the_dot_view():
    """Property: no module under src/ (the sim, the match paths) names the
    dot view or its [debug] switch, so a run with the view on and one with it
    off are the same run. Breaks if a sim module starts reading it."""
    bad = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "swarm_dots" in text:
            bad.append(path.name)
    assert not bad, bad


def test_digests_equal_with_the_view_on_and_off(monkeypatch):
    """Property: a swarm run whose every tick is packed for the dot view (on)
    and the same run never packed (off) have equal tick digests every tick.
    Breaks if building the view writes sim state."""
    bp = pytest.importorskip("breach_physics")
    import field_ab_harness as fab
    import level_loader
    from field_digest import tick_digest
    from simulation.simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    lv = level_loader.load("larva_studio", str(ROOT / "levels"))

    def run(on):
        sim = Simulation(lv, seed=3, breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        flecks, out = sd.DeathFlecks(), []
        for t in range(40):
            sim.set_paused(False)
            sim.step()
            if on:
                g = sim.gmap
                col = lambda c: swarm.swarm_column(g, "larva", c)  # noqa: E731
                sd.pack_dots(col("pos_x"), col("pos_y"), col("heading"),
                             col("ai_state"), col("valid"), PPT)
                flecks.pack(t / 24.0, PPT)
            out.append(tick_digest(fab.capture_snapshot(sim)))
        return out
    assert run(True) == run(False)
