"""#88 (docs/smoke_unit_rescale_design_2026-10-20.md, V7 and V8): what the
finer trace currency BUYS, measured through the real engine.

The trace planes hold 2^(16 + TRACE_FINE_BITS) counts per real unit. Before
#88 a thin smoke was resolved to 1/65536 and the decay (nearest rounding,
floor of one count) and the diffusion (a face flux below 240 counts was dead)
were both coarser than the look and the lethal doses needed. These tests pin
the PROPERTIES, never a count of ticks of a scenario; each names the change
that must break it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402

HULL, AIR = 1, 0
SMOKE, POISON = 5, 6          # re-resolved from the table below


def _box(h, w):
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = HULL
    return tm


def _sim(tm):
    lvl = LevelData(name="trace_res", version="2", path=Path("."), tilemap=tm,
                    tile_size_m=0.333, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _gid(sim, name):
    return list(sim.gmap.gases.names).index(name)


def _ticks_to_reach(sim, gid, level_counts, max_ticks):
    """Steps until the (uniform) plane's interior first falls to <= level."""
    g = sim.gmap
    for k in range(1, max_ticks + 1):
        sim.step()
        if int(g.gas[gid][1:-1, 1:-1].max()) <= level_counts:
            return k
    return None


@pytest.mark.parametrize("name", ["smoke", "poison"])
def test_v7_the_configured_decay_holds_down_to_visible_densities(name):
    """PROPERTY (V7, #88): a uniform trace field decays at its CONFIGURED
    e-fold rate all the way from 1.0 down to 0.03 real units, each level
    reached within 5 % of ``ln(d0/d)/decay`` -- and the same for poison from
    0.2 to its lethal threshold ``poison_min_density``. (Before #88 the
    ceil-rounded linear regime took over at ~0.0037 and a smoke cell took
    80 s to go from black to its last trace instead of the configured law.)

    BREAKS IF: the resolution is lost (k back to 0), the rounding reverts to
    the linear regime (ceil removes whole counts), or the floor of 1 starts
    to dominate above the floor.
    """
    sim = _sim(_box(6, 6))
    g = sim.gmap
    gid = _gid(sim, name)
    decay = float(np.float32(g.gases.decay[gid]))
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    d0 = 1.0 if name == "smoke" else 0.4
    d_end = 0.03 if name == "smoke" else float(CFG.exchange.poison_min_density)
    g.gas[gid][1:-1, 1:-1] = gas_fixed.quantize_trace_scalar(d0)
    elapsed = 0
    for d in (d_end * 4, d_end * 2, d_end):
        want = math.log(d0 / d) / decay / dt
        level = gas_fixed.quantize_trace_scalar(d)
        k = _ticks_to_reach(sim, gid, level, int(want * 1.2) + 50 - elapsed)
        assert k is not None, f"{name} never reached {d}"
        elapsed += k
        got = elapsed          # ticks since the start, not since the last level
        assert abs(got - want) <= 0.05 * want, (name, d, got, want)


def test_v7_a_gas_with_no_decay_loses_nothing():
    """PROPERTY (V7, B6): ``trace_decay::lost(v, 0) == 0`` inside the kit --
    the floor of one count never decays a gas whose decay is 0.

    BREAKS IF: the ``frac_q <= 0`` guard leaves the kit and a caller skips
    differently, or the floor of 1 applies to frac_q == 0.
    """
    sim = _sim(_box(6, 6))
    g = sim.gmap
    gid = _gid(sim, "smoke")
    g.gases.decay[gid] = 0.0
    g.gas[gid][1:-1, 1:-1] = 37          # a few raw counts: where a floor of 1 would bite
    total0 = int(g.gas[gid].astype(np.int64).sum())
    for _ in range(40):
        sim.step()
    assert int(g.gas[gid].astype(np.int64).sum()) == total0
    assert int(sim.physics_runner.engine.eos.trace_decay_sum()[gid]) == 0


def test_v8_diffusion_acts_where_the_old_currency_was_dead():
    """PROPERTY (V8, #88 D-B): in still air a 3x3 patch of smoke at 0.003
    real units -- a face difference of 196 old counts, under the old 240-count
    diffusion edge, so frozen before #88 -- spreads within 10 s (a growing
    second moment), conserves ``sum S`` exactly, and a uniform field of the
    same thin density stays exactly uniform.

    BREAKS IF: the currency is coarsened back (the patch would not move), the
    diffusion loses or mints counts, or a uniform field drifts.
    """
    sim = _sim(_box(15, 15))
    g = sim.gmap
    g.gases.heat_absorb_q16[:] = 0
    g.gases.decay[:] = 0.0
    q = gas_fixed.quantize_trace_scalar(0.003)
    g.gas[_gid(sim, "smoke")][6:9, 6:9] = q
    g.gas[_gid(sim, "poison")][1:-1, 1:-1] = q
    sid = _gid(sim, "smoke")
    yy, xx = np.mgrid[0:15, 0:15]

    def m2(S):
        w = S.astype(np.float64)
        return float((w * ((yy - 7) ** 2 + (xx - 7) ** 2)).sum() / w.sum())

    total0 = int(g.gas[sid].astype(np.int64).sum())
    m2_0 = m2(g.gas[sid])
    for _ in range(10 * int(CFG.clock.ticks_per_second)):
        sim.step()
        assert int(g.gas[sid].astype(np.int64).sum()) == total0
    assert m2(g.gas[sid]) > 1.5 * m2_0, "the thin patch did not spread"
    P = g.gas[_gid(sim, "poison")][1:-1, 1:-1]
    assert int(P.min()) == int(P.max()), "a uniform thin field drifted"
