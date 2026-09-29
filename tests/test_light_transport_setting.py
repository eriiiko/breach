"""P6d -- the light channels' TRANSPORT as a permanent [light] config setting
(ray-engine-v2; docs/ray_engine_v2_p6d_light_transport_setting_brief_2026-
09-29.md, gates G3, G4, G5). Gate G2 (CPU/CUDA parity on the live conductor)
is tests/test_cuda_p6d_light_transport.py; gate G1 is the unmoved suite +
golden.

The sweep itself already carries both transports on light (gate 18, tests/
test_radiation_sweep_light.py) -- P6d's own surface is the ONE place that used
to hardcode the choice (cpp/src/physics_engine.cpp's run_sweep_) and the
config door that now feeds it (src/simulation/physics_runner.py). These tests
hold THAT surface: the door refuses an unknown value (G5), the live conductor
proves the setting reaches the sweep without moving a single heat integer
(G3) while the light field it produces genuinely differs between the two
values (G4, non-vacuous), and a corridor scene reproduces the scheme study's
own finding about the direction that difference runs (G4, the ordering).

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_light_transport_setting.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                    # noqa: E402
from config import CFG                                         # noqa: E402
from level_loader import LevelData                              # noqa: E402
from simulation.physics_runner import PhysicsRunner             # noqa: E402


# ---------------------------------------------------------------------------
# G5 -- the config door
# ---------------------------------------------------------------------------
def test_the_default_config_binds_step():
    """PROPERTY (brief decision 1): the shipped config.toml's [light] transport
    = "step" binds onto PhysicsEngine.light_transport as RadiationSweep.STEP --
    an engine nobody configures behaves exactly as it did before P6d.
    BREAKS IF: the default in config.toml or the binding's fallback changes to
    something other than "step", or the binding reads the wrong config key."""
    pr = PhysicsRunner(bp)
    assert pr.engine.light_transport == bp.RadiationSweep.STEP


def test_shear_binds_when_configured(monkeypatch):
    """PROPERTY: [light] transport = "shear" binds onto PhysicsEngine.
    light_transport as RadiationSweep.SHEAR -- trying shear is a config edit,
    nothing more (brief decision 2, no key).
    BREAKS IF: the door only ever binds STEP, or "shear" is spelled/matched
    case-sensitively against something other than the literal string."""
    monkeypatch.setattr(CFG.light, "transport", "shear")
    pr = PhysicsRunner(bp)
    assert pr.engine.light_transport == bp.RadiationSweep.SHEAR


@pytest.mark.parametrize("bad", ["diagonal", "SHEAR", "Step", "", "none", "shear "])
def test_an_unknown_transport_is_refused_at_load(monkeypatch, bad):
    """PROPERTY: any [light] transport value other than exactly "step" or
    "shear" fails LOUDLY at the door instead of silently falling back to a
    default or being coerced -- case variants and stray whitespace included.
    BREAKS IF: PhysicsRunner's validation of this key is removed, or the
    comparison is made case-insensitive or trimmed."""
    monkeypatch.setattr(CFG.light, "transport", bad)
    with pytest.raises(ValueError, match="is not a light transport"):
        PhysicsRunner(bp)


# ---------------------------------------------------------------------------
# scenario helpers (the live engine, stepped the way the game steps it)
# ---------------------------------------------------------------------------
def _burning_smoky_playground(*, transport, seed=5):
    """The playground with a wood tile heated to the 1263-game plateau and lit
    (CLAUDE.md "Starting a fire": heat delivered, not `fire` alone) plus a
    smoke puff around it (both the heat and light smoke arms ride the shipped
    [gases.smoke] columns), light REQUESTED through the Simulation facade, at
    the given transport."""
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed, gas_fixed
    from simulation.gases import SMOKE
    from simulation.materials import MAT_WOOD
    sim = Simulation(load_level("playground"), seed=seed, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    pick = None
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1 \
                and (not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]):
            pick = (int(y), int(x))
            break
    assert pick is not None, "the playground carries no wood tile beside air"
    g.temperature[pick] = 1263 << 16
    g.fire[pick] = fire_fixed.quantize_scalar(0.8)
    y0, x0 = pick
    air = (~g.thermal_solid) & (~g.solid) & (~g.is_vacuum)
    yy, xx = np.mgrid[0:g.material.shape[0], 0:g.material.shape[1]]
    cloud = air & (np.abs(yy - y0) <= 4) & (np.abs(xx - x0) <= 4)
    dens = gas_fixed.quantize(np.where((yy + xx) % 3 == 0, 0.27, 0.06))
    g.gas[SMOKE][cloud] = dens[cloud]
    sim.set_light(True)
    sim.physics_runner.engine.light_transport = transport
    return sim, pick


def _corridor_scene():
    """The scheme study's own ship-deck scene, geometry reproduced verbatim
    (docs/ray_engine_v2_scheme_study_2026-09-13/room_render.py::deck -- the
    exact scene report.md's "far-room mean" table was measured on): a 64x64
    hull-walled grid, room A and room B joined by a corridor entered and left
    through two ROW-OFFSET single-tile doorways (so light must travel down
    the corridor's length, not just through one aperture), plus a partition
    with its own gap inside room A. A lone hot wood pillar stands in for the
    study's point source at the same cell, (40, 12)."""
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD
    n = 64
    tm = np.full((n, n), MAT_AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = MAT_HULL
    tm[:, 0] = tm[:, -1] = MAT_HULL
    tm[:, 26] = MAT_HULL
    tm[28:34, 26] = MAT_AIR               # doorway 1: room A | corridor
    tm[:, 40] = MAT_HULL
    tm[14:20, 40] = MAT_AIR               # doorway 2: corridor | room B, OFFSET
    tm[20, 1:26] = MAT_HULL
    tm[20, 6:11] = MAT_AIR                # a gap in room A's own partition
    src_y, src_x = 40, 12
    tm[src_y, src_x] = MAT_WOOD           # the fire, standing in room A
    ld = LevelData(name="p6d_corridor", version="2", path=Path("."),
                  tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(ld, seed=11, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    assert g.material[src_y, src_x] == MAT_WOOD and g.thermal_solid[src_y, src_x], \
        "the fixture's wood pillar did not come up as a thermal solid"
    g.temperature[src_y, src_x] = 1263 << 16
    g.fire[src_y, src_x] = fire_fixed.quantize_scalar(0.8)
    return sim, (src_y, src_x)


# ---------------------------------------------------------------------------
# G3 + G4 (non-vacuity half) -- the live conductor, N ticks
# ---------------------------------------------------------------------------
def test_heat_is_bit_identical_and_light_differs_across_transport():
    """PROPERTY (brief G3 + G4): on the live conductor (the playground, a
    burning wood tile, a smoke puff, light requested), swapping [light]
    transport between step and shear moves NOTHING heat -- every SIM_FIELDS
    field (the A/B harness's synced set, a superset of the digest's) plus the
    sweep's four heat planes stay bit-for-bit identical between the two runs,
    tick after tick -- while light_q differs between them at least once (the
    switch is real, not vacuous), and neither run is dark.

    BREAKS IF: PhysicsEngine.light_transport leaks into a heat read (run_sweep_
    reusing LightChannels past the light branch, or PhysicsRunner binding it
    onto a heat-reading member), or the setting fails to reach the sweep (both
    runs would then produce identical light_q).
    """
    s_step, pick = _burning_smoky_playground(transport=bp.RadiationSweep.STEP)
    s_shear, _ = _burning_smoky_playground(transport=bp.RadiationSweep.SHEAR)
    from field_ab_harness import SIM_FIELDS
    heat_fields = tuple(SIM_FIELDS) + ("rad_net_sweep", "rad_flux_sweep",
                                       "rad_amb_sweep", "rad_fluence")
    any_light_diff = False
    for t in range(8):
        for s in (s_step, s_shear):
            s.set_paused(False)
            s.step()
        for k in heat_fields:
            a, b = getattr(s_step.gmap, k), getattr(s_shear.gmap, k)
            eq = (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                  else np.array_equal(a, b))
            assert eq, (k, t)
        if not np.array_equal(s_step.gmap.light_q, s_shear.gmap.light_q):
            any_light_diff = True
    assert any_light_diff, "step and shear produced identical light_q throughout -- vacuous"
    assert np.any(s_step.gmap.light_q != 0), "the step run is dark"
    assert np.any(s_shear.gmap.light_q != 0), "the shear run is dark"
    assert pick is not None


# ---------------------------------------------------------------------------
# G4 (the ordering) -- the corridor
# ---------------------------------------------------------------------------
def test_shear_light_carries_more_through_a_corridor_than_step():
    """PROPERTY (brief G4, the scheme study's corridor-throughput finding --
    report.md "What the heat schemes look like": shear keeps ~0.90 of the far-
    room mean light against step's ~0.52 of exact): on the scheme study's own
    corridor scene, the far room's MEAN light_q (report.md's own metric,
    room_render.py's `far = (~opaque) & (columns > 40)`) is strictly greater
    under shear than under step, and both are non-zero (the far room really is
    reached either way) -- one relight, no ticking needed, since this is a
    statement about the sweep's transport, not accumulation.

    BREAKS IF: [light] transport stops reaching relight()'s sweep call, or the
    two transports stop differing in the direction the scheme study measured
    (a step/shear swap in the ordinate tables, or the setting silently
    inverted).
    """
    def far_room_mean(transport):
        sim, pick = _corridor_scene()
        sim.set_light(True)
        sim.physics_runner.engine.light_transport = transport
        sim.relight()
        g = sim.gmap
        h, w = g.solid.shape
        far_mask = (~g.solid) & (np.arange(w)[None, :] > 40)
        far = g.light_q[far_mask]
        assert far.size > 0 and np.any(far > 0), \
            "the far room is entirely dark -- the corridor let nothing through"
        return float(far.astype(np.float64).mean())

    step_mean = far_room_mean(bp.RadiationSweep.STEP)
    shear_mean = far_room_mean(bp.RadiationSweep.SHEAR)
    assert shear_mean > step_mean > 0, (step_mean, shear_mean)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
