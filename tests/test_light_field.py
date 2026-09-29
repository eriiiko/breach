"""THE LIGHT ACCESSOR and the game's light on the conductor (ray-engine-v2 P6b).

docs/ray_engine_v2_p6b_lit_world_brief_2026-09-25.md sections 1.1, 1.4-1.5 and
3.1, 3.3, 3.5; design v3 sections 4.3, 7.1, 8.5. The renderer, the rules and RL
read light only through src/simulation/light_field.py; the game requests light
(Simulation.set_light), hands the sweep its cone rows and sky, and relights a
paused frame (Simulation.relight). These tests hold the accessor and that path
to the brief's properties, on real scenes, headlessly (the pack is pure numpy:
renderer.lighting.pack_light_view -- no GL context is opened here).

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_light_field.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from config import CFG  # noqa: E402
from level_loader import LevelData, load as load_level  # noqa: E402
from simulation import Simulation, fire_fixed, light_field  # noqa: E402
from simulation.unit import Unit  # noqa: E402

TURN = 65536
LU = 1 << int(bp.L_FINE_BITS)                 # counts per light unit


def _pack(view, gain=1.0):
    """The renderer's pack (pure numpy) into fresh buffers."""
    from renderer.lighting import pack_light_view
    h, w, _ = view.rgb.shape
    out = dict(rgb=np.zeros((h, w, 3), np.float32), dx=np.zeros((h, w), np.float32),
               dy=np.zeros((h, w), np.float32), lmap=np.zeros((h, w), np.float32),
               a=np.zeros((h, w, 4), np.float16), b=np.zeros((h, w, 4), np.float16),
               glow=np.zeros((h, w, 3), np.float32))
    pack_light_view(view, gain, out["rgb"], out["dx"], out["dy"], out["lmap"],
                    out["a"], out["b"], glow_rgb=out["glow"])
    return out


def _lit_playground(ticks=6):
    """The playground, REALLY lit the way main.py lights it: its own [[light]]
    lamps through the one assembly, a flashlight-like cone and a lamp on a
    marine-free floor, a burning wood tile (heat delivered, CLAUDE.md "Starting a
    fire"), a smoke cloud, and an overcast sky -- light requested, ticked."""
    from level_lights import partition_lights
    from renderer import frame_lights as fl
    from simulation.materials import MAT_WOOD
    level = load_level("playground")
    sim = Simulation(level, seed=5, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    h, w = g.solid.shape
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    y0, x0 = int(ys[0]), int(xs[0])
    g.temperature[y0, x0] = 1263 << 16
    g.fire[y0, x0] = fire_fixed.quantize_scalar(0.8)
    air = np.argwhere(~g.solid & ~g.is_vacuum)
    cy, cx = (int(v) for v in air[len(air) // 2])
    g.smoke[max(0, cy - 3):cy + 3, max(0, cx - 3):cx + 3] = fire_fixed.quantize_scalar(0.4)
    lights_in, _off = partition_lights(level.lights, w, h)
    extra = [fl.LightSpec(x=cx + 0.5, y=cy + 0.5, color=(1.0, 0.9, 0.8), intensity=2.0,
                          angle_center=0.7, angle_spread=1.9, kind="flashlight",
                          source="render")]
    specs = fl.frame_light_specs(lights_in, total_tick=0, sim_time_per_tick=1 / 24.0,
                                 extra=extra).specs
    rows = fl.cone_rows(specs, w, h, int(bp.L_FINE_BITS))
    sky = np.full((16, 3), 3 << 28, dtype=np.int64)
    sim.set_light(True, rows, sky)
    for _ in range(ticks):
        sim.set_paused(False)
        sim.step()
    return sim


# ---------------------------------------------------------------------------
# 1. §8.5 -- THE EYE IS MONOTONE IN THE RULES FIELD
# ---------------------------------------------------------------------------
def test_the_eye_is_a_monotone_function_of_the_rules_field():
    """PROPERTY (brief 3.1, design 8.5): on a real scene (the playground lit by
    its own lamps, a cone, a fire, smoke, a sky), the render's field -- the
    accessor's view packed into the 16F textures exactly as LightingPass packs
    them -- is a MONOTONE non-decreasing function of light_q, per channel: order
    the cells by light_q[c] and the packed value never falls. The same holds for
    the scalar (light_map, against the brightest channel) and the glow (against
    light_glow). Non-vacuous: hundreds of distinct light_q levels.

    BREAKS IF: any render-side post-process that reorders brightness lands in
    the accessor or the pack -- a blur, a neighbour mix, a per-cell noise, a
    channel cross-talk (validated: a 3x3 box blur in pack_light_view turns it
    red).
    """
    sim = _lit_playground()
    g = sim.gmap
    view = light_field.read_light(g)
    out = _pack(view, gain=float(CFG.render.lighting.sweep_gain))
    n_levels = 0
    for c in range(3):
        q = g.light_q[..., c].ravel()
        order = np.argsort(q, kind="stable")
        packed = out["a"][..., c].astype(np.float64).ravel()[order]
        assert np.all(np.diff(packed) >= 0.0), f"channel {c}: the eye reorders brightness"
        n_levels = max(n_levels, len(np.unique(q)))
    qmax = g.light_q.max(axis=2).ravel()
    lm = out["lmap"].astype(np.float64).ravel()[np.argsort(qmax, kind="stable")]
    assert np.all(np.diff(lm) >= 0.0)
    for c in range(3):
        qg = g.light_glow[..., c].ravel()
        pg = out["b"][..., c].astype(np.float64).ravel()[np.argsort(qg, kind="stable")]
        assert np.all(np.diff(pg) >= 0.0)
    assert n_levels > 200, f"only {n_levels} distinct light levels -- vacuous"
    assert np.any(g.light_glow > 0), "no glow on the scene -- the glow check is vacuous"
    y, x = np.unravel_index(int(np.argmax(g.light_q[..., 0])), g.light_q.shape[:2])
    assert light_field.light_at(g, y, x) == tuple(int(v) for v in g.light_q[y, x])


def test_the_flux_direction_is_unit_length_or_zero():
    """PROPERTY (design 4.3: "the flux vector is normalised at the render read,
    in light_field.py"): read_light's flux_dir is a unit vector wherever the
    sweep's net flux is non-zero and exactly (0, 0) where it is zero, and it
    points the way the integer flux points (same signs).

    BREAKS IF: the normalisation is dropped or moved outside the accessor, the
    zero-flux case divides by zero, or a sign is flipped in the read (the pack
    negates it for the texture contract, never the accessor).
    """
    sim = _lit_playground(ticks=3)
    g = sim.gmap
    view = light_field.read_light(g)
    f = g.light_flux_q
    nz = (f[..., 0] != 0) | (f[..., 1] != 0)
    norms = np.hypot(view.flux_dir[..., 0], view.flux_dir[..., 1])
    assert np.all(np.abs(norms[nz] - 1.0) < 1e-5)
    assert np.all(view.flux_dir[~nz] == 0.0)
    assert np.all(np.sign(view.flux_dir[..., 0][nz]) == np.sign(f[..., 0][nz]))
    assert np.count_nonzero(nz) > 100


# ---------------------------------------------------------------------------
# 2. RELIGHT: the paused path writes the light planes and nothing else
# ---------------------------------------------------------------------------
def test_relight_writes_only_the_light_planes():
    """PROPERTY (physics_engine.h relight; brief 3.5): Simulation.relight --
    the paused renderer's path -- recomputes light_q / light_flux_q /
    light_glow on the current state and leaves EVERY other GameMap array (heat
    planes, temperature, gas_energy, the sweep's four planes included) exactly
    as it found them; the light it computes equals what a tick's light
    computation gives the same state and inputs (one sweep invocation).

    BREAKS IF: relight hands a caller's heat plane to the sweep as its scratch,
    writes the temperature or the energy field, or runs a different light law
    than step 2b.
    """
    sim = _lit_playground(ticks=4)
    g = sim.gmap
    before = {k: v.copy() for k, v in vars(g).items() if isinstance(v, np.ndarray)}
    serial = sim.light_serial
    for k in ("light_q", "light_flux_q", "light_glow"):
        getattr(g, k)[...] = -7
    sim.relight()
    assert sim.light_serial == serial + 1
    for k, v in before.items():
        if k in ("light_q", "light_flux_q", "light_glow"):
            continue
        assert np.array_equal(v, getattr(g, k), equal_nan=v.dtype.kind == "f"), k
    assert not np.any(g.light_q == -7) and np.any(g.light_q > 0)
    first = g.light_q.copy()
    sim.relight()
    assert np.array_equal(first, g.light_q)          # a relight is a pure function


# ---------------------------------------------------------------------------
# 3. HEAT AND THE GOLDEN ARE UNTOUCHED BY THE GAME'S LIGHT
# ---------------------------------------------------------------------------
def test_heat_and_the_golden_are_untouched_by_the_games_light():
    """PROPERTY (brief 3.5, the P6a gate re-run under the game's own
    conductor): the canonical golden scenario with the GAME's light on -- light
    requested, cone rows (an omni lamp, a narrow beam, a zero-spread cone) and
    an overcast sky handed through Simulation.set_light, and a RELIGHT before
    every tick (the paused path interleaved) -- produces a trajectory whose
    digest IS the committed GOLDEN_AGGREGATE. Non-vacuous: light_q is lit.

    BREAKS IF: requesting light, a cone row, the sky or a relight moves any
    synced integer (a heat plane, temperature, gas, a unit), anywhere in 30
    ticks.
    """
    from field_ab_harness import capture_trajectory, default_scenario_sim
    from field_digest import trajectory_digest
    from _xarch_perfield_digest import GOLDEN_AGGREGATE, N_STEPS
    lit = []

    def make_sim():
        sim = default_scenario_sim()
        rows = np.asarray([[5, 5, 1 << 37, 1 << 37, 1 << 36, 0, TURN],
                           [10, 3, 3 << 36, 3 << 36, 3 << 36, TURN // 8, TURN // 6],
                           [12, 12, 1 << 36, 0, 1 << 36, 40000, 0]], dtype=np.int64)
        sim.set_light(True, rows, np.full((16, 3), 1 << 30, dtype=np.int64))
        step = sim.step

        def step_with_relight():
            sim.relight()
            step()
            lit.append(bool(np.any(sim.gmap.light_q > 0)))
        sim.step = step_with_relight
        return sim

    traj = capture_trajectory(make_sim=make_sim, n_steps=N_STEPS)
    assert trajectory_digest(traj) == GOLDEN_AGGREGATE
    assert all(lit) and len(lit) == N_STEPS


# ---------------------------------------------------------------------------
# 4. THE SKY
# ---------------------------------------------------------------------------
def _open_level(boundary="ambient", n=24):
    """An all-air n x n level (no wall anywhere): every cell is a clear
    boundary-reachable cell."""
    tm = np.full((n, n), 4, dtype=np.int32)
    lv = LevelData(name="p6b_open", version="1", path=Path("."), tilemap=tm,
                   tile_size_m=1.0, diffuse_path=Path("."))
    lv.boundary = boundary
    return lv


@pytest.mark.parametrize("flat", [(0.10, 0.10, 0.13), (0.18, 0.18, 0.22), (0.02, 0.5, 0.9)])
def test_a_uniform_sky_equals_the_old_flat_ambient_at_the_boundary(flat):
    """PROPERTY (brief 3.3, first bullet): a UNIFORM sky derived from a flat
    ambient A (the old u_ambient: main.py's (0.10, 0.10, 0.13), LightingPass's
    (0.18, 0.18, 0.22), a coloured one) through the sky's door
    (frame_lights.sky_for_level, overcast = A / (sweep_gain * master_gain) --
    light units, so the integers never depend on a render dial), the sweep,
    the accessor and the pack renders at every clear boundary cell as A
    within 0.1 % once the shader's master gain applies (the floor excluded:
    it is a separate term).

    BREAKS IF: the door splits the overcast unevenly over the ordinates or
    rounds it off, the ring reads the sky for one upwind share only, a uniform
    sky stops being a fixed point of the transport, or the dequantize / the
    render gain drifts.
    """
    from types import SimpleNamespace as NS
    from renderer import frame_lights as fl
    lcfg = CFG.render.lighting
    exposure = float(lcfg.sweep_gain) * float(lcfg.master_gain)
    cfg = NS(light=NS(sky=NS(ambient=NS(overcast=[v / exposure for v in flat],
                                        sun=[0.0, 0.0, 0.0]))))
    sky = fl.sky_for_level("ambient", cfg, bp)
    sim = Simulation(_open_level(), seed=1, breach_physics=bp, enable_recorder=False)
    sim.set_light(True, None, sky)
    sim.relight()
    out = _pack(light_field.read_light(sim.gmap), gain=float(lcfg.sweep_gain))
    shown = out["a"][..., :3].astype(np.float64) * float(lcfg.master_gain)
    ring = np.zeros(shown.shape[:2], dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    for c in range(3):
        assert np.all(np.abs(shown[..., c][ring] / flat[c] - 1.0) < 1e-3), c
    # a uniform sky is a fixed point: every clear cell, not just the ring
    assert np.all(sim.gmap.light_q == sim.gmap.light_q[0, 0])


def test_space_is_dark_and_the_shipped_planetside_sky_is_uniform():
    """PROPERTY (brief 1.4, the sky's source): the sky is chosen per BOUNDARY
    TYPE -- a space level gets the dark ring (None: P6a's 0, bit for bit), a
    planetside ("ambient") level the shipped overcast, uniform over the
    ordinates and summing to the configured irradiance in L°'s currency; a
    level type with no table is dark.

    BREAKS IF: space lets light in, the planetside sky stops being uniform
    (a sun leaking in by default), or the door stops reading the level's
    boundary type.
    """
    from renderer import frame_lights as fl
    assert fl.sky_for_level("space", CFG, bp) is None
    assert fl.sky_for_level("no_such_boundary", CFG, bp) is None
    sky = fl.sky_for_level("ambient", CFG, bp)
    over = [float(v) for v in CFG.light.sky.ambient.overcast]
    assert sky is not None and np.all(sky == sky[0])
    for c in range(3):
        assert abs(int(sky[:, c].sum()) - over[c] * LU) <= 16


def test_a_sealed_room_is_black_to_the_eye_but_for_the_floor():
    """PROPERTY (brief 3.3, second bullet): on a REAL sealed level
    (bench_two_room: hull on every grid edge) under a bright sky, with no lamp
    and no fire, light_q is EXACTLY 0 in every interior cell -- so the packed
    light the shader adds is 0 and the eye there sees only the floor dial.
    Non-vacuous PAIR: the same sky on an open level lights every cell.

    BREAKS IF: the sky leaks through an opaque hull (a transport or absorption
    bug), a cold wall emits (L°[0] != 0), or the pack adds a constant term of
    its own (the floor must stay the one flat term).
    """
    sky = np.full((16, 3), 1 << 34, dtype=np.int64)
    sim = Simulation(load_level("bench_two_room"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    sim.set_light(True, None, sky)
    sim.relight()
    interior = ~g.solid & ~g.is_vacuum
    assert interior.sum() > 100
    assert np.all(g.light_q[interior] == 0)
    out = _pack(light_field.read_light(g))
    assert np.all(out["a"][interior][:, :3] == 0)
    open_sim = Simulation(_open_level("space"), seed=1, breach_physics=bp,
                          enable_recorder=False)
    open_sim.set_light(True, None, sky)
    open_sim.relight()
    assert np.all(open_sim.gmap.light_q > 0)


def test_light_is_not_requested_by_default_and_the_request_survives_a_reset():
    """PROPERTY (P6a decision 2, brief 1.5): a Simulation computes no light
    until the game requests it (headless runs, benches and RL never do); once
    requested, the request and its inputs survive reset() (which builds a
    fresh runner), and set_light(False) turns it off again.

    BREAKS IF: light becomes always-on (headless training would pay for it),
    or a reset silently drops the renderer's request (the game would go dark
    after a restart).
    """
    sim = Simulation(_open_level(), seed=1, breach_physics=bp, enable_recorder=False)
    assert sim.physics_runner.engine.light_requested is False
    rows = np.asarray([[3, 3, 1 << 37, 1 << 37, 1 << 37, 0, TURN]], dtype=np.int64)
    sim.set_light(True, rows, None)
    sim.reset(seed=2)
    assert sim.physics_runner.engine.light_requested is True
    assert np.array_equal(sim.physics_runner.light_cones, rows)
    sim.set_paused(False)
    sim.step()
    assert sim.gmap.light_q[3, 3, 0] > 0
    sim.set_light(False)
    assert sim.physics_runner.engine.light_requested is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
