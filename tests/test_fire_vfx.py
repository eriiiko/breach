"""Property tests of the flame VFX substrate's pure side (graphics arc G3,
patch F1; docs/g3_fire_design_2026-10-23.md §5, §8, §13.3, §13.10, §13.13,
§13.14). Pyray-free, no GL context, no breach_physics.

Each test names the PROPERTY it protects and the change that must break it.
Nothing pins an image, a count of things designed to grow, or an order. The
GL side (blend/flip traps, upload orientation, the LUT upload, per-stage
parity with the twin, the region clear, zero passes, one tamed wind, digests
VFX on/off) is ``tests/gl_fire_vfx_check.py``, run by
``tests/test_fire_vfx_gl.py``.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_fire_vfx.py -q
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from renderer import fire_vfx as fx  # noqa: E402
from renderer import fire_vfx_ref as ref  # noqa: E402

SRC = ROOT / "renderer" / "fire_vfx.py"


@pytest.fixture(scope="module")
def cfg():
    from config import CFG
    return CFG


@pytest.fixture(scope="module")
def settings(cfg):
    return fx.settings_from_config(cfg)


@pytest.fixture(scope="module")
def scales():
    return ref.load_scales()


# ---------------------------------------------------------------------------
# The clock (§13.3)
# ---------------------------------------------------------------------------

def _ticks_per_second(cfg) -> float:
    return float(cfg.clock.ticks_per_second)


def test_substeps_per_tick_equal_in_shot_mode_and_at_60fps(cfg, settings):
    """Property: the flame advances the same number of substeps per sim tick
    whether frames step one tick each (look shots, alpha = 0) or the game
    renders at 60 fps with the accumulator's alpha -- so shots show the
    game's flame speed. Breaks if the per-frame cap drops below the
    substeps one tick owes (5 at 24 tps and dt 1/120) or alpha stops
    reaching the clock."""
    tps = _ticks_per_second(cfg)
    dt = settings.dials.dt_s
    per_tick = 1.0 / (tps * dt)
    shot = fx.SimClock(dt)
    shot.advance(0, 0.0, tps)
    shot_counts = [shot.advance(k, 0.0, tps) for k in range(1, 200)]
    assert all(n == round(per_tick) for n in shot_counts), set(shot_counts)
    assert shot.dropped == 0

    game = fx.SimClock(dt)
    accum, tick, total = 0.0, 0, 0
    spt = 1.0 / tps
    game.advance(0, 0.0, tps)
    frames = 60 * 20
    for _ in range(frames):
        accum += 1.0 / 60.0
        while accum >= spt:
            accum -= spt
            tick += 1
        total += game.advance(tick, accum / spt, tps)
    assert game.dropped == 0
    assert abs(total - tick * per_tick) <= 1.0, (total, tick * per_tick)


def test_alpha_is_clamped_and_t_vfx_never_steps_back(settings):
    """Property: alpha is clamped to [0, 1) and t_vfx is monotone under any
    alpha (negative, >= 1, NaN, None) and any tick sequence, including a sim
    tick that jumps BACK (a new sim) -- after which the clock keeps moving
    instead of stalling. Breaks if the clamp or the re-anchor is removed."""
    for a in (-3.0, -0.0, 0.0, 0.5, 0.999999, 1.0, 7.5, float("nan"), None):
        c = fx.clamp_alpha(a)
        assert 0.0 <= c < 1.0
    rng = np.random.default_rng(3)
    clk = fx.SimClock(settings.dials.dt_s)
    tick, prev = 100, -1.0
    for i in range(2000):
        tick += int(rng.integers(0, 3))
        if i == 1000:
            tick = 5                                   # a new sim
        clk.advance(tick, float(rng.uniform(-0.5, 1.5)), 24.0)
        assert clk.t_vfx >= prev
        prev = clk.t_vfx
    t_at_jump = prev
    for k in range(1, 30):
        clk.advance(tick + k, 0.0, 24.0)
    assert clk.t_vfx > t_at_jump, "the clock stalled after a backward tick"


def test_idle_when_paused_runs_on_its_own_wall_counter(settings):
    """Property: while paused, idle substeps run on wall time into t_idle and
    t_vfx does not move; with idle off a paused frame runs nothing. Breaks if
    idle time leaks into t_vfx (shots / replays would depend on wall time) or
    the idle switch is ignored."""
    dt = settings.dials.dt_s
    clk = fx.SimClock(dt)
    clk.advance(50, 0.0, 24.0)
    t_vfx = clk.t_vfx
    n = sum(clk.advance(50, 0.0, 24.0, paused=True, idle_when_paused=True,
                        wall_dt=1.0 / 60.0) for _ in range(60))
    assert clk.t_vfx == t_vfx
    assert clk.t_idle == pytest.approx(1.0)
    assert abs(n - 1.0 / dt) <= 1
    off = fx.SimClock(dt)
    off.advance(50, 0.0, 24.0)
    assert sum(off.advance(50, 0.0, 24.0, paused=True, idle_when_paused=False,
                           wall_dt=1.0 / 60.0) for _ in range(60)) == 0


# ---------------------------------------------------------------------------
# The active region (§5, §13.10)
# ---------------------------------------------------------------------------

def _disjoint(boxes):
    return all(not fx._overlap(a, b) for i, a in enumerate(boxes)
               for b in boxes[i + 1:])


def test_boxes_cover_every_burning_tile_and_margin_disjointly():
    """Property: on random tile maps the active boxes are at most MAX_BOXES,
    pairwise disjoint, inside the map, and cover every live tile dilated by
    the margin. Breaks if a cluster is dropped, boxes overlap (double work)
    or the box count grows without bound."""
    rng = np.random.default_rng(11)
    for trial in range(200):
        h, w = int(rng.integers(8, 60)), int(rng.integers(8, 60))
        live = rng.random((h, w)) < rng.choice([0.002, 0.01, 0.05])
        r = int(rng.integers(0, 5))
        boxes = fx.cut_boxes(live, r)
        assert len(boxes) <= fx.MAX_BOXES
        assert _disjoint(boxes)
        for x0, y0, x1, y1 in boxes:
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        cover = fx.boxes_mask(boxes, (h, w))
        ys, xs = np.nonzero(live)
        for y, x in zip(ys, xs):
            assert cover[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].all()


def test_no_fire_plans_no_passes_and_asks_no_wind():
    """Property: with nothing burning (and nothing lingering) the frame plan
    is empty, whatever the substep count. Breaks if the active-region gate
    is lost and an idle map pays for passes."""
    reg = fx.ActiveRegion((20, 30))
    left = reg.update(np.zeros((20, 30), bool), 1.0, 3, 2.0)
    assert not left.any() and not reg.active
    assert fx.plan_frame(reg.boxes, fx.mask_rects(left), [0.0] * 6) == []


def test_a_tile_leaving_the_region_is_cleared_once():
    """Property: when a fire goes out (past its linger), the plan clears
    exactly the tiles that left the domain -- in both fields -- once, and
    the next frame clears nothing. Breaks if departed tiles are not cleared
    (a stale flame reappears when the region returns) or are cleared every
    frame."""
    shape = (30, 40)
    reg = fx.ActiveRegion(shape)
    a = np.zeros(shape, bool)
    a[5, 5] = True
    b = np.zeros(shape, bool)
    b[25, 35] = True
    reg.update(a | b, 0.0, 3, 1.0)
    dom_before = reg.domain.copy()
    left = reg.update(b, 0.5, 3, 1.0)          # A still lingers
    assert not left.any()
    left = reg.update(b, 2.0, 3, 1.0)          # A past its linger
    assert left.any()
    assert np.array_equal(left, dom_before & ~reg.domain)
    rects = fx.mask_rects(left)
    assert np.array_equal(fx.boxes_mask(rects, shape), left)
    plan = fx.plan_frame(reg.boxes, rects, [0.0])
    cleared = {p.field for p in plan if p.kind == "clear"}
    assert cleared == {"vel", "flame"}
    assert not reg.update(b, 2.1, 3, 1.0).any()


def test_pass_count_is_independent_of_box_count():
    """Property: every stage is ONE pass per substep however many boxes the
    region has (all boxes are quads inside one texture mode, §13.10). Breaks
    if the plan issues a pass per box."""
    t = [0.0, 0.01]
    one = fx.plan_frame([(0, 0, 4, 4)], [], t)
    four = fx.plan_frame([(0, 0, 4, 4), (10, 0, 14, 4), (0, 10, 4, 14),
                          (10, 10, 14, 14)], [], t)
    assert len(one) == len(four) == len(t) * len(fx.STAGES)


def test_linger_outlasts_the_flame_decay(settings):
    """Property: a tile stays active at least as long as a full-intensity
    flame needs to decay below r_cut, so a fire that goes out fades instead
    of being cut by its region vanishing. Breaks if linger_s stops tracking
    lifetime_s / s0 / r_cut."""
    d = settings.dials
    r_peak = d.s0 * d.lifetime_s * (1.0 + d.puff_amp)
    assert fx.linger_s(d) >= d.lifetime_s * math.log(r_peak / d.r_cut) - 1e-9


# ---------------------------------------------------------------------------
# The blackbody GPU LUT (§3.5, §13.14)
# ---------------------------------------------------------------------------

def test_lut_data_is_the_one_ramp_at_its_samples(cfg):
    """Property: the GPU LUT's data equals BlackbodyRamp.emission_at_kelvin
    at the LUT's sample points over the ramp's own Kelvin domain (both ends
    included), with LUT_SIZE entries. Breaks if a second T -> colour map
    appears or the domain / size drifts from the ramp's."""
    from renderer.blackbody import BlackbodyRamp
    ramp = BlackbodyRamp.from_config(cfg)
    k = fx.blackbody_lut_kelvins(ramp)
    assert k[0] == ramp.kelvin_floor and k[-1] == ramp.kelvin_ceil
    data = fx.blackbody_lut_data(ramp)
    assert data.shape == (1, fx.LUT_SIZE, 4)
    chroma, inten = ramp.emission_at_kelvin(k)
    assert np.allclose(data[0, :, :3], chroma, rtol=1e-6, atol=1e-7)
    assert np.allclose(data[0, :, 3], inten, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Sources per tile (§13.14) and render-only (§8)
# ---------------------------------------------------------------------------

def test_tile_pack_is_the_twins_source_law_per_tile(scales, settings):
    """Property: the per-tile textures carry exactly the twin's
    ``flame_sources`` (s, q, T_src, cover, puffing) and the same tamed-wind
    scale (tamed * tps) -- nearest per tile, no CPU upsampling (the arrays
    are (h, w)). Breaks if fire_vfx re-implements or rescales the source
    law."""
    rng = np.random.default_rng(5)
    h, w = 14, 18
    fire = np.where(rng.random((h, w)) < 0.15, rng.random((h, w)), 0.0)
    kel = rng.uniform(600.0, 1400.0, (h, w))
    tg = np.array([[scales.tscale.from_kelvin(v) for v in row] for row in kel])
    solid = rng.random((h, w)) < 0.2
    flam = rng.random((h, w)) < 0.6
    vac = rng.random((h, w)) < 0.05
    wind = rng.normal(0, 0.3, (h, w, 2)).astype(np.float32)
    d = settings.dials
    pack = fx.pack_tile_inputs(fire, tg, solid, flam, vac, wind, tps=24.0,
                               tile_size_m=0.333, scales=scales, dials=d)
    src = ref.flame_sources(fire, scales.to_kelvin(tg), solid, flam, vac,
                            t_amb_k=scales.t_amb_k,
                            glow_min_k=scales.glow_min_k,
                            tile_size_m=0.333, dials=d)
    for k, arr in enumerate((src.s, src.q, src.t_src_k, src.cover)):
        assert np.allclose(pack.src_a[..., k], arr, rtol=1e-6)
    assert np.allclose(pack.src_b[..., 0], src.puff_hz, rtol=1e-6)
    assert np.allclose(pack.src_b[..., 1], src.puff_phase, rtol=1e-6)
    assert np.allclose(pack.wind[..., :2], wind * 24.0, rtol=1e-6)
    assert np.allclose(pack.wind[..., 2],
                       ref.local_intensity(src.i_eff, d.i_local_sigma_tiles))
    assert pack.src_a.shape[:2] == (h, w)


def test_fire_vfx_takes_read_only_views_and_writes_no_sim_state(scales,
                                                                 settings):
    """Property: the VFX's CPU side runs on writeable=False views (any write
    into a sim array raises), and its source never imports the sim package
    nor assigns into a gmap / sim / unit object. Breaks if a VFX path writes
    sim state or grows a sim import."""
    h, w = 8, 8
    ro = lambda a: fx.read_only(np.asarray(a))  # noqa: E731
    fire = np.zeros((h, w))
    fire[3, 3] = 1.0
    tg = np.full((h, w), scales.tscale.from_kelvin(1200.0))
    z = np.zeros((h, w), bool)
    fx.pack_tile_inputs(ro(fire), ro(tg), ro(z), ro(~z), ro(z),
                        ro(np.zeros((h, w, 2), np.float32)), tps=24.0,
                        tile_size_m=0.333, scales=scales,
                        dials=settings.dials)
    view = fx.read_only(fire)
    with pytest.raises(ValueError):
        view[0, 0] = 1.0
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            assert not any(n.split(".")[0] in ("simulation", "breach_physics")
                           for n in names), f"sim import: {names}"
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if not isinstance(t, (ast.Attribute, ast.Subscript)):
                    continue                  # a local name, not a write INTO
                base = t
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    assert base.id not in ("gmap", "sim", "unit", "units"), \
                        f"line {node.lineno} assigns into {base.id}"
    assert "writeable = True" not in SRC.read_text(encoding="utf-8")


def test_the_renderer_tames_the_wind_at_one_call_site():
    """Property: GameRenderer has exactly one ``tame_wind`` call, inside the
    per-frame memo every consumer (relief, props, flame) goes through -- so
    at most one tamed-wind product per frame, with the relief's dials
    (§13.13). The GL check counts the calls on the running renderer. Breaks
    if a consumer grows its own tame_wind call."""
    src = (ROOT / "renderer" / "game_renderer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "tame_wind"]
    assert len(calls) == 1, [c.lineno for c in calls]
    memo = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            and n.name == "frame_tamed_wind"]
    assert memo and any(c in list(ast.walk(memo[0])) for c in calls)


# ---------------------------------------------------------------------------
# Config: [render.fire_vfx] is the single source
# ---------------------------------------------------------------------------

def test_config_names_every_dial_and_nothing_else(cfg):
    """Property: the shipped [render.fire_vfx] carries every dial of
    FireVFXDials and every setting of FireVFXSettings, and no unknown key --
    so the dataclass fallbacks are never silently live and a typo'd key is
    caught. Breaks if a dial is added to the twin without a config row (or a
    config row names nothing)."""
    sec = cfg.render.fire_vfx
    have = set(vars(sec))
    want = set(fx.config_keys())
    assert want - have == set(), f"missing in config: {sorted(want - have)}"
    assert have - want == set(), f"unknown config keys: {sorted(have - want)}"


def test_settings_from_config_carry_the_config_values(cfg):
    """Property: every dial the pass uses is the config's value with the
    schema's type (int stays int, bool bool). Breaks if settings_from_config
    reads a key from somewhere else or drops a type."""
    s = fx.settings_from_config(cfg)
    sec = cfg.render.fire_vfx
    for k in fx.config_keys():
        v = getattr(s.dials, k) if hasattr(s.dials, k) else getattr(s, k)
        assert v == getattr(sec, k), k
        assert type(v) is type(getattr(ref.FireVFXDials(), k, None)
                               if hasattr(s.dials, k)
                               else getattr(fx.FireVFXSettings(), k)), k


def test_every_stage_shader_composes_with_the_common_glsl():
    """Property: every pass shader carries the shared GLSL (texel convention,
    cover rule, puffing) through the one include, so no stage re-derives
    them. Breaks if a stage drops the include or inlines its own copy."""
    common = (ROOT / "shaders" / "fire_vfx_common.glsl").read_text(encoding="utf-8")
    for st in fx.STAGES + ("debug",):
        raw = (ROOT / "shaders" / f"fire_vfx_{st}.fs").read_text(encoding="utf-8")
        assert "bool source_covers" not in raw and "float puff_factor" not in raw
        assert common in fx.compose_shader_source(st)
