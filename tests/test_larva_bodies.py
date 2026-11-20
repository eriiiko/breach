"""The swarm larvae's bodies (arc #63, ``renderer/larva_bodies.py``): the
grub mesh, the per-larva instance data, the dead bodies, the door, and that
the view is render-only. The GL face is ``tests/gl_larva_bodies_check.py``
(``tests/test_gl_larva_bodies.py``).

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import larva_bodies as lb  # noqa: E402
from renderer import propgen  # noqa: E402
from simulation import swarm, swarm_fixed  # noqa: E402

ONE = swarm_fixed.FP_ONE
PPT = 48.0
TILE_M = 0.333


def _cfg():
    from config import CFG
    return CFG


def _settings(**over):
    sec = dict(vars(_cfg().render.larvae))
    sec.update(over)
    return lb.larva_settings_from_config(SimpleNamespace(
        render=SimpleNamespace(larvae=SimpleNamespace(**sec))))


def _cols(n, *, valid=None, states=None, walked=None, ids=None):
    """A swarm store of n slots in the roster's dtypes (only READ_COLUMNS)."""
    x = (np.arange(n, dtype=np.int64) % 20 * ONE + ONE // 2).astype(np.int32)
    y = (np.arange(n, dtype=np.int64) // 20 * ONE + ONE // 2).astype(np.int32)
    dt = {c.name: c.dtype for c in swarm.ROSTER}
    return {
        "pos_x": x, "pos_y": y,
        "heading": np.zeros(n, dt["heading"]),
        "ai_state": np.asarray(states if states is not None
                               else [swarm.AI_RUN] * n, dt["ai_state"]),
        "valid": np.asarray(valid if valid is not None else [1] * n,
                            dt["valid"]),
        "dist_walked": np.asarray(walked if walked is not None else [0] * n,
                                  dt["dist_walked"]),
        "unit_id": np.asarray(ids if ids is not None
                              else np.arange(1, n + 1), dt["unit_id"]),
    }


# --- the mesh ---------------------------------------------------------------

@pytest.mark.parametrize("rings", [2, 4, 6, 9])
def test_the_grub_is_a_small_upward_heightfield(rings):
    """Property: ``generate_larva`` gives 6 x rings triangles (36 at the
    shipped 6), every one wound counter-clockwise seen from above (raylib's
    culling keeps it) and none below the floor; its length and width are
    the arguments; the alpha (data) is the along-body coordinate, rising
    tail to head. Breaks if the winding flips, the count grows past the
    budget, or the alpha stops meaning s."""
    v, n, c = propgen.generate_larva(0.1, 0.02, rings, 0.85, 0.35, 1.1)
    assert v.dtype == np.float32 and n.dtype == np.float32 and c.dtype == np.uint8
    assert len(v) == 3 * 6 * rings
    t = v.reshape(-1, 3, 3).astype(np.float64)
    up = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 1]
    assert (up > 0).all()
    assert v[:, 1].min() >= 0.0 and (n[:, 1] >= -1e-6).all()
    assert np.isclose(v[:, 0].max() - v[:, 0].min(), 0.1, atol=1e-6)
    assert 0.8 * 0.02 <= v[:, 2].max() <= 0.02 + 1e-6   # no wider than radius
    order = np.argsort(v[:, 0], kind="stable")
    assert (np.diff(c[order, 3].astype(int)) >= 0).all()
    assert c[:, 3].min() == 0 and c[:, 3].max() == 255


def test_the_grub_is_deterministic_and_the_shipped_one_fits_the_budget():
    """Property: the same arguments give byte-identical arrays, and the
    shipped [render.larvae] grub is 20 to 40 triangles. Breaks if the
    generator draws randomness or the config's rings outgrow the budget."""
    s = _settings()
    a = lb.pack_mesh(s)
    b = lb.pack_mesh(s)
    assert np.array_equal(a, b)
    assert 20 <= a.shape[1] // 3 <= 40


def test_the_mesh_texture_carries_s_exactly():
    """Property: the mesh texture's s is x / length + 1/2 (not the 8-bit
    alpha), in [0, 1]. Breaks if the wave's argument is fed the quantized
    alpha."""
    s = _settings()
    m = lb.pack_mesh(s)
    assert np.allclose(m[0, :, 3], m[0, :, 0] / s.length_m + 0.5, atol=1e-6)
    assert m[0, :, 3].min() >= -1e-6 and m[0, :, 3].max() <= 1 + 1e-6


# --- the instances ----------------------------------------------------------

def test_one_instance_per_valid_larva_in_view():
    """Property: the live pack holds exactly the valid slots inside the view
    (plus a body length of margin), at their world-px positions, and marks
    a larva in coma. Breaks if a freed slot is drawn, the view cull drops a
    visible larva, or the coma flag misreads the state."""
    s = _settings()
    states = [swarm.AI_RUN, swarm.AI_COMA, swarm.AI_EAT, swarm.AI_RUN, 0]
    cols = _cols(5, valid=[1, 1, 1, 0, 0], states=states)
    inst, uid, ph = lb.pack_live(cols, s, PPT, TILE_M)
    assert inst.shape == (2, 3, 4)
    assert uid.tolist() == [1, 2, 3]
    assert np.allclose(inst[0, :, 0], (np.arange(3) + 0.5) * PPT)
    assert inst[1, :, 2].tolist() == [0.0, 1.0, 0.0]
    assert (inst[1, :, 1] == 1.0).all() and (inst[1, :, 0] == 0.0).all()
    # a view to x 1.25 (+ the 0.3-tile margin) holds the first two
    inst, _, _ = lb.pack_live(cols, s, PPT, TILE_M, view=(0.0, 0.0, 1.25, 1.0))
    assert inst.shape[1] == 2


def test_a_still_larva_keeps_its_phase_and_a_crawler_advances():
    """Property: the wriggle phase is a function of dist_walked and the id
    only: the same walk gives the same phase on every frame; walking d metres
    advances it by 2 pi d / stride_m (mod 2 pi). Breaks if the phase reads a
    clock or ignores the walk."""
    s = _settings()
    walked = np.array([5 * ONE, 5 * ONE], np.uint32)
    a = lb.wriggle_phase(walked, [7, 7], s.stride_m, TILE_M)
    b = lb.wriggle_phase(walked.copy(), [7, 7], s.stride_m, TILE_M)
    assert np.array_equal(a, b)
    step_tiles = 0.01
    more = walked + np.uint32(int(step_tiles * ONE))
    c = lb.wriggle_phase(more, [7, 7], s.stride_m, TILE_M)
    want = 2 * np.pi * (int(step_tiles * ONE) / ONE) * TILE_M / s.stride_m
    got = (c.astype(np.float64) - a) % (2 * np.pi)
    assert np.allclose(got, want % (2 * np.pi), atol=1e-4)
    # two larvae with the same walk are not in lockstep
    d = lb.wriggle_phase(walked, [7, 8], s.stride_m, TILE_M)
    assert d[0] != d[1]


def test_the_view_reads_only_its_declared_columns_and_writes_none():
    """Property: ``LarvaBodies.upload`` reads the swarm store only through
    THE column read, only the columns of READ_COLUMNS (plus the presence
    test's own attributes and the tile size), and writes nothing: read-only
    columns pack without error and are byte-identical after. Breaks if the
    view reads an undeclared column or writes the store."""
    cols = _cols(6, walked=np.arange(6) * 1000)
    for a in cols.values():
        a.setflags(write=False)
    before = {k: v.copy() for k, v in cols.items()}
    store = {swarm.store_attr("larva", c): a for c, a in cols.items()}
    seen = []

    class Spy:
        swarm_species = object()
        tile_size_m = TILE_M

        def __getattribute__(self, name):
            seen.append(name)
            if name in store:
                return store[name]
            if name == swarm.high_water_attr("larva"):
                return np.int64(6)
            return object.__getattribute__(self, name)

    view = lb.LarvaBodies()
    view.settings = _settings()
    view.upload(Spy(), 1.0, PPT)
    assert view.instances.shape[1] == 6
    allowed = ({swarm.store_attr("larva", c) for c in lb.READ_COLUMNS}
               | {swarm.high_water_attr(sp) for sp in swarm.SWARM_SPECIES}
               | {"swarm_species", "tile_size_m"})
    assert set(seen) <= allowed, set(seen) - allowed
    for k in cols:
        assert np.array_equal(cols[k], before[k])
    assert set(lb.READ_COLUMNS) <= {c.name for c in swarm.ROSTER}
    assert lb.AI_COMA == swarm.AI_COMA


def test_the_instances_are_exactly_the_living_and_the_fading_dead():
    """Property: a frame's instances are the dead bodies still fading
    (first, drawn under) followed by every living larva; a body is gone
    once limp_s + fade_s have passed on the sim clock and held while the
    clock holds. Breaks if a dead body outlives its fade, a living larva is
    dropped, or the dead are drawn over the living."""
    s = _settings()
    view = lb.LarvaBodies()
    view.settings = s
    cols = _cols(4)
    store = {swarm.store_attr("larva", c): a for c, a in cols.items()}
    gmap = SimpleNamespace(swarm_species=object(), tile_size_m=TILE_M,
                           **store,
                           **{swarm.high_water_attr("larva"): np.int64(4)})
    view.upload(gmap, 0.0, PPT)
    view.on_deaths([2.0, 3.0], [2.0, 3.0], [0.0, 1.0], [101, 102], 0.0)
    view.upload(gmap, 0.1, PPT)
    inst = view.instances
    assert inst.shape[1] == 2 + 4
    assert (inst[1, :2, 0] > 0).all() and (inst[1, 2:, 0] == 0).all()
    life = s.limp_s + s.fade_s
    view.upload(gmap, life - 0.01, PPT)
    assert view.instances.shape[1] == 6
    assert 0.0 < view.instances[1, 0, 1] < 0.01
    view.upload(gmap, life + 0.01, PPT)
    assert view.instances.shape[1] == 4


def test_a_dead_body_goes_limp_then_fades():
    """Property: limp rises 0 -> 1 over limp_s while the body stays opaque,
    then the alpha falls linearly to 0 over fade_s. Breaks if the fade
    starts before the body is limp or never ends."""
    limp, alpha = lb.dead_weights([0.0, 0.25, 0.5, 2.0, 3.5, 9.0], 0.5, 3.0)
    assert np.allclose(limp, [0, 0.5, 1, 1, 1, 1])
    assert np.allclose(alpha, [1, 1, 1, 0.5, 0, 0])


def test_a_death_starts_from_the_phase_it_had_alive_and_the_cap_keeps_the_newest():
    """Property: a dead body's wriggle starts at the phase its larva had in
    the last live pack (no jump at death; an unseen id falls back to its
    hash), and past dead_cap the oldest bodies go first. Breaks if the
    death pops the pose or the store grows without bound."""
    ph = lb.lookup_phase([5, 9, 4], np.array([3, 4, 5]),
                         np.array([0.1, 0.2, 0.3]), [7.0, 8.0, 9.0])
    assert ph.tolist() == [0.3, 8.0, 0.2]
    d = lb.DeadLarvae(cap=3)
    d.add([1, 2], [1, 2], [0, 0], [1, 2], [0, 0], 0.0)
    d.add([3, 4], [3, 4], [0, 0], [3, 4], [0, 0], 1.0)
    assert d.uid.tolist() == [2, 3, 4]


def test_the_instance_texture_layout_wraps_rows():
    """Property: instance i sits at column i % W, rows 2 (i // W) and +1,
    the layout the vertex shader fetches. Breaks if the packing and the
    shader's fetch disagree past the first row."""
    k, w = 7, 3
    inst = np.zeros((2, k, 4), np.float32)
    inst[0, :, 0] = np.arange(k)
    inst[1, :, 0] = 100 + np.arange(k)
    tex = lb.instance_texels(inst, w)
    assert tex.shape == (6, 3, 4)
    for i in range(k):
        assert tex[2 * (i // w), i % w, 0] == i
        assert tex[2 * (i // w) + 1, i % w, 0] == 100 + i


def test_off_reads_nothing_and_packs_nothing():
    """Property: with ``enabled = false`` the upload touches no gmap
    attribute, forgets the dead and packs nothing; deaths are ignored.
    Breaks if the off switch still reads the store."""
    class Spy:
        def __getattribute__(self, name):
            raise AssertionError(f"read gmap.{name} with the view off")

    view = lb.LarvaBodies()
    view.settings = _settings(enabled=False)
    view.dead.add([1.0], [1.0], [0.0], [1], [0.0], 0.0)
    view.upload(Spy(), 1.0, PPT)
    view.on_deaths([1.0], [1.0], [0.0], [7], 1.0)
    assert view.instances is None and len(view.dead) == 0


# --- the draw's place and the shaders ----------------------------------------

def test_compose_world_draws_the_bodies_on_the_floor_under_the_smoke():
    """Property: ``compose_world`` draws the larvae once, after the water
    and before the floor smoke, the debris, the units and the resolve, with
    the dot view (a debug picture) over them. Breaks if the bodies slip over
    the smoke or the units, or past the one tone-map."""
    from renderer.game_renderer import GameRenderer
    src = inspect.getsource(GameRenderer.compose_world)
    order = ["self.water_pass.draw(", "self._draw_larva_bodies()",
             "self._draw_swarm_dots()", 'self.gas_relief.draw("floor"',
             "self._draw_debris(", "self._draw_units_3d_world(",
             "self.world.resolve("]
    pos = [src.index(k) for k in order]
    assert pos == sorted(pos), order
    assert src.count("self._draw_larva_bodies()") == 1


def test_the_shader_is_lit_through_the_lit3d_block_and_never_tone_maps():
    """Property: the fragment shader is composed with THE lit-3D block (the
    field sample, the smoke cast shadow, the light curve), and neither stage
    tone-maps, dithers or reads a clock. Breaks if the GLSL is re-typed, a
    second light read appears, or the pose starts to depend on time."""
    from renderer.lit3d import (_FIELD_SAMPLE_GLSL, _LIGHT_CURVE_GLSL,
                                _SMOKE_SHADOW_GLSL)
    vs, fs = lb.shader_sources()
    for block in (_FIELD_SAMPLE_GLSL, _SMOKE_SHADOW_GLSL, _LIGHT_CURVE_GLSL):
        assert block in fs
    for src in (vs, fs):
        low = src.lower()
        for bad in ("aces", "dither", "u_time", "u_clock"):
            assert bad not in low, bad


def test_nothing_on_the_sim_side_reads_the_bodies():
    """Property: no module under src/ names this view or its config table,
    so a run with the bodies on and one with them off are the same run.
    Breaks if a sim module starts reading it."""
    bad = [p.name for p in (ROOT / "src").rglob("*.py")
           if "larva_bodies" in (t := p.read_text(encoding="utf-8"))
           or "render.larvae" in t]
    assert not bad, bad


def test_digests_equal_with_the_bodies_on_and_off():
    """Property: a larva_studio run packed for the bodies every tick (on,
    deaths included) and the same run never packed (off) have equal tick
    digests every tick. Breaks if building the view writes sim state."""
    bp = pytest.importorskip("breach_physics")
    import field_ab_harness as fab
    import level_loader
    from field_digest import tick_digest
    from simulation.events import SwarmUnitDiedEvent
    from simulation.simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    lv = level_loader.load("larva_studio", str(ROOT / "levels"))
    s = _settings()

    def run(on):
        sim = Simulation(lv, seed=3, breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        view, out = lb.LarvaBodies(), []
        view.settings = s
        for t in range(30):
            sim.set_paused(False)
            sim.step()
            if on:
                view.upload(sim.gmap, t / 24.0, PPT)
                died = [e for e in sim.tick_events
                        if isinstance(e, SwarmUnitDiedEvent)]
                view.on_deaths([e.pos_x_q / ONE for e in died],
                               [e.pos_y_q / ONE for e in died],
                               [e.heading_q / ONE for e in died],
                               [e.unit_id for e in died], t / 24.0)
            out.append(tick_digest(fab.capture_snapshot(sim)))
        return out
    assert run(True) == run(False)


# --- the door ---------------------------------------------------------------

_DROP = object()


def test_config_names_every_larva_dial_and_the_door_refuses():
    """Property: ``[render.larvae]`` names every LarvaSettings dial and
    nothing else, the settings carry its values, and the door refuses a
    missing section or key, a mistyped value and the ranges the shader
    relies on (a peristalsis that folds the body, a sway past 0.3) by name.
    Breaks if a dial falls back silently or the config drifts."""
    cfg = _cfg()
    sec = cfg.render.larvae
    assert set(vars(sec)) == set(lb.larva_config_keys())
    st = lb.larva_settings_from_config(cfg)
    for k in lb.larva_config_keys():
        want, got = getattr(sec, k), getattr(st, k)
        assert (tuple(got) == tuple(want) if k.endswith("_rgb")
                else got == want), k

    def cfg_with(**over):
        d = dict(vars(sec))
        d.update(over)
        d = {k: v for k, v in d.items() if v is not _DROP}
        return SimpleNamespace(render=SimpleNamespace(
            larvae=SimpleNamespace(**d)))

    with pytest.raises(ValueError, match="no \\[render.larvae\\]"):
        lb.larva_settings_from_config(SimpleNamespace(render=SimpleNamespace()))
    with pytest.raises(ValueError, match="missing 'stride_m'"):
        lb.larva_settings_from_config(cfg_with(stride_m=_DROP))
    with pytest.raises(ValueError, match="rings has the wrong type"):
        lb.larva_settings_from_config(cfg_with(rings=6.0))
    with pytest.raises(ValueError, match="enabled has the wrong type"):
        lb.larva_settings_from_config(cfg_with(enabled=1))
    with pytest.raises(ValueError, match="body_rgb has the wrong type"):
        lb.larva_settings_from_config(cfg_with(body_rgb=[1, 2]))
    with pytest.raises(ValueError, match="peristalsis"):
        lb.larva_settings_from_config(cfg_with(peristalsis=0.95))
    with pytest.raises(ValueError, match="sway"):
        lb.larva_settings_from_config(cfg_with(sway=0.5))
    with pytest.raises(ValueError, match="fade_s must be > 0"):
        lb.larva_settings_from_config(cfg_with(fade_s=0.0))


def test_the_dot_view_defaults_off_now_the_larvae_have_bodies():
    """Property: the shipped config draws the bodies and not the debug dots
    (Erik's order 2026-11-19: the dots stay a debug view). Breaks if both
    default on, or the bodies default off."""
    cfg = _cfg()
    assert cfg.render.larvae.enabled is True
    assert cfg.debug.swarm_dots is False
