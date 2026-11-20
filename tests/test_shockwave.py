"""The shock front, the second term of the resolve's DISTORTION BLOCK
(graphics arc G4, patch P2; docs/explosion_look_design_2026-11-07.md §3.3 as
amended by §6, §4).

Pure numpy and source scans, no GL context. The GL side (no ring / a far ring
resolves bit-identical; one ring over a checker equals the twin's resample)
is ``tests/gl_shockwave_check.py`` (wrapper ``tests/test_gl_shockwave.py``).

Every test names the PROPERTY it protects and the change that must break it.
"""
from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from renderer import shockwave as S  # noqa: E402

RESOLVE = ROOT / "shaders" / "world_resolve.fs"
SRC = ROOT / "renderer" / "shockwave.py"
PPT = 48.0                       # world px per tile


def _settings(**kw) -> S.ShockwaveSettings:
    base = dict(enabled=True, speed_tiles_s=40.0, width_tiles=1.0,
                amp_tiles=0.25, strength_power=0.5, life_s=0.5, max_rings=4,
                zoom_zero=14.0,
                zoom_full=32.0)
    base.update(kw)
    return S.ShockwaveSettings(**base)


def _room(h=30, w=30):
    """An open room, a one-tile wall ring round it."""
    m = np.zeros((h, w), bool)
    m[1:-1, 1:-1] = True
    return m


def _code(glsl: str) -> str:
    glsl = re.sub(r"/\*.*?\*/", "", glsl, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", glsl)


def _one(plane, cx, cy, r, amp=10.0):
    return [(plane, cx, cy, r, amp)]


def test_offset_is_zero_outside_the_band_and_where_the_walk_did_not_reach():
    """Property: the twin's offset is EXACTLY zero where the ring's
    distance (bilinear) is 2 w or more from r, and wherever the bilinear read
    touches an unreached tile (a wall), so a pixel there resolves with the
    plain point fetch. Breaks if the band loses its compact support, or an
    unreached tile stops reading far outside it (a -1 or 0 fill)."""
    m = _room()
    m[:, 20] = False                                 # a wall, no gap
    plane = S.distance_plane(m, 15, 10, 30.0)
    assert np.all(plane[:, 20:] == S.UNREACHED)
    ys, xs = np.mgrid[0:30 * 8, 0:30 * 8] / 8.0 + 1.0 / 16.0
    r = 6.0
    off = S.ring_offset_px(xs, ys, _one(plane, 10.5, 15.5, r), 1.0)
    mag = np.linalg.norm(off, axis=-1)
    d = S.bilinear_tiles(plane.astype(np.float64), xs, ys)
    assert np.all(mag[np.abs(d - r) >= 2.0] == 0.0)
    assert mag.max() > 5.0                           # and the band is there
    # beyond the wall's centre line every pixel reads an unreached tile
    assert np.all(mag[:, xs[0] >= 19.5] == 0.0)
    # no matter how far the ring has gone
    for r in (9.0, 10.0, 12.0, 15.0):
        off = S.ring_offset_px(xs, ys, _one(plane, 10.5, 15.5, r), 1.0)
        assert np.all(np.linalg.norm(off, axis=-1)[:, xs[0] >= 19.5] == 0.0)


def test_a_rings_offset_points_away_from_its_centre():
    """Property: every non-zero offset of one ring points away from its
    centre (positive dot with p - c; y south, as world px are). Breaks if a
    sign flips (the ring pulling inward) or the y axis is mirrored."""
    plane = S.distance_plane(_room(), 15, 15, 30.0)
    ys, xs = np.mgrid[0:240, 0:240] / 8.0 + 1.0 / 16.0
    off = S.ring_offset_px(xs, ys, _one(plane, 15.5, 15.5, 7.0), 1.0)
    nz = np.linalg.norm(off, axis=-1) > 1e-9
    assert nz.sum() > 500
    dot = off[..., 0] * (xs - 15.5) + off[..., 1] * (ys - 15.5)
    assert np.all(dot[nz] > 0.0)


def test_two_rings_sum():
    """Property: overlapping rings ADD as vectors (below the clamp): the
    offset of two rings is the sum of each alone. Breaks if the block takes
    the max, or only the first ring, instead of the sum."""
    m = _room(30, 40)
    p1 = S.distance_plane(m, 15, 10, 40.0)
    p2 = S.distance_plane(m, 15, 30, 40.0)
    xs = np.linspace(17.0, 23.0, 61)
    ys = np.full_like(xs, 15.5)
    a = S.ring_offset_px(xs, ys, [(p1, 10.5, 15.5, 10.0, 8.0)], 1.0)
    b = S.ring_offset_px(xs, ys, [(p2, 30.5, 15.5, 10.0, 4.0)], 1.0)
    ab = S.ring_offset_px(xs, ys, [(p1, 10.5, 15.5, 10.0, 8.0),
                                   (p2, 30.5, 15.5, 10.0, 4.0)], 1.0)
    both = (np.linalg.norm(a, axis=-1) > 0.5) & (np.linalg.norm(b, axis=-1) > 0.5)
    assert both.sum() > 5
    assert np.allclose(ab[both], (a + b)[both], atol=1e-9)
    # and the sum is clamped to the strongest ring's amplitude
    same = S.ring_offset_px(xs, ys, [(p1, 10.5, 15.5, 10.0, 8.0),
                                     (p1, 10.5, 15.5, 10.0, 8.0)], 1.0)
    assert np.linalg.norm(same, axis=-1).max() <= 8.0 + 1e-9


def test_a_ring_dies_at_life_s():
    """Property: a ring lives on the sim clock exactly while its age
    ``(tick - tick0 + alpha) / tps`` is under ``life_s``, and a paused clock
    holds its age. Breaks if the ring ages on wall time or outlives life."""
    rs = S.RingSet(_settings(life_s=0.5))
    rs.start((10, 10), 1.0, 100, _room())
    tps = 24.0
    rs.set_clock(111, 0.0, False, tps)               # 11/24 s
    rs.expire()
    assert len(rs.rings) == 1
    rs.set_clock(111, 0.9, True, tps)                # paused: holds
    assert rs.age(rs.rings[0]) == pytest.approx(11 / 24)
    rs.set_clock(112, 0.0, False, tps)               # 12/24 s = life
    rs.expire()
    assert rs.rings == []


def test_max_rings_holds_and_the_oldest_is_dropped():
    """Property: never more than ``max_rings`` live rings; a new ring past
    the cap drops the OLDEST, and the freed slot is reused (one RGBA
    texture). Breaks if the cap is lost or the newest is dropped."""
    rs = S.RingSet(_settings(max_rings=3))
    m = _room()
    for k in range(5):
        rs.start((5 + k, 10), 1.0, 100 + k, m)
        assert len(rs.rings) <= 3
    assert [rg.tick0 for rg in rs.rings] == [102, 103, 104]
    assert sorted(rg.slot for rg in rs.rings) == [0, 1, 2]
    pack = rs.packed_planes(30, 30)
    for rg in rs.rings:
        assert np.array_equal(pack[..., rg.slot], rg.plane)
    assert np.all(pack[..., 3] == S.UNREACHED)


def test_the_ring_is_sharper_than_a_tile():
    """Property: the ring is evaluated PER PIXEL: two pixels inside one tile,
    on either side of the band's peak, get clearly different offsets.
    Breaks if the term is sampled per tile (a blocky ring)."""
    plane = S.distance_plane(_room(), 15, 15, 30.0)
    # along the row through the centre, the tile spanning x in [22, 23)
    xs = np.array([22.1, 22.5, 22.9])
    ys = np.full(3, 15.5)
    r = 22.5 - 15.5
    off = S.ring_offset_px(xs, ys, _one(plane, 15.5, 15.5, r, 10.0), 0.3)
    mag = np.linalg.norm(off, axis=-1)
    assert mag[1] > 9.0
    assert mag[0] < 0.5 * mag[1] and mag[2] < 0.5 * mag[1]


def test_the_ring_follows_the_walk():
    """Property: the ring rides the canonical walk: a tile behind a wall with
    no way round never offsets; the same tile reached through a gap offsets
    when r equals its walk distance (straight line + the way round), not
    when r equals its straight-line distance; and in an open room the
    distance is the straight line (a round ring, not the hop count's
    diamond). Breaks if the ring is a plain circle (shows through walls), or
    rides raw hops (a diamond), or ignores the walk."""
    m = _room(30, 30)
    m[5:25, 15] = False                              # a wall x = 15, rows 5..24
    sealed = m.copy()
    sealed[1:5, 15] = False
    sealed[25:29, 15] = False                        # ... and no way round
    ty, tx = 15, 18                                  # behind the wall
    px, py = tx + 0.5, ty + 0.5
    straight = math.hypot(tx - 12, ty - 15)
    p_sealed = S.distance_plane(sealed, 15, 12, 40.0)
    for r in np.arange(0.0, 45.0, 0.25):
        off = S.ring_offset_px(np.array([px]), np.array([py]),
                               _one(p_sealed, 12.5, 15.5, r), 1.0)
        assert not off.any(), f"offset through a wall at r={r}"
    p_gap = S.distance_plane(m, 15, 12, 40.0)
    d_walk = float(p_gap[ty, tx])
    assert d_walk > straight + 4.0                   # the way round is longer
    at = lambda r: float(np.linalg.norm(S.ring_offset_px(
        np.array([px]), np.array([py]), _one(p_gap, 12.5, 15.5, r), 1.0)))
    assert at(d_walk) > 9.0
    assert at(straight) == 0.0
    # open room: the plane is the straight line from the centre
    p_open = S.distance_plane(_room(), 15, 15, 30.0)
    yy, xx = np.nonzero(p_open < S.UNREACHED)
    assert np.allclose(p_open[yy, xx], np.hypot(yy - 15, xx - 15), atol=1e-5)


def test_the_band_box_never_cuts_a_nonzero_offset():
    """Property: every non-zero offset lies inside ``band_box`` (the shader's
    early-out box), and with no ring in band the box is None. Breaks if the
    box is computed too tight (a ring clipped to a rectangle)."""
    m = _room(30, 40)
    m[5:25, 22] = False
    rings = [(S.distance_plane(m, 15, 10, 40.0), 10.5, 15.5, 9.0, 10.0),
             (S.distance_plane(m, 8, 30, 40.0), 30.5, 8.5, 4.0, 6.0)]
    box = S.band_box([rg[0] for rg in rings], [rg[3] for rg in rings], 1.0)
    ys, xs = np.mgrid[0:30 * 8, 0:40 * 8] / 8.0 + 1.0 / 16.0
    off = S.ring_offset_px(xs, ys, rings, 1.0)
    nz = np.linalg.norm(off, axis=-1) > 0.0
    x0, y0, x1, y1 = box
    assert nz.any()
    assert np.all((xs[nz] >= x0) & (xs[nz] < x1)
                  & (ys[nz] >= y0) & (ys[nz] < y1))
    assert np.array_equal(S.ring_offset_px(xs, ys, rings, 1.0, box=box), off)
    assert S.band_box([rings[0][0]], [200.0], 1.0) is None


def _cfg(**kw):
    base = dict(enabled=True, speed_tiles_s=40.0, width_tiles=1.0,
                amp_tiles=0.25, strength_power=0.5, life_s=0.5, max_rings=4,
                zoom_zero=14.0,
                zoom_full=32.0)
    base.update(kw)
    return SimpleNamespace(render=SimpleNamespace(
        shockwave=SimpleNamespace(**base)))


def test_the_door_refuses_a_missing_or_mistyped_key():
    """Property: ``settings_from_config`` refuses a missing section, a
    missing key and a mistyped value, naming it, and a max_rings beyond the
    one RGBA texture. Breaks if a dial silently falls back to a default."""
    assert S.settings_from_config(_cfg()).max_rings == 4
    for name in S.config_keys():
        cfg = _cfg()
        delattr(cfg.render.shockwave, name)
        with pytest.raises(ValueError, match=name):
            S.settings_from_config(cfg)
    with pytest.raises(ValueError, match="enabled"):
        S.settings_from_config(_cfg(enabled=1))
    with pytest.raises(ValueError, match="max_rings"):
        S.settings_from_config(_cfg(max_rings=4.0))
    with pytest.raises(ValueError, match="speed_tiles_s"):
        S.settings_from_config(_cfg(speed_tiles_s="fast"))
    with pytest.raises(ValueError, match="max_rings"):
        S.settings_from_config(_cfg(max_rings=5))
    with pytest.raises(ValueError, match="shockwave"):
        S.settings_from_config(SimpleNamespace(render=SimpleNamespace()))


def _headless_shockwave(settings):
    """A Shockwave without its GL texture (uniforms / active are pure)."""
    sw = S.Shockwave.__new__(S.Shockwave)
    sw.h, sw.w = 30, 30
    sw.set = S.RingSet(settings)
    sw.zoom_px_per_tile = 64.0
    return sw


def test_no_live_ring_means_the_term_is_inactive():
    """Property: with no live ring (none started, or the last one dead) the
    term is inactive and ``u_shock_n`` is 0, so the shader returns before
    any fetch; a live ring turns it on. Breaks if a dead ring keeps the
    term running (a cost and a non-identical resolve)."""
    sw = _headless_shockwave(_settings())
    assert not sw.active(PPT)
    assert sw.uniforms(PPT)["u_shock_n"] == (0,)
    sw.set.start((15, 15), 1.0, 10, _room())
    sw.set.set_clock(12, 0.0, False, 24.0)
    assert sw.active(PPT)
    u = sw.uniforms(PPT)
    assert u["u_shock_n"] == (1,)
    assert u["u_shock_ring[0]"][3] == pytest.approx(0.25 * PPT)
    sw.set.set_clock(22, 0.0, False, 24.0)
    sw.set.expire()
    assert not sw.active(PPT) and sw.uniforms(PPT)["u_shock_n"] == (0,)
    off = _headless_shockwave(_settings(enabled=False))
    off.set.start((15, 15), 1.0, 10, _room())
    assert not off.active(PPT)


def test_the_amplitude_follows_strength_to_the_dialled_power_and_fades_with_zoom():
    """Property: a ring's amplitude is ``amp_tiles * strength ** strength_power
    * px_per_tile * zoom_fade`` (0 zoomed out): at 0.5 the sqrt, at 0 every
    blast alike. Breaks if strength enters linearly, the power dial is
    ignored, or the zoom fade is lost."""
    for power, factor in ((0.5, 2.0), (0.0, 1.0), (0.25, 4.0 ** 0.25)):
        rs = S.RingSet(_settings(strength_power=power))
        rs.start((10, 10), 4.0, 0, _room())
        rs.set_clock(1, 0.0, False, 24.0)
        (_, _, _, _, amp), = rs.twin_rings(PPT, 64.0)
        assert amp == pytest.approx(0.25 * factor * PPT)
        (_, _, _, _, amp), = rs.twin_rings(PPT, 10.0)
        assert amp == 0.0


def test_the_resolve_carries_the_shock_term_in_the_one_block():
    """Property: ``shock_term`` is a term of the ONE ``distortion_px`` block
    in ``world_resolve.fs`` and returns before any fetch when ``u_shock_n``
    is 0; the renderer hands it only through ``resolve(shock=...)``. Breaks
    if the ring grows its own pass or fetches with no ring live."""
    code = _code(RESOLVE.read_text(encoding="utf-8"))
    block = code[code.index("vec2 distortion_px("):]
    block = block[:block.index("}") + 1]
    assert "shock_term(" in block and "shimmer_term(" in block
    term = code[code.index("vec2 shock_term("):]
    first = term.index("texture(")
    assert term.index("if (u_shock_n == 0) return vec2(0.0);") < first
    gr = (ROOT / "renderer" / "game_renderer.py").read_text(encoding="utf-8")
    assert gr.count("shock=self.shockwave") == 1


def test_config_names_every_shockwave_dial():
    """Property: ``[render.shockwave]`` names every ShockwaveSettings dial
    and nothing else, and the shipped table passes the door. Breaks if a
    dial is added without a config row."""
    from config import CFG
    assert set(vars(CFG.render.shockwave)) == set(S.config_keys())
    S.settings_from_config(CFG)


def test_shockwave_writes_no_sim_state():
    """Property: ``renderer/shockwave.py`` imports from the sim only the
    walk (``blast_expansion``) and assigns into no gmap / sim object. Breaks
    if it grows a sim write or a second walk."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for n in names:
                if n.split(".")[0] in ("simulation", "breach_physics"):
                    assert n == "simulation.blast_expansion", n
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                base = t
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and t is not base:
                    assert base.id not in ("gmap", "sim", "mask"), \
                        f"line {node.lineno} assigns into {base.id}"
