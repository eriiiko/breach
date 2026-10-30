"""Heat shimmer, the first term of the resolve's DISTORTION BLOCK (graphics
arc G3 "Fire", patch F4a; docs/g3_fire_design_2026-10-23.md §6.2, §13.12).

Pure numpy and source scans, no GL context. The GL side (a cold map resolves
bit-identical to no shimmer; a hot tile shifts only the pixels around it) is
``tests/gl_fire_vfx_check.py`` (``shimmer_zero_where_cold``).

Every test names the PROPERTY it protects and the change that must break it.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from renderer import heat_shimmer as H  # noqa: E402

RESOLVE = ROOT / "shaders" / "world_resolve.fs"
SRC = ROOT / "renderer" / "heat_shimmer.py"


def _code(glsl: str) -> str:
    glsl = re.sub(r"/\*.*?\*/", "", glsl, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", glsl)


def test_strength_is_zero_at_or_below_the_threshold_and_off_gas():
    """Property: the per-tile strength is exactly 0 at or below ``t0_k``
    and on solid / vacuum tiles, in [0, 1], and monotone in T. Breaks if
    the threshold leaks (cold gas shimmering) or walls shimmer."""
    t = np.linspace(200.0, 1500.0, 400)
    gas = np.ones_like(t, bool)
    s = H.shimmer_strength(t, gas, 380.0, 700.0)
    assert np.all(s[t <= 380.0] == 0.0)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert np.all(np.diff(s) >= 0.0)
    assert np.all(s[t >= 700.0] == 1.0)
    assert np.all(H.shimmer_strength(t, ~gas, 380.0, 700.0) == 0.0)


def test_offset_is_zero_where_the_gas_is_cold_and_bounded_everywhere():
    """Property: with ONE hot tile in a cold map, the shimmer offset
    (bilinear-upsampled strength, the spec of the resolve's term) is exactly
    zero at every pixel farther than one tile from that tile's centre, is
    nonzero right over it, and never longer than ``amp`` anywhere. Breaks
    if the shimmer leaks into cold areas (a wider filter, a floor on the
    strength) or the clamp is lost."""
    h, w = 12, 14
    tk = np.full((h, w), 293.0)
    tk[6, 7] = 900.0
    st = H.shimmer_strength(tk, np.ones((h, w), bool), 380.0, 700.0)
    m = 16                                         # pixels per tile
    py, px = np.meshgrid((np.arange(h * m) + 0.5) / m,
                         (np.arange(w * m) + 0.5) / m, indexing="ij")
    s = H.bilinear_tiles(st, px, py)
    rng = np.random.default_rng(1)
    noise = rng.uniform(-1.0, 1.0, s.shape + (2,))
    amp = 1.7
    off = H.shimmer_offset_px(s, noise, amp)
    length = np.linalg.norm(off, axis=-1)
    assert np.all(length <= amp + 1e-12)
    far = (np.abs(px - 7.5) >= 1.0) | (np.abs(py - 6.5) >= 1.0)
    assert np.all(length[far] == 0.0), "shimmer leaked into cold gas"
    assert length[~far].max() > 0.3 * amp


def test_the_zoom_fades_it_out():
    """Property: the shimmer is 0 at or below ``zoom_zero`` screen px per
    tile, 1 from ``zoom_full``, monotone between. Breaks if a zoomed-out
    view swims (the fade is lost)."""
    z = np.linspace(4.0, 64.0, 200)
    f = np.array([H.zoom_fade(v, 14.0, 32.0) for v in z])
    assert np.all(f[z <= 14.0] == 0.0) and np.all(f[z >= 32.0] == 1.0)
    assert np.all(np.diff(f) >= 0.0)


def test_flow_phases_stay_bounded_and_change_with_time():
    """Property: the flow map's phases and weight are bounded (the GPU only
    ever sees small numbers, at any game age) and move with the VFX clock.
    Breaks if absolute time reaches the shader."""
    st = H.ShimmerSettings()
    for t in (0.0, 1.3, 3600.0 * 5 + 0.7):
        ph0, ph1, w0, offs, dirs = H.flow_phases(t, st)
        assert 0.0 <= ph0 < 1.0 and 0.0 <= ph1 < 1.0 and 0.0 <= w0 <= 1.0
        assert all(0.0 <= o < 1.0 for o in offs)
        assert np.allclose(np.hypot(dirs[0::2], dirs[1::2]), 1.0)
    assert H.flow_phases(1.0, st)[0] != H.flow_phases(1.1, st)[0]


def test_the_resolve_carries_the_one_distortion_block():
    """Property: ``world_resolve.fs`` has ONE distortion block
    (``distortion_px``) with the shimmer as a term in it; a zero offset
    reads the scene with the plain point fetch, a shifted read does its OWN
    bilinear tap (texelFetch; §13.12); and the distortion happens before
    the tone-map, in the one resolve. Breaks if a distortion grows its own
    pass, the scene RT's filter is changed instead, or a cold pixel stops
    being bit-identical."""
    code = _code(RESOLVE.read_text(encoding="utf-8"))
    assert code.count("vec2 distortion_px(") == 1
    block = code[code.index("vec2 distortion_px("):]
    block = block[:block.index("}") + 1]
    assert "shimmer_term(" in block
    main = code[code.index("void main()"):]
    assert "distortion_px(" in main
    assert "texture(texture0, fragTexCoord)" in main
    assert "scene_bilinear(" in main
    assert main.index("distortion_px(") < main.index("aces_tonemap(")
    assert "texelFetch(texture0" in code
    # the composite never changes the scene RT's filter for it
    wc = (ROOT / "renderer" / "world_composite.py").read_text(encoding="utf-8")
    assert wc.count("TEXTURE_FILTER_POINT") == 1


def test_no_second_distortion_pass_and_the_hud_draws_after():
    """Property: the shimmer reaches the picture only through
    ``WorldComposite.resolve(shimmer=...)``; nothing else in the renderer
    samples the scene RT, and every display-referred layer (the HUD / UI,
    ``overlay_fn``) is drawn after the resolve -- so it is never distorted.
    Breaks if a second full-screen distortion pass appears or UI moves into
    the scene."""
    gr = (ROOT / "renderer" / "game_renderer.py").read_text(encoding="utf-8")
    assert gr.count("shimmer=self.shimmer") == 1
    assert "scene_rt" not in gr
    src = gr[gr.index("def compose_world("):]
    src = src[:src.index("\n    def ", 10)]
    assert (src.index("self.world.resolve(")
            < src.index("overlay_fn(self.world.world_px_per_tile)"))


def test_config_names_every_shimmer_dial():
    """Property: ``[render.shimmer]`` names every ShimmerSettings dial and
    nothing else. Breaks if a dial is added without a config row."""
    from config import CFG
    sec = CFG.render.shimmer
    assert set(vars(sec)) == set(H.config_keys())
    st = H.settings_from_config(CFG)
    for k in H.config_keys():
        assert getattr(st, k) == getattr(sec, k), k


def test_shimmer_is_render_only():
    """Property: ``renderer/heat_shimmer.py`` never imports the sim nor
    assigns into a gmap / sim object. Breaks if it grows a sim write."""
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
                base = t
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and t is not base:
                    assert base.id not in ("gmap", "sim", "t_game", "solid",
                                           "vacuum"), \
                        f"line {node.lineno} assigns into {base.id}"
