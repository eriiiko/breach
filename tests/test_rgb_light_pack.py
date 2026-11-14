"""The RGB light the eye sees -- a lamp's colour survives the sweep and the
accessor, and the 16F pack keeps the shader's layout and sign convention.

Rewritten at ray-engine-v2 P6c (design v3 §11.2 row "test_rgb_light_pack.py: 1
survives (pack layout), 2 rewritten against the accessor"). Until P6c these
cast the old C++ render march and packed its output by hand:

  * "the default LightSource colour is white" -- the LightSource struct is
    gone with the march; a light's colour is its LightSpec's, and the cone
    row's rgb = intensity x colour is held by
    tests/test_frame_lights_cones.py::test_the_cone_row_door_quantizes_once_and_keeps_the_old_conventions;
  * "the deposit matches the colour ratio" -- HERE, end to end through the real
    game path (cone row -> Simulation.set_light / relight -> read_light);
  * "the 16F pack layout and its signed direction" -- HERE, on the pack the game
    runs (renderer.lighting.pack_light_view).

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_rgb_light_pack.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from level_loader import LevelData  # noqa: E402
from renderer import frame_lights as fl  # noqa: E402
from renderer.lighting import pack_light_view  # noqa: E402
from simulation import Simulation, light_field  # noqa: E402

COLOR = (1.0, 0.6, 0.2)


def _lit_room(color=COLOR, n=30):
    """An n x n hull-ringed air room, one omni lamp of `color` at its centre,
    relit through the game's own path; returns (gmap, lamp cell (y, x))."""
    tm = np.full((n, n), 4, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    lvl = LevelData(name="rgb_pack", version="1", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    c = n // 2
    spec = fl.LightSpec(x=c + 0.5, y=c + 0.5, color=color, intensity=1.0)
    rows = fl.cone_rows([spec], n, n, int(bp.L_FINE_BITS))
    sim.set_light(True, rows, None)
    sim.relight()
    return sim.gmap, (c, c)


def test_a_lamps_colour_reaches_the_eye_in_its_ratio():
    """PROPERTY: a (1.0, 0.6, 0.2) lamp in a clear room reaches the renderer's
    read (light_field.read_light) in its own colour ratio -- G/R = 0.6 and
    B/R = 0.2 in every lit cell, to the integer currency's rounding -- so the
    sweep and the accessor carry colour, not just brightness.

    BREAKS IF: the cone door or the sweep swaps, drops or mixes a channel, or
    the accessor dequantizes the channels on different scales.
    """
    g, (cy, cx) = _lit_room()
    rgb = light_field.read_light(g).rgb.astype(np.float64)
    lit = rgb[..., 0] > 1e-6
    lit[cy, cx] = False
    assert lit.sum() > 100, "the lamp must light the room"
    r = rgb[..., 0][lit]
    assert np.allclose(rgb[..., 1][lit] / r, 0.6, atol=1e-3)
    assert np.allclose(rgb[..., 2][lit] / r, 0.2, atol=1e-3)


def test_the_pack_is_16f_with_the_direction_pointing_toward_the_light():
    """PROPERTY (the texture contract the ship shader, the 3D units and the
    water pass read): pack_light_view writes float16 textures with texture A =
    (the calibrated RGB, dir.x) and texture B = (the glow, dir.y); the stored
    direction is SIGNED (no 0.5-centred encode) and points TOWARD the light --
    negative x on the +x side of a lamp, positive on its -x side, negative y
    below it -- and a room without smoke packs zero glow.

    BREAKS IF: the pack stops negating the sweep's flux (which points the way
    light travels), re-centres the direction, changes a texture's slot layout
    or dtype, or packs glow where there is none.
    """
    g, (cy, cx) = _lit_room()
    view = light_field.read_light(g)
    h, w = view.rgb.shape[:2]
    gain = 1.0
    light_rgb = np.zeros((h, w, 3), np.float32)
    ldx = np.zeros((h, w), np.float32)
    ldy = np.zeros((h, w), np.float32)
    lmap = np.zeros((h, w), np.float32)
    pa = np.zeros((h, w, 4), np.float16)
    pb = np.zeros((h, w, 4), np.float16)
    pack_light_view(view, gain, light_rgb, ldx, ldy, lmap, pa, pb)
    assert pa.dtype == np.float16 and pb.dtype == np.float16
    assert np.array_equal(pa[..., 0:3], light_rgb.astype(np.float16))
    assert float(pa[cy, cx + 5, 3]) < 0.0 and float(pa[cy, cx - 5, 3]) > 0.0
    assert float(pb[cy + 5, cx, 3]) < 0.0 and float(pb[cy - 5, cx, 3]) > 0.0
    assert not np.any(pb[..., 0:3])
