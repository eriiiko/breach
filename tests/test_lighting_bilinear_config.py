"""#59 — bilinear filtering on the light texture is a config setting, read
once at ``LightingPass`` construction (``renderer/lighting.py``), not a live
B-key toggle (moved out of ``renderer/game_renderer.py``'s debug-key block;
see ``test_debug_key_b_no_bilinear.py``).

Property under test: ``LightingPass.__init__`` reads
``[render.lighting] bilinear`` from ``config.CFG`` exactly once and applies
it to ``light_tex_a`` / ``light_tex_b`` -- ``bilinear = true`` leaves the
default bilinear filter (set by ``core.create_dynamic_rgba16f_texture``)
alone, ``bilinear = false`` re-filters both textures to POINT. A regression
that ignores the config value, or that only applies it to one of the two
textures, breaks this.

No GL context / window is constructed: ``rl`` and ``core`` (the modules
``renderer.lighting`` calls into) are swapped for lightweight fakes that
record calls, the same technique ``tests/test_onephase_control.py`` uses for
``control_onephase``'s ``rl`` binding.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

from renderer import lighting


class _TextureFilter:
    TEXTURE_FILTER_POINT = "POINT"
    TEXTURE_FILTER_BILINEAR = "BILINEAR"


class _ShaderUniformDataType:
    SHADER_UNIFORM_VEC3 = "VEC3"
    SHADER_UNIFORM_FLOAT = "FLOAT"
    SHADER_UNIFORM_INT = "INT"
    SHADER_UNIFORM_VEC4 = "VEC4"


class _Ffi:
    def new(self, typestr, values):
        return list(values)


class FakeRl:
    """Records every filter change; no-ops everything else LightingPass.__init__
    needs (uniform pushes), so construction never touches a real GL context."""

    TextureFilter = _TextureFilter()
    ShaderUniformDataType = _ShaderUniformDataType()
    ffi = _Ffi()

    def __init__(self):
        self.filter_calls = []  # list of (texture, filter) in call order

    def set_texture_filter(self, tex, filt):
        self.filter_calls.append((tex, filt))

    def get_shader_location(self, shader, name):
        return 1

    def set_shader_value(self, shader, loc, val, dtype):
        pass


class _Texture:
    """A distinguishable sentinel so filter_calls can tell the two light
    textures and the vacuum texture apart."""
    def __init__(self, tag):
        self.tag = tag


class FakeCore:
    def __init__(self):
        self._n = 0

    def create_dynamic_rgba16f_texture(self, w, h):
        self._n += 1
        return _Texture(f"light_{self._n}")

    def create_dynamic_rgba_texture(self, w, h):
        return _Texture("vacuum")

    def load_shader_with_fallback(self, vs_path, fs_path):
        return object()  # opaque shader handle; never inspected


class _LightingCfg:
    def __init__(self, bilinear):
        self.sweep_gain = 1.0
        self.floor = (0.03, 0.03, 0.04)
        self.master_gain = 1.0
        self.bilinear = bilinear


class _RenderCfg:
    def __init__(self, bilinear):
        self.lighting = _LightingCfg(bilinear)


class _FakeCFG:
    def __init__(self, bilinear):
        self.render = _RenderCfg(bilinear)


@pytest.fixture
def faked(monkeypatch):
    def _make(bilinear: bool):
        fake_rl = FakeRl()
        fake_core = FakeCore()
        monkeypatch.setattr(lighting, "rl", fake_rl)
        monkeypatch.setattr(lighting, "core", fake_core)
        monkeypatch.setattr(lighting, "CFG", _FakeCFG(bilinear))
        return fake_rl
    return _make


def test_bilinear_true_binds_flag_and_applies_no_point_filter(faked):
    fake_rl = faked(True)
    lp = lighting.LightingPass(grid_h=4, grid_w=4)

    assert lp.bilinear is True
    # Default creation is already bilinear (renderer/core.py); true means no
    # re-filter to POINT is issued for either light texture.
    point_calls = [c for c in fake_rl.filter_calls
                   if c[0] in (lp.light_tex_a, lp.light_tex_b)]
    assert point_calls == []


def test_bilinear_false_binds_flag_and_repoints_both_light_textures(faked):
    fake_rl = faked(False)
    lp = lighting.LightingPass(grid_h=4, grid_w=4)

    assert lp.bilinear is False
    point = lighting.rl.TextureFilter.TEXTURE_FILTER_POINT
    a_calls = [c for c in fake_rl.filter_calls if c[0] is lp.light_tex_a]
    b_calls = [c for c in fake_rl.filter_calls if c[0] is lp.light_tex_b]
    assert a_calls and a_calls[-1] == (lp.light_tex_a, point)
    assert b_calls and b_calls[-1] == (lp.light_tex_b, point)


def test_toggle_bilinear_method_is_gone():
    """The live B-key toggle method is deleted outright (#59), not merely
    disconnected -- nothing else in the repo calls it (grepped clean)."""
    assert not hasattr(lighting.LightingPass, "toggle_bilinear")
