"""renderer/static_props.py gate (props & vegetation arc #60 P2).

Everything here is HEADLESS — shader SOURCE composition, the cache key, and
the metres->world-px scale contract. The GL-dependent half (mesh upload,
model cache, unload) is exercised by ``tools/lighting_demo.py --props``
(the patch's HUMAN-TEST vehicle) and by its ``--auto`` boot smoke run, which
builds, draws and unloads the whole garden.
"""
from __future__ import annotations

import pytest

from renderer import lit3d
from renderer import static_props as sp


# ---------------------------------------------------------------------------
# The shared lit3d seam is REUSED, not forked (design §4.3 F1)
# ---------------------------------------------------------------------------

def test_prop_shader_uses_the_lit3d_common_glsl():
    assert lit3d._COMMON_GLSL in sp.PROP_FS, (
        "the prop shader must concatenate lit3d's shared sRGB/ACES block — a "
        "second copy would drift from shaders/lighting.fs")


def test_prop_shader_uses_the_lit3d_field_sample_block():
    """The light-field sample/unpack + L/N setup is the SAME text the marine
    compiles (extracted into lit3d at P2). If a future edit forks it, the
    marine and the props stop agreeing about what 'lit' means."""
    assert lit3d._FIELD_SAMPLE_GLSL in sp.PROP_FS


def test_prop_shader_declares_every_uniform_the_shared_block_reads():
    for name in ("u_world_px", "u_light_z", "texture1", "texture2"):
        assert name in sp.PROP_FS
    for name in ("fragWorldPos", "fragWorldNormal"):
        assert name in sp.PROP_FS and name in sp.PROP_VS


def test_prop_shader_takes_albedo_from_vertex_colors():
    assert "in vec4 vertexColor;" in sp.PROP_VS
    assert "vec3 albedo = fragColor.rgb;" in sp.PROP_FS
    assert "texture0" not in sp.PROP_FS, "props carry no albedo texture"


def test_prop_shader_never_uses_vertex_alpha_as_opacity():
    """Vertex-color alpha is the P4 wind-flutter WEIGHT. Output alpha must be
    a hard 1.0 — the world RT is blitted premultiplied, so a translucent prop
    would bleed the background through."""
    assert "finalColor = vec4(lit, 1.0);" in sp.PROP_FS
    assert "fragColor.a" not in sp.PROP_FS


def test_prop_shader_carries_no_sway_uniforms_yet():
    """Sway is P4. A stray sway uniform here would be a half-built feature the
    demo silently leaves at zero."""
    for name in ("u_sway", "u_wind", "u_time", "u_phase"):
        assert name not in sp.PROP_VS


# ---------------------------------------------------------------------------
# Cache key — one entry per distinct LOOK
# ---------------------------------------------------------------------------

def _p(**kw):
    return sp.PropPlacement(x_wpx=0.0, y_wpx=0.0, **kw)


def test_same_look_different_position_shares_a_cache_entry():
    a = sp.PropPlacement(x_wpx=10.0, y_wpx=20.0, seed=3)
    b = sp.PropPlacement(x_wpx=999.0, y_wpx=1.0, seed=3)
    assert a.cache_key() == b.cache_key()


def test_yaw_and_tint_are_per_placement_not_per_model():
    a = sp.PropPlacement(x_wpx=0.0, y_wpx=0.0, seed=3, yaw_deg=0.0)
    b = sp.PropPlacement(x_wpx=0.0, y_wpx=0.0, seed=3, yaw_deg=137.0,
                         tint=(200, 200, 255, 255))
    assert a.cache_key() == b.cache_key()


@pytest.mark.parametrize("field,value", [
    ("generator", "palm"), ("seed", 4), ("palette", "exotic"),
    ("style", "faceted"), ("decor", "fruit"), ("decor_density", 0.5),
    ("height_m", 3.0),
])
def test_every_look_field_changes_the_cache_key(field, value):
    base = _p(seed=3)
    other = sp.PropPlacement(x_wpx=0.0, y_wpx=0.0,
                             **{**{"seed": 3}, field: value})
    assert base.cache_key() != other.cache_key()


def test_height_is_bucketed_so_hand_typed_values_do_not_split_the_cache():
    a = _p(seed=3, height_m=2.23)
    b = _p(seed=3, height_m=2.25)
    assert a.cache_key() == b.cache_key()
    # But a genuinely different size still gets its own model: the generator's
    # tuft density is absolute, so a scaled-up mesh is NOT the same tree.
    assert _p(seed=3, height_m=2.2).cache_key() != \
        _p(seed=3, height_m=2.6).cache_key()


def test_empty_and_none_decor_are_the_same_look():
    a = sp.PropPlacement(x_wpx=0.0, y_wpx=0.0, seed=3, decor="")
    assert a.cache_key()[4] == ""


# ---------------------------------------------------------------------------
# Scale contract: metres -> world pixels via the tile size
# ---------------------------------------------------------------------------

def test_px_per_m_is_world_px_per_tile_over_tile_size_m():
    r = sp.StaticPropRenderer(world_px_per_tile=48.0, tile_size_m=0.333)
    assert r.px_per_m == pytest.approx(48.0 / 0.333)
    # A 3 m tree on a 0.333 m/tile level is ~9 tiles of world pixels tall.
    assert 3.0 * r.px_per_m / 48.0 == pytest.approx(9.009, abs=1e-2)


def test_scale_contract_is_resolution_independent():
    """Doubling world_px_per_tile doubles the pixel height of the same prop —
    the prop stays the same SIZE IN TILES, which is what a level author means
    by 'a 3 m tree'."""
    a = sp.StaticPropRenderer(48.0, 0.333)
    b = sp.StaticPropRenderer(96.0, 0.333)
    assert b.px_per_m == pytest.approx(2.0 * a.px_per_m)


def test_renderer_is_inert_before_load():
    r = sp.StaticPropRenderer(48.0, 0.333)
    assert not r.ready
    # draw_props must no-op (never raise) when the shader is not compiled —
    # a failed shader must not take the ship's render down with it.
    r.draw_props([_p(seed=1)], camera3d=None, ctx=None)
