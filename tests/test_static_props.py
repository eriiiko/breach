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


# ---------------------------------------------------------------------------
# P4 sway — the shader terms, the dials, and the TAMED-wind seam
# ---------------------------------------------------------------------------

def test_prop_vertex_shader_declares_the_sway_uniforms():
    """The sway lives in the VERTEX shader (a displacement, not a texture
    trick), and every dial it obeys is a uniform — nothing baked into GLSL, so
    Erik's tuning pass is config-only."""
    for name in ("u_time", "u_wind", "u_phase", "u_sway", "u_height",
                 "u_flutter", "u_gust_speed", "u_gust_depth",
                 "u_flutter_speed"):
        assert f"uniform float {name};" in sp.PROP_VS or \
               f"uniform vec3  {name};" in sp.PROP_VS, name


def test_sway_uniforms_are_all_looked_up_by_the_renderer():
    """A uniform the shader declares but the renderer never locates is a dial
    that silently does nothing."""
    import re
    declared = set(re.findall(r"uniform \w+\s+(u_\w+);", sp.PROP_VS))
    src = sp.__file__
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    for name in declared:
        assert f'"{name}"' in text, f"{name} is never looked up / set"


def test_sway_bends_by_height_squared_and_uses_alpha_as_flutter_weight():
    """The two load-bearing lines of the ported spike math: trunks stiff /
    crowns bending (h^2), and vertex ALPHA as the flutter WEIGHT — never as
    opacity, which the fragment shader still forces to 1.0."""
    assert "float bend = hn * hn;" in sp.PROP_VS
    assert "vertexColor.a * length(u_wind)" in sp.PROP_VS
    # The fragment shader still never reads alpha, and still writes opaque.
    assert "fragColor.a" not in sp.PROP_FS
    assert "finalColor = vec4(lit, 1.0);" in sp.PROP_FS


def test_sway_settings_defaults_are_a_gentle_visible_motion():
    s = sp.SwaySettings()
    # Crown offset is a FRACTION OF HEIGHT — a few percent reads as wind, tens
    # of percent reads as rubber. Guard the range Erik tunes inside.
    assert 0.0 < s.strength <= 0.15
    assert 0.0 <= s.flutter <= 1.0
    assert 0.0 <= s.idle_wind <= 0.5
    # wind_ref IS gas_detail's saturation ceiling: the largest value tame_wind
    # can return maps to full sway, so the normalized fraction lands in 0..1.
    from renderer.gas_detail import WIND_V_REF
    assert s.wind_ref == WIND_V_REF


def test_sway_settings_from_config_reads_the_render_props_section():
    """The dials come from [render.props], re-read every frame (Ctrl+R)."""
    from config import CFG
    s = sp.SwaySettings.from_config(CFG)
    assert s.strength == pytest.approx(
        float(CFG.render.props.sway_strength))
    assert s.flutter == pytest.approx(
        float(CFG.render.props.flutter_strength))
    # A config with no [render.props] at all still yields the shipped feel.
    fallback = sp.SwaySettings.from_config(object())
    assert fallback == sp.SwaySettings()


def test_sway_strength_zero_disables_sway():
    r = sp.StaticPropRenderer(48.0, 0.333)
    r.sway = sp.SwaySettings(strength=0.0)
    assert r.sway.strength == 0.0
    # model_wind is never consulted when the dial is off (draw_props passes
    # sway_on=False), but even called directly it produces no displacement.
    assert r.model_wind(_p(seed=1), 2.2, None) == (0.0, 0.0)


def test_wind_sampling_is_one_clamped_nearest_tile_lookup():
    import numpy as np
    r = sp.StaticPropRenderer(48.0, 0.333)
    field = np.zeros((4, 5, 2), dtype=np.float32)
    field[2, 3] = (0.05, -0.02)
    # tile (3, 2) -> world px (3*48 .. 4*48, 2*48 .. 3*48)
    assert r.sample_wind(field, 3 * 48.0 + 7.0, 2 * 48.0 + 1.0) == \
        pytest.approx((0.05, -0.02))
    # Out of bounds clamps rather than wrapping or raising.
    assert r.sample_wind(field, -900.0, 1e6) == (0.0, 0.0)
    # No field at all = dead calm (never a crash).
    assert r.sample_wind(None, 10.0, 10.0) == (0.0, 0.0)


def test_sway_amplitude_is_a_fraction_of_the_props_own_height():
    """A shrub and a palm lean by the same visual proportion — the shader's
    displacement is in model units, so the Python side scales by the mesh's
    native height."""
    import numpy as np
    r = sp.StaticPropRenderer(48.0, 0.333)
    r.sway = sp.SwaySettings(strength=0.10, idle_wind=0.0)
    field = np.zeros((2, 2, 2), dtype=np.float32)
    field[0, 0] = (r.sway.wind_ref, 0.0)          # full wind, +X
    small = r.model_wind(_p(seed=1), 1.0, field)
    big = r.model_wind(_p(seed=1), 4.0, field)
    assert small[0] == pytest.approx(0.10)        # 10% of a 1-unit mesh
    assert big[0] == pytest.approx(0.40)          # ... and of a 4-unit one
    assert small[1] == pytest.approx(0.0)


def test_idle_wind_keeps_still_air_breathing_and_zero_turns_it_off():
    r = sp.StaticPropRenderer(48.0, 0.333)
    r.sway = sp.SwaySettings(strength=0.06, idle_wind=0.2)
    wx, wz = r.model_wind(_p(seed=1), 2.0, None)   # dead calm
    assert (wx * wx + wz * wz) ** 0.5 == pytest.approx(0.06 * 2.0 * 0.2)
    r.sway = sp.SwaySettings(strength=0.06, idle_wind=0.0)
    assert r.model_wind(_p(seed=1), 2.0, None) == pytest.approx((0.0, 0.0))


def test_wind_is_rotated_into_the_props_own_frame():
    """Sway displaces in MODEL space (before the yaw), so the world wind must
    be counter-rotated or a yawed tree bends the wrong way."""
    import numpy as np
    r = sp.StaticPropRenderer(48.0, 0.333)
    r.sway = sp.SwaySettings(strength=0.1, idle_wind=0.0)
    field = np.zeros((2, 2, 2), dtype=np.float32)
    field[0, 0] = (r.sway.wind_ref, 0.0)           # world +X
    straight = r.model_wind(_p(seed=1), 2.0, field)
    turned = r.model_wind(_p(seed=1, yaw_deg=90.0), 2.0, field)
    # Same magnitude, rotated by 90 degrees in the model frame.
    assert (turned[0] ** 2 + turned[1] ** 2) ** 0.5 == \
        pytest.approx((straight[0] ** 2 + straight[1] ** 2) ** 0.5)
    assert turned[0] == pytest.approx(0.0, abs=1e-9)


def test_prop_phase_desyncs_neighbours_and_is_deterministic():
    a = sp.PropPlacement(x_wpx=100.0, y_wpx=200.0, seed=1)
    b = sp.PropPlacement(x_wpx=148.0, y_wpx=200.0, seed=2)
    assert sp.StaticPropRenderer.prop_phase(a) != \
        sp.StaticPropRenderer.prop_phase(b)
    assert sp.StaticPropRenderer.prop_phase(a) == \
        sp.StaticPropRenderer.prop_phase(a)


def test_pack_models_draw_rigid():
    """Design §4.3: 'Pack models draw with u_sway = 0' — their geometry has no
    baked flutter weights."""
    class _E:
        class_name = "prop"
        def __init__(self, **f):
            self.fields = f
    gen, mod = sp.placements_from_entities(
        [_E(x=1, y=1, kind="generated"), _E(x=2, y=2, kind="model")], 48.0)
    assert gen.sway == 1.0 and mod.sway == 0.0


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
