"""P1 extraction gate (props & vegetation arc #60): renderer/lit3d.py takes
the shared light-field GLSL, LightFieldCtx, and the top-down camera factory
out of renderer/marine_shader.py + renderer/unit_model_renderer.py.

This is a PURE MOVE — no behavior change. The gate:
  1. The marine's composed shader source (MARINE_VS / MARINE_FS) is
     byte-for-byte identical to the pre-refactor oracle fixtures captured in
     tests/_lit3d_marine_shader_golden/ (see that capture's docstring: the
     sources are plain module-level string constants, no GL context needed
     to compose them).
  2. renderer.lit3d exposes LightFieldCtx, make_camera, and the shared GLSL
     block.
  3. renderer.unit_model_renderer's public names still resolve unchanged
     (import-compat): LightFieldCtx, UnitModelRenderer.make_camera.
"""
from __future__ import annotations

from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "_lit3d_marine_shader_golden"


def _read_golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def test_marine_shader_byte_identical_to_oracle():
    from renderer import marine_shader as ms

    assert ms.MARINE_VS == _read_golden("marine.vs"), (
        "MARINE_VS changed after the lit3d extraction — the refactor must be "
        "behavior-preserving (pure move, no source-text change)."
    )
    assert ms.MARINE_FS == _read_golden("marine.fs"), (
        "MARINE_FS changed after the lit3d extraction — the refactor must be "
        "behavior-preserving (pure move, no source-text change)."
    )


def test_lit3d_exposes_shared_seam_members():
    from renderer import lit3d

    assert hasattr(lit3d, "LightFieldCtx")
    assert hasattr(lit3d, "make_camera")
    assert hasattr(lit3d, "_COMMON_GLSL")
    assert callable(lit3d.make_camera)
    assert isinstance(lit3d._COMMON_GLSL, str) and "aces_tonemap" in lit3d._COMMON_GLSL


def test_marine_shader_common_glsl_is_the_lit3d_instance():
    """marine_shader.py must MOVE (not copy) _COMMON_GLSL — same object, not a
    second definition that could drift from shaders/lighting.fs independently."""
    from renderer import lit3d
    from renderer import marine_shader as ms

    assert ms._COMMON_GLSL is lit3d._COMMON_GLSL


def test_unit_model_renderer_import_compat():
    """Existing importers (game_renderer.py: `from renderer.unit_model_renderer
    import LightFieldCtx`, `UnitModelRenderer.make_camera(...)`) must keep
    working unmodified after the extraction."""
    from renderer import lit3d
    from renderer.unit_model_renderer import LightFieldCtx, UnitModelRenderer

    # Re-exported name resolves to the SAME class lit3d defines (not a copy).
    assert LightFieldCtx is lit3d.LightFieldCtx

    # The staticmethod still calls straight through to lit3d.make_camera.
    assert UnitModelRenderer.make_camera is lit3d.make_camera

    cam = UnitModelRenderer.make_camera(640, 480)
    assert cam.fovy == 480.0


def test_light_field_ctx_constructible():
    """Sanity: the moved dataclass still constructs with the fields
    game_renderer.py passes (tex_a/tex_b/world_px_w/world_px_h/ambient/
    light_gain/normal_y_sign)."""
    from renderer.lit3d import LightFieldCtx

    ctx = LightFieldCtx(tex_a=None, tex_b=None, world_px_w=640.0,
                        world_px_h=480.0, ambient=(0.1, 0.1, 0.1),
                        light_gain=1.5)
    assert ctx.normal_y_sign == 1.0
