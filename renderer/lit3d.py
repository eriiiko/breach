"""lit3d — the shared "lit 3D in the world RT" seam.

Everything drawn as real 3D geometry inside the 2D world render target (today:
the marine's skinned glTF body; next: static props/vegetation; later: 3D
walls) shares three things, extracted here so a second 3D consumer never
forks them:

  * ``_COMMON_GLSL`` — the sRGB DECODE GLSL helper, string-concatenated into
    a consumer's fragment shader. Kept numerically identical to
    ``shaders/lighting.fs`` (which keeps its OWN inline copy) so every
    3D-lit thing decodes its albedo exactly as the 2D ship does. Since
    graphics arc G0 P1 a 3D consumer writes LINEAR, un-tone-mapped HDR into
    the RGBA16F scene RT, like every world layer: the ONE tone-map + encode
    runs at ``WorldComposite.resolve()`` (``shaders/world_resolve.fs``), so
    this block carries no tone-map and no encode -- a consumer that adds
    either is the bug (``tests/test_one_tonemap.py``).
  * ``LightFieldCtx`` — the ship's baked light field (``light_tex_a`` /
    ``light_tex_b`` + world dims + ambient/gain), built once per frame by
    ``game_renderer`` from its ``LightingPass`` + ``WorldComposite`` (single
    source of truth) and handed to every 3D draw call so each consumer
    samples EXACTLY what the ship samples.
  * ``make_camera`` — the top-down orthographic ``Camera3D`` factory, built
    UNCONDITIONALLY (not gated on any per-consumer toggle) so a second
    consumer never needs its own camera.

Coordinate mapping (calibrated in prototypes/scratchpad, see
``calib_camera.py``):
  * The world render target is ``world_px_w × world_px_h`` world-pixels, drawn
    top-left origin, y-down. A top-down ORTHOGRAPHIC ``Camera3D`` measured in
    world-pixels maps 3D X = x_wpx, 3D Z = y_wpx, 3D Y = height-up, with
    ``fovy = world_px_h`` and camera up = (0, 0, -1). Verified: 3D primitives
    land exactly on the same-coordinate 2D draws (calib_camera.py).
  * The RT's own depth buffer occludes correctly (verified: a near model drawn
    FIRST fully occludes a far model drawn SECOND), so NO world-Y painter's
    fallback is needed.

Extracted 2026-09 (props & vegetation arc #60, P1) from
``renderer/marine_shader.py`` (``_COMMON_GLSL``) and
``renderer/unit_model_renderer.py`` (``LightFieldCtx``, ``make_camera``,
``_CAM_HEIGHT``) — pure move, no behavior change. See
``docs/architecture/graphics/props_and_vegetation.md`` §4.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyray as rl

# Shared helper — string-concatenated into a consumer's fragment shader. Kept
# numerically identical to shaders/lighting.fs so every 3D-lit thing decodes
# its albedo as the ship does. No tone-map, no encode: the world resolve owns
# both (G0 P1).
# The SMOKE CAST SHADOW (graphics G2, docs/graphics_direction_2026-10-18.md
# §2.3, §12): THE one copy of the GLSL every receiver of the smoke's shadow
# runs -- the floor (shaders/lighting.fs, spliced in at its
# "// <SMOKE_SHADOW>" marker by LightingPass) and every lit-3D consumer (via
# _COMMON_GLSL below, called from _FIELD_SAMPLE_GLSL). It reads the relief's
# CAST PRODUCT (renderer/gas_relief.py ``GasReliefPass.product_texture``:
# r = ceiling-layer alpha, g = its relief height, b/a = the floor layer's,
# heights in tiles), never a second smoke model.
#
# Geometry: a receiver at height z (tiles) looks toward the light along the
# light field's own planar direction (``light_dir_2d``, toward the light),
# rising one tile per ``parallax`` tiles of run. The CEILING layer is a slab
# from ``lift`` up to ``lift + h(q)``; the FLOOR layer a slab from 0 up to
# h(q). Three taps across each slab's window of run find where the ray is
# inside it; the shadow is the strongest occlusion alpha met there, and the
# light is multiplied by (1 - strength * shadow) per layer. The planar light
# sweep already attenuates light THROUGH smoke in the plane; this adds the
# vertical shadow it cannot know about (the column's shape thrown across the
# floor and over the squad). Strengths 0 (relief off, smoke hidden) return
# 1 before any texture read.
_SMOKE_SHADOW_GLSL = """
// --- smoke cast shadow (graphics G2; renderer/lit3d.py _SMOKE_SHADOW_GLSL) ---
uniform sampler2D u_smoke_cover; // the relief's cast product: r,g = ceiling
                                 // (alpha, height), b,a = floor; tiles
uniform vec2  u_smoke_grid;      // (grid_w, grid_h), tiles
uniform vec4  u_smoke_cast;      // (ceiling strength, floor strength,
                                 //  parallax: run per tile of rise, ceiling lift)
uniform vec4  u_smoke_cast2;     // (tallest column, tiles; unused x3)
vec4 smokeCoverAt(vec2 world_uv) {
    // a RenderTexture's rows run bottom-up
    return texture(u_smoke_cover, vec2(world_uv.x, 1.0 - world_uv.y));
}
float smokeShadow(vec2 world_uv, vec2 to_light, float z_tiles) {
    if (u_smoke_cast.x + u_smoke_cast.y <= 0.0) return 1.0;
    vec2 step_uv = to_light / u_smoke_grid;      // world uv per tile of run
    float k = u_smoke_cast.z;
    float hmax = max(u_smoke_cast2.x, 1e-3);
    float shC = 0.0;
    float shF = 0.0;
    for (int i = 0; i < 3; ++i) {
        float f = (float(i) + 0.5) / 3.0;
        // ceiling slab [lift, lift + h]: the ray is in it for run s with
        // z + s / k in that band
        float sC = (max(u_smoke_cast.w - z_tiles, 0.0) + f * hmax) * k;
        vec4 c = smokeCoverAt(world_uv + step_uv * sC);
        float topC = u_smoke_cast.w + c.g - (z_tiles + sC / max(k, 1e-3));
        shC = max(shC, c.r * smoothstep(0.0, 0.25 * hmax, topC));
        // floor slab [0, h]
        float sF = f * max(hmax - z_tiles, 0.0) * k;
        vec4 q = smokeCoverAt(world_uv + step_uv * sF);
        float topF = q.a - (z_tiles + sF / max(k, 1e-3));
        shF = max(shF, q.b * smoothstep(0.0, 0.25 * hmax, topF));
    }
    return (1.0 - u_smoke_cast.x * shC) * (1.0 - u_smoke_cast.y * shF);
}
"""

_COMMON_GLSL = """
// Cheap sRGB -> linear (gamma 2.2), matching shaders/lighting.fs. Output stays
// linear HDR: WorldComposite.resolve() tone-maps and encodes, once.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
""" + _SMOKE_SHADOW_GLSL


# The light-field SAMPLE + unpack block, shared verbatim by every 3D consumer's
# fragment shader (marine, props, future walls). It reads ``fragWorldPos`` /
# ``fragWorldNormal`` and the uniforms ``u_world_px`` / ``u_light_z`` /
# ``texture1`` (light_tex_a) / ``texture2`` (light_tex_b), and defines
# ``world_uv``, ``tex_a``, ``tex_b``, ``incoming_rgb``, ``light_dir_2d``, the
# 3D light direction ``L`` and the shading normal ``N`` — the contract every
# consumer's main() continues from.
#
# The wording is the FIRST consumer's (the marine), preserved BYTE-FOR-BYTE:
# ``tests/test_lit3d_extraction.py`` gates ``MARINE_FS`` against a pre-
# refactor oracle, so this block may not be re-worded without re-baselining
# that oracle. It applies verbatim to any Y-up mesh drawn in the world RT —
# read "the marine" as "this mesh".
_FIELD_SAMPLE_GLSL = """\
    // Foot-plane world UV: sample the baked field at the marine's XZ ground
    // position (height Y ignored). worldPos.x -> grid X, worldPos.z -> grid Y
    // (y-down). NO v-flip: the marine reads the same texture the ship reads
    // with the same world->grid mapping (numpy row 0 = grid y 0 = texture v 0
    // = screen top under both the RT quad and the top-down 3D camera), so the
    // marine's lighting is glued to the same tiles as the ship beside it.
    vec2 world_uv = fragWorldPos.xz / u_world_px;

    vec4 tex_a = texture(texture1, world_uv);
    vec4 tex_b = texture(texture2, world_uv);
    vec3 incoming_rgb = tex_a.rgb;              // total light colour at this tile
    vec2 light_dir_2d = vec2(tex_a.a, tex_b.a); // signed, already unit-length

    // 2D baked direction -> 3D in the marine's Y-up world frame: dir.x -> X,
    // u_light_z -> Y (up), dir.y -> Z. Same vector the ship builds as
    // vec3(dir, light_z), reordered because the ship's tangent frame is Z-up.
    vec3 L = normalize(vec3(light_dir_2d.x, u_light_z, light_dir_2d.y));
    vec3 N = normalize(fragWorldNormal);

    // Graphics G2: the smoke's cast shadow on this fragment, at its own
    // height (world px -> tiles), through the one shared GLSL above.
    incoming_rgb *= smokeShadow(world_uv, light_dir_2d,
                                fragWorldPos.y * u_smoke_grid.x / u_world_px.x);"""


@dataclass
class LightFieldCtx:
    """The ship's baked light field + scalars, handed to a 3D draw call so its
    shader samples EXACTLY what the ship samples. Built by game_renderer from
    its LightingPass + WorldComposite (single source of truth). ``tex_a`` /
    ``tex_b`` are the ``light_tex_a`` / ``light_tex_b`` RGBA16F textures."""
    tex_a: object
    tex_b: object
    world_px_w: float
    world_px_h: float
    ambient: tuple
    light_gain: float
    normal_y_sign: float = 1.0
    # Graphics G2: the smoke cast shadow -- the relief's cast product
    # (GasReliefPass.product_texture; None = no shadow), the grid in tiles
    # and gas_relief.cast_uniforms' two vec4s. Strength 0 = no shadow.
    smoke_cover: object = None
    smoke_grid: tuple = (1.0, 1.0)
    smoke_cast: tuple = (0.0, 0.0, 0.0, 0.0)
    smoke_cast2: tuple = (0.0, 0.0, 0.0, 0.0)


def set_smoke_shadow_uniforms(shader, grid, cast, cast2) -> None:
    """Push the cast-shadow uniforms of :data:`_SMOKE_SHADOW_GLSL` to
    ``shader`` (safe outside begin_shader_mode: SetShaderValue self-enables
    the program). The sampler ``u_smoke_cover`` is the caller's to bind (a
    3D consumer through a material map, the floor on its own unit). A
    location of -1 (the program compiled the block out) is skipped."""
    for name, vals in (("u_smoke_grid", grid), ("u_smoke_cast", cast),
                       ("u_smoke_cast2", cast2)):
        loc = rl.get_shader_location(shader, name)
        if loc < 0:
            continue
        n = len(vals)
        kind = {2: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2,
                4: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4}[n]
        rl.set_shader_value(shader, loc,
                            rl.ffi.new(f"float[{n}]", [float(v) for v in vals]),
                            kind)


def bind_smoke_cover_map(shader, materials, ctx: "LightFieldCtx") -> None:
    """Wire the cast product into a lit-3D consumer: its sampler
    ``u_smoke_cover`` on the OCCLUSION material-map slot (unit 4, a free
    slot: albedo 0, light textures 1/2, normal map 3), so DrawMesh binds it
    every draw like the light textures, plus the per-frame uniforms. With
    no product (``ctx.smoke_cover`` None) the strengths go to 0 and nothing
    is sampled."""
    MM = rl.MaterialMapIndex
    shader.locs[rl.ShaderLocationIndex.SHADER_LOC_MAP_OCCLUSION] = \
        rl.get_shader_location(shader, "u_smoke_cover")
    cast = ctx.smoke_cast if ctx.smoke_cover is not None else (
        0.0, 0.0, 0.0, 0.0)
    if ctx.smoke_cover is not None:
        for mat in materials:
            mat.maps[MM.MATERIAL_MAP_OCCLUSION].texture = ctx.smoke_cover
    set_smoke_shadow_uniforms(shader, ctx.smoke_grid, cast, ctx.smoke_cast2)


# Top-down camera height above the floor, in world px. Ortho => this does NOT
# affect on-screen size, only near/far framing. It MUST stay below raylib's
# orthographic far-clip (empirically < ~5000 in this build), so it CANNOT scale
# with world size: a tall level (RT up to 5760 px) pushed the old
# max(w,h)*2 height past the far plane and culled every model (the "press M and
# everything vanishes" bug). 500 clears the tallest model (~3*wpt) with margin
# and is safely inside the far plane at every level size (verified 2400x5760).
_CAM_HEIGHT = 500.0


def make_camera(world_px_w: int, world_px_h: int) -> rl.Camera3D:
    """Top-down orthographic Camera3D framed to the world RT in world-px.

    3D X = x_wpx, 3D Z = y_wpx, 3D Y = up. ``fovy = world_px_h`` makes the
    ortho view span exactly the RT (aspect = w/h fills the width), so 3D
    world-px coords land on the same texels as the 2D world-px draws.
    Camera up = (0,0,-1) so world +Z (screen-down, i.e. y-down) reads down.
    """
    cam = rl.Camera3D()
    cx, cy = world_px_w / 2.0, world_px_h / 2.0
    # Fixed height (NOT world-size-scaled): ortho => distance doesn't change
    # on-screen size, only near/far framing, and it must stay under raylib's
    # ortho far-clip. See _CAM_HEIGHT (the old max(w,h)*2 culled everything on
    # tall levels).
    cam.position = rl.Vector3(cx, _CAM_HEIGHT, cy)
    cam.target = rl.Vector3(cx, 0.0, cy)
    cam.up = rl.Vector3(0.0, 0.0, -1.0)
    cam.fovy = float(world_px_h)
    cam.projection = rl.CameraProjection.CAMERA_ORTHOGRAPHIC
    return cam


__all__ = ["LightFieldCtx", "make_camera", "_COMMON_GLSL",
           "_FIELD_SAMPLE_GLSL", "_SMOKE_SHADOW_GLSL", "_CAM_HEIGHT",
           "set_smoke_shadow_uniforms", "bind_smoke_cover_map"]
