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

# Graphics G5 (#64): the light curve, THE one copy every lit surface runs (the
# floor, spliced in at lighting.fs's marker; the marine and props through
# _COMMON_GLSL). Spec and twin: renderer/light_curve.py (``curve_gain``).
# ``lightCurveGain`` is the factor on a pixel's light, in exposure units,
# before albedo and N.L; u_curve.x 0 returns exactly 1 and the light is
# untouched. A uniform array, not a texture: 3D consumers have no free
# material slot, and the map is 256 floats.
_LIGHT_CURVE_GLSL = """
uniform float u_curve_lut[256];  // the curve, y at 256 even x
uniform vec4  u_curve;           // (on, range, dark saturation, dark knee)

float lightCurveLum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

float lightCurveGain(vec3 light) {
    if (u_curve.x < 0.5) return 1.0;
    float L = lightCurveLum(max(light, vec3(0.0)));
    if (L <= 1e-6) return 1.0;
    float u = sqrt(min(L / u_curve.y, 1.0)) * 255.0;
    int i = min(int(u), 254);
    float y = mix(u_curve_lut[i], u_curve_lut[i + 1], u - float(i));
    return y * y * u_curve.y * max(L / u_curve.y, 1.0) / L;
}

// The art's colour in the dark: grey toward u_curve.z where the curved light
// is below the knee (1 = off). The prototype's dark state (#64) without its
// brightness, which is the curve's.
vec3 lightCurveAlbedo(vec3 albedo, vec3 curved_light) {
    if (u_curve.x < 0.5 || u_curve.z >= 1.0) return albedo;
    float y = sqrt(min(lightCurveLum(max(curved_light, vec3(0.0))) / u_curve.y,
                       1.0));
    float sat = mix(u_curve.z, 1.0, smoothstep(0.0, u_curve.w, y));
    return mix(vec3(lightCurveLum(albedo)), albedo, sat);
}
"""

# Water look arc #97 W3 (critique M12): THE WATERLINE, the one copy every
# lit-3D body that stands in water runs (the units and what they hold, the
# props; called at the end of their main()). Spec and twin:
# renderer/water_surface.py ``waterline_tint``. A fragment at WORLD height y
# (world px over the floor, / u_waterline.y px per metre: the consumer's own
# drawn scale) below the water standing ``depth`` m deep on its tile (the
# water pass's own per-tile texture, B = depth, read at the fragment's foot
# point as the water pass reads it) is seen through the column above it by
# the water pass's own law for the floor: the Beer-Lambert fade toward
# water_color and the alpha ramp, with the same [graphics.water] dials. In
# world height, per fragment, so it holds when the camera tilts (Erik,
# 2026-11-20). u_waterline.x 0 (no water: the default of an unset uniform),
# a dry tile or a fragment above the surface return ``lit`` untouched.
_WATERLINE_GLSL = """
uniform sampler2D u_waterline_tex; // the water pass's u_water: B = depth, m
uniform vec4  u_waterline;   // (strength, px per metre, fog_density, height_edge)
uniform vec4  u_waterline2;  // (alpha_scale, alpha_min, alpha_max, 0)
uniform vec3  u_waterline_color;  // water_color: the volume colour at depth
vec3 waterlineTint(vec3 lit, vec3 world_pos, vec2 world_uv) {
    if (u_waterline.x <= 0.0) return lit;
    float depth = texture(u_waterline_tex, world_uv).b;
    float col = depth - world_pos.y / u_waterline.y;
    if (col <= 0.0) return lit;
    float fog = exp2(-u_waterline.z * col);
    float a = clamp(col * u_waterline2.x, u_waterline2.y, u_waterline2.z)
            * smoothstep(0.0, max(u_waterline.w, 1e-4), col) * u_waterline.x;
    vec3 vol = u_waterline_color + (lit - u_waterline_color) * fog;
    return lit + (vol - lit) * a;
}
"""

_COMMON_GLSL = """
// Cheap sRGB -> linear (gamma 2.2), matching shaders/lighting.fs. Output stays
// linear HDR: WorldComposite.resolve() tone-maps and encodes, once.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
""" + _SMOKE_SHADOW_GLSL + _LIGHT_CURVE_GLSL + _WATERLINE_GLSL


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
    # Graphics G5: the light curve's uniforms (LightingPass.curve_uniforms);
    # None = off.
    curve: object = None
    # #97 W3: the waterline (WaterPass.waterline_inputs: the water texture
    # and the water's own dials); None = no water on the map, the waterline
    # is off and nothing is sampled.
    waterline: object = None


def set_light_curve_uniforms(shader, curve) -> None:
    """Push :data:`_LIGHT_CURVE_GLSL`'s uniforms to ``shader``. ``curve`` is
    ``(lut, vec4)``: a cffi ``float[256]`` and the four floats, or None for
    off. A location of -1 (compiled out) is skipped."""
    vec = (0.0, 1.0, 1.0, 1.0) if curve is None else curve[1]
    loc = rl.get_shader_location(shader, "u_curve")
    if loc >= 0:
        rl.set_shader_value(shader, loc, rl.ffi.new("float[4]", list(vec)),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)
    loc = rl.get_shader_location(shader, "u_curve_lut")
    if loc >= 0 and curve is not None:
        rl.set_shader_value_v(shader, loc, curve[0],
                              rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
                              256)


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


def bind_waterline_map(shader, materials, ctx: "LightFieldCtx",
                       px_per_m: float) -> None:
    """Wire the waterline (:data:`_WATERLINE_GLSL`, #97 W3) into a lit-3D
    consumer: the water pass's per-tile texture on the EMISSION material-map
    slot (unit 5, a free slot: albedo 0, light textures 1/2, normal map 3,
    smoke cover 4), so DrawMesh binds it every draw, plus the uniforms --
    ``px_per_m`` the consumer's own drawn scale (world px per metre of its
    model). With no water (``ctx.waterline`` None) the strength goes to 0
    and nothing is sampled: the fragment is untouched, bit for bit."""
    MM = rl.MaterialMapIndex
    shader.locs[rl.ShaderLocationIndex.SHADER_LOC_MAP_EMISSION] =         rl.get_shader_location(shader, "u_waterline_tex")
    wl = ctx.waterline
    if wl is None:
        vals = (("u_waterline", (0.0, 1.0, 0.0, 0.0)),)
    else:
        tex, a, b, color = wl
        for mat in materials:
            mat.maps[MM.MATERIAL_MAP_EMISSION].texture = tex
        vals = (("u_waterline", (a[0], float(px_per_m), a[1], a[2])),
                ("u_waterline2", tuple(b)),
                ("u_waterline_color", tuple(color)))
    kinds = {3: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3,
             4: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4}
    for name, v in vals:
        loc = rl.get_shader_location(shader, name)
        if loc >= 0:
            rl.set_shader_value(shader, loc,
                                rl.ffi.new(f"float[{len(v)}]",
                                           [float(x) for x in v]),
                                kinds[len(v)])


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
           "set_smoke_shadow_uniforms", "bind_smoke_cover_map",
           "_LIGHT_CURVE_GLSL", "set_light_curve_uniforms",
           "_WATERLINE_GLSL", "bind_waterline_map"]
