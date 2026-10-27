"""GLSL + loader for the lit 3D-marine material shader (marine lit-shader arc).

The marine samples the SAME baked light field the 2D ship is lit by
(``LightingPass.light_tex_a`` / ``light_tex_b``) and does real N·L against its
true glTF mesh normals — so it inherits the ship's colour, occlusion and
falloff (and, since graphics G0 P1, the ONE world tone-map at the resolve)
while gaining genuine 3D directional shading. See the design doc
``docs/marine_shader_foundation_design_2026-07-20.md`` (v2 hardened plan) and
the evidence in ``docs/research/marine_shader_foundation_2026-07-20_raw/``.

Binding contract (verified empirically, scratchpad/introspect_model.py, raylib
STATIC 6.0.1.0):
  * ``matModel`` / ``matNormal`` are auto-populated shader loc slots and
    auto-uploaded by DrawMesh every draw — no per-frame matrix push needed.
  * The marine albedo stays in MATERIAL_MAP_ALBEDO -> sampler ``texture0``.
  * ``light_tex_a`` goes in the MATERIAL_MAP_METALNESS slot -> sampler
    ``texture1``; ``light_tex_b`` in MATERIAL_MAP_NORMAL -> ``texture2``. Bound
    as material maps (NOT pre-bound uniforms) so DrawMesh binds them every draw
    and never clobbers our texture units.
  * The custom shader does NOT declare ``boneMatrices`` — the mesh is CPU-
    skinned (positions/normals already posed), so a bone uniform would double-
    transform it.

Shared GLSL lives in ``renderer/lit3d.py`` (the shared lit-3D-in-world-RT
seam, extracted 2026-09 for the props & vegetation arc #60): ``_COMMON_GLSL``
(the srgb decode; its encode + ACES moved to the world resolve at graphics G0
P1) at P1, and ``_FIELD_SAMPLE_GLSL`` (the light-field
sample/unpack + L/N setup) at P2 when ``static_props.py`` became the second
consumer. Both are concatenated into the fragment source here — the composed
``MARINE_FS`` stays byte-identical (``tests/test_lit3d_extraction.py``) and is
NOT copied verbatim from the golden-gated ship shader
``shaders/lighting.fs`` (which keeps its own inline copies; we do not touch
it). Kept numerically identical so tone/colour match the ship.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pyray as rl

from .lit3d import _COMMON_GLSL, _FIELD_SAMPLE_GLSL

# --- Marine-specific tunables (feel knobs; the whole arc is HUMAN-TEST gated) -
# The marine gets its OWN, more grazing key than the ship (ship default 0.5).
# A straight-overhead lamp (light_z -> 1) flattens a top-down character
# (all top faces N·L≈1); an angled key keeps a light side and a shaded side.
MARINE_LIGHT_Z = 0.35
# Cheap Fresnel-ish rim for silhouette pop against the floor (top-down, a
# small token reads as 3D more from its lit rim than from N·L over its top).
MARINE_RIM_STRENGTH = 0.35
MARINE_RIM_POWER = 2.5
# #33: how much the rim takes the albedo (0 = white rim, 1 = fully tinted). The
# shipped value is config.toml [render.unit_shading] rim_albedo, pushed per
# frame by UnitModelRenderer; this is only the compile-time default (white).
MARINE_RIM_ALBEDO = 0.0

# #33 highlight: compile-time defaults only (highlight OFF). The shipped values
# are config.toml [render.unit_shading] gloss_strength / gloss_shininess,
# pushed per frame by UnitModelRenderer. The per-texel gloss is the albedo
# texture's ALPHA (charkit/gameready.py bakes it from the materials' roughness).
MARINE_GLOSS_STRENGTH = 0.0
MARINE_GLOSS_SHININESS = 64.0

# --- P2 normal-map capability -------------------------------------------------
# How hard the tangent-space normal map perturbs the mesh normal: 0 = inert
# (N unchanged), 1 = full map. Compile-time defaults only: since #33 every look
# binds the normal map baked into its own .glb, and config.toml
# [render.unit_shading] normal_map / normal_strength drive the guard and the
# strength per frame.
MARINE_NORMAL_STRENGTH = 1.0
MARINE_USE_NORMAL_DEFAULT = False
# A file of this name next to a model is bound when its .glb carries no normal
# map of its own (none of the shipped looks needs it).
MARINE_NORMAL_MAP_FILENAME = "marine_normal_PLACEHOLDER.png"
# The green-channel sign of the looks' own maps (#33): Blender bakes +Y along
# its UV v, which runs UP the image; glTF (and raylib reading it) flips v to run
# DOWN, and cotangent_frame's B follows the texture coordinate as the shader
# sees it -- so green is negated. Multiplied with the ship's H-toggle sign.
UNIT_NORMAL_Y_SIGN = -1.0

MARINE_VS = """#version 330
// Lit-marine vertex shader. The mesh is CPU-skinned upstream
// (update_model_animation), so vertexPosition/vertexNormal are already POSED —
// no bone skinning here (declaring boneMatrices would double-transform). We
// only lift the posed vertex to world space for the fragment's field lookup.
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;

uniform mat4 mvp;
uniform mat4 matModel;    // auto-uploaded by DrawMesh (loc auto-populated)
uniform mat4 matNormal;   // = transpose(inverse(matModel)), world-space normals

out vec2 fragTexCoord;
out vec3 fragWorldPos;
out vec3 fragWorldNormal;

void main() {
    fragTexCoord = vertexTexCoord;
    fragWorldPos = (matModel * vec4(vertexPosition, 1.0)).xyz;
    fragWorldNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

_MARINE_FS_BODY = """#version 330
// Lit-marine fragment shader — samples the ship's baked light field on the
// marine's TRUE mesh normals. Mirrors shaders/lighting.fs's field unpack +
// composite, adapted to a Y-up 3D mesh with half-Lambert + rim. Writes linear
// HDR into the scene RT; the world resolve tone-maps (G0 P1).
in vec2 fragTexCoord;
in vec3 fragWorldPos;
in vec3 fragWorldNormal;

uniform sampler2D texture0;   // marine albedo (MATERIAL_MAP_ALBEDO)
uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: RGB smoke_glow, A = dir.y
uniform sampler2D texture3;   // P2 marine normal map (MATERIAL_MAP_ROUGHNESS
                              // slot — a FREE slot; light textures own 1 & 2)
uniform vec4  colDiffuse;     // draw colour (white since #33: looks are models)

uniform vec3  u_ambient;       // ship's ambient floor (single source of truth)
uniform float u_light_gain;    // ship's render exposure
uniform float u_light_z;       // marine's OWN grazing key (lower than ship)
uniform float u_normal_y_sign; // P2 normal-map green-channel convention (marine's
                               // OWN copy of the ship's H-toggle)
uniform vec2  u_world_px;      // (world_px_w, world_px_h) for the field UV
uniform vec3  u_view_dir;      // direction toward the eye (ortho ~ (0,1,0))
uniform float u_rim_strength;
uniform float u_rim_power;
uniform float u_rim_albedo;    // #33: rim colour = mix(white, albedo, this)
uniform float u_gloss_strength;  // #33: highlight strength (0 = no highlight)
uniform float u_gloss_shininess; // #33: highlight exponent at gloss 1
uniform float u_wrap;           // #114: 1 = half-Lambert (soft) .. 0 = Lambert (hard dark side)
uniform int   u_srgb_decode;
uniform int   u_use_normal_marine; // P2 guard: 0 = inert (N unchanged), 1 = on
uniform float u_normal_strength;   // P2 perturbation strength (feel knob)
// #114 body light: 1 = the whole body lit from the unit's ONE light read
// (renderer/unit_shadow.py read_unit_light, set per unit before its draw);
// 0 = each fragment from the field tile under it (the pre-#114 A/B).
uniform int   u_body_light;
uniform vec3  u_body_dir_rgb;      // one-sided irradiance (x exposure)
uniform vec3  u_body_iso_rgb;      // balanced irradiance (x exposure)
uniform vec2  u_body_to_light;     // toward the light, unit; (0,0) = none

out vec4 finalColor;

// <COMMON>

// Tangent-free TBN from screen-space derivatives of world-pos + UV (Schüler,
// "Followup: Normal Mapping Without Precomputed Tangents", thetenthplanet.de).
// The Quaternius mesh has NO vertex tangents and CPU skinning would not skin
// them anyway, so we reconstruct a per-fragment cotangent frame from dFdx/dFdy.
// Known degeneracy: on surfaces edge-on to the camera (the marine's vertical
// sides / silhouette) the UV derivatives collapse and the frame is unreliable;
// at true top-down the visible surface is mostly top-facing, where it holds.
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    // Scale-invariant, NaN-guarded (degenerate/edge-on fragments -> ~identity).
    float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-8));
    return mat3(T * invmax, B * invmax, N);
}

void main() {
// <FIELD_SAMPLE>

    // #114 body light. The unit is stamped opaque into the light sweep (its
    // floor shadow), so the field under the body holds light only in the
    // footprint row facing the lamp. In "unit" mode the body's light is the
    // unit's own read instead -- the same read its projected shadow takes --
    // so the whole body is lit and its lit side agrees with the shadow. The
    // smoke's cast shadow stays PER FRAGMENT, toward the read's light.
    // Both parts are already in exposure units (x u_light_gain).
    vec3 light_dir_rgb = incoming_rgb * u_light_gain;
    vec3 light_iso_rgb = vec3(0.0);
    if (u_body_light == 1) {
        float smoke_k = smokeShadow(world_uv, u_body_to_light,
                                    fragWorldPos.y * u_smoke_grid.x / u_world_px.x);
        light_dir_rgb = u_body_dir_rgb * smoke_k;
        light_iso_rgb = u_body_iso_rgb * smoke_k;
        L = normalize(vec3(u_body_to_light.x, u_light_z, u_body_to_light.y));
    }

    // P2 normal map (guarded — zero cost when off). Reconstruct a tangent frame
    // from screen-space derivatives, sample the map as LINEAR data (never sRGB-
    // decode a normal map — its RGB are directions), unpack, apply the marine's
    // own green-channel sign + strength, and perturb N BEFORE the N·L below.
    // Inert if no map is bound (a flat 1x1 default reads (0.5,0.5,1) -> +Z, i.e.
    // N unchanged after transform) or if u_normal_strength is 0.
    if (u_use_normal_marine == 1) {
        vec3 nmap = texture(texture3, fragTexCoord).rgb * 2.0 - 1.0;
        nmap.y *= u_normal_y_sign;
        nmap.xy *= u_normal_strength;
        mat3 TBN = cotangent_frame(N, fragWorldPos, fragTexCoord);
        N = normalize(TBN * nmap);
    }

    // Wrapped Lambert (#114): u_wrap 1 is the half-Lambert of design v2 §Bugs
    // (soft, the side facing away still half lit), 0 plain Lambert (a hard
    // dark side). [render.unit_shading] wrap, read every frame.
    float ndotl = max(0.0, (dot(N, L) + u_wrap) / (1.0 + u_wrap));
    // #114: the balanced share's weight -- the same wrapped term averaged over
    // every horizontal direction at the key's height (exact at u_wrap 1, where
    // it is linear in L; below 1 the clamp is taken after the average), so a
    // ring of equal lamps lights every side alike and the top most.
    float ndotl_iso = max(0.0, (N.y * u_light_z / sqrt(1.0 + u_light_z * u_light_z)
                                + u_wrap) / (1.0 + u_wrap));
    vec3 light_rgb = light_dir_rgb + light_iso_rgb;   // all of it, x exposure

    vec4 albedo_gloss = texture(texture0, fragTexCoord);
    vec3 albedo = albedo_gloss.rgb;
    float gloss = albedo_gloss.a;  // #33: the bake's gloss mask (0 matte .. 1 glossiest)
    if (u_srgb_decode == 1) albedo = srgb_to_linear(albedo);  // else double-dark
    albedo *= colDiffuse.rgb;                                  // draw colour

    vec3 lit = albedo * (u_ambient + light_dir_rgb * ndotl
                         + light_iso_rgb * ndotl_iso);

    // #33 highlight (Blinn-Phong): the LOCAL light's own colour, never the
    // ambient (a dark room puts no shine on a visor), scaled AND tightened by
    // the texel's gloss -- a visor gets a small bright spot, a plate a broad
    // faint sheen, cloth nothing. Faded out where the surface turns away from
    // the light (dot(N, L) < 0) so the lobe never lights a back face.
    vec3 H = normalize(L + u_view_dir);
    float shininess = mix(4.0, u_gloss_shininess, gloss);
    float spec = u_gloss_strength * gloss
               * pow(max(dot(N, H), 0.0), shininess)
               * clamp(dot(N, L) * 4.0, 0.0, 1.0);
    lit += spec * light_rgb;

    // Rim: Fresnel-ish silhouette term, tinted by the LOCAL light so it never
    // brightens a marine the room around it can't (dark room -> faint ambient
    // rim only), and by the surface's own colour (#33, u_rim_albedo) so dark
    // cloth keeps a dark edge instead of reading light grey from above.
    float rim = pow(1.0 - max(0.0, dot(N, u_view_dir)), u_rim_power);
    vec3 rim_col = mix(vec3(1.0), albedo, u_rim_albedo);
    lit += u_rim_strength * rim * rim_col * (u_ambient + light_rgb);

    // Linear HDR out (G0 P1): the world resolve tone-maps and encodes, once.

    // alpha = 1.0: the world RT is blitted premultiplied; a translucent marine
    // would bleed the background through (design v2 §Bugs). The texture's
    // alpha is the gloss mask (#33), never coverage.
    finalColor = vec4(lit, 1.0);
}
"""

MARINE_FS = (_MARINE_FS_BODY
             .replace("// <COMMON>", _COMMON_GLSL)
             .replace("// <FIELD_SAMPLE>", _FIELD_SAMPLE_GLSL))


@dataclass
class MarineShader:
    """A loaded marine shader plus its scalar-uniform locations."""
    shader: rl.Shader
    locs: Dict[str, int]

    def set_frame_uniforms(self, ambient, light_gain: float, world_px_w: float,
                           world_px_h: float, normal_y_sign: float = 1.0,
                           view_dir=(0.0, 1.0, 0.0)) -> None:
        """Push the per-frame scalar uniforms (SetShaderValue self-enables the
        program, so this is safe to call before begin_mode_3d). matModel /
        matNormal / mvp / colDiffuse are uploaded per-draw by DrawMesh."""
        s = self.shader
        L = self.locs
        rl.set_shader_value(s, L["u_ambient"],
                            rl.ffi.new("float[3]", [float(ambient[0]),
                                                    float(ambient[1]),
                                                    float(ambient[2])]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
        rl.set_shader_value(s, L["u_light_gain"],
                            rl.ffi.new("float[1]", [float(light_gain)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value(s, L["u_world_px"],
                            rl.ffi.new("float[2]", [float(world_px_w),
                                                    float(world_px_h)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
        rl.set_shader_value(s, L["u_normal_y_sign"],
                            rl.ffi.new("float[1]", [float(normal_y_sign)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value(s, L["u_view_dir"],
                            rl.ffi.new("float[3]", [float(view_dir[0]),
                                                    float(view_dir[1]),
                                                    float(view_dir[2])]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

    def set_rim_albedo(self, amount: float) -> None:
        """#33: rim tint by albedo, 0 (white rim) .. 1 (fully albedo-tinted).
        Safe to call anytime -- SetShaderValue self-enables the program."""
        rl.set_shader_value(self.shader, self.locs["u_rim_albedo"],
                            rl.ffi.new("float[1]", [float(amount)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def set_gloss(self, strength: float, shininess: float) -> None:
        """#33 highlight: strength (0 = none) and its exponent at gloss 1.
        Safe to call anytime -- SetShaderValue self-enables the program."""
        rl.set_shader_value(self.shader, self.locs["u_gloss_strength"],
                            rl.ffi.new("float[1]", [float(strength)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        rl.set_shader_value(self.shader, self.locs["u_gloss_shininess"],
                            rl.ffi.new("float[1]", [float(shininess)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def set_use_normal(self, enabled: bool) -> None:
        """P2 on/off. Clean, zero-cost when off (the fragment guard skips the
        whole TBN+sample path). Safe to call anytime — SetShaderValue self-
        enables the program."""
        rl.set_shader_value(self.shader, self.locs["u_use_normal_marine"],
                            rl.ffi.new("int[1]", [1 if enabled else 0]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def set_normal_strength(self, strength: float) -> None:
        """P2 feel knob (0 = inert, 1 = full map)."""
        rl.set_shader_value(self.shader, self.locs["u_normal_strength"],
                            rl.ffi.new("float[1]", [float(strength)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def set_light_shape(self, light_z: float, wrap: float) -> None:
        """#114: the key's height (the 2D light direction lifted to 3D;
        lower = more grazing) and the wrap (1 half-Lambert .. 0 Lambert),
        from ``[render.unit_shading] light_height`` / ``wrap``."""
        f1 = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT
        rl.set_shader_value(self.shader, self.locs["u_light_z"],
                            rl.ffi.new("float[1]", [float(light_z)]), f1)
        rl.set_shader_value(self.shader, self.locs["u_wrap"],
                            rl.ffi.new("float[1]", [float(wrap)]), f1)

    def set_body_light(self, from_unit: bool) -> None:
        """#114: True = the body lit from each unit's read (set per unit by
        :meth:`set_unit_light`), False = from the field tile under each
        fragment. Safe to call anytime."""
        rl.set_shader_value(self.shader, self.locs["u_body_light"],
                            rl.ffi.new("int[1]", [1 if from_unit else 0]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def set_unit_light(self, light) -> None:
        """#114: one unit's light read (``unit_shadow.UnitLight``), pushed
        just before that unit's draw. The read's direction is the way the
        light TRAVELS; the shader wants the way TOWARD it."""
        s, L, f3 = self.shader, self.locs, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3
        rl.set_shader_value(s, L["u_body_dir_rgb"],
                            rl.ffi.new("float[3]", [float(c) for c in light.dir_rgb]), f3)
        rl.set_shader_value(s, L["u_body_iso_rgb"],
                            rl.ffi.new("float[3]", [float(c) for c in light.iso_rgb]), f3)
        rl.set_shader_value(s, L["u_body_to_light"],
                            rl.ffi.new("float[2]", [-float(light.dir_x),
                                                    -float(light.dir_y)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)


def load_marine_shader(light_z: float = MARINE_LIGHT_Z,
                       rim_strength: float = MARINE_RIM_STRENGTH,
                       rim_power: float = MARINE_RIM_POWER,
                       srgb_decode: bool = True,
                       use_normal: bool = MARINE_USE_NORMAL_DEFAULT,
                       normal_strength: float = MARINE_NORMAL_STRENGTH
                       ) -> MarineShader:
    """Compile the marine shader from memory and prime its constant uniforms.

    Returns None-safe: if compilation fails raylib logs and returns a shader
    whose id may be the default; the caller should treat a zero/!=expected id
    defensively. The scalar-only uniforms that never change per frame
    (u_light_z, rim, u_srgb_decode) are set once here.
    """
    shader = rl.load_shader_from_memory(MARINE_VS, MARINE_FS)
    names = ["u_ambient", "u_light_gain", "u_light_z", "u_normal_y_sign",
             "u_world_px", "u_view_dir", "u_rim_strength", "u_rim_power",
             "u_srgb_decode", "u_use_normal_marine", "u_normal_strength",
             "u_rim_albedo", "u_gloss_strength", "u_gloss_shininess",
             "u_body_light", "u_body_dir_rgb", "u_body_iso_rgb",
             "u_body_to_light", "u_wrap"]
    locs = {n: rl.get_shader_location(shader, n) for n in names}

    # P2 normal map: bind the ROUGHNESS(3) material slot to the `texture3`
    # sampler. raylib auto-locates ONLY the first three map samplers
    # (ALBEDO->texture0, METALNESS->texture1, NORMAL->texture2); slot 3 is NOT
    # auto-wired (verified: locs[SHADER_LOC_MAP_ROUGHNESS] == -1 in raylib
    # 6.0.1.0), so we wire it explicitly. DrawMesh then binds whatever texture
    # sits in maps[MATERIAL_MAP_ROUGHNESS] to texture unit 3 every draw — the
    # same auto-bind mechanism the light textures ride, in a free slot that
    # never collides with them.
    shader.locs[rl.ShaderLocationIndex.SHADER_LOC_MAP_ROUGHNESS] = \
        rl.get_shader_location(shader, "texture3")

    # Constant-per-run uniforms, set once.
    rl.set_shader_value(shader, locs["u_light_z"],
                        rl.ffi.new("float[1]", [float(light_z)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_wrap"],
                        rl.ffi.new("float[1]", [1.0]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_rim_strength"],
                        rl.ffi.new("float[1]", [float(rim_strength)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_rim_power"],
                        rl.ffi.new("float[1]", [float(rim_power)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_rim_albedo"],
                        rl.ffi.new("float[1]", [float(MARINE_RIM_ALBEDO)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_gloss_strength"],
                        rl.ffi.new("float[1]", [float(MARINE_GLOSS_STRENGTH)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_gloss_shininess"],
                        rl.ffi.new("float[1]", [float(MARINE_GLOSS_SHININESS)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_srgb_decode"],
                        rl.ffi.new("int[1]", [1 if srgb_decode else 0]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
    # P2 normal-map guard + strength (constant per run; flip via set_use_normal).
    rl.set_shader_value(shader, locs["u_use_normal_marine"],
                        rl.ffi.new("int[1]", [1 if use_normal else 0]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
    rl.set_shader_value(shader, locs["u_normal_strength"],
                        rl.ffi.new("float[1]", [float(normal_strength)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    # Safe default so a stray draw before set_frame_uniforms can't divide
    # world_uv by zero (an unset vec2 reads (0,0) -> NaN). Overwritten per frame.
    rl.set_shader_value(shader, locs["u_world_px"],
                        rl.ffi.new("float[2]", [1.0, 1.0]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)
    return MarineShader(shader=shader, locs=locs)


__all__ = ["load_marine_shader", "MarineShader", "MARINE_VS", "MARINE_FS",
           "MARINE_LIGHT_Z", "MARINE_RIM_STRENGTH", "MARINE_RIM_POWER",
           "MARINE_RIM_ALBEDO", "MARINE_GLOSS_STRENGTH", "MARINE_GLOSS_SHININESS",
           "MARINE_NORMAL_STRENGTH", "MARINE_USE_NORMAL_DEFAULT",
           "MARINE_NORMAL_MAP_FILENAME", "UNIT_NORMAL_Y_SIGN"]
