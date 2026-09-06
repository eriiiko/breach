"""GLSL + loader for the lit 3D-marine material shader (marine lit-shader arc).

The marine samples the SAME baked light field the 2D ship is lit by
(``LightingPass.light_tex_a`` / ``light_tex_b``) and does real N·L against its
true glTF mesh normals — so it inherits the ship's colour, occlusion, falloff
and tone-map while gaining genuine 3D directional shading. See the design doc
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

Shared GLSL (srgb decode/encode, ACES) lives in ``renderer/lit3d.py``'s
``_COMMON_GLSL`` (the shared lit-3D-in-world-RT seam, extracted 2026-09 for
the props & vegetation arc #60 P1) and is concatenated into the fragment
source here — NOT copied verbatim from the golden-gated ship shader
``shaders/lighting.fs`` (which keeps its own inline copies; we do not touch
it). Kept numerically identical so tone/colour match the ship.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pyray as rl

from .lit3d import _COMMON_GLSL

# --- Marine-specific tunables (feel knobs; the whole arc is HUMAN-TEST gated) -
# The marine gets its OWN, more grazing key than the ship (ship default 0.5).
# A straight-overhead lamp (light_z -> 1) flattens a top-down character
# (all top faces N·L≈1); an angled key keeps a light side and a shaded side.
MARINE_LIGHT_Z = 0.35
# Cheap Fresnel-ish rim for silhouette pop against the floor (top-down, a
# small token reads as 3D more from its lit rim than from N·L over its top).
MARINE_RIM_STRENGTH = 0.35
MARINE_RIM_POWER = 2.5

# --- P2 normal-map capability (default OFF; see below) ------------------------
# The single feel knob for how hard the tangent-space normal map perturbs the
# mesh normal: 0 = inert (N unchanged), 1 = full map. A future real normal-mapped
# marine asset just replaces marine_normal_PLACEHOLDER.png — no code change.
MARINE_NORMAL_STRENGTH = 1.0
# Default state of the u_use_normal_marine guard. Ships OFF: the bundled asset is
# UNTEXTURED (flat base-colour, no authored albedo/normal map), so the placeholder
# normal map proves the plumbing but does NOT visibly improve this model — the
# payoff awaits a real normal-mapped marine asset. Flip to True (or drive the
# uniform) once such an asset is dropped in. See VERIFY note in the P2 patch.
MARINE_USE_NORMAL_DEFAULT = False
# Filename of the placeholder normal map, resolved next to the model asset.
MARINE_NORMAL_MAP_FILENAME = "marine_normal_PLACEHOLDER.png"

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
// composite + tone-map, adapted to a Y-up 3D mesh with half-Lambert + rim.
in vec2 fragTexCoord;
in vec3 fragWorldPos;
in vec3 fragWorldNormal;

uniform sampler2D texture0;   // marine albedo (MATERIAL_MAP_ALBEDO)
uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: RGB smoke_glow, A = dir.y
uniform sampler2D texture3;   // P2 marine normal map (MATERIAL_MAP_ROUGHNESS
                              // slot — a FREE slot; light textures own 1 & 2)
uniform vec4  colDiffuse;     // group tint (draw color): marines / zombies

uniform vec3  u_ambient;       // ship's ambient floor (single source of truth)
uniform float u_light_gain;    // ship's render exposure
uniform float u_light_z;       // marine's OWN grazing key (lower than ship)
uniform float u_normal_y_sign; // P2 normal-map green-channel convention (marine's
                               // OWN copy of the ship's H-toggle)
uniform vec2  u_world_px;      // (world_px_w, world_px_h) for the field UV
uniform vec3  u_view_dir;      // direction toward the eye (ortho ~ (0,1,0))
uniform float u_rim_strength;
uniform float u_rim_power;
uniform int   u_srgb_decode;
uniform int   u_use_normal_marine; // P2 guard: 0 = inert (N unchanged), 1 = on
uniform float u_normal_strength;   // P2 perturbation strength (feel knob)

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

    // Half-Lambert: softens the pre-normalized-direction instability that
    // side normals get under near-overhead lamps AND reads better on a small
    // top-down token (design v2 §Bugs).
    float ndotl = dot(N, L) * 0.5 + 0.5;

    vec3 albedo = texture(texture0, fragTexCoord).rgb;
    if (u_srgb_decode == 1) albedo = srgb_to_linear(albedo);  // else double-dark
    albedo *= colDiffuse.rgb;                                  // group identity

    vec3 lit = albedo * (u_ambient + incoming_rgb * u_light_gain * ndotl);

    // Rim: Fresnel-ish silhouette term, tinted by the LOCAL light so it never
    // brightens a marine the room around it can't (dark room -> faint ambient
    // rim only).
    float rim = pow(1.0 - max(0.0, dot(N, u_view_dir)), u_rim_power);
    lit += u_rim_strength * rim * (u_ambient + incoming_rgb * u_light_gain);

    lit = aces_tonemap(lit);
    if (u_srgb_decode == 1) lit = linear_to_srgb(lit);

    // alpha = 1.0: the world RT is blitted premultiplied; a translucent marine
    // would bleed the background through (design v2 §Bugs).
    finalColor = vec4(lit, 1.0);
}
"""

MARINE_FS = _MARINE_FS_BODY.replace("// <COMMON>", _COMMON_GLSL)


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
             "u_srgb_decode", "u_use_normal_marine", "u_normal_strength"]
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
    rl.set_shader_value(shader, locs["u_rim_strength"],
                        rl.ffi.new("float[1]", [float(rim_strength)]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(shader, locs["u_rim_power"],
                        rl.ffi.new("float[1]", [float(rim_power)]),
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
           "MARINE_NORMAL_STRENGTH", "MARINE_USE_NORMAL_DEFAULT",
           "MARINE_NORMAL_MAP_FILENAME"]
