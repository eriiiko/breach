#version 330
// Lighting shader for Breach: composites diffuse + normal map + light field.
//
// Input textures:
//   u_diffuse   — RGB diffuse art at high resolution (sRGB-ish, treated linear here)
//   u_normal    — normal map at the SAME resolution as diffuse (linear)
//   u_light     — light field at low (physics) resolution:
//                   R = intensity, G = light_dx, B = light_dy
//                 (G/B are signed; we encode/decode via 0.5 + 0.5*x in CPU)
//
// Uniforms:
//   u_ambient        — base light color when no light source reaches a tile
//   u_normal_strength — 0 disables normal map, 1 = full strength
//   u_use_normal     — bool toggle for normal mapping
//
// Pyray sends texture0 as u_diffuse implicitly (Raylib convention).
// Additional samplers are bound by name via SetShaderValueTexture.

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D u_diffuse;
uniform sampler2D u_normal;
uniform sampler2D u_light;

uniform vec3  u_ambient;
uniform float u_normal_strength;
uniform int   u_use_normal;    // 0 = off, 1 = on

out vec4 finalColor;

void main() {
    vec3 diffuse = texture(u_diffuse, fragTexCoord).rgb;

    // KNOWN BUG: light field is sampled at fragTexCoord (visible-window UV)
    // instead of world UV. When camera scrolls, light/smoke/fire stay anchored
    // at world (0,0) instead of following the camera. To be solved properly
    // via a Camera/coordinate-system design — see todo "camera architecture".
    vec3 light_sample = texture(u_light, fragTexCoord).rgb;
    float intensity = light_sample.r;
    // Decode signed direction: stored as 0.5 + 0.5*x, so (sample - 0.5) * 2.
    vec2 light_dir_2d = (light_sample.gb - 0.5) * 2.0;

    float ndotl = 1.0;  // default: flat (no normal map)
    if (u_use_normal == 1) {
        // Normal map: unpack from [0,1] to [-1,1].
        vec3 N = texture(u_normal, fragTexCoord).rgb * 2.0 - 1.0;
        N = mix(vec3(0.0, 0.0, 1.0), N, u_normal_strength);
        N = normalize(N);

        // Construct 3D light direction. Z is a placeholder (0.5 for v1).
        // See patch_level_pipeline_v1.md "Expert review feedback".
        vec3 L = normalize(vec3(light_dir_2d, 0.5));

        ndotl = max(dot(N, L), 0.0);
    }

    vec3 lit = diffuse * (u_ambient + intensity * ndotl);
    finalColor = vec4(lit, 1.0);
}
