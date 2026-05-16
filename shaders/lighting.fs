#version 330
// Lighting shader for Breach: composites diffuse + normal map + light field.
//
// Input textures:
//   u_diffuse   - RGB diffuse art at high resolution (sRGB-encoded PNG)
//   u_normal    - normal map at the SAME resolution as diffuse (linear)
//   u_light     - light field at low (physics) resolution:
//                   R = intensity, G = light_dx, B = light_dy
//                 (G/B are signed; we encode/decode via 0.5 + 0.5*x in CPU)
//
// Uniforms:
//   u_ambient         - base light color when no light source reaches a tile
//   u_normal_strength - 0 disables normal map, 1 = full strength
//   u_use_normal      - bool toggle (0/1) for normal mapping
//   u_normal_y_sign   - +1 for OpenGL Y-up normals, -1 for DirectX Y-down
//   u_srgb_decode     - 1 = decode sRGB diffuse to linear, re-encode on output
//
// Pyray sends texture0 as u_diffuse implicitly (Raylib convention).
// Additional samplers are bound by name via SetShaderValueTexture.

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D u_diffuse;
uniform sampler2D u_normal;
uniform sampler2D u_light;
uniform sampler2D u_vacuum;        // physics-res mask, R>0.5 = vacuum tile
                                   // (don't draw — let background show through)

uniform vec3  u_ambient;
uniform float u_normal_strength;
uniform int   u_use_normal;
uniform float u_normal_y_sign;
uniform int   u_srgb_decode;
uniform float u_light_z;           // 0..1: 0 = grazing horizontal, 1 = straight down
                                   // (more grazing = stronger normal-map relief)

out vec4 finalColor;

// Cheap sRGB <-> linear approximations (gamma 2.2). Adequate for game
// lighting; not strict sRGB which uses a piecewise function near zero.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

void main() {
    // Vacuum tiles are not part of the ship — discard so the screen-space
    // background (stars, void) shows through.
    float vacuum = texture(u_vacuum, fragTexCoord).r;
    if (vacuum > 0.5) {
        discard;
    }

    vec3 diffuse = texture(u_diffuse, fragTexCoord).rgb;
    if (u_srgb_decode == 1) {
        // PNG textures from image editors / AI tools are sRGB-encoded.
        // Lighting math must happen in linear space.
        diffuse = srgb_to_linear(diffuse);
    }

    // Light field sample: this shader is invoked inside the world RT, where
    // the diffuse covers the full world (0..1 fragTexCoord <-> 0..1 world).
    // The light field also covers the full world, so sampling at the same
    // fragTexCoord gives the right tile. Camera scrolling happens later, as
    // a separate blit from the world RT to the screen — see WorldComposite.
    vec3 light_sample = texture(u_light, fragTexCoord).rgb;
    float intensity = light_sample.r;
    // Decode signed direction: stored as 0.5 + 0.5*x, so (sample - 0.5) * 2.
    vec2 light_dir_2d = (light_sample.gb - 0.5) * 2.0;

    float ndotl = 1.0;  // default: flat (no normal map)
    if (u_use_normal == 1) {
        // Normal map: unpack from [0,1] to [-1,1].
        vec3 N = texture(u_normal, fragTexCoord).rgb * 2.0 - 1.0;
        // Flip Y if the normal map uses DirectX convention (Laigter varies).
        N.y *= u_normal_y_sign;
        N = mix(vec3(0.0, 0.0, 1.0), N, u_normal_strength);
        N = normalize(N);

        // Construct 3D light direction. Z controlled live via u_light_z:
        //   small Z (~0.1)  = light skims along the floor → high relief
        //   medium Z (~0.5) = standing height / overhead lamp feel
        //   large Z (~1.0)  = light from directly above → flat shading
        vec3 L = normalize(vec3(light_dir_2d, u_light_z));

        ndotl = max(dot(N, L), 0.0);
    }

    vec3 lit = diffuse * (u_ambient + intensity * ndotl);
    if (u_srgb_decode == 1) {
        // Encode back to sRGB for the default framebuffer.
        lit = linear_to_srgb(lit);
    }
    finalColor = vec4(lit, 1.0);
}
