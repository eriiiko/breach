#version 330
// Lighting shader for Breach: composites diffuse + normal map + light field.
//
// Input textures:
//   u_diffuse   - RGB diffuse art at high resolution (sRGB-encoded PNG)
//   u_normal    - normal map at the SAME resolution as diffuse (linear)
//   u_light_a   - RGBA16F light field A at low (physics) resolution:
//                   RGB = incoming light colour, A = light_dir.x (signed)
//   u_light_b   - RGBA16F light field B at low (physics) resolution:
//                   RGB = smoke_glow (god-ray glow, ch.03 C16), A = light_dir.y (signed)
//                 (light_dir is stored signed in 16F — no 0.5-centered encode)
//                 NOTE: smoke_glow RGB is carried here for the render contract,
//                 but the god-ray shaft is drawn as a SEPARATE additive overlay
//                 (renderer/overlays.py GlowOverlay) composited with the smoke,
//                 before units (ch.05 §God-rays). This shader only reads tex_b.a.
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
uniform sampler2D u_light_a;       // RGB = incoming light colour, A = light_dir.x
uniform sampler2D u_light_b;       // RGB = smoke_glow (god-ray), A = light_dir.y
uniform sampler2D u_vacuum;        // physics-res mask, R>0.5 = vacuum tile
                                   // (don't draw — let background show through)

uniform vec3  u_ambient;
uniform float u_normal_strength;
uniform int   u_use_normal;
uniform float u_normal_y_sign;
uniform int   u_srgb_decode;
uniform float u_light_z;           // 0..1: 0 = grazing horizontal, 1 = straight down
                                   // (more grazing = stronger normal-map relief)
uniform vec4  u_art_uv_rect;       // art-UV subrect drawn over the world RT
                                   // (xy = origin, zw = size) — the [art.align]
                                   // transform (level format v2 §1.3). The quad's
                                   // fragTexCoord interpolates over THIS rect (art
                                   // space, samples diffuse/normal); world-space
                                   // samplers (light field, vacuum) need world UV
                                   // = (fragTexCoord - xy) / zw. Default (0,0,1,1)
                                   // = legacy full-stretch: world_uv == fragTexCoord
                                   // bit-exactly ((x - 0) / 1 == x in IEEE 754).

out vec4 finalColor;

// Cheap sRGB <-> linear approximations (gamma 2.2). Adequate for game
// lighting; not strict sRGB which uses a piecewise function near zero.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

// ACES filmic tone-map (Narkowicz approximation): compresses HDR over-bright
// colour toward [0,1] while staying punchy/saturated, instead of per-channel
// clipping that hue-shifts bright warm light toward white. ch.05 §Tone-mapping.
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    // World-space UV (0..1 over the world RT / grid). With the default
    // u_art_uv_rect (0,0,1,1) this IS fragTexCoord (bit-exact); with an
    // explicit [art.align] it inverts the art->world src-rect mapping so the
    // grid-resolution samplers stay glued to the grid while the art shifts.
    vec2 world_uv = (fragTexCoord - u_art_uv_rect.xy) / u_art_uv_rect.zw;

    // Vacuum tiles are not part of the ship — discard so the screen-space
    // background (stars, void) shows through.
    float vacuum = texture(u_vacuum, world_uv).r;
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
    // the drawn quad covers the full world. The light field also covers the
    // full world, so it is sampled at world_uv (== fragTexCoord for the
    // legacy full-stretch draw; the [art.align] inverse otherwise). Camera
    // scrolling happens later, as a separate blit from the world RT to the
    // screen — see WorldComposite.
    vec4 tex_a = texture(u_light_a, world_uv);
    vec4 tex_b = texture(u_light_b, world_uv);
    vec3 incoming_rgb = tex_a.rgb;      // total light colour reaching this tile
    // light_dir is stored signed in 16F — reconstruct directly (no decode).
    vec2 light_dir_2d = vec2(tex_a.a, tex_b.a);

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

    // Diffuse: albedo x incoming RGB (a red lamp lights surfaces red).
    // ndotl modulates the directional contribution; ambient is a flat floor.
    vec3 lit = diffuse * (u_ambient + incoming_rgb * ndotl);
    // ACES filmic tone-map on the linear HDR value (ch.05 §Tone-mapping):
    // tames over-bright coloured light without per-channel clip / hue-shift.
    lit = aces_tonemap(lit);
    if (u_srgb_decode == 1) {
        // Encode back to sRGB for the default framebuffer.
        lit = linear_to_srgb(lit);
    }
    finalColor = vec4(lit, 1.0);
}
