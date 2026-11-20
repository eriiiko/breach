#version 330
// larva_body.fs -- THE swarm larvae's bodies (arc #63,
// renderer/larva_bodies.py LarvaBodyDraw). RENDER-ONLY, LINEAR HDR into the
// scene RT, PREMULTIPLIED ALPHA (blend ONE, ONE_MINUS_SRC_ALPHA on RGB and
// alpha). The one tone-map (world_resolve.fs) does the rest.
//
// Lit the props' way: the light arriving at the fragment's tile is read
// from light_tex_a/b through the lit-3D block (renderer/lit3d.py
// _FIELD_SAMPLE_GLSL, spliced in below with _COMMON_GLSL: the smoke's cast
// shadow and the light curve included). Half-Lambert on the bent normal, a
// Blinn gloss highlight and a rim, both of the LOCAL light (a larva is
// never brighter than the floor it lies on).
//
// The look: a pale body, the head end darker (from u_look.z along the
// body), SEGMENTS as shaded grooves (u_look.x of them, depth u_look.y: the
// groove darkens and tilts the normal along the body), a cold tint in chill
// coma, a dead body darkening to u_dead_rgb as it goes limp (and losing
// its gloss).

in vec3 fragWorldPos;
in vec3 v_normal;
in vec3 v_along;
in float v_s;
in float v_shade;
in float v_limp;
in float v_alpha;
in float v_coma;

uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: A = dir.y
uniform vec2  u_world_px;     // (world_px_w, world_px_h) for the field UV
uniform float u_light_z;      // the lit-3D block's grazing key
uniform vec3  u_ambient;      // the ship's ambient floor
uniform float u_light_gain;   // the ship's render exposure
uniform vec4  u_look;         // (segments, groove depth, head start, coma mix)
uniform vec4  u_gloss;        // (gloss, gloss power, rim, -)
uniform vec3  u_body;         // linear RGB
uniform vec3  u_head;
uniform vec3  u_coma;
uniform vec3  u_dead_rgb;

out vec4 finalColor;

// <COMMON>

void main() {
    // the segment grooves: cos peaks at the boundaries between segments
    float ph = 6.28318530718 * u_look.x * v_s;
    float ends = smoothstep(0.0, 0.08, v_s) * smoothstep(1.0, 0.92, v_s);
    float groove = smoothstep(0.55, 1.0, cos(ph)) * u_look.y * ends;
    vec3 fragWorldNormal = normalize(v_normal
                              + v_along * (0.5 * u_look.y * ends
                                                  * sin(ph)));
// <FIELD_SAMPLE>

    float ndotl = dot(N, L) * 0.5 + 0.5;

    // the head capsule: a short, fairly sharp change from u_look.z on
    vec3 albedo = mix(u_body, u_head,
                      smoothstep(u_look.z, min(u_look.z + 0.12, 1.0), v_s));
    albedo = mix(albedo, u_coma, u_look.w * v_coma);
    albedo = mix(albedo, u_dead_rgb, v_limp);
    albedo *= v_shade * (1.0 - 0.5 * groove);

    vec3 light_rgb = incoming_rgb * u_light_gain;
    light_rgb *= lightCurveGain(light_rgb);
    albedo = lightCurveAlbedo(albedo, light_rgb);

    vec3 lit = albedo * (u_ambient + light_rgb * ndotl);

    // the gloss: a Blinn highlight of the local light toward the eye
    // (straight down: V = +Y), lost as the body goes limp
    vec3 H = normalize(L + vec3(0.0, 1.0, 0.0));
    float spec = pow(max(dot(N, H), 0.0), u_gloss.y);
    lit += u_gloss.x * (1.0 - v_limp) * (1.0 - groove) * spec * light_rgb;
    float rim = pow(1.0 - max(0.0, N.y), 3.0);
    lit += u_gloss.z * rim * (u_ambient + light_rgb) * albedo;

    float a = v_alpha;
    finalColor = vec4(lit * a, a);
}
