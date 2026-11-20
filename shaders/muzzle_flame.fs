#version 330
// muzzle_flame.fs -- THE muzzle flame (#87, renderer/muzzle_flame.py).
// RENDER-ONLY, LINEAR HDR into the scene RT, ADDITIVE (blend ONE, ONE on
// RGB; the destination alpha is kept) -- the one tone-map (world_resolve.fs)
// does the rest.
//
// ITS SPEC IS THE NUMPY TWIN, muzzle_flame.flame_radiance: a change lands
// there first. Per flash the CPU drew the shape (muzzle_flame.shot_rows):
//   row 1 = (age, noise u, noise v, px per metre)
//   row 2 = (core m, halo m, halo gain, gain)
//   row 3 = (T core, T edge, tear, half m)
//   rows 4..8 = a jet each: (angle rad, length m, width m, gain); length 0 =
//               no jet
// Here, per pixel: every jet is a tapered tongue with a wobbling centre line;
// a streaked noise (the one fBm bake: R, B tear; G wobbles) eats the body,
// harder with age; a core ball at the muzzle; temperature from the rim to
// the core, cooling with age; emission ONLY through the blackbody GPU LUT
// (rgb = chroma, a = intensity); a soft warm halo (the resolve has no bloom).
//
// Two stages, the flame's and the sparks' (design §13.7 of the G3 doc):
//   u_stage 0  in the scene, after the units, UNDER the ceiling soot;
//   u_stage 1  the COMPENSATION after the soot: add back alpha_ceil (1 -
//              glow_through_ceiling), so the soot hides a muzzle flame as
//              much as it hides a fire.

in vec2 v_local;
in vec2 v_world_uv;
flat in int v_index;

uniform sampler2D u_data;
uniform sampler2D u_lut;      // blackbody GPU LUT (1024 x 1, bilinear)
uniform sampler2D u_noise;    // the fBm bake (REPEAT, bilinear)
uniform vec2 u_lut_k;         // the LUT's Kelvin domain (floor, ceil)
uniform float u_lut_n;        // its entry count
uniform int   u_stage;        // 0 = under the soot, 1 = compensation
uniform float u_glow_through_ceil;

// <SMOKE_SHADOW>

out vec4 finalColor;

// The twin's constants (muzzle_flame.py), by the same names.
const int   MAX_JETS = 5;
const float NOISE_CELLS = 4.0;
const float GROW_FROM = 0.9, GROW_BY = 0.3, GROW_AGE = 1.0 / 3.0;
const float WOBBLE_AT_BIRTH = 0.7, WOBBLE_WITH_AGE = 1.6;
const float TEAR_AT_BIRTH = 0.30, TEAR_WITH_AGE = 1.3;
const float NOISE_CONTRAST = 1.7;
const float HALO_K = 1700.0;
const float EDGE_FROM = 0.85;

vec3 le_at(float k) {
    float x = clamp((k - u_lut_k.x) / (u_lut_k.y - u_lut_k.x), 0.0, 1.0);
    vec4 le = texture(u_lut, vec2((x * (u_lut_n - 1.0) + 0.5) / u_lut_n, 0.5));
    return le.rgb * le.a;
}

void main() {
    vec4 r1 = texelFetch(u_data, ivec2(v_index, 1), 0);
    vec4 r2 = texelFetch(u_data, ivec2(v_index, 2), 0);
    vec4 r3 = texelFetch(u_data, ivec2(v_index, 3), 0);
    float age = r1.x;
    if (age < 0.0 || age >= 1.0) discard;
    vec2 ofs = r1.yz;
    float x = v_local.x, y = v_local.y;
    float half_m = r3.w;
    float edge = max(abs(x), abs(y)) / half_m;
    float win = clamp((1.0 - edge) / (1.0 - EDGE_FROM), 0.0, 1.0);
    win = win * win * (3.0 - 2.0 * win);
    if (win <= 0.0) discard;

    float grow = GROW_FROM + GROW_BY * min(age / GROW_AGE, 1.0);
    float dens = 0.0, nz = 0.0, wsum = 0.0, len0 = 0.0;
    for (int i = 0; i < MAX_JETS; ++i) {
        vec4 jet = texelFetch(u_data, ivec2(v_index, 4 + i), 0);
        if (jet.y <= 0.0) continue;
        float len = jet.y * grow, wid = jet.z, fi = float(i);
        if (i == 0) len0 = len;
        float ca = cos(jet.x), sa = sin(jet.x);
        float s = x * ca + y * sa;
        float t = -x * sa + y * ca;
        float u = s / len;
        float uc = clamp(u, 0.0, 1.0);
        float wob = texture(u_noise, vec2(
            s / (len * 0.45 * NOISE_CELLS) + 0.37 * fi + ofs.x,
            age * 1.5 / NOISE_CELLS + ofs.y)).g - 0.5;
        t += wob * wid * (WOBBLE_AT_BIRTH + WOBBLE_WITH_AGE * age) * uc;
        bool inside = u > 0.0 && u < 1.0;
        float prof = inside ? pow(uc, 0.45) * pow(1.0 - uc, 0.8) * 1.9 : 0.0;
        float wj = wid * max(prof, 1e-4);
        float q = t / wj;
        float body = inside ? exp(-q * q) : 0.0;
        float along = pow(clamp(1.15 - u, 0.0, 1.0), 0.6);
        float n1 = texture(u_noise, vec2(
            (s - age * len * 0.6) / (wid * 5.0 * NOISE_CELLS) + ofs.x
                + 0.21 * fi,
            t / (wid * 0.9 * NOISE_CELLS) + ofs.y)).r;
        float n2 = texture(u_noise, vec2(
            s / (wid * 1.6 * NOISE_CELLS) + ofs.y + 0.13 * fi,
            t / (wid * 0.5 * NOISE_CELLS) + ofs.x)).b;
        float n = clamp((0.6 * n1 + 0.4 * n2 - 0.5) * NOISE_CONTRAST + 0.5,
                        0.0, 1.0);
        float d = body * along;
        dens = max(dens, d * jet.w);
        nz += n * d;
        wsum += d;
    }
    nz = wsum > 1e-6 ? nz / max(wsum, 1e-6) : 0.5;
    float tear = r3.z * (TEAR_AT_BIRTH + TEAR_WITH_AGE * age);
    float body = clamp((dens - tear * nz) / max(1.0 - 0.45 * tear, 1e-3),
                       0.0, 1.0);
    body = body * body * (3.0 - 2.0 * body);
    float r = length(vec2(x, y)) / r2.x;
    float core = exp(-r * r) * pow(1.0 - age, 1.5);
    float d = clamp(max(body, core), 0.0, 1.0);

    float cool = pow(1.0 - age, 1.4);
    float fade = pow(1.0 - age, 0.8);
    float t_c = r3.y + (r3.x - r3.y) * (0.25 + 0.75 * cool);
    float kelvin = r3.y * (0.85 + 0.15 * cool) + (t_c - r3.y) * pow(d, 2.6);
    vec3 c = le_at(kelvin) * (sqrt(d) * r2.w * fade);

    float hr = length(vec2((x - 0.25 * len0) / 1.5, y));
    float h1 = hr / r2.y, h2 = hr / (0.35 * r2.y);
    float halo = exp(-h1 * h1) + 0.6 * exp(-h2 * h2);
    c += le_at(HALO_K) * (halo * r2.z * cool * fade);
    c *= win;

    if (u_stage == 1) {
        vec4 cov = smokeCoverAt(v_world_uv);
        c *= clamp(cov.r, 0.0, 1.0) * (1.0 - u_glow_through_ceil);
    }
    finalColor = vec4(c, 0.0);
}
