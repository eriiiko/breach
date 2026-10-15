#version 330
// Volumetric smoke for the tilted view (issue #90) — THE raymarcher.
// Design: docs/volumetric_smoke_design_2026-10-14.md §2.4. RENDER-ONLY.
//
// Per pixel: reconstruct the view ray from the inverse view-projection, clip
// it to the smoke slab (0 <= y_up <= h_max over the world rectangle), and
// march it front-to-back. Per sample:
//   tau       = the sim's per-tile optical depth (bilinear; the SAME number
//               renderer/gas_medium.py builds — single source of scale);
//   (d, f)    = the baked detail volume (smoke, fire), trilinear + wrap, looked
//               up at world tiles / period and height / height_tiles, two
//               frames lerped on the sim-tick phase;
//   column    = tau grows a column of height h(tau); the detail erodes it
//               Nubis-style (solid cores, wispy edges) and ragged-tops it;
//   sigma     = tau * coverage / h  (so the vertical integral through a
//               column ~ tau: a slanted ray sees MORE, as it should);
//   lit       = (ambient + light_field * gain) * albedo * vertical self-shadow
//   emission  = the blackbody overlay's colour glowing up from the floor,
//               modulated by the fire channel.
// Output is PREMULTIPLIED (ACES(inscatter), 1 - transmittance), composited
// with BLEND_ALPHA_PREMULTIPLY over the ground — the gas medium's operator.
//
// Credit (repo rule — cite what a file implements):
//   - Pharr, Jakob & Humphreys, "Physically Based Rendering", ch. Volume
//     Scattering — Beer-Lambert transmittance, emission/absorption march.
//   - A. Schneider & N. Vos, "The Real-time Volumetric Cloudscapes of Horizon:
//     Zero Dawn" (Nubis), SIGGRAPH 2015 — coarse coverage eroded by a tiled
//     3D detail texture, remap strongest at low density.
//   - Sebastien Hillaire, "Physically Based and Unified Volumetric Rendering
//     in Frostbite", SIGGRAPH 2015 — front-to-back scatter/transmittance.
//   - ACES filmic (Narkowicz) — the same curve every Breach layer applies.

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D u_tau;      // RGBA16F grid-res: R = optical depth (pre-curve),
                              // GBA = tau-weighted scatter albedo of the tile
uniform sampler2D u_light;    // light_tex_a RGBA16F: rgb = incoming light (HDR)
uniform sampler2D u_light_b;  // light_tex_b RGBA16F: rgb = the ray march's
                              // inscatter deposit (smoke_glow: the light the
                              // gas ABSORBED, per channel, already x albedo)
uniform sampler2D u_heat;     // HeatFieldOverlay RGBA8: rgb hue, a intensity
uniform sampler3D u_volume;   // RG8 (X, Y, (Z+1)*T): r = smoke, g = fire
uniform sampler2D u_field;    // RGBA16F grid-res: RG = (cos, sin) of the tile's
                              // PHASE OFFSET (renderer/volume_phase.py) — the
                              // bilinear fetch interpolates it between tile
                              // centres, so every pixel has its own moment in
                              // the loop and the movie's repeat decorrelates

uniform mat4  u_inv_vp;       // inverse (projection * view) of the tilt camera
uniform vec3  u_cam_pos;      // world px (X = x_wpx, Y = up, Z = y_wpx)
uniform vec2  u_world_px;     // (W, H) of the world RT in px
uniform float u_px_per_tile;
uniform vec2  u_vol_zt;       // (Z slices per frame, T frames)
uniform float u_phase;        // frac(tick / loop_ticks) — the sim-tick clock
uniform float u_period_tiles; // volume x/y repeat, in tiles
uniform float u_height_tiles; // volume z extent, in tiles
uniform float u_rise_turns;   // vertical scroll per loop, whole heights (seamless)

uniform int   u_steps;        // CAP on the steps per ray
uniform float u_step_px;      // step length, world px (= step_texels texels)
uniform float u_h_max;        // slab / max column height, tiles
uniform float u_tau_h;        // tau at which the column reaches u_h_max
uniform float u_tau_solid;    // tau at which erosion stops (solid core)
uniform float u_erode;        // erosion depth at zero solidity [0,1]
uniform float u_detail_gamma; // contrast on the smoke channel
uniform float u_top_var;      // column-top variation by detail [0,1]
uniform float u_k_ext;        // extinction multiplier (1 = honest tau)
uniform float u_k_shadow;     // vertical self-shadow strength
uniform int   u_light_from_below; // 1 = lamps at floor height (bright bases)
uniform float u_albedo_floor; // artistic floor under the per-tile albedo (soot
                              // is 0.04 in the table — pure black in any light)
uniform vec3  u_ambient;      // the ship's ambient floor
uniform float u_light_gain;   // the ship's master exposure on the light field
uniform float u_glow_gain;    // [render.gas_medium] glow_gain — the SAME dial
                              // the 2D medium exposes its inscatter with
uniform float u_glow_scale;   // the volume's own factor on top of it (the
                              // deposit is per tile, the column spreads it)
uniform float u_emit_gain;    // fire glow strength
uniform float u_emit_pow;     // fire channel contrast
uniform int   u_heat_on;      // 0 when the blackbody overlay is off
uniform float u_exposure;     // final inscatter exposure before ACES
// Directional light (Erik 2026-10-15): the lamp DIRECTION is the gradient of
// the baked light field at the ray's ground hit (per pixel; lamps are far
// against a column), the smoke NORMAL is the gradient of the baked density
// (three extra volume samples per lit sample). No new raycast.
uniform float u_dir_light;    // lamp-facing side brightens (1+a), the back
                              // darkens (1-a); 0 = off (and no extra samples)
uniform float u_dir_eps;      // normal finite-difference step, in TILES
uniform float u_hg_g;         // Henyey-Greenstein asymmetry (0 = isotropic)
uniform float u_hg_mix;       // how much of the phase function is applied
uniform int   u_debug;        // 1 = paint the ground hit (alignment check)

out vec4 finalColor;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Henyey-Greenstein relative to isotropic: 1 at g = 0.
float hg_phase(float g, float cos_t) {
    float g2 = g * g;
    return (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// The LOCAL loop phase at ground uv g: the global sim-tick clock plus the
// tile field's interpolated offset (unit-vector encoded so the interpolation
// crosses the wrap on the short arc). Continuous over the floor -> no seams.
float localPhase(vec2 g) {
    vec2 cs = texture(u_field, g).rg;
    float off = atan(cs.y, cs.x) * 0.15915494309;  // / 2 pi
    return fract(u_phase + off);
}

// Trilinear + wrap sample of the volume at tile position (tx, ty), height
// hz in volume-height units and loop phase `phase`; z and t wrap in the
// shader (each frame carries a duplicate top slice so the in-frame trilinear
// wrap is exact).
vec2 sampleVolume(vec2 txy, float hz, float phase) {
    float Z = u_vol_zt.x;
    float T = u_vol_zt.y;
    float D = (Z + 1.0) * T;
    vec2 st = txy / u_period_tiles;                // GL_REPEAT in hardware
    float z01 = fract(hz);
    float f = phase * T;
    float f0 = floor(f);
    float fw = f - f0;
    float f1 = mod(f0 + 1.0, T);
    float r0 = (f0 * (Z + 1.0) + z01 * Z + 0.5) / D;
    float r1 = (f1 * (Z + 1.0) + z01 * Z + 0.5) / D;
    vec2 a = texture(u_volume, vec3(st, r0)).rg;
    vec2 b = texture(u_volume, vec3(st, r1)).rg;
    return mix(a, b, fw);
}

// Ray / axis-aligned box (0,0,0)-(W,hmax,H) — returns (t_near, t_far).
vec2 boxHit(vec3 ro, vec3 rd, vec3 bmax) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (vec3(0.0) - ro) * inv;
    vec3 t1 = (bmax - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z),
                min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
    // --- view ray from the inverse view-projection -------------------------
    // fragTexCoord (0,0) is the drawn rectangle's top-left in screen
    // convention, which is NDC (-1, +1).
    vec2 ndc = vec2(fragTexCoord.x * 2.0 - 1.0, 1.0 - fragTexCoord.y * 2.0);
    vec4 pn = u_inv_vp * vec4(ndc, -1.0, 1.0);
    vec4 pf = u_inv_vp * vec4(ndc,  1.0, 1.0);
    vec3 p_near = pn.xyz / pn.w;
    vec3 p_far  = pf.xyz / pf.w;
    vec3 rd = normalize(p_far - p_near);
    vec3 ro = u_cam_pos;

    float hmax_px = u_h_max * u_px_per_tile;
    vec3 bmax = vec3(u_world_px.x, hmax_px, u_world_px.y);
    float inv_tiles = 1.0 / u_px_per_tile;
    vec2 th = boxHit(ro, rd, bmax);
    float tn = max(th.x, 0.0);
    float tf = th.y;
    if (tf <= tn) {
        finalColor = vec4(0.0);
        return;
    }

    if (u_debug == 1) {
        // Ground hit painted as a tile checker — must sit exactly on the
        // ground quad's own grid (alignment oracle for the camera matrices).
        float tg = (0.0 - ro.y) / rd.y;
        vec3 g = ro + rd * tg;
        vec2 tile = floor(g.xz / u_px_per_tile);
        float chk = mod(tile.x + tile.y, 2.0);
        finalColor = vec4(chk * 0.5, 0.0, (1.0 - chk) * 0.5, 0.5);
        return;
    }

    // Step length in WORLD PX from the dial in VOLUME TEXELS (u_step_px),
    // so the march resolves whatever the bake contains; the count follows
    // the ray's own slab length, capped by u_steps.
    float seg = tf - tn;
    int nsteps = int(clamp(ceil(seg / max(u_step_px, 1e-3)), 4.0, float(u_steps)));
    float ds = seg / float(nsteps);
    float t = tn + hash12(gl_FragCoord.xy) * ds;

    if (u_debug == 2) {
        // MOVIE ONLY: the whole slab filled with the baked volume under flat
        // grey light — "show me just the movie", in the game's own view.
        float trans2 = 1.0;
        float acc = 0.0;
        for (int i = 0; i < nsteps; ++i) {
            vec3 p = ro + rd * t;
            t += ds;
            float yt = p.y * inv_tiles;
            float ph = localPhase(p.xz / u_world_px);
            float rise = u_rise_turns * ph;        // whole heights per loop: seamless
            vec2 vol = sampleVolume(p.xz * inv_tiles, yt / u_height_tiles - rise, ph);
            float d = pow(vol.r, u_detail_gamma);
            float sigma = u_k_ext * 2.0 * d / max(u_h_max * u_px_per_tile, 1e-3);
            float a = 1.0 - exp(-sigma * ds);
            acc += trans2 * a * (0.35 + 0.65 * d);
            trans2 *= (1.0 - a);
            if (trans2 < 0.01) break;
        }
        finalColor = vec4(vec3(acc), 1.0 - trans2);
        return;
    }

    // --- lamp direction at the ground hit (once per pixel) -----------------
    // The light field's relative gradient over one tile: its direction is
    // where the lamp is, its size (against the local level) says how
    // directional the light is at all (0 in flat ambient -> the term fades).
    vec3 L = vec3(0.0, 1.0, 0.0);
    float Lw = 0.0;
    if (u_dir_light > 0.0 || u_hg_mix > 0.0) {
        float tg = (abs(rd.y) > 1e-5) ? (0.0 - ro.y) / rd.y : tn;
        vec3 gp = ro + rd * clamp(tg, tn, tf);
        vec2 g0 = gp.xz / u_world_px;
        vec2 du = vec2(u_px_per_tile) / u_world_px;
        float lc = lum(texture(u_light, g0).rgb);
        float lx = lum(texture(u_light, g0 + vec2(du.x, 0.0)).rgb)
                 - lum(texture(u_light, g0 - vec2(du.x, 0.0)).rgb);
        float lz = lum(texture(u_light, g0 + vec2(0.0, du.y)).rgb)
                 - lum(texture(u_light, g0 - vec2(0.0, du.y)).rgb);
        vec2 grad = vec2(lx, lz) * 0.5;              // per tile
        float gl = length(grad);
        Lw = clamp(gl / max(lc, 1e-4) * 1.5, 0.0, 1.0);
        if (gl > 1e-6) {
            vec2 lxz = grad / gl;
            float ly = (u_light_from_below == 1) ? -0.4 : 0.4;
            L = normalize(vec3(lxz.x, ly, lxz.y));
        }
    }
    if (u_debug == 4) {
        // HEAT INPUT: the blackbody overlay's colour x intensity at the
        // ground hit (what the emission term multiplies the fire channel by).
        float tg4 = (abs(rd.y) > 1e-5) ? (0.0 - ro.y) / rd.y : tn;
        vec3 gp4 = ro + rd * clamp(tg4, tn, tf);
        vec4 h4 = texture(u_heat, gp4.xz / u_world_px);
        finalColor = vec4(h4.rgb * h4.a, 0.9);
        return;
    }
    if (u_debug == 3) {
        // LAMP DIRECTION: rg = the direction toward the lamp (x, z) mapped
        // to 0..1, brightness = how directional the light is here (Lw).
        finalColor = vec4((L.x * 0.5 + 0.5) * Lw, (L.z * 0.5 + 0.5) * Lw, 0.0, 0.85);
        return;
    }
    float hg = 1.0;
    if (u_hg_mix > 0.0) {
        // light travels lamp -> sample (-L), then sample -> camera (-rd):
        // cos(theta) between the two propagation directions = dot(L, rd).
        hg = mix(1.0, hg_phase(u_hg_g, dot(L, rd)), u_hg_mix * Lw);
    }

    vec3 inscatter = vec3(0.0);
    float trans = 1.0;

    for (int i = 0; i < nsteps; ++i) {
        vec3 p = ro + rd * t;
        t += ds;
        vec2 g = p.xz / u_world_px;                  // ground uv
        vec4 tau4 = texture(u_tau, g);
        float tau = tau4.r;
        if (tau <= 1e-4) continue;
        vec3 albedo = max(tau4.gba, vec3(u_albedo_floor));

        float yt = p.y * inv_tiles;                  // height in tiles
        float s = clamp(tau / u_tau_solid, 0.0, 1.0);            // solidity
        float h = u_h_max * sqrt(clamp(tau / u_tau_h, 0.0, 1.0)); // column h
        if (yt > h * (1.0 + u_top_var)) continue;

        vec2 txy = p.xz * inv_tiles;
        float ph = localPhase(g);
        float rise = u_rise_turns * ph;            // whole heights per loop: seamless
        vec2 vol = sampleVolume(txy, yt / u_height_tiles - rise, ph);
        float d = pow(vol.r, u_detail_gamma);

        // Nubis-style erosion: cores keep coverage, edges get carved.
        float e = u_erode * (1.0 - s);
        float coverage = clamp((d - e) / max(1.0 - e, 1e-3), 0.0, 1.0);
        // Ragged top: the column's height breathes with the detail.
        float h_eff = h * (1.0 - u_top_var + 2.0 * u_top_var * d);
        float top = 1.0 - smoothstep(0.65 * h_eff, h_eff, yt);
        float dens = coverage * top;
        if (dens <= 1e-4) continue;

        // Extinction per world px: the vertical integral over h gives ~tau.
        float sigma = u_k_ext * tau * dens / max(h * u_px_per_tile, 1e-3);
        float a = 1.0 - exp(-sigma * ds);

        // Lighting: the baked field (what reaches the tile THROUGH the gas)
        // times the tile's albedo, PLUS the ray march's inscatter deposit
        // (the light the gas absorbed on the way — the god-ray glow, already
        // albedo-weighted), both under the vertical Beer-Lambert self-shadow
        // (fraction of the column above this sample).
        // Breach's lamps sit at floor / wall height and the baked field is
        // what reaches the TILE, so by default the column is lit from BELOW:
        // the smoke under a sample shadows it (bright bases, dark tops — a
        // fire's smoke in a dark room). u_light_from_below = 0 flips to the
        // sky-lit case (bright tops) for planetside scenes.
        vec3 lin = u_ambient + texture(u_light, g).rgb * u_light_gain;
        vec3 glow = texture(u_light_b, g).rgb * u_glow_gain * u_glow_scale;
        float frac_h = clamp(yt / max(h, 1e-3), 0.0, 1.0);
        float above = 1.0 - frac_h;
        float occl = (u_light_from_below == 1) ? frac_h : above;
        float shadow = exp(-u_k_shadow * tau * occl);

        // Directional shading: the baked density's gradient is the billow's
        // normal; the lamp-facing side brightens, the back darkens (wrap-lit
        // so thin smoke never goes black), faded by how directional the
        // light is here (Lw).
        float dshade = 1.0;
        if (u_dir_light > 0.0 && Lw > 0.0) {
            float hz = yt / u_height_tiles - rise;
            float e = u_dir_eps;
            float eh = e / u_height_tiles;
            float dx = sampleVolume(txy + vec2(e, 0.0), hz, ph).r - vol.r;
            float dzz = sampleVolume(txy + vec2(0.0, e), hz, ph).r - vol.r;
            float dh = sampleVolume(txy, hz + eh, ph).r - vol.r;
            vec3 grad_d = vec3(dx, dh, dzz);
            float gm = length(grad_d);
            if (gm > 1e-5) {
                vec3 n = -grad_d / gm;                 // dense -> thin
                float ndl = dot(n, L);
                float wrap = clamp((ndl + 0.5) / 1.5, 0.0, 1.0);
                dshade = mix(1.0, mix(1.0 - u_dir_light, 1.0 + u_dir_light, wrap), Lw);
            }
        }
        vec3 lit = (lin * albedo * dshade * hg + glow) * shadow;

        // Fire glow from the floor, through the fire channel.
        vec3 emis = vec3(0.0);
        if (u_heat_on == 1) {
            vec4 heat = texture(u_heat, g);
            float fire = pow(vol.g, u_emit_pow);
            emis = heat.rgb * heat.a * fire * above * above * u_emit_gain;
        }

        inscatter += trans * a * (lit + emis);
        trans *= (1.0 - a);
        if (trans < 0.01) break;
    }

    vec3 rgb = aces_tonemap(inscatter * u_exposure);
    finalColor = vec4(rgb, 1.0 - trans);
}
