#version 330
// Smoke RELIEF shader for Breach (graphics arc G1,
// docs/graphics_direction_2026-10-18.md §2.1, §2.2, §2.4). RENDER-ONLY.
//
// Smoke seen from above is a relief: a height field with a lit surface. Per
// pixel of the world RT this builds the surface of ONE species group (the
// ceiling soot, or the floor haze) from the sim's tile fields and lights it
// with the sweep's light (renderer/gas_relief.py owns the pass):
//
//   coverage  C = smoothstep over ln(extinction depth), pushed by the billow
//               field B -> a narrow, billowed edge
//   height    h = H sqrt(sat(tau / tau_h)) (1 - a + a B) C, tiles: thick
//               columns stand tall, the billows bulge their tops
//   light     N from central differences of h; L = the light field's own
//               direction (toward the light) lifted by a lamp height; the
//               surface = albedo x (floor + E x wrap(N.L) / wrap(flat)), E the
//               tile's incoming light at the one exposure. Thick smoke shows
//               its lit SURFACE, thin smoke the sweep's own in-scatter: blended
//               by the column's opacity 1 - exp(-tau_ext K)
//   shadow    a horizon march toward the light through h: a crest between
//               the pixel and the lamp, above the ray, shades it
//   AO        the lanes between billows (low B) darken, crests do not
//   powder    multiple scattering brightens THIN smoke: (1 - e^-2s) e^-s,
//               peaking at an optical depth near 1/2 -- fluffy bright edges
//   rim       the silver lining: light passing THROUGH a thin edge that faces
//               away from the lamp, forward-scattered (Henyey-Greenstein)
//   inscatter the sweep's glow (god rays) laid where the smoke IS: times the
//               coverage, and textured by the billow field inside the body
//
// OUTPUT: premultiplied LINEAR HDR (inscatter, 1 - T) into the RGBA16F scene
// RT, blended out = src.rgb + dst * (1 - src.a). Never a tone-map here: the
// ONE tone-map is shaders/world_resolve.fs (tests/test_one_tonemap.py).
//
// CLOCK = SIM TICK: u_phase / u_phase_hot are frac(tick / tau_ticks) from
// renderer/advected_noise.py (float64 on the CPU), never wall time.
//
// Credit (repo rule -- cite what a file implements):
//   - Andrew Schneider & Nathan Vos, "The Real-Time Volumetric Cloudscapes
//     of Horizon Zero Dawn", SIGGRAPH 2015 Advances in Real-Time Rendering
//     -- coverage remap of a billow (Perlin-Worley) field.
//   - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 -- the two-layer
//     flow-map crossfade; Fabrice Neyret, "Advected Textures", SCA 2003.
//   - Christian Sigg & Markus Hadwiger, "Fast Third-Order Texture
//     Filtering", GPU Gems 2 ch. 20 -- cubic B-spline via 4 bilinear taps.

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;   // the group's MEDIUM: rgb = glow share, a = tau' (occlusion)
uniform sampler2D u_cover;    // rgb = surface albedo, a = extinction depth
uniform sampler2D u_dyn;      // rg = tamed wind (tiles/tick), b = boil weight,
                              // a = FLOOR group's "smoke within 2 tiles" mask
uniform sampler2D u_toplight; // rgb = the light over a column's top (gas_relief.top_light_field),
                              // a = CEILING group's "smoke within 2 tiles" mask
uniform int   u_group;        // 0 = floor, 1 = ceiling (which mask to read)
uniform sampler2D u_light_a;  // rgb = incoming light, a = light_dir.x (toward the light)
uniform sampler2D u_light_b;  // rgb = glow (unused here), a = light_dir.y
uniform sampler2D u_billow;   // R = billow B, GB = warp, A = fine domes (REPEAT)

uniform vec2  u_grid;         // (grid_w, grid_h): uv * u_grid = tiles
uniform float u_phase;        // base clock phase [0,1)
uniform float u_tau_ticks;    // base clock cycle, ticks
uniform float u_phase_hot;    // hot (boiling) clock phase [0,1)
uniform float u_tau_ticks_hot;
uniform float u_adv_gain;     // wind -> noise advection gain
uniform float u_drift_cap;    // most a layer may drift in one cycle, tiles
uniform float u_billow_tiles; // billow texture period, tiles
uniform float u_billow_mean;  // mean of B over the bake (centres the push)
uniform float u_warp_tiles;   // domain warp, tiles
uniform float u_jitter_wl;    // phase-jitter wavelength, tiles (SMOOTH: a
                              // per-pixel jitter turns the crossfade into grain)
uniform float u_cov_tau;      // extinction depth at the coverage edge
uniform float u_cov_edge;     // edge width where the field has a gradient, tiles
uniform float u_cov_soft;     // least edge width, e-folds (uniform haze stays soft)
uniform float u_cov_push;     // billow displacement of the edge, tiles
uniform float u_haze;         // share of the depth kept OUTSIDE the coverage
uniform float u_height;       // tallest column, tiles
uniform float u_height_tau;   // depth of a full-height column
uniform float u_billow_amp;   // share of the height the billows carve
uniform float u_lamp_z;       // lamp elevation (z of the unnormalised L)
uniform float u_top_light;    // share of the top light that reaches the surface
uniform float u_wrap;         // wrap lighting floor
uniform float u_light_gain;   // the one exposure ([render.lighting] master_gain)
uniform vec3  u_ambient;      // the flat floor term (LightingPass.ambient)
uniform float u_shadow;
uniform float u_shadow_len;
uniform float u_shadow_soft;
uniform float u_powder;
uniform float u_ao;
uniform float u_rim;
uniform float u_rim_g;
uniform float u_glow_texture; // how much the billows texture the in-scatter, [0,1]
uniform int   u_debug;

out vec4 finalColor;

const float TWO_PI = 6.28318530718;

// --- cubic B-spline reconstruction (Sigg-Hadwiger) ------------------------
vec4 cubicWeights(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0 / 6.0);
}

vec4 textureBicubic(sampler2D tex, vec2 texCoords, vec2 texSize) {
    vec2 invTexSize = 1.0 / texSize;
    texCoords = texCoords * texSize - 0.5;
    vec2 fxy = fract(texCoords);
    texCoords -= fxy;
    vec4 xcubic = cubicWeights(fxy.x);
    vec4 ycubic = cubicWeights(fxy.y);
    vec4 c = texCoords.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;
    offset *= invTexSize.xxyy;
    vec4 sample0 = texture(tex, offset.xz);
    vec4 sample1 = texture(tex, offset.yz);
    vec4 sample2 = texture(tex, offset.xw);
    vec4 sample3 = texture(tex, offset.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// --- the billow field, advected on the two clocks ----------------------------
// Set once per pixel in main(): the layers' advection offsets (in billow
// texture units) and crossfade weights, the warp and the boil weight. The
// neighbour taps of the height field reuse them (they lie within a tile).
vec2 g_o0, g_o1, g_h0, g_h1, g_warp;
float g_w0, g_wh0, g_boil;

float billowRaw(vec2 p) {                     // p in tiles -> B in [0,1]
    vec2 q = (p + g_warp) / u_billow_tiles;
    float b = mix(texture(u_billow, q + g_o1 + vec2(0.5)).r,
                  texture(u_billow, q + g_o0).r, g_w0);
    if (g_boil > 0.004) {
        float bh = mix(texture(u_billow, q + g_h1 + vec2(0.25, 0.75)).r,
                       texture(u_billow, q + g_h0 + vec2(0.75, 0.25)).r, g_wh0);
        b = mix(b, bh, g_boil);
    }
    return b;
}

// Coverage as a DISPLACED LEVEL SET of ln(extinction depth): the edge is
// the contour ln(tau / u_cov_tau) = 0, moved along its normal by the billow
// field u_cov_push tiles at most -- q gains |grad ln tau| * displacement.
// Where the field has a gradient (a plume's edge) the transition is
// u_cov_edge tiles wide: sharp, billowed. Where it is flat (a room of thin
// uniform haze at the threshold) the gradient -> 0, so the billows displace
// nothing and the width falls back to u_cov_soft e-folds: an even haze, never
// a leopard of flecks (G1's first plume shots cut exactly that).
const float GRAD_MAX = 8.0;          // |grad ln tau| clamp, e-folds per tile
float lnTauAt(vec2 p) {          // bilinear: the gradient is tile-scale
    return log(max(texture(u_cover, p / u_grid).a, 1e-6));
}
// Returns (C, C_h): the coverage, and the softer coverage the HEIGHT rides
// (H_SOFT x wider) -- a cloud's rim is rounded; a height that fell as sharply
// as C cut near-vertical walls that read as black creases.
const float H_SOFT = 3.0;
vec2 coverageOf(float ln_tau, float grad, float b) {
    float q = ln_tau - log(u_cov_tau) + u_cov_push * grad * (b - u_billow_mean);
    float w = max(u_cov_edge * grad, u_cov_soft);
    return vec2(smoothstep(-0.5 * w, 0.5 * w, q),
                smoothstep(-0.5 * H_SOFT * w, 0.5 * H_SOFT * w, q));
}

// The RELIEF HEIGHT at p (tiles), given the local |grad ln tau| (taken at
// the pixel: the field is tile-scale, the taps sub-tile).
float heightOf(float t, vec2 p, float grad) {
    t = max(t, 0.0);
    float b = billowRaw(p);
    float c = coverageOf(log(max(t, 1e-6)), grad, b).y;
    return u_height * sqrt(clamp(t / u_height_tau, 0.0, 1.0))
         * (1.0 - u_billow_amp + u_billow_amp * b) * c;
}
// Bicubic depth for the normal taps (they must agree with the pixel's own
// reconstruction), bilinear for the shadow march (soft by nature).
float heightAt(vec2 p, float grad) {
    return heightOf(textureBicubic(u_cover, p / u_grid, u_grid).a, p, grad);
}
float heightAtFast(vec2 p, float grad) {
    return heightOf(texture(u_cover, p / u_grid).a, p, grad);
}

void main() {
    vec2 uv = fragTexCoord;
    vec2 tp = uv * u_grid;

    // Early out on the group's mask (1 where this group has smoke within two
    // tiles, CPU-dilated past the bicubic's reach): one bilinear tap, and
    // an empty stretch of ship costs nothing more.
    vec4 dyn = texture(u_dyn, uv);
    vec4 top = texture(u_toplight, uv);
    if ((u_group == 1 ? top.a : dyn.a) <= 0.0) discard;

    vec4 cov = textureBicubic(u_cover, uv, u_grid);
    float tau_ext = max(cov.a, 0.0);
    if (tau_ext <= 1e-5) discard;
    float ln_tau = log(max(tau_ext, 1e-6));

    // --- per-pixel clock state (Vlachos two-layer crossfade, two clocks) ---
    vec2 wind = dyn.rg;                                   // tiles / tick
    g_boil = clamp(dyn.b, 0.0, 1.0);
    // Phase jitter: a smooth periodic fBm (the bake's B channel) at a coarse
    // scale, stretched over two cycles -- neighbouring pixels share nearly
    // the same phase, distant ones do not (no whole-screen pulse, no grain).
    float jph = 2.0 * texture(u_billow, tp / max(u_jitter_wl, 1e-3)
                                        + vec2(0.37, 0.71)).b;
    // The DRIFT CAP: a layer's whole-cycle displacement |wind| * tau is
    // limited to u_drift_cap tiles. Uncapped, a blast's wind (up to ~1.4
    // tiles/tick on the tamed scale) carries a layer ~80 tiles in one 60-tick
    // cycle, and any spatial change in wind or phase shears the noise into
    // streaks and rings (seen in G1's first plume shots). Capped, slow air
    // drifts at its own speed and fast air drifts at the cap; the sim's
    // plume still moves at the real speed underneath.
    vec2 v = wind * u_adv_gain;
    vec2 v_base = v * min(1.0, u_drift_cap / max(length(v) * u_tau_ticks, 1e-6));
    vec2 v_hot = v * min(1.0, u_drift_cap / max(length(v) * u_tau_ticks_hot, 1e-6));
    float ph = fract(u_phase + jph);
    g_w0 = 0.5 - 0.5 * cos(TWO_PI * ph);
    g_o0 = -v_base * (ph * u_tau_ticks) / u_billow_tiles;
    g_o1 = -v_base * (fract(ph + 0.5) * u_tau_ticks) / u_billow_tiles;
    float phh = fract(u_phase_hot + jph);
    g_wh0 = 0.5 - 0.5 * cos(TWO_PI * phh);
    g_h0 = -v_hot * (phh * u_tau_ticks_hot) / u_billow_tiles;
    g_h1 = -v_hot * (fract(phh + 0.5) * u_tau_ticks_hot) / u_billow_tiles;
    g_warp = (texture(u_billow, tp / (u_billow_tiles * 2.0)).gb - 0.5)
           * 2.0 * u_warp_tiles;

    // --- coverage: the sharp, billowed edge --------------------------------
    const float GE = 0.25;           // gradient step, tiles (tile-scale field)
    vec2 g2 = vec2(lnTauAt(tp + vec2(GE, 0.0)) - lnTauAt(tp - vec2(GE, 0.0)),
                   lnTauAt(tp + vec2(0.0, GE)) - lnTauAt(tp - vec2(0.0, GE)))
            / (2.0 * GE);
    float grad = min(length(g2), GRAD_MAX);
    float B = billowRaw(tp);
    vec2 CC = coverageOf(ln_tau, grad, B);
    float C = CC.x;

    vec4 med = textureBicubic(texture0, uv, u_grid);
    float tau_abs = max(med.a, 0.0);
    vec3 glow = max(med.rgb, vec3(0.0));

    if (u_debug == 1) { finalColor = vec4(vec3(C), 1.0); return; }

    // --- the relief height ---------------------------------------------------
    float h = u_height * sqrt(clamp(tau_ext / u_height_tau, 0.0, 1.0))
            * (1.0 - u_billow_amp + u_billow_amp * B) * CC.y;
    if (u_debug == 2) {
        finalColor = vec4(vec3(h / max(u_height, 1e-6)), 1.0); return;
    }

    // --- N from the height field (central differences, NE tiles apart) ----
    const float NE = 1.0 / 16.0;
    float hxp = heightAt(tp + vec2(NE, 0.0), grad);
    float hxm = heightAt(tp - vec2(NE, 0.0), grad);
    float hyp = heightAt(tp + vec2(0.0, NE), grad);
    float hym = heightAt(tp - vec2(0.0, NE), grad);
    vec3 N = normalize(vec3(-(hxp - hxm), -(hyp - hym), 2.0 * NE));

    // --- L: the light field's direction, lifted by the lamp height --------
    vec4 la = texture(u_light_a, uv);
    vec4 lb = texture(u_light_b, uv);
    vec2 dir2d = vec2(la.a, lb.a);                 // toward the light, |.| <= 1
    vec3 L = normalize(vec3(dir2d, u_lamp_z));
    // E: the light on the surface. The tile's own light is what crossed
    // the smoke sideways; a column's top also sees the light passing over
    // it (the top-light field), by u_top_light.
    vec3 E_tile = max(la.rgb, vec3(0.0));
    vec3 E_top = max(top.rgb, E_tile);
    vec3 E = mix(E_tile, E_top, u_top_light) * u_light_gain;

    // Wrap lighting, normalised to a FLAT top: a level smoke deck under a
    // lamp reads albedo x E, crests facing the lamp brighter, the far
    // flanks darker -- the shading moves light around, it does not add any.
    float wrapNL = max((dot(N, L) + u_wrap) / (1.0 + u_wrap), 0.0);
    float wrapFlat = max((L.z + u_wrap) / (1.0 + u_wrap), 1e-3);
    float shade = wrapNL / wrapFlat;
    if (u_debug == 3) { finalColor = vec4(vec3(0.5 * shade), 1.0); return; }

    // --- self-shadow: a horizon march toward the light ---------------------
    // SHADOW_STEPS taps along the light's planar direction, out to
    // u_shadow_len tiles; the ray climbs at the lamp's elevation. Any crest
    // that stands above the ray (by u_shadow_soft for a full shadow) shades
    // this pixel's direct light -- lamp-facing crests stay bright, the
    // crevices behind them go dark. Overhead light (no planar direction)
    // casts nothing.
    const int SHADOW_STEPS = 6;
    float lit = 1.0;
    float dlen = length(dir2d);
    if (u_shadow > 0.0 && dlen > 0.05 && h > 0.0) {
        vec2 d = dir2d / dlen;
        float rise = L.z / max(length(L.xy), 1e-3);   // tan(elevation)
        float occ = 0.0;
        for (int k = 1; k <= SHADOW_STEPS; ++k) {
            float sk = u_shadow_len * float(k) / float(SHADOW_STEPS);
            float hq = heightAtFast(tp + d * sk, grad);
            occ = max(occ, clamp((hq - (h + sk * rise))
                                 / max(u_shadow_soft, 1e-4), 0.0, 1.0));
        }
        lit = 1.0 - u_shadow * occ;
    }
    if (u_debug == 4) { finalColor = vec4(vec3(lit), 1.0); return; }

    // --- AO: the lanes between billows ---------------------------------------
    // The billow field already says where the crevices are: B below its
    // mean is a lane between lumps, the deeper the darker. (A Laplacian of
    // h at the normal taps' scale caught the Worley cells' crease LINES and
    // drew a cracked-mud net over the smoke -- G1 step 5's first shot.)
    float lane = clamp((u_billow_mean - B) / max(u_billow_mean, 1e-3), 0.0, 1.0);
    float ao = 1.0 - u_ao * lane * u_billow_amp;
    if (u_debug == 5) { finalColor = vec4(vec3(ao), 1.0); return; }

    vec3 albedo = max(cov.rgb, vec3(0.0));
    vec3 surface = albedo * (u_ambient + E * shade * lit) * ao;

    // The cut keeps a HAZE share of the depth outside the edge: thin smoke
    // around a plume (and a room of it) still dims the floor, and the edge
    // reads as the step from haze to body.
    float K = u_haze + (1.0 - u_haze) * C;
    float alpha = 1.0 - exp(-tau_abs * K);
    // Thick smoke shows its surface, thin smoke the sweep's in-scatter.
    float thick = 1.0 - exp(-tau_ext * K);
    // In-scatter where the smoke is: the coverage K, and inside it the
    // billow field's sub-tile density D (mean ~1 over a tile), so a beam
    // through haze is textured by the smoke it crosses, not a flat tile.
    float D = max(1.0 + u_glow_texture * (B - u_billow_mean)
                  / max(1.0 - u_billow_mean, 1e-3), 0.0);
    vec3 rgb = surface * thick + glow * u_light_gain * K * D * (1.0 - thick);
    float sig = tau_ext * K;
    // Rim (silver lining): where the surface faces AWAY from the light and
    // the smoke is thin, light crosses the edge and leaves along N, forward-
    // scattered: Henyey-Greenstein at cos = dot(-L, N), normalised to 1 at
    // full forward. Its strength is the SINGLE-scatter albedo w (inverted
    // from the bulk reflectance in u_cover), so soot barely glows, steam
    // does; thin = 4 e^-s (1 - e^-s), 1 at s = ln 2.
    float sR = (1.0 - albedo.g) / (1.0 + albedo.g);
    float w_ss = 1.0 - sR * sR;
    float g = u_rim_g;
    float hg = pow(1.0 - g, 3.0)
             / pow(max(1.0 + g * g - 2.0 * g * dot(-L, N), 1e-4), 1.5);
    float thin = 4.0 * exp(-sig) * (1.0 - exp(-sig));
    rgb += u_rim * w_ss * hg * thin * lit * E;
    // Powder: multiple scattering lifts thin smoke (Schneider & Vos's
    // powder, used here as a brightening of the thin edge, §2.2); the
    // bump (1 - e^-2s) e^-s peaks at 0.385 near s = ln(3)/2.
    float powder = (1.0 - exp(-2.0 * sig)) * exp(-sig) / 0.385;
    rgb *= 1.0 + u_powder * powder;

    finalColor = vec4(max(rgb, vec3(0.0)), clamp(alpha, 0.0, 1.0));
}
