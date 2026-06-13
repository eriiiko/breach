#version 330
// Water surface optics pass for Breach (graphics/water_rendering.md §2, §7).
// CORE look (steps 1+2): perturbed-ripple normal + Schlick Fresnel + GGX glints
// (reusing the light buffer) + normal-driven refraction × depth + Beer-Lambert
// depth tint. MOOD PASS (step 3, phase 2, ADDED below the core and dormant-
// safe): real-surface caustics × light (§3, SIGN load-bearing), foam port
// (§4-foam, composited last), chromatic aberration, and the wave-size dials
// (u_wave_scale / u_ambient_amp). Matcap (§6) + SSR remain out.
//
// This is a SEPARATE pass from lighting.fs (§5): it draws a full-RT quad in
// the WaterFieldOverlay compose slot, AFTER the lit ship, BEFORE units, and
// REPLACES the CPU-tinted WaterFieldOverlay placeholder.
//
// DORMANT-SAFE (the hard requirement): the pass outputs a PREMULTIPLIED vec4
// with alpha = 0 on every DRY tile (water_depth == 0). The world RT composites
// premultiplied (out = rt.rgb + bg*(1-rt.a)); alpha 0 + rgb 0 is a no-op, so a
// ship with no standing water renders bit-identically to no pass at all.
//
// Input textures (all reused from the lighting pass except u_water):
//   u_diffuse  - RGB UNLIT ship art (sRGB PNG), sampled at fragTexCoord
//                (raylib binds texture0 here implicitly).
//   u_light_a  - RGBA16F light field A: RGB = incoming light colour,
//                A = light_dir.x (signed). (lighting.fs:100,104)
//   u_light_b  - RGBA16F light field B: RGB = smoke_glow (unused here),
//                A = light_dir.y (signed). (lighting.fs:101,104)
//   u_water    - RGBA16F water field (renderer/water.py packing):
//                R = ripple height (m), G = ripple_v (reserved, unused here),
//                B = water_depth (m), A = foam/agitation (reserved, unused).
//
// Uniforms ([graphics.water] config block, bound in renderer/water.py):
//   u_roughness_base       GGX roughness floor (still puddles glint sharp)
//   u_roughness_agitation  added roughness per unit local ripple energy
//   u_fog_density          Beer-Lambert extinction (depth tint rate)
//   u_refract_strength     normal-driven floor-warp magnitude (UV units)
//   u_r0                   Fresnel reflectance at normal incidence (~0.02)
//   u_water_color          deep-water tint colour (the volume colour at depth)
//   u_light_z              light-direction z (shared with lighting.fs)
//   u_srgb_decode          1 = decode sRGB diffuse to linear, re-encode out
//   u_ripple_scale         metres-of-ripple -> normal-slope gain
//   u_texel                1/grid (x,y): neighbour tap offset for the gradient
//   u_art_uv_rect          art-UV subrect (matches lighting.fs; default 0,0,1,1)
//   u_time                 render animation clock (s) for the ambient sines
//   u_glint_strength       ADDITIVE GGX-glint HDR multiplier (light off surface)
//   u_alpha_scale          depth->opacity rate (alpha = clamp(depth*scale,...))
//   u_alpha_min            shoreline alpha floor   (transparency dial)
//   u_alpha_max            deep-water alpha ceiling (transparency dial)

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D u_diffuse;
uniform sampler2D u_light_a;
uniform sampler2D u_light_b;
uniform sampler2D u_water;
// Optional per-pixel floor HEIGHTMAP (greyscale relief, 0..1), sampled in art
// space (fragTexCoord, like u_diffuse). ADJUSTS THE PER-PIXEL DEPTH the renderer
// sees: depth = tileDepth - relief*u_height_scale, so raised floor relief makes
// the local water shallower. ALL depth-driven effects (refraction offset,
// Beer-Lambert depth tint, the alpha ramp) compute off this adjusted depth, and
// where it goes <= 0 the floor pokes above the surface (the feature protrudes,
// the water laps around it with a soft u_height_edge shoreline). u_has_height = 0
// (no level heightmap) or u_height_scale == 0 forces relief = 0 so the per-pixel
// depth == the per-tile depth EXACTLY — bit-identical, the dormant default.
uniform sampler2D u_height;

uniform float u_roughness_base;
uniform float u_roughness_agitation;
uniform float u_fog_density;
uniform float u_refract_strength;
uniform float u_r0;
uniform vec3  u_water_color;
uniform float u_light_z;
uniform int   u_srgb_decode;
uniform float u_ripple_scale;
uniform vec2  u_texel;
uniform vec4  u_art_uv_rect;
uniform float u_time;
// Glint strength: the ADDITIVE HDR multiplier on the GGX highlight. The glint
// is light reflecting OFF the surface, so it is added on top of the (see-
// through, premultiplied) base — NOT Fresnel-blended at ~2% nor alpha-
// attenuated. Already gated by × lightRGB (only under a source) and × NdotL
// (no back-lit glint). Default ~2.0 makes glints clearly visible on a flashlit
// rippled puddle. ([graphics.water] glint_strength)
uniform float u_glint_strength;
// Alpha (transparency) ramp dials: alpha = clamp(depth*u_alpha_scale,
//   u_alpha_min, u_alpha_max). u_alpha_scale = depth->opacity rate (old 6.0),
//   u_alpha_min = shoreline floor (old 0.15), u_alpha_max = deep ceiling
//   (old 0.95). The glint does NOT raise alpha — it survives the premultiplied
//   blit as additive light regardless of how transparent the water is.
uniform float u_alpha_scale;
uniform float u_alpha_min;
uniform float u_alpha_max;
// --- Phase 2 (mood pass) uniforms ------------------------------------------
// Caustics (§3): focused light on the floor where the surface is concave-up.
//   caustic ~ +laplacian(ripple) (clamp negatives to 0) * lightRGB * strength,
//   ADDED into the refraction/floor term (rides the floor, × alpha). The SIGN
//   is load-bearing: +laplacian focuses (bright in dips), -div(N) ≡ same; a +
//   div(N) would invert it (bright on bumps). u_caustic_scale drives an
//   optional high-frequency procedural detail modulated by surface energy.
uniform float u_caustic_strength;
uniform float u_caustic_scale;
// Foam (§4-foam, port of overlays.py:261-277): white-ish whitecaps where the
// ripple front is steep (|grad ripple| > threshold) + the wet/dry shoreline
// (high depth gradient). Composited LAST into the base (surface scatter).
uniform float u_foam_threshold;
uniform float u_foam_intensity;
// Chromatic aberration (subtle polish): the floor refraction r/g/b are sampled
// at slightly different offsets (scaled 1 ± u_ca_amount per channel). Tiny.
uniform float u_ca_amount;
// Wave-size dials (Erik: idle shimmer waves read "very big"). u_wave_scale
// MULTIPLIES the ambient-sine spatial frequencies (higher = smaller/tighter
// waves); u_ambient_amp scales the idle amplitude base (the old 0.06 term).
uniform float u_wave_scale;
uniform float u_ambient_amp;
// Global ambient light (matches lighting.fs's u_ambient). The dry ship is lit
// by `diffuse * (ambient + sources)`; the water body must be lit the same way,
// else the refracted floor goes black outside any raycast source and shallow
// water reads as a void only visible inside the flashlight beam. Pushed every
// frame from the LightingPass's ambient (renderer/game_renderer.py) so the
// demo's ambient sliders drive the water ambient too.
uniform vec3  u_ambient;
// Heightmap depth-adjust (graphics/water_rendering.md §2 §8). The per-pixel
// relief = texture(u_height, fragTexCoord).r raises the floor, REDUCING the
// per-pixel water depth the whole shader sees:
//   depth = tileDepth - relief * u_height_scale
// so refraction/tint/alpha all vary per-pixel with it, and where depth <= 0 the
// feature protrudes (transparent) with a soft u_height_edge-wide lap shoreline.
//   u_has_height     1 if the level supplied a heightmap, else 0 (-> relief = 0)
//   u_height_scale   relief metres of floor lift subtracted from depth (0
//                    disables -> relief_term = 0); ~0.4 so furniture pokes
//                    through a ~0.3-0.5 m puddle
//   u_height_edge    soft protrusion-shoreline width (m) — the smoothstep ramp
//                    over the first metres of positive depth (~0.1; min-clamped)
//   u_height_floor   the heightmap value treated as floor level ("level 0"
//                    baseline), subtracted from relief BEFORE scaling so a floor
//                    pixel lifts ~0 and water shows on the floor immediately
//                    (no over-fill). Inert on the no-heightmap path.
uniform int   u_has_height;
uniform float u_height_scale;
uniform float u_height_edge;
// Heightmap BASELINE (the "level 0" floor relief). The art heightmap is a
// RELATIVE depth map: the floor itself sits at a nonzero relief value, so the
// raw `relief * u_height_scale` subtracts a constant everywhere and water must
// over-fill before it clears the floor. u_height_floor is the heightmap value
// treated as floor level — subtracted from relief BEFORE scaling, so a floor
// pixel (relief ≈ u_height_floor) reads ~0 lift and water shows on the floor
// immediately; a raised crate (relief > floor) protrudes; a dip (relief < floor)
// reads slightly deeper. ONLY bites on the heightmap path: with u_has_height==0
// or u_height_scale==0 the relief_term is forced to 0 (depth == tileDepth EXACTLY
// — the no-heightmap path stays bit-identical and u_height_floor cannot leak in).
uniform float u_height_floor;

out vec4 finalColor;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

// Ambient sine lattice (ported from overlays.py:183-185 _WAVES) — three
// directions so the idle shimmer never reads as stripes. Returns a height
// offset in the same (m-ish) units as ripple, so it feeds the same gradient.
//   (kx, ky, omega) rad/tile spatial, rad/s temporal.
float ambientSine(vec2 grid_xy) {
    // u_wave_scale MULTIPLIES the spatial frequencies (the temporal terms keep
    // their original rad/s so the idle motion stays the same speed): higher =
    // smaller/tighter waves, lower = bigger sweeping waves (Erik's dial).
    vec2 g = grid_xy * u_wave_scale;
    float s = 0.0;
    s += sin( 0.55 * g.x + 0.25 * g.y + 1.3 * u_time);
    s += sin(-0.35 * g.x + 0.45 * g.y + 0.9 * u_time);
    s += sin( 0.20 * g.x - 0.60 * g.y + 1.9 * u_time);
    return s * (1.0 / 3.0);
}

// Sample the ripple height (R) of the water texture at a world UV, plus the
// ambient sine lattice scaled by the local idle amplitude. The sine lattice
// gives standing water idle motion even where ripple == 0 (overlays.py §3).
float surfaceHeight(vec2 uv, float amb_amp) {
    float ripple = texture(u_water, uv).r;
    // grid coords for the sine phases (rad/tile): uv * grid = uv / u_texel.
    vec2 grid_xy = uv / max(u_texel, vec2(1e-6));
    return ripple + amb_amp * ambientSine(grid_xy);
}

void main() {
    // World UV (matches lighting.fs:78). Default art_uv_rect (0,0,1,1) makes
    // this bit-exactly fragTexCoord. Grid-resolution samplers (light, water)
    // read at world_uv; the diffuse reads at fragTexCoord (art space).
    vec2 world_uv = (fragTexCoord - u_art_uv_rect.xy) / u_art_uv_rect.zw;

    // --- depth gate (E) — the DORMANT-SAFE early-out ---------------------
    // water_depth (B) is the per-TILE depth — the only thing that says "there
    // is standing water on this tile". On a dry tile we emit a fully-transparent
    // premultiplied fragment, so the pass is a no-op over the composite. This is
    // the gate the safety property checks: no standing water -> identical render.
    // (UNCHANGED from the alpha-only version — the dormant-safe tile gate.)
    float tileDepth = texture(u_water, world_uv).b;
    if (tileDepth <= 0.0) {
        finalColor = vec4(0.0);
        return;
    }

    // --- PER-PIXEL ADJUSTED DEPTH from the floor heightmap relief --------
    // The per-pixel art heightmap RAISES THE FLOOR locally: high relief subtracts
    // from the tile's water depth, so `depth` is the actual water column ABOVE
    // the floor at THIS pixel. Everything depth-driven below computes off this
    // adjusted `depth` (via depthPos, the >= 0 clamp): the refraction offset, the
    // Beer-Lambert tint, and the alpha ramp — a raised crate refracts/tints/laps
    // as the shallower water it sits in, and where the floor pokes ABOVE the
    // surface (depth <= 0) the feature protrudes (handled at output). (The foam
    // wet/dry shoreline still reads the per-TILE depth field gradient — the
    // tile-level water edge — and caustics read ripple curvature; neither is a
    // per-pixel-depth term, so they are intentionally left on the tile field.)
    // The raw relief is measured against u_height_floor (the heightmap value
    // treated as floor level) so the floor itself lifts ~0 and water clears it
    // immediately (no over-fill). When there is no heightmap (u_has_height == 0)
    // or the gain is off (u_height_scale == 0), the relief_term is forced to 0 so
    // `depth == tileDepth` EXACTLY (bit-identical, the no-regression / dormant-
    // safe guarantee the default level relies on; u_height_floor cannot leak in).
    // relief_term is the per-pixel floor LIFT above the baseline floor level:
    // (heightmap relief - u_height_floor). Subtracting u_height_floor before
    // scaling means a floor pixel (relief ≈ u_height_floor) lifts ~0 (water on
    // the floor immediately), a crate (relief > floor) lifts positive (shallower
    // / protrudes), a dip (relief < floor) lifts negative (slightly deeper). With
    // no heightmap or the gain off, relief_term is forced to 0 so depth ==
    // tileDepth EXACTLY (bit-identical; u_height_floor never touches this path).
    float relief_term = (u_has_height == 1 && u_height_scale > 0.0)
                      ? (texture(u_height, fragTexCoord).r - u_height_floor) : 0.0;
    float depth = tileDepth - relief_term * u_height_scale;
    // Effect inputs must never see a negative depth (a protruding floor): clamp
    // to >= 0 for the refraction/fog/alpha math; the protrusion itself is decided
    // by the raw `depth <= 0` test at output.
    float depthPos = max(depth, 0.0);

    // Local ripple energy -> ambient idle amplitude + roughness agitation.
    // (|ripple| + |ripple_v|, the overlays.py energy heuristic, channels R+G.)
    vec4 wtex = texture(u_water, world_uv);
    float energy = abs(wtex.r) + abs(wtex.g);
    // Idle amplitude = u_ambient_amp base (old hardcoded 0.06) + energy gain.
    float amb_amp = clamp(u_ambient_amp + 2.0 * energy, 0.0, 0.40); // overlays.py _AMB_*

    // --- A. perturbed surface normal -------------------------------------
    // Height-field normal from neighbour taps of the ripple texture (+ the
    // ambient sine lattice). N ≈ normalize(-∂h/∂x, -∂h/∂y, 1) (height-field
    // convention, §3). u_ripple_scale converts metres-of-height to a slope
    // gain that reads on screen (the ripples are millimetric vs a 1-unit UV).
    vec2 tx = vec2(u_texel.x, 0.0);
    vec2 ty = vec2(0.0, u_texel.y);
    float hL = surfaceHeight(world_uv - tx, amb_amp);
    float hR = surfaceHeight(world_uv + tx, amb_amp);
    float hD = surfaceHeight(world_uv - ty, amb_amp);
    float hU = surfaceHeight(world_uv + ty, amb_amp);
    float dhdx = (hR - hL) * 0.5 * u_ripple_scale;
    float dhdy = (hU - hD) * 0.5 * u_ripple_scale;
    vec3 N = normalize(vec3(-dhdx, -dhdy, 1.0));

    // --- D. view direction — top-down constant ---------------------------
    vec3 V = vec3(0.0, 0.0, 1.0);

    // --- B/C. reused light buffer: colour + 3D direction -----------------
    vec4 tex_a = texture(u_light_a, world_uv);
    vec4 tex_b = texture(u_light_b, world_uv);
    vec3 lightRGB = tex_a.rgb;                       // incoming light colour (B)
    vec2 light_dir_2d = vec2(tex_a.a, tex_b.a);      // signed (C)
    vec3 L = normalize(vec3(light_dir_2d, u_light_z));

    // --- REFRACTION branch (head-on; dominates top-down) -----------------
    // Warp the floor sample by the surface tilt, scaled by depth so the
    // shoreline (depth->0) doesn't swim. Re-light the UNLIT diffuse in-shader
    // with the reused light buffer at the refracted UV (§4-F) — NEVER sample
    // the world RT this pass draws into (read-while-write feedback).
    vec2 off = N.xy * u_refract_strength * clamp(depthPos, 0.0, 1.0);
    // --- chromatic aberration (subtle polish, phase 2) ------------------
    // Sample the floor diffuse r/g/b at slightly different refraction offsets
    // (the offset scaled by 1 ± u_ca_amount per channel) — a faint prismatic
    // fringe on the refracted floor. u_ca_amount default is tiny; 0 collapses
    // to the single-sample core (the three offsets coincide).
    vec2 offR = off * (1.0 + u_ca_amount);
    vec2 offB = off * (1.0 - u_ca_amount);
    vec3 floorDiffuse = vec3(
        texture(u_diffuse, fragTexCoord + offR).r,
        texture(u_diffuse, fragTexCoord + off ).g,
        texture(u_diffuse, fragTexCoord + offB).b);
    if (u_srgb_decode == 1) {
        floorDiffuse = srgb_to_linear(floorDiffuse);
    }
    // Light the refracted floor by ambient + source, exactly as lighting.fs
    // lights the dry floor (`diffuse * (ambient + sources)`). Without the
    // ambient term the floor is black wherever no raycast source reaches, so
    // shallow water read as a black void only visible inside the flashlight.
    vec3 floorLight = u_ambient + texture(u_light_a, world_uv + off).rgb;
    vec3 floorC = floorDiffuse * floorLight;

    // Beer-Lambert depth tint: the floor fades toward the deep-water colour
    // with depth, reading as a true volume rather than a flat blue wash. Uses
    // the per-pixel ADJUSTED depth — shallower water over a raised feature tints
    // less (the protrusion reads near the floor colour).
    float fog = exp2(-u_fog_density * depthPos);
    vec3 refr = mix(u_water_color, floorC, fog);

    // --- caustics from the REAL surface (§3; SIGN IS LOAD-BEARING) -------
    // Caustics = where the wavy surface focuses the downward light onto the
    // floor. Intensity ∝ +laplacian(ripple): a CONCAVE-UP (focusing) patch is
    // BRIGHT, a convex bump is dark. With the height-field normal
    // N ≈ (-∂r/∂x, -∂r/∂y, 1), div(N_xy) = -laplacian(ripple), so
    //   caustic ~ +laplacian(ripple) ≡ -div(N_xy)   (CLAMP NEGATIVES TO 0).
    // Brightening by +div(N) would INVERT it (bright on bumps) — do NOT.
    // 5-point Laplacian of the ripple texture R channel (the same neighbour
    // taps the normal uses, but the raw R — no ambient sine, no ripple_scale):
    float rC = texture(u_water, world_uv).r;
    float rL = texture(u_water, world_uv - tx).r;
    float rR = texture(u_water, world_uv + tx).r;
    float rD = texture(u_water, world_uv - ty).r;
    float rU = texture(u_water, world_uv + ty).r;
    float lap = (rL + rR + rD + rU - 4.0 * rC) * u_ripple_scale;
    float caustic = max(lap, 0.0);              // focusing only; bumps -> 0
    // Optional subtle hybrid: a touch of high-frequency procedural detail
    // (modulated by surface energy) for crispness — the Laplacian × light is
    // the core; u_caustic_scale drives the detail (0 = pure surface curvature).
    if (u_caustic_scale > 0.0) {
        vec2 g = world_uv / max(u_texel, vec2(1e-6));
        float hf = 0.5 + 0.5 * sin(u_caustic_scale * (g.x + g.y) + 2.7 * u_time)
                              * sin(u_caustic_scale * (g.x - g.y) - 1.9 * u_time);
        caustic *= mix(1.0, hf, clamp(2.0 * energy, 0.0, 1.0));
    }
    // × lightRGB so caustics only appear under a flashlight/fire (correct —
    // focused light needs light); × strength; ride the floor/base (× alpha
    // below, NOT the surface-additive glint).
    refr += caustic * u_caustic_strength * lightRGB;

    // --- REFLECTION branch (the GGX GLINT — light OFF the surface) -------
    // GGX specular, cheap dot(L,H) form (§2). roughness drives α = roughness²;
    // still puddles (low energy) glint sharp, sloshing water (high energy)
    // shimmers broad. The glint is ADDITIVE light reflecting off the surface,
    // so it does NOT go through the Fresnel mix (which top-down on near-flat
    // water sits at ~R0 ≈ 0.02 and crushes the highlight to ~2%); it is added
    // on top of the premultiplied base instead. Gates kept: × lightRGB (only
    // appears under the flashlight/fire) and × NdotL (no back-lit glint).
    float roughness = clamp(u_roughness_base + u_roughness_agitation * energy,
                            0.02, 1.0);
    float a = roughness * roughness;
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float a2 = a * a;
    float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float ggx = a2 / max(3.14159265 * d * d, 1e-6);
    float NdotL = max(dot(N, L), 0.0);
    vec3 glint = ggx * NdotL * lightRGB * u_glint_strength;

    // --- BASE: the see-through refraction + depth tint -------------------
    // The base is the refraction/Beer-Lambert term — the floor seen through
    // the water. Add a small Fresnel reflection back into the base (grazing
    // ripples pick up a touch of reflected light) — this is the SUBTLE, alpha-
    // bound part; the bright sparkle is the additive glint above, NOT this.
    // The reflected colour is ambient + source so the surface has a faint
    // sheen (presence) OUTSIDE any source, not only under the flashlight. This
    // ambient sheen is the placeholder for the deferred matcap/environment
    // reflection — keep it subtle. (The glint stays source-only; ambient gives
    // a flat sheen, not a sharp specular.)
    float NdotV = max(dot(N, V), 0.0);
    float F = u_r0 + (1.0 - u_r0) * pow(1.0 - NdotV, 5.0);
    vec3 sheenRGB = u_ambient + lightRGB;
    vec3 base = mix(refr, refr + sheenRGB * F, F);  // ≈ refr + small ambient sheen

    // --- FOAM (port of overlays.py:261-277, composited LAST) ------------
    // Whitecaps where the ripple front is steep (|grad ripple| over a
    // threshold) + the wet/dry shoreline (where the water depth gradient is
    // high — the advancing/receding edge). Both from the ripple + depth this
    // pass already samples (no new texture). Foam is white-ish SURFACE SCATTER
    // mixed into the (Fresnel) base before the premultiply — NOT additive
    // light (it does not ride on top like the glint) and NOT × the floor
    // (it sits on the surface). It is alpha-bound with the base.
    // |grad ripple| from the RAW ripple R taps (central difference, per UV):
    float gxr = (rR - rL) * 0.5;
    float gyr = (rU - rD) * 0.5;
    float gradRipple = length(vec2(gxr, gyr));
    float crestFoam = clamp(gradRipple / max(u_foam_threshold, 1e-6) - 1.0,
                            0.0, 1.0);
    // Wet/dry shoreline: high depth gradient = the water edge. The neighbour
    // depths from the same water texture (B channel).
    float dL = texture(u_water, world_uv - tx).b;
    float dR = texture(u_water, world_uv + tx).b;
    float dD = texture(u_water, world_uv - ty).b;
    float dU = texture(u_water, world_uv + ty).b;
    float depthGrad = length(vec2(dR - dL, dU - dD)) * 0.5;
    float shoreFoam = clamp(depthGrad / max(u_foam_threshold, 1e-6) - 1.0,
                            0.0, 1.0);
    float foam = max(crestFoam, shoreFoam) * u_foam_intensity;
    foam = clamp(foam, 0.0, 1.0);
    // Composite foam LAST into the base: a white-ish term lerped over the
    // Fresnel base. lightRGB-aware so foam isn't a flat grey in the dark — it
    // catches ambient + the source like real froth (kept simple).
    vec3 foamColor = (u_ambient + lightRGB) * vec3(1.0);
    base = mix(base, foamColor, foam);

    // --- output: PREMULTIPLIED base + ADDITIVE glint ---------------------
    // alpha ramps in over the first ~few cm of depth so the wet/dry shoreline
    // reads as a soft edge rather than a hard step; it never reaches 0 here
    // (depth > 0 already passed the gate), and is 0 on dry tiles (returned
    // above). The base is premultiplied by alpha (see-through, transparency-
    // bound); the GLINT is added on top WITHOUT × alpha, so it survives the
    // premultiplied blit as additive light — a bright highlight visible
    // regardless of how transparent the water is. The glint does NOT raise
    // alpha (a specular highlight does not make the water more opaque).
    float alpha = clamp(depthPos * u_alpha_scale, u_alpha_min, u_alpha_max);
    // --- PROTRUSION + SOFT LAP EDGE (per-pixel depth) -------------------
    // The heightmap adjusted the floor per-pixel above (subtracting relief from
    // the tile depth). Where the adjusted `depth` has gone <= 0 the floor pokes
    // ABOVE the surface — the feature protrudes, so emit a fully-transparent
    // fragment (the crate/console/debris shows through; the water laps around
    // it). For a SOFT shoreline (not a hard cut) we fade the alpha to 0 over the
    // first u_height_edge metres of positive depth: smoothstep is ~0 right at
    // the protrusion edge and ramps to 1 over height_edge, OVERRIDING the
    // alpha_min floor so the waterline against the feature reads as a soft lap
    // rather than a step. With NO heightmap relief == 0 -> depth == tileDepth > 0
    // (passed the gate), and smoothstep(0, edge, depth) == 1 for any depth beyond
    // a sub-cm edge, so the alpha is bit-identical to the pre-heightmap render
    // (and to a no-water ship). This SUBSUMES the b5e7bdb alpha-only block:
    // where depth <= 0 the feature is dry; the soft edge replaces the old wet=...
    // ramp but now drives every effect through the adjusted depth, not just alpha.
    float lapEdge = smoothstep(0.0, max(u_height_edge, 1e-4), depth);
    alpha *= lapEdge;
    if (depth <= 0.0) {
        finalColor = vec4(0.0);
        return;
    }
    if (u_srgb_decode == 1) {
        base  = linear_to_srgb(base);
        glint = linear_to_srgb(glint);
    }
    finalColor = vec4(base * alpha + glint, alpha);
}
