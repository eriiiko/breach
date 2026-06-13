# Water rendering — the surface optics

**Depends on:** [Fluid & water](../engine/07_fluid_and_water.md) (the `water_depth` + `ripple` fields,
and §6's two-layer surface model), [Ray engine](../engine/08_ray_engine.md) (the per-pixel `light_rgb`
+ light-direction buffers this pass reuses), [Rendering](../engine/09_rendering.md) (the world-RT
compose order + the premultiplied-alpha blend discipline). Research + citations:
[`docs/water_rendering_research.md`](../../water_rendering_research.md).

---

## 1. What this is

The fluid system simulates a `water_depth` field and a visual-only `ripple` wave-field on its surface
(engine ch.07 §6). This chapter is how those become *convincing water* on screen: a single fragment
pass that makes the surface reflect the ship's lights, refract the floor beneath it, darken with depth,
and dance with caustics — in a **top-down, mostly-dark ship** where the light is flashlights, fire, and
glowing screens.

The **decoupled-budget principle** (graphics README) licenses doing this properly: this pass never runs
during headless self-play, so its cost trades against nothing the training farm cares about. We spend
freely here — the surfaces we already simulate deserve to be shown off.

**The top-down reframing that drives every choice:** looking straight down, there is no sky to reflect.
"Reflection" means the *specular response of the actual light sources* on the rippled surface — a
flashlight glinting off the wet floor, fire shimmering in a flooded corridor. This makes the cheap,
light-source-driven methods the *correct* ones, not a compromise; and it makes our **ripple normal the
linchpin** — straight down, a flat surface has ~2% Fresnel and reads dead; it is the ripple tilts that
create the grazing angles and therefore the glints.

## 2. The unified Fresnel reflect/refract pass

One fragment shader, one surface normal `N`, one depth `d`. Fresnel is the single dial that unifies the
two halves (reflection at grazing/rippled, refraction head-on).

```glsl
// --- the perturbed surface normal (the only genuinely-new per-pixel build, §4-A) ---
vec3 N = perturbedRippleNormal(uv);   // ∂ripple/∂x,∂y (+ ambient sines), lifted to 3D

// --- REFRACTION branch (head-on; dominates top-down) ---
vec2  off    = N.xy * refractStrength * clamp(d, 0.0, 1.0);   // depth-scaled (no shoreline swim)
vec3  floorC = sampleDiffuse(uv + off) * sampleLight(uv + off);  // §4-F: UNLIT diffuse, re-lit
//   in-shader with the reused light buffer (B) at the refracted position — NEVER sample the world
//   RT this pass draws into (read-while-write feedback). chromatic aberration: r/g/b at slightly
//   offset UVs (subtle; +2 fetches)
float fog    = exp2(-fogDensity * d);                          // Beer–Lambert
vec3  refr   = mix(waterColor, floorC, fog);                  // depth tint → reads as volume
refr        += causticTerm(uv) * lightRGB;                    // §3, ride the light (floor light; phase 2)

// --- REFLECTION: GGX glints, ADDED on top (not Fresnel-mixed — see note) ---
float spec   = ggxSpecular(N, L, V, roughness);              // §4-C reuse L; cheap dot(L,H) form
vec3  glint  = spec * NdotL * lightRGB * glintStrength;      // §4-B reuse light; gated to LIT, forward-facing ripples

// --- BASE: refraction + a faint Fresnel ambient sheen (this is what Fresnel weights) ---
float F      = R0 + (1.0 - R0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);   // R0 ≈ 0.02
vec3  base   = mix(refr, refr + lightRGB * F, F);           // mostly the see-through floor; a faint sheen at crests
base         = mix(base, foamColor, foamMask(uv));          // foam last (surface scatter; §4-foam, phase 2)
//             (+ matcapAmbient(N) under the Fresnel weight when enabled, §6)

// --- OUTPUT: premultiplied base + the glint ADDED on top (the §5 additive discipline) ---
float alpha  = clamp(d * alphaScale, alphaMin, alphaMax);    // transparency ramp (all three tunable)
vec4  water  = vec4(base * alpha + glint, alpha);           // glint = HDR light ON TOP, NOT × alpha
```

- **GGX, not Blinn-Phong**, for the glints (Erik, locked) — its narrow-core/soft-tail lobe attenuates
  the sparkle-aliasing ("fireflies") that Blinn-Phong suffers on a *moving* ripple field under bloom;
  the `dot(L,H)` simplification makes it barely costlier than Blinn-Phong.
- **Roughness is a gameplay/art lever** (locked): drive `α = roughness²` from the local ripple
  agitation — still puddles glint sharp (low roughness), disturbed/sloshing water shimmers broad.
- **Order:** refract + tint (+ floor caustics) → Fresnel-weighted base → foam → premultiply → **add
  the glint (and surface caustics) on top** as additive HDR light.
- **Why the glint is *added*, not Fresnel-mixed (shipped 2026-06-13, `a5d177a`).** Blending the
  reflection in by `F` weights it ~2% head-on (top-down), which crushes the glint *invisible* on calm
  water — the near-flat normal means `F` never spikes. A specular highlight is reflected *source*
  light, so it must be **added** on top: an HDR term gated by `lightRGB`·`NdotL`, scaled by a
  `glintStrength` dial, surviving the premultiplied blit as additive light regardless of the water's
  transparency. Only the faint *ambient* sheen stays Fresnel-weighted. This is the §5 additive-RGB-only
  discipline, applied to the glint from v1 (not deferred to caustics). The naïve `mix(refr, refl, F)`
  is the trap that made the first build's glints invisible.
- Top-down, geometric `F ≈ 0.02` (we mostly see the floor — correct); the **ripple normals locally
  spike `F`** to draw glints riding the wave crests.

Full per-method rationale, tiers, and citations: the research doc. All of 2b ships in v1 (refraction,
depth-tint, caustics, chromatic aberration — Erik); SSR is held as polish (low return straight-down).

## 3. Caustics from the *real* surface (not procedural noise)

Caustics are where the wavy surface acts as a **lens**: concave (focusing) patches concentrate the
downward light into bright filigree on the floor; convex patches spread it into dark gaps. The
intensity is governed by the surface **curvature**.

Because we *simulate the surface*, we compute the real thing rather than a decoupled noise loop. The
caustic intensity tracks the surface **curvature** — `+∇²ripple` (the Laplacian of the ripple height,
a few neighbour taps of the same ripple texture §4-A already samples). **Sign convention — pin it:**
with the height-field normal `N ≈ (−∂ripple/∂x, −∂ripple/∂y, 1)`, the screen-space divergence
`div(N_xy) = −∇²ripple`, so brighten by **`+∇²ripple ≡ −div(N_xy)`** — *concave-up (focusing) tiles
brighten*, convex bumps darken (clamp negatives to 0). The ray-density Jacobian fixes the sign: floor
irradiance ∝ `1/(1 − c·d·∇²ripple) ≈ 1 + c·d·∇²ripple`. (Brightening by `+div(N)` inverts the caustics —
bright on the bumps — so the sign is load-bearing.) The payoff: caustics that **dance with our actual physics** — a
grenade splash sends a real flash of focused light skittering across the flooded floor, a wading
marine trails caustics in its wake, the *Titanic* tilt-slosh drags the whole pattern.

**Hybrid (the recommendation, Erik's question 2026-06-12):** our ripples resolve at ~art/tile
resolution, so pure surface-curvature caustics may read soft (real sparkle comes from fine
high-frequency ripples we don't fully resolve). So: the **real surface curvature drives where/when**
caustics intensify (reactivity to events), and we optionally layer a little **procedural
high-frequency detail modulated by the real surface energy** for crispness. Both **× the per-pixel
light** — caustics in darkness make no sense; caustics on the flashlight pool are gorgeous and
motivated. Phase 2 (separable; shouldn't gate the v1 core).

## 4. Reuse map — what the shader reuses vs. computes new (Erik's check, code-verified)

| Input | Status | Source / note |
|-------|--------|---------------|
| **A** perturbed surface normal | **compute-new (GPU)** | numpy recipe already proven CPU-side (`overlays.py:241` ∂ripple + sine lattice `:183-185`); re-derive per-pixel in GLSL from a ripple float texture |
| **B** per-pixel light RGB | **reuse directly** | already GPU texture `tex_a.rgb`, `lighting.fs:100,102` (from `raycaster.h:192-206`) |
| **C** light direction (GGX `L`) | **reuse directly** | `vec2(tex_a.a, tex_b.a)` → 3D `L`, `lighting.fs:104,119` (`raycaster.h:208-213`) |
| **D** view direction | **trivial constant** | top-down `V = (0,0,1)` |
| **E** water depth | **NEEDS PLUMBING** | `gmap.water_depth` (`gamemap.py:250`) is CPU-only / lossy RGBA8 tint today; needs a real float channel on the GPU |
| **F** floor diffuse (refraction sample) | **reuse directly** | the **unlit** ship diffuse `TextureSet.diffuse`, sampled at `fragTexCoord` `lighting.fs:87` (no separate floor layer); **re-light it in-shader with the reused `light_rgb` (B)** — never sample the world RT this pass draws into (feedback) |
| **G** normal divergence (caustics) | **derive cheap** | ∇²ripple from the same ripple texture A needs (phase 2) |
| **foam** | **derive / port** | already implemented CPU-side `overlays.py:261-277` (`|∇ripple|` crests + wet/dry front via `flow_vx/vy`) |

**Genuinely new:** A (the GPU normal) + the Fresnel/GGX/refraction/tint/CA combine. Everything else
reuses an existing GPU texture, is a constant, or derives from a field we already have.

**Plumbing gaps (the only real new uploads):**
- **`ripple` (+ `ripple_v`)** — CPU-only today (`gamemap.py:262`); upload as a float texture so the
  shader can take its gradient (A) and Laplacian (G).
- **`water_depth`** — currently only baked into the RGBA8 tint overlay; upload as a real per-pixel
  depth for the refraction scale + Beer–Lambert tint (E).
- The `update_rgba16f_texture` upload primitive already exists (`core.py:133,150`). **Suggested
  packing** (mirrors the existing two-RGBA16F light-texture scheme): one new RGBA16F "water texture" =
  `ripple` (R) · `ripple_v` or ∇²ripple (G) · `water_depth` (B) · foam/agitation (A) — one upload per
  frame. `light_rgb`/`light_dir`/`diffuse` need **no** new plumbing.
- New uniforms: `roughness` base + agitation scale, `fogDensity`, `refractStrength`, `R0`, caustic
  strength, CA amount, foam threshold — a `[graphics.water]` config block (look-tuning lives here per
  the graphics README).

## 5. Architecture — a separate water fragment pass

**Separate GLSL pass, not an extension of `lighting.fs`** (verified): the lighting shader writes opaque
`vec4(lit,1.0)` over the whole ship in one full-RT quad and discards vacuum (`lighting.fs:83,134`);
folding water in would push every *dry* pixel through the water branches and need a water-mask
early-out — a tax on the common case, with no sharing a separate pass doesn't also get by binding the
same textures.

- **Compose slot:** exactly where `WaterFieldOverlay` draws today — after the lit ship + smoke/fire/glow
  + pressure, **before units** (`game_renderer.py:381-384`). Units/projectiles/effects stay on top
  (the squad reads over the water); the lit ship beneath is the floor sample. This *replaces* the
  CPU-tinted `WaterFieldOverlay` placeholder (the W6b overlay) with the GLSL pass in the same slot.
- **Blend discipline (the god-ray-fix rule):** the world RT composites to screen premultiplied
  (`world_composite.py:120`, `out = rt.rgb + bg·(1−rt.a)`). Every **additive** sub-term — specular
  glints (shipped), caustics (phase 2) — **must not write destination alpha** (or vacuum goes
  opaque-black under the blit). The shipped v1 does this *within the single premultiplied draw*:
  `finalColor = vec4(base·alpha + glint, alpha)` — the glint adds to the premultiplied RGB while
  `alpha` is left untouched, so under the premultiplied blit it rides as additive light that never
  blackens vacuum. (A term drawn as a *separate* pass would instead use the
  `_begin_additive_rgb_only_blend()` helper, `overlays.py:294-314`.) Net: a premultiplied `vec4` with
  `alpha < 1` only on water tiles; dry tiles return `vec4(0)`.

### 5.1 Relationship to the CUDA migration — build now, don't wait

This pass is **independent of the CUDA migration** and should not wait for it. The CUDA work moves the
*physics computation and state* (GameMap fields, the solvers) to GPU-resident CUDA buffers for
parallel-sim throughput + fixed-point determinism — all **sim-side**. This is a **render-side** GLSL
pass that only *reads* fields. The Fresnel/GGX/refraction/caustics math is invariant to where its
source data came from; it samples GL textures either way, and graphics never run in self-play so the
determinism pass has zero bearing on it.

The *only* seam CUDA touches is the field→texture **upload**: today numpy → CPU pack → `glTexImage`;
after CUDA residency, the same fields are CUDA buffers shared to GL via **CUDA–GL interop** (no CPU
round-trip) — a faster fill behind the *same* "this GL texture holds this field" interface. So the
packed-RGBA16F single-upload design (§4) is deliberately a clean swap point: the shader is permanent,
the upload is a one-function later optimization, and the CPU-upload path is needed now regardless (CUDA
is a large, gated, later phase). Building water rendering now is not throwaway work — it stands up the
consumer interface that CUDA–GL interop later accelerates.

## 6. The matcap / ambient-environment hook (optional)

A faint additive reflection term sampled from a baked "lit sphere" by the screen-space normal
(`muv = N.xy*0.5+0.5`). It fakes a *static* environment, so **indoors it stays ~off** (the dynamic GGX
glints already do moving flashlights correctly). Its real win is **outdoors** — the EVA band or any
water open to space, where it reflects sky/stars/hull cheaply (one texture fetch). The architecture
carries the slot (composes under the Fresnel weight); v1 ships it disabled/subtle, trivially enableable
by dropping in a matcap texture. (Erik wants to play with it; not a v1 authoring deliverable.)

## 7. Implementation staging (each step eyeball-verified)

1. **Plumbing** — upload `ripple` + `water_depth` as a packed RGBA16F water texture (reuse
   `core.update_rgba16f_texture`); add the `[graphics.water]` uniforms; stand up the empty water
   fragment pass in the `WaterFieldOverlay` compose slot, premultiplied, water-tile-gated.
2. **Core look (v1 must-have)** — perturbed normal (A) → Schlick Fresnel → GGX glints (reuse B/C) +
   normal-driven refraction×depth (reuse F/E) + Beer–Lambert tint. This is "flashlight glints off the
   wet floor, floor warps and darkens with depth." Tune `roughness`/`fogDensity`/`refractStrength`.
3. **Mood + polish** — real-surface caustics × light (G, hybrid §3), foam port (composite last),
   chromatic aberration (subtle). **Shipped (phase 2, 2026-06-13).**
4. **Matcap hook** (§6) — optional term, off by default. *(Not built — gated on a matcap texture.)*
SSR remains out (polish, revisit only for literal reflected shapes).

## 8. Implementation status

**v1 CORE shipped (`ab2bd25` + `a5d177a`, 2026-06-13).** The GLSL water fragment pass is live
(`shaders/water.fs`, `renderer/water.py`, drawn in the `WaterFieldOverlay` compose slot — the CPU
placeholder is retired): the RGBA16F water texture (ripple + depth packed/uploaded per frame), the
`[graphics.water]` uniforms, the perturbed-ripple normal build, Schlick Fresnel, **additive GGX
glints** (the corrected design, §2), normal-driven refraction × depth, and Beer–Lambert depth tint.
Dormant-safe (dry tiles return `vec4(0)`; verified bit-identical on a dry ship). **Live tuning:** all
params are sliders in the lighting demo (`tools/lighting_demo.py`), pushed to per-frame `WaterPass`
setters — drag + pour with `U`, no restart; the values that look right become the `[graphics.water]`
config defaults.

**Reused (built, no new plumbing):** `light_rgb` + light-direction GPU textures (ray engine Tier-1);
the ship diffuse texture; the `ripple`/`water_depth` fields (engine ch.07, W6); the RGBA16F upload
helpers (`core.py:133,150`); the premultiplied/additive blend helpers (`overlays.py:294-314`).

**Phase 2 — the mood pass — shipped (2026-06-13).** Added to the same `shaders/water.fs` pass, all
dormant-safe (dry tiles still return `vec4(0)`):
- **Real-surface caustics × light (§3).** `caustic = max(+∇²ripple, 0) · u_caustic_strength · lightRGB`,
  the Laplacian from a 5-point neighbour-tap of the ripple R channel (the SIGN is load-bearing —
  `+∇²ripple ≡ −div(N)`, concave-up/focusing brightens; clamped ≥ 0). **Added into the refraction/floor
  term** so it rides the floor (× alpha), seen through the water — not the surface-additive glint.
  × `lightRGB` gates it to under a flashlight/fire. Optional `u_caustic_scale` high-frequency procedural
  detail (× ripple energy) for crispness.
- **Foam port (§4-foam).** The CPU algorithm at `overlays.py:261-277` moved into the shader: whitecaps
  where `|∇ripple| > u_foam_threshold` + the wet/dry shoreline (high depth gradient), composited **last**
  into the (Fresnel) base before the premultiply (surface scatter, alpha-bound — not additive). Dials
  `u_foam_threshold`, `u_foam_intensity`.
- **Chromatic aberration.** The refracted floor r/g/b sampled at offsets scaled `1 ± u_ca_amount`
  per channel (default tiny). One dial `u_ca_amount`.
- **Wave-size dials.** `u_wave_scale` multiplies the ambient-sine spatial frequencies (Erik: idle waves
  read "very big" — default 2.0 tightens them); `u_ambient_amp` is the idle-shimmer amplitude base (the
  old hardcoded `0.06`).

All seven new knobs are `[graphics.water]` config keys, `WaterPass` per-frame setters, and live sliders
in the lighting demo (Water section), per the v1 pattern. **Still deferred:** the matcap hook (§6, gated
on a matcap texture) and SSR (polish).
