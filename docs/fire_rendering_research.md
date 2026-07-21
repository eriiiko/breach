# Fire & Explosion Rendering — Technique Search for Breach

*Research pass, 2026-07-11. Focused, single-pass literature/technique scan. Scope: modern real-time + film/VFX approaches for fire and explosions, mapped onto Breach's actual physical fields. Cost tags are relative to a 2D tile grid: **cheap** = per-tile arithmetic or a screen-space pass; **medium** = per-particle work or a few extra buffer passes; **expensive** = anything that fights the real-time budget.*

Breach's unfair advantage, restated so every technique below can be judged against it: the renderer can **read real physical fields** — Kelvin temperature per tile, a real compressible wind velocity field, per-species gas densities (O2, N2, soot/black smoke), and a per-channel volumetric ray-marched light engine with per-gas absorption **and** scatter coefficients plus a `smoke_glow` god-ray buffer. Almost every technique the games/VFX world invented exists to *fake* one of those fields. Breach doesn't fake them — so most of the work here is **reading a field that's already there**, not synthesizing a fake one. That inversion is the whole opportunity.

---

## 1. Executive summary — highest-value, real-time-feasible picks

- **Make fire color physical, not fixed-orange.** Replace the constant orange emitter with a **blackbody Kelvin→RGB** mapping driven by the real temperature field, then push the very-bright end through **exposure + tone-map + bloom** so a white-hot core reads as white-hot rather than clipping to flat orange. This is the single highest-leverage change and it is **cheap** (a per-tile lookup or a short polynomial). Everything else is polish on top of this.
- **Don't simulate detail — advect it.** The tile field is coarse; get sub-tile flame licking by **advecting a detail noise texture along the real wind field** and/or adding **curl noise** on top. Fire's characteristic irregular motion is cheap synthetic turbulence riding a coarse true flow, not a finer solve.
- **Fire particles ride the real wind vector + per-particle curl noise.** This is exactly the Bridson curl-noise recipe, except Breach supplies the *base* velocity field for free (real advection). Particles sample the tile wind, add divergence-free curl noise for organic swirl, and are colored by the local blackbody temperature. **Medium** cost, very high payoff for the "alive" look.
- **God-rays are already 90% built.** Breach has per-gas scatter/absorption and a `smoke_glow` buffer — that *is* volumetric light. The film reference (the Alien escape-pod beams that glow through haze without dimming) is precisely the **scatter ≫ absorption** regime Breach already models per-gas. The cheap real-time trick (screen-space radial-blur light shafts, GPU Gems 3 ch.13) is a fallback/booster, not the main event.
- **Explosions = one transient super-bright light source.** The look is carried by **light**, not by the sprite: a brief high-intensity flash that (a) blows out the tone-map for a frame or two, (b) shock-lights the expanding smoke from inside, and (c) casts **very long, fast-moving shadows** because the source is bright and localized. Breach's ray-marched light engine can do all three literally — most games can only fake the shadows.
- **Incomplete-combustion soot is a real, free art asset.** Low-O2 → more soot is genuine field data in Breach. That means smoke *color and opacity* respond to real chemistry: an oxygen-starved fire genuinely blackens and chokes its own light. No other game has this as physics rather than a scripted material swap.

---

## 2. Per-question findings

### Q1 — Temperature → color (blackbody). *Core "glow" primitive.*

**The physically-correct path** (offline-accurate, do it once to build a table): evaluate **Planck's law** across the visible spectrum (380–780 nm) for a given Kelvin temperature, integrate against the **CIE XYZ color-matching functions**, convert XYZ → **linear sRGB** (do *not* gamma-correct yet), then apply exposure + tone-map, then gamma. The catch is dynamic range: emitted radiance scales roughly as T⁴, so 1000 K vs 8000 K spans ~10 orders of magnitude ([Macklin](https://blog.mmacklin.com/2010/12/29/blackbody-rendering/), [ScratchAPixel](https://www.scratchapixel.com/lessons/cg-gems/blackbody/blackbody.html)). You **must** carry intensity, not just hue, or every fire looks the same brightness.

**Real-time approximations (pick one):**
- **1D lookup texture indexed by Kelvin** — bake the Planck→XYZ→RGB curve offline into a small gradient texture (say 1000–10000 K). **Cheapest possible** per-tile: one texture read. Recommended default for Breach since temperature is already a tile scalar.
- **Polynomial/piecewise approximation** — Tanner Helland's Kelvin→RGB (piecewise log/power fits, valid ~1000–40000 K) is the canonical cheap closed-form; Neil Bartlett refined it. Good for a shader with no texture dependency ([Tanner Helland](https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html), [Neil Bartlett](https://github.com/neilbartlett/color-temperature)). Note: these are tuned for *chromaticity* (photography white balance) and do **not** give you the T⁴ intensity — multiply by a separate brightness ramp.
- **HLSL blackbody function** — zubetto's `BlackBodyRadiation` returns both luminance and chromaticity in-shader, purpose-built for real-time CG ([zubetto/BlackBodyRadiation](https://github.com/zubetto/BlackBodyRadiation)).

**The bright end (this is where it reads right or wrong):** compute emission in **linear HDR**, then **Reinhard (or similar) tone-map + bloom**. The chromaticity of a blackbody *desaturates toward white* as T rises, and a properly exposed hot core should push past the display gamut into white with colored bloom fringing — that's what sells "white-hot." Macklin explicitly uses a reference temperature (~3000 K) to set exposure and Reinhard tone-maps the rest. **Cost: cheap** (LUT read) + **cheap** (a bloom pass Breach likely already wants).

**Maps onto Breach:** directly replaces the fixed-orange fire emitter. Feed tile Kelvin → LUT → set the light source's per-channel RGB *and* intensity. Fire at 1500 K reads deep orange-red; explosion cores at higher K read yellow-white; the light engine then propagates that real color through the smoke. This one change makes the existing light engine "physical" for free.

### Q2 — Flame detail from a coarse field.

The universal modern answer: **coarse solve + procedural fine detail, never a finer solve.** High-res fire in film/GPU pipelines (Horvath & Geiger's "Directable, High-Resolution Simulation of Fire on the GPU") is a **coarse simulation + fine view-oriented refinement**; the coarse stage enforces the physics, the fine stage adds looks-detail ([Horvath & Geiger / SIGGRAPH](https://history.siggraph.org/learning/directable-high-resolution-simulation-of-fire-on-the-gpu)). For Breach the coarse stage is *already done and real*.

Concrete detail generators, cheapest first:
- **Advected detail texture** *(cheap)* — sample a tiling noise/flow texture, scroll its UVs *along the real wind vector*, and use it to modulate emission/density between tile centers. Because the advection direction is the true field, the detail moves correctly (licks downwind, curls around obstacles) with zero extra simulation.
- **Curl noise turbulence** *(cheap–medium)* — add a divergence-free synthetic vorticity field (curl of a Perlin/simplex potential evaluated at x,y,t) on top of the tile velocity when advecting detail or particles. Frequency/octaves/time-scale are the three tuning knobs: high frequency = tight swirls, more octaves = fine structure over broad flow ([Bridson SIGGRAPH 2007](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf), [Unity curl-flow demo](https://github.com/Parallel-Cascades/curl-flow-simulation)).
- **Vorticity confinement** *(cheap, field-side)* — if you want the *sim* itself to look livelier without more resolution, amplify existing curl in the solver. Even low levels (0–15) noticeably sharpen turbulent features, especially with semi-Lagrangian advection ([Andrew Chan](https://andrewkchan.dev/posts/fire.html)). This is a physics knob, not a render trick — flag for Erik since it touches the solver.

**Maps onto Breach:** detail lives entirely in the renderer reading `wind` + `temperature`. The licking motion = noise UVs advected by the real velocity; the irregularity = curl noise; the color of the detail = blackbody of the local T. No change to the deterministic solver required (keep it that way unless you want vorticity confinement, which does change sim state).

### Q3 — Particles riding the velocity field.

This is the **Bridson curl-noise particle** recipe, and Breach is the ideal host for it because the base flow is real:
- Each fire/smoke/ember particle **samples the tile wind vector** at its position (bilinear across tile centers) as base advection velocity.
- Add **curl noise** (divergence-free, so particles neither clump nor scatter — they swirl in incompressible-looking streams) as per-particle organic detail; add a small constant **buoyant up-vector** for flame rise ([Bridson 2007](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf), [kbladin GPU impl](https://github.com/kbladin/Curl_Noise), [freder curl-noise explainer](https://freder.github.io/UnityGraphicsProgrammingBook1/html-translated/vol2/Chapter%206%20_%20Curl%20Noise-Explanation%20of%20Noise%20Algorithms%20for%20Pseudo-Fluids.html)).
- **Color each particle by the local blackbody temperature** (Q1 LUT) and let it cool along its lifetime — an ember fades orange→red→dark exactly as its sampled T drops. Fade opacity by local soot density.
- GPU-side, particle positions live in a buffer updated per frame (transform feedback / compute), no CPU roundtrip.

**Cost: medium** (per-particle field sample + noise eval). **Maps onto Breach:** particles are pure render-layer garnish on top of the authoritative field — they read `wind`, `temperature`, `soot`, and never write back, so determinism is untouched. This is exactly the "fire particles ride the wind vector + per-particle noise" Erik asked for, and the wind vector is *real advection*, so embers genuinely stream correctly around walls and get pushed by the compressible solver's gusts. That behavior is unfakeable with canned particle forces.

### Q4 — Volumetric light / god-rays through smoke.

The physics: volumetric light = **absorption + in-scattering** along the view/light ray, governed by extinction μ_t = absorption μ_a + scattering μ_s ([NVIDIA GPU Gems 3 ch.13](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process), [Grokipedia: volumetric lighting](https://grokipedia.com/page/Volumetric_lighting)). **Breach already ray-marches this per-gas with separate scatter and absorb coefficients** — which is the expensive part everyone else skips. The Alien escape-pod reference look (beams that *glow* through haze but barely dim the beam behind) is the **scatter ≫ absorption** case: steam/mist scatter light into the eye (bright shafts) while absorbing little (beam survives). Breach models exactly this per-gas — steam scatters, black soot absorbs — so the escape-pod look falls out of setting the right coefficients, not a special effect.

Techniques to layer on:
- **Use the real per-gas coefficients as the art control** *(already built)* — steam/water vapor: high scatter, low absorb → glowing god-rays that don't dim. Black smoke/soot: high absorb → hard shadow shafts and dark cores. This is the whole ballgame and Breach has it.
- **Screen-space radial-blur light shafts** *(cheap booster/fallback)* — GPU Gems 3 ch.13: from the bright source in screen space, march samples outward accumulating `L = Exposure · Σ(Weight · Decay^i · sample_i)`. 64–128 taps/pixel, downsample-friendly. Useful to *cheaply amplify* the `smoke_glow` buffer into long cinematic shafts without more ray-march cost.
- **6-way / 6-point smoke lighting** *(medium, mostly for particle smoke)* — bake light response from 6 directions so sprite/mesh smoke shows internal shadowing and back-lit rims ([Unity 6-way lighting](https://unity.com/blog/engine-platform/realistic-smoke-with-6-way-lighting-in-vfx-graph)). Relevant only if Breach adds particle smoke on top of the volumetric field; the field-based smoke already gets true lighting from the ray-marcher, so this is optional.

**Maps onto Breach:** feed the blackbody fire color (Q1) as the light source RGB into the existing ray-marcher; let per-gas scatter build the `smoke_glow`/god-ray buffer; optionally run the screen-space radial blur on that buffer to exaggerate shafts. The novel part is that the shaft **color** is the real fire temperature and the shaft **shape** is carved by real soot density — both are data, not painted.

### Q5 — Explosions specifically.

Explosion impact is **carried by light, not the sprite** — the community consensus and the tool-of-record (EmberGen) both treat the flash + shock-lighting as the payload ([realtimevfx: EmberGen explosion](https://realtimevfx.com/t/how-to-achieve-a-realistic-game-explosion-with-embergen/27306), [Mad-VFX: light & VFX interplay](https://www.mad-vfx.com/blogs/the-interplay-of-light-and-vfx-in-games-how-they-work-together)). Three components, all of which Breach can do *literally* via its light engine + fields:
- **The brief bright flash** — a transient, very-high-Kelvin (white/blue-white) light source for 1–3 frames, intensity far above the tone-map's white point so it **blooms hard and briefly desaturates the frame**. Cheap: it's a Q1 emitter with a spiked temperature/intensity and a fast decay curve.
- **Shock-lit expanding smoke** — the flash lights the newly-created soot/smoke *from inside* as it expands on the real velocity field. Because the light is volumetric, the expanding shell self-shadows and rim-lights correctly. Free consequence of Q4 + the compressible solver actually pushing the gas out.
- **Very long, fast shadows** — a bright, localized, moving source throws long shadows that sweep as the fireball rises/expands. Breach's ray-marched light casts these for real; most games script or bake them. This is the single most "expensive-looking" cue and Breach gets it as physics. *(Watch cost: many transient dynamic shadow-casters is the one place explosions can blow the budget — cap the number of simultaneous bright transient sources.)*

**Cost:** flash + bloom **cheap**; shock-lit volumetric smoke **medium** (rides existing ray-march); long transient shadows **medium–expensive** depending on how many casters. **Maps onto Breach:** an explosion is just a scripted spike into the *same* fields — inject heat (high T), inject soot, inject outward velocity — and the renderer's existing blackbody + ray-march machinery turns that into flash, shock-light, and shadows with no bespoke explosion shader.

### Q6 — see the Novel Opportunity section below.

---

## 3. The novel opportunity — opinionated top picks

Breach's edge is that **the render can read real T, wind, O2/soot, and a true volumetric light solve.** The distinctive look comes from techniques that are *impossible* without those fields. My top picks, ranked:

**① Temperature-is-color, everywhere, physically.** Kill fixed-orange. Every emissive thing — fire, embers, explosion cores, molten debris, hot gas — gets its RGB *and* brightness from its real Kelvin via the blackbody LUT, then HDR-bloomed. The payoff isn't "prettier orange"; it's that **you can read the physics by eye**: a fire visibly runs hotter (yellowing) when it gets more O2, and cools to sullen red as it starves — because that's the real temperature field driving the color. No sprite-based game can show temperature because it has no temperature. *This is the foundational pick; do it first.*

**② Oxygen-starvation as a visible, physical smoke story.** Breach knows real O2 and real soot yield. So a fire in a sealed room genuinely **chokes itself**: as O2 drops, combustion goes incomplete, soot rises, the smoke blackens and grows *opaque*, which (through the real light engine) **absorbs its own firelight** and the scene goes dark and murky — then crack a door, O2 rushes in on the wind field, the fire flares yellow-hot and the smoke thins to glowing haze. That entire dramatic arc is *emergent from the physics*, not scripted. I have never seen a game where smoke color/opacity and self-shadowing are driven by real combustion chemistry. **This is the "no one has seen this" pick.**

**③ Real god-rays whose color is fire temperature and whose shafts are carved by real soot.** Combine Q1 + Q4: fire emits its true blackbody color into the volumetric solve; steam scatters it into glowing shafts (Alien escape-pod look) while soot absorbs it into hard dark beams — and both the shaft *color* and the shaft *geometry* are live data reacting to the wind pushing gas around. A torch carried through drifting smoke throws colored, moving light shafts that bend as the compressible solver gusts the smoke. Faked god-rays are static screen-space gradients; these are a physical light field.

**④ Explosions as a field-spike, read out as light.** No bespoke explosion asset: inject heat + soot + outward velocity into the real fields and let the existing blackbody + ray-march render it — white-hot flash that blooms and briefly desaturates the frame, a shock-lit smoke shell that self-shadows as it expands on the real velocity field, and long sweeping shadows from the transient bright core. Because it's all fields, **explosions interact with everything**: the blast wind bends nearby flames, pushes existing smoke, and fans or snuffs other fires (via the O2/wind coupling) — consequences that ripple through the same simulation instead of being canned.

**Do-first order:** ① (unlocks all others) → ② (uniqueness, needs only field reads + smoke opacity from soot) → ③ (coefficient tuning on an existing solve) → ④ (scripted field injection reusing ①–③). Notably none of ①–④ require touching the deterministic solver except optional vorticity confinement — they're render-layer reads of fields that already exist, which keeps cross-GPU determinism safe.

---

## 4. Sources

- [Macklin — Blackbody Rendering](https://blog.mmacklin.com/2010/12/29/blackbody-rendering/) — Planck→XYZ→sRGB, dynamic range, Reinhard tone-map with a reference temperature.
- [ScratchAPixel — Blackbody radiation for realistic color](https://www.scratchapixel.com/lessons/cg-gems/blackbody/blackbody.html) — spectrum-to-RGB pipeline.
- [Wikipedia — Planckian locus](https://en.wikipedia.org/wiki/Planckian_locus) — the chromaticity curve blackbody color follows with temperature.
- [zubetto/BlackBodyRadiation (HLSL)](https://github.com/zubetto/BlackBodyRadiation) — real-time in-shader luminance+chromaticity function.
- [Tanner Helland — Convert Temperature (K) to RGB](https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html) — canonical cheap piecewise approximation.
- [Neil Bartlett — color-temperature](https://github.com/neilbartlett/color-temperature) — refined Kelvin→RGB approximation.
- [Bridson, Hourihan, Nordenstam — Curl-Noise for Procedural Fluid Flow (SIGGRAPH 2007)](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf) — divergence-free noise advection, the particle-on-field foundation.
- [Parallel-Cascades — curl-flow Unity demo](https://github.com/Parallel-Cascades/curl-flow-simulation) — procedural curl-noise flow via render textures.
- [kbladin/Curl_Noise (GPU/OpenGL)](https://github.com/kbladin/Curl_Noise) — GPU particle curl-noise implementation.
- [freder — Curl Noise explainer](https://freder.github.io/UnityGraphicsProgrammingBook1/html-translated/vol2/Chapter%206%20_%20Curl%20Noise-Explanation%20of%20Noise%20Algorithms%20for%20Pseudo-Fluids.html) — advect + rising vector for flame look.
- [Andrew Chan — Simulating Fluids, Fire, and Smoke in Real-Time](https://andrewkchan.dev/posts/fire.html) — semi-Lagrangian advection, vorticity confinement, blackbody-for-fire, cost breakdown.
- [Horvath & Geiger — Directable, High-Resolution Simulation of Fire on the GPU (SIGGRAPH)](https://history.siggraph.org/learning/directable-high-resolution-simulation-of-fire-on-the-gpu) — coarse solve + fine refinement paradigm.
- [Fuller, Krishnan et al. — Real-time Procedural Volumetric Fire (I3D 2007)](https://web.cs.ucdavis.edu/~hamann/FullerKrishnanMahrousHamannJoyFirePaperFor_I3D2007AsSubmitted11012006.pdf) — procedural volumetric fire detail.
- [NVIDIA GPU Gems 3, ch.13 — Volumetric Light Scattering as a Post-Process](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process) — cheap screen-space radial-blur god rays; the L = Exposure·Σ(Weight·Decay^i·sample) equation.
- [Grokipedia — Volumetric lighting](https://grokipedia.com/page/Volumetric_lighting) — extinction/absorption/scattering (μ_t, μ_a, μ_s) and ray-march framing.
- [Unity — Realistic smoke with 6-way lighting in VFX Graph](https://unity.com/blog/engine-platform/realistic-smoke-with-6-way-lighting-in-vfx-graph) — 6-way smoke lighting for internal shadow/back-light rims.
- [realtimevfx — Realistic game explosion with EmberGen](https://realtimevfx.com/t/how-to-achieve-a-realistic-game-explosion-with-embergen/27306) — explosion look-dev, flash + shock lighting.
- [Mad-VFX — Interplay of Light & VFX in Games](https://www.mad-vfx.com/blogs/the-interplay-of-light-and-vfx-in-games-how-they-work-together) — transient flash light + glow driving explosion drama.
