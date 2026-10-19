# Breach graphics direction — the look, the gap, and the plan

> **Status**: LIVING DOC, graphics lead: Claude (Erik's ruling 2026-10-18 —
> "I promote you to graphics lead designer"). Erik judges every look at the
> screen (HUMAN-TEST gate, unchanged); this document owns *what we build, in
> what order, and why*. Rulings land in §9 as they are made. Render-only
> throughout: nothing here touches sim state, digests or goldens.
>
> Written after reading the renderer, the shaders, the ray engine, the
> `#90` volumetric branch (`90-volumetric-smoke-fire12`), the open render
> issues (#6 #51 #52 #64 #74 #75 #88 #90 #91) and `docs/plan-for-tuning-and-
> graphics.md`. Everything in §0 is a statement about the code as it is on
> 2026-10-18, not an opinion.

Erik's brief (2026-10-18, paraphrased): exploit the physics engine and the
light engine as *inputs* to the graphics; bulging smoke with sharp edges;
"I want to be impressed"; the `#90` result was nice but not modern, not
totally perfect; the demos online that use similar computations look much
better. Big picture and strategy here; Opus implements.

---

## 0. Where we stand — what the code does today

Inventory of the look layer, one line each, so the gap in §0.2 is checkable.

| Piece | What it does | Resolution it works at |
|---|---|---|
| Ray engine (`cuda_raycaster.cu`, `raycaster.cpp`) | per-source DDA fans; deposits RGB irradiance, a weighted light **direction** (`light_dx/dy`), per-gas Beer-Lambert extinction, and single-scatter **inscatter** (`smoke_glow`) | tile (1/3 m) |
| `shaders/lighting.fs` | floor = diffuse × (ambient + light × N·L), normal-mapped, ACES per pixel | art (48 px/tile) lit by tile light |
| `renderer/gas_medium.py` | ONE premultiplied layer: RGB = tone-mapped tile inscatter, A = 1−exp(−τ) | tile, RGBA8 |
| `shaders/gas_medium.fs` | bicubic upsample of that layer; two-phase advected fBm **erodes** coverage at low density; domain warp; dither | sub-tile |
| `renderer/gas_detail.py::tame_wind` | the one render wind: smoothed, saturating |−∇P| direction — **mis-scaled** (#51) | tile |
| `HeatFieldOverlay` + `blackbody.py` + `speckle.py` | fire = tile ΔT → Planck colour × T⁴ intensity, ACES, additive; dirty-Planck mottle moves with the sim clock | tile, RGBA8 |
| `fire_lights.py` | brightest-16 hot tiles become real ray-cast omni lights (long range) | tile |
| `#90` branch: `gas_volume.fs`, `volume_bake.py`, `tilt_view.py`, `volume_phase.py` | tilted perspective camera; raymarch of a column heightfield h(τ) carved by a baked periodic 4D turbulent scalar; lit by tile light × albedo with vertical self-shadow; lamp direction = ∇(light field); phase field decorrelates the loop | sub-tile in 3D |
| World RT + compose | RGBA8 world RT at 48 px/tile; order: lit floor → smoke → fire → temperature → water → doors → units/props (3D) → effects | — |

What is good and must be kept: the light field is physical (1/r power, per
channel, with a direction); one blackbody ramp; one optical depth definition
(`gas_optical_depth`); the sim-tick clock for every animated texture; the
studio harness (`tools/lighting_demo.py --auto --shot`) that renders a
screenshot headless; the founding principle that graphics cost is decoupled
from sim cost (`docs/architecture/graphics/README.md`, "spend freely on the
look").

### 0.2 The gap — why it does not look like the demos yet

Ranked by how much of the distance each one covers.

1. **The smoke is lit at tile resolution and without direction.** In both
   the 2D medium and the volume pass the colour of a smoke pixel is a *per-
   tile* value (inscatter, or light × albedo). The sub-tile detail only carves
   alpha. Nothing below 33 cm is *lit*: no normals, no self-shadow at the
   scale of the billows, no shadow cast onto the floor, no bright rim where a
   lamp sits behind a thin edge. Every reference that looks "modern" gets its
   beauty from lighting at the detail scale, not from the detail itself.
   (The volume pass has a first directional term, but L comes from the
   gradient of a tile-resolution field and N from a homogeneous noise, so it
   reads as a faint tint.)
2. **The detail is texture, not form.** fBm erosion (2D) and a periodic
   turbulent passive scalar (the bake) are both *homogeneous, Gaussian-ish
   noise*. Smoke reads as smoke because of **convex lumps** — billows with a
   solid core and a fairly hard boundary. That shape comes from Worley-type
   cellular noise (inverted) shaped by a narrow coverage transition, not from
   fBm or a stationary turbulent scalar. This is the "bulging with sharp
   edges" Erik asked for, and it is a density-*construction* question before
   it is a lighting question.
3. **The detail does not move with the physics.** The 2D pass advects by a
   wind that is five orders of magnitude off and tamed to taste (#51); the
   volume pass does not advect at all (the phase field offsets the loop per
   tile). Motion that is not the flow reads as a screen effect; motion that
   is the flow reads as physics. Pillar 4 below makes this a rule.
4. **The macro field is blurry in space and binary in time.** Donor-cell
   upwind transport smooths plumes to tile-scale mush; the smoke unit is so
   coarse that opacity is ~1 until it vanishes (#88, #6). The renderer can
   *sharpen* edges but cannot invent the 1–4 tile plume structure the sim
   lost. Decide where that band of structure comes from (sim fix vs. a
   synthetic low band in the detail field) — §2.5.
5. **Fire has colour but no shape.** Blackbody tile blobs with mottle; no
   tongues, no lean, no embers, no shimmer. `#52` is the right design and it
   is parked.
6. **Draw order has no height.** Smoke is drawn under units and under fire,
   so a marine is never in smoke and a fire never burns through its own
   smoke. Species have no vertical identity (hot soot at the ceiling vs.
   teargas on the floor).
7. **Tone-mapping is applied per layer, in 8 bit.** `lighting.fs`, `lit3d`,
   `gas_medium.py`, `HeatFieldOverlay` and `gas_volume.fs` each ACES their
   own output and composite in an RGBA8 world RT (`water.fs` does not, so
   water and floor already disagree about exposure). Adding a
   tone-mapped emissive on top of a tone-mapped floor re-curves the sum and
   clips to white; the god-ray inscatter is quantised to 8 bits before it is
   added. This is the most likely root of "fires read too white" (#74) and
   it means every look so far was tuned on a pipeline that cannot represent
   the thing being tuned. Fix first (tune-once principle, §8 G0).

Not on the list on purpose: resolution (the bake at 32 texels/tile is
plenty; 48 px/tile on screen is plenty), the ray engine (it is the strongest
asset we have), bloom (Erik's ruling stands: none).

---

## 1. Art direction — what "impressive" means for Breach

Five pillars. A patch that violates one is wrong even if the screenshot is
pretty.

1. **Darkness is the canvas.** Ambient is faint starlight. Lamps carve
   pools; outside them the ship is near-black and desaturated. The
   prototype's two-state lit/dark reveal (#64, "I've never seen any other
   game do graphics like this") is the signature, brought back as a shader
   family with one dial for edge crispness.
2. **Light is made visible by the medium.** Beams in steam, shafts through
   soot, the shadow a smoke column throws across a corridor, a marine
   silhouetted in a doorway full of haze — the *Aliens* escape-pod scene. The
   ray engine is the star of the show; smoke, steam and dust exist to let
   you see what it does.
3. **Fire is physics you can read.** Colour is temperature, dirt is soot,
   motion is wind, a choking fire blackens and re-flares when air comes back.
   (Standing ruling from the Fire & Heat Beauty arc; kept.)
4. **Everything moves with the sim.** Every animated detail — billow noise,
   flame tongues, embers, shimmer, speckle — is advected by the real
   velocity (m/s, #51 fixed) and clocked on the sim tick. A static or
   wall-clock-driven pattern is a bug (the speckle hard rule, generalised).
5. **Scale honesty.** One tile is 33 cm and the camera sits at 48 px/tile:
   we are close enough to see centimetre texture. Detail is authored at the
   2–10 cm scale and the sim owns everything above a metre. The gap between
   those two (the 30 cm–1 m billow band) is the one place the renderer may
   *synthesise* structure, and only with coherent, flow-advected form.

### 1.1 Reference board (to assemble in `docs/graphics_refs/` as we go)

- Cloud tops from above (satellite / aircraft photographs): the exact view
  geometry of a top-down smoke layer — convex billows, crisp sunlit crests,
  dark crevices, **shadows on the ground**.
- *Aliens* (1986), the Sulaco and the escape-pod scene: beams through haze,
  near-black rooms, light that reveals rather than illuminates.
- Horizon Zero Dawn "Nubis" clouds (Schneider & Vos 2015): Perlin-Worley
  base shape, Worley erosion, the powder term, in-scatter lobes — the
  canonical modern volumetric look.
- Teardown's smoke/fire (voxel heightfield smoke lit by cast light; shadows
  cast by smoke onto the scene) — the closest *gameplay* analogue to what
  §2 proposes.
- Noita, Rain World, Pixel-fluid demos: 2D sims whose charm is that every
  pixel of the effect *is* the physics.

---

## 2. Smoke — a lit relief seen from above (the core proposal)

**Thesis.** In a top-down game, three-dimensional smoke seen from above is a
*relief*: a height field with a surface. What the eye reads is (a) the
silhouette and coverage, (b) the shading of that surface by lamps and fire,
(c) the shadow it throws on the floor and on units, (d) what glows through
it from below. A full raymarch adds information only when the camera tilts.
So we build the smoke as a **surface model** — density construction and
lighting that are view-independent — and render it with one of two
integrators: a 2D relief shader for the canonical top-down view, and the
`#90` raymarcher for the tilted cinematic mode. The model is shared; the
integrators are thin.

### 2.1 Density construction (per pixel of the world RT)

- `tau` — the sim's optical depth, bicubic, unchanged (the single source).
  After `#88` it has temporal resolution; until then `tau_curve_b` steepens
  it in τ-space as now.
- `B(x, y, t)` — the **billow field**. Two or three octaves of *inverted
  Worley* (cellular) noise warped by a low-frequency fBm — the Nubis
  "Perlin-Worley" recipe — authored at wavelengths of roughly 1/4, 1/8 and
  1/16 tile (8, 4, 2 cm). Convex lumps, not filaments. Advected by the
  **real** wind with the two-phase flow-map crossfade the 2D pass already
  has (keep `advected_noise.py`'s clock), plus a per-tile **boil rate** from
  temperature (hot smoke churns faster — the phase-field idea from `#90`,
  now applied to the advection clock, not to a loop offset).
- `C = smoothstep(t0, t1, tau' + k_b · (B − 0.5))` — **coverage** with a
  *narrow* transition `t1 − t0`. This is the sharp edge. One dial for
  crispness, one for how far the billows push the edge in and out.
- `h = h_max · sat(tau / tau_h)^½ · (1 − a + a · B)` — the **relief height**.
  Thick tiles grow tall columns; the billow field gives them a bulging top.
- **Species have heights.** Hot soot (`smoke`) is a *ceiling layer*: tall,
  drawn **above** units. Steam, teargas, poison are *floor layers*: low,
  drawn **below** units so marines wade in them. Two draws of the same pass
  from two τ planes (`gas_optical_depth` per species group). This is
  physically right (hot combustion products stratify at the ceiling) and
  it is the single cheapest way to make the world read as 3D.

### 2.2 Lighting the surface

- `N = normalize(−∂h/∂x, −∂h/∂y, 1)` from a Sobel on `h` (cheap; `h` is in
  the same shader).
- `L` per pixel from the tile light direction (`light_tex_a.a`,
  `light_tex_b.a`), lifted to 3D with a **lamp height** assumption: ship
  lamps are at the ceiling, so a ceiling-layer smoke is lit from *above and
  the side* (cloud tops), a floor-layer haze from above. Fire lights come
  from below: their contribution shades the *underside*, which from above
  shows as glow through thin coverage (§2.3), not as surface shading.
- Diffuse = wrap-lit `N·L` (thin smoke never goes black) × the tile's
  incoming RGB × albedo (soot 0.22 per #91; steam ≈ 0.9).
- **Self-shadow**: 4–8 tap horizon march along `L` in the height field —
  crevices dark, lamp-facing crests bright. This is what makes a billow a
  billow.
- **Powder / multi-scatter**: `(1 − exp(−2σ)) · exp(−σ)` on the in-scatter so
  thin edges look fluffy-bright rather than grey, plus an ambient-occlusion
  term from the Laplacian of `h`.
- **Rim**: Henyey-Greenstein forward lobe against the lamp when the lamp is
  behind a thin edge (silver lining); only where coverage is thin.
- **Transmittance** `T = exp(−tau' · C)`: the floor is darkened by `T`; the
  tile inscatter from the ray engine (god rays) is added, *modulated by `C`*
  so the beam scatters where the smoke actually is, not uniformly over the
  tile.
- Output stays premultiplied `(inscatter, 1 − T)` in **linear HDR** (§6);
  the compositing contract of `gas_medium.py` is unchanged.

### 2.3 Fire through smoke and the shadow on the floor

- **Glow through**: emissive `E = blackbody(ΔT_tile) · (1 − C_core)` — fire
  under a thin or ragged part of the smoke glows through; under a solid
  core it does not. The soot *mottles the flame* (the dirty-Planck idea)
  because `C` is the mottle.
- **The cast shadow** (this one sells the 3D on its own): in `lighting.fs`
  (floor) and `lit3d` (marines, props), sample the smoke's height/coverage
  at `p − L_xy · h(p) · k_parallax` and multiply the incoming light by that
  column's transmittance. A smoke column in a lamp-lit corridor throws its
  shape across the floor and over the squad. The ray engine already
  attenuates light *through* smoke in the plane; this adds the vertical
  shadow the plane cannot know about. Cheap: one extra texture read per
  floor pixel.

### 2.4 The two integrators

- **Top-down (canonical)**: the relief shader replaces `gas_medium.fs` in
  its slot — same inputs (τ texture, dynamics texture, light textures),
  same premult output, new noise assets. The erosion-only look is kept
  behind the toggle for A/B.
- **Tilt (cinematic, `#90`)**: `gas_volume.fs` keeps its march but the
  *density* it marches becomes `C · column(h)` with the billow field
  sampled in 3D (the bake's z, or a 3D Worley), and the lighting uses the
  same N·L / horizon-shadow / powder terms. The bake becomes a *detail
  modulator*, not the shape. Phase field and stochastic tiling stay as
  anti-repetition tools.

### 2.5 Where the 30 cm–1 m band comes from

The sim is diffusive at this scale and the renderer must not pretend
otherwise. Two honest options, to be judged on screen in G1:

- **A — synthetic low band**: the billow field gets a coarse octave at
  ~1 tile, *gated by `tau` and its gradient* (only inside plumes and only
  where the sim says the field is changing), advected by the real wind.
  Render-only, no ceremony.
- **B — sim sharpness**: MacCormack/BFECC on the trace transport (the
  written trigger in the B2 design: "reopen only if plumes lose silhouette
  while travelling"). Physics-side, digest ceremony, Erik's physics session.

Recommendation: ship A in G1 and measure whether B's trigger has fired by
looking at plumes in motion with A on. If the plumes still lose their shape
while travelling, B goes on the physics list.

---

## 3. Fire — shape, embers, shimmer

`docs/fire_vfx_plan_2026-08-24.md` (#52) is the right design: a render-only
mini-Stam loop on a finer grid, one-way fed by the sim, rendered through the
blackbody ramp. Two changes to its plan:

1. **Move it up.** Fire is the hero of every screenshot and every video in
   the outreach plan. It was parked behind the physics tuning lid; that lid
   is about *amounts*, and flame structure is about *shape*. They are
   independent, and the VFX sim's input signal (`I`, `T`) will not change
   kind in tuning. G3 in §8.
2. **Add the three cheap things that sell fire** in the same arc, after the
   loop works:
   - **Embers** — the canonical particle primitive (#52 F4): spawned from
     the hottest tiles, ride the real wind + curl noise, cool along the
     blackbody ramp, fade; a few hundred sprites, CPU.
   - **Heat shimmer** — a refraction pass over the whole world RT: displace
     by `∇T` upsampled with advected noise, so hot air above a fire wobbles
     the floor and the smoke. (Already in the plan; it is a 30-line shader
     once the HDR RT exists.)
   - **Flicker** — modulate each fire light's intensity by the local
     reaction `R` of the VFX sim (resolves the "steady glow" open question:
     steady light from the sim, flicker from the VFX, render-only).

Top-down flames: tongues lick *outward* from the burning solid's open faces
(the plan's injection rule), lean with the wind, and "rise" is drawn as
brightness and width, not displacement — plus the relief machinery of §2 if
we want a flame *surface*: the same N·L on a flame height field gives
tongues a lit side. Try it; it may be the cheaper path to structure than a
tall VFX grid.

---

## 4. Explosions (#75)

Built from the parts above, so it comes after them:

- **One discrete light** per detonation with a predetermined envelope
  (rise < 50 ms, decay ~300–600 ms by payload class), colour from the
  blackbody ramp at the payload's peak T, very long range: the ray engine
  does the long rays and long shadows for free.
- **Chaotic shadows** fall out of §2.3: the blast's own smoke column throws
  a moving shadow from a light that is itself moving in brightness.
- **The shock front** is *refraction*, not colour: displace the world RT by
  `∇P` (the pressure field) through a ring mask — the old pressure colormap
  washed out because it painted pressure; a refractive ring is invisible
  until it crosses something, which is exactly what a shock wave looks like.
- Debris and sparks from the particle primitive.
- Explosion smoke: heavy, dark, ceiling layer (§2.1) — it should visibly
  *hang* and then thin (needs #88).

---

## 5. Darkness and beams (#64)

The two-state family in `lighting.fs` and `lit3d.py` (one copy of the
function in the lit3d seam, per the canonical-systems rule):

```
dark = (luma + (albedo − luma) · u_dark_sat) · u_dark_gain   // prototype: 0.15, 0.08
lit  = mix(dark, albedo · light, smoothstep(u_lit_t0, u_lit_t1, lum(light)))
```

`u_lit_t1 − u_lit_t0 → 0` is the prototype; wider is a soft reveal. Judged
in the studio with the flashlight and the beacon. Beams: the tile inscatter
modulated by sub-tile coverage (§2.2) already makes beams *textured*; a
screen-space radial march from each lamp through `C` (classic "sunrays")
can come later if the tile-resolution shafts still look blocky.

---

## 6. Pipeline — one HDR world RT, one tone-map

Every layer currently tone-maps itself into an RGBA8 target (§0.2 item 7).
The fix is mechanical and it is the foundation everything in §2–§5 is tuned
on:

- World RT → RGBA16F (`core.create_dynamic_rgba16f_texture` exists; the RT
  needs `rlLoadTexture`-level creation — raylib's `load_render_texture` is
  RGBA8, so this is a small `gl3d.py`-style shim, which `#90` already
  started).
- All layer shaders output **linear, un-tone-mapped** HDR; the CPU-packed
  layers (`gas_medium.py`, `HeatFieldOverlay`) become 16F uploads or move
  their packing into the shader.
- ONE ACES at the final blit to the screen (`WorldComposite`), plus the
  dither there. `u_light_gain` stays the single exposure dial.
- No bloom (ruling). What people like about bloom on fire — the warm halo —
  is *physically* the inscatter of the fire's light in the smoke/haze around
  it, which the ray engine already computes; with HDR intact it will show.

Gate: a pixel-diff harness shot before/after with the fire studio at default
dials should look *the same or better* in the dark and *less clipped* around
fire. This is G0 and it is small.

---

## 7. How we work (lead's rules for every graphics arc)

- **Look-driven loop.** Every patch renders a fixed set of studio shots
  (`tools/lighting_demo.py --level fire_studio --auto --shot`, fixed camera,
  fixed seed, fixed tick count) *before and after*, and the implementing
  agent **looks at the PNGs** before claiming the patch works. The shot set
  lives in `tools/look_shots.toml` (to create in G0): dark corridor with one
  lamp; fire + plume; steam in a flashlight beam; explosion at +4/+12/+40
  ticks. Erik judges the same shots live (HUMAN-TEST), so the two of us are
  looking at the same thing.
- **Single sources, no parallels** (CLAUDE.md): one τ (`gas_optical_depth`),
  one blackbody ramp, one light field, one wind seam (`tame_wind`, rescaled
  by #51), one tone-map, one particle primitive, one noise clock.
- **Render-only, by construction.** Reads dequantized copies; writes
  nothing synced; never runs headless. The `fire_lights` `heat = 0.0` pin is
  the precedent.
- **Budget**: ≤ 2.0 ms/frame for smoke at 1080p on Erik's desktop, ≤ 1 ms
  for fire VFX at M=8, ≤ 0.3 ms shimmer; laptop tier = half the step counts.
  Measured in the studio (`--perf`) at the end of each arc.
- **Design → adversarial critique → patches with gates → HUMAN-TEST** (the
  repo's arc shape). Graphics critiques are *visual*: the critic must render
  the shots and argue from them.
- **Credit the source** in every file header (repo rule).

---

## 8. Roadmap — order and rationale

| Arc | Deliverable | Gate | Depends on | Size |
|---|---|---|---|---|
| **G0 Foundations** | HDR world RT + single tone-map (§6); #51 wind in m/s with `tame_wind` recalibrated on the honest scale; the look-shot harness (§7); soot albedo 0.22 on main (#91) | shots identical-or-better; perf unchanged | — | 1–2 sessions |
| **G1 Smoke relief** | §2.1–§2.2 in the top-down integrator: billow noise assets (Worley/fBm bake + 3D variant), coverage + height, N·L, horizon self-shadow, powder, rim, coverage-modulated inscatter; species layers (ceiling soot above units) | **HUMAN-TEST** on the shot set; ≤ 2 ms | G0 | 2–3 sessions |
| **G2 Shadows and glow-through** | §2.3: smoke shadow on floor and units; fire glow through coverage; draw-order change; synthetic low band (§2.5 A) | **HUMAN-TEST**; plume-in-motion judgement for §2.5 B | G1 | 1–2 sessions |
| **G3 Fire** | #52 F0–F3 (design session with Erik first) + embers, shimmer, flicker | **HUMAN-TEST** | G0 (+G1 if the flame-relief variant is tried) | 3–4 sessions |
| **G4 Explosions** | §4 | **HUMAN-TEST** | G2, G3, #31 generic explosion | 1–2 sessions |
| **G5 Darkness** | §5 two-state family | **HUMAN-TEST** | G0 | 1 session |
| **G6 Tilt reconciliation** | §2.4: `#90` raymarcher on the shared model; decide whether tilt is a shipping mode | **HUMAN-TEST** | G1, G2 | 1–2 sessions, optional |

Physics-side prerequisites that are *not* graphics work but gate the look:
`#88` smoke unit rescale (decay, thinning — without it every plume is
opaque-then-gone, and G1's edges will be tuned on a field that has no
gradient in time); `#6` saturation; `#12` fire tuning. Erik's physics
sessions; G1 can be built before them but should be *judged* after #88.

What this retires or re-sequences: `#52` moves ahead of the tuning lid
(§3); `#90`'s "v2 ideas" list (flow-map advection, lamp shadows, depth-
composited marines) is absorbed into G1/G2/G6; the 2D erosion look stays
as an A/B toggle until G1 is blessed, then goes.

### 8.1 Opus kickoff — G0 (ready to run)

> You are implementing arc G0 of `docs/graphics_direction_2026-10-18.md`
> (read it first, then CLAUDE.md's Render/UI rows). Three patches, each its
> own commit, render-only:
> **P1 HDR pipeline.** Make the world RT RGBA16F (raylib's
> `load_render_texture` is RGBA8 — add a `gl3d.py`-style creation shim, or
> `rlLoadTexture` + `rlLoadFramebuffer` via rlgl), remove `aces_tonemap` from
> every layer that applies it (`lighting.fs`, `lit3d.py` GLSL,
> `gas_medium.py`, `HeatFieldOverlay`/`blackbody.pack_emissive_rgba`; check
> `water.fs` outputs linear and stays so), convert the two CPU-
> packed layers to 16F uploads (`core.create_dynamic_rgba16f_texture`), and
> apply ONE ACES + dither in `WorldComposite`'s final blit. Keep
> `u_light_gain` as the only exposure. Gate: `tools/lighting_demo.py --level
> fire_studio --auto --shot` before/after at default dials; the dark scene
> must match within dither and the fire core must clip less. Look at both
> PNGs.
> **P2 Wind units (#51).** `tame_wind` reads `wind_x/y` as Q16.16 m/s;
> tiles/tick = v / tile_size_m / tps from config, no magic constants; then
> retune `v_ref`/`v_sens` so the smoke's visual drift matches the plume's
> centroid drift measured on the same run (write the measurement down).
> **P3 Look-shot harness.** `tools/look_shots.toml` + a `--shots` flag on
> the demo that renders the named shots (fixed camera/seed/ticks) to
> `scratchpad/look/<name>_<git-sha>.png`; the four shots in §7.
> Also: `[gases.smoke] scatter_albedo` 0.22 (#91, Erik's ruling on the
> volumetric branch) ported to main.
> HUMAN-TEST gate before merge (feel-adjacent). Never `git add -A`.

### 8.2 Opus kickoff — G1 (after G0 is merged)

> Implement §2.1–§2.2 and §2.4's top-down integrator. Start with the noise
> assets: a Python baker (`renderer/noise_bake.py`, pyray-free, tested for
> tileability) producing a 2D Perlin-Worley "billow" texture (3 octaves,
> inverted Worley cells, fBm warp) and a 3D variant for the tilt mode; keep
> `advected_noise.py`'s two-phase clock. New shader `shaders/gas_relief.fs`
> in the `gas_medium.fs` slot behind `[render.gas_relief] enabled`; inputs =
> the existing τ/dynamics/light textures plus the billow texture and a
> temperature texture for the boil rate. Implement in this order, one
> screenshot per step, looking at each: coverage with narrow transition →
> height field → N·L with lamp-height L → horizon self-shadow → powder/AO →
> rim → coverage-modulated inscatter → species layers (ceiling soot drawn
> after units). Dials in config, Ctrl+R live. Budget ≤ 2 ms at 1080p.
> HUMAN-TEST.

---

## 9. Decisions needed from Erik (and my recommendation)

1. **Top-down stays the canonical camera; tilt is an optional cinematic
   mode built on the same smoke model.** (Recommend yes — the RL, the 3D
   marines, walls, UI and levels are all built for top-down; the tilt view
   cost a flat squad and decal walls.)
2. **Hot soot is a ceiling layer drawn above units; steam/teargas/poison are
   floor layers below units.** (Recommend yes.)
3. **G0 (HDR + one tone-map) goes first, before any look is tuned further.**
   (Recommend yes — tune-once.)
4. **`#88` smoke-unit rescale is scheduled as a physics session before G1 is
   judged.** (Recommend yes; G1 can be built before it.)
5. **`#52` fire VFX moves ahead of the fire-tuning lid.** (Recommend yes.)
6. **Bloom stays off**; the warm halo comes from HDR + inscatter. (Confirm.)
7. **Synthetic 30 cm–1 m band (§2.5 A) is allowed** as render-only
   structure, gated by τ and advected by the real wind. (Recommend yes,
   with B's trigger re-checked afterwards.)

Rulings land here with a date.

### 9.1 Rulings

**2026-10-18 — Erik: YES to all seven, as recommended.** In force from this
date:

1. Top-down is the canonical camera; tilt (`#90`) is an optional cinematic
   mode on the shared smoke model (§2.4), never a schedule driver.
2. Hot soot is a ceiling layer drawn above units; steam, teargas and poison
   are floor layers drawn below units.
3. G0 (one HDR world RT, one tone-map) goes first; no further look tuning
   until it is merged.
4. `#88` smoke-unit rescale is a physics session scheduled before G1 is
   *judged* (G1 may be built before it).
5. `#52` fire VFX moves ahead of the fire-tuning lid (G3).
6. Bloom stays off; the warm halo comes from HDR + the ray engine's
   inscatter.
7. The synthetic 30 cm–1 m billow band (§2.5 A) is allowed, render-only,
   gated by τ and advected by the real wind; §2.5 B's trigger is re-checked
   on plumes in motion once A is in.

Next action: hand §8.1 (the G0 kickoff) to Opus on a fresh worktree branch
off `main`; this branch carries only the doc and merges first. (Superseded: see §10, G0 runs on `fire-12`.)

---

## 10. Rebase note: G0 runs on `fire-12`, not `main` (2026-10-19)

This doc was written against `main`'s renderer. In this repo the live line
is `fire-12`: about 460 commits ahead of `main`, never merged back, and
Erik's #12 line. Only `main` has the old light path that §0 describes.
On `fire-12`, ray-engine v2 (closed 2026-09-30, tag `ray-engine-v2`)
**deleted** `raycaster.cpp`, `cuda_raycaster.cu` and `fire_lights.py`.
Light now rides the radiation sweep (`cpp/src/radiation_sweep.*`,
`cuda_radiation_sweep.*`, P6a–P6d). So, where §0 and §7 disagree with
`fire-12`, the code wins:

- §0's "Ray engine" and "`fire_lights.py`" rows describe `main`. On
  `fire-12`, the light field, its direction and the inscatter (`smoke_glow`)
  come from the sweep. Re-read the renderer's light inputs on this tree
  before G1 builds on them.
- §7's "`fire_lights` `heat = 0.0` pin" precedent no longer exists as a
  file. The render-only principle it illustrated still stands.
- **G0's spec (§8.1) holds unchanged on `fire-12`.** Checked on
  2026-10-19: the same layers call `aces_tonemap` (`lighting.fs`,
  `lit3d.py`, `marine_shader.py`, `static_props.py`, `gas_medium.py`,
  `blackbody.py`). `WorldComposite` still makes an RGBA8 RT with
  `rl.load_render_texture` and applies no tone-map. `fire_studio` and
  `tools/lighting_demo.py` exist. #51 is still open.
- Read §8.1's "ported to main" (the #91 albedo) as "ported to this branch".
  It comes from `90-volumetric-smoke-fire12` (`fd8a6f5`).
- **#90 is not merged** (it waits on HUMAN-TEST). Its `gas_volume.fs` calls
  `aces_tonemap` too. Whichever of G0 and #90 lands second removes that call
  as part of its merge. G0 does not pull in #90.
- `smoke_light_studio` (on `fire-12` only) is the better smoke shot. Add it
  to the §7 shot set.

Base: G0 is cut from `fire-12` on branch `93-g0-foundations`
(tracker #93) and merges back into `fire-12` `--no-ff` after Erik's
HUMAN-TEST. `main` gets it when `fire-12` merges, and that is Erik's call.

- **2026-10-19, G0 P2 (#51), the wind measurement.** `tame_wind` now reads
  the planes as m/s and converts with `v / tile_size_m / tps` (0.333 m,
  24 tps: 1 m/s = 0.125 tiles/tick). `tools/wind_drift_probe.py` on the
  smoke-light studio (ticks 0-168) and a fire_studio frag blast (ticks
  24-120): the plume's tau-weighted centroid drifts **0.12-0.78 tiles/tick
  while a blast moves the air and ~0.01 once it settles**; the plume-weighted
  honest speed is 0.5-4.6 (blast) and 0.07-0.4 (settled sloshing). The old
  dials capped the visual drift at 0.08, ~10x too slow in a blast (median
  |log(visual/centroid)| 0.81 over 215 source-free 6-tick windows). Retuned to
  `v_ref 1.4, v_sens 3.0` tiles/tick: median 0.25, mean 0.40, a broad
  valley along `v_ref/v_sens` ~0.35-0.5. Smears at blast speeds (the
  crossfade cycle was sized for 0.08) are a HUMAN-TEST item; `cycle_seconds`
  is the dial.

### 10.1 Systems (G0)

**(a) Existing canonical systems G0 must use:**
- GameRenderer + WorldComposite: the one world RT, made HDR here.
- LightingPass: `light_tex_a/b`, never a second light path.
- Gas medium: the only smoke look.
- Tamed wind (`tame_wind`): rescaled by #51, never re-derived. Correct its
  CLAUDE.md row in the same patch (#51's ruling).
- Blackbody: the one ΔT→colour map.
- Lit-3D seam (`lit3d.py`): its tone-map is removed here. The marine
  byte-identity gate `test_lit3d_extraction.py` must be re-baselined in
  the same commit, with written rationale.
- The dequantize convention.
- `tools/lighting_demo.py`: the studio harness, extended rather than
  copied.
- Config, via `CFG`.

**(b) New systems G0 creates, with draft rules** (they go into CLAUDE.md
at implementation, pointing at the real code):
- **One tone-map**: "Every world layer writes linear HDR into the RGBA16F
  world RT. `WorldComposite`'s final blit is the ONLY place ACES and the
  dither run, and `u_light_gain` is the only exposure. A layer that
  tone-maps itself is the bug."
- **Look shots**: "`tools/look_shots.toml` holds the named shot set. Every
  graphics patch renders it before and after with `--shots`, and the
  implementer looks at the PNGs before claiming the patch works. Never a
  second screenshot harness."
