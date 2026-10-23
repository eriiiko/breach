# G3 Fire — design (F0), 2026-10-23

*Graphics arc G3 of `docs/graphics_direction_2026-10-18.md` (§3, §8), issue
#52 under tracker #93. Builds on `docs/fire_vfx_plan_2026-08-24.md` (the
approach) and revisits its locked decisions in the light of G0–G2 and the
#12 fire + smoke session (2026-10-22). Base: `main` at `cfcba40` (fire-12
merged). Status: DRAFT for adversarial critique, then Erik's approval.*

---

## 0. What G3 is for

A fire today is a hot area: the tile glows in black-body colours (the heat
layer), it lights the room (the radiation sweep), and it smokes (G1/G2). It
is physically right and has **no shape** — no tongues, no lean, no motion
between two sim ticks. G3 adds flame structure, render-only, then the three
cheap things that sell fire: embers, heat shimmer, flicker.

## 1. Erik's decisions (2026-10-23, design session)

1. **Camera: straight down.** G3 is designed for the top-down camera. A later
   tilt of **at most 15–20°** is kept open (G6): the flame's shape must come
   from a field that can later be given height, not from a top-down-only
   trick.
2. **No "lit relief" for flames.** A flame emits its own light; it has no lit
   and shadowed side (Erik's objection, accepted). G1's N·L machinery is not
   used on the flame. The physically honest structure cues are emission ones
   (§3.4).
3. **The comparison he judges:** *glow by temperature only* (A) vs *glow by
   temperature and thickness* (B) — one config toggle, look shots of both.
4. **Scope:** flames first (F1–F3) → HUMAN-TEST 1; then embers, shimmer,
   flicker (F4) → HUMAN-TEST 2; then merge. One arc, one branch
   (`93-g3-fire`).
5. **No reference board up front.** Erik judges by reacting to look shots of
   the real fire in `fire_smoke_studio`, as in G1/G2.

6. **Flicker = option A (§6.3), Erik 2026-10-23:** the flame, embers and
   glow flicker; the room light stays the sim's steady light. G3 does not
   touch the sim, which decides it (C would have been his other pick). The
   related darkness under smoke is #100, not G3 (below).

**Related ruling, not G3 (Erik, 2026-10-23):** smoke over a fire swallowing
all of its room light is a physics gap. The direction is #100's option 3,
stratified smoke inside the sweep (hot soot fills from the ceiling down;
light at occupant height sees only the part that has come down). It is for a
later physics session and is out of scope here.

## 2. What the code does today (facts, `cfcba40`)

- **Heat layer** (`renderer/overlays.py::HeatFieldOverlay`,
  `renderer/blackbody.py::pack_emissive_rgba`): reads `gmap.temperature`
  only, one texel per tile (RGBA8, bilinear), CPU-packed every frame through
  `BlackbodyRamp.chroma_intensity` → `aces_curve` (the CPU twin of the
  resolve's ACES). Drawn **after** the resolve, display-space, additive, and
  through the smoke via `glow_through_fs` (G2). It is G0's **named temporary
  exception "until the G3 fire arc replaces it"** (CLAUDE.md, One tone-map).
  Speckle (`renderer/speckle.py`, dirty Planck) modulates it at pack time.
- **Fire state**: `GameMap.fire`, Q16.16 logistic intensity `I ∈ [0,1]` on
  `flammable` solid tiles (wood, furniture, kindling, foliage). Only the
  legacy flat-orange `FireOverlay` reads it (off by default).
- **Room light**: the radiation sweep's `light_q` / `light_glow`, one
  combined field per tick; every cell emits `a·L°[T]` from the checked-in
  table generated from the blackbody ramp. Fire is **not a separable source**
  — lamps, sky and hot cells sum into one field. Uploaded as `light_tex_a/b`
  once per tick.
- **Clocks**: sim 24 tps, render 60 fps, no interpolation — frames between
  ticks see identical state. `advected_noise.advection_phase` is the one
  noise clock (tick-locked).
- **Wind**: `gas_detail.tame_wind` → per-tile tiles/tick, smoothed,
  saturating (`v_ref` 1.4). THE render-side wind seam.
- **GPU passes**: `core.load_render_texture_rgba16f`,
  `create_dynamic_rgba16f_texture`, the `GasReliefPass` pattern (manual
  sampler units from 5, `tile_rect` view clip, product RT at 8 px/tile).
  **No ping-pong / iterative pass exists yet.**
- **No particle system, no screen-space distortion pass** (the
  `world_composite.py` header reserves the post-FX slot).
- `fire_lights.py` is gone (ray-engine-v2 P6c). The blackbody ramp has no
  GPU LUT.
- Levels: studios 48×32 to 56×56; largest test level 128×256. World RT 48
  px/tile. RTX 3070 on the work desktop.

## 3. The flame model

### 3.1 Layer split (unchanged from the plan)

- **Layer A — the game (authoritative, untouched):** `fire` (I),
  `temperature`, wind, solids.
- **Layer B — the flame VFX (this arc, render-only):** fields on a finer
  grid, read from dequantized copies, writing nothing synced, never running
  headless. Digests cannot move; a test proves it (§8).

### 3.2 Fields (Layer B)

| Field | Grid | Meaning |
|---|---|---|
| `u_vfx` (vec2) | M_u = 4 px/tile | flame-scale air velocity, tiles/s |
| `p`, `div` | M_u | projection scratch |
| `R` | M = 8 px/tile | flame density: "how much flame is in this column of air" |
| `T_vfx` | M | flame temperature (game units), carried with the flame |
| `solid` | M (per tick) | boundary mask from the game's blocking tiles |

**Why `T_vfx` exists** (plan's open question 2): the game's temperature is
one value per tile, so it cannot make a tongue white at its root and red at
its tip. `T_vfx` is seeded at the source from the game's own temperature and
only **cools** as the flame travels (`T_vfx ≤ T_source`), so a flame is never
hotter than the sim says, and its colour stays consistent with the room light
the sweep computes from the same temperatures.

**Why two resolutions:** velocity and pressure carry the large swirls and
cost most (the Jacobi solve); flame density needs the crisp edges. Half-res
velocity is standard practice (Harris, GPU Gems ch. 38) and halves the solve.

### 3.3 The loop (per render frame, fragment passes)

```
per tick (24/s):  upload sources (I, T at open faces), solid mask, tamed wind
per frame (60/s), dt = frame dt (fixed 1/60 in shot mode):
  1. advect u_vfx (semi-Lagrangian, bilinear); relax toward tamed wind (k_follow)
  2. inject: at OPEN FACES of burning tiles, add R, T_vfx, and an outward
     velocity kick ∝ I  ("tongues lick outward from the open faces")
  3. vorticity confinement (ε)            ← the curls and tongues
  4. curl-noise forcing, small (curl_amp) ← breaks symmetry between ticks
  5. divergence → Jacobi (n_jacobi, ~20) → subtract ∇p; solids = no-through
  6. advect R, T_vfx (SL; BFECC/MacCormack if SL looks soft at F2)
  7. decay R (lifetime), cool T_vfx toward the local game T
```

- **Source law** `f(I, T)`: rate ∝ `I` (a dying fire shrinks for free),
  `T_vfx` seeded from the adjacent gas cell's game temperature. Open face =
  a face of a burning tile whose neighbour is non-blocking. Pure numpy
  function, tested (§8).
- **Top-down has no buoyancy** in the plane. "Rise" is drawn as emission
  (§3.4), never as a sim force.
- **Wind** comes only through `tame_wind` (canonical seam), converted to
  tiles/s.
- **Domain:** whole-map, world-anchored (plan decision 1 kept), but every
  pass is **scissored to the active region**: the bounding box of burning
  tiles plus a margin that covers a flame's lifetime of travel. No fire →
  zero cost. Off-screen fires keep simulating.
- **Between ticks** the VFX runs at frame rate, so the flame moves smoothly
  even though the game state steps at 24 Hz. This is new motion the game
  never had.

### 3.4 How it is drawn

Emission only — a flame is a light source, not a lit surface (decision 2):

```
L_flame = BlackbodyRamp(T_vfx) · E(R)       (linear HDR, into the scene RT)
  A  "temperature only":       E = smoothstep(r0, r1, R)        — a crisp mask
  B  "temperature + thickness": E = 1 − exp(−κ·R)                — column emissivity
```

B is the physically honest one: from straight above, a taller tongue is a
longer column of glowing gas, so more light reaches the eye. A treats every
part of the flame as equally bright for its temperature. Both get:

- **ragged edges**: the billow noise (`noise_bake`, advected on the flame's
  own velocity, the one noise clock) erodes `R` at the edge, as G1's
  coverage edge does;
- **colour from the one ramp**: a GPU LUT of `BlackbodyRamp`'s chroma and
  intensity curves (new, §9), so the flame, the glowing solids and the
  sweep's L° all come from one map;
- **soot darkening** (later, optional): the ceiling soot over a flame
  occludes it through the normal composite (§3.5); no extra term.

**Tilt-readiness (decision 1):** `R` and `T_vfx` are a 2D density field. A
15–20° tilt can extrude it as a short height column (`R` → height), the
same way the smoke relief turns τ into height. Nothing in F1–F3 assumes the
camera is vertical except the final draw.

### 3.5 Where it sits in the frame — and the heat-layer exception retires

- The flame pass draws into the **linear HDR scene RT** inside
  `compose_world`: after the floor and the floor gas layer, **before the
  units and the ceiling soot** (flames are crate height; a marine's head and
  the ceiling soot are above them).
- The ceiling soot's normal alpha composite then hides the flame by its own
  transmittance. In linear HDR that is the physically right glow-through;
  G2's display-space `glow_through_fs` weight was needed only because the
  heat layer was drawn after the resolve.
- **The heat layer moves into linear HDR** at the same slot: glowing solids
  and hot gas keep their black-body glow, now written linear, pre-resolve,
  and through the one ACES. `pack_emissive_rgba`'s display-space path,
  `aces_curve`'s use in it, and the "ONE named exception" clause in CLAUDE.md
  are deleted. Speckle keeps modulating the glow's intensity.
- **Gate:** the existing fire shots (`fire_plume`, `fire_thin_smoke`,
  `fire_thick_smoke`, `explosion_t*`) look the same or better with the flame
  pass off — hue, chroma and halo extent compared to `cfcba40`, not only
  brightness (G0's P1 lesson: orange washed to beige).
- The legacy flat `FireOverlay` (off by default, superseded) is deleted.

## 4. Budget

Direction doc §7: **≤ 1 ms** fire VFX at M=8, **≤ 0.3 ms** shimmer, RTX 3070,
1080p; laptop tier = M 4 / M_u 2, half the Jacobi iterations.

Estimate (bandwidth-bound, RGBA16F): `fire_smoke_studio` active box ≈ 20×12
tiles → 80×48 velocity texels; 20 Jacobi passes are ~0.02 ms. The worst case
is a whole 128×256 map ablaze (512×1024 velocity texels) at ~1.5–2 ms. The
active-region scissor and `n_jacobi` are the dials; F1 measures it, F2 gates
on it.

## 5. Config

`[render.fire_vfx]`, Ctrl+R live: `enabled` (false = today's look, the A/B),
`emission = "thickness" | "mask"` (Erik's A/B), `m`, `m_u`, `n_jacobi`,
`k_follow`, `kick`, `vorticity`, `curl_amp`, `lifetime_s`, `cool_rate`,
`kappa`, `r0`/`r1`, `edge_*`, `margin_tiles`. A debug view key cycles
R / T_vfx / u_vfx / sources.

## 6. F4 — embers, shimmer, flicker

### 6.1 Embers — the canonical particle primitive

`renderer/particles.py`: CPU, numpy-vectorised, a few hundred particles.
They are emitted from the hottest flame texels, ride tamed wind plus curl
noise (Bridson 2007), cool along the black-body ramp and fade, and are drawn
as additive linear-HDR point sprites in the scene RT. They use a seeded RNG,
so look shots are reproducible. **This is the one particle system**:
sparks, debris and muzzle smoke (#34/#40) extend it later. The existing
`_effects` queue (tracers, rings) stays as is; migrating it is not G3.

### 6.2 Heat shimmer — the one post-FX slot

One fragment pass over the scene RT before the resolve, in the slot that
`world_composite.py` reserves. It offsets UVs by advected noise scaled by hot
gas (game T above a threshold, upsampled), so the floor and smoke behind hot
air wobble. Strength and threshold are in `[render.fire_vfx.shimmer]`. Budget
≤ 0.3 ms. Every later screen-space distortion (blast shockwave, G4) is a term
in this pass, never a second pass.

### 6.3 Flicker — RULED: A (Erik, 2026-10-23)

Room light comes from the radiation sweep as one combined field per tick
(§2), so the fire's share of the light on the floor cannot be separated from
the lamps' share.

- **A (recommended): the flame flickers; the room light stays steady.** The
  flame, embers and glow move at 60 fps; the light on the floor is the sim's
  light, updating at 24 Hz with the fire's real intensity. It is honest and
  cheap, and the motion in the flame carries most of the "flicker" feel.
  Revisit at HUMAN-TEST 2 and in #95 ("what glows should emit light"), the
  session that already goes through every light source.
- **B: a render-only pulse on the room light near fires.** A per-tile
  multiplier from the flame's local `R` scales `light_tex_a`. Because the
  light is not separable, a lamp next to a fire would pulse too. It is a
  visual hack on a physical field.
- **C: flicker in the physics.** The sweep's emission gets deterministic
  noise. Then heat transfer flickers too, at 24 Hz steps; digests and the
  golden move. This is Erik's physics, and out of scope for a render arc.

## 7. Patches and gates

| Patch | Content | Gate |
|---|---|---|
| **F1 substrate** | blackbody GPU LUT; source extraction (pure numpy); VFX RTs + ping-pong helper; solid mask; debug view of sources; perf harness; look-shot rows on `fire_smoke_studio` (`fss_overview`, `fss_close`, `fss_lone`, a windy variant) | digests/golden unchanged; source-extraction property tests; perf measured |
| **F2 loop** | §3.3 passes 1–7, debug views | stable: 10 min run with no NaN/Inf, resize, F5 reload, a fire burning out; budget ≤ 1 ms on `fire_smoke_studio`; the debug view shows tongues leaving the open faces and leaning with the vents' wind |
| **F3 look** | §3.4 draw, A/B emission toggle, §3.5 heat layer → linear HDR, exception and FireOverlay deleted, CLAUDE.md rows | old fire shots same-or-better (hue/chroma/halo); `test_one_tonemap` drops the exception; **HUMAN-TEST 1** (Erik picks A or B) |
| **F4 extras** | §6.1 embers, §6.2 shimmer, §6.3 per ruling | budget incl. shimmer; **HUMAN-TEST 2**; merge `--no-ff` |

Each patch renders the shot set before and after, and the implementer looks
at the PNGs before claiming the patch works (direction doc §7).

## 8. Tests (properties, not snapshots)

- **Render-only by construction:** running the same scenario with
  `fire_vfx.enabled` on and off leaves the field digest identical. *Breaks
  if* any VFX code writes `gmap` or sim state.
- **Sources are open faces of burning tiles only:** for random tile maps, no
  source sits on a blocking neighbour or a non-burning tile, and the source
  strength rises with `I`. *Breaks if* the injection rule leaks into walls or
  ignores intensity.
- **Never hotter than the sim:** the extracted `T_vfx` source is at most the
  game's local temperature. *Breaks if* the flame's colour detaches from the
  physics.
- **One ramp:** the GPU LUT equals `BlackbodyRamp`'s curves sample for
  sample. *Breaks if* a second temperature→colour map appears.
- **One tone-map, no exception:** `test_one_tonemap` asserts that no
  display-space world layer remains. *Breaks if* the heat layer slips back
  past the resolve.
- **Particle bounds:** embers never outlive `lifetime`, never exceed the
  cap, and cool monotonically. *Breaks if* the primitive leaks or brightens.

## 9. Systems

**(a) Existing canonical systems G3 must use:**

- GameRenderer + WorldComposite: one new draw call in `compose_world` into
  the one scene RT.
- One tone-map: G3 retires its named exception (§3.5).
- Blackbody: the only T→colour map. The GPU LUT is an extension of
  `BlackbodyRamp`, not a second ramp. Retuning still goes through
  `gen_light_table.py`.
- Tamed wind: the only wind input.
- `advected_noise` clock and `noise_bake`: the flame edge noise, embers' curl
  noise and shimmer noise. A new noise shape is a function in `noise_bake`.
- Smoke cast product and the relief: the ceiling soot occludes the flame
  through the composite; the relief's `own_light_cut` /
  `glow_below_fraction` are re-checked when the heat layer moves.
- Dequantize convention: `fire_fixed` / temperature `dequantize_f32` at
  `upload_state`.
- `core.py` RGBA16F helpers and the `GasReliefPass` sampler pattern.
- Look shots: new rows, never a second harness.
- Config via `CFG`.

**(b) New systems, with draft CLAUDE.md rules:**

- **FireVFX pass** — `renderer/fire_vfx.py` + `shaders/fire_vfx_*.fs`: "THE
  flame look: a render-only fluid sim on a finer grid, fed by `fire`,
  temperature and tamed wind; reads dequantized copies, writes no synced
  state, never runs headless; flame colour only through the blackbody LUT."
- **Ping-pong helper** — `renderer/core.py`: "the one double-buffered RT
  helper for iterative passes; never hand-roll a second FBO pair."
- **Blackbody GPU LUT** — `BlackbodyRamp.lut_texture()`: "the GPU face of
  the one ramp; shaders sample it, never re-derive Planck."
- **Particle primitive** — `renderer/particles.py`: "the one particle system
  (embers, sparks, debris); consumers extend it, never build a second."
- **Scene post-FX pass** — `world_composite` post-FX slot: "every
  screen-space distortion is a term here (shimmer now, shockwave in G4),
  never a second pass."

## 10. Credits (file headers + `docs/papers/`, none archived yet)

Stam, *Stable Fluids* (SIGGRAPH 1999) and *Real-Time Fluid Dynamics for
Games* (GDC 2003); Fedkiw, Stam & Jensen, *Visual Simulation of Smoke*
(SIGGRAPH 2001) for vorticity confinement; Harris, *Fast Fluid Dynamics
Simulation on the GPU* (GPU Gems ch. 38); Selle et al., *An Unconditionally
Stable MacCormack Method* (2008) if used; Bridson, Hourihan & Nordenstam,
*Curl-Noise for Procedural Fluid Flow* (SIGGRAPH 2007); Sørb, the Ignitement
fire-VFX breakdown (Unity blog, 2026).

## 11. Out of scope

- Flames on burning gas (`fuel_gas`) and the flamethrower jet. The source
  law extends to them later as another source row.
- Smoke from the VFX: the plan's decision 2 is kept, so smoke stays G1/G2's.
- Room-light flicker beyond the ruling in §6.3.
- The tilted camera (G6).
