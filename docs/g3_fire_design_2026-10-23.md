# G3 Fire — design (F0), 2026-10-23

*Graphics arc G3 of `docs/graphics_direction_2026-10-18.md` (§3, §8), issue
#52 under tracker #93. Builds on `docs/fire_vfx_plan_2026-08-24.md` (the
approach) and revisits its locked decisions in the light of G0–G2 and the
#12 fire + smoke session (2026-10-22). Base: `main` at `cfcba40`.*

*Status: **v2**. Claude is the approver as lead designer (Erik delegated the
approval, 2026-10-23).*

*Critique rounds:*

| Round | Doc | Lens | Verdict | File |
|---|---|---|---|---|
| 1 | v1 | all (facts, physics, composite, GPU risk, systems, tests, flicker, look) | REVISE: 3 blockers, 12 major, 13 minor; all folded in or answered in §12 | `docs/g3_fire_design_critique_2026-10-23.md` |
| 2 | v2 | resolution check: are round 1's blockers and majors actually resolved, and did the fixes create new ones? | APPROVE-WITH-FIXES: 1 new blocker, 7 new majors, 8 minors, all doc-level, none needing Erik. Folded in as §13; no third round needed | `docs/g3_fire_design_critique2_2026-10-23.md` |

***v3 = v2 + §13.** §13 is binding and OVERRIDES the earlier sections
wherever they differ. Implementers read §13 before §3–§9.*

***APPROVED, v3, by Claude as lead designer (2026-10-23).** F0b may start.*

---

## 0. What G3 is for

Today a fire is a hot area:
- the tile glows in black-body colours (the heat layer);
- it lights the room (the radiation sweep);
- it smokes (G1/G2).

That is physically right, but it has **no shape**: no tongues, no lean, and no
motion between two sim ticks. G3 adds flame structure, render-only. Then it
adds the three cheap things that sell fire: embers, heat shimmer and flicker.

## 1. Erik's decisions (2026-10-23, design session)

1. **Camera: straight down.** A later tilt of **at most 15–20°** is kept
   open for G6. The flame's shape comes from a field that can later be given
   height, never from a top-down-only trick.
2. **No "lit relief" for flames.** A flame emits its own light, so it has no
   lit side and no shadowed side (Erik's objection, accepted). Structure
   comes from emission only (§3.4).
3. **Erik judges an A/B:** *glow by temperature only* (A) against *glow by
   temperature and thickness* (B), with one config toggle.
4. **Scope:**
   - F1–F3 flames → HUMAN-TEST 1;
   - F4 embers, shimmer and flicker → HUMAN-TEST 2;
   - then merge. One branch, `93-g3-fire`.
5. **No reference board.** Erik judges the look shots of the real fire in
   `fire_smoke_studio`.
6. **Flicker = D** (§6.3), Erik 2026-10-23, superseding his A from earlier
   the same day, which was ruled on a wrong menu: the fire's own room light
   flickers through a light-only multiplier, with no heat and no digest.
   Erik: "happy to try it". His two worries go into the F4 gate:
   - fires must **not flicker in sync**;
   - a single tile flickers the same in every direction (isotropic), which
     may look flat. Larger fires vary naturally.

**Standing look rulings G3 honours:**
- `glow_through_ceiling = 0.2` (Erik, 2026-10-22: "the fire's own smoke
  closed over it and it went dark");
- hot gas glow is not hidden by its own depth (G2; #95's);
- soot is a ceiling layer above the units (ruling 2, 2026-10-18).

**Related, but not G3:** #100 (smoke swallows a fire's room light), Erik's
direction for which is stratified smoke in the sweep, in a later physics
session.

## 2. What the code does today (facts, `cfcba40`)

- **Heat layer** (`overlays.py::HeatFieldOverlay`,
  `blackbody.py::pack_emissive_rgba`):
  - reads `gmap.temperature` only, one texel per tile (RGBA8, bilinear);
  - CPU-packed every frame through `BlackbodyRamp.chroma_intensity` →
    `aces_curve`;
  - drawn after the resolve, in display space, additive, **above the 3D
    units**, and through the smoke via `glow_through_fs` (G2). Only the BODY
    share (`glow_below_fraction`) is dimmed;
  - the relief's `own_light_cut` reads the same display-space brightness;
  - it is G0's named temporary exception, "until the G3 fire arc replaces
    it";
  - speckle (`renderer/speckle.py`) modulates it at pack time.
- **Fire state:**
  - `GameMap.fire` is a Q16.16 logistic intensity `I` on `flammable` tiles;
  - **only wood is `solid`**; furniture, kindling and foliage are permeable
    (`gamemap.py:1451`);
  - the sim lands heat and soot **on the burning tile itself**
    (`physics_runner.py:1235`);
  - look-shot fires force-flammable a 3×3 patch of air
    (`lighting_demo.py:900`);
  - readers today: the legacy `FireOverlay` (off by default) and
    `hover_readout`.
- **Temperatures (probe, `fire_smoke_studio`, 5–20 s):**
  - burning tile: 1331 → 1061 K;
  - hottest gas next to it: 457–761 K, below the 800 K glow floor (the gas
    cell averages a thin flame over a 2.5 m column; #68).
- **Room light:**
  - the sweep's light channels form one combined field per tick;
  - they move **no heat** (`radiation_sweep.h:79-85`), and `light_q` is not
    digested (`light_field.py:31-33`);
  - render-state emitters (cones, cursor lamp, muzzle flash) already feed
    them;
  - uploaded as `light_tex_a/b` per tick, plus 4 Hz relights while paused.
- **Clocks:**
  - sim 24 tps, render 60 fps, no interpolation;
  - `advected_noise.advection_phase` is tick-locked;
  - look shots run one tick per frame (`look_shots.toml:14`).
- **Wind:** `gas_detail.tame_wind`, computed once per frame in
  `upload_state` for the relief.
- **GPU:**
  - RGBA16F texture and RT helpers in `core.py`;
  - the `GasReliefPass` pattern (manual sampler units from 5, `tile_rect`
    clip, 8 px/tile product);
  - **no ping-pong pass exists**, and no particle or post-FX system;
  - `world_composite.resolve` is the one place ACES runs.
- **Sizes and hardware:** studios 48×32 to 56×56, largest level 128×256;
  world RT 48 px/tile; RTX 3070.

## 3. The flame model

### 3.1 Layer split

- **Layer A, the game, is authoritative and untouched:** `fire`,
  `temperature`, tamed wind, solids.
- **Layer B, the flame VFX, is render-only:**
  - it reads dequantized **read-only** copies;
  - it writes nothing synced;
  - it never runs headless.

### 3.2 Fields (Layer B)

| Field | Grid | Meaning |
|---|---|---|
| `u_vfx` (vec2) | M_u = 8 px/tile (laptop 4) | flame-scale velocity, tiles/s |
| `p` | M_u, fp32 if the RT path allows, else fp16 with warm start | pressure |
| `R` | M = 8 px/tile | flame density, "how much flame stands in this column" |
| `RT = R·T_vfx` | M | premultiplied flame temperature (minor 4) |
| `src` (per tick) | M | source mask: rate `s`, `T_src`, expansion `q` |
| `solid` (per tick) | M | `solid` tiles (wood, hull, doors): no-through, `R = 0` |

`T_vfx = RT / R` where `R > ε`. The temperature has to live in Layer B
because the game holds one value per tile, and the flame needs to cool along
its tongue.

**M_u = 8, not 4** (M12): at 4, a lone burning tile is 4×4 velocity cells,
which gives an orange blob. At studio sizes, full resolution is cheap (§4).

### 3.3 The source law (B1, B3)

Per tick, a pure numpy function `flame_sources(fire, temperature, solid,
flammable, vacuum)` produces the source texture. It is tested (§8).

- **Where.** Each source sits in its own tile, or beside it, depending on
  the fuel:
  - **permeable fuel** (furniture, kindling, foliage, air fires): over the
    burning tile's own area, where the sim lands its heat and soot;
  - **solid fuel** (wood): in a strip in the open neighbour cell along each
    face that touches non-blocking air;
  - **never** in a vacuum (`game_renderer.py:568` gate) or inside a solid.
- **How much:** rate `s ∝ I`, so a dying fire shrinks for free.
- **How hot:** `T_src` is **the burning tile's own `temperature`**. That is
  the value the sweep emits `a·L°[T]` from and the heat layer glows with, so
  the flame's colour and the room light agree.
  - **Invariant:** `T_vfx ≤ T` of the burning tile that fed it.
  - **Look consequence:** burning tiles run at roughly 900–1330 K, so the
    flames will be **deep orange to orange-red**, with no yellow-white
    roots. Yellow-white needs hotter tiles, which is physics (#68), not a
    render dial. Erik is told before HUMAN-TEST 1.
- **Motion source (B3):** the burning gas **expands**. The source adds a
  prescribed divergence `q ∝ I·(T_src/T_amb − 1)` to the pressure solve,
  `∇²p = ∇·u* − q`, so the outflow survives the projection.
  - An outward velocity kick alone would not survive: projection removes a
    radial kick almost entirely.
  - The active region's edge is an **open boundary** (`p = 0`), which gives
    the expanding gas a place to go.
  - Sign: expansion (outflow) at flame height. The inward entrainment at the
    floor is not modelled; it is invisible from above.
- **Structure:** vorticity confinement (Fedkiw 2001) on the expansion's
  shear against the tamed wind and solids, plus a small curl-noise forcing
  (Bridson 2007) that breaks the symmetry between ticks.

**F0b, a numpy prototype before any GPU code:** a 64² grid with one burning
furniture tile and one wood-wall fire, with and without wind. It must show
tongues leaving the fuel and leaning downwind, using the real source law and
the loop below. It is also the **spec twin**: the GL loop is checked against
it (§8).

### 3.4 The loop (fixed substeps on the sim clock — M1)

```
clock: t_vfx = (tick + α) · (1/tps); α = sub-tick fraction from main's accumulator
       substep dt_s = 1/120 s; run n = steps due since last frame (cap 4)
per tick:      upload src, solid, tamed wind (the relief's per-frame array, reused)
per substep:
  1. advect u_vfx (SL); relax toward tamed wind (k_follow)
  2. vorticity confinement (ε) + curl noise (curl_amp)
  3. divergence − q → Jacobi (n_jacobi, warm-started from last p) → subtract ∇p;
     solid = no-through, active-region edge = open
  4. advect R and R·T (MacCormack with Selle's min/max limiter — minor 3;
     SL fallback); inject s, s·T_src
  5. decay R (lifetime_s); cool T_vfx toward the local game gas T (cool_rate);
     R = 0 in solids
```

- **Paused game:** the sim clock stops, so the flame would freeze.
  - **Lead decision:** a render-only `idle_when_paused = true` keeps the
    substeps running on the frozen sources, so a fire stays alive during
    planning phases. It never touches the sim.
  - Erik judges it at HUMAN-TEST 1 and can flip it.
- **Look shots** advance exactly one tick per frame, with α = 0 and a fixed
  seed. The flame therefore runs at the same speed in shots and in the game,
  and the shots are reproducible.
- **Smoke:** the relief keeps its tick-locked clock. A sub-tick α for the
  relief is noted for later (minor 12) and is not G3's.

### 3.5 How it is drawn (emission only)

```
L_flame = Le(T_vfx) · E(R)                         linear HDR
  A  "temperature only":        E = smoothstep(r0, r1, R)
  B  "temperature + thickness":  E = 1 − exp(−κ·R)
```

- **`Le(T)`** comes from the one ramp, as a GPU LUT:
  - 1024 entries, **linearly filtered**, so there is no banding;
  - indexed on the ramp's own Kelvin domain;
  - the data is `BlackbodyRamp`'s float arrays;
  - the upload lives in `renderer/fire_vfx.py`, so **`blackbody.py` stays
    pyray-free** (M5).
- **Ragged edges:** a **flame noise** shape, i.e. stretched streaks along
  `u_vfx`, a new function in `noise_bake`. It is not the cauliflower billow
  (M12). It erodes `R` at the edge, on the one noise clock.
- **No double counting (M4):** where the flame stands, it **covers** the
  burning tile's body glow instead of adding to it. The flame is composited
  as an emitter with opacity `E`: `out = mix(under, Le, E)`, written
  premultiplied as `Le·E + under·(1−E)` for the glow pass. The tile is never
  brighter than its own temperature allows.
- **A vs B are compared on shots where they actually differ** (minor 9). The
  steep `((K−800)/500)²` intensity curve may make temperature, not R, set the
  visible edge in both. F2's prototype measures this first; if A and B come
  out the same, κ and r0/r1 are retuned until the A/B is meaningful.
- **Tilt-readiness:** `R` and `R·T` are a 2D column field. A 15–20° tilt can
  extrude `R` into a short height, the way the relief turns τ into height.

### 3.6 Where it sits, and the heat layer's move (B2, M2, M3)

**Draw order inside `compose_world`** (scene RT, linear HDR):

```
floor → BODY glow (heat layer, body share) → floor gas layer → flame →
units/props → ceiling soot (relief) → GLOW-THROUGH compensation →
GAS glow (heat layer, gas share, undimmed) → [resolve: ACES + shimmer]
```

- **Body glow before the floor gas** keeps `glow_through_floor = 0.3`
  (minor 2).
- **Flame and body glow below the ceiling soot.** Physical occlusion through
  the relief's composite would equal `glow_through_ceiling = 1.0`, which
  Erik rejected. So a **compensation term** right after the ceiling pass adds
  back `glow·α_ceil·(1 − glow_through_ceiling)·below`, read from the cast
  product through the lit3d `smokeCoverAt` seam. This is G2's glow-through
  shader moved before the resolve. Net occlusion = Erik's 0.2.
- **Gas glow above the soot, undimmed:** a hot slab is not hidden by its own
  depth (G2 rule).
- **The units now stand above the fire glow.** Today the heat layer is drawn
  over them. This is a look change, and it is named for HUMAN-TEST 1.
- **The relief's own-light cut** keeps a dedicated brightness input,
  computed from the same ramp and normalised as today (a 0–1 brightness
  share, not linear radiance). It is fed from a function in `blackbody.py`
  instead of the display pack, so it does not saturate (M3).

**The hue gate, before anything is deleted (G0's P1 lesson: orange washed to
beige):**
- the heat layer's linear path renders the existing fire shots (`fire_plume`,
  `fire_thin_smoke`, `fire_thick_smoke`, `explosion_t*`, the new `fss_*`)
  with the flame pass **off**;
- a script compares them to `cfcba40` in a colour metric: mean hue,
  chroma and halo extent (the radius at which the glow falls to 10 %) in
  CIELAB on the masked fire region;
- **gate:** the hue shift is ≤ 5° and the chroma is ≥ the old value.

The emitter-cover composite (§3.5) is what keeps chroma: a covering emitter
shows its own colour, while an additive emitter on a grey floor drifts toward
white.

**If the gate fails:**
- the heat layer stays the display-space exception for the rest of G3, with
  the reason written down;
- the flame is still drawn linear;
- the exception's retirement becomes its own patch.

**What `enabled = false` means after F3:** the flame pass is off, and the
heat layer (now linear) still draws. "Today's look" is the shot set at
`cfcba40`, kept as PNGs in the arc's shot folder. A toggle back to the
display-space path is **not** kept.

The legacy `FireOverlay` is deleted. `hover_readout` keeps reading `fire`.

## 4. Budget

The direction doc's targets, on the RTX 3070 at 1080p:
- **≤ 1 ms** of GPU time for the fire VFX;
- **≤ 0.3 ms** for shimmer;
- **CPU time is measured separately** (M6).

**Per substep:** about 8 + n_jacobi fragment passes (n_jacobi ≈ 15,
warm-started), each costing:
- one `begin/end_texture_mode`;
- the uniform sets;
- a quad.

The Python and rlgl overhead per pass (~20–40 µs) dominates small grids. At 2
substeps per frame that is about 46 passes, roughly 1–2 ms of CPU.

**Dials** (F1 measures them on `fire_smoke_studio` and the 128×256 level):
- run substeps only for the **view ∪ active region**;
- n_jacobi;
- one substep per frame on the laptop tier;
- batching more passes into a single shader where the math allows (for
  example divergence and inject, decay and cool).

**GPU estimate:** a studio's active box of about 20×12 tiles is 160×96 cells
at M_u = 8. At that size each pass is a few µs, so the GPU stays far under
budget. The CPU pass overhead is the real cost, so it is the F1 gate.

## 5. Active region (M7)

- The active region is a **per-tile mask**: burning tiles, dilated by
  `margin_tiles` (a flame's lifetime of travel). It is cut into at most 4
  bounding boxes, so two distant fires do not make one map-wide box.
- Substeps run only on those boxes.
- A tile that **leaves** the mask is cleared once (`R = 0`, `u = 0`,
  `p = 0`), so no stale flame reappears when it rejoins.
- No fire → no passes → zero cost.

## 6. F4 — embers, shimmer, flicker

### 6.1 Embers: the canonical particle primitive

`renderer/particles.py`, CPU, numpy-vectorised.

- **Spawning:**
  - spawned from the **CPU list of burning tiles** (the per-tick `src`
    array), never by reading back GPU texels (M11);
  - the rate is ∝ `I`;
  - birth temperature `T_src`.
- **Motion and life:**
  - they ride the tamed wind plus curl noise;
  - they cool on the ramp, fade, and are capped;
  - they use a seeded RNG.
- **Drawing:** additive linear-HDR point sprites in the scene RT, drawn after
  the flame and before the units.

**Rule:** "the one particle system (embers, sparks, debris); consumers extend
it." The existing `_effects` queue (tracers, rings) is **legacy display-space
effects**. It is named in the rule and left alone; migrating it is not G3
(minor 7).

### 6.2 Heat shimmer: inside the resolve (M10)

The shimmer is a UV offset inside `world_resolve.fs`, applied when the
resolve samples the scene RT, so there is no read-while-write.

- **Offset:** advected noise × hot-gas strength (game gas T above
  `shimmer_t0`, upsampled per tile).
- **Fades with the camera's zoom.**

**Rule:** "every screen-space distortion is a term in the resolve's distortion
block (shimmer now, the G4 shockwave next), never a second pass."

### 6.3 Flicker — RULED D (Erik, 2026-10-23)

The critique corrected v1's menu:
- the sweep's **light channels move no heat**;
- **`light_q` is not digested**;
- render-state emitters already enter the sweep (cones, cursor lamp, muzzle
  flash, `[render.muzzle_flash]`: "render state, never sim state, no
  digest").

The corrected options:
- **A (ruled):** the flame, embers and glow flicker; the room light stays the
  sim's steady light.
- **B:** a render-side pulse on `light_tex_a` near fires. It also pulses a
  lamp next to a fire.
- **D (new):**
  - a per-tick, per-burning-tile multiplier from the VFX's local `R` scales
    the **light-channel emission only** of burning cells, entering the sweep
    as render state, like the muzzle flash;
  - the room light flickers at the fire only, and lamps are untouched;
  - there is no heat, no digest and no golden change;
  - it costs C++ work in the sweep's input;
  - it steps at 24 Hz, which is enough for 1–10 Hz flicker.

v1's option C ("heat flickers, the golden moves") was only the expensive way
to do it.

**RULED: D (Erik, 2026-10-23).** It touches the sweep's code but not the
simulation's state.

**Design notes for D (Erik's two worries):**
- **No sync.** Each burning tile's multiplier comes from its **own** local
  flame (the VFX's R and T around that tile), not from a shared clock.
  Neighbouring tiles' flames are separate turbulent patches, so they
  decorrelate by construction.
  - **Gate:** the correlation between two burning tiles' multiplier time
    series is low (|ρ| < 0.3) for tiles three or more apart. A property test
    on the F0b twin.
- **Direction.** v1 of D is **isotropic per tile**: the multiplier scales
  the tile's emission equally in every sweep ordinate. A lone tile then
  pulses in place, while a row of tiles flickers unevenly across the room.
- **Possible follow-up, D2 (directional):** weight the per-ordinate emission
  by the flame's lean, i.e. the offset of R's centroid within the tile. A
  tongue leaning east would then light east a little more, and nearby
  shadows would sway.
  - Only if HUMAN-TEST 2 finds lone fires flat.
  - The sweep already resolves ordinates, so it is an extension, not a new
    system.
- **Clamp:** keep the multiplier mean-preserving over a second (mean ≈ 1),
  so the fire's average room light still equals the physics' light. Erik's
  1300 K exposure anchor and the brightness he tuned on 2026-10-22 stay
  true.

## 7. Patches and gates

| Patch | Content | Gate |
|---|---|---|
| **F0b prototype** | numpy spec twin: source law + loop (§3.3–3.4), PNG strips, a pytest on its properties | tongues leave permeable and wood fuel and lean downwind (Claude looks at the PNGs); `R ≥ 0`, `T_vfx ≤ T_src`, `R = 0` in solids, no fire → no flame |
| **F1 substrate** | `flame_sources` (from F0b); ping-pong helper (§9 traps); blackbody GPU LUT; active-region masks; sim-clock substeps; debug view of sources; perf harness (CPU and GPU split); `fss_*` look-shot rows | digests/golden unchanged; source-law tests; CPU+GPU ms measured on `fire_smoke_studio` and on the largest level |
| **F2 loop** | §3.4 in GLSL, debug views R / T / u / p | GL readback within tolerance of the F0b twin on its scenes; 10 min run with no NaN/Inf; resize; config reload; a fire burning out leaves no stale flame; budget |
| **F3 look** | §3.5 draw + A/B; §3.6 order, compensation, heat layer to linear behind the hue gate; `FireOverlay` deleted; CLAUDE.md rows | hue gate (or the documented fallback); `test_one_tonemap` updated; shots look the same or better; **HUMAN-TEST 1**: A vs B, idle-when-paused, units above the glow, deep-orange flames |
| **F4 extras** | embers, shimmer, flicker per Erik's re-ruling | budget incl. shimmer; **HUMAN-TEST 2**; merge `--no-ff` into `main` |

Every patch renders the shot set before and after, and the implementer looks
at the PNGs before claiming it works (direction doc §7).

**Execution:** every patch runs as one worktree subagent on `93-g3-fire`, one
at a time, committing as it goes. **Tier: Opus, high effort, for all five.**
- F0b and F3 are look-judged and have no oracle.
- F1 and F2 are GL plumbing whose traps (§9) the gates catch only partly.
- F4 is look-judged.

**HUMAN-TEST:** F3 (HUMAN-TEST 1) and F4 (HUMAN-TEST 2). F0b–F2 run hands-off
and merge into the arc branch on green. Only F4's close merges the arc into
`main`.

## 8. Tests (properties, not snapshots; minor 8–9)

Each test names the property it protects and the change that must break it.

- **Render-only, structurally:**
  - `fire_vfx` receives `writeable=False` array views;
  - a lint asserts it never assigns into `gmap`/sim objects and never
    imports a sim writer;
  - a GL-subprocess harness runs one scenario with the pass on and off and
    compares field digests.
  - *Breaks if* any VFX code writes sim state.
- **The source law** (randomised tile maps):
  - sources are only on or beside burning tiles, by fuel class;
  - none in a solid or a vacuum;
  - the rate is monotone in `I`;
  - `T_src` equals the burning tile's `T`.
  - *Breaks if* injection leaks, ignores intensity, or detaches its colour
    from the tile.
- **The spec twin's invariants** (numpy loop):
  - `R ≥ 0`;
  - `max T_vfx ≤ max T_src`, which needs the limiter;
  - `R = 0` in solids after every substep;
  - no fire → `R ≡ 0` after `lifetime`.
  - *Breaks if* advection makes new extrema or flame passes into walls.
- **GL parity:** the GL loop matches the twin within tolerance on 3 scenes.
  *Breaks if* the shaders drift from the spec.
- **One ramp:** the GPU LUT's data equals `BlackbodyRamp`'s curves at their
  sample points. *Breaks if* a second T→colour map appears.
- **One tone-map:**
  - `test_one_tonemap` covers the draw-order contract;
  - a recording pyray stub asserts every world layer draws before the
    resolve.
  - *Breaks if* a layer slips past the resolve.
- **Zero cost with no fire:** no VFX passes are issued without fire. *Breaks
  if* the active-region gate is lost.
- **Particles:** never past `lifetime` or the cap; cooling is monotone.
  *Breaks if* the primitive leaks or brightens.
- **A ≠ B:** on the `fss_close` shot the A and B renders differ by more than
  a threshold. *Breaks if* the A/B becomes meaningless.

## 9. Ping-pong traps (M8)

These go into the helper's docstring and are covered by its test.

- **Blending:** set it explicitly for every pass (`BLEND_ADD_COLORS` with
  ONE/ZERO, or blending off). raylib's default alpha blend corrupts
  simulation data.
- **Y flip:** each pass through an RT flips Y. Sample in texel space with
  one documented convention.
- **Depth:** no depth attachment is needed. Create the FBO without one.
- **Pressure precision:** fp16 pressure loses convergence. Use fp32 if
  `rlLoadTexture` supports R32F/RGBA32F on the target; otherwise use fp16
  with a warm start and a convergence check in the debug view.
- **Sampler units:** start from `_FIRST_UNIT` as `GasReliefPass` does,
  because raylib's batch binds at most 4.
- **Config reload:** changing `m`/`m_u` reallocates the RTs. Every other dial
  is live on the existing reload key (minor 10).

## 10. Systems

**(a) Existing canonical systems G3 must use:**

- **GameRenderer + WorldComposite:** the flame and glow draws go in
  `compose_world`, into the one scene RT.
- **One tone-map:** G3 retires the named exception, behind the hue gate. The
  CLAUDE.md row text is edited: "never converted to linear" is removed and
  the exception clause deleted.
- **Smoke cast product:** the compensation term reads it through lit3d
  `smokeCoverAt`. The row's "hot gas is #95's" stays true.
- **Blackbody:** the only T→colour map. The GPU LUT carries its data;
  retuning still goes through `gen_light_table.py`.
- **Temperature scale:** for T→Kelvin.
- **Tamed wind:** reuse the relief's per-frame array; never call it again
  with other dials.
- **`advected_noise` clock and `noise_bake`:** flame noise is a new
  function there.
- **Dequantize convention.**
- **`core.py` RGBA16F helpers, and the `GasReliefPass` sampler pattern.**
- **Look shots:** new `fss_*` rows, never a second harness.
- **Config:** via `CFG`, under `[render.fire_vfx]`.
- **`frame_lights.py` and the sweep's render-state inputs:** only if Erik
  picks D.

**(b) New systems, with draft CLAUDE.md rules:**

- **FireVFX pass** — `renderer/fire_vfx.py` + `shaders/fire_vfx_*.fs`:
  "THE flame look. A render-only fluid sim fed by `fire`, the burning
  tile's temperature and tamed wind. Its numpy spec twin is
  `renderer/fire_vfx_ref.py`. It reads read-only dequantized copies, writes
  no synced state, never runs headless, and colours the flame only through
  the blackbody LUT."
- **Ping-pong helper** — `renderer/core.py`: "the one double-buffered RT
  helper for iterative passes, with its traps documented; never hand-roll a
  second FBO pair."
- **Blackbody GPU LUT** — in `fire_vfx.py`, from `BlackbodyRamp` arrays:
  "the GPU face of the one ramp; shaders sample it, never re-derive Planck."
- **Particle primitive** — `renderer/particles.py`: "the one particle system;
  the `_effects` queue is legacy display-space effects, not a second
  system."
- **Resolve distortion block** — `world_resolve.fs`: "every screen-space
  distortion is a term here, never a second pass."

## 11. Credits (file headers + `docs/papers/`, to archive in F0b)

- Stam, *Stable Fluids* (SIGGRAPH 1999); *Real-Time Fluid Dynamics for
  Games* (GDC 2003).
- Fedkiw, Stam & Jensen, *Visual Simulation of Smoke* (SIGGRAPH 2001):
  vorticity confinement.
- Nguyen, Fedkiw & Jensen, *Physically Based Modeling and Animation of Fire*
  (SIGGRAPH 2002): expansion at the reaction front as a divergence source.
- Harris, *Fast Fluid Dynamics Simulation on the GPU* (GPU Gems, ch. 38).
- Selle, Fedkiw, Kim, Liu & Rossignac, *An Unconditionally Stable MacCormack
  Method* (J. Sci. Comput. 2008).
- Bridson, Hourihan & Nordenstam, *Curl-Noise for Procedural Fluid Flow*
  (SIGGRAPH 2007).
- Sørb, the Ignitement fire-VFX breakdown (Unity blog, 2026).

## 12. Out of scope, and critique items answered rather than adopted

**Out of scope:**
- **Burning gas and the flamethrower jet:** a later source row.
- **Smoke from the VFX:** smoke stays G1/G2's.
- **The tilted camera:** G6.
- **#100:** stratified smoke for room light.
- **A sub-tick clock for the smoke relief:** minor 12, noted for later.
- **Foliage props drawn over their own fire:** minor 11, noted on #60.

**Critique items answered rather than adopted:**
- **B2's alternative, physical occlusion at 1.0:** not offered. Erik ruled
  on 2026-10-22; a flame with shape does not change why.
- **M1's "Erik's call" on pause:** the lead set the default
  (`idle_when_paused = true`, render-only) because the planning phases are
  paused most of the time, and a frozen flame reads as a still picture. It is
  on the HUMAN-TEST 1 checklist.

## 13. Round-2 amendments (binding; they override §3–§9)

Each item names the round-2 finding it answers.

**13.1 The pressure solve is screened (N-B1).** Solve
`(∇² − 1/ℓ²) p = ∇·u* − q`, with `ℓ = screen_len_tiles` (a dial, ~3 tiles).
The screening term is the plume leaving the plane upward (§3.3).
- A sealed room inside a box, or a box with closed doors, then has a unique
  bounded solution. With the old pure-Neumann problem and Σq > 0, warm-started
  Jacobi grows p until it overflows.
- The active-region edge keeps `p = 0`.
- **F0b gates:**
  - a sealed-room scene (a fire in a closed 6×6 room inside the box);
  - `max|p|` stays bounded over 10⁴ substeps.

**13.2 Velocity scales and structure sources (N-M5).**
- Wind enters as `u_wind = tamed · tps`. `tame_wind` returns tiles **per
  tick**; `u_vfx` is in tiles/s.
- `q = q₀ · I · (T_src/T_amb − 1)`, with `q₀` in 1/s.
  - **`q₀` is a look dial**, about 100× the physical in-plane expansion.
  - Physical in-plane expansion is ~0.2 tiles/s at the tile edge, which is
    too weak for tongues. The dial is honest about that.
  - Tongues 1–2 tiles long within ~0.4 s need `q` ≈ 20–30 /s.
- The projection removes the wind's divergent part every substep, so wind and
  expansion do not fight on divergence; they compete only in magnitude.
- **The structure sources, named honestly.** Expansion and pressure-driven
  wind are both irrotational, so vorticity confinement only amplifies what
  these two seed:
  - **curl noise** scaled by local `I` (Bridson 2007);
  - **pool-fire puffing**: `s` and `q` pulse at `f ≈ 1.5/√D` Hz, with D the
    fire's extent in m (Cetegen & Ahmed, *Combust. Flame* 1993), each fire
    cluster at its own phase.
- F0b's "with wind" scene is driven by the **real** tamed-wind field, dumped
  from `fire_smoke_studio` with the vents running, not by a constant.

**13.3 The clock (N-M1, minor 8).**
- `upload_state(..., alpha=)` is a new keyword.
  - Both glues pass it: `main.py`'s accumulator and
    `tools/lighting_demo.py` (the `--auto` shots, its own accumulator, and
    `--perf`).
  - The render side clamps it to [0, 1).
  - `t_vfx` never steps backward.
- The substep is `dt_s = 1/120 s`. The cap is **6 per frame** (≥
  `ceil(1/(tps·dt_s)) + 1`), so a shot frame (one tick) runs exactly 5,
  the same as the game.
  - **Test:** substeps per tick are equal in shot mode and at 60 fps. *Breaks
    if* the cap drops below 5.
- `idle_when_paused` runs on its own wall-dt counter `t_idle`, never written
  into `t_vfx`.
  - Curl-noise and edge-noise phases key on `t_vfx + t_idle`, not on the
    tick-locked `advection_phase`, so the noise never freezes on a moving
    flame.
  - Shots never pause (`lighting_demo` un-pauses every frame), so they stay
    reproducible.
  - The game's VFX state after a pause depends on wall time. It is
    **ACCEPTED GAP: not replay-reproducible**, which is fine for a render-only
    layer.

**13.4 Flame opacity is tied to emission (N-M7, the M4 bound).**
- The cover weight is `ε = E(R) · w(T_vfx)`, with `w` a smoothstep from 0 at
  `kelvin_glow_min` to 1 at `kelvin_glow_min + 100 K`. A flame cooler than
  the glow floor covers nothing, so there are no dark tips.
- No source is injected where `T_src < kelvin_glow_min`.
  - The ~300 K air fires of the look-shot injector (probe: 283–306 K while
    `fire` → 0.10) therefore make no flame.
  - That is honest: the sim says the patch is not hot.
  - Those shots keep the heat layer's look.
- **Tests:**
  - *the flame pass never lowers masked luminance below the flame-off
    render by more than a tolerance* (breaks if ε decouples from emission);
  - *flame radiance at a pixel never exceeds `Le(T_src)` of its feeding
    tile* (round 1's M4 bound).

**13.5 Cooling never heats (minor 1).** `T_vfx ← min(T_vfx, relax(T_vfx,
T_gas_local))`.

**13.6 Vacuum (minor 2).** `R = 0` in vacuum tiles, as in solids, after
every substep. Every VFX draw keeps the destination alpha.

**13.7 Draw order and occlusion, exact (N-M2, N-M3).** The §3.6 order is
replaced by:

```
floor → water → floor gas layer → BODY glow · (1 − glow_through_floor·cov.b) →
flame (ε-cover) → units/props → ceiling soot (relief) →
COMPENSATION (a second evaluation of body glow + flame, additive,
              weight α_ceil·(1 − glow_through_ceiling) from smokeCoverAt;
              below_flame = 1, body uses glow_below_fraction) →
GAS glow (undimmed) → [resolve: ACES + shimmer]
```

- **The body glow draws after the floor gas** with G2's own floor factor.
  That is exact: the floor relief's composite would otherwise hide it fully.
- **The compensation is approximate.** The cast product is 8 px/tile, the
  drawn soot 48 px/tile. **Gate:** on `fire_thin_smoke` and
  `fire_thick_smoke`, the masked fire-region luminance with the flame off is
  within ±4 % of `cfcba40` (G0's tolerance).
- **If it fails:** the term moves inside the ceiling relief shader, which
  knows its own α, with the glow passed as a texture.

**13.8 The hue gate tests what ships (N-M4).**
- The gated configuration is the **flame on**, covering the body tiles.
- The body share also uses the over operator, with ε from its own emission,
  so it keeps chroma the way the flame does.
- **Metric:**
  - CIELAB on the shot PNGs;
  - hue = atan2 of the **chroma-weighted** mean (a\*, b\*);
  - mask = the burning tiles, dilated by 1 and projected through the shot's
    camera, the same for both runs;
  - halo = the radius at which luminance falls to 10 %, against a
    `hide = ["heat"]` twin row;
  - pass if the hue shift is ≤ 5° and chroma is within G0's ±4 %.
- **Fallback keeps M4.** If the gate fails, the heat layer stays display-space
  after the resolve, but its shader samples the flame's ε texture and draws
  `glow · (1 − ε)`, so the flame still covers the tile and nothing is counted
  twice. The draw order and units-above-glow changes are then deferred with
  it, and the reason is written down.

**13.9 GL parity is per operator (N-M6).**
- F2 compares GL with the twin **one stage at a time**: upload the same
  state, run one stage (advect, confine, divergence, Jacobi sweep, project,
  inject, decay/cool), read back, compare within fp16/fp32 tolerance.
- Over long runs, only invariants and statistics are checked: `R ≥ 0`,
  `T ≤ T_src`, total R within 10 % of the twin's.
- Trajectory parity is dropped; the loop is chaotic by design.

**13.10 Budget and memory (M6 gaps).**
- **Substeps run on the active boxes only.** Off-screen fires keep
  simulating, which is cheap. **Draws are clipped to active ∩ view.** v2's
  "view ∪ active" was wrong.
- **All boxes share one pass per stage:** up to 4 quads inside one
  `begin/end_texture_mode`. The pass count is independent of the box count.
- **VRAM:** whole-map RTs at M = 8 on the largest level (128×256) are about
  300 MB. That is acceptable on the 3070's 8 GB, and about 75 MB on the
  laptop tier at M = 4. F1 measures it.

**13.11 Ping-pong blend (M8 correction).** Every simulation pass draws with
**blending disabled**, or a custom ONE/ZERO blend, so the shader's output
replaces the texel. raylib's `BLEND_ADD_COLORS` is ONE/ONE, which v2
mis-stated. Composite passes choose their blend explicitly.

**13.12 Shimmer sampling (M10 note).** The scene RT is point-filtered, so
the resolve's shimmer offset does its own bilinear tap.

**13.13 One tamed wind (minor 3).** `upload_state` computes **one** tamed-wind
product per frame, with the relief's dials, whenever any consumer needs it:
the relief, props or the flame. Every consumer takes that array; no second
`tame_wind` call with other dials. Props' current default-dial call joins
it, and the change is checked in the prop-sway shots.

**13.14 Smaller items.**
- **Speckle (minor 4):** keeps modulating the body glow's intensity in the
  linear path.
- **Cold overlay (minor 5):** stays display-space; F3 checks halo overlap in
  the shots.
- **`test_one_tonemap` (minor 6):** add every Python-composed fire,
  compensation and body-glow shader string to its scan. Remove
  `CPU_TWIN_HOME`, or re-justify it if M3's own-light input still uses
  `aces_curve`.
- **The own-light input (round-2 M3):** a named function in `blackbody.py`,
  `glow_share(temp)`, the 0–1 brightness share the relief's cut has always
  used, with its own test. It is not "as today" by implication.
- **Source upload (minor 7):** `src` and `solid` are uploaded per tile and
  sampled nearest. No CPU upsampling.
- **LUT (M5 note):** the 1024-entry GPU LUT is built from
  `BlackbodyRamp.emission_at_kelvin`; `lut_size` is not raised.

**Work order by patch:**
- **F0b:** 13.1, 13.2, 13.5.
- **F1:** 13.3, 13.10, 13.11, 13.13.
- **F2:** 13.9.
- **F3:** 13.4, 13.7, 13.8, 13.14.

F0b must show the sealed room stays bounded, and tongues forming under the
real wind, before F1 starts.

## 14. F3 as built (2026-10-23) — what the measurements changed

*Implementation record of patch F3 (the look), for HUMAN-TEST 1. Code and
the commit messages are the source of truth; this lists only where the
design above turned out wrong or was decided by a gate.*

- **The hue gate failed on the heat layer's linear move → the §13.8
  fallback ships.** `tools/fire_look_gate.py` (CIELAB, chroma-weighted hue,
  masks from the shot's own META) on F3's first commit (`2fe0589`, the
  linear §13.7 order) against `cfcba40`: the heat-only blasts gained
  +51 % / +62 % / +142 % masked luminance (`explosion_t40/t12/t04`), chroma
  −28 … +18 %, `explosion_t04` shifted hue +17.5°, and the rim went hard and
  blocky (bilinear per-tile linear radiance saturates through ACES right up
  to the tile edge). So the heat layer stays G0's display-space exception,
  drawn by the one fire shader's "heat" stage through G2's glow-through and
  under the flame's cover (`glow·(1−ε)`); with the flame off it matches
  `cfcba40` within the run-to-run noise on every gated shot, and the §13.7
  compensation gate (flame off, `fire_thin/thick_smoke`) passes at −0.3 % /
  +0.0 %. The body-glow-before-units / units-above-the-glow change is
  deferred with it; the FLAME is linear, under the units, with its
  compensation after the ceiling soot. The exception's retirement is its own
  patch (start from `2fe0589`).
- **With the flame on, "chroma within ±4 %" cannot hold on the fss shots:**
  the old look there is a thin glowing line; the flame fills the burning
  tiles' region with a saturated body (chroma +450 … +860 %, i.e. the
  opposite of G0's wash), while the hue stays within 3.1° (A) / 0.4° (B).
  The gate as written measures the heat layer's move; it is reported, not
  bent.
- **§13.4's cover weight was wrong.** `w = smoothstep(glow_min, +100 K)`
  let 850–950 K tongues darken a lit floor by up to 42 % (GL check). The
  cover is now the ramp's own intensity capped at 1
  (`BlackbodyRamp.cover_weight`: 0 at the glow floor, 1 at `kelvin_ref`);
  measured darkening 0.00 %.
- **A white-hot flame fills its active box.** At ~60× intensity even faint R
  saturates, and `fire_thin_smoke` showed the box as a straight white cut.
  The draw fades the flame toward its box edge (a ragged window + an
  exponential brightness ceiling, `fire_glow.fs` `edge_fade`); B's faint
  fringe also needed `e_min` 0.04 → 0.08.
- **A vs B differ, but modestly** (6.7 % of the flame's RGB on the GL
  check's row; on `fss_close` A is crisp and opaque, B soft and
  translucent). Not retuned: it is Erik's call at HUMAN-TEST 1.
- **Look risk for Erik:** the flame reads as tongues, but at the studio's
  overview zoom the row is an even radial "sunburst" (F2's dynamics, not the
  draw); and the colour after ACES is yellow-orange (as F0b's strips).

## 15. F4a as built (2026-10-23) — embers and heat shimmer

*Implementation record of patch F4a, for HUMAN-TEST 2 (with F4b's flicker).
Code and the commit messages are the source of truth; this lists only where
the design above turned out wrong or was decided by a measurement.*

- **The embers vanished under their own fire's soot.** §6.1 put them after
  the flame and before the units, i.e. under the ceiling soot, and the
  studio fire's own soot covers the area its sparks fly through: in the
  first shots no spark showed outside the flame. They keep §6.1's height
  and get the flame's §13.7 COMPENSATION: a second sprite stage after the
  ceiling soot adds back `α_ceil·(1 − glow_through_ceiling)` of them
  (`smokeCoverAt`), so their net occlusion is Erik's 0.2, exactly the
  flame's.
- **Cooling had to be slow for a spark to leave its flame.** With a ~1 s
  e-fold a 1250 K ember reached the 800 K glow floor (where it shows
  nothing) while still inside the flame's ±3-tile extent. `cool_s = 5`,
  `kick_speed = 3.5` tiles/s, `drag = 1/s`: a spark is ~1070 K after 1 s,
  dark by ~3 s, 2–4 tiles out. Sparks born `≤ T_src` keep the flame's
  colour (deep orange to yellow after ACES); they read as sparks by being
  points ~10× brighter per pixel than the surface (`brightness`), not
  hotter.
- **raylib culls back faces.** The direct-GL sprite quads (built from
  `gl_VertexID`, like `pass_quad.vs`) drew nothing until wound
  counter-clockwise after the y-down → NDC flip. The trap is noted in
  `shaders/particle_sprite.vs`.
- **The shimmer's cost follows the heat.** The resolve covers the whole
  world RT (2688² px on the studio), so the term first cost 0.21 ms GPU
  even for two small fires; it now skips every pixel outside the hot
  tiles' bbox (+1 tile) before any fetch: 0.05 ms on `fire_smoke_studio`,
  ~0.21 ms when the whole world is hot (`fire_plume`'s hall).
- **Config homes.** `[render.embers]` and `[render.shimmer]` are their own
  sections, not sub-tables of `[render.fire_vfx]` (whose test holds that
  section to exactly the flame's dials); the shimmer is a resolve term for
  any hot gas, not only the flame's.
- **Accepted gap:** the shimmer's noise runs on the flame's VFX clock; with
  `[render.fire_vfx] enabled = false` that clock stops and the shimmer
  pattern freezes (still shifted, no longer moving).
- **Measured (fire_smoke_studio, RTX 3070, 600 frames at 60 fps,
  `--fire-vfx-perf`):** fire VFX incl. embers CPU 1.38 ms mean (embers
  0.20 update + 0.14 draw, ~17 alive); flame + embers GPU 0.62 ms (embers
  0.012); shimmer GPU 0.05 ms, its CPU pack 0.10 ms.

## 16. F4b as built (2026-10-23) — the room-light flicker (D)

*Implementation record of patch F4b, for HUMAN-TEST 2 (with F4a). Code and
the commit messages are the source of truth; this lists only where the design
above turned out wrong or was decided by a measurement.*

- **How it crosses.** A per-cell EMISSION MULTIPLIER on the sweep's light
  group (`LightChannels.emit_mult`, rows `(y, x, mult_q)` Q16): the cell's
  thermal light emission becomes `(em·mult_q) >> 16` in every ordinate —
  heat never reads it, a cone in the cell is not scaled, ONE is the identity
  and no rows is the pre-F4b sweep. Door: in the grid, `[0, 2·ONE]` (one bit
  of the over-driven headroom), one row per cell. The CPU sweep, the CUDA
  twin (per-call and `(N, h, w)` launch core) and the reference
  (`sweep_ref_q.LightGroup.emit_mult`) agree at tol 0. Python side: the rows
  are render state handed with the cone rows (`Simulation.set_light(...,
  emit_mult=)` → `PhysicsRunner.set_light_emit_mult`), converted from floats
  once at `frame_lights.emit_mult_rows`.
- **"The VFX's local R and T" was not the driver.** R lives on the GPU and a
  readback is out (M11), so the multiplier is built from the flame's CPU
  state: each burning tile's own cluster PUFFING (`fire_vfx_ref.puff_theta`,
  the pulse the flame injects with) plus a per-tile turbulence (8 sinusoids
  hashed from 0.8–5 Hz). `m = 1 + a·clip(z/2, −1, 1)`, `a = 0.3`.
- **No-sync and "follow the puffing" pull against each other.** The puffing
  is shared by every tile of a cluster, so two tiles of one fire correlate at
  about the puffing's share of the variance; the §6.3 gate (|ρ| < 0.3 for
  tiles ≥ 3 apart) holds only with a small share — `puff_share = 0.1`
  (a dial; above ~0.25 one fire's tiles begin to move together).
- **Chance correlation sets the oscillator count.** Two independent
  narrow-band signals still correlate by chance over a short window: with 3
  strongly pink sinusoids per tile ~10 % of tile pairs passed |ρ| = 0.3 over
  10 s. Eight flatter ones (amplitude ∝ f^−1/4) pass on every pair of the
  test layout (a 2×9 fire, a 3×3 fire, five equal lone tiles) over 30 s with
  max |ρ| < 0.3 and mean |ρ| ≈ 0.1; the gate is measured over 30 s (≥ 10 s).
- **`fire_vfx_ref._hash01` is weak** (no first xor-shift: map keys 572 / 574
  hash to 0.3282 / 0.3276). The flicker uses its own full splitmix64. The
  FLAME's puffing phases still come from `_hash01`, so two equal-sized fires
  can draw nearly the same puffing phase and puff in step — F0b's law, not
  changed here (a finding for the arc, one line).
- **Paused.** The flicker rides the flame's VFX clock (`t_noise`), so it
  idles with the flame; main.py relights a paused frame at the TICK rate
  whenever the flicker rows moved (the 4 Hz paused relight would step it
  visibly) — the cost of a running tick's sweep. With `idle_when_paused` off
  the rows freeze and the 4 Hz cadence returns. Without the flame VFX there
  are no rows. `lighting_demo` hands the rows per tick but keeps its 4 Hz
  paused relight.
- **Look: the flicker is real in the field and nearly invisible on screen at
  the shipped tuning.** `main.py --level fire_smoke_studio`, 10 s from 1.5 s,
  flicker on vs off (same sim, cursor lamp excluded):
  - field `light_q`, floor 2–4 tiles from the fire, high-passed (2 s):
    CV 1.4 % → 6.1 % (lone tile 1.0 % → 11 %); the region series west /
    east of the row / around the lone tile go from ρ 0.97–0.99 (they only
    follow the fire's slow growth) to 0.39–0.67 — uneven, never in sync;
    means equal within 0.8 %;
  - SCREEN floor luminance 1.5–4 tiles from the flame: mean 18.1 vs 17.0
    (8-bit) far away — the fire adds ~1 level to the visible floor — so the
    flicker is 0.11 levels RMS frame to frame (0.6 %) against 0.014 off:
    below what the eye resolves. The floor right next to the fire, where its
    light is strong, is under the flame draw. No strobe, no whole-room pulse
    — and at this exposure, no visible breathing. Making it read is the fire's
    room-light brightness (master / sweep gain, #100's smoke), Erik's call at
    HUMAN-TEST 2, not this patch's.
- **Measured cost (this machine, RTX 3060).** Flicker + door per tick: 0.13 ms
  (6 tiles), 0.23 ms (200), 0.62 ms (1000). CPU sweep (relight, 56×56): 2.11
  vs 2.11 ms (noise); GPU per-call sweep 3.21 → 3.29 ms (+ the plane's
  build and one int32 upload).
