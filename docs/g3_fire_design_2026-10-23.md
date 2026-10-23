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
| 2 | v2 | resolution check: are round 1's blockers and majors actually resolved, and did the fixes create new ones? | *pending* | `docs/g3_fire_design_critique2_2026-10-23.md` |

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
