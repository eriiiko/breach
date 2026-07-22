# B2 design — smoke honesty: physical gas medium + the fire studio (render + coefficients)

Arc: Fire & Heat Beauty + Tuning (living plan: `docs/plan-for-tuning-and-graphics.md`,
bottom section). Beat B2 of B1–B4. Written by Fable 2026-07-21 for an Opus build
session (autonomous-patch-workflow). Status: DESIGN — Erik ruled on all seven
open B2 decisions 2026-07-21 (see the living plan) and approved this scope.

Research base: `docs/research/smoke_render_litsearch_2026-07-21.md` (NEW,
untracked — **P0 commits it**) + `docs/fire_rendering_research.md` (merged in B1).
Prior threads: `docs/blackbody_smoke_and_rendering_brainstorm.md` (speckle /
"dirty Planck"), `docs/architecture/engine/05_smoke.md`, `08_ray_engine.md`.
Erik's addition 2026-07-21 (mid-design): the studio level carries **one rotating
beacon** so a sweeping beam rakes through smoke.

## 0. Iron constraints

- **Render + config + tools + levels + docs only.** ONE deliberate exception:
  P0's mechanical species rename (§1) edits sim-side **name strings and
  constant names only**, at the exact surface enumerated in §1 — gas ids,
  slice order, array shapes, and all numeric behavior are UNCHANGED. Everything else reads sim fields
  (`gmap.gas`, `gmap.temperature`, wind, `smoke_glow`) and writes only
  render-side structures. **Goldens/digests byte-identical across every
  patch** — run the digest gate every patch as insurance; if it moves, STOP.
- **No solver changes.** MacCormack/BFECC is DEFERRED with a written trigger
  (§7 non-goals). The gas-chemistry wishes (fuel-gas ignition, condensation,
  smoke heat absorption, poison thermal breakdown) are a SEPARATE future
  sim-side beat — never smuggle them in here.
- **Traffic:** own worktree, branch `fire-b2-smoke-honesty` off main. Arc B
  (logic layer: entities/SignalBus) is concurrently in flight in its own
  worktree — do not touch its files. `tools/lighting_demo.py` and
  `renderer/*` are B2's lane.
- **Feel-adjacent → HUMAN-TEST gate.** Build, gate, push; Erik plays before
  merge. No auto-merge, ever, for any patch in this beat.
- **Credit the source** (repo rule): file headers cite what they implement —
  `gas_medium.py`/`gas_medium.fs`: Beer–Lambert transmittance (pbr-book),
  Frostbite unified volumetrics (Hillaire 2015), premultiplied compositing
  (Quílez); the detail shader: Vlachos SIGGRAPH 2010 flow maps, Neyret
  SCA 2003 advected textures, Perlin–Neyret flow noise, Sigg–Hadwiger
  GPU Gems 2 ch. 20 bicubic. Links live in the research report §4.

## 1. P0 — species rename (pre-approved by Erik 2026-07-21)

`black_smoke → smoke`, `white_smoke → steam`. "Smoke" means fire-smoke and
nothing else from now on.

- The rename is wider than one file — the critique pass mapped the TRUE
  surface (~210 name occurrences across ~36 files, almost all mechanical
  comments/tests). Runtime-critical sites that MUST move atomically in ONE
  commit:
    • `src/simulation/gases.py` — `GAS_NAMES` strings + the
      `WHITE_SMOKE`/`BLACK_SMOKE` constants → `STEAM`/`SMOKE` (ids 0/1 and
      slice order EXACTLY as they are — index-bound: `gmap.smoke = gas[1]`,
      digest field order, EOS append rule).
    • `src/simulation/gamemap.py:44-49` — re-exports the constants (~16
      test files + `tools/eos_p5_bake.py` import from there; sweep them).
    • `src/simulation/physics_runner.py` (×4 sites) — `name_to_id[...]`
      lookups by literal string; a missed one is a KeyError on tick 1.
    • `src/simulation/weapons.py` — validates payload `gas_species` against
      `GAS_NAMES`; the `[weapons.*] gas_species = "white_smoke"` strings in
      config.toml must flip IN THE SAME COMMIT or the weapons table refuses
      to load.
    • config `[gases.*]` table keys themselves.
- C++ identifiers/comments stay as-is this beat (verified: bindings pass gas
  planes positionally/by index; no runtime gas-name strings in `cpp/`; no
  rebuild needed) — leave a TODO comment, sweep another day.
- This surface is DISJOINT from Arc B's lane (entities/SignalBus) — the
  concurrent-worktree rule holds.
- Gate: full suite green (after the test-import sweep), digest/golden gate
  byte-identical, `main.py` boots, one weapons gas-payload smoke test fires.

## 2. P1 — the fire studio (level + harness; Erik's "dark room" ruling)

**Level:** `tools/gen_fire_studio.py` writes `levels/fire_studio/` (generator
script, not hand-painted — reproducible, reviewable; follow
`tools/gen_playground_level.py` and the existing level.toml schema).

Layout (single hull-sealed box, roughly 48×32; exact dims Opus's call):
- **Main hall** (~24×16): near-dark (starlight ambient floor). Two lamps on
  one toggleable group. A cluster of wood crates/furniture mid-left (the
  primary fire). **ONE rotating beacon** (`[[light]] kind="beacon"`, the
  levels-w1 machinery — `src/level_lights.py`) mounted center-hall so its
  beam sweeps 360° through whatever gas is drifting.
- **Sealed side room** (~8×8) off the hall, one door: wood furniture inside.
  This is the O2-starvation stage: ignite, close the door, watch the fire
  choke and blacken; open — reflare. (The fire-doc pick ② scene.)
- **Corridor** (~16×3) with a single lamp at the far end: clean
  beam-through-haze compositions.
- **Water pool**: a few tiles in a hall corner (level water carrier, .npy) —
  standing water for steam context and the wet-floor look.
- **2–3 marine spawns** in the hall (visual scale + flashlight carriers +
  the 3D-marine lighting showcase).

**Harness:** extend `tools/lighting_demo.py` (live sim + raygui sliders +
presets + grenade/water/tilt keybinds exist). HONEST SIZING (critique
finding): the demo's per-frame light-source list is currently the mouse
flashlight ONLY — level statics, beacon tick-angle evaluation, fire lights
+ the HUD count all live in main.py's sources block (~:385-409:
`FireLightSelector`, `monotonic_total_tick`, `_build_light_source`).
**P1's first task is extracting that block into a shared helper** (e.g.
`renderer/frame_lights.py: build_frame_light_sources(...)`) consumed by
BOTH main.py and the demo — a mechanical extraction, zero behavior change
in main.py, and the studio gets beacons + fire lights with no drift risk.
- `--level <name>` argument (default = `CFG.display.level`, as today;
  studio session runs `--level fire_studio`).
- Injection keybinds at cursor: ignite tile; puff `steam`; puff `smoke`
  (the direct-write debug pattern of `src/input_handler.py:255-301` —
  TOOLS may write sim fields; the RENDERER never does).
- Door open/close at cursor: doors are `[[entity]]` `DoorRuntime`s
  (`src/simulation/door_system.py`) — toggle through the door system, not
  tile paint. Lamp toggle: DEMO-SIDE grouping (the harness owns the lamp
  list and rebuilds sources) — the `[[light]]` schema has NO group field
  and gets none this beat. Beacon on/off likewise demo-side.
- AUDIT THE KEYMAP before binding (known collision: `O` is already the
  water-depth overlay); print the final bindings in the on-screen help.
- **Hover-tile readout panel** (the TODO item Erik queued — this beat's
  microscope): T in game units AND pseudo-Kelvin, fire intensity, material
  name, per-gas densities (all five), **O2**. Read-only gmap reads.
- Sliders for every §6 dial + `soot_yield` / `smoke_emission` (the handover
  pair Erik tunes by feel). Preset load must TOLERATE renamed/removed old
  slider keys (`tools/lighting_presets.toml` carries them today).
- Level-feature precedents, per feature (all verified to exist): beacon
  `[[light]] kind="beacon"` (levels-w1, `src/level_lights.py`), `[[entity]]`
  door (`levels/door_test`), `[water]` .npy carrier (`levels/aquarium_demo`),
  unit spawns (the demo already consumes them). The generator composes
  EXISTING schema features only; if one resists generation, hand-write that
  block in level.toml rather than extending the schema.
- Gate: generator is deterministic (fixed content, no RNG); level loads
  headless; readout functions unit-tested headless (pyray-free packing, the
  B1 `pack_emissive_rgba` pattern); harness boots; main.py boots UNCHANGED
  after the sources-block extraction.

## 3. P2 — the physical gas medium pass (the heart of B2)

Replace the flat-grey `smoke_overlay` + additive `glow_overlay` pair with
**one premultiplied layer** built per frame in `renderer/gas_medium.py`:

- **Optical depth, per tile:** `tau = plume_k_scale * Σ_s k_s · ρ_s` over the
  five trace gases, where `k_s = mean(absorption[s])` **from the existing
  GasTable optics columns** — the same data that drives the ray march, so the
  plume body and its god-rays can never disagree about what a gas is. No new
  per-species render dials (soot already carries high absorption → it
  dominates and blackens a mix; steam's is low → thin haze). Single-channel
  alpha (mean extinction) — the same panchromatic collapse `beam_absorb_q16`
  already uses. SINGLE SOURCE OF SCALE (critique finding): the ray march
  already scales absorption by `smoke_absorb_scale` (default 1.4) — P2
  reads THAT as the shared base scale, and `plume_k_scale` is a RELATIVE
  multiplier (default 1.0), so plume body and beam reach track by
  construction instead of by two dials agreeing.
- **Artistic remap in τ-space, never on alpha:** `tau' = tau_curve_a *
  tau^tau_curve_b` (defaults 1.0/1.0 = honest), then
  `alpha = 1 − exp(−tau')`. Linear in thin smoke, saturating in thick.
  This REPLACES `smoke_render_gamma` (delete the dial; the curve pair
  subsumes it).
- **RGB = the lit half:** `aces_tonemap(glow_gain · smoke_glow)` — the ray
  march's inscatter buffer already carries per-gas `scatter_albedo` color
  summed density-weighted (steam scatters near-white, soot barely at all),
  i.e. the species' *lit* identity is already computed. Premultiply by
  nothing further: pack `(RGB, alpha)` and draw ONE premult-over blend.
  Premult-over IS the volume compositing operator `C = inscatter + T·bg`;
  glow-through-soot double-counting and smoke-dimming-its-own-glow become
  impossible by construction. Unlit smoke in a dark room = a black occluder
  (physically right); steam in darkness is invisible (also right — the
  beacon/flashlight reveals it).
- **Per-species hue polish:** poison/teargas keep their identity via their
  existing `scatter_albedo` hues (sickly green / pale warm). A tiny emissive
  legibility floor for gameplay gases is a DESIGN override — expose it as a
  slider defaulting to 0, flag it in the HUD as non-physical if raised.
- **Keep `lighting.py`'s smoke_glow → `light_tex_b` texture packing
  untouched** — the marine shader samples it (contract documented at
  `renderer/marine_shader.py:113`). B2 replaces only the OVERLAY consumer
  of `smoke_glow` (the glow_overlay draw); the light-field texture path to
  the marine/world shaders stays exactly as it is.
- **Accepted gap (documented, deliberate):** the layer draws at the existing
  overlay stage, i.e. OVER the already-ACES-tonemapped world, with its own
  RGB tone-mapped by the same Narkowicz fit (the established B1 pattern —
  every layer internally ACES-consistent). Mathematically, over-blending in
  tonemapped space ≠ tonemapping the blended linear scene; the visible cost
  is slight contrast distortion of bright HDR content seen THROUGH dense
  smoke. The full fix — a linear 16F world RT with one final tonemap pass —
  touches every shader and hits raylib/pyray float-RT platform risk, so it
  is STAGED as its own follow-up patch (candidate B2.5), not smuggled into
  this beat. If the gap reads badly in the studio, that follow-up gets
  promoted; note it in the build log either way. (Accuracy note from the
  critique: today only `lighting.fs` and the B1 `HeatFieldOverlay` apply
  ACES — the old glow overlay is a plain clamp and the old smoke tint is
  untonemapped. B2's layer RAISES tonemap consistency; it doesn't merely
  inherit an existing pattern.)
- **Legacy A/B:** keep the old smoke+glow path behind `[render]
  legacy_smoke_on = false` and a live harness/game toggle, exactly like
  B1's F3 demotion — Erik compares old vs new in one session. `glow_gain`
  defaults to whatever re-matches the current beloved god-ray brightness
  (calibrate by eye against a saved scene before/after PNG pair).
- Gate: unit tests (thin-limit slope ≈ `plume_k_scale·k_s`, soot-dominates-
  steam crossover, alpha monotone and bounded in [0,1], all-zero gas + zero
  glow → fully transparent black, and the ADDITIVE LIMIT: zero optical depth
  with nonzero `smoke_glow` yields alpha≈0 with RGB > 0 — premult's
  additive case, this is steam glowing in a beam and MUST be allowed, do
  NOT assert `RGB ≤ alpha`), before/after PNGs, digest gate untouched.

## 4. P3 — sub-tile detail: the advected-noise smoke shader

New fragment shader `shaders/gas_medium.fs` for the P2 layer's draw (P2 ships
with a plain pass-through sample; P3 turns detail on). The Vlachos/Neyret
recipe, transcribed:

- **Inputs:** the P2 packed layer texture; the density field texture (for
  erosion weighting); a **wind texture** (new per-frame upload: gmap's wind
  vector field packed RG float — render-layer read of a sim field, allowed);
  one tiling fBm noise texture (~256², 3–4 octaves, persistence ~0.56,
  baked at startup); one small static jitter-noise texture. WIND UNITS
  (critique finding): `gmap.wind_x`/`wind_y` are int32 **Q16.16 planes of
  raw `-grad(P)`**, NOT tiles/tick — the sim only turns them into motion
  via `gas_advection_rate` (900). Dequantize (÷2¹⁶) and scale by
  `gas_advection_rate·dt` before packing the RG float wind texture, or the
  advection will be invisibly slow and read as a shader bug. CALIBRATION
  ANCHOR: on the studio doorway jet, the noise drift must visually match
  the plume's own drift — if not, the scaling is wrong, not the shader.
- **Two UV layers** advected by the sampled wind: `uv_i = p·f_base −
  wind(p)·(t − t0_i)·k_adv`, phase-offset by τ/2, crossfaded with
  `w = ½ − ½·cos(2π t/τ)` **plus a per-pixel jitter phase** (kills the
  whole-screen pulse), **half-texture UV offset between layers** (kills
  repetition). Keep accumulated distortion under ~⅓ of a noise tile:
  `τ ≈ (L_noise/3)/v_typical` → default τ ≈ 2–3 s. Base noise wavelength
  2–4 tiles (octaves span ~96 px → ~6 px at 24 px/tile).
- **Noise ERODES optical depth, strongest where density is low** (Nubis-style
  remap → wispy ragged edges, solid cores; macro shape stays 100% the sim's
  field). A few-px **domain-warp** of the sampling UV from the same noise
  hides the tile lattice.
- **Bicubic (Catmull-Rom via 4 bilinear taps)** sampling of the
  density/layer texture (GPU Gems 2 ch. 20) — kills the bilinear diamond
  stars; keep POINT/BILINEAR choices deliberate per texture (this closes the
  old "accidental double-bilinear" TODO for the smoke path).
- **Clock = sim tick, never wall time** (`t = tick · dt`): replays and
  spectators render identical smoke. Dither the thin-gradient range (the
  jitter texture doubles as the dither source) against 8-bit banding.
- Fallback: if the shader seam fights pyray (uniform limits, texture units),
  escalate rather than half-ship; the P2 layer without P3 is still a
  complete, mergeable look.
- Gate: shader-param plumbing unit-tested headless (uniform packing, wind
  texture layout); visual PNG set (still, windy, doorway jet); perf
  spot-check ≤ ~1 ms added at 1080p.

## 5. P4 — speckle A/B, gameplay-gas hues, stretch

- **Dirty-Planck speckle** on the blackbody overlay (`HeatFieldOverlay`),
  Erik's spräcklig idea, two variants behind one `speckle_mode` toggle:
  (a) `noise` — pure render noise modulating the LUT color/intensity;
  (b) `soot` — amplitude SEEDED BY THE REAL LOCAL SOOT density (dirty
  Planck: chemistry decides where the flame is dirty; the O2-choke story
  shows in flame color for free). Both must MOVE with the flow (a static
  speckle reads as a screen overlay — the lit-search's hard rule).
  Cross-layer seam resolved by the critique: `HeatFieldOverlay` is a
  CPU-packed additive texture with NO fragment shader, so speckle does NOT
  sample the P3 shader — the two-layer advected-phase recipe is factored
  into a small shared module and ALSO evaluated CPU-side at grid resolution
  (256² numpy is cheap, and per-tile flame mottle is grid-scale anyway),
  modulating the overlay's color/intensity at pack time.
  `speckle_amp` slider; keep steam-side mottle ≤ ~10% and low-frequency.
- **Gameplay-gas hue check** in the studio: poison/teargas legibility under
  the new medium (albedo hue, optional non-physical floor slider).
- **Stretch, default OFF, skip if timebox pressures:** fuel-gas heat-haze —
  near-zero `k_s` (invisible body) but density drives a small screen-space
  domain-warp of the already-rendered scene (the lit-search's refraction
  identity). Config-gated `fuel_haze_on = false`.
- Gate: A/B toggle live in harness + game; PNG pairs of both speckle modes
  on the same scene; suite green.

## 6. Config summary (all NEW/CHANGED, all render-layer)

```toml
[render]
legacy_smoke_on = false     # old flat smoke+glow pair, kept for A/B

[render.gas_medium]
plume_k_scale = 1.0         # RELATIVE multiplier on the shared smoke_absorb_scale base
tau_curve_a = 1.0           # τ' = a·τ^b — artistic remap IN τ-SPACE
tau_curve_b = 1.0           #   (b>1 steepens edges; 1/1 = fully honest)
glow_gain = 1.0             # re-match beloved god-ray brightness (tune)
effect_gas_floor = 0.0      # NON-PHYSICAL legibility floor, poison/teargas
fuel_haze_on = false        # stretch: fuel_gas refraction shimmer

[render.gas_detail]
enabled = true
noise_octaves = 4           # fBm, persistence ~0.56 (Kolmogorov-flavored)
noise_wavelength_tiles = 3.0
adv_gain = 1.0              # k_adv: wind → UV advection rate
cycle_seconds = 2.5         # τ, from the ⅓-distortion rule
erode_strength = 0.6        # low-density erosion depth
warp_px = 3.0               # sampling-UV domain warp
dither_on = true

[render.speckle]
mode = "soot"               # "off" | "noise" | "soot"  (A/B in studio)
amp = 0.25
```

(`[smoke] smoke_render_gamma` is DELETED — subsumed by the τ curve. The
delete sweep is bigger than the key: `tests/test_smoke_render_gamma.py`
asserts the key exists (rework into τ-curve tests), `tools/lighting_demo.py`
carries old smoke sliders at several sites (rework for the legacy/new
pair), and `engine/05_smoke.md` + `graphics/tuning_guide.md` mention it
(doc sweep at canon fold). `soot_yield` / `smoke_emission` are existing sim
config, exposed as harness sliders only — values change ONLY in Erik's feel
session, as their own deliberate config commit.)

## 7. Patch plan (autonomous-patch-workflow) + non-goals

- **P0** — species rename sweep (§1). (The design doc + lit-search report
  are ALREADY committed on main by Fable at design close — the
  commit-docs-before-worktrees rule; a worktree cut from main sees them.)
  Digest gate green.
- **P1** — fire studio generator + level + harness extension + hover
  readout (§2).
- **P2** — `renderer/gas_medium.py` single-layer medium pass + legacy A/B
  toggle (§3). Before/after PNGs.
- **P3** — `shaders/gas_medium.fs` advected-noise detail + wind texture +
  bicubic + dither (§4).
- **P4** — speckle A/B + gas hues + optional fuel haze (§5).
- **P5** — perf pass (frame-time budget with beacon + fire + full smoke in
  the studio), HUD annotations (active medium path, speckle mode), the
  HUMAN-TEST build, push. Erik plays; only he merges.

Per-patch gates: `pytest tests -q` green; digest/golden gate byte-identical;
dependency direction: `src/simulation/` must NEVER import `renderer/` (the
renderer importing simulation is fine and existing); screenshot artifacts
at P2–P4.

**Non-goals (written down so they stay out):** MacCormack/BFECC — deferred;
REVISIT TRIGGER: if, after B2, plumes measurably lose peak density /
silhouette while TRAVELING (transport dissipation — no render trick recovers
it), open it as a gated sim change (digest re-baseline + rationale +
HUMAN-TEST). Gas chemistry (fuel-gas ignition M3, steam condensation,
heat-driven boil, smoke heat absorption, poison thermal breakdown) — future
sim beat, coefficients may live in GasTable columns but every PROCESS is a
solver/exchange-row change with full ceremony. Linear-HDR world RT — staged
follow-up (§3 accepted gap). 2.5D smoke layers — out of arc (§8 item 7).

## 8. Human-test script (Erik's studio session)

Launch `tools/lighting_demo.py --level fire_studio`. Dark hall: beacon ON —
the sweeping beam should carve a visible moving shaft through a steam puff
(Aliens escape-pod read: beam barely dims, haze lights up). Flashlight
marine in the corridor: beam through drifting smoke vs steam — soot should
EAT the beam, steam should GLOW in it. Ignite the crate cluster: watch the
plume — wispy advected edges (P3), no tile diamonds, no grey film; toggle
`legacy_smoke_on` for the old look, live. Dying fire: soot handover —
`soot_yield`/`smoke_emission` sliders until a starving fire visibly blackens
its own room. Sealed room: ignite, shut door — choke, blacken, go murky;
open — reflare (O2 readout on hover tells the story). Speckle A/B: `noise`
vs `soot` mode on the same blaze — pick by eye. Grenade into smoke (the
existing harness key): shock-lit plume sanity check. Watch the HUD frame
time with everything burning at once. Expectation-setting from the numbers:
at defaults, max-density soot alone reads ≈ 0.6 alpha — the full
black-occluder look lives in the `plume_k_scale` / τ-curve dials; that is
what they are for, not a bug.

## 9. Kickoff prompt (paste into a fresh Opus session)

---

You are building **B2 of the Fire & Heat Beauty arc** on the breach project:
the physical gas-medium render pass, the fire-studio level + harness, and
the speckle experiments. **Render + config + tools + levels only — the sole
exception is P0's mechanical name-string rename (ids/slices/behavior
unchanged). Zero solver changes. Goldens byte-identical throughout.**

The design is LOCKED: `docs/fire_b2_smoke_honesty_design_2026-07-21.md`
(read ALL of it; §7 is your patch plan P0–P5, §6 the config schema, §0 the
iron constraints). Context: the arc's living plan is the bottom section of
`docs/plan-for-tuning-and-graphics.md`; the research base is
`docs/research/smoke_render_litsearch_2026-07-21.md` and
`docs/fire_rendering_research.md` (both already on main). Workflow: autonomous-patch-workflow
(fresh subagent per patch, own full context; memory checkpoint at every
patch boundary; surface surprises, don't plough ahead).

Environment: Python = conda env `data` (per-machine specifics in
docs/dev_setup.md / docs/lenovo_dev_setup.md); pytest = `pytest tests -q`;
no C++ build needed (P0 leaves C++ identifiers alone). Work in your OWN
worktree on branch `fire-b2-smoke-honesty` off main. Arc B (logic layer:
entities/SignalBus) is in flight in its own worktree — its files are
off-limits; P0's rename surface (§1: gases.py, the gamemap re-export,
physics_runner name lookups, weapons `gas_species`, config, tests) is
disjoint from it.

Gates: per-patch tests; digest/golden gate byte-identical EVERY patch (run
it even though the beat is render-only by construction); before/after PNG
pairs at P2–P4; perf + HUD at P5. The beat is FEEL-ADJACENT: finish =
built, gated, pushed, HUMAN-TEST build ready — **Erik plays before any
merge. Never auto-merge.**

Escalate (STOP, bring Erik/Fable) if the work seems to require: touching
`cpp/`, the solver, recorder/digest surface, or any sim behavior; any
golden moving; the pyray shader seam not supporting the P3 plumbing (the
`WaterPass` precedent at `renderer/water.py:399-411` binds 5 textures —
follow it); the glow re-calibration failing (old god-ray character
unreachable via `glow_gain`); or the P0 rename demanding BEHAVIOR edits
beyond the §1-enumerated name-string surface (the ~36-file mechanical
sweep itself is EXPECTED — do not escalate over its size).

---
