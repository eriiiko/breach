# The explosion look (G4, #75 / #94)

Graphics arc G4 of `docs/graphics_direction_2026-10-18.md` §4, after the
standard explosion (#31), the fire (G3) and the smoke (G1, G2). Everything in
it is render-only: no sim state, no digest, no golden moves, bar one new
field on a render-only event. Erik judges every piece at the screen
(HUMAN-TEST). Design session 2026-11-07, Erik and Claude.

## 1. What Erik ruled (2026-11-07)

1. **No separate explosion flash as a principle.** The explosion lights its
   room with its own hot soot through the sweep; the situation falls out of
   the physics. The flash machinery is built because the **flashbang** needs
   it (not hot, so an emitter or nothing), and because a row may still need a
   top-up: so **every payload row may carry a flash, as a row property**, 0
   where the fireball's own light is enough. The #94 measurement (light at
   the units' tiles beside each blast, flash at 0) sets each row's number.
2. The flash is **white light, very bright, think of a flashbang**; it comes
   from **more than one tile** so the sweep's 16-direction star is smeared
   into a disc.
3. The blast smoke **absorbs the flash and casts its shadow** ("more dynamic
   lighting, otherwise the same result each time"). This is free: the sweep
   reads every emitter and every absorber of the same tick.
4. Smoke is settled by #31 and the 2026-11-06 retune (`smoke_scale` 0.25,
   "pretty good"). #94's options (a)/(b)/(c) are closed unless the
   measurement shows the soot swallowing the flash.
5. The legacy grenade ring is deleted (his words on #75, 2026-10-29).
6. The light budget: #173 is merged (`98eb9f8`): the door charges each
   ordinate, an omni lamp pays 1/16, so an omni level holds about 2048 light
   units; muzzle flashes cap themselves at 1024.
7. Erik is most interested in the shock front, the sparks and debris, and the
   ring deletion.
8. **The scripted ring is the one piece Erik has not accepted on paper**
   ("I'd like to see it before it's merged"): P2 is a HUMAN-TEST on that
   question. If he does not like it, the fallback experiment is the
   pressure field's own "flawed shockwave" (§2.2: a whole-room pulse from
   |v| or P), tried as a second term before anything else is built on the
   ring.

## 2. Before pictures and measurements

### 2.1 The look shots (`tools/lighting_demo.py --shots blast_final_t52,...`)

Rendered at `674ed59` into this doc's folder
(`docs/explosion_look_design_2026-11-07/`): `blast_final_t52`, `_t60`, `_t84`,
`_t168`. Four rooms, nine 1.2-unit lamps each; frag, breach on wood, magazine,
C4 at tick 48.

- **t52 (tick +4)**: every room washes warm or white. That is the heat layer
  (`blackbody.pack_emissive_rgba`, display-space, G0's named exception): it
  paints hot air and lights nothing, no marine, no shadow.
- **t60 (tick +12)**: the frag room's floor is as grey as the untouched breach
  room. The magazine and C4 rooms are warm from their soot's emission. So a
  blast lights its room exactly as far as its hot soot emits (Kirchhoff: gas
  emits only through the smoke term, `emitted = a_c · L°[T]`); #94's
  complaint holds for the frag.
- The faint circle in the breach and magazine panes at t52/t60 is the legacy
  ring (`renderer/game_renderer.py`, the `"explosion"` effect, life 0.6 s).
- **t168**: the hanging soot and the fires; the smoke itself is right.

### 2.2 The pressure plane after a blast (probe, blast_final, row through each centre)

| tick after | frag P (atm) | C4 P (atm) | frag v (m/s) | C4 v (m/s) |
|---|---|---|---|---|
| +1 | 1.09..1.23 room-wide | 2.1..2.5 room-wide | 213 at the centre, 54, 30, 24, 19 ... outward | 78.7 everywhere (the cap) |
| +2 | 0.89 flat (undershoot) | 1.41 flat | 44 at the centre, 31, 28 ... | 14..68 |
| +3..+10 | 1.01 flat | 1.4 down to 1.1 | under 10 | 60 decaying |

The projection equalises the room in one tick: pressure jumps everywhere at
once, undershoots, then decays flat. The velocity is a radial outflow over
the whole room from the first tick, monotone outward, decaying in TIME (the
frag's centre 213 → 44 → 10) with calm nowhere behind it. §4's "refract by
∇P through a ring mask" has no travelling front to refract, and |v| would
warp the whole room as a pulse, not a ring. The C4 fills its room at the
cap; a bigger room would show the fall-off, still everywhere at once.

**Ruling from the data**: the shock front is a scripted ring from the
detonation event; its LIFETIME may follow the outflow's decay (it lives
while the blast's air still moves), its SHAPE cannot come from the planes.

## 3. The design

### 3.1 Delete the legacy ring

`GameRenderer._convert_events`' `ExplosionEvent` → `"explosion"` effect and
the `"explosion"` branch of the effects drawer go. The event feeds §3.2–3.4
instead. The `_effects` queue stays legacy display-space for tracers, hits,
jets (the particle rule).

### 3.2 The event carries the row's strength and flash

`ExplosionEvent(pos, radius, kind)` gains `strength: float` (the row's, 0 for
a row without one) and `flash: float`, `flash_decay_s: float`,
`flash_color: tuple` (the row's flash columns, §3.5). Render-only event,
outside the digest (only UnitHit/UnitKilled hash); `execute_payload` fills
it from the `PayloadDef`. Existing consumers (`combat.py`,
`heat_source_system.py`, `timed_charge_system.py`, tests) read `pos`,
`radius`, `kind` unchanged.

### 3.3 The shock front: a term in the resolve's distortion block

`renderer/shockwave.py` (`ShockwaveSettings`, `Shockwave`) on the heat
shimmer's pattern (`renderer/heat_shimmer.py`), the term `shock_term(wpos)`
one more line in `distortion_px()` of `shaders/world_resolve.fs`. Never a
second pass.

- **Rings.** A detonation starts a ring at its centre: radius `c·t` (t from
  the detonation, on the VFX clock `SimClock.t_noise` so it idles with the
  flame while paused and a look shot reproduces), band width `w` tiles,
  amplitude `A·sqrt(strength)` world px (zoom-faded as the shimmer is). A
  volley is several live rings; a ring dies when its band has left its bbox
  or at `life_s`. Cap `max_rings` live rings, oldest dropped.
- **Resolution: the ring is PER PIXEL, the mask per tile.** (Erik asked
  whether the shimmer is per tile: its strength is per tile, bilinear, its
  noise per pixel.) The ring must be sharper than a tile, so it is evaluated
  analytically in the shader: per live ring a uniform row `(cx, cy, r, w, A)`
  (tiles, tiles, tiles, tiles, px; up to `max_rings` rows), the term
  `Σ_rings g(|p − c| − r) · unit(p − c) · A` at the PIXEL's tile position
  `p`, `g` a smooth bump of width `w`. The per-tile part is only a GAS MASK:
  one `(h, w)` texture (1 on gas, 0 on solid / vacuum, the shimmer's masks),
  sampled bilinearly at `p`, so walls clip the ring for free, a corridor
  shows a front that follows the corridor, and a pixel whose four nearest
  tile centres are not gas gets EXACTLY zero. The mask is static per level
  (re-uploaded when `render_dirty` says a tile changed); nothing else is
  packed per frame. The numpy twin `ring_offset_px(...)` (per-pixel formula
  × bilinear mask) is THE spec. The summed offset is clamped to `A_max·zoom_fade`.
  Outside the rings' union bbox no fetch and no arithmetic at all; without a
  live ring the term is `vec2(0)` (`u_shock_n == 0`), so the resolve is the
  plain point fetch bit for bit.
- **Dials** `[render.shockwave]`: `enabled`, `speed_tiles_s` (40: a frag
  ring crosses a 10-tile room in a quarter second), `width_tiles` (1.0),
  `amp_tiles` (0.25 at strength 1, times strength^`strength_power`; 0.5 = sqrt,
  Erik's dial for how far a big charge out-rings a grenade, 0.25 shipped), `life_s` (0.5),
  `max_rings` (6), `zoom_zero` / `zoom_full` (the shimmer's). Read every
  frame through a refusing door; F5 retunes live. A key is NOT added (switches
  are settings).
- **Cost** follows the rings' bbox; budget 0.2 ms GPU on blast_final's
  volley (measured with `lighting_demo --perf`, the G3 way).

### 3.4 Sparks and debris: the particle primitive, extended

Both consumers of `renderer/particles.py` (`Particles`, `integrate`,
`pack_sprites`, `ParticleDraw`), never a second system. Spawned on the CPU
from the detonation event (never a GPU readback), seeded RNG on the VFX
clock, capped, every particle with a lifetime.

- **Sparks** (first). `SparkSystem`, `[render.sparks]`: `n = rate·sqrt(strength)`
  particles born at the centre's open tiles with a radial speed
  `v0 ± spread` tiles/s (uniform direction), drag toward the frame's tamed
  wind × tps plus the twin's curl noise (`integrate` as the embers), born at
  `t_born_k` (2600 K) cooling in closed form, coloured ONLY through the
  blackbody GPU LUT, life `life_s ± jitter` (0.3..0.8 s). Drawn by the
  existing additive pass under the units, with the ember's compensation
  after the ceiling soot. Gas tiles only: a spark that enters a solid tile
  dies (the integrate's air_fn already answers that).
- **Debris** (second). Dark chips cannot be drawn additively (additive HDR
  never darkens a floor), so `ParticleDraw` gains a second blend mode,
  `stage="debris"`: premultiplied alpha, under the units, after the floor
  and the floor smoke group. `DebrisSystem`, `[render.debris]`: `n =
  rate·strength` chips, radial `v0`, strong drag (they stop within ~0.5 s
  and lie there), a slow fade over `life_s` (2..4 s), size and a per-chip
  rotation from the hash (`fire_vfx._hash01`, THE fire's hash, never a
  second copy), albedo a dark grey with the tile's light applied in the
  sprite shader from `light_tex_a/b` (the lit3d read, so debris in the
  blast's shadow is dark). A chip's shape is a small rotated quad; no
  texture.
- The particle data array `(n, 8)` gains what the two need (a kind /
  rotation column) additively; `tests/test_particles.py` extends.

### 3.5 The flash: a payload-row property, through the frame lights

- **Row columns**, `[payloads.*]`: `flash` (light units at peak, default 0 =
  none), `flash_decay_s` (default 0.3), `flash_color` (default white
  `[1.0, 1.0, 1.0]`). The door (`weapons.py`'s payload loader) refuses
  `flash < 0`, `flash_decay_s <= 0` with `flash > 0`, a colour outside
  [0, 1]³. A row with `strength` keeps its soot's own light and adds its
  flash on top; a row WITHOUT `strength` (the flashbang) is a flash alone.
  Starting PLACEHOLDERS, Erik tunes at the screen: frag 60, breach 60,
  magazine 0 (its soot lights the room), C4 400, flashbang 1500.
- **`[payloads.flashbang]`**: `flash = 1500`, `flash_decay_s = 1.5`, no
  `strength`, `radius = 0`, no heat, no smoke; `emit_blast_smoke = false`.
  Fires through `execute_payload` like any row (a timed charge in the studio;
  a throwable flashbang is a `[weapons.*]`/loadout row for the weapons
  session, not this arc). Its blinding of units is gameplay, not here.
- **The transient**: `frame_lights.blast_flash_lights(effects, ...)` beside
  `muzzle_flash_lights`, pure: from the renderer's live `"blast"` effects
  (the event's `pos`, `flash`, `flash_decay_s`, `flash_color`, age), each
  flash is `flash · exp(−age / decay)` light units, full on the first frame,
  dropped below 1 % of peak. SOURCE: the centre tile plus its open
  4-neighbours (solid / vacuum skipped), the intensity split equally, so one
  flash is up to five omni cone rows; the star of one cell becomes a disc.
  Through `frame_light_specs(extra=)` and `cone_rows`, so the soot absorbs
  it and casts its shadow. `EMITTER_INPUTS` gains the row ("sim event, render
  envelope").
- **Budget**: the door is per ordinate (128 units; an omni row pays 1/16),
  so an all-omni frame holds 2048. A volley's flashes cap themselves as a
  group at `BLAST_FLASH_MAX_TOTAL = 1024` (the brightest kept proportionally);
  `MUZZLE_FLASH_MAX_TOTAL` drops from 1024 to 512 (85 simultaneous full
  flashes was far beyond any shipped match; 42 still is), leaving 512 for
  lamps, flashlights, jets and the cursor lamp (a large recipe ship's lamps
  are about 60). Nothing raises in the frame path.
- A flash is on the FRAME clock (wall-clock seconds, like the muzzle flash):
  paused, it holds.

### 3.6 Smoke: no change, one measurement

`tools/fire_smoke_probe.py`-style, headless: for each blast_final row, at
ticks +1..+12, `light_field.light_at` at the units' tiles 3 and 6 tiles from
the centre, (a) as shipped, (b) with blast smoke off (`emit_blast_smoke =
false` on a copied `PayloadDef`, the lighting_demo rule), (c) with the row's
starting flash. (a) vs (b) answers #94's "is the smoke hiding it"; (a) vs
(c) sets the row's flash. Numbers into this doc's as-built section; if (b)
is many times (a) at tick +1, option (a) of #94 (a broken pattern) comes
back to Erik, otherwise #94 closes with this arc.

## 4. Verification (properties)

- `tests/test_shockwave.py`: the twin's offset is zero outside the band and
  where the bilinear gas mask is zero; a ring's offset points away from its
  centre; two rings sum; the ring is sharper than a tile (two pixels of one
  tile on either side of the band differ); a ring dies at `life_s`; `max_rings` holds; the door refuses a
  missing / mistyped key. `tests/gl_shockwave_check.py` + wrapper: a pixel
  whose four nearest tiles are zero is bit-identical to the plain resolve
  (the shimmer's `shimmer_zero_where_cold` pattern).
- `tests/test_particles.py` extended: spark count scales as sqrt(strength),
  debris as strength; every particle has a lifetime it never outlives; the
  cap holds; a spark entering a solid tile dies; the debris stage is drawn
  premultiplied, the sparks additive (the draw's blend factors asserted).
- `tests/test_blast_flash.py`: the envelope (full on frame 0, exp decay,
  gone under 1 %); the source cells are the centre + open 4-neighbours
  only; the group cap scales a volley proportionally; the payload door's
  refusals; `blast_flash_lights` + `muzzle_flash_lights` + a recipe ship's
  lamps pass `cone_rows` at their caps (no raise, by construction).
- `tests/test_payloads.py`: the event carries `strength` and the flash
  columns; `flashbang` executes with no field edit, one event.
- `tests/test_one_tonemap.py`, `test_no_blas_in_frame_path.py`,
  `test_ingress_lint.py` unchanged and green; the digest goldens untouched
  (a render-only arc: `pytest tests -q` proves it).
- Look shots: the four `blast_final_*` before/after, plus two new rows:
  `blast_final_t50` (tick +2, the ring mid-room) and `flashbang_t50` on a
  new `levels/flashbang_studio` (one flashbang at tick 48 in a lamp-lit room
  with four marines and a smoke column from a smouldering crate, so the
  shadow is seen). The implementer LOOKS at the PNGs before claiming a patch.

## 5. Patches

| # | patch | gate | merge |
|---|---|---|---|
| P1 | event fields (+ `approach`) + `reach_depth` + ring deletion + main.py consuming events every sim step + the §3.6 (a)/(b) measurement | tests; the measured table in the as-built | auto on green (nothing visible but the ring's absence; Erik sees it at P2) |
| P2 | shock front (`shockwave.py`, the shader term, dials, GL check, `blast_final_t50` shot) | tests + GL check + shots | HUMAN-TEST |
| P3 | sparks | tests + shots | HUMAN-TEST |
| P4 | debris (the second blend mode) | tests + shots | HUMAN-TEST |
| P5 | the flash: row columns + door, `flashbang` row + studio, `blast_flash_lights`, the shared transient cap, the §3.6 (c) measurement, `flashbang_t50` shot | tests + shots | HUMAN-TEST |

One implementation agent at a time, Opus, each in this worktree's branch;
a critic on this doc before P1 (verdict to disk). P2–P5 are independent of
each other after P1; order as Erik wants to look.

## 6. Critique

One critic pass (Opus, against `60ef6a2`), verdict in
`docs/explosion_look_design_2026-11-07/critique.md`: 3 blockers, 9 shoulds,
5 nits. Resolved on paper here (v2); the sections above are read WITH this
table, which wins where they disagree.

| finding | resolution (v2) |
|---|---|
| B1 the caps: halving the muzzle cap re-derives gen_ship's lamp budget and can pass the per-ordinate door | **No cap moves.** Muzzle and blast flashes share ONE group cap, `TRANSIENT_MAX_TOTAL = 1024` (the constant `MUZZLE_FLASH_MAX_TOTAL` is renamed, its value kept, gen_ship's `MUZZLE_FLASH_LIGHT` and `test_levelspec` follow the rename only); within the group the muzzle flashes scale down first, the blast flashes last. A level's lamp budget stays what #173 made it. The flashbang's placeholder is 800 (under the cap, as every placeholder must be) |
| B2 the ring shows through walls; the canonical Blast reach walk is ignored | **The ring rides the walk.** At detonation the ring takes the canonical `blast_expansion.reach_tiles` walk from its centre (the event carries `approach` for the seed rule, so a charge on a wall seeds beside it) extended, never copied, with a BFS depth output (`reach_depth`, the same module: hops along the open mask, 4-neighbour). Per live ring a per-tile plane of that geodesic distance `d_g` (tiles; unreached tiles hold a large value, never −1, so the bilinear read fades the front at a wall instead of inventing one); up to `max_rings = 4` planes packed as the RGBA channels of ONE texture. The shader evaluates `g(d_g(p) − r)` per pixel on the bilinear `d_g`, direction the straight line from the centre, so the front follows a corridor and never enters the room behind a wall. The per-level gas mask and its `render_dirty` read are dropped |
| B3 the "smoke off" measurement is vacuous (a strength row always deposits smoke) | (b) is measured on a THROWAWAY sim after the fact: step to tick +k, zero the smoke trace plane in place (`gmap.gas[..., SMOKE] = 0`, a test fixture's write on a sim that is then discarded, no energy seam involved), `sim.relight()`, read `light_at`. (c) moves to P5 where `blast_flash_lights` exists. #94 closes only on a measurement that could have failed |
| S1 pause: effects age by wall clock | The flash and the ring are on the SIM clock: age = `(tick − t0 + alpha) / tps` from `upload_state(alpha=, paused=)`, so paused holds and a look shot reproduces. Sparks and debris on the VFX clock like the embers (idle drift while paused, the ember's accepted behaviour) |
| S2 look shots of a wall-clock flash do not reproduce | Resolved by S1 |
| S3 detonations in an earlier sim step of a slow frame are lost (`tick_events` cleared per step, main.py runs up to 5 steps per frame) | Pre-existing, but every piece of this arc depends on the event, so P1 fixes it: main.py consumes events after EVERY sim step inside its step loop (a one-line move; lighting_demo checked for the same) |
| S4 the gas mask bypasses the wall seam | Gone with B2 (the plane is per ring, built once at detonation from the walk) |
| S5 debris: `integrate` kills cold particles, drags toward the wind, `pack_sprites` is ember-shaped | Debris has its own twin stage in the same module, `integrate_debris` (drag toward REST, no glow-floor kill, a settle time), and `pack_sprites` takes a kind column (ember / spark / chip); the ember path stays bit-identical (gated) |
| S6 P1's (c) needs P5 | Moved (B3) |
| S7 flashbang 1500 over the cap | 800 (B1) |
| S8 the GL check proves only the zero case | `gl_shockwave_check.py` also renders ONE ring over a synthetic checker scene RT and compares the resolved image to the twin's CPU resample of the same checker, tolerance 1/255 per channel over the band, which also catches the y-flip sign |
| S9 the payload door never refuses an unknown key | The door refuses any `flash_*` key that is not one of the three (by prefix) |
| N1 `consume_events`, the `fresh` flag | Named right in the briefs; the `"blast"` effect carries `fresh` as tracers do |
| N2 the +0.5 tile-centre offset | Ring centre, flash cells and spark births at tile centres (+0.5) |
| N3 `test_payloads.py:174` compares event reprs | The new fields get defaults (`strength=0.0`, `flash=0.0`, `flash_decay_s=0.3`, `flash_color=(1.0, 1.0, 1.0)`, `approach=None`) so existing constructions and reprs hold |
| N4 undefined ring terms | bbox = the walk's tiles' bbox + 1; `A_max` = `amp_tiles · sqrt(strength_max_live) · px_per_tile`; `g(x) = exp(−(x / w)²)` on support `|x| < 2w`, 0 outside |
| N5 a blast centred on a wall | The walk's own seed rule (`approach`, else every open 4-neighbour) gives the ring its tiles, the flash its cells (the seeds, up to 5) and the sparks their birth tiles |
| N6 the debris draw slot, exposure, shader splice | Drawn after the floor smoke group and BEFORE the flame (chips lie on the floor), linear HDR `albedo × incoming light` from `light_tex_a/b` through the lit3d block (`_FIELD_SAMPLE_GLSL`, the props' way), the one tone-map untouched |
| N7 `cone_rows` enforces no budget | The flash budget test evaluates the sweep's own per-ordinate door (`RadiationSweep.cone_emission` sums ≤ `LIGHT_CONE_BUDGET`), not `cone_rows` |


## 7. Systems

**Existing canonical systems this design uses** (CLAUDE.md inventory):
Resolve distortion block (the ring is a term); Particle primitive (sparks and
debris are consumers); Frame lights (the flash is a transient in the one
assembly; `EMITTER_INPUTS`); Blackbody GPU LUT (sparks' colour); Tamed wind
(sparks' drag); FireVFX twin's `_hash01` (debris rotation, spark jitter);
Light accessor / `light_tex_a/b` (debris lit by the field); Look shots (two
new rows); One tone-map (nothing tone-maps itself; the ring shifts the
resolve's read); Payload executor + the strength door (the flash columns are
row columns, refused at the one door); Generic explosion (untouched);
Config via `CFG` with refusing doors; Test conventions (GL check in a
subprocess).

**New systems it creates**, draft rules for CLAUDE.md at implementation:
- *Shock front*: `renderer/shockwave.py` + `shaders/world_resolve.fs::shock_term`
  — THE blast's screen-space ring: a term of the resolve's distortion block
  from the detonation event, analytic per pixel times a bilinear per-tile gas
  mask (walls clip it), its spec the module's numpy twin; never from the pressure
  or velocity planes (measured 2026-11-07: the projection has no travelling
  front), never a second pass.
- *Blast flash*: `frame_lights.blast_flash_lights` + the `[payloads.*] flash`
  columns — THE detonation light: a row property, white by default, the
  centre + open 4-neighbours as its source, capped as a group at
  `BLAST_FLASH_MAX_TOTAL`; a hot row's room light stays its soot's own
  emission, the flash is a top-up the #94 measurement sizes; the flashbang is
  a row with `flash` and no `strength`.
- *Sparks / debris*: `particles.SparkSystem` / `DebrisSystem` + the draw's
  `"debris"` stage — the particle primitive's second and third consumers; a
  dark particle is drawn premultiplied in the debris stage, never additive,
  never a third blend path.
- *Legacy effects queue*: the `"explosion"` kind no longer exists; a new
  explosion visual is one of the three above.

## 8. As built

### P2, the shock front (2026-11-07)

- `renderer/shockwave.py` + `shaders/world_resolve.fs::shock_term`, one line
  in `distortion_px`; `WorldComposite.resolve(shock=)`; `[render.shockwave]`
  (the §3.3 dials, `max_rings` 4 per B2), a refusing door.
- **The distance plane is the straight line plus the walk's detour**, not the
  raw hop count: `reach_depth`'s hops are an L1 metric, whose equal-depth
  lines are diamonds, so a ring on raw hops would be a square turned 45°.
  `d = hypot(tile − centre) + (depth − L1 to the nearest seed)`: exactly the
  straight line where the walk went straight (a round ring in a room), longer
  by the way round behind a wall. Membership is still the walk's alone.
  Claude's call; Erik judges at the HUMAN-TEST.
- The ring is born on the tick after the detonating step (`tick0` = the
  monotonic tick `consume_events` is stamped with), so `blast_final_t50` is
  ring age 1 tick (r ≈ 1.7 tiles), not 2.
- **Zoom**: the fit-the-map shots sit at ~16.8 screen px per tile, just above
  `zoom_zero` 14, so the ring is faded to ~6 % there and not visible; four
  close-up rows (`blast_final_t50_frag/_c4`, `_t52_frag/_c4`, zoom 40) show
  it. Whether a coherent travelling ring needs the shimmer's zoom fade at all
  is Erik's question.
- **The C4's ring is hidden** at every tick it lives: by the heat layer, which
  is display-space and drawn AFTER the resolve (G0's named exception), so it
  covers the ring and is not itself distorted. The frag's ring shows on the
  floor at +4 (max 13/255 against the same shot without the ring: subtle on
  blast_final's low-contrast floor at `amp_tiles` 0.25).
- GPU: `lighting_demo --level blast_final --shock-perf 600`, the volley's 4
  rings held live and their ages swept: resolve 0.469 ms on vs 0.344 ms off,
  **0.125 ms** for the term (world RT 3600×3024; budget 0.2 ms).

### P3, the sparks (2026-11-07)

- `renderer/particles.py::SparkSystem` (+ `SparkSettings`,
  `spark_settings_from_config`, `spark_count`, `spark_rng`, `spark_spawns`),
  `[render.sparks]` behind a refusing door (the shock front's), owned by
  `GameRenderer` beside the embers and stepped in `upload_state` right after
  them. The particle primitive's second consumer: `Particles`, `integrate`,
  `pack_sprites` and `ParticleDraw` are used unchanged; the ember path is
  byte-identical (`tests/test_particles.py` untouched and green, plus
  `tests/test_sparks.py`'s ember-only-frame assertion).
- **Consumption**: each `"blast"` effect is read once, `sparks_seen`; while
  the system is off (or the flame VFX, whose clock it runs on, is off) a
  blast is marked seen and never thrown later.
- **Births at the walk's seeds**: the renderer hands in
  `walk_seeds(open_mask(gmap), fy, fx, approach)` as `seeds_fn` (so
  `particles.py` stays sim-free, its render-only test unchanged); each spark
  takes one seed's tile centre (+0.5), so a blast centred on a wall throws
  from beside it. The RNG is `default_rng([seed, tick0, fx, fy])`: the same
  effect throws the same sparks.
- **No spark crosses a wall**: a spark at 18 tiles/s in a slow frame
  (`MAX_STEP_S` 0.25 s) would step 4.5 tiles over a one-tile wall in one
  `integrate` call, so `SparkSystem.update` substeps it to at most
  `SPARK_MAX_HOP` = 0.5 tile per step (the ember call is untouched); the
  wall test fails without the substeps.
- **One draw**: `GameRenderer._draw_embers` packs both systems and
  concatenates them into one sprite array, one `ParticleDraw.draw` per stage
  (scene under the units, compensation after the ceiling soot); a frame with
  no spark hands it the embers' own array. The curl field is the embers'
  (`EmberSystem.modes()`: one air, one drift), so the sparks have no curl
  wavelength / rate / modes dials of their own.
- **Dials beyond the brief** (`drag`, `curl`, `size_tiles`,
  `min_screen_px`, `streak_s`, `brightness`, `seed`: the names `integrate`
  and `pack_sprites` read; `size_px` became the embers' `size_tiles`).
  Claude's starting values, set from the shots: `drag` 1.0 (at 2.0 the
  sparks stayed inside the fireball's white glow at every tick they lived),
  `size_tiles` 0.06, `streak_s` 0.06, `brightness` 4 (a streak spreads its
  light over its length, the shader's `r / (r + seg)`; at 1 they did not read
  on the lit floor). `cap` 512: blast_final's volley throws 457 (frag 60,
  breach 19, magazine 170, C4 208).
- **What the shots show** (`p3/`, zoom 40 rows + `blast_final_t52`): at +2
  the frag's sparks are a tight star at its centre; at +4 they are a fan
  ~1.5 tiles out, visible only where the floor is dark (over the crate and
  its soot): on the lit floor and inside the fireball's display-space heat
  layer (G0's named exception, drawn after the resolve) they are covered. A
  debug pass at +10 (not a committed row) shows them as yellow streaks 3-4
  tiles out all round, none past a wall. The C4's and the magazine's are
  hidden by the heat layer at every tick their sparks live (the shock
  front's finding again). The breach on wood's 19 read as a crisp white
  starburst at +4. The frag's 60 read as sparse but right for a grenade;
  the count is Erik's.
- `look_shots.HIDEABLE_LAYERS` gains `sparks` (`SparkSystem.enabled_override`,
  set only by a shot's `hide`; no key).
- GPU: `lighting_demo --level blast_final --spark-perf 600` (new, the shock
  front's method: the volley's 457 sparks held at birth, the particle draw's
  two stages timed with `GL_TIME_ELAPSED`, sparks on vs off alternated):
  **0.051 ms** (0.053 on, 0.001 off; world RT 3600x3024). CPU: 0.44 ms on the
  spawn frame (four walks + 457 births).

### P4, the debris (2026-11-07)

- `renderer/particles.py::DebrisSystem` (+ `DebrisSettings`,
  `debris_settings_from_config`, `debris_count`, `debris_keys`,
  `debris_spawns`, `integrate_debris`, `debris_weight`, `pack_debris`),
  `[render.debris]` behind a refusing door (the sparks'), owned by
  `GameRenderer` beside the sparks and stepped in `upload_state` right after
  them; `look_shots.HIDEABLE_LAYERS` gains `debris` (`enabled_override`, set
  only by a shot's `hide`; no key). The particle primitive's THIRD consumer:
  `Particles` is extended additively (`add(temp=)`: a chip's `t0` column is
  its hashed rotation, `temp` its hashed size; every other caller is
  unchanged), `integrate` and `pack_sprites` are untouched, the ember and
  spark packs byte-identical (`tests/test_debris.py`, `test_particles.py`,
  `test_sparks.py`).
- **Consumption and births** as the sparks': each `"blast"` effect read once
  (`debris_seen`), `rate · strength` chips (linear, §3.4; the volley throws
  25 + 2 + 200 + 300, over `cap` 512, so a fair subset) at the walk's seed
  centres, the RNG `default_rng([seed, tick0, x, y])` (the sparks' key rule,
  the debris' own seed), the rotation and size from THE fire's hash
  (`fire_vfx_ref._hash01`, the vectorised twin of `fire_vfx._hash01`, bit for
  bit).
- **The motion law is Erik's, ruled at the screen (2026-11-07)**, which
  amends §3.4's "strong drag" and S5's "drag toward REST": in his words, a
  chip's motion is (1) drag toward the local air's motion, scaled by the
  tile's gas density, so in vacuum the air term is zero; plus (2) a ground
  friction term, density-independent, that brings a sliding chip to rest,
  so a draught never moves a chip lying on the floor but the blast's outflow
  carries it; (3) a chip in vacuum slides until a wall stops it -- it stops
  at the wall rather than dying: a chip is a thing on the floor.
  `integrate_debris`: `v += (air − v)(1 − e^{−drag·ρ·dt})` with `air` the
  frame's ONE tamed wind × tps and `ρ` the tile's bulk N over the ambient
  (`static_props.gas_frac_at_tiles`, the many-tile form of the prop sway's
  `sample_gas_frac`: the same accessor `GameMap.gas_bulk_n_at`, the same
  `ambient_seed()[0]`, asked once per distinct tile), clamped to [0, 2] (the
  props' `density_max`); then `|v| −= friction·dt`, never below 0; a step
  into a blocked tile (solid | vacuum, the embers' mask) or off the map
  restores the last position with velocity 0, alive. Substepped to half a
  tile (`DEBRIS_MAX_HOP`) on max(chip speed, air speed). A lying chip moves
  only in air faster than `friction / (drag·ρ)`. The sparks keep their law
  (giving them the density factor is a one-line follow-up, not done here).
- **Dials, Claude's starting values, set from the shots**: `drag` 2.5 /s and
  `friction_tiles_s2` 15, i.e. a threshold of 6 tiles/s. Measured on
  blast_final (headless, `tame_wind` × 24 over the frag's room): p90 7.0,
  max 12.1 tiles/s at +4; p90 3.5-4.7, max 5-6 at +8..+12; p90 < 3.3 from
  +16. At the brief's suggestion (drag 1.5, friction 6: threshold 4) and at
  1.0 / 8, the shots showed the frag's outermost chip still moving between
  +12 and +36 (max radius 4.6 → 6.8 tiles: the fast chips were still in
  flight at +12 and the settling air kept nudging them); at 2.5 / 15 it is
  3.7 → 4.1 and the median 2.2 → 2.1: lying. In still air the fastest chip
  (12.8 tiles/s) stops in ~0.46 s about 2.4 tiles out; in vacuum it slides
  `v²/2f`, ~2.1 tiles for a v0 chip. `size_tiles` is the half-width (0.08:
  ~3.8 world px); `fade_s` 0.8; `albedo` [0.08, 0.07, 0.06], `alpha` 0.9.
- **The draw**: `ParticleDraw.draw_debris`, the `"debris"` stage, its own
  program (`shaders/particle_debris.vs/.fs`) and data texture: a rotated
  quad per chip from `gl_VertexID` (the sprite's corner order and winding),
  a one-pixel antialiased edge, colour `albedo × (ambient + exposure ×
  light)` with the light read through the lit-3D block spliced in
  (`lit3d._COMMON_GLSL` + `_FIELD_SAMPLE_GLSL`: `light_tex_a/b`, the smoke
  cast shadow, the light curve; the context is the one
  `GameRenderer._light_field_ctx()` the 3D units and props now share),
  written premultiplied `(lit·a, a)` under `(ONE, ONE_MINUS_SRC_ALPHA)` on
  RGB and alpha. Drawn in `compose_world` after the floor smoke group and
  before the flame (3b), so the flame, the embers / sparks, the units and
  the ceiling soot are over it, and the heat layer (display space, after the
  resolve) over all of it.
- **What the shots show** (`p4/`; five brief rows + two new close-ups,
  `blast_final_t60_frag` / `_t84_frag`; numbers from a diff against the same
  shots with `hide = ["debris"]`): at +2 and +4 the frag's chips are
  invisible (0 and 0 changed pixels: inside the flame and the display-space
  heat layer, the sparks' finding); the full +4 shows only the breach's
  chips beside the wood (20 px). At +12 the frag's chips lie 0.6-3.7 tiles
  out (p10-max, 23 visible) all round the crate, the ones in the fireball's
  light lit orange; at +36 0.6-4.1 tiles, 19 visible, fading. In the frag
  close-ups a chip on the smoke-shadowed half of the floor reads 22-25
  (8-bit) against 89-124 on the lit half; chip / floor 0.36-0.86 (p10-p90)
  at +36. Whole map at +36: 233 chips visible, the magazine's scatter in the
  dark beside its soot, the C4's seen through its heat layer as orange
  flecks (the heat layer is added over them on screen). None is inside a
  wall (by construction: births at open seeds, a blocked step is undone;
  gated). The count, the throw and the albedo are Erik's.
- GPU: `lighting_demo --level blast_final --debris-perf 600` (new, the
  sparks' method: the volley's 512 chips held 12 ticks after birth, the
  debris stage timed with `GL_TIME_ELAPSED`, on vs off alternated):
  **0.012 ms** (budget 0.1; world RT 3600x3024). CPU 1.4 ms on a frame with
  512 chips (the per-tile density read in each substep).
