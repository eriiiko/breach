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
  `amp_tiles` (0.25 at strength 1, times sqrt(strength)), `life_s` (0.5),
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
| P1 | event fields + ring deletion + the §3.6 measurement + before shots into the doc folder | tests; the measured table in the as-built | auto on green (nothing visible but the ring's absence; Erik sees it at P2) |
| P2 | shock front (`shockwave.py`, the shader term, dials, GL check, `blast_final_t50` shot) | tests + GL check + shots | HUMAN-TEST |
| P3 | sparks | tests + shots | HUMAN-TEST |
| P4 | debris (the second blend mode) | tests + shots | HUMAN-TEST |
| P5 | the flash: row columns + door, `flashbang` row + studio, `blast_flash_lights`, the caps, `flashbang_t50` shot | tests + shots | HUMAN-TEST |

One implementation agent at a time, Opus, each in this worktree's branch;
a critic on this doc before P1 (verdict to disk). P2–P5 are independent of
each other after P1; order as Erik wants to look.

## 6. Critique

Pending (one critic pass, written to `docs/explosion_look_design_2026-11-07/critique.md`).

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
