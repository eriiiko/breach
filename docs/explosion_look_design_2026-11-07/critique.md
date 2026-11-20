# Critique: explosion look design (G4, #75 / #94), 2026-11-07

Critic: Opus, read-only, against `60ef6a2`. Erik's §1 rulings, PLACEHOLDERs
and ACCEPTED GAPs are taken as given.

| lens | verdict | findings |
|---|---|---|
| 1 systems reuse | mostly sound; one canonical walk ignored, one seam bypassed | 3 |
| 2 correctness / failure modes | two blockers (budget, ring through walls) | 7 |
| 3 scope / regression | one blocker (P1's measurement is vacuous), one unbuilt dependency | 3 |
| 4 verification | properties, no snapshots; three gates missing | 4 |

Totals: 3 BLOCKER, 9 SHOULD, 5 NIT.

## BLOCKER

**B1 (lens 2/3). The flash budget leaves gen_ship's lamp budget out, so a lamp-heavy recipe ship plus a volley trips the sweep's door.**
gen_ship works out its lamp budget from the muzzle cap alone:
`LAMP_LIGHT_BUDGET = (CONE_LIGHT_UNITS - MUZZLE_FLASH_LIGHT/OMNI_SHARE - LAMP_RESERVE) * OMNI_SHARE` (`tools/gen_ship.py:144-148`), which is 768 units today.
- `tests/test_levelspec.py:320` pins `MUZZLE_FLASH_LIGHT == fl.MUZZLE_FLASH_MAX_TOTAL`, so §3.5's drop to 512 forces a gen_ship edit.
- That edit, done naively, raises the lamp budget to (128-32-16)*16 = 1280.
- Per ordinate: 1280/16 + 512/16 + 1024/16 + 16 (beams) = 192, against the 128-unit door (`radiation_sweep.h:161`, `frame_lights.py:84`). The door raises mid-frame. This is the #136 frame-3 incident again.
- §3.5's "leaving 512 for lamps" also forgets that `LAMP_RESERVE` (16/ordinate = 256 omni units) already sits inside that 512. The honest lamp budget is 256, down from 768. That partly undoes #173's "ships keep all their lamps, brighter".

The design must:
- add `BLAST_FLASH_MAX_TOTAL` to gen_ship's formula and to `test_levelspec.py:310-322` / `:326-345`;
- state whether the large freighter's lamps (§3.5 says "about 60") fit in 256 without dimming;
- if they don't fit, accept that regenerated ships change.

**B2 (lens 1/2). The ring is not occluded by walls. §3.3's "walls clip the ring, a corridor shows a front that follows the corridor" is false as specified.**
- The term is `g(|p-c| - r)` in straight-line distance, times a per-tile gas mask.
- The mask only zeroes pixels that are themselves in a wall. Every gas pixel at distance r gets the ring, including the room behind a 1-tile wall.
- A front in a bent corridor is a circular arc cut by the corridor, not a front following it.
- The repo already has THE walk for "what a blast touches": `blast_expansion.reach_tiles` (CLAUDE.md "Blast reach"; called at `src/simulation/payloads.py:442`). The design neither uses it (for example a reach mask per ring) nor states the leak as an ACCEPTED GAP for Erik's P2 judgement.
- No test in §4 would catch it.

**B3 (lens 3). The §3.6 measurement (b) "blast smoke off" changes nothing for every blast_final row, so #94 would close on a vacuous experiment.**
- All four blast_final rows (`tools/gen_blast_quad.py:84-89`) are `strength` rows (`config.toml` frag 1.0, breach 0.1, C4 12, magazine 8).
- The generic branch deposits its smoke whatever `emit_blast_smoke` says (`payloads.py:452-457`, gated only on `generic`).
- The strength door requires `emit_blast_smoke = true` anyway (`weapons.py:582-585`).
- So (a) and (b) are identical by construction.
- The "lighting_demo rule" cited does not exist: `tools/lighting_demo.py:835-840` copies the row and overrides radius and damage only.
- The probe must suppress the smoke another way, and add a check that (b)'s smoke plane really is zero (the vacuous-fixture pattern of CLAUDE.md's "Starting a fire" incidents).

## SHOULD

**S1 (lens 2). "Paused, it holds" is false for the flash.**
- `_advance_effects(dt)` ages every effect by the wall-clock frame dt, paused or not (`main.py:745-746`, `:895`).
- The paused relight re-assembles the cone rows whenever they move (`main.py:833-841`).
- So a flash decays and relights while paused, while the ring and the sparks follow `t_noise`. `t_noise` also advances while paused when `idle_when_paused` is on (`fire_vfx.py:246-258`), so "idles" means it keeps moving.
- The three pieces need one stated pause behaviour.

**S2 (lens 2/4). The flash clock makes the `flashbang_t50` look shot irreproducible.**
- `lighting_demo` ages effects by `perf_counter` dt even in `--auto` shots (`tools/lighting_demo.py:1912-1914`, `:2206`).
- At tick +2 a 1.5 s envelope's value therefore depends on machine load.
- The ring and the sparks on the VFX clock are deterministic; the flash is not.

**S3 (lens 2). Detonations in all but the last tick of a multi-tick frame never reach the renderer.**
- `tick_events.clear()` runs on each step (`src/simulation/simulation.py:1554`).
- main.py runs up to 5 catch-up steps per frame (`main.py:796-819`) and consumes events once, after them (`:894`).
- Blast frames are exactly the slow frames, so a flashbang or a ring can vanish.
- The bug predates this design, but the design hangs three systems on the event. Name it as a gap or have the loop collect per step (`follow.observe` already reads per step, `:813`).

**S4 (lens 1). The gas mask's update path bypasses the wall seam.**
- `render_dirty` is a set that `TileArtSeam` drains in `upload_state` (CLAUDE.md "Wall-destruction render seam"). A second reader of `render_dirty` will usually see it empty.
- Re-upload from the seam's drained patch list, or CRC-key the mask per sim tick as the shimmer does (`renderer/heat_shimmer.py:224-236`).

**S5 (lens 2). `integrate` and `pack_sprites` are ember-shaped. Debris will not survive them unchanged.**
- `integrate` kills every particle at `TEMP <= glow_min_k` (`renderer/particles.py:376`), and a cold chip dies on its first step.
- `pack_sprites` packs T plus brightness only (`:417-421`).
- `drag` pulls a particle toward the tamed wind × tps (`:373`), not toward rest. Chips in a blast's outflow drift instead of "lying there".
- §3.4 should name the primitive's extension (a per-kind kill rule, a rest target, a third data row for rotation and albedo) instead of "additively".

**S6 (lens 3). P1's gate (c) "with the row's starting flash" needs `blast_flash_lights`, which is P5.**
- The flash is a render-side cone row. It does not exist headless until P5 builds it.
- Move (c) to P5, or build the pure function in P1.

**S7 (lens 2). The flashbang PLACEHOLDER (1500) can never be reached.**
- `BLAST_FLASH_MAX_TOTAL = 1024` scales any single flash above it, so a lone flashbang always lands at 1024.
- Decide whether the cap or the row is meant (a lone flashbang plus muzzle 512 = 1536 omni, which fits the 2048 door only if the lamps get the rest; see B1).

**S8 (lens 4). The GL check only proves the zero case.**
- `gl_shockwave_check` asserts bit-identity where the mask is zero. Nothing holds the shader's non-zero offset to the twin `ring_offset_px`.
- The resolve flips y (`shaders/world_resolve.fs:131`, `vec2(d.x, -d.y)`), so a sign or flip error would pass every listed gate.
- G3 practice is GL-vs-twin per pass. Also assert zero for gas pixels outside the band (the `d == vec2(0)` path at `:129`).

**S9 (lens 4). The payload door does not refuse unknown keys.**
- `col(c, default)` reads with defaults (`weapons.py:687-689`), so `flash_decay = 1.5` (a typo) silently gives 0.3, and `flsah` gives no flash.
- §3.5 lists only range refusals. Add a key-presence check for the `flash*` family, or say it is an accepted gap.

## NIT

- N1: the method is `consume_events`, not `_convert_events` (`renderer/game_renderer.py:1645`). The new `"blast"` effect needs the `fresh` flag that tracers use (`:1672`, `:1722-1735`), or a slow first frame ages the flash before any tick lights it. §7's "the legacy queue no longer carries explosions" contradicts the `"blast"` entry that §3.5 reads from that queue.
- N2: `ExplosionEvent.pos` is the integer tile `(fx, fy)` (`payloads.py:486`). The ring centre, spark origin and flash cell need `+0.5`, as the legacy ring did (`game_renderer.py:1488`). The new fields need defaults equal to what an unflashed row fills, because `tests/test_payloads.py:174` compares `repr(ev_a) == repr(ev_b)` against a replica that builds the event with three arguments (`:115`). `flash_color` should be a tuple in both places.
- N3: the ring's "its bbox" (the death test) is undefined. `A_max` is in the clamp but not among the dials. `unit(p-c)` is undefined at `p == c`. `g` must have compact support (a Gaussian breaks both "exactly zero" and the bit-identical resolve).
- N4: the flash source cell set when the centre is solid (a breach on intact steel) is unstated. It is unclear whether the 1/5 share is lost in the wall or re-split. Sparks born "at the centre's open tiles" need the same rule, or `blocked` kills them all on the first step (`particles.py:378-379`).
- N5: the debris stage's slot relative to the flame and the embers (`game_renderer.py` compose steps 4-4b, ~`:905-926`) is unstated. Premultiplied dark chips drawn after the flame would darken it. Lit debris must multiply by the one exposure `u_light_gain`, and should splice lit3d's `_FIELD_SAMPLE_GLSL` (Lit-3D seam rule), not re-read `light_tex_a/b` by hand. "Draw blend factors asserted" must run in a mocked rl or a `gl_*_check` (no GL context in pytest). `cone_rows` itself enforces no budget, so `test_blast_flash`'s "pass cone_rows at their caps" must evaluate the sweep door (`RadiationSweep.cone_emission` sum, as `test_levelspec.py` does).

## What holds

- The ring as a distortion-block term, the particle consumers and the flash through `frame_light_specs`/`cone_rows` all reuse the canonical systems correctly.
- The event field is outside every digest: `field_ab_harness._event_record` hashes only UnitHit/Killed (`tests/field_ab_harness.py:238-269`).
- Replay re-simulates intents, so the recording format is untouched.
- No sim field, golden or C++ TU moves.
- The §2.2 measurement does justify a scripted shape over a plane-derived one.
