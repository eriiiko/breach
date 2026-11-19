# Swarm units P4 -- the brood entity and the dot view (as built, 2026-11-17)

Arc #63, patch P4 (`docs/swarm_units_patch_plan_2026-09-10.md`, the P4 row;
`docs/architecture/engine/17_swarm_units.md` sections 7, 9, 11). Branch
`63-p4-brood` off `63-swarm-units` at `382733f7`. Rules: `.claude/rules/swarm.md`
(Brood entity, Brood spawn, Swarm dot view, and the salt-regeneration line).

## What was built

### The brood entity
`swarm_brood` (`src/simulation/entities/zones.py`), a third zone class in the
one paint-id namespace (`level_loader.ZONE_CLASSES`), so the map editor's zone
paint and the binding validators take it unchanged. Fields: `zone_id`
(required), `species` (`KIND_STR`, default `"larva"`, validated at load
against `swarm.SWARM_SPECIES` -- a string, not an enum, so the import-light
entity package does not copy the species set), `count` (default 200,
minimum 1). No seed column. It never ticks: `EntityExport` takes its tail
once at build (the "no device table" branch); the export gate
(`tests/test_hazard_tables.py::test_the_export_from_the_tables_is_the_serializers_bytes`)
now also drives `larva_studio` (two broods and a heat source).

### The brood spawn -- the one seam
`src/simulation/swarm_brood.py::spawn_broods`, called once at the end of the
tick-state block of `Simulation._reset_internal` (after the entity build, so
a closed door is solid, and after `self.tick = 0`). Every match path
constructs a `Simulation` there: the played game (`main.py`),
`tools/run_match.py` and `MatchSession` (`MatchSession.__init__` builds the
sim), and each batched env (`rl_env.EnvRunner.reset` builds a `MatchSession`
per episode). No second site.

Draws: one new salt, `Salt.SWARM_BROOD` ("BROD"). Unit k of the brood with
ordinal o takes the draws `(tick 0, index 4k + j, BROD, o)`, j = tile, x in
the tile, y in the tile, heading. Tile = `below(w, n_free)` over the zone's
free tiles in row-major order; position in the tile = the word's top 16 bits
(so `[t, t + 1)`); heading = an angle draw, then `wrap_heading_q16`. The words
come from a new `MatchDraws.words_at(salt, ident, indices)`: the same blocks
`at()` gives, formed in one vectorised Philox pass (`philox32_4x32_10_np`) --
600 larvae cost 2 400 blocks, which one by one in Python would be ~70 ms per
reset. `cpp/src/unit_draws_layout.h` regenerated for the new salt.

Refusals, all before the first write: `BroodRefused` (a zone with no free
tile, an unknown species; both also printed by the level lint,
`tools/level_airtight.py`, from the same `brood_problems` list) and
`SwarmFull` (the broods' total per species over the free slots).

### The dot view
`renderer/swarm_dots.py` + `shaders/swarm_dots.vs` / `.fs`, switched by the new
`[debug] swarm_dots = true` (read every frame; Ctrl+R live). Built as briefed:
dot radius 0.12 tile, tick 0.28 tile (half-width 0.025 tile, never under
0.6 px), bone white `(232, 220, 184)` crawling (RUN and EAT), blue-grey
`(120, 150, 185)` and no tick in COMA, dark red-brown `(110, 40, 30)` death
flecks fading linearly over 1.5 s of SIM time, newest 2048 kept. NO amber: the
larva law has no alarmed / tumbling state (RUN, EAT, COMA only). Colours are
converted sRGB -> linear once; drawn unlit, premultiplied, after the water
pass and before the floor smoke layer (step 2b of `compose_world`), in one
direct-GL draw from an RGBA32F data texture (the particle draw's
`gl_VertexID` idiom; not a particle consumer: nothing moves or cools).
The sprites are packed in `upload_state` by numpy in one pass; the flecks are
fed by `consume_events` from `SwarmUnitDiedEvent`.

The tile inspector gains `larvae on this tile: n` through
`swarm.count_on_tile` (a pure read of the host mirror over `high_water`).

### Frame cost at 8192 larvae
`tools/e2e_drive.py --level <studio with 2 x 4096> --control direct --profile
--glfinish`, 400 frames, Home Desktop: `_upload_swarm_dots` 0.54 ms/frame,
`_draw_swarm_dots` 0.46 ms/frame (GPU included by glFinish): **1.0 ms per
frame**, at the brief's line. The frame held 60 fps (16.7 ms mean). Side
note: the recorder truncates the swarm ring at `swarm_units_cap` 1024
(existing behaviour, it says so).

### The larva studio
`tools/gen_larva_studio.py` -> `levels/larva_studio` (80 x 36, 0.333 m,
through level_lib; deterministic):
- the NURSERY (west, 26 x 22) with two broods of 300 at room temperature;
- a 14-tile DOORWAY into the WARM ROOM (east, 26 x 22), its air seeded by
  `temp_init.npy` as a ramp from ambient at its west wall to +14 game degrees
  (34 C) at its east wall, and held by a `heat_source` in its north-east corner
  (`heat_per_s` 20, `radius` 0, `max_temp` 12 = 32 C);
- a furniture FUEL stack against the east wall beside the source (dev key I);
- one marine `[[spawn]]` in the nursery's south-west corner; four lamps.
- NO COLD ROOM: no honest way exists. `temp_init.npy` refuses sub-ambient (a
  parked physics ruling), the ambient temperature is global
  (`[physics.temperature_scale] eos_t_amb_k`), and nothing cools a room.

A first version had `heat_per_s` 40 with `radius` 3: the corner passed 45 C
(the cap reads the source tile only, and #72's checkerboard keeps that tile
under the cap half the time), so the source was cut to its own tile.

### The first look
`docs/swarm_P4_first_look.png`: `tools/e2e_drive.py --level larva_studio
--control direct --press SPACE@5 --press F6@6 --wheel 2@3 --frames 1000`, the
frame at tick ~404 (17 s). The camera is centred on the marine in the
nursery's south-west corner: the nursery floor is covered with ~600 bone-white
dots, each with its short heading tick pointing every which way (a crawling
mass, the two brood rectangles already dissolved into one spread), thinning
toward the doorway on the right edge; a handful of blue-grey tickless dots
(coma) sit by the marine's feet in a #72 cold cell. No fleck yet (no death by
17 s). The warm room is off-screen to the right.

### Measured in the studio (CPU, seed 1)
Warm-room larvae (of 600): 5 at 20 s, 28 at 60 s, 54 at 120 s, 104 at 300 s;
the first arrive within ~20 s. Deaths: 12 by 300 s, comas 0 to 19 at a time.
So larvae DO reach the warm room within a minute, but the bulk drifts slowly:
the nursery has no gradient (they random-walk until they meet the warm air),
and the temperature field carries #72's checkerboard (below). `speed_mps`
stays 0.2: the limit is the missing gradient and the sensing noise, not the
speed. No value proposed.

### Finding in the reference: #72 decides what the larvae feel
Any non-uniform temperature in the studio (the warm seed alone, or the heat
source alone) grows a +-10 K tile-scale CHECKERBOARD over the whole connected
air within ~7 s (a uniform studio stays clean). It is open issue #72 (EOS
second law + checkerboard; the 2026-09-25 triage already noted "#63's
grid-scale T pattern is #72's checkerboard"). For the larvae it means: cells
below 10 C appear in a 20 C room (the comas seen above are these), the
south-east corner below the fuel stack becomes a persistent cold pocket, and
the head sweep reads noise as large as the warm room's gradient. Not fixed
here (out of scope); it is the first thing Erik will see in the studio.

## Numbers set here
- A larva is ~6 cm long (a fifth of a tile): written as the reason on
  `[swarm.larva] speed_mps` (0.2 m/s ~ 3 body lengths a second). Not a
  species row: nothing reads a body length yet.
- Brood `count` default 200 (`zones.BROOD_DEFAULT_COUNT`).
- Still `PLACEHOLDER (P4)` in `[swarm.larva]`: `tumble_rate_base` 0.1,
  `tumble_rate_alarm` 1.0, `turn_bias` 0.8, `turn_min` 30 deg, `turn_max`
  120 deg, `sweep_angle` 45 deg (`hp_max` 1.0 is its own placeholder).

## Tests (each names the property and the change that breaks it)
- `tests/test_swarm_brood.py`: the round trip (registry, writer, loader,
  serializer); count / zone / free tile / in-tile / heading / fill order; the
  seed alone decides (and a reset); an unrelated entity after the broods moves
  nothing; two ordinals never share a key; `words_at` == `at().word()`; the
  three refusals by name with nothing spawned (+ the lint); the lint passes a
  good brood; a brood-free level draws and stores nothing; a batched env's
  reset spawns from its own seed, equal to a lone MatchSession, and re-spawns
  at the next episode; the inspector's count.
- `tests/test_swarm_dots.py`: one sprite per live unit, colours by state, no
  coma tick; the tick follows the heading; the fleck fade, colour and cap;
  the builder writes nothing; nothing under `src/` reads the view; tick
  digests equal with the view packed every tick and never packed.
- `tests/test_larva_studio.py`: byte-deterministic and equal to the
  checked-in level; the generator's output passes the lint; after a headless
  minute at least 3 larvae are in the warm room within 5 K of T_prefer.
- `tests/test_hazard_tables.py`: the export test also drives `larva_studio`.

## Breaks (a driver that fails when its pattern is absent, then restores)
All 20 went red; two only after the test was improved:
key drops the ordinal; key reads the entity count; position spills out of
its tile; free tiles include walls; fill order reversed; capacity checked
after a write; the scatter reads a shared seed (batched env); the lint
ignores the broods; `words_at` forms another counter; a brood-free level
draws; the inspector counts empty slots (GREEN first: a freed slot sits at
tile (0, 0); the test now asserts (0, 0) stays 0 after a reclaim); coma draws
a tick; the tick off the heading; the fleck fades over 2 s; the fleck cap
keeps the oldest; the builder writes the store; a sim module names the view;
the studio edited without regenerating; a studio brood painted on a wall
(GREEN first: the lint test read the checked-in level; it now lints the
generator's own output); the warm room cold (seed and source off).

## Left out
The bite, corpses, food, bodies, any larva-unit interaction, any change to
the law or the physics (brief). The look-shot set was not re-rendered: no
shot level holds a swarm, so the view draws nothing there (the still above is
the e2e driver's).
