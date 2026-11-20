# Muzzle flame — design (2026-11-12, #87)

The fire that leaves a gun's barrel when it fires. Erik's brief: "extremely
cool", cinematic (every shot flashes visibly), a size dial per gun, "a little
bigger than one would expect", and a way to attach it to many guns later.
The room LIGHT of a shot already exists (`[render.muzzle_flash]`, #33) and is
not changed.

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| 1 | A PROCEDURAL flame drawn by a shader, not sprites and not a fluid sim | A flash is 1–2 m and lives ~55 ms (3 frames): a fluid has neither the time nor the pixels to show motion; 3–4 sprites repeat visibly at 12 rounds a second. A law evaluated per pixel from a per-shot seed never repeats and still reproduces in a look shot |
| 2 | Cinematic character (Erik, 2026-11-12) | Every shot flashes; white-hot core, jets, torn edges |
| 3 | Colour only through the blackbody ramp's GPU LUT | One T → colour map (the Blackbody GPU LUT row); the flash matches the fire |
| 4 | The halo is part of the flame | The resolve has no bloom; a bloom pass is its own patch |
| 5 | Additive linear HDR into the scene RT, after the units, under the ceiling soot, with the flame's compensation after the soot | The One tone-map row; the flash is at gun height, so it is drawn over the bodies |
| 6 | WHERE = a `muzzle` point on the held-prop row; WHAT = a kind row chosen by weapon id | Position is the prop's geometry, the look is the weapon's. A new gun is two table entries |
| 7 | Sparks are the particle primitive's fourth consumer | The Particle primitive row: consumers extend it |
| 8 | The flash ages on the SIM clock from the first frame it is drawn | Paused holds it; every shot shows its full first frame however a tick falls in a frame |

## 2. The law (`renderer/muzzle_flame.py`, the numpy twin = the shader's spec)

Local frame in METRES: x along the barrel, y across, origin at the muzzle.

- **Per shot, on the CPU** (`shot_rows`, a generator seeded by `(seed, unit id,
  tick)`): up to 5 jets `(angle, length, width, gain)` — the forward tongue
  and the side jets, each with its own length, width and angle draw — whether
  this shot is a FIREBALL (shorter, fatter, a bigger core), and the quad's
  half size. The kind's `size` scales every length.
- **Per pixel** (twin `flame_radiance`, shader `muzzle_flame.fs`): each jet is
  a tapered tongue (fat at ~20 % of its length, a pointed tip) whose centre
  line wobbles more with age; a streaked noise (the one fBm bake,
  `advected_noise.bake_fbm_rgba`, stretched along the jet and sliding outward)
  eats the body, harder with age, so the flash tears into shreds; a core ball
  at the muzzle; temperature from `t_edge_k` at the rim to `t_core_k` in the
  core, cooling with age; emission = the ramp's `Le(T)`; a soft warm halo.
- **Age** `a = (t_sim − t_first) / duration_s` in [0, 1), `t_first` the sim
  time of the first frame the flash is drawn.

## 3. Attachment

- `[render.held_props.props.<prop>] muzzle = [x, y, z]`: the barrel tip in the
  prop file's units. Its world place and direction come from the SAME matrix
  that draws the prop in the hand (`held_props.prop_world`), recorded per
  unit by the unit renderer each frame (`UnitModelRenderer.muzzle_poses`), so
  the flame sits on the barrel through walk, turn and recoil.
- The flame's DIRECTION is the round's own (the tracer's `from` → `to`), not
  the drawn barrel's: the first in-game look showed the aim pose holding the
  gun some degrees off the shot, and a flame that parts from its tracer reads
  as a bug. The barrel's direction is the fallback for an effect with no
  path.
- A unit with no drawn prop (sprites, pose layer off, no socket): the
  footprint's edge along its facing — the light's own rule.
- `[render.muzzle_flame] weapons = { <weapon id> = "<kind>" }`, `default_kind`
  for any other weapon that fires a round; `""` = no flame.
- `[render.muzzle_flame.kinds.<name>]`: the dials (size, lengths, jets,
  temperatures, tear, fireball chance, duration, sparks).

## 4. Sparks

Each new flash throws `kind.sparks` (± half) sparks from the muzzle in a
forward cone, through `particles.MuzzleSparkSystem` (a `SparkSystem` whose
spawns are queued by the flame instead of read from a blast): the same
`integrate`, wind, curl, cooling, walls, `pack_sprites` and draw.
`[render.muzzle_flame.sparks]` holds their dials.

## 5. Not in this patch

- The light is not tied to the flame's per-shot seed (a fireball shot does
  not light the room more).
- No lingering gun smoke; no bloom pass.
- Tracers still start where they did (#141); #85's event issue is unchanged
  (the flame dedupes on `(unit, tick)` and so shows one flash per shot).

## 6. Gates

- `tests/test_muzzle_flame.py`: the door's refusals; the twin's properties
  (dark outside its quad, fades and cools with age, `size` scales it,
  reproducible per seed, different per shot); the muzzle point's place; the
  system's lifecycle (one flash per unit and tick, capped, never outlives its
  duration, held while paused).
- `tests/gl_muzzle_flame_check.py` (+ wrapper): the shader equals the twin;
  no flash = no draw.
- HUMAN-TEST: Erik plays it before merge (feel-adjacent).

## 7. Systems

**(a) Existing systems used**: Blackbody GPU LUT (colour); Particle primitive
(sparks; the direct-GL sprite idiom); Held props (the muzzle point); One
tone-map (linear HDR, no tone-map of its own); Smoke cast product (the
compensation's cover read through the lit3d block); the fBm bake.

**(b) New system** — draft rule: *Muzzle flame — `renderer/muzzle_flame.py` +
`shaders/muzzle_flame.vs/.fs` + `[render.muzzle_flame]`: THE fire at a gun's
barrel. A new gun's flash is a `kinds` row and a `weapons` entry, its place
the held prop's `muzzle`; a new flash shape is a term of the one law (twin
first, then the shader), never a sprite sheet or a second flash pass.*
