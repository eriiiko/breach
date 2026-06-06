# Ray Engine

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md), [World State & Ownership](02_state_and_ownership.md), [Material System](03_material_system.md)

One DDA ray-march primitive serves every directional energy query in Breach — light,
heat, vision, and energy weapons. The core is a **deposit-only accumulator over a
read-only world**: rays walk outward from sources, deposit energy into per-tile buffers
as they go, attenuate per material as they cross each tile, and die when their energy is
spent. There is no recursion, no per-ray branching, and nothing the kernel writes except
the accumulation buffers. That shape is deliberate — it is the shape that ports to one
GPU thread per ray unchanged.

## Why the ray engine is simulation, not rendering

In most games light is purely cosmetic, so the renderer can own it and the simulation
need never know it exists. In Breach light is **physical**: a ray carries heat that raises
tile temperature, ignites wood, melts hull, and drives the firestorm. The moment light has
gameplay consequences, *computing* light becomes a simulation task — it must run headless
(for training and replay), and its heat output must be deterministic and cross-machine
reproducible.

This fixes the dependency direction. The ray engine produces a gameplay field (`heat`) and,
as a byproduct of the same march, the visual fields the renderer reads (`light_rgb`,
`light_dir`, `smoke_glow`). The renderer is a downstream pure consumer of those buffers.
**Render reads sim; sim never reads render.** The boundary between the two is the buffer set —
not a function call, not a shared object, just four arrays on `GameMap`.

> **Implementation reality:** the cast currently runs in `renderer/lighting.py`
> (`LightingPass.compute_light_field`, invoked from `game_renderer.upload_state`). Moving it
> into the deterministic sim step is a planned, mechanical relocation — the C++ kernel and the
> buffer contract are already in their final form, so the move is a caller change, not a
> redesign. See *Implementation status*.

## The ray and its payload

A ray carries an RGB light triple plus a scalar heat multiplier, with an origin, a
direction, a range, and a falloff profile. Light is **RGB**; heat rides the *same* ray as a
scalar channel. This unifies every emitter into one type:

| Emitter | RGB | heat |
|---------|-----|------|
| Lamp / point light | warm white | 0 |
| Fire | orange | > 0 |
| Invisible heater / radiator | 0 | > 0 |
| Energy-weapon glow | beam colour | 0 (damage rides the weapon pre-phase, below) |

Because attenuation is per channel (see below), one ray naturally gives light and heat
different effective ranges — heat can punch deeper than the visible glow, or vice versa,
from a single march.

### Source definition

A source is a `LightSource` (`cpp/src/raycaster.h`): position in tile coordinates,
`max_range` (tiles; one tile = 1/3 m), `ray_count` (0 = auto), `angle_center` /
`angle_spread` (an omni light is `spread = 2π`; a cone is narrower), `intensity`,
`heat`, `jitter`, an RGB `color`, and a `Falloff` (`UNIFORM`, `COSINE`, `SHARP`).

`ray_count = 0` auto-computes enough rays that adjacent ray endpoints land roughly one
tile apart at `max_range`, scaled down for cones so a narrow flashlight does not pay for a
full circle of rays:

```
full_circle = ceil(2π · max_range)
fraction    = angle_spread / 2π
ray_count   = max(8, ceil(full_circle · fraction))
```

A 40° flashlight at range 30 costs ~22 rays. An omni lamp at range 20 costs ~126. The cost
is linear in (ray_count × range), which is what makes a roomful of lamps and a dozen fires
affordable on the CPU today and trivially parallel on the GPU later.

**Source profiles** are a thin convention, not engine machinery: callers build a
`LightSource` and override the fields they care about. The intended default profiles are:

| Profile | range | spread | intensity | color | heat | jitter | falloff |
|---------|-------|--------|-----------|-------|------|--------|---------|
| point   | 20 | 2π | 1.0 | warm white | 0 | 0 | uniform |
| fire    | 15 | 2π | 0.8 | orange | 1.0 | ~0.05 | uniform |
| flashlight | 30 | ~0.7 | 1.5 | cool white | 0 | 0 | cosine |
| energy weapon | 40 | ~0.14 | 3.0 | cyan | 2.0 | 0 | sharp |
| muzzle flash | 12 | 2π | 5.0 | yellow-white | 0 | ~0.02 | uniform |
| emergency | 12 | 2π | 0.4 | red | 0 | ~0.01 | uniform |

These are defaults to copy and tweak, not enum cases the kernel knows about.

## Output buffers (owned by GameMap)

The march **accumulates every source** into a small, fixed-size, **summed** buffer set.
No per-source or per-direction breakdown ever crosses to the renderer — the set is the same
size whether one lamp or a hundred fires are burning. This summed set is the entire
sim→render interface.

| Buffer (`gmap.*`) | Shape | Type | Nature | Determinism | Headless |
|-------------------|-------|------|--------|-------------|----------|
| `light_rgb` | (h, w, 3) | f32 → RGBA16F | render byproduct | float | skipped |
| `light_dir` (dx, dy) | 2 × (h, w) | f32 → RGBA16F | render byproduct (normal maps) | float | skipped |
| `smoke_glow` | (h, w, 3) | f32 → RGBA16F | render byproduct (god-rays, RGB) | float | skipped |
| `heat` | (h, w) | **Q16.16 int32** | **simulation** | int saturating-add | **computed** |

Two design rules govern these:

- **`heat` is the only sim-affecting output.** Everything else is render-only and feeds no
  gameplay threshold, so float is correct for it — fixed-point would band the near-dark
  gradient at ambient (~0.01) and fight the HDR sum the tone-mapper expects.

- **No damage channel.** Damage is *derived*, never deposited. Walls fail through
  `heat → temperature → thermal-failure` ([Temperature & Heat](temperature.md)); units
  sample `heat` at their footprint tiles and take damage in serial CPU logic. This keeps the
  kernel deposit-only and inherits determinism from `heat` being integer. (Kinetic/bullet
  damage rides the projectile-entity path, not rays — rays are energy/thermal only.)

### Fixed-point heat

`heat` is a **Q16.16 fixed-point int32**: 16 integer bits, 16 fractional bits, so one unit
of heat energy is `HEAT_SCALE = 65536` raw counts. The deposit is quantized (round-to-nearest)
and added with a **saturating add** — it clamps at `INT32_MAX` and never wraps, so a
firestorm depositing many sources into one cell can never roll past the ignition threshold
into a cold value.

The reason is determinism. Many rays deposit into the same cell; integer `+=` is
order-independent, so the result is identical regardless of the order the rays are summed —
the property that lets the deposit become a CUDA `atomicAdd` later and stay bit-exact across
machines. Float would be order-dependent and could diverge in lockstep multiplayer or replay.
Because `heat` is integer, the damage *derived* from it is deterministic for free.

The render channels accumulate in f32 (the renderer down-converts to RGBA16F at pack time —
see [Lighting & Render](lighting.md)) and are exempt: their non-determinism is harmless with
no downstream threshold.

## The march

A DDA tile walk from the source outward. At each tile:

1. **Deposit** the per-channel surviving energy, scaled by a distance falloff
   `1 / (1 + d²·0.01)`, into `light_rgb`.
2. **Accumulate direction** into `light_dir`, weighted by the aggregate deposited intensity
   and pointing *toward the source* (`-dx, -dy` of travel). It is normalized to a unit vector
   in a separate full-grid pass after all sources are cast (see below).
3. **Deposit heat** (only if the source emits heat): `heat_quantize(dep_aggregate · src.heat)`,
   saturating-added into `heat`.
4. **Attenuate per channel** by the material's static attenuation, then by live smoke.
5. **Deposit god-ray glow** — the energy the smoke just absorbed goes into `smoke_glow`.
6. **Step** to the next tile; terminate when range is exceeded or aggregate energy drops below
   a small epsilon.

### Occlusion is attenuation, not `is_wall`

The march reads the **material attenuation field**, never the binary `is_wall`. Each tile
carries a per-channel RGB attenuation coefficient from the material table
([Material System](materials.md)). As the ray crosses a tile, each channel is multiplied by
`(1 − atten[c])`:

- opaque hull / door → `[1,1,1]` → every channel driven to 0 → ray dies (exactly the old
  hard wall-stop)
- air → `[0,0,0]` → ray passes untouched
- glass → `[0.1, …]` → ray transmits, dimmed
- a tinted window → an asymmetric triple → "blocks two of three colours" for free, tinting the
  survivor

This is the single most important correctness fix in the engine. `is_wall` is occlusion for
*collision, smoke, and pathfinding*; rays need *optical* occlusion, and the two disagree on
glass — glass is a wall you cannot walk through but light passes. Driving the ray march off
the attenuation field resolves that cleanly, and it subsumes the old `block_light` bool: an
opaque tile is just attenuation `1.0`.

The ray terminates on the **aggregate** remaining energy (max over channels), not per channel.
A per-channel early-out would create warp divergence on the GPU; marching all three channels
in lockstep to one aggregate range keeps the kernel uniform. The cost is small — an opaque
tile zeroes all channels at once, so a wall still stops the ray the same step.

### God-rays via smoke_glow

When a ray crosses smoke, the smoke absorbs a fraction of the surviving energy. That
*absorbed* energy is deposited into `smoke_glow`, per channel. This is god-rays by
construction: it is exactly the energy the march removes from the ray, so it is
energy-conserving (no double-count), and it is RGB, so a red beam casts a red shaft. Because
the ray deposits nothing past opaque tiles, shafts terminate at walls automatically. This
replaces the old surface-tint glow hack — there is one mechanism, and it lives in the march.

### Direction field

`light_dir` is accumulated as a sum of per-deposit vectors weighted by deposited intensity,
then normalized to unit length in one post-pass over the grid (`normalize_directions`).
Tiles where opposing rays cancel, or where no ray arrives, end at `(0,0)` and the shader must
handle that. The result is a single **dominant-direction approximation**: it is correct for
one strong light and averages toward flat relief when lights oppose. That is an accepted
limitation of summing into a fixed-size buffer — recovering per-light coloured relief would
require storing every source, which the summed contract deliberately refuses. Normalization
is by **vector magnitude**, not by accumulated intensity (an earlier intensity-normalization
was a bug — it scaled the unit vector by brightness).

## Units and rays

- **Units occlude as read-only obstacles.** Each tick `stamp_units` rebuilds a *dynamic*
  attenuation field, `gmap.dyn_light_atten`: a copy of the static material attenuation,
  combined per channel via `max` with each living unit's opacity stamped over its footprint
  (default `[1,1,1]` = a full shadow). The march reads this field; it never writes units. An
  occluder can only *add* opacity, never remove it. Because the field is RGB, a unit can occlude
  per colour — a creature transparent to green light is just a per-channel opacity value.

- **Units receive by sampling.** After the pass, units read `heat` at their footprint tiles
  (reduction = max over occupied tiles) and apply damage in serial, id-ordered CPU logic. The
  kernel never touches the unit list.

### Energy weapons are a weapon pre-phase

Lasers do **not** run inside the lighting kernel. A laser is a single ray resolved in a CPU
**pre-phase before the lighting pass**, reading the CPU-authoritative `material` field only:
it resolves hits along its path (skewering multiple enemies, since a beam passes through
bodies), deposits weapon heat, and lets walls breach through the normal thermal-failure
reaction — a strong beam spikes a tile past its failure threshold the *same* tick. The map
mutation (damage, destruction, heat) lands first; *then* the lighting/heat raycaster runs on
the frozen, updated map, so a laser's breach lights the room it just opened on the same tick.

The beam's **glow is ordinary light** — emit it as a normal transient source and the lighting
pass deposits it as usual. The weapon beam shares only the DDA march *primitive* with lighting;
it carries damage and mutates the world, lighting deposits and reads only. Principle: the DDA
march is a shared primitive with two distinct consumers — lighting (deposit, read-only) and
weapons (mutate, serial pre-pass).

## The three ray passes

1. **Emitter lighting** — lamps, last tick's fire, this tick's weapon glow cast rays and
   deposit the summed buffers. Fire sources read *last* tick's fire state to break the
   fire → heat → fire feedback loop within a tick.

2. **Line of sight** — LoS is a pairwise query behind a `has_los(a, b)` interface: a ray from
   observer to target, blocked by occlusion. The principled rule is "you have LoS if any light
   gets through," backed by the same attenuation march — so you see and shoot through glass and
   dimly through smoke, but not through opaque walls. **Infravision is the identical query on
   the heat channel.** The interface lets faster backings (PVS, hierarchical) slot in later.
   The v1 backing is a binary Bresenham walk on `is_wall`; migrating it to the attenuation-aware
   check is a small, isolated change.

3. **Entity re-emission (deferred)** — prisms, mirrors, and refractive glass read the light
   field and **spawn new rays** in a second pass. There is deliberately no in-kernel
   recursion or ray forking: a reflection is a new source emitted by an entity, not a recursive
   `march_ray` call. This honours the CUDA contract (one thread per ray, no forking) and gives
   reflections a clean home as a future chapter. Straight-through dimming and tinting are
   attenuation; direction-changing optics are re-emission.

## Tick order (target)

When the cast moves into the sim, the deterministic step orders rays like this:

1. Advance projectiles
2. Unit movement + facing (player + AI)
3. Register shooting intent (this tick's hitscan/laser shots)
4. AI decisions
5. **`stamp_units`** → rebuild `obstacles` (wave/smoke) **and** `dyn_light_atten` (rays) in one pass
6. **Weapon pre-phase** (CPU, `material`-only): resolve beams → skewer units, deposit weapon heat
7. **Physics** (frozen, updated map): atmosphere/wind → smoke advection → **raycaster**
   (deposit `light_rgb`/`light_dir`/`heat`/`smoke_glow`) → temperature relaxation (reads `heat`
   non-destructively) → fire spread
8. **Field reactions:** ignition, wall thermal-failure (`wall_hp` → `destroy_wall`), smoke burn-off
9. **Unit damage:** sample `heat` at footprints, apply (serial)
10. Cleanup: deaths, events, pending `on_tile_changed` patches, **clear the `heat` deposit buffer**

`heat` is a per-tick *deposit* buffer, read non-destructively by both temperature (7) and unit
damage (9), then cleared at cleanup (10) — so unit damage is independent of conduction order.
The invariant: headless runs the identical ordering over the identical in-process arrays (there
is no "frame"), so training and rendered runs produce identical trajectories, enforced by the
determinism regression test.

## CUDA contract

Every choice above is shaped to port to the GPU unchanged:

- one thread per ray
- read-only world (material table bound as a constant/texture, indexed by the int8 material id —
  no random-access stalls)
- `atomicAdd` deposits — render channels via f32 atomics down-converted at pack time; **heat via
  integer atomics** (order-independent, the determinism guarantee)
- no in-kernel forking or recursion; bounded ray length
- entities sample buffers, never participate in the kernel
- march all channels in lockstep to the aggregate range (no per-channel early-out → no warp
  divergence)

Per-tick ray-list construction (host list vs device launch grid) and near-source atomic
contention are CUDA-phase tuning items, not contract changes.

## Lightning arcs (design, unbuilt)

Electrical arcs are a separate, event-driven effect that shares nothing with the lighting
march except a destination point — they are spawned by game events (damaged electronics,
energy-weapon impacts on metal, exposed wiring, a future water-conduction hazard), not run by a
continuous system.

`spawn_arc(origin, energy, radius, gmap)` selects a target by *path of least resistance* —
nearest metal first, then water (future), then a unit, falling back to a random nearby air tile
for a wild spark — then draws the bolt with **recursive midpoint displacement** (the classic
jagged-bolt algorithm), recomputed each frame for a live flicker. Energy sets damage, brightness,
and visual thickness. This has no newer design home and is recorded here as the canonical
lightning spec; it is not yet implemented.

## Implementation status

**Built and shipping (Tier 1):**

- `cpp/src/raycaster.{h,cpp}` — the full directional march. `LightSource` carries RGB `color`,
  `heat`, `jitter`, falloff. `cast_source_directional` / `march_ray_directional` deposit
  `light_rgb`, `light_dir` (dx/dy), Q16.16 `heat` (quantized, saturating-add), and RGB
  `smoke_glow`. Occlusion reads the per-channel `light_atten` field; aggregate-energy
  termination; god-rays from smoke-absorbed energy; `normalize_directions` post-pass
  (vector-magnitude).
- `cpp/src/bindings.cpp` — `LightSource` and `Raycaster` exposed to Python; `heat` /
  `smoke_glow` are optional (`None` skips that deposit), so render-only callers pass only what
  they need.
- `src/simulation/gamemap.py` — owns `light_rgb`, `light_atten` (static, table-derived),
  `dyn_light_atten` (rebuilt each tick in `stamp_units`), `heat` (int32 Q16.16), `smoke_glow`,
  `conductivity`. Buffers are filled in place, never reassigned, so C++ views stay valid.
  `is_wall` is the occlusion mask derived from `light_atten`; walkability is the separate
  `is_passable` predicate.
- `src/simulation/materials.py` — material table carries the per-channel `light_atten` column
  and `occludes()`; `conductivity` / `ignition_temp` columns present.
- `shaders/lighting.fs` — consumes the two packed RGBA16F textures: `albedo × incoming_RGB`,
  normal-map `N·L` with the `u_light_z` fake out-of-plane term, ACES filmic tone-map, sRGB
  encode, vacuum discard.
- `renderer/lighting.py` — casts all sources, normalizes, derives the legacy scalar `light_map`
  (max over channels) for render-side tinting consumers, packs into two RGBA16F textures.
- Tests: `tests/test_rgb_light_atten.py`, `test_dyn_light_atten.py`, `test_heat_smoke_glow.py`,
  `test_rgb_light_pack.py` cover the per-channel attenuation, unit-shadow stamping, heat +
  god-ray deposit, and pack contract.

**Designed but not yet built:**

- **Raycaster relocation into the sim.** The cast still runs in `renderer/lighting.py` from
  `game_renderer.upload_state`; `src/simulation/simulation.py` does not yet call it. The kernel
  and buffer contract are final, so this is a caller move plus extending the determinism test to
  the heat/temperature fields — not a redesign.
- **Heat consumer.** `heat` is deposited but nothing reads it yet; the temperature pass
  ([Temperature & Heat](temperature.md)) and unit-damage sampling are the consumers.
- **Energy-weapon pre-phase** — no laser pre-phase exists; weapons do not yet mark the map or
  deposit heat.
- **Attenuation-aware LoS** — `gamemap.has_los` is still binary Bresenham on `is_wall`; the
  `has_los` interface and infravision-on-heat are designed, not wired.
- **Headless skip** — the conditional allocation that drops the render-only buffers in a headless
  run is designed but not yet implemented (today the renderer owns the cast).
- **Lightning arcs** — design only (above).
- **Entity re-emission** (prisms/mirrors) and **hot-tile emission** (glowing tiles as sources)
  — deferred to their own chapters; the attenuation-only march and the deposit-only contract are
  built to accommodate them without kernel changes.

**Gaps / known limitations:**

- The march has deterministic per-source angular jitter seeded from source position (used by
  fire flicker). Once `heat` is sim-affecting and the cast is in the sim, this jitter must draw
  from `sim.rng` (or be removed) to keep the heat field replay-deterministic.
- `light_dir` is a single dominant-direction aggregate; opposing lights average toward flat
  normal-map relief and per-light coloured relief is not recoverable. Accepted, not a bug.
- Source profiles are a copy-and-tweak convention in caller code, not centralized; callers
  currently build `LightSource` ad hoc (`main.py`, `tools/lighting_demo.py`).
- A legacy scalar `light_map` and the `cast_source` / `update_from_fire` scalar API survive for
  render-side tinting consumers during the RGB migration; they are not part of the canonical
  directional contract.
