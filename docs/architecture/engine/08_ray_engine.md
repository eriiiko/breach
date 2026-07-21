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

> **Implementation reality:** the sim-side **heat** cast already runs inside the deterministic
> step (`PhysicsRunner.cast_fire_heat`, depositing `gmap.heat`). The **render light** cast still
> runs in `renderer/lighting.py` (`LightingPass.compute_light_field`, invoked from
> `game_renderer.upload_state`). Moving that render cast into the sim step is a planned, mechanical
> relocation — the C++ kernel and the buffer contract are already in their final form, so the move
> is a caller change, not a redesign. See *Implementation status*.

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

A DDA tile walk from the source outward. Each ray carries a **fixed energy budget**
`intensity / N` (N = the source's ray count — see *Falloff is density*, below) and a per-channel
**survival** ∈ [0,1] that starts at 1 and is reduced **only by occlusion** (never by distance).
At each tile:

1. **Deposit** each channel's `energy · color[c] · survival[c]` into `light_rgb`. There is **no
   per-ray distance falloff** — the `1/r` intensity falloff emerges from ray *density* (below).
2. **Accumulate direction** into `light_dir`, weighted by the aggregate deposited intensity
   and pointing *toward the source* (`-dx, -dy` of travel). It is normalized to a unit vector
   in a separate full-grid pass after all sources are cast (see below).
3. **Deposit heat** — only if the source emits heat **and `heat_survival > ε_heat`**:
   `heat_quantize(energy · src.heat · heat_survival)`, saturating-added into `heat`.
4. **Attenuate survival per channel** by the material's static attenuation, then by the live
   gases — a per-channel Beer-Lambert transmission summed (density-weighted) over every gas
   sharing the tile (see [Smoke & Gases](smoke.md)), so mixed gases tint the survivor
   automatically. **Heat survival attenuates by `heat_atten` only** — gases do not block heat.
   **Source-tile self-occlusion skip (load-bearing invariant):** on the *source tile*
   (`distance == 0`, the first marched cell) heat is *deposited* but heat survival is **not**
   attenuated. A heat emitter sits *inside* a solid — fire only ever burns on a flammable,
   heat-opaque tile (`heat_atten ≈ 1`) — and a radiating surface emits *outward*, it does not
   absorb its own emission. Without the skip, `heat_survival` would hit 0 on tile 0 and the
   fire would deposit nothing past its own cell (no ignition-at-a-distance). Every downrange
   tile attenuates normally, so a wall still blocks the fire's heat beyond it. This is inert
   for air-sourced rays (their source `heat_atten == 0`). **The GPU kernel must replicate this
   exactly** — a doc-/code-faithful port that attenuates heat on the source tile silently
   breaks fire radiation, and the air-cast unit tests (which start in vacuum) would not catch it.
5. **Deposit god-ray glow** — the energy the gases scatter back (their per-channel scatter
   budget, decoupled from absorption) goes into `smoke_glow`.
6. **Step** to the next tile. Each channel **stops depositing once its own survival drops below
   its threshold** (`ε_rgb` for light, `ε_heat` for heat); the **ray** terminates when *every*
   channel is below threshold (fully absorbed) or `max_range` is exceeded.

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

The **ray** marches to the **aggregate** range — it continues while *any* channel is still above
its threshold — so the march length is uniform across a warp (no per-channel early-out → no GPU
divergence). The cost is small: an opaque tile zeroes every channel at once, so a wall stops the
ray the same step. But each channel's **deposit** is gated by its own survival (step 3/6 above):
heat deposits only while `heat_survival > ε_heat`, *independent* of where the RGB channels die.
That gating is a per-thread branch on a value the thread already holds — it does not change the
march length, so it adds no divergence. It is what **decouples the heat output from the light
path**, the core of the determinism contract below.

### Falloff is density, not per-ray distance (the propagation model)

A ray is a physics abstraction that **carries constant energy** along its path — only occlusion
removes energy, never distance. The `1/r` intensity falloff is *not* a per-ray multiplier; it
emerges from **ray density**. A point source fans `N` rays over angular spread `Δ`. At distance
`r` a unit cell subtends angle `≈ 1/r`, so it is crossed by `≈ N/(Δ·r)` rays. Give each ray the
budget `energy = P/N` (P = the source's total emitted power) and the cell accumulates

```
  k(r) · (P/N) · survival  =  [N/(Δr)] · [P/N] · survival  =  P·survival / (Δ·r)
```

— the **N cancels**: brightness is independent of ray count (ray count is a *quality* knob, not a
brightness knob) and falls as `1/r`, the faithful 2D law. Two equal-power sources with different
ray counts deposit identically. This is why there is **no `dist_atten`** in the march: the old
`1/(1+d²·0.01)` per-ray factor stacked a second falloff on top of the density falloff
(`~1/r³` total — far too steep) and was a band-aid for the source-cell pile-up. Removing it makes
the model both more physical *and* cheaper (one fewer per-tile op; the `1/N` folds into the ray's
initial energy).

The honest cost of pure density is **sampling noise** where rays separate past ~1 tile (the wide
far field): a cell may catch 0, 1, or 2 rays → faint banding in the `1/r` field. It is fully
*deterministic* (identical CPU/GPU — it does not touch the bit-identity gate), so it is a quality
matter, tuned by ray count (the auto count keeps ~1-tile spacing out to `max_range`) and an
optional blur. **`intensity` now means total emitted power** (N-independent), so sources re-tune
once when this lands.

### Determinism: heat is decoupled from light

`heat` feeds unit damage, so its integer output must be bit-identical CPU↔GPU (and cross-machine,
eventually). Three things guarantee it, and they compose with the survival model above:

1. **Per-channel survival termination** (above) — heat deposits while `heat_survival > ε_heat`,
   a heat-only quantity driven by `heat_atten` (material). It never depends on the RGB survival,
   which carries the gas-optics `exp` (a transcendental that differs CPU↔GPU). So the
   heat-touched tile *set* is independent of `exp`. This is also what makes a **heat-shield
   material** (low-E glass: passes light, blocks heat — physically real) safe to add: the old
   aggregate termination would have let the surviving light drag a heat tail past the shield,
   desyncing; decoupled heat stops at the shield, deterministically.
2. **Host-precomputed ray directions** — `dx, dy = cos θ, sin θ` are transcendentals too, and a
   1-ULP difference flips a DDA step → a different tile path → different heat. So directions are
   computed once on the host (per ray) and handed to the kernel; the device march is then pure
   `+ / ÷ / compare` (deterministic). No `cos`/`sin` on the device. (Cross-*machine* heat needs a
   deterministic integer direction too — that lands with the combat/HP integerization, not here.)
3. **No fast-math on the deposit path** (`--fmad=false`, no FMA contraction) — the deposit
   arithmetic (`energy · survival`, the `(1-atten)` decays) is correctly-rounded IEEE `+,-,*,/`,
   which is bit-identical across MSVC `/fp:strict` and nvcc. The CPU `raycaster.cpp` is compiled
   `/fp:strict` (it now produces `heat`, sim state — not the old render-only `/fp:fast`). The
   render channels (`light_rgb`, `light_dir`, `smoke_glow`) are exempt — gate `heat` only.
   **Pinned truncation path:** the final float→Q16.16 quantize (`heat_quantize`) runs in
   **double precision** — promote `energy` to `double`, multiply by `HEAT_SCALE`, add `0.5`,
   truncate toward zero (round-half-up), clamping at `INT32_MAX`. The GPU kernel **must** use the
   identical double-precision round — *not* `rintf`/`__float2int_rn` (round-half-to-even) and
   *not* a float multiply — or heat diverges at the LSB on boundary values (the "pin one
   truncation path cross-toolchain" lesson). The saturating integer accumulate is order-free
   (non-negative deltas under a monotone clamp), so the device may scatter `heat` with an integer
   atomic — but it must be a **saturating** atomic (CAS-loop or add-then-clamp), since a plain
   signed `atomicAdd` wraps and breaks the overflow guarantee.

**Defaults** (tune by eye in `tools/lighting_demo.py`): `ε_rgb = 0.01`, `ε_heat = 0.01` (a
channel stops when ~99% absorbed — only meaningful behind occluders, since survival decays only
by occlusion), ray count = the existing auto formula (`≈ max_range · spread`, ~1-tile spacing at
range). All three are `config.toml` dials on the `[graphics.lighting]` / raycaster path
(no-recompile).

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
- **ray directions precomputed on the host** (`dx,dy = cos θ, sin θ` per ray) and uploaded — no
  `cos`/`sin` on the device, so the DDA path is pure `+/÷/compare` and bit-identical (see
  *Determinism* above)
- read-only world (material table bound as a constant/texture, indexed by the int8 material id —
  no random-access stalls)
- `atomicAdd` deposits — render channels via f32 atomics down-converted at pack time; **heat via
  integer atomics** (order-independent, the determinism guarantee)
- **per-channel deposit gating, not per-channel ray-termination** — heat deposits while
  `heat_survival > ε_heat` (decoupled from the RGB/`exp` path); the ray still marches the uniform
  aggregate range, so the gate is a branch on a held value, not warp divergence
- `--fmad=false`/no-fast-math on the deposit math; render channels exempt, **gate `heat` only**
- no in-kernel forking or recursion; bounded ray length
- entities sample buffers, never participate in the kernel

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

> **⚙ Propagation-model redesign — CPU done on branch `cuda-s2-cpu-march`, feel-check + GPU port
> pending (CUDA-S2).** The sections above (*The march*, *Falloff is density*, *Determinism*) are
> now the **implemented** model on that branch: pure-density `1/r` falloff, per-ray energy
> `intensity/N`, per-channel survival termination (`light_cull`/`heat_cull`), heat decoupled from
> the RGB/`exp` path, `raycaster.cpp` on `/fp:strict`. The old per-ray `dist_atten = 1/(1+d²·0.01)`
> and the single aggregate-energy cull are **gone**. Re-tunes that landed with it: a render
> exposure `u_light_gain` (intensity is now total power, ~`ray_count`× dimmer field) and
> `k_fire_heat` 200→1600 (×`fire_ray_count`, restoring ignition/kill after the `/N`). Remaining
> sequence: (1) **Erik feel-check** of the brightness/falloff on the demo (then regenerate the
> heat/xarch golden — it legitimately moved); (2) **GPU** port (one thread/ray, host-precomputed
> directions, saturating integer `atomicAdd` heat, float-atomic render) + the heat bit-identity
> gate. **`main` still ships the OLD model** until this branch merges. The bullets below are being
> updated to the new model as parts land.

**Built and shipping (Tier 1):**

- `cpp/src/raycaster.{h,cpp}` — the full directional march. `LightSource` carries RGB `color`,
  `heat`, `jitter`, falloff. `cast_source_directional` / `march_ray_directional` deposit
  `light_rgb`, `light_dir` (dx/dy), Q16.16 `heat` (quantized, saturating-add), and RGB
  `smoke_glow`. Per-tile optics now read the **multi-gas** field, not a single smoke array: the
  kernel takes `gas_field` (shape `(n_gases, h, w)`) plus per-gas `gas_absorption` /
  `gas_scatter` tables and `n_gases`, and sums a density-weighted per-channel Beer-Lambert
  transmission (absorption) and god-ray scatter over every gas sharing the tile (a single
  populated gas reproduces the old single-`smoke` path). Heat is the **independent 4th channel**:
  a scalar heat survival attenuated per tile by `heat_atten` exactly as each RGB channel is
  attenuated by `light_atten`, so heat and light occlusion diverge (heat-shield vs smoked glass);
  gases never attenuate heat. The deposit is `src.heat · heat_survival · falloff`, decoupled from
  RGB. Aggregate-energy termination over ALL FOUR channels {R,G,B,heat}; god-rays from the
  per-gas scatter budget; `normalize_directions` post-pass (vector-magnitude).
- `cpp/src/bindings.cpp` — `LightSource` and `Raycaster` exposed to Python; `heat` /
  `smoke_glow` / `heat_atten` are optional (`None` skips that deposit / attenuation), so
  render-only callers pass only what they need.
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
- **Heat consumers.** `heat` is read, not just deposited. Two consumers run each tick:
  the `TemperatureSolver` converts this tick's `gmap.heat` to `temperature`, conducts, and cools
  it (`physics_runner.step`); then unit heat damage samples the still-occluded `heat` at each
  unit's footprint (`apply_environmental_damage`, called from `Simulation.step`). Temperature
  ignition derives from the resulting field, and the per-tick `heat` deposit is cleared at
  end-of-tick after both readers (see [Temperature & Heat](temperature.md)).
- Tests: `tests/test_rgb_light_atten.py`, `test_dyn_light_atten.py`, `test_heat_smoke_glow.py`,
  `test_rgb_light_pack.py` cover the per-channel attenuation, unit-shadow stamping, heat +
  god-ray deposit, and pack contract.

**Designed but not yet built:**

- **Render light cast relocation into the sim.** The sim-side **heat** cast already ships:
  `PhysicsRunner.cast_fire_heat` (`src/simulation/physics_runner.py`) casts every burning tile
  as a heat `LightSource` into `gmap.heat` at the START of the deterministic `step()`
  (`physics_runner.step` → called from `Simulation.step`). What remains renderer-side is the
  **render light** cast: it still runs in `renderer/lighting.py` from `game_renderer.upload_state`
  (the RGB/`light_dir`/`smoke_glow` buffers are not yet filled inside the sim step). The kernel
  and buffer contract are final, so this is a caller move plus extending the determinism test to
  the render fields — not a redesign.
  - **As-built (S8c item 1, the fire-FPS fix):** on `--cuda` the heat cast is **batched**.
    `cast_fire_heat` builds the whole burning-tile source list, then issues **one**
    `bp.cuda_raycaster_cast_batch` — it concatenates `build_ray_list` over every source (in the
    `/fp:strict` TU) and marches them in a **single** `raycaster_cast_directional` (one H2D of the
    inputs + running `heat` plane, one march, one D2H). This replaces the old one-round-trip-**per
    source** loop that ran hundreds of whole-plane transfers per tick (~3 fps with hundreds of
    fires, 2026-07-20 B5). `heat` is **byte-identical** to the per-source path — heat deposits are
    order-free saturating integer atomic adds, so batching changes only the atomic interleave.
    Render channels differ in float-atomic order (determinism-exempt) and are discarded by this
    caller (`smoke_glow=None`). Payoff: a 600-fire firestorm goes from ~424 ms (~2.4 fps) to
    ~1.5 ms for the cast (277×). Gate: `tests/cuda_s2_check.py` batch-vs-per-source witness +
    `tests/test_s8c_fire_heat_bench.py`. The CPU path stays the per-source
    `cast_source_directional` loop (no transfer tax to amortise).
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
  from `sim.rng` (or be removed) to keep the heat field replay-deterministic. **As-built:** the
  sim heat cast sets `jitter = 0` (removed) — so `build_ray_list`'s per-source `mt19937` is never
  drawn. The S8c batched cast **depends on** this: a nonzero jitter would couple sources through
  the RNG sequence and desync the batch from the per-source loop. A future flickering heat source
  must draw its jitter from `sim.rng` before re-enabling it (invariant noted in `cast_fire_heat`).
- `light_dir` is a single dominant-direction aggregate; opposing lights average toward flat
  normal-map relief and per-light coloured relief is not recoverable. Accepted, not a bug.
- Source profiles are a copy-and-tweak convention in caller code, not centralized; callers
  currently build `LightSource` ad hoc (`main.py`, `tools/lighting_demo.py`).
- A legacy scalar `light_map` and the `cast_source` / `update_from_fire` scalar API survive for
  render-side tinting consumers during the RGB migration; they are not part of the canonical
  directional contract.
