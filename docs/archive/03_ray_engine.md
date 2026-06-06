# 03 — Ray Engine

_Depends on: [01 State & Ownership](01_state_and_ownership.md),
[02 Material System](02_material_system.md). Status: DRAFT (rev.2, post-review)._

One DDA ray-march **primitive** serves light, heat, vision, and energy weapons. The core is a
**deposit-only accumulator over a read-only world.** (Reconciliation: C3–C6, C12–C14, C16, C17.
Review items: #1, #3, #8/#9, #11, #15, #16, #19.)

## Why the raycaster is *simulation*, not rendering

In most games light is purely visual, so sim and render are cleanly separate. **In Breach you
made light physical** — rays carry *heat* that ignites wood, melts walls, drives the firestorm.
Once light has gameplay consequences, *computing* it is a **simulation** task (it must run
headless, deterministic). So the raycaster **moves out of the renderer** (`LightingPass`, where
it lives today) **into the sim**. It produces a gameplay field (`heat`) and, as a byproduct of
the same march, the **visual** fields the renderer reads. The renderer becomes a *thinner* pure
consumer (ch.05). Dependency direction stays clean: **render reads sim; sim never reads render.**

## The ray & its payload (C5)

A ray carries `(R, G, B, heat)` plus origin, direction, range, falloff. Light is **RGB**; heat
is a scalar channel on the *same* ray. One system for all rays — an invisible heater is
`RGB=0, heat>0`. Per-channel attenuation (ch.02) gives light and heat different effective
ranges from one ray.

## Output buffers (owned by GameMap, ch.01)

The ray pass **accumulates every source** into a small **summed** set (the entire sim→render
interface — review, your point): total light, one aggregate direction, glow. No per-source or
per-direction breakdown ever crosses to the renderer; it's pre-summed, so the set is fixed-size
regardless of light count.

| Buffer | Type | Nature | Determinism | Headless |
|--------|------|--------|-------------|----------|
| `light_rgb` | f32 acc → 16F tex | render byproduct | float | **skipped** |
| `light_dir` (dx,dy) | f32 acc → 16F tex | render byproduct (normal maps) | float | **skipped** |
| `smoke_glow` | f32 acc → 16F tex (**RGB**) | render byproduct | float | **skipped** |
| `heat` | **fixed-point int** | **simulation** | int `atomicAdd` (C8) | **computed** |

**Accumulation ≠ storage (re-review).** The render channels **accumulate in f32** (`atomicAdd`)
and down-convert to the RGBA16F render textures (ch.05) **at pack time** — "16F" is the *storage*
format, not the accumulator (scalar f16 atomics are CC≥7.0-only and mis-fit 3-channel RGB).
`light_dir` is accumulated (signed, intensity-weighted), then **normalized in a separate
full-grid pass** before packing (an extra GPU kernel — not free).

- **`heat`** is the only sim-affecting ray output; headless computes only it.
- **No stealth/LoS light scalar** (review #1). Stealth is deferred and **image/field-stack
  based** ("what the NN sees is what it gets") — there is no sim-side light scalar. The float
  light buffers feed *nothing* with a gameplay threshold, so float is correct for them.
- **`light_dir`** is accumulated **weighted by deposited intensity** (review #16) and normalized
  after; it's a single dominant-direction approximation (opposing lights average out) — a known,
  accepted limitation documented in ch.05.

**No damage channel (C6):** damage is derived — walls via heat→temperature→thermal failure
(ch.04); units sample `heat` at their tiles and apply damage in CPU logic.

## The march (occlusion via attenuation) (C12)

DDA tile walk from source outward. At each tile: deposit (distance falloff) into the buffers;
then **attenuate per channel** by `material_atten (static) × dynamic_atten (live)` (ch.02).
Opaque (1.0) kills the ray (the old hard wall-stop). **The march reads `material` + the
attenuation field, never the binary `is_wall`.** The ray terminates on **aggregate**
remaining-energy, not per-channel (review #6, avoids per-channel warp divergence). **Range unit
= tiles** (tile = 1/3 m); falloff function per the source profile.

**God-rays (C16, review #8/#9):** deposit the light a tile *absorbs* into `smoke_glow` (**RGB**,
so a red beam casts a red shaft). This **supersedes** the existing surface-tint
`light_modulation` path in `overlays.py` — one energy-conserving mechanism, no double-count.

## Units & rays (C14, review #19)

- **Occlude:** in the `stamp_units` pass, units stamp into **both** `obstacles` (wave/smoke
  physics, as today) **and** the **dynamic per-channel attenuation field** (rays) — two outputs of
  one pass, **not** a new `block_light` array. The kernel reads the attenuation field
  **read-only**. Because it's RGB, a unit can occlude *per colour* (programmable — e.g. a creature
  that passes green light); opacity is just an attenuation contribution.
- **Receive:** units sample the `heat` buffer at their footprint tiles **after** the pass
  (reduction = **max over occupied tiles**); apply damage in CPU logic, iterating the stable
  id-ordered unit list. The kernel never writes units.

### Energy weapons = a weapon pre-phase (C14, review #15)

Lasers run as a **CPU pre-phase, before the lighting pass**, reading **`material` only**
(CPU-authoritative — **no GPU download**, no mid-tick stall; `wall_hp` stays GPU-side, so the
pre-phase never needs it — re-review): one ray per shot resolves hits against material occlusion
(skewer enemies along its path) and **deposits weapon heat**. **Walls breach via the GPU
thermal-failure reaction** (ch.04, step 8) — not a CPU `wall_hp` write — and a strong beam spikes
the tile past failure the **same** tick. **Then** the lighting/heat raycaster runs on the frozen,
updated map, so a laser's breach lights the room the same tick. The beam's **glow is ordinary light** (emit a
normal/transient source); the **weapon beam shares only the ray-march primitive** with lighting.

## The three passes

1. **Emitter lighting** — lamps, last-tick's fire, this-tick's weapon glow cast rays → deposit
   the summed buffers. *(Fire sources use last tick's state to break the fire→heat→fire cycle.)*
2. **Line of sight (C17, review #3):** **LoS is preserved if any light gets through** —
   backed by the attenuation march, not the binary `is_wall`. So you see/shoot through glass
   (light passes), not through opaque walls; through smoke, dimly. v1 may simplify to "blocked
   only by fully-opaque tiles"; detection-by-brightness is a later layer. Behind a `has_los(a,b)`
   interface (pairwise, CPU — entity-count-bounded, not a GPU kernel); infravision = the same on
   the heat channel; faster methods (PVS) deferred behind the interface.
3. **Entity re-emission (deferred)** — prisms/mirrors/refractive glass read the light field and
   **spawn new rays** in a second pass (C13). **No in-kernel recursion/forking.** Own later
   chapter.

## Unified tick order (review #11)

**Target** order — requires a `step()` refactor (the live `step()` resolves shooting inside
`process_shooting` *before* `stamp_units`; this splits it into intent (3) + a post-stamp resolve
(6), and moves AI before stamp). Validate against `simulation.py`:

1. Advance projectiles
2. Unit movement + facing (player + AI)
3. Register shooting intent (this tick's laser/hitscan shots)
4. AI decisions
5. **`stamp_units`** → rebuild `obstacles` + stamp the dynamic attenuation field (one pass)
6. **Weapon pre-phase** (CPU, **`material`-only**): resolve beams against material → skewer
   units, deposit weapon heat *(walls breach via thermal failure, step 8 — not here)*
7. **Physics (GPU, frozen updated map):** atmosphere/wave/wind → smoke advect →
   **raycaster** (deposit light_rgb/dir/heat/smoke_glow) → **temperature** (relaxation, reads
   heat **non-destructively**, ch.04) → **fire** spread
8. **Field reactions:** ignition (temp ≥ ignition_temp & O₂), wall thermal-failure
   (`wall_hp`→`destroy_wall`), smoke burn-off
9. **Unit damage:** sample `heat` at footprints → apply (CPU, serial)
10. Cleanup: deaths, events, pending `on_tile_changed` patches, **clear the `heat` deposit buffer**

`heat` is the per-tick **deposit** buffer, read **non-destructively** by both temperature (7) and
unit-damage (9), cleared at cleanup (10) — so unit damage is independent of conduction order.
Rules: weapon pre-phase **after** `stamp_units`, **before** the raycaster; all structural edits
funnel through `destroy_wall`→`on_tile_changed`; laser mutations are visible to the **same** tick's
heat raycaster. *Post-refactor invariant:* headless runs the identical ordering (same in-process
arrays, no "frame") → training and rendered runs produce identical trajectories, **enforced by the
determinism test.**

## CUDA contract

One thread per ray · read-only world · `atomicAdd` deposits (render RGB via **f32** atomics →
16F store; **heat = int** atomics) · **no in-kernel forking/recursion** · bounded ray length ·
entities sample buffers,
never participate in the kernel · material attenuation via a **constant/texture-bound table**
indexed by the int8 material id (review #6, avoids random-access stalls) · march all channels in
lockstep to the aggregate range (no per-channel early-out). *(Per-tick ray-list construction —
host-built list vs device 2D launch grid — and near-source atomic contention are CUDA-phase
items; see README doc-debt.)*

## Determinism note (review)

Once `heat` is sim-affecting, the raycaster must have **no unseeded internal jitter** — any
jitter draws from `sim.rng` or is removed. Update the `simulation.py` determinism docstring's
enumeration; extend the determinism regression test to the heat/temperature/ignition fields.

## Current code (where this lands)

- `cpp/src/raycaster.{h,cpp}` — DDA + the unused `heat` field + a directional variant.
  Changes: scalar → RGB payload; add `heat`/`smoke_glow` outputs; deposit smoke-absorbed energy
  into `smoke_glow`; attenuation from material table + dynamic field instead of `bool* is_wall`;
  fixed-point heat; remove internal jitter (or seed it).
- Move the per-frame cast out of `renderer/lighting.py` into the sim/PhysicsEngine step.
- `src/simulation/gamemap.py:has_los` is binary Bresenham on `is_wall` today → migrate to the
  attenuation-aware check (v1: stop on fully-opaque tiles).

## Open / deferred

- Hot-tile emission (glowing tiles as sources) + entity re-emission (prisms/mirrors) — own chapters.
- Per-tick ray-list construction + atomic-contention — CUDA phase.
