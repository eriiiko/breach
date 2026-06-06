# 02 — Material System

_Depends on: [01 State & Ownership](01_state_and_ownership.md). Status: DRAFT (rev.2, post-review)._

Materials are **data-driven**: every property derives from the material id via a single
**property table**. Adding a material = one table row + one CSV mapping. (Reconciliation: C9,
C10, C11. Review items: #10/#12, #13, #14.)

## The property table (C9, review #14)

Named-key dict per material (supersedes the flat `[hp, reflectivity, absorption, flammable,
passable]` arrays). **Light optics and pressure acoustics are kept as DISTINCT columns** —
they were conflated in the old flat format. One row per material id:

```toml
[materials.hull]
hp            = 300
flammable     = false
passable      = false           # unit walkability
# --- optics (light rays, ch.03) ---
light_atten   = [1.0, 1.0, 1.0] # per-channel RGB attenuation; 1.0 = opaque, 0.0 = clear
heat_atten    = 1.0             # heat-ray attenuation
# --- thermal (ch.04) ---
conductivity  = 50.0            # high = heat spreads fast along metal
ignition_temp = 0.0             # 0 = non-flammable
# --- acoustics (pressure-wave boundary, the wave solver) ---
wave_reflect  = 0.9             # fraction of shockwave energy bounced back
wave_absorb   = 0.1             # fraction damped; transmit = 1 - reflect - absorb (carries through)
# --- structural ---
blast_resist  = 0.0

[materials.glass]
hp = 15 ; flammable = false ; passable = false
light_atten = [0.1, 0.1, 0.1]   # transmits most light (colour-tint via unequal channels)
heat_atten  = 0.3
conductivity = 1.0 ; ignition_temp = 0.0
wave_reflect = 0.3 ; wave_absorb = 0.1   # rest transmits — boom carries through, then it shatters
blast_resist = 0.0

[materials.air]
hp = 0 ; flammable = false ; passable = true
light_atten = [0,0,0] ; heat_atten = 0
conductivity = 0.0 ; ignition_temp = 0.0
wave_reflect = 0.0 ; wave_absorb = 0.0
```

(Values illustrative — tune in the lighting demo. `wood`, `door`, `steel` follow the same
shape.)

### Column meanings (so nothing is ambiguous — review #14)

- **`light_atten` (RGB) / `heat_atten`** — *optics*: how much a ray loses crossing the tile.
  These replace the old light "absorption". Subsumes the `block_light` bool and the wall
  hard-stop (opaque = 1.0 → ray dies). **Not** the same as acoustic absorption.
- **`wave_reflect` / `wave_absorb`** — *acoustics*: pressure-wave boundary behaviour, for the
  wave solver. **Independent** (not `reflect = 1 − absorb`) because the remainder
  **transmits**: `transmit = 1 − reflect − absorb` (a muffled boom carries into the next
  room). hull → high reflect; soft → high absorb; thin → high transmit. *(Status: the wave
  solver currently uses `is_wall` as a hard reflective boundary; these columns wire into its
  boundary conditions when we do the pressure-physics pass — verify exact current usage before
  finalizing.)*
- **`conductivity` / `ignition_temp`** — thermal (ch.04). `ignition_temp` is **quantized to
  the fixed-point domain once at load** (ch.04 C8).
- **`light_reflect`** (specular, for the deferred entity-reflection chapter) and **`emissivity`**
  (deferred with hot-tile emission) are **future columns** — listed so the schema is complete.

## Two derived masks, not one (review #12)

`is_wall` was wrongly treated as the collision source. In the code, collision/pathfinding use
`material in {AIR, open-DOOR}`, while `is_wall` includes closed doors (so they occlude). So:

- **`occludes`** — physics/light/smoke/vision boundary (includes doors). GPU-side.
- **`walkable`** — unit movement/pathfinding (AIR + doors). **CPU-only** (review #10).

Today a `DOOR` tile **always** occludes **and** is **always** walkable — there is **no open/closed
state yet** (matches `gamemap.py`'s TODO). The dynamic door system that adds an open/closed bit is
**deferred**; until then both masks treat every door identically. Door duality (occludes
light/smoke, passable to units) lives in the table as `passable` + `light_atten`, retiring the
hardcoded `np.isin` special-case.

## Per-channel attenuation, static × dynamic (C11, review #13)

The ray, crossing a tile, multiplies its RGB by **two attenuations**:

1. **Static (material)** — from the table, **cached**, changes only on structural edit (wall
   destroyed). Walls, glass.
2. **Dynamic (field)** — a **live** per-channel field recomputed each tick: **smoke, water,
   and stamped units** (ch.03 §units). Smoke density changes every tick, so this *cannot* be a
   structural-change cache.

`total_atten[ch] = material_atten[ch] (static) × dynamic_atten[ch] (live)`. Because the dynamic
field is **per-channel RGB**, anything can occlude *per colour* — an aquarium tints light
blue-green, a creature can pass green light — and it stays read-only-GPU-safe because writers
(units/water/smoke) stamp it in the CPU pre-phase and the kernel only reads it (ch.03 §units).

| Material | `light_atten` | Behaviour |
|----------|---------------|-----------|
| opaque wall | `[1,1,1]` | ray dies (== hard block) |
| air | `[0,0,0]` | passes untouched |
| glass | `[0.1,0.1,0.1]` | passes, dimmed |
| tinted window | `[0.9,0.9,0.1]` | "blocks 2 of 3 colours" for free |

**Not** covered by attenuation: **direction-changing** optics (refraction, prisms, mirrors) —
the deferred entity re-emit pattern (ch.03), never in-kernel.

## Cache invalidation (review #10)

`destroy_wall` and the laser pre-phase MUST funnel structural edits through a single
**incremental `on_tile_changed(x, y)`** that patches *all* derived static caches for that tile
(`occludes`, `walkable`, `flammable`, `conductivity`, static `light_atten`/`heat_atten`) — **not**
a full `_update_caches()` rebuild (which is O(grid) and won't scale when a firestorm melts many
walls/tick). This patch is the CPU-side delta the ch.01 seam uploads.

## Material set is open (C10)

The table makes the set open-ended; adding a material = one row + a CSV→id mapping. *Prereq
(review nit):* unify the duplicated `MAT_*` id definitions (currently in both `gamemap.py` and
`level_loader.py`) into one source so the "two places" promise holds.

## Current code (where this lands)

- `config.toml` `[materials]` → the named-key format above (incl. acoustic + optics columns).
- `src/simulation/gamemap.py` `_update_caches()` → table lookups + the incremental
  `on_tile_changed`; `destroy_wall` routes through it.
- `src/level_loader.py` → CSV→id map covering the full set; shares the unified `MAT_*` ids.
- **Config hot-reload** (old §14, non-negotiable): the table must hot-reload; on reload, rebuild
  static caches + re-sync the GPU material mirror.

## Open / deferred

- `emissivity`, `light_reflect` columns — deferred with hot-tile emission / entity reflection.
- Final attenuation/conductivity/acoustic **values** — tuned in the lighting demo + a wave test.
- **Liquid** dynamic attenuation enters via the same dynamic field as smoke (review gap closed).
