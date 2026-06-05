# 01 — State & Ownership

_Depends on: none. Status: DRAFT (rev.2, post-review 2026-06-05)._

How world state is represented, who owns it, and how that survives the CUDA port.
(Reconciliation: C1, C2. Review items: #2, #18, walkable, conditional buffers.)

## Two kinds of state

- **Field state (grids)** — `material`, `wall_hp`, `atmosphere`, `wave_*`, `wind_*`,
  `smoke`, `fire`, `temperature`, the dynamic attenuation field, and the ray buffers
  (`light_rgb`, `light_dir`, `heat`, `smoke_glow`). Dense, array-shaped, touched every tick.
  **GPU-resident when CUDA lands.**
- **Entity / logic state** — the unit list, orders, projectiles, RNG, turn/phase, and the
  `walkable`/pathfinding masks. Small, branchy, serial. **Stays CPU/Python.** Owned by the
  `Simulation` facade.

> Rule of thumb: *the GPU owns the world's fields; the CPU owns the world's actors.*
> Corollary (review #10): **`walkable` is a CPU-only concern** (pathfinding A*, unit-movement
> collision) and never goes GPU-side. The GPU kernels need `occludes`, the attenuation field,
> and `obstacles` — not `walkable`.

## Ownership (C1)

**Logical owner = the `GameMap` interface.** You always reach world state through
`gmap.<field>`. Today those are CPU numpy arrays; after CUDA they become numpy *views* onto
GPU-resident buffers a C++ owner holds. **Callers never change** and must **never assume who
allocated a field or where it lives.** This single discipline makes the GPU migration a
localized change inside `GameMap`.

- **Live physics fields** → the GPU copy is authoritative; the CPU gets read-only snapshots.
- **`material`** → the CPU keeps the editable source of truth (where `destroy_wall` runs) and
  pushes deltas to a GPU mirror kernels read.
- **`wall_hp`** (review #2) → **GPU-resident and GPU-written.** It is depleted on-GPU by the
  thermal-failure reaction (ch.04) and by explosions; `destroy_wall` (CPU, from
  weapons/explosions/AI) issues a **delta** that patches it. `wall_hp` is downloaded with the
  field snapshot for any CPU reads. (Supersedes rev.1's "wall_hp is CPU-authoritative.")
- **`PhysicsEngine`** holds `GameMap`'s grids **and** the stateless solvers. *(Doc-debt:
  reword `cuda_integration_plan.md` §7 to match — also §3/§4, see README.)*

## The CPU↔GPU seam (per tick)

- **Up (tiny deltas):** moved unit footprints → rebuild `obstacles` + stamp the dynamic
  attenuation field; destroyed walls → patch `material`/`wall_hp` (one delta each).
- **Down (one snapshot/frame):** the field buffers, once, for the renderer **and** any
  CPU-side reads.

**Freshness split (review #18).** The once-per-frame download may be **one tick stale for the
renderer** (invisible on screen). The **simulation must read current-tick values** — its field
reactions (ignition, wall failure, unit damage) read the buffers the GPU **just computed**,
on-GPU, before any staleness is introduced. Headless has no "frame" and always reads
current-tick values. So: *render may read stale; sim never does.*

## Authoritative shape = arrays + table, NOT tile-objects (C2)

- **Structural truth = numerical arrays:** `material` (int8), `wall_hp`, future
  `liquid_type`/`liquid_depth`.
- **A material-property table** (indexed by material id) holds *all* per-material constants
  (ch.02).
- **Derived caches** are projections rebuilt incrementally on structural change (see ch.02
  `on_tile_changed`): the **static** masks/arrays (`occludes`, `walkable`, `flammable`,
  `conductivity`, the static material attenuation). The **dynamic** attenuation contributors
  (smoke/water/units) are live fields, not caches (ch.02 §static-vs-dynamic).
- **Physics fields** are array-authoritative.
- **No tile-object grid.** Rare unique-tile state (named door, hackable terminal) → **sparse
  side-structures keyed by (x,y)**.

## Render buffers are conditional (review, cross-chapter)

The ray buffers split by nature: **`heat` (and `temperature`) are simulation fields, always
allocated.** The **visual byproducts (`light_rgb`, `light_dir`, `smoke_glow`) exist only when
a renderer is attached** — headless skips them (ch.03/05). So `gmap.light_rgb` is a
render-only field; in headless it is absent/unallocated, and only render-path code touches it.
This is a *flag* ("produce render byproducts?"), **not** a sim→render dependency (ch.03 §why
light is simulation).

## Serialization / replay (review gap)

World state serializes trivially (it's arrays): `np.save` the field set + the entity/logic
state. **Replay = initial seed + input/command log** (no need to store per-tick state).
**Cross-machine replay and lockstep multiplayer work because the sim is fixed-point
deterministic** (ch.04 C8) — same seed + same inputs → bit-identical trajectory on any
machine. The deleted tile-object "serialization for free" is replaced by "arrays serialize for
free, and even cheaper."

## Current code (where this lands)

- `src/simulation/gamemap.py` is the `GameMap` interface; it gains the table-driven caches
  (ch.02), the ray buffers (ch.03, render-conditional), the `temperature` field (ch.04), and
  the dynamic attenuation field.
- C++ solvers are already stateless (pybind zero-copy) — matches this model; CUDA moves field
  memory GPU-side behind the same interface.

## Open / deferred

- GPU-residency mechanics (view vs staging) — CUDA phase; this chapter fixes only the contract.
- **VRAM is not a constraint** (review gap): the full field set is ~100–150 B/tile → ~17 MB at
  240×480, ~150 MB at 1000×1000 (≈2% of an 8 GB card). The limit is compute/bandwidth, not
  memory; only massively-parallel training (N instances) makes memory relevant, and hundreds
  still fit.
