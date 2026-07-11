# FieldEdit — the Canonical Write Primitive

**Depends on:** [Grid & coordinates](01_grid_and_coordinates.md), [State & ownership (GameMap)](02_state_and_ownership.md), [Material System](03_material_system.md).

Every system that *injects or removes* a continuous field — a grenade depositing
pressure and smoke, a fire emitting smoke and heat, a laser tunnelling through a
plume, a gas emitter releasing poison — is doing the **same operation**: take a
field, a region, an amount; combine it into the field with a mode and a falloff.
Before this primitive, three call sites re-derived that operation inline, each
with a slightly different sign, clamp, and disc loop — the "an `if` for one
scenario" smell the architecture README warns against.

`FieldEdit` is that operation, written once. It is the **third leg of a pattern
the engine already names twice**:

- `wind = -grad(p)` is the canonical **read** primitive — "one field, many
  readers" ([Atmosphere & pressure](04_atmosphere_and_pressure.md)).
- the DDA march is "one primitive, two consumers" ([Ray engine](08_ray_engine.md)).
- **`FieldEdit` is the canonical write primitive** — "many systems write many
  fields through one operator."

> **EOS refactor (2026-07) — as-built.** The §3 policy table below predates the EOS refactor
> and is stale in two ways: (1) `atmosphere` is now the Q16.16 derived pressure `P` (a zero-copy
> alias, *not* float), and pressure *deposits* remap to **N/T feeds** — an explosion's `pressure`
> payload becomes an energy/gas-mass scale, since nothing "injects pressure" any more (ch.04
> as-built); (2) `wave_source` (with `wave_p`) is **retired** — the row is dead. The per-gas
> `FIELD_POLICY` rows (O₂/N₂/traces) are the live write surface now. A full rewrite of the §3
> table to the EOS field set is a flagged follow-up; treat ch.04's as-built as authoritative for
> the field model in the meantime.

Home: `src/simulation/field_edit.py`.


## 1. The three composing parts

### `FieldEdit` — a frozen description of one edit

A pure, stateless value object. No state, no application logic — it only
*describes* an edit so it can be queued, sorted, and applied later.

```python
@dataclass(frozen=True)
class FieldEdit:
    field: str            # "smoke" / "atmosphere" / "wave_source" / "fire" / "heat" / per-gas (pending)
    region: Region        # TILE · DISC · BEAM · RECT
    coords: tuple         # TILE:(r,c) · DISC:(r,c,radius) · BEAM:(r0,c0,r1,c1,width) · RECT:(r0,c0,r1,c1)
    amount: float
    mode: EditMode = ADD              # ADD · REMOVE · MAX
    falloff: Falloff = FLAT           # FLAT · LINEAR
    channel: int | None = None        # None = scalar; 0/1/2 = R/G/B of an (h,w,3) field
    clamp: tuple | None = None        # post-combine clamp; None = the field-policy default
    noise: float = 0.0                # >0 = per-tile multiplier in [1-noise, 1], drawn from sim.rng
    source_id: int = 0                # bookkeeping + stable-sort grouping (one emitter = one id)
```

**Modes (`EditMode`).** Only the three a live consumer needs today:

| Mode | Combine | Used by |
|------|---------|---------|
| `ADD` | `field += contribution` | pressure / smoke / heat deposits |
| `REMOVE` | `field -= contribution` | burn-off, clearing (a REMOVE-to-0 is a large amount + clamp floor) |
| `MAX` | `field = max(field, contribution)` | ignite (never lower an existing fire) |

`SET` (lerp-to-value) and `MIN` are **deferred** — added the day a consumer
needs them, not before (Erik's call: don't ship modes with no caller).

**Regions (`Region`).** `TILE` (a single cell), `DISC` (a filled disc), `BEAM`
(a thick line — the laser case), `RECT` (an axis-aligned box). The disc/beam/rect
loop is written **once**, in `_iter_region`, which yields `(row, col, weight)` in
a deterministic row-major order. DISC uses a strict `dist < radius` membership;
the `dist == radius` ring has weight 0 under LINEAR anyway, so excluding it
changes no additive/max result while keeping the per-tile RNG draw count exact.

**Falloff (`Falloff`).** `FLAT` (weight 1 everywhere) and `LINEAR`
(`weight = 1 - dist/radius`, == today's explosion falloff). Others (`SHARP`,
`GAUSS`) are added only when a migrated or new site uses one.

### `apply_field_edit(gmap, edit, rng)` — the only writer

The **single** function that writes a field through this path. It resolves the
field array and its policy, iterates the region, applies the per-cell skip-mask
veto, draws noise (if any), computes `contribution = amount × weight × noise`,
and combines. `_combine` does float `+=` / `-=` / `max` for float fields and a
**Q16.16 saturating branch** for the `heat` field — `heat_quantize` then
`heat_saturating_add`, **never a float `+=`** (§3).

### `EditQueue` — the deterministic flush

Consumers `enqueue` edits during the tick; the Simulation `flush`es the whole
queue at one fixed point, applying every edit in a **stable sort** by
`(field, source_id, region, seq)`, where `seq` is the monotonic enqueue index.


## 2. Queued, not immediate — the determinism-critical decision

For Level-2 lockstep (a hard requirement), the answer is a
deterministically-ordered queue, **not** immediate mutation:

- **Order independence is the whole determinism story.** Two grenades overlapping
  a tile give `clamp(clamp(s+a)+b)` — order-dependent the moment a clamp or `MAX`
  is involved. A stable sort makes the applied order identical on every machine
  regardless of projectile / AI / container iteration order. This is the same
  principle the `heat` buffer relies on (integer saturating add is
  order-independent), lifted to *all* field edits and made explicit rather than
  accidental. Within one emitter (same field/source_id/region) the `seq`
  tie-break preserves enqueue order, so e.g. the explosion's many `wave_source`
  ADDs sum in exactly their original disc-iteration order (bit-identical float
  accumulation).
- **One flush = one RNG consumer.** A noisy deposit's per-tile multiplier must
  come from the seeded `sim.rng`. With one flush site, the flush is the single
  RNG consumer, drawing in sorted order — the seeded-rollout guarantee is
  structural, not a per-caller convention. A skip-masked tile draws **nothing**
  (the veto precedes the draw), so the RNG sequence depends only on the
  surviving-tile order, which is deterministic.
- **Solvers see a settled pre-state.** A laser burn-off and a grenade cloud
  issued the same tick both land before smoke advection runs, so the solver
  advects the net result once.

### Flush slot in `Simulation.step()`

The flush runs **after** the weapon / fire / explosion phases enqueue their edits
and **before** the physics solvers run:

```
… update projectiles (enqueue explosion edits) → movement → shooting → zombie AI
   → stamp_units → EditQueue.flush(gmap, rng) → physics solvers → heat consumers → clear
```

`sim.edit(FieldEdit(...))` is the enqueue API. The queue is a per-tick deposit
list, exactly like the `heat` buffer — recreated empty each `reset`.


## 3. Per-field policy table — consumers stop knowing the rules

`FIELD_POLICY` declares, per field, the three things a consumer would otherwise
have to remember:

| Field | dtype | default clamp | skip-mask |
|-------|-------|---------------|-----------|
| `smoke` | float | `[0, 1]` | skip `solid` |
| `atmosphere` | float | — | skip `solid` |
| `wave_source` | float | — | skip `solid` + `is_vacuum` |
| `fire` | float | `[0, 1]` | skip non-`flammable` |
| `heat` | Q16.16 | — | (none) |
| `water_depth` | float | `[0, ∞)` | skip `solid` |

- **dtype** selects the `_combine` branch: `float` (`+=` / `-=` / `max`) or
  `heat` (Q16.16). The fixed-point discipline is implemented **once** here, not
  re-derived at every future heat-deposit site. `heat_quantize` (saturating
  float→Q16.16, round-to-nearest) and `heat_saturating_add` (clamp at
  `INT32_MAX`, never wrap) mirror the C++ `raycaster.h` helpers exactly, so the
  Python write path is bit-compatible with the kernel's deposit.
- **clamp** is the post-combine bound used when the `FieldEdit` sets none.
- **skip-mask** is a per-cell veto (`smoke`/`atmosphere` never enter a wall;
  `wave_source` never sources in a wall or vacuum; `fire` only ignites flammable
  tiles). The N gas fields now exist (`gmap.gas` is `(N, h, w)`, with `smoke` a
  view into the black-smoke slice); only the black-smoke deposit has a policy row
  (`"smoke"`) today, so each additional gas is a new `FIELD_POLICY` row when it
  needs a deposit path — *zero* consumer code changes.

This is the write-side mirror of the data-driven gas table: `field` is a string
key, so a poison grenade is `FieldEdit("poison", …)` and "new gas = new field
name, not new edit code."


## 4. How the call sites migrated (behaviour-preserving)

The migration is a refactor: the existing explosion / smoke tests are the
regression guard and stay green.

- **`physics.apply_explosion`** — the inline field mutations now enqueue:
  - `atmosphere +=` → one `DISC` `ADD` with `LINEAR` falloff.
  - smoke-clear (inner 40 %) → a `DISC` `REMOVE` (large amount, smoke's `[0,1]`
    clamp drives it to exactly 0) — the REMOVE-to-0 idiom.
  - `wave_source +=` 3×3 smoothed kernel → per-tile `TILE` `ADD`s emitted in the
    original disc-iteration order; `seq` preserves the exact float summation.
  - in-radius ignite → per-tile `TILE` `MAX` (its membership radius `0.7 r`
    differs from the falloff radius `r`, so a single DISC edit would couple the
    two — a per-tile edit with the pre-computed `0.5·falloff` amount is faithful).
  - **wall damage stays immediate and structural** — see §5.
- **`physics.add_explosion_smoke`** — the noisy disc deposit → one `DISC` `ADD`
  with `LINEAR` falloff, `noise` set, `[0,1]` clamp. The per-tile noise draw moves
  from an inline `rng.uniform` to the flush, in the queue's deterministic tile
  order. Because the draw order matches the legacy nested-loop order and a
  skipped solid tile consumes no draw, the deposit is **bit-identical** for a
  fixed seed. (One intended consequence: a tick where a grenade detonates *and*
  shooting fires now consumes the smoke-noise RNG after the shooting RNG, since
  the draw moved from the projectile phase to the flush — the single-RNG-consumer
  property the queue is built to provide.)


## 5. The one honest carve-out — topology is not a field

Edits that change **topology** — `destroy_wall` (which retriggers the
conductivity/occlusion cache patch via `on_tile_changed`, marks the smoke sink
field dirty, refills atmosphere) — are **not** FieldEdits. They stay immediate
and structural. `FieldEdit` is strictly continuous scalar/vector values on a
*fixed* grid (smoke, atmosphere, wave_source, fire, heat, the other gas slices).

`wall_hp -= dmg` *is* a clean `REMOVE` FieldEdit in shape, but the destruction it
triggers must run as a separate **post-flush structural sweep**: collect tiles at
`<= 0`, destroy them in sorted order — the same pattern fire burn-through and the
over-pressure relief valve already use (the solver/scan returns coords, the
runner destroys them). This is **designed, not built** here: it lands with the
fire phase, where `wall_hp` becomes both the fuel source and the structural
health, and a single sweep collapses fuel-burnout and wall-failure into one
mechanic. Until then, `apply_explosion` keeps its `wall_hp -= …` + `destroy_wall`
inline (the only non-FieldEdit write it makes).


## 6. CUDA-readiness

A flat, pre-sorted array of `FieldEdit` records is a kernel-launch list — one
thread per edit, atomic combine — the same shape as the ray-list and gather
stencils. The stable sort moves to a GPU sort; the `heat` saturating add is an
`atomicMax`/`atomicAdd` on the Q16.16 buffer. Nothing in the contract forecloses
the port.


## Implementation status

**Built:**

- `FieldEdit` (frozen), `EditMode {ADD, REMOVE, MAX}`, `Region {TILE, DISC, BEAM,
  RECT}`, `Falloff {FLAT, LINEAR}`.
- `apply_field_edit` + `_iter_region` (the disc/beam/rect loop, once) +
  `_combine` (float and Q16.16 `heat` branch); `heat_quantize` /
  `heat_saturating_add` mirroring the C++ helpers.
- `FIELD_POLICY` for the live fields (`smoke`, `atmosphere`, `wave_source`,
  `fire`, `heat`, `water_depth`, and — weapons W3 — `gas`).
- `EditQueue` with the stable-sorted, single-RNG-consumer flush; one flush point
  in `Simulation.step()` before the solvers; `sim.edit(...)` enqueue API.
- Migration of `apply_explosion` and `add_explosion_smoke` (behaviour-preserving;
  existing explosion/smoke tests green). Test suite `tests/test_field_edit.py`
  covers each mode, each region, LINEAR falloff, clamp, per-field skip-mask, the
  heat Q16.16 saturating branch, queue order-independence (the stable sort), the
  seeded-noise determinism, and before/after migration equivalence.
- **The gas emitter (weapons W3, 2026-07-05):** `field="gas"` targets the
  `(N, h, w)` multi-gas array with `channel = <gas slice id>`
  (`gmap.gases.name_to_id`), resolved to the contiguous `(h, w)` int32 view at
  apply time; same `"gas"` combine + `[0, 1]` clamp + solid skip as the
  `smoke` (BLACK_SMOKE) view. First consumer:
  `simulation.payloads.emit_gas` — the gas-payload DISC deposit
  (deliberately `noise = 0`: a deterministic radial cloud, no RNG;
  tests/test_payloads.py pins the per-tile Q16.16 exactness).

**Designed, not built:**

- `wall_hp` damage via a `REMOVE` FieldEdit + the post-flush `<= 0` destruction
  sweep (lands with the fire phase — §5).
- The laser emitters (BEAM burn-off) — built on `FieldEdit` when they land.
  (The gas emitter LANDED with weapons W3, above.)
- `SET` (lerp-to-value) and `MIN` modes; additional falloffs (`SHARP`, `GAUSS`)
  — added when a consumer needs them.
- Fire's smoke emission and the §3-plume `atmosphere` deposit re-expressed as
  FieldEdits (land with the fire/temperature feedback work).
