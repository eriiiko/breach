# 17 — Swarm units: GPU-resident population-scale units

> **Name (Erik's ruling 2026-09-10): "swarm units"** — many simple units that
> move as a population, versus "units" (CPU actors). The working nickname
> "swarm-unit" survives only in issue #63 / memory; code never says a vendor.

> **Depends on:** 02 (state & ownership — amended by §1 below), 14 (determinism
> & number ingress — amended by §3: Philox is a second sanctioned raw-draw
> source), 06 (temperature & fire), 16 (entity system),
> `docs/rl_env_arc_proposal_2026-08-27.md` §A (resident-path habits, Erik-ruled)
> and §B3 (device unit store — piloted in part, see §1).
> **Status:** BLESSED 2026-09-10 (Erik, after a 3-pass review — see the
> Review log). Implementation: branch `63-swarm-units`, build order §11.
> Design v2, 2026-09-07 — post-adversarial-critique
> (`docs/cudaunit_critique_2026-09-07.md`; all BLOCKER/MAJOR findings folded
> in). Forks marked **[ERIK]** await ruling. Pilot species: larvae (issue
> #63). Research inputs: `docs/cudaunit_lit_brief_2026-09-07.md` +
> `docs/cudaunit_canon_brief_2026-09-07.md`.

## 1. What a swarm-unit is — and the engine/02 amendment

A **swarm species** is not a class. It is three data-oriented artifacts:

1. a **species table row** (all tunables, Q16.16; never a per-species
   if-chain),
2. a set of **fixed-capacity SoA per-unit arrays** (one device array per
   attribute), and
3. a **behavior kernel** with a bit-identical CPU twin.

Chapter 02 says "GPU owns fields, CPU owns actors." Swarm units cross that
line under a real criterion (not the circular "whatever is array-shaped"):
**homogeneous populations whose full per-agent state fits a fixed schema and
whose per-tick behavior is uniform and branch-light are swarm units; agents
with orders, plans, inventory, or dialogue stay CPU units.** Marines gaining
device-side reflex state someday does not make marines swarm units.
Ownership: **Simulation drives swarm-unit lifecycle** (spawn, reset,
serialize) exactly as it owns the unit list; **GameMap stores the arrays**
(fork 2a).

Scope honesty vs §B3: this chapter pilots B3's *storage layout and
determinism story only*. Not piloted here: Python-Unit-as-view, velocity,
EnvironmentProfile/coupling-row porting, obs kernels. The RL arc must not
treat B3 as done.

## 2. State layout

Arrays are born `(N_env, max_units)` per §A rule 1 (N_env = 1 today),
allocated once through the residency path (§A rule 3), written in place.
`max_units` is config; the **digested and recorded extent is
`[0, high_water)`** per env (high_water = highest slot ever spawned, itself
synced), so capacity is a pure knob that moves no golden.

Per-unit synced state (v1 roster — additions bump the section version, §6):

| Array | dtype | Meaning |
|---|---|---|
| `pos_x`, `pos_y` | int32 Q16.16 | position in tile units (Q16 from birth, per issue #63) |
| `heading` | int32 Q16.16 | radians, canonically wrapped to (−π, π] at every write (one representation per direction — digest-stable); movement via `sin_q16`/`cos_q16` |
| `hp` | int32 Q16.16 | hit points |
| `ai_state` | int32 | RUN / EAT / COMA / DEAD (enum field like `life_state`; EAT is a state, not a timer — §4) |
| `T_prev` | int32 Q16.16 | own-tile temperature sampled last step — the temporal-sensing memory the tumble law reads (critique B2) |
| `dist_walked` | uint32 Q16.16 | cumulative path length, wraps mod 2^32 by definition (no UB; twin uses uint32). Render crawl wavelengths use power-of-two raw λ so phase is continuous across wrap |
| `unit_id` | int32 | persistent monotone id, distinct from slot; RNG/recorder identity across slot reuse |
| `faction` | int32 | present from v1 (B3 lists it; targeting arrives later without a section bump) |
| `valid` | uint8 | slot occupied (spawn sets, reclaim clears; §7) |

Per-env synced scalars in the same section: `next_unit_id`, `high_water`.

Species-table row (Q16.16 unless noted): `radius`, `speed`, `hp_max`,
`T_prefer`, `T_hot`, `T_ctmin` (+ `T_coma_hyst`, ≥1 count, against boundary
flapping), `T_lethal`, `tumble_rate_base`, `tumble_thermal_bias`,
`food_threshold`, `eat_start_prob`, `eat_continue_prob`, `eat_rate`,
`heat_damage_coeff`. **Load-time validation: `speed·Δt < 1 tile` is a hard
error** (the movement law's no-tunneling bound, §4).

**Fork 2a — RULED (Erik 2026-09-09): arrays live on GameMap** as
`gmap.swarm_*`. Honest cost note (critique M9): residency alloc, targeted
transfer lists, digest snapshot, and serializer are all `(h, w)`-shaped
today — `(N, max_units)` actor arrays need extension work in each; GameMap
makes that work smaller, not free. Alternative: a new store that replicates
those contracts itself. Recommendation stands: GameMap.

**[ERIK] Fork 2b — scale target** — now GROUNDED by the density study
(`docs/larva_density_study_2026-09-09/`, larvae drawn to scale on the real
UNHCR vessel: 420 m² walkable floor, 9 art rooms, median room 23 m²):

| larvae | reads as | ship packed |
|---:|---|---:|
| 250 | sparse | 10 % |
| 500 | present everywhere, harmless | 20 % |
| **1,000** | **infestation** — packs cargo hold + both hydroponics bays to 87 % | 40 % |
| 2,000 | nearly every floor tile | 79 % |
| ~2,500 | physical ceiling of this hull (1 per 1.5 tiles) | 100 % |

Erik's gut (~1k tops) matches the geometry. Erik 2026-09-10: several full
rooms must be allowed; "10k is definitely enough, 5k probably". Where the
real thresholds are (none in the sim): the kernel, memory (~360 KB at
10k), digest hashing and the CPU twin (~1 ms at 10k) are all trivial below
~1M; the ONE binding budget is **render** — instanced bodies at ~1–2k
triangles each put 10k larvae at 10–20M triangles per frame, the edge of
comfortable on an RTX 3070, so swarm bodies need an LOD row anyway.
Recommendation: design headroom stays at 100k (memory is a knob);
**`max_units` default = 8,192** — a power of two (clean block sizing),
above Erik's 5k, ~3× this hull's ceiling. The
default is load-bearing: the CPU twin is O(max_units) per tick regardless of
population, and recorder rings are `(capacity, max_units)` per attribute
(recording bounded to `[0, high_water)` and capped by config).

**[ERIK] Fork 2c — species-table regime** (critique M6; constrains your
tuning workflow, so genuinely your call): (a) registry-style, hash-pinned at
match setup, no hot reload — determinism-pure, tuning-hostile; or
(b, recommended) **CFG-style F5 hot reload** (`[swarm.larva]` in
config.toml), with the table hash captured per run and a mid-run reload
invalidating the replay from that tick, like any config change. Fire-12
just demonstrated how much creature tuning wants live dials.

## 3. The RNG kit — new canon, pays an old debt

**Philox-4x32-10** (Salmon et al., SC'11): a stateless keyed bijection
`out = f_key(counter)`. Integer-only (32×32→64 via `__umulhi` on device —
the `mul128_shr_signed` portability trick).

- **Key (2×32)** = `(match_seed_lo, match_seed_hi ^ unit_id)` — the match
  seed (the same seed material `sim.rng` is built from; match-setup
  material) enters every stream (critique B1). Same seed + same id ⇒ same
  stream; new match ⇒ new behavior.
- **Counter (4×32)** = `(tick, draw_index, stream_salt, 0)` — the salt
  names the purpose (tumble / turn / eat / spawn).
- **`draw_index` is never stored**: it is local to `(unit, salt, tick)`,
  restarts at 0 each tick, and is assigned by static code position. A unit
  consuming a variable number of draws per tick is therefore safe by
  construction — no shared or persisted counter exists to diverge
  (critique m1).
- Output mapping: uniform Q16.16 by integer shift of the 32-bit word; the
  turn-angle draw multiplies by a checked-in 2π Q16 constant (the door-2
  idiom). Never through float.
- Home: **`cpp/src/philox_q16.h`**, `FP_HD`, included by both twins
  (`fixed_point.h` stays the arithmetic kit). This is an **amendment to
  chapter 14's door 4**: Philox joins `sim.rng`'s PCG64 as the second
  sanctioned raw-draw source, with its own case-log entry. Sequenced
  independently of `sim.rng` — device draws never desync host draw counts.

The kit is also the owed deterministic sampler from the stats redesign (the
Ada/LAPACK incident): the spawn-stat work adopts it, not a second sampler.

## 4. Pilot behavior: larvae

Erik's spec (random-walk until collision; food density per painted area;
stop-and-eat; flee heat; freeze when cold), mapped to citable biology:

- **RUN:** advance `speed·Δt` along `heading` (integer sequence in §4.1).
  Tumble draw each tick: `u < tumble_rate`, where the rate *drops* while
  `T_now` moved toward `T_prefer` since last tick (the `T_prev` comparison —
  temporal sensing, Berg & Brown, *Nature* 1972: gradients are climbed by
  modulating run length, never by computing a gradient) and *rises* when
  moving toward danger — **heat escape emerges from the same mechanism as
  food-seeking**, no separate state (Luo et al., *J. Neurosci.* 2010 for the
  turn-direction bias; Klein et al., PNAS 2015).
- **EAT (a state, not a timer):** in RUN, on a tile with
  `food ≥ food_threshold`, enter EAT on `u < eat_start_prob`. Each EAT tick:
  decrement the food cell by `eat_rate` (order-free integer `atomicAdd`,
  allowed to undershoot below zero; a follow-up clamp pass zeroes negatives —
  deterministic, mildly non-conserving, and stated as such per critique M1),
  and remain on `u < eat_continue_prob` (memoryless geometric duration — no
  countdown state). v1 eating nourishes nothing (no satiety state) — a
  deliberate cut, revisit with growth/molting.
- **COMA:** `T < T_ctmin` enters (chill coma — MacMillan & Sinclair,
  *J. Insect Physiol.* 2011); exit at `T > T_ctmin + T_coma_hyst`.
  `T < T_lethal` → DEAD.
- **Heat damage** (fork 7b): applied after the move, so fleeing can outrun
  the burn — feel-adjacent, HUMAN-TEST at tuning time.

### 4.1 Movement law — the exact integer sequence (this IS the twin spec)

Per tick, per valid unit (critique M3):
1. Sample `T_now` at `tile(pos)`; run state transitions (COMA/DEAD/EAT
   checks, tumble draw) using `T_now` vs `T_prev`.
2. If moving: `step = mul_q16(speed, dt_q16)` (validated < 1 tile at load);
   `target = pos + (cos_q16(heading)·step, sin_q16(heading)·step)`.
3. Probe `tile(target)`, coordinates clamped to the grid (**out-of-grid
   reads as solid**). Vacuum tiles are walkable (v1 larvae don't breathe —
   stated, not accidental). Diagonal corner-cutting between two orthogonally
   solid tiles is **accepted** for soft-bodied larvae (documented choice).
4. If probe is solid: position unchanged this tick; draw a new heading
   (turn-biased away from the wall side); move next tick. No sliding.
5. Else commit `pos = target`; `dist_walked += step` (uint32 wrap);
   `heading` unchanged unless the tumble draw fired, in which case the new
   heading (gradient-sign-biased draw) applies **next** tick.
6. Apply heat damage from `T_now` (fork 7b law); write `T_prev = T_now`.

## 5. Tick integration & orchestration

**Placement (critique B3, settled): the swarm-unit step runs at the END of the
physics chain inside slot 7** — the last stage of `PhysicsEngine`'s
orchestration (resident: last kernels before the once-per-tick D2H, which
then carries larva arrays + food with the fields for free; CPU backend: the
twin at the same point in `PhysicsRunner.step`). There is no host conductor
line and no mid-tick transfer.

- **Input snapshot contract:** larvae read this tick's post-solver
  temperature/fields, and **last tick's `solid`/`material`** — wall
  destructions at slots 9/9b reach larvae one tick late, on both backends
  identically (the same lag every solver has). This is a written, scoped
  exception to 02's freshness clause, and it is what makes "bit-identical by
  construction" true of the *inputs*, not just the arithmetic.
- The 9c exchange-ordering comment block in `simulation.py` gains one line
  noting larva damage happens in slot 7, before unit damage — amend the
  comment, it is the canon for within-tick order.
- If larvae ever eat *gas*, that flux rides the 9e(d) pump primitives + the
  gas-energy seam with a named books channel — R6 stands.
- Kernel discipline: per-unit independent work; writes to own slot only;
  world writes only as order-free integer atomics; launch geometry never
  leaks into results; the CPU twin is a sequential loop over slots, `.cpp`
  on the `/fp:strict` list.
- Field access: larvae gather raw field cells in-kernel (the combustion
  pattern). The sensor accessor remains the *entity* seam — this is its
  device-side sibling, not a bypass (critique m7, answered).

## 6. Determinism gates, recorder, save

- **Digest:** swarm-unit synced state joins as presence-gated section
  **`SWARM_SECT_V1`** (carrier semantics copied exactly from
  `__entity__`; carrier key `__swarm__`; presence = any slot ever
  spawned this run). Honest accounting (critique M7): **the section itself
  moves no golden; the food field does** — a mutable food field is a new
  `DIGEST_FIELDS` row ⇒ spec v6 ⇒ every golden regenerates in that commit
  (the v5 precedent). **Sequencing rule: the digest bump lands only after
  fire-12's pending re-baselines merge, from a rebased branch** — two
  branches regenerating goldens concurrently is a blessing hazard.
- Digested extent `[0, high_water)`; `next_unit_id` and `high_water` are in
  the section (id reassignment after load would fork every RNG stream).
- **Gate shape:** clone the fire solver's 3-part CUDA gate (forced-branch
  fuzz byte-identity on all mutated arrays incl. `T_prev`; a lockstep
  trajectory asserted to drive every state transition incl. COMA hysteresis
  and slot reclaim; golden reproduction on the CUDA build's CPU path).
- **Recorder:** per-larva arrays additive + presence-gated, int32 idiom, raw
  Q16 (no casts); recorded extent `[0, high_water)` with a config cap.
- **Save/load:** swarm-unit arrays + section scalars + the species-table hash
  join the (designed, unbuilt) 02 save contract.

## 7. Lifecycle & world-coupling scope (v1)

**Lifecycle:** fixed capacity + `valid`; **no compaction** (reordering
breaks per-slot digests and RNG keys); **never an atomic ticket counter**
for spawns (the one provably order-nondeterministic pattern — Safari et al.
2022). Spawn v1 = load-time zone-roster idiom: a count + seeded in-tile-set
scatter drawn from host `sim.rng` before tick 0, slots filled sequentially,
`unit_id` assigned in zone/row order. In-sim births (nests, later): k-th
request pairs with k-th free slot via prefix-sum. **Death: `ai_state=DEAD`
and `valid=0` in the same step (slot immediately reclaimable), plus a tick
event carrying (unit_id, pos, heading, species)** — the render husk is
driven entirely by the event, never by the freed slot (critique M2: `valid`
= occupancy, `ai_state` = behavior; no husk timer in sim state).

**Fork 7a — RULED (Erik 2026-09-09): v1 = walls only**;
larvae overlap everything. Crowding later = commutative per-tile integer
counters.

**Fork 7b — RULED (Erik 2026-09-09): yes, and as simple as possible** — the
law reads own-tile `temperature` ABSOLUTELY; v1 form = a single threshold
(`T > T_lethal_hot` ⇒ DEAD, no drain coefficient; `hp` stays in the roster for
future targeting). Honestly framed
(critique M5): this is a **NEW coupling row**, not a device copy of the 9c
unit row. The unit law reads per-tick radiant `heat` deposits
(max-over-footprint); the larva law reads absolute own-tile `temperature`,
linear over `T_hot` — a *contact/ambient* law. A marine and a larva in the
same hot room are deliberately damaged by different physics; the coupling
table documents both rows side by side. Larvae die to fire — half the fun
of a flammable nest.

**Out of scope v1 — the complete list** (critique M4), each a staged
extension, none a trap: stamping (`dyn_permeability`/`dyn_light_atten`),
vision presence, **blast/overpressure damage and wave push** (a grenade in
the brood room kills nothing in v1 — known, temporary), **being
shot/targeted** (faction column exists, targeting doesn't), statuses,
larva-larva interaction, gas eating/breathing, per-segment damage,
satiety/growth. **The closure mechanism for all CPU-side queries is one
device-written per-tile occupancy/count field** riding the standard D2H,
which vision/targeting/zombie-AI then read like any field — no per-tick
position round-trips ever.

**Fork 7c — RULED (Erik 2026-09-10): one larva is one point** — a
single-tile sim anchor; the segmented body is render-only (R10). The 10 m
specimens are CPU units by Erik's earlier ruling.

## 8. Food density

- Authoring: painted sidecar (`food.npy`, zones idiom, written only by
  `level_lib`) — paint cafeterias, warehouses, the garden. No "room"
  concept exists; painted areas are exactly Erik's per-area intent.
- **Fork 8a — RULED (Erik 2026-09-10): food is a real per-tile synced
  field, editable, and larvae deplete it.** Details:
  - Resolution: per tile (coarser gains nothing once it is a field — the
    kernel reads one cell either way).
  - Value range 0–255 (int8 semantics, a few discrete levels). Storage dtype
    is decided at implementation by path reuse, not size: the recorder has
    only bool / int64 / float32 ring branches, so a uint8 field would be a
    NEW dtype class (its own ring branch, per the recorder rule) — int32
    storage rides the existing paths and costs 24 KB on this grid.
  - Three writers, each through its lawful seam: (1) level load from the
    painted sidecar; (2) **gameplay events add food** (Erik: corpses,
    broken food crates, …) through the FieldEdit queue at slot 6b like any
    event→field write; (3) the larva kernel depletes (order-free integer
    `atomicAdd`, undershoot + clamp pass, §4). Dead larvae adding food is
    biologically apt (larval cannibalism is common) — an optional row.
  - Behavior tie-in (Erik): dwell longer where food is dense —
    `eat_continue_prob` scales with the cell's food level, so even without
    depletion larvae linger in pantries; with depletion they also move on
    when a cell is eaten out.

## 9. Render seam

Positions/heading/dist_walked/ai_state ride the slot-7 D2H; render reads the
mirror at `upload_state` (may be one tick stale — lawful). Bodies are
`larvagen`-style meshes, instanced per species; **v1 swarm bodies animate by
the traveling crawl wave only** (phase `k·(s·L − dist_walked)`, sim-driven,
replay-identical). The fauna-spike's follow-the-path spine (per-larva path
ring buffers) is **not** in the v1 swarm renderer — at swarm scale it is the
frame budget (critique M8); it remains the law for large CPU specimens.
COMA renders as stillness. **DEAD — persistent corpses (Erik's ask, §9
proposal):** no decal / persistent-marks layer exists in the renderer today
(checked 2026-09-10). Proposed **world decal layer**: one persistent
world-space RGBA render target the size of the world RT, never cleared
within a match; on a death tick event the renderer stamps a baked corpse
sprite (the larva body rendered once, top-down, at its heading) into it;
the world composite blends it over the floor. Unlimited corpses at the cost
of one texture, zero sim state (render-only, R10; deterministic because the
death events are), and the same layer serves **bullet marks, blood pools,
soot/scorch marks** — a canonical-row candidate for chapter 09 (rendering),
not a fauna-private system. Corpses as *food* (fork 8a) is the sim-side
half of the same event. The tile inspector gains larva-count-on-tile via
its existing seam (bincount over the downloaded positions — fine at 100k).

## 9b. What is shared vs per species (the reuse contract)

Every swarm species is *"sense some fields → small state machine → move"*;
species differ mainly in **which fields they sense**. Larvae: temperature +
food. Moths (later): the light field. A zombie horde: a scent/noise field.
A fish school: water depth + a per-tile crowding counter (the one later
extension — neighbour interaction — rides the same store).

**Shared foundation — written once (`cpp/src/swarm.h` + `.cu`/`.cpp`,
`src/simulation/swarm.py`):** the SoA store + residency, `SWARM_SECT_V1`,
recorder/save integration, spawn/reclaim lifecycle, the Philox kit, the
wall-probe movement step (§4.1 steps 2–5), the field-sampling primitive
(head-sweep: compare a field ahead-left vs ahead-right of the heading),
death → tick event → decal/husk, the instanced render bridge with LODs.

**Per species — one file each (`swarm_larva.cu` + `swarm_larva.cpp`
twin):** the state-machine kernel, a species-table row (`[swarm.larva]`),
a body generator with LOD levels, and optional *extension arrays* when the
core schema (§2) lacks state that species needs (fish: velocity). The core
schema is fixed for all species; extensions are per-species columns in the
same section, so adding a species never bumps another species' digest.

Python side: one small class Simulation drives for spawn / reset /
serialize (like the unit list) — the only "class" in the design.

## 10. Open rulings requested (summary)

| # | Fork | Recommendation |
|---|---|---|
| 2a | Arrays on GameMap vs new store | **RULED: GameMap** |
| 2b | Design scale | **RULED: 100k headroom, default 8,192** |
| 2c | Species table: hot-reload vs hash-pinned | **RULED: CFG hot-reload**, hash per run, reload invalidates replay |
| 7a | v1 collision | **RULED: walls only** |
| 7b | Heat damage | **RULED: yes — single absolute-temperature threshold kill, a new coupling row** |
| 7c | Body in sim | **RULED: single-tile anchor, body render-only** |
| 8a | Food mutable | **RULED: per-tile synced field, 0–255 levels, events add / larvae deplete** |

Plus the lasting name: **"swarm-unit"** (vs "swarm unit" / "microunit").

## 11. Level-editor integration & build order

**Editor integration (two registry-driven pieces, "the registry IS the
editor"):**
- **Food paint layer** — a painted sidecar `food.npy` (the zones/water
  idiom) written only by `level_lib`; the map editor gains one paint tool
  (value 0–255, few levels) as a pure core + thin shell like every editor
  feature.
- **Swarm spawn zone** — a registry entity type (e.g. `swarm_brood`: species,
  count, seed) placed like any zone entity; at load the zone-roster idiom
  scatters that many larvae inside the painted zone. Nests as flammable
  props (the #60 system) are a later, separate row.

**Build order (prerequisites first):**
1. **Philox RNG kit** — its own tiny shared package (C++ header + Python
   twin, gated on Random123 known-answer vectors; civulator consumes it
   too), then `cpp/src/philox_q16.h` adoption in breach + the chapter-14
   door-4 amendment. Nothing else starts before this is green.
2. **Swarm store scaffold** — SoA `(N, max_units)` on GameMap, residency,
   `SWARM_SECT_V1`, recorder/save, spawn/reclaim, with an *empty* species
   (no behavior) — gates: existing goldens byte-identical (presence-gated),
   A/B harness extended.
3. **Larva kernel + CPU twin** — §4 state machine with head-sweep sensing on
   temperature only (no food yet), §4.1 movement, 3-part CUDA gate.
4. **Food field** — sidecar + editor paint + FieldEdit writer + kernel
   depletion; **digest spec v6 = all goldens regenerate — sequenced AFTER
   fire-12's re-baselines merge, from a rebased branch** (§6).
5. **Render bridge** — instanced low-poly larvagen (50–300 tris) + LODs +
   crawl shader; corpses via the world decal layer (chapter-09 row; if it is
   not built yet, v1 fallback = event-driven fading husk).
6. **Playground** (Erik, for arc close) — a brood + painted food on the
   playground level so it shows a bit of every system in the game.

Independent work that can go first or in parallel (own branch): the world
decal layer (rendering, serves bullet marks / blood / soot too).

## Systems

**Uses (existing canon):** GameMap + residency path (02, §A) · fixed-point
kits incl. trig · PhysicsEngine orchestration + bindings · field digest /
GOLDEN_AGGREGATE / A/B harness / cuda_harness 3-part gate · recorder
additive contract · level_lib (food sidecar, spawn rosters) · tick events ·
coupling table (new larva row) · species-table pattern · hover readout seam
· propgen/larvagen + sway machinery (render).

**Creates (draft canonical rules, into CLAUDE.md at implementation):**
- *Swarm store* — SoA `(N, max_units)` + species table + kernel/twin
  pair; a species is a table row + kernel, never a class; stable slots, no
  compaction, no atomic-ticket spawns; digested extent `[0, high_water)`.
- *Philox device RNG kit* (`philox_q16.h`) — THE deterministic per-agent
  RNG; key carries the match seed; draw_index never stored; second
  sanctioned door-4 source (14 amended); all future device randomness and
  the owed spawn-stat sampler use it.
- *SWARM digest section* — presence-gated, section-versioned; per-unit
  synced state never joins `DIGEST_FIELDS` directly.
- *Food field* — painted-sidecar-initialized synced field; writers = level
  load, FieldEdit events (corpses, crates), larva kernel depletion — no
  fourth.
- *World decal layer* (proposed, chapter 09 row) — the one persistent
  marks texture: corpses, bullet marks, blood, soot; stamped from tick
  events, render-only.
- *Occupancy field* (when targeting/vision arrive) — the one lawful bridge
  from swarm-unit positions to CPU-side queries.

## Papers (docs/papers/, per the credit rule)

Archived: `richmond2023_flamegpu2.pdf`. To archive with the first
implementation commit: Salmon et al. SC'11 (Philox), Widynski (Squares,
documented fallback), Luo et al. 2010 (larval thermotaxis). URLs in the lit
brief.

## Review log (Erik reads the chapter bit by bit; nothing is blessed until the log closes)

**2026-09-09 — Erik's pass 1, §1–§7b. Resume at fork 7c.**

Ruled: 2a GameMap · 7a walls only · 7b yes, simplest form (absolute tile
temperature, threshold kill). Tentative: 2c CFG hot-reload *if* it costs no
performance (answer: it doesn't — a reload is one tiny H2D of the species
table; kernels read it from constant memory).

Open for discussion (settle before blessing):
- **Sensing model — temporal (`T_prev`) vs spatial gradient.** Erik's scene:
  a brood room catches fire and every larva streams out down −∇T. His
  objection to temporal sensing is the grid: tiles are 1/3 m and a larva
  needs many ticks to cross one, so within a tile dT/dt reports the *room*
  heating, not a direction — directional information only arrives at tile
  crossings. Claude's re-assessment: Erik is right, and the biology agrees —
  real larvae (Luo 2010) sense by *head sweeps*, i.e. spatial left/right
  samples, not bacterial temporal integration. Proposed v1: **spatial
  sensing** — compare the field one tile ahead-left vs ahead-right of the
  heading (the head sweep), bias the turn toward the better side, raise
  tumble rate when both are worse than here. Same primitive serves food
  (food gradient) and temperature; `T_prev` leaves the roster. Optionally
  keep temporal sensing as a species-table switch (`sense_mode`) for an
  A/B experiment — Erik's "twin classes" are one table row, not two
  classes — at the cost of gate coverage for both paths.
- **Fork 2b (scale)** — GROUNDED 2026-09-10: density study delivered (see
  §2 table; `docs/larva_density_study_2026-09-09/`). Erik's gut (~1k) =
  the infestation threshold; hull ceiling ~2.5k; proposed default 4,096.
  Side finding worth its own issue: the physics `tilemap.csv` does NOT wall
  the art's rooms — the `DDD` door rows at y=42/45/75/95 stand in open
  floor, so gas, fire, LOS and pathing see one open hall from row 42 to
  118 (only the bridge door and the 5-wide throat at rows 80–84 partition
  the ship). Clarified: the CPU
  twin exists for the determinism gates regardless; "GPU is a flag" only
  requires the CPU backend to run the same law, not to be *fast* at swarm
  scale. Training impact = memory like one small extra field per env
  (1k units × ~9 int32 ≈ 36 KB vs 64 KB for a 128² field) + an occupancy
  channel in the observation if agents must see larvae — "one extra
  pressure-like field" is the right mental model.
- **RNG kit as a shared package** (Erik): civulator needs the same
  cross-machine determinism → build Philox-4x32-10 as a standalone tiny
  library (header-only C++ + a pure-Python/numpy twin, gated against
  Random123's published known-answer vectors) consumed by both repos.
- v2 candidates Erik raised, all OUT of v1: water (float on `water_depth`?
  — Fable's water domain), pressure/blast, larvae generating (poison) gas
  (would have to ride the pump primitives + gas-energy seam, minted at
  ambient — a device-side inject path, not trivial).
- "Tumble draw each tick" clarified: one uniform draw per larva per tick;
  a new heading iff `u < tumble_rate` — run lengths come out geometric
  (the discrete twin of Berg's exponential runs), not a tumble every tick.

**2026-09-10 — Erik's pass 2 (partial, from phone). Resume at fork 7c
(question answered below; ruling still pending), then §8–§10 + Systems.**

- 7c clarified: "single-tile sim anchor" = in the SIM a larva is one point
  (one Q16.16 position, one tile of presence — the marine precedent: a
  point in the sim, a full 3D model in the render); the segmented body is
  drawn around that point by the renderer only. The alternative gives each
  body segment its own sim position (fire can burn the tail while the head
  is elsewhere) at digest + kernel cost.
- Bodies stay **3D** but Erik wants them much smaller: 10–50× fewer
  triangles than the spike's ~2–3k (i.e. ~50–300 tris: icosphere
  subdivision 0–1 per segment, fewer segments), and/or several LOD levels.
  Effect: 10k larvae ≈ 0.5–3M triangles — the render budget stops being
  the binding constraint at the ≤10k target.

**2026-09-10 — Erik's pass 3 (§7c–§9).** Ruled: 7c one point · 8a food =
per-tile field, 0–255 levels, editable by events (corpses, crates), larvae
deplete, dwell-probability scales with food. §9: Erik asks for persistent
corpses painted on the map alongside bullet marks / blood / soot → proposed
world decal layer (see §9; no such layer exists yet — a chapter-09
canonical row, not fauna-private). **Erik has further comments pending —
resume at §9 decal proposal → §10 naming → Systems.**

**2026-09-10 — naming RULED: swarm units.** File renamed to
`17_swarm_units.md`; §9b added (shared vs per-species reuse contract).
Erik's notes exhausted. Erik asks: level-editor integration? build order
(RNG first)? and, for arc close: apply swarm units + food to the
**playground** level so it shows a bit of every system.

**2026-09-10 — BLESSED.** Erik agreed to all remaining recommendations: 2b
default 8,192 · 2c CFG hot-reload · sensing = spatial head-sweep only in v1
(no temporal switch). Side finding (tilemap vs art rooms) is known to Erik —
no issue; his lean is regenerating the art or a new, more 3D asset system
later. **Graphics are on hold until RL is running** → step 5 of §11 (render
bridge) is deliberately minimal for now (existing instanced path, low-poly
bodies, husk fallback); the decal layer waits. The review log closes here.
