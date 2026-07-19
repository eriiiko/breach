# Boundary conditions — planetside AMBIENT ring (2026-07-19)

**Status:** DRAFT for Erik's review (physics close-out, priority ledger stack #1).
**Sequencing:** lands **BEFORE** the S8a residency build (Erik, 2026-07-19 — residency
freezes final kernel content; `cuda_s8a_residency_spec_2026-07-19.md` §5c).
**Sources:** Topic 4 survey (`notes_2026-07-17_topics_backlog.md`), A9 format hook
(`level_loader.py:468-533`), Erik's decisions this session (below).

**Goal:** simulate planetside levels as well as space. Planetside: outgoing pressure
waves are absorbed at the map edge (as if the atmosphere continued forever), and air
exchanges freely across the boundary — particles exit and enter; the outside is an
infinite ambient reservoir. Pragmatic simplest implementation; determinism is iron.

## 0. Decisions locked (Erik, 2026-07-19)

1. **Whole-map single boundary type** — the A9 `boundary` field (`"space" | "ambient"`).
   No per-edge modes; exotic cases are solved with level-design cheats.
2. **Ambient dials per level** — pressure/composition configurable, defaults Earth-like.
3. **No water boundary condition.** Ocean levels are built as a big indestructible
   reservoir filled at author time (the sanctioned cheat). AMBIENT ring tiles behave
   for water exactly as SPACE ring tiles do today. Recorded as a decision, not a gap.
4. **Wind-in-from-boundary is NOT a boundary mode** — it's a source term (FieldEdit /
   level source), a separate future feature. Out of scope here.
5. **Sponge band ships in v1 as a dial** (see §3) — Erik's wish is *perfect* absorption;
   a pinned ring alone partially reflects (sign-flipped, weakened echo), the sponge is
   the standard cheap fix and the cheat IS the proper technique. Default expected ON
   for planetside; Erik's eyes at the feel bake confirm.

## 1. The mechanism (symmetric to SPACE, all local per-tile edits)

Today: the literal grid edge is closed/reflective (`mirror_idx`, eos_solver.cpp + CUDA
mirror); "space" behavior comes from LEVEL DATA — the border ring of SPACE tiles →
`is_vacuum`, which the MG solve pins as Dirichlet P=0 and bulk transport treats as a
mass sink.

AMBIENT is the twin:

- **Ring representation:** reuse the SPACE tile code. At load, when
  `boundary == "ambient"`, the tilemap's ring tiles populate a new **`is_ambient`**
  bool mask instead of `is_vacuum` (one branch in `GameMap.__init__` after
  `materials_from_tilemap`). One level = one interpretation; no new tile vocabulary,
  no editor change.
- **MG solve:** generalize the exclusion-pin to carry a value — SPACE pins P=0,
  AMBIENT pins **P=P_amb** (one Q16 constant, plumbed like the existing pin).
- **Bulk transport:** each tick, after the flux/advection updates and before P
  materialization, **reset `gas[O2]`/`gas[INERT_N2]` at ring tiles to the ambient
  composition** — the infinite reservoir, both directions (outflow swallowed, inflow
  supplied). Traces reset to 0 (absorbed).
- **Temperature:** ring `temperature` reset to ambient (0 ΔT) — the outside is an
  infinite heat bath too.
- **Consistency rule (important):** the pinned P_amb and the reset (N_amb, T_amb) MUST
  satisfy the EOS identity `P_amb == C · N_total_amb · T_abs_amb` exactly in Q16.16 —
  quantize ONCE at level load (engine/14 ingress rules) and derive one from the other
  (dial is `p_amb` + `o2_frac`; N planes derived). Otherwise materialization and the
  pin fight, minting a standing artificial gradient at the ring.
- **Velocity at ring:** mirror whatever the solver does at `is_vacuum` tiles today
  (STEP-A audit item), with ambient values.

## 2. The `is_vacuum` consumer audit (the real STEP A)

Every `is_vacuum` read in sim code gets classified and, where needed, extended:

- **(a) "no gas here" physics reads** (e.g. fire's open-neighbour O2 count, seeding
  skips) → stay vacuum-only; ring tiles HOLD real gas (reset each tick) and should
  participate naturally where gas values are read.
- **(b) "off-map / no-entry" reads** (movement, spawn, targeting, unseal edge rules,
  FieldEdit skip masks) → become `is_vacuum | is_ambient` (units don't walk off-world).
- **(c) boundary-pin reads** (MG exclusion-pin, sink behavior) → become mode-valued
  (0 vs P_amb / sink vs reservoir).

The audit table (file:line → class → change) is a REQUIRED deliverable of the build,
reported before kernels are touched. `destroy_wall`'s edge-hull/`was_hull` rule and
`unseal_tiles`' vacuum-join rule are audit items, not pre-decided here.

## 3. The sponge band (the absorber)

- At load, for ambient levels, compute a static per-tile damping coefficient grid:
  integer BFS distance `d` from the nearest ring tile through open air; for `d < W`,
  coefficient `k(d) = k_sponge · (W − d)/W`, quantized once to Q16. Static per level —
  it joins the load-time caches (and later the S8a resident-set statics).
- Per tick, inside the EOS step where velocity updates: `u *= (FP_ONE − k(d))` — one
  Q16 multiply per sponge-band cell, both axes. **v1 damps u only** (that's the wave
  absorber); N/T relaxation toward ambient is a fallback dial if the bake shows
  drift/ringing the u-damping doesn't kill.
- Dials: `sponge_width` (0 == hard ring), `k_sponge`. Determinism-clean by
  construction (static integer coefficients, pure Q16 muls, no libm).

## 4. Level format (backward compatible)

- `boundary = "ambient"` (A9 field, semantics now live).
- Optional `[ambient]` table: `p_amb` (atm, default 1.0), `o2_frac` (default 0.21),
  `sponge_width` (tiles, default TBD at bake — spec placeholder 6), `k_sponge`
  (default TBD at bake). Absent table = all defaults. `[ambient]` present with
  `boundary = "space"` is a hard error (path-bearing message, loader style).
- All values quantized once at ingress. `air_init.npy` composes as today (authored
  interior override; ring rules win at ring tiles).

## 5. Conservation bookkeeping (open system, by design)

The ring deliberately breaks global mass/energy conservation — the map is open.
- Existing exact-conservation guarantees are untouched on all existing maps (no
  AMBIENT tiles exist in them; the new branches never execute — **zero re-baseline**).
- New AMBIENT-map tests assert INTERIOR behavior, never global mass.
- **Counted rail (P5 spirit):** `boundary_flux` — per-tick net N added/removed by the
  ring reset, accumulated per run and exposed like the existing rail counters. The
  exchange is watched, not invisible.
- Free emergent corollary (document in canon at fold): O2 replenishes at the ring —
  outdoor fires near the boundary don't suffocate.

## 6. CUDA lockstep + gates

BC lands on BOTH paths in one patch (CPU reference + CUDA kernels:
`cuda_mg_solve` pin branch, bulk-transport reset, trace absorb, sponge multiply —
same-tick lockstep, the P6 pattern; this precedes S8a so kernels are final before
launch-core extraction).

- **E2E gates (new AMBIENT fixture level):**
  1. Sealed planetside room holds equilibrium (interior trajectory flat).
  2. Breach to ambient → air rushes IN, room recovers toward P_amb (the inverse of
     the space money-shot vent).
  3. Wave absorption: detonation echo amplitude at a probe tile, sponge ON vs OFF —
     ON must cut the reflected amplitude below a pinned threshold.
  4. O2 replenishment: sustained fire near the ring keeps burning.
  5. Rail: `boundary_flux` matches the interior mass delta exactly (LSB-level) —
     conservation holds once the ring exchange is counted.
- **Lockstep gate:** CPU vs CUDA A/B trajectory on the AMBIENT fixture, all synced
  fields byte-identical (tol 0), per the `cuda_*_check` pattern; new golden committed
  for the fixture. Existing space goldens must come out untouched (assert in CI by
  running the standard digest suite).
- Full suite green: `pytest tests -q` (conda env `data`).

## 7. Scope discipline

- No solve-structure change; per-tile edits in existing kernels only. If the STEP-A
  audit or the MG pin generalization balloons, STOP and report (the survey says it
  won't — trust but verify).
- Feel-adjacent: the sponge default + k values are a HUMAN-TEST item (Erik plays a
  planetside fixture). Mechanical parts (audit, pin, reset, rail) are digest-gated.
- Out of scope: per-edge modes, boundary trace species (toxic atmospheres — future
  `[ambient]` extension), water BC (ocean-reservoir cheat), wind-in (source term),
  rain/weather, render treatment of the ring (render-layer, deterministic-exempt).

## 8. Relation to the Arc B pump (concept unity — Erik, 2026-07-19)

The ring reset and the Arc B pump N-feed are the SAME concept at two extremes:
the ring is a degenerate pump (unconditional, infinite-rate N-clamp to a setpoint,
static, level-authored, in-kernel); the pump is the dynamic sibling (rate-limited,
signal-driven, entity-owned, placeable — a life-support vent is a wireable,
destroyable piece of ambient reservoir, and is how a SPACE map gets free particle
exchange without an ambient boundary). BC deliberately ships the in-kernel form
(lands pre-Arc-B, pre-residency; ring tiles as entities would be pure overhead for
identical physics). Arc B should build the pump knowing this kinship — same
setpoint/composition vocabulary, same `boundary_flux`-style counted rail.
