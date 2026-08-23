# Vent system — design v2 (2026-08-23, post-critique)

**Status**: design v2 — adversarial critique pass DONE (two independent
critics, physics/determinism + architecture/scope), all confirmed findings
folded in. **Issue**: #48 (siblings: #43 haze venting, #4 drag law, #6 smoke
taste; deferred → #49 states/conditioning, #50 auto-placement).
**Depends on**: #4's measured quantization dead-zone threshold (bench folded
into this arc, §7). **Lifecycle**: capture doc — durable rules move to the
project CLAUDE.md canonical-systems table at implementation; archives at arc
close.

Designed Erik + Claude 2026-08-23; rulings inline are Erik's. v2 changes are
marked ⟨crit⟩.

---

## 1. Motivation

The ship should have air *movement* during normal play. Smoke/gas drift is the
visual instrument for the whole atmosphere; today it only moves when something
explodes. Vents give a permanent background circulation, subsume #43's
lingering-haze fix (an intake vent IS the sustained continuity wind toward an
opening), and unlock a mission-mechanic family (poison the ducts, sabotage
life support, deplete reserves).

## 2. The entities

⟨crit⟩ **Two registry rows, not one** — the duct itself is an entity:

**`duct`** — an `INTANGIBLE = True` registry entity (no tile; the pattern of
`breach_site` / the logic nodes). Authoring fields: reserve size (real units,
quantized at load), filter row name (loader-validated reference), tuning.
Runtime state (= its `runtime_digest_rows`): the plenum — see §5. This gives
ducts a serialization home, loader validation, and an inspector pane for free;
`duct_id: int` (v1 draft) is replaced by a real reference.

**`vent`** — fields:

| Field | Type | Notes |
|---|---|---|
| `mount` | enum `floor` \| `wall` | floor = on an open tile (canon: ceiling/floor — the top-down sim can't tell); wall = anchored to a wall tile + `facing` |
| `facing` | dir enum | wall mounts only |
| `duct` | `KIND_ENTITY_REF` | the duct entity this vent belongs to (day-1, ruling #2) |
| `role` | enum `supply` \| `return` | editor-assigned, never self-negotiated (ruling #3) |
| `q_circ` | real units/s | circulation throughput; quantized to the per-tick quantum ONCE at load (door-2 / pump `rate` idiom — never a raw per-tick Q16 in the editor) |
| `q_makeup_max` | real units/s | rate limit on the makeup term (schema-reserved; the controller itself is patch 3, §7) |
| `state` | enum | v1: always `open`. Reserved so #49 adds `closed / damaged-open / welded-shut` without migration |

Vent runtime rows: `P_ctrl` (Q16), the flux error accumulator (Q16).

**Aperture model** — the single mechanism. The sim has no sub-tile gas
geometry: fields are tile-resolution, so every vent, however mounted, reduces
to *gas-mass deltas on one open tile* (the aperture). Floor mount: aperture =
own tile — the physically correct model, not a shortcut (a ceiling jet is
out-of-plane; an isotropic source/sink is its honest 2D projection). Wall
mount: aperture = the open tile in front of the face. One code path; a helper
resolves placement → aperture. Schema stores an aperture *tile list* (v1
always length 1) so wide vents later are additive.

**Directionality for free**: even mass-only, a wall vent is directional — the
wall blocks backflow, so induced wind is biased away from the face. No
momentum injection needed for readable drift direction.

**Multi-tile / wide vents** (settled, not load-bearing): v1 covers them by
**adjacency composition** — several vents on neighboring tiles/faces sharing
one duct; the plenum ledger unifies any number of vents per duct, adjacent
sensors agree by construction, flux adds. Zero machinery. A *true* wide-vent
entity (one sprite, one controller sensing the aperture-mean with fixed
integer rounding, flux split across the aperture list by fixed weights) is a
later additive patch riding the reserved tile-list field — never a migration.

## 3. Flux mechanism

⟨crit⟩ **The write path is the pump primitives, not FieldEdit.** The repo
already has this system class: `pump_system.py` (Arc B B4) sweeps entity
actuators at conductor slot 9e(d), injecting/extracting gas mass via the
integer-native GameMap primitives `inject_gas_n` / `extract_gas_n` — no RNG,
no float, no dequantize. `extract_gas_n` already implements our intake
verbatim (proportional removal across ALL slices, pinned-order integer
shortfall cascade, zero-clamped, returns the withdrawn total). The v1 draft's
FieldEdit route would have been a parallel copy of this — the exact iron-rule
failure. Vent flux therefore = **extended pump primitives**:

- `extract_gas_n` grows a variant returning the **per-slice withdrawal
  vector** (to credit the plenum composition and apply the filter);
- `inject_gas_n` grows a variant taking an **arbitrary composition vector**
  (not the fixed 21/79 ambient split) **plus the T-mix write** (§4).

These extensions are named as new systems in §Systems. FieldEdit remains the
canonical write for *event*-shaped edits (payload executor); per-tick entity
actuators are the 9e pattern — that seam distinction is now stated, not
implied.

⟨crit⟩ **Conductor slot: 9e(d), a sibling of `sweep_pumps`** — no new
numbered slot. `build_vents` at reset (like `build_pumps`); sweep gated on
`self._vents`, iterating in **entity-ordinal order** (the serializer's own
order) so all duct arithmetic is machine-identical. The 2-tick field-effect
contract applies (edits land after this tick's physics, materialize next
tick), and sensing reads the pre-edit `atmosphere` — the honest reading, per
the pump docstring. This *answers* v1's open question 2: there is no
"between EOS steps 1–5" — steps 0–5 are one C++ call; 9e(d) sits cleanly
before next tick's step 1, where injected N is ordinary initial state to the
donor-cell transport (conserves whatever it's given), p* sees it once, and
compression work acts only on div(u) — no double-count path exists.

**Two flux terms, not one** (real HVAC separates recirculation from makeup):

1. **Circulation term** (always on): ⟨crit⟩ *intakes first, then distribute.*
   Per duct per tick: every return vent extracts up to its `q_circ`
   (zero-clamped by the aperture's actual content — `extract_gas_n`
   semantics); the duct then distributes `min(Σ actual intake, Σ q_circ)`
   across its supply vents by fixed integer weights, Bresenham remainder
   carried in the plenum. "Balanced by construction" was v1 wishful thinking
   — a return over a thin room under-delivers; with intakes-first the
   imbalance lands *visibly in the plenum ledger* instead of silently
   draining the reserve. Circulation never touches the reserve — now true by
   bookkeeping, not by hope.
2. **Makeup term** (P-controller, bounded, drawing on the finite reserve):
   corrects filtered pressure toward the 1 atm setpoint, clamped to
   `±q_makeup_max`. Overpressure = the same term negative. ⟨crit⟩ **Moved to
   patch 3** (§7): it carries the one underived number in the design (the Kp
   stability bound, old Q4) and the reserve gameplay; it is not needed for
   v1's demo (ambient drift + scrubbing + poison recirculation all come from
   term 1). Schema reserves its fields now; no migration later.

**Sensing**: each vent reads its own aperture tile — no compartment graph
(topology is dynamic; a room graph goes stale at the first breach; the field
does the spatial averaging). ⟨crit⟩ Open choice flagged by critique: raw
`gmap.atmosphere` read at 9e (the pump precedent, legal for actuators) vs
registering apertures in the sensor accessor's `SiteIndex` and reading
`Channel.PRESSURE` (residency-proof — S8 won't have to unpick it).
**Recommendation: the accessor** — cheap now, and vents are exactly the
"static tiles read every tick" case SiteIndex was built for. Decide at patch
1 review.

Filtered by an exponential moving average, power-of-two α, pure shift
arithmetic, exact in Q16:

```
P_ctrl += (P_meas - P_ctrl) >> 4      # α = 1/16;  P_ctrl[0] = 1 atm
```

⟨crit⟩ **Known EMA facts** (they shape the patch-3 deadband): arithmetic
shift truncates toward −inf, so equilibrium is the stall band
`P_ctrl ∈ [P_meas−15, P_meas]` raw — a persistent ≤15-raw positive bias in
the error. And the solver's nominal-map P is not exactly 65536 raw everywhere
(MG residual; the frozen gate measured worst-dev ≈ 430 raw). Therefore the
makeup deadband is a **derived bound, not a tuned one**:
`deadband ≥ 16 raw + measured nominal-map P dispersion`, and the error
accumulator **zeroes whenever the error is inside the deadband** (else a
stale sub-quantum balance emits on the next micro-excursion). Only with both
is "provably quiescent at 1 atm" a theorem. Also honest: α=1/16 passes ~half
of a 10-tick transient into P_ctrl — the real shockwave protection is the
`±q_makeup_max` clamp, not the EMA.

**Cadence**: everything every tick inside the 9e(d) sweep. No modulo
scheduling; the EMA constant alone sets the slow timescale.

**Quantization**: per-vent Q16 **error accumulator** — accrue owed flux each
tick, emit whole quanta on crossing (the Bresenham pattern from the
fixed-point arc). Kills the tiny-flux underflow cliff (same class as #4's
dead zone). The primitives take integer quanta; the accumulator is what makes
per-tick emission smooth.

## 4. What flows — composition, temperature, filters

- **Intake** (return vent): `extract_gas_n` removes proportionally across all
  species — bulk (`o2`, `inert_n2`) and trace planes alike (a grille can't
  suck only clean air). ⟨crit⟩ **The ledger books the MEASURED per-slice
  withdrawal vector the primitive returns — never the intended amount.**
  Clamps, floors, and skip-vetoes then can't desync the books (intended ≠
  applied happens routinely at the LSB and wholesale when a guard fires).
- ⟨crit⟩ **The plenum's bulk store is a PAIR (o2, n2), not one scalar.** v1's
  anonymous "bulk N" was an oxygen fountain: a fire-vitiated room (5% O2)
  inhaled as anonymous N and re-emitted at ambient 21% would let the duct
  manufacture O2 and defeat suffocation gameplay. Intake credits each bulk
  plane separately; deposit emits at the plenum's own bulk ratio (fixed-
  rounding integer split, remainder carried plenum-side).
- **Filter**: the plenum filters intake per species — a **filter is a table
  row** (per-gas-id efficiency vector), global `[filters.<name>]` rows in
  config; per-ship-ness lives in the duct's loader-validated row *reference*.
  Physical anchor: real filters catch particulates (smoke ≈ HEPA) but pass
  gases — so one table gives both headline behaviours: **smoke is scrubbed**
  (counted mass sink → #43's haze gets a physical mechanism, rooms clear
  over ~a minute) and **poison recirculates** (a chemical grenade fed into a
  return re-emerges at every supply on that duct — mission mechanic from
  physics, symmetric, enemy ships included). Quality is data: rusty freighter
  0.4, military ~1.0, derelict all-zeros (ducts merely redistribute).
  ⟨crit⟩ **Scrubbed smoke KEEPS its heat** (decision): the filter removes
  mass, not energy — soot's thermal share stays in `E_plenum`, so deposits
  run slightly warm under smoky intake. Physically defensible (a clogged
  filter is hot) and one less counted channel; the R3 energy gate knows the
  smoke sink is mass-only.
- **Deposit** (supply vent): bulk at the plenum ratio + the plenum's
  unfiltered trace composition, via the extended `inject_gas_n`.
- **Temperature** (grounded in `eos_solver.h`: state = (N, T), T intensive,
  P derived once per tick from `p* = C·N_total·T_abs`):
  - **Intake writes nothing to T.** Removing a parcel leaves the remaining
    gas at the same temperature (a pumped-down hot room stays hot, thinner);
    the energy leaves implicitly as `ΔN·T_tile`, credited to `E_plenum`.
  - **Deposit mixes by the mass-weighted mean**:
    `T_new = (N_old·T_old + ΔN·T_dep)/(N_old+ΔN)`, `T_dep = E_plenum/N_plenum`.
    Weights sum to 1, so the delta-over-ambient representation mixes
    correctly in delta units (affine offset passes through; critic-verified)
    — and ambient reserve air legitimately carries E = 0.
  - ⟨crit⟩ **Integer discipline** (the v1 gaps): `E_plenum` is **int64 in the
    engine's raw N·T energy currency** (Q16.16² — the `bulk_transport.cpp`
    e-books convention; an int32 Q16 scalar overflows immediately). Both
    divides are **floordiv toward −inf** (`floordiv_q` idiom — truncation
    toward zero mints energy on sub-ambient mixes). `E_plenum` is debited by
    the **measured tile-side energy change** `(N_old+ΔN)·T_new − N_old·T_old`,
    which banks both division remainders back into the plenum automatically —
    conservation exact by construction, no counted leak. `T_new` is clamped
    to the [T_MIN, T_MAX_PHYS] rails with counter-tracked hits like every
    other T writer. **`N_plenum`** = the plenum's total bulk pair; below an
    N_EPS-style floor, residual E is wiped into a signed counted channel
    (`e_wipe` idiom) and deposits go out at ambient (T_dep = 0) — reserve
    depletion is a *designed* game state, the divide must survive it.
  - **Vents NEVER write P.** `atmosphere` is the EOS-derived alias; vents
    write species N and the T mix; pressure follows at the next EOS
    materialization (a direct P write would be clobbered at step 5 anyway).
  - ⟨crit⟩ **T-mix ownership guards**: `temperature[]` on `thermal_solid`
    tiles is owned by the TemperatureSolver (P-EOS ruling) — the T mix never
    writes there.
- ⟨crit⟩ **Aperture guards are RUNTIME, not just lint**: the vent no-ops
  (counted) when the aperture is solid, `thermal_solid`, `is_vacuum` or
  `is_ambient` (the bulk-transport step-4 vacuum-zeroing / ambient-ring clamp
  would destroy a deposit outside the ledger next tick), or flooded
  (`water_depth` over threshold — the W3 flooded seal; pumping into a sealed
  cell spikes pressure). Topology mutates mid-game (slot 9 wall deaths,
  explosion payloads); a load-time lint alone is insufficient — the lint
  (`level_airtight.py` host) catches authoring errors, the runtime guard
  catches battle damage. Measured-delta booking (above) makes the mass side
  self-healing regardless.
- ⟨crit⟩ **The conservation invariant is scoped to bulk + energy.** Trace
  planes are semi-Lagrangian, clamped, and decayed — non-conservative by
  construction; their plenum ledger is *telemetry + gameplay bookkeeping*
  (poison recirculation needs determinism, not conservation), unit =
  density·tiles. The R3 gate checks: bulk-pair field totals + plenum pairs +
  counted sinks/wipes = const, and the energy books likewise.

## 5. Duct network / plenum

Per duct entity, as runtime digest rows: **reserve pair** (o2, n2 raw Q16) +
**circulating-credit / Bresenham remainders** + **trace composition vector**
+ **`E_plenum`** (int64 raw N·T) + the filter row reference (authoring
field). Ducts are logic, not tiles — no duct geometry in the map; distant
rooms connect by reference.

**Reserve gameplay** (patch 3, with the makeup term): the makeup term only
draws the reserve — after a hull breach the system fights the leak, visibly,
then runs dry: the ship slowly loses the ability to repressurize. Missions
hang objectives on it.

⟨crit⟩ **Digest/serialization — the ENTITY_SECT route, no spec bump.** v1's
"join the digest, version bump, regenerate goldens" was wrong twice: it
invoked the field-plane membership rule for what is entity state, and it
forfeited the dormancy proof in the same commit that needed it. The correct
instrument exists: per-class `runtime_digest_rows` in `ENTITY_SECT_V1`
(`entities/serialize.py`) — **absence-transparent** (folds only when the
entity exists) and independent of `DIGEST_SPEC_VERSION`. Vent rows: `P_ctrl`,
accumulator. Duct rows: the plenum scalars (the serializer's row encoding is
already signed int64 — `E_plenum` costs nothing). A vent-free level is then
*byte-identical by construction*: no spec bump, no golden regen, dormancy
proof free and honest.

## 6. Placement doctrine

Stored here for the manual test map; the durable copy for automation lives in
#50 (references #27 levelgen + #48).

1. **Supply into cabins/rooms, extraction from corridors** (marine/spacecraft
   practice: rooms at slight positive pressure → smoke and smells always flow
   room → doorway → corridor → return grille). Dramatic payoff: doorways
   breathe, corridors go moody, rooms clear first.
2. **Never short-circuit**: supply and return far apart, so the current
   sweeps the space (the cardinal real-HVAC sin is placing them adjacent).
3. **Sealed rooms get a diagonal pair**: no return path through a sealed
   door, so the room gets its own supply + return at opposite corners (the
   honest version of the real-world door-undercut hack). When the door opens,
   the room's loop couples to the corridor's and the drift reorganizes —
   free, from the physics.

**Playground plan** (edit the existing playground, no new map — ⟨crit⟩
verified feasible: `levels/playground/level.toml` carries six door entities):
a diagonal pair in one sealed room + a supply-in-room / return-in-mid-corridor
pair where a door exists. Exact tiles chosen in the editor at the feel patch;
airtightness checked with `tools/level_airtight.py`, which also hosts the
blocked-aperture lint. Placement is editor data and iterates freely.

## 7. Patches, determinism, gates

⟨crit⟩ **Three patches, not two** — the makeup controller is not mechanical
while its stability bound is underived:

1. **Mechanism patch** (auto-merge on green): duct + vent entities,
   circulation term, plenum ledger (bulk pair + E + composition), filter
   table, extended primitives, 9e(d) sweep, runtime guards.
   Dormant-by-default: no vents in levels ⇒ byte-identical (ENTITY_SECT
   absence transparency, §5); digest-gated; corridor bench included.
2. **Feel patch** (HUMAN-TEST): playground placement + `q_circ` sizing —
   Erik plays it before merge.
3. **Makeup + reserve patch**: the P-controller with the *derived* deadband
   (§3 bound) and Kp stability/overflow analysis as its entry gate, plus
   reserve-depletion gameplay. HUMAN-TEST for the breach-response feel.

- Q16.16 only in the synced path; all arithmetic integer-native through the
  extended primitives (the EMA and accumulator are shift-and-add; the two
  plenum divides are floordiv toward −inf; E in int64 raw currency). New C++
  TUs (if any) go on the `/fp:strict` list.
- **Corridor drift bench** (folded from #4 per ruling): synthetic scenario in
  `tests/` — one supply, one return, a corridor; sweep `q_circ`; measure
  steady-state |u| against the drag law's quantization floor. The #4
  interlock made measurable: if honest mass-flux drift dies in the dead zone
  at plausible rates, plan B is a small directed momentum term at supply
  vents — strictly a fallback, only if the bench forces it.
- **Tuning waits** until one vent demonstrably works (ruling #4). #4's
  VENTING gate (drag must leave venting alive) applies to this arc's tests.

## 8. Deferred (each lives as a GitHub issue, not here)

- **v2 state machine** (#49): `damageable` / `console_controllable`; states
  `open / closed / damaged-open / welded-shut`; asymmetry (console opens +
  closes, damage only forces open, welding forces shut). Dead vent = no flux,
  ledger unaffected.
- **Air conditioning** (parked in #49): relax `E_plenum` toward
  `N_plenum·T_setpoint` at limited power, a named counted energy channel —
  the ship's thermostat (rooms recover ambient after a fire; dead heaters =
  cold ship). One-liner on top of the v1 energy ledger.
- **Fans** — the *momentum-injection* entity, separate from vents. Wall
  mounts only can be "fanned" (in-plane jets are real on a wall, fictional
  on a floor/ceiling mount — floor vents stay mass-only forever, by physics).
- **Duct & vent auto-placement** for levelgen (#50) — own design session.
- Filter degradation/clogging; multi-duct tactical routing; breached ducts as
  uncontrolled leaks; vent alarms / smoke detection at returns; true
  wide-vent entity (§2).

## 9. Open questions (post-critique residue)

1. **Kp stability + overflow bound** — patch 3's entry gate: derive against
   the EMA lag, tick rate, and Q16 headroom of `Kp·Δp`; never tune it.
2. **Self-sensing bias** — a supply vent's aperture P is biased high by its
   own standing injection (return: low) before the solve spreads it; a
   systematic offset feeding an integrating controller. Bound it in the
   patch-3 derivation, or subtract the vent's own known injection from its
   reading.
3. **Sensor path** — accessor `SiteIndex`/`Channel.PRESSURE` (recommended,
   residency-proof) vs raw `gmap.atmosphere` at 9e (pump precedent). Decide
   at patch 1 review.
4. **Trace composition cap** — does the plenum's trace vector need a cap
   (a duct holding unbounded poison), or is intake-rate × filter the natural
   bound? Patch 1 review.

## Systems

**Existing canonical systems this design must use**: ⟨crit⟩ **pump system +
GameMap gas-N primitives** (`pump_system.py`, `gamemap.py::inject_gas_n` /
`extract_gas_n`) — THE per-tick gas mass flux path (v1's FieldEdit route was
a parallel copy; incident logged, CLAUDE.md table row added); conductor slot
9e(d) entity-actuator sweep (no new numbered slot); entity registry + one
serializer + `runtime_digest_rows` / ENTITY_SECT_V1 absence transparency;
`KIND_ENTITY_REF` + intangible entities (the duct); sensor accessor
(`sensor_accessor.py`) if Q3 lands on it; gas table (`gases.py`) row pattern
for filters; config via `CFG` (`[physics.vents]` for Kp/deadband/α-shift;
per-entity rates authored in real units, quantized at load — door-2/pump
idiom); Q16 conventions incl. `floordiv` toward −inf and int64 raw-e books
(`bulk_transport.cpp`); field digest + GOLDEN_AGGREGATE for gating; editor UI
from registry; level data layer; `level_airtight.py` as lint host. FieldEdit
+ payload executor remain the seam for *event* edits — explicitly NOT this
design's flux path.

**New systems this design creates** (draft one-line rules for the CLAUDE.md
table at implementation):

| System | Where (planned) | Draft rule |
|---|---|---|
| Vent + duct entities | `entities/` rows + runtime in `simulation/` | The only ambient-airflow mechanism; flux only via the extended gas-N primitives at 9e(d); plenum ledger R3-counted (bulk pair + int64 energy), measured-delta booked, ENTITY_SECT-digested |
| Gas-N primitive extensions | `gamemap.py` | Per-slice withdrawal vector return; arbitrary-composition inject + mass-weighted T-mix (floordiv −inf, railed) — never a second flux path, never a FieldEdit mode |
| Filter table | config `[filters.*]` | A filter is a table row (per-gas efficiency); ducts reference by validated name; never a hardcoded per-gas if |
