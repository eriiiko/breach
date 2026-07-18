# A5 impl design — EOS evacuation rule: `seal_tiles` / `unseal_tiles` (v1, pre-critique, 2026-07-18)

Arc A patch A5 (plan: `arc_a_patch_plan_2026-07-18.md`, A5 row). Design-gated:
this doc goes through an adversarial pass (conservation + determinism lenses)
before anything builds. Scope is the **pure `GameMap` primitive + its tests** —
no door entity, no structural sweep, no entity wiring (all A6). Nothing in the
sim calls the new methods in A5; dormancy is structural.

Binding requirements (entity doc `entity_system_design_2026-07-18.md` §7,
"Mass conservation prerequisite", lines 238–244):

- closing a door **evacuates** its tiles' gas to neighbors;
- opening uses the existing joins-open-air rule;
- a test cycling a door in a sealed room asserts **EXACT N conservation**;
- door flips are STRUCTURAL (`on_tile_changed` cache patch; effects reach
  solvers next tick via the step-6 restamp);
- water rule v1: refuse to close over `water_depth > 0`.

Canon this design builds against: `docs/eos_refactor_design.md` §2.2
occupancy-transition rule (lines 154–162): *"when a cell leaves the open-air
mask … its `N_i` is evacuated conservatively into adjacent open cells … before
the cell is masked; it is never zeroed. (Zeroing remains correct only for
vacuum cells.)"*

---

## 0. Why the evacuation must be synchronous (the load-bearing code fact)

The bulk-flux solver **defensively zeroes `N` on solid tiles every flux pass**:
`cpp/src/bulk_transport.cpp:175–188` — *"Solid never holds N (defensive — a
stale value from before a tile became solid must not linger)"* — and likewise
zeroes vacuum (the deliberate breach sink). So a tile sealed without
evacuation has its gas **silently deleted** on the next tick's first flux
pass. The trace planes are similarly clamped/zeroed against the wall mask in
the SL smoke pass (`cpp/src/physics_engine.cpp:322` — "diffusion Laplacian →
… → SL back-trace → clamp/zero"). Evacuation therefore cannot be deferred to
the solver: it must complete **inside the primitive, at the moment the tile
turns solid**, before any solver pass sees the new mask. That is exactly the
§2.2 occupancy-transition contract, and it is why A5 exists as a prerequisite
for doors.

---

## 1. The unseal-direction conflict, and how this design resolves it

The entity doc requires both (a) "opening uses the existing joins-open-air
rule" and (b) "cycling a door in a sealed room asserts exact N conservation."
These are **irreconcilable as literally written**: the existing joins-open-air
rule is `destroy_wall`'s neighbor-mean seed
(`src/simulation/gamemap.py:1042–1044` + `_seed_bulk_gas_neighbor_mean`,
`gamemap.py:892–901`), which **mints** gas — it writes the neighbor mean into
the new tile without debiting the neighbors. Cycling close→open with that rule
adds ~one tile's mean `N` per cycle; a toggling door is a mass pump, and total
`N` grows without bound. For an RL project this is a discoverable exploit, not
a cosmetic wart.

**Resolution (this design's call, flagged for the critique):**

- A5 ships the symmetric pair `seal_tiles` **and** `unseal_tiles`.
  `unseal_tiles` implements the joins-open-air rule's *shape* — the tile is
  seeded at the integer neighbor mean, preserving the anti-vacuum-pulse intent
  of engine/04 §2.3 — but **conservatively**: the seeded amount is *withdrawn*
  from the donor neighbors, so grid-total `N` is exactly unchanged. Physically
  this is the correct statement anyway: opening a door does not create air;
  the air in the doorway comes from the room.
- `destroy_wall` (`gamemap.py:994–1044`) is **untouched canon**. The minting
  neighbor-mean seed remains the rule for *destruction* events (breach, fire
  burn-through, bullet chew, burst) — do-not-redesign per the arc plan. The
  asymmetry (destruction mints, door-cycling doesn't) is deliberate and
  documented: destruction is rare, un-cyclable by an agent at zero cost, and
  already shipped behavior under golden gates.
- Fallback if the critique rejects conservative unseal: keep the minting open
  rule and weaken the cycle test to "exactly *predicted*" (assert
  `N_after == N_before + minted`, minted computed from the rule). Rejected as
  primary because it leaves the mass-pump exploit in place.

A6's door open/close will call this pair; the A5 conservation test cycles the
primitives directly.

---

## 2. API

New code in `src/simulation/gamemap.py` (the same file as `destroy_wall` — it
is the state-topology seam; engine/04 §2.3 keeps topology edits on the Python
side), placed next to `destroy_wall`:

```python
class SealBlocked(ValueError):
    """A seal/unseal precondition failed. State is untouched (atomic)."""

def can_seal_tiles(self, tiles) -> bool:
    """Policy query: True iff seal_tiles(tiles, ...) would succeed.
    Checks bounds / not-already-solid / water / receiver availability.
    Does NOT check unit occupancy — that is caller policy (A6)."""

def seal_tiles(self, tiles, material_id) -> None:
    """Seal a span of open tiles to `material_id` (a solid material),
    evacuating their gas conservatively to open neighbors. Atomic:
    validates everything, then mutates; raises SealBlocked / ValueError
    with no partial mutation."""

def unseal_tiles(self, tiles) -> None:
    """Open a span of solid tiles to MAT_AIR, seeding each from its open
    neighbors CONSERVATIVELY (withdrawn, not minted). Atomic like seal."""
```

- `tiles`: iterable of `(fy, fx)`. Internally normalized to a **row-major
  sorted** span (`sorted(set(...))`); duplicate tiles raise `ValueError`
  (caller bug). The caller's ordering can never matter (determinism §7).
- `material_id` must be solid in the material table —
  `materials.permeability[material_id] <= 0` (the `solid` derivation,
  `gamemap.py:441`; table default derivation `materials.py:186–192`) — else
  `ValueError`. Sealing to a non-solid material is incoherent (the tile would
  stay open to flow while its gas was evacuated). For doors v0 this will be
  `MAT_DOOR` (`materials.py:32`), whose `permeability` is 0 in the shipped
  table (the door-stamp-leak fix made doors flow-solid).
- Return `None`; failures raise. `SealBlocked` for state-dependent refusals
  (water, sealed pocket), `ValueError` for caller bugs (bounds, duplicates,
  already-solid, non-solid material). A6's close check composes as:
  `occupancy_clear(span) and gmap.can_seal_tiles(span)`.
- **Not FieldEdits.** Door flips are structural, the engine/13 carve-out
  (entity doc §7); these methods mutate state directly, exactly like
  `destroy_wall`. The FieldEdit flush-once contract is untouched.

---

## 3. `seal_tiles` — exact algorithm (integer only)

All arithmetic is on plain Python ints (arbitrary precision; `divmod` exact).
Every numpy read goes through `int(...)`; every write is an int store into the
existing int32 arrays. **No floats anywhere** — deliberately *not* reusing
`_neighbor_mean` (`gamemap.py:878–890`), which computes a float mean
(`total / count`, line 890); that float path stays canon for `destroy_wall`
only.

### 3.1 Validation pass (no mutation)

For the row-major span `S`:

1. Every `(fy, fx)` in bounds, else `ValueError` (strict — unlike
   `destroy_wall`'s silent OOB return at `gamemap.py:1012–1013`, which is
   event-driven leniency; a primitive caller passing OOB is a bug).
2. `not self.solid[fy, fx]`, else `ValueError` ("already sealed" — catches
   double-close bugs; idempotence, if a caller wants it, is caller-side).
3. `material_id` solid per §2, else `ValueError`.
4. **Water rule v1:** `self.water_depth[fy, fx] == 0`, else `SealBlocked`.
   Layering pinned: the *primitive* enforces this as a hard invariant guard —
   `water_depth` is a bit-conserved Q16.16 field (`gamemap.py:283–288`) and
   the water solver zeroes depth on solid (`gamemap.py:359–363` comment), so
   sealing over water is silent conserved-mass deletion; the primitive must
   make that impossible unconditionally. The *caller* (A6 door) additionally
   pre-checks via `can_seal_tiles` to implement the polite
   "refuses to close" (same blocked semantics as the unit-occupancy rule,
   which is purely caller-side — the primitive never sees units).
5. Compute each tile's **receiver set** (order = `_FACE_DIRS` N,S,E,W,
   `gamemap.py:476` — the conduction seam's canonical order; NOT
   `_neighbor_mean`'s up/down/left/right at line 883):

   ```
   receivers(t) = [(ny,nx) for (dy,dx) in N,S,E,W
                   if in-bounds
                   and not self.solid[ny,nx]
                   and (ny,nx) not in S]          # span members never receive
   ```

   - **Exposed-vacuum neighbors qualify** (`is_vacuum & ~solid`): a breach is
     an open side (engine/04 §2.3). Gas pushed there vents on the next flux
     pass via the deliberate vacuum sink (`bulk_transport.cpp:177–181`) —
     physically, a door slamming next to a breach shoves some air out. The
     primitive itself stays exactly conservative (the loss happens later,
     through the sanctioned sink, as venting). Alternative (prefer
     non-vacuum receivers) rejected for rule simplicity; revisit only if the
     critique finds a gameplay foot-gun.
   - **Span members are excluded** even though they are still open at
     validation time: the span seals simultaneously, so evacuating tile A
     into tile B of the same span would delete B's inflow when B seals.
     Receivers are defined against the **post-span** solidity.
6. **Sealed-pocket rule:** if `total_gas(t) > 0` (sum of `int(gas[g][t])`
   over all `N_GASES` slices, `gases.py:46–70`) and `receivers(t)` is empty →
   `SealBlocked`. Pinned: **refuse**, never delete — §2.2's "it is never
   zeroed" is LOCKED canon; a counted-sink alternative is rejected for v1
   (documented gap §10 for future wall-*building* features, which will need a
   vent-first policy or a counted sink). A tile with zero gas (e.g. vacuum)
   seals fine with no receivers.
   Gameplay consequence, accepted: you cannot seal the last open tile of a
   gas-holding room; a functional door always has an open side by
   construction, so this bites only degenerate geometry.
7. **Overflow pre-check:** for each receiver `r`,
   `int(gas[g][r]) + Σ_{t ∈ S adjacent to r} int(gas[g][t]) < 2**31` per
   slice (a generous over-bound: assumes `r` receives each neighbor's whole
   load). Violation raises `OverflowError` *before* mutation (atomicity).
   Rationale: `N` is a conserved field — a saturating store (the `heat`
   buffer's convention, `gamemap.py:233–235`) would *silently* break the iron
   conservation invariant, so overflow must be loud. Unreachable in practice:
   worst shipped densities (~200× ambient ≈ 1.3e7 counts, eos design §3.4)
   sit ~160× under `INT32_MAX`.

### 3.2 Mutation pass

For each `t = (fy, fx)` in row-major span order:

```
k = len(receivers(t))
for g in range(N_GASES):                    # slice order = gas ids, gases.py:46–55
    n = int(self.gas[g][fy, fx])
    if n == 0: continue
    q, r = divmod(n, k)
    for j, (ny, nx) in enumerate(receivers(t)):
        share = q + (1 if j < r else 0)     # remainder → first r receivers, N,S,E,W order
        if share:
            self.gas[g][ny, nx] = int(self.gas[g][ny, nx]) + share
    self.gas[g][fy, fx] = 0

self.material[fy, fx] = material_id
self.on_tile_changed(fy, fx)                # gamemap.py:536–570 — see §5

# steady-state solid values for the solver-owned fields (see §6):
self.atmosphere[fy, fx] = 0                 # P: solver holds 0 on solid (eos_solver.cpp:464)
self.wave_p[fy, fx]     = 0                 # P_prev store (recorder.py:58–59); solid ≡ 0
self.wind_x[fy, fx] = 0; self.wind_y[fy, fx] = 0    # u: zero-on-solid (eos_solver.cpp:421–423)
self.flow_vx[fy, fx] = 0; self.flow_vy[fy, fx] = 0  # water solver zeroes on solid
self.ripple[fy, fx] = 0.0; self.ripple_v[fy, fx] = 0.0  # zeroed on dry/solid (gamemap.py:296–301)
```

**Exactness proof (per slice, per tile):** the tile loses exactly `n`; the
receivers gain `Σ_j (q + [j<r]) = k·q + r = n`. Integer addition commutes, so
two span tiles sharing a receiver interact exactly; grid-total `N_g` is
unchanged to the LSB. Distribution is an **equal split** with the remainder
going to the first `r` receivers in N,S,E,W order — not pressure-weighted
donor-cell flux. Justification: the primitive is a topology edit, not a
transport step; the solver re-equilibrates from the next tick's real
`P = C·N_total·T`, so any deterministic placement converges to the same
physics; equal-split is the simplest rule with an exact remainder story.
(§2.2's "via the same donor-cell/limiter machinery" is read as *the same
conservation discipline* — every subtraction has a matching addition — not as
literally invoking the C++ flux kernel from a Python structural edit.)

**Fields deliberately NOT touched:** `temperature` (§4), `fire` (§10 gap:
a burning tile becomes a burning door — coherent, walls burn via the
burn-through machinery), `is_vacuum` (§5.2), `heat` (per-tick deposit buffer,
cleared each tick), `wall_hp`/`flammable`/caches (owned by
`on_tile_changed`), `wave_v`/`wave_source` (retired acoustic fields,
`recorder.py:58–59`; allocated but not consumed by the EOS path —
`run_substeps` does not take them, `physics_runner.py:470–479`).

### 3.3 Multi-tile simultaneous seal

Covered above by construction: receivers exclude span members (no
intra-span evacuation), per-tile operations are commutative integer adds, and
the span is internally sorted — so `seal_tiles([(a),(b)])` and
`seal_tiles([(b),(a)])` produce byte-identical state. A 2-tile door closing
is ONE `seal_tiles` call (A6 contract), not two.

---

## 4. Energy stance — argued, and deliberately not conserved in v1

Moving `N` without moving thermal energy is not energy-conserving: the
evacuated gas arrives at each receiver and thereafter counts at the
*receiver's* `T`, so the thermal energy ledger changes by
`Σ_j ΔN_j·c_v·(T_j − T_s)` per seal. **v1 accepts this.** The sealed tile
keeps its `T` value and simply switches to solid-mask temperature rules
(convert/conduct/cool — a hot gas tile sealed becomes a hot door that
conducts; physically pleasant and free). Rationale:

1. **Exact-N is the hard requirement; exact-E is not.** `T` is already a
   designed non-conserved field with counted sinks and rails (`T_MIN` floor,
   `T_MAX_PHYS` clamp, cooling — eos design §4, v2.4), and no exact-energy
   gate exists anywhere in the suite.
2. **The error is zero in the common case.** Door tiles sit in rooms near
   local thermal equilibrium; `T_j ≈ T_s` ⇒ error ≈ 0. Cycling a door in an
   isothermal sealed room is exactly energy-neutral. The error is one-shot
   per flip (no compounding mechanism).
3. **The canon join direction is already worse.** `destroy_wall`'s
   neighbor-mean seed mints `N` *and* its implied energy; an exact-energy
   seal next to a minting unseal buys nothing observable.
4. An energy-conserving variant (receiver
   `T_j' = (Ñ_j·T_j + ΔN_j·T_s) / (Ñ_j + ΔN_j)` with `Ñ` the
   `trace_mass_scale`-weighted Dalton total, wide int64 mul + one truncating
   divide per receiver) is fully deterministic and is the documented upgrade
   path — but it still rounds (≤1 LSB residue per receiver), so it cannot be
   *exact* either; it converts a zero-in-practice error into permanent
   machinery. Documented gap (§10), not built.

Same stance on unseal: the opened tile's `T` (the solid's temperature)
becomes the gas `T` — the warm door warms the doorway air. The withdrawn gas
likewise keeps its donors' `T`. Compression is instantaneous and isothermal
at the primitive (sealing 1 tile of a 2-tile room doubles the neighbor's `N`
with no temperature rise; real adiabatic compression would heat it). The
solver's compression-work term (eos design §3.2 step 4c) only sees flow it
computes itself — accepted v1 simplification, listed in §10.

---

## 5. Structural mechanics

### 5.1 What `on_tile_changed` covers (and what it doesn't)

`on_tile_changed` (`gamemap.py:536–570`) is **direction-agnostic** — it
projects the *new* material id through the table. The seal direction (AIR →
solid) needs **no extension**: it already patches `light_atten`, `heat_atten`,
`flammable`, `wall_hp` (Q16.16 re-quantize, line 558), `conductivity`,
`heat_inv_shift`, `face_shift` (this tile + 4 neighbor back-faces, lines
563–566 / `_patch_face_shift` 506–531), `permeability`, `wave_absorb`, and
`solid` (line 570). Its documented carve-out — *"Does NOT touch
atmosphere/vacuum — those carry edit-specific semantics owned by the caller"*
(lines 543–546) — is precisely where `seal_tiles`' evacuation and steady-state
writes (§3.2) live, mirroring how `destroy_wall` owns its seed.

**`obstacles` is intentionally NOT patched** — neither here nor in
`destroy_wall`. It is rebuilt every tick at step 6 (`simulation.py:721–722` →
`stamp_units`, rebuild at `gamemap.py:751` Python / C++ engine reset), which
is the **step-6 restamp**: a seal applied at slot 9e (A6) or between ticks
(tests) reaches the solvers' masks next tick, exactly the one-tick-delay
story the entity doc pins. Within-tick readers of `self.solid` (LOS
`gamemap.py:868`, `find_burst_walls` `gamemap.py:951`) see the new state
immediately — but both burst (slot 9b) and physics (slot 7) run *before*
slot 9e in `Simulation.step` (`simulation.py:635–764`), so no solver-class
consumer observes a half-applied tick.

Restamp interaction, unseal direction: at the next tick's step 6 the tile
leaves `obstacles`, so the **freed-tile refill** fires
(`gamemap.py:697–702` C++ path / `802–807` Python path) and overwrites
`atmosphere[t]` with the float neighbor mean. This touches only the derived-P
alias — which the solver re-materializes the same tick (eos design §3.2
step 5) — and never the `gas` planes (`_seed_bulk_gas_neighbor_mean` is
called only from `destroy_wall`), so **no mass is minted**; deterministic;
same window `destroy_wall` has always had. Noted, not fought.

### 5.2 `is_vacuum` bookkeeping — symmetric to `exposes_vacuum`

- **Seal never writes `is_vacuum`.** Sealing an interior tile: stays `False`.
  Sealing an exposed-vacuum tile (a breach): gas is already 0 there
  (`bulk_transport.cpp:183–186`), nothing to evacuate, no receivers needed;
  the tile becomes `solid ∧ is_vacuum` — exactly the **sealed-hull** state of
  engine/04 §2.3 ("sealed border/hull tiles are `is_vacuum` and obstacles").
  Hull-patching therefore works out of the box: seal a breach and the room
  holds pressure again.
- **Unseal applies the `exposes_vacuum` canon** (mirror of
  `destroy_wall:1033–1041`): if the tile is itself `is_vacuum` or any
  4-neighbor is exposed vacuum (`is_vacuum ∧ ¬solid`), set
  `is_vacuum[t] = True` and **seed nothing** — zeroing is correct for vacuum
  (§2.2); the solver's Dirichlet P=0 + donor-cell venting take over natively.
  The `on_edge_hull` special (`gamemap.py:1028–1029`) is **not** replicated:
  it is `was_hull`-gated destruction semantics; a door authored on the
  outermost ring is a level-authoring error (§10 gap).

---

## 6. Sealed/opened tiles vs the solver-owned fields (match what solids carry)

Every write in §3.2 reproduces the steady state the solvers themselves impose
on solid tiles, so the primitive leaves no "haunted door" values for the
recorder snapshot (which runs at end-of-tick, after slot 9e; `recorder.py:63`
`DEFAULT_FIELDS = ('atmosphere','temperature','gas_o2','smoke','fire',…)`):

| Field | Solid steady state | Enforced by (per tick) | Seal writes |
|---|---|---|---|
| `gas[*]` | 0 | `bulk_transport.cpp:175–188`; SL clamp/zero | 0 (after evacuation) |
| `atmosphere` (=P) | 0 | `eos_solver.cpp:464` (`pstar_=0` on solid); init `gamemap.py:449–451` | 0 |
| `wave_p` (=P_prev) | 0 | overwritten from P at tick top (eos design §3.2 step 0) | 0 |
| `wind_x/y` (=u) | 0 | `eos_solver.cpp:421–423` step-1f + `:511` post-kick | 0 |
| `flow_vx/vy`, `ripple/ripple_v` | 0 | water solver zero-on-solid/dry (`gamemap.py:296–301`) | 0 |
| `temperature` | live (solid rules) | conduct/cool pipeline | untouched (§4) |
| `water_depth` | 0 | precondition (§3.1.4) — solver would zero it (mass sink) | — (guarded) |

Unseal sets `atmosphere[t]` to the **integer** neighbor mean
(`Σ donors // k`) as the same consumer-stopgap `destroy_wall` provides
(display/read consistency until the next materialize), and
`wave_p[t] = atmosphere[t]` so the `|P − P_prev|` ripple-splash transient
(`recorder.py:158–164`, `physics_runner.py:838–840`) sees no phantom spike in
the window before step 0 overwrites it. `wind/flow` stay 0 (a fresh doorway
starts quiescent).

---

## 7. `unseal_tiles` — exact algorithm (integer only)

Validation: in-bounds, `solid[t]` true (else `ValueError`), duplicates raise.
Snapshot `pre_open = ~self.solid` **before any mutation** — donors are drawn
from pre-existing open air only (a 2-tile door's second tile never seeds from
the first's fresh gas; no chaining).

For each `t` in row-major span order:

```
self.material[t] = MAT_AIR
self.on_tile_changed(t)

if is_vacuum[t] or any 4-neighbor (N,S,E,W) has is_vacuum & ~solid:
    self.is_vacuum[t] = True          # joins vacuum — §5.2; no seed
    continue

donors = [(ny,nx) for (dy,dx) in N,S,E,W
          if in-bounds and pre_open[ny,nx] and not is_vacuum[ny,nx]
          and (ny,nx) not in span]
if not donors:
    continue                          # opens empty (vacuum pocket) — NEVER mint

k = len(donors)
for g in range(N_GASES):
    avail = [int(self.gas[g][d]) for d in donors]
    target = sum(avail) // k          # integer neighbor mean (floor)
    if target == 0: continue
    q, r = divmod(target, k)          # balanced two-pass withdrawal:
    take = [min(q + (1 if j < r else 0), avail[j]) for j in range(k)]
    short = target - sum(take)        # cascade shortfall in donor order
    for j in range(k):
        if short == 0: break
        extra = min(short, avail[j] - take[j])
        take[j] += extra; short -= extra
    for j, d in enumerate(donors):
        self.gas[g][d] = avail[j] - take[j]
    self.gas[g][t] = target           # Σ take == target, exactly

self.atmosphere[t] = sum(int(self.atmosphere[d]) for d in donors) // k
self.wave_p[t] = int(self.atmosphere[t])
```

**Exactness:** `target = ⌊Σavail/k⌋ ≤ Σavail`, so the cascade always
terminates with `short == 0`; the tile gains exactly what the donors lose.
The withdrawal is *balanced-then-greedy*: equal shares first, clamped to each
donor's holdings, shortfall taken left-to-right in N,S,E,W donor order —
deterministic, mostly symmetric, ≤2 passes over ≤4 donors. Unlike seal,
per-tile unseal operations sharing donors do **not** commute (clamping
depends on remaining holdings), which is exactly why the row-major span order
and the pre_open donor snapshot are pinned, load-bearing rules.

Round-trip invariants (open→close→open, any number of cycles):

- **Grid-total `N_g` per slice: exactly unchanged, forever** — each primitive
  is internally exact, so any composition is.
- Per-tile values are **not** restored (redistribution is not an undo);
  conservation is global, positional recovery is the solver's job (the next
  ticks' pressure solve re-equilibrates). The cycle test asserts totals, not
  layouts.
- Structural caches: exact round-trip (pure functions of the material grid).

---

## 8. Edge cases — pinned behavior (summary table)

| Case | Behavior | Where |
|---|---|---|
| All 4 neighbors solid, tile holds gas | `SealBlocked` (refuse; never delete) | §3.1.6 |
| All 4 neighbors solid, tile gas-free | seals fine (nothing to move) | §3.1.6 |
| Neighbor is exposed vacuum | counts as receiver; that share vents next flux pass (sanctioned sink) | §3.1.5 |
| Sealing a tile that IS exposed vacuum | seals; `is_vacuum` stays True → sealed-hull state (hull patch) | §5.2 |
| Sealing over `water_depth > 0` | `SealBlocked` at the primitive (invariant guard); caller pre-checks via `can_seal_tiles` for the polite refusal | §3.1.4 |
| Sealing last open tile of a room | = pocket case: refuse if gas present | §3.1.6 |
| Unseal into all-sealed surroundings | opens empty (N=0); never mint | §7 |
| Unseal adjacent to / on vacuum | joins vacuum, no seed, `is_vacuum=True` | §5.2 / §7 |
| Duplicate span tiles / OOB / already-solid / non-solid material | `ValueError` (caller bug; atomic) | §3.1 |
| Receiver int32 overflow | `OverflowError` pre-mutation (loud, atomic; unreachable at shipped densities) | §3.1.7 |
| Unit standing on the tile | **not the primitive's concern** — occupancy is A6 caller policy | §2 |
| Fire on the sealed tile | carried over (burning door); v1 gap | §10 |

---

## 9. Determinism, digest dormancy, and the GPU story

**Determinism.** The primitives use: Python-int arithmetic only (`divmod`,
`+`, `min` — exact at any width), pinned iteration orders (row-major span;
N,S,E,W receiver/donor order = `_FACE_DIRS`, `gamemap.py:476`), no floats, no
RNG, no dict-order dependence, no numpy reductions in the arithmetic path
(sums are Python-int over ≤4 values). Two machines executing the same call
sequence produce bit-identical arrays. The one float wrinkle nearby —
`_neighbor_mean`'s `total/count` (`gamemap.py:890`) and the freed-refill
(`gamemap.py:697–702`) — is pre-existing canon, IEEE-deterministic in
practice, and deliberately **not** entered by the new code.

**Digest dormancy (A4 contract, `a4_digest_impl_note_2026-07-18.md`).** A5
adds no digest sections and changes no hashed bytes: the primitives have
**zero call sites** in the sim (A6 wires them), so an entity-free — indeed,
any existing — level runs a byte-identical trajectory. Dormancy is
structural, not behavioral. A5 tests build their own tiny programmatic
fixtures (the `test_eos_p1_species_transport.py:53–58` `LevelData` idiom);
per A4's critique-11 corollary, no digest-suite level is touched and no
golden is re-baselined.

**GPU / CUDA lockstep (pre-S8a).** The edit lands in the CPU numpy arrays
between solver invocations — exactly like `destroy_wall`. Pre-S8a, every
`PhysicsRunner.step` passes the live arrays into
`engine.run_substeps(...)`/`step_tail(...)` (`physics_runner.py:470–479`,
`543–552`) and the CUDA dispatch round-trips fields GPU↔CPU per tick, so both
backends consume identical post-edit state; the primitive itself never runs
on the GPU. Lockstep holds because the edit is pure-integer, order-pinned,
and happens at a pinned slot (9e / between ticks).

**S8a-contract rider (flagged for the ledger's S8a spec rewrite):** once
fields are GPU-resident, structural primitives (`destroy_wall`,
`seal_tiles`, `unseal_tiles`) must push their touched-tile dirty set to the
device before the next kernel reads: the `on_tile_changed` cache set
(permeability/solid/wall_hp/face_shift/…) **plus** `gas` (all 7 slices),
`atmosphere`, `wave_p`, `wind_x/y`, `flow_vx/vy`, `ripple/ripple_v`,
`is_vacuum`. Sparse per-tile upload or device-side execution — S8a's choice;
the contract just requires the list be honored. This rider extends the
sensor-gather contract note already standing in the arc plan
(`arc_a_patch_plan_2026-07-18.md` standing constraints).

---

## 10. Accepted gaps v1 (simplest honest design; revisit at arc close)

1. **Energy not conserved at seal/unseal** (§4): `T` untouched; error
   `Σ ΔN_j·c_v·(T_j − T_s)`, ≈0 in isothermal rooms, one-shot per flip.
   Upgrade path documented (N-weighted T mixing, wide int64).
2. **Instant isothermal compression** (§4): teleported mass raises next
   tick's `p*` with no adiabatic heating and no authored acoustic transient;
   the `|P − P_prev|` transient fires naturally next tick instead.
3. **`destroy_wall` still mints** (§1): the destruction direction keeps the
   canon neighbor-mean seed. Only the seal/unseal pair is exactly
   conservative. Asymmetry documented, deliberate.
4. **Fire carried onto sealed tiles** (§3.2): a burning tile becomes a
   burning door; no extinguish/transfer rule.
5. **Equal-split distribution** (§3.2): not pressure-weighted; the solver
   re-equilibrates. Same for the balanced withdrawal on unseal.
6. **Traces treated like bulk** (§3.2): all 7 slices move by the same rule;
   no per-species physics (a smoke cloud is displaced, not filtered).
7. **No partial-permeability seal** (§2): the primitive only seals to fully
   solid materials; a "cracked door" conductance is the face-flux forward
   idea (engine/04 §4), out of scope.
8. **Sealed-pocket refusal, not counted sink** (§3.1.6): future
   wall-building features need a policy; doors never hit it.
9. **No edge-hull special on unseal** (§5.2): doors on the outermost ring
   are authoring errors; only `destroy_wall` has the `was_hull` rule.
10. **Water displacement** (entity doc §7): refuse-over-water stands in for
    displacement physics — future work, per the locked design.

---

## 11. Test plan (`tests/test_a5_seal_evacuation.py`, gate: `pytest tests -q`)

Fixtures: programmatic sealed-box `LevelData` (the
`test_eos_p1_species_transport.py:53` idiom), variants with a breach
(exposed-vacuum ring), a 2-tile room, and a 1-tile pocket. Totals measured as
`int(gas[g].astype(np.int64).sum())` per slice (the `_isum2` idiom,
`test_eos_p1_species_transport.py:193–194`).

1. **Sealed-room door-cycle EXACT N conservation** (the A5 gate): record per-
   slice totals; `seal_tiles` one doorway tile → totals exactly unchanged,
   tile fields at solid steady state (§6 table); `unseal_tiles` → totals
   exactly unchanged; repeat ×100 cycles → totals constant to the LSB.
   Variant with nonzero trace gas (smoke in the doorway) — all 7 slices.
2. **Full-tick cycle**: same, but with `PhysicsRunner.step` between flips —
   totals conserved every tick (sealed room, no vacuum: the solver's own
   conservation gate domain, `bulk_transport.cpp:180–182`).
3. **Remainder placement**: craft indivisible loads (e.g. n=7, k=3) → assert
   exact shares `[3,2,2]` in N,S,E,W order.
4. **Multi-tile span**: 2- and 3-tile spans; no intra-span evacuation
   (span-member gas ends 0, receivers outside the span); permutation
   invariance — shuffled input tile list ⇒ byte-identical field state.
5. **Sealed pocket**: gas-holding tile, all neighbors solid → `SealBlocked`
   and full-state equality with the pre-call snapshot (atomicity). Gas-free
   pocket seals.
6. **Vacuum**: breach neighbor receives its share; sealing an exposed-vacuum
   tile keeps `is_vacuum`, seals, then the room holds pressure over ticks
   (hull-patch E2E). Unseal adjacent to vacuum → joins vacuum, no seed.
7. **Water refusal**: `water_depth > 0` → `SealBlocked`, no mutation;
   `can_seal_tiles` False; drained tile seals.
8. **Two-tile room compression**: sealing one tile moves its entire load to
   the other (exact doubling); next materialized P rises (physics smoke
   check, not a golden).
9. **Conservative unseal withdrawal**: unequal donors incl. a zero donor →
   cascade correctness, per-slice totals exact; donor clamping never goes
   negative.
10. **Round-trip invariants**: open→close→open — totals exact forever;
    structural caches byte-equal to a fresh `_update_caches` build.
11. **Freed-refill interaction**: unseal, then `stamp_units` — gas totals
    unchanged (refill touches only `atmosphere`).
12. **Determinism**: run an identical seal/unseal/step script on two fresh
    maps → all synced fields byte-identical (`np.array_equal`).
13. **Strictness**: duplicates, OOB, already-solid, non-solid material, and
    the overflow pre-check each raise with no mutation.
14. **Dormancy**: full existing suite green with zero golden edits
    (structural: no sim call sites — asserted by grep-level review, exercised
    by the suite itself).

---

*A5 design v1 — pre-critique. Author: Claude (Arc A design agent), 2026-07-18.
Critique lenses requested by the plan: conservation, determinism,
neighbor-distribution order, all-neighbors-solid, GPU-resident-field
interaction — §3, §7, §9 are the load-bearing sections.*
