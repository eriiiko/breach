# P-M3 — `destroy_wall` seeding: design (2026-08-18, **v2**)

**Arc:** mass-books (`docs/mass_books_arc_kickoff_2026-08-18.md`).
**Status:** v2, post-critique. Feel-adjacent → **HUMAN-TEST before merge**.
**Decision owner:** Erik. The decisions here were taken with him in session; this
document records them, their reasoning, and their gates — it does not re-open
them.

> **v1 → v2.** v1 (commit `2d4c6d8`) went through a three-lens adversarial
> critique — determinism/integer-exactness, conservation-books/physics, and
> scope/regression. All three independently found the same **blocker**: v1's
> instruction not to write `atmosphere` would have replaced a mass amplifier
> with a *burst* amplifier in the same tick. Four further findings changed the
> spec. §8 records every finding and its resolution. v1's central energy claim
> (§4) was verified correct against the code and survives unchanged.

---

## 1. The defect

`destroy_wall` ends by seeding the newly-opened tile with the neighbour mean of
its bulk gas ([`gamemap.py:1752-1754`](../src/simulation/gamemap.py#L1752-L1754)):

```python
self.atmosphere[fy, fx] = self._neighbor_mean(self.atmosphere, fy, fx)
self._seed_bulk_gas_neighbor_mean(fy, fx)      # gas[O2], gas[INERT_N2]
```

`_neighbor_mean` **reads** the open 4-neighbours and **writes** their mean into
the new cell. Nothing is withdrawn from the donors, so every destroyed tile
creates one neighbour-mean cell of air out of nothing.

Measured (`tools/repro_destroy_wall_mint.py`, no weapon, no explosion, not even
a solver step): 4 walls destroyed in the sealed two-room fixture mint exactly
+4.000 cell-equivalents, scaling linearly with local density — ×10 → +40.000,
×100 → +400.000.

**Why it blows up rather than drifts.** `find_burst_walls` fires on a pressure
*differential* above threshold, so a bursting wall is high-pressure **by
definition**, and the mint scales with exactly that pressure:

```
high pressure → wall bursts (the relief valve) → mint ∝ that pressure
              → higher pressure → more bursts → …
```

The emergent pressure-relief valve is a pressure amplifier. `_neighbor_mean`
also skips vacuum neighbours ([`gamemap.py:1571-1573`](../src/simulation/gamemap.py#L1571-L1573)),
so at a hull between a pressurised room and space the mean is taken over the
pressurised side only — peak seed at peak differential.

In one recorded session this path carried **87.7% of a 2.201× total mass
growth** (`docs/mass_books_arc_kickoff_2026-08-18.md` §1.2).

**The codebase already knew.** The A5 evacuation block twelve lines below
records the asymmetry as deliberate
([`gamemap.py:1764-1767`](../src/simulation/gamemap.py#L1764-L1767)): *"the
symmetric open half (`unseal_tiles`) withdraws its seed from the donors instead
of minting (destroy_wall's neighbor-mean seed stays the rule for DESTRUCTION
events only)."*

## 2. The decision, and why it is not "withdraw from the donors"

The conservative option — take the seed out of the donor cells, as
`unseal_tiles` does — closes the books exactly. It was considered and
**rejected on physical grounds**, by Erik:

> *"An explosion does not eliminate matter, it just redistributes it."*

At a 1/3 m cell, destroying a tile is a drastic geometry change, and the wall's
own material does not vanish in reality — it becomes rubble that still occupies
volume. The sim carries no rubble, so withdrawing gas from the neighbours models
an expansion into void that the model has no way to represent. Withdrawal would
be right if the simulation were ground truth; it is not.

The cave case makes it concrete: a map 80% solid, blasted open to 40% solid,
would under withdrawal have its pressure fall by more than half — a cave that
suffocates you for digging it out. Real caves stay breathable because they are
connected to a reservoir. **Seeding ambient models the reservoir that is
actually there**, and on `boundary = ambient` maps it is not even a fiction.

**Decision: seed a CONSTANT TOTAL at the map's ambient, and book it.**

The load-bearing property is *constant total N*, not the particular value and
not the composition. A constant total breaks the feedback loop with the burst
valve dead — the seed no longer scales with the pressure that triggered the
burst. A neighbour-scaled total of any size keeps the loop closed.

> **Critique correction (§8, books lens).** v1's §2 also argued that withdrawal
> "models an expansion into void that does not physically happen". That half is
> **wrong** and is withdrawn: the transport pass performs the expansion anyway
> on the next tick — a seeded cell at N=1.0 beside donors at N=5.0 is filled by
> donor-cell flux, drawing the neighbours down regardless. The constant seed
> does not prevent the drawdown; it adds `ambient_N` on top of it. The
> reservoir/cave justification stands on its own and is the real argument.

## 3. Specification

### 3.1 The seed

| field | value |
|---|---|
| `gas[O2]` + `gas[INERT_N2]` | **total** = the map's ambient N; **split** = the local O₂ mole fraction (see below) |
| `temperature` | **0**, explicitly written, on **every** destroyed tile including breach tiles |
| `atmosphere` | **written**, to the pressure consistent with `(N_seed, T=0)` — see §3.1.2 |

#### 3.1.1 One accessor, no fifth copy of the constant

`self._ambient` is **`None` on every space map** — it is only populated under
`if self._boundary == "ambient"` ([`gamemap.py:638-646`](../src/simulation/gamemap.py#L638-L646)).
`playground`, the map this entire audit ran on, is `boundary = space`. So v1's
"read from the map's ambient" is an `AttributeError` in the common case.

There are already four places that derive this constant and two distinct
derivations that agree **only** at default dials. Add exactly one accessor —
`GameMap.ambient_seed()` returning `(n_total_q, o2_q, n2_q, pin_q)` — with the
existing fallback logic ([`gamemap.py:1043-1049`](../src/simulation/gamemap.py#L1043-L1049)):

- `_ambient` present → `derive_ambient`'s values ([`ambient.py:150-153`](../src/simulation/ambient.py#L150-L153)),
  where `n_total = quantize(p_amb)` and the split uses the level's authored
  `o2_frac` — **not** a hard-coded 0.21/0.79;
- `_ambient is None` → the space fallback: `FP_ONE`, `quantize_scalar(0.21)` =
  13763, `quantize_scalar(0.79)` = 51773, which sum to 65536 exactly.

Every other site (`_update_caches`, `inject_gas_n`, `field_edit`, this seed)
routes through it. Gate 1 keys off the accessor's value, not off `FP_ONE`.

#### 3.1.2 The `atmosphere` write — restored, and why

**v1 was wrong to drop it.** A solid tile's `atmosphere` is hard 0 — the MG
solve zeroes it ([`eos_solver.cpp:1409`](../cpp/src/eos_solver.cpp#L1409)) and
that is what gets stored. Two `destroy_wall` callers run **after** the physics
step and are refilled by nothing in that tick: fire burn-through
([`simulation.py:1415`](../src/simulation/simulation.py#L1415)) and the burst
valve ([`:1431`](../src/simulation/simulation.py#L1431)). `find_burst_walls`
then runs at [`:1429`](../src/simulation/simulation.py#L1429) and reads
`float(atm[ny, nx])` for any neighbour that is not solid, vacuum or ambient
([`gamemap.py:1674-1675`](../src/simulation/gamemap.py#L1674-L1675)).

Worked case at this arc's own target pressure: a wood/door wall between a
2.1 atm room and a 1.0 atm corridor holds spread 1.1 — safe. Punch one hole with
`atmosphere` unwritten and its along-wall neighbours see sides {2.1, 1.0, **0.0**}
→ spread 2.1 > `burst_threshold` 2.0 → they pop, up to `burst_max_per_tick = 16`
per event. **v1 would have removed the amplifier at the mass end and installed
one at the burst end**, on the same code path, in the same tick.

The dependency is documented: [`gamemap.py:1638-1641`](../src/simulation/gamemap.py#L1638-L1641)
says the ambient branch exists because *"a just-breached ring tile is seeded by
neighbour-mean, not yet re-pinned"*.

Three further same-tick readers of the stale 0 confirm it is not display-only:
the MG **warm start** (`p_prev := atmosphere`), `apply_wave_push` (a ~1 atm
phantom ∇P that can fire `KNOCKED_DOWN` on units beside a burnt-through wall),
and the `PRESSURE` sensor channel — an RL observation.

v1's rationale ("a third independent seed is how the current pair got out of
sync") argued for the wrong conclusion: writing the pressure **consistent with**
the seeded N is not a third independent seed, it is the consistent triple.
Leaving it at 0 alongside `N = ambient, T = 0` is the inconsistent one.

Write `atmosphere := pin_q` from the accessor. Note the lattice: the effective
pin is 65540 raw (1.000061 atm) at Earth defaults, not 65536
([`ambient.py:21-24`](../src/simulation/ambient.py#L21-L24)) — so state the seed
purely in N and do **not** assert `p* == N` exactly.

#### 3.1.3 Composition — inherited, not ambient

**Erik's ruling, accepted this revision.** A fixed 0.21/0.79 seed injects fresh
oxidizer: combustion is linear in the O₂ *mole fraction* with a 2-hop draw
radius, so each destroyed tile would drop a 21%-O₂ cell inside the draw radius
of the rest of a burning wall run — re-feeding fires that should be suffocating.
Today's neighbour-mean seed cannot do this because it inherits the room's
depleted mixture.

**Seed the constant TOTAL, split by the local O₂ fraction:**

```
n_total := ambient_seed().n_total_q              # CONSTANT — breaks the loop
X_local := ΣO2(open 4-neighbours) / ΣN(open 4-neighbours)
           (fall back to the ambient o2_frac when the denominator is 0)
o2  := round(n_total · X_local);  n2 := n_total − o2      # exact complement
```

The magnitude is what closed the feedback loop, not the ratio, so this keeps the
fix fully intact while removing the oxidizer injection. Cost is four neighbour
reads — the same reads `_neighbor_mean` does today, so no performance change.
The `n2 := n_total − o2` complement keeps the pair summing to `n_total` exactly.

### 3.2 Breach tiles — skip the gas seed, still write T

A destroyed tile that joins the vacuum/ambient boundary (`on_edge_hull or
exposes`) is Dirichlet-pinned. Measured on a **space** map: it is seeded +10
cell-eq and the next tick deletes exactly −10 — a mint-then-delete round trip
that nets zero through two broken books. Skip the gas seed there.

> **Critique correction (books lens).** v1 stated this generally; it is
> **space-map-only**. On an ambient map `breach_mask` is `is_ambient` and the
> clamp does the *opposite* — it **fills** the tile to `N_amb` each substep and
> books the difference to `boundary_flux`
> ([`bulk_transport.cpp:231-233`](../cpp/src/bulk_transport.cpp#L231-L233)). So
> on ambient maps the skipped tile is filled by the ambient rail, not wiped, and
> Gate 2's "0 then 0" is a space-map assertion. Both paths are correct to skip;
> the *reason* differs and the gate must say which map it is on.

`temperature := 0` is written on breach tiles too. On an ambient map a burning
hull tile would otherwise join `is_ambient` still carrying the wall's hot T, and
the `c_local` scan skips only `solid || is_vacuum` — inflating map-wide sound
speed and substep count for a tick. Costs nothing on space maps.

Use `breach_mask[fy, fx]` as the skip predicate (mirroring `unseal_tiles`'
`joins_boundary` check) rather than re-deriving `on_edge_hull or exposes`, so the
two predicates cannot drift apart. The mask is set at
[`gamemap.py:1751`](../src/simulation/gamemap.py#L1751), unconditionally before
the seed, on every path — verified.

### 3.3 Clear `fire` — corrected justification

> **Critique correction (scope + determinism lenses).** v1's premise was
> **false** for the path it described. `fire_simulation.cpp:317-320` already
> sets `fire[i] = 0` on every tile it emits in the burn-through list, and the
> CUDA twin mirrors it. The stale `fire` is real, but only on the *other*
> callers — burst, explosion, bullet-chew, door — because `on_tile_changed`
> patches ten caches and never `fire`.

Keep the change; restate what it buys. It is **not** a gameplay change to fire
spread: cellular spread no longer exists, `apply_temperature_ignition` is
`flammable`-gated so it cannot relight an air tile, and a destroyed burning tile
is already dropped as a radiation emitter. There is no exploit — shooting walls
extinguishes nothing that was still doing work.

What it actually buys: a stale display/sensor cleanup, and it stops
`fire_simulation.cpp:311-316` decrementing a now-air tile's `wall_hp` forever.
It **does** change the `FIRE` sensor channel, which is an RL observation change
and is the honest justification to record.

### 3.4 Booking — `n_destruction_seed_sum`, measured and signed

> **Critique correction (both lenses).** v1 specified this `≥ 0` and equal to
> `ambient_N × tiles_seeded`. Both are **wrong**. `destroy_wall`'s gate is
> `material != MAT_AIR`, widened deliberately for W2 bullet-chew on furniture.
> Furniture ships `permeability = 0.5` → **not solid** → it already holds bulk
> N. Writing a constant into a tile that already holds gas is `seed − prior`,
> which is **negative** whenever the room is above ambient: chewing a crate at
> 5 atm *deletes* ~4 cell-equivalents.

The channel is **measured, not computed from a formula**: sample `Σ N` over the
map immediately before and after each `destroy_wall` call and accumulate the
difference into a **signed** counter. v1's "trivially predictable / cheapest
channel to gate" claim is retracted.

Gate 1 must therefore exercise a **furniture** tile as well as a solid wall, and
its "independent of local density" assertion applies to the *solid* path only —
on the furniture path the seeded total is constant but the *delta* is not.

### 3.5 The books seam — gated separately, stated explicitly

Every energy and mass bracket lives inside `EOSSolver::step`. Destruction runs
from Python **after** `physics_runner.step()` returns and before the next tick's
first bracket. A Python-incremented counter therefore **can never appear in a
C++ per-tick identity** — both the seed and the counter fall in the gap.

The kickoff §3.4 demanded this be resolved and written down rather than chosen
silently. **Decision: gate the seam separately, at the Python level.** The
destruction seam is asserted by sampling `Σ N` on both sides of the destruction
block within `Simulation.step` and requiring the difference to equal
`n_destruction_seed_sum`. The C++ per-tick identity is left alone and continues
to assert only what happens inside the solver.

Rationale: destruction is a Python-side event on a Python-owned mirror; forcing
it into the C++ identity would require plumbing a seam sample through the
binding for no gain. This is the simplest honest split, and it is now recorded
rather than implicit.

### 3.6 `stamp_units` — the second neighbour-mean writer

A **second** `atmosphere` neighbour-mean seed exists that v1 never accounted for:
`stamp_units` re-seeds every tile freed by a unit moving off it
([`gamemap.py:1380-1388`](../src/simulation/gamemap.py#L1380-L1388) C++ path,
[`:1488-1494`](../src/simulation/gamemap.py#L1488-L1494) Python path), at
step 6 — *before* the EOS at step 7.

Three consequences:

1. **For the explosion path this patch is a no-op on `atmosphere`.**
   `physics.py:115` destroys mid-tick, before step 6, so `stamp_units` refills
   the hole before any reader including the EOS. Only the step-9/9b paths (fire
   burn-through, burst) are exposed to §3.1.2's hazard.
2. Its guard is `is_vacuum` only — `is_ambient` is not checked — so on a
   planetside map it writes a neighbour mean over a Dirichlet-pinned breach
   tile, contradicting §3.2's "the seam is already handled by the pin".
3. **Gate 1 is structurally blind to it**: `repro_destroy_wall_mint.py` calls
   `destroy_wall` directly and never steps a tick.

**Scope decision:** `stamp_units` writes only `atmosphere`, never `gas`, so it
cannot mint bulk N and is not part of this arc's defect. It is **out of scope
for the fix**, but Gate 1b below exists so the interaction is *measured* rather
than assumed, and finding (2) is recorded as a known inconsistency for the
retune pass.

### 3.7 Ordering — a constraint, not an assumption

> **Critique correction (both lenses).** v1's §7 Q4 claimed a constant seed makes
> the multi-tile coupling disappear. It is **false**. The coupling never lived
> in the seed *value*: `destroy_wall` writes `breach_mask[fy,fx] = True` and the
> next tile's `exposes` test reads the live mask, so §3.2 makes that mask decide
> *whether a tile is seeded at all* — a step function per tile, a **strictly
> larger** order sensitivity than the value-level one that already produced a
> measured CPU≠GPU **and** GPU≠GPU divergence
> ([`cuda_fire.cu:58-70`](../cpp/src/cuda_fire.cu#L58-L70)), whose repair is the
> host-side sort at [`:482`](../cpp/src/cuda_fire.cu#L482).

Determinism holds today, but only because four independent callers each happen
to pin their order: the sorted CUDA burn-through list, `find_burst_walls`'
descending-spread sort over a row-major scan, `physics.py`'s nested `dy/dx`
loop, and `door_system`'s row-major span. **No test asserts any of them except
the CUDA one.** This is recorded as a standing constraint and gated (Gate 8).

## 4. Why `T := 0` keeps the energy books closed — VERIFIED

The energy books sum `n_bulk · T_game` over an accountable set that skips
`solid || ts || is_vacuum || is_ambient`. **Verified against the code this
revision:** [`eos_solver.cpp:300`](../cpp/src/eos_solver.cpp#L300) is
`acc += nb * (int64_t)temperature[i]` — no `C`, no `s_eos_q`, no `+ t_amb_q`.
The `T_abs` conversion exists only in the `p*` law and the `c_local` scan,
neither of which feeds the books. `T_MIN = -289.0f`, so 0 is nowhere near the
floor, and `on_tile_changed` clears `thermal_solid` from the air row, so there
is no stale-`ts` exclusion.

A cell joining the accountable set with `T = 0` therefore contributes exactly
`nb × 0 = 0` → **Δ(energy books) is exactly zero, and no energy channel is
needed.**

`destroy_wall` writes no temperature today, so a burning wall currently joins
the books hot. **This patch closes a pre-existing energy-seam hole as a side
effect** — a claim Gate 3 must measure, not assume.

## 5. Scope

**In:** §3.1–§3.4 and §3.7's gate.

**Out, deliberately:**

- **The grenade payload** (260.5 cell-equivalents per throw, 12.3% of the
  recorded mint). Erik's ruling. Untouched by this fix and still open.
- **`burst_threshold` retuning.** With the amplifier gone the valve is
  near-unreachable: popping a wooden interior wall needs the room above **3.0
  atm**, above the intended 2–3 range; hull never bursts (threshold 0.0),
  furniture is not scanned (not solid), steel is 10.0. After this patch the
  relief valve fires essentially only on glass — a canon mechanic
  (`engine/04 §5`) going quiet. **Erik's ruling: retune later.** Recorded on the
  post-arc retune list, which this arc unblocks.
- **The unbound velocity clamp.** |u| = 862 m/s against `c_local` ≈ 640 on 976
  cell-snaps, `U_MAX = 1000` never binding. Separate defect, still open.
- **The full P-M1 mass ledger.** This patch ships the one channel it needs.
- **Rubble porosity.** The principled way to model "the fragments still occupy
  volume" is reduced `dyn_permeability`, not gas from nowhere. Deferred.
- **`unseal_tiles` / doors.** Already conservative and dormant; Erik's reasoning
  distinguishes them. Left alone, but see §7 for its stale comment.
- **`stamp_units`** (§3.6) — measured, not changed.

### ACCEPTED GAPs

- `ACCEPTED GAP:` **Solid thermal energy is not accounted anywhere.** Destroying
  a burning wall discards the heat stored in the wall material, which lives in
  the thermal-solid domain the books never tracked. Pre-existing.
- `ACCEPTED GAP:` **The seed mints on a depressurised map.** Blowing a wall in a
  vacuum-filled ship produces a small puff of air. Bounded at `ambient_N` per
  tile, not self-amplifying, named in the books.
- `ACCEPTED GAP:` **Total minted mass can still be large on destruction-heavy
  maps.** The cave case mints ~2,400 cell-equivalents on a 6,000-cell map. Large
  but benign: spread at ambient, no concentration, no proportionality. P-M1's
  gate asserts `Δ(Σ N) == Σ named channels`, **not** `Δ(Σ N) == 0`.

## 6. Gates

1. **Conservation with attribution, solid path.** `repro_destroy_wall_mint.py`
   promoted to an asserting test: `Δ(Σ N)` from a destruction equals
   `n_destruction_seed_sum` exactly, and the seeded **total** is independent of
   local density — the ×1 / ×10 / ×100 sweep must give the *same* total. This is
   the direct assertion that the feedback loop is broken.
1b. **`stamp_units` interaction.** The same sweep run through a full
   `Simulation.step` rather than a bare `destroy_wall` call, so §3.6's second
   writer is exercised. Measures; does not assert a fix.
2. **Breach path**, stated per map class: on a **space** map, seeded gas is 0 and
   no compensating wipe follows (today: +10 then −10); on an **ambient** map the
   tile is filled by the ambient rail and the difference lands in
   `boundary_flux`, not in `n_destruction_seed_sum`.
3. **Energy books.** `Δ(energy books)` across a destruction is exactly zero,
   including for a **burning** wall. Needs a Python-visible books sum —
   `eth_books_sum` is a function-local lambda today and `bindings.cpp` exports
   only deltas. **Add the binding; do not transcribe the four-flag skip-set into
   Python**, or it will drift.
4. **Furniture path.** A bullet-chewed crate in a pressurised room books a
   **negative** delta, and the books still close.
5. **Full suite green.** Meaningful only after P-M0b; see §9 for why that is now
   two distinct red populations.
6. **Determinism.** Digests unchanged where behaviour is unchanged; the seeded
   values are exact Q16.16 integer writes (`quantize_scalar(0.21) = 13763`,
   `quantize_scalar(0.79) = 51773`, sum `= 65536` exactly), so no new
   quantisation path is introduced.
7. **CPU↔GPU.** Re-run `test_cuda_p64_kick_compression` PART 2 against the fix
   (P-M5). A Python-side seed that both backends inherit is not an explanation
   for a divergence, so if this does not move it, they are two bugs.
8. **Ordering.** A test that destroys the same multi-tile set in two different
   orders and asserts identical `Σ N` and identical seeded/skipped decisions —
   §3.7's constraint made falsifiable.
9. **HUMAN-TEST.** Erik plays: grenades in a sealed room, then breach a
   pressurised one. Expected: no blowup, peak room pressure in the low
   single-digit atmospheres.

## 7. Comments that must be updated in the same patch

Three go stale on merge, and one is load-bearing:

- [`gamemap.py:1764-1767`](../src/simulation/gamemap.py#L1764-L1767) — the A5
  block this design quotes as its evidence.
- [`gamemap.py:2016`](../src/simulation/gamemap.py#L2016) — "*Unlike
  `destroy_wall`, which mint-seeds unconditionally*".
- [`gamemap.py:2098-2100`](../src/simulation/gamemap.py#L2098-L2100) — **load-
  bearing**: it is `unseal_tiles`' documented *reason* for writing `atmosphere`
  + `wave_p` ("*this is the minted display value `destroy_wall` also
  provides… `wave_p` matches so the |P − P_prev| ripple splash sees no phantom
  spike*"). §3.1.2 keeps an `atmosphere` write, so the justification survives in
  altered form and must be re-argued rather than left dangling.

## 8. Critique findings and resolutions

| # | lens(es) | finding | resolution |
|---|---|---|---|
| B1 | **all three** | dropping the `atmosphere` write re-arms `find_burst_walls` in the same tick | §3.1.2 — write it, consistent with the seeded N |
| B2 | scope | `self._ambient` is `None` on space maps → `AttributeError` on `playground` | §3.1.1 — one accessor with the existing fallback |
| B3 | books | the channel cannot appear in a C++ per-tick identity | §3.5 — gate the seam separately, stated explicitly |
| M1 | det + books | furniture is a non-solid destructible holding gas; channel sign/formula wrong | §3.4 — measured, signed; Gate 4 |
| M2 | books | constant 0.21/0.79 re-oxygenates suffocating fires | §3.1.3 — inherit the local O₂ fraction |
| M3 | scope | `stamp_units` is a second neighbour-mean `atmosphere` writer | §3.6 — out of scope, Gate 1b measures it |
| M4 | det + scope | order coupling does not disappear; it moves to the seed *decision* | §3.7 — constraint recorded, Gate 8 |
| M5 | scope + books | the relief valve becomes near-unreachable | §5 — Erik's ruling: retune later |
| M6 | det + scope | goldens will move (`GOLDEN_AGGREGATE`'s scenario breaches an edge hull) | §9 |
| M7 | scope | §3.3's premise false — C++ already clears `fire` on the burn-through path | §3.3 — corrected justification, change kept |
| M8 | books + scope | ambient maps *fill* the breach tile rather than wiping it | §3.2 — stated per map class |
| m1 | all | v1's "0.21/0.79" contradicted its own "read from the map's ambient" | §3.1.1 |
| m2 | det + scope | `p* = N` is off by 4 raw counts on the quantisation lattice | §3.1.2 — state the seed in N only |
| m3 | det | seed representation is exactly Q16.16; the float came from `_neighbor_mean` | Gate 6 records it |
| m4 | scope | one semantic test pins the old behaviour; comments go stale | §7, §9 |

**Verified and dismissed:** §4's energy claim (books lens confirmed
`Σ n_bulk·T_game`); §7 Q3 of v1 (the mask *is* set before the seed on every
path); no CPU↔GPU hazard from the seed itself; no `fire` sync hazard; no
scoring/termination/unit-damage reader of `fire`.

## 9. Test and golden ledger

- **One semantic re-point:** `tests/test_eos_p1_species_transport.py:164`
  `test_destroy_wall_seeds_bulk_gas_by_neighbor_mean` asserts the old
  neighbour-mean values against deliberately off-ambient neighbours. Rename and
  re-point — do not delete; it is the only test that pins this behaviour.
- **One deliberate golden re-baseline:** `GOLDEN_AGGREGATE`
  (`tests/_xarch_perfield_digest.py:155`), whose scenario calls
  `g.destroy_wall(8, 0)` — an edge-hull tile, exactly §3.2's breach-skip case.
  Shared by `test_w6_armory.py` and the PART-3 leg of 11 CUDA check modules.
  **One rebase event, written rationale, lineage entry**, regenerating both
  `_xarch_perfield_DESKTOP-0E98HUV.txt` and `_xarch_perfield_erik_lenovo.txt` in
  the same commit. Do **not** bump `DIGEST_SPEC_VERSION` — values only.
- **Non-vacuousness guards whose margin this eats** — measure, do not blind-
  rebase: `test_a5_seal_evacuation.py:269` (partly guaranteed today by the very
  mint-then-delete round trip §3.2 removes), `cuda_ambient_check.py:180-183` and
  `cuda_s8a_check.py:344-347` (both "scenario too tame" guards driven by
  ring-adjacent breaches that now seed nothing).
- **Two distinct pre-existing red populations, keep them separate** or "green"
  becomes unfalsifiable: P-M0b's fire-signature reds, and the 12 CUDA golden-leg
  reds that this patch re-baselines.

## 10. Execution note

Agent worktrees cannot run this suite — the built `breach_physics*.pyd` is
gitignored build output, and a fresh worktree collects 0 tests (72 collection
errors). Any agent running gates must **copy the built extension from the
primary worktree as its first step**. Valid because this patch is Python-only
(`gamemap.py`); a patch touching `cpp/` must build for real instead.
