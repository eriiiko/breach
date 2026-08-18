# P-M3 — `destroy_wall` seeding: design (2026-08-18, **v3**)

**Arc:** mass-books (`docs/mass_books_arc_kickoff_2026-08-18.md`).
**Status:** v3, post-critique-round-2. Feel-adjacent → **HUMAN-TEST before merge**.
**Decision owner:** Erik. Decisions recorded here were taken with him in session;
this document records them and their gates, it does not re-open them.

> **Revision history.**
> **v1** (`2d4c6d8`) — first draft. Three-lens critique (determinism,
> conservation-books, scope/regression) found a **blocker**: v1's instruction not
> to write `atmosphere` would have replaced a mass amplifier with a *burst*
> amplifier in the same tick, plus 14 further findings.
> **v2** (`30aa029`) — resolved those. A second, narrow critique of only the
> material v2 *added* found **two more blockers** and four majors, all in the new
> material: a tautological gate, a seam premise that is factually wrong, and an
> unbooked mass-loss path created by the interaction of two v2 fixes.
> **v3** (this) — resolves those. §8 carries both rounds' findings.
>
> Surviving unchanged since v1 and independently verified against the code:
> §4's energy claim.

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
also skips vacuum neighbours, so at a hull between a pressurised room and space
the mean is taken over the pressurised side only — peak seed at peak
differential.

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
volume. The sim carries no rubble, so a rule that withdraws gas from the
neighbours charges the player for volume the model never gave them.

The cave case makes it concrete: a map 80% solid, blasted open to 40% solid,
would under withdrawal have its pressure fall by more than half — a cave that
suffocates you for digging it out. Real caves stay breathable because they are
connected to a reservoir. **Seeding ambient models the reservoir that is
actually there**, and on `boundary = ambient` maps it is not even a fiction.

**Decision: seed a CONSTANT TOTAL at the map's ambient, and book it.**

The load-bearing property is *constant total N*, not the value and not the
composition. A constant total breaks the feedback loop with the burst valve
dead — the seed no longer scales with the pressure that triggered the burst.

> **Correction carried from v2.** v1 also argued withdrawal "models an expansion
> into void that does not physically happen". That half is **withdrawn**: the
> transport pass performs the expansion anyway on the next tick — a seeded cell
> at N=1.0 beside donors at N=5.0 is filled by donor-cell flux, drawing the
> neighbours down regardless. The constant seed does not prevent the drawdown;
> it adds `ambient_N` on top of it. The reservoir/cave argument stands alone.

## 3. Specification

### 3.1 The seed

| field | value |
|---|---|
| `gas[O2]` + `gas[INERT_N2]` | **total** = the map's ambient N (constant); **split** = the local O₂ mole fraction (§3.1.3) |
| `temperature` | **0**, explicitly written, on **every** destroyed tile including breach tiles |
| `atmosphere` | **written**, to the map's effective pin (§3.1.2) |

#### 3.1.1 One accessor, no fifth copy of the constant

`self._ambient` is **`None` on every space map** — it is only populated under
`if self._boundary == "ambient"`
([`gamemap.py:638-646`](../src/simulation/gamemap.py#L638-L646)). `playground`,
the map this entire audit ran on, is `boundary = space`.

Four places already derive this constant, by two derivations that agree **only**
at default dials. Add exactly one accessor — `GameMap.ambient_seed()` returning
`(n_total_q, o2_q, n2_q, pin_q)` — reusing the existing fallback logic
([`gamemap.py:1043-1049`](../src/simulation/gamemap.py#L1043-L1049)):

- `_ambient` present → `derive_ambient`'s values, where `n_total =
  quantize(p_amb)` and the split uses the level's authored `o2_frac`;
- `_ambient is None` → `FP_ONE`, `quantize_scalar(0.21)` = 13763,
  `quantize_scalar(0.79)` = 51773, summing to 65536 exactly.

Every other site routes through it. Gates key off the accessor, never `FP_ONE`.

#### 3.1.2 The `atmosphere` write — restored

A solid tile's `atmosphere` is hard 0 — the MG solve zeroes it
([`eos_solver.cpp:1409`](../cpp/src/eos_solver.cpp#L1409)) and that is what gets
stored. Two `destroy_wall` callers run **after** the physics step and are
refilled by nothing in that tick: fire burn-through
([`simulation.py:1415`](../src/simulation/simulation.py#L1415)) and the burst
valve ([`:1431`](../src/simulation/simulation.py#L1431)). `find_burst_walls`
then runs at [`:1429`](../src/simulation/simulation.py#L1429) and reads
`float(atm[ny, nx])` for any neighbour that is not solid, vacuum or ambient.

Worked case at this arc's own target pressure: a wood/door wall between a
2.1 atm room and a 1.0 atm corridor holds spread 1.1 — safe. Punch one hole with
`atmosphere` unwritten and its along-wall neighbours see sides {2.1, 1.0,
**0.0**} → spread 2.1 > `burst_threshold` 2.0 → they pop, up to
`burst_max_per_tick = 16` per event. **v1 would have removed the amplifier at
the mass end and installed one at the burst end**, same code path, same tick.

The dependency is documented at
[`gamemap.py:1638-1641`](../src/simulation/gamemap.py#L1638-L1641): the ambient
branch exists because *"a just-breached ring tile is seeded by neighbour-mean,
not yet re-pinned"*.

Three further same-tick readers of a stale 0 confirm it is not display-only: the
MG **warm start** (`p_prev := atmosphere`), `apply_wave_push` (a ~1 atm phantom
∇P that can fire `KNOCKED_DOWN` on units beside a burnt-through wall), and the
`PRESSURE` sensor channel — an RL observation.

Write `atmosphere := pin_q` from the accessor. **Do not assert `p* == N`**: the
effective pin is 65540 raw (1.000061 atm) at Earth defaults, not 65536
([`ambient.py:21-24`](../src/simulation/ambient.py#L21-L24)). State the seed
purely in N.

#### 3.1.3 Composition — inherited, with an exactly-specified integer form

A fixed 0.21/0.79 seed injects fresh oxidizer: combustion is linear in the O₂
mole fraction with a 2-hop draw radius, so each destroyed tile would drop a
21%-O₂ cell inside the draw radius of the rest of a burning wall run.

**Donor set (PINNED, not inherited by accident):** the 4-neighbours that are
`not solid and not is_vacuum` — the same predicate `_neighbor_mean` uses today.
`is_ambient` cells **are** included: on a planetside map the reservoir
composition is the right thing to inherit toward.

**Arithmetic — one exact int64 form, one rounding, no intermediate ratio:**

```python
sum_o2 = int(Σ gas[O2][donors])        # int64, never narrowed
sum_n  = int(Σ (gas[O2]+gas[N2])[donors])
if sum_n > 0:
    o2 = (n_total_q * sum_o2 + sum_n // 2) // sum_n     # round-half-up, exact
else:
    o2 = o2_q                                          # §3.1.4 fallback
n2 = n_total_q - o2                                    # exact complement
```

`0 ≤ o2 ≤ n_total_q` is guaranteed because `sum_o2 ≤ sum_n` (per-plane N ≥ 0 is
held by `bulk_transport.cpp:234-235`), so `n2` can never go negative. **No float
enters the sim path** — the current float comes only from `_neighbor_mean`'s
`total / count`, which this replaces.

> This closes the two hazards critique round 2 raised: written naively as
> `sum_o2 / sum_n` this is a float against the iron rule; written via the house
> `mul_q16(reciprocal_q16(...))` idiom the reciprocal's rounding can push the
> fraction to `FP_ONE` and make `n2` **negative**, which `bulk_transport` then
> silently clamps to 0 — an unbooked mint one substep after the books recorded
> a smaller delta.

#### 3.1.4 The fallback, and what it costs

`sum_n == 0` is **not an edge case**. It fires whenever no donor holds gas —
most importantly when all four neighbours are solid, which is the interior tile
of a ≥2-thick slab, i.e. ordinary blast geometry. It also fires when all
neighbours are vacuum, and when the room is fully vented but not vacuum-flagged.

**Erik's ruling: fall back to the map's ambient composition.** The alternative
(pure N₂ — "no information ⇒ inject no oxidizer") is safer for fire but breaks
the cave case in §2, which is the entire motivation for the ambient seed:
digging out a cave would fill it with nitrogen and suffocate the player.

- `ACCEPTED GAP:` **blasting a burning slab briefly feeds the fire.** Each
  destroyed interior tile of a burning slab seeds one cell at the map's ambient
  O₂ fraction inside combustion's 2-hop draw radius. Bounded at `ambient_N` per
  tile, does **not** scale with pressure, and is arguably the behaviour wanted
  anyway — breach a burning wall, air rushes in, the fire flares. Recorded as a
  decision, not a silent consequence.

### 3.2 Breach tiles — skip the seed, but evacuate what is already there

A destroyed tile that joins the vacuum/ambient boundary is Dirichlet-pinned.
Skip the gas seed there — on a **space** map it would otherwise be seeded and
then wiped (measured: +10, then −10 next tick).

> **Per map class.** On an **ambient** map `breach_mask` is `is_ambient` and the
> clamp does the *opposite*: it **fills** the tile to `N_amb` each substep and
> books the difference to `boundary_flux`
> ([`bulk_transport.cpp:231-233`](../cpp/src/bulk_transport.cpp#L231-L233)).
> Both are correct to skip; the *reason* differs and the gate must name the map
> class.

**A skip alone is not sufficient — it opens an unbooked mass sink.** Critique
round 2 found this by composing two of v2's own fixes. `destroy_wall`'s gate is
`material != MAT_AIR`, and furniture is a **non-solid** destructible that
**already holds N** (§3.4). Chew a crate adjacent to vacuum: `breach_mask` is
set, §3.2 skips the seed, the crate's existing atmosphere-or-more of N is never
booked, and the next transport pass zeroes it via
[`bulk_transport.cpp:229-230`](../cpp/src/bulk_transport.cpp#L229-L230) — which
carries **no `boundary_flux` credit**. Mass vanishes with a channel on neither
side of the seam.

**Therefore: on a breach tile, explicitly zero `gas[O2]`/`gas[INERT_N2]` at
destroy time** rather than leaving them for the transport clamp. The removal
then lands inside the measured delta (§3.4) and is attributed. Nothing is left
for the unbooked wipe to take.

`temperature := 0` is written on breach tiles too: on an ambient map a burning
hull tile would otherwise join `is_ambient` still carrying the wall's hot T, and
the `c_local` scan skips only `solid || is_vacuum` — inflating map-wide sound
speed and substep count for a tick.

Use `breach_mask[fy, fx]` as the skip predicate (mirroring `unseal_tiles`'
`joins_boundary`) rather than re-deriving `on_edge_hull or exposes`, so the two
cannot drift. The mask is set at
[`gamemap.py:1751`](../src/simulation/gamemap.py#L1751), unconditionally before
the seed, on every path — verified.

### 3.3 Clear `fire` — corrected justification

> **Corrected in v2.** v1's premise was false for the path it named:
> `fire_simulation.cpp:317-320` already zeroes `fire` on every tile it emits in
> the burn-through list, and the CUDA twin mirrors it. Stale `fire` is real only
> on the *other* callers — burst, explosion, bullet-chew, door — because
> `on_tile_changed` patches ten caches and never `fire`.

Keep the change; the honest justification is: a stale display/sensor cleanup,
and it stops `fire_simulation.cpp:311-316` decrementing a now-air tile's
`wall_hp` forever. It is **not** a change to fire spread — cellular spread no
longer exists, `apply_temperature_ignition` is `flammable`-gated so it cannot
relight an air tile, and a destroyed burning tile is already dropped as a
radiation emitter. No exploit: shooting walls extinguishes nothing that was
still doing work. It **does** change the `FIRE` sensor channel — an RL
observation change, and that is the reason to record.

### 3.4 Booking — `n_destruction_seed_sum`, measured and signed

> **Corrected in v2.** v1 specified this `≥ 0` and equal to `ambient_N ×
> tiles_seeded`. Both wrong. Furniture ships `permeability = 0.5` → **not
> solid** → it already holds bulk N, so writing a constant into it is
> `seed − prior`, **negative** whenever the room is above ambient: chewing a
> crate at 5 atm *deletes* ~4 cell-equivalents.

The channel is **measured, not derived from a formula**: bracket each
`destroy_wall` call, sampling `Σ N` before and after, and accumulate the
difference into a **signed** counter. v1's "trivially predictable / cheapest
channel to gate" claim is retracted.

On the **solid** path specifically, the prior N is 0, so the seeded delta *is*
predictable at `n_total_q` per tile — and that is what Gate 1 asserts, because a
gate that compares the measured channel against the measured delta is a
tautology (§8, round 2 B1).

### 3.5 The seam — per-call-site brackets, because there is no "destruction block"

> **v2 was factually wrong here.** It asserted destruction runs *after*
> `physics_runner.step()`. It does not. Critique round 2 enumerated all six call
> sites against `Simulation.step`:

| site | tick slot | vs `physics_runner.step()` (`simulation.py:1406`) |
|---|---|---|
| grenade / explosion (`physics.py:115`) | 2 | **before** |
| bullet chew (`combat.py:178`) | 2 | **before** |
| beam chew (`combat.py:178`) | 4 | **before** |
| fire burn-through (`simulation.py:1415`) | 9 | after |
| burst valve (`simulation.py:1431`) | 9b | after |
| door assembly death (`door_system.py:241`) | 9e | after |

Destruction is scattered across **three regions straddling the physics step**, so
"bracket the destruction block" is unimplementable. A bracket around slots 9/9b
misses slots 2 and 4 entirely; a bracket around the whole of `Simulation.step`
encloses `physics_runner.step()`, whose own `Σ N` legitimately changes via the
ambient rail and the (separately unbooked) vacuum sink, so the identity would
fail on any map with a breach.

**Decision: bracket inside `destroy_wall` itself** — one sample pair per call,
accumulating into the signed counter. That is the only form that covers all six
sites without enclosing solver behaviour, and it makes the channel correct by
construction regardless of where in the tick a caller fires.

The C++ per-tick identity is left alone and continues to assert only what
happens inside the solver. **This split is now stated rather than implicit** —
the kickoff §3.4 required exactly that.

### 3.6 `stamp_units` — out of scope, with its reach stated accurately

A **second** `atmosphere` neighbour-mean seed exists: `stamp_units` re-seeds
every tile freed by a unit moving off it
([`gamemap.py:1380-1388`](../src/simulation/gamemap.py#L1380-L1388)), at step 6,
*before* the EOS at step 7. Consequences:

1. **For the explosion path this patch is a no-op on `atmosphere`** —
   `physics.py:115` destroys at slot 2, so `stamp_units` refills the hole before
   any reader including the EOS. Only slots 9/9b/9e are exposed to §3.1.2.
2. Its guard is `is_vacuum` only — `is_ambient` is not checked — so on a
   planetside map it writes a neighbour mean over a Dirichlet-pinned breach tile.
3. `repro_destroy_wall_mint.py` calls `destroy_wall` directly and never steps a
   tick, so no bare-repro gate can see it.

**Scope call: out of scope.** Verified: `stamp_units` writes only
`self.atmosphere` on both paths and never `self.gas`, and nothing anywhere
converts `atmosphere` back into `gas` — so it **cannot mint bulk N**.

> **But do not overstate it** (round 2, m8). An `atmosphere` write is a *wind*
> source, and it reaches the MG warm start (`p_prev := atmosphere`,
> [`eos_solver.cpp:320`](../cpp/src/eos_solver.cpp#L320)), which drives the
> substep-count estimate and the warm start, hence velocity, hence donor-cell
> flux, hence **how much mass vents at a breach** — and vacuum venting is itself
> unbooked. So: *cannot mint; can redistribute, and can change vented mass.*
> Caveat for the burst path: `simulation.py:1399` guards the physics step on
> `physics_runner is not None`, so with physics disabled the neighbour mean
> **does** survive into `find_burst_walls`.

### 3.7 Ordering — a constraint, not an assumption

> **Corrected in v2, and strengthened in v3.** v1 claimed a constant seed makes
> the multi-tile coupling disappear. False on two counts.

**Decision-level coupling.** `destroy_wall` writes `breach_mask[fy,fx] = True`
and the next tile's `exposes` test reads the live mask, so §3.2 makes that mask
decide *whether a tile is seeded at all* — a step function per tile, a strictly
larger sensitivity than the value-level one that already produced a measured
CPU≠GPU **and** GPU≠GPU divergence
([`cuda_fire.cu:58-70`](../cpp/src/cuda_fire.cu#L58-L70)), whose repair is the
host-side sort at [`:482`](../cpp/src/cuda_fire.cu#L482).

**Value-level coupling, reintroduced by §3.1.3.** `on_tile_changed` clears
`solid` *before* the seed runs, so destroying tile A then its neighbour B means
B's donor set includes A's freshly-seeded composition. Different order → same
total, **different O₂/N₂ split** → different combustion. `Σ N` is order-invariant
by construction and therefore cannot detect this; the gate must compare the
per-tile pair.

Determinism holds today only because four independent callers each pin their
order: the sorted CUDA burn-through list, `find_burst_walls`' descending-spread
sort over a row-major scan, `physics.py`'s nested `dy/dx` loop, and
`door_system`'s row-major span. **No test asserts any of them except the CUDA
one.** Gate 8 gates the four pins; Gate 8b gates the per-tile split.

## 4. Why `T := 0` keeps the energy books closed — VERIFIED

The energy books sum `n_bulk · T_game` over an accountable set that skips
`solid || ts || is_vacuum || is_ambient`. **Verified against the code:**
[`eos_solver.cpp:300`](../cpp/src/eos_solver.cpp#L300) is
`acc += nb * (int64_t)temperature[i]` — no `C`, no `s_eos_q`, no `+ t_amb_q`.
The `T_abs` conversion exists only in the `p*` law and the `c_local` scan,
neither of which feeds the books. `T_MIN = -289.0f`, so 0 is far from the floor,
and `on_tile_changed` clears `thermal_solid` from the air row, so there is no
stale-`ts` exclusion.

A cell joining with `T = 0` contributes exactly `nb × 0 = 0` → **Δ(energy books)
is exactly zero; no energy channel is needed.**

`destroy_wall` writes no temperature today, so a burning wall currently joins
the books hot. **This patch closes a pre-existing energy-seam hole as a side
effect** — Gate 3 must measure that, not assume it.

## 5. Scope

**In:** §3.1–§3.5, §3.7's gates.

**Out, deliberately:** the grenade payload (Erik's ruling); **`burst_threshold`
retuning** (Erik's ruling — with the amplifier gone the valve is
near-unreachable: wood needs the room above 3.0 atm, hull never bursts,
furniture is not scanned, so it fires essentially only on glass; a canon
mechanic going quiet, recorded on the post-arc retune list); the unbound
velocity clamp; the full P-M1 ledger; rubble porosity; `unseal_tiles`/doors;
`stamp_units` (§3.6).

### ACCEPTED GAPs

- `ACCEPTED GAP:` **Solid thermal energy is not accounted anywhere** — destroying
  a burning wall discards heat stored in the wall material, which lives in the
  thermal-solid domain the books never tracked. Pre-existing.
- `ACCEPTED GAP:` **The seed mints on a depressurised map** — blowing a wall in a
  vacuum-filled ship produces a small puff of air. Bounded, not amplifying,
  named in the books.
- `ACCEPTED GAP:` **Total minted mass can be large on destruction-heavy maps** —
  the cave case mints ~2,400 cell-equivalents on a 6,000-cell map. Benign:
  spread at ambient, no concentration, no proportionality. P-M1 asserts
  `Δ(Σ N) == Σ named channels`, **not** `Δ(Σ N) == 0`.
- `ACCEPTED GAP:` **Blasting a burning slab briefly feeds the fire** — §3.1.4.

## 6. Gates

Every gate below must be able to go **red**. Round 2 found three that could not;
each is now stated as a falsifiable proposition.

1. **Solid path, predicted value.** `Δ(Σ N)` over a destruction of *k* solid wall
   tiles equals **`k × ambient_seed().n_total_q`** exactly, and each seeded tile
   holds `gas[O2] + gas[INERT_N2] == n_total_q`.
   *Not* `Δ(Σ N) == n_destruction_seed_sum` — with §3.4's measured channel that
   is `A == A` and stays green against today's unfixed mint.
2. **Density independence + composition.** Run the ×1 / ×10 / ×100 sweep with
   **`atmosphere` scaled alongside `gas`** (today the tool scales gas only, so an
   `atmosphere`-sourced seed would be invisible), and assert the seeded
   **`(O2, N2)` pair** — not merely the total — is identical at every scale.
   A total-only assertion has zero discriminating power over composition: a split
   implemented as `o2 := ΣO2/4` with `n2 := n_total − o2` gives the right total
   at every scale while injecting density-scaled oxidizer and a negative `n2`.
3. **Breach path**, stated per map class: on a **space** map the tile ends with
   zero gas and no compensating wipe follows (today: +10 then −10); on an
   **ambient** map the tile is filled by the ambient rail with the difference in
   `boundary_flux`, not in `n_destruction_seed_sum`.
4. **Breach × furniture** (§3.2's escape). Chew a crate adjacent to vacuum: the
   crate's prior N appears in `n_destruction_seed_sum` as a **negative** delta,
   and no mass is left for the transport clamp to take unbooked.
5. **Furniture in a pressurised room.** A signed, negative delta; books close.
6. **Energy books.** `Δ(energy books)` across a destruction is exactly zero,
   including for a **burning** wall. Requires a Python-visible books sum —
   `eth_books_sum` is a function-local lambda and `bindings.cpp` exports only
   deltas. **Add the binding; do not transcribe the four-flag skip-set into
   Python**, or it will drift.
7. **Determinism.** Seeded values are exact Q16.16 integer writes; §3.1.3's form
   introduces no float and no new quantisation path. Note `quantize_scalar(0.21)
   = 13763` / `(0.79) = 51773` are the **fallback** values only — on the
   inherited path the split is whatever §3.1.3 yields.
8. **Caller order pins.** Assert the four orders §3.7 depends on: the CUDA
   burn-through sort, `find_burst_walls`' descending-spread sort, `physics.py`'s
   `dy/dx` nesting, `door_system`'s row-major span. Three of the four are
   currently ungated.
   **8b. Per-tile split under reordering.** Destroy an adjacent pair in both
   orders and compare the per-tile `(O2, N2)` — *not* `Σ N`, which is
   order-invariant by construction and cannot fail.
9. **No new failures** against the diagnosed baseline (31 failed / 2205 passed at
   `de5dfd9`). "Full suite green" is unreachable and always was; §9 keeps the
   two red populations distinct.
10. **CPU↔GPU.** Re-run `test_cuda_p64_kick_compression` PART 2 (P-M5). A
    Python-side seed both backends inherit cannot explain a *divergence*, so if
    it does not move, they are two bugs.
11. **HUMAN-TEST.** Erik plays: grenades in a sealed room, then breach a
    pressurised one. Expected: no blowup, peak room pressure in the low
    single-digit atmospheres.

**Measurement, not a gate:** run the sweep through a full `Simulation.step` to
observe §3.6's `stamp_units` interaction. It is explicitly not assert-bearing —
and note the legs are not comparable across scales, because the burst valve
fires on a different tile set at each scale (capped at 16/tick), so the ×1/×10/
×100 runs destroy different tiles. Report it; do not count it toward gates green.

## 7. Comments that must be updated in the same patch

- [`gamemap.py:1764-1767`](../src/simulation/gamemap.py#L1764-L1767) — the A5
  block this design quotes as its evidence.
- [`gamemap.py:2016`](../src/simulation/gamemap.py#L2016) — "*Unlike
  `destroy_wall`, which mint-seeds unconditionally*".
- [`gamemap.py:2098-2100`](../src/simulation/gamemap.py#L2098-L2100) — **load-
  bearing**: `unseal_tiles`' documented *reason* for writing `atmosphere` +
  `wave_p` cites "*the minted display value `destroy_wall` also provides*".
  §3.1.2 keeps an `atmosphere` write, so the justification survives in altered
  form and must be re-argued rather than left dangling.

## 8. Critique findings and resolutions

**Round 1** (on v1 — determinism / books / scope lenses):

| # | finding | resolution |
|---|---|---|
| B1 | dropping the `atmosphere` write re-arms `find_burst_walls` in the same tick | §3.1.2 |
| B2 | `self._ambient` is `None` on space maps → `AttributeError` on `playground` | §3.1.1 |
| B3 | the channel cannot appear in a C++ per-tick identity | §3.5 |
| M1 | furniture is a non-solid destructible holding gas; channel sign/formula wrong | §3.4 |
| M2 | constant 0.21/0.79 re-oxygenates suffocating fires | §3.1.3 |
| M3 | `stamp_units` is a second neighbour-mean `atmosphere` writer | §3.6 |
| M4 | order coupling does not disappear | §3.7 |
| M5 | the relief valve becomes near-unreachable | §5 — Erik: retune later |
| M6 | goldens will move | §9 |
| M7 | §3.3's premise false — C++ already clears `fire` on burn-through | §3.3 |
| M8 | ambient maps *fill* the breach tile rather than wiping it | §3.2 |
| m1–m4 | 0.21/0.79 contradiction; `p*` off by 4 counts; representation; stale comments | §3.1.1, §3.1.2, Gate 7, §7 |

**Round 2** (on v2's new material only):

| # | finding | resolution |
|---|---|---|
| **B1** | Gate 1 was a tautology (`A == A`) and stays green against the unfixed mint | Gate 1 asserts the **predicted** solid-path value |
| **B2** | §3.5's premise false — destruction spans three regions straddling the physics step; and **breach × furniture loses mass unbooked** | §3.5 brackets inside `destroy_wall`; §3.2 evacuates on breach; Gate 4 |
| M3 | the fallback fires on the *most common* case, reinstating M2's failure mode | §3.1.4 — Erik's ruling + `ACCEPTED GAP:` |
| M4 | §3.1.3's arithmetic unspecified — float in the sim path, or negative `n2` | §3.1.3 — one exact int64 form |
| M5 | Gate 8 blind to the value-level coupling §3.1.3 reintroduces | §3.7, Gate 8b |
| M6 | Gate 1's sweep scales `gas` only; total-only assertion cannot see a split bug | Gate 2 |
| m7 | Gate 1b was not a gate | demoted to a measurement |
| m8 | §3.6's "not part of this arc's defect" overstates | §3.6 — "cannot mint; can redistribute and change vented mass" |

**Verified and dismissed:** §4's energy claim; that §3.1.3 does **not** re-close
the feedback loop (`X_local` is a ratio over one donor set, hence scale-
invariant, so absolute O₂ per tile is bounded by `ambient_N` regardless of
pressure); that the seeded mole fraction matches what combustion reads, since
`n2 := n_total − o2` is an exact complement; the seed's Q16.16 representation;
no CPU↔GPU hazard from the seed itself; no `fire` sync hazard; no
scoring/termination/unit-damage reader of `fire`.

## 9. Test and golden ledger

- **One semantic re-point:** `tests/test_eos_p1_species_transport.py:164`
  `test_destroy_wall_seeds_bulk_gas_by_neighbor_mean` — the only test pinning
  this behaviour. Rename and re-point; do not delete.
- **One deliberate golden re-baseline:** `GOLDEN_AGGREGATE`
  (`tests/_xarch_perfield_digest.py:155`), whose scenario calls
  `g.destroy_wall(8, 0)` — an edge-hull tile, exactly §3.2's case. Shared by
  `test_w6_armory.py` and the PART-3 leg of 11 CUDA check modules. **One rebase
  event, written rationale, lineage entry**, regenerating both
  `_xarch_perfield_DESKTOP-0E98HUV.txt` and `_xarch_perfield_erik_lenovo.txt` in
  the same commit. Do **not** bump `DIGEST_SPEC_VERSION` — values only.
- **Non-vacuousness guards whose margin this eats** — measure, do not blind-
  rebase: `test_a5_seal_evacuation.py:269` (partly guaranteed today by the very
  mint-then-delete round trip §3.2 removes), `cuda_ambient_check.py:180-183` and
  `cuda_s8a_check.py:344-347` (both "scenario too tame" guards driven by
  ring-adjacent breaches that now seed nothing).
- **Keep the two red populations distinct** or "green" is unfalsifiable: the 31
  diagnosed baseline reds (fire re-tune drift — `docs/TODO.md` item 3), and the
  12 CUDA golden-leg reds this patch re-baselines.

## 10. Execution note

Agent worktrees cannot run this suite — the built `breach_physics*.pyd` is
gitignored build output, and a fresh worktree collects 0 tests (72 collection
errors); an agent there will assert fixes it cannot verify, which has already
happened once. Any agent running gates must **copy the built extension from the
primary worktree as its first step**. Valid because this patch is Python-only
(`gamemap.py`); a patch touching `cpp/` must build for real.
