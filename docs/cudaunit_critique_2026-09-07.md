# Adversarial critique — `docs/architecture/engine/17_cudaunits.md` (draft 2026-09-07)

Reviewed against: the chapter itself; `docs/cudaunit_canon_brief_2026-09-07.md`;
`docs/cudaunit_lit_brief_2026-09-07.md`; `docs/rl_env_arc_proposal_2026-08-27.md`
§A/§B3; `docs/architecture/engine/02_state_and_ownership.md`;
`14_determinism_and_number_ingress.md`; repo CLAUDE.md; and code spot-checks of
`src/simulation/simulation.py::step` (slots 9–9f), `tests/field_digest.py`,
`src/simulation/exchange.py` (COUPLING_TABLE / 9c row), `cpp/src/fixed_point.h`
(FP_HD + trig kit), `docs/fauna_design_inputs_2026-09-07.md`, and issue #63.

Verified-clean claims, for the record: no slot "9c4" exists today (9c/9c2/9c3/9d
only — no collision); the `__entity__` presence-gated digest precedent is real
and works as described; `sin_q16`/`cos_q16` wrap internally (arbitrary q16
radians are fine as *inputs*); Philox-4x32-10 genuinely needs only 32×32→64
multiplies; the fauna doc's Erik rulings (larvae = first cudaunit, queen = CPU)
are honored; the 10 m larva is legitimately routed to the CPU-unit queen/large-
specimen ruling, not dropped.

---

## BLOCKERS

### B1. The RNG key has no seed. (§3)
Key = `(unit_id, stream_salt)`, counter = `(tick, draw_index)`. **The match seed
appears nowhere.** Consequence: every run of every level with the same unit_ids
draws identical random streams — every larva tumbles at the same ticks with the
same turn angles in every playthrough forever. This also breaks the RL story
(§B4 resets take a seed buffer; the cudaunit streams would ignore it) and makes
the kit unusable as the owed door-4 stats sampler (§3's own claim), whose entire
purpose is seed-dependent deterministic draws. The canon brief's open fork 9
explicitly specified `(seed, unit_id, tick, draw_idx)`; the design silently
dropped the seed.
**Fix:** Philox-4x32 has a 2×32 key and 4×32 counter — there is room. E.g.
key = `(seed_lo ^ unit_id, seed_hi ^ stream_salt)` or counter =
`(tick, draw_index, salt, 0)` with key = `(seed_lo, seed_hi ^ unit_id)`. Pick
one, write it down, and state which seed (the `sim.rng` seed? a separate
cudaunit seed that is match-setup material like the table hash?).

### B2. The behavior law requires per-unit state the "complete roster" does not contain. (§2 vs §4)
§4 RUN: "the tumble rate *drops* while temperature **changes** toward
`T_prefer`" — temporal gradient sensing (Berg & Brown, correctly cited as the
whole point: no spatial gradient computed). Sensing a *change* requires memory
of the previous sample. The §2 table — declared "the pilot's complete roster" —
has no `T_prev` (or low-passed temperature memory) field. As written the
headline mechanism is unimplementable from the declared state. The same table
also lacks anything for "an eat draw stops the larva for a few ticks" (see M1).
An implementer will quietly add a scratch array, and per-unit scratch that
survives ticks IS synced state: it forks the tumble stream, hence positions,
hence the digest — the exact class of miss `dem_acc` (digest v3) was added to
catch.
**Fix:** add `T_prev` (int32 Q16.16) to the synced roster and to
`CUDAUNIT_SECT_V1`, and re-audit §4 for any other cross-tick memory (eat-stop,
COMA entry hysteresis). Alternatively rule the sensing spatial (read the tile
ahead vs own tile) and rewrite §4's biology framing honestly — but then the
turn-direction bias and tumble modulation both become spatial and the Berg/Luo
citations need re-scoping.

### B3. Slot 9c4 vs "orchestration lands in PhysicsEngine" is a contradiction the design papers over — and it hides a real CPU/GPU divergence. (§5)
Three intertwined problems:

1. **Placement is double-booked.** §5 says both "one conductor line: new slot
   9c4" (in `Simulation.step`, i.e. *after* `physics_runner.step` returns) and
   "orchestration lands in PhysicsEngine (§A rule 2)". PhysicsEngine runs
   *inside slot 7*. A 9c4 conductor line calling a solver member is a second
   per-tick C++ entry outside the resident chain — legal, but then the canon
   brief's explicitly flagged question ("ordering vs the once-per-tick D2H at
   `_step_resident` step 6 must be settled") is left unsettled. The design
   never mentions the D2H at all.
2. **The divergence:** at slots 9/9b (before 9c4), `destroy_wall` mutates
   `material`/`solid` on the **host**. The CPU twin at 9c4 reads host arrays —
   post-destruction. The GPU kernel at 9c4 reads the **device** copy of
   `solid`, which is not refreshed until next tick's H2D. On any
   wall-destruction tick, CPU and GPU larvae probe different walls →
   bit-divergence. The 3-part gate would eventually catch it, but the design's
   "bit-identical by construction" claim (§5) is false as specified, because
   the *inputs* are not pinned, only the arithmetic.
3. **Mirror staleness downstream:** if the 9c4 kernel mutates device food /
   larva arrays after the resident D2H, the host mirror is stale for the
   recorder snapshot at end of tick (recorder reads gmap mirrors) — either a
   second targeted D2H is needed (design silent; cost per tick) or the
   recorder records stale food.

**Fix:** pick ONE placement and specify the input snapshot exactly. The clean
option: run the cudaunit step at the **end of the resident physics chain,
inside slot 7** (device fields maximally fresh, one D2H covers everything,
CPU-backend twin runs at the same point in `PhysicsRunner.step`) and accept
that larvae read *last* tick's wall destructions (one-tick lag, same as every
solver — document it as the contract). If 9c4-after-9c is truly wanted (it
buys nothing visible: larvae don't interact with units), the design must
specify the targeted H2D (solid/material deltas) before the kernel and the
targeted D2H (food, larva arrays) after it, and account both under the
"zero mid-tick transfers" ambition of §B2. Either way, the "sim never reads
stale" clause of engine/02 needs an explicit, scoped exception written down.

---

## MAJOR

### M1. EAT is underspecified to the point of being unimplementable order-free. (§4, §8)
- "an eat draw stops the larva for a few ticks" — for *how many*? If a stored
  countdown: missing synced state (see B2). If memoryless (each tick on food,
  draw `u < eat_stop_prob` to stay stopped — geometric duration): fine and
  stateless, but then SAY so; the current wording implies a timer.
- "decrements the food cell (integer atomicAdd)" — by how much (`eat_rate` per
  stopped tick?), and against what threshold ("where the food field is high" —
  there is no `food_threshold` row in the species table).
- **Concurrent over-consumption:** fork 7a lets unlimited larvae share a tile.
  Ten larvae each atomicAdd −eat_rate into a cell holding 3: the cell goes
  negative. A saturating decrement is NOT order-free (which larva "got" the
  last unit depends on atomic timing — harmless in v1 only because eating has
  no per-larva effect, but the moment satiety exists this is the
  nondeterminism). Design must pick the law now: (a) allow negative, clamp to
  zero in a follow-up pass (deterministic, breaks food conservation —
  acceptable, but state it), or (b) two-pass: count eaters per tile
  (atomicAdd), then a second kernel divides stock deterministically.
- Eating gives the larva **nothing** (no satiety, hp, growth). Legitimate v1
  scope, but the doc should state it — "foraging that never consumes is
  scenery" cuts the other way too: consuming that never nourishes is a decal
  with a ledger.
- "dist_walked ... drives eat-cadence draws" (§2) — nowhere defined. Draws per
  tick, or triggered at distance thresholds? If distance-triggered, specify
  the draw_index accounting.

### M2. `valid` vs `ai_state == DEAD` — two fields, one fact, and no reclaim lifecycle. (§2, §7)
Engine/02's own state-economy rule: never store two independent fields encoding
one fact. When a larva dies: DEAD state, but is `valid` still 1 (it must be, if
DEAD is digested per-slot and the husk renders from `ai_state`)? Then when does
the slot become reusable — who flips `valid` to 0, after what (a husk timer? a
render concern leaking into sim state?), and what prevents a spawn overwriting
a still-rendering husk? §7 says "death = mask flip + tick event" — *which*
mask, and if death immediately flips `valid=0`, how does §9's "DEAD renders as
a fading husk" work when the downloaded arrays say the slot is free? The two
sections contradict each other.
**Fix:** define the lifecycle state machine precisely: e.g. `valid` = slot
occupied (spawn sets, reclaim clears), `ai_state` = behavior incl. DEAD;
reclaim = host-side, N ticks after death (N in the species table, synced), or
immediately with the husk driven purely by the death tick event render-side.
One paragraph; without it, twin and kernel authors will guess differently.

### M3. Movement law holes: tunneling, corner-cutting, map edges, overflow. (§2, §4)
- **No bound on `speed·Δt`.** The wall test is a "probe one step ahead". With
  speed a TOML-tunable table row, nothing stops `speed·Δt > 1` tile (a tuning
  typo, or a fast future species), and the single-probe law then steps through
  1-tile walls. Fix: validate `speed·Δt < 1 tile` at table load (hard error),
  or specify a DDA sub-stepped probe.
- **Diagonal corner-cutting:** probing only the destination tile lets a
  diagonal move pass between two orthogonally-adjacent solid tiles. Decide:
  acceptable for soft-bodied larvae (defensible! say so), or probe both
  orthogonal neighbors on a diagonal crossing.
- **"On wall contact" is not defined in integer math.** Does the larva move 0
  this tick and re-draw heading, or slide? Where exactly is the probe point —
  `tile(pos + dir·step)`? Is the new heading applied same tick (move again?)
  or next tick? Each answer changes the digest. Write the exact integer
  sequence; it IS the twin's law.
- **Map edge / hull breach:** a breached edge tile is walkable-ish air; a larva
  walking off-grid indexes out of bounds in the kernel. Specify: out-of-grid
  probes read as solid (clamp), and whether vacuum tiles are walkable (v1 has
  no breathing, so larvae happily stroll in vacuum — fine, but say it).
- **`dist_walked` int32 Q16.16 is monotone → overflows at 32768 tiles.** At
  ~1.5 tiles/s that is ~6 h of walking; signed overflow is UB in the C++ twin
  and wraps on device — a genuine CPU/GPU divergence bomb on long runs / RL
  training. Fix: make it uint32 with defined wraparound, or wrap at an exact
  multiple of the crawl wavelength (keeps the render phase continuous) — one
  line, but it must be in the spec. `heading` accumulation should likewise be
  wrapped to ±π at write time (the trig kit tolerates any input, but the
  *stored* synced value should have one canonical representation — otherwise
  two histories reaching "the same" heading digest differently).

### M4. Larvae are untouchable by every gameplay system, and the out-of-scope list hides two of the omissions. (§7)
The v1 exclusion list names stamping, vision, statuses, larva-larva, gas,
per-segment damage. It does NOT name:
- **Blast/wave coupling** — larvae take no overpressure damage and no wave
  push. A grenade in the nest room kills nothing ("larvae die to fire; that is
  half the fun" — but explosions, the game's signature verb, do nothing).
- **Being shot / targeted** — no faction/team column (§B3 lists team; the
  roster omits it), no presence in `vision.py`/engagements/attack_resolver.
  Marines cannot fight the swarm at all.

Both may be correct v1 cuts, but they must be ON the list, with a sketched
closure path — and the closure path has a structural trap the design should
name now: every CPU-side gameplay query (vision, shooting, zombie AI targeting)
needs *fresh* larva positions, which under the freshness split means a per-tick
D2H of the position arrays — reinstating exactly the host round-trip §B2
exists to kill. The likely lawful shape (a device-written per-tile
occupancy/crowding count field that rides the existing field D2H, which CPU
systems then read like any field) is one sentence; write it, or the "later,
explicit extension" list is a trap, not a plan.

### M5. The heat-damage "device analogue of the 9c coupling row" is not an analogue — it is a new, different law. (§7 fork 7b)
The 9c row (`exchange.COUPLING_TABLE[0]`, verified) reads the per-tick radiant
`heat` deposit, max-over-footprint, through a T_felt band. Fork 7b reads
absolute `temperature` at the own tile with a linear-over-threshold law. Field,
reduction, and response all differ. Consequence of calling it an analogue:
units and larvae disagree about what a hot room does (a marine in 200 °C air
with no radiant source takes zero damage; the larva dies) and nobody documented
that this is intentional. It may well BE the better law (contact/ambient vs
radiant) — but then it is a NEW coupling row and the coupling-table rule ("a
physics→unit coupling is one row") says it gets registered/documented as one,
with the divergence from the unit law stated. Also unspecified: within the
kernel, is damage sampled before or after this tick's move? (Determines whether
fleeing can outrun the burn — feel-adjacent, HUMAN-TEST territory.)

### M6. Species-table home: the design cites two mutually exclusive precedents and picks neither. (§1, §2)
"Schema in code / numbers in TOML, like materials/gases" → `config.toml` via
`CFG`, **F5 hot-reloadable**. "The table hash joins match-setup material like
`registry_content_hash()`" → entities-registry style, hash-pinned at match
start. These conflict: a hot-reloaded `T_hot` mid-match changes behavior under
a hash that was already committed. Pick one: (a) registry-style, hash-pinned,
no hot reload (determinism-clean, tuning-hostile — and #12 just demonstrated
how much fire tuning wants live dials), or (b) CFG-style hot reload in play,
with the hash captured per-run for recorder/replay validity (a reload
invalidates the replay from that tick, like any config change). Also decide the
file home (a `[cudaunits.larva]` config.toml block? `cudaunits.toml`?) — this
is a lasting name, i.e. exactly the kind of call Erik enters the loop for.

### M7. Digest claims oversell, and the food-field bump collides with fire-12. (§6, §8)
- §6 bullet 1: "**no existing golden moves**" — directly contradicted by the
  recommended fork 8a: a mutable food field joins `DIGEST_FIELDS` → spec v6 →
  **every golden regenerates** (v5 precedent, verified in `field_digest.py`).
  The headline claim is only true for the fork the design recommends against.
  Reword honestly: the *section* is free; the recommended *food field* is not.
- **fire-12 interaction:** branch `fire-12` carries tuning commits marked
  UNDER REVIEW with pending re-baselines. Two branches each planning golden
  regeneration is a merge-order hazard (whichever lands second regenerates
  again, and an interleaved regeneration can silently bless the other branch's
  unreviewed behavior). Sequence explicitly: cudaunit digest bump lands only
  after fire-12's re-baseline is merged, from a rebased branch.
- **Digest depends on config:** per-slot arrays digest directly at shape
  `(N, max_units)`; `max_units` is config (§2 fork 2b). Changing the default
  capacity — a pure capacity knob — would then move every cudaunit-bearing
  golden. Fix: digest only slots `[0, high_water)` per env (high_water = max
  slot ever spawned, itself synced), or pin the digested extent in the section
  header deliberately and document that capacity is digest-affecting.

### M8. Scale honesty at the 100k design target. (§2 fork 2b, §6, §9)
- **Recorder:** per-larva ring arrays are `(capacity, max_units)` per
  attribute (verified idiom). At 100k units × 5 arrays × int32 × a 600-tick
  ring ≈ **1.2 GB** of ring. Fine at the 2–4k default, quietly absurd at the
  design target. Specify a recorded-slot cap or high-water-bounded recording.
- **CPU twin:** O(max_units) per tick regardless of alive count (masked
  slots still iterate). ~100k trivial iterations/tick is likely fine (<1 ms)
  but it means a CPU-backend player pays for capacity, not population — the
  2–4k default is load-bearing; say so.
- **Render:** the fauna doc's requirement 3 (follow-the-path spine — segments
  trace the head's recorded trajectory) needs a per-larva head-path ring
  buffer on the render side. §9 mentions crawl phase (requirement 1) and drops
  requirement 3 without a word. At 2–4k it is fine; at 100k it is the render
  frame budget. Either scope v1 bodies to rigid/sinusoidal (no path-following)
  or name the ring buffer and its budget.

### M9. "Inherits the field contract for free" (fork 2a) oversells; and the §B3-pilot claim needs honest scoping. (§1, §2)
`GameMap.enable_residency` / `from_host` / `to_host` / the digest snapshot /
FieldEdit / the recorder field path are all built for `(h, w)` grids keyed by
field name. `(N, max_units)` actor arrays need real extension work in each
(residency alloc, targeted transfer lists, snapshot plumbing, serializer) —
GameMap residence makes that work *smaller*, not free. Relatedly, §1 claims
this chapter "is the pilot" of §B3, but it pilots only the storage layout: no
Python-Unit-as-view story, no team column, no velocity, no
EnvironmentProfile/coupling-row porting, no obs kernel. Add one sentence
listing what of B3 is explicitly NOT piloted, so the RL arc doesn't inherit a
false "done" checkbox.

---

## MINOR

### m1. draw_index semantics are unstated — and the stateless answer must be written down. (§3)
The design's scheme actually dissolves the classic divergent-draw-count problem
*if and only if* `draw_index` is local to `(unit, salt, tick)`, restarting at 0
each tick, numbered by static code position — then a unit that tumbles (2
draws) and one that doesn't (1 draw) are still bit-reproducible CPU-vs-GPU,
because no counter is shared or persisted. But the doc never says this. If an
implementer persists a per-unit running draw counter instead, it becomes
synced state missing from the roster (B2's failure class). One sentence: "no
draw counter is ever stored; draw_index is per-(unit, salt, tick), assigned by
code position; a variable number of draws per tick is safe." Also: "outputs map
to Q16.16 by integer shift" is only true for uniform [0,1); the turn-angle draw
needs a multiply by a checked-in 2π constant (the door-2 idiom) — specify it.

### m2. `next_unit_id` is synced state. (§2, §7)
The monotone id allocator must survive save/load and be digest-visible (an id
re-assignment after load forks every RNG stream). It appears nowhere. Add it to
the section (and to the future save contract).

### m3. Spawn bootstrap underspecified. (§7)
"`[[spawn]]`-style / zone roster idiom" — 2000 larvae as 2000 TOML rows is not
a plan; the zone-roster (count + seeded in-zone scatter) is. Which RNG draws
spawn positions — `sim.rng` (host, load-time — fine) or the Philox kit keyed by
the ids being assigned (bootstrap ordering must then be stated)? One paragraph.

### m4. The RNG kit amends engine/14 and the design doesn't say so. (§3)
Chapter 14's door 4 currently sanctions one raw-draw source (`sim.rng`'s
PCG64 bitstream). Philox is a second sanctioned source — that is an amendment
to 14 (and its case log / L1 lint story), not just a new header. The chapter's
Depends-on lists 14 but claims no amendment. Also decide the header home:
`fixed_point.h` is already ~1000+ lines of arithmetic kit; a sibling
`philox_q16.h` (FP_HD, included by both twins) keeps concerns separable —
either is fine, but "(+ device header include)" in §3 is confused: FP_HD in
`fixed_point.h` already compiles both sides; no separate device header is
needed unless the kit lives outside it.

### m5. The engine/02 amendment's boundary is circular. (§1)
"Object-shaped actors are CPU; array-shaped actors are cudaunits" defines the
category by the implementation choice it licenses — anything becomes
array-shaped the moment someone writes it as arrays (B3 eventually wants marine
reflex state device-side; is a marine then a cudaunit?). State the real
criterion: *homogeneous populations whose full per-agent state fits a fixed
schema and whose per-tick behavior is uniform and branch-light* are cudaunits;
agents with orders/plans/inventory/dialogue stay CPU. Also name the ownership
consequences: engine/02 says Simulation owns "the unit list"; who owns cudaunit
lifecycle (spawn/reset/serialize) if the arrays live on GameMap — presumably
Simulation drives, GameMap stores; write the sentence.

### m6. Within-tick order phrase is wrong. (§5)
"Inside the existing 9c damage-then-push ordering" — 9c4 is *after* the whole
9c/9c2/9c3 chain, not inside it. If the slot survives B3's resolution at all,
the documented within-tick exchange-order contract (the 9c comment block,
simulation.py:1463–1479) must be amended to name the larva step's place —
that comment is the canon for this ordering, not this chapter.

### m7. Sensor-accessor vocabulary question ducked. (canon brief §3/§9)
The brief explicitly asked the design to reuse the frozen `Channel` enum
vocabulary for the larva field gather or say why not. The design does neither.
The answer is probably "no — larvae gather raw fields in-kernel like combustion
does, the accessor is the *entity* seam"; fine, but write it, or someone will
later flag the kernel as a parallel sensor path.

### m8. Save/load of cudaunit arrays unaddressed. (§6)
Recorder and digest are covered; the (designed, unbuilt) save/load contract of
engine/02 (`np.save` the field set + entity/logic state) is not — cudaunit
arrays + `next_unit_id` + the species-table hash belong in it. One line in §6.

## NIT

- Species table lacks `food_threshold` (used by §4 EAT) and any husk/reclaim
  timer (M2); `eat_stop_prob` naming suggests memoryless — if M1 resolves that
  way, rename to `eat_continue_prob` or document the direction.
- COMA at a T_ctmin boundary will flap state each tick as temperature dithers
  ±1 count; cosmetic (render stillness flicker) but a 1-count hysteresis is
  one table row.
- §9 hover-readout larva count: `np.bincount` over the downloaded positions is
  fine even at 100k; no action, just confirming it was checked.
- "Larvae-free levels hash byte-identically" — the presence carrier's trigger
  (capacity allocated but zero spawned? first spawn mid-run?) should copy the
  entity carrier's exact semantics; name the carrier key.

---

## Verdict — what must change before implementation may start

1. **B1**: put the seed in the Philox key/counter and specify the exact word
   layout. Non-negotiable; everything downstream (goldens, replays, RL resets,
   the stats-sampler adoption) depends on it.
2. **B2**: close the missing-state audit — add `T_prev` (or re-rule the
   sensing spatial), settle M1's eat-stop mechanism, M2's valid/DEAD
   lifecycle, m2's `next_unit_id` — and re-issue the §2 roster as actually
   complete. The roster is section-versioned; getting it right pre-v1 is the
   cheap moment.
3. **B3**: settle the tick placement (recommendation: end of the resident
   physics chain inside slot 7, not a 9c4 host line), define the input
   snapshot (which tick's `solid`/`material` both twins read), and account the
   transfers. Until this is written, "bit-identical by construction" is a
   hope, not a property.
4. **M3**: write the movement law as the exact integer sequence (probe point,
   contact behavior, step bound validated at table load, edge clamp, overflow
   semantics for `dist_walked`/`heading`). This IS the twin spec; the kernel
   and twin cannot be written from prose.
5. **M5–M7**: re-frame heat damage as a new documented coupling row; pick the
   species-table regime (hot-reload vs hash-pinned) — flag for Erik, it
   constrains tuning workflow; fix the §6 "no golden moves" claim and add the
   fire-12 sequencing note.
6. **M4**: extend the out-of-scope list (blast coupling, shootability/faction)
   and add the one-sentence closure mechanism (per-tile occupancy field) so
   v1's isolation is a staged plan, not a dead end.

The skeleton is sound — SoA + mask + counter-based RNG + presence-gated section
+ 3-part gate is the right architecture, and the literature grounding is
genuinely load-bearing rather than decorative. But the draft's two proudest
claims ("bit-identical by construction", "the pilot's complete roster") are
both false as written, and both for the same reason: the design specifies the
arithmetic carefully and the *inputs and state* casually. Fix the state table,
the seed, and the tick placement, and the rest is implementable.
