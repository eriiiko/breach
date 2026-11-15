SOUND WITH FIXES

# Critique of design 4.4c (P6b and P7 re-planned at `ca2c85a9`), 2026-11-15

Reviewer: adversarial critic (Opus), read-only, on branch `29-p7-replan`.
Subject: `docs/residency_batching_design_2026-11-07.md` section 4.4c, against
the code and `docs/residency_p7_inventory_2026-11-15.md`.

The cut's shape holds: the stages respect most real dependencies and the
nine decisions mostly match the reference. No blocker. Nine majors, each
fixed by amending the text; most are dependencies or orderings the
inventory did not name.

## BLOCKER

None.

## MAJOR

**M1. Slot 5's BASH cannot be a deferred record.**
Claim (decision 1): "BITE and BASH (P7d) ... slot 5's pass 3 ... [is a] STAGE
of the same kernel over the same record". §4.3 row 5 says "one thread per
zombie; bites recorded, applied by the applier in id order".
Code: the bash is applied at once inside pass 3: `_bash` -> `chew_wall` ->
`destroy_wall` (`ai_zombie.py:118-137`, `combat.py:253-259`). Later zombies
in the same pass read the live map. `descend` tests `is_passable_block` on the
live `material` (`horde.py:476-498`, `gamemap.py:2353-2365`). `bash_tile` reads
the live `material` (`horde.py:453-473`). `block_feared` reads live
(`horde.py:291-297`). `_sees_prey` reads `has_los` over the live `solid`
(`ai_zombie.py:73-81`). The wander tests `is_passable_block` (`ai_zombie.py:403`).
The lazy mask and flood are built from the live map when a zombie first needs
them (`horde.py:377-399`, the closure at `ai_zombie.py:346-354`). So a door one
zombie breaks lets a higher-id zombie step through it on the same tick, or
changes the mask that zombie floods. The BITE alone is safe as a record:
`players` is fixed at the slot head (`ai_zombie.py:221`) and nothing later in
the pass reads prey hp or `alive`.
The horde-cache bullet's alternative ("flood at the stage's head ... is equal
to it") also fails whenever a bash destroys a tile before the first need.
Fix: P7d's pass 3 is a serial per-env walk. Thread 0 of the applier block
decides each zombie in id order and applies its bash at once, or else the pass
runs wide and is re-run serially from the first zombie whose bash destroyed a
tile. BITE may stay a recorded kind. Delete "flood at the stage's head" as a
candidate, or limit it to ticks with no destroying bash.

**M2. The noise list is not a per-tick list.**
Claim (decision 4): "`noise_log`, `blast_log` and the credit claims ... cleared
with the tick-end clears".
Code: `noise_log` is cleared by nothing at tick end. The only place it empties
is the drain at the slot-5 head, `noise.take` (`simulation.py:692`,
`noise.py:54-63`). `_clear_tick_buffers` does not touch it
(`simulation.py:2458-2515`). A slot-5 bash (`ai_zombie.py:137`) and every 9e
detonation (`payloads.py:473-474`) are heard at the NEXT tick's slot 5
(`.claude/rules/horde.md` "Noise"). If the device clears the list at tick end,
those noises are lost.
Fix: the noise list is drained at the slot-5 stage head and persists across the
tick boundary. It is cleared only at env reset; add it to §4.6's list of
queues cleared at reset. The claims stay with the tick-end clears. The blast
log may stay with them too, which is equivalent: 9e(d'') or its `elif` empties
it (`simulation.py:2391-2394`, `heat_source_system.py:222-223`), and nothing
writes it after 9e.

**M3. Slot 9d (ignition) is ported by no patch.**
Claim (P7e): "every slot's kernel in the reference's order for N envs with no
host slot between them".
Code: `apply_temperature_ignition` runs every tick on the continuous path
(`simulation.py:2314-2324`). It has no device form: no CUDA source names it,
there is no twin-manifest row, and the profiler label `9d ignition` is
unported (`tools/tick_stage_profile.py:93`; inventory "9b and 9d have no
kernel"). P6.3 as built ported 2b, 9c and 9c3 (§4.4b). Nothing in 4.4c names
9d.
Fix: give 9d to a patch. It is a per-cell law over `temperature`, the O2
fraction and `fire`, with `ignition_seed` and `o2_frac_ext` as table rows. It
fits P6b.3, or its own small row before P7e.

**M4. Stage 2's pair (P7a.2 and P7b) collides in three places.**
(a) Slot 2 is ONE callable, `_update_projectiles`, with grenades and rounds in
one body (`simulation.py:2520-2569`). It has one profiler and harness label,
"2 projectiles" (inventory C14). P7a.2 ports the rounds and P7b ports the
grenades, so both must split or partly wrap the same function, label and
twin row at once.
(b) Decision 3's table machinery (schema rows into a generated header, `pack`
/ `unpack`, append, stable compaction, a counted capacity refusal) is listed
under P6b.1 in stage 3. Its first users are P7a.2 (the rounds) and P7b (the
grenades and charges), side by side in stage 2. That invites two copies of
the machinery, against the iron rule.
(c) P7a.2's march records the payload round's detonation
(`combat.py:611-625`), but the PAYLOAD kind is P7b's.
Fix: P7b owns the slot-2 split (`_update_grenades` / `_update_rounds`, two
labels), the generic table module and the PAYLOAD record, and P7a.2 depends
on P7b. Alternatively, move the table module and the split into P7a.1.
P7a.2 refuses payload rounds by name until P7c.

**M5. P6b.1 ports 9e(c), which reads the bus that P6b.2 builds later.**
Claim: P6b.1 builds "9e(c) and the door sweep", needs only P7a.1. P6b.2 (stage
4) builds "the signal bus".
Code: `_resolve_door_inputs` aggregates `pub` from the bus
(`simulation.py:1926-1946`) and runs only inside `if self._signal_bus is not
None` (`simulation.py:2349-2355`).
Fix: move 9e(c) to P6b.2. P6b.1 keeps the door table, the door sweep, seal and
unseal, the gas-N primitives and 1b's use write.

**M6. SPRAY_CELLS is variable-length and decided in phase A.**
Claim (decision 1): SPRAY_CELLS is a kind of the one record, built on
`UnitRecordList` (a fixed record width and row cap).
Code: the cone's cells are computed in phase A, occluded by the walls standing
at the slot's start. Then the whole list is handed to the WORLD effect
(`combat.py:1713-1734`; `_enqueue_spray_cells` walks it,
`combat.py:1737-1760`). One sprayer has up to about range² x
(half-angle / pi) tiles. Recomputing the cone during the walk would read walls
that an earlier CHEW in the same resolve destroyed.
Fix: the SPRAY_CELLS record carries a span into a per-env cell buffer that
phase A fills, with its own capacity, counted and refused by name like
`door_spans`. State that the walk never recomputes the cone.

**M7. Decision 7's `seq` needs three things it does not state.**
Code: `EditQueue._seq` resets at the flush (`field_edit.py:818-824`), so one
`seq` window runs from the previous tick's 9e to this tick's 6b. `seq` is
also the noise key, packed as `int(seq)` (`field_edit.py:978`; §4.8 "each
edit row carries its `seq` as the noise key"). So for noisy edits (blast
smoke, `payloads.py:509-520`, `physics.py:197-202`) the ABSOLUTE value
matters, not only the order.
(a) The per-env counter resets at the 6b flush and survives the tick boundary.
It is never reset at the tick-end clears or the tick head. It is cleared at
env reset.
(b) "Rows equal `pack_for_device` of the edits the host producer enqueued":
`pack_for_device` packs the host-sorted window with absolute keys and
edit-row indices (`field_edit.py:927-1007`). A device producer's rows match
only given a seq base (the edits other producers enqueued earlier in the
window) and rebased edit-row indices. While host and device producers
interleave inside one walk (4c with a host payload round beside a device
spray, before P7c), the base cannot be known before the walk. The gate must
refuse such ticks by name, or compare with keys rebased (exact only for
noise-free edits; sprays carry `noise = 0`, `combat.py:1836-1838`).
(c) Before step (c), a CHAINED run with a device producer needs the device
rows merged into the 6b serve in `(field, source, region, seq)` order. Name
how, or say chained gates wait for (c).

**M8. P7e needs planes that stay non-resident until P8b.**
Claim: decision 6 says "`sight_screen` is a slot input until P8b makes it
resident". P7e is "the input of G2 and of P8b".
Code: the device topology seam writes `sight_screen`, `heat_atten` and
`light_atten` (design 4.4b P6.4b "NOT RESIDENT TODAY ... P8b must hold them";
`applier.non_resident_planes()`). With no host step between slots, a
destroy_wall at 2, 4c, 5 or 9 must reach the `has_los` reads in 4b, 5, 9e and
the flash, and slot 7's sweep, from device memory. The same holds for
`div_source`: today it is built on the host by `take_source_plane`, which
drains the multi-tick `gmap.expansions`, and is uploaded every tick
(`physics_runner.py:995-998`, inventory C10). No patch names a device form of
`take_source_plane`.
Fix: P7e (or a named step before it) makes these planes and the expansions
table resident. Name `take_source_plane`'s device form in P7c or P7e, or put
the residency half of P8b ahead of P7e.

**M9. P7c is too big to gate in one piece.**
P7c holds every sub-step of `execute_payload` (`payloads.py:458-558`): the
disc and its wave TILE edits, blast damage, the flash over sight, two reach
walks, the air read, the `div_source` writer and expansions, DISC and TILES
deposits with linear weights, bulk gas through the gas-N primitives,
ignition, deposit_heat, the noise, the blast log, the claims. It also carries
the applier's first fan-out and ExplosionEvent. P6b.3 and P7e wait on it.
The clean seam is `generic = strength > 0` (`payloads.py:488`):
- P7c.1, the classic path: `apply_explosion`, blast damage, flash, `emit_gas`,
  `ignite_ring`, `deposit_heat` (DISC regions), noise, blast log, claims.
  Gated on flash and smoke grenades, the studio rows and `frag_heat`.
- P7c.2, the strength path: the reach walk, the air read, `start_source`,
  `start_expansion`, `deposit_over_tiles` with linear weights, the fan-out,
  and bulk gas. Gated on `frag_standard` (a strength row, `config.toml`
  `[payloads.frag_standard] strength = 1.0`), `breach_focus` and C4.

## MINOR

- m1. Decision 2 contradicts itself: "the mode is the stage's, never the
  caller's guess" against "a payload inside a resolve keeps its immediate
  form". The mode is per (stage, kind). A HIT in a resolve stage uses the
  landing; a PAYLOAD's blast packets are immediate in every stage
  (`exchange.py:540-545`; `damage.py:445-472`). Say so.
- m2. FACE has no continuous-path producer. `fire_burst` snaps only when
  `aim_q is None` (`combat.py:1348-1355`, the WEGO paths). The melee verb
  passes `snap_facing=False` (`combat.py:2126`). `start_spray_burst` is WEGO
  (`combat.py:1781`). Decision 9's census will find FACE unreachable: name it,
  or drop the kind.
- m3. Decision 6: the spray cone calls `has_los(..., physical=True)`, which
  tests `solid` only (`combat.py:1679`, `gamemap.py:2459-2481`). The device
  function needs both forms.
- m4. Decision 8 says the ring "keeps only synced events" and also that
  "ExplosionEvent joins with P7c". ExplosionEvent is not synced
  (`field_ab_harness.py:207-219`; `payloads.py:549` "render-only,
  undigested"), no `src/` reader outside the producers consumes it, and its
  fields (pos, radius, strength, flash floats, payload name) do not fit the
  ring record. Drop it and amend the Event ring rule, or name its consumer.
- m5. `vent_system._aperture_blocked` (`vent_system.py:240`) is needed by both
  P7c (`bulk_gas_tiles`, `payloads.py:248-252`) and P6b.2 (vents,
  `vent_system.py:387`, `425`), side by side in stage 4. Assign it to P6b.1
  with the gas-N primitives.
- m6. P6b.2's need on P7a.3, and its place in decision 7(b), are spurious.
  Pumps and vents enqueue no FieldEdit; the only enqueue sites are
  `combat.py:1745`, `1754`, `payloads.py`, `physics.py` and
  `simulation.py:2012`. Remove both. P6b.2 could then move to stage 3.
- m7. P7d does not name two slot-5 inputs:
  - the match's `drive_zombies` request set (`simulation.py:693-701`): a host
    latch that P7e's "the host writes only the action latches" must carry;
  - the early returns before `begin_tick`: an empty request set
    (`simulation.py:697-698`) or no prey (`ai_zombie.py:221-223`). The cache
    is not dropped on those ticks (inventory Part F).
- m8. `start_spray_burst_directional` and its noise (slot 4's spray branch,
  `combat.py:1204-1211`) are owned by neither P7a.2 nor P7a.3. Name it under
  P7a.2.
- m9. The refusal list should also refuse by name, at the device door:
  - 9b, when `burst_enabled` is true (`simulation.py:2237`; off today,
    `config.toml:133`);
  - a unit holding WEGO `orders`, since slot 4 reads
    `get_fire_order_in_phase` (`combat.py:1034`).
  The ruleset check alone does not prove either.
- m10. Field sensors sample through `EntityFieldAccessor`
  (`sensor_system.py:88`), the old P6b row's "#25 gather". P6b.2 needs a device
  `sample` for them, and the amendment does not name one.
- m11. P7a.2's tables leave out the per-material cover exposure
  (`attack_resolver.cover_exposure_at`, `combat.py:503`) and the zombie bullet
  multiplier (`_bullet_mult_q`, `combat.py:536`). Name them, beside the weapon
  and ammo rows.
- m12. The "in no digest" bullet leaves out two pieces of cross-tick hidden
  state: the noise log across the tick boundary and the horde cache
  (`horde.py:346-375`).
- m13. The float-conversion bullet leaves out the heat source's FLOAT
  per-tick deposit, which has no `amount_q` (`heat_source_system.py:246-247`;
  inventory Part F).
- m14. Side-by-side pairs regenerate the same generated headers
  (`unit_tables_layout.h` with its T count, `applier_layout.h` with its kinds).
  State the P6.4a/b practice: regenerate after the merge, and rebuild.
- m15. Decision 1 says slot 9e and pass 3 are stages "over the same record".
  They walk tables, not records. Reword them as stages of the same kernel
  that walk the entity tables (9e) and the zombie rows (5).

## Answers in short

1. **Dependencies.**
   - Missing: 9d (M3); the bus for 9e(c) (M5); the table machinery before
     P6b.1 (M4b); the PAYLOAD kind for P7a.2 (M4c); resident planes and
     `take_source_plane` for P7e (M8); a device accessor `sample` for P6b.2
     (m10).
   - Spurious: P6b.2 needing P7a.3 (m6).
2. **Side by side.**
   - Stage 1: no conflict beyond regenerating headers (m14). P7a.1 needs
     nothing from P7.0: `melee_adjacent` is Chebyshev (`combat.py:1931-1955`)
     and the verb draws nothing.
   - Stage 2: collides (M4).
   - Stage 3: only once 9e(c) leaves P6b.1 (M5).
   - Stage 4: `_aperture_blocked` (m5).
   - Stage 5: no conflict, given M2.
3. **The nine decisions.**
   - (1) holds for slot 2, 4c and 9e. It fails for 5's BASH (M1) and needs
     spans for SPRAY_CELLS (M6).
   - (2) is faithful (see below), apart from the wording (m1).
   - (3) holds.
   - (4) is wrong for the noise log (M2).
   - (5) is exact as the code stands.
   - (7) is underspecified (M7).
4. **Order claims.** All six check, except the horde cache's head-flood
   alternative (M1).
5. **Scope.**
   - Left out: 9d (M3); the zombie request latch (m7); the directional spray
     start (m8).
   - The items named as WEGO-only are WEGO-only.
6. **Size.**
   - Split P7c at `strength > 0` (M9).
   - Split P6b.2 into the signals part (bus, sensors, logic, pumps, which
     exist only with a bus, `simulation.py:352-353`) and the vents part
     (bus-independent, `simulation.py:2364-2365`).
   - No pair merges with no loss. The closest is the timed charges' countdown
     joining P7b's `charges.tick`.

## Checked and found correct

- Grenade flight reads no map state (`combat.py:195-216`: integer flight
  only; the detonation reads the target, `combat.py:182-186`).
- Slot 2's order is every grenade, detonations immediate in list order, then
  the rounds' phase A against the post-grenade state, then the landing
  resolve (`simulation.py:2525-2569`).
- The march's first hit is the first unit in list order whose footprint holds
  the tile, with the shooter excluded by identity (`combat.py:480-486`). Row r
  is the r-th unit of `sim.units` (`.claude/rules/device-units.md`). A
  converted shooter keeps its id.
- The record order is shooters by id x rounds, then melee by id, then
  sprayers by id (`simulation.py:2155-2180`; `combat.py:1010`, `2115`,
  `1851`). Within a round, the recoil PUSH precedes the march's records
  (`combat.py:1377-1383`), and the push follows its hit (`combat.py:566-579`).
- The exposure landing as per-target scratch is faithful to
  `ExposureLanding`:
  - the adds commute under the cap;
  - raises skip a unit dead at landing, then the dead are zeroed
    (`exposure.py:286-314`);
  - `on_hit` runs after the kill only for the was-alive (`damage.py:445-472`);
  - blast packets bypass the landing (`exchange.py:540-545` via the
    inventory).
- Stable compaction keeps every reference walk's order:
  - bullets: survivors, then this tick's appends (`simulation.py:2559-2569`,
    `combat.py:1384-1385`);
  - projectiles (`ruleset.py:289-291`);
  - planted charges (`charges.py:63-75`, `133-137`).
- The `.next` sites:
  - `SHOT_CONE` (`combat.py:1366`) and `BEAM_CONE` (`combat.py:1480`), once
    per shooter per tick on the continuous path;
  - `MELEE_CRIT` (`combat.py:2038`), on the order path only, since the
    loadout door refuses crit melee rows;
  - the count resets per tick (`draws.py:146-156`).
  A row-local counter is exact.
- The disc loop reads `solid` after earlier `destroy_wall` calls in the same
  disc (`physics.py:123-158`), so it must stay serial.
- The heat-source chain is bounded by the initial discs plus the source count
  (`heat_source_system.py:226-239`).
- The bite can be recorded and applied in id order (see M1).
- The claims: first claim wins as a first-match scan (`damage.py` `setdefault`;
  `credit_at`).
- The scope list's WEGO items are dead on the continuous path:
  - targeted and auto-fire, which need orders (`combat.py:1034-1054`, `1099`,
    `1267`);
  - `start_spray_burst` (`combat.py:1763`);
  - physical cover (`march_cover` None);
  - the order path's melee crit.
- Payload rows ship strength rows (`frag_standard`, `breach_focus`), so the
  generic path is on the shipped critical path (M9).

## Could not check

- The inventory's scene and level census (Part E) and the corpus counts:
  taken as given.
- `exchange.apply_blast_damage`, sensors, logic nodes and the vent ledgers:
  read through the inventory only.
- The CUDA internals (`cuda_applier.cu`, `cuda_edit_queue.cu`,
  `cuda_unit_lib.cuh`) beyond the cited signatures. Whether the existing
  unit-impulse add is a library function or private to the verbs TU (it
  matters for P7a.1's PUSH).
- Performance of a serial pass 3 (M1) and of `credit_at`'s scan over a longer
  claims list. Nothing was built or run.
