# Swarm units P3 — implementation doc: the larva behaviour kernel + CPU twin

> **v2 — 2026-09-24, post-critique. Design complete; build PAUSED until
> fire-12 merges (Erik 2026-09-24); refresh file:line refs, the §9 merge
> table and the Kelvin scale (k 3→1) against merged code before building.**
> Arc #63, branch `63-swarm-units` (patch branch `63-p3-larva-kernel` per
> the plan, cut when the build starts). v1 was `4528571`; the two critique
> lenses are `docs/swarm_P3_critique_2026-09-24.md` (Lens 1 `799b9ae`,
> Lens 2 `ea4c58f`), whose `## Synthesis → v2` section maps every finding to
> a resolution R1–R24 and to the § that applies it. Written against HEAD
> `ea4c58f` (P1 + P2 merged). Executes
> `docs/architecture/engine/17_swarm_units.md` (BLESSED) §4 (state machine;
> §4.1 re-specified here as the exact integer sequence), §5 (placement,
> input snapshot), §6 (gate shape), §7 (death event, fork 7b), §9b (shared
> vs per-species files), under plan row P3
> (`docs/swarm_units_patch_plan_2026-09-10.md`, re-planned 2026-09-24). Takes
> the P2 hand-off (`docs/swarm_P2_impl.md` §9) as its contract.
>
> **Model tiers:** this doc = Opus; the build = Sonnet, oracle-gated. The feel
> test is P4, not P3.
>
> **P3 in one sentence:** one FP_HD per-slot larva law (head-sweep thermotaxis
> on the own-tile temperature, threshold heat kill, chill-coma hysteresis,
> wall-probe movement, Philox draws) run by three loops (CPU twin, per-call
> CUDA, resident CUDA) that a new `PhysicsEngine::step_swarm` orchestrates as
> the last stage of slot 7 on every backend, fed by Kelvin-authored species
> rows and host-derived launch words, emitting (species, env, slot)-ordered
> death events — and a 3-part CUDA gate that proves the loops byte-identical.

## Erik's rulings for P3 (locked 2026-09-24 — recorded, not re-opened)

| # | Ruling |
|---|---|
| 1 | **Temperature only.** No food (P6). The head-sweep sampler is field-generic so P6 can feed it food without reopening it. `AI_EAT` stays unused. |
| 2 | **Sensing = spatial head sweep (Luo 2010).** Every tick a RUN larva reads temperature at its own tile ("here") and at two sweep points one tile ahead at ±`sweep_angle` of its heading. Comfort = \|T − T_prefer\| (smaller is better): larvae move toward T_prefer from both sides. Per-tick tumble probability = `tumble_rate_base`, raised to `tumble_rate_alarm` when BOTH sweep points are worse than here. On a tumble the new heading turns toward the better side with probability `turn_bias` (else the other side), by a random magnitude in [`turn_min`, `turn_max`]; equal sides → 50/50. Straight runs between tumbles. **ACCEPTED GAP:** no weathervaning (continuous steering). **ACCEPTED GAP:** the law responds to the sign of the comparison only, not its magnitude. |
| 3 | **T_prefer = 30 °C = 303.15 K.** |
| 4 | **Heat kill: T > 50 °C = 323.15 K → DEAD, instantly** (fork 7b: one absolute threshold on own-tile temperature, no drain, `hp` unused by the law). |
| 5 | **Coma: enter at T < 10 °C = 283.15 K; exit at T > 12 °C = 285.15 K.** COMA = no movement, no sensing, no tumble draws. |
| 6 | **No cold death in v1** — no `T_lethal` row. |
| 7 | Temperature rows authored in **Kelvin** in `[swarm.larva]`, converted through `temperature_scale.from_kelvin` and quantized ONCE at table load (door 2); immune to fire-12's `k_temp_to_kelvin` 3 → 1 by construction. |
| 8 | **The death event is built in P3:** emitted from the PRE-ZERO values (unit_id, pos_x, pos_y, heading, species), then the P2 empty-slot form is written; event order deterministic (slot order, no atomic append), CPU==GPU gated. |
| 9 | Speed and tumble/turn dials are placeholders for the P4 play-test (defaults below, one-line rationale each). `SPECIES_RULES` is created with its first rule `speed·Δt < 1 tile` (hard error at load and on reload). |
| 10 | Tiers: doc Opus, build Sonnet (oracle-gated); feel test at P4. |
| 11 | **(2026-09-24, E1) The larva reads the RAW own-tile `temperature`** — `felt_T = temperature[tile(x, y)]`, the same reader for "here", both sweep points, coma and kill. Premise: assume the temperature field is good (fire-12 is changing the temperature physics; the Appendix-A2 pattern is recorded, not a design driver). |
| 12 | **(2026-09-24) Swarm larvae are much smaller than one tile (0.333 m); their real size is still to be given.** Nothing in P3 is derived from a body size (the 0.60 × 0.15 m figure was the fauna-spike "garden larva", not the swarm unit). |
| 13 | **(2026-09-24) Build paused until fire-12 merges into main and main into the arc branch;** a refresh pass (§13 step 0) precedes the build. |

---

## Deviations from the plan / chapter / P2 hand-off (resolution → evidence)

| # | Plan/chapter/P2 said | P3 does | Why (evidence) |
|---|---|---|---|
| V1 | "`SwarmSolver` in PhysicsEngine" (plan row P3); "last stage of PhysicsEngine's orchestration" (ch. §5) | **v1's runner-only `_run_swarm` is WITHDRAWN (R1).** The orchestration is a `PhysicsEngine` METHOD, `step_swarm`, whose body is `swarm::step_all` in the shared swarm TU; there is no `SwarmSolver` member class (the step has no tunables to own; its one piece of state is a record scratch member). PhysicsRunner calls it in ONE line after `step_tail` on both tick paths. | CLAUDE.md "PhysicsRunner / PhysicsEngine" ("new C++ orchestration lands in PhysicsEngine, not Python glue") and Erik's §A rule 2 ("no new host-side tick logic"). A method is the `step_tail` / `run_substeps_resident` idiom (the latter forwards to `breach_cuda::eos_step_resident`, `physics_engine.cpp:607`). Forwarding keeps the `physics_engine.{h,cpp}` hunks at ~25 lines in spans fire-12 does not edit (§9). Chapter §5 is then true as written. |
| V2 | "species-table H2D keyed on hash" (P2 §9) | Species params passed **by value** as a kernel argument (an 8-word POD), re-read from `gmap.swarm_species` by the Python gatherer at every launch. | Meets P2's purpose (no flag; a reload reaches the device from the next tick) with no device copy to go stale and no cache key to get wrong. |
| V3 | "the P1 salt enum" (task) | **Created in P3**: `src/simulation/philox_streams.py` + `cpp/src/philox_streams.h`. | No salt enum exists (grep `salt` over `src/`, `cpp/`, `tests/`: only philox32's own API names; confirmed by Lens 1). Chapter 14's amendment requires ONE breach-wide enum. |
| V4 | Chapter §4.1 step 3: "coordinates clamped to the grid (out-of-grid reads as solid)" | Bounds-check before any read; out-of-grid = blocked; no clamp for the solid decision. | A clamp would make an out-of-grid target read the edge tile's solidity, contradicting "out-of-grid reads as solid" whenever the edge tile is open. Routed to ch.17 §4.1 step 3 (§9b). |
| V5 | Philox counter `tick` = `sim.tick` (implied, ch.17 §3 + the ch.14 door-4 amendment) | Counter tick = **`Simulation.total_tick`** (a new property, `round_index * ticks_per_round + round_tick`). | Under the default `TwoPhaseWEGO`, `sim.tick` rewinds to 0 every round (`simulation.py:1846`, `ruleset.py:115-121`); keying Philox on it would replay every larva's draw sequence every round. Routed to ch.17 §3, the ch.14 amendment and the CLAUDE.md "Deterministic RNG" row (§9b). |
| V6 | P2 §9: "the kernel/twin loop runs over `[0, high_water)`" with the device keeping its own `high_water` copy | The loop bound is the **host** `high_water[e]`, carried per env in the launch block; the device copies of `swarm_<sp>_high_water` / `swarm_next_unit_id` are never read by any kernel. | One source for the mask, the launch extent and the compaction, so a missed dirty flag can never turn stale records into phantom deaths (L1-M1, R2). |
| V7 | P2 §9 R9: "P3's upload consumes `swarm_host_dirty[sp]` … and clears it" | Kept, and extended: the engine also SETS the flag after every host-path launch (CPU twin or per-call CUDA wrote the host store, so any device copy is stale). | Makes a CPU tick followed by a resident tick (a `set_residency` toggle) upload before it launches. |

---

## 0. Scope

**In:**

- the shared swarm kit (species-agnostic, ch.17 §9b "written once"):
  `cpp/src/swarm.h` (roster/enum/heading twins, tile lookup, the
  field-generic `TileReader` + head-sweep sampler, side preference, wall
  probe, turn draw, the store/launch types, the host loop template, the
  species-law registry type), `cpp/src/swarm.cpp` (the registry, the
  engine body `step_all`, launch-block composition, death compaction, the
  CPU counter), `cpp/src/swarm_launch.cuh` (the kernel + per-call + resident
  launch templates), `cpp/src/swarm.cu` (backend flag, counters, device
  scratch, transfers) (§3, §4);
- the larva law (per species): `cpp/src/swarm_larva.h` (the FP_HD per-slot
  step, its params POD, its draw table), `swarm_larva.cpp` (CPU entry),
  `swarm_larva.cu` (per-call + resident entries); `src/simulation/swarm_larva.py`
  (param/derived word order, the host derive of the tumble probabilities,
  the larva rules) (§3, §5);
- `PhysicsEngine::step_swarm` + its one-line calls in `PhysicsRunner` (§4);
- the breach-wide Philox stream-salt enum + match key + `Simulation.total_tick`
  (§6);
- species rows, `SPECIES_UNITS` (Kelvin), per-species `SPECIES_RULES`,
  `SPECIES_KERNELS`, the ingress gatherer `swarm.engine_args` (§4.3, §5);
- the death event (§7);
- the larva coupling rows in the ONE `COUPLING_TABLE` (additive `CouplingRow`
  extension) + the 9c comment (§8);
- bindings, stub, CMake, float ratchet incl. the sim-law headers (§9, §10);
- the verification plan incl. the 3-part CUDA gate (§11) and Erik's first-look
  plot (§12).

**Out** (each a later patch or an ACCEPTED GAP):

- food, EAT, eat draws (P6); `swarm_brood` + level spawning + the dot view
  (P4); bodies/husks/decals (P5);
- **ACCEPTED GAP:** no weathervaning; sign-only response (ruling 2);
- **ACCEPTED GAP:** turn *size* is not regulated by the comparison (Luo 2010
  finds larvae regulate turn size as well as direction; v1 regulates direction
  (`turn_bias`) and rate (`tumble_rate_alarm`) only);
- **ACCEPTED GAP:** larvae read gas temperature regardless of gas density — a
  vented room's frozen residual gas puts them in coma (§1 F1); a physiological
  body temperature with thermal inertia is the natural later fix (a ROSTER
  column → `SWARM_SECT_V2`);
- **ACCEPTED GAP:** a door that closes on a larva (`seal_tiles` checks no
  swarm occupancy) leaves it in a solid tile; it keeps its state, every move
  whose target stays in that tile is blocked, so it is effectively frozen until
  the door reopens. P4 decides refuse / eject / crush;
- **ACCEPTED GAP:** larvae walk onto `is_vacuum` SPACE tiles (chapter §4.1 "v1
  larvae don't breathe") and onto thermal-solid furniture (not `solid`), where
  they read the crate's object temperature;
- **ACCEPTED GAP:** no larva–larva interaction (fork 7a); no blast/shot
  damage (chapter §7 list);
- **ACCEPTED GAP (R5):** `match_key` refuses a spawned `SeedSequence`
  (non-empty `spawn_key`); the RL arc folds `spawn_key` into the key when it
  needs sibling seeds (§6);
- the `hp` column is untouched except by death (fork 7b);
- any body-size-derived number (ruling 12): the real larva size is Erik's to
  give before P4's speed tuning and P5's render.

---

## 1. Findings this design rests on (each with evidence)

**F1 — What `temperature` holds on vacuum and solid tiles (Erik asked).**
Both backends are gated bit-identical, so one answer covers both.

- **`is_vacuum` tiles (SPACE outside the hull, and breach tiles joined to
  space):** `TemperatureSolver` Pass 0 wipes every non-thermal-solid vacuum
  cell to `temperature = 0` each tick (`temperature_solver.cpp:141-171`). 0 is
  the ambient reference: `to_kelvin(0) = kelvin_ambient = 293 K` (19.85 °C) on
  HEAD and on fire-12. A larva on a SPACE tile reads 20 °C: no coma, no death.
- **Interior gas that a breach is emptying (not flagged `is_vacuum`):** the
  expanding gas cools adiabatically and the recovery floors it at `T_MIN`
  (`eos_solver.cpp:1661-1667`; `config.toml:617` `T_MIN = -289.0` on HEAD,
  `-292.0` ≙ 1 K on fire-12). Probe (Appendix A1, 16×16 room, one breach tile):
  the room interior falls below 10 °C within **one tick** (173 of 196 cells at
  t = 0) and sits at the floor (mean T_game ≈ −270) from t ≈ 25 on; the
  residual gas **never fully leaves** (mean N ≈ 1.6 % of ambient after 400
  ticks, because at the floor p ≈ 0 drives no outflow). Cells whose bulk N
  drops below 1 raw count are wiped to T = 0 by the recovery
  (`eos_solver.cpp:1654-1659`) and read 20 °C; the probe shows such cells
  toggling (8 → 2 → 0 → 15) as the residue sloshes.
- **So: yes — venting a room puts its larvae into coma within about a second,
  and keeps them there** while the thin frozen residue remains; larvae on
  hard-vacuum cells (and on SPACE tiles) read 20 °C and wake, and a larva on a
  toggling cell flaps COMA↔RUN (the 2 K hysteresis cannot help against a
  293 K ↔ floor jump). No death (ruling 6). Re-probe after fire-12 (its T_MIN
  and frame change) at the refresh pass.
- **Solid tiles:** a wall's `temperature` is its own thermal truth, derived by
  `TemperatureSolver` alone (CLAUDE.md "Temperature solver" row). **Larvae
  never sense a solid tile** in this design (§2 D4): a sweep point on a solid
  tile is "blocked", not a reading. The own tile is solid only in the
  door-closes-on-a-larva gap (§0), where "here" is the door's temperature.

**F2 — Recorded, not a design input (ruling 11).** A one-shot heat seed in a
sealed room leaves a grid-scale pattern in the gas `temperature` field on HEAD
(Appendix A2). Erik's premise for P3 is that the field is good; the pattern is
re-measured after fire-12 merges (§13 step 0) and appears once in the
"recorded, not P3" list of the critique synthesis. P3's own consumer of this
fact is the first-look tool's second assertion (§12), which will report it if
it persists.

**F3 — `sim.tick` is round-local under `TwoPhaseWEGO`** (V5):
`_end_round` sets `self.tick = 0` and `turn_number += 1`
(`simulation.py:1846-1849`); the base ruleset defines
`round_index = turn_number - 1`, `round_tick = sim.tick`
(`ruleset.py:111-121`); `OnePhaseWEGO` is free-running (`round_index = tick
// tpr`, `round_tick = tick % tpr`, `ruleset.py:369-378`), and
ContinuousRealtime keeps `turn_number == 1`. So `round_index *
ticks_per_round + round_tick` is monotone and gap-free under all three
(Lens 1 checked). `level_lights.monotonic_total_tick` (`level_lights.py:55-65`)
is a second, ruleset-blind definition that double-counts under OnePhase —
recorded for its own issue; P3 does not touch it (R6).

**F4 — No salt enum exists** (V3).

**F5 — On the resident path the tail runs on the host mirror.** Step 6's D2H
(`physics_runner.py:1299-1300`) happens BEFORE the combustion and tail
brackets (`:1303-1336`), which write `temperature` on the mirror only. So a
resident swarm launch must upload `temperature` first (§4.4). RL-arc B2
("Port combustion + tail to resident") deletes that upload; §9b routes a
bullet to `rl_env_arc_proposal_2026-08-27.md` §B2.

**F6 — Float ratchet.** `test_no_float_in_sim_tu.py` counts LINES containing
the words `float`/`double` (comments included) per file listed in `SIM_TUS`
and fails if a count rises (`:378-396`); it scans no header anywhere (Lens 1
L1-m1, Lens 2 L2-M5). The whole larva law is FP_HD-inline in headers, so P3
adds a `SIM_HEADERS` list held at 0/0/0 (R11): **no line of `swarm.h`,
`swarm_larva.h`, `philox_streams.h`, `swarm_launch.cuh`, `swarm.cpp` or
`swarm_larva.cpp` may contain the words "float" or "double", not even in a
comment**; the new `physics_engine.{h,cpp}` hunks add no such line either.

**F7 — P2's test fixtures pin the species row.** `_valid_row` /
`_swarm_cfg_namespace` build `{"max_units", "hp_max"}` literally
(`test_swarm_species.py:43-46`, `test_swarm_digest.py:47-50`,
`test_swarm_recorder.py:43-46`, `test_swarm_store.py:46-50`). With P3's required
rows every valid-path P2 test would fail on "missing key". Fixed in P3 by the
testkit (§11); T22's message-matching is tightened at the same time (R10).

**F8 — The coupling-table tests pin a set designed to grow**
(`test_exchange_reductions.py:266-267` pins the exact field list, `:275, :291`
unpack exactly five rows). `test_wave_push.py:426-430` pins index 2 only and
survives an append (v1 misstated it). The pinning is a finding; P3 rewrites
those three tests as properties (R7) — a listed pre-existing-test change —
and appends the larva rows to the ONE `COUPLING_TABLE` (§8).

**F9 — `tools/run_on_cuda.py::_BACKEND_SETTERS` omits
`set_combustion_backend`** (`:92-108`), so `--cuda` never runs combustion on
the GPU. Finding in another system; P3 only appends its own setter.

**F10 — Interpreter.** On the desktop the cp311 `.pyd` runs under
`C:/Users/steen/anaconda3/python.exe` (3.11.7, NumPy 2.4.6); the conda env
`data` here is 3.12 and cannot import it. CLAUDE.md's "always the conda env
`data`" is Lenovo-true, desktop-false. Finding; the build uses the base
interpreter on this desktop.

**F11 — v1's Appendix A3 "bulk ≈ 7 °C with the cold sink on" does not
reproduce.** The synthesis re-ran A3 in its own geometry (20×28 room, rows
7–12 × cols 10–17, 1440 ticks): bulk time-mean **21.0 °C** with the cold
sink, 22.1 °C with the hot hold alone (Appendix A3′). The v1 §12 protocol was
therefore not predicted to be comatose by its mean. §12 is recalibrated on a
new probe (A4) anyway, because a held patch's influence on the gas decays
within ~3 tiles: a 4×4 hot corner leaves mid-room row 12 at its ambient
pair-mean (≈ 22 °C) right up to the west wall, whereas a held warm strip along
that wall raises cols 3–4 to a ≈ 35 °C pair-mean — a band larvae can find.
(On HEAD the RAW cells themselves are dominated by the recorded A2 pattern;
see §12's second assertion.)

---

## 2. Decisions (with reasons)

**D1 — One law, three loops.** The per-slot law is ONE `FP_HD inline`
function, `swarm::larva_step_slot` in `cpp/src/swarm_larva.h`, wrapped in a
`LarvaLaw` functor. The CPU entry (a sequential loop over `(env, slot)`), the
per-call CUDA entry and the resident CUDA entry (one thread per `(env, slot)`)
are instantiations of species-agnostic templates (`swarm.h`,
`swarm_launch.cuh`) around that functor. Precedent: `gas_energy.h`
"transcribes ONCE for every C++ caller". Bit-identity is then by construction
(all-integer, own-slot writes, read-only fields, no atomics); the gate proves
the loops, the launch geometry and the transfers. There is **no Python copy
of the law**; the CPU entry is the reference.

**D2 — Sweep geometry.** Distance = 1 tile (`SWEEP_DIST_Q16 = FP_ONE`, a larva
constant: ruling 2 says "one tile ahead"). **One tile is a sampling distance
on the field grid — the nearest distinct cell ahead — not a body length**
(ruling 12). Angle = `±sweep_angle` (row, default 45°). At 45° from a tile
centre heading along +x the two points land on the two diagonal-ahead tiles.
The sampler takes distance and angle as arguments (field-generic).

**D3 — Turn law (tumble and wall hit share it).** Side from one Philox word,
magnitude uniform integer in `[turn_min, turn_max]` via `philox32_below` from
another; new heading `wrap(h ± m)`. Tumble and wall turn use different salts,
so their draws are independent.

**D4 — A sweep point on a solid or out-of-grid tile is BLOCKED: no reading.**
(a) It never counts as "worse" — the alarm needs both sides OPEN and worse, so
geometry alone never raises the tumble rate (no doorway/corridor jitter).
(b) For the side choice an open side beats a blocked side; two blocked sides
= no preference. That gives the wall turn "biased away from the wall side"
(chapter §4.1 step 4) from the same two samples, and nudges tumbles away from
walls. Rejected: "blocked = worst comfort" (turns walls into thermal
repellers, makes 1-wide doors filters — a larva approaching a 1-wide door
head-on would sit at the alarm rate for the ~1.7 s approach and pass straight
only ~18 % of the time); "blocked = read the wall's T" (a wall's temperature
is not the air a larva senses; out-of-grid has none).

**D5 — Kill reads the POST-move tile and applies in every state incl. COMA.**
Fork 7b: "applied after the move, so fleeing can outrun the burn". Physics does
not spare a comatose body. Honest note: at the default 40 ticks per tile the
post-move rule rarely changes an outcome; the real escape is sensing — the
sweep points see heat one tile ahead, before the own tile crosses 50 °C.

**D6 — Tumble and wall turn in one tick: the wall turn wins.** Both draws are
taken (static positions); the heading written is the wall turn's.

**D7 — New headings apply next tick.** The move uses the heading the tick
started with (§4.1 step 5); the death record carries that heading.

**D8 — Coma entry/exit consume the tick.** Entering COMA: no sensing, no
draws, no move. Waking: state → RUN, movement resumes next tick. Comparisons
strict: enter iff `T < T_ctmin`, exit iff `T > T_coma_exit`, kill iff
`T > T_lethal_hot`.

**D9 — `AI_EAT` has no branch in P3.** The state dispatch is `if COMA … else
RUN-law`, so an EAT slot (possible only through a hand-built store) is
processed by the RUN law and leaves as RUN. P6 owns the EAT branch. The fuzz
includes EAT so both backends provably agree.

**D10 — Rates are authored per second, turned into per-tick probabilities by
integer multiply.** `p = (rate_q * dt_q) >> 16` — first-order Poisson, exact
enough for `rate·dt ≪ 1`; no `expm1` (banned by the ingress lint). House
rule: per-second tunables are tick-rate independent (`config.toml:80-82`).

**D11 — Speed is authored in m/s (meters-first).** The marine precedent:
`move_speed_mps`, derived per level through `tile_size_m`
(`config.toml:28-40`, `timeline.py:118-139`). A tiles/s speed would triple on
the 1.0 m/tile test levels. Per-tick step in tiles:
`step_q = (((speed_mps_q * inv_tile_q) >> 16) * dt_q) >> 16`.

**D12 — Displacements round sign-symmetrically.** Sweep offsets and moves use
`narrow_round_signed(mul_wide(trig, len))` (`fixed_point.h:146-151`) —
`mul_q16`'s floor would bias every walk by −0.5 raw count per axis per tick.

**D13 — The Philox key is the match seed's SeedSequence entropy (R5).**
`match_key(rng)`: `ss = rng.bit_generator.seed_seq`; if `ss.spawn_key != ()`
→ `ValueError`; `e = operator.index(ss.entropy)` (accepts `int` and numpy
integers — `default_rng(np.int64(5))` stores a `numpy.int64` — and raises
`TypeError` for a list/float entropy); key = `(e & 0xFFFFFFFF, (e >> 32) &
0xFFFFFFFF)`. Draws nothing from `sim.rng`. Bits ≥ 64 are dropped (collision
only between seeds equal mod 2⁶⁴). Every `Simulation(seed=…)` caller in the
repo passes an int literal or `None` (survey of `src/`, `tests/`, `tools/`).

**D14 — Dormancy = "nothing present in any env", decided in C++.**
`swarm::step_all` returns before any upload, launch or scratch growth unless
some species has `high_water[e] > 0` in some env (L1-m5). A present species
launches over `max_e high_water[e]`; per-env masking happens in-kernel
(§A rule 4 — the host only skips when EVERY env of a species is empty).

**D15 — Launch words are derived on the host at every launch (R14).** The
Python gatherer computes `step_q`, `p_base_q`, `p_alarm_q` from the row and
the actual tick length / tile size, re-runs the species rules on exactly
those ints, and passes them as per-env launch words. The kernel never
derives; there is no C++ derive to twin.

**D16 — The host `high_water` is the one source (V6, R2).** The engine copies
`high_water[e]` into each env's launch block; the CPU loop, the kernel mask,
the launch extent and the death compaction all read it.

**D17 — Orchestration lives in `PhysicsEngine::step_swarm` (V1, R1).** Python
keeps only the ingress gather (`swarm.engine_args`) and the conversion of
death rows into `SwarmUnitDiedEvent`s. The backend choice is made in C++ (the
`step_tail` precedent reads its backend flags in C++).

**D18 — The absolute tick is `Simulation.total_tick` (V5, R6).**

---

## 3. The exact per-tick integer sequence (THIS is the twin spec)

All values are the store's native types (ROSTER, `swarm.py:64-74`). "Q16" =
int32 Q16.16. `FP_ONE = 65536`. Arithmetic in int64 wherever stated.
**No `std::` helper appears in any `FP_HD` function** — nvcc runs without
`--expt-relaxed-constexpr` (`CMakeLists.txt:125-135`), so `std::min/max/
clamp/abs` are not callable from device code; write hand-written ternaries,
as `fixed_point.h` does (R15).

### 3.1 Constants and twins (`cpp/src/swarm.h`, namespace `swarm`, unless noted)

| C++ | Value | Python source (identity-tested, T-P3-18) |
|---|---|---|
| `AI_DEAD, AI_RUN, AI_EAT, AI_COMA` | 0, 1, 2, 3 | `swarm.AI_*` |
| `FIRST_UNIT_ID` | 1 | `swarm.FIRST_UNIT_ID` |
| `PI_Q16` | 205887, `static_assert(PI_Q16 == ((fixedpoint::PI_Q30 + (1 << 13)) >> 14))` | `swarm_fixed.PI_Q16` |
| `HEADING_PERIOD_Q16` | `2 * (int64_t)PI_Q16` = 411774 | `swarm_fixed.HEADING_PERIOD_Q16` |
| `N_COLUMNS` + roster C types | 9; `int32_t` ×5, `uint32_t dist_walked`, `int32_t` ×2, `uint8_t valid` (ROSTER order), each with a `static_assert(sizeof)` | `len(ROSTER)`, `ROSTER` dtype itemsizes |
| `ENV_TICK, ENV_SEED_LO, ENV_SEED_HI, ENV_WORDS` | 0, 1, 2, 3 (the `(N, ENV_WORDS)` uint32 block Python passes) | `swarm.ENV_*` |
| `LAUNCH_TICK, LAUNCH_SEED_LO, LAUNCH_SEED_HI, LAUNCH_HW, LAUNCH_FIXED_WORDS` | 0, 1, 2, 3, 4 (C++-internal per-env launch block; derived words follow at `LAUNCH_FIXED_WORDS + i`) | — (C++ only; exported in `SWARM_CONSTANTS` for inspection) |
| `DEATH_UNIT_ID, DEATH_POS_X, DEATH_POS_Y, DEATH_HEADING, DEATH_WORDS` | 0, 1, 2, 3, 4 (per-slot record) | `swarm.DEATH_*` |
| `ROW_SPECIES, ROW_ENV, ROW_SLOT, ROW_UNIT_ID, ROW_POS_X, ROW_POS_Y, ROW_HEADING, ROW_WORDS` | 0 … 6, 7 (compacted death row) | `swarm.ROW_*` |
| `SPECIES_LAWS[i].name / param_names / derived_names / field reductions` | registry, §3.6 | `SWARM_SPECIES[i]`, `SPECIES_KERNELS[sp].params / .derived`, the `COUPLING_TABLE` swarm rows |
| `philox_streams::SALT_RESERVED_NONE` (in `philox_streams.h`) | 0 (never drawn) | `philox_streams.StreamSalt.RESERVED_NONE` |
| `philox_streams::SALT_SWARM_TUMBLE` | 1 | `StreamSalt.SWARM_TUMBLE` |
| `philox_streams::SALT_SWARM_WALL_TURN` | 2 | `StreamSalt.SWARM_WALL_TURN` |
| `SWEEP_DIST_Q16` (in `swarm_larva.h`) | `fixedpoint::FP_ONE` | — (C++ only; no Python consumer, R14) |

**Heading wrap (C++ twin of `swarm_fixed.wrap_heading_q16`, floor-mod in
int64 because C++ `%` truncates):**

```cpp
FP_HD inline int32_t wrap_heading_q16(int64_t h) {
    int64_t r = ((int64_t)PI_Q16 - h) % HEADING_PERIOD_Q16;
    if (r < 0) r += HEADING_PERIOD_Q16;
    return (int32_t)((int64_t)PI_Q16 - r);
}
FP_HD inline int64_t abs64(int64_t v) { return (v < 0) ? -v : v; }
```

Draw with `philox32.TWO_PI_Q16 = 411775` (not used in P3 — no full-circle
draw), wrap with 411774; never unify (CLAUDE.md "Swarm heading law").

### 3.2 The field reader (RAW, ruling 11) — field-generic

`tile_of(x_q, y_q, h, w, &idx)`: `tx = x_q >> 16`, `ty = y_q >> 16`
(arithmetic shift on int32 = floor, so −0.5 → −1); returns
`0 <= tx < w && 0 <= ty < h`, sets `idx = ty * w + tx`. **Never reads.**

```cpp
struct TileReader {                 // any int32 per-tile field (temperature now, food at P6)
    const int32_t* f; int w;
    FP_HD int32_t operator()(int32_t x_q, int32_t y_q) const {
        return f[(y_q >> 16) * w + (x_q >> 16)];     // precondition: (x_q, y_q) in-grid
    }
};
```

Every call site is in-grid by construction: the own position (the kernel
contract, §3.5 precondition; `check_invariants` now checks it, R16) and open
sweep points (`tile_of` true). The coupling rows label this reader
`"center"` — `exchange.reduce_center` over the one-tile footprint returns
exactly that tile — and the law registry exports the same label (§3.6), so
T-P3-22 ties the documentation to the kernel.

### 3.3 Shared kit functions (`swarm.h`)

```cpp
enum : int32_t { SIDE_MINUS = -1, SIDE_NONE = 0, SIDE_PLUS = 1 };

struct SweepSample {           // one head sweep
    int32_t here;              // R(px, py)
    bool    open_plus, open_minus;
    int32_t val_plus, val_minus;   // R(point) iff open, else 0 (never read)
};

// PLUS side = heading + angle, MINUS = heading - angle. int32 adds: |heading|
// <= PI_Q16 (canonical, §3.5 precondition) and angle <= PI_Q16/2 keep
// |a| < 1.5*PI_Q16 (inside the trig kit's pinned 4*pi accuracy band).
template <class R>
FP_HD inline SweepSample head_sweep(const R& read, const bool* solid, int h, int w,
                                    int32_t px, int32_t py, int32_t heading,
                                    int32_t angle_q, int32_t dist_q) {
    SweepSample s; s.here = read(px, py);
    for (side in {PLUS, MINUS}) {      // PLUS first (static order)
        int32_t a  = heading + side * angle_q;
        int32_t sx = px + fixedpoint::narrow_round_signed(fixedpoint::mul_wide(fixedpoint::cos_q16(a), dist_q));
        int32_t sy = py + fixedpoint::narrow_round_signed(fixedpoint::mul_wide(fixedpoint::sin_q16(a), dist_q));
        int idx; bool in = tile_of(sx, sy, h, w, &idx);
        bool open = in && !solid[idx];
        (open_side, val_side) = (open, open ? read(sx, sy) : 0);
    }
    return s;
}

// Comfort distances: smaller is better. Open beats blocked; two blocked or
// two equal = no preference.
FP_HD inline int32_t side_preference(bool op, int64_t dp, bool om, int64_t dm) {
    if (op && !om) return SIDE_PLUS;
    if (!op && om) return SIDE_MINUS;
    if (!op && !om) return SIDE_NONE;
    return (dp < dm) ? SIDE_PLUS : (dm < dp) ? SIDE_MINUS : SIDE_NONE;
}

// The turn draw (tumble AND wall turn). w_side, w_mag: two Philox words.
FP_HD inline int32_t turn_heading(int32_t heading, int32_t pref, uint32_t w_side,
                                  uint32_t w_mag, int32_t bias_q,
                                  int32_t turn_min_q, int32_t turn_max_q) {
    uint32_t u = philox32_uniform_q16(w_side);                 // 0..65535
    int32_t side;
    if (pref == SIDE_NONE) side = (u < 32768u) ? SIDE_PLUS : SIDE_MINUS;
    else                   side = (u < (uint32_t)bias_q) ? pref : -pref;
    uint32_t span = (uint32_t)(turn_max_q - turn_min_q) + 1u;   // >= 1 by rule R4
    int32_t m = turn_min_q + (int32_t)philox32_below(w_mag, span);
    return wrap_heading_q16((int64_t)heading + (int64_t)side * (int64_t)m);
}

// The wall probe (chapter §4.1 steps 2-4). No sliding.
struct Probe { bool blocked; int32_t nx, ny; };
FP_HD inline Probe probe_move(const bool* solid, int h, int w,
                              int32_t px, int32_t py, int32_t heading, int32_t step_q) {
    int32_t dx = fixedpoint::narrow_round_signed(fixedpoint::mul_wide(fixedpoint::cos_q16(heading), step_q));
    int32_t dy = fixedpoint::narrow_round_signed(fixedpoint::mul_wide(fixedpoint::sin_q16(heading), step_q));
    Probe p; p.nx = px + dx; p.ny = py + dy;
    int idx; bool in = tile_of(p.nx, p.ny, h, w, &idx);
    p.blocked = !in || solid[idx];
    return p;
}
```

(The `for (side in …)` / tuple-assignment lines are pseudocode for two
unrolled blocks, PLUS then MINUS.)

**No-tunnelling proof.** `|cos_q16|, |sin_q16| <= FP_ONE` (kit contract,
`fixed_point.h:987-988`), so `|dx|, |dy| <= step_q < FP_ONE` (rule R1). The
target tile differs from the own tile by at most one per axis: it is the own
tile or one of its 8 neighbours, and it is the tile probed. A move can never
skip a tile. Corner-cutting between two orthogonally solid tiles is accepted
(chapter §4.1 step 3) and positively tested (T-P3-1b). A larva on an open tile
therefore only ever enters open tiles; it can stand on a solid tile only if the
tile became solid under it (§0 gap).

### 3.4 Launch words (host-derived, D15)

Per species and env the engine composes the launch block (§4.4) from the
Python-provided words. For the larva:

| launch word | value (Python ints, `swarm_larva` / `swarm` helpers) |
|---|---|
| `LAUNCH_TICK` | `sim.total_tick` (masked to uint32 after a `< 2**32` check) |
| `LAUNCH_SEED_LO`, `LAUNCH_SEED_HI` | `match_key(sim.rng)` |
| `LAUNCH_HW` | `high_water[e]` (host array, D16) |
| `LAUNCH_FIXED_WORDS + LARVA_STEP_Q` | `step_q = (((speed_mps_q * inv_tile_q) >> 16) * dt_q) >> 16` |
| `LAUNCH_FIXED_WORDS + LARVA_P_BASE_Q` | `p_base_q = (tumble_rate_base_q * dt_q) >> 16` |
| `LAUNCH_FIXED_WORDS + LARVA_P_ALARM_Q` | `p_alarm_q = (tumble_rate_alarm_q * dt_q) >> 16` |

with `dt_q = quantize_scalar(sim_time)`, `inv_tile_q = quantize_scalar(1 /
gmap.tile_size_m)`. Python `>>` on ints is floor (= SAR). Rules R1/R2 run on
these exact ints before they are packed (§5.3), so the kernel consumes only
checked values.

### 3.5 Per-slot law — `swarm::larva_step_slot` (env `e`, slot `s < hw_e`)

**Kernel contract (preconditions; the gate's inputs lie inside it, R3):**
every slot `s < hw_e` with `valid == 1` has `pos` in-grid
(`[0, w<<16) × [0, h<<16)`), a canonical heading in `(−PI_Q16, PI_Q16]`,
`ai_state ∈ {RUN, EAT, COMA}`; every slot with `valid == 0` is the all-zero
form; the launch words satisfy the rules (`1 <= step_q < FP_ONE`,
`0 <= p_base_q <= p_alarm_q <= FP_ONE`); the params satisfy R3/R4 and the
field bounds. Slots `>= hw_e` are never read or written.

Inputs: the store view at flat index `k = e * max_units + s`;
`T_e = T + e*h*w`; `solid_e = solid + e*h*w`; `L = launch + e*launch_words`
(uint32; `tick = L[LAUNCH_TICK]`, `lo = L[LAUNCH_SEED_LO]`,
`hi = L[LAUNCH_SEED_HI]`, `step_q = (int32_t)L[LAUNCH_FIXED_WORDS +
LARVA_STEP_Q]`, `p_base_q`, `p_alarm_q` likewise); the params `P`
(`SwarmLarvaParams`, by value); `R = TileReader{T_e, w_}`. Output also
`death = death_rec + k*DEATH_WORDS`.

```
L0  if valid[k] == 0:  death[0..3] = 0; return.            // write own record every launch
L1  px = pos_x[k]; py = pos_y[k]; h = heading[k]; st = ai_state[k]; uid = unit_id[k]
    h_new = h; st_new = st; moved = false
L2  T_here = R(px, py)
L3  if st == AI_COMA:
        st_new = (T_here > P.T_coma_exit_q) ? AI_RUN : AI_COMA     // D8: wake, move next tick
        goto L7
    // RUN law (also AI_EAT, D9)
    if T_here < P.T_ctmin_q:  st_new = AI_COMA;  goto L7          // no sensing, no draws, no move
    st_new = AI_RUN
L4  sw = head_sweep(R, solid_e, h_, w_, px, py, h, P.sweep_angle_q, SWEEP_DIST_Q16)
    d_here = abs64((int64)sw.here - P.T_prefer_q)
    d_plus = abs64((int64)sw.val_plus - P.T_prefer_q)      (meaningful iff sw.open_plus)
    d_minus= abs64((int64)sw.val_minus - P.T_prefer_q)     (meaningful iff sw.open_minus)
    alarm  = sw.open_plus && sw.open_minus && d_plus > d_here && d_minus > d_here
    pref   = side_preference(sw.open_plus, d_plus, sw.open_minus, d_minus)
L5  philox32_draw(lo, hi, (uint32)uid, tick, /*draw_index*/ 0, SALT_SWARM_TUMBLE, Wt)
    p = alarm ? p_alarm_q : p_base_q                      // in [0, 65536] by R2
    if philox32_uniform_q16(Wt[0]) < (uint32)p:
        h_new = turn_heading(h, pref, Wt[1], Wt[2], P.turn_bias_q, P.turn_min_q, P.turn_max_q)
L6  pr = probe_move(solid_e, h_, w_, px, py, h, step_q)   // moves along the OLD heading (D7)
    if pr.blocked:
        philox32_draw(lo, hi, (uint32)uid, tick, 0, SALT_SWARM_WALL_TURN, Ww)
        h_new = turn_heading(h, pref, Ww[0], Ww[1], P.turn_bias_q, P.turn_min_q, P.turn_max_q)   // D6: overrides
    else:
        px = pr.nx; py = pr.ny; moved = true
L7  if R(px, py) > P.T_lethal_hot_q:                   // post-move (D5); every state
        death = { uid, px, py, h }                     // h = the stored (pre-update) heading
        pos_x[k] = pos_y[k] = heading[k] = hp[k] = ai_state[k] = unit_id[k] = faction[k] = 0
        dist_walked[k] = 0; valid[k] = 0               // the P2 empty-slot form (AI_DEAD == 0)
        return (died)
    death[0..3] = 0
L8  pos_x[k] = px; pos_y[k] = py
    if moved: dist_walked[k] = (uint32)(dist_walked[k] + (uint32)step_q)    // wraps mod 2^32
    heading[k] = h_new                                  // canonical: stored value or a wrap() output
    ai_state[k] = st_new
```

(`h_`, `w_` = grid height/width; the store column `h` is the heading.)
`hp[k]` and `faction[k]` are written only by L7. The argument order of
`philox32_draw` matches `philox32.h:175` (Lens 1 checked).

**Draw table** — every Philox block P3 takes (counter = `(tick, draw_index,
salt, unit_id)`, key = `(seed_lo, seed_hi)`; `draw_index` is never stored and
every block below is `draw_index 0` of its salt; a later need for more words
of the same purpose takes `draw_index 1`, never reorders these). **This table
is copied verbatim into `swarm_larva.h`'s header comment, which is its canon
at build (R21):**

| salt | when | word | use |
|---|---|---|---|
| `SWARM_TUMBLE` (1) | every RUN larva that reaches L5 | `W[0]` | `uniform_q16 < p` decides the tumble |
| | | `W[1]` | turn side (`turn_heading`) |
| | | `W[2]` | turn magnitude (`philox32_below`) |
| | | `W[3]` | reserved, unused |
| `SWARM_WALL_TURN` (2) | only when the probe is blocked | `W[0]` | turn side |
| | | `W[1]` | turn magnitude |
| | | `W[2..3]` | reserved, unused |

Integers come only through `philox32_uniform_q16` and `philox32_below` —
never `%`, never float (chapter 14 door 4).

### 3.6 The species-law registry and the three loops

**Per-species law file (`swarm_larva.h`):**

```cpp
struct SwarmLarvaParams {          // 8 words, by value (V2); order == swarm_larva.KERNEL_PARAMS
    int32_t T_prefer_q, T_lethal_hot_q, T_ctmin_q, T_coma_exit_q,
            turn_bias_q, turn_min_q, turn_max_q, sweep_angle_q;
};
constexpr int LARVA_N_PARAMS = 8;
enum : int { LARVA_STEP_Q = 0, LARVA_P_BASE_Q = 1, LARVA_P_ALARM_Q = 2, LARVA_N_DERIVED = 3 };
FP_HD inline SwarmLarvaParams larva_params_from_words(const int32_t* w);  // field-by-field copy, no reinterpret_cast
struct LarvaLaw { SwarmLarvaParams P;
    FP_HD int step_slot(const swarm::SlotIO& io, int e, int s) const; };   // returns 1 iff the slot died
extern const char* const LARVA_PARAM_NAMES[LARVA_N_PARAMS];     // defined in swarm_larva.cpp
extern const char* const LARVA_DERIVED_NAMES[LARVA_N_DERIVED];
// entries (out of line):
int swarm_larva_cpu(const swarm::LawIO&);                 // swarm_larva.cpp
int swarm_larva_cuda_percall(const swarm::LawIO&);        // swarm_larva.cu  (CUDA build)
int swarm_larva_cuda_resident(const swarm::LawIO&);       // swarm_larva.cu  (CUDA build)
int32_t swarm_larva_felt_T_host(const int32_t* T, int h, int w, int32_t x_q, int32_t y_q);  // swarm_larva.cpp, for the binding only
```

**Shared types (`swarm.h`):**

```cpp
struct StoreView {                 // one species' (n_env, max_units) SoA, host OR device
    int32_t* pos_x; int32_t* pos_y; int32_t* heading; int32_t* hp; int32_t* ai_state;
    uint32_t* dist_walked; int32_t* unit_id; int32_t* faction; uint8_t* valid;
};
struct SlotIO {                    // what one slot step reads/writes (host or device pointers)
    StoreView st; int max_units, h, w;
    const int32_t* T; const bool* solid;         // (n_env, h, w)
    const uint32_t* launch; int launch_words;    // (n_env, launch_words)
    int32_t* death;                              // (n_env, max_units, DEATH_WORDS)
};
struct LawIO {                     // one law invocation
    SlotIO io; int n_env; const int32_t* params;
    const uint32_t* launch_host;   // the host launch block (extent + hw per env)
    int32_t* death_host;           // host records; only [0, hw[e]) per env are written
};
struct SpeciesLaw {
    const char* name; int n_params, n_derived;
    const char* const* param_names; const char* const* derived_names;
    const char* field; const char* field_reduction;     // larva: "temperature", "center"
    int (*cpu)(const LawIO&);
    int (*cuda_percall)(const LawIO&);    // nullptr on the CPU build
    int (*cuda_resident)(const LawIO&);   // nullptr on the CPU build
};
extern const SpeciesLaw SPECIES_LAWS[]; extern const int N_SPECIES_LAWS;   // swarm.cpp
template <class Law> int host_loop(const Law& law, const LawIO& L);          // the CPU loop
```

`swarm.cpp` defines the registry — ONE line per species, index == position in
`SWARM_SPECIES` (identity-tested):

```cpp
const SpeciesLaw SPECIES_LAWS[] = {
    {"larva", LARVA_N_PARAMS, LARVA_N_DERIVED, LARVA_PARAM_NAMES, LARVA_DERIVED_NAMES,
     "temperature", "center", &swarm_larva_cpu, SWARM_CUDA_ENTRY(swarm_larva_cuda_percall),
     SWARM_CUDA_ENTRY(swarm_larva_cuda_resident)},
};
```

(`SWARM_CUDA_ENTRY(f)` is `&f` on the CUDA build and `nullptr` otherwise.)

**The three loops:**

- **CPU (`host_loop<Law>`, `swarm.h`; instantiated in `swarm_larva.cpp`):**
  `for e in [0, n_env): hw_e = (int)launch_host[e*W + LAUNCH_HW]; for s in
  [0, hw_e): deaths += law.step_slot(io, e, s)`; returns the death count.
- **CUDA kernel (`swarm_launch.cuh`):** `template <class Law> __global__ void
  swarm_kernel(Law law, SlotIO io, int n_env, int launch_hw)`:
  `const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  e = i / launch_hw; s = i % launch_hw; if (e >= n_env) return; if (s >=
  (int)io.launch[e*io.launch_words + LAUNCH_HW]) return; law.step_slot(io, e,
  s);`. Block size 256. `launch_hw = max_e hw[e]` from the host launch block.
- **Per-call wrapper (`percall<Law>`) and resident wrapper (`resident<Law>`)**
  (`swarm_launch.cuh`): both first do **`if (n_env == 0 || launch_hw == 0)
  return 0;` before any allocation, transfer or launch** (R3 — a 0-block grid
  is `cudaErrorInvalidConfiguration`). Per-call: H2D the 9 host columns
  (whole arrays), `T`, `solid`, the launch block into scratch; launch; D2H
  the 9 columns (whole arrays — the kernel never touches slots `≥ hw`, whose
  staged copy equals the host) and the records **`[0, hw[e])` per env
  only** (R4); count deaths by a host scan of `[0, hw[e])`. Resident: H2D the
  launch block only; launch on the caller's device columns/`T`/`solid`; D2H
  records `[0, hw[e])` per env into `death_host`; host-scan count. Neither
  touches record slots `≥ hw[e]` on the host.
- All three write only their own slot's 9 columns and 4 record words; `T`,
  `solid`, the launch block and the params are read-only. Launch geometry never
  reaches a result.

**Shared counted wrappers (`swarm.cpp` / `swarm.cu`):** `run_cpu(law, L)`,
`run_cuda_percall(law, L)`, `run_cuda_resident(law, L)` call the registry entry
and increment the process-global counters `swarm_cpu_calls` /
`swarm_cuda_calls` / `swarm_resident_calls` once per invocation with
`launch_hw > 0` (the `eos_step_cuda_calls` precedent: dispatch-fired
telemetry, so a gate cannot pass on a silently-CPU run). Both the engine and
the direct bindings go through them.

---

## 4. Placement, orchestration, input snapshot, dormancy, residency

### 4.1 Where it runs

At the **end of the slot-7 chain on every backend**, as the last physics
stage (chapter §5, now true as written):

- `PhysicsRunner.step()`: after `destroyed = self.engine.step_tail(...)` and
  its comment blocks, immediately before `return destroyed`
  (`physics_runner.py:974`), ONE statement (§4.7).
- `PhysicsRunner._step_resident()`: after the tail bracket, immediately before
  `return destroyed` (`:1351`), the same statement with `resident=True`.
- The dispatch line (`:772`) forwards `philox=philox`.

`Simulation.step` gets no new conductor slot: the swarm is part of slot 7. It
passes the Philox clock into the existing physics call and turns the runner's
death rows into tick events on the next line (§7).

### 4.2 Input snapshot contract (what "this tick" means, concretely)

- **`temperature`:** as it stands at the end of slot 7 (after `step_tail`'s
  temperature pass). Later writers — 9d ignition (`fire` only), 9e
  pump/vent/door seams (which refresh gas-cell mirrors) — reach larvae next
  tick.
- **`solid`:** the mask every slot-7 solver read this tick. It already
  includes pre-physics topology edits made THIS tick (grenade blasts at slot 2,
  `physics.py:115`; weapon wall damage at slot 4, `combat.py:178`); burn-through
  (9), over-pressure bursts (9b) and door seal/unseal (9e) reach larvae next
  tick — the chapter's "one tick late, the same lag every solver has".
- **Philox clock:** `PhiloxClock(seed_lo, seed_hi, tick)` with `tick =
  sim.total_tick` (§6), masked to uint32 after a `< 2**32` check (≈ 5.6 years
  at 24 tps).
- **Identical on both backends by construction, and self-contained:** the
  resident path uploads exactly those mirror values (`temperature` AND
  `solid`) inside `step_swarm` before it launches (R2) — it relies on no
  earlier upload in the tick.

### 4.3 The Python side: `swarm.engine_args(gmap, sim_time, clock, *, resident)` — INGRESS ONLY

Gathers the host arrays / device pointers `PhysicsEngine.step_swarm` takes and
converts the tick length, tile size and Philox clock to integer words (door
2). **It transfers nothing, launches nothing and writes nothing.** Its only
branch on world state is "does this species have `high_water.max() > 0`", and
that decides only whether the species' launch words must be valid; the engine
makes the launch decision from the same arrays.

```python
class EnginePack(NamedTuple):
    species: str
    columns: tuple          # the 9 host (N_ENV, max_units) arrays, ROSTER order
    high_water: np.ndarray  # (N_ENV,) int32 host array
    params: np.ndarray      # (len(SPECIES_KERNELS[sp].params),) int32
    derived: np.ndarray     # (N_ENV, len(SPECIES_KERNELS[sp].derived)) int32
    dev_columns: tuple      # 9 device pointers (int); all 0 unless resident

def engine_args(gmap, sim_time, clock, *, resident: bool) -> dict:
    dev = gmap.device_ptrs() if resident else None
    ctx = None
    packs = []
    for sp in SWARM_SPECIES:                                # a table walk, no if-chain
        row = gmap.swarm_species.row(sp)
        kern = SPECIES_KERNELS[sp]
        hw = getattr(gmap, high_water_attr(sp))
        derived = np.zeros((N_ENV, len(kern.derived)), dtype=np.int32)
        if int(hw.max()) > 0:
            if clock is None:
                raise ValueError("PhysicsRunner.step: a swarm is present but no philox clock "
                                 "was passed -- Simulation passes PhiloxClock(...); a direct caller must too")
            if ctx is None:
                ctx = launch_context(gmap, sim_time)
            errs = rule_violations(sp, row, ctx)             # SPECIES_RULES[sp] on the exact ints
            if errs:
                raise RuntimeError("launch-time swarm species rule failure (the tick length or tile "
                                   "size changed after load): " + "; ".join(errs))
            derived[:] = kern.derive(row, ctx)
        packs.append(EnginePack(
            sp, tuple(swarm_column(gmap, sp, c.name) for c in ROSTER), hw,
            np.array([getattr(row, a) for a in kern.params], dtype=np.int32), derived,
            tuple(dev[store_attr(sp, c.name)] for c in ROSTER) if resident else (0,) * len(ROSTER)))
    return dict(packs=tuple(packs),
                temperature=gmap.temperature[None], solid=gmap.solid[None],   # (N_ENV, h, w) views
                env=None if clock is None else pack_env(clock),               # (N_ENV, ENV_WORDS) uint32
                host_dirty=gmap.swarm_host_dirty, resident=resident,
                d_temperature=dev["temperature"] if resident else 0,
                d_solid=dev["solid"] if resident else 0)
```

`pack_env(clock)`: checks `0 <= clock.tick < 2**32` and both seed words in
`[0, 2**32)` (else `ValueError`), returns `np.array([[clock.tick,
clock.seed_lo, clock.seed_hi]] * N_ENV, dtype=np.uint32)`.
`launch_context(gmap, sim_time)`: `SpeciesContext(dt_q =
quantize_scalar(float(sim_time)), inv_tile_q = quantize_scalar(1.0 /
float(gmap.tile_size_m)))`, each in `[1, INT32_MAX]` else `ValueError`.
`rule_violations(sp, row, ctx)` returns the `"[swarm.<sp>] <rule>: <msg>"`
strings of `SPECIES_RULES[sp]` (§5.3).

### 4.4 The engine: `PhysicsEngine::step_swarm` → `swarm::step_all` (`swarm.cpp`)

```cpp
// physics_engine.h, new block after :330 (no float/double token):
std::vector<int32_t> step_swarm(
    const std::vector<swarm::SpeciesStep>& species,
    const int32_t* temperature, const bool* solid, int n_env, int h, int w,
    const uint32_t* env, bool resident,
    std::uintptr_t d_temperature, std::uintptr_t d_solid) const;
mutable std::vector<int32_t> swarm_records_;   // host per-slot record scratch, (N, M, DEATH_WORDS)

// swarm.h:
struct SpeciesStep {
    int            law;          // index into SPECIES_LAWS (== SWARM_SPECIES position)
    StoreView      host;         // host mirror columns (authoritative)
    StoreView      dev;          // resident device columns (all null unless resident)
    const int32_t* high_water;   // (n_env,) HOST -- the one source (D16)
    int            max_units;
    const int32_t* params;       // SPECIES_LAWS[law].n_params words
    const int32_t* derived;      // (n_env, SPECIES_LAWS[law].n_derived) words
    uint8_t*       host_dirty;   // IN/OUT: the R9 flag (the binding marshals the Python dict)
};
```

`PhysicsEngine::step_swarm` (defined after `physics_engine.cpp:644`) is a
forwarding body: `return swarm::step_all(species, temperature, solid, n_env,
h, w, env, resident, d_temperature, d_solid, swarm_records_);`.
`swarm::step_all(species, temperature, solid, n_env, h, w, env, resident,
d_temperature, d_solid, std::vector<int32_t>& rec) -> std::vector<int32_t>`
does exactly this, in this order:

1. `rows.clear()`.
2. **Presence (D14, L1-m5):** if no species has `high_water[e] > 0` for any
   `e`, return `rows` — no upload, no launch, no scratch growth.
3. If `env == nullptr`: throw `std::invalid_argument("step_swarm: a swarm is
   present but no Philox clock was passed")` (pybind → `ValueError`; the
   gatherer raises first on the tick path; this guards direct callers).
4. `dev = resident && backend_is_cuda()`. On the CPU build `resident == true`
   throws `std::runtime_error("step_swarm(resident=True) requires the CUDA
   build")`, and `backend_is_cuda()` is a constant `false`.
5. If `dev`: `upload_fields(temperature → d_temperature, solid → d_solid,
   n_env*h*w)` once (F5 and R2 — self-contained); `++swarm_field_uploads`.
6. For each species `sp` in registry order: `hw_max = max_e
   sp.high_water[e]`; if `0`, continue. `law = SPECIES_LAWS[sp.law]`. Compose
   the host launch block (`n_env × (LAUNCH_FIXED_WORDS + law.n_derived)`
   uint32: `env[e*3 .. e*3+2]`, `(uint32)high_water[e]`, then
   `(uint32)derived[e*nd + i]`). Size `swarm_records_` to
   `n_env*max_units*DEATH_WORDS` (contents irrelevant: every read slot is
   written this launch).
   - **`dev` path:** if `*sp.host_dirty`: `upload_store(sp.host → sp.dev,
     n_env*max_units)` (the 9 columns, whole), `*sp.host_dirty = 0`,
     `++swarm_store_uploads` (R9). Then `run_cuda_resident(law, L)` on the
     device columns + `d_temperature` + `d_solid`. Then `download_store(sp.dev
     → sp.host)` (the 9 columns, whole — `high_water` / `next_unit_id` are
     host-authoritative and never downloaded).
   - **host path:** `backend_is_cuda() ? run_cuda_percall(law, L) :
     run_cpu(law, L)` on the host columns, then `*sp.host_dirty = 1` (V7: the
     host store was written; any device copy is stale).
   - **Compaction:** `for e: for s in [0, high_water[e]): if
     rec[(e*M + s)*4 + DEATH_UNIT_ID] != 0: rows += {sp.law, e, s, rec[0..3]}`
     — (species, env, slot) order, no atomics anywhere.
7. Return `rows` (flattened `n × ROW_WORDS`).

The resident kernel reads the `temperature` and `solid` it just uploaded;
nothing else in the tick is assumed.

### 4.5 Device side (`swarm.cu`, `swarm_launch.cuh`)

- `swarm.cu`: file-local `cuda_check`; `set_swarm_backend(bool)` /
  `get_swarm_backend()` / `backend_is_cuda()`; the counters
  (`swarm_cuda_calls`, `swarm_resident_calls`, `swarm_store_uploads`,
  `swarm_field_uploads`); a file-static `SwarmDeviceScratch` (per-call staging
  for the 9 columns, `T`, `solid`; `d_launch`; `d_death`), grown on demand and
  keyed by the byte sizes of `(N, M)` and `(N, h, w)` (§A rule 5; the
  `EOSResidentScratch` precedent, `cuda_eos_resident.cu:552-665`); the
  transfer helpers `upload_fields`, `upload_store`, `download_store`,
  `upload_launch`, `download_records` (per env, `[0, hw[e])`).
- The engine's resident transfers are plain `cudaMemcpy` between the numpy
  mirror and the CuPy-owned device buffers (`gmap.device_ptrs()`), on the
  legacy default stream — the same stream the existing resident C++ launches
  and CuPy's `.set()/.get()` use (no `--default-stream per-thread` in
  `CMakeLists.txt`).
- `swarm_launch.cuh` (device-only include): `swarm_kernel<Law>`,
  `percall<Law>`, `resident<Law>` (§3.6). `swarm_larva.cu` instantiates them
  with `LarvaLaw{larva_params_from_words(L.params)}`.

### 4.6 §A habits (Erik's ruling 2026-08-27)

Rule 1: every new entry takes `(N, …)`-shaped inputs (`T`/`solid` as `(N, h,
w)` via `[None]` views today; store `(N, M)`; `high_water (N,)`; env `(N, 3)`;
derived `(N, D)`; records `(N, M, 4)`); per-env scalars live in the launch
block. Rule 2: orchestration is C++ (`PhysicsEngine::step_swarm`); Python only
gathers. Rule 3: no mirror-only field (the store is resident since P2).
Rule 4: per-env masks in-kernel (`LAUNCH_HW`); the host skips only when every
env of a species is empty. Rule 5: scratch keyed by `(N, M)` / `(N, h, w)`.

### 4.7 The exact Python lines

`physics_runner.py` (module import after `:28`: `from simulation import
swarm  # arc #63 P3: the swarm step's ingress (engine_args)`):

```python
    def step(self, gmap, sim_time, tick=0, philox=None):          # :751
        ...
            return self._step_resident(gmap, sim_time, tick=tick, philox=philox)   # :772
        ...
        # arc #63 P3 (docs/swarm_P3_impl.md §4): the swarm step, the LAST stage of slot 7.
        self.swarm_death_rows = self.engine.step_swarm(**swarm.engine_args(gmap, sim_time, philox, resident=False))
        return destroyed                                           # :974

    def _step_resident(self, gmap, sim_time, tick=0, philox=None):  # :1144
        ...
        # arc #63 P3 (docs/swarm_P3_impl.md §4): the swarm step, the LAST stage of slot 7.
        self.swarm_death_rows = self.engine.step_swarm(**swarm.engine_args(gmap, sim_time, philox, resident=True))
        return destroyed                                           # :1351
```

`simulation.py` (the physics call `:1427-1428`):

```python
            destroyed = self.physics_runner.step(
                self.gmap, sim_time_per_tick, tick=self.tick,
                philox=PhiloxClock(*self._philox_key, self.total_tick))   # arc #63 P3
            # arc #63 P3: swarm deaths (the last stage of slot 7) -> tick events, (species, env, slot) order.
            self.tick_events.extend(death_events(self.physics_runner.swarm_death_rows))
```

---

## 5. Species table: rows, units, rules, kernels

### 5.1 New `SPECIES_FIELDS` rows (only what P3 consumes; all REQUIRED, all `hot`)

| key | `Field` | unit | stored | consumer |
|---|---|---|---|---|
| `speed_mps` | `KIND_REAL_Q16, minimum=2**-16, maximum=100.0` | m/s | `speed_mps_q` | host derive (`step_q`) |
| `T_prefer` | `KIND_REAL_Q16, minimum=1.0, maximum=2000.0` | **kelvin** | `T_prefer_q` (game ΔT) | comfort |
| `T_lethal_hot` | same | **kelvin** | `T_lethal_hot_q` | L7 |
| `T_ctmin` | same | **kelvin** | `T_ctmin_q` | L3 entry |
| `T_coma_exit` | same | **kelvin** | `T_coma_exit_q` | L3 exit |
| `tumble_rate_base` | `KIND_REAL_Q16, minimum=0.0, maximum=1000.0` | 1/s | `tumble_rate_base_q` | host derive (`p_base_q`) |
| `tumble_rate_alarm` | same | 1/s | `tumble_rate_alarm_q` | host derive (`p_alarm_q`) |
| `turn_bias` | `KIND_REAL_Q16, minimum=0.0, maximum=1.0` | probability | `turn_bias_q` | `turn_heading` |
| `turn_min` | `KIND_REAL_Q16, minimum=0.0, maximum=math.pi` | rad | `turn_min_q` | `turn_heading` |
| `turn_max` | same | rad | `turn_max_q` | `turn_heading` |
| `sweep_angle` | `KIND_REAL_Q16, minimum=2**-16, maximum=math.pi / 2` | rad | `sweep_angle_q` | L4 |

`math.pi` quantizes to 205887 = `PI_Q16`; `math.pi/2` to 102944. Not added:
`radius` (nothing reads it; the size is Erik's to give, ruling 12),
`T_lethal` (ruling 6), food rows (P6). `SPECIES_RELOAD` gains all eleven as
`"hot"`; `SwarmSpeciesRow` gains the eleven `<name>_q` int fields (the P2
import-time assertion enforces the match).

**Known limit (recorded, not P3's):** `SPECIES_FIELDS` is one tuple shared by
every species (P2 design), so these larva fields would be required of a
second species too. `SPECIES_RULES` is per species from P3 on (R9); the field
set is not. The second-species patch splits it into shared + per-species field
sets.

### 5.2 Kelvin authoring — `SPECIES_UNITS`

A separate mapping beside `SPECIES_RELOAD` (the R5 precedent — a per-field
authoring property is a mapping, not a new schema kind):

```python
SPECIES_UNITS: dict = {"T_prefer": "kelvin", "T_lethal_hot": "kelvin",
                       "T_ctmin": "kelvin", "T_coma_exit": "kelvin"}
```

Import-time assertions: keys ⊆ `SPECIES_FIELDS` names; values ∈ `{"kelvin"}`;
every kelvin field is `KIND_REAL_Q16`. (`schema.py` is untouched.)

**`SwarmSpeciesTable.from_config(cfg)` becomes two-pass** (the order is
load-bearing for T22):

1. **Validate** (unchanged P2 structure checks, then `field_value_error` for
   every field of every species) — all before any conversion.
2. **Convert/quantize:** `KIND_INT` → `int`; `KIND_REAL_Q16` → if
   `SPECIES_UNITS.get(name) == "kelvin"`:
   `q = swarm_fixed.quantize_scalar(ts.from_kelvin(float(value)))` with
   `ts = temperature_scale.load(cfg)` loaded once (lazily, on the first kelvin
   field); **a `RuntimeError` or `AssertionError` from `temperature_scale.load`**
   (the migration guards raise the first, `_assert_invariants` the second when
   `phi_exp·k ≠ 1`, `temperature_scale.py:102-113`; R17) is re-raised as
   `ValueError("[swarm.<sp>].<key> = <v> K: cannot convert through
   [physics.temperature_scale]: <e>")`, so a Ctrl+R with a broken scale is
   rejected by the seam, never a crash; then `q` must lie in
   `[INT32_MIN, INT32_MAX]` else `ValueError` naming the key; otherwise
   `quantize_scalar(value)` as in P2.

`from_config` runs no cross-field rule (§5.3). Door 3: `from_kelvin` is
`(K − kelvin_ambient) / k` — algebraic, quantized at the write.

**Immunity to fire-12 (ruling 7), by construction:** the rows are Kelvin; the
stored game-ΔT integers are *derived* from the scale in force. Under HEAD's
k = 3 the defaults store T_prefer 221730, T_lethal_hot 658637, T_ctmin
−215177, T_coma_exit −171486 (Lens 1 reproduced them); under fire-12's k = 1
they store 665190, 1975910, −645530, −514458 — the same Kelvin (re-check at the
refresh pass). The hash covers stored values (`swarm.py:228-244`), so a scale
change changes `.hash`.

**Reload through the ONE seam:** Ctrl+R → `CFG.reload()` →
`sim.on_config_reload()` → `reload_species(gmap)` → `from_config(CFG)` →
`temperature_scale.load(CFG)` reads the fresh CFG (no cache). A
`[physics.temperature_scale]` edit therefore re-derives the four rows, changes
`.hash` (a "replay comparability ends here" line prints), and the next launch
reads the new ints (V2 — no device copy to re-upload). Honest limit: other
scale consumers (`gmap`'s cached `gas_t_amb_raw`, the EOS binds) do not all
follow a mid-run scale change; a scale edit mid-match is not a supported play
action.

### 5.3 `SPECIES_RULES` — invariants the kernel relies on (never tuning taste), per species (R9)

```python
class SpeciesContext(NamedTuple):
    dt_q: int          # quantize_scalar(tick seconds)
    inv_tile_q: int    # quantize_scalar(1 / tile_size_m)

def species_context(gmap, cfg=CFG) -> SpeciesContext     # load/reload: dt from cfg.clock.ticks_per_second
def launch_context(gmap, sim_time) -> SpeciesContext      # launch: dt from the tick actually being stepped
    # both: each word in [1, INT32_MAX], else ValueError

def movement_step_q(row, ctx) -> tuple[int, int]:         # SHARED (every species moves via probe_move)
    v_q = (row.speed_mps_q * ctx.inv_tile_q) >> 16
    return v_q, (v_q * ctx.dt_q) >> 16

class SpeciesRule(NamedTuple):
    name: str
    check: Callable   # (row, ctx) -> str | None   (None = holds)

SHARED_RULES = (SpeciesRule("no_tunnelling", _check_no_tunnelling),)
SPECIES_RULES: dict = {"larva": SHARED_RULES + tuple(SpeciesRule(n, c) for n, c in swarm_larva.RULES)}
def rule_violations(sp, row, ctx) -> list[str]           # "[swarm.<sp>] <rule>: <msg>" per violation
def check_species_rules(table, ctx) -> list[str]         # every row, SPECIES_RULES[row.species]
```

Import-time assertions: `set(SPECIES_RULES) == set(SWARM_SPECIES)`; every
species' tuple starts with `SHARED_RULES`; rule names unique per species.

| rule | owner | holds iff | protects |
|---|---|---|---|
| **R1 `no_tunnelling`** (ruling 9's first rule) | shared (`swarm.py`) | `v_q <= INT32_MAX` and `1 <= step_q < FP_ONE` (from `movement_step_q`) | §3.3's proof (`speed·Δt < 1 tile`); RUN means moving; the launch word fits int32 |
| **R2 `tumble_probability`** | larva (`swarm_larva.py`) | `0 <= p_base_q <= p_alarm_q <= FP_ONE` (from `swarm_larva.tumble_probabilities`) | the L5 compare is a probability; "raised to" alarm (ruling 2) |
| **R3 `coma_hysteresis`** | larva | `T_coma_exit_q >= T_ctmin_q + 1` | chapter §2 "≥ 1 count against boundary flapping" |
| **R4 `turn_range`** | larva | `turn_min_q <= turn_max_q` | `span >= 1` in `philox32_below` |

Callers (the only two installers, plus the launch):

- `allocate_swarm_store(gmap, cfg)`: after `from_config`, `errs =
  check_species_rules(table, species_context(gmap, cfg))`; any error →
  `ValueError("; ".join(errs))` (**hard error at load**, ruling 9).
- `reload_species(gmap, cfg)`: after P2's step 2 (load-class merge) and before
  step 3 (install): any error → print `[swarm] RELOAD REJECTED: <errs[0]>`,
  keep the table in force, return `False` (the P2 rejection contract).
- `engine_args` (§4.3): for each present species, on the exact launch ints →
  `RuntimeError` (unreachable unless the tick length or tile size changed
  after load).

`GameMap.__init__` sets `tile_size_m` (`gamemap.py:666`) before
`allocate_swarm_store(self)` (`:845`), so the context exists at load.

With the defaults on a 0.333 m level at 24 tps: `speed_mps_q = 13107`,
`inv_tile_q = 196805`, `dt_q = 2731`, `v_q = 39360`, `step_q = 1640`
(0.0250 tile/tick, 40 ticks/tile); on 1.0 m tiles `step_q = 546`.
`p_base_q = 273` (0.1 /s), `p_alarm_q = 2731` (1.0 /s). All rules hold (Lens 1
reproduced these numbers).

### 5.4 `SPECIES_KERNELS`, `swarm_larva.py`, and the C++ POD

`src/simulation/swarm_larva.py` (larva-only; imports only the stdlib and
`swarm_fixed` — **it must not import `swarm`**; the import direction is
`swarm → swarm_larva`):

```python
KERNEL_PARAMS = ("T_prefer_q", "T_lethal_hot_q", "T_ctmin_q", "T_coma_exit_q",
                 "turn_bias_q", "turn_min_q", "turn_max_q", "sweep_angle_q")   # == the C++ POD order
DERIVED_WORDS = ("step_q", "p_base_q", "p_alarm_q")                             # == the C++ LARVA_* order
FIELD, FIELD_REDUCTION = "temperature", "center"                                # the RAW reader (ruling 11)

def tumble_probabilities(row, ctx) -> tuple[int, int]:
    return (row.tumble_rate_base_q * ctx.dt_q) >> 16, (row.tumble_rate_alarm_q * ctx.dt_q) >> 16

RULES = (("tumble_probability", _check_tumble_probability),
         ("coma_hysteresis", _check_coma_hysteresis),
         ("turn_range", _check_turn_range))
```

`swarm.py` (shared):

```python
class SpeciesKernel(NamedTuple):
    params: tuple     # SwarmSpeciesRow attrs == the C++ params-POD word order
    derived: tuple    # the derived launch words' names == the C++ order
    derive: Callable  # (row, ctx) -> tuple[int, ...], len == len(derived)

def _derive_larva(row, ctx):
    return (movement_step_q(row, ctx)[1],) + swarm_larva.tumble_probabilities(row, ctx)

SPECIES_KERNELS = {"larva": SpeciesKernel(swarm_larva.KERNEL_PARAMS, swarm_larva.DERIVED_WORDS, _derive_larva)}
```

Import-time assertions: `set(SPECIES_KERNELS) == set(SWARM_SPECIES)`; every
`params` entry is a `SwarmSpeciesRow` field; `derived` names unique. A species
is a `SWARM_SPECIES` entry + a row + a `SPECIES_KERNELS` / `SPECIES_RULES`
entry + a law file pair + one C++ registry line — never an if-chain. T-P3-18
asserts `bp.SWARM_SPECIES_LAWS` equals these, in `SWARM_SPECIES` order.

The C++ POD is `SwarmLarvaParams` (§3.6), built on the host from the 8 words
by `larva_params_from_words` and passed to the kernel by value (V2).

### 5.5 `config.toml` — appended to `[swarm.larva]` (after `hp_max`, EOF, `:2451`)

```toml
# --- arc #63 P3: the larva law (docs/swarm_P3_impl.md). All HOT (Ctrl+R ->
# Simulation.on_config_reload, applied from the next tick). Temperatures are
# KELVIN, converted through [physics.temperature_scale] at load (immune to the
# k_temp_to_kelvin scale). Rates are per SECOND. Speed is metres per second.
speed_mps         = 0.2      # PLACEHOLDER (P4). A slow crawl: 40 ticks per 0.333 m tile at 24 tps, well inside speed*dt < 1 tile. Real larva size (hence a size-scaled speed) still to be given by Erik
T_prefer          = 303.15   # 30 C (ruling 3): comfort = |T - T_prefer|
T_lethal_hot      = 323.15   # 50 C (ruling 4): instant death above, post-move, any state
T_ctmin           = 283.15   # 10 C (ruling 5): chill-coma onset (enter below)
T_coma_exit       = 285.15   # 12 C (ruling 5): recovery (exit above); 2 K hysteresis
tumble_rate_base  = 0.1      # PLACEHOLDER (P4). 1/s: mean run 10 s when no sweep side is worse (larval runs ~5-20 s)
tumble_rate_alarm = 1.0      # PLACEHOLDER (P4). 1/s: mean run 1 s when BOTH sweep points are worse than here
turn_bias         = 0.8      # PLACEHOLDER (P4). P(turn toward the better side); Luo 2010: turns biased toward favourable
turn_min          = 0.5236   # PLACEHOLDER (P4). rad (30 deg): smallest tumble/wall turn
turn_max          = 2.0944   # PLACEHOLDER (P4). rad (120 deg): largest turn (larval turns span ~30-120 deg)
sweep_angle       = 0.7854   # PLACEHOLDER (P4). rad (45 deg): sweep points one tile ahead at +-45 deg (the diagonal-ahead tiles); 1 tile = a field-grid sampling distance, not a body length
```

---

## 6. Philox stream salts, the match key, and the absolute tick (V3, V5)

**`src/simulation/philox_streams.py`** (stdlib + nothing heavy; scanned by the
ingress lint):

```python
class StreamSalt(IntEnum):          # THE one breach-wide enum (engine/14 door-4 amendment).
    RESERVED_NONE   = 0             # APPEND-ONLY: a value is never reused or renumbered;
    SWARM_TUMBLE    = 1             # consumers of different id spaces (CPU units, swarm
    SWARM_WALL_TURN = 2             # units) share the key, so they must never share a salt.

class PhiloxClock(NamedTuple):
    seed_lo: int; seed_hi: int; tick: int

def match_key(rng) -> tuple[int, int]      # D13 / R5
```

`match_key` (R5, decision 4): `ss = rng.bit_generator.seed_seq`;
`if ss.spawn_key != (): raise ValueError("match_key: a spawned SeedSequence
(spawn_key=…) shares its parent's entropy, so sibling sims would draw
identical Philox words -- pass an int seed (the RL arc folds spawn_key into
the key when it needs sibling seeds)")`; `e = operator.index(ss.entropy)`
(TypeError for a list/float entropy, with a message naming the seed); return
`(e & 0xFFFFFFFF, (e >> 32) & 0xFFFFFFFF)`. **ACCEPTED GAP:** sibling-seed
sims are refused, not supported.

Import-time assertion: values unique. **`cpp/src/philox_streams.h`**: the same
three `constexpr uint32_t` constants, namespace `philox_streams`. Identity:
`bp.PHILOX_SALTS == {s.name: int(s) for s in StreamSalt}`.

`Simulation._reset_internal` adds, after `self.swarm = SwarmUnits(self.gmap)`
(`simulation.py:275`): `self._philox_key = match_key(self.rng)` — reads the
SeedSequence, draws nothing. Computed unconditionally (a bad seed is loud at
reset, swarm or not).

**`Simulation.total_tick`** (R6, decision 5) — a property beside
`round_index` / `round_tick` (after `simulation.py:1106`):

```python
    @property
    def total_tick(self) -> int:
        """Ticks since match start, monotone and gap-free under every ruleset
        (arc #63 P3). THE clock for tick-keyed determinism inputs (Philox
        counters); `self.tick` is round-local under TwoPhaseWEGO."""
        return self.round_index * self.ticks_per_round + self.round_tick
```

The conductor passes `PhiloxClock(*self._philox_key, self.total_tick)` in one
line (§4.7). **Save-format note (routed to ch.17 §6):** `total_tick` depends
on `turn_number`, `tick` and the ruleset; the chapter-02 save contract must
carry them (or `total_tick` itself), or a restored TwoPhase match would replay
round 0's Philox counters. `level_lights.monotonic_total_tick` is not touched
(its OnePhase double count is recorded for its own issue).

---

## 7. The death event (ruling 8)

`src/simulation/events.py` gains (appended before `__all__`, and to it):

```python
@dataclass
class SwarmUnitDiedEvent:
    """A swarm unit died this tick (arc #63 P3, engine/17 §7). Carries the
    PRE-ZERO values of its slot — the post-move position and the heading it
    moved along — because the slot itself is already the all-zero freed form.
    Renderers read this event, never the freed slot. Emitted in
    (SWARM_SPECIES order, env, ascending slot) order. Not in the synced event
    digest (field_ab_harness._SYNCED_EVENT_TYPES is unchanged): its content is
    gated CPU==GPU by tests/cuda_swarm_check.py, and cross-machine it is a
    pure function of the previous tick's digested swarm section."""
    species: str
    unit_id: int
    pos_x_q: int      # Q16.16 tile units, raw
    pos_y_q: int
    heading_q: int    # Q16.16 radians, canonical (-PI_Q16, PI_Q16]
```

- **Records (C++):** the per-slot record (§3.5 L0/L7) — every slot in
  `[0, hw[e])` writes its 4 words every launch (zeros = no death), so a stale
  record from an earlier tick is never read. No atomic append anywhere.
- **Rows (C++):** `step_all` compacts the records into `(n, ROW_WORDS)` rows in
  (species, env, slot) order (§4.4 step 6).
- **Events (Python):** `swarm.death_events(rows)`: after an explicit `N_ENV ==
  1` check (`NotImplementedError` otherwise — the carrier idiom), one
  `SwarmUnitDiedEvent(SWARM_SPECIES[row[ROW_SPECIES]], *map(int, row[ROW_UNIT_ID:ROW_WORDS]))`
  per row, in row order.
- **Into the tick:** `Simulation.step`, on the line after the physics call
  (§4.7) — slot 7, before 9's `WallDestroyedEvent`s.
- **Death and slot reuse:** the empty-slot form makes the slot the lowest free
  one for the next spawn (P2 §2.2); `high_water` and `next_unit_id` are
  unchanged by death, so a reused slot gets a new `unit_id`.

---

## 8. The larva coupling rows — in the ONE coupling table (R7)

**`CouplingRow` extended additively** (`exchange.py:661-684`; fire-12 edits
this file only at `:296-317`):

```python
@dataclass(frozen=True)
class CouplingRow:
    field: str
    reduction: Optional[str]
    response: Optional[Callable] = None   # None: executed outside Python (a swarm species kernel)
    note: str = ""
    consumer: str = "unit"                # "unit" = the CPU-unit exchange; "swarm:<species>" = that species' kernel
    reads: tuple = ()                     # species-table fields (SPECIES_FIELDS names) the response's numbers come from
```

Docstring additions: a `"unit"` row has a callable `response`; a
`"swarm:<species>"` row has `response=None` and is executed in-kernel as the
last stage of slot 7; **the future EXCHANGE-READ executor iterates only rows
with `consumer == "unit"`**. Every existing row constructs with keywords, so
the change is behaviour-neutral for them.

Three rows appended to `COUPLING_TABLE` (before its closing `)` at `:759`; the
indices `simulation.py` comments cite, `[0]`…`[4]`, are unchanged):

```python
    CouplingRow(
        field="temperature", reduction="center", consumer="swarm:larva",
        reads=("T_prefer",),
        note=("Thermotaxis head sweep (arc #63 P3): own tile + two sweep points one tile "
              "ahead at +-sweep_angle; comfort |T - T_prefer| steers the tumble rate and "
              "side (cpp/src/swarm_larva.h)."),
    ),
    CouplingRow(
        field="temperature", reduction="center", consumer="swarm:larva",
        reads=("T_lethal_hot",),
        note=("Contact/ambient heat kill (fork 7b): felt T > T_lethal_hot -> DEAD, post-move, "
              "any state. Deliberately NOT the heat|max unit row: a marine and a larva in the "
              "same hot room are damaged by different laws."),
    ),
    CouplingRow(
        field="temperature", reduction="center", consumer="swarm:larva",
        reads=("T_ctmin", "T_coma_exit"),
        note="Chill coma with hysteresis: enter below T_ctmin, exit above T_coma_exit.",
    ),
```

No new names in `__all__`. T-P3-22 ties each swarm row to the kernel (§11).
Three pre-existing tests in `test_exchange_reductions.py` are rewritten as
properties (§11 "Pre-existing test changes").

**The 9c comment amendment** (`simulation.py:1458-1468`, chapter §5): append
one line to the block, before `if self.physics_runner is not None:`:

```python
        # Swarm units (arc #63 P3; the consumer="swarm:*" rows of exchange.COUPLING_TABLE) were already judged IN slot 7 (the swarm step is physics' last stage), so within a tick larva heat/coma happen BEFORE this unit row.
```

(Split over two comment lines if the implementer's line length demands it —
still one sentence, still in that block.)

---

## 9. Files touched + merge-friendliness vs fire-12

Checked with read-only `git diff -U0 $(git merge-base HEAD origin/fire-12)
origin/fire-12` (merge-base `fdc986a`, fire-12 `0d4e07b`). Every P3 hunk is ≥ 1
unchanged line away from every fire-12 hunk. **These line numbers are HEAD
`ea4c58f`; the refresh pass (§13 step 0) re-derives them against the merged
code** — after the merge this table's overlap column is moot and only the
anchors matter.

| file | fire-12 hunks (merge-base coords) | P3 footprint (HEAD coords) | overlap? |
|---|---|---|---|
| `cpp/src/swarm.h`, `swarm.cpp`, `swarm.cu`, `swarm_launch.cuh`, `swarm_larva.{h,cpp,cu}`, `philox_streams.h` | — | new | — |
| `src/simulation/philox_streams.py`, `swarm_larva.py` | — | new | — |
| `cpp/src/physics_engine.h` (HEAD = merge-base) | +2 after 30; +6 after 47; +25 after 61; 119-125; 130-131; 158; 166 (+52); +10 after 492; +2 after 500 | `#include "swarm.h"` after `:28`; the `step_swarm` declaration + `swarm_records_` member after `:330` | No (`:29-30` separate the include; `:330` is in the quiet span 198-491) |
| `cpp/src/physics_engine.cpp` (HEAD = merge-base) | +1 after 20; +14 after 29; 64-385 (step_tail, many); +2 after 937; +2 after 946; +1 after 950; +6 after 980 | `PhysicsEngine::step_swarm` forwarding definition (~10 lines, no float/double token) after `:644` | No (quiet span 386-936) |
| `src/simulation/physics_runner.py` (HEAD = merge-base) | +1 after 27; 89-94; 149-158; 161-162; 246-253; +25 after 256; 288-305; 328-344; 348-362; 370-409; 455-456; +16 after 594; 619; 683-684; 774-786; 911-917; +23 after 934; 938-943; 998; +17 after 1018; +15 after 1078; 1160-1162; 1319; +13 after 1323; 1325-1326; 1391-1396; 1454-1624 (deleted); 1738 | import after `:28`; def `:751` (+`philox=None`); dispatch `:772`; one statement before `:974`; def `:1144` (+`philox=None`); one statement before `:1351` | No (`:28` separates the import; `:773` separates the dispatch) |
| `src/simulation/simulation.py` | 1616 (+9) (HEAD ≈ 1629) | P2 import line `:101` (+`death_events`) and a new import after `:101`; 1 line after `:275`; `total_tick` after `:1106`; the physics call `:1427-1428` + 1 line; 9c comment `:1458-1468` | No |
| `src/simulation/swarm.py` | — (branch-only) | §4.3, §5, §7, R16 | — |
| `src/simulation/events.py` | — | dataclass + `__all__` | No |
| `src/simulation/exchange.py` (HEAD = merge-base) | +16 after 298; 301 | `CouplingRow` `:661-684`; 3 rows before `:759` | No |
| `cpp/CMakeLists.txt` (HEAD = merge-base) | +2 after 86; 95; 105; +7 after 187 | `src/swarm.cpp` + `src/swarm_larva.cpp` after `:88`; a NEW `list(APPEND BREACH_SOURCES src/swarm.cu src/swarm_larva.cu)` immediately before `endif()` at `:109`; a NEW `set_source_files_properties(src/swarm.cpp src/swarm_larva.cpp PROPERTIES COMPILE_OPTIONS "${BREACH_FP_STRICT}")` after `:188` | No (never edits fire-12's lines 86/105/187) |
| `cpp/src/bindings.cpp` | many; quiet spans merge-base 1067-1574 and 3224-3640 (HEAD 1068-1639, 3289-3705) | `#include "swarm_larva.h"` after `:17`; the §10 module block after `:1493`; `.def("step_swarm", …)` after `:3540` (end of the `run_substeps_resident` def) | No |
| `stubs/breach_physics.pyi` | regenerated | regenerated (pybind11-stubgen, CPU build) | expected regen conflict → regenerate after the merge |
| `config.toml` | 1469-1498, 2350 (+33) | rows appended to `[swarm.larva]` at EOF (`:2451`, branch-only) | No |
| `tests/test_no_float_in_sim_tu.py` (HEAD = merge-base) | 47-48; +2 after 55; +22 after 326; 330; +2 after 333 | after `:334`: `SIM_TUS = SIM_TUS + ("swarm.cpp", "swarm_larva.cpp")` and `BASELINE.update({"swarm.cpp": {"float": 0, "double": 0, "fp:fast": 0}, "swarm_larva.cpp": {…0/0/0}})`; after `:344`: `MIGRATED_FLOOR_TUS = MIGRATED_FLOOR_TUS + ("swarm.cpp", "swarm_larva.cpp")` and `SIM_HEADERS = ("swarm.h", "swarm_larva.h", "philox_streams.h", "swarm_launch.cuh")`; a new `test_sim_headers_have_no_float` before `if __name__` | No |
| `tools/run_on_cuda.py` | 26-29, 84-86, 97 | `"set_swarm_backend",` after `:107` | No |
| `tests/field_ab_harness.py` | +5 after 87; 511 | `swarm_thermal_scenario_sim` after `:174` | No |
| `tests/_xarch_perfield_digest.py` | 225 (+131) | `build_swarm_lines` after `:315`; a block in `main()` before `:373` | No |
| `tests/xarch_digest.py` | — | `--scenario` + suffix | No |
| `tests/test_exchange_reductions.py` | — | `:260-300` rewritten as properties (R7) | No |
| `CLAUDE.md` | 3-4; 27-31; +2 after 69; 76; 90-91; +1 after 115 | amend "Coupling table" (`:79`) and "Ingress lint + float ratchet" (`:109`); amend branch-only rows "Deterministic RNG" (`:93`), "Swarm facade" (`:96`), "Swarm species table" (`:97`); new rows after "Swarm heading law" (`:99`) | No (`:77-78` separate `:79`; `:92` is NOT edited — it touches fire-12's 90-91) |

**Doc routing (R21 — each deviation into the passage it amends):**

- `docs/architecture/engine/17_swarm_units.md`: §3 "Counter" bullet → the
  counter tick is `sim.total_tick` (V5); §4 banner → one "As built (P3)" line
  (sensing as specified here; RAW reader, ruling 11); §4.1 step 3 → "out-of-grid
  is blocked, no clamp" (V4); §5 → "As built: `PhysicsEngine::step_swarm`
  (body `swarm::step_all`)"; §6 "Save/load" → the save contract carries what
  `total_tick` depends on (R6).
- `docs/architecture/engine/14_determinism_and_number_ingress.md`: the door-4
  amendment gains "the enum is `simulation/philox_streams.py` +
  `cpp/src/philox_streams.h`; the counter tick is `sim.total_tick`".
- `docs/rl_env_arc_proposal_2026-08-27.md` §B2: one bullet — "the swarm step's
  per-tick `temperature`/`solid` H2D and store D2H (`PhysicsEngine::step_swarm`,
  F5) disappear when the tail is ported to resident".

---

## 10. Bindings (`bindings.cpp`) + stub

**Rule (R11, the fire-12 `emissive_table.cpp` precedent):** `bindings.cpp` is
compiled `/fp:fast`, so it never calls an `FP_HD` law function inline; every
binding reaches the law through an out-of-line entry defined in a strict TU
(`swarm.cpp`, `swarm_larva.cpp`) or a `.cu`.

After HEAD `:1493`, a block tagged `// arc #63 P3 (docs/swarm_P3_impl.md §10)`:

**Both builds:**

- `swarm_species_step(species: str, columns: tuple[9 arrays], high_water,
  temperature, solid, env, params, derived, death_rec) -> int` — the CPU entry
  through the registry (`run_cpu`); composes the launch block with the same
  helper as `step_all`; returns the death count.
- `swarm_larva_felt_T(T, x_q, y_q) -> int` — `swarm_larva_felt_T_host` (the RAW
  reader); `ValueError` if `(x_q, y_q)` is out of grid.
- `swarm_wrap_heading_q16(h: int) -> int` — out-of-line in `swarm.cpp` (int64 in).
- `swarm_cpu_calls() -> int`.
- `m.attr("SWARM_CONSTANTS")` = dict of every §3.1 constant + the roster type
  sizes; `m.attr("PHILOX_SALTS")` = dict name → value;
  `m.attr("SWARM_SPECIES_LAWS")` = list, registry order, of dicts `{name,
  params: tuple, derived: tuple, field, field_reduction}`.

**CUDA build only (inside a new `#ifdef BREACH_HAS_CUDA … #endif` block):**

- `set_swarm_backend(use_cuda: bool)`, `get_swarm_backend() -> bool`.
- `cuda_swarm_species_step(...)` — same signature as `swarm_species_step`,
  per-call GPU (`run_cuda_percall`).
- `swarm_cuda_calls()`, `swarm_resident_calls()`, `swarm_store_uploads()`,
  `swarm_field_uploads() -> int`.

**PhysicsEngine method** (after HEAD `:3540`):
`.def("step_swarm", …, py::arg("packs"), py::arg("temperature"),
py::arg("solid"), py::arg("env"), py::arg("host_dirty"), py::arg("resident"),
py::arg("d_temperature") = 0, py::arg("d_solid") = 0)` → returns an
`(n, ROW_WORDS)` int32 array (empty `(0, ROW_WORDS)` when dormant). The lambda:
checks `len(packs) == N_SPECIES_LAWS` and `packs[i].species ==
SPECIES_LAWS[i].name`; derives `n_env` from `high_water.shape[0]` (every pack
the same) and `(h, w)` from `temperature.shape[1:]`; builds one
`swarm::SpeciesStep` per pack; marshals `host_dirty`: reads
`bool(host_dirty[name])` into a `uint8_t` per species before the call and
writes `py::bool_(…)` back after it (so `gmap.swarm_host_dirty` stays the P2
dict of Python bools); `env=None` → `nullptr`.

**Argument discipline (no silent copies), every array argument of every new
binding:** exact dtype (the ROSTER dtype per column, `int32` T, `bool` solid,
`uint32` env, `int32` params/derived/records), C-contiguity and the §4.6
shapes; a mismatch raises `TypeError`/`ValueError`. Take `py::array` and
validate (or `py::array_t<T, py::array::c_style>` plus an explicit
`dtype().is(...)` check): the default `forcecast` would turn a mismatched
IN/OUT column into a temporary copy whose writes are lost.

**Stub:** regenerate `stubs/breach_physics.pyi` with pybind11-stubgen against
`cpp/build/Release` (`docs/dev_setup.md:134-147`, the P1 precedent `e9e427f`);
the diff adds only the P3 symbols. CUDA-only symbols stay absent (the
documented gap, `dev_setup.md:144-147`).

---

## 11. Verification plan (tests are written from this list)

**The standard.** Each test names the property it protects and the change that
must break it. Loops run over `ROSTER`, `SWARM_SPECIES`, `SPECIES_FIELDS`,
`SPECIES_RULES[sp]`, `StreamSalt`, `COUPLING_TABLE`; nothing pins a
set/count/order designed to grow. A failing pre-existing test is a finding.
Baseline (re-taken at the refresh pass on the merged code): **no new reds**.

**Law tests call the real C++ entries through `bp.swarm_species_step("larva",
…)` on numpy arrays** (synthetic `T`/`solid` arrays are test inputs, never
`gmap.temperature`/`gmap.solid` — the "Gas temperature is a mirror" rule and
the GameMap topology rule). Explicit test params, not the config, unless the
property is about the config. The RAW reader's oracle is `T[y_q >> 16, x_q >>
16]`.

**`tests/_swarm_testkit.py`** (F7, R10, R18): captures, at import, each
species' repo row `dict(vars(getattr(CFG.swarm, sp)))`; `species_row(sp,
**overrides)` and `swarm_cfg(**overrides)` return FRESH objects per call;
`swarm_cfg()` is a full config wrapper carrying `swarm` (every species, full
rows) AND `physics.temperature_scale` copied from `CFG`, so a rejection can
only come from the validator. `larva_world(h, w, max_units, n_env=1, …)` builds
the direct-binding arrays (9 columns, `high_water`, `T`, `solid`, `env`,
`params`, `derived`, records) with `step_cpu(world)` / `step_cuda(world)`;
`larva_words(**named)` builds `params`/`derived` in `SPECIES_KERNELS` order;
`as_store_namespace(world, e)` is a `SimpleNamespace` with the gmap attribute
names (`swarm_<sp>_<col>`, `swarm_<sp>_high_water`, `swarm_next_unit_id`,
`swarm_species`, `_h`, `_w`) holding env `e` as `(1, M)` views, so tests call
the real `swarm.check_invariants`, never a re-derived copy.

### Pre-existing test changes (each: property + the change that must break it)

| test | change | property it now protects | must break if… |
|---|---|---|---|
| the four P2 fixture helpers (`test_swarm_species.py:43-55`, `test_swarm_digest.py:47-50`, `test_swarm_recorder.py:43-46`, `test_swarm_store.py:46-50`) | become thin calls into the testkit, keeping their explicit `max_units=64/8192`, `hp_max=1.0` as overrides (L2-M4) | a P2 test's rows grow with `SPECIES_FIELDS` and never depend on a P4 placeholder | a new required field has no repo value, or a P2 test starts reading a tunable it does not set |
| T22 (`test_swarm_species.py` strict schema) | every rejection case asserts `match=re.escape(f"[swarm.{sp0}].{f.name}")` (structural cases match their own token: `"[swarm] missing"`, `"[swarm].totally_unknown_species"`, `f"[swarm].{sp0} must be a table"`, `"unknown_key_xyz"`, `f"[swarm.{sp0}].hp_max missing"`); configs come from `testkit.swarm_cfg()`; one positive case: the full valid cfg passes `from_config` | the strict schema rejects each bad shape FOR ITS OWN REASON | a Namespace pass-through, a lax `KIND_REAL_Q16` branch, a dropped bounds check — each previously masked by the Kelvin conversion's missing-scale `ValueError` (L2-M4) |
| `test_exchange_reductions.py::test_coupling_table_registers_the_shipped_rows_in_order` | → the five shipped unit rows (found by `response` identity) all have `consumer == "unit"` and appear in the relative order heat < blast < push < teargas < poison; the table is a tuple of `CouplingRow` | registration and chapter order of the shipped rows, with the table free to grow | a shipped row is dropped, reordered, or loses its consumer |
| `…::test_coupling_table_reductions_name_the_vocabulary` | → every row's `reduction` is in `REDUCTIONS`, or `None` with a non-empty `note`; the shipped rows keep their reductions (heat max, blast None, push grad, teargas max, poison max), looked up by `response`, not by position | the reduction vocabulary and the shipped rows' reads | a reduction name leaves the vocabulary, or a shipped row's reduction changes |
| `…::test_coupling_table_responses_are_the_shipped_implementations` | → each shipped response function is the `response` of exactly one `"unit"` row; every `"unit"` row's response is callable; every non-unit row has `response is None` and `consumer == f"swarm:{sp}"` for some `sp in SWARM_SPECIES`; the `combat` re-exports resolve to the same objects | the executor contract (unit rows run in Python, swarm rows in-kernel) | a response is wrapped/replaced, a swarm row gains a Python response, or a consumer names no species |

### `tests/test_swarm_larva_law.py` (CPU build)

| # | property | must break if… |
|---|---|---|
| T-P3-1 | **(a) No tunnelling at max speed:** `step_q = FP_ONE − 1`; 1-thick walls — axis-aligned and a **4-connected** (edge-sharing) staircase; larvae on both sides at many sub-tile offsets and all 64 headings `k·PI_Q16/32`; over 200 ticks no larva stands on a solid tile it did not start on and none changes side of a wall line (tile-coordinate test). **(b) Corner-cut is allowed:** solid `(tx+1, ty)` and `(tx, ty+1)`, open `(tx, ty)` and `(tx+1, ty+1)`; a larva at `((tx+1)<<16) − 1, ((ty+1)<<16) − 1` heading `PI_Q16/4` with `step_q = 1000` ends the tick in tile `(tx+1, ty+1)`. | (a) the probe skips the target tile, uses the pre-move tile, or R1 is relaxed; (b) someone "fixes" the accepted corner-cut |
| T-P3-2 | **Out-of-grid = blocked:** a grid with NO solid tiles; larvae 1 step from each edge heading out; pos stays in `[0, w<<16) × [0, h<<16)` every tick, and each edge hit takes the wall-turn path (heading changes, pos unchanged that tick). | clamping instead of refusing, or an unchecked read |
| T-P3-3 | **Every write keeps the store valid:** a 500-larva, 300-tick run with high tumble rates; after every tick `swarm.check_invariants(testkit.as_store_namespace(world, 0)) == []` (the real checker: canonical headings, in-grid positions, states, empty-slot form). | a raw `h ± m` store, an int32 wrap, `TWO_PI_Q16` as the period, a move out of grid |
| T-P3-4 | **`dist_walked` wraps:** a mover with `dist_walked = 2**32 − 3` and `step_q = 10` reads 7 after one unblocked tick (uint32), and is unchanged after a blocked tick. | signed/overflowing accumulation, or accumulation on blocked ticks |
| T-P3-5 | **Coma hysteresis, both halves:** on a flat field set in turn to `T_ctmin−1`, each value in `[T_ctmin, T_coma_exit]`, `T_coma_exit+1`: RUN→COMA only below `T_ctmin`; COMA→RUN only above `T_coma_exit`; inside the band RUN stays RUN and COMA stays COMA; a COMA larva's pos/heading/dist are unchanged, and its columns are identical under two different seeds (no draw reaches it). | a single threshold, `<=` for `<`, sensing, drawing or moving in COMA |
| T-P3-6 | **Wake consumes the tick:** a COMA larva whose field rises above `T_coma_exit` becomes RUN with pos unchanged that tick and moves on the next. | waking and moving in one tick |
| T-P3-7 | **Kill is post-move and state-blind:** a lethal block of tiles (value `T_lethal_hot_q + 65536`) in a safe field: (a) a RUN larva `step_q/2` before the block's edge, heading in: dies on the tick it crosses into the block, record pos == the post-move pos (the RAW oracle says lethal there, not at the pre-move pos); (b) a larva inside the block `step_q/2` from its edge heading out: survives; (c) a COMA larva whose tile is made lethal dies. Record heading == the stored heading (the one it moved along), `unit_id` == its id. | pre-move sampling, a COMA exemption, a record from the zeroed slot |
| T-P3-8 | **Death writes the P2 empty form, records every slot:** after a kill, all 9 columns of the slot equal a never-spawned slot's; for every slot in `[0, hw)` without a death the record is 4 zeros (records pre-filled with garbage); record slots `≥ hw` keep their garbage. | leaving `valid=0` with other columns set; reading stale records; writing beyond `hw` |
| T-P3-9 | **Blocked sweep = no reading (D4), single-tick independent trials (R8):** (a) K = 2000 larvae each `0.5` tile from a wall, facing it (both sweep points at `+0.707` tile inside the wall; the probe target `step_q < 0.5` tile ahead is open), flat field; J = 40 trials, each restoring the initial arrays and setting the env tick to `j`: tumbles (heading changed; `turn_min > 0`) / (K·J) within a binomial 4σ band of `p_base/65536`, and outside the band of `p_alarm/65536` (test params: alarm = 10 × base). (b) Wall hits with one side open: e.g. a larva at `(11<<16) − step_q/2, (10<<16) + 32768` heading 0, wall column `tx = 11` solid except tile `(11, 11)` (the PLUS sweep point lands there, the MINUS one on solid `(11, 9)`, the probe on solid `(11, 10)`); over K·J independent trials the wall-turn side is PLUS with frequency `turn_bias` (4σ). | blocked-as-worst, blocked-as-reading, a side choice ignoring blockage |
| T-P3-10 | **Wall turn wins over tumble (D6):** with `p_base = p_alarm = FP_ONE` (always tumble) and a blocked probe, the written heading equals `turn_heading` evaluated on the WALL_TURN words (computed in the test from `bp.philox32_4x32_10` with `StreamSalt.SWARM_WALL_TURN`), not the TUMBLE words. | tumble overriding, or both applied |
| T-P3-11 | **Draw layout is the draw table:** for a RUN larva in open space, the tumble decision and the new heading equal the values computed in the test from `bp.philox32_4x32_10((tick, 0, SWARM_TUMBLE, uid), (lo, hi))` words 0/1/2 through `philox32_uniform_q16`/`philox32_below` and `swarm_fixed.wrap_heading_q16`. | a salt swap, a draw_index ≠ 0, `%` or float mapping, a word reorder |
| T-P3-12 | **Per-slot independence:** larva A's full trajectory (all 9 columns every tick) alone equals A's trajectory with 200 other larvae present in other slots (same uid, same slot index for A). | any cross-slot read, a shared counter, a draw keyed on slot or population |
| T-P3-13 | **Counter keys:** changing only `tick`, only `seed_lo`, only `seed_hi`, or only `unit_id` changes the tumble decision sequence of a larva over 500 ticks (≥ 1 difference each). | a counter/key word ignored |
| T-P3-14 | **Invalid and out-of-hw slots are inert:** random garbage in slots ≥ hw survives untouched; valid=0 slots stay all-zero; hw = 0 → nothing touched and 0 returned. | the loop bound is capacity, or empty slots are processed |

### `tests/test_swarm_larva_sensing.py` (CPU build; statistical, fixed seeds, paired controls — R8)

| # | property | must break if… |
|---|---|---|
| T-P3-15 | **Toward T_prefer from BOTH sides, paired against the ablated law:** 64×64 grid, solid border; `T(tx) = T_prefer_q + g·(tx − 32)` per tile column (g so every T stays inside `(T_coma_exit, T_lethal_hot)`); 400 larvae uniform on `x ∈ [4, 20]`, 400 on `[44, 60]`, headings uniform; run A = law (turn_bias 0.8, alarm 10× base), run B = ablated (turn_bias 0.5, alarm = base), **identical placements, headings, uids, seeds and ticks**; per larva `d = Δx_A − Δx_B` after 2400 ticks; mean `d` > 4 SE(`d`) for the left group and < −4 SE(`d`) for the right. The wall drift (a random walk started near a wall drifts off it: +7.5 tiles, z ≈ 10 in Lens 2's model) cancels in the pairing. | the alarm/bias sign flipped, comfort = T instead of \|T − T_prefer\| (up-gradient only: the right group's `d` turns positive), sweep sides swapped |
| T-P3-16 | **Flat field is unbiased:** flat T; 800 larvae placed symmetrically about the grid centre (uniform over `[2, 62)²`), so wall drift cancels in expectation; mean Δx and mean Δy after 2400 ticks each within 4 SE of 0; and, as single-tick independent trials (K = 2000 larvae ≥ 2 tiles from any wall, J = 40 ticks, arrays restored per trial), the tumble frequency within a binomial 4σ band of `p_base/65536`. | a floor-biased displacement (D12), an alarm from geometry, a directional draw bias |
| T-P3-17 | **The drift is the two dials:** the ablated law on T-P3-15's gradient vs the ablated law on a flat field, paired (identical placements/headings/uids/seeds/ticks): per-group mean paired Δx difference within 4 SE of 0. | a gradient response through anything other than `turn_bias` / `tumble_rate_alarm` |

### `tests/test_swarm_constants.py` (CPU build)

| # | property | must break if… |
|---|---|---|
| T-P3-18 | **Twins are identical:** every `bp.SWARM_CONSTANTS` entry with a Python source equals it (§3.1); `bp.PHILOX_SALTS == {s.name: int(s) for s in StreamSalt}`; roster type sizes equal the ROSTER itemsizes; `[d["name"] for d in bp.SWARM_SPECIES_LAWS] == list(SWARM_SPECIES)` and each entry's `params`/`derived` equal `SPECIES_KERNELS[sp].params`/`.derived` in order, and its `field`/`field_reduction` equal `swarm_larva.FIELD`/`FIELD_REDUCTION` for the larva. | a drifted constant, a renumbered salt, a param-word or registry reorder |
| T-P3-19 | **Wrap identity:** `bp.swarm_wrap_heading_q16(h) == swarm_fixed.wrap_heading_q16(h)` for P2's T17 inputs (±PI_Q16, ±3·PI_Q16, 411774, 411775, 0, ±1, INT32_MIN, INT32_MAX, Philox endpoints) plus 10⁵ seeded int64 values in `[−2³³, 2³³]`. | a truncating `%`, an int32 intermediate |
| T-P3-20 | *(v1 "derived identity" — WITHDRAWN by R14: there is no C++ derive.)* | — |
| T-P3-21 | **Salt enum is append-only-safe:** values unique, `RESERVED_NONE == 0`, no salt value reused by two names. | a duplicated value |
| T-P3-31 | **The RAW reader:** `bp.swarm_larva_felt_T(T, x, y) == T[y >> 16, x >> 16]` over seeded random fields and in-grid points incl. tile edges; out-of-grid raises `ValueError`. | a reader other than the own tile (ruling 11), an off-by-one tile lookup |

### Species table / coupling / engine / tick integration (`tests/test_swarm_species.py` additions + `tests/test_swarm_tick.py`)

| # | property | must break if… |
|---|---|---|
| T-P3-22 | **Swarm coupling rows cannot drift from the kernel (R7):** for every `COUPLING_TABLE` row with `consumer != "unit"`: `consumer == f"swarm:{sp}"`, `sp ∈ SWARM_SPECIES`, `response is None`, `reduction ∈ exchange.REDUCTIONS`, `reads ⊆ {f.name for f in SPECIES_FIELDS}`, `{f"{r}_q" for r in reads} ⊆ SPECIES_KERNELS[sp].params`, and `(row.field, row.reduction)` equals the registry's `(field, field_reduction)` for `sp`; and every species in `SWARM_SPECIES` has ≥ 1 row. | a renamed field or species, a row reading a number the kernel does not take, a reduction label that is not the kernel's reader |
| T-P3-23 | **Kelvin conversion:** for every `SPECIES_UNITS` kelvin row, `ts.to_kelvin(swarm_fixed.dequantize(stored_q))` equals the authored Kelvin within `k / 2¹⁷` K (no literal 65536). | a missing `from_kelvin`, a double conversion, a unit applied to the wrong field |
| T-P3-24 | **Scale immunity + seam:** with `CFG.physics.temperature_scale.k_temp_to_kelvin` monkeypatched 3.0 → 1.0 (keeping `phi_exp·k == 1`), a fresh GameMap stores different game-ΔT ints for the kelvin rows but the same Kelvin (T-P3-23 holds); `sim.on_config_reload()` on a running sim changes `.hash`, prints one "replay comparability ends here" line, and the next tick's launch uses the new ints (a larva on a tile whose T is lethal under one scale and not the other); a monkeypatched `phi_exp` breaking `phi_exp·k == 1` makes the reload print one `RELOAD REJECTED` line (the `AssertionError` branch, R17) and keeps the table. | cached conversion, conversion outside the seam, hashing authored text, an `AssertionError` escaping the seam |
| T-P3-25 | **Rules at load, reload and launch:** for every species and every `SPECIES_RULES[sp]` entry, a config violating only that rule raises `ValueError` naming it at `GameMap(...)`, and through `sim.on_config_reload()` prints exactly one `RELOAD REJECTED` line and keeps the table object; R1 is exercised both by speed and by a small `tile_size_m`. Launch: with a swarm present, changing `sim._tps` so the launch `step_q` breaks R1 makes the next `sim.step()` raise `RuntimeError` naming `no_tunnelling`; the same change on a swarm-free sim steps cleanly. | a rule not wired to an installer or to the launch; a launch check on a swarm-free map |
| T-P3-26 | **Swarm-free runs launch nothing and move no golden:** `capture_trajectory(default_scenario_sim, 30)` → `bp.swarm_cpu_calls()` unchanged, `physics_runner.swarm_death_rows.shape == (0, ROW_WORDS)` every tick, `trajectory_digest == GOLDEN_AGGREGATE`; plus the P2 T1 set (w6_armory golden + canary, B6, vent/B1/B2) green with zero golden edits. | an unconditional launch, a field write, an RNG draw |
| T-P3-27 | **The Philox clock is `sim.total_tick`:** in a `TwoPhaseWEGO` sim run through two round boundaries, `sim.total_tick` strictly increases by 1 per step while `sim.tick` rewinds, and the `env[0, ENV_TICK]` word `engine_args` packs (spy) equals the pre-step `total_tick`; in a `OnePhaseWEGO` sim `total_tick == sim.tick` every tick (no double count). The key equals `match_key(sim.rng)`; `sim.rng.bit_generator.state` is unchanged by `match_key`. | keying Philox on `sim.tick`, a ruleset-blind formula, drawing the key from `sim.rng` |
| T-P3-28 | **Events in (species, env, slot) order, from pre-zero values:** a sim with larvae spawned into a lethal seeded patch (`seed_gas_temperature`, the sanctioned seeding call) emits `SwarmUnitDiedEvent`s in ascending slot order on the tick of death, each matching the slot's post-move pose, and the slot is reused by the next spawn with a new id. | an unordered/atomic append, events from the zeroed slot |
| T-P3-29 | **Larvae are inert on fields:** P2's T8 (`field_digest` per tick equal with and without larvae) stays green now that the kernel runs. | the kernel writing any field |
| T-P3-30 | **`check_invariants(sim.gmap) == []` after every tick** of `swarm_thermal_scenario_sim` (CPU) for 130 ticks. | any lifecycle break |
| T-P3-32 | **`check_invariants` position rule is not vacuous (R16):** a store with one live larva at `pos_x = w << 16` (and, separately, `pos_y = −1`) reports a violation naming the position; the uncorrupted fixture is clean. | the position rule dropped or bounded by capacity instead of the grid |
| T-P3-33 | **`match_key` (R5):** `Simulation(seed=np.int64(k))` constructs and its key equals `match_key(default_rng(int(k)))`; `Simulation(seed=np.random.SeedSequence(42).spawn(2)[0])` (`default_rng` accepts a `SeedSequence`) raises `ValueError` naming `spawn_key`; `Simulation(seed=[1, 2])` (a list entropy) raises `TypeError`. | `isinstance(e, int)`, spawned siblings silently sharing a key |
| T-P3-34 | **Engine presence spans every env (L1-m5):** a direct `engine.step_swarm(...)` with hand-built `n_env = 2` packs, env 0 empty and env 1 holding one larva on a lethal tile, advances `bp.swarm_cpu_calls()` and returns one row with `ROW_ENV == 1`; with both envs empty it returns `(0, ROW_WORDS)` and advances nothing. | presence read from env 0 only |
| T-P3-35 | **R9 on the host path (V7):** after a CPU tick with a swarm present, `gmap.swarm_host_dirty[sp] is True`; after a swarm-free tick the flag is untouched (still `False` on a fresh map). | the host path leaving a stale device copy undetectable |
| T-P3-36 | **Binding argument discipline:** for `swarm_species_step` and `engine.step_swarm`, a column with the wrong dtype, a non-contiguous column, a wrong shape, or packs out of registry order each raise `TypeError`/`ValueError` and leave every input array byte-identical. | `forcecast` silently copying an IN/OUT array |

### The 3-part CUDA gate — `tests/cuda_swarm_check.py` + `tests/test_cuda_swarm.py`

Cloned from `cuda_fire_check.py` / `test_cuda_p68_fire.py` (subprocess via
`cuda_harness.run_cuda_script`, `skipif not cuda_harness.cuda_available("swarm")`,
marker `SWARM_P3_RESULT: PASS`, timeout 600 s). The wrapper satisfies
`test_cuda_check_scripts_are_wired.py`. Backend and residency flags are
process-global: each world toggles them around its own step (the
`cuda_s8a_check.py` `_mode` idiom).

- **PART 1 — forced-branch fuzz, byte identity (R3, R4).**
  `bp.swarm_species_step("larva", …)` vs `bp.cuda_swarm_species_step("larva",
  …)` on identical copies; compare all 9 columns and the FULL record arrays
  byte-for-byte (both pre-filled with identical garbage, so the untouched
  record slots `≥ hw[e]` are asserted too) and the returned counts;
  `swarm_cuda_calls()` advanced. **Preconditions (the kernel contract):** per
  env, the `[0, hw[e])` extent satisfies `check_invariants` (via
  `testkit.as_store_namespace`) — headings drawn from `(−PI_Q16, PI_Q16]` with
  extremes `−PI_Q16 + 1` and `PI_Q16`, positions in-grid, `unit_id <
  next_unit_id ≤ INT32_MAX`, states {RUN, EAT, COMA}; slots `≥ hw[e]` all-zero
  (except in the dedicated garbage forcer). Random: `n_env ∈ {1, 2, 3}` with
  unequal `hw[e]` including 0 (L1-M3); grid sizes incl. 1×N and N×1;
  `max_units` 1..300; hw < and == capacity; fields straddling every threshold;
  solid masks (none, all, random, larvae inside solid tiles); positions incl.
  edges and exact tile boundaries; `dist_walked` near 2³²; unit_ids near
  INT32_MAX; env extremes (tick 0 / 2³²−1, seeds 0 / 2³²−1); launch words
  `step_q` 1 and 65535 and `p` 0 and FP_ONE (inside R1/R2); params extremes
  (bias 0 and FP_ONE, turn span 1 and full `[0, PI_Q16]`, sweep angle min/max).
  Forcers: everyone dies; everyone enters coma; everyone blocked; everyone
  tumbles; hw = 0 in every env (nothing touched, 0 returned, **no CUDA error**
  — the zero-extent guard); garbage in all 9 columns at slots `≥ hw[e]`
  (identity only; invariants not applied to that forcer). After every
  non-garbage case the output `[0, hw[e])` satisfies `check_invariants`.
- **PART 2 — 130-tick lockstep, every transition asserted.** Two GameMaps from
  one level supply the stores and the facade (store A → `swarm_species_step`,
  store B → `cuda_swarm_species_step`); the scripted synthetic `T` field and
  interior walls are **standalone numpy arrays owned by the gate script**
  (identical copies, driven between ticks — never `gmap.temperature` /
  `gmap.solid`, R18): a cold band that rises through the hysteresis band, a hot
  front that sweeps in. Larvae spawned through `SwarmUnits` on both maps. Per
  tick: byte identity of all store arrays + records, `check_invariants(A) ==
  check_invariants(B) == []`. At a scripted tick after deaths, spawn k larvae on
  both: the slots are the freed ones (lowest free) and the ids are new.
  **Non-vacuity, each asserted > 0 from pre/post diffs:** RUN tumble (heading
  changed AND moved), wall hit (heading changed AND pos unchanged, RUN→RUN),
  RUN→COMA, COMA held with T inside `[T_ctmin, T_coma_exit]`, RUN held inside
  the band, COMA→RUN, heat kill, slot reclaim + reuse by spawn.
- **PART 2b — the sim paths, three-way (R2, R19).** Three
  `swarm_thermal_scenario_sim` worlds for 130 ticks: **CPU** (residency off,
  every backend off), **per-call via the sim** (residency off, only
  `set_swarm_backend(True)`), **resident** (`set_residency(True)`, every
  backend on incl. swarm). Scripted on all three: tick 40 `sim.swarm.spawn` of
  k larvae; tick 70 `g.destroy_wall(8, 15)` (the east-hull tile in front of the
  scenario's row-8 wall larva — a solid change the resident world must see,
  L1-M1); tick 90 `sim.swarm.reclaim` of a
  live larva, and on the next tick its slot is all-zero on every world (a
  stale device `valid=1` must never resurrect it, L1-m8). Per tick:
  `swarm_section_bytes` equal across the three, death-event lists equal,
  `field_digest` equal, `check_invariants == []` on all three. Counters:
  `swarm_cuda_calls()` advanced on every present per-call tick;
  `swarm_resident_calls()` and `swarm_field_uploads()` advanced on every
  present resident tick; `swarm_store_uploads()` advanced on the resident
  world's first tick and on the tick after each scripted facade write, and on
  no other tick. Non-vacuity: ≥ 1 death, ≥ 1 coma, ≥ 1 wall hit,
  `dist_walked` grew. **Swarm-free resident leg (L2-m7):** `default_scenario_sim`
  resident for 10 ticks: `swarm_resident_calls()`, `swarm_store_uploads()`,
  `swarm_field_uploads()` unchanged; `swarm_death_rows` empty every tick.
- **PART 3 — golden on the CUDA build's CPU path:** `trajectory_digest(
  capture_trajectory(n_steps=30)) == GOLDEN_AGGREGATE` (imported from
  `_xarch_perfield_digest`).

### Existing gates that must stay green untouched

`test_ingress_lint.py`; `test_no_float_in_sim_tu.py` (with the P3 TUs and
headers at 0/0/0); `test_w6_armory.py`, `test_b6_logic_golden.py`,
`test_field_ab_harness.py`, `test_thermostat_books.py`; all four
`test_swarm_*.py` (with the testkit), **including P2's T7/T8/T9, which now
exercise the kernel through `swarm_scenario_sim` (R20)**; `test_philox32.py`;
`test_wave_push.py` (its index-2 check survives the append);
`test_cuda_check_scripts_are_wired.py`; every existing CUDA gate (S8a's
scenarios are swarm-free, so the swarm step is dormant there).

### The thermal scenario + xarch (R20)

- **Why a second scenario:** P2's `swarm_scenario_sim` must keep its fields
  identical to `default_scenario_sim` (P2's T8 asserts exactly that), so it
  cannot carry the cold/hot forcing a P3 lockstep needs; and it never drives a
  coma. It stays as P2 built it and now also exercises the kernel.
- **`tests/field_ab_harness.py::swarm_thermal_scenario_sim()`** (no RNG):
  `Simulation(_scenario_level(), seed=SEED, breach_physics=bp,
  enable_recorder=False)` (16×16, 1.0 m tiles, `step_q = 546`); hot patch rows
  2–5 × cols 2–5 and cold patch rows 10–13 × cols 2–5 seeded once via
  `gmap.seed_gas_temperature(sel, swarm_fixed.quantize_scalar(ts.from_kelvin(K)))`
  with K = 363.15 and 268.15; 24 larvae at fixed raw-Q16 positions/headings: 4
  inside the hot patch, 4 inside the cold patch, 4 wall larvae at `x = (15 << 16)
  − 200`, `y = (ty << 16) + 32768`, `ty ∈ {6, 8, 10, 12}`, heading 0 (into the
  east hull), 12 spread over rows 7–12 × cols 6–13 with headings
  `k·PI_Q16/6`; `set_paused(False)`. (PART 2b adds its scripted mid-run
  actions on top; the xarch run uses the scenario as returned.)
- **`_xarch_perfield_digest.py`:** the canonical block is unchanged (its file
  and `GOLDEN_AGGREGATE` are byte-identical). Then `traj =
  capture_trajectory(make_sim=swarm_thermal_scenario_sim,
  n_steps=SWARM_N_STEPS)` with `SWARM_N_STEPS = 60`, and a new
  `build_swarm_lines(traj)` emitting per tick, for each species present in
  that tick's carrier, one `<tick>\t__swarm__.<sp>.<col>\t<blake2b of the
  column bytes>` line per ROSTER column and one `__swarm__.scalars` line
  (next_unit_id, each hw). `tests/_xarch_swarm_<host>.txt` interleaves, per
  tick, `build_perfield_lines(traj)`'s lines for that tick and then
  `build_swarm_lines(traj)`'s, so the existing `first_divergence` reports a
  FIELD divergence before its swarm consequence. No deaths line: a death
  event's content is a pure function of the previous tick's digested section
  (its uid/heading are there; its pos is that pos plus the deterministic move)
  (R20).
- **`xarch_digest.py`:** `--scenario {canonical,swarm}` (default canonical →
  the line is unchanged); with `swarm`, digest the swarm scenario and append
  `\tswarm=<sp>:hw<hw>,ssect_v1,sp=<species_hash[:12]>` when the last
  snapshot's carrier is present.

---

## 12. Erik's first look — `tools/swarm_larva_first_look.py` → `docs/swarm_P3_first_look.png`

Produced by the REAL sim path (CPU build, `Simulation.step`), committed with
the patch. **Recalibrated in v2 on probe A4 (R12); re-run A4 on the merged
code at the refresh pass before building the tool.**

- **Level:** synthetic `LevelData`, 24 × 36 tiles at 0.333 m (8 × 12 m), hull
  border, air interior (tilemap codes 1/4, the harness idiom);
  `seed = 20260924`.
- **Forcing — lawful only (the FieldEdit queue and the gas-energy seam, never
  a bare `temperature[...] =`):** before every `sim.step()` the tool enqueues via
  `sim.edit(...)`, each a `FieldEdit(field="wave_source", region=Region.RECT,
  …)` (the `wave_source` policy's target is the temperature energy deposit,
  booked through `gas_energy_deposit(..., "field_edit_wave")` on an accountable
  gas cell, `field_edit.py:641-662`):
  - **warm strip** rows 5–22 × cols 1–2: `amount=ts.from_kelvin(313.15)`,
    `mode=EditMode.MAX` (a 40 °C hold along the west wall — a gradient band
    larvae can find from the bulk);
  - **lethal corner** rows 1–4 × cols 1–4: `amount=ts.from_kelvin(343.15)`,
    `mode=EditMode.MAX`;
  - **cold patch** rows 21–22 × cols 33–34: `amount=1000.0,
    mode=EditMode.REMOVE, clamp=(ts.from_kelvin(278.15),
    field_edit.T_MAX_PHYS)` (held at `T_ctmin − 5 K`, L2-M6's suggestion).
  - Probe A4 (HEAD, k = 3): bulk (cols 8–27) mean 22.6 °C over the warm-up,
    23.7 °C over the following 60 s; along mid-room row 12 the pair-mean (each
    cell with its right-hand neighbour — the A2 pattern's period is 2 tiles)
    is ≈ 56 °C on the strip, ≈ 35 °C at cols 3–4, ≈ 24–25 °C beyond; the v1
    corner-only forcing leaves that row at ≈ 22 °C everywhere. The strip's own
    tiles and the corner read above 50 °C (lethal, by design).
- **Assertions after the 10 s warm-up — the tool exits non-zero and prints the
  numbers, never plotting silently (L2-M6):**
  1. the bulk (cols 8–27, all interior rows) mean over the warm-up lies in
     `[T_coma_exit + 3 K, T_prefer] = [15 °C, 30 °C]` (read from the species
     rows in Kelvin): the room sits inside the RUN band and below the
     preference, so there is a gradient to climb. A4: 22.6 °C → passes.
  2. fewer than 20 % of bulk cells read felt-T `< T_ctmin`, averaged over the
     warm-up ticks. **A4 on HEAD: 31.4 % → FAILS** (the recorded A2 pattern:
     on HEAD the RAW cells alternate ≈ 35 °C / ≈ 10 °C column by column even as
     60-s time-means).
     At the refresh pass: if it still fails on the merged code, stop and take
     it to Erik (it is the recorded gas-T pattern line, now blocking P3's
     exit) — never retune the forcing or the threshold around it.
- **Protocol:** spawn 200 larvae uniformly on open interior tiles with col ≥ 5,
  excluding rows 20–22 × cols 32–34 (the cold patch + 1-tile margin),
  positions/headings from a seeded `np.random.default_rng` in the tool
  (callers own randomness), raw Q16 into `sim.swarm.spawn`; run 120 s;
  `sim.set_paused(False)` before every step (TwoPhase auto-pause). Record per
  tick: live positions, `ai_state`, deaths (from `sim.tick_events`), and the
  felt-T map `T[y >> 16, x >> 16]` (the RAW reader, numpy, display only).
- **Cross-check the numpy felt-T against the kernel's decisions, every tick
  (L2-M6):** every `SwarmUnitDiedEvent` has felt(pos) `> T_lethal_hot_q`; every
  slot RUN at t−1 → COMA at t has felt(pos) `< T_ctmin_q`; COMA → RUN has
  felt(pos) `> T_coma_exit_q`; COMA → COMA has felt(pos) `<= T_coma_exit_q`. A
  mismatch exits non-zero. Valid because this room has no temperature writer
  after slot 7 (no pumps, vents or doors; 9d ignition writes `fire` only), so
  the post-step `gmap.temperature` is the kernel's input.
- **Control:** the identical run with the ablated law (`turn_bias = 0.5`,
  `tumble_rate_alarm = tumble_rate_base`, set through `CFG.swarm` before
  construction — the species table is the only door).
- **Figure (publication quality, 300 dpi, labelled axes in metres, units on
  every colorbar):** (a) time-mean felt-T map in °C (diverging colormap centred
  on 30 °C, contours at 10, 12, 30, 50 °C) with all trajectories (thin,
  translucent), start points, death sites (×), coma onsets (dots); (b) stacked
  state fractions RUN/COMA/DEAD vs time; (c) mean \|T_felt − 30 °C\| of RUN
  larvae vs time, law vs ablation. Caption: seed, dials, reader (RAW), achieved
  strip/corner/cold/bulk temperatures, and both assertion values. No marker or
  glyph is scaled to a body size (ruling 12).

---

## 13. Build brief (ordered; a Sonnet implementer executes this without re-deriving)

Python: `C:/Users/steen/anaconda3/python.exe` on this desktop (F10). Build the
CPU `.pyd` (`cpp/build/Release`) and the CUDA `.pyd` (`cpp/build_cuda.bat`).
Stage explicit paths only; never `git add -A`.

0. **Refresh pass (after fire-12 merges into main and main into the arc
   branch — not before):** re-derive every file:line anchor in §4.7, §8, §9,
   §10 against the merged code; re-check the Kelvin ints (§5.2, k 3 → 1);
   re-run Appendix A1 and A4 (the vent and the first-look calibration) and
   update §12's numbers; re-take the red baseline; commit the refreshed doc
   before cutting `63-p3-larva-kernel`.
1. `cpp/src/philox_streams.h` + `src/simulation/philox_streams.py` (§6).
2. `cpp/src/swarm.h` (§3.1-3.3, §3.6, §4.4): constants + static_asserts,
   `wrap_heading_q16`, `abs64`, `tile_of`, `TileReader`, `head_sweep<R>`,
   `side_preference`, `turn_heading`, `probe_move`, `StoreView`, `SlotIO`,
   `LawIO`, `SpeciesLaw`, `SpeciesStep`, `host_loop<Law>`, declarations of the
   registry, `step_all`, the counted wrappers, the counters and (under
   `#ifdef BREACH_HAS_CUDA`) the `swarm.cu` entry points. FP_HD where device
   code needs it; no `std::` in FP_HD; no line containing "float"/"double".
   Header comment: engine/17 §4/§9b, this doc, the credit block (§17).
3. `cpp/src/swarm_larva.h` (§3.5, §3.6): `SwarmLarvaParams`,
   `larva_params_from_words`, `LARVA_*` indices, `SWEEP_DIST_Q16`,
   `larva_step_slot` (verbatim order), `LarvaLaw`, the entry declarations, and
   the draw table in the header comment. Credit block.
4. `cpp/src/swarm.cpp` (registry, `step_all`, launch-block composition,
   compaction, `run_cpu`, `swarm_cpu_calls`, out-of-line
   `wrap_heading_q16` for the binding) and `cpp/src/swarm_larva.cpp`
   (`swarm_larva_cpu` = `host_loop<LarvaLaw>`, the name arrays,
   `swarm_larva_felt_T_host`). **No line may contain "float" or "double"** (F6).
5. `cpp/src/swarm_launch.cuh`, `cpp/src/swarm.cu`, `cpp/src/swarm_larva.cu`
   (§3.6, §4.5) incl. the zero-extent guard and the `[0, hw[e])` record copies.
6. `cpp/src/physics_engine.h` (include + declaration + member) and
   `physics_engine.cpp` (the forwarding definition) (§4.4, §9).
7. `cpp/CMakeLists.txt` (§9 rows, exact placement).
8. `cpp/src/bindings.cpp` (§10), then rebuild both builds.
9. `src/simulation/swarm_larva.py` (§5.4) and `src/simulation/swarm.py`:
   `import math`,
   `temperature_scale`, `swarm_larva`; `ENV_*`, `DEATH_*`, `ROW_*`; the eleven
   `SPECIES_FIELDS` rows, `SPECIES_RELOAD`, `SPECIES_UNITS` + assertions,
   `SwarmSpeciesRow` fields; two-pass `from_config` (§5.2); `SpeciesContext`,
   `species_context`, `launch_context`, `movement_step_q`, `SpeciesRule`,
   `SHARED_RULES`, `SPECIES_RULES`, `rule_violations`, `check_species_rules`;
   the rule calls in `allocate_swarm_store` and `reload_species` (§5.3);
   `SpeciesKernel`, `SPECIES_KERNELS`; `EnginePack`, `pack_env`,
   `engine_args` (§4.3); `death_events` (§7); the position rule in
   `check_invariants` (R16: live `pos_x ∈ [0, gmap._w << 16)`, `pos_y ∈ [0,
   gmap._h << 16)`, rule 7 in the docstring). Update the module docstring's
   scope line.
10. `src/simulation/events.py`: `SwarmUnitDiedEvent` (§7).
11. `src/simulation/exchange.py`: `CouplingRow` extension + the three rows (§8).
12. `src/simulation/physics_runner.py` (§4.7; anchors in §9).
13. `src/simulation/simulation.py`: imports; `self._philox_key` (§6);
    `total_tick` (§6); the physics call + events line (§4.7); the 9c comment
    line (§8).
14. `config.toml` (§5.5).
15. `tools/run_on_cuda.py`: `"set_swarm_backend",` appended to
    `_BACKEND_SETTERS` (after `:107`). Do not add the combustion setter (F9 is
    reported, not fixed here).
16. Tests (§11): `tests/_swarm_testkit.py` + the pre-existing test changes;
    `test_swarm_larva_law.py`, `test_swarm_larva_sensing.py`,
    `test_swarm_constants.py`, `test_swarm_tick.py`, additions to
    `test_swarm_species.py`; `tests/cuda_swarm_check.py` +
    `tests/test_cuda_swarm.py`; the float-ratchet additions; the thermal
    scenario; the xarch additions.
17. `tools/swarm_larva_first_look.py`; run it; if both assertions pass, commit
    `docs/swarm_P3_first_look.png`; if assertion 2 fails, stop and report
    (§12).
18. Stub regeneration (§10).
19. Verify: full suite (`python -m pytest tests -q`) — no new reds vs the
    refreshed baseline; the CUDA gate PASS; `git diff` shows no golden value
    changed; `_xarch_perfield_digest.py` reports `matches golden = True` and
    writes `_xarch_swarm_<host>.txt`; `field_ab_harness.py __main__`
    self-matches.
20. Papers → `docs/papers/` (§17).
21. `CLAUDE.md` (§16 b rows + amendments); the doc routing in §9 (chapter 17,
    chapter 14, the RL arc proposal).

**Do NOT touch:** `schema.py`; `GOLDEN_AGGREGATE` or any golden;
`tests/_pg5_base_trajectory_45050f3.pkl`; the stale "F5" wording (P2 R19);
`recorder.py`; `SimState`; `level_lights.py`; `test_wave_push.py`; any
fire-12 hunk region (§9). `COUPLING_TABLE`'s existing rows and
`test_exchange_reductions.py` change ONLY as §8/§11 specify.

---

## 14. For Erik

- **No open build-blocking question** (E1 ruled RAW, 2026-09-24). The build is
  paused until fire-12 merges (ruling 13).
- **Your question — does venting put them into coma?** Yes (§1 F1): within
  ~1 s, and they stay comatose while the frozen residual gas remains; SPACE
  tiles and hard-vacuum cells read 20 °C (they wake there, and flap on cells
  that toggle). No death (ruling 6). A heat-capacity-aware body temperature is
  the natural later fix (ACCEPTED GAP, §0). Re-probed at the refresh pass.
- FYI (decision 2): the swarm step's orchestration now lives in
  `PhysicsEngine::step_swarm` (CLAUDE.md-compliant; three small hunks in
  `physics_engine.{h,cpp}`/`bindings.cpp`, all in spans fire-12 does not edit).
- FYI (decision 4): a spawned `SeedSequence` is refused by `match_key` with a
  loud error — siblings share the parent's entropy and would draw identical
  Philox words; the RL arc folds `spawn_key` into the key when it needs it.
- FYI: larva body size — to be given before P5 (render) and before tuning
  speed at P4.
- FYI: the first-look tool's second assertion fails on HEAD (31.4 % of bulk
  cells below 10 °C — the recorded gas-T pattern); it is re-run on the merged
  code, and comes back to you only if it still fails there.

---

## 15. Critique lenses (critics fill this in, one at a time)

| lens | verdict | doc | status in v2 |
|---|---|---|---|
| Determinism & twin spec (integer sequence, draws, wrap, CPU==GPU, events) | SOUND WITH FIXES (1 blocker: L1-B1) | `swarm_P3_critique_2026-09-24.md §Lens 1` | resolved (R3, R4, R5, R6, R11, R15, R16, R17, R18) |
| Residency & CUDA (placement, snapshot contract, uploads/D2H, dormancy, §A) | SOUND WITH FIXES (majors L1-M1, L1-M3) | `swarm_P3_critique_2026-09-24.md §Lens 1` | resolved (R1, R2, R4, R19) |
| Systems reuse & scope (canon rows, new systems, merge-friendliness) | SOUND WITH FIXES (1 blocker: L2-B1; majors L2-M1, L2-M2, L2-M3) | `swarm_P3_critique_2026-09-24.md §Lens 2` | resolved (R1, R6, R7, R9, R13, R14, R20-R22) |
| Behaviour law & biology (sensing, turns, coma/kill, units, defaults, E1) | Erik — E1 ruled RAW (2026-09-24); size premise corrected; P4 play-test | — | resolved (R13, R23) |
| Verification (property tests, non-vacuity, gate coverage) | SOUND WITH FIXES (1 blocker: L2-B2; majors L2-M4, L2-M5, L2-M6) | `swarm_P3_critique_2026-09-24.md §Lens 2` | resolved (R8, R10, R11, R12, R18, R19) |

---

## 16. Systems

**(a) Existing canonical systems P3 uses:**

- **PhysicsRunner / PhysicsEngine** — `PhysicsEngine::step_swarm` is the new
  orchestration method (the `step_tail` / `run_substeps_resident` idiom); the
  runner calls it in one line on both tick paths.
- **Swarm store / facade / digest section** — the three law entries are the
  sim-side writers the facade row names; spawn/reclaim unchanged; the section
  and `check_invariants` (now with a position rule) gate every tick; the
  empty-slot form is P2's; `swarm_host_dirty` is consumed on resident ticks and
  set after host-path launches.
- **Swarm species table + Config reload seam** — eleven rows; Kelvin via
  `SPECIES_UNITS`; `SPECIES_RULES[sp]` at both installers and at launch; the
  reload seam is the only way a new table reaches the kernel (read per launch).
- **Swarm heading law** — the C++ twin, identity-tested; every heading write
  passes it.
- **Deterministic RNG (Philox)** — `philox32_draw`, `philox32_uniform_q16`,
  `philox32_below`; key = match seed, counter = `(total_tick, 0, salt, unit_id)`.
- **Fixed-point kits** — `mul_wide`, `narrow_round_signed`, `cos_q16`/`sin_q16`
  (first device use of the trig kit).
- **Temperature scale** — `from_kelvin` at table load; `to_kelvin` in the
  tool/tests.
- **Q16 boundary modules** — `swarm_fixed.quantize_scalar` / `dequantize` (no
  new copy).
- **Temperature solver / Gas temperature is a mirror** — larvae only READ
  `temperature`; the tool and scenarios seed/hold it via
  `seed_gas_temperature` and `wave_source` FieldEdits.
- **FieldEdit** — the tool's strip/corner/cold holds.
- **Coupling table** — the larva rows are `consumer="swarm:larva"` rows of the
  ONE `COUPLING_TABLE` (additive `CouplingRow` extension).
- **Tick events** (`events.py`) — `SwarmUnitDiedEvent`.
- **Field digest / GOLDEN_AGGREGATE / A/B harness / CUDA harness / xarch** —
  unchanged goldens; a new harness scenario; the 3-part gate; presence-gated
  xarch lines built from `capture_trajectory` + `build_perfield_lines` +
  `first_divergence`.
- **Ingress lint + float ratchet** — new modules scanned; the swarm TUs and the
  sim-law headers at 0/0/0.
- **RL-batch habits (§A)** — §4.6.
- **Test conventions** — property tests, paired statistical controls,
  non-vacuity counters.
- **Not used:** the recorder (unchanged), `schema.py`,
  `level_lights.monotonic_total_tick`.

**(b) New systems P3 creates — DRAFT one-line CLAUDE.md rules** (written at
implementation, pointing at the real code, appended after "Swarm heading law"):

- **Absolute tick** — `Simulation.total_tick` (`round_index * ticks_per_round
  + round_tick`, ruleset-routed). THE monotone, gap-free tick for every
  tick-keyed determinism input (Philox counters, phase rotations) — never
  `sim.tick` (round-local under TwoPhase) and never
  `level_lights.monotonic_total_tick` (double-counts under OnePhase); a save
  format must carry what it depends on (`turn_number`, `tick`, the ruleset).
- **Philox stream salts** — `src/simulation/philox_streams.py` (`StreamSalt`,
  `match_key`, `PhiloxClock`) + `cpp/src/philox_streams.h`. THE one
  breach-wide salt enum: append-only, a value is never reused or renumbered,
  every Philox consumer (CPU units included) takes a new value here; the key is
  `match_key(sim.rng)` (the match seed via `operator.index`, no draw; a
  spawned `SeedSequence` is refused); the counter tick is `sim.total_tick`;
  identity-tested against the C++ header.
- **Swarm step** — `PhysicsEngine::step_swarm` (body `swarm::step_all`,
  `cpp/src/swarm.cpp`), called by `PhysicsRunner` in ONE line after
  `step_tail` on both tick paths with `swarm.engine_args(...)`. THE only
  orchestration of swarm species: species walk, dormancy (nothing present in
  any env → no transfer, launch or allocation), the R9 dirty upload, the
  field uploads, launch, store D2H, (species, env, slot)-ordered death
  compaction. Python only gathers (ingress: arrays, pointers, launch words)
  and turns death rows into events — never host-side swarm tick logic.
- **Swarm species laws** — `swarm::SPECIES_LAWS` (`swarm.cpp`) ↔
  `swarm.SPECIES_KERNELS` / `SPECIES_RULES`. A species is a `SWARM_SPECIES`
  entry + a table row + one law file set (`swarm_<sp>.{h,cpp,cu}` with ONE
  FP_HD per-slot function) + one registry line; the CPU, per-call CUDA and
  resident CUDA entries are loops around that one law, never a copy; params
  and launch words cross as int32 words in the registry's name order
  (identity-tested); the law's draws are the draw table in its header.
- **Swarm shared kit** — `cpp/src/swarm.h` (+ `swarm.cpp`, `swarm.cu`,
  `swarm_launch.cuh`): `TileReader`, `head_sweep<Reader>`, `side_preference`,
  `turn_heading`, `probe_move`, `wrap_heading_q16`, the store/launch types,
  the loop and kernel templates, scratch and transfers. Every species senses
  through `head_sweep` with a reader and moves through `probe_move`
  (out-of-grid and solid = blocked, no sliding, `step < 1 tile`); a blocked
  sweep point is never a reading; no `std::` in FP_HD; the law headers and
  swarm TUs sit on the float ratchet at 0/0/0, and `bindings.cpp` reaches the
  law only through out-of-line entries in strict TUs.
- **Swarm species rules / units / kernels** — `swarm.SPECIES_RULES` (per
  species: the shared rules + the species' own; checked by
  `allocate_swarm_store`, `reload_species` and at every launch on the exact
  launch ints), `SPECIES_UNITS` (Kelvin rows converted through
  `temperature_scale` at load only), `SPECIES_KERNELS` (param word order,
  launch-word names, the host derive). A rule protects an invariant the kernel
  relies on, never tuning taste; launch words are derived on the host so the
  rules check exactly what the kernel consumes.
- **Swarm death events** — `events.SwarmUnitDiedEvent`, built from per-slot
  records that `step_all` compacts in (species, env, slot) order. Renderers
  read the event, never the freed slot; a sim-side consumer (P6 food) acts at
  the emission point, and if one ever reads the event the type joins the
  synced event digest in that patch.

**Amendments to existing rows** (merge-safe per §9):

- "Coupling table": "A physics→unit coupling is one row; **a swarm row names
  its `consumer` (`"swarm:<species>"`), has no Python response and is executed
  in its species kernel; the future EXCHANGE-READ executor iterates
  `consumer == "unit"` rows only**".
- "Deterministic RNG" (branch-only row): "… counter = `(sim.total_tick,
  draw_index, salt, unit_id)`; salts from `philox_streams.StreamSalt` only".
- "Swarm facade" (branch-only): "… Every host write sets
  `gmap.swarm_host_dirty[sp]`; `PhysicsEngine::step_swarm` consumes it on
  resident ticks and sets it after every host-path launch".
- "Swarm species table" (branch-only): "… Kelvin rows via `SPECIES_UNITS`,
  cross-field invariants via `SPECIES_RULES[sp]`".
- "Ingress lint + float ratchet": "…; `SIM_HEADERS` holds sim-law headers at
  0/0/0".

(v1's draft "Swarm coupling rows" row is DROPPED — R7.)

---

## 17. Credit + papers (`docs/papers/`)

Header credit block (in `swarm.h`, `swarm_larva.h/.cpp/.cu`, and the
`swarm_larva.py` module docstring) — citations verified via PubMed:

- Luo L, Gershow M, Rosenzweig M, Kang K, Fang-Yen C, Garrity PA, Samuel ADT.
  "Navigational decision making in *Drosophila* thermotaxis." *J Neurosci*
  2010;30(12):4261-72. doi:10.1523/JNEUROSCI.4090-09.2010 (PMC2871401) — run
  length, turn direction and turn size are all regulated; larvae turn toward
  favourable temperature. Honest framing for the header: Luo describes the
  input as *temporal* variation during head sweeps; the two-point spatial
  comparison is the discrete stand-in for one head sweep, not a claim that
  larvae compute a spatial gradient.
- Klein M, Afonso B, Vonner AJ, … Garrity PA, Samuel ADT. "Sensory determinants
  of behavioral dynamics in *Drosophila* thermotaxis." *PNAS*
  2015;112(2):E220-9. doi:10.1073/pnas.1416212112 (PMC4299240) — sensory
  dynamics → run/turn decisions (the graded response v1 omits: ACCEPTED GAP).
- MacMillan HA, Sinclair BJ. "Mechanisms underlying insect chill-coma."
  *J Insect Physiol* 2011;57(1):12-20. doi:10.1016/j.jinsphys.2010.10.004 —
  CTmin, reversible chill coma (the `T_ctmin`/`T_coma_exit` names).
- Monzon MA, Weidner LM, Rusch TW, Nehrozoglu S, Hamilton G. "High Temperature
  Limits of Survival and Oviposition of *Phormia regina* (Meigen) and *Lucilia
  sericata* (Meigen)." *Insects* 2022;13(11):991. doi:10.3390/insects13110991
  (PMC9693050) — adult blowflies, 24 h exposures: ≈100 % survival at 37 °C,
  ≈50 % at 41 °C, 0 % at 44 °C. The header states that the instant 50 °C kill
  is Erik's game compression of an exposure-time law, above the 24 h
  total-mortality point.
- Salmon et al. SC'11 (Philox) — already archived
  (`salmon2011_random123_sc11.pdf`).

Archive (not yet in `docs/papers/`, checked): `luo2010_drosophila_thermotaxis
_jneurosci.pdf`, `klein2015_thermotaxis_sensory_dynamics_pnas.pdf`,
`monzon2022_blowfly_high_temperature_limits_insects.pdf` (all three open
access via PMC); `macmillan2011_chill_coma_jip.pdf` if a legal copy is
obtainable (Elsevier, not in PMC) — otherwise a `README_swarm_2026-09.md`
beside `README_radiation_2026-08.md` records its full citation and DOI.

---

## Appendix A — the probes behind F1, F2, F11 and §12 (reproducible, scratch scripts not committed)

All on the CPU build `cpp/build/Release`, base interpreter,
`temperature_scale.load()` (k = 3, kelvin_ambient 293). A1/A2 on HEAD
`83b7fbd` (v1); A3′/A4 on HEAD `ea4c58f` (synthesis). Every number here is
HEAD physics; the refresh pass re-runs A1 and A4 on the merged code.

**A1 — vent (F1).** `tm = ones((16,16)); tm[1:15,1:15] = 4`,
`LevelData(..., tile_size_m=0.333)`, `Simulation(seed=1)`,
`g.destroy_wall(8, 0)`, `set_paused(False)`, step 400 ticks; per tick over
`interior[1:15,1:15]`: `T = g.temperature`, `N = gas[O2] + gas[INERT_N2]`.
Result: t = 0: T_game min −19.7, 173/196 cells below 283.15 K; t = 25: mean
−224; from t ≈ 125: mean N ≈ 0.016 (never 0), cells with N < 1 raw count
toggling 0–15, those read T = 0. Six-tile breach: same floor, faster.

**A2 — the recorded gas-T pattern (F2; a record, not a design input).**
Sealed room `tm[1:19,1:27] = 4` on a 20×28 grid, `tile_size_m=0.333`,
`seed=3`; `g.seed_gas_temperature(sel, round(ts.from_kelvin(333.15) * 65536))`
on rows 1–4 × cols 1–4 at t = 0 only; step with `set_paused(False)` each tick.
Bulk = rows 7–12 × cols 10–17, °C via `ts.to_kelvin(T/65536) − 273.15`. At
t = 48, 120, 240, 480, 719: bulk mean 20.5/19.8/20.1/20.2/20.7 °C, spatial sd
25.2/13.8/16.3/14.9/14.9 K, min/max e.g. −14.3/+54.9 °C (t = 48); mean \|ΔT\|
to the right-hand neighbour 27.7 K; per-cell 5 s EMA sd 14.0 K. Unforced room:
sd 0.00 for 480 ticks. Re-measure after fire-12 merges.

**A3′ — held corners, v1's A3 re-run (F11).** Same 20×28 room, `seed=3`; each
tick `wave_source` MAX to `from_kelvin(333.15)` on rows 1–4 × cols 1–4 and
(with the sink) `wave_source` REMOVE 1000 clamped at `from_kelvin(273.15)` on
rows 15–18 × cols 23–26; 1440 ticks. Bulk (rows 7–12 × cols 10–17) time-mean
**21.0 °C with the cold sink**, 22.1 °C hot-only (whole-interior 22.3 / 24.3
°C). v1 reported ≈ 7 °C with the sink; that does not reproduce.

**A4 — first-look calibration (§12).** 24×36 room, `tile_size_m=0.333`,
`seed=20260924`; 240-tick warm-up then a 1440-tick window. Bulk = all interior
rows × cols 8–27 for the strip rows; for the corner-only rows, the interior
minus a 6-tile margin around each patch. "RAW < 10 °C" = the per-tick fraction
of bulk cells with `T < from_kelvin(283.15)`, averaged over the window (the
warm-up value, which §12's assertion 2 uses, in brackets where measured).

| forcing | bulk mean, window (warm-up) | RAW < 10 °C, window (warm-up) | RAW > 50 °C, window | row-12 pair-mean, cols 1-2 / 3-4 / 5-6 / 7-12 |
|---|---|---|---|---|
| hot 4×4 MAX 343.15 + cold 4×4 REMOVE→273.15 (v1 §12) | 20.6 °C | 34.6 % | 0.3 % | — |
| hot 4×4 MAX 343.15 + cold 4×4 REMOVE→278.15 | 20.8 °C | 34.1 % | 0.3 % | — |
| hot 4×4 MAX 343.15 only | 21.4 °C | 32.6 % | 0.4 % | — |
| hot 4×4 MAX 343.15 + cold 2×2 REMOVE→278.15 | 21.3 °C (20.9 °C) | 32.9 % (35.5 %) | 0.3 % | 16 / 22 / 22 / 21.5 °C |
| **warm strip (rows 5–22 × cols 1–2) MAX 313.15 + hot 4×4 MAX 343.15 + cold 2×2 REMOVE→278.15 (§12)** | **23.7 °C (22.6 °C)** | **24.9 % (31.4 %)** | — | **56 / 35 / 24.5 / 23.5–25 °C** |
| same with the strip at MAX 318.15 | 23.9 °C | 23.9 % | — | — |

RAW 60-s time-means along row 12 with the strip: 63 49 56 14 37 12 39 9 35
12 39 11 … °C for cols 1, 2, 3, … — the A2 pattern (odd/even columns ≈ 35 /
≈ 10 °C) persists as a time-mean; 166 of 748 interior cells have a RAW
time-mean below 10 °C (329 of 748 with the corner-only forcing). The held
patches read above their MAX target (the 4×4 corner held at 70 °C reads ≈ 88
°C), and a patch's influence on the pair-mean decays within ~3 tiles — which
is why the strip, not a corner, is the §12 forcing. Everything here is HEAD
physics and is re-run at the refresh pass.
