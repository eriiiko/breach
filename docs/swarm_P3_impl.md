# Swarm units P3 — implementation doc: the larva behaviour kernel + CPU twin

> **v1 — 2026-09-24, pre-critique.** Arc #63, branch `63-swarm-units` (patch
> branch `63-p3-larva-kernel` per the plan). Written against HEAD `83b7fbd`
> (P1 + P2 merged). Executes `docs/architecture/engine/17_swarm_units.md`
> (BLESSED) §4 (state machine; §4.1 re-specified here as the exact integer
> sequence), §5 (placement, input snapshot), §6 (gate shape), §7 (death event,
> fork 7b), §9b (shared vs per-species files), under plan row P3
> (`docs/swarm_units_patch_plan_2026-09-10.md`, re-planned 2026-09-24). Takes
> the P2 hand-off (`docs/swarm_P2_impl.md` §9) as its contract.
>
> **Model tiers:** this doc = Opus; the build = Sonnet, oracle-gated. The feel
> test is P4, not P3.
>
> **P3 in one sentence:** one FP_HD per-slot larva law (head-sweep thermotaxis,
> threshold heat kill, chill-coma hysteresis, wall-probe movement, Philox
> draws) shared by a sequential CPU loop and a CUDA kernel, run at the end of
> the slot-7 chain on every backend, fed by Kelvin-authored species rows,
> emitting slot-ordered death events — and a 3-part CUDA gate that proves the
> two loops byte-identical.

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

## ⚠ BUILD-BLOCKING QUESTION FOR ERIK — E1: which temperature does a larva read?

Ruling 2/4 say "own-tile temperature". Probing the real engine (HEAD CPU
build; method in Appendix A) shows that **any heat disturbance in a closed room
leaves a persistent grid-scale checkerboard in the gas `temperature` field**:
after a one-shot 60 °C seed on a 4×4 corner of a 20×28 sealed room, the room
bulk (6×8 tiles, far from the seed) keeps its mean at 20 °C but has a
**spatial sd of ≈15 K for ≥ 30 s**, with **mean |ΔT| to the right-hand
neighbour ≈ 28 K** and single cells from −10 °C to +53 °C. The pattern is
quasi-static (per-cell 5 s EMA still 14 K) and grid-scale (a 2×2 mean
collapses it to 2.4 K, a 3×3 mean to 2.0 K). An unforced room stays flat
(sd 0). This is an engine property (most likely an entropy/odd-even mode of
the compressible EOS — mechanism not investigated here; the render already
"tames" the raw wind for a related reason), not P3's to fix.

Consequence for the ruled law read raw: a larva in an ordinary 20 °C room that
has seen any fire would **die on a +50 °C mosaic cell, fall into coma on a
−5 °C cell, and steer by checkerboard noise** — the 2 K coma hysteresis is
swamped. Two readers, both fully specified in §3.2 so the build proceeds on
Erik's answer alone:

- **RAW (as ruled):** `felt_T = temperature[tile(x, y)]`.
- **FOOTPRINT (recommended):** `felt_T = floor(mean of the 2×2 tiles around
  the grid vertex nearest (x, y))` — ≈ the larva's 0.6 × 0.15 m body footprint
  (1.8 × 0.45 tiles), four integer reads + `floordiv_q(sum, 4)`, residual
  mosaic 2.4 K. The same reader serves "here", both sweep points, coma and
  kill, so it is ONE decision.

Only the chosen reader is written. Numbers are HEAD (k = 3); fire-12's frame
collapse (k 3 → 1, φ 1, eos T_amb 293) changes the EOS frame, so the mosaic's
Kelvin amplitude must be re-measured after that merge. **If Erik does not
answer before the build, the builder implements RAW (the locked ruling) and the
first-look plot (§12) will show the effect.**

---

## v1 deviations from the plan / chapter / P2 hand-off (resolution → evidence)

| # | Plan/chapter/P2 said | P3 does | Why (evidence) |
|---|---|---|---|
| V1 | "`SwarmSolver` in PhysicsEngine" (plan row P3); "last stage of PhysicsEngine's orchestration" (ch. §5) | A stateless free-function family in `swarm_larva.{h,cpp,cu}`, bound directly (the `sky_exchange_step` / `cuda_combustion_step` precedent), called by `PhysicsRunner` right after `engine.step_tail(...)` on both tick paths. **Zero hunks in `physics_engine.{h,cpp}`.** | The slot-7 chain's tail is orchestrated by `PhysicsRunner` (`step_tail`, `_run_combustion`, `_run_sky_exchange` are runner calls, `physics_runner.py:863-953`); the pass has no state to own; fire-12 rewrites `physics_engine.h:119-197, 492-500` and `physics_engine.cpp` (15 hunks); `physics_engine.cpp` is on the float ratchet (`test_no_float_in_sim_tu.py:333`). |
| V2 | "species-table H2D keyed on hash" (P2 §9) | Species params passed **by value** as a kernel argument (a ≤ 64-byte POD, the CUDA constant-bank idiom), re-read from `gmap.swarm_species` at every launch. | Meets P2's purpose (no flag; a reload reaches the device from the next tick) with no device copy to go stale and no cache key to get wrong. The per-env block (§A rule 1) is a device array; the species block is per species, not per env. |
| V3 | "the P1 salt enum" (task) | **Created in P3**: `src/simulation/philox_streams.py` + `cpp/src/philox_streams.h`. | No salt enum exists (grep `salt` over `src/`, `cpp/`, `tests/`: only philox32's own API names). Chapter 14's amendment requires ONE breach-wide enum. |
| V4 | Chapter §4.1 step 3: "coordinates clamped to the grid (out-of-grid reads as solid)" | Bounds-check before any read; out-of-grid = blocked; no clamp for the solid decision. | A clamp would make an out-of-grid target read the edge tile's solidity, contradicting "out-of-grid reads as solid" whenever the edge tile is open. |
| V5 | Philox counter `tick` = `sim.tick` (implied) | Counter tick = the **absolute tick since reset**: `sim.round_index * sim.ticks_per_round + sim.round_tick`. | Under the default `TwoPhaseWEGO`, `sim.tick` rewinds to 0 every round (`simulation.py:1846`, `ruleset.py:115-121`). Keying Philox on it would replay every larva's draw sequence every round — a periodic walk, not a random one. |

---

## 0. Scope

**In:**

- the shared swarm kit `cpp/src/swarm.h` (roster/enum/heading twins, tile
  lookup, the field-generic head-sweep sampler, side preference, wall probe,
  turn draw) (§3);
- the larva law `cpp/src/swarm_larva.h` (FP_HD per-slot step), its CPU loop
  `swarm_larva.cpp` and CUDA kernel + per-call + resident launches
  `swarm_larva.cu` (§3, §4);
- the breach-wide Philox stream-salt enum + match key (§6);
- species rows, `SPECIES_UNITS` (Kelvin), `SPECIES_RULES`, `SPECIES_KERNELS`
  (§5);
- placement in `PhysicsRunner` on the CPU, per-call-CUDA and resident paths,
  dormancy, host-dirty upload, D2H (§4);
- the death event (§7);
- the documenting swarm coupling rows + the 9c comment (§8);
- bindings, stub, CMake, float ratchet (§9, §10);
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
- the `hp` column is untouched except by death (fork 7b).

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
  `-292.0` ≙ 1 K on fire-12). Probe (Appendix A, 16×16 room, one breach tile):
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
  293 K ↔ floor jump). No death (ruling 6).
- **Solid tiles:** a wall's `temperature` is its own thermal truth, derived by
  `TemperatureSolver` alone (CLAUDE.md "Temperature solver" row): ~ambient,
  raised by radiation/fire deposits, relaxing with the material's `cool_shift`
  (hull 5 → e-fold 1.3 s). **Larvae never sense a solid tile** in this design
  (§2 D4): a sweep point on a solid tile is "blocked", not a reading. The own
  tile is solid only in the door-closes-on-a-larva gap (§0), where "here" is the
  door's temperature.

**F2 — The temperature mosaic (E1 above).** Appendix A.

**F3 — `sim.tick` is round-local under `TwoPhaseWEGO`** (deviation V5):
`_end_round` sets `self.tick = 0` and `turn_number += 1`
(`simulation.py:1846-1849`); the base ruleset defines
`round_index = turn_number - 1`, `round_tick = sim.tick`
(`ruleset.py:111-121`); `OnePhaseWEGO` is free-running (`ruleset.py:372-378`).
`round_index * ticks_per_round + round_tick` is monotone and gap-free under
both. (The same rewind also repeats fire-12's D4 fan phase every round —
harmless there, noted as a finding.)

**F4 — No salt enum exists** (V3).

**F5 — On the resident path the tail runs on the host mirror.** Step 6's D2H
(`physics_runner.py:1299-1300`) happens BEFORE the combustion and tail
brackets (`:1303-1336`), which write `temperature` on the mirror only. At the
end of the resident tick the device `temperature` is the post-EOS value, stale
by the tail. So a resident larva launch must upload `temperature` first
(§4.3). RL-arc B2 ("Port combustion + tail to resident",
`docs/rl_env_arc_proposal_2026-08-27.md` §B2) deletes that one upload.

**F6 — Float ratchet.** `test_no_float_in_sim_tu.py` counts LINES containing
the words `float`/`double` (comments included) per sim TU and fails if a count
rises (`:378-396`). P3 adds `swarm_larva.cpp` at 0/0/0 as a hard gate (§9) —
so **no line of `swarm_larva.cpp` may contain the words "float" or "double",
not even in a comment**.

**F7 — P2's test fixtures pin the species row.** `_valid_row` /
`_swarm_cfg_namespace` build `{"max_units", "hp_max"}` literally
(`test_swarm_species.py:43-46`, `test_swarm_digest.py:47-50`,
`test_swarm_recorder.py:43-46`, `test_swarm_store.py:46-50`). With P3's required rows every valid-path P2
test would fail on "missing key", and T22's rejection cases would pass
vacuously for the wrong reason. This is a snapshot-pinning defect (the tests
pin a set designed to grow), fixed in P3 by deriving the rows from the repo's
own `[swarm.<species>]` (§11, `tests/_swarm_testkit.py`). Reported as a finding.

**F8 — Unit coupling-table tests pin the row count and order**
(`test_exchange_reductions.py:266-267, 275, 291`; `test_wave_push.py:426-430`),
and a future EXCHANGE-READ slot will iterate `COUPLING_TABLE` over CPU units.
So the larva rows go into a sibling table, not into `COUPLING_TABLE` (§8).
The pinning itself is a finding (left untouched).

**F9 — `tools/run_on_cuda.py::_BACKEND_SETTERS` omits
`set_combustion_backend`** (`:92-108`; grep: only tests set it), so `--cuda`
never runs combustion on the GPU despite the comment at `:136-138`. Finding in
another system; P3 only appends its own setter.

**F10 — Interpreter.** On the desktop the cp311 `.pyd` runs under
`C:/Users/steen/anaconda3/python.exe` (3.11.7, NumPy 2.4.6); the conda env
`data` here is 3.12.14 (NumPy 2.5.3) and cannot import it. CLAUDE.md's
"always the conda env `data`" is Lenovo-true, desktop-false (P2's brief used
the base interpreter too). Finding; the build uses the base interpreter.

---

## 2. Decisions (with reasons)

**D1 — One law, two loops.** The per-slot law is ONE `FP_HD inline` function,
`swarm::larva_step_slot` in `cpp/src/swarm_larva.h`. The CPU twin
(`swarm_larva.cpp`) is a sequential loop over `(env, slot)`; the CUDA kernel
(`swarm_larva.cu`) is one thread per `(env, slot)`. Both call the same
function. Precedent: `gas_energy.h` "transcribes ONCE for every C++ caller".
Bit-identity is then by construction (all-integer, own-slot writes, read-only
fields, no atomics); the gate proves the loops, the launch geometry and the
transfers. There is **no Python copy of the law**; the CPU twin is the
reference.

**D2 — Sweep geometry.** Distance = 1 tile (`SWEEP_DIST_Q16 = FP_ONE`, a larva
constant: ruling 2 says "one tile ahead"); angle = `±sweep_angle` (row,
default 45°). At 45° from a tile centre heading along +x the two points land on
the two diagonal-ahead tiles. The sampler takes distance and angle as
arguments (field-generic).

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
walls. Interpretable: a wall has no air temperature, touch is a separate real
cue; deterministic: two flags. Rejected: "blocked = worst comfort" (turns
walls into thermal repellers, makes 1-wide doors filters — a larva approaching
a 1-wide door head-on would sit at the alarm rate for the ~1.7 s approach and
pass straight only ~18 % of the time); "blocked = read the wall's T" (a wall's
temperature is not the air a larva senses; out-of-grid has none).

**D5 — Kill reads the POST-move tile and applies in every state incl. COMA.**
Fork 7b: "applied after the move, so fleeing can outrun the burn". Physics does
not spare a comatose body. Honest note: at the default 40 ticks per tile the
post-move rule rarely changes an outcome; the real escape is sensing — the
sweep points see heat one tile ahead, before the own tile crosses 50 °C.

**D6 — Tumble and wall turn in one tick: the wall turn wins.** Both draws are
taken (static positions); the heading written is the wall turn's.

**D7 — New headings apply next tick.** The move uses the heading the tick
started with (§4.1 step 5); the death record carries that heading (the pose
the larva actually had).

**D8 — Coma entry/exit consume the tick.** Entering COMA: no sensing, no
draws, no move. Waking: state → RUN, movement resumes next tick. Comparisons
strict: enter iff `T < T_ctmin`, exit iff `T > T_coma_exit`, kill iff
`T > T_lethal_hot`.

**D9 — `AI_EAT` has no branch in P3.** Nothing writes it (ruling 1). The
state dispatch is `if COMA … else RUN-law`, so an EAT slot (possible only
through a hand-built store) is processed by the RUN law and leaves as RUN. P6
owns the EAT branch. The fuzz includes EAT so both backends provably agree.

**D10 — Rates are authored per second, turned into per-tick probabilities by
integer multiply.** `p = mul_q16(rate_q, dt_q)` — first-order Poisson, exact
enough for `rate·dt ≪ 1`; no `expm1` (banned by the ingress lint,
`test_ingress_lint.py:52-56`). House rule: per-second tunables are tick-rate
independent (`config.toml:80-82`).

**D11 — Speed is authored in m/s (meters-first).** The marine precedent:
`move_speed_mps`, derived per level through `tile_size_m`
(`config.toml:28-40`, `timeline.py:118-139`). A tiles/s speed would triple on
the 1.0 m/tile test levels. Per-tick step in tiles:
`step_q = mul_q16(mul_q16(speed_mps_q, inv_tile_q), dt_q)`.

**D12 — Displacements round sign-symmetrically.** Sweep offsets and moves use
`narrow_round_signed(mul_wide(trig, len))` (the kit's non-conserved-deposit
idiom, `fixed_point.h:146-151`) — `mul_q16`'s floor would bias every walk by
−0.5 raw count per axis per tick (a population drift toward −x/−y). Derived
constants (`step_q`, `p`, all non-negative products) use `mul_q16`.

**D13 — The Philox key is the match seed's SeedSequence entropy.**
`match_key(rng) = (e & 0xFFFFFFFF, (e >> 32) & 0xFFFFFFFF)` with
`e = rng.bit_generator.seed_seq.entropy` — for an int seed `e == seed`
(verified NumPy 2.4.6: `default_rng(12345)` → 12345); for `seed=None` it is the
OS entropy `sim.rng` itself used (a non-replayable run stays self-consistent).
Draws nothing from `sim.rng`. Bits ≥ 64 are dropped (collision only between
seeds equal mod 2⁶⁴).

**D14 — Dormancy = "nothing present".** `_run_swarm` returns before any
transfer or launch unless `swarm.swarm_present(gmap)`. The launch extent is
`max_e high_water[e]`; per-env masking happens in-kernel (§A rule 4 — the host
decides only whether there is anything at all to do).

---

## 3. The exact per-tick integer sequence (THIS is the twin spec)

All values are the store's native types (ROSTER, `swarm.py:64-74`). "Q16" =
int32 Q16.16. `FP_ONE = 65536`. Arithmetic in int64 wherever stated.

### 3.1 Constants and twins (`cpp/src/swarm.h`, namespace `swarm`)

| C++ (`swarm.h` / `philox_streams.h`) | Value | Python source (identity-tested) |
|---|---|---|
| `AI_DEAD, AI_RUN, AI_EAT, AI_COMA` | 0, 1, 2, 3 | `swarm.AI_*` |
| `FIRST_UNIT_ID` | 1 | `swarm.FIRST_UNIT_ID` |
| `PI_Q16` | 205887, `static_assert(PI_Q16 == ((fixedpoint::PI_Q30 + (1 << 13)) >> 14))` | `swarm_fixed.PI_Q16` |
| `HEADING_PERIOD_Q16` | `2 * (int64_t)PI_Q16` = 411774 | `swarm_fixed.HEADING_PERIOD_Q16` |
| roster C types | `int32_t` ×7, `uint32_t dist_walked`, `uint8_t valid`, with `static_assert(sizeof(...))` | `ROSTER` dtypes |
| `ENV_TICK, ENV_SEED_LO, ENV_SEED_HI, ENV_DT_Q, ENV_INV_TILE_Q, ENV_WORDS` | 0,1,2,3,4,5 | `swarm.ENV_*` |
| `DEATH_UNIT_ID, DEATH_POS_X, DEATH_POS_Y, DEATH_HEADING, DEATH_WORDS` | 0,1,2,3,4 | `swarm.DEATH_*` |
| `philox_streams::SALT_RESERVED_NONE` | 0 (never drawn) | `philox_streams.StreamSalt.RESERVED_NONE` |
| `philox_streams::SALT_SWARM_TUMBLE` | 1 | `StreamSalt.SWARM_TUMBLE` |
| `philox_streams::SALT_SWARM_WALL_TURN` | 2 | `StreamSalt.SWARM_WALL_TURN` |
| `swarm::SWEEP_DIST_Q16` (in `swarm_larva.h`) | `fixedpoint::FP_ONE` | `swarm.LARVA_SWEEP_DIST_Q16` |

**Heading wrap (C++ twin of `swarm_fixed.wrap_heading_q16`, floor-mod in
int64 because C++ `%` truncates):**

```cpp
FP_HD inline int32_t wrap_heading_q16(int64_t h) {
    int64_t r = ((int64_t)PI_Q16 - h) % HEADING_PERIOD_Q16;
    if (r < 0) r += HEADING_PERIOD_Q16;
    return (int32_t)((int64_t)PI_Q16 - r);
}
```

Draw with `philox32.TWO_PI_Q16 = 411775` (not used in P3 — no full-circle
draw), wrap with 411774; never unify (CLAUDE.md "Swarm heading law").

### 3.2 Field readers — E1 picks ONE

`tile_of(x_q, y_q, h, w, &idx)`: `tx = x_q >> 16`, `ty = y_q >> 16`
(arithmetic shift on int32 = floor, so −0.5 → −1); returns
`0 <= tx < w && 0 <= ty < h`, sets `idx = ty * w + tx`. **Never reads.**

- **RAW (the locked ruling):**
  `felt_T(x, y) = T[ (y >> 16) * w + (x >> 16) ]` — called only for in-grid
  points.
- **FOOTPRINT (recommended, E1):**
  ```
  vx = (x + (FP_ONE >> 1)) >> 16          // nearest grid vertex; x >= 0 for every in-grid point
  vy = (y + (FP_ONE >> 1)) >> 16
  x0 = clamp(vx - 1, 0, w - 1); x1 = clamp(vx, 0, w - 1)
  y0 = clamp(vy - 1, 0, h - 1); y1 = clamp(vy, 0, h - 1)
  s  = (int64)T[y0*w+x0] + T[y0*w+x1] + T[y1*w+x0] + T[y1*w+x1]
  felt_T = (int32) fixedpoint::floordiv_q(s, 4)
  ```
  (solid tiles in the window contribute their own temperature — contact.)

The reader is a functor type `R` with `FP_HD int32_t operator()(int32_t x_q,
int32_t y_q) const`, so the sampler below is a template over it (P6 passes a
raw food reader without touching the sampler).

### 3.3 Shared kit functions (`swarm.h`)

```cpp
enum : int32_t { SIDE_MINUS = -1, SIDE_NONE = 0, SIDE_PLUS = 1 };

struct SweepSample {           // one head sweep
    int32_t here;              // R(px, py)
    bool    open_plus, open_minus;
    int32_t val_plus, val_minus;   // R(point) iff open, else 0 (never read)
};

// PLUS side = heading + angle, MINUS = heading - angle. int32 adds: |heading|
// <= PI_Q16 and angle <= PI_Q16/2 keep |a| < 1.5*PI_Q16 (inside the trig
// kit's pinned 4*pi accuracy band).
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
    uint32_t span = (uint32_t)(turn_max_q - turn_min_q) + 1u;   // >= 1 by SPECIES_RULES
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

**No-tunnelling proof.** `|cos_q16|, |sin_q16| <= FP_ONE` (kit contract,
`fixed_point.h:987-988`), so `|dx|, |dy| <= step_q < FP_ONE` (SPECIES_RULES
R1). The target tile differs from the own tile by at most one per axis: it is
the own tile or one of its 8 neighbours, and it is the tile probed. A move can
never skip a tile. Corner-cutting between two orthogonally solid tiles is
accepted (chapter §4.1 step 3). A larva on an open tile therefore only ever
enters open tiles; it can stand on a solid tile only if the tile became solid
under it (§0 gap).

### 3.4 Derived per-(species, env) constants (FP_HD `larva_derive`; Python twin `swarm.larva_derived`)

```
v_q       = mul_q16(speed_mps_q, inv_tile_q)          // tiles per second
step_q    = mul_q16(v_q, dt_q)                        // tiles per tick
p_base_q  = mul_q16(tumble_rate_base_q,  dt_q)        // per-tick probability, Q16 of 1
p_alarm_q = mul_q16(tumble_rate_alarm_q, dt_q)
```

`mul_q16` = `(int32)(((int64)a * b) >> 16)`. The Python twin is
`(a * b) >> 16` on Python ints (floor = SAR). They agree whenever no `mul_q16`
result exceeds int32, which SPECIES_RULES R1/R2 guarantee on the exact launch
inputs (R1 checks `v_q <= INT32_MAX` in Python's unbounded ints).

### 3.5 Per-slot law — `swarm::larva_step_slot` (env `e`, slot `s < high_water[e]`)

Inputs: the store view at flat index `k = e * max_units + s`; `T_e = T + e*h*w`;
`solid_e = solid + e*h*w`; `env = env_params + e*ENV_WORDS` (uint32 words;
`dt_q = (int32_t)env[ENV_DT_Q]`, `inv_tile_q = (int32_t)env[ENV_INV_TILE_Q]`,
both in `[1, INT32_MAX]` by `pack_env_params`); the species params `P` (by
value, §5.4); `D = larva_derive(P, dt_q, inv_tile_q)` (per slot or hoisted per
env — identical ints either way); the reader `R` bound to `T_e` (E1). Output
also `death = death_rec + k*DEATH_WORDS`.

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
    d_here = |(int64)sw.here - P.T_prefer_q|
    d_plus = |(int64)sw.val_plus - P.T_prefer_q|      (meaningful iff sw.open_plus)
    d_minus= |(int64)sw.val_minus - P.T_prefer_q|     (meaningful iff sw.open_minus)
    alarm  = sw.open_plus && sw.open_minus && d_plus > d_here && d_minus > d_here
    pref   = side_preference(sw.open_plus, d_plus, sw.open_minus, d_minus)
L5  philox32_draw(env[SEED_LO], env[SEED_HI], (uint32)uid, env[TICK],
                  /*draw_index*/ 0, SALT_SWARM_TUMBLE, Wt)
    p = alarm ? D.p_alarm_q : D.p_base_q                 // in [0, 65536] by R2
    if philox32_uniform_q16(Wt[0]) < (uint32)p:
        h_new = turn_heading(h, pref, Wt[1], Wt[2], P.turn_bias_q, P.turn_min_q, P.turn_max_q)
L6  pr = probe_move(solid_e, h_, w_, px, py, h, D.step_q)       // moves along the OLD heading (D7)
    if pr.blocked:
        philox32_draw(env[SEED_LO], env[SEED_HI], (uint32)uid, env[TICK], 0, SALT_SWARM_WALL_TURN, Ww)
        h_new = turn_heading(h, pref, Ww[0], Ww[1], P.turn_bias_q, P.turn_min_q, P.turn_max_q)   // D6: overrides
    else:
        px = pr.nx; py = pr.ny; moved = true
L7  if R(px, py) > P.T_lethal_hot_q:                   // post-move (D5); every state
        death = { uid, px, py, h }                     // h = the stored (pre-update) heading
        pos_x[k] = pos_y[k] = heading[k] = hp[k] = ai_state[k] = unit_id[k] = faction[k] = 0
        dist_walked[k] = 0; valid[k] = 0               // the P2 empty-slot form (AI_DEAD == 0)
        return
    death[0..3] = 0
L8  pos_x[k] = px; pos_y[k] = py
    if moved: dist_walked[k] = (uint32)(dist_walked[k] + (uint32)D.step_q)    // wraps mod 2^32
    heading[k] = h_new                                  // canonical: stored value or a wrap() output
    ai_state[k] = st_new
```

(`h_`, `w_` = grid height/width; the store column `h` is the heading.)
`hp[k]` and `faction[k]` are written only by L7.

**Draw table** — every Philox block P3 takes (counter = `(tick, draw_index,
salt, unit_id)`, key = `(seed_lo, seed_hi)`; `draw_index` is never stored and
every block below is `draw_index 0` of its salt; a later need for more words
of the same purpose takes `draw_index 1`, never reorders these):

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

### 3.6 The two loops

- **CPU (`swarm_larva.cpp::swarm_larva_step_cpu`):** `for e in [0, n_env): for
  s in [0, high_water[e]): larva_step_slot(...)`; returns the number of deaths
  (counted in the loop).
- **CUDA (`swarm_larva.cu`):** `__global__ larva_kernel(...)`: `i = blockIdx.x
  * blockDim.x + threadIdx.x; e = i / launch_hw; s = i % launch_hw; if (e >=
  n_env || s >= d_high_water[e]) return; larva_step_slot(...)`. Block size 256;
  `launch_hw = max_e high_water[e]` from the host mirror. Death count = a host
  scan of the downloaded records (no device atomic).
- Both write only their own slot's 9 columns and 4 record words; `T`, `solid`,
  `high_water`, `env` are read-only. Launch geometry never reaches a result.

---

## 4. Placement, input snapshot, dormancy, residency

### 4.1 Where it runs

At the **end of the slot-7 chain on every backend**, as the last physics
stage, in `PhysicsRunner` (V1):

- `step()`: after `destroyed = self.engine.step_tail(...)` and its comment
  blocks, immediately before `return destroyed` (`physics_runner.py:974`):
  `self._run_swarm(gmap, sim_time, philox)`.
- `_step_resident()`: after the tail bracket, immediately before `return
  destroyed` (`:1351`): the same call.
- Dispatch line (`:772`) forwards `philox=philox`.

`Simulation.step` gets no new conductor slot: the swarm is part of slot 7. It
passes the Philox clock into the existing physics call and turns the runner's
death records into tick events on the next line (§7).

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
- **Philox clock:** `PhiloxClock(seed_lo, seed_hi, tick_abs)` with
  `tick_abs = sim.round_index * sim.ticks_per_round + sim.round_tick` (V5),
  masked to uint32 after a range check (`< 2**32`, ≈ 5.6 years at 24 tps).
- Identical on both backends by construction: the resident path uploads
  exactly those mirror values before launching (§4.3).

### 4.3 `PhysicsRunner._run_swarm(gmap, sim_time, philox)` — exact behaviour

New method appended at the END of the class (after `_eos_t_amb_raw`,
`:1733-1748`; fire-12's last hunk touches `:1738` only). Local imports inside
the method (no hunk in the import block fire-12 rewrites).

1. `self.swarm_deaths = []` (every tick, first statement — never stale).
2. If not `swarm.swarm_present(gmap)`: return. **Nothing else happens on a
   swarm-free map: no upload, no launch, no allocation** (D14).
3. If `philox is None`: raise `ValueError("PhysicsRunner.step: a swarm is
   present but no philox clock was passed — Simulation passes
   PhiloxClock(...); a direct caller must too")`.
4. `env = swarm.pack_env_params(gmap, sim_time, philox)` → `(N_ENV, 5)`
   uint32 `[tick, seed_lo, seed_hi, dt_q, inv_tile_q]`, with
   `dt_q = quantize_scalar(float(sim_time))` and
   `inv_tile_q = quantize_scalar(1.0 / float(gmap.tile_size_m))`. It re-runs the
   context rules (R1, R2) on these exact ints and raises `RuntimeError` naming
   the rule if one fails — unreachable unless `[clock] ticks_per_second` was
   edited mid-run (the load/reload check used the config's tps).
5. `on_cuda = get_swarm_backend()` if the binding exists else `False` (the
   `_combustion_on_cuda` idiom, looked up here, not in `__init__`);
   `resident = gmap.residency_on() and on_cuda`.
6. If `resident`: `gmap.from_host(["temperature"])` — once, before any
   species (F5: the tail wrote it on the mirror).
7. For each `sp` in `SWARM_SPECIES` (a table walk, no if-chain):
   `hw = getattr(gmap, swarm.high_water_attr(sp))` (the `(N_ENV,)` host
   mirror — host-authoritative, the kernel never writes it); if
   `int(hw.max()) == 0` continue; `launch_hw = int(hw.max())`.
   `kern = swarm.SPECIES_KERNELS[sp]`;
   `kw = swarm.kernel_kwargs(gmap.swarm_species.row(sp), kern)`;
   `death = self._swarm_death_buffer(sp, gmap)` (lazy `(N_ENV, max_units, 4)`
   int32, re-allocated when the shape changes — §A rule 5 key).
   - **resident:** if `gmap.swarm_host_dirty[sp]`:
     `gmap.from_host(swarm.upload_names(sp))` (the 9 columns + that species'
     `high_water` + `swarm_next_unit_id`), then
     `gmap.swarm_host_dirty[sp] = False` (R9). Then
     `bp.<kern.resident>(dev pointers of the 9 columns + high_water +
     temperature + solid, n_env, max_units, h, w, launch_hw, env, death, **kw)`,
     then `gmap.to_host(swarm.column_names(sp))` (**the 9 columns only** —
     `high_water`/`next_unit_id` are host-authoritative; the kernel never
     writes them).
   - **host (CPU twin or per-call CUDA):** `fn = bp.<kern.cuda>` if `on_cuda`
     else `bp.<kern.cpu>`; `fn(9 column arrays, hw, gmap.temperature[None],
     gmap.solid[None], env, death, **kw)`; then
     `gmap.swarm_host_dirty[sp] = True` (a host-side store write, R9).
   - `self.swarm_deaths.extend(swarm.death_events(sp, death, hw))`.
   - `self.swarm_launches = getattr(self, "swarm_launches", 0) + 1`
     (telemetry for the dormancy test).

The resident kernel reads the device `solid` uploaded at step 4
(`physics_runner.py:1242-1244`); nothing between step 4 and this launch writes
`solid` (EOS, traces, combustion, sky exchange and the tail do not).

**§A habits.** Every new entry takes `(N, …)`-shaped arrays (`T`/`solid` as
`(N, h, w)` via `[None]` views today; store `(N, M)`; `hw (N,)`; env `(N, 5)`;
records `(N, M, 4)`); per-env scalars are the env block (rule 1); orchestration
is C++ except the backend choice (the `_run_combustion` precedent; rule 2);
no mirror-only field (the store is resident since P2; rule 3); per-env masks
in-kernel (rule 4); scratch keyed by `(N, M)` (rule 5).

### 4.4 Resident device scratch

`swarm_larva.cu` owns a file-static `SwarmResidentScratch { uint32_t* d_env;
int32_t* d_death; size_t env_cap, death_cap; }` grown on demand (the
`EOSResidentScratch` precedent, `cuda_eos_resident.cu:552-665`). The resident
launch copies `env` H2D (20·N bytes), launches, then copies each env's
`launch_hw` death records D2H into the host `death` array (the host compaction
reads only `[0, hw[e])`).

---

## 5. Species table: rows, units, rules, kernels

### 5.1 New `SPECIES_FIELDS` rows (only what P3 consumes; all REQUIRED, all `hot`)

| key | `Field` | unit | stored | consumer |
|---|---|---|---|---|
| `speed_mps` | `KIND_REAL_Q16, minimum=2**-16, maximum=100.0` | m/s | `speed_mps_q` | `larva_derive` |
| `T_prefer` | `KIND_REAL_Q16, minimum=1.0, maximum=2000.0` | **kelvin** | `T_prefer_q` (game ΔT) | comfort |
| `T_lethal_hot` | same | **kelvin** | `T_lethal_hot_q` | L7 |
| `T_ctmin` | same | **kelvin** | `T_ctmin_q` | L3 entry |
| `T_coma_exit` | same | **kelvin** | `T_coma_exit_q` | L3 exit |
| `tumble_rate_base` | `KIND_REAL_Q16, minimum=0.0, maximum=1000.0` | 1/s | `tumble_rate_base_q` | L5 |
| `tumble_rate_alarm` | same | 1/s | `tumble_rate_alarm_q` | L5 |
| `turn_bias` | `KIND_REAL_Q16, minimum=0.0, maximum=1.0` | probability | `turn_bias_q` | `turn_heading` |
| `turn_min` | `KIND_REAL_Q16, minimum=0.0, maximum=math.pi` | rad | `turn_min_q` | `turn_heading` |
| `turn_max` | same | rad | `turn_max_q` | `turn_heading` |
| `sweep_angle` | `KIND_REAL_Q16, minimum=2**-16, maximum=math.pi / 2` | rad | `sweep_angle_q` | L4 |

`math.pi` quantizes to 205887 = `PI_Q16`; `math.pi/2` to 102944. Not added:
`radius` (nothing reads it), `T_lethal` (ruling 6), food rows (P6). The
chapter's `tumble_thermal_bias`/`heat_damage_coeff` are superseded by
`tumble_rate_alarm`/`turn_bias` (ruling 2) and the threshold kill (ruling 4).
`SPECIES_RELOAD` gains all eleven as `"hot"`; `SwarmSpeciesRow` gains the
eleven `<name>_q` int fields (the P2 import-time assertion enforces the match).

**Known limit (not P3's):** `SPECIES_FIELDS` is one tuple shared by every
species (P2 design), so these larva rows would be required of a second species
too. The second-species patch splits it into shared + per-species field sets.

### 5.2 Kelvin authoring — `SPECIES_UNITS`

A separate mapping beside `SPECIES_RELOAD` (the R5 precedent — a per-field
authoring property is a mapping, not a new schema kind):

```python
SPECIES_UNITS: dict = {"T_prefer": "kelvin", "T_lethal_hot": "kelvin",
                       "T_ctmin": "kelvin", "T_coma_exit": "kelvin"}
```

Import-time assertions: keys ⊆ `SPECIES_FIELDS` names; values ∈ `{"kelvin"}`;
every kelvin field is `KIND_REAL_Q16`. (`schema.py` is untouched; T22 covers
the rows as ordinary `KIND_REAL_Q16` fields for free.)

**`SwarmSpeciesTable.from_config(cfg)` becomes two-pass** (the order is
load-bearing for T22's bare-`Namespace` configs):

1. **Validate** (unchanged P2 structure checks, then `field_value_error` for
   every field of every species) — all before any conversion.
2. **Convert/quantize:** `KIND_INT` → `int`; `KIND_REAL_Q16` → if
   `SPECIES_UNITS.get(name) == "kelvin"`:
   `q = swarm_fixed.quantize_scalar(ts.from_kelvin(float(value)))` with
   `ts = temperature_scale.load(cfg)` loaded once (lazily, on the first kelvin
   field); a `RuntimeError` from `temperature_scale.load` is re-raised as
   `ValueError("[swarm.<sp>].<key> = <v> K: cannot convert through
   [physics.temperature_scale]: <e>")` (so a Ctrl+R with a broken scale is
   rejected by the seam, never a crash); then `q` must lie in
   `[INT32_MIN, INT32_MAX]` else `ValueError` naming the key; otherwise
   `quantize_scalar(value)` as in P2.

`from_config` runs no cross-field rule (§5.3). Door 3: `from_kelvin` is
`(K − kelvin_ambient) / k` — algebraic, quantized at the write.

**Immunity to fire-12 (ruling 7), by construction:** the rows are Kelvin;
the stored game-ΔT integers are *derived* from the scale in force. Under
HEAD's k = 3 the defaults store T_prefer 221730, T_lethal_hot 658637,
T_ctmin −215177, T_coma_exit −171486; under fire-12's k = 1 they store 665190,
1975910, −645530, −514458 — the same Kelvin. The hash covers stored values
(`swarm.py:228-244`), so a scale change changes `.hash`.

**Reload through the ONE seam:** Ctrl+R → `CFG.reload()` →
`sim.on_config_reload()` → `reload_species(gmap)` → `from_config(CFG)` →
`temperature_scale.load(CFG)` reads the fresh CFG (no cache,
`temperature_scale.py:12-15`). A `[physics.temperature_scale]` edit therefore
re-derives the four rows, changes `.hash` (a "replay comparability ends here"
line prints), and the next launch reads the new ints (V2 — there is no device
copy to re-upload). Honest limit: other scale consumers (`gmap`'s cached
`gas_t_amb_raw`, the EOS binds) do not all follow a mid-run scale change; a
scale edit mid-match is not a supported play action.

### 5.3 `SPECIES_RULES` — invariants the kernel relies on (never tuning taste)

```python
class SpeciesContext(NamedTuple):
    dt_q: int          # quantize_scalar(tick seconds)
    inv_tile_q: int    # quantize_scalar(1 / tile_size_m)

def species_context(gmap, cfg=CFG) -> SpeciesContext:
    dt_q = swarm_fixed.quantize_scalar(1.0 / float(cfg.clock.ticks_per_second))
    inv_tile_q = swarm_fixed.quantize_scalar(1.0 / float(gmap.tile_size_m))
    # both must be in [1, INT32_MAX], else ValueError

class SpeciesRule(NamedTuple):
    name: str
    check: Callable   # (row, ctx) -> str | None   (None = holds)

SPECIES_RULES = (R1, R2, R3, R4)
def check_species_rules(table, ctx) -> list[str]   # "[swarm.<sp>] <rule>: <message>" per violation
```

| rule | holds iff | protects |
|---|---|---|
| **R1 `no_tunnelling`** (ruling 9's first rule) | `v_q = (speed_mps_q * inv_tile_q) >> 16 <= INT32_MAX` and `1 <= step_q = (v_q * dt_q) >> 16 < FP_ONE` | §3.3's proof (`speed·Δt < 1 tile`); `mul_q16` exactness (§3.4); RUN means moving |
| **R2 `tumble_probability`** | `p_base_q <= p_alarm_q <= FP_ONE` | the L5 compare is a probability; "raised to" alarm (ruling 2) |
| **R3 `coma_hysteresis`** | `T_coma_exit_q >= T_ctmin_q + 1` | chapter §2 "≥ 1 count against boundary flapping" |
| **R4 `turn_range`** | `turn_min_q <= turn_max_q` | `span >= 1` in `philox32_below` |

Callers (the only two installers):

- `allocate_swarm_store(gmap, cfg)`: after `from_config`, `errs =
  check_species_rules(table, species_context(gmap, cfg))`; any error →
  `ValueError("; ".join(errs))` (**hard error at load**, ruling 9).
- `reload_species(gmap, cfg)`: after P2's step 2 (load-class merge) and before
  step 3 (install): any error → print `[swarm] RELOAD REJECTED: <errs[0]>`,
  keep the table in force, return `False` (the P2 rejection contract).

`GameMap.__init__` sets `tile_size_m` (`gamemap.py:666`) before
`allocate_swarm_store(self)` (`:845`), so the context exists at load.

With the defaults on a 0.333 m level at 24 tps: `speed_mps_q = 13107`,
`inv_tile_q = 196805`, `dt_q = 2731`, `v_q = 39360`, `step_q = 1640`
(0.0250 tile/tick, 40 ticks/tile); on 1.0 m tiles `step_q = 546`.
`p_base_q = 273` (0.1 /s), `p_alarm_q = 2731` (1.0 /s). All rules hold.

### 5.4 `SPECIES_KERNELS` and the kernel parameter block

```python
class SpeciesKernel(NamedTuple):
    cpu: str; cuda: str; resident: str     # breach_physics attribute names
    params: tuple                           # SwarmSpeciesRow attrs, == the pybind kwarg names

LARVA_KERNEL_PARAMS = ("speed_mps_q", "T_prefer_q", "T_lethal_hot_q", "T_ctmin_q",
                       "T_coma_exit_q", "tumble_rate_base_q", "tumble_rate_alarm_q",
                       "turn_bias_q", "turn_min_q", "turn_max_q", "sweep_angle_q")
SPECIES_KERNELS = {"larva": SpeciesKernel("swarm_larva_step", "cuda_swarm_larva_step",
                                          "swarm_larva_step_resident", LARVA_KERNEL_PARAMS)}
def kernel_kwargs(row, kern) -> dict: return {a: int(getattr(row, a)) for a in kern.params}
```

Import-time assertions: `set(SPECIES_KERNELS) == set(SWARM_SPECIES)`; every
`params` entry is a `SwarmSpeciesRow` field. A species is a `SWARM_SPECIES`
entry + a row + a kernel entry here — never an if-chain.

The C++ POD (by value, V2):

```cpp
struct SwarmLarvaParams {
    int32_t speed_mps_q, T_prefer_q, T_lethal_hot_q, T_ctmin_q, T_coma_exit_q,
            tumble_rate_base_q, tumble_rate_alarm_q, turn_bias_q,
            turn_min_q, turn_max_q, sweep_angle_q;
};
```

### 5.5 `config.toml` — appended to `[swarm.larva]` (after `hp_max`, EOF, `:2451`)

```toml
# --- arc #63 P3: the larva law (docs/swarm_P3_impl.md). All HOT (Ctrl+R ->
# Simulation.on_config_reload, applied from the next tick). Temperatures are
# KELVIN, converted through [physics.temperature_scale] at load (immune to the
# k_temp_to_kelvin scale). Rates are per SECOND. Speed is metres per second.
speed_mps         = 0.2      # PLACEHOLDER (P4). 1/3 body length/s for the 0.6 m larva (Drosophila crawls ~0.25-0.5 BL/s); 40 ticks per 0.333 m tile at 24 tps
T_prefer          = 303.15   # 30 C (ruling 3): comfort = |T - T_prefer|
T_lethal_hot      = 323.15   # 50 C (ruling 4): instant death above, post-move, any state
T_ctmin           = 283.15   # 10 C (ruling 5): chill-coma onset (enter below)
T_coma_exit       = 285.15   # 12 C (ruling 5): recovery (exit above); 2 K hysteresis
tumble_rate_base  = 0.1      # PLACEHOLDER (P4). 1/s: mean run 10 s when no sweep side is worse (larval runs ~5-20 s)
tumble_rate_alarm = 1.0      # PLACEHOLDER (P4). 1/s: mean run 1 s when BOTH sweep points are worse than here
turn_bias         = 0.8      # PLACEHOLDER (P4). P(turn toward the better side); Luo 2010: turns biased toward favourable
turn_min          = 0.5236   # PLACEHOLDER (P4). rad (30 deg): smallest tumble/wall turn
turn_max          = 2.0944   # PLACEHOLDER (P4). rad (120 deg): largest turn (larval turns span ~30-120 deg)
sweep_angle       = 0.7854   # PLACEHOLDER (P4). rad (45 deg): sweep points one tile ahead at +-45 deg (the diagonal-ahead tiles)
```

---

## 6. Philox stream salts + the match key (breach-wide, V3)

**`src/simulation/philox_streams.py`** (stdlib + nothing heavy; scanned by the
ingress lint):

```python
class StreamSalt(IntEnum):          # THE one breach-wide enum (engine/14 door-4 amendment).
    RESERVED_NONE   = 0             # APPEND-ONLY: a value is never reused or renumbered;
    SWARM_TUMBLE    = 1             # consumers of different id spaces (CPU units, swarm
    SWARM_WALL_TURN = 2             # units) share the key, so they must never share a salt.

class PhiloxClock(NamedTuple):
    seed_lo: int; seed_hi: int; tick: int

def match_key(rng) -> tuple[int, int]      # D13; TypeError if the entropy is not an int
```

Import-time assertion: values unique. **`cpp/src/philox_streams.h`**: the same
three `constexpr uint32_t` constants, namespace `philox_streams`. Identity test:
`bp.PHILOX_SALTS == {s.name: int(s) for s in StreamSalt}`.

`Simulation._reset_internal` adds, after `self.swarm = SwarmUnits(self.gmap)`
(`simulation.py:275`): `self._philox_key = match_key(self.rng)` — reads the
SeedSequence, draws nothing (T18-style canary in §11).

---

## 7. The death event (ruling 8)

`src/simulation/events.py` gains (appended before `__all__`, and to it):

```python
@dataclass
class SwarmUnitDiedEvent:
    """A swarm unit died this tick (arc #63 P3, engine/17 §7). Carries the
    PRE-ZERO values of its slot — the post-move position and the heading it
    moved along — because the slot itself is already the all-zero freed form.
    The render husk/decal (P5) is driven by this event only, never by the
    freed slot. Emitted in (SWARM_SPECIES order, env 0, ascending slot) order.
    Not in the synced event digest (field_ab_harness._SYNCED_EVENT_TYPES is
    unchanged): its content is gated CPU==GPU by tests/cuda_swarm_check.py and
    cross-machine by the swarm xarch lines."""
    species: str
    unit_id: int
    pos_x_q: int      # Q16.16 tile units, raw
    pos_y_q: int
    heading_q: int    # Q16.16 radians, canonical (-PI_Q16, PI_Q16]
```

- **Device side:** the per-slot record (§3.5 L0/L7) — every slot in
  `[0, hw)` writes its 4 words every launch (zeros = no death), so a stale
  record from an earlier tick is never read. No atomic append anywhere.
- **Host compaction** (`swarm.death_events(sp, death_rec, hw)`): after an
  explicit `N_ENV == 1` raise (the carrier idiom), `slots =
  np.flatnonzero(death_rec[0, :hw0, DEATH_UNIT_ID])` (ascending = slot order),
  one event per slot with `int()`-converted words.
- **Into the tick:** `Simulation.step`, on the line after the physics call:
  `self.tick_events.extend(self.physics_runner.swarm_deaths)` — slot 7,
  before 9's `WallDestroyedEvent`s.
- **Death and slot reuse:** the empty-slot form makes the slot the lowest free
  one for the next spawn (P2 §2.2); `high_water` and `next_unit_id` are
  unchanged by death, so a reused slot gets a new `unit_id`.

---

## 8. Coupling rows + the 9c comment

**Fork 7b's "new coupling row, documented beside the unit row"** — honestly
represented as a **sibling table** in `src/simulation/exchange.py`, appended
after `COUPLING_TABLE` (`:759`), because (F8) the unit table's future executor
iterates it over CPU units and its tests pin its count:

```python
@dataclass(frozen=True)
class SwarmCouplingRow:
    """One physics->swarm-unit coupling (engine/17 fork 7b), documented BESIDE
    the unit rows and EXECUTED in-kernel (the swarm step, last stage of slot
    7). It has no Python response: the law is cpp/src/swarm_larva.h, its
    numbers are the species-table fields named in `reads`."""
    field: str        # GameMap field the law samples
    reduction: str    # a REDUCTIONS name: "center" (RAW, own tile) or "mean" (FOOTPRINT, E1)
    species: str      # a SWARM_SPECIES name
    reads: tuple      # SPECIES_FIELDS names the law's thresholds come from
    note: str

SWARM_COUPLING_TABLE: tuple = (
    SwarmCouplingRow("temperature", <E1>, "larva", ("T_lethal_hot",),
        "Contact/ambient heat kill: felt T > T_lethal_hot -> DEAD, post-move, any state. "
        "Deliberately NOT the unit row's radiant heat|max physics: a marine and a larva "
        "in the same hot room are damaged by different laws."),
    SwarmCouplingRow("temperature", <E1>, "larva", ("T_ctmin", "T_coma_exit"),
        "Chill coma with hysteresis: enter below T_ctmin, exit above T_coma_exit."),
)
```

Plus `"SwarmCouplingRow", "SWARM_COUPLING_TABLE"` in `__all__`. No new
plumbing: the rows are data that a test cross-checks against the species table
(§11 T-P3-22), so they cannot drift from what the kernel reads.

**The 9c comment amendment** (`simulation.py:1458-1468`, chapter §5): append
one line to the block, before `if self.physics_runner is not None:`:

```python
        # Swarm units (arc #63 P3, exchange.SWARM_COUPLING_TABLE) were already judged IN slot 7 (the swarm step is physics' last stage), so within a tick larva heat/coma happen BEFORE this unit row.
```

(Split over two comment lines if the implementer's line length demands it —
still one sentence, still in that block.)

---

## 9. Files touched + merge-friendliness vs fire-12

Checked with read-only `git diff -U0 $(git merge-base HEAD fire-12) fire-12`.
For `physics_runner.py`, `physics_engine.*`, `CMakeLists.txt`,
`test_no_float_in_sim_tu.py`, `run_on_cuda.py`, `_xarch_perfield_digest.py`,
`xarch_digest.py`, `exchange.py`, `events.py` HEAD equals the merge-base, so
line numbers are valid on both.

| file | fire-12 hunks (merge-base) | P3 footprint | overlap? |
|---|---|---|---|
| `cpp/src/swarm.h`, `swarm_larva.h/.cpp/.cu`, `philox_streams.h` | — | new | — |
| `src/simulation/philox_streams.py` | — | new | — |
| `cpp/src/physics_engine.{h,cpp}` | many | **none** (V1) | — |
| `src/simulation/physics_runner.py` | 28, 89-117, 149-…, 774, 911, 934-943, 998, 1018, 1078, 1160, 1319-1325, 1391, 1454-1624, 1738 | def lines `:751`, `:1144` (+`philox=None`); dispatch `:772`; one call before `:974`; one call before `:1351`; `_run_swarm` + `_swarm_death_buffer` appended after `:1748` | No (each ≥ 1 unchanged line from a fire-12 hunk) |
| `src/simulation/simulation.py` | 1616 (+9) | import after `:101`; 1 line after `:275`; the physics call `:1427-1428` + 1 line; 9c comment `:1458-1468` | No |
| `src/simulation/swarm.py` | — (P2 file) | §5, §7 | — |
| `src/simulation/events.py` | — | dataclass + `__all__` | No |
| `src/simulation/exchange.py` | 296-317 | appended after `:759`; `__all__` `:766-768` | No |
| `cpp/CMakeLists.txt` | +2 after 86; 95; 105; +7 after 187 | `src/swarm_larva.cpp` inserted after `:88` (`src/bindings.cpp`); a NEW `list(APPEND BREACH_SOURCES src/swarm_larva.cu)` line immediately before the `endif()` at `:109`; a NEW `set_source_files_properties(src/swarm_larva.cpp PROPERTIES COMPILE_OPTIONS "${BREACH_FP_STRICT}")` statement after `:188` | No (never edits fire-12's lines 86/105/187) |
| `cpp/src/bindings.cpp` | many; quiet merge-base 1067-1574 | `#include "swarm_larva.h"` after HEAD `:17`; the §10 block after HEAD `:1493` (`PHILOX32_TWO_PI_Q16`), which is inside the quiet zone | No |
| `stubs/breach_physics.pyi` | regenerated | regenerated (pybind11-stubgen, CPU build) | expected regen conflict → resolve by regenerating after the merge |
| `config.toml` | … | rows appended to `[swarm.larva]` at EOF | No |
| `tests/test_no_float_in_sim_tu.py` | +2 after 55; +22 after 326; 330; +2 after 333 | after `:334` (closing `}` of BASELINE): `SIM_TUS = SIM_TUS + ("swarm_larva.cpp",)`, `BASELINE["swarm_larva.cpp"] = {"float": 0, "double": 0, "fp:fast": 0}`; after `:344`: `MIGRATED_FLOOR_TUS = MIGRATED_FLOOR_TUS + ("swarm_larva.cpp",)` | No |
| `tools/run_on_cuda.py` | 26-29, 84-86, 97 | `"set_swarm_backend",` appended after `:107` | No |
| `tests/field_ab_harness.py` | 87 (+5), 511 | `swarm_thermal_scenario_sim` after `swarm_scenario_sim` (HEAD `:174`) | No |
| `tests/_xarch_perfield_digest.py` | 225 only | `build_swarm_lines` after `:303`; a block in `main()` before `:373` | No |
| `tests/xarch_digest.py` | — | `--scenario` + suffix | No |
| `CLAUDE.md` | 3-4, 27-31, 69-70, 76, 90-91, 115 | new rows appended after the "Swarm heading law" row (HEAD `:99`); edits only P1/P2 rows (`:93`, `:97`), which exist on this branch alone | No |
| `docs/architecture/engine/17_swarm_units.md` | — | one "As built (P3)" line under the §4 banner | — |
| `docs/architecture/engine/14_determinism_and_number_ingress.md` | — | one sentence in the door-4 amendment: "the enum is `simulation/philox_streams.py` + `cpp/src/philox_streams.h`" | — |

---

## 10. Bindings (`bindings.cpp`) + stub

After HEAD `:1493`, a block tagged `// arc #63 P3 (docs/swarm_P3_impl.md §10)`:

**Both builds (plain defs):**

- `swarm_larva_step(pos_x, pos_y, heading, hp, ai_state, dist_walked, unit_id,
  faction, valid, high_water, temperature, solid, env_params, death_rec, *,
  speed_mps_q, T_prefer_q, T_lethal_hot_q, T_ctmin_q, T_coma_exit_q,
  tumble_rate_base_q, tumble_rate_alarm_q, turn_bias_q, turn_min_q, turn_max_q,
  sweep_angle_q) -> int` — the CPU twin; returns the death count.
- `swarm_larva_derived(speed_mps_q, tumble_rate_base_q, tumble_rate_alarm_q,
  dt_q, inv_tile_q) -> tuple[int, int, int, int]` (`v_q, step_q, p_base_q,
  p_alarm_q`) — `larva_derive` exposed for the identity test.
- `swarm_wrap_heading_q16(h: int) -> int` — the C++ wrap (int64 in).
- `m.attr("SWARM_CONSTANTS")` = dict of every §3.1 C++ constant + roster type
  sizes; `m.attr("PHILOX_SALTS")` = dict name → value.

**CUDA build only (inside a new `#ifdef BREACH_HAS_CUDA … #endif` block):**

- `set_swarm_backend(use_cuda: bool)`, `get_swarm_backend() -> bool`.
- `cuda_swarm_larva_step(...)` — same signature as `swarm_larva_step`, per-call
  GPU (H2D all inputs, launch the resident core, D2H the 9 columns + records).
- `swarm_larva_step_resident(d_pos_x, …, d_valid, d_high_water, d_temperature,
  d_solid, n_env, max_units, h, w, launch_hw, env_params, death_rec, **params)
  -> int` — `uintptr_t` device pointers, host `env_params`/`death_rec`.
- `swarm_cuda_calls() -> int`, `swarm_resident_calls() -> int` — dispatch-fired
  telemetry (the `eos_step_cuda_calls` precedent) so the gate cannot pass on a
  silently-CPU run.

**Argument discipline (no silent copies).** Every array arg is checked for
exact dtype (the ROSTER dtype, `int32` T, `bool` solid, `uint32` env, `int32`
records), C-contiguity and the §4.3 shapes; a mismatch raises
`TypeError`/`ValueError`. `py::array_t` defaults to `forcecast`, which would
turn a mismatched IN/OUT column into a temporary copy whose writes are lost —
so the binding takes `py::array` and validates, or uses
`py::array_t<T, py::array::c_style>` plus an explicit `dtype().is(...)` check.

**Stub:** regenerate `stubs/breach_physics.pyi` with pybind11-stubgen against
`cpp/build/Release` (`docs/dev_setup.md:134-147`, the P1 precedent `e9e427f`);
the diff adds only the P3 symbols. The CUDA-only symbols stay absent (the
documented gap, `dev_setup.md:144-147`).

---

## 11. Verification plan (tests are written from this list)

**The standard.** Each test names the property it protects and the change that
must break it. Loops run over `ROSTER`, `SWARM_SPECIES`, `SPECIES_FIELDS`,
`SPECIES_RULES`, `StreamSalt`; nothing pins a set/count/order designed to
grow. A failing pre-existing test is a finding. Baseline: 4 pre-existing reds
(`test_bench_two_room` import, 2× `test_cool_shift_axis`, `test_p3_direct_e2e`
spray cone) — **no new reds**.

**Law tests call the real C++ twin through `bp.swarm_larva_step` on numpy
arrays** (synthetic `T`/`solid` arrays are test inputs, not GameMap fields —
the "Gas temperature is a mirror" rule governs `gmap.temperature`). Explicit
test params, not the config, unless the property is about the config.

**Fixture fix (F7): `tests/_swarm_testkit.py`** — captures, at import, each
species' repo row `dict(vars(getattr(CFG.swarm, sp)))`; exposes
`species_row(sp, **overrides)` and `swarm_cfg(**overrides)` (every species,
full rows; a FRESH dict per call — T22 mutates what it gets). The four P2
helpers (`test_swarm_species.py:43-55`, `test_swarm_digest.py:47-50`,
`test_swarm_recorder.py:43-46`, `test_swarm_store.py:46-50`) become thin
calls into it. Property: a P2 test's rows grow with `SPECIES_FIELDS`.

### `tests/test_swarm_larva_law.py` (CPU build)

| # | property | must break if… |
|---|---|---|
| T-P3-1 | **No tunnelling at max speed:** params with `step_q = FP_ONE − 1`; a grid with 1-thick walls (axis-aligned and a diagonal staircase) and larvae on both sides at many sub-tile offsets and all 64 headings `k·PI_Q16/32`; over 200 ticks no larva ever stands on a solid tile it did not start on, and no larva ever changes side of a wall line (tile-coordinate test, not a whole-grid mean). | the probe skips the target tile, uses the pre-move tile, or R1's bound is relaxed |
| T-P3-2 | **Out-of-grid = blocked:** a grid with NO solid tiles; larvae 1 step from each edge heading out; pos stays in `[0, w<<16) × [0, h<<16)` every tick, and each edge hit takes the wall-turn path (heading changes, pos unchanged that tick). | clamping instead of refusing, or an unchecked read |
| T-P3-3 | **Heading canonical after every write:** a 500-larva, 300-tick run with high tumble rates; after every tick every live heading ∈ `(−PI_Q16, PI_Q16]` and `check_invariants`-style checks pass on the arrays. | a raw `h ± m` store, an int32 wrap, `TWO_PI_Q16` as the period |
| T-P3-4 | **`dist_walked` wraps:** a mover with `dist_walked = 2**32 − 3` and `step_q = 10` reads 7 after one unblocked tick (uint32), and is unchanged after a blocked tick. | signed/overflowing accumulation, or accumulation on blocked ticks |
| T-P3-5 | **Coma hysteresis, both halves:** on a flat field set in turn to `T_ctmin−1`, each value in `[T_ctmin, T_coma_exit]`, `T_coma_exit+1`: RUN→COMA only below `T_ctmin`; COMA→RUN only above `T_coma_exit`; inside the band RUN stays RUN and COMA stays COMA; a COMA larva's pos/heading/dist are unchanged, and its columns are identical under two different seeds (no draw reaches it). | a single threshold, `<=` for `<`, sensing, drawing or moving in COMA |
| T-P3-6 | **Wake consumes the tick:** a COMA larva whose field rises above `T_coma_exit` becomes RUN with pos unchanged that tick and moves on the next. | waking and moving in one tick |
| T-P3-7 | **Kill is post-move and state-blind:** (a) a RUN larva one step from the boundary of a lethal region, heading in: dies on the tick it enters, record pos == the post-move pos; (b) heading out of a lethal tile into a safe one in one step: survives; (c) a COMA larva whose tile turns lethal dies. Record heading == the stored heading (the one it moved along), unit_id == its id. | pre-move sampling, a COMA exemption, a record from the zeroed slot |
| T-P3-8 | **Death writes the P2 empty form, records every slot:** after a kill, all 9 columns of the slot equal a never-spawned slot's; for every slot in `[0, hw)` without a death the record is 4 zeros (pre-fill the record array with garbage first); slots ≥ hw untouched. | leaving `valid=0` with other columns set; reading stale records |
| T-P3-9 | **Blocked sweep = no reading (D4):** a larva facing a wall head-on (both sweep points solid) with `here` in a flat field: over many ticks its tumble rate equals `p_base` (not `p_alarm`), measured as tumbles/ticks within a binomial 4σ band; a larva with one open and one blocked side turns toward the open side with frequency `turn_bias` (4σ) on wall hits. | blocked-as-worst, blocked-as-reading, side choice ignoring blockage |
| T-P3-10 | **Wall turn wins over tumble (D6):** with `p = FP_ONE` (always tumble) and a blocked probe, the written heading equals `turn_heading` evaluated on the WALL_TURN words (computed in the test from `bp.philox32_4x32_10` with `StreamSalt.SWARM_WALL_TURN`), not the TUMBLE words. | tumble overriding, or both applied |
| T-P3-11 | **Draw layout is the draw table:** for a RUN larva in open space, the tumble decision and the new heading equal the values computed in the test from `bp.philox32_4x32_10((tick, 0, SWARM_TUMBLE, uid), (lo, hi))` words 0/1/2 through `philox32_uniform_q16`/`philox32_below` and `swarm_fixed.wrap_heading_q16`. | a salt swap, a draw_index ≠ 0, `%` or float mapping, a word reorder |
| T-P3-12 | **Per-slot independence:** larva A's full trajectory (all 9 columns every tick) alone equals A's trajectory with 200 other larvae present in other slots (same uid, same slot index for A). | any cross-slot read, a shared counter, a draw keyed on slot or population |
| T-P3-13 | **Counter keys:** changing only `tick`, only `seed_lo`, only `seed_hi`, or only `unit_id` changes the tumble decision sequence of a larva over 500 ticks (≥ 1 difference each). | a counter/key word ignored |
| T-P3-14 | **Invalid and out-of-hw slots are inert:** random garbage in slots ≥ hw survives untouched; valid=0 slots stay all-zero; hw = 0 → nothing touched and 0 returned. | the loop bound is capacity, or empty slots are processed |

### `tests/test_swarm_larva_sensing.py` (CPU build; statistical, fixed seeds, thresholds derived from theory)

| # | property | must break if… |
|---|---|---|
| T-P3-15 | **Toward T_prefer from BOTH sides:** 64×64 open grid with a solid border; `T(x) = T_prefer_q + g·(x − 32)` (Q16, g chosen so all T stay inside `(T_coma_exit, T_lethal_hot)`); 400 larvae uniform on `x ∈ [4, 20]`, 400 on `[44, 60]`, headings uniform; after 2400 ticks the left group's mean x has increased and the right group's decreased, each by > 4 standard errors of the flat-field control's displacement. Test params: `turn_bias` 0.8, alarm 10× base. | the alarm/bias sign flipped, comfort = T instead of \|T − T_prefer\| (up-gradient only), sweep sides swapped |
| T-P3-16 | **Flat field is unbiased:** same grid, flat T; 800 larvae; mean Δx and mean Δy after 2400 ticks each within 4 standard errors of 0, and the tumble frequency within a binomial 4σ band of `p_base`. | a floor-biased displacement (D12), an alarm from geometry, a directional draw bias |
| T-P3-17 | **Ablation control:** with `turn_bias = 0.5` and alarm = base, T-P3-15's drift is gone (each group's mean displacement within 4 SE of 0) — so T-P3-15 measures the law, not the grid. | the drift coming from something other than the two dials |

### `tests/test_swarm_constants.py` (CPU build)

| # | property | must break if… |
|---|---|---|
| T-P3-18 | **Twins are identical:** every `bp.SWARM_CONSTANTS` entry equals its Python source (§3.1 table); `bp.PHILOX_SALTS == {s.name: int(s) for s in StreamSalt}`; roster type sizes equal the ROSTER dtypes' itemsizes. | a drifted constant, a renumbered salt |
| T-P3-19 | **Wrap identity:** `bp.swarm_wrap_heading_q16(h) == swarm_fixed.wrap_heading_q16(h)` for P2's T17 inputs (±PI_Q16, ±3·PI_Q16, 411774, 411775, 0, ±1, INT32_MIN, INT32_MAX, Philox endpoints) plus 10⁵ seeded int64 values in `[−2³³, 2³³]`. | a truncating `%`, an int32 intermediate |
| T-P3-20 | **Derived identity:** `bp.swarm_larva_derived(...) == swarm.larva_derived(...)` over the rule-satisfying region (seeded grid of speeds, rates, dt_q, inv_tile_q incl. the R1/R2 boundaries). | the Python rule twin and the kernel disagreeing |
| T-P3-21 | **Salt enum is append-only-safe:** values unique, `RESERVED_NONE == 0`, and no salt value is reused by two names. | a duplicated value |

### Species-table / coupling / tick integration (`tests/test_swarm_species.py` additions + `tests/test_swarm_tick.py`)

| # | property | must break if… |
|---|---|---|
| T-P3-22 | **Coupling rows cannot drift:** every `SWARM_COUPLING_TABLE` row's `species ∈ SWARM_SPECIES`, `reduction ∈ exchange.REDUCTIONS`, every `reads` name ∈ `SPECIES_FIELDS`. | a renamed field or species |
| T-P3-23 | **Kelvin conversion:** for every `SPECIES_UNITS` kelvin row, `to_kelvin(stored_q / 65536)` equals the authored Kelvin within `k/2¹⁷` K. | a missing `from_kelvin`, a double conversion, a unit applied to the wrong field |
| T-P3-24 | **Scale immunity + seam:** with `CFG.physics.temperature_scale.k_temp_to_kelvin` monkeypatched 3.0 → 1.0 (keeping `phi_exp·k == 1`), a fresh GameMap stores different game-ΔT ints for the kelvin rows but the same Kelvin (T-P3-23 holds), and `sim.on_config_reload()` on a running sim changes `.hash`, prints one "replay comparability ends here" line, and the next tick's launch uses the new ints (observable: a larva sitting at a T that is lethal under one scale and not the other). | cached conversion, conversion outside the seam, hashing authored text |
| T-P3-25 | **Rules at load and reload:** for every `SPECIES_RULES` entry, a config violating only that rule raises `ValueError` naming it at `GameMap(...)`, and through `sim.on_config_reload()` prints exactly one `RELOAD REJECTED` line and keeps the table object. R1 is exercised both by speed and by a small `tile_size_m`. | a rule not wired to both installers |
| T-P3-26 | **Swarm-free runs launch nothing and move no golden:** `capture_trajectory(default_scenario_sim, 30)` → `physics_runner.swarm_launches` absent/0, `swarm_deaths == []` every tick, `trajectory_digest == GOLDEN_AGGREGATE`; plus the P2 T1 set (w6_armory golden + canary, B6, vent/B1/B2) green with zero golden edits. | an unconditional launch, a field write, an RNG draw |
| T-P3-27 | **The Philox clock is monotone across round rewinds:** in a `TwoPhaseWEGO` sim run through two round boundaries, the `tick` word the runner packs (spy on `pack_env_params`) strictly increases by 1 per step while `sim.tick` rewinds; the key equals `match_key(sim.rng)` and `sim.rng.bit_generator.state` is unchanged by `match_key`. | keying Philox on `sim.tick`, drawing the key from `sim.rng` |
| T-P3-28 | **Events in slot order, from pre-zero values:** a sim with larvae spawned into a lethal seeded patch (`seed_gas_temperature`, the sanctioned seeding call) emits `SwarmUnitDiedEvent`s in ascending slot order on the tick of death, each matching the slot's post-move pose, and the slot is reused by the next spawn with a new id. | an unordered/atomic append, events from the zeroed slot |
| T-P3-29 | **Larvae are inert on fields:** P2's T8 (`field_digest` per tick equal with and without larvae) stays green now that the kernel runs. | the kernel writing any field |
| T-P3-30 | **`check_invariants(sim.gmap) == []` after every tick** of `swarm_thermal_scenario_sim` (CPU) for 130 ticks. | any lifecycle break |

### The 3-part CUDA gate — `tests/cuda_swarm_check.py` + `tests/test_cuda_swarm.py`

Cloned from `cuda_fire_check.py` / `test_cuda_p68_fire.py` (subprocess via
`cuda_harness.run_cuda_script`, `skipif not cuda_harness.cuda_available("swarm")`,
marker `SWARM_P3_RESULT: PASS`, timeout 600 s). The wrapper satisfies
`test_cuda_check_scripts_are_wired.py`.

- **PART 1 — forced-branch fuzz, byte identity.** `bp.swarm_larva_step` vs
  `bp.cuda_swarm_larva_step` on identical copies; compare all 9 columns + the
  death records byte-for-byte and the returned counts; `swarm_cuda_calls()`
  advanced. Random: grid sizes incl. 1×N and N×1; `max_units` 1..300; hw < and
  == capacity; fields straddling every threshold; solid masks (none, all,
  random, larvae inside solid tiles); positions incl. edges and exact tile
  boundaries; headings incl. ±PI_Q16; states {RUN, EAT, COMA}; `dist_walked`
  near 2³²; unit_ids near INT32_MAX; env extremes (tick 0 / 2³²−1, seeds 0 /
  2³²−1, `step_q` 1 and 65535, dt/inv_tile extremes inside R1/R2);
  params extremes (bias 0 and FP_ONE, turn span 1 and full `[0, PI_Q16]`,
  sweep angle min/max, p 0 and FP_ONE). Forcers: everyone dies; everyone
  enters coma; everyone blocked; everyone tumbles; hw = 0 (nothing touched).
  After every case the output satisfies the P2 invariants.
- **PART 2 — 130-tick lockstep, every transition asserted.** Two GameMaps from
  one level (store A → CPU twin, store B → per-call GPU); a scripted synthetic
  `T` field and interior walls (identical copies, driven by an external driver
  between ticks — the fire gate's idiom): a cold band that rises through the
  hysteresis band, a hot front that sweeps in; larvae spawned through
  `SwarmUnits` on both maps. Per tick: byte identity of all store arrays +
  records, `check_invariants(A) == check_invariants(B) == []`. At a scripted
  tick after deaths, spawn k larvae on both: assert the slots are the freed
  ones (lowest free) and the ids are new. **Non-vacuity, each asserted > 0 from
  pre/post diffs:** RUN tumble (heading changed AND moved), wall hit (heading
  changed AND pos unchanged, RUN→RUN), RUN→COMA, COMA held with T inside
  `[T_ctmin, T_coma_exit]`, RUN held inside the band, COMA→RUN, heat kill,
  slot reclaim + reuse by spawn.
- **PART 2b — resident leg.** Two `swarm_thermal_scenario_sim` worlds, CPU
  (residency off, backends off) vs GPU-resident (`set_residency(True)`, all
  backends incl. swarm on — `cuda_s8a_check.py`'s `_mode` toggle idiom), 130
  ticks: per tick `swarm_section_bytes` equal, death-event lists equal,
  `field_digest` equal, `check_invariants == []` both; `swarm_resident_calls()`
  advanced every present tick; the host-dirty path exercised by a scripted
  `sim.swarm.spawn` mid-run (upload happened, identity held). Non-vacuity:
  ≥ 1 death, ≥ 1 coma, `dist_walked` grew.
- **PART 3 — golden on the CUDA build's CPU path:** `trajectory_digest(
  capture_trajectory(n_steps=30)) == GOLDEN_AGGREGATE` (imported from
  `_xarch_perfield_digest`).

### Existing gates that must stay green untouched

`test_ingress_lint.py`; `test_no_float_in_sim_tu.py` (with the P3 TU at
0/0/0); `test_w6_armory.py`, `test_b6_logic_golden.py`,
`test_field_ab_harness.py`, `test_thermostat_books.py`; all four
`test_swarm_*.py` (with the §11 testkit); `test_philox32.py`;
`test_exchange_reductions.py`, `test_wave_push.py` (COUPLING_TABLE unchanged);
`test_cuda_check_scripts_are_wired.py`; every existing CUDA gate (S8a's
scenarios are swarm-free, so the swarm step is dormant there).

### xarch (presence-gated, with the lockstep scenario)

- **`tests/field_ab_harness.py::swarm_thermal_scenario_sim()`** (no RNG):
  `Simulation(_scenario_level(), seed=SEED, breach_physics=bp)`; a 4×4
  interior corner seeded to 90 °C via `gmap.seed_gas_temperature(sel,
  quantize_scalar(ts.from_kelvin(363.15)))`; a vent `g.destroy_wall(12, 15)`;
  24 larvae at fixed tile centres and headings (4 inside the hot corner, 4
  against walls heading in, the rest spread); `set_paused(False)`.
- **`_xarch_perfield_digest.py`:** after the canonical block (unchanged; its
  file and `GOLDEN_AGGREGATE` are byte-identical), run
  `swarm_thermal_scenario_sim` for `SWARM_N_STEPS = 60` ticks in its own loop
  and write `tests/_xarch_swarm_<host>.txt`: per tick, for each species present
  in that tick's carrier, one line per ROSTER column
  (`<tick>\t__swarm__.<sp>.<col>\t<blake2b of the column bytes>`), one
  `__swarm__.scalars` line (next_unit_id, each hw), and one `__swarm__.deaths`
  line (hash of the tick's `SwarmUnitDiedEvent` tuples in order). Built-in
  first-divergence diff against other hosts' `_xarch_swarm_*.txt`.
- **`xarch_digest.py`:** `--scenario {canonical,swarm}` (default canonical →
  the line is unchanged); with `swarm`, digest the swarm scenario and append
  `\tswarm=<sp>:hw<hw>,ssect_v1,sp=<species_hash[:12]>` when the last
  snapshot's carrier is present.

---

## 12. Erik's first look — `tools/swarm_larva_first_look.py` → `docs/swarm_P3_first_look.png`

Produced by the REAL sim path (CPU build, `Simulation.step`), committed with
the patch.

- **Level:** synthetic `LevelData`, 24 × 36 tiles at 0.333 m (8 × 12 m), hull
  border, air interior (tilemap codes 1/4, the harness idiom);
  `seed = 20260924`.
- **Corners held lawfully — through the FieldEdit queue and the gas-energy
  seam, never a bare `temperature[...] =`:** before every `sim.step()` the
  tool enqueues via `sim.edit(...)`:
  - hot corner (4×4 at one end): `FieldEdit(field="wave_source",
    region=Region.RECT, coords=(r0, c0, r1, c1),
    amount=ts.from_kelvin(343.15), mode=EditMode.MAX, source_id=…)` — the
    `wave_source` policy's target is the temperature energy deposit; on an
    accountable gas cell it goes through `gas_energy_deposit(...,
    "field_edit_wave")` (`field_edit.py:641-662`), i.e. booked;
  - cold corner (4×4 at the other end): `FieldEdit(field="wave_source", …,
    amount=1000.0, mode=EditMode.REMOVE, clamp=(ts.from_kelvin(273.15),
    field_edit.T_MAX_PHYS))` — holds the cells at 0 °C through the same seam.
  - Probe (Appendix A): this holds the hot corner ≈ 66–76 °C and the cold
    corner at ≈ 0 °C; note the bulk settles well below 20 °C when the cold sink
    is on (≈ 7 °C time-mean in a 20 × 28 room) — the tool reports the achieved
    bulk mean in the figure caption rather than assuming 20 °C.
- **Protocol:** 10 s warm-up without larvae; spawn 200 larvae uniformly on open
  interior tiles outside both corners, positions/headings from a seeded
  `np.random.default_rng` in the tool (callers own randomness), raw Q16 into
  `sim.swarm.spawn`; run 120 s; `sim.set_paused(False)` before every step
  (TwoPhase auto-pause). Record per tick: live positions, `ai_state`, deaths
  (from `sim.tick_events`), and the felt-T map (E1's reader, computed in
  numpy from `gmap.temperature` for display only).
- **Control:** the identical run with the ablated law (`turn_bias = 0.5`,
  `tumble_rate_alarm = tumble_rate_base`, set through `CFG.swarm` before
  construction — the species table is the only door).
- **Figure (publication quality, 300 dpi, labelled axes in metres, units on
  every colorbar):** (a) time-mean felt-T map in °C (diverging colormap centred
  on 30 °C, contours at 10, 12, 30, 50 °C) with all trajectories (thin,
  translucent), start points, death sites (×), coma onsets (dots); (b) stacked
  state fractions RUN/COMA/DEAD vs time; (c) mean \|T_felt − 30 °C\| of RUN
  larvae vs time, law vs ablation. Caption: seed, dials, E1 reader, achieved
  corner and bulk temperatures.

---

## 13. Build brief (ordered; a Sonnet implementer executes this without re-deriving)

Python: `C:/Users/steen/anaconda3/python.exe` (F10). Build the CPU `.pyd`
(`cpp/build/Release`) and the CUDA `.pyd` (`cpp/build_cuda.bat`). **Confirm
Erik's E1 answer first; default RAW.**

1. `cpp/src/philox_streams.h` + `src/simulation/philox_streams.py` (§6).
2. `cpp/src/swarm.h` (§3.1-3.3): constants + static_asserts, `wrap_heading_q16`,
   `tile_of`, the E1 reader functor, `head_sweep<R>`, `side_preference`,
   `turn_heading`, `probe_move`. All `FP_HD inline`; includes `fixed_point.h`,
   `philox32.h`, `philox_streams.h`. Header comment: engine/17 §4/§9b, this
   doc, and the credit block (§17).
3. `cpp/src/swarm_larva.h`: `SwarmLarvaParams`, `LarvaDerived larva_derive(...)`,
   `SwarmStoreView`, `larva_step_slot(...)` (§3.5, verbatim order),
   `SWEEP_DIST_Q16`, the CPU entry declaration, and under
   `#ifdef BREACH_HAS_CUDA` the `breach_cuda` declarations (backend flag,
   per-call, resident, call counters). Credit block.
4. `cpp/src/swarm_larva.cpp`: the CPU loop (§3.6). **No line may contain the
   words "float" or "double"** (F6). Credit block.
5. `cpp/src/swarm_larva.cu`: file-local `cuda_check`, `larva_kernel`, the
   shared launch core, per-call wrapper, resident launch + scratch (§4.4),
   backend flag + counters. Credit block.
6. `cpp/CMakeLists.txt` (§9 rows, exact placement).
7. `cpp/src/bindings.cpp` (§10), then rebuild both builds.
8. `src/simulation/swarm.py`: `import temperature_scale`; `ENV_*`, `DEATH_*`,
   `LARVA_SWEEP_DIST_Q16`; the eleven `SPECIES_FIELDS` rows (`import math`
   for the bounds), `SPECIES_RELOAD`, `SPECIES_UNITS` + assertions,
   `SwarmSpeciesRow` fields; two-pass `from_config` (§5.2); `SpeciesContext`,
   `species_context`, `larva_derived`, `SpeciesRule`, `SPECIES_RULES`,
   `check_species_rules`; the rule calls in `allocate_swarm_store` and
   `reload_species` (§5.3); `SpeciesKernel`, `SPECIES_KERNELS`,
   `kernel_kwargs`; `column_names(sp)`, `upload_names(sp)`;
   `pack_env_params(gmap, sim_time, clock)`; `death_events(sp, death_rec, hw)`.
   Update the module docstring's scope line.
9. `src/simulation/events.py`: `SwarmUnitDiedEvent` (§7).
10. `src/simulation/exchange.py`: `SwarmCouplingRow`, `SWARM_COUPLING_TABLE`
    (reduction per E1: RAW → `"center"`, FOOTPRINT → `"mean"`), `__all__`.
11. `src/simulation/physics_runner.py` (§4.1, §4.3; exact anchors in §9).
12. `src/simulation/simulation.py`: import `match_key, PhiloxClock`;
    `self._philox_key` (§6); the physics call:
    ```python
            destroyed = self.physics_runner.step(
                self.gmap, sim_time_per_tick, tick=self.tick,
                # arc #63 P3: the swarm step's Philox clock -- the ABSOLUTE tick
                # (sim.tick rewinds every TwoPhase round; draws must not repeat).
                philox=PhiloxClock(self._philox_key[0], self._philox_key[1],
                                   self.round_index * self.ticks_per_round
                                   + self.round_tick))
            # arc #63 P3: swarm deaths (slot 7's last stage) as tick events, slot order.
            self.tick_events.extend(self.physics_runner.swarm_deaths)
    ```
    and the 9c comment line (§8).
13. `config.toml` (§5.5).
14. `tools/run_on_cuda.py`: `"set_swarm_backend",` appended to
    `_BACKEND_SETTERS` (after `:107`). Do not add the combustion setter (F9 is
    reported, not fixed here).
15. Tests (§11): `tests/_swarm_testkit.py` + the four P2 fixture helpers;
    `test_swarm_larva_law.py`, `test_swarm_larva_sensing.py`,
    `test_swarm_constants.py`, `test_swarm_tick.py`, additions to
    `test_swarm_species.py`; `tests/cuda_swarm_check.py` +
    `tests/test_cuda_swarm.py`; the float-ratchet additions; the harness
    scenario; the xarch additions.
16. `tools/swarm_larva_first_look.py`; run it; commit
    `docs/swarm_P3_first_look.png`.
17. Stub regeneration (§10).
18. Verify: full suite (`C:/Users/steen/anaconda3/python.exe -m pytest tests
    -q`) — no new reds vs the 4-red baseline; the CUDA gate PASS; `git diff`
    shows no golden value changed; `_xarch_perfield_digest.py` reports
    `matches golden = True` and writes `_xarch_swarm_<host>.txt`;
    `field_ab_harness.py __main__` self-matches.
19. Papers → `docs/papers/` (§17).
20. `CLAUDE.md` (§16 b rows, appended after "Swarm heading law"; plus the
    two small edits there); chapter 17 "As built (P3)" line; chapter 14 enum
    sentence.

Stage explicit paths only; never `git add -A`.

**Do NOT touch:** `physics_engine.{h,cpp}`; `COUPLING_TABLE` and its tests;
`schema.py`; `GOLDEN_AGGREGATE` or any golden; `tests/_pg5_base_trajectory_
45050f3.pkl`; the stale "F5" wording (P2 R19); fire-12's hunk regions (§9);
`recorder.py`; `SimState`.

---

## 14. For Erik

- **E1 (build-blocking): raw own-tile temperature vs the 2×2 body footprint**
  — see the box at the top. Recommendation: FOOTPRINT.
- **Your question — does venting put them into coma?** Yes (§1 F1): within
  ~1 s, and they stay comatose while the frozen residual gas remains; SPACE
  tiles and hard-vacuum cells read 20 °C (they wake there, and flap on cells
  that toggle). No death (ruling 6). A heat-capacity-aware body temperature is
  the natural later fix (ACCEPTED GAP, §0).
- FYI (non-blocking): V1 (no PhysicsEngine member), V2 (species params by
  value), V5 (the Philox tick is the absolute tick — `sim.tick` rewinds every
  TwoPhase round).

---

## 15. Critique lenses (critics fill this in, one at a time)

| lens | verdict | doc |
|---|---|---|
| Determinism & twin spec (integer sequence, draws, wrap, CPU==GPU, events) | | |
| Residency & CUDA (placement, snapshot contract, uploads/D2H, dormancy, §A) | | |
| Systems reuse & scope (canon rows, new systems, merge-friendliness) | | |
| Behaviour law & biology (sensing, turns, coma/kill, units, defaults, E1) | | |
| Verification (property tests, non-vacuity, gate coverage) | | |

---

## 16. Systems

**(a) Existing canonical systems P3 uses:**

- **Swarm store / facade / digest section** — the kernel and twin are the
  sim-side writers the facade row names; spawn/reclaim unchanged; the section
  and `check_invariants` gate every tick; the empty-slot form is P2's.
- **Swarm species table + Config reload seam** — eleven rows; Kelvin via
  `SPECIES_UNITS`; `SPECIES_RULES` at both installers; the reload seam is the
  only way a new table reaches the kernel (read per launch).
- **Swarm heading law** — the C++ twin, identity-tested; every heading write
  passes it.
- **Deterministic RNG (Philox)** — the vendored `philox32_draw`,
  `philox32_uniform_q16`, `philox32_below`; key = match seed, counter =
  `(tick_abs, 0, salt, unit_id)`.
- **Fixed-point kits** — `mul_q16`, `mul_wide`, `narrow_round_signed`,
  `cos_q16`/`sin_q16` (first device use of the trig kit), `floordiv_q`.
- **Temperature scale** — `from_kelvin` at table load; `to_kelvin` in the
  tool/tests.
- **Q16 boundary modules** — `swarm_fixed.quantize_scalar` (no new copy).
- **PhysicsRunner** — `_run_swarm` beside `_run_combustion` /
  `_run_sky_exchange`; the resident path's `from_host`/`to_host`/`device_ptrs`.
- **Temperature solver / Gas temperature is a mirror** — larvae only READ
  `temperature`; the tool and scenarios seed/hold it via
  `seed_gas_temperature` and `wave_source` FieldEdits.
- **FieldEdit** — the tool's hot/cold corners.
- **Coupling table** — the sibling `SWARM_COUPLING_TABLE`.
- **Tick events** (`events.py`) — `SwarmUnitDiedEvent`.
- **Field digest / GOLDEN_AGGREGATE / A/B harness / CUDA harness / xarch** —
  unchanged goldens; a new harness scenario; the 3-part gate; presence-gated
  xarch lines.
- **Ingress lint + float ratchet** — new modules scanned; `swarm_larva.cpp` at
  0/0/0.
- **RL-batch habits (§A)** — §4.3.
- **Test conventions** — property tests + non-vacuity counters.
- **Not used:** `PhysicsEngine` (V1), the recorder (unchanged), `schema.py`.

**(b) New systems P3 creates — DRAFT one-line CLAUDE.md rules** (written at
implementation, pointing at the real code, appended after "Swarm heading law"):

- **Philox stream salts** — `src/simulation/philox_streams.py` (`StreamSalt`,
  `match_key`, `PhiloxClock`) + `cpp/src/philox_streams.h`. THE one breach-wide
  salt enum: append-only, a value is never reused or renumbered, every Philox
  consumer (CPU units included) takes a new value here; the key is
  `match_key(sim.rng)` (the match seed, no draw); the counter tick is the
  ABSOLUTE tick, never the round-local `sim.tick`; identity-tested against the
  C++ header.
- **Swarm kernel law** — `cpp/src/swarm_larva.h::larva_step_slot` (FP_HD).
  ONE per-slot law per species; the CPU twin (`swarm_larva.cpp`, on the
  float ratchet at 0/0/0) and the CUDA kernel (`swarm_larva.cu`) are loops
  around it, never a second copy; own-slot writes only, read-only fields, no
  atomics; every draw is a row of the doc's draw table (salt, draw_index 0,
  fixed word); runs as the last stage of slot 7 via `PhysicsRunner._run_swarm`,
  dormant (no transfer, no launch) when no swarm is present.
- **Swarm shared kit** — `cpp/src/swarm.h` (`head_sweep<Reader>`,
  `side_preference`, `turn_heading`, `probe_move`, `wrap_heading_q16`,
  roster/enum twins). Every species senses through `head_sweep` with its own
  reader functor and moves through `probe_move` (out-of-grid and solid =
  blocked, no sliding, `step < 1 tile`); a blocked sweep point is never a
  reading.
- **Swarm species rules / units / kernels** — `swarm.SPECIES_RULES`
  (invariants the kernel relies on, checked by `allocate_swarm_store` and
  `reload_species` against a `SpeciesContext`), `SPECIES_UNITS` (Kelvin rows
  converted through `temperature_scale` at load only), `SPECIES_KERNELS` (a
  species' kernel is a table entry, never an if-chain). A rule protects an
  invariant, never tuning taste.
- **Swarm death events** — `events.SwarmUnitDiedEvent`, built from per-slot
  device/CPU records compacted on the host in slot order. The ONLY source of a
  dead swarm unit's pose; renderers and later food consumers read the event,
  never the freed slot.
- **Swarm coupling rows** — `exchange.SWARM_COUPLING_TABLE`: a physics→swarm
  coupling is documented here beside the unit rows and executed in its species
  kernel; each row names the species-table fields it reads (test-checked).

Edits to existing P2/P1 rows (branch-only rows, no fire-12 overlap): the
"Swarm species table" row gains "Kelvin rows via `SPECIES_UNITS`, cross-field
invariants via `SPECIES_RULES`"; the "Deterministic RNG" row gains "salts
from `philox_streams` only".

---

## 17. Credit + papers (`docs/papers/`)

Header credit block (in `swarm.h`, `swarm_larva.h/.cpp/.cu`, and the
`swarm.py` P3 section) — citations verified via PubMed:

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

## Appendix A — the probes behind F1 and E1 (reproducible, scratch scripts not committed)

All on HEAD `83b7fbd`, CPU build `cpp/build/Release`, base interpreter,
`temperature_scale.load()` (k = 3, kelvin_ambient 293).

**A1 — vent (F1).** `tm = ones((16,16)); tm[1:15,1:15] = 4`,
`LevelData(..., tile_size_m=0.333)`, `Simulation(seed=1)`,
`g.destroy_wall(8, 0)`, `set_paused(False)`, step 400 ticks; per tick over
`interior[1:15,1:15]`: `T = g.temperature`, `N = gas[O2] + gas[INERT_N2]`.
Result: t = 0: T_game min −19.7, 173/196 cells below 283.15 K; t = 25: mean
−224; from t ≈ 125: mean N ≈ 0.016 (never 0), cells with N < 1 raw count
toggling 0–15, those read T = 0. Six-tile breach: same floor, faster.

**A2 — mosaic (E1).** Sealed room `tm[1:19,1:27] = 4` on a 20×28 grid,
`tile_size_m=0.333`, `seed=3`; `g.seed_gas_temperature(sel, round(ts.from_
kelvin(333.15) * 65536))` on rows 1–4 × cols 1–4 at t = 0 only; step with
`set_paused(False)` each tick. Bulk = rows 7–12 × cols 10–17, °C via
`ts.to_kelvin(T/65536) − 273.15`. At t = 48, 120, 240, 480, 719: bulk mean
20.5/19.8/20.1/20.2/20.7 °C, spatial sd 25.2/13.8/16.3/14.9/14.9 K, min/max e.g.
−14.3/+54.9 °C (t = 48). Over the next 480 ticks: raw sd 15.0 K; 3×3 box
mean 2.0 K; 2×2 vertex mean 2.4 K; per-cell 5 s EMA 14.0 K; mean \|ΔT\| to
the right-hand neighbour 27.7 K. Unforced room: sd 0.00 for 480 ticks.

**A3 — held corners (§12).** Same 20×28 room; each tick
`wave_source` MAX to `from_kelvin(333.15)` on rows 1–4 × cols 1–4 and
`wave_source` REMOVE 1000 clamped at `from_kelvin(273.15)` on the far 4×4;
1440 ticks: hot corner time-mean ≈ 66–68 °C, cold corner ≈ 0 °C, bulk
time-mean ≈ 7 °C with the cold sink on (≈ 22 °C with only the hot MAX);
within-bulk spatial sd ≈ 15 K under every forcing tried (`heat` ADD 0.25/1/4,
`wave_source` MAX).
