# Smoke transport v2: trace gases ride the air (design v2, 2026-10-04)

**Issue:** #12, the fire + smoke session, step 3.
**Branch:** `12-smoke-transport`, cut off `fire-12`.
**Status:** v2. Three critique lenses have been run and folded in (§12). One ruling
is still open for Erik (§11, D1).
**Depends on:** `docs/smoke_transport_handoff_2026-10-04.md` (the problem) and the
#12 comments of 2026-10-04, which hold the measurements and the look spike
(`prototypes/smoke_transport_spike/`, untracked).

## 1. Problem, measured

The five non-bulk gas planes (smoke, steam, poison, teargas, fuel_gas; together
"the trace planes") are moved once per tick by an integer semi-Lagrangian (SL)
step (`cpp/src/smoke_dynamics.cpp`, CUDA `cuda_smoke.cu`). A per-tick replay
attribution, run 2026-10-04, found:

| mechanism | measured effect |
|---|---|
| SL keeps a tile's value as its air expands (advective form on a divergent flow, H1) | Creation tracks `Σ c·div(u)·dt` tick by tick. The studio peaks at 100 units against 8.9 deposited |
| SL resampling over a 20–37-tile back-trace at blast speeds | Single ticks destroy 13–52 units |
| diffusion `mul_q16(coeff, lap)` floors per cell | About ½ count is lost per cell per tick: −11 % in 10 s on a quiet blob |

**Erik's tolerance (2026-10-04).** Deterministic non-conservation is acceptable,
and so is wind or a breach clearing smoke. What is not acceptable is explosions
inventing smoke, and a room going black so nothing can be seen.

**Look spike (offline, on the captured studio flows):**
- **A, today's scheme:** 91 units at 1.75 s.
- **B, SL per unit of air:** 40 units at the peak, then 0.1 by 6 s. Rejected.
- **C, conservative donor-cell:** follows the deposit exactly. Smooth, with
  blast rings early and swirls later. **Chosen by Erik.**

Diffusion is invisible in B and C (the same totals and picture without it).

## 2. The law

### 2.1 Advection: stage 3b, once per EOS substep

On every face, in every EOS substep, the trace crossing the face is that face's
applied bulk mass flux, priced at the donor's trace-per-air ratio:

    phi_f = price_face(dq_f, S_pre[donor], N_pre[donor])   = floordiv(dq_f · S, N), exact in 64 bits

- **`dq_f`** is the post-limiter per-face bulk dq that stage 2 already banks
  (`dqsum_e` / `dqsum_s`, summed over the conservative planes).
- **`N_pre`** is the pre-flux bulk sum over the same conservative planes the
  limiter reads, taken BEFORE stage 2 (which mutates `gas` in place).
  - On CPU, `n_pre` is extended to hold the true N on thermal-solid cells. This
    is energy-safe: stage 3 reads `n_pre` only behind `e_participates`,
    `bulk_transport.cpp:356-368`.
  - On device, `d_nb` already holds the true N there.
  - The `< 2^30` assert extends to those cells.
- **`S_pre`** is a pre-pass snapshot of the trace plane.

This is the shape in which `gas_energy` already rides the same faces (arc #54,
`bulk_flux_energy_transport_cached` stage 3). It reuses the gather form
(each cell writes only its own S, with no atomics on S), the pinned face order
E, W, S, N, and `price_face`. Both sides of a face price the same triple, so the
pass telescopes.

The limiter is per plane on its own pre-flux N, so `Σ_f dq_f ≤ N_i`, and
therefore `Σ_f phi_f ≤ S_i`: a donor never goes negative.

The donor's trace-per-air ratio is invariant as air leaves it. That is the H1
fix: expanding air thins its smoke instead of multiplying it. The `dq·S < 2^61`
product fits.

**Participants:** `!solid && !is_vacuum && !ring`, including thermal-solid
(crate) tiles. Air seeps through their pores, and the trace goes with it.

**Boundary faces:**
- A participating donor priced onto a vacuum or ring receiver exports that trace.
  The donor books it (it is the only participating side), in the vent channel
  (§3).
- Inflow from vacuum or ring carries no trace.

**Order within the substep:** stage 3 (energy apply), then **3b (trace apply)**,
then stage 4 (mirror refresh, which on device re-accumulates `d_nb` into the
post-flux N), then **3c (trace wipe)**. In 3c, a participating cell whose
post-flux N falls below `N_EPS_RAW` has its trace destroyed and booked in the
wipe channel: no air is left to carry it. CPU uses the same order.

**Plane skipping:** a plane that is all zero stays all zero, with all its
counters zero. Skipping such planes is an optimisation only:
- CPU and chained may skip a plane after a host scan.
- Resident skips nothing (RL habits §A).

### 2.2 Diffusion: once per tick, conservative, Jacobi

This runs where the SL call was (after the substep loop), from a snapshot
`S0` of the plane. Each face is evaluated once in canonical orientation
(`i` = lower index):

    c_ij  = mul_q16(dd_q[g], quantize(min(perm_i, perm_j)))   dd_q[g] = quantize(d_g · dt), folded once on the host
    |F|   = (c_ij · |S0_i − S0_j|) >> 16                       magnitude truncation, the scale_mag idiom
    F     flows from the larger to the smaller

Each cell gathers its four faces and applies ± from the snapshot.

**Why magnitude truncation and not a floor.** A floor toward −∞ would break
positivity and pump thin smoke toward the north-west (lens 1). With magnitude
truncation:
- `|F| ≤ c·|ΔS| / ONE`, so together with `Σ_j c_ij ≤ ONE` positivity is exact;
- the flux is symmetric in orientation;
- there is no drift.

**Stability:** `4 · dd_q ≤ FP_ONE`, checked on the integer at the gases door and
re-checked where dt is bound (PhysicsRunner). Shipped dials: 0.004 per tick.

Faces to vacuum and ring cells diffuse like any other face. Their outflow is
booked in the vent channel (§3). This keeps `test_air_boundary.py` gate 4 true
(the ring is a trace sink). Solid faces carry `perm = 0`, so no flux.

Diffusion stays. It is the only thing that spreads smoke in still air.

### 2.3 Decay: once per tick, after diffusion, with the floor fixed

Today `lost = mul_q16(v, frac_q)` is 0 below `v < 2^16 / frac_q` counts. That is
about 0.045 units for smoke and 0.091 for poison. Until now the SL diffusion
floor's leak hid this. Under conservation it becomes permanent haze, and
permanent **lethal poison** in a band above `poison_min_density = 0.05`
(lens 3, finding 3).

The fix: `lost = ceil(v · frac_q / 2^16)` for `v > 0`. Above the threshold this
keeps the configured exponential e-fold. Below it, it removes 1 count per tick,
so thin gas is gone in at most `v` ticks. Decay is booked in the decay channel.
This is a numeric fix that honours the configured dial (§11, D4).

### 2.4 Clamps and stranded trace

**No trace plane is clamped anywhere outside a reader.** Three clamps go:
- **The SL step's [0, 1] clamp:** deleted with SL.
- **The fire step's smoke clamp:** deleted, CPU `fire_simulation.cpp:401-407` and
  CUDA `cuda_fire.cu:336-345`. It cut every compressed tile back to 1 each tick,
  silently undoing conservation (lens 3, finding 1). Fire keeps its own clamp.
- **The FieldEdit deposit ceiling:** see §11, D1, Erik's ruling.

The lower bound becomes a debug assert, since transport, diffusion and decay
cannot produce a negative.

**Stranded trace.** Once per tick, trace sitting on a `solid ∨ vacuum ∨ ring`
cell is zeroed and booked in the sink channel. Such trace arrives when a tile is
breached, sealed, or deposited onto. The old SL clamp did this silently, every
step (`smoke_dynamics.cpp:309-314`).

**int32 narrow.** With S ≥ 0, a cell's S is at most the map total. The narrow is
asserted at `Σ_map S < 2^31` per plane, which is 32768 units, far above any
deposit.

## 3. Books

The trace slots of the existing per-gas int64 rail `EOSSolver::boundary_flux_`
(`eos_solver.h:511-518`, sized `n_gases`, not digested, today filled only for the
bulk planes at the ring reset) carry the **vent** channel. That is everything
priced or diffused onto vacuum or ring. This avoids a parallel per-gas counter.

Three sibling per-gas int64 arrays sit next to it in `EOSSolver`:
- **`trace_wipe_sum`:** the N_EPS wipe (3c).
- **`trace_sink_sum`:** stranded trace (§2.4).
- **`trace_decay_sum`:** decay (§2.3).

All are accumulated per tick, are diagnostics only, and are not digested. On
device they use the unsigned-long-long `atomicAdd` idiom.

**The identity**, exact in int64 for any tick and plane:

    Σ S(after) − Σ S(before) == deposits − vent − wipe − sink − decay

Deposits are FieldEdit, payloads, combustion soot, debug keys, vents and water W5.
They are outside transport. The tests measure them by bracketing.

**Other existing movers of trace** are unchanged and already conservative:
vents, which price trace by air (`vent_system.py:476`); W3 water evacuation
(`physics_engine.cpp:1147-1177`); and the `destroy_wall` / `seal_tiles`
redistribution (`gamemap.py:2770-2917`).

**The trace never touches the air.** It stays out of bulk N, out of p*, and out
of the energy books. `GasTable.conservative` keeps its single meaning, bulk
membership.

## 4. Readers (what changes for gameplay)

Conserved trace lasts longer and can exceed 1 in compressed pockets. Audit
(lens 3):

| reader | behaviour above 1 / longer life | feel-adjacent |
|---|---|---|
| Radiation sweep `a_gas`, light smoke term | `min(ONE, Σ…)` per tile: saturates safely | yes (the look) |
| Hitscan laser `combat.py:1145-1156` | `trans = ONE − Σ beam_absorb·ρ` saturates. Longer-lived haze shortens laser reach | yes |
| Poison dose `exchange.py:666-683` | linear and uncapped above 1 (pockets dose harder). The decay fix ends the permanent band | yes |
| Teargas BLINDED | threshold 0.15, safe above 1. The cloud lasts longer | yes |
| Smoke sensor `sensor_accessor.py:130-132` | reads the density; no shipped level uses it | no |
| `gas_medium` / speckle | clip for glow and speckle only, safe | yes (the look) |
| combustion soot emission | adds without a clamp (`combustion.cpp:1004`) | no |
| Tile inspector | shows the value | no |

The poison dose above 1 stays linear. That is today's law; pockets above 1 are
real compressed gas (§11, D5).

## 5. Backends

**CPU.**
- Stage 3b and 3c go in `bulk_flux_energy_transport_cached`.
- The diffusion, stranded zeroing and decay form a once-per-tick trace tail in
  `PhysicsEngine::run_substeps`, where SL was.

**Chained GPU EOS** (the only live GPU EOS path; there is no per-call CUDA EOS).
- Stage 3b and 3c go once into `bulk_flux_energy_transport_device`
  (`cuda_bulk_transport.cu:537`), between stage 3 and stage 4's `d_nb` memset
  (3b) and after stage 4's re-accumulation (3c). This serves chained and resident
  alike.
- Chained gains H2D/D2H for the trace planes, an `S_pre` scratch plane, and the
  counter block. The D2H lands before `digest_bulk_flux`, which hashes every gas
  plane.
- The once-per-tick tail stays in the shared host code after the dispatch, so
  CPU and chained agree automatically.

**Resident.**
- Every plane is already on device. Stage 3b and 3c need trace pointers and
  persistent scratch in `g_eos_res`.
- The once-per-tick trace tail (diffusion, stranded zeroing, decay) becomes a
  resident launch, `(N, h, w)`-shaped (§A). It is called where
  `trace_smoke_resident` is called now. `do_traces` is redefined as "the tail
  runs on the host".

**Deleted** (each listed in the patch report):
- `SmokeDynamics` (cpp, `engine.smoke`, its binding `bindings.cpp:3757`, and
  `physics_runner.py:209-212`).
- The `#ifdef` `smoke_step` branch in `run_substeps`.
- `cuda_smoke.cu` (`smoke_step`, `trace_smoke_resident`).
- `set_smoke_backend`, together with its callers: `tools/run_on_cuda.py:95`,
  `cuda_ambient_check.py:42`, `cuda_p6d_light_transport_check.py:94`,
  `cuda_radiation_sweep_check.py:767`, `cuda_sky_exchange_check.py:34`,
  `cuda_thermal_mass_check.py:411`, `cuda_thermal_mass_eos_check.py:350`,
  `cuda_s8a_check.py:62`.
- The `n_smoke` kit (`fixed_point.h:720-760`) and `test_bedrock_cliff_counts.py`.
- The config keys `advection_rate`, `wind_diffusion_scale` and `d_smoke`
  (`config.toml:100-104`).
- The float-ratchet TU row (`test_no_float_in_sim_tu.py:58, 366`) and the
  CMakeLists entries (`:78, 102, 182`).

**Rewritten:**
- `cuda_s8a_check.py` PART 2, together with `test_cuda_s8a_residency.py`.
- `test_multigas_structure.py:244-249, 294`, as properties.
- `test_gas_weapons.py:87`'s docstring.
- `test_field_edit.py:231,239` and `test_payloads.py:211`, following D1.

**Intermediate red.** The worktree has no CUDA build, so P1's suite cannot see
the CUDA gates. About 10 `GOLDEN_AGGREGATE`-reading CUDA checks are expected red
from P1 to P2a. The branch merges only at P3.

## 6. Cost (answered to Erik 2026-10-04)

Measured on this PC's CPU build:
- **Tick, median:** 7.0 ms in the studio (8 EOS substeps after a blast), 10.6 ms
  on the playground (1–4 substeps).
- **One stage-3b pass:** costs about one bulk mass-flux pass, 0.1–0.3 ms per
  substep. The SL step it replaces cost 0.14–0.29 ms per plane per tick.
- **Net:** about equal in quiet rooms. About +0.6 ms per active plane per tick
  after a blast (roughly +9 %).
- **Several active planes** may share one face pass.
- **GPU:** negligible.

## 7. Verification plan (properties)

Each item names the property it pins and the change that must break it.
Nothing pins a count or a set of gases.

- **V1, exact conservation.** Sealed room, blast, decay overridden to 0, one tile
  seeded at 2.0. Then `Σ S + wipe` is constant to the count across N full
  `Simulation.step`s. *Breaks if* any stage floors asymmetrically, skips a face,
  or any clamp (fire, SL, deposit) returns.
- **V2, no expansion mint.** Uniform ratio `S = r·N`, a heat-only blast,
  diffusion at 0. The ratio stays within `faces · substeps / N_min` of r.
  *Breaks if* the trace is moved as a per-tile value (H1).
- **V3, venting closes.** Seal, then breach: the identity of §3 holds exactly.
  *Breaks if* a vent, diffusion or sink path bypasses its counter.
- **V4, non-negativity.** No trace cell is ever negative across the blast
  scenarios. *Breaks if* a donor is over-priced or diffusion floors.
- **V5, the air does not see the smoke.** Bulk N, `gas_energy`, wind and T are
  bit-identical with the trace planes populated or empty. *Breaks if* trace
  leaks into bulk membership, p* or the energy books.
- **V6, CPU == CUDA at tolerance 0**, chained and resident (including the
  thermal-mass-eos resident leg), on every plane and counter. Every
  `GOLDEN_AGGREGATE` consumer is re-greened.
- **V7, diffusion conserves, spreads and does not drift.** In still air,
  `Σ S` is exact, a blob's second moment grows, and a uniform thin field stays
  uniform. *Breaks if* diffusion is removed, loses mass, or floors (which would
  bring back the north-west pump).
- **V8, decay reaches zero.** A thin uniform poison field (0.06) in a sealed room
  goes to 0 in bounded time. *Breaks if* the decay floor returns.
- **V9, readers at density 2.** The poison dose and teargas BLINDED are well
  defined at 2.0 (linear dose; threshold). *Breaks if* a reader starts assuming
  ≤ 1.
- **Reading, not a gate:** the studio's total after the three charges sits within
  the deposits minus decay. Reported in P1.

## 8. Patches

| # | content | mode | tier | gate | HUMAN-TEST |
|---|---|---|---|---|---|
| P1 | CPU: stage 3b and 3c, the trace tail (Jacobi magnitude-truncated diffusion, stranded zeroing, ceil decay), the books, clamp removal (SL, fire CPU, deposit per D1), the gases door and PhysicsRunner stability check, all CPU deletions and rewrites in §5, V1–V5 and V7–V9, GOLDEN re-baselined ONCE with written rationale. Report states which CUDA checks are expected red | worktree subagent | Opus | V1–V5, V7–V9, and the CPU suite green | no |
| P2a | CUDA: stage 3b and 3c in `bulk_flux_energy_transport_device`, the chained H2D/D2H and counters, the fire CUDA clamp, `smoke_step` / `set_smoke_backend` and their callers deleted, every `GOLDEN_AGGREGATE` CUDA check re-greened. Builds CUDA in the worktree | worktree subagent | Sonnet (tol-0 oracle) | V6 chained | no |
| P2b | Resident: the trace tail launch `(N, h, w)`, `trace_smoke_resident` and `cuda_smoke.cu` deleted, `cuda_s8a_check` PART 2 rewritten, V6 resident | worktree subagent | Opus | V6 resident | no |
| P3 | CLAUDE.md rows and survey row 20 (§9), #12 and Roadmap update, merge into fire-12 | inline | — | suite green + Erik | **YES**, see below |

**P3's HUMAN-TEST, Erik plays:**
1. The smoke-light studio.
2. The playground fire.
3. The smoke_screen, tear_burst and poison_cloud grenades in a sealed room,
   watched for 60 s or more.
4. A laser fired through settled smoke.
5. Water boiling in a drained room (steam riding near-zero N).
6. The long-run haze after a fire.

## 9. Systems

**(a) Existing canonical systems this design uses:**
- **Bulk transport** (the dq planes, N snapshot, limiter, `price_face`).
- **The gas-energy stage-3 shape and the face-flux idiom.**
- **Fixed-point kits** (`floordiv_q`, the `scale_mag` truncation, `mul_q16`,
  `quantize`).
- **`EOSSolver::boundary_flux_`** (the per-gas rail; its trace slots carry the
  vent channel).
- **FieldEdit** (the deposit seam; its trace ceiling per D1).
- **Material / gas tables** (diffusion and decay columns, and the door's
  stability check).
- **Config / PhysicsRunner** (dials bound once).
- **Vent / duct system, W3 water evacuation, `destroy_wall` / `seal_tiles`**:
  existing conservative trace movers, unchanged.
- **Determinism gates** (field digest, `GOLDEN_AGGREGATE` re-baselined once, the
  CUDA harness).
- **RL-batch habits §A.**

**(b) New, with draft rules for CLAUDE.md:**
- **Trace transport** (`bulk_transport.cpp` stages 3b/3c + the trace tail in
  `run_substeps`; the CUDA twins in `cuda_bulk_transport.cu` + the resident
  tail): "A trace plane's TRANSPORT (advection and diffusion) is ONLY the
  stage-3b ride on the bulk face flux (`price_face(dq, S, N)` at the donor) plus
  the once-per-tick Jacobi, magnitude-truncated face diffusion. No trace plane is
  clamped anywhere outside a reader. Every loss is booked: vent in the trace
  slots of `boundary_flux_`, then wipe, sink, decay. Other movers (vents, W3,
  `destroy_wall` / `seal_tiles`, deposits through FieldEdit) are separate,
  already-booked systems." This also replaces survey row 20
  (`docs/canonical_systems_survey_2026-08-22.md:32`).

## 10. Recorded, not fixed (other systems)

- **The air.** After one heat-only frag in the sealed studio, the wind sits on its
  rails (160–300 m/s) for more than 2 s. The EOS runs at its 8-substep cap with a
  per-substep CFL of about 4.7. The trace inherits that coarseness; it does not
  cause it. Drag is revisited after this arc (Erik, 2026-10-04).
- **The physical amount.** Even conserved, the studio's 99 g of soot gives a
  per-tile τ of 0.3–0.7 once spread, which is 2–4 m visibility. The lever is the
  soot yield (TNT's 0.193 g/g assumes no afterburn). Erik's call, after this arc.

## 11. Decisions

**Open, for Erik:**
- **D1, the deposit ceiling.** FieldEdit's `"smoke"` / `"gas"` policies clamp the
  WHOLE tile to [0, 1] after old + contribution (`field_edit.py:219,227,488-498`;
  `payloads.py:87`, `physics.py:180`, `combat.py:1372`). Under conservation, a
  deposit onto a compressed tile above 1 destroys smoke that was already there.
  - **(a)** Drop the ceiling for trace ADD edits. A deposit adds exactly its
    amount. The smoke_screen grenade (`gas_amount = 1.5`, "saturates the core")
    then puts more smoke in its core than today, by up to 1.5× at its centre
    tile.
  - **(b)** Keep the ceiling, counted as destruction.
  - Recommendation: **(a)**. It is the only option under which "smoke is
    conserved" is true, and smoke_screen's amount is a payload row Erik retunes
    anyway.

**Statements, Erik may veto:**
- **D2:** all five trace gases ride, not only smoke. They share the defect;
  vents already price trace by air; steam's puff and combustion add without a
  clamp; nothing in the sim reads fuel_gas's flammability.
- **D3:** crate (thermal-solid) tiles carry trace in their pores, as they carry
  air.
- **D4:** decay's rounding is fixed (`ceil`). This ends the permanent thin haze
  and the permanent lethal poison band. The configured e-folds are unchanged.
- **D5:** poison dose stays linear above 1 (today's law).
- **D6:** diffusion stays, conservative, with today's dials.

## 12. Critique log

| lens | critic | verdict | resolution (v2) |
|---|---|---|---|
| conservation / integer arithmetic | Opus, 2026-10-04 | PASS WITH FIXES. Stage 3b is confirmed exact and non-negative against the code. Findings: **(1) BLOCKER**, the diffusion floor breaks positivity and pumps thin smoke north-west; **(2) BLOCKER**, stranded trace on solid / vacuum / ring cells; (3) diffusion via vacuum/ring faces unspecified; (4) diffusion must be Jacobi; (5) the coefficient must be an integer chain, with the stability check on the integer and at dt binding; (6) the int32 narrow bound; (7) V2's bound wrong; (8) V1 needs the wipe term; (9) the N snapshot needed specifying | (1) §2.2 magnitude truncation; (2) §2.4 booked sink; (3) §2.2 diffuses and books the vent; (4) §2.2 Jacobi; (5) §2.2 `dd_q` chain, door plus PhysicsRunner; (6) §2.4; (7) V2 at diffusion 0, bound per substep; (8) V1; (9) §2.1 |
| backends / resident / CUDA lockstep | Opus, 2026-10-04 | PASS WITH FIXES. **(1) BLOCKER**, the resident path loses diffusion, decay and zeroing when `trace_smoke_resident` goes; (2) no per-call CUDA EOS exists, stage 3b goes in `bulk_flux_energy_transport_device` once; (3) chained needs trace H2D/D2H before `digest_bulk_flux`; (4) the device `d_nb` reuse fixes the 3b/3c order; (5) the deletion list was incomplete (`set_smoke_backend` callers including `run_on_cuda.py`); (6) P1 cannot see the CUDA gates, so about 10 checks are red until P2; (7) the zero-skip is only an optimisation; (8) atomics for the counters; (9) no silent CPU bypass. Recommends the P2a/P2b split | (1) §5 resident tail, P2b; (2)–(4) §5 and §2.1 order; (5) §5 full list; (6) §5 and P1's report; (7) §2.1; (8) §3; split adopted in §8 |
| systems reuse / readers / scope | Opus, 2026-10-04 | PASS WITH FIXES. **(1) BLOCKER**, the fire step clamps smoke to [0, 1] every tick on both backends; **(2) BLOCKER**, the FieldEdit deposit clamps the whole tile; **(3) BLOCKER**, the decay floor makes thin poison permanent and lethal once SL's leak is gone; (4) missing gameplay readers (laser, poison dose, teargas, sensor); (5) the draft rule was inaccurate (other movers exist; no SmokeDynamics rows in CLAUDE.md, survey row 20 instead); (6) `boundary_flux_` already exists as a per-gas rail; (7) tests not listed; (8) the HUMAN-TEST was too narrow; (9) moving all five gases is safe | (1) §2.4, P1/P2a; (2) §11 D1, open for Erik; (3) §2.3 ceil decay, D4, V8; (4) §4 table, V9; (5) §9 rule reworded; (6) §3 reuses the rail; (7) §5 rewrite list; (8) §8 HUMAN-TEST widened; (9) D2 |
