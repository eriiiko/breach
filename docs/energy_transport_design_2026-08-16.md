# Energy-books arc — energy-conservative thermal transport (design v1, 2026-08-16)

**Status: DRAFT v1, submitted to a 4-lens adversarial critique panel. Not
blessable until the panel's blockers are resolved on paper (v2, v3, …).**
Authored by Claude (Fable) with Erik in-session; every ruling marked
`RULING (Erik 2026-08-16)` was made by Erik in that session with the recorded
reason. Branch: `storm-damping` (continues the storm/thermal line; the audit
and P-S1 live here). Supersedes the never-built "E1 options 2+3" rails patch
(see §9 history note).

---

## 0. Problem and evidence

The engine transports temperature as an **intensive scalar copied by
semi-Lagrangian sampling** (`cpp/src/eos_solver.cpp:457-508`; the file
self-documents the consequence at `:423`: the source is a "FREE-ENERGY channel
(SL sampling copies without debiting)"). Temperature-changing terms whose
denominators collapse at evacuated cells (compression work `T·(1−k)` at
`eos_solver.cpp:726-784`; deposit conversions `ΔT = ΔE/(N·c_v)` at
`combustion.cpp:774-808` and `temperature_solver.cpp:367`) therefore write
enormous T *values* into near-empty cells, and the transport happily copies
those values onto real mass later. Temperature times mass is energy: the copy
is a mint.

Measured consequences (`docs/storm_audit_2026-08-14.md`):
- §4.4 — Erik's in-game blowup: a starved, evacuated fire cell climbs
  ×1.4957/tick (the 1+T_WORK_CLAMP rate-rail signature) to T_MAX_PHYS = 16000,
  then 47–65 atm pressure spikes when a ~50×-ambient gas slug meets the
  phantom-hot pocket. Happens at shipped dials.
- §4.3 — the bench §5 window: the same engine on the cold rail (supercooling
  spiral to T_MIN, ambient reservoir back-feeding the pocket through hull
  conduction: +25,150 eth after the fire died).
- §4 table — the leak is not a corner case: **even post-P-S1, EOS transport
  injects +7,805 eth over a normal 200 s two-room bench run** (pump-off row).
  Normal operation sits on an open mint held down by cooling laws.

**RULING (Erik 2026-08-16): the bounded-rails patch (trust-gate + floor raise
alone) is rejected as a destination** — it caps the mint rate but leaves the
books open, "correct only while cooling outpaces the leak; we are building a
very unstable system." The structural fix is chosen instead. Reason recorded:
these explosions are extremely unphysical; the project's doctrine is
physically interpretable models, and now is the cheapest window (goldens under
standing deferral; fire recalibration already scheduled).

## 1. The principle (one sentence + four rules)

**Every thermal exchange is denominated in energy; temperature is what energy
looks like through a cell's actual mass.**

- **R1 — Transport conserves.** Thermal energy rides the same conservative
  donor-cell face fluxes that bulk mass already rides. Mixing is therefore
  mass-weighted by construction; a phantom-hot near-empty cell carries ~no
  energy and dilutes to nothing on contact with real gas.
- **R2 — Conversions are local and honest.** Energy→temperature conversion at
  any endpoint divides by that endpoint's actual thermal capacity (gas: N·c_v;
  object: its `thermal_mass` via `heat_inv_shift`). No global or assumed
  denominators.
- **R3 — Floors may only destroy, and are counted.** Every value-hygiene
  guard (divisor floor, wipe at ~vacuum) is one-directional: it may drop
  energy (bounded, counter-tracked), it may never create it. After this arc,
  NO channel can create thermal energy except combustion and radiation
  deposits (chemistry), and none silently.
- **R4 — Determinism unchanged.** Q16.16/int64 integer arithmetic only,
  order-pinned loops, CPU↔CUDA bit-identical, no new synced planes (the
  energy plane is transient within a substep).

## 2. Mechanism, pass by pass

### 2.1 EOS transport (the core change) — replaces the SL T-sample

Inside the existing substep loop (`eos_solver.cpp:453-534`):

1. The fused SL advection keeps its velocity job (u self-advection); **the
   `.t` slot is deleted** (T-WRITE SITE 1/2 retires). The A2 `t_occlude`
   mask's transport job retires with it — in flux form, energy moves with
   mass through whatever the *flow* permeability admits, which is the
   physical statement (an object's own T remains TemperatureSolver-owned and
   is never written here; the `ts`/vacuum/ambient guards keep their exact
   current skip semantics). [Panel: attack this retirement — Q6.]
2. Build the transient energy accumulator, exact and unshifted:
   `e[i] = (int64)n_bulk_raw[i] · (int64)T_raw[i]` where
   `n_bulk = Σ gas[gi] over gas_conservative[gi]` (the O2/inert-N2 pair —
   same predicate the flux loop branches on; `src/simulation/gases.py:89`).
   Worst-case magnitude ~4.5e15 < 2^62 — int64-safe with headroom (§3).
3. Extend `bulk_flux_transport_cached` (`eos_solver.cpp:510-528` entry): for
   every face, alongside each bulk plane's donor-cell mass flux φ_gi, move
   energy `φ_e = (Σ_bulk φ_gi_raw) · T_raw[donor]` (int64, exact — donor
   chosen by the same upwind sign the mass flux already uses). One extra
   accumulation in the existing loop; no new stencil; antisymmetric per face
   by construction.
4. Recover `T[i] = floordiv(e[i], n_bulk_new[i])` (floor division pinned on
   both backends — see §3 rounding). Guard: `n_bulk_new < N_EPS` (proposed
   1 count = 2^-16 ambient) → `T := 0` and the residual e is dropped into a
   counter (`e_wipe_hits` + int64 `e_wipe_sum`) — R3: bounded (< N_EPS·T_max
   per cell), one-way, counted. Ring/vacuum cells keep today's `T := 0` wipe.

Ledger property: the transport pass's Σ eth contribution is **≤ 0 every tick,
bounded by 1 LSB of T per active cell per substep** (floor-division bias is
one-way down) — versus **+7,805 unbounded-positive today**. The mint is dead;
what remains is a counted, bounded, one-directional truncation loss of the
same class the trace SL already accepts (Q-S2-1 precedent).

### 2.2 Deposits (combustion + radiative) — already energy-form; floors become hygiene

Recognition (sharpened during design): `ΔT = ΔE/(N·c_v)` **already deposits
exactly ΔE of energy** when the divisor is the cell's true N — the deposit was
never the mint; the transport was. With 2.1 in place, deposit spikes at thin
cells become energetically honest and dilute on contact. Therefore:

- Combustion (`combustion.cpp:799-803`) and radiative Pass-1
  (`temperature_solver.cpp:367`) **keep their ΔT form**. The `n_floor_heat`
  floor is retained purely as VALUE hygiene (it bounds the written T; when it
  engages it under-deposits — R3-compliant destruction, already counted via
  `heat_floor_hits`).
- **RULING (Erik 2026-08-16): `n_floor_heat` 0.05 → 0.25 ambient, one shared
  constant** with the step-4c trust gate (§2.4). Reason recorded: healthy
  flame cells sit at N ≈ 0.37–0.43 (Charles's law at the blessed plateau
  band), so 0.25 is the largest floor that leaves blessed physics untouched
  (~40% margin) while capping phantom-value gain at 4×. 0.5 would engage at
  the plateau; 1.0 would re-tune the whole fire law. Config rationale block
  (`config.toml:157-169` — records a 0.2 trial perturbing ignition timings)
  to be extended; anchors gate at P-E5 verifies the margin claim empirically.
- The object branch (`deposit >> heat_inv_shift`) is untouched — objects were
  always energy-honest through thermal mass.
- **Kirchhoff acceptance re-run (hard gate):** equal-T pairs net `rad_net`
  EXACTLY 0 on both backends (the P-R4 headline gate). The deposit fold is
  downstream of the net-flux computation, so this should be structurally
  unaffected — the gate proves it. Design task at P-E2: inventory which paths
  deposit ray heat INTO GAS (the Pass-1 floor site implies a gas-receiving
  branch) before touching anything near it.

### 2.3 Conduction passes (Pass 2 air↔air and solid↔air) — to energy form

The audit's cold-rail loop (§4.3: hull ring back-feeds a supercooled pocket
"for free") is a ΔT-denominated exchange minting energy wherever the two
endpoints' capacities differ. Rework (its own patch, P-E2):

- Per face: compute an **energy quantum** `ΔE_face = k·(T_i − T_j)·(face
  capacity factor)` — antisymmetric by construction (what leaves i enters j,
  exactly). Endpoint conversions per R2 (gas: /N·c_v with the shared 0.25
  value floor, counted; object: heat_inv_shift). Exact current Pass-2 law to
  be transcribed into the P-E2 spec before rewriting — this doc fixes only
  the constraints: face-antisymmetric ΔE, endpoint-local conversion, counted
  one-way floors.
- Self-limiting replaces back-feed: a near-empty pocket receiving ΔE has its
  T rise steeply (small N) → the gradient closes → flow stops. The reservoir
  can no longer pump indefinitely against a floored T.
- Pass-3 cooling laws / sky / ambient ring pinning are OPEN BY DESIGN (game
  cooling levers) and unchanged — the books close on *exchange* channels, not
  on deliberate sinks. The ledger records them as named legal channels.

### 2.4 Compression work — stays T-form, gains the trust gate

`dT/T = −(γ−1)·div(u)·dt` is exact per-parcel physics in which N cancels;
the term stays multiplicative on T (`eos_solver.cpp:726-784`). Two additions:

- **Trust gate (the old option-2 rider, now hygiene):** fade
  `k ← mul_q16(k, clamp01_q(recip_mul(n_bulk[i], recip_n_ref)))` with
  N_ref = 0.25 (the shared constant), applied BEFORE the ±T_WORK_CLAMP
  compare (the clamp sees the trusted k; placement changes `work_clamp_hits`,
  which only CPU↔GPU parity tests compare — safe if all three twins move
  together). Physical reading: at an unbacked cell, div(u) is not a
  measurement of real compression (the velocity field claims convergence the
  mass ledger contradicts) — the input loses authority, not the formula.
  Division-free idiom precedent: `combustion.cpp:181-192`, `:309-314`;
  `make_recip`/`recip_mul`/`clamp01_q` are FP_HD.
- The T_MIN / T_MAX_PHYS rails stay (counted; expected hit rate post-arc: 0).

### 2.5 What retires / is born / is unchanged

- **Retires:** the SL `.t` sample + `t_src_`/`tcmask_` transport role; the
  ":423 free-energy channel" comment (rewritten to state the conservation
  property); `n_floor_heat`'s stability role (remains as counted hygiene).
- **Born:** transient int64 `e_` scratch plane (not synced, not saved);
  `N_REF` = 0.25 shared constant (config-plumbed, one physical meaning:
  "quarter-ambient — the edge of thermodynamic trust"); counters
  `e_wipe_hits`/`e_wipe_sum`; the ledger's transport-conservation gate.
- **Unchanged:** mass transport, EOS solve/kick, cool laws, Huggett anchors
  (`burn_rate` 0.02, `fuel_per_o2` 0.7, `o2_frac_ext` 0.13, gas-side `H_fuel`
  4.0), the radiation law itself, all Q16.16/digest infrastructure.

## 3. Q16.16 arithmetic and determinism

- **Ranges:** `n_bulk_raw` ≤ ~65 × 65536 ≈ 4.3e6 (50×-ambient slug + margin);
  `|T_raw|` ≤ 16000 × 65536 ≈ 1.05e9 (T_MAX_PHYS). Products ≤ 4.5e15 < 2^62.
  Face flux accumulation adds ≤ 4 faces × φ_max·T_max of the same order —
  int64 throughout, no 128-bit needed (compare: mul128_shr exists if a bound
  tightens). Panel L2: verify these bounds under N_SUB_MAX=8 substeps.
- **Rounding, pinned:** e built exact (no shift); flux products exact;
  recovery is **floor division** (toward −∞, explicitly implemented — C++
  `/` truncates toward zero and MUST be adjusted for negative e; identical
  code CPU + CUDA). One-way-down bias is the R3 direction.
- **Order-pinning:** face loops iterate in the existing pinned order; energy
  accumulation is add-only into int64 (associativity-safe for the pinned
  order; CUDA twin uses the same single-writer gather pattern as
  `mg_build_levels`' device port precedent — no atomics on e).
- **No new synced state:** e is rebuilt from (N, T) inside each substep;
  digests continue to hash N and T. `digest_advect` keeps its position
  (now hashing u after SL + T after recovery — same fields, same tick point).

## 4. Edge semantics (re-derivation of the thermal-mass-axis rulings)

- **thermal_solid (`ts`) tiles:** the EOS never writes their T (ruling A1
  upheld): they are excluded from e build/recovery exactly as they are
  excluded from the T-sample today; gas flowing THROUGH a permeable object
  tile carries its energy in the flux (the object's pore gas is thin by
  ruling A3 and is not separately modeled — ACCEPTED GAP below).
- **Vacuum + ambient ring:** keep the `T := 0` wipe; ring N-clamp already
  records `boundary_flux_` per plane — energy through the ring is implied by
  (φ_gi, T_donor) and gets its own rail column only if the BC audit gates
  demand it (panel L3 to rule).
- **is_ambient interior cells (planetside):** skipped exactly as today
  (compression work skip set unchanged).

## 5. T-threshold consumer inventory (design Q9 — RULING (Erik 2026-08-16))

**Ignition STAYS temperature-based** — autoignition temperature of the fuel
surface is the physics; solid T is already energy-backed via thermal mass.
(Erik initially proposed energy-based ignition, then withdrew it on the
argument that energy-form conduction makes solid-T ignition de facto
energy-based — a phantom-hot empty cell cannot deliver the joules.)
P-E2's spec must include the verification inventory: every consumer of a
temperature threshold — solid ignition checks, any path where hot GAS
directly ignites fuel or flashes the `fuel_gas` trace plane (if one exists),
unit heat damage (the D3 radiant-flux sensor vs any legacy gas-T read),
rails/telemetry — each ruled: reads solid T (safe) / reads energy-backed gas
T (safe post-arc) / reads raw gas T where N can vanish (fix or gate).
Criterion: **no threshold may act on a temperature that is not backed by
energy.**

## 6. Patch contract (per the autonomous-patch-workflow skill)

Merge semantics: green gates commit to the arc branch; standing
auto-merge-on-green applies only WITHIN the branch (the arc is a deliberate
behavioral change — oracles are parity + conservation, not bit-identity vs
main). ONE merge to main, after P-E5 HUMAN-TEST.

| # | patch | mode | tier | oracle / gate | HUMAN-TEST |
|---|---|---|---|---|---|
| D | this doc + 4-lens panel + v2 | inline (orchestrator + Erik) | top tier, xhigh synthesis | panel blockers resolved on paper | no |
| P-E0 | hot-rail repro `tests/test_e1_hot_rail.py` (dump-anatomy E2E: sealed tight room, oversized fire load → O2 exhaustion + evacuation; asserts `t_max_phys_hits > 0` / ×~1.5-per-tick climb on HEAD) | subagent | Opus-class (defines the oracle) | RED on HEAD, deterministic, committed; if bench dials can't reach the rail → back to Erik, no harness forcer | no |
| P-E1 | CPU energy transport (§2.1) | subagent | Opus-class | ledger: transport eth ≤ 0, bounded LSB; O2 conservation unchanged; P-E0 green/bounded; suite set-diff explained | no |
| P-E2 | conduction to energy form (§2.3) + `n_floor_heat` 0.25 + threshold inventory (§5) + Kirchhoff re-gate (§2.2) | subagent | Opus-class | conduction face-antisymmetry exact; Kirchhoff exact-0 both backends; ledger; set-diff | no |
| P-E3 | CUDA twins + lockstep (§3) | subagent | Sonnet 5 (bit-identity oracle) | CPU↔CUDA tol 0: 40-tick full-engine A/B space + ambient, per-call + resident, counter parity | no |
| P-E4 | trust gate in the three step-4c twins (`eos_solver.cpp:726-784`, reference `:1500-1528`, `cuda_kick_compression.cu:213-255` + folds + `cuda_resident.h:105-125`), N_REF plumbed once | subagent | Sonnet 5 (parity oracle) | lockstep tol 0; rails bounded on P-E0 scenario | no |
| P-E5 | recalibration + bless: `fire_tune_loop` scorecard vs pre-arc; storm-ledger battery (rails zero/strictly reduced); canon fold + archive | inline (Erik + orchestrator) | Opus-class | **HUMAN-TEST: Erik plays** blowup level, two-room in-game, space/venting map, smoke/fire look, HeatMapOverlay read | **YES — merge gate** |

Memory checkpoint at every patch boundary. One git-touching agent per tree;
implementation subagents get worktrees if run concurrently (default: serial,
this tree).

## 7. Ledger acceptance (the arc's headline gate, measured by `tools/storm_ledger.py`)

After P-E1..P-E4, on the two-room bench (4800 ticks, P-F1b dials):
- transport contribution to Σ eth: ≤ 0, |·| ≤ active-cells LSB bound (today:
  +7,805 unbounded-positive);
- conduction channel: exact 0 modulo counted floor destruction;
- only combustion + radiation deposits create eth; every named sink (cool
  laws, sky, ring, floors, wipes) is one-way and counted;
- rails: `t_max_phys_hits = 0`, `energy_floor_hits = 0`, `e_wipe_sum` bounded
  as derived, on every battery row (baseline / damped / window-restored).

## 8. ACCEPTED GAPS (decisions, not findings — panel: do not re-litigate)

- **ACCEPTED GAP: KE↔eth stays uncoupled.** The kick creates kinetic energy
  from pressure without debiting eth; compression work changes eth without a
  KE counterpart beyond the div-u coupling. Closing that loop is a full
  compressible-energy reform — out of scope. The rails + trust gate bound
  its abuse; the ledger names it.
- **ACCEPTED GAP: trace gases carry no thermal energy** (0.02 pressure
  weight; their SL transport keeps its documented lossy form, Q-S2-1).
- **ACCEPTED GAP: object pore gas** (ruling A3) is not separately
  energy-modeled; the object branch's heat_inv_shift conversion stands.
- **ACCEPTED GAP: floor/wipe destruction** (deposit floors at 0.25, e-wipe at
  N_EPS, floor-div bias) — one-way, counted, bounded; preferred over
  machinery that redistributes every remainder.

## 9. Out of scope + history note

Out of scope: damping (audit §5 A/B/C — parked by Erik pending tuning; this
arc removes the explosion tail, not the Helmholtz ring); golden re-bless
(standing deferral; the fuel-bearing golden waits for blessed dials); other
ex-nihilo trace sources (P-S1 as-built §5 queue); MacCormack/BFECC sharpening
(named fallback if P-E5 reads the donor-cell T diffusion as smeared — the
consistency argument says T should be exactly as diffusive as the mass it
rides, which today it is not).

History: this arc supersedes the "E1 = options 2+3" bounded-rails patch
planned earlier the same day (2026-08-16). Its repro became P-E0 verbatim;
its trust gate became P-E4 (hygiene, not destination); its floor raise
folded into §2.2. The audit decision sheet's option E is hereby answered:
value guard alone was refuted by the dump itself (T_MAX_PHYS fired and the
blowup happened anyway); the div-u-consistency clamp survives as P-E4; the
books close at the transport layer where the actual mint lived.
