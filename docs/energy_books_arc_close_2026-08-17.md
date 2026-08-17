# Energy-books arc — CLOSE (as-built, 2026-08-17)

**Status: the arc is CLOSED. Erik played the build and blessed it at the P-E5
HUMAN-TEST (2026-08-17), with the shipped dials `k_drag = 0.5` /
`k_drag_heat_frac = 0.0014` (`39e07f6`). Branch `storm-damping`, ~21 commits
from `7e6be0d` (post-P-S1 HEAD) to the close.**

This is the single document a future reader needs. The arc's own working
record — design v2.2, the round-1 critique panel with its committed WRITER
TABLE addendum, and the seven per-patch as-builts — is archived intact under
`docs/archive/`:

| doc | what it holds |
|---|---|
| `archive/energy_transport_design_2026-08-16.md` (v2.2) | the contract: the four rules, every pass's mechanism, §2.9 the deferred cold-rail engine, §8 the accepted gaps |
| `archive/energy_transport_critiques_round1_2026-08-16.md` | the 4-lens panel verdicts + the L3 T-WRITER TABLE (the completeness oracle) |
| `archive/e1_p_e0_asbuilt_2026-08-17.md` | repro + instruments |
| `archive/e1_p_t0_asbuilt_2026-08-17.md` | traces leave the physics books |
| `archive/e1_p_e1_asbuilt_2026-08-17.md` | energy transport, CPU+CUDA — the core patch |
| `archive/e1_p_e2a_asbuilt_2026-08-17.md` | conduction in energy form + the per-face limiter |
| `archive/e1_p_e2b_asbuilt_2026-08-17.md` | deposit dial + the T-threshold consumer inventory |
| `archive/e1_p_e3_asbuilt_2026-08-17.md` | interior drag with a heat counterparty |
| `archive/e1_p_e4_asbuilt_2026-08-17.md` | trust gate + reversible work |

The arc's origin — `docs/storm_audit_2026-08-14.md` — stays in `docs/`
alongside this record. The as-built LAW is folded into canon:
`architecture/engine/04_atmosphere_and_pressure.md`,
`05_smoke.md`, `06_temperature_and_fire.md`.

*(Pointer note: a handful of code and test comments still cite these docs at
their pre-archive `docs/<name>.md` paths — `physics_runner.py`'s stale-key
guard and three test module docstrings. Harmless; fix at the next touch of
those files.)*

---

## 1. The problem, as measured

Temperature was transported by semi-Lagrangian **copy**
(`eos_solver.cpp`'s fused SL sample; the source itself called the mechanism a
"FREE-ENERGY channel"). Terms with collapsing denominators wrote enormous T
into near-empty cells, and the copy then pasted those values onto real mass.
T × mass is energy, so the copy was a **mint** — energy created from nothing,
every tick, invisible to every conservation gate the repo had, because no gate
was denominated in energy.

Three measurements defined the arc:

- **The bench mint.** EOS transport injected **+7,805 eth per 200 s bench
  run** (audit §4 pump-off row) — and it survived P-S1, the patch that killed
  the ex-nihilo smoke scatter. Re-measured on-tree after P-T0 on the same
  command: **+8,022**.
- **The hot rail (in-game blowups, reproduced at bench scale by P-E0).** In a
  starved, fully evacuated combustion pocket, step-4c compression work
  multiplied one cell's T by **×1.4972/tick for 19 consecutive ticks** (the
  ×1.5 = 1 + `T_WORK_CLAMP` rail signature), pinning T at the `T_MAX_PHYS`
  ceiling — `t_max_phys_hits = 2130` in 2,000 ticks — and throwing a **97.5
  atm** pressure spike when gas slammed back into the pocket.
- **The cold rail (the window row).** A supercooled pocket descended to the
  T_MIN floor (`t_min_gas` −288.78 on HEAD), then the pinned hull conducted
  into it for free and the kick converted the standing pressure deficit to
  wind.

**Erik's ruling at the design gate:** bounded rails were rejected as the
destination ("we still mint energy, just less — a very unstable system").
Close the books *structurally*, while goldens were already deferred and a
recalibration was already scheduled.

**The principle that replaced the copy** (design §1): *every thermal exchange
is denominated in energy; temperature is what energy looks like through a
cell's actual mass.* Transport conserves (energy rides the same conservative
donor-cell face fluxes bulk mass rides); conversions are endpoint-local;
one-way guards are all counted **in energy units**; determinism is unchanged
(Q16.16/int64, order-pinned, CPU↔CUDA bit-identical, no new synced planes).

---

## 2. Headline results

### 2.1 The mint is closed, and the books close to an identity

| observable (committed HOT scenario, 2,000 ticks) | HEAD (P-E0) | after P-E1 |
|---|---:|---:|
| `eth_transport_delta` run total (raw Q16.16²) | **+3.72e16** | **−5.33e14** |
| worst single tick | +3.80e15 | **0** |
| ticks with a positive delta | 916 / 2000 | **0 / 2000** |
| §7 truncation-allowance violations | 901 | **0** |
| measured active-flux fraction | — | 0.4374 |

The transport pass is now **one-way non-positive on every tick** — not merely
bounded. Better than a bound: on a sealed map the new gate
(`cuda_bulk_flux_check` PART 3) asserts an **identity** every tick on both
backends — `eth_transport_delta = −e_ts_residual − e_wipe_sum + e_floor_sum +
trunc`, with `trunc ∈ (−n_bulk_active_sum, 0]` — and all five new counters are
bit-identical CPU↔GPU.

Conduction closes the same way after P-E2a: `Σ ΔE == 0` **exactly** in int64
across the grid, so `Σ ΔT·C_real == e_cond_trunc_sum + e_cond_cap_sum` per
tick, gated on 150 ticks of a heterogeneous field, 400 ticks of the sealed-room
E2E, and 121 synthetic CUDA configs. Face antisymmetry is exact — worst
residual **0** over 218,396 live-face samples, verified against an independent
Python transcription that reproduces the C++ field bit-for-bit.

### 2.2 The ledger

EOS-pass gas-thermal injection on the 200 s pump-off row:

| | audit (origin) | P-T0 | **P-E1** | P-E2a | P-E4 |
|---|---:|---:|---:|---:|---:|
| `eos` row `eth_gas` | +7,805 | +8,022 | **+275.1** | +291.9 | (row stable) |
| `eos` row bulk-N creation | — | **0 LSB** | 0 LSB | 0 LSB | 0 LSB |

A **29× fall**, closing the audit's original grievance to ~3.5 % of its value.
What remains on that row is the compression pass, not transport — the named,
accepted §2.9 gap (§5 below).

### 2.3 The hot rail is gone

| observable | P-E0 (HEAD) | P-E1 | **P-E4** |
|---|---:|---:|---:|
| `t_max_phys_hits` | 2130 | 46 | **0** |
| `work_clamp_hits` | 5681 | 247 | **0** |
| peak gas T (game) | 15984.5 | 15981.0 | **3702.35** |
| longest sustained climb | 19 ticks ×1.4972 | — | **5 ticks ×1.05–1.08** |

The geometric runaway is gone; what survives is a five-tick linear-ish drift
at the trust band's half-to-full transition (`n_bulk` 0.153–0.225), exactly the
partial fade the design predicted and did not claim to eliminate.
`test_no_rail_hits` — carried as a declared cross-rung xfail from P-E0 through
P-E2b — **flipped strict and green at its owning patch, P-E4**.

### 2.4 The cold rail's free leg is gone

The reservoir loop's conduction leg — the pinned hull conducting into the
supercooled pocket "for free" — was the old ΔT-form law moving equal ΔT across
a hull↔air face whose endpoint capacities differ by ~32×, i.e. moving 32× more
energy into the light side than it took out of the heavy one. Energy-form
conduction priced both ends by their own capacity, and the leg **flipped sign**:

| window row (4800 ticks, damp 0.005, strip 0.5) | PRE (P-E1) | POST (P-E2a) |
|---|---:|---:|
| `t_min_gas` minimum over the run | **−0.1908** | **0.0000** |
| `tail` pass `eth_gas` | −20.67 | **+0.8128** |
| `eth_compression_delta` run total | +3.358e10 | +9.612e9 → **1.044e9** (P-E4) |

`t_min_gas` never leaves ambient on that row again, through P-E3 and P-E4.
§2.9's engine (compression acting on a negative game-T) has nothing to act on
there — it is documented, not fixed, and owns its own queued patch (§5).

### 2.5 The anchor scorecard (the fire's own numbers)

Measured at the P-E1 boundary — the MacCormack-fallback decision point — with
**no dial retuned on either side**:

| metric | PRE (pre-arc HEAD) | POST | target | verdict |
|---|---:|---:|---|---|
| peak I | 0.748 | 0.751 | 0.4–0.6 | MISS → MISS (pre-existing) |
| **peak time** | 2.29 min | **2.00 min** | ~3 min (2–5 ok) | **PASS → MISS** |
| fire death | nan | nan | 6–8 min | MISS → MISS (never dies, both) |
| flame plateau T (game) | 390 | **387** | 400–500 | MISS → MISS (−0.8 %) |
| **far-field T rise (game)** | 0.6 | **0.0** | ≤ 20 | PASS → PASS |
| room N_total min | 1.000 | 1.000 | ≥ 0.9 | PASS |
| far-field X min | 0.2061 | 0.2065 | ≥ 0.19 | PASS |

**One verdict flip in the whole scorecard** (`peak time`, out at the low edge
of its band), against a −0.8 % plateau temperature and one genuine win: the
far-field warming went to **exactly zero**. That 0.6 game-deg of far-field rise
*was the mint made visible* — cells warming at range with no energy ever having
been delivered to them.

**Flame-cell `n_bulk` histogram** (the question the whole law change turns on:
what is the denominator doing under the flame?): mean **0.4476 → 0.4346**
(0.4344 after P-E2b's floor change), p0 0.3954 → 0.3938, and **not one sample
migrated below 0.25** — every bin under 0.25 stayed empty. The failure mode
MacCormack was named as insurance against — flame cells sitting where `T = e/n`
is rounding-dominated — does not materialise at these dials, so the fallback
was not taken.

---

## 3. The three latent bugs found en route

None of these were the arc's target. All three were found by building the
instruments the design demanded, and all three are reported here because each
was silent — no gate in the repo could see it.

1. **The deposit-narrowing overflow, at BOTH deposit sites (P-E2b, CPU+CUDA).**
   The old two-step chain `mul_q16(deposit, recip_n)` → `recip_mul(·, recip_cv)`
   narrowed `deposit/N` to Q16.16 int32 (magnitude ceiling ~32,768) *before*
   dividing by `c_v`. At the new `n_floor_heat = 0.01` a routine ~330 per-tick
   deposit is already `330/0.01 = 33,000` — past the ceiling. The narrow wrapped
   (typically to a large negative), and `heat_saturating_add`'s `delta <= 0`
   early-return then **dropped the entire deposit — no clamp, no counter,
   nothing.** It was already reachable at the *pre-arc* 0.05 default for a
   stacked-firestorm deposit (`2600/0.05 = 52,000`); the config's own comment
   believed `T_MAX_PHYS` would bound that case, which it cannot, because the
   corruption happens in an intermediate before `T_MAX_PHYS` ever sees the
   value. Fixed with a shared wide helper (`deposit_dT_wide_q16` + its CUDA
   twin): one 128-bit product, narrowed exactly once, to int64. An honestly
   huge deposit now reaches the counted rail through a value that was never
   corrupted on the way there.

2. **`mul128_shr` / `mul128_shr_signed` undefined behaviour at `shift == 0`
   (P-E3, CPU+CUDA).** Both helpers' hi:lo recombine does `hi << (64 − S)` — a
   64-bit shift by 64 when `S == 0`. Every pre-existing call site used 16/32/48;
   the drag oracle's value-sum counters are the codebase's **first caller
   needing shift 0**. On the GPU it produced silently, reliably wrong counters
   (`−576` = one −1 per cell of a 24×24 grid) while the CPU path — sharing the
   identical UB expression — happened not to diverge on the tested inputs. Fixed
   by special-casing `S == 0` on both backends, purely additively (no existing
   caller's behaviour changes).

3. **A parity gate that had stopped asserting what it claimed
   (found P-E2a, root-caused and repaired P-E4).**
   `test_cuda_p64_kick_compression` PART 2 compared an advection-replay digest
   against `EOSSolver.digest_advect`. P-E1 made the replay reference **u-only**
   and moved `digest_advect` across the flux call to hash T-after-recovery —
   a 2-field chain against a 3-field chain, **structurally incomparable**, not
   merely stale. Worse, its field-level "ground truth" check was also broken:
   the tick-start T snapshot it fed the isolated tail is no longer advanced by
   anything, so it is stale from tick 1. Instrumented reproduction showed
   `digest_velocity` matching the real solver **exactly** while only
   `digest_compression` diverged — proof that the kick (which never reads T) was
   fine and only 4c was downstream of the unreconstructible T. PART 2 was
   narrowed to what remains true: wind ground-truth against the real engine,
   plus full CPU-ref-vs-GPU bit-identity on every field, digest and counter.
   The T-side coverage was never lost — it lives at its rightful owner, P-E1's
   `cuda_bulk_flux_check` PART 3. **The gate no longer claims, and must never
   again be read as claiming, that the isolated tail reproduces the engine's
   post-tick temperature.**

---

## 4. What shipped, and what Erik rejected

**Shipped dials** (`39e07f6`, config.toml `[physics.eos]`):

| dial | shipped | note |
|---|---:|---|
| `k_drag` | **0.5** | interior momentum drag, per-second rate |
| `k_drag_heat_frac` | **0.0014** | the physical-air anchor |
| `n_work_ref` | 0.25 | trust-gate reference N (power-of-two reciprocal is exact here) |
| `n_floor_heat` | 0.01 | was 0.05; now a low value-hygiene dial |

**`k_drag = 0.5` is a STARTING value, not a tuned one** (Erik's P-E5 ruling):
real tuning waits until the pressure-transient arc lands, then one more retune
pass over everything. The audit's own suggested band (0.02–0.05) is far too
weak — P-E3 **measured** the free-momentum e-fold at `k_drag = 0.02` as
**≈49.6 s**, where the design doc had estimated ≈2.1 s (the estimate omitted
the `dt = 1/24 s` factor that `kd_q = quantize(k_drag·dt)` requires). A ~2 s
e-fold needs `k_drag ≈ 0.5`. The same correction applies to the neck-heating
figure: measured **≈8.05 game-deg/s** at a sustained 20 m/s draft, not the
design's ~96.

**Why `k_drag_heat_frac = 1.0` was REJECTED — and this is the arc's sharpest
lesson.** The full deposit kept the conservation oracle exact through every
gate, and every gate passed. Then the in-game HUMAN-TEST **detonated**: the
deposit scales with **u²**, so at blast velocities an explosion's own wind
self-immolates into heat. The dump (`debug_blowup_20260817_051730`) shows T
pinned at the 16000 ceiling across **739 cells**, with pressure following it to
**66 atm**. The two-room bench never sees this — it peaks at **7.7 m/s**
against blast-scale hundreds — which is exactly why every automated gate was
green at 1.0. This is the failure the round-1 thermo lens predicted. The
physical-air anchor 0.0014 is what ships.

**The arc's one declared new red** (47 → 48 failed):
`tests/test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing`.
With damping live, a spray cone rides the wind and damped air shortens its
throw. Feel-adjacent and expected; **owned by the post-pressure retune pass**,
not by this arc. Left honestly red rather than xfail-papered.

**Blessed suite state: 48 failed / 2186 passed / 5 skipped.** Of the 48, the
large families are pre-existing and unrelated to this arc (the shared canonical
CUDA golden across 12 files — unmoved through the whole arc, because that A/B
scenario carries `temperature` identically 0 on every cell and every tick,
making any T-based law an exact no-op there — plus the fire-realism,
logic-golden and cool-shift-axis families that were red before the arc began).

**One red resolved itself.** `test_air_boundary::test_ambient_gate3_udamp_band_absorbs_reflection`
went red at P-E1 (its *relative* leg failed because the pin-only reflection
baseline collapsed from 2.19 % to 0.81 % — the absolute ≤2 % safety leg passed
throughout with margin) and was carried unfixed, awaiting Erik's
gate-semantics ruling rather than being retuned by an implementation agent. At
P-E4 it **passed again without being touched**: the law change restored the
baseline to 1.95 %. The ruling was never made and is now moot in practice.

---

## 5. Open items — what this arc deliberately did not close

### 5.1 The cold-rail engine: compression work on absolute temperature (design §2.9)

Step 4c multiplies **ambient-relative** T. Below ambient (`T_rel < 0`) the
compression branch makes a cold cell **colder** — compression *freezes*
sub-ambient gas. This is the game-T-vs-T_abs gap not merely omitting physics
but **inverting** it below ambient, and it is the cold-rail window's true
engine. (The earlier "standing-wave refrigerator" attribution was retracted at
design v2.2, on measurement.)

The honest fix is specified: run the (now reversible) work on absolute
temperature — `T_new = (T + 290)·(1±w) − 290` — so compression warms
sub-ambient cells and ambient air finally heats under compression at all,
restoring the missing acoustic thermalization that is the one physical damping
channel of the Helmholtz mode.

**RULING R1 (Erik):** not in this arc. It re-opens a ruled accepted gap
mid-arc and is feel-adjacent everywhere (breach rarefaction becomes genuinely
cold — ~97 game-deg at the clamp versus 0 today; venting acoustics change). It
lands as **its own short designed patch with its own critique round and its own
HUMAN-TEST**. Within this arc the window died operationally (§2.4) and the
engine is documented rather than left a mystery.

### 5.2 Three third-class T-threshold consumers (P-E2b's inventory)

P-E2b walked every read-side consumer of a gas-temperature threshold, using the
critique's L3 writer table as the completeness oracle. Every ignition and
emission decision in the engine is masked to `thermal_solid`/`flammable`
(object T, thermal-mass-backed) or gated by Kirchhoff absorptivity — **safe**.
Three consumers read raw, N-unguarded gas temperature. All three are
**reported, not fixed** — a threshold change is feel-adjacent and Erik's call:

1. **The EOS CFL sound-speed max-reduction — `eos_solver.cpp:347-351` + its
   CUDA reference twin. This one is SIM-AFFECTING and is the serious member of
   the set.** It takes a MAX of `t_abs` over all `!solid && !is_vacuum` cells —
   **no `n_bulk`/N-floor guard** — and that maximum feeds the local sound speed,
   which caps the velocity estimate and therefore **determines `n_sub`, the
   substep count for the whole tick's advection**. Post energy-books
   (`T = e/n_bulk`), one thin-N cell with a rounding-dominated T can dominate
   that reduction over the entire open-air field and change the substep count
   for every cell. Contrast `p*` in the same function (`p* = C·N·T_abs`), which
   IS N-weighted, so the N in the product cancels the thin-N reciprocal back
   down — benign by construction. This reduction has no such weighting. Any fix
   belongs to whoever owns the CFL/substep design.
2. **The `temperature` sensor's area-mean** — `sensor_accessor.py:154-175`,
   reachable from any level's `temperature` sensor with `area_m > 0`. The
   single-tile path deliberately avoids gas T (its own docstring explains that
   "a faced-air sample would read a plume-advected gas value"), but `area()` is
   channel-agnostic and takes an unweighted arithmetic mean of raw
   `gmap.temperature[]` over surrounding open-air tiles, with no N-weighting and
   no guard. Not exercised by any test today; wire-able by a level author.
3. **Render fire-light selection** (`renderer/fire_lights.py`,
   `renderer/blackbody.py`) — cosmetic, no gameplay consequence, but a noisy
   thin-N spike could in principle out-select genuinely energetic tiles.

### 5.3 The pressure transient — Erik has scoped this as its OWN NEXT ARC

**The arc closed the THERMAL books, and the in-game dumps confirm it**: the
2026-08-17 05:10 dump peaks at **T = 741** — normal fire range, with **zero
cells near the 16000 ceiling** that defined every old blowup. But **pressure
transients remain**: ~**98 atm at a normal ~700 game-T**. Run that through
`p* = C·N·T_abs` and it implies **~29× ambient density in one cell** — plus a
**negative `P_min` (−0.98)**. That is a **mass/momentum event, not a thermal
one**, and nothing in this arc addresses it.

The next arc is instrumented and ready: the recorder's `DEFAULT_FIELDS` now
captures **`wind_x`, `wind_y` and `inert_n2`** (`df088f1`), so the next pop is
diagnosable offline. Wind is **not** recoverable from the pressure field — the
gradient gives the per-tick *acceleration* while `u` is its accumulated
history, and the two run ~90° out of phase in the Helmholtz mode; the storm
audit named this exact gap. `inert_n2` joins `gas_o2` so `p* = C·N·T_abs` is
decomposable offline. Cost: the ring buffer grows ~336 MB → ~500 MB at the
default 2400 slots. **Audit first, per Erik's standing ruling.**

### 5.4 Accepted gaps carried forward (design §8 — decisions, not findings)

- **KE↔eth is HALF-coupled.** The kick still mints KE with no eth debit;
  §2.8's drag now launders that mint into eth. Small at bench scale, but it is
  a positive-feedback path at blast scale — named here rather than discovered
  later. The kick-side debit is the open half.
- **Compression work multiplies game-T, not T_abs** — sharpened and scheduled
  as §5.1 above.
- **§2.7's per-cycle residual** — ≤1 LSB, one-way, measured at P-E4 (exactly 0
  at the clamp in both cycle orders and both signs of T); a bounded named
  integer ratchet replacing an unbounded proportional leak.
- **Traces carry no thermal energy** — now trivially true (§6).
- **Object pore gas not separately modeled**; the ts-face rule (d) residual is
  counted; the honest gas↔object convective exchange is named as a future
  upgrade, not built.
- **Water/steam thermal coupling absent** (verified: `water_solver.cpp` has no
  thermal contact). Steam citizenship is the water arc's decision, via the
  full-citizenship recipe in design §2.6.
- **The `eth_transport_delta` / `eth_compression_delta` bracket counters are
  CPU-only.** Deliberate: pure instrumentation, digest-inert, read by no parity
  gate, and every consumer runs on the CPU backend. The five load-bearing P-E1
  counters ARE on both backends and gated bit-identical. Named gap if a
  GPU-side ledger is ever wanted.

### 5.5 Sequencing note (design §9)

**This arc landed BEFORE any recorder milestone that snapshots physics for
training.** That was the point of the sequencing: the recorder must capture a
substrate whose books close, or every trajectory in the replay buffer carries
a mint. Recorded in the priority ledger for the RL push.

---

## 6. What retired

- **Both semi-Lagrangian T-copiers**: the fused SL sample's `.t` slot (live
  step, CPU reference twin, both GPU dispatch paths) and the dormant second
  copier, `TemperatureSolver` Pass 0b, with its ~110-line file-local sampler
  and its CUDA twin — deleted identically on both backends, so a wind-passing
  caller now simply gets no advection and the twins still agree.
- **The A2 `t_occlude` / `tcmask` machinery**, which existed only to serve that
  T write.
- **`trace_mass_scale` and the trace-decay→N₂ credit.** Traces left the physics
  books entirely at P-T0 (Erik's 0 % ruling) — see the ch.05 canon fold.
- **The old ΔT-form conduction law** and the plain-`Σ T` metric that gated it
  (that metric was blind to the sealed room's largest energy channel; the
  replacement asserts an identity against named counters and would have caught
  the original defect).
- **The v1 "0.25 shared constant" ruling**, retired by the 0 % + low-floor
  rulings.
- **The P4 doctrine "decay is oxidation, not deletion"** — deliberately
  retired, with written rationale (design §2.6): at zero pressure weight there
  is no mass to conserve.

---

## 7. The gate ladder, for the record

| patch | what landed | headline gate |
|---|---|---|
| P-E0 | hot-rail repro + cold-rail window + N≈0.15 pocket + the bracket counters | counters provably inert (suite set + bench digests byte-identical, twice); reds on HEAD documented |
| P-T0 | traces out of all three Dalton families + the decay credit deleted (incl. a CUDA-resident twin the design's site list missed) | bulk-N creation inside the EOS pass = **0 LSB**, with fire alive |
| P-E1 | energy transport, CPU+CUDA in one patch | mint closed (§2.1); books close as an identity; CPU↔CUDA tol 0; anchor scorecard + histogram at the boundary |
| P-E2a | conduction in energy form + the per-face limiter | `Σ ΔE == 0` exactly; face antisymmetry exact; Kirchhoff re-gate green both backends |
| P-E2b | `n_floor_heat` → 0.01, `n_work_ref` plumbing, deposit drop counters, the T-consumer inventory | dial change measured behaviorally INERT on every live scenario; the overflow bug (§3.1) |
| P-E3 | interior drag with a heat counterparty | drag identity exact per tick (worst relative slack 1.48e-7); dormancy byte-identical at the then-default 0.0 |
| P-E4 | trust gate + reversible work + the P6.4 gate repair | `t_max_phys_hits` → **0**; the §2.7 unit oracle measured; window row sets §7's expectations |
| P-E5 | recalibration + **Erik's HUMAN-TEST** | **blessed** — and the `k_drag_heat_frac` 1.0 detonation that only in-game play could find (§4) |

Every rung carried CPU and CUDA in the same patch (or, at P-E1, in two
back-to-back commits with an explicitly labelled one-commit parity window), so
the branch is parity-whole at HEAD.
