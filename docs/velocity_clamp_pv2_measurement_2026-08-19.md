# Velocity-clamp arc — P-V2 measurement (2026-08-19)

Measurement patch. Reports numbers; changes no engine code, tunes no dial,
re-baselines nothing. Compares P-V1's fixed build against
`docs/velocity_clamp_audit_2026-08-19.md`'s pre-fix numbers, per the patch
contract in `docs/velocity_clamp_impl_2026-08-19.md`.

## 1. The scripted scenario

Erik's manual session cannot be replayed bit-for-bit (it was human input on
the pre-fix build), so this measurement drives a **deterministic headless
scripted blast on `playground`** — the same level and the same south-hull
vacuum-breach geometry (`(67,10)` hull / `(68,10)` vacuum) already asserted
by `tests/test_playground_level.py`. Script:
`tools/velocity_clamp_pv2_measure.py` (seed `20260819`, 420 ticks, reusable
— reruns are bit-identical).

Mechanism, exercising both instructed paths ("destroy_wall events and/or
grenade-scale deposits"):

| tick | event | detail |
|---|---|---|
| 2 | deposit | r=4 disc @ (62,10): gas x9, T=820 K, 49 cells |
| 6 | destroy_wall | (67,10) — punches the pressurised pocket into TRUE vacuum |
| 90 | deposit | r=4 disc @ (62,36): gas x9, T=780 K, 49 cells |
| 94 | destroy_wall | (67,36) — second breach, same south-hull line |
| 180 | deposit | r=4 disc @ (62,27): gas x9, T=900 K, 49 cells |
| 184 | destroy_wall | (67,27) — third breach |
| 280 | deposit | r=4 disc @ (30,50): gas x9, T=850 K, 49 cells, **no wall break** — pure interior transport stress |

"Deposit" = pressurise+heat a disc directly (scale `gas`/`atmosphere` by the
factor, set `temperature`), mimicking an explosive payload without driving
the weapon/combat stack — the same style of direct `GameMap` poke
`tools/repro_destroy_wall_mint.py` and `tests/test_destroy_wall_conserves_mass.py`
already use. `destroy_wall` is the real engine method, the same one
`find_burst_walls`/grenades/bullet-chew call in play.

**Capture**: RAW (undequantized int32 Q16.16) fields read directly off
`GameMap` every tick — not through `PhysicsRecorder`'s float32 dequant
round-trip — so the core symptom count runs in **exact int64 /
Python-bignum arithmetic**, matching gate 1's own `rad > cap2` test rather
than a lossy float comparison. `EOSSolver` telemetry
(`dbg_last_c_local_q`, `dbg_last_n_sub`, the nine rail counters,
`ke_drag_removed`) is read every tick too; the five rail counters are
**cumulative** members (never reset by `step()`), so the script diffs them
itself, while `ke_drag_removed`/`e_drag_*` reset per tick (confirmed against
`eos_solver.cpp:324-327`).

**Apples-to-apples, stated plainly**: the pre-fix seed dump
(`debug_manual_20260818_194038_velocity_clamp_seed.npz`, Erik's session) and
this post-fix scripted run are **different scenarios on different builds** —
there is no pre-fix build available to replay this exact script against.
Every comparison below is "pre-fix session dump" vs "post-fix scripted
scenario," not a same-scenario A/B. Two things make the comparison
meaningful anyway: (1) the same audit formulas run against both datasets
verbatim, and (2) this script's numbers were validated by recomputing the
audit's own pre-fix figures from the seed dump first (see below) — every
one matches the published audit/human-test numbers, confirming the formula
implementation is correct before trusting it on new data.

**Validation against the audit's own published numbers** (pre-fix dump,
recomputed here):

| metric | audit/human-test doc | this script |
|---|---|---|
| supersonic violations | 52,865 | 52,923 |
| violation snapshots | 255/775 | 255/775 (exact) |
| max own-cell Mach | 2.47 | 2.4746 |
| P_min | −1.324 | −1.3237 |
| worst cell x-ambient | 433.5 | 433.46 |

(The 58-count difference in the violation total is explained below — a D4
approximation this script cannot avoid on either dataset.)

## 2. Symptom table, pre-fix vs post-fix

Same formulas throughout: `c_amb=300`, `T_amb=290`, `s_eos=1`, D1 ambient
floor, cap² computed in int64/Python-bignum per cell
(`(⌊√cap²⌋+2)²` slack, matching gate 1's assertion form exactly).

**Two stated approximations, both from data the dumps don't carry:**
- **D4 (ts routing)**: neither dataset carries a separate `thermal_solid`
  mask, so a furniture/door cell's cap is folded from *its own* stored
  temperature (the object's T) rather than routed to the ambient cap the
  live engine uses (D4). This inflates the violation count slightly on
  both sides — it is the source of the 52,923-vs-52,865 gap above.
- **is_vacuum exclusion**: the kick's skip-set is `solid||is_vacuum||
  ambient-ring` — a cell that just joined `is_vacuum` (a fresh breach)
  keeps whatever wind it held the instant before and is never touched by
  the kick again. This script's own post-fix capture carries `is_vacuum`
  directly and excludes it (confirmed to matter: total open cell-snapshots
  dropped from 2,641,340 to 2,360,817 once applied, though the violation
  count was unaffected — see below). The pre-fix dump predates that field
  (`PhysicsRecorder.DEFAULT_FIELDS` has no `is_vacuum`) and stays
  solid-only, the same limitation the original audit had.

### 2a. Own-cell supersonic violations — the patch's central claim

The audit measured cap violations using each snapshot's **own** (post-tick)
temperature. This script reproduces that exactly for comparability, and
**additionally** recomputes the post-fix run using the cap's actual fold
basis — **tick-entry** temperature (D2v2: "the cap derives from
TICK-ENTRY temperature") — since this script, unlike the dump, has the
tick-entry state available.

| basis | pre-fix (dump) | post-fix (scripted) |
|---|---|---|
| same-snapshot T (audit's method) | **52,923** violations, 255/775 snapshots, max ratio **2.4746** | **73** violations, 29/421 snapshots, max ratio **1.3178** |
| tick-entry T (the fix's actual fold basis) | n/a (dump doesn't retain entry-vs-exit distinction per field) | **0** violations, 0/421 snapshots, max ratio **0** |

The 73 same-snapshot-T "violations" were investigated cell-by-cell
(low-T cells at 1.00–1.14x, one at 1.32x, all near the fresh breach
during the first ~20 ticks — see the diagnostic trace in the session
log). Recomputing those exact cells against **tick-entry** T instead —
the basis the engine's own kick clamp actually reads — makes every one of
them vanish, confirming they are the same tick-entry-vs-tick-exit
temperature drift the audit itself named as the explanation for its own
residual 10% ("nothing exceeds 1.1x cap... no unexplained mechanism
remains"); this run's single 1.32x outlier is fully absorbed by the same
correction. **On the basis the clamp actually uses, zero stored velocities
exceeded their own cell's cap anywhere in 421 snapshots x up to ~5,600 open
cells each.** This is the strongest evidence in this report: gate 1's claim
holds not just in the C++ unit harness but in a full, real, multi-event
blast run through the live engine.

### 2b. The rest of the table

| metric | pre-fix (dump, 775 snaps) | post-fix (scripted, 421 snaps) |
|---|---|---|
| P_min | **−1.324 atm** | **−0.310 atm** |
| worst cell x-ambient | **433.5x** | **299.4x** |
| peak single-tick cell-gain (excl. deposit ticks) | **327.75** (snap 540, a 12-wall-break event) | **196.78** (tick 18, cell (66,19)) |
| u_clamp_hits over the run | n/a (pre-fix counter not meaningful — the clamp fired against the wrong, global number) | **7,432** |
| u_max_hits over the run | n/a | **0** |
| work_clamp_hits over the run | n/a | **20,728** |
| energy_floor_hits / t_max_phys_hits | n/a | **0 / 0** |

`u_max_hits = 0` matches P-V1's own prediction exactly: U_MAX becomes
reachable only where `T_abs/T_amb >= 11.1`, i.e. `T >~ 2930`; this
scenario's hottest deposit was 900 K, so the rail structurally cannot
bind here. Not a gap — the as-built doc already named this as the thing
P-V2 should go measure, and it measures as expected.

`u_clamp_hits = 7,432` over 420 ticks (vs the P-V1 golden trajectory's 4
hits in 30 quiet ticks) confirms the clamp is genuinely load-bearing on a
real blast path, not a dormant safety net — consistent with the audit's
own "it fires constantly" framing, now firing against the *right* number.

**Pile-up magnitude dropped (433x -> 299x worst-cell; 328x -> 197x peak
single-tick gain) but did not vanish** — exactly the design doc's stated
non-claim ("DOES NOT CLAIM that the >=100x pile-ups... vanish").

## 3. Required n_sub vs the N_SUB_MAX=8 rail

Reconstructed per-tick from `eos_solver.cpp:470-536`'s own formula
(`u_est = min(max|u| + max(K|deltaP|/N_hat)*dt, max(c_local, U_MAX))`,
`required = ceil(dt*u_est/(CFL_ADV*dx))`), replayed in real (dequantized)
units — no int64 overflow risk exists at this scale, since real-unit
magnitudes stay small (T <= ~1000, cap <= 1000 m/s); the overflow the
design doc warns about is specific to the C++ raw-Q16.16 pipeline, not this
recompute. **Validated**: `min(required, N_SUB_MAX)` matched the engine's
own `dbg_last_n_sub` telemetry on 419/420 ticks (the one mismatch is a
`ceil()`-boundary rounding artifact of this reconstruction, not a real
divergence).

```
required n_sub: min=1  max=251  mean=249.6  median=251
fraction of ticks where required > N_SUB_MAX(8): 0.995  (418/420)

  [    0,    1) : 0
  [    1,    2) : 2      <- the two calm pre-blast ticks
  [    2,    4) : 0
  [    4,    8) : 0
  [    8,   16) : 0
  [   16,   32) : 0
  [   32,   64) : 0
  [   64,  128) : 0
  [  128,  256) : 418    <- almost the entire run
  [  256, 100000): 0
```

**This is worse (more saturated) than the audit's earlier estimate of
"required ~140 at spike ticks"** — and the likely reason is a real
difference in scenario shape, stated honestly: the audit's critique
measured required n_sub at transient blast-spike ticks in a live-fire
session, whereas this script's `destroy_wall` calls punch a hole directly
into **true, permanent vacuum** (`P=0`, Dirichlet-pinned) that stays open
for the rest of the run. A punctured hull is not a one-off pulse — it is
an ongoing decompression front, and the near-zero-N cell right at the
breach edge (floored at `N_FLOOR_SOLVER`) keeps the local `|deltaP|/N_hat`
term enormous for as long as the hole stays open, which is most of this
420-tick run once the first breach lands at tick 6. This is a **genuine,
physically real regime** (a real breached hull behaves exactly this way)
but it is a different regime from "spike at the moment of the blast," and
the two numbers (140 vs ~250) should not be read as the same measurement
repeated — they bracket the rail-pressure problem from two ends: a
transient spike, and a sustained open breach.

**Either way, the rail binds essentially always once anything breaches.**
This patch does not touch `n_sub`/`N_SUB_MAX` (by design, out of scope) and
the numbers confirm that ceiling, not the velocity ceiling, is now the
active bottleneck on resolving these events.

## 4. Clamp-energy note (best-effort estimate)

The clamp removes kinetic energy with no ledger counterparty
(`u_clamp_hits` counts *events*, not the energy each one removes). This
cannot be isolated cleanly from outside the engine — total system KE moves
for many reasons each tick (pressure-gradient acceleration, drag, advective
transport out of the measured region, compression work), and the clamp's
own share is mixed into all of that. What can be measured:

```
total system KE (open cells, real units)     : peaked 21.9M (tick 5, first deposit)
sum of every single-tick KE DROP over the run : 80.9M
sum of ke_drag_removed over the run           : 144.1M   (booked, per-tick reset, confirmed)
```

The booked drag channel alone (`ke_drag_removed`, `k_drag=0.5` shipped) is
larger in total than the sum of all measured KE drops across the run — i.e.
drag is not a minor channel here, and the clamp's unbooked share is
happening inside a budget that a large, *already-booked* term dominates.
This does **not** prove the clamp's own loss is small (a single clamp event
can still remove real energy at a shock front, and 7,432 events over 420
ticks is not nothing), only that it is not obviously the dominant term next
to drag at this scenario's density factor. **Labeled explicitly as an
estimate**: a precise clamp-energy ledger would need a counter inside the
kick itself (comparing pre-/post-clamp `|u|²` the way `ke_drag_removed`
already does for drag) — not something this measurement patch can add.

## 5. Verdict

**The fix delivered its central claim.** Measured on a real, deterministic,
multi-event blast run through the live post-fix engine — not just the C++
unit gates — zero stored gas velocities exceeded their own cell's
sound-speed cap when compared on the cap's actual tick-entry-temperature
basis (421 snapshots, thousands of open cells each); the small residual
that appears under the audit's own same-snapshot-T methodology (73
violations, max 1.32x) is fully explained by the same tick-entry-vs-tick-
exit temperature drift the audit itself predicted, and disappears entirely
once the correct basis is used. `u_max_hits` stayed structurally zero,
exactly as P-V1 predicted for a scenario this cool. **What remains is
exactly what the design doc said would remain, and it is not small**: pile-up
magnitude dropped (433x -> 299x worst cell; 328x -> 197x peak single-tick
gain) but stayed well above ambient — spikes attenuated, not gone, as
promised — and the substep rail is saturated almost the entire run once
anything breaches (99.5% of ticks needed `n_sub` far past `N_SUB_MAX=8`,
median ~251 required against a rail of 8), somewhat worse than the audit's
earlier transient-spike estimate of ~140 because a punctured hull to true
vacuum is a sustained condition, not a momentary one. `u_clamp_hits` firing
7,432 times over 420 ticks confirms the clamp is genuinely load-bearing on
this path, and its own energy removal remains unaudited (best-effort
estimate only, no hard number) though not evidently dominant next to the
already-booked drag channel. For HUMAN-TEST: expect the illegal-velocity
symptom (Mach 1.8-2.5 own-cell flow, negative pressure driven by it) to be
gone, and the flashing/pile-up symptom to be visibly smaller but still
present, especially right at any fresh breach — which is exactly what P-V3
should confirm by eye. For the N_SUB_MAX ruling: the rail is the load-
bearing bottleneck now, and it binds on nearly every tick of any blast at
this scale, not just a rare spike.

## Reproduction

```
conda run -n data python tools/velocity_clamp_pv2_measure.py --n-ticks 420
```

Deterministic (fixed seed `20260819`); reruns are bit-identical. The pre-fix
comparison loads `debug_manual_20260818_194038_velocity_clamp_seed.npz` from
the repo root by default (`--pre-fix-dump` to override) — that file is
untouched by this patch.
