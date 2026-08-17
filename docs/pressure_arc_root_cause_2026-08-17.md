# Pressure/momentum arc — ROOT CAUSE FOUND (2026-08-17)

**The storm is not physics. It is the pressure solve running under-converged.**
`mg_cycles = 2` is too few at real map sizes; the unconverged residual is
re-injected every tick and drives the whole phenomenon.

Status: **diagnosis, measured and reproduced headlessly. No engine code
changed. Nothing merged.** Erik blesses before anything lands.

---

## 1. The headline

On `levels/playground` — the level Erik played when the storm was recorded —
8 furniture fires, 3600 ticks (150 s), CPU backend, everything else shipped:

| | `mg_cycles = 2` (SHIPPED) | `mg_cycles = 8` |
|---|---|---|
| **P_max** | **103.239 atm** | **1.405 atm** |
| **P_min** | **−4.536 atm** | 0.995 atm |
| RMS\|P−1\| deciles | 0.133 … 0.421 → **0.749** | 0.0024 → 0.080 |
| `dbg_last_n_sub` | **8** (pinned at cap) | **1** |
| `u_clamp_hits` | **69,672** | **0** |
| `work_clamp_hits` | **386,835** | **0** |
| fire total, peak | 79.0 | 32.9 |
| **tick ms (p50 / mean)** | **7.78 / 8.30** | **6.39 / 6.93** |

Note the last row: the converged solver is **17% FASTER**, not slower.

Erik's in-game dump `debug_blowup_20260817_052531` measured **103.961 atm**
with a negative `P_min`. This run measures **103.239 atm** with a negative
`P_min`, on the same level. That is the storm, reproduced headlessly and
switched off by one solver dial.

## 2. How it was isolated

Synthetic two-room fixture, 6 crates, 7200 ticks (300 s), sealed:

| config (70×99, tile 0.333) | RMS d0 | RMS d9 | P_max | P_min |
|---|---|---|---|---|
| `mg_cycles=2` (shipped) | 0.1217 | **0.2803** | **9.633** | **−0.033** |
| `mg_cycles=3` | 0.0023 | 0.0050 (growing ×2.2) | 1.016 | 0.984 |
| `mg_cycles=4` | 0.0024 | 0.0027 (flat) | 1.001 | 0.996 |
| `mg_cycles=8` | 0.0026 | 0.0030 (flat) | 1.001 | 0.997 |
| `c_max=75` instead | 0.0021 | 0.0023 | 1.007 | 0.997 |

**Two independent knobs each kill it**: raise solver effort, or lower the
sound speed. That is the signature of an under-converged solve — and *more*
cycles making it *better* rules out the amplifying-coarse-correction failure
mode that the old non-variational form had (`eos_solver.h:425-436`).

C=8 buys nothing over C=4 ⇒ **converged at 4**. C=3 is marginal (RMS grows).

**Grid size is the dominant variable, not tile size or geometry:**

| grid | tile | RMS d9 | P_max |
|---|---|---|---|
| 14×27 | 0.5 | 0.0072 | 1.010 |
| 70×99 | 0.5 | 0.1776 | 6.195 |
| 70×99 | 0.333 | 0.2803 | 9.633 |

25× worse from grid size alone at fixed tile size; tile 0.5→0.333 adds 1.74×.
That is why **every bench we own was blind to this** — they are all small.

## 3. The project already measured this and the note was not carried forward

`docs/eos_p3_gate_measurements.md` §B, 2026-07-10:

> Single-tick convergence (16² vent, vs a C=64 deep reference):
> C=1: 0.44 → **C=2: 0.24** → C=4: 0.084 atm (**~×0.55/cycle**).
> Warm start … buys ~2 cycles: the schedule drops from V(2,2)×C=4 (cold start;
> **C=3 was UNSTABLE at 19.9 atm worst-dev**) to **V(2,2)×C=2 — FROZEN** …
> **300-tick durability** at the frozen schedule.

Three things stand out, all confirmed by the measurements above:

1. **×0.55/cycle is poor for multigrid** (textbook geometric MG is ~×0.1), so
   2 cycles removes only ~70% of the error. The gate's own single-tick error
   at C=2 is **0.24 atm**; the RMS measured on the big fixture is **0.28 atm**.
   Same number.
2. **C=4 was the durably-stable cold-start schedule and C=3 was UNSTABLE.**
   That ladder reproduces exactly here — at map scale, with no warm-start
   credit surviving.
3. The drop to C=2 rested on warm-start "buying ~2 cycles", validated for
   **300 ticks** on small scenarios. The warm start is only a good initial
   guess when the field is slow; with live fires on a 70×100 map it is not,
   and the credit evaporates.

§4 of the same doc had already listed **"a much lower c_max"** as one of the
three candidate resolutions — which is independently what Erik proposed this
session, and what run E confirms works.

## 4. Why it looked like everything except what it is

Every earlier read is explained by this one cause:

- **Growth at low spatial wavenumber** (k=1–8, λ 17–140 tiles) — an
  under-converged multigrid leaves precisely the *smooth* error components;
  those are the ones needing the most cycles.
- **Growth fastest at high temporal frequency** (286,486× near Nyquist vs 295×
  at 0–0.5 Hz) — a per-tick error source is broadband in time.
- **Negative `P_min`** — an unconverged solve returns unphysical pressures.
- **~98× ambient density in one cell** — a failed projection does not remove
  divergence, so mass piles up. It really was a "mass/momentum event"; the
  cause was just upstream of the momentum.
- **Volume-filling, not corner-hugging** — solver residual is everywhere.
- **Every sealed bench quiet** — small system, 2 cycles suffice.
- **The ★ corner lead, the CFL-desync guess, and O2 suffocation** — all wrong,
  all falsified by measurement before they could cost anything.

## 5. Consequence Erik must weigh: the fire dials were tuned against the storm

Same geometry, same crates, only `mg_cycles` differs:

| | mg2 | mg4 |
|---|---|---|
| fire total, peak | 6.000 (saturated) @ 42 s | 3.928 @ 106 s |
| T_mean, peak | 3.08 | 0.40 |

The spurious wind was delivering O2 and driving combustion. With the solver
converged, **fires burn cooler and slower**. So this fix *requires* the fire
retune already queued as TODO item 3 — and it plausibly explains a long tail
of tuning pain (anchors missing, `cool_shift` 16× off its own anchor).

It also re-opens **`k_drag = 0.5`**: that dial was added to damp a storm that
was largely a solver artifact. It may be unnecessary, or much smaller.
UNTESTED — listed in §7.

## 6. Recommendation

**`mg_cycles = 2 → 8`**, and it costs nothing. Measured on `playground`,
3600 ticks, CPU:

| `mg_cycles` | p50 ms | p99 ms | max ms | mean ms | P_max | `n_sub` |
|---|---|---|---|---|---|---|
| 2 (shipped) | 6.77 | 7.92 | 18.11 | 6.74 | **103.239** | 8 |
| 4 | **4.82** | **5.86** | **12.01** | **4.86** (−28%) | 1.443 | 2 |
| **8** | 5.43 | 6.67 | 15.79 | **5.51** (−18%) | **1.405** | **1** |
| 16 | 7.37 | 8.86 | 14.87 | 7.49 (+11%) | 1.409 | 1 |

(Run strictly sequentially — an earlier concurrent sweep gave inflated and
inconsistent times and should not be quoted.)

**The converged solver is 18% faster than the shipped one.** More cycles cost
more per solve, but a converged solve collapses `n_sub` from 8 to 1, and the
gate had already measured substeps as the dominant cost (`substeps(16) ≈
16.5 ms` vs `MG C=2 ≈ 3.2 ms` at 160²). We have been paying for eight advection
substeps to chase velocities that were solver error.

C=8 and C=16 agree to 0.3% ⇒ **converged at 8**. C=4 is *faster still* (−28%)
and gets 97% of the way (P_max 1.443 vs 1.405) — but C=4 sits exactly on the
stability edge the gate previously measured (C=3 UNSTABLE, C=4 the cold-start
minimum), so **C=8 is the recommendation**: fully converged, still 18% faster
than shipped, with real margin. C=4 is the fallback if perf ever gets tight.

It also fixes the gate's own unresolved perf complaint (§D: "the cost driver is
SUSTAINED sonic venting … pins n_sub at the cap") and its `N_SUB_MAX` re-pin
request — both were symptoms of this, not independent problems.

Prefer this over lowering `c_max`: it makes the solver *correct* rather than
altering the physics to be easier to solve, and `c_max = 300` is a deliberate
physical choice of Erik's (with `c` scaling as √T, which he wants to keep).

This is **not a free change**: it moves every digest, so it needs a deliberate
golden re-baseline with written rationale, plus CUDA lockstep (`mg_cycles` is
already plumbed to the device path, `bindings.cpp:1240`).

**Deeper follow-up worth its own arc (HYPOTHESIS, not measured):** ×0.55/cycle
is poor because prolongation is **piecewise-constant injection** — chosen as
the exact transpose of the summing restriction to keep the transfers
variational. Standard bilinear prolongation + full-weighting restriction is
also variational against the *symmetric* operator now in use, and would
plausibly restore ~×0.1/cycle, buying the cycles back. The gate rejected
bilinear when the operator was still *nonsymmetric* — that objection no longer
applies. Worth measuring before assuming.

## 7. Open / untested

- **Determinism holds**: two `mg_cycles=8` runs on `playground` reproduced
  every RMS decile and both P extremes exactly; and the baseline fixture run
  was reproduced bit-identically across two invocations.
- Does C=8 hold under a **breach** (sonic venting) and under explosions? The
  gate's marginal cases were exactly those.
- `k_drag` re-evaluation with the solver converged.
- CUDA lockstep at the new cycle count.
- The T_abs compression-work patch (Erik approved, timing mine) — note it
  makes step 4c act on ambient air where today it is inert, so it should land
  *after* this and be re-measured against it.

## 8. Reproduction

Scratchpad harness (not committed): `repro.py` builds the fixture or loads a
shipped level, applies `--solver ATTR=VAL` to the live `EOSSolver`, and reports
RMS deciles, P extremes, rail counters, fire totals and tick times.

```
python repro.py PG_mg2.npz --level playground --crates 8 --ticks 3600
python repro.py PG_mg4.npz --level playground --crates 8 --ticks 3600 \
       --solver mg_cycles=4
```
