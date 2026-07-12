# EOS refactor P3 microbenchmark results (deliverables #2 and #3)

> Executed 2026-07-10 on the dev desktop. Measurement only — **no solver code was
> written**; `AtmosphereSolver.gs_iters` was already a pybind read-write attribute
> and `wave_substep`/`diffuse_solve` were already separately callable, so no new
> C++ binding was needed. Script: `tests/_eos_p3_bench.py` (throwaway diagnostic,
> mirrors `tests/_xarch_*.py`). See `docs/eos_refactor_design.md` §3 (tick order,
> CFL estimate) and §8 patch P3 for the deliverables this closes.
>
> Design constants used (task-specified, **not** the shipped engine's 1/24s
> physics tick — these are the future solver's own numbers): `dt = 0.083 s`,
> `dx = 1/3 m`, `CFL_ADV = 0.5`.

## M1 — shipped-engine baseline wall-clock (worst-case-ish load)

Scenario: hull ring + a grid of interior rooms (door-connected partitions),
`enable_recorder=False`, 300 ticks, real `Simulation.step()` (all systems —
combat/movement/physics/fire — running, matching production). Load: 5 staggered
grenade-equivalent explosions (`grenade_frag` payload, radius x2 / pressure x3)
spread across the map and across the run, a true hull breach to vacuum
(`gmap.destroy_wall` on an edge hull tile) at 55% through the run, and a
water release (0.4 m flood over a room) at 75% through the run.

| Scenario | p50 (ms) | p99 (ms) | max (ms) | ticks |
|---|---|---|---|---|
| **160x160** (primary) | 6.6 | **18.97** | 21.0 | 300 (10 warmup skipped) |
| ~50x120 (shipped ship scale, reference) | 1.6 | 9.78 | 27.0 | 300 (10 warmup skipped) |

Gate: p99 <= 25% of the 83 ms budget = **20.75 ms**. 160x160 p99 = 18.97 ms ->
**PASS**, but with only ~1.8 ms (9%) of headroom left in the *whole-tick*
budget — everything else in the tick (combat/movement/other physics/fire) plus
the *current* atmosphere group already consumes 91% of the allowed budget.

Note: the ~50x120 reference's **max** (27.0 ms) exceeds the 160x160 grid's max
(21.0 ms) despite the smaller grid — a single-tick spike from the `destroy_wall`
breach event (its Dinv-cache invalidation forces a one-time full per-cell RB-GS
coefficient rebuild) landing on that map's specific breach tick, not a steady-state
number; p99 is the load-bearing statistic here, per the design's own "p99/max, not
mean" gate philosophy.

**Verdict:** M1 confirms the current engine has real but thin headroom at 160x160
under worst-case-ish load. The gate is judged against *this* baseline: the new
solver's atmosphere-group replacement (M2) must fit inside roughly
`20.75 - (18.97 - old_atmosphere_group_cost)` ms to keep the whole tick under
budget — see the M2 verdict below for the number.

## M2 — RB-GS per-sweep cost (pins sweep count S) + legacy wave-core verification

Measured by snapshotting a turbulent 160x160 field state (post-explosion) and
calling `AtmosphereSolver.diffuse_solve` / `.wave_substep` directly and repeatedly
from fresh copies of that snapshot (no cross-call drift), restoring `gs_iters` to
its shipped default (8) afterward — never left mutated.

| gs_iters | median (ms) | p90 (ms) | min (ms) |
|---|---|---|---|
| 8  | 2.70 | 2.92 | 2.45 |
| 16 | 4.81 | 5.06 | 4.22 |
| 24 | 6.77 | 11.49 | 5.99 |
| 40 | 10.37 | 11.11 | 9.84 |

Linear fit (medians): `ms = 0.917 + 0.238 * gs_iters` -> **ms/sweep @ 160x160 =
0.238 ms**.

**Legacy wave-core verification (the napkin's claimed saving):** `wave_substep`
costs **0.240 ms/call** (median). The design's napkin assumed **~50** wave
substeps/tick (from `max_dt = 0.5/c` at the C++ default `c=300`). The **shipped
config** (`config.toml wave_c = 66.0`) gives a different `max_dt` — the actual
configured substep count at `dt=0.083s` is `ceil(0.083 / (0.5/66)) = `**11**, not
50:

| | substeps | wave-core cost |
|---|---|---|
| napkin's assumed `c=300` | 50 | 12.02 ms |
| **actual shipped `c=66`** | **11** | **2.64 ms** |

This is the run's headline surprise: **the napkin overestimated the deleted
wave-core cost by ~4.5x** because it used the C++ struct default wave speed, not
the shipped config value. The real legacy cost being deleted is much smaller.

**Projected Helmholtz cost** (`ms/sweep x 1.5` wide-int64 factor x S, per
§3.4's amortized-divide spec):

| S | projected Helmholtz ms/tick |
|---|---|
| 8  | 2.86 |
| 16 | 5.72 |
| 24 | 8.58 |
| 40 | 14.30 |

**Old-vs-new, using the corrected (actual-config) legacy numbers:** old total
atmosphere-group cost/tick = wave core (11 substeps, 2.64 ms) + `diffuse_solve`
(gs=8, 2.70 ms) = **5.34 ms**.

- vs new Helmholtz-only @ S=8 (2.86 ms): **1.87x cheaper**
- vs new Helmholtz-only @ S=16 (5.72 ms): **0.93x** — i.e. *slightly more
  expensive*, not cheaper, before even adding the new solver's advection
  substeps (which don't exist yet and aren't measured here)

**Verdict:** M1+M2 do **not** confirm the napkin's "~2-3x cheaper at S=8-16"
claim once the shipped wave-speed config (not the C++ default) is used for the
legacy side. At the shipped config, the deleted wave core is cheap (~2.6 ms,
not ~12 ms), so the new solver's margin is much tighter than advertised:
- **S=8** (2.86 ms) is comfortably supported: cheaper than the old
  atmosphere-group total, and fits inside the ~1.8 ms of whole-tick headroom
  M1 measured plus the ~2.5 ms freed by deleting the old group — leaves room
  for the new advection substeps too.
- **S=16** (5.72 ms) is a wash against the old group's total (5.34 ms) *before*
  counting the new advection substeps' cost — it likely blows the whole-tick
  budget once those are added. Do not lock S=16 without measuring the real
  advection-substep kernel first.
- **S=24/S=40** clearly exceed the freed budget (8.58 ms / 14.30 ms vs 5.34 ms
  freed) and are not supported by this data.

**Recommendation for the P3 gate: target S=8**, and re-run this measurement
once the real advection-substep kernel exists (patch P3 build) before
considering S=16 — the napkin's cost-honesty premise (§8 P3 change log) is
exactly why this correction matters: re-derive, don't inherit.

## M3 — substep-count distribution (pins N_SUB_MAX)

`n = ceil(dt / (CFL_ADV*dx / (u_est+eps)))`, `u_est = max|u| + (max|gradP|/N_hat)*dt`,
computed every tick from the CURRENT engine's fields as proxies (`u` ~
`wind_x`/`wind_y`, `P` ~ `atmosphere + wave_p`, `N_hat` ~ `atmosphere` floored at
1e-3). The gradient/accel max is taken over cells **not adjacent to a structural
wall** — the current engine hard-zeros `atmosphere` at wall cells (its own BC
pass), which is a raw-field-read artifact this proxy must not confuse with real
physics (the design's actual operator uses a Neumann mirror at solid, i.e. zero
cross-wall flux); true vacuum (breach) cells stay in the domain since `P=0`
there is real Dirichlet physics, not an artifact.

| Scenario | p50 | p99 | max |
|---|---|---|---|
| M1 160x160 (full run) | 1 | 96.1 | 124 |
| M1 ~50x120 (full run) | 30 | 159.0 | **159** |
| Nasty: simultaneous multi-explosion stack (9 grenades, 1 tick) | 4.5 | 37.0 | 37 |
| Nasty: O2-tank rupture (200x-ambient pressure + 2000K spike, direct field write) | 1 | 8.3 | 13 |

Overall observed max n = **159** (from the ~50x120 reference run, likely at the
breach or water event). `p50=1` in the quiescent 160x160 run confirms the design's
own point (§3.2): most ticks need only the CFL-cheap single substep; the tail is
driven entirely by the breach/water/explosion events, exactly the "native
venting" transient the design expects.

**N_SUB_MAX recommendation: 512** (next power of two >= observed max x 2 margin
= 318 -> 512).

**Verdict:** the substep count is heavy-tailed exactly as the design's own
qualitative claim predicts (near-1 in the quiet steady state, spiking hard at
discrete events), and the spike magnitude (up to 159 in this A/B pair of
scenarios) is materially larger than the "~2-3 substeps" the P3 napkin's cost
model assumed for its *typical*-case advection-substep count. That assumption is
still fine for the **typical/median** case (p50=1 here), but a **fixed cap**
(§3.4's "never adaptive" rule) sized only for the typical case would silently
truncate the CFL estimate during exactly the breach/blast transients the new
solver is supposed to handle natively — this is the concrete number the cap
needs to be checked against. Recommend **N_SUB_MAX = 512** as the gate value,
with the explicit caveat that it should be re-validated once the real
solver's `u`/`P`/`N` fields exist (this measurement uses the *current* engine's
fields as a proxy, per the task's own framing — not the future solver's actual
state, which will differ once bulk-species donor-cell transport and the true
Kwatra pressure evolution replace today's wave+diffusion split).

## Summary for the P3 gate

| Deliverable | Result |
|---|---|
| #2 baseline (M1) | 160x160 p99 = 18.97 ms, PASSES the 20.75 ms (25%-of-83ms) gate with ~9% headroom |
| #2 sweep cost (M2) | 0.238 ms/sweep @ 160x160; **napkin's legacy-cost baseline was ~4.5x too high** (used c=300 not the shipped c=66); corrected comparison supports **S=8**, makes **S=16 a wash** (not "cheaper"), rules out S>=24 |
| #3 substep cap (M3) | heavy-tailed, p50=1, max=159 across the measured scenarios; recommend **N_SUB_MAX=512** |
