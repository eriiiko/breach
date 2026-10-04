# Smoke transport v2: trace gases ride the air (design v1, 2026-10-04)

**Issue:** #12 (fire + smoke session, step 3). **Branch:** `12-smoke-transport`, cut
off `fire-12`. **Status:** DRAFT v1, awaiting critique. **Depends on:**
`docs/smoke_transport_handoff_2026-10-04.md` (the problem), and the #12 comments of
2026-10-04 that hold the measurements and the look spike
(`prototypes/smoke_transport_spike/`, untracked).

## 1. Problem, measured

Every non-bulk gas plane (smoke, steam, poison, teargas, fuel_gas: the "trace
planes") is moved once per tick by an integer semi-Lagrangian (SL) step
(`cpp/src/smoke_dynamics.cpp`, CUDA twin `cuda_smoke.cu`). The per-tick replay
attribution on 2026-10-04 found three effects:

| mechanism | measured effect |
|---|---|
| SL keeps a tile's value as its air expands (the advective form on a divergent flow; H1) | creation tracks `Σ c·div(u)·dt` tick by tick. One small charge's 1.27 units grow to 61 in 0.5 s, and the studio peaks at 100 units against 8.9 deposited |
| SL resampling over a 20–37-tile back-trace at blast speeds | single ticks destroy 13–52 units; this is not predicted by the linear term |
| the diffusion update `mul_q16(coeff, lap)` floors per cell | about ½ count is lost per cell per tick, independent of the amount: −11 % in 10 s on a quiet blob |

The [0, 1] clamp never binds (max tile 0.24). The only mass these schemes are
meant to remove is decay and venting, and they are not alone in doing it.

**Erik's tolerance (2026-10-04):** deterministic non-conservation is acceptable,
and so is strong wind or a breach clearing smoke. Explosions **inventing** smoke
is not, and a room going black so that nothing can be seen is the complaint.

**Look spike (2026-10-04, captured studio flows, offline):** three options were tried.

| option | outcome |
|---|---|
| A: today's scheme | 91 units at 1.75 s |
| B: SL on smoke per unit of air, then multiplied by the new N (the "compensate" idea) | still 40 units at the peak and 0.1 by 6 s. **Rejected** |
| C: conservative donor-cell | holds the deposit exactly and looks smooth: blast rings early, swirls later. **Chosen by Erik** |

Diffusion is irrelevant to B and C: running both without it changes nothing
visible. C's smoothness is the donor-cell's own numerical spreading.

## 2. The law

**A trace plane moves only with the bulk air.** On every face, in every EOS
substep, the trace that crosses the face is that face's applied bulk mass flux
priced at the donor's trace-per-air ratio:

    phi_f = price_face(dq_f, S_donor_pre, N_donor_pre)
          = floordiv(dq_f * S_donor_pre, N_donor_pre)      (exact, 64-bit)

The terms:
- `dq_f` is the APPLIED per-face bulk dq the mass pass already banks (`dqsum_e` /
  `dqsum_s`: post-limiter, summed over the bulk planes, positive = toward the
  higher index).
- `N_pre` is the pre-flux bulk sum.
- `S_pre` is a pre-flux snapshot of the trace plane.

This is exactly how `gas_energy` has ridden the same faces since arc #54
(`bulk_flux_energy_transport_cached`, stage 3). It reuses that pass's dq, its
N snapshot, its gather shape, and its pinned face order E, W, S, N. Both sides of
a face compute the same int64 from the same triple, so the pass telescopes:
**Σ over the participating cells of a trace plane is conserved exactly**, apart
from the counted boundary channels in §3.

`Σ_f dq_f ≤ N_i` (the mass limiter) gives `Σ_f phi_f ≤ S_i`, so a donor can
never go negative. A donor's trace-per-air ratio is invariant as air leaves it,
which is what removes H1: an expanding parcel spreads the same smoke over more
air, so its tiles thin out instead of multiplying it.

**Range.** A trace plane is int32 Q16.16 per tile, so `S < 2^31`. `dq ≤ N < 2^30`
(the stated map invariant, asserted at the energy pass). The direct product
`dq·S < 2^61` therefore fits without `price_face`'s split. Still, the twin uses
`price_face` itself (one pricing idiom in the TU, and the split is exact for any
`S ≥ 0`).

**Participation.** Trace participates on `!solid && !is_vacuum && !ring` cells,
INCLUDING thermal-solid (crate) tiles. Bulk air already seeps through a crate's
pores, and the trace goes where the air goes. (Energy excludes those tiles because
their `temperature` is the object's; smoke has no such reason.) So the trace pass
needs `N_pre` on thermal-solid cells too, where the energy pass stores 0. P1
either computes its own `n_pre_trace` or fills those cells in `n_pre`, whichever
keeps the energy books bit-identical (gate V5).

**Once per substep, inside the loop.** This is new stage 3b of
`bulk_flux_energy_transport_cached`, after the energy apply and before the mirror
refresh, for every trace plane that is not all zero this tick. The all-zero skip is
decided once per tick, before the loop, identically on every backend: a zero plane
stays zero under this law. The once-per-tick SL call in
`PhysicsEngine::run_substeps` is deleted.

**Diffusion: conservative, once per tick**, where the SL call was. On each open
face, in canonical orientation (i = lower index):

    F_ij = floordiv(c_ij * (S_i - S_j), 2^16),   c_ij = q16(d_g * dt * min(perm_i, perm_j))

Then `S_i -= F_ij` and `S_j += F_ij`. The flux is computed once and applied with
opposite signs, so the update is exact (the arc #54 face-flux idiom); the floor
loses nothing because the same int moves both ways. Stability needs
`4·c ≤ ONE`, i.e. `d_g·dt ≤ 0.25`. This is refused at the gases door (load time)
rather than substepped; the shipped dials are 0.004 per tick. Diffusion stays:
it is what spreads smoke in still air, where nothing else moves it.
`wind_diffusion_scale` has been forced to 0 since EOS P3 and is retired together
with SmokeDynamics.

**No upper clamp.** Compression can legitimately put more than 1 unit of smoke on
a tile, and a clamp would destroy it. Transport cannot produce a negative, so the
lower clamp becomes a debug assert. P1 audits every reader of a trace plane for an
assumed ≤ 1, and puts any `min` it needs AT THE READ. Known readers to check:
- the radiation sweep's `a_gas` (already `min(ONE, …)`)
- the light smoke term (Beer-Lambert: unbounded is fine)
- `gas_medium` / `gas_detail`
- combustion's O2 / smoke reads
- the tile inspector
- `debug_keys` (writes `SMOKE_MAX_Q`; harmless)

**Decay is unchanged**: once per tick, after diffusion, as today.

## 3. Boundaries and books

Each trace plane gets four int64 counters, accumulated per tick:
- **`trace_vent_sum[g]`**: trace priced onto a face whose receiver is vacuum or
  the ambient ring. It leaves the system with its air. This is Erik's "breaches
  vent smoke out", now exact.
- **Inflow from vacuum or the ring** carries zero trace: the ring holds no smoke,
  which is today's rule.
- **`trace_wipe_sum[g]`**: a participating cell whose post-flux bulk N falls below
  `N_EPS_RAW` has its trace destroyed and counted (no air left to carry it), the
  energy pass's wipe rule.
- **`trace_decay_sum[g]`**: what the decay removes.
- Deposits (payloads, combustion soot, debug keys) are outside transport and are
  not booked here.

The transport identity, exact in int64 for any tick:

    Σ S(after transport+diffusion) − Σ S(before) == −trace_vent_sum − trace_wipe_sum

The counters are diagnostics: no gate reads them except the identity tests, and
they are not digested.

**Trace never touches the air.** The trace planes stay out of bulk N, out of p*,
and out of the energy books ("only BULK shares carry energy", `physics_engine.cpp`).
`GasTable.conservative` keeps its one meaning, bulk membership. No flag split is
needed, because every non-bulk plane now rides.

## 4. Backends

- **CPU**: stage 3b plus the diffusion in `bulk_transport.cpp` / `physics_engine.cpp`.
  The new TU code goes on `/fp:strict` (it already is, as the same TU).
- **CUDA per-call and chained** (`cuda_bulk_transport.cu`, `cuda_eos_step.cu`): a
  stage 3b kernel transcribed from the CPU loop expression for expression, in
  gather form with no atomics (each cell writes only its own S). The trace planes
  must be on the device through the substep loop.
- **Resident** (`trace_smoke_resident`, S8a Path B): replaced by the same kernels
  inside the resident EOS orchestration. New launches follow the RL-batch habits
  (§A: born `(N, h, w)`-shaped, per-env masks).
- **Deleted**: `SmokeDynamics` (CPU, bindings, tests), `cuda_smoke.cu`'s step, and
  `trace_smoke_resident`, with the tests that pinned the SL behaviour
  (`test_smoke_semilagrangian.py`, `test_cuda_s4a_smoke.py`, the
  `cuda_trace_smoke_check` key). Each deletion is listed in the patch report.

## 5. Cost (Erik's question)

Measured on this PC's CPU build (2026-10-04):

| | studio 56×56 | playground 70×100 |
|---|---|---|
| tick, median | 7.0 ms | 10.6 ms |
| EOS substeps per tick | 8 on most ticks after a blast, otherwise 1 | 1–4 |
| today's SL step, one plane, once per tick | 0.14 ms | 0.29 ms |
| bulk mass-flux pass, one substep | 0.09 ms | 0.29 ms |

A stage-3b pass per plane per substep costs about one mass-flux pass. That is
0.1–0.3 ms per substep, minus the deleted SL step.

- **Quiet room, one active plane:** about the same as today.
- **Studio after a blast (8 substeps):** about +0.6 ms per active plane per tick,
  roughly +9 % of the tick.
- **Several active trace gases at once** add up linearly. P1 may price all active
  planes in one pass over the faces (dq and N are read once), which roughly halves
  that.
- **GPU:** these are per-cell kernels next to the energy kernel, small next to the
  radiation sweep.

## 6. Verification plan (properties the tests pin)

Each test names its property and the change that must break it. No test pins a
count or a set of gases.

- **V1, exact conservation.** A sealed room with any wind (a blast in it) and decay
  overridden to 0: Σ S per trace plane is constant to the count across N ticks.
  *Breaks if* any stage floors asymmetrically, skips a face, or clamps.
- **V2, no expansion mint.** Seed a uniform trace-per-air ratio (`S = r·N`) and
  fire a heat-only blast. The ratio stays uniform within the rounding bound,
  max|S/N − r| ≤ (cells' face count) / N. *Breaks if* the trace is moved as a
  per-tile value again (H1).
- **V3, venting closes.** Seal a room, then breach it: Σ S before − Σ S after ==
  `trace_vent_sum + trace_wipe_sum`, exactly. *Breaks if* a vent path bypasses the
  counter or mass is lost silently.
- **V4, non-negativity.** No trace cell goes negative across the blast scenarios.
  *Breaks if* a donor is over-priced.
- **V5, the air does not see the smoke.** Bulk N, `gas_energy`, wind and
  temperature are bit-identical with trace planes populated or empty. *Breaks if*
  the trace leaks into bulk membership, p*, or the energy books.
- **V6, CPU == CUDA, tolerance 0**, per-call, chained and resident, on every trace
  plane and every counter (`cuda_harness`, a `cuda_*_check.py` + `test_*` wrapper).
- **V7, diffusion conserves and spreads.** In still air, Σ S is exact and the
  blob's second moment grows. *Breaks if* diffusion is removed or loses mass.
- **The studio as a reading, not a gate:** total smoke after the three charges
  stays within the deposits minus decay. Reported in the P1 result, not asserted
  on a config row.

## 7. Patches

| # | content | mode | tier | gate | HUMAN-TEST |
|---|---|---|---|---|---|
| P1 | CPU law (stage 3b), conservative diffusion, counters, clamp removal + reader audit, the gases-door stability check, SL CPU path deleted, V1–V5 + V7, GOLDEN re-baselined ONCE with written rationale | worktree subagent | Opus | V1–V5, V7 + full suite green | no |
| P2 | CUDA per-call, chained and resident twins; `cuda_smoke` / `trace_smoke_resident` deleted | worktree subagent | Sonnet (tol-0 oracle) | V6 | no |
| P3 | CLAUDE.md rows (§8), close-out, merge into fire-12 | inline | — | suite green + Erik | **YES**: Erik plays the smoke-light studio and the playground fire before merge |

The golden moves once, at P1, for every scenario with a trace gas in it. P2 must
match P1's CPU numbers bit for bit.

## 8. Systems

**(a) Existing canonical systems this design uses:**
- **Gas pump primitives / bulk transport:** the applied dq planes (`dqsum_e/s`)
  and the N snapshot. No second flux path.
- **Gas energy field + seam:** untouched; its stage-3 shape is the template.
- **Face-flux energy step idiom:** canonical orientation, computed once, applied
  ±.
- **Fixed-point kits:** `floordiv_q`, `price_face`. No new rounding.
- **Material / gas tables:** diffusion and decay stay `[gases.*]` columns; the
  door gains the stability check.
- **Config / PhysicsRunner:** dials bound there only.
- **Determinism gates:** field digest, GOLDEN_AGGREGATE (re-baselined once), the
  CUDA harness.
- **RL-batch habits §A:** new resident launches.
- **Vent / duct system:** unchanged. It moves trace through the `_vec`
  primitives, a separate, already-booked path.

**(b) New, with draft rules for CLAUDE.md:**
- **Trace transport** (`bulk_transport.cpp` stage 3b + the conservative trace
  diffusion; CUDA twins): "Every non-bulk gas plane moves ONLY by riding the bulk
  face flux, priced `floordiv(dq·S, N)` at the donor, plus the once-per-tick
  conservative face diffusion. Never an advection of its own, never a clamp in
  transport. Vent and wipe losses are counted (`trace_vent_sum` /
  `trace_wipe_sum`)." This REPLACES the SmokeDynamics references in the Sim-core
  rows.

## 9. Recorded, not fixed (other systems; one system at a time)

- **Decay floors to 0** below about 0.045 units per tile (`v·frac_q >> 16`), so
  thin smoke never decays. Kept as is; an issue for the smoke-look session.
- **After a single heat-only frag** in the sealed studio, the wind sits on its
  rails (160–300 m/s) for more than 2 s, and the EOS runs at its 8-substep cap with
  a per-substep CFL of about 4.7 at those speeds. The air's own transport limiter
  holds it conservative. The trace inherits that coarseness; it does not cause it.
  Drag is revisited after this arc (Erik, 2026-10-04).
- **Physical amount:** even conserved, the studio's 99 g of soot gives a per-tile
  τ of 0.3–0.7 once evenly spread (2–4 m visibility). The lever is the soot yield
  (TNT's 0.193 g/g assumes no afterburn): Erik's call, after this arc.

## 10. Decisions (Erik-vetoable statements)

1. **All five trace gases ride**, not only smoke. They share the defect, and one
   transport is simpler than two.
2. **Thermal-solid tiles carry trace** in their pores, as they carry air.
3. **The [0, 1] upper clamp is removed** from transport; readers clamp at the read
   if they need to.
4. **Diffusion stays**, conservative, with today's dials.

## 11. Critique log

| lens | critic | verdict | resolution |
|---|---|---|---|
| conservation / integer arithmetic | Opus, 2026-10-04 | PASS WITH FIXES. Stage 3b is confirmed exact and non-negative, cited against the code (the per-plane limiter on the pre-flux N; `dqsum` after the limiter; sign(dq) == sign(v); the N_EPS guard; one call path, `eos_solver.cpp:848`; filling `n_pre` on thermal-solid cells is energy-safe because it is read only behind `e_participates`). Findings: **(1) BLOCKER**: the diffusion floor toward −∞ breaks positivity and pumps thin smoke north-west; the fix is magnitude truncation (the `scale_mag` idiom). **(2) BLOCKER**: trace stranded on cells that become solid, vacuum or ring (the old step zeroed them, `smoke_dynamics.cpp:309-314`); zero it once per tick, booked. (3) Diffusion through vacuum and ring faces is unspecified. (4) Diffusion must be Jacobi (a snapshot), not in place. (5) Build the coefficient as an integer chain (the `coeffE` idiom), and run the stability check on the integer value and where dt is bound. (6) State and assert the int32 narrow bound. (7) V2's bound is wrong: run it with diffusion at 0 and count substeps. (8) V1 must include `trace_wipe_sum`. (9) Specify the trace's N snapshot (before stage 2, same planes, assert extended) | pending synthesis |
| backends / resident path / CUDA lockstep | | | |
| systems reuse / readers / scope | | | |
