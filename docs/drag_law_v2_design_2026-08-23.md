# Drag law v2 — two-term interior damping (linear + implicit quadratic)

> **Status**: DESIGN v2.1 (post-critique — three adversarial lenses, all
> SURVIVES WITH FIXES, resolutions in §11 — plus a delta-verifier pass,
> FIX THEN BUILD, all fixes applied. Build authorized.)
> **Date**: 2026-08-23 · **Issue**: #4 · **Arc branch**: `drag-law-v2`
> **Depends on**: `docs/energy_books_arc_close_2026-08-17.md` (P-E3 as-built),
> `docs/archive/e1_p_e3_asbuilt_2026-08-17.md`,
> `docs/archive/velocity_clamp_pv1_asbuilt_2026-08-19.md` (P-V1 as-built),
> `docs/architecture/engine/04_atmosphere_and_pressure.md`,
> `docs/architecture/engine/14_determinism_and_number_ingress.md`
> Archived to `docs/archive/` at arc close.

## 0. Rulings (Erik, 2026-08-23 — all locked; do not re-derive)

- **R1 — Two-term law.** `F = −k1·u − k2·|u|·u`, dials `k_drag` (existing) +
  `k_drag2` (new), independent. Every "which exponent?" outcome is a dial
  setting; `u³` stays kit-reachable later (needs `|u|² = rad` only, no sqrt)
  — ACCEPTED GAP in v1 of the mechanism.
- **R2 — Implicit discretization.** `u ← u/(1 + k2·|u|·dt)`: factor ∈ (0,1]
  for any dial/speed/dt — never reverses flow, never overshoots zero.
  Stronger than stability (critique-verified): the semi-implicit map is the
  **exact solution** of `du/dt = −k2·u²` sampled at ticks — by induction
  `u_n = u0/(1 + k2·u0·n·dt)` — so stage Q's only discretization error is
  integer rounding, and the decay trajectory is dt-independent.
- **R3 — Land dormant now, tune later.** P1 ships the mechanism byte-inert at
  `k_drag2 = 0` (auto-merge on green — pre-authorized by Erik 2026-08-23
  together with this plan; the P-E3 pattern). All dial-turning is HUMAN-TEST,
  waits behind #48 (vents) and #6 (smoke), folds into #5's retune. No golden
  re-baseline before that. **#4 stays open after P1/P2 merge** (P3 pending).
- **R4 — sqrt determinism verdict (retires #4's lockstep fear only).**
  `sqrt_q16` is a fixed 32-iteration branch-identical integer isqrt
  (`fixed_point.h:722`); `sqrt_q16_dev` is a verbatim device port, attested.
  Determinism risk: none. **Cost is NOT free by precedent** (v1 misstated
  this): fire's per-cell sqrt runs on lit flammable cells only
  (`fire_simulation.cpp:161-164`), the clamp's only on clamped cells
  (`:856` / cuda `:207`). Stage Q is a new always-on cost class, mitigated by
  the calm-cell fast path (§7) and gated by an **armed, windy** timing bench.
- **R5 — Tick rate (Erik, 2026-08-23).** `ticks_per_second` stays a free
  config variable (currently 24; `config.toml:2`). All design formulas are
  symbolic in dt; worked numbers below are labeled anchors at dt = 1/24.
  Gates derive thresholds from the live integer chain, never from printed
  numbers. If a future arc ever needs to lock the tick rate, Erik's blessed
  value is 24.
- Standing rulings inherited from #4: the VENTING gate is mandatory (§6 —
  and per §1 it is load-bearing, not ceremonial); the small-|u| cutoff is a
  named, measured threshold (§4).

## 1. The law, and what each term honestly does

Continuous: `du/dt = −k1·u − k2·|u|·u`. Crossover speed `u* = k1/k2`.

Three observables, three different stories — v1 conflated them; sizing at P3
must not:

**(a) Time domain — standing/recirculating flow** (a room's swirl, ambient
circulation). Quadratic decay is algebraic, not exponential:
`u(t) = u0/(1 + k2·u0·t)`; **half-life** `= 1/(k2·u0)`, e-fold
`= (e−1)/(k2·u0) ≈ 1.72/(k2·u0)`. Anchors at k2 = 1:

| regime | \|u\| | half-life | e-fold |
|---|---|---|---|
| storm | 20 | 50 ms | 86 ms |
| vent breeze | 0.2 | 5 s | 8.6 s |
| ambient drift | 0.02 | 50 s | 86 s |

Here k2 genuinely bites storms and spares drift. Note the tail: at k1 = 0
residual drift decays only as 1/t — in floats forever; in our integers the
trunc-toward-0 divide removes ≥1 raw count per component per tick above
`u_dead`, so decay terminates in finite time (§4). The linear term's e-fold
is `1/k1` (measured: 49.6 s at k1 = 0.02, `config.toml:651`; the SHIPPED
value is k1 = 0.5, `config.toml:634`) — when comparing dials, compare
e-folds to e-folds, not to half-lives.

**(b) Transport domain — venting/equalization through a path.** Along a
streamline in steady flow, `u·du/dx = −k2·u²` ⇒ `u(x) = u0·e^(−k2·x)`:
per-metre attenuation is **speed-independent**, and discretely the slow flow
loses *more* per cell crossed (it sits in the damper longer). Per-cell
momentum retention at k2 = 1, dt = 1/24, tile 0.333 m: breeze 0.2 → 71.7%,
neck 20 → 78.5%, blast 300 → 93.3%. **Quadratic drag does not structurally
protect venting** — v1's claim to the contrary was wrong. The venting neck is
the max-|u| cell (the storm audit's own Helmholtz-neck observation), and
per-tick neck damping reaches molasses parity (k_drag = 10 ⇒ retention
0.5833) at **k2 ≈ 0.86** (from `1/(1+k2·20·dt) = 0.5833`). Orifice estimate:
equalization time stretches by `√(1 + 2·k2·L)`; the §6 bound of 1.5× implies
**k2·L_neck ≲ 0.625** — the venting-safe band at ~1 m necks is roughly
`k2 ≲ 0.6`, an order of magnitude below naive sweep ceilings.

**(c) The ceiling — a named hard constant.** The map `u ↦ u/(1+k2·dt·u)` has
supremum **`u_ceil = 1/(k2·dt)`** (= `u_dead · 2^16`, the exact dual of §4's
floor): the *stored* wind field can never exceed it, whatever the kick
produced. Anchors at dt = 1/24: k2 = 0.1 → 240; 0.25 → 96; 0.5 → 48;
1 → 24; 10 → 2.4 (units of speed). Where `u_ceil < min(c_local, U_MAX)`
(P-V1 plane: 300/1000) the drag, not the clamp, is the effective cap, and
every downstream wind consumer (fire's W, smoke advection, wave push) sees
capped blast winds — and the advection substep count `n_sub` (derived from
max|u| + the pressure term, `eos_solver.cpp:477-479`) drops across its
discrete cliff: a behavior *and* perf change for the P3 table. Ledger consequence: at blast cells with `k2·u·dt ≫ 1`
stage Q removes ~99% of injected KE per tick — `e_drag_drop_sum` becomes the
dominant sink and the heat-deposit step grows ~24× vs shipped linear (§6's
blast watch exists for this).

**Sizing implication (binding on P3, not on the mechanism):** k2 is the
storm-killer and lives in a narrow honest band (venting bound + ceiling);
the vent-drift protection Erik wants comes from sizing **k1 down** (its
time-domain e-fold at drift speeds), not from k2's shape. The two-dial
structure is exactly what makes that split possible — R1 stands.

## 2. Discrete update (per tick, in the kick-loop drag block)

Sites — the ONE existing drag block, all mirrors: CPU `eos_solver.cpp:882-`
and `:1785-` (two kick-loop variants), GPU kernel
`cuda_kick_compression.cu:113` (`kick_kernel`, drag block at `:220-257`),
reached from `cuda_eos_step.cu:645-663` and `cuda_eos_resident.cu:699-708`
via the shared launch core + wrapper (`:421`, `:443-470`). Applied once per
TICK, after the |u| clamp, before the store — unchanged.

Restructure (identical structure in all mirrors; **each mirror keeps its own
existing spellings** — CPU: file-local `mul128_shr` (`eos_solver.cpp:46`,
NOT in `fixed_point.h`), `sqrt_q16`, and `!ts[i]`; GPU: `mul128_shr_signed`,
`sqrt_q16_dev`, and `!(ts && ts[i])`):

```
if ((kd_q > 0 || kd2_q > 0) && !ts[i]) {                 // widened outer branch
    u_old = u;                                            // capture (unchanged)

    // Stage L — linear: the EXISTING lines verbatim, inner-branched,
    // INCLUDING the saturation guard:
    if (kd_q > 0) {
        kk_drag = (kd_q < FP_ONE) ? (q16)(FP_ONE - kd_q) : 0;   // verbatim
        u = sign_symmetric_mul(u, kk_drag);                      // verbatim
    }

    // Stage Q — implicit quadratic, NEW, dormant by branch:
    if (kd2_q > 0) {
        rad1 = ux*ux + uy*uy;              // int64; |comp| ≤ RAD_SAFE upstream
        if (rad1 >= rad_dead_q32) {        // §7 calm-cell fast path — EXACTLY
                                           // the prod>0 condition, precomputed
            umag  = sqrt_q16(rad1);
            prod  = mul128_shr(kd2_q, umag, 16);      // trunc(k2·dt·|u|), ≥ 1
            denom = (int64)FP_ONE + prod;             // ≥ FP_ONE + 1
            ux = (ux * (int64)FP_ONE) / denom;        // trunc-toward-0 int64
            uy = (uy * (int64)FP_ONE) / denom;        //   division (clamp idiom)
        }
    }

    // Energy booking + heat counterparty — the EXISTING block, verbatim,
    // booking the COMBINED du² = |u_old|² − |u|² across both stages.
}
```

Pinned choices with reasons:

- **Division, not `reciprocal_q16`**: single rounding, exact sign symmetry
  (`trunc(−a/b) = −trunc(a/b)`), shrink-only (`denom > FP_ONE` ⇒
  `|u_new| ≤ |u|−1` counts or 0), and the identical idiom the clamp rescale
  ships cross-attested for negative numerators
  (`eos_solver.cpp:867-868`, `cuda_kick_compression.cu:216-217`).
- **Order L→Q**, corrected justification (critique): the order sensitivity is
  bounded by `k1·dt` **alone** (≈2.1% at shipped k1 = 0.5, dt = 1/24),
  independent of k2 and |u| — NOT by `(k·dt)²`, which fails at the neck
  (`k2·u·dt` reaches 0.83 there). Bound: `(L→Q)/(Q→L) = (1+bu)/(1+bu(1−a))`
  with `a = k1·dt`. Pinned L-first; keeps the linear lines byte-verbatim.
- **`rad1` is recomputed** — the clamp block's `rad` (`:206`/CPU `:855`) is
  STALE when the clamp fired (rescale runs after it). Do not reuse.
- **Booking wraps both stages**: `du² ≥ 0` holds structurally (both stages
  shrink magnitude — critique-verified incl. the 1-LSB corner).
  `k_drag_heat_frac` applies to combined removed KE. Tourniquet skip
  unchanged. ACCEPTED GAP: the ledger does not attribute removal to k1 vs k2
  (conservation needs totals; attribution comes from differencing bench runs;
  extra counter slots rejected as harness churn).
- **Folds**: `kd2_q = quantize(k_drag2 · dt)` beside the kd_q folds
  (`eos_solver.cpp:463`, `:1679`; CUDA fold fn `cuda_kick_compression.cu:
  360-400` region), plus **`rad_dead_q32`** — an `int64_t` member of
  `KickScalarFolds` under exactly that name, mirrored in the CPU folds:
  `rad_dead_q32 = (kd2_q > 0) ? (int64)U0*U0 : 0` with
  `U0 = ceil(2^16 / kd2_q)`. **The dormant-dial guard is mandatory** — an
  unconditional ceil-divide is a divide-by-zero at the SHIPPED config
  (kd2_q = 0). `kd2_q ≥ 1` ⇒ `U0 ≤ 2^16` ⇒ `U0² ≤ 2^32`, int64-trivial;
  computed host-side per tick. Dormancy branches on the QUANTIZED kd2_q,
  never the float (P-E3 idiom).
- **Behavior note (P3-visible, named per critique):** at `k1 = 0, k2 > 0`
  the widened branch makes the block's temperature-write reachable where it
  never ran before — a cell already above `T_MAX_PHYS` gets railed by
  `t_candidate` clamping (`eos_solver.cpp:928-940` region) with `dT = 0`.
  Not reachable in any shipped P1/P2 config; P3 sizing must know it exists.

## 3. Determinism, ingress, dormancy

- **Byte-inert at `k_drag2 = 0`**: `kd2_q == 0` ⇒ stage Q never executes;
  the outer condition reduces to today's `kd_q > 0` and the executed integer
  arithmetic is bit-identical to main (extra compares emit; values don't
  change). Gate: full-suite digest byte-identity vs main (P-E3 standard).
- **Ingress**: one new config float crossing at the quantize folds — the
  number-doors law (engine/14). Honest scope note (critique): neither
  `test_no_float_in_sim_tu.py` (SIM_TUS excludes `eos_solver.cpp`) nor the
  Python-AST ingress lint can see this change — the folds are guarded by
  review + this doc, not by a lint. `/fp:strict` membership verified: no new
  TU; `eos_solver.cpp` and all `.cu` host passes already strict.
- **`quantize` is unguarded** (`fixed_point.h:94-97`, no saturation): a
  `k_drag2 > ~2^31/(2^16·dt)` (≈7.9e5 at dt = 1/24) double→int32 cast is UB
  and on MSVC lands INT32_MIN ⇒ stage Q silently dormant. Same exposure as
  `k_drag` today — no new machinery (ACCEPTED GAP, matching status quo); the
  config comment states the sane range and this cliff in one sentence.
- No digest-membership or recorder change (`wind_x`/`wind_y` already
  recorded); no golden re-baseline in P1/P2 (R3).
- **No defaulted parameters on any new internal/C-ABI signature** (the
  silent-desync trap): the new fields thread through `KickScalarFolds` and
  every signature compile-forced. The pybind kwarg **defaults to 0.0f**
  (pinned — 7+ existing callers omit the drag dials), which is exactly why
  the documented P-V1 incident rule applies in full (§8 gate 3:
  live-solver read + exhaustive caller audit).

## 4. The named floor — `u_dead(k2)` — and its dual ceiling

`prod = trunc(kd2_q·umag / 2^16) = 0` ⇔ `umag < U0 = ceil(2^16/kd2_q)`
⇔ **`|u| < u_dead ≈ 1/(k2·dt·2^16)`** real units (exact statement is the
integer one). Anchor: k2 = 1, dt = 1/24 → `u_dead = 3.66e-4` — sub-mm/s.
Monotonicities (critique): cranking k2 *shrinks* the floor (safe direction);
small k2 grows it — at k2 = 0.01, `u_dead = 0.037`, above the ambient-drift
row, harmless only while k1 is live; if P3 sizes k1 → 0, re-check. Below the
floor stage Q is an EXACT no-op (the fast-path branch); above it the divide
removes ≥1 raw count per component per tick, so decay reaches the floor in
finite time — and at k1 = 0 the residual then **freezes inside
`[0, u_dead)`** (no 1/t tail in integers, and no decay to zero either).

The dual constant `u_ceil = u_dead·2^16 = 1/(k2·dt)` is §1(c)'s ceiling —
both derive from the same fold and BOTH are named outputs of this design.

Gate (P1 property test): derive the threshold FROM THE LIVE INTEGER CHAIN
(`kd2_q`, `sqrt_q16` floor semantics — i.e. locate the smallest integer
`umag_raw` with `mul128_shr(kd2_q, umag_raw, 16) > 0`, then cells at
`rad ⋚ U0²`), never from a printed number (R5); assert bit-unchanged below /
changed above; include **negative components and the diagonal case**, which
is what extends the trunc-division's negative-numerator attestation to
stage Q. `u_ceil`, the other named constant, is gated by §8 gate 6's
one-tick ceiling assert.

## 5. Overflow headroom

Velocity chain (all verified by the determinism critique's independent
recomputation): `|comp| ≤ RAD_SAFE = 2^30` enforced pre-clamp in all mirrors
(`cuda_kick_compression.cu:201-205`, `eos_solver.cpp:850-854`, `:1767-1771`)
and stage L only shrinks ⇒ holds at stage Q. `rad1 ≤ 2^61`;
`umag < 2^31` (self-clamp dead here); `prod ≤ 2^45.5` via 128-wide mul;
`denom ≤ ~2^45.5`; `|u|·2^16 ≤ 2^46`; quotient ≤ |u| (shrink-only) — the
store-narrow argument is unchanged. Every `kd2_q ∈ [1, INT32_MAX]` is
arithmetically safe (the DIAL-value cliff is §3's quantize note — a
different, pre-fold hazard).

**Ledger chain (new bound, from critique):** stage Q at sweep dials drives
`du2_raw → |u|²` (vs ≈4% of it under shipped linear) — up to ~24× the
per-cell term entering `ke_drag_removed`/`cnt[5]`. int64 accumulator chain
(verifier-corrected, shown): `n_bulk = N·2^16`, `du2_raw = Δu²·2^32` ⇒ each
add is `N·Δu²·2^32`; signed-int64 overflow needs
`Σ_cells N·Δu² ≥ 2^63/2^32 = 2^31 ≈ 2.147e9` (speed²·cells at N = 1) — the
whole 256² grid at ~181 simultaneously; a realistic blast (hundreds of
cells at ≤300) sits ~30× under it. **P2's assert uses the 2^31 constant.**
CPU accumulates signed int64 (overflow = UB), GPU
unsigned wrap, and the harness asserts them bit-exact — so the bound must
HOLD, not just usually hold: the P2 sweep bench asserts counter headroom
margin on its worst blast leg.

## 6. The VENTING gate (mandatory, load-bearing) + blast/heat watch

Host instrument: **transcribe `tools/tabs_pw2_venting_capture.py`** (the
committed blast + 4-tile-breach venting scenario, already deterministic,
already printing rail counters) — do NOT author fresh geometry (that exact
option was evaluated and rejected in its header; the Benches reuse rule).

Legs (P2 authors; P3 re-runs at every candidate dial):

1. **Regression fence** at shipped dials (k2 = 0): equalization profile
   captured; bound expressed as a RATIO to this same-run baseline so it
   re-derives automatically whenever k1 moves (no frozen N).
2. **k2 legs** {0.25, 0.5, 1.0}: time to **50%-equalization** (criterion
   pinned — a tight tolerance would sit in the k2-insensitive slow tail and
   have no power) ≤ 1.5× the k2 = 0 leg.
3. **Quadratic negative control**: k2 = 10 must FAIL the bound (per §1(b)
   it sits deep in the choke regime) — proves the gate can catch a
   quadratic venting death, not only the linear one.
4. **Linear negative control**: k_drag = 10 must FAIL (the 2026-08-20
   molasses, revert-the-fix validation — kept).
5. **Blast/heat watch** (closes the documented P-E5-class coverage hole —
   the two-room bench never sees blast speeds): on the blast leg,
   accumulate the four P-E3 drag counters **per tick inside the loop** —
   they are per-tick, reset at every `step()` entry (`eos_solver.h:665-
   667`; precedent `tools/velocity_clamp_pv2_measure.py:159`); a single
   read after the loop returns only the last tick's values. The set MUST
   include **`ke_drag_removed`** (`cnt[5]`, §5's assert target) plus
   `e_drag_deposit` / `e_drag_drop_sum` / `e_drag_rail_clipped`, and read
   `t_max_phys_hits`; assert deposit and rail ratios vs the k2 = 0 leg
   within measured-and-pinned factors, and §5's 2^31 headroom margin. The
   instrument has no dial knob: set `runner.eos.k_drag2` post-construction
   (the tool's own `dx` precedent at `:96`). This is where the ~24×
   deposit step is watched, ahead of P3.

**Scope caveat (pinned):** the scenario's breach is one tile ⇒
`L_neck ≈ 0.333 m` (`tabs_pw2_venting_capture.py:78`), so a green
k2 = 1.0 leg (orifice stretch 1.29 < 1.5 there) does NOT establish
§1(b)'s `k2 ≲ 0.6` band — that band is set by longer necks. P3 must not
read gate-green-at-1.0 as venting-safe-at-1.0.

## 7. Cost — honest class + the calm-cell fast path

Stage Q armed is a NEW always-on per-open-cell cost (isqrt + 2 int64
divisions; GPU int64 divide is emulated multi-instruction — the GPU is the
exposed side). Mitigation, bit-exact by construction: **skip stage Q when
`rad1 < rad_dead_q32 = U0²`** — for integer floor-isqrt,
`isqrt(rad1) ≥ U0 ⇔ rad1 ≥ U0²`, so the skip fires EXACTLY when `prod`
would be 0 and the stage a no-op. Calm cells (the vast majority, most
ticks) pay one int64 compare; sqrt + divisions run only where drag actually
acts. Warp-level: calm-majority warps skip coherently.

Timing gate (P1): CPU wall-clock via the existing tick instrument pattern
(`tests/_eos_p3_bench.py` — the actual timing instrument; `tools/bench_*`
do not time) AND a CUDA-path leg via the `run_cuda_script` harness pattern.
The scenario must be **armed (kd2_q > 0) and windy** (storm-scale field) so
the divisions execute — a calm field measures only the fast path. Budget:
< 3% tick time on BOTH backends.

## 8. Patch plan

| Patch | Content | Mode | Tier | Gate → merge |
|---|---|---|---|---|
| **P1** | Full plumbing + restructured drag block, all mirrors | subagent | Sonnet 5 | see gate list below → **auto-merge on green** (pre-authorized, R3) |
| **P2** | §6 venting/blast gate legs + k2 sweep bench extension + §5 headroom assert | subagent | Sonnet 5 | suite green (tests are the deliverable) → **auto-merge** |
| **P3** | Dial-turning (k2 live, k1 resized) | Erik at screen | — | **HUMAN-TEST**; deferred behind #48 + #6; folds into #5; §6 re-run per candidate; golden re-baseline with written rationale. EOS dials are restart-bound (convention stated at `physics_runner.py:663-668`; the EOS binds are `:472-492`) — sweep via `bench_two_room.py --set` or restart-per-value |

**P1 exhaustive site list** (critique-enumerated; a missed mirror = the
documented silent-desync incident):
`config.toml` key + comment placed ADJACENT to the k_drag/forbidden-band
paragraph (cross-referencing it; §10 states why k2 keeps its contract) ·
`eos_solver.h` member (~:211) + reference-twin signature (`eos_solver.h:
674-687`, `eos_solver.cpp:1632-1649`) · `bindings.cpp` `.def_readwrite`
(:2221) AND both free-function signatures/kwargs (:1129/:1180,
:2435/:2503) · `src/simulation/physics_runner.py:486` `_ep` bind ·
CPU folds `eos_solver.cpp:463`, `:1679` (+ `rad_dead_q32`) · CPU drag
blocks `:882-`, `:1785-` · C ABI `cuda_kick_compression.h:60-75`
(non-defaulted) · `KickScalarFolds` members + `kick_scalar_folds()`
signature (`cuda_resident.h:114-149`) · kernel params + launch + wrapper +
fold (`cuda_kick_compression.cu:113-131`, `:421`, `:443-470`, `:360-400`) ·
GPU dispatch call sites `cuda_eos_step.cu:645-663`,
`cuda_eos_resident.cu:699-708` · artifact stamp
`tools/velocity_clamp_pv2_measure.py:169` (add k_drag2 so sweeps stay
attributable).

**P1 gates (concrete — replaces v1's vacuous set):**
1. Full suite + inertness digest: byte-identical vs main at k2 = 0.
2. **Armed cross-mirror gate**, TWO pinned legs through
   `tests/cuda_kick_check.py::_run_pair`, drag-forcer scenario, BOTH cap
   regimes: `CONSTS_DRAG2 = dict(CONSTS_DRAG, k_drag2=1.0)` (main leg) and
   a `k_drag2=0.01` leg for the dead-zone straddle (U0 = 2428 raw makes
   the straddle comfortably constructible; at 1.0, U0 = 24 is too tight).
   Assert bit-identical `wind_x`/`wind_y`/`temperature` + digests + all
   counters (`cnt[0..8]`); include negative-component and diagonal cases
   (§4). Without these legs every other gate passes with stage Q switched
   off — the v1 hole both critics found independently.
3. **Pybind-default audit** (the P-V1 incident,
   `velocity_clamp_pv1_asbuilt_2026-08-19.md:79-91`): extend the live-solver
   read `tests/cuda_kick_check.py:520` with `k_drag2=float(eos.k_drag2)`;
   audit every `**CONSTS`-style caller — exhaustively:
   `tests/test_p_e3_drag.py:81,188,198,246`,
   `tests/test_p_e4_reversible_work.py:163`,
   `tests/test_thermal_mass_axis.py:645` AND its second dict `:688-691`
   (feeds `:694`/`:697`),
   `tests/cuda_thermal_mass_eos_check.py:182-186` (feeds `:191`, `:194`,
   `:197`), `tests/test_velocity_clamp_property.py:46-50` (feeds `:111`),
   `tools/velocity_clamp_pv2_measure.py:169`. The `bindings.cpp`
   forwarding sites (`:1160-1165`, `:2484-2489`) are compile-forced.
4. Dead-zone property test per §4 (live-chain-derived, signs, diagonal).
5. Timing per §7 (armed + windy, CPU and CUDA legs, < 3%).
6. **Stage-Q law gate** (closes the both-mirrors-wrong hole — bit-identity
   alone cannot catch two mirrors implementing the same wrong formula): a
   Python replay of the exact integer chain —
   `denom = 65536 + ((kd2_q·isqrt(rad1)) >> 16)`,
   `u' = trunc(u·65536/denom)` — bit-matched against
   `eos_kick_compression_ref` on a random-plus-corners field; assert
   per-component sign preservation, `|u'| ≤ |u|`, and the one-tick ceiling:
   from `|u| = U_MAX = 1000`, one armed tick at k2 = 1 lands below
   `u_ceil = 24`.

Test-conventions note: new gates follow `tests/` naming (`test_*` collected,
`_*`/`cuda_*_check.py` harness + `test_*` wrapper; GPU via `run_cuda_script`
subprocess, never importing the CUDA .pyd into pytest).

Checkpoint to memory at each boundary; re-plan between patches is normal.

## 9. Systems section (rules lifecycle)

**(a) Existing canonical systems used**: fixed-point kits (`fixed_point.h`
`sqrt_q16`/`quantize`; the CPU mirrors' file-local `mul128_shr`; device kit
`sqrt_q16_dev`/`mul128_shr_signed`; the attested trunc-division idiom);
Config via `CFG` bound in PhysicsRunner (`_ep`, restart-bound semantics);
field digest + GOLDEN_AGGREGATE; CUDA harness (`run_cuda_script` pattern,
`cuda_kick_check.py`); ingress doctrine engine/14 (with §3's honest note on
lint reach); `/fp:strict` list (no new TU); instruments:
`tools/tabs_pw2_venting_capture.py` (venting/blast host),
`tests/_eos_p3_bench.py` (timing pattern), `tools/bench_two_room.py --set`
(P3 sweeps).

**(b) New reusable systems**: none — extends the existing drag block.
**Draft CLAUDE.md rule** (corrected per critique — the v1 wording was false
against three code sites): *Interior air **momentum drag** (the storm sink)
lives only in the kick-loop staged drag block (`eos_solver.cpp` kick loops +
`cuda_kick_compression.cu`); extend its stages, never add a parallel
damping site. Separately-scoped neighbours in the same loop, NOT under this
rule: the `dyn_wave_absorb` chain and the B3c space-sponge band.* Lands at
implementation as a new EOS/atmosphere row in CLAUDE.md's Sim-core table
(which currently has none).

## 10. Accepted gaps & standing notes (decisions, not findings)

- `u³` term not shipped (R1). · Single global k2 dial (sponge-plane
  precedent if regional damping is ever wanted). · L→Q order pinned (§2's
  corrected bound). · Ledger k1-vs-k2 attribution by bench differencing
  (§2). · `quantize` dial-cliff matches k_drag status quo (§3). · Dead-zone
  floor accepted as sub-perceptual in the upward sweep direction; re-check
  if k1 → 0 at small k2 (§4). · Heat-deposit re-modeling deferred to P3/#5,
  now WITH the §6 blast watch in front of it.
- **Forbidden-band inheritance** (`config.toml:655-659`, `:1196-1203`):
  k2 keeps k_drag's contract — stage Q is sign-symmetric and strictly
  dissipative (`factor ∈ (0,1]`, trunc toward 0 on both signs, denom built
  from |u|), so it cannot rectify oscillation or pump KE; the wave_absorb
  window stays closed. Stated, not silently inherited.
- **u_ceil is a P3 sizing constraint**, not a mechanism defect: the table in
  §1(c) goes in front of Erik with the dials.

## 11. Critique resolution ledger (three lenses, 2026-08-23)

| Finding (lens: D=determinism, P=physics, S=systems) | Resolution |
|---|---|
| D1+S1 BLOCKER armed cross-mirror gate missing | §8 gate 2 |
| D2+S2 BLOCKER/MAJOR pybind-default incident + caller audit + live read | §8 gate 3 |
| P1+D6 BLOCKER dt = 1/24 not 1/60 | R5; all numbers re-anchored (§1, §4) |
| P2 BLOCKER transport inversion | §1(b) rewritten; v1 claim retracted |
| P3 BLOCKER neck choke / molasses parity k2≈0.86 | §1(b) band; §6 legs 2–3 |
| P4 BLOCKER unnamed ceiling | §1(c), §4 dual constant, §10 |
| P5 MAJOR half-life vs e-fold | §1(a) corrected table + comparison rule |
| P6 MAJOR blast/heat coverage hole (P-E5 class) | §6 leg 5 |
| P7+P8 MAJOR venting gate never runs k2 / criterion unspecified | §6 legs 1–4 (ratio bound, 50% criterion) |
| P9+S6 MAJOR cost precedent false / wrong bench home | R4 corrected; §7 fast path + armed-windy timing at the real instruments |
| D3+P14+S10 MAJOR CPU multiply name / per-mirror spellings | §2 preamble |
| D4+S3+S5 MAJOR plumbing undercount (resident folds, ABI, physics_runner, def_readwrite) | §8 site list; §3 no-defaulted-params |
| D5 MAJOR ledger headroom + CPU/GPU accumulator types | §5 ledger bound; §6 leg 5 assert |
| D7 MAJOR dead-zone gate vs floor-isqrt + signs | §4 gate |
| D8 MINOR T-rail write reachable at k1=0 | §2 behavior note |
| D9 MINOR booking attribution | §2 accepted gap |
| D10+P11 MINOR quantize unguarded | §3 note + config comment |
| P10 MINOR split-order bound wrong reason | §2 corrected |
| P12 MINOR dead-zone monotonicity + integer termination | §4 |
| P13 MINOR forbidden band + comment placement | §10; §8 site list |
| S4 MAJOR CLAUDE.md rule false | §9(b) rewritten |
| S7 MAJOR venting scenario already exists | §6 host instrument |
| S8 MINOR quadratic negative control | §6 leg 3 |
| S9 MINOR stage-L saturation guard dropped | §2 pseudocode restored |
| S11 MINOR sweep artifact attribution | §8 site list (pv2_measure) |
| S13/S14/S15/S16, D11-13, P15-16 NOTEs | R5, §6 ratio bound, §8 gates/conventions, R3 (#4 stays open; pre-auth recorded), citations fixed (`:882`, `:867-868`, wrapper call path) |
| **Verifier pass (v2.1, FIX THEN BUILD — all applied)** | fold dormant-guard `rad_dead_q32 = 0` at kd2_q=0 (§2); stage-Q law gate incl. ceiling assert (§8 g6, §4); per-tick counter accumulation + `ke_drag_removed` + dial knob (§6 leg 5); ledger chain corrected to 2^31 (§5); pinned CONSTS legs 1.0/0.01 (§8 g2); audit completions incl. `test_velocity_clamp_property` (§8 g3); floor-freeze wording (§4); 0.333 m neck caveat (§6); n_sub coupling (§1c); pybind default pinned + cite/spelling pins (§1-§3, §8) |
