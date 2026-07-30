# EMA high-pass sponge — design (2026-07-23, Fable)

**Decision context:** fire-tuning §7 Q2 resolution (`docs/fire_tuning_plan_2026-07-22.md`).
Erik's call 2026-07-23: the boundary absorber must keep eating acoustic
transients (shockwave reflections, the B3c ≤2% gate) **and pass steady wind**,
because windy levels are planned — outdoors and indoors. Chosen mechanism: damp
the **deviation from a slow per-tile running mean** of the velocity, not the
velocity itself. Fronts are fast → fully damped; steady/slow flow tracks into
the mean → passes.

**Standing technique credit (iron rule):** this is a sponge-zone damping toward
a reference state — D. J. Bodony, *"Analysis of sponge zones for computational
fluid mechanics"*, J. Comput. Phys. 212(2):681–702, 2006 — with the reference
chosen as a per-tile exponential moving average (a self-calibrating reference
rather than a prescribed target flow). The implementing file carries this
citation in its header; archive the paper under `docs/papers/`.

**Status: DESIGN — awaiting adversarial critique + Opus build (§7 kickoff).
Its OWN engine item on its own branch; NOT part of the fire-tuning arc and NOT
blocking it** (rationale in the Q2 resolution: post-sky-exchange, O₂ is
volumetric; forced-wind harness runs bypass the sponge; sealed rooms have no
band).

---

## 1. Current state (verified in code 2026-07-23)

- **Band construction** (`src/simulation/gamemap.py` `_build_sponge_grids`,
  ~line 540): multi-source BFS distance `d` from the ring through open air;
  quadratic ramp `k(d) = k_max·(W−d)²/W²` over `1 ≤ d < W`; `W = sponge_width ·
  res_factor` (base tiles); walls block the BFS, so sealed rooms carry no band.
  `k_max = sponge_u_damp` (authored, default 0.9·FP_ONE = 58982, the B3c knee).
- **Application — the kick, per substep**, immediately after the absorb chain:
  - CPU: `cpp/src/eos_solver.cpp:588` (inside the kick loop).
  - CUDA step path: `cpp/src/cuda_kick_compression.cu:168`.
  - CUDA resident path: same kernel via `cuda_eos_resident.cu` (device-resident
    `d_sponge_udamp`).
  - Form: per-component, **magnitude-first sign-symmetric shrink**
    `|u| *= (1 − k(d))` (`mul128_shr_signed` on `|u|`, sign reapplied) — the
    established idiom that avoids the arithmetic-shift negative bias.
- **Ring tiles themselves:** `u ≡ 0` hard (the still-boundary idiom,
  `cuda_kick_compression.cu:123`); mass exchange at the ring happens via the
  per-substep N clamp, not via velocity. Unchanged by this design.
- **σ pressure-sponge** (`sponge_sigma`): ships 0 (B3b measured it reflects).
  Untouched here; grids stay wired.

**The problem:** `k = 0.9` per substep annihilates ANY velocity in the band —
transient or steady. Steady wind cannot cross a `W`-deep band (multiple 0.9
bites per crossing), and in small domains the band covers most of the interior
(the fire-bench finding). Reflection absorption and steady-flow transmission
are currently the same dial; this design separates them by TIMESCALE.

---

## 2. The design

### 2.1 State

Two new **synced sim-state planes**, full-grid int32 Q16.16:

- `ubar_x`, `ubar_y` — per-tile EMA of the realized wind.

Full grid (not band-only): ≤256² × 4 B × 2 = 512 KB, trivial; keeps indexing
branchless and the digest simple. Tiles with `k(d) == 0` (everything outside
the band) are **never read or written** → they stay 0 forever and the
outside-band arithmetic is byte-identical to today (space-map identity trivially
preserved: no band, no touch).

Initialization: zeros. Cold start therefore behaves exactly like the current
sponge until ū accumulates (~one τ) — a deliberate, safe degradation: the first
seconds of a level over-damp steady wind slightly, never under-damp a transient.

### 2.2 The kick modification (both CPU + CUDA, identical staging)

Replace the current band clause. Per band tile (`kd > 0`), per substep, on the
post-absorb wide values:

```
dx = u_x − ubar_x[i]                     # deviation, int64 wide
dy = u_y − ubar_y[i]
dx' = sign_symmetric_shrink(dx, 1 − kd)  # the EXISTING magnitude-first idiom,
dy' = sign_symmetric_shrink(dy, 1 − kd)  #   applied to the DEVIATION
u_x = ubar_x[i] + dx'
u_y = ubar_y[i] + dy'
```

Then (still inside the band clause, after the u-cap chain settles the final
narrowed `u`):

```
ubar_x[i] += sign_symmetric_shift(u_x_final − ubar_x[i], S_EMA)   # α = 2^−S_EMA
ubar_y[i] += sign_symmetric_shift(u_y_final − ubar_y[i], S_EMA)
```

Design choices, fixed here (escalate if a gate contradicts them):

- **ū updates from the POST-damp, post-cap (realized) velocity.** The mean must
  track the flow the tile actually carries; tracking the pre-damp signal would
  let a sustained oscillation walk ū off zero. At small α a fast front
  contributes negligibly to ū either way.
- **α is a power-of-two** (`S_EMA` a plain shift) via the sign-symmetric shift
  idiom (shift the magnitude, reapply the sign) — a raw arithmetic `>>` on the
  signed difference floors toward −inf and would give ū a permanent negative
  creep. Same discipline as every other magnitude-first op in the kick.
- **Update per substep** (where the damping already lives — no new sync point,
  resident-path friendly). Substep count varies with CFL, so τ in wall-seconds
  varies too; acceptable because the timescale separation is ~2 orders of
  magnitude (front-crossing ≪ 1 s vs plume/weather build-up over tens of
  seconds). Deterministic regardless (substep count is itself deterministic).
- **u-cap unchanged and applied to the recomposed u** — ū + deviation can
  never exceed the cap post-clause, so no new overflow surface beyond the
  existing RAD_SAFE staging (ū is itself a capped-u average, so |ū| ≤ u_cap).

### 2.3 Timescale target

τ ≈ 5–10 s wall-time in the typical substep regime. `τ ≈ dt_sub · 2^S_EMA`;
with dt_sub ≈ 1/192 s (24 Hz × 8 substeps), `S_EMA = 10` → τ ≈ 5.3 s. **P3
calibrates the final `S_EMA`** against both gates; it ships as a config key
(`[ambient]` `sponge_ema_shift`, per-level authored like the other sponge
dials, default the calibrated value; `0` = EMA off = today's behavior, the
escape hatch).

### 2.4 What this deliberately does NOT do

- No directional (one-way) logic — rejected: crude asymmetry, damps steady
  outflow.
- No σ-rung revival, no weather system, no ring-tile change (`u ≡ 0` stays;
  steady wind lives in the interior from the first non-ring tile inward).
- No band-geometry change: `k(d)` ramp, width semantics, res_factor scaling all
  unchanged.

---

## 3. Behavior consequences (accepted)

- **Wind onset** (weather ramping up): partially damped for ~τ, then passes.
  Reads as the wind "arriving" — acceptable, arguably nice.
- **Sub-τ-frequency oscillations** (a standing wave slower than τ): leak
  partially. The spec's own standard is "imperceptibility, not perfection"; the
  reflection gate is the arbiter.
- **Sustained shock TRAINS** (many fronts over > τ): ū picks up a small mean
  bias, slightly reducing absorption of later fronts. Gate (b) below bounds
  this.

---

## 4. Gates (all must pass; goldens re-baselined once, deliberately)

a. **Space-map byte-identity** — no band ⇒ no state touched ⇒ existing space
   goldens byte-identical. Hard gate, zero tolerance.
b. **Transient reflection ≤ 2%** — re-run the B3c reflection harness
   (`tests/_ambient_reflection.py`) at the calibrated `S_EMA`, width 16;
   including a 3-front train variant for the ū-bias concern.
c. **Steady-flow transmission (NEW gate)** — impose a sustained interior flow
   across the band (harness: forced upstream u or an authored pressure
   gradient); after ≥ 5τ, band-exit |u| ≥ **80%** of band-entry |u| (vs ~0%
   today). Threshold provisional — Erik may re-set it at review.
d. **Sealed-room conservation** — untouched (the clause never writes N), the
   existing conservation gate must stay green as-is.
e. **CPU↔CUDA lockstep** — step path AND resident path bit-identical over the
   gate scenarios (extend `cuda_eos_step_check` / the resident check with a
   banded ambient case exercising ū accumulation).
f. **Determinism digest** — `ubar_x/y` enter the synced-state digest, save
   format, and the undo log (they are true sim state: not derivable, they
   accumulate history).

---

## 5. Patch plan (Opus; own branch `ema-sponge`; autonomous-patch-workflow)

- **P1 — CPU.** State planes on GameMap (+ save/digest/undo wiring), kick
  clause rewrite in `eos_solver.cpp`, unit tests (shift symmetry, cold-start
  identity-to-today for the first substep, off-switch `S_EMA=0` byte-identity),
  gate (a), gate (d).
- **P2 — CUDA.** `cuda_kick_compression.cu` + resident-path plumbing
  (`d_ubar_x/y` device-resident, H2D/D2H at the existing sync seams), gate (e).
- **P3 — Calibration.** Sweep `S_EMA` ∈ {8..12} against gates (b) + (c); pin
  the default; config key + level.toml plumbing (`level_loader` validation like
  `sponge_u_damp`'s); re-run the full B3c reflection curve for the record.
- **P4 — Close.** Golden re-baseline (one deliberate pass, written rationale),
  fold into `docs/architecture/engine/04_atmosphere_and_pressure.md` (the §
  quoted at line 86 currently describes the plain `|u| *= (1−k)` band), archive
  this doc, HUMAN-TEST: Erik feels a windy bench + a grenade at the boundary.

**Escalation triggers (stop and come back to Fable/Erik):**
1. No `S_EMA` satisfies gates (b) AND (c) simultaneously — the timescale
   separation assumption failed; design revisit.
2. Digest/save schema change turns out to ripple beyond adding two planes
   (recorder, netcode framing, undo-log size math).
3. Resident-path throughput regression > 5% on the S8a benchmark scenario.
4. Long-run drift: any nonzero ū on a tile whose true mean flow is zero
   (shift-bias symptom) in a 10-min still-air soak.
5. Anything forcing a ring-tile (`u ≡ 0`) or band-geometry change.

**Interaction note for scheduling:** independent of the sky-exchange (Q2
Option A) build — different state, different pass, no shared dials. Either may
land first; the fire-tuning arc depends on NEITHER for its remaining queue
except the final windy-phase dials (`k_wind_fan` re-tune happens after this
lands).
