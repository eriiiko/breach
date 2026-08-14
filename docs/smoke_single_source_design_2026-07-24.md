# Smoke single-source + O₂-dependent soot — design (2026-07-24, Fable)

**Context:** fire-tuning §7 **Q6** (`docs/fire_tuning_plan_2026-07-22.md`), plus
**Erik's architectural call 2026-07-24**: fire smoke should have ONE source —
the physically-bookkept combustion channel — not two. Decisions (Erik):
1. **DELETE source A** (the fire logistic's `smoke_emission·I` scatter) —
   phenomenological, conjures mass from nothing, duplicates B's purpose.
2. **Source B (combustion soot) becomes the sole fire-smoke source**, with the
   Q6 law on its yield: starved burn → dirty burn.
3. **Soot mass comes from O₂ AND fuel** — physically truer (real smoke is
   mostly fuel carbon) and it widens B's small mass budget.

**Physics (the Q6 verdict):** soot/smoke yields rise steeply under
ventilation-controlled (O₂-starved) burning — Tewarson, *SFPE Handbook*,
"Generation of Heat and Chemical Compounds in Fires" (cite in the implementing
file header; archive under `docs/papers/`). Emergent shape worth stating: total
soot = consumption (falls as the fire starves) × yield (rises) → smoke output
**peaks mid-choke and stops at death** — the real "billows black, then gone"
arc, for free.

**Status: DESIGN — blessed direction, awaiting build. Depends on
`o2-continuous-law` (needs `o2f`); build AFTER it, stacked on the same
integration line. NOT part of the running builds' scope.**

---

## 1. Current state (verified in code 2026-07-24)

- **Source A** — fire step scatter: `delta = smoke_emission·dt·I` into open
  4-neighbours (`cpp/src/fire_simulation.cpp` + `cuda_fire.cu:264`;
  `smoke_emission = 0.8`). The dominant visible smoke today (~0.5 N/s at
  I≈0.6). To be DELETED.
- **Source B** — combustion pass (`cpp/src/combustion.h:47`): per burn site,
  `SOOT += round(burn·soot_yield)`, `inert += burn − soot` (decision #12
  identity), `soot_yield = 0.3` fixed. Post-Q1 magnitude:
  `burn = burn_rate·I·o2f ≈ 0.012 N/s` steady crate — ~40× below A.
- **Explosion smoke** (`_SRC_EXPLOSION_SMOKE`) — separate legitimate source,
  UNTOUCHED.
- **Plane facts** (`src/simulation/gases.py`): `smoke` (id 1) is a TRACE
  plane — NOT in the conservative pair, therefore **not in the EOS N_total:
  trace deposits are pressure-neutral** (verified, the design's load-bearing
  fact). Decision #12's identity today moves `soot` mass OUT of the bulk pair
  (inert gets `burn − soot`) into the trace plane — bulk pressure drops by the
  soot share; that semantic is KEPT for the O₂ share below. Smoke `decay` is
  loaded but NOT applied in transport (M1 note) — density is
  advection/diffusion-limited.
- C++ still spells the plane `black_smoke` (cosmetic, gases.py TODO(B2)).

## 2. The design

### 2.1 Yield law (Q6 proper)

One dial-pair replaces fixed `soot_yield`; `o2f` is the Q1 factor, already
available at the burn site (Q1 doc §2.3 reserved the signature):

```
y(o2f) = y_clean + (y_starved − y_clean) · (FP_ONE − o2f)     # Q16, linear
```

`[physics.combustion] soot_yield_clean` default **0.05**,
`soot_yield_starved` default **0.5** (placeholders — felt out at the joint
re-tune; tombstone the old `soot_yield`).

### 2.2 Two-term soot deposit (per burn site, inside the existing pass)

```
soot_o2   = min( mul_q16(y, burn), burn )      # O₂-mass share — decision #12
inert    += burn − soot_o2                     #   identity UNCHANGED, cap keeps
                                               #   inert non-negative by design
soot_fuel = mul_q16(k_fuel_soot, mul_q16(y, burn))   # fuel-mass share: NEW,
                                               #   trace-only deposit, pressure-
                                               #   neutral (§1); the fuel mass
                                               #   is already drained by the
                                               #   existing fuel_per_o2 path
smoke    += soot_o2 + soot_fuel
```

`[physics.combustion] k_fuel_soot` default **1.0** (fuel share ≈ O₂ share;
re-tune dial). No hp↔N exchange rate is invented: `wall_hp` stays in hp units,
the fuel drain is unchanged — `k_fuel_soot` is the authored coupling.

### 2.3 Source A deletion

Remove the smoke scatter from the fire step (CPU + CUDA); retire
`smoke_emission` with a tombstone comment. The fire step no longer writes the
gas array at all through that path (check nothing else rode the scatter loop).

### 2.4 The scale migration (the one real cost)

B-only smoke is ~10–40× thinner than A's field. Consumers of smoke DENSITY
re-gain ONCE, at their read seams — **prefer the single natural seam**: the
gas table's per-plane optics (`[gases.smoke]` absorption / scatter_albedo,
density-weighted in the raycaster) carries most of it. Build task: INVENTORY
all readers (raycaster optics, `smoke_glow`, renderer smoke draw/alpha, any
vision/AI or gameplay thresholds, B2 medium hooks) and re-gain each — with a
bench A/B screenshot pair as the acceptance (a steady crate's plume reads
comparably to today; a choked room reads BLACKER than today).

## 3. Gates

a. **Conservation** — bulk-pair budget unchanged in FORM (inert += burn −
   soot_o2, cap enforced); trace smoke excluded from the bulk budget as today;
   `test_atmosphere_conservation` / `test_multigas_structure` green.
b. **Pressure neutrality of `soot_fuel`** — a unit test that deposits a large
   fuel-share soot and asserts EOS P bit-identical (the §1 fact, enforced).
c. **CPU↔CUDA lockstep** — combustion + fire-step mirrors bit-identical
   (extend `cuda_combustion_check` with a starved-yield case; fire check loses
   its smoke-scatter assertions).
d. **Behavioral acceptance (harness + eyeball):** well-fed crate → thin/clean
   plume; sealed-room burn → smoke output PEAKS during the choke, stops at
   death; burnt-out cold tile emits nothing.
e. **Digest/goldens** — move by design; ride the SAME single deliberate
   rebase as the joint re-tune (this build stacks on the integration line, not
   straight onto main — see §4).

## 4. Build order & scope

Branch `smoke-single-source`, stacked ON `o2-continuous-law` (needs `o2f`).
- **P1** — yield law + two-term deposit + config keys + gates a/b (CPU).
- **P2** — source-A deletion (CPU+CUDA fire step) + combustion CUDA mirror +
  gate c.
- **P3** — consumer inventory + re-gain (render/optics seams; render-layer
  changes are determinism-exempt) + gate d screenshots for Erik.
- Joins the joint re-tune (`y_clean`/`y_starved`/`k_fuel_soot` felt on the
  bench) → ONE golden rebase → HUMAN-TEST → main, together with the rest.

**Escalation triggers:** the §1 pressure-neutrality fact turns out wrong
anywhere (a consumer sums trace planes into N_total); the scatter-loop removal
breaks something that secretly rode it; consumer inventory finds a gameplay
system whose re-gain changes balance (vision/AI); soot LSB underflow at
smolder-scale burns (burn·y < 1 LSB per tick — check at I≈0.05).
