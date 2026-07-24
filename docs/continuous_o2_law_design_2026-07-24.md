# Continuous O₂ → combustion law — design (2026-07-24, Fable)

**Context:** fire-tuning §7 **Q1**, folded together with the **Q3 burn_rate
decision** (`docs/fire_tuning_plan_2026-07-22.md`, "Q3 MEASURED"). Erik's
intent: O₂ dependence should be proportional, not a gate; `burn_rate` drops to
**0.02** (the `ceiling_h`-anchored physical value, ~1/50) and is then never
touched again without a real reason. The 2026-07-23 harness run proved these
cannot land separately — the blessed still-air flame was O₂-self-starvation-
limited, so the law change + the rate change + the dial re-fit are **ONE
build + ONE re-tune + ONE golden rebase + ONE HUMAN-TEST**.

**Realism verdict (the design's justification):** a continuous law is MORE
realistic than the current gate, on three counts.
1. Measured compartment-fire burning rates decline ~**linearly** with O₂
   volume fraction below ambient — Peatross, M.J. & Beyler, C.L.,
   *"Ventilation effects on compartment fire behavior"*, Fire Safety Science
   5:403–414, 1997 (the linear burning-rate–vs–O₂ correlation). Not a step.
2. Real flaming combustion has an **extinction limit**: flames cannot be
   sustained below ~13–16 vol-% O₂ (Beyler, *SFPE Handbook*, flammability
   limits chapter). Today's gate keeps FULL burn down to 0.03 absolute
   (≈14% of one atmosphere's O₂ *density*, far below any real flame's limit)
   and only cuts off at 0.01 — physically indefensible in both directions.
3. O₂ consumption should scale with how hard the fire burns. Today's draw is
   flat per site (combustion.h: `demand_i = burn_rate*dt`, uniform) — an
   I=0.1 smolder consumes like an I=1.0 blaze.
Erik's "10% O₂ → 10% burn" is the right *shape*; physically flames die at
~13%, so the law below carries an extinction-limit dial `X_ext` — at the
physical default it is Peatross–Beyler-realistic, at `X_ext = 0` it
degenerates to Erik's pure proportional. Both regimes in one law.

**Credit (iron rule):** implementing files cite Peatross & Beyler 1997 (the
linear law) and Huggett 1980 (oxygen-consumption calorimetry, ~13.1 MJ/kg O₂ —
the `burn_rate` anchor) in their headers; archive both under `docs/papers/`.

**Status: DESIGN — awaiting Erik's read + Opus build. Own branch
(`o2-continuous-law`). Independent of the sky-exchange build (different
files); the joint RE-TUNE waits for BOTH to merge.**

---

## 1. Current state (verified in code 2026-07-24)

- **Intensity gate** — `cpp/src/fire_simulation.cpp:134–165` (CUDA mirror
  `cuda_fire.cu`): per burning tile, mean **absolute** `n_o2` over open
  4-neighbours (order-free sum + `mean_round`), then Hermite
  `smoothstep(P_min=0.01, P_full=0.03)` → `o2 ∈ [0,1]`; `avail = F · o2`.
- **Combustion draw** — `cpp/src/combustion.h` (+ CUDA kernels): sites with
  `O2[j] > o2_thresh_burn` draw `demand = burn_rate·dt` **uniform** (no I, no
  O₂ proportionality), yield `H_fuel` heat, `soot_yield` soot, rest → inert.
- **The density trap:** the gate reads absolute density, so THERMAL EXPANSION
  (hot gas, P=C·N·T → low local N) reads as "no oxygen" even at ambient
  *composition*. This forced the whole v2.4 "hot-zone-equilibrium" rescale of
  `P_min`/`P_full` (config.toml comment block) and produced harness quirk #1
  (a crate deep in a room self-starves planetside). The gate conflates HOT
  with VITIATED.

## 2. The law

### 2.1 One shared O₂ factor, computed on the MOLE FRACTION

Per evaluation site, from the open-4-neighbour sums (both sums order-free,
exact int64, as today):

```
X      = Σ n_o2 / Σ n_total            # O₂ mole fraction of the local mix
                                       # (fraction of the sums — one division,
                                       #  via the existing reciprocal_q16 on
                                       #  max(Σ n_total, n_floor))
o2f    = clamp01( (X − X_ext) / (X_amb − X_ext) )     # linear, Q16
```

- `X_amb = 0.21` (reads the authored `[ambient] o2_frac` when present; 0.21
  fallback — one source of truth with the BC).
- `X_ext` — **new config dial** `[physics.fire] o2_frac_ext`, default
  **0.13** (physical flame-extinction limit). `0` = pure proportional.
- The reciprocal of the span `(X_amb − X_ext)` is hoisted once per tick
  (host-side), like today's `recip_P_span`.
- Fraction is INVARIANT under thermal expansion → the density trap and the
  v2.4 rescale saga close; only true vitiation (combustion products
  displacing O₂) starves a fire. Harness quirk #1 substantially resolved.

### 2.2 Intensity logistic — drop-in replacement

`o2f` replaces the smoothstep `o2` in `avail = F · o2f`. Nothing else in the
logistic changes. (The `smoothstep_q` helper itself stays — other users.)

### 2.3 Combustion draw — proportional in BOTH intensity and O₂

```
demand_i = burn_rate · I_i · o2f_i · dt        # was: burn_rate · dt, gated
```

- `burn_rate = 0.02` lands HERE (config edit inside this build, with the
  written rationale; the Q16 quantization of the per-tick demand must be
  checked for underflow — 0.02/tick·… stays ≫ 1 LSB at 24 Hz, verify in P1).
- `o2_thresh_burn` is RETIRED as a gate; keep only as an epsilon floor
  (skip-site guard at effectively-zero O₂) or delete — implementer's call,
  documented either way.
- Heat/soot/inert yields per unit O₂ unchanged — heat output now scales with
  `I·o2f` automatically, which is the physical statement "a choked fire is a
  cool fire". (Q6's O₂-dependent `soot_yield` builds on this same `o2f` later
  — design the signature so `o2f` is available to the yield computation.)

### 2.4 Ignition seeding

`apply_temperature_ignition`'s "has local O₂" check switches to the same
fraction test (`X > X_ext`) — ignition and sustain read one law.

**Bundled fix, NEEDS ERIK'S EXPLICIT NOD (small, same files): zombie
smolder** (§5.3 quirk 2) — re-seeding requires remaining fuel
(`wall_hp > 0`), so a burnt-out tile stops re-igniting every tick while hot.
Flagged as its own patch (P1b) so it can be dropped from scope cleanly.

## 3. Gates

a. **Law unit tests** — `X = X_amb → o2f = 1`; `X ≤ X_ext → 0`; midpoint
   linearity; `X_ext = 0` degenerates to `X/X_amb`; Q16 rounding
   sign-symmetric.
b. **CPU↔CUDA lockstep** — fire step + combustion bit-identical
   (extend `cuda_fire_check` / `cuda_combustion_check` with vitiated-mix and
   hot-thin-gas cases — the trap case: hot tile at ambient COMPOSITION must
   now burn).
c. **Determinism digest** — moves by design (sim-path change); ONE deliberate
   golden re-baseline at the END of the joint re-tune, not per-patch.
d. **Behavioral acceptance (harness, pre-re-tune):** hot tile at ambient
   composition burns (trap closed); sealed-box burn now shows monotonic
   decline of I as X falls (continuous choke, no cliff); a burnt-out crate
   goes OUT and stays out (if P1b lands).

## 4. Patch plan (Opus; branch `o2-continuous-law`)

- **P1** — CPU: `o2f` (fraction + linear law) in fire_simulation + combustion
  + ignition; config keys (`o2_frac_ext`, burn_rate=0.02 + rationale;
  P_min/P_full retired from the fire block with a tombstone comment); unit
  tests (gate a).
- **P1b** — zombie-smolder fix (Erik-gated scope).
- **P2** — CUDA mirrors (cuda_fire.cu, combustion CUDA) + lockstep (gate b).
- **P3** — harness acceptance runs (gate d) + handoff notes for the joint
  re-tune (measured peak/T/burnout at the new law, locked dials untouched).
- **NO golden rebase in this build** — goldens move once, after the joint
  re-tune (see §5).

**Escalation triggers:** Q16 underflow in the per-tick demand at
burn_rate=0.02; the fraction division perturbing the resident-path seam;
any need to touch the EOS/temperature side; P1b turning out non-local.

## 5. The joint re-tune (AFTER this + sky-exchange both merge)

One session, old fire-tuning chat, sponge-safe bench (≥ 80×36):
`burn_rate` FIXED at 0.02 → re-fit `k_grow/k_die` (peak ≈ 0.6 @ ~3 min;
expect the ratio to move toward ~1:1 since `avail` now saturates at F≈0.7 —
I_eq math in §5.2 of the tuning plan), `wall_damage` (≈ 0.083 for ~7.5 min
burnout at Ī≈0.8 — bench decides), `X_ext` (feel: how visibly fires gasp),
sky `τ` (from the sky-exchange doc). Then: ONE golden rebase with rationale,
HUMAN-TEST, merge.
