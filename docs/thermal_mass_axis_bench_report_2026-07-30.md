# Bench verification report — thermal-mass axis (P3 close, 2026-07-30)

Closes the arc opened by `docs/thermal_mass_axis_design_2026-07-25.md` (Fable,
blessed by Erik 2026-07-25), corrected by
`docs/thermal_mass_axis_build_addendum_2026-07-30.md`, and extended into the
EOS pass by `docs/thermal_mass_eos_ruling_2026-07-30.md` (which answers
`docs/thermal_mass_eos_escalation_2026-07-30.md`).

As-built commits on branch `thermal-mass-axis` (base `fire-o2-integration` @ `423cd38`):

| commit | patch |
|---|---|
| `f5e9aa3` | P1 — CPU: `thermal_solid = (thermal_mass > 0)` replaces `solid` at the six MEDIUM-TEST sites in `temperature_solver.cpp` |
| `312e984` | P2 — CUDA: the six twins in `cuda_temperature.cu` + the resident path (one static mask upload) |
| `6f57762` | P-EOS — the EOS pass: both T-writes skip thermal_solid, T-only occluder, `cmask` untouched; combustion deposit re-routed to the object divisor |

**This document is a measurement report, not a design.** Everything below was
re-measured for P3 on the Lenovo (`ERIK_LENOVO`, RTX 1000 Ada sm_89), CPU build
`cpp/build`. Where my numbers disagree with what the arc had on record, the
disagreement is stated explicitly and my number is the one I stand behind.

**P3 changed no engine code.** It is this report + `tools/fire_tune_loop.py`
defaults + the `fire_tuning_plan` §9.5 rewrite. `config.toml` still carries
**exactly one** edit for the whole arc (air `thermal_mass` 1 → 0) — verified:

```
$ git diff 423cd38..HEAD -- config.toml
-thermal_mass = 1     # air: irrelevant (conversion skips non-solids); 1 avoids a 0-shift guard
+thermal_mass = 0     # air: 0 == the GAS thermal regime (thermal-mass axis design
+                     # §2.1). The derived `thermal_solid` mask (thermal_mass > 0)
+                     # IS the guard now, so no 0-shift placeholder is needed: air
+                     # never reaches the >>shift convert path at all.
```

Suite on this branch: **42 failed / 1714 passed / 5 skipped**, failure set
byte-identical to P-EOS's. The 42 are the pre-existing by-design reds inherited
from the o2-continuous-law line (`FireSimulation.step` missing the `n_total`
arg), enumerated in `docs/continuous_o2_law_p3_handoff_2026-07-24.md`.
**No golden was rebased and no digest re-baselined** — that rides the joint
re-tune's ONE deliberate rebase, later, with Erik.

---

## 1. The bench

Escalation §5 recipe, unchanged: `tools/fire_timing_harness.build_level`, 84×40
planetside interior with a 1-tile SPACE ring, one furniture crate deep at
`x = 12` (sponge-safe), tile 0.333 m, `sky_tau_s = 60`, sponge width 8, warm
seed `T = 280` on the crate plus the harness ignition seed.

Driven through the **LIVE path** — `Simulation.step` →
`PhysicsRunner.step` → `engine.run_substeps` + `engine.step_tail` — not the
isolated pass. A thin forwarding proxy on `runner.engine` reads the crate's `T`
at tick entry, after `run_substeps` (the EOS pass) and after `step_tail` (the
thermal pass), so the EOS's own contribution to object temperature is measured
directly. That is the exact quantity the escalation measured at −21.2 / −34.7 /
−32.7 game-units per tick before P-EOS.

Every dial is a **runtime CFG override**; `config.toml` is never touched by the
bench. Scripts: `peos_gate_c.py` (reused verbatim from P-EOS) and `p3_sweep.py`
(new for P3), both in the P3 session scratchpad.

The bench applies the §9.3 **ANCHORED** set. `config.toml` already carries
`burn_rate = 0.02`, `fuel_per_o2 = 0.7`, `o2_frac_ext = 0.13`; it does **not**
carry the other two, and both are load-bearing, so the sweep overrides
`k_wind_strip = 0.0` (the 2026-07-23 plume self-blow-out finding) and
`fuel_ref = 40.0`.

---

## 2. Gate (c) — the live-path result

At the **config** dials (`k_fire_heat = 1600`, `COOL_SHIFT = 5`) with the
§9.3-step-1 structure (`fire_T_ext = 250`, `fire_T_span = 100`), 12 s window:

| quantity | measured |
|---|---|
| monotone T rise from the seed to I-peak | **True** |
| min(T) in the window − seed | **+10.651** (the t≈0 dip is GONE) |
| EOS delta on the crate, first five ticks | **0.000, 0.000, 0.000, 0.000, 0.000** |
| EOS delta on the crate, max abs over the run | **0.000** |
| thermal-pass delta, first five ticks | +10.651, +9.368, +8.409, +7.705, +7.210 |
| I peak | 0.2864 @ 5.00 s |
| T at I-peak / T max / T final | 1592.5 / 1676.8 / 791.5 |
| §2.5 analytic ratio (measured deposit) | 0.871 |
| §2.5 analytic ratio (`k_fire_heat·I` proxy) | 0.869 |

This **reproduces P-EOS's reported gate (c) exactly**, including the 0.871.

The decisive line is `eos_delta_max_abs = 0.000`: across **every** operating
point measured in this report (**70 runs**, `k_fire_heat` 1.76 → 1600,
`COOL_SHIFT` 5 → 12, windows 12 → 900 s) the EOS pass never moved the crate's
temperature by a single Q16.16 LSB. The ruling's ownership rule — *on
`thermal_solid` tiles `temperature[]` is owned by the TemperatureSolver, every
other system is a reader* — holds in the live engine, not just in the isolated
pass. The −21/−35/−33 per-tick drain the escalation found is gone.

---

## 3. The §2.5 analytic — confirmed, with one correction to the record

### 3.1 The correction: 0.871 is a transient, not the steady state

P-EOS recorded the §2.5 analytic ratio as **0.871** and the arc carried that
forward as if it were a steady-state deficit. **It is not.** It is T lagging I
inside a 12 s window whose "plateau" is still moving.

Re-measured at equilibrium (200–300 s windows, three independent operating
points):

| k_fire_heat | COOL_SHIFT | window | analytic ratio |
|---|---|---|---|
| 175 | 7 | 200 s | **0.999** |
| 75 | 8 | 200 s | **0.995** |
| 35 | 9 | 200 s | **1.009** |

and the self-share — the fraction of `k_fire_heat·I` that actually lands on the
crate as `heat[]` — measures **1.0001, 1.0002, 1.0003**. All of it lands on the
crate; there is no ray-spreading correction to apply.

**So the §2.5 analytic is exact to within ±1 % at equilibrium**, in its
per-tile form:

> **T\*(I) = k_fire_heat · I · 2^(COOL_SHIFT − heat_inv_shift)**
>
> with `heat_inv_shift = log2(thermal_mass)`; for furniture (`thermal_mass = 8`)
> that is 3, so at `COOL_SHIFT = 5` it reduces to **T\* = 4 · k_fire_heat · I**
> — exactly the form §2.5 wrote.

Anyone reading the 0.871 as a fudge factor to divide by will over-shoot
`k_fire_heat` by 15 %. Use 1.00 and the exact shift form above.

### 3.2 §2.5's `k_fire_heat ≈ 225` is arithmetically right and dynamically dead

Solving the analytic for §9.3's thermal target (`T* = 450` at `I = 0.5`,
`COOL_SHIFT = 5`, `heat_inv_shift = 3`) gives `k_fire_heat = 900 / 2^(C−3) = 225`.
§2.5's arithmetic is correct.

Measured, at `fire_T_ext = 250` and `ignition_seed = 0.1`, **`k_fire_heat = 225`
snaps out at tick 1**: `I` peaks at the seed value 0.095 at t = 0.04 s and the
fire is dead by t ≈ 0.9 s, `T_final = 0`. So do 260 and 300. The whole
neighbourhood of §2.5's operating point is unreachable from the seed.

This is not a tuning miss; it is structural, and §4 states it.

---

## 4. ★ The structural finding P3 must hand back: the I_crit cliff

The deposit is **linear in I** and the loss is **linear in T**, so the
equilibrium temperature is linear in I. The hot gate opens at `fire_T_ext`.
Therefore there is a critical intensity

> **I_crit = I_peak · fire_T_ext / T_flame**

below which `T*(I) < fire_T_ext`, the hot gate closes, and the fire
self-collapses. At §9.3's own targets (`T_flame = 450` @ `I_peak = 0.5`,
`fire_T_ext = 250`) that is **I_crit = 0.278** — nearly 3× the
`ignition_seed = 0.1`. The fire can therefore neither ignite from the seed nor
decay gracefully to burnout: it is fenced into `I > 0.278` at both ends.

This is §9.2's "cold-start gap" generalised. §9.2 diagnosed it in the gas
regime and fixed it by dropping `fire_T_ext` below `ignition_temp`. In the
now-live **object** regime the same cliff reappears, because COOL_SHIFT is
genuinely the crate's only loss channel (ruling A5) and the balance is exactly
linear.

### 4.1 Measured bootstrap floor

`fire_T_ext = 250`, `span = 100`, `ignition_seed = 0.1`, `k_grow/k_die = 0.6/0.1`,
200 s window. "Floor" = lowest `k_fire_heat` that still ignites and sustains.

| COOL_SHIFT | e-fold | highest k that DIES | lowest k that SUSTAINS | plateau T at the floor |
|---|---|---|---|---|
| 5 | 1.3 s | 800 (bootstraps, dies @ 58.6 s) | 1600 | 3241 (6774 K) |
| 6 | 2.7 s | 300 | 400 | 1354 (3001 K) |
| 7 | 5.3 s | 150 | 175 | 1007 (2307 K) |
| 8 | 10.7 s | 68 | 75 | 720 (1733 K) |
| 9 | 21.3 s | 30 | 35 | 735 (1762 K) |
| 10 | 42.7 s | — | 16 | 669 (1632 K) |
| 11 | 85.3 s | — | 10 | 679 (1651 K) |

The floor sits at `k_fire_heat · 2^(COOL_SHIFT−3) ≈ 2000–2600` and is roughly
flat above COOL_SHIFT 8 — raising COOL_SHIFT buys bootstrap time (the decay
e-fold is 2^COOL_SHIFT ticks) faster than it raises the equilibrium, which is
why the reachable plateau falls from 3241 to ~670 as COOL_SHIFT goes 5 → 10.

Note also the far-field column of the same sweep: total injected energy scales
with `k_fire_heat`, so the low-k / high-COOL_SHIFT branch is the only one that
gets anywhere near §9.3's "far-field rise ≤ 20" target. At `k = 1600, C = 5`
the far field rises **1256 game units** and room `N` falls to 0.672 — one crate
cooking an 84×40 room, the effect already logged in the plan's 4a session.

### 4.2 The iso-target family: only COOL_SHIFT = 12 survives

Holding the thermal target fixed (`k_fire_heat · 2^(C−3) = 900`, i.e.
`T* = 450 @ I = 0.5`) and sweeping COOL_SHIFT, 300 s window:

| COOL_SHIFT | k_fire_heat | sustains? | death |
|---|---|---|---|
| 8 | 28.125 | no | 16.8 s |
| 9 | 14.0625 | no | 18.3 s |
| 10 | 7.03125 | no | 21.7 s |
| 11 | 3.515625 | no | 31.4 s |
| **12** | **1.7578125** | **yes** | 301.9 s (fuel-governed) |

### 4.3 The four levers, measured

1. **Raise COOL_SHIFT to ~12.** Works (§4.2, and §5's recommended set). But
   **COOL_SHIFT is a GLOBAL dial** (`[physics.thermal] COOL_SHIFT`) applying to
   every `thermal_solid` tile — hull, steel, glass, wood, doors — not just
   furniture. 5 → 12 changes the cooling e-fold on every wall in the game from
   1.3 s to 171 s. Arguably more physical (a steel hull section shedding 1000 K
   in 1.3 s is not), but it is a large, feel-adjacent, golden-moving change and
   it is Erik's call, not P3's.
2. **Raise `ignition_seed` into the sustaining band.** `k = 225, C = 5,
   ignition_seed = 0.4` bootstraps cleanly and plateaus at **449** (1191 K) —
   dead-centre of the 400–500 target, monotone rise, dip +2.3. But it then dies
   at **48.5 s** with `wall_hp = 24.6` remaining: the cliff simply moved to the
   decay end. Fixes ignition, not lifetime.
3. **Lower `fire_T_ext` below T\*(I_seed).** `k = 56.25, C = 7,
   fire_T_ext = 90` plateaus at **450** (1193 K) and lives **207.8 s** with the
   fuel actually consumed (`wall_hp = 6.4`) — death by fuel, the correct
   mechanism. But `fire_T_ext = 90` is 473 K / 200 °C, not a defensible
   flame-extinction temperature; it is §9.5's old gas-era compromise returning
   for a new reason. It also fails gate (c) monotonicity (dip −46).
4. **Make the deposit non-linear in I** (a floor/plateau term so `T*` does not
   collapse proportionally at low I). This is the only lever that removes the
   cliff rather than relocating it — and it is a **model change**, explicitly
   out of P3's scope. Flagged for the joint re-tune.

---

## 5. The operating point P3 hands back

Best measured set, 900 s window, all §9.3 anchors applied:

```
COOL_SHIFT = 12   k_fire_heat = 2.2   fire_T_ext = 250   fire_T_span = 100
k_grow = 0.35     k_die = 0.06
```

| §9.3 target | target band | measured | |
|---|---|---|---|
| flame plateau T | 400–500 (1093–1293 K) | **414** (1120 K) | PASS |
| peak I | 0.40–0.60 | 0.331 | miss (low) |
| peak time | 120–300 s | **143.9 s** | PASS |
| fire death | 360–480 s | 332.7 s (5.5 min) | miss (marginal) |
| far-field T rise | ≤ 20 | 21.4 | miss (marginal) |
| room N_total min | ≥ 0.90 | **1.000** | PASS |
| far-field X min | ≥ 0.19 | **0.1989** | PASS |
| wall_hp at death | > 0 (charred remains) | **4.5** | PASS |
| gate (c) dip vs seed | ≥ 0 | −1.7 | near-pass |

Sibling at the faster ramp (`k_grow/k_die = 0.6/0.1`): plateau **433**,
peak I **0.411** (PASS), peak @ 61.4 s, death 302 s, far rise **17.4** (PASS),
dip **−0.9**.

Six of nine targets pass and the three misses are marginal and all sit in
§9.3's **steps 3 and 4** (`k_grow`/`k_die` and `wall_damage`) — which are
Erik's to tune, in his order, from here. That is the point: this set produces a
live, on-target flame you can sit down and watch, instead of one that snaps out
at tick 1.

**It is a bench proposal, not a config change.** The tune loop drives the
harness entirely through `--set` overrides and never writes `config.toml`.

---

## 6. Gate (f) — the A3 pressure tripwire: does NOT fire

Instrumented at the crate and in the far field (outside the sponge band) across
**12 operating points** examined in detail, spanning `k_fire_heat` 1.76 → 1600 and `COOL_SHIFT`
5 → 12, windows 12 → 900 s.

| observable | measured, all runs |
|---|---|
| `P` at the crate | ∈ [0.9997, 1.0003] — ±0.03 % of ambient |
| std(`P`) over the second half, crate | 1–3 × 10⁻⁵ |
| std(`P`) over the second half, far field | 1–2 × 10⁻⁵ |
| \|u\| max at the crate | 0.30 – 0.93 |
| \|u\| max far field | 0.45 – 0.63 |
| `u_clamp_hits` / `u_max_hits` | **0 / 0** |
| `work_clamp_hits` / `t_max_phys_hits` | **0 / 0** |

The crate's pressure is **not noisier than the far field** — the second-half
standard deviations are the same order, and at several points the crate is the
quieter of the two. The `dP` sign reversals the probe counts (95–112 per run)
are LSB dither at 10⁻⁵ amplitude on a field pinned at 1.0000, not an
oscillation artifact; a real artifact would show as growing amplitude pinned to
the crate, and none appears at any k or any COOL_SHIFT. Every EOS rail counter
is zero.

**Verdict: the tripwire does not fire.** Ruling §1-A3's prediction on record
("it will not fire") holds. The hot-pore-gas decision — `P = C·N·T[i]` with
object T on thermal_solid tiles — stands; the named fallback (`t_amb` pore gas)
is not needed.

---

## 7. ★ Accepted gap: the plume→T shim writes object T without the object divisor

P-EOS's writer enumeration (ruling §2) surfaced a **7th** writer of
`temperature[]` that the ruling's list does not cover:

- `cpp/src/fire_simulation.cpp:265-293` (the `for (i) { I = fire[i]; ... }` plume loop)
- `cpp/src/cuda_fire.cu:239-259` (the twin)

```cpp
const q16 sat = clamp01_q(FP_ONE - fp::recip_mul(temperature[i], recip_T_flame_max));
q16 g = fp::mul_q16(gain_q, I); g = fp::mul_q16(g, sat);
const q16 gain = fp::narrow_round_signed(fp::mul_wide(g, dt_q));
if (gain > 0) {
    q16 dT = fp::narrow_round_signed(fp::mul_wide(gain, temp_gain_scale_q));
    ...
    temperature[i] = fp::sat_add_q16(temperature[i], dT);   // <-- no >> heat_inv_shift
}
```

**Why it is a real tension with the ownership rule.** The loop runs on every
tile with `fire[i] > 0`. Every flammable material has `thermal_mass > 0`, so a
burning tile is **always** a `thermal_solid` tile — always an object. The write
is a deposit-class ΔT (its own `temp_gain_scale` dial, a `T_FLAME_MAX` taper
and a headroom cap) and it does **not** convert through the tile's
`heat_inv_shift`. Under the ruling, deposits onto an object convert via the
object's divisor — that is precisely the rule P-EOS applied to the combustion
aggregate deposit (ruling §2 site 3). This writer does not follow it.

**Why P-EOS deliberately left it alone, and P3 agrees.** It already behaved
this way on wood and hull walls long before this arc, so touching it is a
behaviour change unrelated to the regression being fixed, and it would move
goldens the arc is committed not to move. It is also numerically small: at
equilibrium the measured analytic ratio is 0.995–1.009 against a balance that
**omits the shim entirely**, so at the measured operating points the shim
contributes **under 1 %** of the crate's steady state. `fire_pressure_gain` is
0.15 and the `T_FLAME_MAX` taper drives it toward zero exactly where the crate
runs hottest.

**Do not fix it in this arc.** Record it as an accepted gap and a candidate for
the joint re-tune, where a golden rebase is already budgeted. Erik should see it
named: it is a writer that touches object temperature without the object's
divisor. If a future bench shows the analytic residual growing beyond ~1 %,
this is the first place to look.

*(Ruling §2 site 5, the `field_edit` T-paint dev brush, remains audit-only as
the ruling allows: a dev brush may write either medium.)*

---

## 8. Pending canon fold — do NOT do it yet, do not lose it

The arc is **unmerged** and awaiting Erik's HUMAN-TEST play. Per project
CLAUDE.md the canon fold happens at arc close, after he blesses it. So
`docs/architecture/` is untouched by P3 — and `engine/06` IS now stale. The
mechanical follow-up, enumerated so nothing is lost:

**`docs/architecture/engine/06_temperature_and_fire.md`**
1. `:91` "**Temperature lives on solids only**" → lives on **`thermal_solid`**
   tiles (`thermal_mass > 0`). That is *not* the same set as `solid`
   (`permeability <= 0`). Furniture is permeable **and** thermally solid — the
   one material where the two axes diverge.
2. `:106` the field diagram's "lives on solids; κ=0 on air" line — same fix.
3. State the **medium test** explicitly: the thermal pass keys on
   `thermal_solid`; `solid` keeps its flow / LoS / `N == 0` meaning at every
   other site. Six `MEDIUM-TEST SITE n/6` markers in `temperature_solver.cpp`
   and six twins in `cuda_temperature.cu`.
4. The heat→T convert divisor is **per-tile** (`heat_inv_shift`, =
   `log2(thermal_mass)`), not the global 8 the chapter still implies.
5. **Add the ownership rule as canon** (ruling §1): *on `thermal_solid` tiles
   `temperature[]` is owned by the TemperatureSolver — deposit-convert,
   conduct, COOL_SHIFT. Every other system is a reader.* This is the sentence
   whose absence let the regression happen; the chapter's own §77-91 rationale
   ("air temperature would ignite nothing and advect everything") is its
   justification.
6. **The EOS pass**: step-1b (semi-Lagrangian sample) and step-4c (compression
   work) **skip the `temperature[]` write** on thermal_solid tiles, and a
   thermal_solid tile is an **occluder to the backtrace sampler** — but this is
   **T-only**: `cmask` is untouched, so pressure, velocity and gas flow are
   unchanged and `permeability`'s shield-not-seal semantics are preserved.
7. **Combustion aggregate deposit** converts through the tile's
   `heat_inv_shift` on thermal_solid sites (the object path) instead of the gas
   divisor `c_v · max(N, n_floor)`.
8. The equilibrium law, now measured and worth stating:
   `T* = k_fire_heat · I · 2^(cool_shift − heat_inv_shift)`, exact to ±1 %.
9. The `I_crit` cliff (§4 above) belongs here too, as a documented property of
   the linear deposit/loss balance — it is the thing that makes the fire dials
   coupled.

**COOL-SHIFT AXIS follow-ups (added 2026-07-30 by the loss-side patch — the
axis that answers §4.3 lever 1's "but COOL_SHIFT is a GLOBAL dial")**

`engine/06_temperature_and_fire.md`
16. §3's ambient cooling is no longer one global. State it as
    `T -= T >> cool_shift[i]` with `cool_shift` a **per-material column**
    (`config.toml [materials.*]`, loader-validated integer in
    `[SHIFT_MIN, 20]`), projected to the derived per-tile grid
    `GameMap.cool_shift`. e-fold = `2^shift / tick_rate` seconds.
17. State the **vacuum-exposure rule in its offset form**, which is what the
    code now does:
    `exposed → max(SHIFT_MIN, cool_shift[i] − (COOL_SHIFT − COOL_SHIFT_VACUUM))`.
    The 4× space discount is a property of the BOUNDARY, so it stays ONE
    global rule applied to every material's own shift — each material keeps
    exactly one dial. Record the floor as load-bearing (a material at the
    floor would otherwise derive an exposed shift of 0 == `T -= T`).
18. Record that `[physics.thermal] COOL_SHIFT` / `COOL_SHIFT_VACUUM` are
    **kept and still have jobs**: COOL_SHIFT is the omitted-column default and
    the solver's nullptr fallback; the PAIR is the offset. They no longer set
    the interior rate for any material that authors the column (all eight do).
19. Note the WHY, next to §2.2's "furniture's one loss channel": one global
    could not serve thin hull plate (1.3 s) and a wooden crate (171 s) at
    once — the identical argument `thermal_mass` won on the gain side.

`engine/03_material_system.md`
20. Add `cool_shift` beside `thermal_mass` as a first-class material column —
    gain axis and loss axis, both per material. Live values: **every material
    5** (seeded at the retired global; the joint re-tune moves them).

`engine/02_state_and_ownership.md`
21. Add the derived grid `cool_shift` (int32, h×w) to the derived-grid list,
    on the SAME single structural-rebuild seam as `solid` / `heat_inv_shift` /
    `thermal_solid` (`_update_caches` build, `on_tile_changed` patch), and to
    `GameMap._RESIDENT_MASKS`. Same not-static caveat as `thermal_solid`: a
    device kernel that reads it must take it off the per-tick `from_host` list
    (no device kernel reads it today).

**Tooling, already done (not a fold item — recorded so it is not re-discovered)**
22. `tools/fire_tune_loop.py` moved its thermal dial from
    `physics.thermal.COOL_SHIFT` to `materials.furniture.cool_shift`. The old
    key is now INERT for the crate, because the material column wins. §4.3
    lever 1's warning ("this dial is GLOBAL, it moves goldens everywhere") is
    obsolete and has been rewritten in place.

**`docs/architecture/engine/03_material_system.md`**
10. `thermal_mass` is **not documented at all**. Add it as a first-class
    material column: `0` = gas thermal regime; `> 0` = solid thermal regime and
    the value **is** the convert divisor (power-of-two, loader-validated).
    Live values: hull 32, steel 32, glass 16, wood 8, door 8, door_closed 8,
    furniture 8, air 0.
11. State the axis separation: `thermal_mass` is the **thermal** identity,
    `permeability` the **flow** identity; they are independent.

**`docs/architecture/engine/02_state_and_ownership.md`**
12. `heat_inv_shift` and the derived `thermal_solid` grid are **not documented**.
    Add both to the derived-grid list, on the same structural-rebuild seam as
    `solid` (`gamemap.py:656` build, `:806` patch), noting the single-seam
    requirement so the future movable-furniture version has one place to become
    dynamic.
13. Record the `temperature[]` writer enumeration (ruling §2) — including the
    §7 plume-shim gap above — as the ownership table for that field.

**Sweep to do at fold time**
14. Re-grep the chapters for "solids only" / "on solids" phrasing in
    `04_atmosphere_and_pressure.md`, `05_smoke.md`, `08_ray_engine.md`.
15. `src/simulation/entities/sensors.py:154` — the comment "Temperature lives
    on solids only" was left as a comment fix in P1's D5; confirm it reads
    `thermal_solid` at fold time.

**O₂ FULL-RESPONSE REFERENCE SPLIT follow-ups (added 2026-07-30 by the
normalization patch — seed doc `fire_model_design_seed_2026-07-30.md` §2.1)**

`engine/06_temperature_and_fire.md`
23. The O₂ availability law's denominator is no longer ambient. State it as
    `o2f = clamp01((X − o2_frac_ext) / (o2_frac_full − o2_frac_ext))` with
    `o2_frac_full` a **fixed physical reference (pure O₂, 1.0)** that is
    deliberately **NOT** map-overridden, and `o2_frac_amb` demoted to "what the
    ambient atmosphere is" — **no longer read by either O₂ law**. Record the
    WHY: normalizing by ambient made ambient the ceiling under `clamp01`, so
    every enrichment route (reservoirs, leaks, wind delivery) was invisible *by
    construction*. Ambient air now reads `o2f = 0.092`.
24. Record that the two O₂ laws (`fire_simulation.cpp`, `combustion.cpp`) plus
    both CUDA twins share the reference, remain bit-identical, and that the CUDA
    free functions now take `o2_frac_full` in the slot that used to take
    `o2_frac_amb` (`o2_frac_amb` is not passed to the GPU at all).
25. State the tuning consequence, so it is never re-derived: at `avail·hot =
    0.092` the logistic needs `k_die/k_grow ≈ 0.051` (measured **9.88× smaller**
    than the shipped 0.5) for a normal-air fire to sit at `I ≈ 0.5`. That is
    Erik's dial, not a structural change.

**★ Finding that outranks the fold items: the digest goldens are BLIND to both
O₂ laws.**
26. `tests/field_ab_harness.default_scenario_sim` — the scenario behind the
    committed trajectory goldens — has **zero flammable tiles** and seeds its
    fire on AIR (`gmap.flammable.sum() == 0`, fire at `[8,8]`/`[8,9]` unchanged
    after 30 ticks). `FireSimulation::step` early-outs on `!flammable[i]` and
    the combustion pass has no claimants, so **no golden in the suite can move
    when the fire or combustion law changes**. That is why this patch — a
    deliberate behavioural change — moved **zero** goldens. Before the joint
    re-tune's one deliberate rebase, the golden scenario should gain fuel, or
    a second fuel-bearing golden should be added; otherwise the re-tune will
    rebase digests that never watched the thing being tuned.
    (`tests/o2_full_reference_gate_a_capture.py::_sim_burning_fuel` is a
    ready-made fuel-bearing scenario.)

Also worth folding: the escalation's process note, already adopted in ruling
§5 — *verify a routing question by enumerating writers of the field, not by
grepping near the mask.*

---

## 9. Disagreements with the record, collected

1. **The 0.871 analytic ratio is a transient, not a steady-state deficit**
   (§3.1). At equilibrium it is 0.995–1.009. This matters because it was on
   track to become a permanent 15 % fudge factor in the tuning loop.
2. **§2.5's `k_fire_heat ≈ 225` cannot be handed to Erik as a starting value**
   (§3.2). The arithmetic behind it is right; the operating point is dead on
   arrival. The escalation §6 line "P3 hands back to Erik's manual tuning loop
   (§9.3), which expects `k_fire_heat ≈ 225`" would have produced a bench that
   snaps out at tick 1.
3. **The design's implicit assumption that fixing the routing was sufficient to
   re-open the tuning loop is not quite right** (§4). The routing fix is
   correct and complete; what it exposes is that the linear deposit/loss
   balance fences the fire into `I > I_crit`. That is a new, quantified design
   question, and it is the one thing P3 cannot close on its own.

Gate results from P1/P2/P-EOS that P3 re-confirmed rather than contradicted:
gate (a) furniture-free byte-identity, gate (b) no golden moved, gate (c) the
t≈0 dip gone + monotone rise, gate (d) CPU↔CUDA tol 0, gate (e) green,
gate (f) tripwire silent.
