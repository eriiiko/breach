# P-F4b — THE SIZING PACKAGE: supply-vs-radius, smother curves, wind level (2026-08-02)

**Status: EVIDENCE, not a ruling.** This document delivers the measurements
`docs/fire_realism_design_2026-08-01.md` v5.2's execution order calls
"P-F4b (supply-vs-radius sweep + sealed-room smother check + FORCED-WIND
level with a literature slot)" and names as the prerequisite for "ERIK:
radius + fire-power sizing call". It makes **no recommendation** — R=2 vs
R=3, the plateau target, and the wind-level design fork are Erik's calls at
the sizing session. Every number below is measured on this branch
(`pf4b-sizing-package`, base `4642508`); every CSV cited lives in
`_fire_tuning_artifacts/` (untracked by design — regenerate with the tool
each file names).

No sim code changed. `[physics.combustion] draw_r`/`max_claimants` are
config OVERRIDES applied and restored per run by the sweep tools (the
same `apply_overrides`/`restore_overrides` idiom every bench in this family
already uses); config.toml itself is untouched (still ships `draw_r = 2`).

Tools added (all thin drivers over the existing P-F4a bench family — see
each file's own docstring for full methodology):

- `tools/fire_o2_supply_baseline.py` — REFACTORED (not rewritten): the pin-I
  measurement loop is now `measure_supply_on_level(level, cy, cx, ...)`,
  level-agnostic; `measure_supply(...)` (the P-F4a CLI entry point) is now a
  thin wrapper over it, byte-for-byte the same behavior as before this
  patch.
- `tools/fire_supply_radius_sweep.py` — task 1, the supply-vs-radius sweep.
- `tools/fire_smother_curve_sweep.py` — task 2, the sealed-room smother
  curves.
- `tools/fire_wind_level_probe.py` — task 3's feasibility evidence (the task
  STOPPED; see §3).

---

## 1. The supply-vs-radius curve

**Methodology**: the P-F4a/P-O2b pin-I diagnostic (`measure_supply_on_level`)
— intensity pinned at the design's blessed operating point (I = 0.192,
`tune_r5_lone_wd020.csv` post-P-R4) and `wall_hp` pinned to full, each tick,
so the run isolates diffusive/extended-draw O2 TRANSPORT from the fire's own
intensity dynamics and fuel depletion (see the module docstring for the full
rationale, including the honest caveat about the two delivery columns
below). Two numbers are reported per point:

- **"analytic radius-1 formula"** — a Python re-implementation of the
  UNCONTESTED radius-1 demand sum over the tile's own 4 open faces. It
  literally cannot see an extended draw, so at DRAW_R > 1 it under-reports
  (kept only as a cross-check against the pre-P-O2b baseline).
  Delivery converts counts/s → kW via the Huggett-anchored
  `J_PER_COUNT = 73.7` (1 O2 unit = 11.53 mol = 369 g O2 = 4.83 MJ; the
  constant `tools/fire_o2_supply_baseline.py` fixed post-P-F4a).
- **"TRUE draw"** — the number that matters: law-agnostic, recovered by
  re-running the real combustion pass (real dials, real backend, whatever
  `draw_r` is configured) on the settled per-tick state and reverting every
  plane it touches. This is what the fire actually consumes under 2b's
  extended draw. **All headline numbers below are this column.**

Swept: `draw_r` ∈ {1, 2, 3} × material ∈ {kindling, furniture} × 3
environments — the open still-air arena (`fire_timing_harness.build_level`,
planetside, sky-exchange ON), a sealed 12×12 SHIP room
(`fire_room_bench.build_room_level`, boundary="space", no sky, no vent), and
the same room with a 2-wide vent open **from tick 0**.

### 1.1 Delivery table (TRUE draw, quasi-steady, last 5 s of a 30 s pinned run)

| env | material | R=1 | R=2 (shipped) | R=3 (sweep ceiling) |
|---|---|---|---|---|
| open_arena | kindling | 5.59 kW (75.8 counts/s) | 12.62 kW (171.2 counts/s) | 21.59 kW (293.0 counts/s) |
| open_arena | furniture | 5.19 kW (70.4 counts/s) | 12.85 kW (174.4 counts/s) | 21.37 kW (290.0 counts/s) |
| sealed_room (12×12) | kindling | 4.42 kW (60.0 counts/s) | 11.39 kW (154.6 counts/s) | 19.74 kW (267.8 counts/s) |
| sealed_room (12×12) | furniture | 4.13 kW (56.0 counts/s) | 11.19 kW (151.8 counts/s) | 19.52 kW (264.8 counts/s) |
| vented_room (12×12, vent open @ t=0) | kindling | 0.00 kW (all R) | 0.00 kW | 0.00 kW |
| vented_room (12×12, vent open @ t=0) | furniture | 0.00 kW (all R) | 0.00 kW | 0.00 kW |

CSVs (one per point, per-tick trace):
`_fire_tuning_artifacts/supply_radius_sweep_{open_arena,sealed_room,vented_room}_{kindling,furniture}_R{1,2,3}.csv`
(18 files); ring-by-ring X profiles and the full per-point breakdown:
`_fire_tuning_artifacts/supply_radius_sweep_summary.txt`.

**The vented_room row is a real, measured zero, not a gap.** Opening ANY
vent to true vacuum (`boundary="space"`, the SHIP-style hull) from tick 0
drains a 12×12 room to near-total vacuum (`room N_total` → ≈0.007–0.01 of
ambient) within about one tick — an acoustic-scale blowdown the Kwatra
pressure solve resolves almost instantly, not a slow leak (§3 measures this
directly and quantifies it). By the time the pin-I run's quasi-steady
window is sampled, the room has been near-vacuum the whole time and the
combustion pass's `o2_thresh_burn` ABSOLUTE-count epsilon floor skips every
donor cell — hence exactly 0.00 counts/s at every radius. (The "analytic
radius-1 formula" column in the CSVs still reads a deceptively normal
0.08–8 kW there: it is built from X, the density-invariant FRACTION, which
can look near-ambient even as the absolute gas content the real law reads
has collapsed to nothing — a live illustration of why the TRUE column
exists.) **Reading**: "vent open" is not a moderating, partial-supply
condition for a ship room — it is a near-total and immediate supply cutoff,
independent of draw radius.

### 1.2 Gate (b) — R2/R1 ratio reproduction

The design doc's own cited reproduction target: "Verify the shipped R=2
numbers reproduce (≈2.26×/2.48× over R=1 in the open arena)."

| material | measured R2/R1 (open arena) | expected | error |
|---|---|---|---|
| kindling | 2.2586 | 2.26 | −0.06% |
| furniture | 2.4773 | 2.48 | −0.11% |

**GATE (b): PASS** (well inside the ±2% band). This is also the falsifier
v5.2 names for F-O2b — "quasi-steady delivery scales with the draw
BOUNDARY (~2–3× at R=2–3), not the area" — confirmed: R3/R1 lands at
3.87×/4.12× (kindling/furniture), consistent with boundary-count scaling
(slot_count 4/12/24 at R=1/2/3 is a 3×/6× AREA ratio; the measured 2.3–4.1×
DELIVERY ratio is well below that, i.e. genuinely boundary-shaped, not
area-shaped).

### 1.3 Delivery vs the design's required-HRR books (§0)

Design doc §0's own books, at the blessed plateau (peak I 0.192, steady_T
385–544 game):

| channel | power |
|---|---|
| chemical (burn_rate 0.02) | 4.85–6.7 kW |
| H_bed to the bed | 2.6–3.6 kW |
| honest sky sink (ε=0.5, T 385–544 game) | 30–45 kW |
| h-anchored convection (F3) at the plateau | 15–29 kW |
| **required HRR for the blessed plateau** | **60–97 kW** |
| real crate fire (Babrauskas) | 100–250 kW |

Measured TRUE delivery at the shipped R=2: **≈11–13 kW** (sealed room /
open arena). At the sweep's upper point R=3: **≈19–22 kW**. Against the
book's low end (60 kW):

- R=2 delivers **18–21%** of the low-end requirement (a **5–5.5×
  shortfall**).
- R=3 delivers **32–36%** of the low-end requirement (a **2.8–3.2×
  shortfall**) — better, but still short of even the cheapest end of the
  required range, before CUDA cost (§4) is weighed at all.

**Read plainly**: at today's `burn_rate = 0.02` and the "blessed" 385–544 K
plateau target, the 2b extended draw ALONE — at either shipped or
sweep-ceiling radius — does not supply the required HRR. Closing that gap
(if the 385–544 K plateau is kept) needs either a higher `burn_rate`, a
lower plateau target (the R-SCALED path the v4/plain-edition rulings
already flagged as newly legitimate under REQ-11's scaling ruling), or an
additional supply channel. This arithmetic is evidence for the sizing call,
not a verdict on it.

---

## 2. Sealed-room smother curves

**Task order (locked)**: sealed 12×12 (and one ~20×20), UNPINNED natural
burns at DRAW_R=2, kindling + furniture, recording I/T/hp/room-total-O2
until death; report cause (knee vs O2), time, part-burn %, room O2 at death.

**Result — a genuine, measured finding, not a tooling defect**: every run
below dies in under one second, at peak I ≈ 0.09, cause **"knee (T-gate
limited)"**, with the room's O2 mole fraction essentially unchanged from
ambient (0.2100) at death. This is INDEPENDENT of room size (12×12 vs
20×20) and of material.

| material | room | peak I | death cause | death time | part-burn % | room O2 X at death |
|---|---|---|---|---|---|---|
| kindling | 12×12 | 0.093 | knee (T-gate limited) | 0.92 s | 0.22% | 0.2100 |
| furniture | 12×12 | 0.093 | knee (T-gate limited) | 0.79 s | 0.05% | 0.2100 |
| kindling | 20×20 | 0.093 | knee (T-gate limited) | 0.92 s | 0.22% | 0.2100 |
| furniture | 20×20 | 0.093 | knee (T-gate limited) | 0.79 s | 0.05% | 0.2100 |

CSVs: `_fire_tuning_artifacts/smother_curve_sealed_{12x12,20x20}_{kindling,furniture}.csv`;
full breakdown: `_fire_tuning_artifacts/smother_curve_summary.txt`.

**Why**: this is the SAME bootstrap-floor condition `simulation/materials.py`'s
own load-time check already warns about on every load of this branch —
`ignition_seed = 0.1` sits well below the 15%-margin `I_sustain` floor every
flammable material needs to bootstrap at the current `k_grow=4.0/k_die=2.0`
tempo and the P-R4 `H_bed` fuel-bed gain chain (kindling: need ≥0.2097, has
0.1; furniture: need ≥3.3554, has 0.1). It reproduces even on
`fire_timing_harness.py`'s own flagship still-air run (STALLS, peak
I=0.092, snap-out 0.8 s) and `fire_room_bench.py`'s own `--demo` (peak_I
0.093, snap-out 0.8–0.9 s) — i.e. this is not something the sweep
environment introduced.

**Consequence for the sizing package**: the O2-DRIVEN smother curve T2's
pre-measurement charter actually wants — "does the room's O2 inventory, not
the growth tempo, decide when this fire dies" — **cannot be measured at
today's tune**. Every death observed is a bootstrap-floor knee, seconds
before O2 supply could ever become the limiting factor; the room's O2 never
moves. Per gate (d) this is reported as such, not extrapolated or patched
around (no law change is in scope for this patch). Task 1's pin-I curves
sidestep this by construction (pinning I every tick is exactly why that
methodology exists) and remain the meaningful supply signal in this
package; task 2's natural-burn smother behavior is a real, additional
finding for the ignition-seed/tempo re-tune that has to land before a
natural-burn smother curve is measurable — flagged here for whoever owns
that pass, not fixed by this one (no law changes, per this patch's
charter).

---

## 3. The pressure-driven wind level — STOPPED, feasibility evidence delivered

**Task order's own escape clause, invoked**: "IF a sustained honest flow
cannot be built from existing machinery, STOP that task only, document why,
and deliver tasks 1–2 + the write-up... the wind level then becomes a
design item — do not hack the WindForcer back in." That is what happened.

### 3.1 Why the two named candidates are not constructible

1. *"A corridor with one end open to the planetside ambient ring and the
   other vented to SPACE (vacuum draft)"* — **not buildable**. `GameMap`'s
   `boundary` is a single LEVEL-GLOBAL flag (`src/simulation/gamemap.py`
   `__init__`): every SPACE(9)-coded tile on one level routes WHOLESALE to
   either `is_vacuum` (`boundary="space"`) or `is_ambient`
   (`boundary="ambient"`) — never a per-tile mix. There is no way to make
   one end of one level a P=0 sink and the other a P=P_amb source in the
   same map.
2. *"Two ambient regions if the engine supports differing pressures"* —
   **the engine does not**. `[ambient]` (`src/simulation/ambient.py`)
   carries exactly one `p_amb` per level, applied uniformly to the whole
   `is_ambient` ring.

### 3.2 What IS buildable — measured directly, and it is not sustained

The only assembly existing machinery supports is a sealed SHIP room
(`boundary="space"`) breached to vacuum mid-run via the existing
`GameMap.destroy_wall` path (`fire_room_bench`'s own "breach" mode,
unchanged). `tools/fire_wind_level_probe.py` builds exactly this — a 12×12
sealed room, a pin-I furniture fire established under closed hull for 15 s
(reaching a genuine quasi-steady 288 counts/s TRUE draw, confirming the
methodology is sound), then a real 2-wide vent breached at t=15 s — and
logs the SOLVER's OWN `wind_x`/`wind_y` at the fire's tile (no forced field
write anywhere in this script) plus the room's bulk O2/N2 inventory.

Measured (`_fire_tuning_artifacts/wind_level_probe_sealed_breach.csv`):

| | pre-breach (mean, t<15s) | +0.25s | +0.5s | +0.75s | +1.0s | t=25s (run end) |
|---|---|---|---|---|---|---|
| room N_total (frac. of ambient) | 1.0008 | 0.4212 | 0.1723 | 0.0530 | 0.0198 | 0.0073 |
| `\|wind_x\|` at the fire | 0.01–0.15 (natural plume noise) | 164.76 (blowdown spike) | 25.71 | 6.21 | 26.54 | noisy, 0–6 (residual sloshing in a near-empty room) |
| O2 drawn by the fire | 72–288 counts/s (pinned-I quasi-steady) | 0 | 0 | 0 | 0 | 0 |

The room drains to ~5% of its starting air content within one second of the
breach and to <1% within ~4 s; there is no multi-second plateau, let alone
one comparable to a fire's lifetime, for a "wind level" to be swept against.
This confirms directly what
`docs/architecture/engine/04_atmosphere_and_pressure.md`'s as-built
description already implies: breach venting is an acoustic-scale
TRANSIENT (an elliptic, whole-domain pressure solve — "the front passes,
the dome lingers," not a slow leak), and the ONLY continuously-refilled
reservoir the engine has (the `is_ambient` ring) cannot coexist with a
vacuum sink on the same level (§3.1).

### 3.3 Conclusion

**No sustained, honest, pressure-driven cross-flow can be built from
today's existing boundary machinery.** Per the task order and per F-BO's
own escape clause ("if the supply channel alone proves too weak, re-siting
the wind fan is a DESIGN question — bring it back, do not dial it"), the
wind level is now a **design item**, not a measurement this patch can
deliver. No WindForcer-style forced field write was used anywhere in this
investigation; every number in §3.2 is the solver's own field, read only.

---

## 4. R=2 vs R=3 — the trade, stated neutrally

No recommendation; the sizing call is Erik's. The two axes:

- **Delivery gain** (§1.1, TRUE draw, open arena): R=3 over R=2 is
  **1.71×** (kindling, 21.59 vs 12.62 kW) / **1.66×** (furniture, 21.37 vs
  12.85 kW) — i.e. roughly two-thirds more delivery, still 2.8–3.2× short
  of the design's own low-end required HRR (§1.3).
- **CUDA cost** (`cpp/src/cuda_combustion.cu`, measured with `ptxas -v` on
  sm_89, the shipping flags): pass_a's per-thread register count goes
  **70 → 168** at R=2 → R=3 (zero register spill at either radius — the
  cost is occupancy, not correctness). At 256-thread blocks that is
  **~50% → ~17% SM occupancy**. The file's own note: "R=3 ... is not the
  ship value, and if P-F4b's sweep ever wants R=3 in production THAT is
  when the plane-form restructure earns its keep" — i.e. the 17% figure is
  the cost of shipping R=3 AS-IS; a register-budget restructure (already
  scoped, not yet built) is the mitigation path if R=3 is chosen.

---

## 5. Gates

- **(a) suite failure-set == 27 names, unchanged**: baseline (this branch,
  before any P-F4b edit) and post-edit runs both show the SAME 27 failing
  tests (pre-existing, unrelated to this patch — see the commit body for
  the full diff). No sim code was added; the sweep tools are config
  overrides only.
- **(b) R2/R1 open-arena ratio reproduction**: PASS, §1.2. Measured R2/R1 =
  2.2586 (kindling), 2.4773 (furniture) against 2.26/2.48 expected —
  errors −0.06% / −0.11%, both inside ±2%.
- **(c) every CSV lands in `_fire_tuning_artifacts/`, cited by filename**:
  24 files (18 supply-radius-sweep + 4 smother-curve + 1 wind-probe + the
  2 summary .txt); every one is cited by name in §1–§3 above.
- **(d) any non-quasi-steady scenario reported as such**: the vented_room
  supply rows (§1.1) and the wind-level probe (§3.2) are both reported as
  transient/collapsed, not extrapolated into a steady number.

## 6. Deviations from the literal task order

- Task 2's natural-burn smother curves are all degenerate (bootstrap-floor
  knee deaths, <1s, room O2 untouched) — see §2's "Why" and "Consequence."
  Delivered as specified (the CSVs and cause/time/part-burn/O2-at-death
  columns are all there); the O2-DRIVEN behavior the task order actually
  wants is not observable until a separate ignition-seed/tempo re-tune
  lands. Not fixed here (no law changes, per this patch's charter).
- Task 3 STOPPED per its own escape clause; §3 is the required
  documentation of why, with direct measurement rather than only
  architectural argument.

---

**Appended 2026-08-14 (supersession note).** Any ×2 game-T→Kelvin map referenced
above is superseded by the unified canonical map in
`[physics.temperature_scale]` (`K = 293 + 3·T_game`; EOS pressure calibration
keeps a named, deliberate exception at `eos_t_amb_k = 290`). See
`docs/temperature_scale_unification_design_2026-08-13.md`.
