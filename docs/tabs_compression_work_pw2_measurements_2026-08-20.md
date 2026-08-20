# T_abs compression work — P-W2 measurements (2026-08-20)

**Arc:** `tabs-compression-work`. **Patch:** P-W2 (design §6 row 4; contract
`docs/tabs_compression_work_design_2026-08-20.md`, evidence
`docs/tabs_compression_work_manifest_pw1b_2026-08-20.md`,
`docs/tabs_compression_work_baseline_2026-08-20.md`). This doc is the AFTER
half of every P-W0 BEFORE row, plus the two new instrument runs the design
asked for (quiet-room long horizon, venting-bench mach-census). Machine:
`erik_lenovo`. Both pyds current for this tree — not rebuilt (no `cpp/src`
touched this patch).

Write surface used: `tools/tabs_pw2_venting_capture.py` (new),
`tools/quiet_room_drift.py` (extended: `mean_t_rel` series/summary field, the
R-3 mint-guard quantity that was test-local at P-W1b because `tools/` was off
that patch's edit surface — folded back into the tool proper now that it's
back on-surface). No `cpp/`, no sim-path changes.

## 1. Storm-ledger battery (`tools/storm_ledger.py --ticks 4800 --damp 0.005 --pf1b`)

| field | BEFORE (P-W0 §5) | AFTER |
|---|---|---|
| `eos.u_clamp_hits` / `u_max_hits` / `work_clamp_hits` / `energy_floor_hits` / `t_max_phys_hits` | all 0 | all 0 |
| `eos.eth_transport_delta` (run total) | −7,671,929 | −610,660,422 |
| `eos.eth_compression_delta` (run total) | −16,339,797 | **−10,021,816,567** |
| `eos.e_ts_residual` | 558,051 | 601,855,175 |
| `eos.e_wipe_sum` / `e_floor_sum` | 0 / 0 | 0 / 0 |
| `eos.n_active_flux` / `n_bulk_active_sum` | 288 / 18,860,554 | 288 / 18,858,386 |
| `temp.e_vac_wipe_sum` / `e_ring_pin_sum` | 0 / 0 | 0 / 0 |
| amplifier max gain | 1.1x (4788 ticks) | 1.2x (4789 ticks) |

No rail ever engages in this "clean" (no window dial) battery, before or
after — expected, this is the calm baseline row, not the window row. The
headline change is **`eth_compression_delta`'s magnitude jumping ~613x**
(−16.3M → −10.02B): design §3's "ambient cells contribute for the first
time" — under the old law every T_rel≈0 cell contributed exactly `k·0 = 0`
to compression work; under the new law they contribute the `+w·290` term.
`eth_transport_delta` also grew substantially (−7.67M → −610M); not
independently investigated (out of scope for a measurement patch) but
consistent with the same root cause — SL transport now interacts with a
temperature field that evolves everywhere, not only near the fire seed.

## 2. Cold-rail window (`tools/bench_two_room.py::run_bench`, WINDOW dials)

Dials: `dict(storm_probe.PF1B, k_wind_strip="0.5")`, `damp=0.005`, 4800 ticks
— the exact P-W0 §6 recipe. **AFTER-ONLY characterization per design §0 A-1**
(the −288.65 spiral is dormant on HEAD; there is no BEFORE spiral to compare
warming against — this row instead answers "does sub-ambient gas now exist,
and does compression warm it honestly").

| field | BEFORE (P-W0 §6, old law) | AFTER (T_abs law) |
|---|---|---|
| probe cell (y,x) | (7, 6) | (7, 6) |
| `T_probe` min / final (game-deg) | 0.0 (tick 1) / 0.1516 | **−6.1894** (tick 2) / **+60.4020** |
| `p_probe` min / final (atm) | 0.9957 / 0.9965 | 0.9989 / 1.0339 |
| `ke_peak` / `ke_final` | 19.16 / 1.05 | 17.76 / 1.64 |
| `umax_peak` / `umax_final` (m/s) | 1.85 / 0.297 | 2.22 / 0.564 |
| all `run_bench` rail counters | all 0 | all 0 |
| domain-wide `t_min_gas` (`storm_ledger.run_ledger`, same dials) | 0.0 the entire run | **min −44.8385 @ tick 43**, final −27.8062 |
| `temp.e_vac_wipe_sum` / `e_ring_pin_sum` (domain) | not captured | 0 / 0 |

**Answers to the design's questions:** (a) *does sub-ambient gas now
appear?* Yes — domain min T reaches −44.84 game-deg (was exactly 0.0 for
every tick pre-arc); the probe cell itself dips to −6.19 briefly. (b) *does
compression warm it?* Yes at the probe — it recovers from −6.19 (tick 2) to
+60.40 by tick 4800, a clean net warming with no sign of the old inverted
spiral. Domain-wide the picture is gentler: min T recovers from −44.84 to
−27.81 by run end — still net negative at the domain scale (some pocket
hasn't fully conducted/advected back to ambient in 4800 ticks) but *bounded*,
nowhere near T_MIN (−289), and not diverging. (c) *rails:* none engage —
this scenario is far too mild to stress any counted rail; it is not the
2026-08-14 audit's extreme spiral (already established dormant, P-W0
finding A-1), just the honest, small-amplitude thermalization the new law
adds everywhere.

## 3. Hot-rail HOT scenario — cited, not re-run

Fresh numbers already measured at P-W1b (`docs/tabs_compression_work_manifest_pw1b_2026-08-20.md`
§3), no missing field:

| field | value |
|---|---|
| `t_max_phys_hits` | 4 (gate ≤ 8) |
| ticks with any cell T > 15000 | 7 (gate ≤ 14) |
| `peak_T` | 15975.98 game-deg |
| mean of last 1000 ticks' peak T (equilibrium) | 5341.90 game-deg (old-law all-run peak was 5553.30) |

## 4. Ambient gate-2 AFTER (re-run for the missing counters)

The manifest cited `work_clamp_hits`/`u_clamp_hits`/peak-T/`t_max_phys_hits`
(6014/4080/629.2/0); `u_max_hits`/`e_vac_wipe_sum`/`e_ring_pin_sum` were not
in that table, so this patch re-ran the test module's own scenario body
(reused, not transcribed — `tests/test_air_boundary.py::_ambient_gmap(40,40)`,
80 ticks, same recipe) to fill them in:

| field | BEFORE (P-W0 §8, old law) | AFTER (T_abs law) |
|---|---|---|
| `work_clamp_hits` | 4345 | 6014 |
| `u_clamp_hits` | 2816 | 4080 |
| `u_max_hits` | 0 | 0 |
| `energy_floor_hits` | 0 | 0 |
| `t_max_phys_hits` | **0** (real green both before/after) | **0** |
| peak interior T (game-deg) | 24.46 (mean @ t80: 3.76) | 629.23 (mean @ t80: 40.27) |
| `e_vac_wipe_sum` / `e_ring_pin_sum` | 0 / 0 | 0 / 0 |

Matches the manifest's cited numbers exactly (6014/4080/629.2/0); no silent
weakening of the rail the design worried most about.

## 5. Counter exposure table (u_max_hits / u_clamp_hits / e_vac_wipe_sum / e_ring_pin_sum), AFTER

| counter | storm-ledger (§1) | window row (§2, domain) | gate-2 (§4) | venting bench (§6) |
|---|---|---|---|---|
| `u_clamp_hits` | 0 | 0 (run_bench) | 4080 | 327 (1500 ticks) |
| `u_max_hits` | 0 | not captured | 0 | 0 |
| `e_vac_wipe_sum` | 0 | 0 | 0 | 0 |
| `e_ring_pin_sum` | 0 | 0 | 0 | 0 |

**Vac/ring creation-channel pricing (design §3: bounded ≤ 290·N_vented per
tick).** Measured actual across **every** P-W2 scenario, including a
1500-tick sustained-venting run through a permanent breach into vacuum with
deep sub-ambient excursions (§6: min T −283 game-deg, up to 2136 open cells
sub-ambient at once): **`e_vac_wipe_sum` = `e_ring_pin_sum` = 0, always.**
This is not evidence the channel is safe at its bound — it is evidence the
channel was never *exercised*. Reading `cpp/src/temperature_solver.cpp:117-137`,
the wipe/pin only prices a NONZERO temperature being found on an
already-`is_vacuum`/`is_ambient`-flagged cell; every scenario measured here
either carves its breach once at level-load (static from tick 0, so the
vacuum cells' T is 0 for the whole run — nothing to wipe) or never breaches
at all. **The channel needs a mid-run solid→vacuum TRANSITION (a live
`destroy_wall` that breaks a wall carrying nonzero T) to fire at all** — none
of the P-W0/P-W2 candidate scenarios have one. Flagged as an open
measurement gap, not resolved here: the ≤290·N_vented/tick bound is a
design-time argument, not yet an observed one.

## 6. Mach census on a venting-bench recorder capture (design §3 B-F7 / D-1)

**Scenario choice (documented):** `tools/tabs_pw2_venting_capture.py`
transcribes `tests/cuda_kick_check.py::part2_trajectory`'s "blast + venting"
geometry verbatim — 48×48, a hull ring, a 4-tile breach carved through the
east wall into an outer vacuum band, a 5000 K hot core + O2 overpressure
pocket, plus a near-ceiling 15500 K pocket (so T_MAX_PHYS/U_MAX are
reachable). Considered and rejected: `cuda_s8a_check`'s ring-adjacent breach
world (built for a short residency check, not a sustained multi-hundred-tick
vent); a fresh scripted `destroy_wall` on a sealed room (would need new
geometry authored from scratch, and — see §5 — a mid-run wall break is
exactly the untested case worth a FUTURE dedicated capture, not blended into
this one). The kick-check geometry's breach is permanent from tick 0 and
never re-seals, so it vents the longest of the three candidates.

Run twice (600 and 1500 ticks) via `PhysicsRecorder` (`DEFAULT_FIELDS`, so
`--mach-census` gets `gas_o2`+`inert_n2` for measured N, plus `wind_x`/
`wind_y` for the Mach ratio). The system reaches steady state by ~tick
300–600 and both runs agree to 4 significant figures on every summary
statistic — the 1500-tick numbers are cited below as canonical.

| field | value |
|---|---|
| sub-ambient (N < 1) open cells | peak 2136 (snap 73); 3,201,050 total cell-snapshots over the run |
| min T over run | **−283.19 game-deg** @ snap 170 (t_abs = 6.81 K vs the T_MIN floor's 1 K — within ~7x of the floor, does not reach it) |
| min P over open cells | **−0.00026 atm** @ snap 291 (quantization-scale negative — NOT a deep collapse) |
| max\|grad P\| proxy | 1.00 atm @ snap 0 (the initial seed's own discontinuity, not a rail event; settles below 0.05 atm by snap ~60) |
| \|u\|/c_own (Mach vs OWN local sound speed) | p50 0.0001, p90 0.083, p99 0.326, p99.9 1.02, **max 2.99** |
| fraction of open cell-snapshots with u/c_own > 1.0 / > 1.5 | 0.108% / 0.018% |
| `u_clamp_hits` / `u_max_hits` (1500 ticks) | 327 / 0 |
| `work_clamp_hits` / `energy_floor_hits` / `t_max_phys_hits` | 1843 / 0 / 0 |
| `e_vac_wipe_sum` / `e_ring_pin_sum` | 0 / 0 (see §5) |

**B-F7 flash-route verdict: NOT OBSERVED in this scenario.** Cells approach
close to the T_MIN floor (t_abs down to 6.81 K, i.e. within an order of
magnitude of the 1 K floor) but pressure never collapses — min P stays
within a few×10⁻⁴ atm of zero, nowhere near the deep-negative signature
(−0.98 atm) `tools/analyze_blowup_dump.py`'s own header cites for the
velocity-clamp arc's original mass/momentum flash class. Supersonic cells do
occur (max Mach 2.99, ~0.1% of cell-snapshots above Mach 1) but stay
localized and counted — `u_clamp_hits` (327) is the bounded channel doing
exactly its designed job, not a sign of an uncounted pile-up. **This is a
measured "no" for THIS geometry/duration, not a general clearance** — the
hazard is named and instrumented per design §3, and a cell 7x above the
floor in t_abs is one sustained expansion tick away from being much closer
(§3's `ln(290)/ln(1.5) ≈ 14 rail ticks` estimate) — D-6 (raise T_MIN) stays
the cheap lever if a different scenario finds the flash this one didn't.

## 7. Quiet-room long horizon (the open finding, `tools/quiet_room_drift.py --ticks 10000`)

The P-W1b manifest's open finding: the design cited "mean T_rel ≈ +0.004
game-deg at tick 2000, near-canceling"; fresh measurement instead found a
monotonic +4.646 at tick 2000, and re-keyed the mint-guard gate to
`|mean T_rel| ≤ 10` at that horizon (~2x headroom). **This run asks whether
that drift saturates, grows linearly, or decays past 2000 ticks.**

Mean-T_rel trajectory (signed, spatial mean over open interior cells):

| tick | mean T_rel (game-deg) | max\|T_rel\| envelope (game-deg) |
|---|---|---|
| 1 | −0.017 | 6.80 |
| 100 | 2.799 | 13.48 |
| 500 | 3.364 | 19.08 |
| 1000 | 3.805 | 16.46 |
| 1500 | 4.233 | 16.49 |
| **2000** | **4.646** | **16.27** (matches the P-W1b manifest's 4.646 exactly) |
| 2500 | 5.063 | 17.13 |
| 3000 | 5.476 | 17.23 |
| 4000 | 6.238 | 17.79 |
| 5000 | 7.007 | 17.99 |
| 6000 | 7.710 | 18.61 |
| 7000 | 8.429 | 20.48 |
| 8000 | 9.096 | 20.06 |
| 9000 | 9.750 | 21.76 |
| **9999** | **10.388** | **22.42** |

Local slope (Δmean_T_rel / Δtick, game-deg/tick):

| window | slope |
|---|---|
| 1000–2000 | 0.000841 |
| 2000–5000 | 0.000787 |
| 5000–10000 | 0.000676 |
| 8000–10000 | 0.000646 |

**VERDICT: the drift is close to LINEAR, decelerating only mildly (~23%
slope reduction from the first measured window to the last) — it does
NOT saturate (the slope stays clearly positive throughout, showing no sign
of flattening) and does NOT decay (never approaches zero or reverses sign)
within 10000 ticks (~417 s / ~7 real minutes at 24 tps).** The envelope
(max|T_rel|) also keeps slowly climbing (16.27 → 22.42) rather than settling,
though far more gently than the mean.

**This crosses the existing gate's own bound.** At tick ~10000, mean T_rel
(10.39) already EXCEEDS the P-W1b mint-guard's `≤ 10.0` game-deg budget —
the same recipe, run 5x longer than the gate's own horizon, would fail the
gate that was sized from the 2000-tick measurement with "2x headroom." The
committed gate (`tests/test_quiet_room_drift_smoke.py::
test_quiet_room_drift_t_abs_split_gate`) only runs 2000 ticks and stays
green — this is not a red anywhere in the suite — but the headroom claim
does not hold at longer, still gameplay-plausible horizons (a sealed room
left quiet for ~7 minutes). **Flagged for Erik's brief verbatim** as
load-bearing for any decision about the mint-guard's margin or a general
drift-tolerance ruling.

## 8. Render instruments (D-7) — state

Both instruments are wired into the main game render path, render-only, on
existing toggles — no new keybinding introduced:

- **Hover readout** (`renderer/hover_readout.py`, previously built+tested
  but unwired): folded into `GameRenderer.draw_debug_hud` (F6 toggle,
  `docs/TODO.md:814-819`'s ask) — the panel now shows T in game-deg +
  pseudo-Kelvin + fire/O2/trace-gas densities under the existing
  tile/material line, using the SAME `BlackbodyRamp._kelvin_from_tgame`
  conversion the heat overlay uses (so the readout and the emissive glow
  agree by construction). Verified: `tests/test_hover_readout.py` (existing,
  unchanged, still green) plus `tests/test_renderer_smoke.py` (instantiates
  the real `GameRenderer` with a live GL context — confirms the wiring
  imports and constructs cleanly).
- **Cold tier** (`renderer/cold_overlay.py`, new): a diverging blue ramp for
  `T_rel < 0`, reusing `PressureOverlay`'s stop-table linear-interpolation
  idiom, alpha-blended (`BLEND_ALPHA_PREMULTIPLY`) and drawn immediately
  BEFORE `HeatFieldOverlay.draw()` on the same `show_temperature` toggle (T
  key). Placeholder constants, explicitly provisional per D-7 — Erik judges
  the look at P-W3. Verified visually-by-numbers per the design's explicit
  instruction (no screenshot needed): `tests/test_cold_overlay.py` asserts
  the mapping function is correct in isolation (ambient/warm → fully
  transparent; cold → nonzero, monotonically deepening alpha, clamped at the
  deepest stop) AND that a real scenario (the quiet-room recipe, 5 ticks —
  cheap, deterministic) produces sub-ambient interior cells that the cold
  pass packs to nonzero alpha, not just a synthetic array.

**Kelvin-frame note (for Erik's brief, D-7):** the hover readout shows
game-deg — the honest number sub-ambient. The canonical render Kelvin map
(`K = 293 + 3·T_game`) is misleading below ambient: −96.67 game-deg (the
work-clamp figure) reads as "193 K" in the EOS's own T_abs frame but as "3 K"
through that render map. Sub-ambient, trust the game-deg number, not Kelvin.
