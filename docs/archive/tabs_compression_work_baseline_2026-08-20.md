# T_abs compression work — P-W0 baseline (2026-08-20)

**Arc:** `tabs-compression-work`. **Patch:** P-W0 (instruments + BEFORE
baselines — design §6 row 1; contract: `docs/tabs_compression_work_design_2026-08-20.md`,
evidence: `docs/tabs_compression_work_critiques_2026-08-20.md`, esp. C2/C11/C15).
This doc is the artifact every later patch's set-diff cites (C2). No sim-source
changed to produce it — `git diff --stat` at capture time touched only
`tools/analyze_blowup_dump.py` (new `--mach-census` mode, additive) plus two
new files, `tools/quiet_room_drift.py` and `tests/test_quiet_room_drift_smoke.py`.

Machine: `erik_lenovo` (RTX 1000 Ada / sm_89, MSVC 14.44, CUDA 12.9, miniconda
`data` env, py3.12.11). Git HEAD at capture: `e833192e2bb5ad184262cc0347f5aac77956da59`
(`docs(critique): T_abs compression-work critique round 1 capture`), branch
`tabs-compression-work`, working tree clean before this patch's own edits.
Capture window: 2026-08-20, ~04:00-06:10 UTC.

## 1. Build stamp

Both backends were rebuilt from a stale state before any measurement below —
the existing `.pyd`s predated this branch's HEAD by weeks (CUDA: 2026-07-24;
CPU: 2026-07-24), while `cpp/src` was last touched 2026-08-19 (the
velocity-clamp arc close). Running the baseline against a stale build would
have measured the WRONG engine, so both were rebuilt even though the P-W0
brief only named the CUDA script explicitly.

- **CUDA**: `cpp\build_cuda_lenovo.bat` (the Lenovo per-machine script, per
  `docs/lenovo_dev_setup.md`). `CONFIGURE` + `BUILD` both exit 0.
  `cpp/build_cuda/breach_physics.cp312-win_amd64.pyd` mtime: **2026-08-20
  05:48** (was 2026-07-24 09:07 before the rebuild).
- **CPU**: `cpp\build_cpu_data.bat` (the sibling per-machine CPU script —
  invoking `cmake --build` directly failed with `fatal error C1083: Cannot
  open include file: 'cstdint'` because it skips the script's `vcvars64.bat`
  call; the dedicated script is the correct entry point and is what
  `docs/dev_setup.md`'s raw `cmake` recipe implicitly assumes a Developer
  Command Prompt for). `cpp/build/Release/breach_physics.cp312-win_amd64.pyd`
  mtime: **2026-08-20 05:49** (was 2026-07-24 09:18).

## 2. Baseline red-set artifact (full suite)

Command: `conda run -n data python -m pytest tests -q` (also verified via the
direct `data`-env interpreter to sidestep a `conda run` quoting limitation
noted in §7 — same environment either way), from `C:\Users\steen\projects\breach`.
Run **twice** (once for the counts, once with `-rs` for skip reasons) —
identical counts both times: **24 failed, 2251 passed, 5 skipped** (126.66s
and 118.00s respectively; the counts, not the wall-clock, are the oracle).

### 2a. Failed (24) — ALL PRE-EXISTING, unrelated to this arc (fire/materials axis)

```
tests/test_cool_shift_axis.py::test_every_material_carries_the_column_seeded_at_the_old_global
tests/test_cool_shift_axis.py::test_a_crate_grid_from_config_is_uniform_today_but_addressable
tests/test_eos_p4_combustion.py::test_combustion_pass_conserves_o2_n2_soot_exactly
tests/test_eos_p4_combustion.py::test_e2e_1_sealed_room_fire_self_starves
tests/test_eos_p4_combustion.py::test_e2e_2_breach_vents_o2_and_kills_fire
tests/test_eos_p4_combustion.py::test_e2e_4_inert_flood_smothers_fire
tests/test_eos_p4_combustion.py::test_payoff_orderings_perturbation_robust
tests/test_eos_p5_1_stoich.py::test_fuel_decrement_exact_and_deterministic
tests/test_eos_p5_1_stoich.py::test_one_lsb_floor_never_crossed
tests/test_eos_p5_1_stoich.py::test_no_destruction_originates_from_combustion
tests/test_eos_p5_1_stoich.py::test_lifecycle_ember_reignite_charout
tests/test_eos_p6_9_isotropy.py::test_isotropy_bit_exact_zero_remainder
tests/test_eos_p6_9_isotropy.py::test_isotropy_bounded_bias_nonzero_remainder
tests/test_fire_feedback.py::test_cold_fire_decays_to_zero
tests/test_fire_feedback.py::test_low_o2_fire_decays_to_zero
tests/test_fire_feedback.py::test_vented_room_extinguishes
tests/test_fire_feedback.py::test_burnout_when_wall_hp_runs_out
tests/test_fire_feedback.py::test_wind_blows_out_a_small_fire
tests/test_fire_feedback.py::test_plume_raises_own_atmosphere_wind_points_outward
tests/test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood
tests/test_fire_o2_invariant.py::test_production_ignition_matches_cpp_gate_off_tie
tests/test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing
tests/test_pr3_capacity_law.py::test_fire_T_ext_is_derived_from_ignition_temp
tests/test_s3b_fire_determinism.py::test_fire_field_and_burnthrough_list_bit_identical_run_twice
```

None touch `eos_solver.cpp`'s step-4c arithmetic, the EOS trust gate, or
anything else this arc's P-W1a/P-W1b plumb — they cluster entirely in the
fire/combustion/materials axis (a `MaterialTable.ignition_to_ext_delta`
value drift — test expects 100.0, measures 200.0 — looks like the likely
root cause thread; not investigated further, out of scope for P-W0). **This
is the block every later patch's set-diff must treat as pre-existing debt,
not arc damage.**

### 2b. Skipped (5) — also pre-existing

```
tests/test_s2a_mean_reduction.py:83   — no C++ toolchain found (proof-script self-check, unrelated to the vcvars fix above)
tests/test_s2b_reciprocal.py:80       — no C++ toolchain found
tests/test_s3b_sqrt.py:85             — no C++ toolchain found
tests/test_wave_push.py:360           — EOS P3: k_push recalibration deferred to a P5 feel pass (documented, deliberate)
tests/test_wave_push.py:377           — same (knockdown-ring calibration)
```

Non-vacuous-green check (C15): 2251 passed is a real, large majority of the
~2280-test suite — not a suite that silently collected nothing.

## 3. SURPRISES — two findings that contradict the design/critique's stated

**expectations.** Both are measured, reproducible, and reported rather than
adjusted for — the brief says surprises are findings, not noise.

### 3a. The cold-rail WINDOW scenario does **not** currently reproduce the −288.65 spiral

The critique's C11/lineage command (`docs/archive/e1_p_e0_asbuilt_2026-08-17.md:119`,
originally from `docs/storm_audit_2026-08-14.md` row "window" / d′): `conda run
-n data python tools/storm_ledger.py --ticks 4800 --damp 0.005 --pf1b --set
k_wind_strip=0.5` was re-run verbatim (same fixture, same dials — `storm_ledger.py`
and `bench_two_room.py::run_bench` both load the identical committed
`bench_two_room` fixture). **Measured on HEAD today: `t_min_gas` stays
EXACTLY 0.0 for all 4800 ticks** (global min and final both 0.0), and
`work_clamp_hits = energy_floor_hits = t_max_phys_hits = u_clamp_hits =
u_max_hits = 0` throughout — no rail ever engages. This directly contradicts
the storm audit's 2026-08-14 measurement (`T_min gas −288.65 (floor)`,
work-clamp 649, T-floor 4324) that the design doc's §1/§3 narrative (the
"measured −91 → −288.65 → T_MIN spiral") and B-F12's "measured n_bulk
1.7–9.3" both cite as ground truth.

`tools/bench_two_room.py::run_bench`'s own fixed probe cell (`h//2, w//4` =
(7,6) on this fixture) shows the same story: `T_probe` never goes negative
(min 0.0 at tick 1, final 0.1516) and every counter in `run_bench`'s
`res["counters"]` is 0. `ke_peak` (19.16) and `umax_peak` (1.85 m/s) are also
roughly 1000x and 40x smaller than the audit's window-row KE/umax figures —
consistent with a qualitatively different, much calmer trajectory, not a
unit-scale artifact (checked: `ke` here is the same `0.5*sum(N*|u|^2)`-style
quantity the audit table used).

**Most likely explanation (not confirmed — flagging for Erik/orchestrator,
not resolving unilaterally):** P-E4's compression-work trust gate (fading k
toward 0 below `n_work_ref/2`) landed 2026-08-17, three days after the audit
that discovered this window, and the velocity-clamp arc (P-V1, landed
2026-08-19) added the per-cell velocity clamp closing "the diagonal leak and
the global-cap bug" — either or both could plausibly have closed this
specific runaway path. This matters because **the design's RISK-1
re-founding (B-F3) and the whole "the −288.65 spiral should now warm" P-W2
framing assume this scenario currently misbehaves** — on today's HEAD it
does not misbehave at all, so P-W2's planned "cold-rail window re-run"
comparison currently has no BEFORE spiral to show warming relative to. This
should be reconciled before P-W1 proceeds on the RISK-1 narrative as written.

### 3b. Ambient gate-2 is **not** vacuous today — contradicts the design's stated "T ≡ 0" premise

The design (§3, B-F3) and this patch's own brief both state gate 2
(`test_ambient_gate2_rush_in_recovers_and_rails_bounded`) "runs at T ≡ 0 on
HEAD (`_ambient_gmap` never seeds T)" and that its `t_max_phys_hits == 0`
assertion "was VACUOUS until now." **Measured by re-running the test's own
body (imported the test module directly and reused its own `_ambient_gmap`
and `DT_TICK` — not a transcription) for the full 80 ticks:**

```
t_max_phys_hits:   0     (the only value the test itself asserts — still green)
work_clamp_hits:   4345
u_clamp_hits:      2816
u_max_hits:        0
energy_floor_hits: 0
peak interior T:   24.46 game-deg   (mean at tick 80: 3.76 game-deg)
e_vac_wipe_sum:    0
e_ring_pin_sum:    0
```

So `t_max_phys_hits == 0` is a real, non-vacuous green (T reaches tens of
game-degrees, nowhere near the 16000 ceiling) — but `work_clamp_hits` and
`u_clamp_hits` are already firing thousands of times, meaning **the
compression-work RATE rail is already live on HEAD in this exact rush-in
scenario**, not dormant. Traced the mechanism one level (not exhaustively):
temperature is 0 everywhere at tick 0 (confirmed), and HEAD's step-4c is
analytically exact-zero at T_rel=0 for both branches (`mul_q16(k, 0) == 0`;
`floordiv_q(0<<16, ...) == 0`, verified against `eos_solver.cpp:1006-1018`)
— so 4c alone cannot be the seed. Within tick 1 the ambient RING itself
already reads a few raw Q16.16 counts nonzero at some cells (not exactly
pinned to 0 the way the atmosphere pin is), and that tiny leak is enough for
HEAD's existing compounding compression branch (`T_new = T*(1+w)`, the same
mechanism the hot-rail file calls "the old ×1.5-at-the-rail compounding") to
amplify geometrically under the sustained rush-in's negative divergence —
same qualitative mechanism the design attributes to the NEW T_abs law's
ambient entry point (B-F3), just already reachable today via a much smaller
seed. Root cause of the ring's non-exact-zero T not chased further (out of
scope for an instruments-only patch); flagging because B-F3's "re-founded on
the ambient-runaway entry point" argument may need to account for an
entry point that already exists pre-arc, at lower amplitude.

## 4. Quiet-room drift instrument (`tools/quiet_room_drift.py`)

28×28 ambient-bounded box (`_ambient_gmap` recipe transcribed from
`tests/test_air_boundary.py:749`, cited in the tool's docstring), +0.1 atm
Gaussian bump (sigma 4 tiles) at centre, 2000 ticks, `PhysicsRunner(bp)` /
`DT_TICK = 1/24` — the same harness gates 1/2 use.

**Run-twice-identical: VERIFIED at full scale.** Two independent 2000-tick
runs, compared via `np.array_equal` over every series array in the `--out`
npz (`tick`, `eos_energy_books_sum`, `eth_compression_delta`, `max_abs_t_rel`,
`t_min_gas`, all five rail counters): **byte-identical.** Also gated going
forward by `tests/test_quiet_room_drift_smoke.py` (50-tick prefix, part of
the full-suite green count above).

Baseline (2000 ticks):

| field | value |
|---|---|
| `eos_energy_books_sum` (tick 1) | 5,963,776 |
| `eos_energy_books_sum` (tick 2000) | 0 |
| `eth_compression_delta` (run sum) | −6,091,536 |
| max\|T_rel\| over run | 0.0000305 game-deg (2 raw Q16.16 counts) |
| `t_min_gas` over run | 0.0 |
| `work_clamp_hits` / `energy_floor_hits` / `t_max_phys_hits` / `u_clamp_hits` / `u_max_hits` (final) | 0 / 0 / 0 / 0 / 0 |

**Not exactly the idealized "expect exact zeros" the design states (§C11)** —
step 4c itself is analytically exact at T_rel=0 (proven, see §3b above), but
a tiny (±1-2 raw Q16.16 count, ≤ ~6e-5 game-deg) transient appears around the
pressure bump in ticks 1-3 and fully self-heals to exact 0 by tick 4. This
is the pre-existing, already-bounded P-E1 SL-transport LSB truncation noise
(`test_e1_hot_rail.py`'s `test_no_transport_mint` /
`test_transport_delta_is_one_way_negative` govern it — the RUN TOTAL is a
provable loss, but an individual active-flux cell can round either
direction), not a step-4c effect and not new to this arc. Recorded as the
true measured baseline rather than an idealized zero so P-W2's diff is
honest about what changed.

## 5. Storm-ledger battery (P-E0 lineage, `--ticks 4800 --damp 0.005 --pf1b`)

Command run exactly as pinned (`tools/storm_ledger.py --help` confirms the
flag set is unchanged from the P-E0 lineage):

```
final counters: eos.u_clamp_hits=0  eos.u_max_hits=0  eos.work_clamp_hits=0
  eos.energy_floor_hits=0  eos.t_max_phys_hits=0
  eos.eth_transport_delta=-7,671,929   eos.eth_compression_delta=-16,339,797
  eos.e_ts_residual=558,051  eos.e_wipe_sum=0  eos.e_floor_sum=0
  eos.n_active_flux=288  eos.n_bulk_active_sum=18,860,554
  eos.ke_drag_removed=124,731,042  eos.e_drag_deposit=0
  eos.e_drag_drop_sum=39,962,612  eos.e_drag_rail_clipped=0
  comb.heat_floor_hits=0  comb.e_deposit_drop_sum=0
  temp.e_cond_trunc_sum=-488,127,046,577  temp.e_cond_cap_sum=0
  temp.cond_limit_hits=0  temp.e_cool_sum=-1,512,834,547,056,640
  temp.e_vac_wipe_sum=0  temp.e_ring_pin_sum=0
  temp.t_max_phys_hits=0  temp.t_low_rail_hits=0  temp.e_deposit_drop_sum=0
amplifier: max gain 1.1x (4788 ticks with combustion heating)
open_cells=288
```

No rail engagement anywhere in the undamped/no-strip PF1B battery — this is
the CLEAN baseline row (matches the audit's "baseline"/"damped" rows, not
the "window" row — see §3a for the window-dial re-run, which is where the
audit expected rails to fire and today doesn't).

## 6. Cold-rail window row (`tools/bench_two_room.py::run_bench`, WINDOW dials)

Dials: `dict(storm_probe.PF1B, k_wind_strip="0.5")`, `damp=0.005`, 4800
ticks. See §3a for the full discussion — summary numbers:

| field | value |
|---|---|
| probe cell (y,x) | (7, 6) |
| `T_probe` min / final | 0.0 (tick 1) / 0.1516 game-deg |
| `p_probe` min / final | 0.9957 / 0.9965 atm |
| `ke_peak` / `ke_final` | 19.16 / 1.05 |
| `umax_peak` / `umax_final` | 1.85 / 0.297 m/s |
| all `run_bench` counters (`u_clamp_hits`...`t_max_phys_hits`) | all 0 |
| domain-wide `t_min_gas` (via a `storm_ledger.run_ledger` re-run at the same dials — the tool the original audit numbers came from) | 0.0 the entire run |

**Not the −288.65 spiral described in the design** — see §3a. Recorded as
measured; not adjusted to match the expected number.

## 7. Hot-rail, fresh (`test_e1_hot_rail.run_scenario(**HOT)`, 2000 ticks)

Docstring numbers are explicitly stale (pre-energy-books) per the brief;
measured fresh via the test module's own `run_scenario`:

| field | value |
|---|---|
| `t_max_phys_hits` | 0 (the STOP-set gate this patch must keep green) |
| `work_clamp_hits` | 0 |
| `peak_T` | 5553.30 game-deg (well under the 16000 ceiling) |
| `o2_burned_frac` | 0.3410 (34.1%, above the 15% fizzle line — non-vacuous) |
| `n_active_flux` / `n_bulk_active_sum` / `n_cell_substeps` | 105,000 / 8,707,741,591 / 240,064 |
| `e_ts_residual` | 886,443,153,581,471 |
| transport-mint violations (`n_viol`) | 0 |
| worst per-tick overshoot vs allowance | −4,194,304 (negative = under the allowance, healthy) |
| transport delta net total | −886,447,278,141,336 (pure loss, as `test_transport_delta_is_one_way_negative` requires) |

Both `test_no_transport_mint` and `test_no_rail_hits` are currently green
(not in the §2a red list) — consistent with these fresh numbers.

## 8. Ambient gate-2 counters

See §3b for the full discussion and the counter table there. Restated: NOT
all-zero as the brief expected — `work_clamp_hits=4345`, `u_clamp_hits=2816`,
peak interior T 24.46 game-deg — while `t_max_phys_hits=0` (green, and now a
genuinely non-vacuous green) and `energy_floor_hits=e_vac_wipe_sum=e_ring_pin_sum=0`.

## 9. Counter exposure map (u_max_hits / u_clamp_hits / e_vac_wipe_sum / e_ring_pin_sum)

| counter | storm-ledger battery (§5) | window row (§6, via storm_ledger re-run) | hot-rail (§7) | gate-2 (§8) | quiet-room (§4) |
|---|---|---|---|---|---|
| `u_clamp_hits` | 0 | 0 | not exposed by `run_scenario` (eos-level only: `t_max_phys_hits`, `work_clamp_hits`) | **2816** | 0 |
| `u_max_hits` | 0 | 0 | not exposed | 0 | 0 |
| `e_vac_wipe_sum` (temp.) | 0 | not captured (storm_ledger's own counters() reads it; not re-queried for the window re-run) | not exposed | 0 | not applicable (no temp.-holder query in this tool) |
| `e_ring_pin_sum` (temp.) | 0 | not captured (same as above) | not exposed | 0 | not applicable |

`test_e1_hot_rail.run_scenario` only threads the `eos`-level counters listed
in its own return dict (`t_max_phys_hits`, `work_clamp_hits`) plus the P-E1
`eth_ticks`/`e1` block — it does not expose `u_clamp_hits`/`u_max_hits`/the
`temp.*` channels at all, so those cells are genuinely "not exposed by this
run" rather than a zero.

## 10. Commands run (for reproducibility)

```
cmd /c cpp\build_cuda_lenovo.bat
cmd /c cpp\build_cpu_data.bat
conda run -n data python -m pytest tests -q
conda run -n data python -m pytest tests -q -rs
conda run -n data python tools/quiet_room_drift.py --ticks 2000 [--out ...]   # run twice, npz-diffed
conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.005 --pf1b
conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.005 --pf1b --set k_wind_strip=0.5
# bench_two_room WINDOW row + the test-module-reused gate-2 body + the fresh
# hot-rail run were driven via short scratchpad scripts calling
# bench_two_room.run_bench / test_e1_hot_rail.run_scenario / the
# test_air_boundary module's own _ambient_gmap+DT_TICK directly (not
# transcribed) -- not committed, per the brief's tools/+tests/+docs/ write
# surface; the exact call sites are quoted inline in §§4-8 above.
```
