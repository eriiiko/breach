# Velocity-clamp arc — P-V1 as-built (2026-08-19)

Implements `docs/velocity_clamp_impl_2026-08-19.md` (v3) sites 1-18, exactly
as spec'd. This doc records what was built, gate results, and the two
incidental (pre-existing, unrelated) bugs found and fixed along the way
because they blocked gate 4 as written.

## What was built (matches the spec verbatim)

**A. Plane fold — two transcription sites, no third** (`eos_solver.cpp`'s
step() scan, `cuda_eos_step.cu`'s `eos_host_prestage` scan). Both compute
`cap2_plane[i] = c_amb²·t_abs_cap/T_AMB` (Q32.32, `mul128_shr`), railed at
`u_max²` above `ratio_umax`, D1-floored and D4-routed to ambient for
`ts[i]`, filled with `u_max²` on `solid||is_vacuum`. `t_max_abs_raw`/
`c_local_q` are unchanged (D7 note below). `cuda_eos_resident.cu`'s
`eos_step_resident` calls the SAME `eos_host_prestage` (adds the missing
`thermal_solid` argument at its call site) — the resident path only
uploads the host-folded plane, it does not re-fold.

**B. Kick clamp — all three sites, identical shape.** Replaced the
`u_cap_q = min(c_local_q, u_max_q)` + component-Chebyshev-pretest +
`reciprocal_q16` rescale with: `cap2_q32 = cap2_plane[i]` (D5, trusted
verbatim) → `rad = ux²+uy²` (RAD_SAFE-guarded, unchanged) → `rad > cap2_q32`
exact test (D3 counter semantics) → D6 exact int64-divide rescale
(`ux = ux*u_cap_q/umag`). Sites: `eos_solver.cpp` step()'s live kick,
`eos_solver.cpp`'s P6.4 reference (`eos_kick_compression_reference`),
`cuda_kick_compression.cu`'s `kick_kernel`. `u_max2_q32` folds from
`u_max_q` identically at every site (host scalar fold or kernel-local).

**D7** — the `u_est` clip (n_sub/CFL derivation) widens from `c_local_q`
alone to `max(c_local_q, u_max_q)`, both sides (`eos_solver.cpp`,
`cuda_eos_step.cu`'s `eos_host_prestage`).

**Plumbing**: `EOSHostPrestage` gained `std::vector<int64_t> cap2` (the
coeffE/coeffS host-vector idiom) and a `thermal_solid` parameter (+ `ts`
fallback fold + `u_max_q` fold); `eos_kick_compression`/
`eos_kick_compression_reference`/`kick_compression_launch_resident`/
`kick_kernel` all take `cap2_plane`/`d_cap2_plane` instead of `c_local_q`;
`KickScalarFolds` gained `u_max2_q32`; `EOSResidentScratch` gained an
`int64_t* cap2` member (P-E1's `a64` idiom), uploaded every tick alongside
`coeffE`/`coeffS`/`absorb_q` (measured cost below). Both pybind entries
(`cuda_eos_kick_compression`, `eos_kick_compression_ref`) take
`py::array_t<int64_t> cap2_plane` (the `fuel_recip` marshalling idiom) in
place of `c_local_q`; both docstrings state the `cap2_plane >= 0` hard
contract. All comment corrections the spec named were made, including the
√2-slack line (now "max(u_cap, RAD_SAFE)") and the P6.4 contract-block
inversion (the cap is now derivable from the replay's own t0, not
`dbg_last_c_local_q` telemetry).

**Python callers** (sites 12-16): `tests/cuda_kick_check.py` — `_run_pair`
takes `cap2_plane`; PART 1 rewritten to two constructed UNIFORM planes per
regime (`u_max²` / `c_amb²`) instead of scalar `c_local_q`, including the
drag forcer's u_max² plane (D5: not a "2300²-style" plane, which would have
disengaged the clamp under D5's no-re-min contract); PART 2 rebuilds the
plane from `t0` via formula A, ported in plain Python ints (`_fold_cap2_plane`,
the `c_amb2*ratio` overflow the spec named). `tests/cuda_thermal_mass_eos_check.py`,
`tests/test_thermal_mass_axis.py` (x2), `tests/test_p_e3_drag.py` (x2, plus
`_run`), `tests/test_p_e4_reversible_work.py` — all converted their scalar
`c_local_q` kwarg/positional to a uniform `cap2_plane` array at the same
numeric cap (signature-only where the spec said so; `test_p_e4`'s 300.0
default explicitly NOT "fixed").

**New property tests** (`tests/test_velocity_clamp_property.py`, gates 1-2):
`test_gate1_diagonal_leak_closed_exact` (deterministic 0.75×cap diagonal
forcer + 40-case random fuzz, `rad <= (floor(sqrt(cap2))+2)²` in int64),
`test_gate2_cap_locality_hot_cell_does_not_raise_a_remote_cap` (formula-A
fold from a hot pocket, asserts the cool region's cap stays EXACTLY
ambient), `test_gate2_full_engine_smoke_playground_blast` (5-tick real
engine run, loose `cap2(T_snap)·1.5` bound).

## Two pre-existing, unrelated bugs found and fixed

Both were discovered while chasing gate 4's PART 2 wind-side requirement
and are **not** velocity-clamp arithmetic — confirmed by reproducing both
on the unmodified pre-P-V1 tree (`git stash` + rebuild + rerun) before
touching anything, per the "don't improvise on the locked spec" instruction
— these are test-harness bugs, not sim-behavior changes.

1. **`cuda_kick_check.py` PART 2's `consts` dict never passed
   `k_drag`/`k_drag_heat_frac`/`c_v`**, so the CPU reference always ran drag
   DORMANT while the live engine (shipped `config.toml`
   `[physics.eos] k_drag = 0.5`) runs drag ACTIVE. Reproduced bit-for-bit on
   HEAD: `digest_velocity` and `wind_x`/`wind_y` diverged from tick 0 by a
   uniform ~2.13% factor (`1/(1-kd_q/65536)` at `kd_q = quantize(0.5/24)`) —
   exactly the drag shrink the reference was silently skipping. The stale
   comment above the dict ("k_drag left at the pybind default — matches the
   live engine's shipped default") was true only before P-E3 shipped a
   nonzero default; nobody updated this test when that landed. Fixed by
   reading `k_drag`/`k_drag_heat_frac`/`c_v` from the live `eos` object —
   confirmed this alone makes PART 2's wind side byte-identical, 120/120
   ticks, on the unmodified tree.
2. **My own edit** to PART 2's "required in-engine rails" vacuousness check
   (below) introduced an index bug: after dropping `"u_max"` from the
   3-tuple, `enumerate()` no longer lined up with the `totals` array's fixed
   9-wide position, so `totals[1]` (`u_max`, correctly 0) was checked under
   the label `"work_clamp"` (`totals[2]`, actually 1495). Fixed by indexing
   via `COUNTER_NAMES.index(name)` instead of the loop's own enumerate index.

## Gate results

**Gate 1** (diagonal leak, exact) — PASS.
**Gate 2** (cap locality, exact + loose smoke) — PASS.
**Gate 3** (P6.4 reference parity) — PASS, via `cuda_kick_check.py` PART 2:
reference and step() now agree exactly, 120/120 ticks, wind-side.

**Gate 4** (CUDA lockstep):
- `cuda_kick_check` PART 1 (rewritten): PASS — 40/40 configs bit-identical
  (fields, both digests, all 9 rail counters); coverage: every rail counter
  engaged at least once across Part 1.
- PART 2 wind-side (plane from t0): PASS — **restored to full validity**,
  120/120 ticks zero divergence (`digest_velocity`, `wind_x`/`wind_y`,
  T-independent counters == solver; GPU == CPU ref on everything). In-engine
  rail engagements over the run: `u_clamp=915, u_max=0, work_clamp=1495,
  energy_floor=0, t_max_phys=0`. **`u_max` is deliberately no longer a
  required rail** (was: `("u_clamp", "u_max", "work_clamp")`; now:
  `("u_clamp", "work_clamp")`) — under D2v2 the near-ceiling hot pocket
  (a static thermal anomaly with no local pressure driver) never carries
  fast local flow, so `u_max` never binds there; this is *correct* per-cell
  behavior, not a scenario weakness (the old global-scalar bug is exactly
  what made `u_max` trivially bind everywhere before). P-V2's job is to
  measure how often the hot+fast-flow combination occurs in a real blast.
- PART 2 `digest_compression`: not checked against ground truth — this was
  **already true before P-V1** (P-E4's repair, `cuda_kick_check.py`'s own
  docstring). Nothing to measure before/after within this specific
  assertion; P-V1 did not touch or attempt to restore that coverage.
- PART 3 golden: FAIL (`d575df33de5c2af3...` vs the committed
  `a18e0dfb017b98cb...`) — **expected**: sim digests move, that is the
  point (behavioral-change note, no re-baseline this patch). Accounted for
  under gate 5's GOLDEN_AGGREGATE exception below.
- Full-engine CPU-vs-CUDA A/B, both modes: `tests/test_cuda_eos_step.py`
  PART 1 (per-call, chained P6.5 dispatch) PASS — 120 ticks bit-identical
  across all EOS fields, all six digests, all five (of the nine) legacy
  rail counters; `tests/test_cuda_s8a_residency.py` (resident) PASS —
  per-tick field + all 9 counters (`_COUNTERS` includes `u_clamp_hits`)
  lockstep, zero divergence. `tests/test_cuda_thermal_mass_eos.py` (the
  thermal-mass-axis EOS gate, exercises site 13's `cap2_plane` conversion
  directly, both per-call and resident) PASS.
- **Resident-path H2D cost**: one `int64_t[h*w]` upload per tick alongside
  `coeffE`/`coeffS`/`absorb_q` — 56 KB at the 70×100 playground scale
  (`n=7000, 8 bytes/cell`), scales linearly with map area.

**Gate 5** (suite delta): `pytest tests -q` — post-patch
**42 failed, 2231 passed, 5 skipped** (125.20s) vs the step-0 baseline's
**31 failed, 2239 passed, 5 skipped** (126.18s).
Diff: **11 newly-red, 0 fixed, 0 other new red.**

The golden trajectory (`capture_trajectory(n_steps=30)` on
`default_scenario_sim`) **does engage the new clamp**: measured
`u_clamp_hits = 4` over 30 ticks (all in tick 0 — a startup transient;
`u_max_hits = 0`), confirming the 11 flips are the predicted
GOLDEN_AGGREGATE cascade, not scattered regressions. All 11 fail on the
**identical** digest pair (`d575df33de5c2af3...` vs
`a18e0dfb017b98cb...`) with every other assertion in each file (PART 1
CPU-vs-GPU bit-identity, in every case) passing — confirmed by inspecting
each failure's full output, not just the pytest summary line. Exact
flipped-test list (11, EXPECTED-red-until-arc-close):

```
tests/test_cuda_eos_step.py::test_p65_eos_step_chained_bit_identity
tests/test_cuda_mg_solve.py::test_p63_mg_solve_bit_identity
tests/test_cuda_p62_sl_advection.py::test_p62_sl_advection_bit_identity
tests/test_cuda_p66_conduction.py::test_p66_conduction_bit_identity
tests/test_cuda_p68_fire.py::test_p68_fire_bit_identity
tests/test_cuda_p69_combustion.py::test_p69_combustion_bit_identity
tests/test_cuda_s2b_raycaster_live.py::test_s2_raycaster_live_heat_bit_identity
tests/test_cuda_s3_water.py::test_s3_water_bit_identity
tests/test_cuda_s4a_smoke.py::test_s4a_smoke_bit_identity
tests/test_cuda_trace_smoke.py::test_trace_smoke_bit_identity
tests/test_w6_armory.py::test_canonical_scenario_matches_sanctioned_golden
```

(`test_cuda_p64_kick_compression.py::test_p64_kick_compression_bit_identity`
was already in the step-0 baseline red list — same golden dependency, not
newly red.) The +3 in `2231 = 2239 - 11 + 3` is
`tests/test_velocity_clamp_property.py`'s three new passing tests.

## Accepted deviations from the spec

None in the implementation itself (sites 1-18 built exactly as specified).
Two test-harness bugs were fixed beyond the spec's explicit site list (see
above) — both were pre-existing, unrelated to velocity-clamp arithmetic,
and blocking gate 4 as literally written; both were verified against the
unmodified tree before being attributed to anything other than P-V1.

## Close-out note (for the arc-close re-baseline)

At close, re-baseline `GOLDEN_AGGREGATE` in `tests/_xarch_perfield_digest.py`
(one event, with this doc as the lineage entry) — this fixes all 11 flipped
tests plus `test_cuda_p64_kick_compression.py` in one commit. No other
golden/digest debt was touched by this patch.
