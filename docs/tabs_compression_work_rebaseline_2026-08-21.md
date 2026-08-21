# Golden re-baseline — T_abs compression-work arc close (2026-08-21)

Written rationale for the ONE deliberate golden re-baseline this arc is
allowed, per the standing ruling (Erik, 2026-08-19, re-confirmed in
`docs/tabs_compression_work_design_2026-08-20.md` §6): re-baseline happens
ONCE, after the bless, with a written rationale doc, and scenario-expectation
re-pins (e.g. b6 latency ticks) get their own measured rationale line in the
same doc rather than riding silently on the digest event. Erik HUMAN-TESTED
and blessed the arc on 2026-08-21 (build-vs-main, per design D-2 — no dial).
This is that record, following the format and procedure of the velocity-clamp
arc's `docs/golden_rebaseline_2026-08-20.md`.

## What changed physically

Step 4c (the EOS's reversible compression/expansion work term,
`eos_solver.cpp:984-1022` + its CUDA and reference twins) previously ran on
**ambient-relative** T: below ambient (T_rel < 0) the compression branch
multiplied a negative number by `(1+w)`, so compression made cold gas
COLDER — an inverted sign, not merely a missing term — and ambient air
(T_rel = 0) was an exact fixed point of the whole step (`k·0 = 0` on both
branches), so acoustics deposited no temperature signal anywhere near
ambient and breach rarefaction was thermally invisible. RULING R1
(Erik, 2026-08-17) executed: the interior of the 4c arithmetic now runs on
**absolute** T (`t_abs = T_rel + t_amb_q`, int64, A7-floored fold), then
shifts back out to the stored ambient-relative convention, which stays
load-bearing everywhere else (books, vac/ring wipe, render, ignition).
Compression now honestly warms cold gas, ambient air participates in 4c for
the first time, and rarefaction cooling registers (the work-clamp figure,
~-96.67 game-deg at the shipped clamp). The canonical A/B scenario
(`field_ab_harness.default_scenario_sim`, 30 ticks) exercises the EOS every
tick, so this lawfully moves every pressure/thermal-coupled field
trajectory — the predicted `GOLDEN_AGGREGATE` cascade (the 12 importers +
`test_w6_armory`) plus `test_b6_logic_golden`'s inline golden (its loop
closes through the atmosphere solver), not a scattered regression.
`DIGEST_SPEC_VERSION` is unchanged — values moved; no field added, removed,
or retyped.

Full physics narrative, the four measured rulings (R-1..R-4) that resolved
P-W1b's STOP-set reds, and the accepted gaps are in
`docs/tabs_compression_work_design_2026-08-20.md` §0b/§2/§3/§7; the
HUMAN-TEST brief content is design §8 + manifest §8
(`docs/tabs_compression_work_manifest_pw1b_2026-08-20.md`).

## What was re-baselined, and the old -> new values

### `tests/_xarch_perfield_digest.py::GOLDEN_AGGREGATE`

Regenerated per the file's own documented procedure (`main()`, module
docstring): `conda run -n data python tests/_xarch_perfield_digest.py`,
reading the printed `aggregate digest` value. Run twice, independently:
identical both times.

```
old: d575df33de5c2af37108d29b73853b465eda761b148c6b812f4a4c4da40e0bb0
new: a2cbc77ac324db99e0fcf2dc76e9ca15b3187c220a6d5abc5f4a110022c65cea
```

This single constant is imported by `tests/test_w6_armory.py` and is the
value the 11 CUDA/EOS lockstep-golden test files (plus
`test_cuda_p64_kick_compression.py`) compare against
(`trajectory_digest(traj)`), so one edit fixes all 12:

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
tests/test_cuda_p64_kick_compression.py::test_p64_kick_compression_bit_identity
```

Per the manifest (§4), every one of these rows had already been verified
with the mismatched hash single-sourced (measured `a2cbc77a...` vs
committed `d575df33...` on every row) and every lockstep (CPU<->GPU
differential) PART bit-identical green — confirming one underlying digest
event, not independent regressions.

Determinism sanity: ran the dumper script twice after the P-W1b law
landed; both runs produced the identical new digest, matching the value
independently measured via the full-suite `pytest tests -q` assertion
diff.

**Not regenerated: the per-machine diagnostic artifact**
`tests/_xarch_perfield_erik_lenovo.txt` (written as a side effect of running
`tests/_xarch_perfield_digest.py`). Checked precedent: the velocity-clamp
arc's close (`62eb119`) did not touch this file either (last committed at
`bae8871`, eos-p3fix, July) — it is a THROWAWAY cross-machine diagnostic
(per its own module docstring), not part of the golden re-baseline
procedure. Mirrored here: the local modification from running the dumper
was reverted (`git checkout -- tests/_xarch_perfield_erik_lenovo.txt`)
before committing.

### `tests/test_b6_logic_golden.py::LOOP_GOLDEN_TRAJ_DIGEST`

The sensor -> filter -> decider -> door closed-loop golden (seed=1,
`breach_physics`, 30 ticks) — closes through the atmosphere solver, so it
is exactly the kind of trajectory step 4c's law change moves. Regenerated
the same way as the velocity-clamp close (actual value read from pytest's
assertion diff), re-run twice independently to confirm stability
(identical both times; the test's own `test_logic_loop_digest_is_reproducible_and_2x_bit_identical`
also confirms two independent captures within one run agree with each
other).

```
old: ed42914ebe44d355ab311e0346ce8d9602dd9728887f1fe35fe7a377dc5cb189
new: a631c182c5669ebefd390dd321868874bbe17db1cd1f3e3195be1c276ede05dd
```

Consumed by both `test_logic_loop_trajectory_digest_matches_committed_golden`
and `test_logic_loop_digest_is_reproducible_and_2x_bit_identical` — one edit
fixes both.

### `tests/test_b6_logic_golden.py` latency pins — measured, UNMOVED

```
LOOP_FILTER_CROSS_TICK:  11 -> 11  (unmoved)
LOOP_DECIDER_HIGH_TICK:  12 -> 12  (unmoved)
LOOP_DOOR_OPEN_TICK:     13 -> 13  (unmoved)
```

Measured rationale: `test_logic_loop_latency_pins_the_per_hop_contract`
passed green on this arc's own build with no edits (confirmed by running
`tests/test_b6_logic_golden.py -q` before touching the file: 2 failed —
both digest-only — 5 passed, including the latency test). The chamber's
pressure trajectory up to first threshold-cross is dominated by the sealed
room's fill rate against the fixed 0.75 atm decider threshold, not by the
ambient-air 4c term this arc changes (the loop's chamber air starts well
above ambient and the new law's ambient-participation term is a small
perturbation against that fill dynamic) — so the tick-level actuation
latency and the 1-tick-per-hop contract (§2c) survive unchanged. No re-pin
was needed; recorded here per the design's standing instruction that
latency re-pins get their own line even when the line says "unmoved."

## Suite counts — before / after

Command: `conda run -n data python -m pytest tests -q`, repo root, working
tree `tabs-compression-work` at `06bb973` (both pyds current, not rebuilt
by this close).

| | before this commit | after this commit |
|---|---|---|
| failed | 38 | **24** |
| passed | 2242 | 2256 |
| skipped | 5 | 5 |
| wall clock | 126.27s | 124.37s |

`38 = 24 (P-W0 baseline, pre-existing fire/materials debt, unchanged) + 14
(the EXPECTED sub-tests this re-baseline settles: the 12 `GOLDEN_AGGREGATE`
importers + `test_b6_logic_golden`'s two golden-digest sub-tests)`. After
this commit the red set is **EXACTLY** the P-W0 baseline's 24 pre-existing
reds (`docs/tabs_compression_work_baseline_2026-08-20.md` §2a) — verified by
full-suite name comparison, zero GRAY/unexplained reds, zero deviation from
the manifest's predicted target.

The `test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood`
`XPASS(strict)` is still present in this run — confirmed pre-existing
baseline debt (same test name, same `MaterialTable`/rad-books drift
suspected in the P-W0 baseline, unrelated to the T_abs law), not a new red
caused by this arc. Reported per protocol, not silently un-xfailed; the
open question of whether warmer ambient air under compression work is what
tips this scenario into ignition was already flagged for Erik at
P-F1b triage in the P-W1b manifest §5 and is unchanged by this close.

## What was explicitly NOT touched

No other golden/digest constant in the repo was regenerated.
`test_b1_signal_bus`/`test_b2_nodes`'s dormancy digests (`physics=None`)
were not re-run or edited — confirmed green and unchanged in both the
before and after full-suite runs, as required (a flip there would mean a
non-physics leak). `tests/_xarch_perfield_erik_lenovo.txt` was regenerated
locally as a side effect of running the dumper's documented procedure, then
reverted before committing (see above) — not part of this event, mirroring
the velocity-clamp close's precedent.

## Standing-ruling statement

This is the `tabs-compression-work` arc's single sanctioned golden
re-baseline event, executed once, after Erik's 2026-08-21 HUMAN-TEST bless,
per the deferred-canon-fold-cadence ruling (Erik, 2026-08-19) and its
re-confirmation in `docs/tabs_compression_work_design_2026-08-20.md` §6.
