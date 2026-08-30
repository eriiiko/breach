# Gas-energy conservation arc — P-G3 golden re-baseline (2026-08-30)

Arc #54. Patch P-G3 (design v4, `docs/gas_energy_conservation_design_2026-08-29.md`
§6). This is **value-move event 2** for this arc — the arithmetic move, as
opposed to P-G0's schema-only bump (event 1: digest spec v4 → v5, +`gas_energy`
int64). Executed on branch `gas-energy-arc`, on top of `df4b988` (P-G2b).

## Why every golden moved

Four patches landed on this branch between the P-G0 schema bump and this
re-baseline, each moving the canonical A/B scenario's arithmetic (not its
shape):

- **P-G1a** — the kick-loop KE brackets (RAD_SAFE guard moved above the
  brackets and tightened to 2^27 raw/component; ∇p kick, `dyn_wave_absorb`,
  the B3c sponge band, the velocity cap, and staged drag each individually
  booked) and the sub-cycled, two-pass-rail face-flux energy step (design
  §2.3/§2.4) replace step 4c's per-cell temperature-form compression work on
  the CPU EOS path. Every pressure- and thermal-coupled field the canonical
  scenario's fire + smoke drive through the EOS moves.
- **P-G1b** — the writer seam (`gas_energy_move` / `gas_energy_deposit`) went
  live for combustion, the thermal solver's gas side, pump primitives, and
  seal/unseal/`destroy_wall`/`on_tile_changed`; the EOS-entry re-sync that had
  kept `gas_energy := N·T_abs` transitional was deleted, so the field is
  **D1-live across whole ticks** instead of being re-derived from the T
  mirror every tick.
- **P-G1d** — the pressure solve's divergence stencil moved to **face form**
  (`û = 0` at solid faces, the exact discrete adjoint of the kick's pressure
  gradient). Interior cells are bit-identical (the `u_i` terms cancel before
  the shift), but every field downstream of a wall-adjacent pressure cell
  moves — this is the "feel-adjacent" fix named in the design (BLAST peak
  |u| 8.7 → 18.9 m/s, AS glass bursts 3 → 16 tiles at the arc's own bench).
- **P-G2** — CUDA twins only (K1/K3, bulk transport, combustion, temperature
  kernels; resident buffers; counter array). **Zero CPU source touched**
  (`git diff d3c6689 -- '*.cpp'` is empty), so this leg contributes nothing to
  the CPU trajectory the goldens below pin — named for completeness since it
  sits between P-G1d and this re-baseline on the branch.

`DIGEST_SPEC_VERSION` is unchanged at v5 (set at P-G0): every move below is
values, not membership/dtype/shape.

## Procedure

Per the design's R3-#11 procedure and this file's own convention
(`tests/_xarch_perfield_digest.py`): ran the canonical A/B scenario on the
clean CPU build (`cpp/build/Release`, rebuilt via `cpp/build_cpu_data.bat`)
and captured the 30-tick aggregate trajectory digest directly (no committed
per-tick reference file survives from the P-G0 rebase to diff against
key-by-key here, since P-G0 already verified the schema-only move by
per-field array comparison, not by this digest tool — this event is a values
move, verified instead by tracing the cause to four already-gated, already-
measured patches, not by a blind digest replace).

## Before -> after (aggregate / inline goldens)

| constant | file | before | after |
|---|---|---|---|
| `GOLDEN_AGGREGATE` | `tests/_xarch_perfield_digest.py` | `df1f5153c9ce60a4de8e9c2198ff8eab3eb8d8267cf8be43d3ede03650b236bd` | `f6daf44f4c2f563fc88bdb4465fb681a776141a9079d0e7c0f62f5c2b7fbb306` |
| `DOORTEST_NOPHYS_TRAJ_DIGEST` | `tests/test_b1_signal_bus.py` | `d256de5eb8094e03877e300e98dbe8a19746ce89df73299922a68bed3d7b993e` | `76ba6dc1c2800eae16f9f98f27abd1646c656e4068773a8150e94465e614cc35` |
| `DOORTEST_NOPHYS_TRAJ_DIGEST` (own copy) | `tests/test_b2_nodes.py` | `d256de5eb8094e03877e300e98dbe8a19746ce89df73299922a68bed3d7b993e` | `76ba6dc1c2800eae16f9f98f27abd1646c656e4068773a8150e94465e614cc35` |
| `LOOP_GOLDEN_TRAJ_DIGEST` | `tests/test_b6_logic_golden.py` | `4fa67f37383c9c3abeedef73699f480e2d7f30d35d37397b34719ad653778769` | `38a47454a12b09b7815c9b95b672e815f9291bb0b3e42c30386fdb2577b3b6b3` |

`test_b1_signal_bus.py`'s door-present, wire-free scenario runs
`breach_physics=None` (no EOS step ever executes) — its move is the
**one-time level-load `gas_energy` seed** (`N_raw × T_abs_raw`, refreshed at
load per P-G1b) and the constants folded at that seed (§2.1's `C` derived
from `T_AMB_K`), not a per-tick divergence; verified every other field in
that trajectory is untouched. `test_b6_logic_golden.py`'s loop scenario runs
LIVE physics (it is a pump/vent loop), so it carries the full P-G1a/b/d move.

`test_b5_airlock.py` and `test_b6_logic_golden.py`'s own
`test_b1_dormancy_door_present_wire_free_still_byte_identical` both call
`test_b1_signal_bus`'s function directly (no separate copy of the constant),
so they picked up the fix automatically once `test_b1_signal_bus.py` moved.
`test_w6_armory.py::test_canonical_scenario_matches_sanctioned_golden` and
all 11 `cuda_*_check.py` golden Parts import `GOLDEN_AGGREGATE` from
`tests/_xarch_perfield_digest.py` (single-sourced since 2026-08-18) and
likewise needed no value edit — only their stale
`# EXPECTED RED until P-G3 re-baseline (#54)` comments, removed.

ENTITY_SECT: no separate golden rows moved. The vent plenum ledger's
conversion at the pump seam (design §2.7) is exercised by
`tests/test_vent_determinism_and_serialize.py`, `tests/test_vent_dormancy.py`,
and `tests/test_vent_mechanism.py`, all of which assert properties (exact
determinism / dormancy transparency / mechanism behaviour), not golden
hashes, and were already green going into P-G3.

## Files touched (golden values / stale-comment cleanup)

- `tests/_xarch_perfield_digest.py` — `GOLDEN_AGGREGATE` + lineage paragraph.
- `tests/test_b1_signal_bus.py` — `DOORTEST_NOPHYS_TRAJ_DIGEST` + lineage comment.
- `tests/test_b2_nodes.py` — its own copy of `DOORTEST_NOPHYS_TRAJ_DIGEST` + lineage comment.
- `tests/test_b6_logic_golden.py` — `LOOP_GOLDEN_TRAJ_DIGEST` + lineage comment.
- `tests/cuda_combustion_check.py`, `cuda_conduction_check.py`,
  `cuda_eos_step_check.py`, `cuda_fire_check.py`, `cuda_kick_check.py`,
  `cuda_mg_solve_check.py`, `cuda_p62_check.py`,
  `cuda_s2b_raycaster_live_check.py`, `cuda_s3_check.py`, `cuda_s4a_check.py`,
  `cuda_trace_smoke_check.py` — removed the stale `EXPECTED RED until P-G3`
  comment block (no value edit needed; single-sourced from
  `GOLDEN_AGGREGATE`).

## Verification

- `pytest tests/test_b1_signal_bus.py tests/test_b2_nodes.py
  tests/test_b5_airlock.py tests/test_b6_logic_golden.py
  tests/test_w6_armory.py -q` → 110 passed.
- `pytest tests/test_cuda_*.py -q` (all 24 CUDA-wrapper tests spanning the 11
  re-baselined check scripts) → 24 passed.
- Full suite (`pytest tests -q`) after this patch's golden work plus the
  mindful restatements (test decisions delegated by Erik, tracked separately)
  → 24 failed (2 pre-existing `test_cool_shift_axis` + 22 pre-existing
  fire/combustion-family, both verified identical on `main`), 2318 passed.
