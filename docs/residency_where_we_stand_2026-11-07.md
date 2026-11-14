# Residency + batching — where we stand (2026-11-07)

The orientation page for the arc that finishes S8 and makes the sim a batched,
device-authoritative environment for RL training (issue #29, with #77, #24,
#25, #71 under it). Written before the benchmark and the design doc, from the
prior docs, the issues and the code at main `ea7ee92`. Nothing here is new
design; it is what exists, what was measured, and what the design must settle.

**Mandate (Erik, 2026-11-07):** Claude owns the arc and runs it autonomously,
one agent at a time; Erik checks in. **Decisions that day:** the word is
*residency* (state lives in VRAM across ticks, the host never owns it) plus
*batching* (N games per launch, every field `(N, h, w)`); the traffic problem
is *authority*, not bandwidth; **snapshots with catch-up** is how game dev
continues (the RL trains on a tagged version in its own checkout and build;
game dev goes on on main on the reference path; device twins catch up in
batches); zombie pathing may become a per-env distance field instead of
per-unit A* ("absolutely no problem").

## 1. Vocabulary

| term | meaning here |
|---|---|
| per-call CUDA (`--cuda`) | each solver's `*_step` allocates, uploads, launches, downloads, frees — every call |
| resident (`--cuda --resident`) | persistent CuPy buffers on the device; the launch cores (`*_launch_resident`) do launches only; `GameMap.from_host` / `to_host` move named fields in place |
| host mirror | the numpy fields; **authoritative today**: every host writer (FieldEdits, `stamp_units`, combat, `destroy_wall`, the brackets) writes it, and the device is refreshed from it |
| bracket | a stage that still runs on the host mirror inside the resident tick (combustion, sky exchange, the tail: ripple + fire + temperature + the radiation sweep) |
| batch-shaped | a launch core whose planes are `(N, h, w)` and whose per-env scalars are device arrays of length N (CLAUDE.md "RL-batch habits", N = 1 today) |

## 2. The tick today

### 2.1 The conductor (`Simulation.step`, all host, Python)

1 events · 1b intents · 2 projectiles · 2b statuses · 3 movement · 4 shooting
· 4b sprays · 4c resolve (#169) · 5 zombie AI · 6 `stamp_units` (C++) ·
6b FieldEdit flush · **7 physics** · 9 burn-through walls · 9b over-pressure
wall failure · 9c unit heat damage · 9d ignition from temperature · 9e
entities (vents, pumps, charges, heat sources, doors) · 10 tick advance.
Every slot but 7 reads and writes the host mirror and the Python `Unit`
objects. The bots and the horde run before the step as controllers through
the intent facade.

### 2.2 Slot 7 on the resident path (`PhysicsRunner._step_resident`)

| step | runs on | crosses the bus |
|---|---|---|
| 1 host pre-physics (water pre-step, ambient args) | host | — |
| 3 water substeps | device (`water_substeps_resident`, `(h, w)`) | H2D 5 fields, D2H 3 fields; then the W5/W3 water tail on the host |
| 4 EOS stage (Path A, 2026-07-21) | host pre-stage reductions on the mirror, then the device chain: substeps with the trace planes riding stages 3b/3c, div_u / N / p*, on-device MG build, vcycle, kick, store | H2D 11 fields; zero mid-stage plane transfers |
| 5 trace tail | device (`run_trace_tail_resident`, `(N, h, w)`) | small H2D of folds, small D2H of books |
| 6 synced-set D2H | — | D2H 7 fields (`atmosphere, wave_p, wind_x, wind_y, temperature, gas, gas_energy`) |
| 7 combustion | host (`CombustionSolver`; the CUDA twin is NOT bit-identical, #71) | — |
| 7b sky exchange | host | — |
| 8 tail: ripple, fire, temperature, radiation sweep (`step_tail`) | host, with fire / temperature / the sweep on their per-call CUDA twins when the backends are on | the per-call twins' own H2D / D2H of their planes, every tick |

So one resident tick today still carries about 16 field uploads, 10 field
downloads, the per-call twins' transfers inside the tail, and a host sync at
every bracket. The resident set is `GameMap._RESIDENT_SYNCED` (20 fields) +
`_RESIDENT_MASKS` (13), allocated by `enable_residency` as CuPy copies;
`residency_on()` is never cleared once set (#77 pointer, 2026-09-30).

### 2.3 What is batch-shaped and what is not

| launch core | shape | source |
|---|---|---|
| `water_launch_resident` / `water_substeps_resident` | `(h, w)` | `cuda_water.cu` |
| `eos_step_resident`, `eos_mg_vcycle_resident`, the SL advection, bulk flux, kick / compression cores | `(h, w)`; the MG hierarchy is built per grid | `cuda_eos_resident.cu`, `cuda_mg_solve.cu`, `cuda_sl_advection.cu`, `cuda_bulk_transport.cu`, `cuda_kick_compression.cu` |
| `trace_tail_launch_resident` | `(N, h, w)`, N = 1 | `cuda_resident.h` |
| `div_source_launch_resident` | `(N, h, w)` | `cuda_resident.h` |
| `energy_flux_launch_resident` | `(h, w)` | `cuda_resident.h` |
| `radiation_sweep_launch_resident` (heat + light) | `(N, h, w)`, per-env scalars as device arrays | `cuda_radiation_sweep.cu` |
| temperature, fire, combustion | per-call twins only (`*_step`), `(h, w)` | `cuda_temperature.cu`, `cuda_fire.cu`, `cuda_combustion.cu` |

Every kernel is a grid-stride loop over `n = h*w`, so batching is `n = N*h*w`
plus a per-env scalar lookup (the 2026-08-27 survey). Scratch is keyed
`(h, w)` in the older cores.

## 3. Measured (issue #125, 2026-10-29, this machine)

Headless 4 v 4 + 4 zombies, 24 ticks per second, 60 s of sim; "real-time x"
is sim seconds per wall second.

| level (grid, cells) | CPU | per-call CUDA | resident |
|---|---|---|---|
| small (60 x 40, 2 400) | 5.9x | 2.6x | 4.8x |
| medium (100 x 70, 7 000) | 2.0x | 1.8x | 3.0x |
| large (160 x 110, 17 600) | 0.7x | 1.0x | 1.3x |

- Physics is 85 to 90 % of a tick and linear in area on the CPU
  (2.5 to 2.85 us per cell per tick); the bots cost 1 / 2 / 8.5 ms per tick
  and spike to 280 ms on Python A* (partly fixed by #143).
- The resident path at N = 1 is only 1.5x the CPU on medium: it is bound by
  the brackets, the transfers and the launches, not by the device's compute.
- The second CUDA match in one process runs 10 to 20 % slower (not
  investigated). The tick digest is identical on CPU, per-call and resident.
- S8a Path B's own bench (2026-07-20): resident over per-call 1.75x at a 128
  grid, 3.2x at 256, 4.8x at 384; the gain scales with the grid because it
  deletes the substep-multiplied transfer tax.

What this says: at one env the GPU is nearly idle; the arc's gain comes from
N envs per launch and from deleting the host round trip, not from faster
kernels.

## 4. Known gaps and debts that the arc inherits

- **#71** the CUDA combustion twin differs from the CPU solver on a burning
  scene (pre-#54 twin, no gas-energy ledger). Erik's ruling: fixed with #77,
  before any all-GPU run that claims determinism.
- **S8c items 2 and 3** deferred (render CUDA-GL interop, recorder kernels):
  `docs/s8c_items_2_3_deferred_2026-07-21.md`. The recorder is not the replay
  path (replay is re-simulation, #126), so item 3 matters less now.
- **#24** CUDA graphs: spike proven (graph replay bit-identical, stable over
  100 replays; capture needs a non-blocking stream + thread-local capture
  mode), nothing shipped. `docs/cuda_s8_graph_spike_salvage_2026-08-22.md`.
- **#25** resident sensor gather: the `sensor_accessor` channel order and
  int32 contract are frozen; the `(n_sites x n_channels)` kernel is unbuilt.
- `is_ambient`, `thermal_solid`, `solid` are not static (breaches, burn-outs,
  edits) and ride the per-tick upload; `dyn_permeability` / `dyn_wave_absorb`
  / `dyn_light_atten_q` are restamped every tick for body shielding — a
  batched tick must keep every one of these current on the device.
- `set_residency(False)` never clears `residency_on()` (#77 pointer).
- The build scripts are Windows `.bat`; no Linux build has ever been made
  (Berzelius needs one; `docs/rl_env_arc_proposal_2026-08-27.md` §D).
- Headless tooling never imports `renderer/`; `tools/run_match.py` is THE
  headless loop and `tools/bench_bot_matches.py --timing` its instrument.

## 5. Instruments and gates that exist

- `tools/bench_bot_matches.py --timing [--cuda --cuda-build DIR [--resident]]`
  (ctrl vs step per tick, repeats, idle / horde / bots).
- `tools/e2e_drive.py --profile` (the window's frame profiler; not headless).
- `tests/cuda_s8a_check.py`: the resident A/B gate (PART 1a space map, 1b
  ambient map, 1c device MG-build parity) + the payoff bench (PART 2).
- `tests/cuda_harness.py::run_cuda_script`: every GPU test runs in a `data`
  subprocess; the CUDA `.pyd` is never imported into pytest.
- `tests/field_ab_harness.py` (lockstep A/B), `tests/field_digest.py` + spec
  (the determinism gates), `tests/_xarch_perfield_digest.py` (the golden).
- `tests/_bench_cuda_payoff.py` (untracked, June): CPU vs per-call GPU per
  solver at 32 to 512 grids.

## 6. Prior docs (read in this order)

1. `docs/architecture/engine/02_state_and_ownership.md` — canon: GameMap stays
   the interface, fields migrate behind it; "GPU owns fields, CPU owns actors"
   (this arc revisits the second half).
2. `docs/rl_env_arc_proposal_2026-08-27.md` — the diagnosis (authority), §A
   habits (adopted), B1–B6 (batch cores → finish residency → device unit store
   → reset / obs / reward / done + zero-copy torch export → first agent task),
   §D Berzelius checklist, the Systems section with draft rules.
3. `docs/cuda_s8a_residency_spec.md` (+ `_2026-07-19`) and
   `docs/cuda_s8a_path_a_impl_2026-07-21.md` — the resident plumbing as
   built; §0 of the Path A doc is the load-bearing observation (the EOS
   pre-stage reductions consume tick-entry state, so they stay on the host
   for free — which a batched, device-authoritative tick must redo).
4. `docs/tick_budget_2026-10-29.md` — the numbers above.
5. `docs/rl_ready_scope_2026-10-29.md` — the v1 match and the five things
   built so both agent shapes fit (control seam, fixed roster slots, one
   sensor accessor, credited events, simultaneous actions).
6. `docs/research/rl_litsearch_2026-07-20.md` — the RL training literature
   (multi-agent, hierarchical); for the agent design, not this arc.
7. `docs/cuda_migration_plan.md` — the integer-atomic deterministic-reduction
   contract (integer `atomicAdd` is order-free), the residency table.

## 7. What the design must settle

1. **The boundary.** Which slots of §2.1 move to the device, in what order,
   and what the host keeps (per-episode and per-decision-batch work only, by
   the stated end state). The benchmark orders them by cost.
2. **The unit store.** SoA `(N, max_units, column)`, schema-driven from the
   entity registry, Q16 where synced, with a `valid` mask; Python `Unit`
   objects as views or absent in training mode (B3). Which columns are
   synced (digested) and which are training-only.
3. **The EOS pre-stage.** Its reductions (`t_max_abs`, `max_rad`, `n_total`,
   `max_du` → `n_sub`) run on the host today over tick-entry state. Batched
   and device-authoritative, they become per-env device reductions; the
   substep count `n_sub` is per env, so the substep loop must be masked per
   env or the loop count must be a per-env maximum with masked work.
4. **Batch invariance as a gate.** Env i of N equals env i alone, bit for bit,
   on every synced field and every book; integer atomics make it feasible.
5. **Reset.** A masked, device-side reset from a template bank of pre-baked
   levels (B4) — levelgen stays offline.
6. **The agent seam.** An obs gather that reproduces the sensor accessor's
   channels, per-unit crops and a team union (both agent shapes), an action
   scatter from the intent facade's verbs, rewards and terminals from the
   credited events; zero-copy export to torch via `__cuda_array_interface__`.
7. **The snapshot system.** Tag + own checkout + own build, refused on
   mismatch; the xfail-with-issue rule for drifted twins.
8. **Determinism gaps to close first.** #71, and the CUDA second-run slowdown.
9. **Linux build and Berzelius.** CMake without the `.bat` wrappers, WSL2 as
   the proving ground, Apptainer or a module stack on the cluster.
10. **CUDA graphs (#24)** — after batching, when launch overhead is measured.

## 8. Next steps in this arc

1. Benchmark (this week): the resident tick's per-stage cost, transfer bytes
   and sync count; a stacked-grid scaling probe (N copies of a ship in one
   grid) for the cost curve vs N; the per-slot cost of the gameplay in a
   headless match. Doc: `docs/residency_benchmark_2026-11-*.md`.
2. Targeted literature (one agent, one day): Madrona, CuLE, Isaac Gym / Lab,
   Brax, PufferLib / EnvPool, CUDA graphs and persistent kernels, GPU flow
   fields. Doc: `docs/research/batched_gpu_sims_litsearch_2026-11-*.md`.
3. Design doc with a Systems section, then one adversarial critic per lens,
   one at a time, each verdict to disk.
4. Patches with gates, one implementation agent at a time.
