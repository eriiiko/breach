# CUDA-S8a — GPU residency, post-EOS rewrite (2026-07-19)

**Status:** DRAFT for Erik's review (physics close-out, priority ledger stack #1).
**Supersedes:** `cuda_s8a_residency_spec.md` (pre-EOS: it specs the retired two-field
`atmosphere`+`wave_p` IMEX tick and the S1–S7 solver chain; kept for the record, banner added).
**Sequencing (Erik, 2026-07-19):** the **boundary-conditions spec/patch lands FIRST**
(AMBIENT border-ring, `notes_2026-07-17_topics_backlog.md` Topic 4) so residency freezes the
final kernel content. If ordering ever flips, launch-core extraction keeps kernel bodies
editable — BC-after is allowed, just one more careful pass.

**Goal (unchanged from the original S8a intent):** keep the synced physics fields
**GPU-resident across the whole tick** — persistent device buffers, launch-only kernel cores,
no per-call malloc/H2D/D2H — and copy the synced set down to the CPU mirror **once per tick**
(the locked Q4 decision) so combat/recorder/render read the mirror exactly as today. This
deletes the per-call transfer tax (~0.6–0.95 ms/call, measured) that makes the per-call GPU
path lose to the CPU at 160²×1env (`eos_p6_gpu_alignment_review.md` §0). CUDA graphs are
**S8b**; render CUDA-GL interop + recorder kernels + the `cast_fire_heat` device port are
**S8c**; batched many-env training builds on all of it.

**Bit-identity is the gate:** the resident path must produce byte-for-byte the same synced
trajectory as the CPU reference (which the per-call GPU path already matches, kernel by
kernel). No physics change anywhere in this patch.

**Proven foundation (still valid):** the S8 spike (`cpp/src/cuda_spike.*`, `tests/_spike_s8.py`)
proved a kernel given a CuPy array's `int(arr.data.ptr)` (as `uintptr_t` through pybind)
mutates CuPy-owned device memory in place — CuPy + the breach `.pyd` share one CUDA primary
context. Keep fields contiguous; `.data.ptr` is stable unless the array is reassigned.
Reuse the `reinterpret_cast<int32_t*>(dev_ptr)` pattern.

---

## 1. What changed since the pre-EOS spec (why this is a rewrite)

The EOS refactor (P1–P7, closed 2026-07-11) replaced the tick this spec's predecessor was
written against:

- **One derived pressure.** `P = C·N_total·T`, Kwatra semi-implicit Helmholtz/MG solve.
  `atmosphere` IS the materialized `P`; `wave_p` is repurposed as the **P_prev** buffer;
  `wave_v`/`wave_source` are **retired** from the physics path (still allocated; no solver
  reads them). The old spec's wave/diffuse substep chain and float bridges do not exist.
- **Conserved species.** `gas` is a dense `(N_GASES, h, w)` int32 array: bulk `O2` +
  `INERT_N2` (donor-cell conservative flux inside the EOS step) + 5 trace planes advected
  once per tick on the final wind, each with `decay → inert_N2`.
- **Combustion** is a new pass (real-O₂ stoichiometric burn) between the EOS step and the tail.
- **The EOS step is already internally device-chained** (`cuda_eos_step.cu` orchestrates
  sl-advection → bulk flux → MG solve → kick/compression on device). Residency extends that
  pattern to the whole tick — it does not invent it.
- **`step_ripple` moved into C++ `step_tail`** (float, render-only, no `.cu` kernel).
- The smoke sink-direction BFS field is deleted (native venting); `obstacles` passed to
  `run_substeps` is ignored (solver reads `solid`).

## 2. The resident set (device buffer inventory)

**Persistent device fields = CuPy arrays, Python-owned**, allocated once, C-contiguous,
written ONLY in place (`__setattr__` guard, §7). From the 2026-07-19 code inventory
(`gamemap.py`), the set the resident block reads/writes:

| group | fields |
|---|---|
| pressure | `atmosphere` (=P), `wave_p` (=P_prev), `wind_x`, `wind_y` |
| species | `gas` `(N_GASES,h,w)` (`smoke` stays a host-side view of the mirror) |
| thermal | `temperature`, `heat` |
| fire | `fire`, `wall_hp` |
| water | `water_depth`, `flow_vx`, `flow_vy`, `floor_height` |
| masks/coeffs | `solid`, `obstacles`, `is_vacuum`, `flammable`, `dyn_permeability`, `dyn_wave_absorb`, `conductivity`, `heat_inv_shift`, `face_shift` `(h,w,4)` |

**Stays host-only:** `ripple`/`ripple_v` (float32, render-only — §3 step 5), `light_atten`/
`dyn_light_atten`/`heat_atten` (raycaster inputs; `cast_fire_heat` stays host in the slice),
`wave_v`/`wave_source` (retired), water scalars/`water_sources` (host control flow).

**Persistent scratch = C++-owned:** every per-solver temporary the per-call wrappers
currently malloc — including the **MG `levels_` hierarchy** (the alignment review §7 names
this explicitly as the S8a persistent-scratch case), water's flux/scale temporaries, trace
advection scratch, combustion scratch, reduction accumulators. Allocate lazily on first
resident call keyed by `(h,w)`, reuse every tick, never per-call malloc in the resident path.

## 3. The tick under residency (the slice boundary — Erik-approved 2026-07-19)

`cast_fire_heat` stays a host Python loop (S8c ports it); ripple stays host float. The
tax-heavy field→field block goes resident:

1. **Host pre-physics (unchanged):** FieldEdits apply, masks restamp, structural edits from
   last tick's combat have already mutated the mirror; `cast_fire_heat` runs on the mirror
   and writes `heat` (+ render light/glow).
2. **H2D once (batched):** upload the resident set from the mirror into the persistent
   device buffers. **Rung 1 uploads the full set** — one batched transfer, correctness
   first; this automatically covers every host writer (FieldEdits, `heat`, structural
   edits, `apply_temperature_ignition`'s `fire` writes, mask restamps). Rung 2 (§5b)
   narrows it.
3. **Resident block — `step_resident(...)`, launch-only cores, zero mid-tick transfers:**
   water (substeps + W5 flash-boil + W3 displacement/seal) → EOS step (advection substeps,
   one MG solve, kick/compression, trace advection ×5 + decay) → combustion → fire →
   temperature. Same order, same substep counts, same arithmetic as
   `physics_runner.step` / `physics_engine.cpp` today. Substep counts stay host-computed
   (integer ceil_div; the couple of scalars that need a device value are a tiny pinned D2H
   — the alignment review's accepted sync point).
4. **D2H once (batched):** full synced set → pinned numpy mirrors. Combat, recorder,
   renderer, `find_burst_walls`, dosing/env-damage read the mirror unchanged (Q4 baseline).
5. **Ripple on host, post-mirror:** run `step_ripple` on the freshly downloaded mirror.
   Its inputs (`water_depth`, `atmosphere`, `wave_p`) are not written by fire/temperature,
   so computing it after the mirror lands is value-identical to its canonical `step_tail`
   slot. **Build step must CONFIRM that input-invariance in code** (and the A/B gate proves
   it end-to-end); fallback if it ever fails: split the tail and D2H the three ripple
   inputs mid-tick. `heat.fill(0)` stays where it is (end of `Simulation.step`).

**Determinism argument:** host ops (`cast_fire_heat`, ripple, FieldEdits, structural edits)
run the same CPU code in both paths; the resident block runs the same kernels as the
per-call GPU path (already == CPU per the `cuda_*_check` gates). Bit-identity therefore
reduces to the boundary sequencing — every host op sees the right field values, the
resident block sees every host write. The 30-tick A/B gate is the arbiter.

## 4. Build steps

- **STEP A — confirm the inventory.** Re-verify against `physics_runner.py` /
  `physics_engine.cpp` at build time: the exact field set each kernel touches, every
  host-side op between C++ solver calls (post-EOS there should be few — the old float
  bridges are gone), the ripple input-invariance claim (§3.5), and whether unit footprints
  still stamp `obstacles` (3b note says no — confirm). Report scope early; if the glue list
  is much more than a handful, stop and report.
- **STEP B — launch-only cores.** For each solver entry (`water_step`, the four EOS
  sub-kernels — already chained device-side, so mostly strip the per-call wrapper's
  malloc/copy — trace/smoke, combustion, fire, temperature): factor a non-anonymous
  `*_launch_resident(...)` taking device pointers + persistent scratch + scalars, declared
  in a shared internal header. Per-call `*_step` entries keep working (the live fallback +
  the existing gates must still pass).
- **STEP C — glue kernels** for whatever STEP A finds between solver calls (expected: near
  zero post-EOS; anything found becomes a device kernel, bit-identical to the host op).
- **STEP D — `step_resident(...)`** orchestrator: owns the persistent scratch, runs §3.3 in
  order on the passed-in field pointers.
- **STEP E — GameMap residency mode** (`gamemap.py`): synced fields as CuPy arrays,
  `device_ptrs()` accessor, `to_host()` batched D2H into pinned mirrors, `from_host()`
  batched H2D, and a `__setattr__` guard making reassignment of a resident field a hard
  error (in-place `[:]` only — stale-pointer protection). CPU path and numpy fields
  untouched when residency is OFF.
- **STEP F — dispatch + flag.** `set_residency(True)` process-global setter in the existing
  `bp.set_*_backend` family; `tools/run_on_cuda.py` / `--cuda` gains a `--resident` opt-in.
  Default OFF — game + suite unchanged.

## 5. Normative contracts this spec carries (cross-arc, binding)

### 5a. The sensor-gather contract (Arc B is gated on this)

Entity design doc §7 / canon `engine/16` §8: **sensor sample sites are static per level.**
At level load, build a device-side site index `(site → tile, field-channel)`. Each tick the
resident path runs **one compact gather kernel** writing a `(n_sites × n_channels)` int32
buffer, downloaded in **one small D2H** alongside (later: instead of) the mirror.
- Arc B sensors MUST read the gather buffer (resident path) or the host mirror (CPU path)
  through one accessor — deterministic site order (id order), Q16.16 values, no dequantize
  in the sim path.
- **No new consumer may be built that depends on per-tick full-field streaming.** The Q4
  full mirror is a transitional service for today's readers; the end-state (S8c, batched
  training) drops it, and the gather buffer is what survives.

### 5b. The structural dirty-set rider (a5 doc §9, binding at Rung 2)

**Rung 1** (full batched H2D per tick, §3.2) covers structural edits implicitly — land
correctness first. **Rung 2 (S8a.2, same branch or immediate follow-up)** replaces the full
upload with host-produced deltas only; from that moment this rider is binding: after
`destroy_wall` / `seal_tiles` / `unseal_tiles`, the touched set must reach the device
before the next kernel read. The verified 2026-07-19 list:
- `on_tile_changed` caches: `flammable`, `wall_hp`, `conductivity`, `heat_inv_shift`,
  `face_shift` (this tile's 4 faces **+ the facing entry of each 4-neighbour**),
  `permeability`→`dyn_permeability`, `solid` (raycaster-side `light_atten`/`heat_atten`
  stay host — the raycaster is host in the slice);
- plus per primitive: `atmosphere`, `wave_p`, `gas` (**all planes, including neighbour
  receiver/donor tiles** — seal evacuates into neighbours, unseal withdraws from them),
  `wind_x/y`, `flow_vx/vy`, `is_vacuum`, `temperature` (seal writes solid-neighbour mean),
  `material`-derived masks. `ripple`/`ripple_v` are host-side — no upload needed.
- The dirty region is the edited tiles **∪ their 4-neighbourhoods.** Sparse scatter H2D or
  device-side execution — implementer's choice; the contract is only that the list is
  honored. Rung 2's gate: the same A/B trajectory bit-identity with scripted
  seal/unseal/destroy events mid-run.

### 5c. Boundary-conditions seam (BC-first)

The AMBIENT border-ring patch (MG pin P=P_amb, reservoir reset in bulk transport, smoke
absorb, optional sponge band) lands **before** this build; STEP B then extracts launch
cores from the post-BC kernels. If a sponge band adds scratch, it joins the persistent
scratch inventory. Space-map goldens are untouched by BC by construction (no AMBIENT tiles
in existing levels).

## 6. The gate

- `tests/cuda_s8a_check.py` + pytest wrapper `tests/test_cuda_s8a_residency.py`, following
  the `cuda_eos_step_check.py` pattern (GPU subprocess, A/B flags-off vs flags-on):
  - **PART 1 — bit-identity:** ≥30-tick full-engine A/B on the canonical seeded scenario
    (events included: detonations, water, fire, a scripted structural edit), residency ON
    vs the CPU path; per-tick trajectory of ALL synced fields byte-identical (tol 0),
    including host-path `heat`/`ripple`/`ripple_v`; reproduce the current committed
    full-engine golden (read it from the repo at build time — do NOT re-baseline).
  - **PART 2 — the payoff:** benchmark N ticks CPU vs per-call GPU vs resident at ≥2 grid
    sizes; the per-call transfer tax must be gone (resident clearly beats per-call GPU;
    beats CPU where the alignment-review cost model predicts). Print `S8A_RESULT: PASS/FAIL`.
- Full suite green: `pytest tests -q` (conda env `data` — per-machine specifics live in
  the dev-setup docs, not here).

## 7. Discipline

- Bit-identical, tol 0, no physics change, no re-baseline. CPU path + per-call GPU path
  remain the live defaults; residency is opt-in behind the flag until Erik promotes it.
- In-place-only resident fields (the `__setattr__` guard enforces it).
- If STEP A's glue inventory or the boundary sequencing balloons, STOP and report with the
  inventory — a partial-but-green slice beats a sprawling half-state. Never commit red.
- Build: `cpp/build_cuda*.bat` (per-machine variant); run/tests via the conda `data` env.

## 8. Out of scope (deliberately)

S8b CUDA graphs · S8c render interop / recorder kernels / `cast_fire_heat` device port /
dropping the Q4 mirror · batched many-env training · any kernel content change (BC lands
before; blast-threshold material column is a separate post-residency rider) · sensor
*implementation* (Arc B builds it; §5a only fixes the interface).
