# RL-env arc proposal — 2026-08-27

Capture doc from a brainstorm (Erik + Claude, Linköping, phone). Read-only
survey of `eriiiko/breach` at HEAD 2026-08-27 plus discussion. Proposes an
arc that turns the resident CUDA sim into a batched, device-authoritative RL
environment for Berzelius. **Ordering ruling from Erik: physics completion
stays #1.** This doc therefore splits into (A) cheap habits to keep *during*
physics completion so the later retrofit is mechanical, and (B) the arc
itself, to be sequenced against Roadmap #46 after physics.

> **Stitched into the repo 2026-08-27** (authored on claude.ai with partial
> repo knowledge; survey claims spot-verified against the tree — the five
> resident TUs' `*_launch_resident`, the `physics_runner.py` mirror
> docstring, `docs/s8c_items_2_3_deferred_2026-07-21.md`, and
> `docs/stamp_units_to_cpp_plan.md` all check out). Issue home: **#29**
> (RL substrate umbrella — B1–B6 are its concrete plan); also touches
> #24 (S8b CUDA graphs), #25 (resident sensor-gather → ancestor of B4's
> obs kernel), #28 (TM communication contract), #33/#34 (animation, §C).
> §A habits ruling (Erik, 2026-08-27): adopted now as a lean CLAUDE.md row —
> the deliberate early exception to "rules enter at implementation", because
> §A constrains code written during physics completion.

Companion topics captured at the end: animation strategy (humanoid rig,
MDM + PD-tracked ragdoll) and Berzelius readiness, since they hang off the
same architecture.

---

## 0. Survey findings (as-built, HEAD 2026-08-27)

Scored against the six properties a vectorized GPU RL env needs:

| # | Property | Status | Evidence |
|---|---|---|---|
| 1 | Env batch dimension in every buffer | **NO** | Every field is `(h, w)`; every `*_launch_resident(...)` takes `int h, int w`; persistent scratch keyed by `(h, w)`. One world per process. |
| 2 | No per-world host state in the trainable path | **NO** | `Simulation.units` is a Python list of objects; combat, FieldEdit flush, `stamp_units`, coupling rows, recorder run on the host. `_step_resident` docstring: *"the numpy fields are the authoritative mirror throughout"* — H2D of resident-loop inputs at step 2, batched D2H at step 6, every tick. Combustion + tail still bracketed (S8c). |
| 3 | Masked reset kernel | **NO** | `Simulation.reset` is host-side, whole-world. |
| 4 | Determinism given seed | **YES, strong** | Q16.16 integer sim path, `/fp:strict`, digest/golden gates, A/B lockstep harness. Better than most published RL sims. |
| 5 | Observation kernel | **NO** | `get_state()` returns host `SimState`. |
| 6 | Zero-copy tensor export (DLPack / `__cuda_array_interface__`) | **PARTIAL** | Resident fields are CuPy-owned → CuPy already exposes `__cuda_array_interface__`, so `torch.as_tensor(cupy_arr, device='cuda')` is zero-copy *today*. Nothing consumes it yet. |

Two things that make the retrofit cheap:

- All CUDA kernels are already **grid-stride loops over `n = h*w`**
  (`for (i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)`).
  Batching over `N` envs = `n = N*h*w` plus a per-env scalar lookup. No
  kernel restructuring.
- The S8a factoring (`*_launch_resident` = launches only, caller owns
  alloc/transfer/sync) is exactly the seam a batched `step_resident` needs.

Diagnosis of the "GPU↔CPU streaming bottleneck" Erik observed: it is not
PCIe bandwidth, it is the **host-authoritative mirror**. The tick must round-
trip because units/combat/edits live in Python. Fixing bandwidth won't help;
inverting authority will.

---

## A. Habits to keep DURING physics completion (cost ≈ 0 now, saves a rewrite later)

These are not new work items. They are constraints on how the remaining
physics kernels get written, so that arc B is a mechanical pass.

1. **Every new `*_launch_resident` takes `(N, h, w)` even if N is always 1.**
   Index as `env = i / (h*w); cell = i % (h*w)`. Per-env scalars (dt, tilt,
   ambient args, …) become `const T* d_scalar` arrays of length N, read once
   per thread by env index. Existing cores get this in arc B step 2; new
   ones should be born with it.
2. **No new host-side tick logic.** If a remaining physics feature needs
   orchestration, put it in `PhysicsEngine` (C++), not `physics_runner.py`.
   CLAUDE.md already says this; it matters more now.
3. **No new fields that only exist on the numpy mirror.** Any new field is
   allocated through the residency path (`GameMap.enable_residency` /
   CuPy-owned) from day one, even if a mirror copy exists for render.
4. **Prefer masked kernels over early-return-on-host.** E.g. "water dormant
   → skip substeps" is a host decision today (`_water_pre_resident`). In a
   batch, some envs are dormant and some aren't: write the skip as a per-env
   `active` mask read inside the kernel, or accept running the kernel on all
   envs. Don't add more host-side gating.
5. **Keep scratch keyed by `(N, h, w)`.** Persistent scratch that is
   lazily allocated keyed by `(h, w)` today should key by `(N, h, w)` when
   touched.
6. **Golden discipline unchanged.** N=1 batched must be bit-identical to
   today's resident path. That is the gate for arc B step 2, so leave the
   digest/golden machinery exactly as it is.

---

## B. The arc (after physics completion)

Goal: `step()` advances N worlds with one launch sequence, no host round-trip;
obs/reward/done are persistent device tensors torch already holds views of;
reset is a masked kernel. Single-instance play keeps working and gets faster.

### B1. Batch the launch cores (N=1 golden)
- Add `N` to every `*_launch_resident` and to `step_resident`.
- Persistent fields: `(h, w)` → `(N, h, w)`; scratch likewise.
- Per-env scalar arrays where scalars can vary; broadcast where they can't.
- Gate: A/B lockstep harness, N=1 vs current resident, tol 0.
- Then N=8 sanity: env k with seed s must equal N=1 with seed s (per-env
  determinism, no cross-env leakage). Add this as a new golden.

### B2. Finish residency (kill the mirror round-trip)
- Port combustion + tail (S8c items 2–3, currently deferred) to resident.
- Move `FieldEdit` flush (6b) to a device queue: host enqueues
  `(env, field, x, y, value, policy)` into a pinned buffer; one kernel
  applies. Same semantics, no D2H.
- Move `stamp_units` to a kernel (there is already a
  `stamp_units_to_cpp_plan.md` — extend it to device).
- Result: a full tick with **zero mid-tick transfers**. The numpy mirror
  becomes a *debug/render view* refreshed on demand, not the truth.
- Gate: goldens unchanged (bit-identity).

### B3. Device unit store (the real design decision — do with the RL collaborator)
- SoA `[N, max_units, ...]` with `valid` mask, Q16 where synced. Fixed
  capacity; no allocation mid-episode.
- Which unit fields go device-side: position, velocity, team, alive, HP,
  `EnvironmentProfile` tolerances, the exposure/coupling state, and whatever
  the reflex tiers act on. Deliberative-tier state (orders, plans) may stay
  host-side for the *game*; for *training* the agents are the reflex tiers.
- The two coupling rows (`exchange.py`) become kernels reading device fields.
- Combat: start with the subset needed for the first agent task (see B5);
  don't port all of combat up front.
- Python `Unit` objects become views over the device store (or are absent
  in headless-training mode).
- Ragdoll hook: the in-sim "torso body" per unit is one more row block here
  (`torso_pos, torso_vel, torso_ang`). PD gains are per-style scalars. This
  is where sim-side impact/shockwave reaction lives.

### B4. Reset / obs / reward / done kernels + export
- `reset(mask, seeds)`: masked re-init from a seed buffer, on device.
  Level instantiation must be device-side or pre-baked: pre-bake a pool of
  seeded levels into a `[L, h, w]` template bank on device; reset copies
  template `seed % L`. (Graph-grammar levels compile offline anyway.)
- `obs`: per-agent local crop of selected fields around the unit + own stats
  → `[N, max_agents, obs_dim]`. Fixed crop radius; field subset from a table.
- `reward`, `done`: small per-env kernels, task-specific, from a table.
- Export: `torch.as_tensor(cupy_array, device='cuda')` (zero-copy via
  `__cuda_array_interface__`, works already) or DLPack. `step()` launches on
  a stream and returns; no `cudaDeviceSynchronize` in the loop — sync via
  stream/event only when torch reads.
- Trainer: use a GPU-tensor-native library (rl_games, RSL-RL, or the Isaac
  Lab vectorized-env pattern). SB3 is CPU-numpy at heart; fine for a first
  smoke test at N=1, wrong for scale.

### B5. First agent task
- Tier: **reactive zombie / fauna reflex**, not marines. Short horizon,
  local obs, simple reward (reach heat/O2/target, avoid fire, survive
  pressure). Stress-tests B3/B4 without deliberative-tier porting.
- Second task: DeepMimic-style clip tracking for the humanoid rig — makes
  animation and agents one Berzelius workload (see C).

### B6. Physics completion inherits the batch
Any physics work landing after B1 is batched for free. Hence the habits in A.

---

## C. Animation strategy (captured; not on the critical path)

Two layers, deliberately separated:

**Layer 1 — offline generative clip library.** Human Motion Diffusion Model
(MDM, Tevet et al., ICLR 2023) or successors: text/action → clip on the
HumanML3D 22-joint skeleton → retarget once to the Breach humanoid rig →
bake. Style classes (heavy armor, normal, old/damaged, zombie) via, cheapest
first: (a) prompt conditioning; (b) procedural modifiers on a base clip
(time-scale, joint-range clamp, lowered CoM, sway); (c) motion style-transfer
models only if (a)+(b) fall short. Weapon-carry variants via MDM joint-space
inpainting (arms only, rest unchanged). Fits the existing ComfyUI
batch-harness mindset.

**Layer 2 — runtime physics reaction (control, not animation).** Active
ragdoll: per-joint PD tracking of the baked clip as reference trajectory,
external forces from the pressure/wind/shockwave fields as disturbances.
Gains per style are the visible character (stiff = armored, sloppy = zombie).
Two resolutions of one design: in-sim = torso/root only (B3 row block, affects
gameplay); presentation = full limb ragdoll (cosmetic). Octopus = same
architecture with the Tsetlin arms as the reference generator instead of a
clip. Upgrade path: RL policy tracking the clips (B5 second task).

Rendering is orthogonal: 2D top-down sim, sprites + 3D models both supported,
possible future z-layers and 3D-rendered maps — the rig is 3D regardless.

**First animation class (normal humanoid), when reached:**
1. MDM locally → walk/run/idle/aim → retarget → confirm pipeline end to end.
2. Play clips on the rig in presentation, no physics.
3. PD-tracked torso in sim + limb ragdoll in presentation, fed by the wind
   field. This is the milestone that looks alive.
4. Variants via prompt + procedural modifiers; inpainting for weapon carry.

---

## D. Berzelius readiness checklist
- Confirm: can we build CUDA extensions (nvcc + pybind11 + CuPy) on the
  login/compute nodes, or do we bring a container (Apptainer)? Check early.
- Headless build with no Raylib / no renderer import path — already a design
  goal; verify the training entrypoint never imports `renderer/`.
- Q16 integer sim means results are bit-reproducible across GPU models — a
  strong property worth stating in any WASP/Berzelius proposal text.
- Per-node budget: A100 40/80 GB. At current grid sizes N in the thousands
  is plausible; measure after B1.
- Civulator is the weaker cluster candidate for now: no physics substrate,
  long horizons, slow per-step. Revisit after Breach B4 exists.

---

## Open questions
- B3 scope: which unit fields are "synced state" (Q16, digested) vs
  training-only float? Digest spec version bump will be needed.
- Level template bank vs on-device grammar instantiation for reset.
- Trainer library choice — with the RL collaborator.
- Whether the deliberative marine tier ever gets a device-side twin, or
  stays host-only forever (implies two "Breach" builds: game and gym).

---

## Systems (rules-lifecycle section — added at stitch, 2026-08-27)

**(a) Existing canonical systems this arc must use** (from CLAUDE.md):

- **Simulation facade** — `step()`/`get_state()`/`apply_action()` stays the
  seam; the batched env wraps it, never bypasses it for single-instance play.
- **PhysicsRunner / PhysicsEngine** — B2's "no new host-side tick logic" is
  this rule sharpened; new orchestration lands in PhysicsEngine (C++).
- **Fixed-point kits** — device unit store's synced columns are Q16 via
  `cuda_fixedpoint_device.cuh`; never re-derive arithmetic.
- **Field digest + GOLDEN_AGGREGATE + A/B lockstep harness** — B1's gate is
  the existing A/B harness at tol 0 (N=1 vs current resident); B3's synced
  unit columns enter the digest spec with a version bump.
- **CUDA harness** — all batched-path tests via `run_cuda_script`
  subprocess, never importing the CUDA .pyd into pytest.
- **FieldEdit** — B2's device edit queue is FieldEdit's flush relocated, not
  a second edit path; host-side enqueue semantics unchanged.
- **Entity system / sensor accessor** — B4 obs kernels are the device twin
  of `sensor_accessor.py`; the registry stays the source of unit schema.
- **Coupling table (`exchange.py`)** — B3 ports the rows to kernels; a
  coupling stays one row, never plumbing.
- **Config** — batch size N, obs crop radius, field subsets: `config.toml`
  rows bound in PhysicsRunner, like solver params.

**(b) New systems this arc creates** (draft rules — enter CLAUDE.md at
implementation, pointing at real code):

- **Batched resident step** — draft rule: every `*_launch_resident` takes
  `(N, h, w)`; per-env scalars are device arrays; N=1 is bit-identical to
  the unbatched golden. *(§A habit adopted early by Erik's 2026-08-27
  ruling — see the CLAUDE.md RL-batch habits row.)*
- **Device unit store** — draft rule: SoA `[N, max_units, …]` with valid
  mask is THE unit truth in training mode; Python `Unit` objects are views;
  no allocation mid-episode.
- **Obs/reward/done kernel tables** — draft rule: a task is table rows
  (field subset, crop radius, reward id), never a bespoke kernel per task.
- **Level template bank** — draft rule: reset copies from the pre-baked
  device `[L, h, w]` bank; level instantiation never runs host-side
  mid-training.
