# ML & Training

**Depends on:** [State & Ownership](../engine/02_state_and_ownership.md) (the `Simulation` facade), [Material System](../engine/03_material_system.md), [Atmosphere & Pressure](../engine/04_atmosphere_and_pressure.md), [Temperature & Fire](../engine/06_temperature_and_fire.md), [Ray Engine](../engine/08_ray_engine.md), [Turn & Control](../mechanics/04_turn_and_control.md), [Units & Entities](../mechanics/01_units_and_entities.md), [Combat & Weapons](../mechanics/03_combat_and_weapons.md).

Breach is built so that a neural-network agent can learn to play it without a
second, divergent implementation of the rules. The same headless,
deterministic `Simulation` that drives human play is the reinforcement-learning
environment: the renderer is a read-only consumer bolted on top, never a
prerequisite. This chapter specifies how that environment is shaped, how state
becomes a tensor, how rollouts stay deterministic, and what an agent does and
does not yet have to work with.

The guiding constraint is stated once and obeyed everywhere: **training and
gameplay run identical logic.** There is no "AI mode" fork of the rules. A
policy trained against `Simulation.step()` is, by construction, trained against
the game the player plays.

---

## 1. The environment IS the Simulation facade

`src/simulation/simulation.py` defines `Simulation`, the single place world
state is mutated. Human play (`main.py`) and a future `train.py` talk to it
through the same small API. That API is deliberately Gymnasium-shaped:

| Method | Role | RL contract |
|--------|------|-------------|
| `reset(seed=None)` | Re-initialise from the same level | `env.reset()` |
| `apply_action(unit_id, order)` | Queue an order; returns accepted/rejected | action application |
| `step()` | Advance exactly one tick | `env.step()` core |
| `get_state() -> SimState` | Read-only snapshot of world + entities | observation source |
| `get_reward(unit_id) -> float` | Per-agent scalar reward | reward signal |
| `is_terminal() -> bool` | Episode boundary | `done` flag |
| `get_legal_actions(unit_id) -> list` | Valid orders for a unit now | action masking |
| `rng` | The one `numpy.random.Generator` | determinism root |

The boundary is the facade, not the renderer and not the C++ physics module.
An agent never imports pyray, never opens a window, and never depends on a frame
having been drawn. It constructs a `Simulation`, optionally passing the compiled
`breach_physics` module and `enable_recorder=False` (the recorder's ~80 MB ring
buffer is pure waste in training), then loops `apply_action`/`step`/`get_state`.

Why a facade rather than a thin C++ `BoardingEnv`: the *rules* — orders, AP,
combat, zombie conversion, phase boundaries — live in Python; only the field
physics is C++. Wrapping the rules behind one Python class keeps the
environment authored where the game logic is authored, and lets the same object
serve the human UI. The early architecture sketch imagined a C++ `sim.BoardingEnv`
returning zero-copy NumPy dicts; the shipped reality is the Python `Simulation`
returning a `SimState` that *references* the live NumPy grids the C++ solvers
already bind to — the zero-copy benefit without a second environment class.

### Pause is invisible to training

`set_paused(True)` makes `step()` a no-op so the human can plan. The auto-pause
at the end of each round exists for the planning UI. **Training never calls
`set_paused`**, so it never sees a halt: it drives a continuous stream of ticks,
reading `is_terminal()` to find episode ends. Pause is strictly a human
convenience layered on top of the same tick loop, not a branch in the rules.

---

## 2. Observation: what the agent sees

### 2.1 Stacked feature planes ("what you see is what you get")

World state is arrays plus a material table — no tile-objects — which maps
directly onto the AlphaStar-style stacked-feature-plane encoding. Each field is
one or more channels of an `[C, H, W]` tensor, all sharing the grid's `(H, W)`:

| Source field | dtype | Channels | Meaning |
|--------------|-------|----------|---------|
| `gmap.material` | int8 | 1 (or one-hot) | material id per tile |
| `gmap.is_wall` | bool | 1 | static occlusion |
| `gmap.obstacles` | bool | 1 | dynamic occlusion (incl. units) |
| `gmap.wall_hp` | float32 | 1 | structural integrity |
| `gmap.atmosphere` | float32 | 1 | pressure / vacuum |
| `gmap.smoke` | float32 | 1 | smoke density |
| `gmap.fire` | float32 | 1 | fire intensity |
| `gmap.heat` | int32 | 1 | deposited thermal energy |
| unit footprints | — | per-team | rasterised positions/teams |

Units are not arrays today; the encoder rasterises them into planes (one per
team, plus per-unit scalar planes such as HP) at encode time. Because the world
is already arrays, building the tensor is a stack-and-cast, not a traversal of
an object graph — the data-driven world model was chosen partly so this encoding
stays a near-trivial projection.

A second, more radical option is on the table and recorded here as canon intent:
feed the agent **the same view the player gets** — the physics grid *as seen*, or
even the rendered image. If smoke makes a tile hard for the human to read, it
should be hard for the agent too. This naturally extends to **per-species views**:
a unit with infravision observes the `heat` channel where a human sees darkness,
so different creatures literally receive different observation tensors derived
from the same world. This is a design direction, not a decision; the feature-plane
stack above is the concrete v1.

### 2.2 Render-only channels are skipped in training

The ray engine ships with per-channel `light_atten`, an RGB `light_rgb` buffer,
`light_dir`, `smoke_glow`, and an ACES tone-map. Most of that is **render-only**.
The raycaster is physics — it runs inside the deterministic sim step and produces
`heat` (which feeds wall thermal failure and unit heat damage) — but the RGB
colour and glow buffers exist only to be drawn.

Therefore headless training computes only what affects the rules: the `heat`
channel, and a scalar **light-intensity** field for stealth and line-of-sight
(a dark tile is harder to see into). It **skips** the RGB lighting and glow
passes and the tone-map entirely. Light-of-sight uses the same ray-march
primitive behind a `has_los(a, b)` query. This keeps a rollout doing the minimum
ray work the rules require, and keeps the heavy perceptual shading out of the
training inner loop.

---

## 3. Actions

An action is an `Order` queued via `apply_action(unit_id, order)`, which returns
`True` if accepted and `False` on a rules violation (no AP, empty inventory,
blocked tile, or a zombie attempting to take orders). Order types are movement
(attack/cover/sprint move), fire, grenade, and door-explosive; placement spends
AP and decrements inventory, and `undo_last_order` refunds both.

`get_legal_actions(unit_id)` is the intended action-masking hook. **It is a stub
today** — it returns `[]`. A full enumeration is large: movement spans every
reachable tile, fire spans every visible enemy in range, grenades span every
tile within throw range, each crossed with mode and phase. Until `train.py`
needs it, the accepted/rejected return value of `apply_action` doubles as a
per-action legality probe. The first real masking implementation will enumerate
reachable tiles (the A* used for path preview already exists), visible enemies,
and in-range throw targets.

Action timing — how often an agent samples the world and issues orders relative
to the 12-tick/second sim — is an open experiment for when training begins, not
a fixed part of the contract.

---

## 4. Reward and termination

`is_terminal()` is implemented and authoritative. An episode ends when the round
completes (`tick >= ticks_per_round`), or one side is wiped out (all marines
dead, or all zombies dead), or the world is degenerate (no units at all).

`get_reward(unit_id)` is a placeholder returning `0.0`. The API commits to the
hook — per-agent, keyed by unit id — but reward shaping is intentionally
deferred to the experiment that needs it. Subclassing or wrapping `Simulation`
to define a reward is the expected path, so reward design never forces a change
to the core engine. Stable unit ids matter here: `add_unit` assigns a stable,
monotonically increasing id, so reward and any future per-unit training-data
pipeline can track an agent across a rollout without ambiguity.

---

## 5. Determinism

Reproducible rollouts require a single controlled source of randomness. A lone
`numpy.random.Generator` lives on `sim.rng`, created in `_reset_internal` from
the constructor/`reset` seed. `reset(seed)` re-runs the same level identically;
the same seed plus the same action sequence yields the same trajectory. `reset`
also reuses the C++ solvers and re-allocates fresh grids without reconstructing
the object, which is what vectorised environments need to recycle workers cheaply.

The non-determinism sites are plumbed to take `self.rng` explicitly rather than
calling the process-global `numpy.random`/`random`:

- **combat** — bullet-cone offsets in firing,
- **explosion smoke** — per-tile placement noise (`add_explosion_smoke`),
- **door explosives** — `process_door_explosives` receives `self.rng`.

The raycaster's fire jitter is the one deliberate exception: fire light sources
run with zero jitter (smoke advection supplies the flicker), and the C++
raycaster keeps its internal seeding, to be revisited when training starts.

One subtlety reaches beyond seeding into number representation. Values that
**cross a discrete threshold into sim state** are moving to fixed-point integers
so they are deterministic *across machines*, not merely within one. The `heat`
channel is already `int32` for exactly this reason: many rays deposit into one
cell, and integer `atomicAdd` is order-independent, so the deposit is
machine-independent. The rule is: **fixed-point where a value crosses a
threshold into sim state; float where it is continuous and perceptual.** Light
colour stays float (no downstream threshold); heat is integer. This is not a
one-way door — heat diffusion can fall back to float while keeping the deposit
integer if fixed-point harmonic-mean conductivity proves painful. The principle
exists so distributed self-play does not silently diverge between hosts.

A determinism test is part of the plan: construct two seeded `Simulation`s, step
each 100 times, and assert identical state — the canary that protects every later
rollout.

---

## 6. Snapshot, save/load, and tree search

`get_state()` returns a `SimState` holding *references* into the live arrays —
cheap, and safe for the renderer to read each frame, but **not** a durable
snapshot. An agent that needs a permanent checkpoint (or save/load) must pickle
or deep-copy explicitly. The snapshot contract is forward-compatible: real
save/load will be built on top of `get_state()` without changing the API.

Full **snapshot/restore** — cloning a mid-rollout state and branching from it —
is deferred. It is not blocking for a first training pass but is a prerequisite
for AlphaZero-style tree search (MCTS needs to fork the environment at a node).
It is flagged here so it is not forgotten when search-based training arrives.

---

## 7. Scaling: parallel self-play

Headless, deterministic, render-free rollouts are the foundation for scale.
Because the renderer is optional and each `Simulation` is self-contained,
many environments run in parallel with no shared mutable state. The longer-term
direction, drawn from the GPU plan, is AlphaZero-style **multi-game simulation**:
run many independent games at once on the GPU (each game is independent — ideal
for the device), move **state encoding** into a CUDA kernel (one thread per
channel/row/col rather than ~0.85 s of NumPy), and **batch A*** so dozens of
units path-plan in the time of one. None of this changes the facade; it changes
where the same computation runs.

---

## Implementation status

**Built and shipping (the environment skeleton):**

- The `Simulation` facade with `apply_action`, `step`, `get_state`, `reset(seed)`,
  `is_terminal()`, `get_reward()` (stub returning `0.0`), `get_legal_actions()`
  (stub returning `[]`), `undo_last_order`, and a single `self.rng`
  (`numpy.random.Generator`) created from the seed in `_reset_internal`.
- `enable_recorder=False` path so training skips the ring buffer.
- RNG plumbed through combat, `add_explosion_smoke`, and `process_door_explosives`.
- World state as arrays + material table (`gmap.material`, `is_wall`, `obstacles`,
  `wall_hp`, `atmosphere`, `smoke`, `fire`, `heat`) — i.e. the raw material for a
  feature-plane encoder already exists in directly-stackable form.
- The ray engine in the sim step producing the rules-affecting `heat` channel,
  plus the render-only `light_rgb`/`light_dir`/`smoke_glow` buffers and ACES
  tone-map; `heat` is `int32` (fixed-point deposit) for cross-machine determinism.
- `is_terminal()` is fully implemented (round end, side wipe-out, degenerate).

**Designed but not built:**

- **No neural network and no training loop.** There is no `train.py`, no
  Gymnasium wrapper, no policy, no agent — only the environment it would target.
- **State encoding is unbuilt.** No code constructs the `[C, H, W]` feature-plane
  tensor or rasterises units into planes; the channel table in §2.1 is a design,
  not a function. The "rendered-image / per-species-view" observation idea is a
  direction only.
- **`get_legal_actions` is a stub** (`[]`); real action enumeration/masking is
  unwritten.
- **`get_reward` is a stub** (`0.0`); no reward shaping exists.
- **Snapshot/restore and save/load are deferred**; `get_state()` returns live
  references, not a durable, forkable snapshot, so tree-search-style branching is
  not yet possible.
- **The light-intensity scalar for stealth/LoS in headless mode** and the
  `has_los(a, b)` interface are designed in the ray reconciliation but not yet a
  separated headless path; today the raycaster computes the render buffers too.
- **Cross-machine fixed-point** is realised only for the `heat` *deposit*;
  fixed-point heat *diffusion* and any other sim-threshold channels are unconfirmed.
- **GPU multi-game self-play, CUDA state-encoding, and batch A*** are plan-stage
  (and the source note frames the GPU training section as cross-project), with no
  CUDA build path wired for training.

**Gaps / risks to watch:**

- The accepted/rejected return of `apply_action` is the *only* legality signal
  until `get_legal_actions` is real; an agent currently cannot mask actions
  without duplicating the validation rules.
- The determinism test (two seeded sims, 100 steps, assert identical) is specified
  but must actually exist and run in CI before any rollout is trusted — C++ physics
  determinism is assumed, not yet asserted here.
- The raycaster fire-jitter exception means the render path is not bit-identical
  to a strict headless path; this must be reconciled before training depends on
  the ray channels.
- Units live as Python objects, not arrays, so the encoder must rasterise them
  every observation — a per-step cost the GPU plan intends to remove but which is
  unaddressed today.

**Net status: design-only.** The deterministic, headless environment boundary is
real and usable; everything above it — encoding, actions/masking, reward, the
agent, and parallel training infrastructure — is designed but not implemented.
