# Breach — Architecture

The canonical architecture: **one focused design document per system**, grouped into
**engine** (the deterministic world), **mechanics** (game logic on the engine),
**graphics** (the look layer — how the world is shown; see [graphics/README.md](graphics/README.md)),
and **ml**.
Each chapter declares the chapters it **depends on** and ends with an honest
**implementation-status** section. Process artifacts (reviews, patch records, research) live
outside this folder and are archived once their work lands.

---

## 1. Design philosophy

These principles govern every design decision. They are non-negotiable.

- **Systems, not scripts.** Every mechanic is a system interacting with others through shared
  fields. No special-case code for specific scenarios — an `if` for one scenario means you're
  doing it wrong.
- **Emergent complexity from simple rules.** Explosion breaks hull → atmosphere vents → smoke is
  sucked out → fire starves near the breach. Zero scripting — it falls out of systems reading and
  writing shared fields.
- **Full physical simulation.** While something is happening, every in-game second is fully
  simulated (no "simulate a fraction to save compute"). Skipping simulation when *nothing* is
  happening is fine — that's smart, not cheating.
- **Game time = real time.** One in-game second = one real second during execution; slow-motion is
  purely aesthetic (the sim still runs fully).
- **Data-driven parameters.** All tunable values live in `config.toml`, not source — enabling
  hot-reload and iteration without recompilation.
- **Prototype in Python, ship in C++.** Python+numpy for rapid iteration; C++ (via pybind11,
  shared memory) for performance. The architecture works in both; the port is incremental, one
  system at a time.
- **Neural-network-compatible.** No decision forecloses training an NN agent: grid state maps to
  CNN feature planes, and clean serialization + headless simulation serve that goal.

---

## 2. System overview — two layers

**Simulation (the engine).** Self-contained, **headless-capable**, deterministic; owns all rules
and state; depends on no game engine. Interface: `get_state()` (read-only world snapshot —
**numerical arrays + a material-property table**, not objects), `apply_action(action)`, `step()`,
`reset(seed)`. This is also the RL environment interface (see `ml/01_ml_and_training.md`).

**Presentation (the renderer).** pyray reads sim state each frame and draws it; it captures input,
translates it to actions, and forwards them to the simulation. It **never mutates game state**.

```
            config.toml
                │
            GameMap  ── world state: numerical arrays + material table (numpy now, GPU-resident later)
                │       the single gmap.<field> interface every system reads/writes
   ┌─────┬──────┼───────┬───────┬────────┐
 atmosphere  smoke   fire   temperature  …      (engine physics, on shared fields)
   └─────┴──────┼───────┴───────┴────────┘
            Ray engine  ── light (RGB) + heat + vision + energy, deposit-only over the read-only world
                │
            Renderer (pyray)  ── reads buffers → shader → screen
```

Enables: headless self-play for NN training at scale, parallel sim/render development, and testing
without a renderer.

---

## 3. Chapters

Legend: ✅ implemented · ⚠️ partial (foundation built, advanced parts designed) · 🧪 prototype-only · 📝 design-only

### Engine — the deterministic world (read in order; each depends on the ones above)
| # | Chapter | Status |
|---|---------|--------|
| 01 | [Grid & coordinates](engine/01_grid_and_coordinates.md) | ⚠️ (core built, is_wall dropped; vision-via-atten, physics scaling owed) |
| 02 | [State & ownership (GameMap)](engine/02_state_and_ownership.md) | ⚠️ (permeability/soft-unit fields built; GPU residency owed) |
| 03 | [Material system](engine/03_material_system.md) | ✅ (permeability, wave_absorb, burst_threshold, mobility consumed; is_wall retired, passable→mobility) |
| 04 | [Atmosphere & pressure](engine/04_atmosphere_and_pressure.md) | ⚠️ (permeability boundary, 4a absorption, over-pressure burst built; venting/4b owed) |
| 05 | [Smoke](engine/05_smoke.md) | ⚠️ (smoke v2 + multi-gas M1/M2 coloured optics shipped; per-gas decay/flammability owed) |
| 06 | [Temperature & fire](engine/06_temperature_and_fire.md) | ⚠️ (heat→temperature→ignition + conduction + cooling + fire-as-heat-ray shipped; thermal wall-failure + unit-damage tuning owed) |
| 07 | [Fluid & water](engine/07_fluid_and_water.md) | ⚠️ (core built: C++ solver + tick + displacement/pressure-head couplings + flash-boil→steam + ripple + debug overlay; conduction/oil/ice/gameplay-reads designed) |
| 08 | [Ray engine](engine/08_ray_engine.md) | ⚠️ (Tier-1 shipped) |
| 09 | [Rendering](engine/09_rendering.md) | ✅ |
| 10 | [Pathfinding](engine/10_pathfinding.md) | ⚠️ (A* built, speed-blind enterability via mobility; temporal A* unused) |
| 11 | [Electricity / lightning](engine/11_electricity.md) | 📝 |
| 12 | [Config & hot-reload](engine/12_config_and_hot_reload.md) | ✅ |
| 13 | [FieldEdit (write primitive)](engine/13_field_edit.md) | ⚠️ (queue + applier + explosion/smoke migration built; wall_hp/destruction-sweep + laser/gas consumers designed) |
| 14 | [Determinism & the number-ingress rule](engine/14_determinism_and_number_ingress.md) | ⚠️ (the LAW + four doors + L1 lint + L3 digest/attestation shipped — `cuda-breached` proven Ampere↔Ada; L2 int-backed unit attrs + deterministic stat sampler owed to the units redesign) |

### Mechanics — game logic on the engine
| # | Chapter | Status |
|---|---------|--------|
| 01 | [Units & entities](mechanics/01_units_and_entities.md) | ⚠️ (mobility speed_fn seam shipped; footprint variants/cadence owed) |
| 02 | [AI & line-of-sight](mechanics/02_ai_and_los.md) | ⚠️ |
| 03 | [Combat & weapons](mechanics/03_combat_and_weapons.md) | ⚠️ |
| 04 | [Turn system & control](mechanics/04_turn_and_control.md) | ⚠️ |
| 05 | [Physics↔unit exchange](mechanics/05_physics_unit_exchange.md) | ⚠️ (heat/blast rows + stamp + FieldEdit shipped; formal coupling table + reductions + ordering principles designed; push/water/gas/O2 rows + statuses owed via combat chapter) |
| 06 | [Damage, health & conditions](mechanics/06_damage_health_and_conditions.md) | 📝 (designed 2026-07-04: packet pipeline, flat-then-mult mitigation, life/condition split, status system w/ behavior flags, RPG attack-resolver seam; wiring patches next — heat/blast/melee/HP-snap already shipped) |

### Graphics — the look layer (chapters by topic, not numbered; see [graphics/README.md](graphics/README.md))
| Chapter | Status |
|---------|--------|
| [Water rendering](graphics/water_rendering.md) | 📝 (in progress — Fresnel reflect/refract surface pass; research landed) |
| Smoke & gas optics · Lighting mood & post · Surface materials · Decals & particles | (planned — migrated/written as each look-pass is done) |

### ML
| # | Chapter | Status |
|---|---------|--------|
| 01 | [ML & training](ml/01_ml_and_training.md) | 📝 |

---

## 4. Principles (how we keep this folder honest)

- **Canon over prototype.** Prototypes (`prototypes/`, prototype-authored levels, scratch tools)
  may diverge freely; on any prototype-vs-canon conflict, **canon wins** — fix the prototype, not
  the engine or these docs. Canon changes only if canon itself is found wrong, deliberately.
- **Canonize bottom-up.** A system is locked as canon only once the systems it *depends on* are
  canon — docs sit on settled ground.
- **Design docs are canon chapters.** New design lands directly here as a chapter (right bucket, a
  "Depends on:" header, an implementation-status section) — not as a loose `docs/*.md` note.

---

## 5. Conventions

- **Reading order:** foundation first — engine `01→12`, then mechanics, then ml.
- **Dependencies:** every chapter's "Depends on:" header names what it builds on; the engine list
  above is already in dependency order.
- **Status is honest:** the implementation-status section reflects the *code*, not aspiration. A
  chapter being canon does not mean the system is fully built — see its status line.

_The CUDA migration plan (`docs/cuda_integration_plan.md`) and the living `docs/TODO.md` sit
alongside this folder; the GPU port re-implements the engine chapters once they're settled._
