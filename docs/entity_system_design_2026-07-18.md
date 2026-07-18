# Entity system — design (DRAFT, in design phase)

**Status: DESIGN PHASE.** Joint arc with `level_editor_v3_design_2026-07-18.md`
(the editor is the authoring VIEW over this MODEL). One adversarial critique
runs over both docs; build order is decided only after both are approved.
Erik's framing (2026-07-18): "fully programmable entities" — Factorio-style
sensors and logic, not just dumb wiring.

---

## 1. What this is

One model for everything placeable that isn't paintable matter: doors,
buttons, lights, sensors, logic nodes, the data chip, creatures, spawns.
Classes defined once (registry), placed as instances with overrides, connected
by signals. The level editor renders and edits this model; the game runs it.

The design bet, extended from the editor doc: **the registry is the game's
vocabulary** — and the *logic layer* is dataflow over signals, not a scripting
language.

## 2. The three layers — where "programming" lives (Erik's question)

Erik: "I don't mean everything has to be programmable inside the level editor
— I can do the programming in Python or C++ and prepare the entities first."
Correct instinct. The rational split is three layers:

| Layer | Lives in | Authored by | Example |
|---|---|---|---|
| **L0 — behavior** | Python/C++ code | Erik + Claude, per class | What "door closed" *means* physically; how a pressure sensor measures; what `blink` does |
| **L1 — vocabulary** | `entities.toml` (registry) | data edit | door class exists, has `state`, emits `is_open`, accepts `toggle`; a new critter variant |
| **L2 — composition** | `level.toml` (editor or text editor) | level design | THIS button wires to THIS door via THIS decider; the pen releases when pressure drops |

**Principle: intrinsic behavior in code, vocabulary in the registry,
composition in the level.** "Preparing entities first" = L0+L1 work in the
repo, exactly as Erik imagined. The editor only ever touches L2 (and reads
L1 to build its palettes/inspectors). There is NO embedded scripting language
(L3) — see §8; the bet is that dataflow (§4) + Erik-writes-L0-in-code covers
the design space without us inventing a language.

## 3. Classes & instances

As in the editor doc §4 (that section becomes a pointer here at fold time):
`entities.toml` classes with typed fields (`length_m`, enums, refs...);
instances in level.toml are class + overrides. New in THIS doc: classes also
declare **signals out** and **inputs in** — the wiring surface:

```toml
[entity.door]
category = "mechanisms"
[entity.door.fields]
width      = { type = "length_m", default = 1.0 }
state      = { type = "enum", options = ["open", "closed"], default = "closed" }
locked     = { type = "bool", default = false }
[entity.door.signals]           # outputs: readable by wires, every tick
is_open    = "bool"
[entity.door.inputs]            # inputs: writable by wires
open       = {}
close      = {}
toggle     = {}

[entity.sensor_pressure]
category = "sensors"
[entity.sensor_pressure.fields]
# measures at its own tile; no config needed for v1
[entity.sensor_pressure.signals]
pressure   = "q16"              # the tile's P, Q16.16 — integer, always
```

## 3b. L0 architecture — the schema lives in CODE (proposal, round 3)

Erik's instinct (2026-07-18): an abstract base class, concrete entity kinds
implemented one at a time, "loaded into the editor somehow." Proposal:

- **Python `Entity` abstract base class**; each kind (Door, Button,
  SensorMotion, DeciderNode...) is a subclass that *declaratively* states its
  schema — fields (name, type, default, constraints), signals, inputs — as
  class-level declarations, plus its L0 behavior (what `toggle` does, how the
  sensor measures). Registering the subclass (a decorator) adds it to the
  runtime registry.
- **The editor's registry is EXPORTED from code**, not hand-written: the
  editor imports the entity module (or a generated `entity_registry.json`)
  at launch — palette and inspectors always match the code, typos die at
  import time, and "I prepare entities in Python first" is literal: write
  the class, the editor sees it next launch. One source of truth.
- **`entities.toml` becomes the TUNING overlay, not the definition** — the
  weapons/materials pattern: code defines behavior + schema, TOML overrides
  class default *numbers* (hot-reloadable dials). No schema in TOML.
- C++/CUDA note: none needed. Dozens of entities, one integer logic pass per
  tick — CPU Python is ample; port later only if profiling ever says so.
  (Terminology: Python calls it an ABC; C++ an abstract base class; C#/Java
  an "interface" — same idea everywhere: a contract subclasses must fill.)

## 3c. Wiring at scale — tags + the authoring API (proposal, round 3)

Erik: "one button wired to 100 doors — I'd want to write a script, not drag
100 wires. This sets the limit for how cool levels we can build." Two
complementary mechanisms, both authoring-layer (runtime dataflow unchanged):

1. **Tags (groups) are first-class.** Every instance has `tags = ["..."]`;
   a wire target is an instance ref OR a tag — `targets = ["tag:lockdown"]`
   fires the input on every member. Resolved at runtime (destroyed members
   just go dead), readable in the TOML, and the editor authors it by
   multi-select → assign tag → one wire. Precedent: Source's targetname
   wildcards (one output → "door_*") — proven at exactly our scale.
2. **The authoring API — levels as a Python library.** `level_lib`: load a
   level, create/query/modify entities, wires, tilemap, save (the baker is
   already procgen-callable — this extends that door). "Write a script for
   it" becomes the OFFICIAL second authoring path beside the editor: author-
   time Python emitting plain data — the no-runtime-scripting decision (§8)
   is untouched, determinism never sees the script.
   **The ML tie-in makes this load-bearing:** training wants thousands of
   level variants (randomized pens, chip positions, breach sites). That IS
   the authoring API. We were going to need it; mission authorship gets it
   for free.

## 4. Signals & logic — the Factorio model, and why it fits us perfectly

Factorio's circuit network is the right reference for a reason deeper than
taste: **Factorio is a deterministic-lockstep game**, like us, and its logic
system is shaped by that. It is *dataflow*, not script:

- Entities put **signal values** on wires; **combinators** read signals and
  emit new ones — *decider* (if signal ⋚ value → output), *arithmetic*,
  *constant*. AND/OR/NOT are compositions of deciders. Everything re-evaluates
  every tick.
- **Each node reads its inputs from the PREVIOUS tick and writes outputs for
  the next — a one-tick delay per node.** This one rule is the whole
  implementation miracle:
  - **evaluation order is irrelevant** → no topological sort, no
    order-dependence bugs, trivially deterministic;
  - **cycles are legal** → feedback is allowed, so *memory emerges* (Factorio
    players build SR latches from two deciders — "alarm stays on until
    reset") — we get stateful mission logic without designing state;
  - cost is one array pass per tick over a few dozen nodes — nothing.
- **Signals are integers** (we use Q16.16 where physical) — the determinism
  iron rule holds; the logic layer adds zero float surface.

### v1 node/op set (deliberately small)

- **Sources:** every entity's declared signals (§3) + `constant` +
  `clock` (pure function of sim tick — the beacon trick, replay-exact).
- **Decider:** `out = (a ⋚ b) ? 1 : 0` where a = a signal, b = signal or
  constant; ops `> < >= <= == !=`.
- **Gates:** AND / OR / NOT (sugar over deciders, but worth first-class UI).
- **Filter:** `out = EMA(in, τ)` — exponential moving average with time
  constant τ (seconds). Integer/Q16.16, shift-based alpha, deterministic.
  A NODE, not a sensor variant (Erik 2026-07-18): any signal can be
  smoothed, which is more composable than baking smoothing into sensors.
  The canonical use — **auto-closing blast doors that ignore grenades**:
  `filter(sensor_pressure.area_mean, τ=3s) < 0.8 → close tag:blast_doors`.
  A grenade's under-pressure transient lasts under a second and barely
  moves a 3 s EMA; a real hull breach drags it down and latches the doors.
- **Sinks:** every entity's declared inputs. An input fires on a rising edge
  or while-held — per-input flag in the registry (`open` = while-high,
  `toggle` = rising-edge; prevents 24 toggles/second).
- Arithmetic combinator: **v2**, not v1 (add when a real level needs it).

### Example — Erik's airlock, in full

```
sensor_pressure(chamber).pressure < 0.1  ──┐
                                           ├─ AND ── door_inner.open
button(cycle).pressed ─────────────────────┘
button(cycle).pressed ── door_outer.close
clock.blink_2s ── AND(door_outer.is_open) ── warn_light.blink
```

Pen release on hull breach: `sensor_pressure(pen) < 0.5 → pen_door.open`.
The chip objective itself is expressible: `chip.in_zone(extract_A) →
win(team_a)` — the objective rule (stack 2) can BE a dataflow node, which
unifies mission logic and wiring into one system.

## 5. Logic is physical — with an invisible escape hatch (DECIDED, Erik 2026-07-18)

Factorio's combinators are physical, placeable, **destructible** objects.
Breach follows: **logic nodes are physical by default** — sensors, consoles,
relays, deciders are in-world entities with tiles, hp, and materiality.
Blowing the bridge console severs door control; a fire in the relay room
disables the pen locks. Emergent sabotage — matter-first applied to logic.

**The escape hatch:** physicality is a property, not an architecture. A node
class (or single instance) can set `intangible = true` — no tiles, no hp,
untouchable by physics — for mission logic that must not break (win
detection, ML reward taps). Physical is the superset; invisible is the
degenerate case; per-node choice is free. (Erik: "physical with the ability
to do invisible.") A destroyed node's signals read 0 and its inputs go dead
— honest wire semantics, no special cases.

## 6. Sensor catalog (draft — Erik's wishlist wanted)

All read existing fields at the sensor's tile(s), integer, one per tick:
`pressure` (P), `temperature` (T), `o2`, `smoke`, `water_depth`, `fire`,
`presence` (see below), `door.is_open`, `hp_below(x)` on any entity,
`chip.carried_by(faction)`, `clock` (tick fn). What else does the Contested
Chip ship need?

**Spatial mean:** field sensors (pressure/temperature/O₂/...) take an
optional `area` (length_m radius): emit the mean over the disc instead of
one tile. Integer mean over a precomputed tile set — cheap, deterministic.
Pairs with the filter node (§4) for the breach-door pattern.

**`sensor_motion` (unit sensor — Erik 2026-07-18 yes):** emits count of
units inside its range. Fields: `radius` (length_m), `faction_filter`
(any | one faction | hostiles-of), `min_footprint` (length_m — small
critters slip under it), `needs_los` (bool: line-of-sight sensor vs
through-wall proximity plate). Motion-triggered lights = `sensor_motion →
light.on` — and the emergent texture is free: zombies trip corridor lights
and betray themselves; a skitter-sized vent-crawler doesn't; cutting power
to the sensor (destroying it) darkens the trap.

## 6b. Actuators — entities that push the physics (Erik 2026-07-18)

Sensors read fields; **actuators write them** — through the FieldEdit
plumbing, so determinism and the number-ingress rule hold automatically.
Day-1 actuator (Erik's ask):

**`pump`** — pressurizes/depressurizes its enclosure. L0 behavior: each
active tick, move bulk gas N at its tile toward the target — inject standard
O₂/N₂ mix from an infinite ship reserve (tanks are a v2 field), or extract
(dump overboard). The EOS conserves mass; a pump is a deliberate
source/sink, exactly like explosion deposits and breach venting today.
Fields: `rate` (atm/s), `target_high` (default 1.0), `target_low` (default
0.0). Signals: `pressure` (own tile), `at_target` (bool). Inputs:
`pressurize`, `depressurize`, `off`.

**The airlock — the system's showcase.** Two doors + pump + terminal +
interlock. The sequencing ("close both → pump → open far door") is a small
state machine — buildable from latches in pure dataflow (the Factorio-purist
way), but the clean answer is the three-layer split doing its job: write an
**`airlock_controller`** entity class at L0 (a ~40-line Python state
machine: inputs `cycle_in`/`cycle_out`; drives its wired doors + pump;
signal `busy`), and every airlock in every level is then terminal →
controller → (doors, pump) wiring. This is the archetype of §8's rule —
"anything truly computational becomes a node class in code" — and a natural
first L0 class for Erik to write.

Future actuators, format-safe, not v1: heater, water valve, gravity/engine
systems. **Prefabs/blueprints** (stamp a saved multi-entity assembly — THE
airlock — into a level, Factorio-blueprint style) are the natural v2 editor
feature this example begs for; noted in the editor doc §10.

## 7. Runtime sketch (L0)

A `SignalBus` on Simulation: flat int arrays (signal id → value), double-
buffered (read prev / write next — the one-tick delay is literally a buffer
swap, the pattern the physics already uses). Tick order: physics → sensors
sample fields → logic pass (one array sweep) → inputs fire (door flips are
FieldEdits, same plumbing as today) → units act. Entity instances live in a
plain registry-typed list beside `Unit` (units CONVERGE into this model
later — `breach_unit_class_design.md` stays authoritative for unit stats;
we do not redesign units in this arc, we only reserve their seat: a unit IS
an entity with signals like `hp`, `alive`, `faction`).

## 8. What stays OUT (and why we can afford that)

- **Embedded scripting (Lua/Python-in-level):** the WC3/Roblox endpoint.
  Deferred indefinitely — dataflow covers reactive mission logic, and
  anything truly computational Erik writes at L0 as a new node class *in
  code* (a first-class, testable, reusable citizen — better than level-blob
  scripts for an ML-training game where determinism is law).
- **Arithmetic combinators, signal networks with multiple values per wire
  (Factorio's channels), displays/speakers:** v2+, format-safe.
- **AI/behavior trees for creatures:** separate design (beastiary/unit AI).

## 9. Decisions log + open questions

**Decided 2026-07-18 (Erik):**
- Restructure: entity model + editor view designed jointly, one critique
  over both, build order after both approved.
- Logic = dataflow (Factorio model, §4); no embedded scripting language.
- Programming split = three layers (§2); Erik prepares entities at L0/L1.
- Logic is physical by default, `intangible` per node as needed (§5).
- Meters-first (editor doc §4) — confirmed.

**Round 3 (2026-07-18):**
- §3b schema-in-code — **ACCEPTED** (Erik: "I love the python idea").
- §3c tags + authoring API — proposed, no objection; treat as accepted
  unless Erik flags it.
- §6 sensor catalog closed for v1 (Erik: nothing more to add) + area-mean
  fields; §4 filter node; §6b pump actuator + airlock_controller pattern;
  prefabs flagged as editor-v2.

**Open:**
1. Editor doc round-2 leftovers: door AP cost + operating classes; entity
   icon pipeline (placeholder chips?). Park for critique unless Erik has
   opinions sooner.
2. Mockup layout — Erik deferred judgment (2026-07-18); revisit once the
   entity discussion settles.
