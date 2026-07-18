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
`presence` (any unit / faction-filtered unit inside a zone or radius),
`door.is_open`, `hp_below(x)` on any entity, `chip.carried_by(faction)`,
`clock` (tick fn). What else does the Contested Chip ship need?

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

**Open:**
1. **Sensor wishlist** (§6) — what's missing for the Contested Chip ship?
2. Editor doc round-2 leftovers: door AP cost + operating classes; entity
   icon pipeline (placeholder chips?). Park for critique unless Erik has
   opinions sooner.
