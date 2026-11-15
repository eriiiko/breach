---
paths:
  - "src/simulation/unit.py"
  - "src/simulation/unit_schema.py"
  - "src/simulation/unit_traits.py"
  - "src/simulation/turning.py"
  - "src/simulation/unit_impulse.py"
  - "src/simulation/movement.py"
  - "src/simulation/speeds.py"
  - "src/simulation/exchange.py"
  - "src/simulation/factions.py"
  - "src/simulation/unit_fixed.py"
---

# Units: what a unit is, how it turns, moves and is pushed

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Unit
- One class; `is_zombie` is a state; subclassing is an anti-goal.
- `team` is any int and hostility is a different team. Zombies live on `factions.ZOMBIE_TEAM` and `is_zombie` is the ONLY zombie check: never a team literal. "The human's squad" is `Simulation.player_team` / `controlled_units()`. A level file's legacy `team = 1` becomes `ZOMBIE_TEAM` at the loader (`factions.team_from_level`), the one door.
- `unit.slot` is the fixed index within its spawn team: never reused, kept at death and conversion, derived from spawn order and so not digested (#119).
- Facing is the int `facing_q`; `facing` is a READ-ONLY float view written only by the Turning law. Position is the ints `x_q` / `y_q` (Q16 tiles); `x` / `y` are read-only views written only through `Unit.set_pos_q` (the float door `Unit.set_pos` quantizes once, half away from zero); `tile_x` / `tile_y` truncate (`tests/test_unit_positions_q16.py` scans `src/`).
- Its impulse state is written only through `unit_impulse`, its order columns only by `unit_orders`; its attributes are `__slots__` from the Unit schema (an undeclared attribute raises); the hp is a float held exactly on the Q16 grid.
- Its full health is `unit_traits.max_hp_q(u)`: the spawn, the conversion, the panel and `bots_v1` read it. The species' vitality is sheet input read by no tick.

## Unit schema
- THE one table of what a unit holds: group S (synced columns: kind, default, digest rule, integer encoding), group D (constants derived from S, never written after spawn), REFERENCE_ONLY (host state never decided on by the continuous match path, each with its reason).
- `Unit.__slots__` is generated from it, `apply_defaults` is the one place a column's default lives, and the unit digest is `digest_record` (each column while it differs from its default): never a hand-listed field.
- A new unit attribute is a schema row (a column, a derivation, or REFERENCE_ONLY with its reason) in the same patch; a group-S change also regenerates the device layout (Unit store).
- The status block is fixed-width: a stacking kind is capped at `STATUS_STACK_CAP`, refused and counted at `status.apply_status`.

## Unit traits
- How one unit differs from its class is a HOOK: a Q16 multiplier in `Unit.traits` (the `HOOKS` order, appended), written only by `compile_unit` at spawn (never inside a tick). A conversion KEEPS the block: the zombie's difference is a CLASS value selected on `is_zombie` at the law. Refused outside `[ONE/8, 8 ONE]` and, for TURN, past the turning law's trial bound.
- Each hooked value has ONE accessor and every reader calls it (the law, the bots, the observation, the masks, the panel), never the raw row. `tests/test_unit_traits.py` scans `src/`: a line naming a hooked class value outside its accessor carries `traits-class: <reason>`.
- An accessor is `round(class value × hook)`: ONE more integer rounding (`unit_fixed.mul_q16_round`; the speed hooks inside `default_speed`'s own), skipped at ONE so a neutral unit is the class value bit for bit; never a float; fixed for a tick.
- A stat, an item, a wound, a drug or a meter acts through the compile step or as a term of an accessor, never as code on one unit. A new hook is a row of `HOOKS` with its accessor and every reader through it; its device twin follows the twin ledger.
- Every hook is ONE today (no stat formula yet, #199); only HP may differ from ONE on a training run until an observation spec shows hooks.

## Turning law
- THE only writer of a unit's facing (`docs/unit_impulse_design_2026-11-10.md` §5.1, §13). A new reason to face somewhere is a TARGET handed to `set_target`, never a facing write; `place` (spawn, test fixtures) is the one other writer.
- The discrete law in slot 1a lands exactly and never overshoots; at `a ≥ 2w` it IS `turn_toward_q16` bit for bit (`tests/test_turning_law.py`). Its top speed bounds the OWN torque only; an incoming omega (the impulse law's outside spin, added first) is clamped below half a turn.
- The dials are read ONLY through `Simulation.unit_turn_rate_q16` / `unit_turn_accel_q16` (the class dials × the TURN hook); a zombie reads the marine's. The door refuses more than 32 trials.
- A controller that needs where this tick's shot goes reads `facing_after_turn_q16` (`turning.predict` + `unit_impulse.pending_dw`); `ticks_to_within` is the law iterated. Never a copy of the law.
- Hashed: `facing_q` always, `turn` and `omega` only while non-zero. A conversion settles the turn.

## Unit impulse
- Anything that pushes a unit hands it an IMPULSE through `add(...)`: never a position, velocity or facing write from a force. A new pusher is a caller of `add` (inside an attack slot through the `push` effect), a new response a term of the integration.
- Division truncates (`unit_fixed.div_trunc`, C's `/`; twins in `cpp/src/fixed_point.h`) and the REMAINDER stays in the accumulator, so the momentum books close exactly (`books_identity`); `V_MAX_Q` clamps, booked.
- ONLY the pushed velocity moves a unit, so soft input never does. Slot 3 is ONE displacement loop over every living unit (a walker's walk and its push as one move). The walk reads the law twice: `effort_scale_q16` (a `FootprintSamples` member) and `walk_allowed` (no walk while `react_ticks > 0`, at all three walk sites; the unit can still shoot).
- The shove's impulse and the knockdown stability are read through their accessors (`unit_traits.push_impulse_q`, `stability_q16`).
- DORMANT: the four multipliers at 0 and no shove = every impulse state exactly 0 on every tick. State hashed only while non-zero; zeroed at death and conversion; `StaggerEvent` / `ShoveEvent` render-only.
- ACCEPTED GAPS: the shover feels no reaction; wind does not turn a unit; one drag area for every unit; above stability sqrt(F / hold) (no shipped unit) a steady over-hold force slides a unit at its excess until its fall limit; the WEGO / onephase plans overwrite a push (legacy).

## Speed table
- Every unit speed is a `[speeds]` row in m/s, quantized ONCE per map: never a ticks-per-tile literal (Erik, 2026-11-10). The marine's walk and full speed become Q16 ticks per tile (the currency `default_speed` reduces in); the zombie's shamble its integer walk cadence (`Unit.speed_ticks_per_tile` None = this row, so a converted marine shambles too).
- The direction rule (`movement.move_speed_tiles_per_tick`, `facing_cos_q16`) is the ONE place facing changes a speed, against THIS tick's facing after slot 1a's turn, for every walker. Nobody sprints backwards.
- The door refuses a missing or mistyped key and holds Erik's rules: the zombie sprint within 70..90 % of the marine walk and ABOVE `backward_frac × walk`; `walk ≤ full`; `0 < back ≤ side ≤ 1`; a map on which one step would cover a whole tile is refused.
- LEGACY: WEGO's orders read `[movement] marine_*_seconds_per_tile`, onephase `[onephase] move_speed_mps`; neither is the continuous path's.

## Throttle
- The commanded speed is formed as ONE exact integer rational between the table's walk and full speed (each reduced by `default_speed`), extra speed FORWARD ONLY, divided once; the walk step is rounded half up ONCE onto the Q16 grid (`move_speed_q16`), so every position a walk writes is an exact n/65536. Never a speed-mode id on the continuous path. The door CLAMPS and the tap records the clamp.

## Footprint samples builder
- Every `default_speed` caller in `src/` takes its samples from `movement.footprint_samples` (`tests/test_footprint_samples_builder.py` scans `src/`; tests may build their own).
- A speed output is a Q16 scale member on `FootprintSamples`, composed in `default_speed`, which stays pure: samples + speed class, never the unit, never config.
- Every `footprint_samples` call in `src/` passes the walker's own `effort_scale_q16=` and `speed_hook_q16=` (scanned by `tests/test_unit_impulse.py` and `tests/test_unit_traits.py`); a caller reducing both costs builds the second set with `dataclasses.replace`.

## Coupling table
- A row whose effect on a unit is a PUSH hands the unit impulse law an impulse (`unit_impulse.add`) and owns no outcome of its own.
- The heat row's felt temperature is ONE function, `exchange.felt_temperature`: the row calls it and so does the observation. Never a copy of the chain.
- A swarm row names its `consumer` (`"swarm:<species>"`), has no Python response and is executed IN its species kernel (the last stage of slot 7); its `reads` are species-table fields the kernel's table rows carry. The future EXCHANGE-READ executor iterates `consumer == "unit"` rows only.

## Wind load on a body
- THE read of what the air does to a unit (#140): dynamic pressure as the int `W = N_raw * (v^2 >> 16)` (Python ints: an int32 square wraps), max over the footprint's gas tiles; `footprint_wind` also hands back the force's DIRECTION. Never read the planes.
- Its effect is a FORCE on the unit impulse law (`add_wind` at slot 9c2), priced once per map from the `[materials.air]` density row (never a dial), with no lever (wind does not turn a unit, v1). Below the hold nothing moves (soft wind never displaces, RULED). Prone, the knockdown is judged on the wind the body WOULD take standing (`Unit.stand_x` / `stand_y`).
- #140's cadence scale and bespoke knockdown are DELETED: never a speed member, never a status write from a coupling row. Never a pressure-gradient push. A new wind effect on a unit is a term of the impulse law, never a second reader. Render sway stays the tamed wind's.
- `GameMap.wind_load_plane()` is the same read over the whole map as a READ-SIDE float64 product for the observation; nothing in the sim reads it; its footprint max equals `footprint_wind_load` (`tests/test_wind_load_plane.py`).
