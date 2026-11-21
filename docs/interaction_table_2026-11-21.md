# What meets what: the interaction table of today's game

2026-11-21, issue #237 (track B step 1 of #238). Paper only.

This is the picture of the game as the code stands on `main` (639cf284): for every pair of kinds of thing, what the first does to the second. It is read from the code and the rule files in `.claude/rules/`, not from memory. Each rule cell names the file and function where it lives, and the config table that tunes it.

How to read it:

- A row is the ACTOR, a column the RECEIVER. "Fire acts on Walls" is a different cell from "Walls act on Fire".
- Each cell has one of four marks: **R** a rule; **0** none, on purpose (with the why); **?** not decided (with an issue number, or "no issue"); **.** not applicable (nothing there to meet).
- A rule that only works through another kind says so ("through heat"); the cell it passes through is named.
- Anything not confirmed in the code is marked UNVERIFIED.

The sections are: 1 the kinds of thing, 2 the overview grid, 3 one section per kind, 4 the holes, 5 findings, 6 how a new kind adds its row and column.

## 1. The kinds of thing

A kind earns its own row only if some rule treats it differently from its neighbours. The list follows what the code distinguishes, not the words we use.

| Code | Kind | What it is, and where it lives in the state | Why it is its own kind |
|---|---|---|---|
| AIR | Air | The bulk gas (O2 and inert N2), its pressure and its wind. The bulk planes of `gmap.gas`, `gmap.gas_energy`, the pressure mirror `gmap.atmosphere`, `gmap.wind_x` / `wind_y`. Moved by the EOS solver (`cpp/src/eos_solver.cpp`). | Everything that floats rides it; wind and pressure are its alone. |
| GAS | Trace gases | Smoke, steam, poison, teargas, fuel gas: the trace planes of `gmap.gas` (int32, the trace currency). | They ride the air but add no pressure, and each carries its own effect (blinds, doses, dims light). |
| HEAT | Heat | The temperature of gas and solids and the heat moving between them: `gmap.gas_energy` (gas), `gmap.temperature` (solids; the gas mirror), `gmap.heat` (this tick's deposit), the sweep's `rad_*_sweep` planes. | One energy ledger for gas and solids; units, larvae, fire and charges all read it. |
| FIRE | Fire | Flame intensity on flammable tiles, `gmap.fire`; its fuel is the tile's `gmap.wall_hp`. | Its own law (`cpp/src/fire_simulation.cpp`, `cpp/src/combustion.cpp`). |
| WAT | Water | `gmap.water_depth`, `gmap.flow_vx` / `flow_vy` (int32 Q16), solved by the pipe model (`cpp/src/water_solver.cpp`). The `ripple` pair is render-only. | Its own solver and its own couplings to air. |
| WALL | Walls | Tiles whose material is sealed to gas (`gmap.solid`): hull, steel, wood, glass. Hit points in `gmap.wall_hp`. | Every rule that tests `solid` treats them alike. Glass is merged here: no rule branches on glass; it differs only by row values (it passes most light but blocks sight). |
| FURN | Furnishings | Material tiles that are NOT sealed: crates (`furniture`), `kindling`, the `grille`, and static props, each of which is a load-time stamp of the `foliage` row (`prop_system.stamp_prop_tiles`). | Air, water, rounds and larvae pass them because they are not `solid`; everything else is read through the material row's columns, so no rule branches on which of these it is. Props carry render-only state (sway, look) and no sim state of their own, so they do not need a row apart. |
| DOOR | Doors | Door entities (`Simulation._doors`, `door_system.DoorRuntime`): closed is a `door_closed` span (solid), open is air; `state`, `want_open`, `hp`. | Their own laws: seal and unseal, the use verb, the zombie bash, "never close on a living unit". |
| DEV | Devices | Vents and ducts, pumps, sensors and logic nodes: entity runtimes swept at slot 9e. | They move air or read fields on their own, and talk to doors by wire. |
| HAZ | Hazards | Timed charges and heat sources (entities, `timed_charge_system`, `heat_source_system`). | They make blasts and heat by themselves and react to heat and blasts. |
| MAR | Marines | Living units that are not zombies (`Unit` with `alive` and not `is_zombie`). | The players' units. |
| ZOM | Zombies | Living units with `is_zombie`. Same class, a different state. | Different laws: the horde field, the bite, the bash, hearing, poison immunity, heat x4, the fear of fire, they observe nothing. |
| CORP | Corpses | Dead units (`alive` false); the hp below zero is the body (`simulation/corpse.py`). | Only the blast and the heat row reach them; they are not stamped; they can rise. |
| SWM | Swarm units | The larvae: the swarm store (`gmap.swarm_*`, written through `sim.swarm`), never a `Unit`. | Erik's ruling of 2026-11-14: two stores that meet only through fields. Their own law (`cpp/src/swarm_larva.h`). |
| RND | Rounds | Rounds in flight (`Simulation.bullets`, `combat.BulletInFlight`), the hitscan beam (`combat.fire_beam`), payload rounds. | The march has its own stops and its own hit rule. |
| SPR | Sprays | Flamer and gas jets: a cone laid each tick (`combat.process_sprays`, slot 4b). | It writes fields only and reads no unit. |
| GRN | Grenades and charges | Thrown grenades (`Simulation.projectiles`, `combat.Projectile`) and planted charges (`charges.PlantedCharge`). | They fly or wait, then become a blast; their flight has its own walk. |
| BLST | Blasts | One detonation: `payloads.execute_payload` on a `[payloads.*]` row, plus a strength row's expansion source (`gmap.expansions`, `div_source`). | The one entry for an event perturbing every field at once. |
| LIT | Light | The sweep's RGB light channels, read by the renderer through `simulation/light_field.py`. Lamps, flashlights and flashes are render inputs. | It is computed in the sim but no game rule may read it (#81). |
| NOI | Noise | Entries of `GameMap.noise_log` made by `noise.emit`. (The sound you hear is the audio layer, presentation only.) | Only zombies hear it. |
| SEE | Sight | Lines of sight (`GameMap.has_los`) and the match's sight rule (`vision.can_see`). | Not a body: its column says what blocks or changes a line of sight, its row what seeing does. |

Merges and splits against the brief's list: glass into Walls (above); props that are tile stamps, crates and the grille into Furnishings (above); vents, ducts, pumps and sensors into Devices; timed charges and heat sources into Hazards; bullets and beams into Rounds; thrown grenades and planted charges into one kind. "Heat in the air" became Heat for gas and solids together, because it is one ledger. Smoke and the other trace gases are one kind (their effects differ per gas row, not per rule). Added from the issue's list: Sprays and Sight. Zombies stay apart from Marines because their laws differ.
