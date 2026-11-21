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

## 2. The overview grid

Row = the actor, column = the receiver. **R** rule, **0** none on purpose, **?** not decided, **.** not applicable. The codes are those of section 1.

| acts on: | AIR | GAS | HEAT | FIRE | WAT | WALL | FURN | DOOR | DEV | HAZ | MAR | ZOM | CORP | SWM | RND | SPR | GRN | BLST | LIT | NOI | SEE |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **AIR** | R | R | R | R | R | R | 0 | R | R | . | R | R | R | 0 | . | . | ? | R | 0 | 0 | . |
| **GAS** | 0 | R | R | ? | ? | . | . | . | R | . | R | R | 0 | 0 | R | . | . | . | R | . | ? |
| **HEAT** | R | . | R | R | ? | R | R | . | R | R | R | R | R | R | . | . | . | . | R | . | . |
| **FIRE** | R | R | R | 0 | ? | R | R | ? | R | . | R | R | R | R | . | . | ? | . | R | ? | . |
| **WAT** | R | R | R | ? | R | ? | ? | R | R | . | ? | ? | ? | ? | . | . | ? | ? | 0 | . | . |
| **WALL** | R | R | R | R | R | . | . | . | R | . | R | R | . | R | R | R | R | R | R | 0 | R |
| **FURN** | R | R | R | R | ? | . | . | . | . | . | R | R | . | 0 | R | R | R | R | R | . | R |
| **DOOR** | R | R | R | . | R | . | . | . | R | . | R | R | 0 | ? | R | R | R | R | R | 0 | R |
| **DEV** | R | R | R | . | ? | . | . | R | R | . | . | . | . | . | . | . | . | . | . | . | . |
| **HAZ** | . | . | R | 0 | . | . | . | . | R | . | . | . | . | . | . | . | . | R | . | . | . |
| **MAR** | R | R | R | . | 0 | . | . | R | R | . | R | R | 0 | 0 | R | R | R | . | R | R | ? |
| **ZOM** | R | R | R | . | 0 | . | . | R | R | . | R | 0 | . | 0 | . | . | . | . | R | R | ? |
| **CORP** | 0 | . | 0 | . | . | . | . | 0 | . | . | . | R | . | . | 0 | . | . | . | 0 | . | . |
| **SWM** | . | . | . | . | ? | 0 | . | ? | ? | . | R | 0 | R | 0 | . | . | . | . | . | . | . |
| **RND** | . | . | . | 0 | 0 | R | R | R | . | . | R | R | 0 | 0 | . | . | . | R | 0 | . | . |
| **SPR** | . | R | R | 0 | ? | . | . | . | . | . | R | R | R | R | . | . | . | . | . | . | . |
| **GRN** | . | . | . | . | . | . | . | . | . | . | ? | ? | . | . | . | . | . | R | . | . | . |
| **BLST** | R | R | R | 0 | R | R | ? | R | . | R | R | R | R | 0 | . | . | ? | R | 0 | R | . |
| **LIT** | . | . | 0 | . | . | . | . | . | . | . | 0 | 0 | . | 0 | . | . | . | . | . | . | 0 |
| **NOI** | . | . | . | . | . | . | . | . | . | . | ? | R | . | 0 | . | . | . | . | . | . | . |
| **SEE** | . | . | . | . | . | . | . | . | . | . | R | R | . | . | . | . | . | . | . | . | . |

Counts over the 441 cells: 147 rules, 43 none on purpose, 32 not decided, 219 not applicable.

## 3. The cells, kind by kind

Each section lists what one kind does to every other kind. Where = the file and function, then the config table that tunes it.

### Air (AIR) acts on

- **Air**: *rule*. Air moves itself: pressure, density and wind are one compressible solve each tick. Where: `cpp/src/eos_solver.cpp` `EOSSolver::step` (CUDA twin `cuda_eos_step.cu`); dials `[physics]` bound in `PhysicsRunner`.
- **Trace gases**: *rule*. A trace gas moves only by riding the air's face flux, plus one diffusion step a tick. Where: `cpp/src/bulk_transport.cpp` (stage 3b) + `trace_decay.h`; dials `[gases.*] diffusion`, `decay`.
- **Heat**: *rule*. Moving air carries its heat with it: gas energy crosses each face once, at the donor's temperature. Where: `cpp/src/eos_solver.cpp` step 6 (face-flux energy step).
- **Fire**: *rule*. The air feeds and fans fire: sustain and ignition need the O2 mole fraction above the extinction limit, scaled by the pressure factor; wind fans or strips a flame. Where: `cpp/src/fire_simulation.cpp` `FireSimulation::step` (`k_wind_fan`, `k_wind_strip`) + `cpp/src/o2_pressure_factor.h`; ignition gate `combat.apply_temperature_ignition` (slot 9d); dials `[physics.fire]`, `[physics.combustion]`.
- **Water**: *rule*. Air pressure pushes the water surface (the pressure head), and water in air below the boil pressure flash-boils into steam. Where: `cpp/src/water_solver.cpp` (head `k_p * P`) and `PhysicsEngine::step_water` (W5 boil), called from `PhysicsRunner._step_water`; dials `[physics.water] k_p`, `boil_p_thresh`, `boil_rate`.
- **Walls**: *rule*. A one-tile wall holding a pressure difference above its row's burst threshold fails (capped per tick). Thick walls never burst. Where: `GameMap.find_burst_walls` via `Simulation._burst_walls` (slot 9b) then `destroy_wall`; column `[materials.*] burst_threshold`.
- **Furnishings**: *none, on purpose*. None in the sim: the burst scan walks `solid` tiles only, so a crate's `burst_threshold` is never read; a gas-open tile holds no difference. The props' sway in wind is render-only (tamed wind times gas density). Where: `GameMap.find_burst_walls` (`np.nonzero(solid)`); sway `renderer/static_props.py` (`.claude/rules/render-units.md`).
- **Doors**: *rule*. A closed door is a wall tile to the burst scan: `door_closed` bursts at 2.0, and the door sweep then destroys the whole door. Where: `GameMap.find_burst_walls`; `door_system.sweep_doors` (whole-door rule); `[materials.door_closed] burst_threshold`.
- **Devices**: *rule*. Pressure and O2 sensors read the air; a vent aperture on a vacuum or ambient tile is blocked. Where: `sensor_system.sample_sensors` over `EntityFieldAccessor` (`Channel.PRESSURE`, `Channel.O2`); `vent_system._aperture_blocked`.
- **Marines**: *rule*. Wind pushes a marine: the dynamic pressure over its footprint is a force on the impulse law (slowed, pushed past the hold, knocked down). The air's pressure and temperature also set its body temperature. Low O2, vacuum and low pressure do nothing to a marine. Where: `exchange.apply_wind_force` (slot 9c2) -> `unit_impulse.add_wind`, read `GameMap.footprint_wind_load`; body `body_temp.advance` in `exchange.apply_environmental_damage`; dials `[impulse]`, `[materials.air]` density, env profiles.
- **Zombies**: *rule*. As for marines (wind force, body air), with the zombie's own stability. Where: `exchange.apply_wind_force` (`species.ZOMBIE_STABILITY`).
- **Corpses**: *rule*. The body-temperature row reads the air a corpse lies in. Wind does not move a corpse: the wind row takes the living only. Where: `exchange.apply_environmental_damage` (runs on the dead, `.claude/rules/units.md` Body temperature); `exchange.apply_wind_force` (`if not u.alive: continue`).
- **Swarm units**: *none, on purpose*. None: larvae sense temperature only (Erik's P3 ruling 1, 2026-09-24) and do not breathe, so they also walk onto vacuum (ACCEPTED GAP). Where: `cpp/src/swarm_larva.h` `larva_step_slot`; `docs/swarm_P3_impl.md` rulings and gaps.
- **Rounds**: *not applicable*. A round's march reads no field but `solid` and the gas planes for a beam (see Trace gases). Where: `combat.BulletInFlight.advance`.
- **Grenades and charges**: *not decided*. None today: a thrown grenade flies its legs regardless of wind or pressure. No ruling found. Where: `combat.flight_walk`, `Projectile.fly`; no issue.
- **Blasts**: *rule*. A charge reads the air it goes off in, once: pressure and O2 over the tiles it reached scale its smoke and its heat. Where: `blast_air.read_blast_air` in `payloads.execute_payload`; dials `[blast_air]`.
- **Light**: *none, on purpose*. None: the bulk gases carry no optics (all-zero absorption), so clear air does not dim light. Where: `[gases.o2]`, `[gases.inert_n2]` rows; `gases.py` docstring.
- **Noise**: *none, on purpose*. None: hearing is by squared distance alone, no propagation through air (rule). Where: `.claude/rules/horde.md` Noise; `noise.emit`.
- **Sight**: *not applicable*. Lines of sight read `solid` and the sight screen only. Where: `GameMap.has_los`.
- *Not applicable*: Hazards, Sprays. Air does nothing of its own to a hazard entity (it reads only temperature and blasts) or to a spray cone (laid whatever the wind).

### Trace gases (GAS) acts on

- **Air**: *none, on purpose*. None: a trace gas rides the air and adds no pressure of its own; only the bulk pair is what the air is counted in. Where: `.claude/rules/fields-gas.md` Trace transport / FieldEdit (`gas` policy); `bulk_transport.cpp`.
- **Trace gases**: *rule*. Each trace gas diffuses and decays on its own plane; gases do not react with each other. Where: `cpp/src/bulk_transport.cpp` face diffusion; `cpp/src/trace_decay.h`; `[gases.*] diffusion`, `decay`.
- **Heat**: *rule*. Smoke absorbs radiant heat in the sweep. Exactly one gas absorbs heat: every other row is 0 (ACCEPTED GAP: band absorbers not modelled). Where: `cpp/src/radiation_sweep.cpp` gas pre-pass; `[gases.smoke] heat_absorb` (derived, `tests/test_gas_heat_absorb.py`).
- **Fire**: *not decided*. None in code. `[gases.fuel_gas]` says `flammable = true` and "IGNITES", but no engine code reads a gas's `flammable`, so fuel gas never burns. Where: `config.toml` `[gases.fuel_gas]`; `gases.py` ("read by fire and mechanics at M2/M3"); no issue.
- **Water**: *not decided*. None: steam never condenses back into water. Where: #219.
- **Devices**: *rule*. The smoke sensor reads the smoke plane; a vent's duct filter takes trace gas out by its `[filters.*]` row. Where: `sensor_system.sample_sensors` (`Channel.SMOKE`); `vent_system._duct_sweep` with `[filters.*]`.
- **Marines**: *rule*. Teargas above its threshold BLINDS a marine (no aimed fire); poison above its threshold is a dose of the poison law. Smoke, steam and fuel gas do nothing to a body. Where: `exchange.apply_teargas_blind`, `exchange.apply_poison_dose` (slot 9c3) -> `poison.dose`; `[exchange] teargas_blind_*`, `[damage.poison]`.
- **Zombies**: *rule*. Teargas blinds a zombie too, which changes nothing (the zombie law reads `can_act` and `can_move`, never `can_aim`). Poison gives a zombie no dose (immune). Where: `exchange.apply_teargas_blind`; `ai_zombie.update_zombies_tick`; `exchange.poison_immune` + the dose door.
- **Corpses**: *none, on purpose*. None: gas doses stop at death (Corpse law, Erik 2026-11-19). Where: `.claude/rules/units.md` Corpse law.
- **Swarm units**: *none, on purpose*. None: larvae sense temperature only (P3 ruling 1). Poison gas does not harm a larva. Where: `swarm_larva.h`; `docs/swarm_P3_impl.md`.
- **Rounds**: *rule*. Gas dims a laser beam tile by tile (Beer-Lambert in integers); a smoke grenade is a laser countermeasure. Bullets ignore gas. Where: `combat.fire_beam`; `GasTable.beam_absorb_q16`.
- **Light**: *rule*. Gases absorb and scatter light in the sweep's light channels, density-weighted per gas. Where: `cpp/src/radiation_sweep.cpp` light channels; `GasTable.light_absorb_q16` / `light_glow_q16`; `[smoke]` dials folded at the gases door.
- **Sight**: *not decided*. None: smoke, however thick, does not block a line of sight (`has_los` reads `solid` and the sight screen). The 2026-06-14 field map asked Erik whether smoke should hide; no ruling found. Where: `GameMap.has_los`; `docs/physics_field_interaction_map.md` section 6; no issue.
- *Not applicable*: Walls, Furnishings, Doors, Hazards, Sprays, Grenades and charges, Blasts, Noise. Gases have nothing to act on in walls, furnishings, doors, hazards, sprays, grenades, blasts or noise: no rule reads a gas there and none is plausible at the game's scale.

### Heat (HEAT) acts on

- **Air**: *rule*. Hot gas is high-energy gas: the air's pressure comes from its stored energy, so heating air raises its pressure and drives wind. Where: `GameMap.gas_energy` + `cpp/src/eos_solver.cpp` step 7 (recovery) and the EOS.
- **Heat**: *rule*. Heat moves by conduction through solids and by radiation (one sweep per ordinate); every joule is booked. Where: `cpp/src/temperature_solver.cpp`; `cpp/src/radiation_sweep.cpp`; closure identity `.claude/rules/energy-books.md`.
- **Fire**: *rule*. A flammable tile whose temperature reaches its `ignition_temp`, with enough O2, catches fire; this is the only way fire spreads. A fire below its row's T_ext dies. Where: `combat.apply_temperature_ignition` (slot 9d); `FireSimulation::step` (`fire_T_ext_plane`); `[materials.*] ignition_temp`.
- **Water**: *not decided*. None: water carries no temperature, so heat neither warms nor boils it (only low pressure boils water). Where: #219.
- **Walls**: *rule*. Walls are thermal solids: they take heat by conduction and radiation, hold a temperature and radiate at their emissivity. Where: `cpp/src/temperature_solver.cpp`; `[materials.*] heat_atten`, `conductivity`, `specific_heat`.
- **Furnishings**: *rule*. Crates and foliage are thermal solids that heat up and then ignite at their `ignition_temp`. Where: `cpp/src/temperature_solver.cpp`; `combat.apply_temperature_ignition`.
- **Devices**: *rule*. The temperature sensor reads a tile's temperature. Where: `sensor_system.sample_sensors` (`Channel.TEMPERATURE`).
- **Hazards**: *rule*. A heat-triggered timed charge fires once when its tile's temperature reaches `trigger_temp`; a heat source stops depositing while its tile is at `max_temp`. Where: `timed_charge_system.sweep_timed_charges` (`trigger = "heat"`); `heat_source_system.sweep_heat_sources`.
- **Marines**: *rule*. The felt temperature (the larger of the tile's heat deposit and the radiant flux, hottest footprint tile) and the air warm the body; a body past its band takes heat damage. One protection number per suit. Where: `exchange.apply_environmental_damage` (slot 9c), `exchange.felt_temperature`, `body_temp.advance`; `[combat]` body dials, env profile `heat_absorption`.
- **Zombies**: *rule*. As for marines, with the zombie's heat weakness x4. Where: `exchange.apply_environmental_damage`; `species.ZOMBIE_MITIGATION`.
- **Corpses**: *rule*. The heat row runs on corpses: heat still destroys a body after death. Where: `exchange.apply_environmental_damage` -> `damage.apply_packet`; `corpse.destroyed`.
- **Swarm units**: *rule*. Larvae steer toward their preferred temperature, fall into a chill coma below 10 C and die at once above 50 C (raw own-tile temperature). Where: `cpp/src/swarm_larva.h` `larva_step_slot`; the three `consumer="swarm:larva"` rows of `exchange.COUPLING_TABLE`; `[swarm.larva]` T rows.
- **Light**: *rule*. Hot cells glow: the light channels emit by temperature through the L table (Kirchhoff per channel); light never enters the heat books. Where: `cpp/src/radiation_sweep.cpp` light channels; `cpp/src/light_emission_table.inc`.
- *Not applicable*: Trace gases, Doors, Rounds, Sprays, Grenades and charges, Blasts, Noise, Sight. Heat has nothing of its own for trace gases (a hot gas is the air's energy, see Heat -> Air), doors (a closed span is a wall tile, see Heat -> Walls), rounds, sprays, grenades, blasts, noise or sight.

### Fire (FIRE) acts on

- **Air**: *rule*. A fire draws O2 from the air around it in proportion to its intensity; contested O2 is split evenly. Where: `cpp/src/combustion.cpp` `CombustionSolver::step`.
- **Trace gases**: *rule*. Burning makes smoke: `soot_yield` units per unit of O2 burned. Where: `CombustionSolver::step`; `[physics.combustion] soot_yield`.
- **Heat**: *rule*. Burning releases heat into the air and into the burning tile, booked in the energy ledger. Where: `CombustionSolver::step` (`e_comb_*` counters).
- **Fire**: *none, on purpose*. No cell-to-cell spread: it was deleted, and fire spreads only by heat reaching a neighbour's ignition temperature. Where: `Simulation._step_post_physics` slot 9d comment; fire design proposal section 1.
- **Water**: *not decided*. None: fire does not evaporate water. Where: #219.
- **Walls**: *rule*. Fire eats a flammable wall's hp (wood); at 0 the wall burns through and opens. Where: `FireSimulation::step` (`wall_hp`, returns destroyed) -> `Simulation._burn_through_walls` (slot 9) -> `destroy_wall`.
- **Furnishings**: *rule*. Fire eats a crate's or foliage tile's hp, but never destroys it: only solid tiles burn through, so a burnt-out crate keeps negative hp and burns on. Where: `cpp/src/fire_simulation.cpp` (`if (wall_hp[i] <= 0 && is_wall[i])`); #130, #23.
- **Doors**: *not decided*. None: the door rows are not flammable. Row values are placeholders until Erik's material session (Erik, 2026-09-22). Where: `[materials.door]`, `[materials.door_closed]` `flammable = false`; `docs/thin_material_rows_design_2026-09-20.md`; no issue.
- **Devices**: *rule*. The fire sensor reads a tile's fire. Where: `sensor_system.sample_sensors` (`Channel.FIRE`).
- **Marines**: *rule*. Only through heat: a marine in flames is hurt by the felt temperature. The BURNING status is registered but no rule applies it ("fire coupling row, later"). Where: `exchange.apply_environmental_damage`; `status.BURNING` in `status.py`.
- **Zombies**: *rule*. A zombie with `fear_fire` never steps onto a burning or hot tile; otherwise as marines, through heat. Where: `horde.block_feared` in `ai_zombie`; `[zombie] fear_fire`.
- **Corpses**: *rule*. Through the heat row on its tiles (corpses are not stamped, so they take no radiation of their own). Where: `exchange.apply_environmental_damage`; `.claude/rules/units.md` Corpse law.
- **Swarm units**: *rule*. Through temperature only: a larva steers away from heat and dies above 50 C on its tile. Where: `swarm_larva.h`.
- **Grenades and charges**: *not decided*. None: a grenade or planted charge in fire never cooks off. Where: #218 is the nearest (explosive cargo); no issue for grenades.
- **Light**: *rule*. A burning tile emits light through its temperature; the room light's gain and flicker are a light-only multiplier. Where: `radiation_sweep` thermal emission; `emit_mult` from `renderer/fire_flicker.py`, `[render.fire_light]`.
- **Noise**: *not decided*. None: a fire is not a noise source for zombies (the fire's sound loop is presentation only). Where: `audio/model.py` `fire_gains`; no issue.
- *Not applicable*: Hazards, Rounds, Sprays, Blasts, Sight. Fire has nothing to act on in hazards (only through heat, see Heat), rounds, sprays, blasts or sight.

### Water (WAT) acts on

- **Air**: *rule*. Rising water squeezes the air above it (isothermal) and pushes gas out of a flooding tile; a flooded tile is sealed to air. Where: `PhysicsEngine::step_water` W3 (displacement, flooded `dyn_permeability = 0`); `[physics.water] ceiling_h`, `ratio_cap`, `flood_eps`.
- **Trace gases**: *rule*. Boiling water makes steam; a flooded tile stops smoke too (sealed). Where: `PhysicsEngine::step_water` W5 (`steam_yield`).
- **Heat**: *rule*. Gas pushed out by water carries its energy at the donor's temperature; what leaves the books is exported. Water itself holds no heat. Where: `PhysicsEngine::step_water` (`gas_energy`, `e_water_evac_export_sum`).
- **Fire**: *not decided*. None: water does not put out fire, and a flooded flammable tile can still burn. Where: #219.
- **Water**: *rule*. Water flows by the pipe model over floor height and tilt, conserving depth. Where: `cpp/src/water_solver.cpp`; `[physics.water]`.
- **Walls**: *not decided*. None: water pressure never loads or bursts a wall (the burst scan reads air only), so a full tank never breaks its glass. Where: `GameMap.find_burst_walls`; #111 is the nearest.
- **Furnishings**: *not decided*. None: a crate tile is open, so water flows through it; nothing floats. Floating by density is the movable-props and bodies-in-water work. Where: #226, #236.
- **Doors**: *rule*. A door cannot close over standing water: the seal refuses, and the door retries every tick. Where: `GameMap.seal_tiles` / `can_seal_tiles` (water rule v1); `door_system.sweep_doors`.
- **Devices**: *rule*. A flooded vent aperture is blocked; the water-depth sensor reads depth. Where: `vent_system._aperture_blocked`; `sensor_system.sample_sensors` (`Channel.WATER_DEPTH`).
- **Marines**: *not decided*. None: a marine walks through any depth at full speed, is not pushed, does not drown. Only the picture reacts (wake foam, the waterline). Where: `movement.py` ("the planned `water_depth` speed row"); `renderer/water_wake.py`; #236.
- **Zombies**: *not decided*. None, as for marines. Where: #236.
- **Corpses**: *not decided*. None: a corpse neither floats nor drifts. Where: #236.
- **Swarm units**: *not decided*. None: the larva law reads only temperature and `solid`, so a larva crawls along the bottom of a flooded room at its normal speed, steering by temperature, and keeps biting. It does not drown, float or drift. Where: `swarm.engine_args` (hands `temperature` and `solid` only); `swarm_larva.h`; #236.
- **Grenades and charges**: *not decided*. None: a grenade flies and detonates in water exactly as in air. Where: no issue.
- **Blasts**: *not decided*. None: a blast in water is a blast in air (the picture's splash is render-only). Where: #222 (the picture), no issue for the sim.
- **Light**: *none, on purpose*. None in the sim: the sweep does not see water. The caustics, the waterline and the surface are render-only. Where: `.claude/rules/render-effects.md` Water flow detail; `lit3d._WATERLINE_GLSL`.
- *Not applicable*: Hazards, Rounds, Sprays, Noise, Sight. Water has nothing to act on in hazards, rounds, sprays, noise or sight at the game's scale; none is designed.

### Walls (WALL) acts on

- **Air**: *rule*. A solid tile is a wall to the air: no face, no gas inside, a hard boundary. Where: `eos_solver.cpp` masks; `GameMap.solid` / `obstacles`.
- **Trace gases**: *rule*. Same boundary for trace gases. Where: `bulk_transport.cpp`.
- **Heat**: *rule*. Walls conduct, absorb and radiate at their row's emissivity; glass lets heat partly through (`heat_atten` 0.3). Where: `temperature_solver.cpp`; extinction planes `GameMap.heat_atten_q`; `[materials.*] heat_atten`.
- **Fire**: *rule*. A flammable wall row (wood) is fuel: its hp is the fire's fuel store. Where: `FireSimulation::step` (`flammable`, `wall_hp`, `fuel_recip`).
- **Water**: *rule*. Water cannot enter a solid tile; walls bound every pool. Where: `water_solver.cpp` (zero on solid, no face).
- **Devices**: *rule*. A vent aperture on a solid or thermal-solid tile is blocked. Where: `vent_system._aperture_blocked`.
- **Marines**: *rule*. A unit cannot overlap a wall (`mobility` 0). Where: `GameMap.is_passable_block`; `[materials.*] mobility`.
- **Zombies**: *rule*. As for marines; the horde's distance field floods around walls. Where: `horde.block_passable`, `horde.flood`.
- **Swarm units**: *rule*. A larva's move into a solid tile is blocked and it turns; a solid sweep point is no reading. Where: `cpp/src/swarm.h` `probe_move`, `head_sweep`.
- **Rounds**: *rule*. A round stops at the first solid tile and chews it (glass included, which breaks at 15 hp). A beam stops there too. Penetration per material is Erik's open session. Where: `combat.BulletInFlight.advance` -> `combat.chew_wall`; `combat.fire_beam`; `.claude/rules/sight.md`.
- **Sprays**: *rule*. A spray cone stops at a solid tile (the physical line). Where: `combat.spray_cone_tiles` over `has_los(physical=True)`.
- **Grenades and charges**: *rule*. A thrown grenade never passes a wall: it bounces off solid tiles. Where: `combat.flight_walk`, `flight_blocked`, `Projectile.bounce`; `[weapons.hand_grenade] bounce_*`.
- **Blasts**: *rule*. A blast's heat, smoke and push walk stops at walls (a corridor concentrates it). Its direct damage to units does not: that is a disc through walls (see Blasts). Where: `blast_expansion.reach_tiles`; `exchange.apply_blast_damage`.
- **Light**: *rule*. Walls block light by their row (glass passes most of it, `light_atten` 0.1). Where: `GameMap.light_atten_q`; `[materials.*] light_atten`.
- **Noise**: *none, on purpose*. None: walls are ignored by hearing (no sound propagation). Where: `.claude/rules/horde.md` Noise; #201 proposes muffling.
- **Sight**: *rule*. A solid tile blocks sight. Glass is solid, so it blocks sight although light passes (the `blocks_sight` column is derived from being sealed to gas). Where: `GameMap.has_los`; `materials.py` `blocks_sight`.
- *Not applicable*: Walls, Furnishings, Doors, Hazards, Corpses. A wall does nothing to another tile, a door, a hazard or a corpse.

### Furnishings (FURN) acts on

- **Air**: *rule*. Crates and kindling slow air (permeability 0.5); the grille and foliage let it through (1.0). Where: `GameMap.permeability` / `dyn_permeability`; `[materials.*] permeability`.
- **Trace gases**: *rule*. Same permeability for smoke. Where: `bulk_transport.cpp`.
- **Heat**: *rule*. Crates and foliage are thermal solids with their own capacity and emissivity. Where: `temperature_solver.cpp`; `[materials.furniture]`, `[materials.foliage]`.
- **Fire**: *rule*. Crates, kindling and foliage are fuel; the grille is not. Where: `FireSimulation::step`; `[materials.*] flammable`, `hp`.
- **Water**: *not decided*. None: water flows through these tiles as if they were floor (not `solid`). Where: `water_solver.cpp`; #226, #236.
- **Marines**: *rule*. Crates and kindling slow a walker to 40%; foliage does not; the grille stops every unit. Where: `GameMap.footprint_mobility`, `is_passable_block`; `[materials.*] mobility`.
- **Zombies**: *rule*. As for marines. Where: `horde` mobility read.
- **Swarm units**: *none, on purpose*. None: larvae walk onto crates, grilles and foliage (not solid) and read the crate's own temperature there (ACCEPTED GAP). Where: `docs/swarm_P3_impl.md` gaps.
- **Rounds**: *rule*. Rounds pass through these tiles. A crate (or kindling) crossed just before a target's footprint is soft cover: one roll at 0.55, and the crate eats the round. A round through a grille is undecided. Where: `combat.BulletInFlight.advance` + `attack_resolver.cover_exposure_at`; `[materials.*] cover_exposure`; `.claude/rules/hand-verbs.md`.
- **Sprays**: *rule*. Not solid, so a spray cone passes them. Where: `combat.spray_cone_tiles` (physical line).
- **Grenades and charges**: *rule*. A grenade bounces off a grille (`blocks_thrown`); it passes crates and foliage. Where: `combat.flight_blocked`; Erik 2026-11-18.
- **Blasts**: *rule*. The blast's walk passes through passable fuel tiles (they stay in its open mask). Where: `blast_expansion.reach_tiles`; `.claude/rules/explosions-fire.md` Blast reach.
- **Light**: *rule*. Crates pass part of the light (0.55), the grille some (0.15), foliage all of it. Where: `[materials.*] light_atten`.
- **Sight**: *rule*. The grille blocks sight (Erik, 2026-11-06); crates and foliage do not. Where: `GameMap.sight_screen`; `[materials.grille] blocks_sight`.
- *Not applicable*: Walls, Furnishings, Doors, Devices, Hazards, Corpses, Noise. Furnishings do nothing to other tiles, doors (a prop may not be placed on a door span: a load rule), devices, hazards, corpses or noise.

### Doors (DOOR) acts on

- **Air**: *rule*. Closing seals the span gas-tight and moves its gas to the receivers; opening unseals it. Where: `GameMap.seal_tiles` / `unseal_tiles` via `door_system.sweep_doors` (slot 9e(d)).
- **Trace gases**: *rule*. Smoke moves with the sealed gas and is stopped by a closed door. Where: `GameMap.seal_tiles` (`gas_proportional_split`).
- **Heat**: *rule*. A closed span is a wall tile to heat (the door rows' emissivity). Where: `temperature_solver.cpp`; `[materials.door_closed]`.
- **Water**: *rule*. A closed door is solid, so it holds water back. Where: `water_solver.cpp`.
- **Devices**: *rule*. A door publishes `is_open` and `alive` on the signal bus. Where: `Simulation._signal_emit` (slot 9e(a)).
- **Marines**: *rule*. A closed door blocks a unit; a door will not close on a living unit's footprint. Where: `is_passable_block`; `door_system._occupancy_clear`.
- **Zombies**: *rule*. As for marines; a closed non-hull door is bashed (see Zombies). Where: `door_system._occupancy_clear`; `horde.is_hull_door`.
- **Corpses**: *none, on purpose*. None: a door closes on a corpse (corpses never block). Where: `door_system._occupancy_clear` ("corpses never block").
- **Swarm units**: *not decided*. A door closing on a larva leaves it inside the solid span, frozen until the door reopens. "P4 decides refuse, eject or crush" was never decided. Where: `docs/swarm_P3_impl.md` ACCEPTED GAP; `GameMap.seal_tiles` checks no swarm occupancy; no issue.
- **Rounds**: *rule*. A closed door stops a round and is chewed; at 0 hp the whole door dies. Where: `combat.chew_wall`; `door_system.sweep_doors` (whole-door rule).
- **Sprays**: *rule*. A closed door stops a spray cone. Where: `combat.spray_cone_tiles`.
- **Grenades and charges**: *rule*. A grenade bounces off a closed door. Where: `combat.flight_blocked`.
- **Blasts**: *rule*. A closed door stops the blast's walk. Where: `blast_expansion.reach_tiles`.
- **Light**: *rule*. A closed door blocks light. Where: `[materials.door_closed] light_atten`.
- **Noise**: *none, on purpose*. None: a door is no wall to hearing, and opening one makes no noise. Where: `.claude/rules/horde.md` Noise.
- **Sight**: *rule*. A closed door blocks sight. Where: `GameMap.has_los`.
- *Not applicable*: Fire, Walls, Furnishings, Doors, Hazards. A door does nothing to fire, to other tiles or to hazards.

### Devices (DEV) acts on

- **Air**: *rule*. Vents circulate air through ducts and pumps inject, extract or transfer gas, all through the gas-N primitives. Where: `vent_system.sweep_vents`, `pump_system.sweep_pumps` (slot 9e(d)); `GameMap.inject_gas_n` / `extract_gas_n_vec` / `transfer_gas_n`.
- **Trace gases**: *rule*. Trace gas rides the pumped air, priced by air; a filter removes its gases. A pump destroys the trace it takes, unbooked (ACCEPTED GAP). Where: `GameMap.extract_gas_n_vec`; `[filters.*]`; #98.
- **Heat**: *rule*. Moved gas carries its energy at the donor's temperature through the energy seam. Where: pump primitives in `gamemap.py` (`.claude/rules/fields-gas.md`).
- **Water**: *not decided*. None: no device moves water; running sources and drains do not exist in a level. Where: `.claude/rules/levels.md` Water studio; no issue.
- **Doors**: *rule*. Wired inputs drive a door's `want_open` (close beats open). Where: `Simulation._resolve_door_inputs` (slot 9e(c)).
- **Devices**: *rule*. Sensors and logic nodes feed each other on the signal bus, one tick per hop. Where: `logic_nodes.sweep_logic_nodes`; `SignalBus.swap_node_signals`.
- *Not applicable*: Fire, Walls, Furnishings, Hazards, Marines, Zombies, Corpses, Swarm units, Rounds, Sprays, Grenades and charges, Blasts, Light, Noise, Sight. Devices act only on the air and its gases, doors and each other; everything else reaches them or not through those.

### Hazards (HAZ) acts on

- **Heat**: *rule*. A heat source delivers its heat to its tile every tick, through the counted deposit. Where: `heat_source_system.sweep_heat_sources` -> `payloads.deposit_heat`; entity `heat_per_s`.
- **Fire**: *none, on purpose*. Not directly: fire is started only by delivering heat (rule), so a heat source lights fuel through Heat -> Fire. Where: `.claude/rules/explosions-fire.md` Starting a fire.
- **Devices**: *rule*. A hazard's `alive` can be wired to the bus (a heat source's wired `alive` reads 1 after its death: a finding in the rule file, not changed). Where: `Simulation._signal_emit`; `.claude/rules/device-units.md` Signal slots.
- **Blasts**: *rule*. A timed charge detonates its payload row on its timer, or once when its tile gets hot; a heat source runs its death payload when a blast kills it. Where: `timed_charge_system.sweep_timed_charges`, `heat_source_system._die` -> `payloads.execute_payload`.
- *Not applicable*: Air, Trace gases, Water, Walls, Furnishings, Doors, Hazards, Marines, Zombies, Corpses, Swarm units, Rounds, Sprays, Grenades and charges, Light, Noise, Sight. Hazards act only through their heat and their blasts; every other kind is reached through those rows.

### Marines (MAR) acts on

- **Air**: *rule*. A living body is a soft, porous block to the air: its footprint slows flow (permeability 0.5) and soaks pressure waves (0.5). A walking body pushes no air. Where: `GameMap.stamp_units` (slot 6); `[physics] unit_permeability`, `unit_wave_absorb`.
- **Trace gases**: *rule*. Smoke seeps past a body at the same permeability. Where: `GameMap.stamp_units`.
- **Heat**: *rule*. A living body is opaque to radiant heat (it casts a heat shadow). Where: `GameMap.stamp_units` (`dyn_heat_atten_q`).
- **Water**: *none, on purpose*. None: water flows under feet (Erik's explicit call). Only the picture shows a wake. Where: `PhysicsRunner._step_water` docstring; `renderer/water_wake.py`.
- **Doors**: *rule*. A marine opens or closes the door it faces in reach (the use verb); standing in a span stops it closing. Where: `Simulation.use_intent` -> `verb_gates.use_door`; `door_system._occupancy_clear`.
- **Devices**: *rule*. A motion sensor counts living units in its radius (optionally in its line of sight). Where: `sensor_system._MotionSensorRuntime`.
- **Marines**: *rule*. Melee blows and shoves hit and push other units through the impulse law; rounds, sprays and grenades are their own rows. Bodies pass through each other. Where: `combat.process_melee_intents`, `Simulation.shove_intent` -> `unit_impulse.add`; pass-through: `Simulation._update_player_movement` (#142); collision #16.
- **Zombies**: *rule*. The same verbs strike zombies (an enemy of every team). Where: `combat.process_melee_intents`.
- **Corpses**: *none, on purpose*. None: melee strikes the living, and rounds, statuses and sprays stop at death (Corpse law). Where: `.claude/rules/units.md` Corpse law.
- **Swarm units**: *none, on purpose*. None: a marine cannot step on or hurt a larva, and larvae do not see marines (ACCEPTED GAPS of arc #63 P3). Where: `docs/swarm_P3_impl.md`.
- **Rounds**: *rule*. A marine fires rounds and beams along its facing, from its weapon row. Where: `combat.process_shooting` (slot 4); `[weapons.*]`, `[ammo.*]`.
- **Sprays**: *rule*. A marine starts a spray burst; slot 4b lays it. Where: `combat.process_sprays`.
- **Grenades and charges**: *rule*. A marine throws grenades (after its wind-up, from its hand) and plants charges. Where: `Simulation.throw_grenade_intent`, `combat.release_point`; `Simulation.plant_charge_intent` -> `charges.place`.
- **Light**: *rule*. A living body blocks light (units cast shadows). Its flashlight is render state: nothing in the sim reads it. Where: `GameMap.stamp_units` (`dyn_light_atten_q`); `Unit.flashlight_on`.
- **Noise**: *rule*. Firing a weapon (bullet, beam or spray start) is a noise at the shooter, of the row's `loudness`. Walking and melee are silent. Where: `combat._dispatch_trigger` / `_directional_fire` -> `noise.emit`; `[weapons.*] loudness`.
- **Sight**: *not decided*. None: a body does not block a line of sight (though it blocks light). Where: `GameMap.has_los`; no issue.
- *Not applicable*: Fire, Walls, Furnishings, Hazards, Blasts. Marines do nothing directly to fire, walls, furnishings, hazards or blasts: those happen through their rounds, sprays and grenades.

### Zombies (ZOM) acts on

- **Air**: *rule*. As for marines (the stamp takes every living unit). Where: `GameMap.stamp_units`.
- **Trace gases**: *rule*. As for marines. Where: `GameMap.stamp_units`.
- **Heat**: *rule*. As for marines. Where: `GameMap.stamp_units`.
- **Water**: *none, on purpose*. None: water flows under feet. Where: `PhysicsRunner._step_water` docstring.
- **Doors**: *rule*. A zombie blocked by a closed non-hull door bashes it at its bite cadence through the wall-chew seam; it never bashes a hull door. Where: `ai_zombie._bash` -> `combat.chew_wall`; `horde.is_hull_door`; `[zombie] bash_doors`, `door_bash_damage`.
- **Devices**: *rule*. A motion sensor counts zombies too. Where: `sensor_system._MotionSensorRuntime`.
- **Marines**: *rule*. An awake zombie bites the nearest marine (a kinetic packet); a marine a zombie kills rises as a zombie. Where: `ai_zombie.update_zombies_tick`; `ai_zombie.convert_marines_to_zombies` (`killed_by_zombie`).
- **Zombies**: *none, on purpose*. None: zombies do not fight each other and pass through each other. Where: `ai_zombie.update_zombies_tick` (prey = non-zombies).
- **Swarm units**: *none, on purpose*. None, as for marines. Where: `docs/swarm_P3_impl.md`.
- **Light**: *rule*. A zombie's body blocks light. Where: `GameMap.stamp_units`.
- **Noise**: *rule*. Each bash strike is a noise at the struck tile. Where: `ai_zombie._bash` -> `noise.emit`; `[zombie] bash_loudness`.
- **Sight**: *not decided*. None: a zombie's body does not block sight. Where: `GameMap.has_los`; no issue.
- *Not applicable*: Fire, Walls, Furnishings, Hazards, Corpses, Rounds, Sprays, Grenades and charges, Blasts. Zombies fire nothing, plant nothing and act on no field but through their bodies' stamp.

### Corpses (CORP) acts on

- **Air**: *none, on purpose*. None: corpses are not stamped, so a body on the floor does not slow air (ACCEPTED GAP). Where: `GameMap.stamp_units` (living only); `.claude/rules/units.md` Corpse law.
- **Heat**: *none, on purpose*. None: a corpse absorbs no radiation of its own (ACCEPTED GAP). Where: same.
- **Doors**: *none, on purpose*. None: corpses never block a door. Where: `door_system._occupancy_clear`.
- **Zombies**: *rule*. A corpse infested to the threshold rises as a zombie, whole again, through the one conversion door (any corpse, a zombie's included, never a destroyed one). Where: `ai_zombie.convert_marines_to_zombies`, `rises_as_zombie`, `corpse.rises`; `[corpse]`.
- **Rounds**: *none, on purpose*. None: rounds pass over corpses (the march skips the dead). Where: `combat.BulletInFlight.advance` (`not e.alive`).
- **Light**: *none, on purpose*. None: a corpse casts no shadow in the sweep (not stamped). Where: same.
- *Not applicable*: Trace gases, Fire, Water, Walls, Furnishings, Devices, Hazards, Marines, Corpses, Swarm units, Sprays, Grenades and charges, Blasts, Noise, Sight. A corpse is acted on; it acts on nothing else (it is invisible to larvae, sensors and every field).

### Swarm units (SWM) acts on

- **Water**: *not decided*. None. Where: #236.
- **Walls**: *none, on purpose*. None: larvae eat nothing (P3 ruling 1: no food until P6). Where: `docs/swarm_P3_impl.md` ruling 1.
- **Doors**: *not decided*. None: a larva in a door span does not stop the door closing (and is then trapped, see Doors -> Swarm units). Where: `GameMap.seal_tiles`; no issue.
- **Devices**: *not decided*. None: a motion sensor counts units, never larvae. Where: `sensor_system._MotionSensorRuntime`; no issue.
- **Marines**: *rule*. Every larva not in coma adds its expected bite dose to its tile; a body standing there takes the sum over its footprint as a poison dose (uncredited, cause `larva_bite`). Two bodies on one tile each take it all. Where: `swarm::bites_on` (`cpp/src/swarm.h`) -> bite-dose plane -> `exchange.apply_bite_dose` (slot 9c3) -> `poison.dose`; `[swarm.larva] bite_*`.
- **Zombies**: *none, on purpose*. None: zombies take no poison dose, so larvae cannot hurt them. Where: `exchange.poison_immune`; `.claude/rules/combat.md` Poison.
- **Corpses**: *rule*. Larvae on a corpse raise its infestation level (the count plane under its footprint); the level decays by a fraction of itself. Where: `swarm::active_on` -> count plane -> `exchange.apply_corpse_infestation` -> `corpse.infest`; `[corpse]`.
- **Swarm units**: *none, on purpose*. None: no larva-larva interaction (fork 7a, ACCEPTED GAP). Where: `docs/swarm_P3_impl.md`.
- *Not applicable*: Air, Trace gases, Heat, Fire, Furnishings, Hazards, Rounds, Sprays, Grenades and charges, Blasts, Light, Noise, Sight. Larvae are not stamped and make no sound: they act on nothing but through their two planes (bite, count).

### Rounds (RND) acts on

- **Fire**: *none, on purpose*. Not directly: an incendiary or plasma round lights fuel only through its payload's heat. Where: `.claude/rules/explosions-fire.md` Starting a fire.
- **Water**: *none, on purpose*. None in the sim; a round stopping at a wall beside water throws a render-only splash. Where: `renderer/particles.py` SpraySystem, `water_wake.wet_near`.
- **Walls**: *rule*. A round that stops on a wall deposits its `wall_damage`; at 0 hp the tile is destroyed (glass, wood, steel and hull alike). Where: `combat.chew_wall` -> `GameMap.destroy_wall`; `[ammo.*] wall_damage`.
- **Furnishings**: *rule*. A crate that wins its cover roll eats the round's wall damage and breaks when spent. Grilles and foliage are never chewed. Where: `combat.chew_wall` via the exposure roll.
- **Doors**: *rule*. A round chews a closed door; the door dies whole. Where: `combat.chew_wall`; `door_system.sweep_doors`.
- **Marines**: *rule*. A round that stops on a unit hits it (a packet, crit by facing if the row has crit) and pushes it with its momentum; a beam skewers every unit on its line. Where: `combat.BulletInFlight.advance`, `push_on`; `combat.fire_beam`; `[ammo.*] damage`, `bullet_g`, `muzzle_mps`, `[impulse] bullet_push_mult`.
- **Zombies**: *rule*. As for marines, with the zombie's bullet multiplier. Where: `combat.BulletInFlight.advance` (`_bullet_mult_q`).
- **Corpses**: *none, on purpose*. None: rounds in flight stop at death (Corpse law). Where: `.claude/rules/units.md` Corpse law.
- **Swarm units**: *none, on purpose*. None: no shot damage to larvae (ACCEPTED GAP, arc #63 P3). Where: `docs/swarm_P3_impl.md`.
- **Blasts**: *rule*. A payload round (the 40 mm, a plasma bolt) executes its payload at the tile it stops on. Where: `combat.BulletInFlight.advance` -> `payloads.execute_payload`.
- **Light**: *none, on purpose*. None in the sim: muzzle flashes and tracers are transient render lights, never a sim field. Where: `.claude/rules/render-core.md` Frame lights.
- *Not applicable*: Air, Trace gases, Heat, Devices, Hazards, Rounds, Sprays, Grenades and charges, Noise, Sight. A round carries nothing an air, gas, heat or sound field can feel (the shot's noise is the shooter's); it meets no device, hazard, grenade or line of sight.

### Sprays (SPR) acts on

- **Trace gases**: *rule*. A spray lays its gas (fuel haze, poison, teargas) on the cone's tiles. Where: `combat.deposit_spray_cone`; `[ammo.*] gas_species`, `gas_amount`.
- **Heat**: *rule*. A spray lays heat on every tile of its cone (over the physical line), into this tick's heat deposit. Where: `combat.deposit_spray_cone` (slot 4b) via the edit queue; `[ammo.*] heat_deposit`.
- **Fire**: *none, on purpose*. Not directly: a flamer lights fuel through the heat it lays (fire starts only by delivered heat). Where: `.claude/rules/explosions-fire.md` Starting a fire.
- **Water**: *not decided*. None: a flamer over water heats nothing in the water. Where: #219.
- **Marines**: *rule*. Through what it lays: its heat and gas reach bodies by the heat and gas rows, credited to the sprayer this tick. The cone itself reads and writes no unit. Where: `combat.deposit_spray_cone` (claims); `exchange` heat and gas rows.
- **Zombies**: *rule*. As for marines (heat x4, poison immune). Where: same.
- **Corpses**: *rule*. Through the heat row only (gas stops at death). Where: `exchange.apply_environmental_damage`.
- **Swarm units**: *rule*. Through temperature only. Where: `swarm_larva.h`.
- *Not applicable*: Air, Walls, Furnishings, Doors, Devices, Hazards, Rounds, Sprays, Grenades and charges, Blasts, Light, Noise, Sight. A spray touches only heat and gas; walls, doors and blasts are not changed by it directly, and its picture and sound are presentation.

### Grenades and charges (GRN) acts on

- **Marines**: *not decided*. None: a thrown grenade passes over units (its walk asks only walls and grilles). Where: `combat.flight_blocked`; no issue.
- **Zombies**: *not decided*. None, as for marines. Where: no issue.
- **Blasts**: *rule*. A grenade detonates on its fuse at its tile (never a blocked one); a planted charge on its timer. Each is one payload execution. Where: `Simulation._update_grenades` (slot 2), `combat.detonation_tile`; `charges.tick` (slot 9e(d')) -> `payloads.execute_payload`.
- *Not applicable*: Air, Trace gases, Heat, Fire, Water, Walls, Furnishings, Doors, Devices, Hazards, Corpses, Swarm units, Rounds, Sprays, Grenades and charges, Light, Noise, Sight. Before it goes off a grenade or charge acts on nothing: every effect is its blast's.

### Blasts (BLST) acts on

- **Air**: *rule*. A blast drives the air outward: a pressure kick and wave, and for a strength row the divergence source of its expansion, over the tiles its walk reached. Where: `physics.apply_explosion`; `blast_expansion.start_source` / `start_expansion` -> `EOSSolver.div_source`; `[blast]`, `[payloads.*]`.
- **Trace gases**: *rule*. A blast clears smoke in its inner 40% and lays its soot over its reach; a gas row emits its gas. Where: `physics.apply_explosion` (clear); `payloads.deposit_over_tiles`, `add_explosion_smoke`, `emit_gas`.
- **Heat**: *rule*. A strength row lands its fixed heat over the reached tiles, scaled by the air it read. Where: `payloads.deposit_over_tiles` (`"heat"`), `heat_base_q`; `[blast]`.
- **Fire**: *none, on purpose*. None: a blast has no fire-starting rule; fuel catches only when the heat takes it to its ignition temperature (Erik, 2026-11-06). Where: `.claude/rules/explosions-fire.md` Starting a fire.
- **Water**: *rule*. Only through the air: the water head reads the air pressure, so a blast's pressure pushes the surface. How far a real blast moves water is UNVERIFIED (not measured here). The splash in the picture is render-only and currently drifts (#222). Where: `water_solver.cpp` (`k_p`); #222.
- **Walls**: *rule*. A blast damages hull, wood and grille tiles in its disc (through walls), opening those at 0 hp. Steel and glass take nothing: a hardcoded material tuple. Where: `physics.apply_explosion` (`_BLAST_WALL_MATERIALS`); #22.
- **Furnishings**: *not decided*. None except the grille: crates, kindling and foliage are not in the blast's material tuple, so a grenade never breaks a crate (heat may light it). Where: `physics._BLAST_WALL_MATERIALS`; #22.
- **Doors**: *rule*. A blast damages door tiles in its disc; a closed door that loses a tile dies whole. Where: `physics.apply_explosion`; `door_system.sweep_doors`.
- **Hazards**: *rule*. Every blast is logged; a heat source in range takes its damage and at 0 hp runs its death payload. Where: `GameMap.blast_log` -> `heat_source_system._blast_damage_q16`, `_die`.
- **Marines**: *rule*. A blast damages every unit in its radius with a linear falloff, through walls (a disc, no line test); it blinds those who see a flash row; it pushes only through its own wind (Air -> Marines). Where: `exchange.apply_blast_damage`; `flash_blind.apply_flash_blind` (`has_los`); `.claude/rules/explosions-fire.md`.
- **Zombies**: *rule*. As for marines. A blast kill never converts. Where: `exchange.apply_blast_damage`.
- **Corpses**: *rule*. The same disc damages a corpse's body, uncredited and with no event; a destroyed corpse is skipped. Where: `exchange.apply_blast_damage`; `corpse.destroyed`.
- **Swarm units**: *none, on purpose*. None: no blast damage to larvae (ACCEPTED GAP, arc #63 P3); only its heat reaches them, through temperature. Where: `docs/swarm_P3_impl.md`.
- **Grenades and charges**: *not decided*. None: a blast never sets off a grenade in flight or a planted charge. Where: #218 is the nearest; no issue.
- **Blasts**: *rule*. Blasts of one tick go off in list order; a later one reads the air the earlier one left. Where: `Simulation._update_grenades`; `blast_air.read_blast_air`.
- **Light**: *none, on purpose*. None in the sim: a blast's flash is one transient render light. Where: `.claude/rules/explosions-fire.md` Generic explosion (flash columns).
- **Noise**: *rule*. A detonation is a noise at its tile, of the row's `loudness` (0 = silent). Where: `payloads.execute_payload` -> `noise.emit`; `[payloads.*] loudness`.
- *Not applicable*: Devices, Rounds, Sprays, Sight. A blast does nothing to devices (vents and sensors have no hp), rounds, sprays or lines of sight.

### Light (LIT) acts on

- **Heat**: *none, on purpose*. None: light channels never enter the heat books, and heat is bit-identical with light on or off. Where: `.claude/rules/radiation-light.md` Radiation sweep.
- **Marines**: *none, on purpose*. None: no game rule may read light until light is game state. Where: `.claude/rules/radiation-light.md`; #81.
- **Zombies**: *none, on purpose*. None, same rule. Where: #81.
- **Swarm units**: *none, on purpose*. None: larvae sense temperature only. Where: `docs/swarm_P3_impl.md` ruling 1.
- **Sight**: *none, on purpose*. None: darkness hides nobody; sight is geometry only until light is game state. Where: #81.
- *Not applicable*: Air, Trace gases, Fire, Water, Walls, Furnishings, Doors, Devices, Hazards, Corpses, Rounds, Sprays, Grenades and charges, Blasts, Light, Noise. Light acts on nothing else in the sim; the renderer is its only reader (`light_field`).

### Noise (NOI) acts on

- **Marines**: *not decided*. None: marines and their bots do not hear noises (sound is presentation only). Where: `audio/` (presentation only); no issue.
- **Zombies**: *rule*. A noise wakes every non-awake zombie within its radius into HEARING (walls ignored); it walks to the nearest such noise. Where: `ai_zombie._hear` (slot 5, drains `GameMap.noise_log`); `[zombie] hearing`, `noise_*`.
- **Swarm units**: *none, on purpose*. None: larvae sense temperature only. Where: `docs/swarm_P3_impl.md` ruling 1.
- *Not applicable*: Air, Trace gases, Heat, Fire, Water, Walls, Furnishings, Doors, Devices, Hazards, Corpses, Rounds, Sprays, Grenades and charges, Blasts, Light, Noise, Sight. Only zombies hear; nothing else reads the noise log.

### Sight (SEE) acts on

- **Marines**: *rule*. What a team sees enters its team picture once a tick, under the match's sight rule; a flash blinds only those who see it. Where: `vision.can_see`, `team_picture.update` (slot 11); `flash_blind.apply_flash_blind`.
- **Zombies**: *rule*. A zombie wakes for good when it sees a marine within range along a clear line. Where: `ai_zombie._sees_prey` (`has_los`).
- *Not applicable*: Air, Trace gases, Heat, Fire, Water, Walls, Furnishings, Doors, Devices, Hazards, Corpses, Swarm units, Rounds, Sprays, Grenades and charges, Blasts, Light, Noise, Sight. Sight is a relation, not a body: it changes only what zombies and teams know.

